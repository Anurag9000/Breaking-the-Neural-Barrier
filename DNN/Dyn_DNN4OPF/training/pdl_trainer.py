import logging
from typing import Callable

import torch

# ─────────────────────────── internal imports ────────────────────────────
from Dyn_DNN4OPF.utils.constraint_losses import (
    inequality_residuals,
    compute_violation,
)
from Dyn_DNN4OPF.utils.optim_sched import get_optimizer_scheduler
from config import SCHEDULER_PARAMS
# ─── add after the other imports ──────────────────────────────
from Dyn_DNN4OPF.utils.pdl_constraints import (
    init_from_case, compute_g, compute_h, objective,
)

def _missing(name: str) -> Callable:
    """Return a stub that reminds the user to implement the missing helper."""

    def _fn(*_args, **_kwargs):  # pylint: disable=unused-argument
        raise NotImplementedError(
            f"`{name}` is not defined. Either add methods `compute_{name}` etc. "
            "to the batch object **or** make a global function with the same "
            "signature that accepts `y_pred` and returns the correct tensor."
        )

    return _fn


# ─────────────────────────── logging config ───────────────────────────────
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s • %(levelname)s • %(message)s",
)

DELTA = 1e-6  # tolerance for validation‑improvement check


class PDLTrainer:
    """Primal‑Dual Learning trainer (Park & Van Hentenryck 2022, Alg. 1).

    The class is *batch‑agnostic*:
    • If your DataLoader yields a custom object with attributes
      `features`, `compute_g`, `compute_h`, `objective`, everything just works.
    • If it yields a plain tuple `(x, …)` you must provide *global* helpers
      `compute_g`, `compute_h`, `objective` that turn a prediction `y_pred`
      into the corresponding tensors. These globals can live anywhere on the
      import path; see the fallback stubs above for guidance.
    """

    # ───────────────────────── static helpers ────────────────────────────
    @staticmethod
    def _x_from(batch):
        """Extract *x* regardless of whether `batch` is an object or a tuple."""
        return batch.features if hasattr(batch, "features") else batch[0]

    @staticmethod
    def _g_h_obj_from(batch, y_pred):
        """Return *(g, h, obj)* no matter the batch layout."""
        if hasattr(batch, "compute_g"):
            g = batch.compute_g(y_pred)
            h = batch.compute_h(y_pred)
            obj = batch.objective(y_pred)
        else:  # fall back to globals defined somewhere in the project
            g = compute_g(y_pred)  # type: ignore[arg-type]
            h = compute_h(y_pred)  # type: ignore[arg-type]
            obj = objective(y_pred)  # type: ignore[arg-type]
        return g, h, obj

    # ───────────────────────── init ──────────────────────────────────────
    def __init__(self, model, train_loader, val_loader, cfg):
        # networks
        self.primal, self.dual = model.primal, model.dual
        # data
        self.train_loader, self.val_loader = train_loader, val_loader
        # device & opts
        self.device = torch.device(cfg.device)
        self.primal.to(self.device)
        self.dual.to(self.device)

        self.opt_P, self.sched_P = get_optimizer_scheduler(
            self.primal.parameters(), lr=cfg.optimizer.lr_primal, **SCHEDULER_PARAMS
        )
        self.opt_D, self.sched_D = get_optimizer_scheduler(
            self.dual.parameters(), lr=cfg.optimizer.lr_dual, **SCHEDULER_PARAMS
        )

        # PDL hyper‑params
        p = cfg.pdl
        self.rho = p.rho_init
        self.rho_max = p.rho_max
        self.alpha = p.alpha
        self.tau = p.tau
        self.outer_iters = p.outer_iters
        self.inner_iters = p.inner_iters
        self.max_epochs = p.max_epochs
        self.patience = p.patience

    # ────────────────────────── training loop ────────────────────────────
    def train(self, *, delta: float = DELTA):
        """Run the outer PDL loop. Returns the (possibly updated) networks."""
        counter, best_v = self.patience, float("inf")
        prev_v = None

        for epoch in range(self.max_epochs):
            # —— 1. Primal update ——
            self.primal.train()
            for _ in range(self.inner_iters):
                for batch in self.train_loader:
                    x = self._x_from(batch).to(self.device)
                    loss_p = self._primal_loss(x, batch)
                    self.opt_P.zero_grad()
                    loss_p.backward()
                    self.opt_P.step()
                    self.sched_P.step()

            # —— 2. Measure violation ——
            v = self._evaluate_violation(self.val_loader)

            # —— 3. Dual update ——
            self.dual.train()
            for _ in range(self.inner_iters):
                for batch in self.train_loader:
                    x = self._x_from(batch).to(self.device)
                    loss_d = self._dual_loss(x, batch)
                    self.opt_D.zero_grad()
                    loss_d.backward()
                    self.opt_D.step()
                    self.sched_D.step()

            # —— 4. Penalty schedule + early‑stopping ——
            if prev_v is not None and v > self.tau * prev_v:
                self.rho = min(self.rho * self.alpha, self.rho_max)

            logger.info(
                f"[PDL] epoch {epoch + 1:03d}: violation = {v:.3e} | ρ = {self.rho:.2e}"
            )

            if v < best_v - delta:
                best_v, counter = v, self.patience
            else:
                counter -= 1
                if counter == 0:
                    logger.info("[PDL] early‑stopping triggered — no improvement.")
                    break
            prev_v = v

        return self.primal, self.dual

    # ────────────────────────── losses ───────────────────────────────────
    def _primal_loss(self, x, batch):
        y_pred = self.primal(x)
        # Detach µ, λ so that gradients do not flow into the dual network here.
        mu, lam = (t.detach() for t in self.dual(x))

        g, h, obj = self._g_h_obj_from(batch, y_pred)

        loss = obj.mean()
        # Augmented Lagrangian — Eq. (5) in the PDL paper.
        loss += (mu * g).mean() + (lam * h).mean()
        violation = inequality_residuals(g)
        loss += 0.5 * self.rho * (
            violation.pow(2).mean()
        + h.pow(2).mean()
        )
        return loss

    def _dual_loss(self, x, batch):
        with torch.no_grad():
            y_pred = self.primal(x)
            g, h, _ = self._g_h_obj_from(batch, y_pred)
            mu_k, lam_k = self.dual(x)

        # Target values according to Alg. 1 (lines 8–9).
        tgt_mu = torch.relu(mu_k + self.rho * g)
        tgt_lam = lam_k + self.rho * h

        mu_pred, lam_pred = self.dual(x)
        return ((mu_pred - tgt_mu) ** 2).mean() + ((lam_pred - tgt_lam) ** 2).mean()

    # ───────────────────── violation metric ─────────────────────────────


    def _evaluate_violation(self, loader):
        """
        Maximum *scalar* violation over the given loader (validation),
        as defined in Eq. (8) of the PDL paper:
        v_k = max_x { max( ||h_x(y)||_∞ , ||σ_x(y)||_∞ ) }
        where σ_x,j(y) = max{ g_x,j(y), –μ_k,j / ρ }.
        """
        self.primal.eval()
        self.dual.eval()
        max_v = 0.0
        with torch.no_grad():
            for batch in loader:
                x = self._x_from(batch).to(self.device)
                y = self.primal(x)
                g, h, _ = self._g_h_obj_from(batch, y)
                mu_k, lam_k = self.dual(x)  # mu_k: inequality duals, lam_k: equality duals

                # σ_x,j(y) = max{ g, –μ_k / ρ } for each inequality constraint
                sigma = torch.max(g, -mu_k / self.rho)

                # Compute per-sample ∞-norm violations
                v_g = sigma.abs().max(dim=1)[0]  # max over inequality constraints
                v_h = h.abs().max(dim=1)[0]      # max over equality constraints

                # Violation for each sample, then track the overall maximum
                v_sample = torch.max(v_g, v_h)
                max_v = max(max_v, v_sample.max().item())

        return max_v
