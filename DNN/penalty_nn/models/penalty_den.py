
"""
===============================================================================
Penalty-DEN (Single-Head) — built on top of the base DEN in dnn_den.py
===============================================================================

This module keeps the *single-head architecture and dynamic expansion logic*
from your `DEN` (in dnn_den.py), and adds **physics-aware penalty training**:

    L = λ_mse · MSE(ŷ, y) + λ_eq · ‖power-balance residuals‖²
                          + λ_ineq · ‖box-constraint violations‖²

Key features
------------
• **Single head**: output ordering remains [PG | QG | Va | Vm] just like your
  non-penalty `DEN` (same head, same forward pass & gating).
• **Case constants registry**: call `register_case_constants(...)` once to put
  OPF constants on the model's device (bounds, indices, Y-bus).
• **Penalty loss**: `loss_fn(x, y_true, y_pred, metadata)` computes composite loss.
  - If `metadata` includes `pd` and `qd`, those are used; otherwise, it assumes
    `x` begins with [PD (n_bus), QD (n_bus)].
• **Test-only clipping**: set `model.clip_test=True` and set `model.eval_stage = "test"`
  before evaluation on your test set; validation uses `eval_stage="val"` (no clipping).
• **fit_task**: optional convenience trainer that:
    - trains with the composite loss,
    - validates on *pure MSE* (sample-weighted, batch-size agnostic),
    - expands the *last* hidden layer when patience runs out **and** best val MSE
      remains above `loss_thr`, then rebuilds the optimizer.

Integration
-----------
• Requires your existing `DEN` implementation in `dnn_den.py`.
• Uses `constraint_losses.power_balance_residuals` already present in your repo.
• No external scheduler helper is required; uses Adam by default.

Usage
-----
    from penalty_den_singlehead import PenaltyDEN

    model = PenaltyDEN(config).to(device)
    model.register_case_constants({
        "p_min": p_min, "p_max": p_max,
        "q_min": q_min, "q_max": q_max,
        "v_min": v_min, "v_max": v_max,
        "gen_buses": gen_bus_idx, "load_buses": load_bus_idx,
        "y_bus": ybus_sparse_or_tuple,
    })
    model.current_task = 0

    # Train
    best_val_mse = model.fit_task(train_loader, val_loader)

    # Validate (no clipping)
    model.eval_stage = "val"
    val_mse = model.evaluate_mse(val_loader)

    # Test (clip only if desired)
    model.clip_test = True
    model.eval_stage = "test"
    test_mse = model.evaluate_mse(test_loader)

"""
from __future__ import annotations
from typing import Optional, Dict, Tuple, Union
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import base DEN and the residuals helper with robust fallbacks
from Dyn_DNN4OPF.models.dnn_den import DEN
from Dyn_DNN4OPF.utils.constraint_losses import power_balance_residuals
Tensor = torch.Tensor

class PenaltyDEN(DEN):
    """Single-head DEN with composite physics-aware penalty loss."""

    __all__ = ["PenaltyDEN"]

    def __init__(
        self,
        config,
        *,
        lambda_mse: float = 1.0,
        lambda_eq: float = 1.0,
        lambda_ineq: float = 1.0,
        clip_test: bool = False,
    ):
        super().__init__(config)
        # penalty weights & clip flag stored as buffers (moved with .to(device))
        self.register_buffer("lambda_mse",  torch.as_tensor(float(lambda_mse)), persistent=False)
        self.register_buffer("lambda_eq",   torch.as_tensor(float(lambda_eq)),  persistent=False)
        self.register_buffer("lambda_ineq", torch.as_tensor(float(lambda_ineq)), persistent=False)
        self.clip_test = bool(clip_test)

        # Eval stage selector to support "clip only during *test*"
        # Valid values: "val", "test" (default to "val" → no clipping during validation)
        self.eval_stage: str = "val"

        # --- placeholders populated by register_case_constants(...) ---
        self._ybus_kind = None
        self._ybus_shape = None
        self.register_buffer("p_min", torch.empty(1, 0), persistent=False)
        self.register_buffer("p_max", torch.empty(1, 0), persistent=False)
        self.register_buffer("q_min", torch.empty(1, 0), persistent=False)
        self.register_buffer("q_max", torch.empty(1, 0), persistent=False)
        self.register_buffer("v_min", torch.empty(1, 0), persistent=False)
        self.register_buffer("v_max", torch.empty(1, 0), persistent=False)
        self.register_buffer("gen_bus_idx", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("load_bus_idx", torch.empty(0, dtype=torch.long), persistent=False)
        # Y-bus buffers are created lazily in register_case_constants

    # ------------------------------------------------------------------
    # Constants registry
    # ------------------------------------------------------------------
    @torch.no_grad()
    def register_case_constants(self, const: Dict[str, torch.Tensor] | SimpleNamespace):
        """Move case constants to the model device and store as buffers.

        Expected keys in `const` (dict or SimpleNamespace):
            • p_min, p_max: (n_gen,)
            • q_min, q_max: (n_gen,)
            • v_min, v_max: (n_bus,)
            • gen_buses:    (n_gen,)   long
            • load_buses:   (n_loads,) long
            • y_bus:        torch.sparse_coo_tensor  *or*
                             tuple(vals, rows, cols, shape)
        """
        dev = next(self.parameters()).device
        if isinstance(const, SimpleNamespace):
            const = const.__dict__

        def _row(t):
            t = torch.as_tensor(t, device=dev)
            return t.view(1, -1) if t.dim() == 1 else t.to(device=dev)

        for k in ("p_min","p_max","q_min","q_max","v_min","v_max"):
            self.register_buffer(k, _row(const[k]).to(dev), persistent=False)

        self.register_buffer("gen_bus_idx",  torch.as_tensor(const["gen_buses"], device=dev, dtype=torch.long), persistent=False)
        self.register_buffer("load_bus_idx", torch.as_tensor(const["load_buses"], device=dev, dtype=torch.long), persistent=False)

        yb = const["y_bus"]
        if torch.is_tensor(yb) and yb.is_sparse:
            self.register_buffer("y_bus_sparse", yb.to(dev), persistent=False)
            self._ybus_kind = "sparse"
            self._ybus_shape = yb.shape
        else:
            vals, rows, cols, shape = yb
            self.register_buffer("ybus_vals", torch.as_tensor(vals, device=dev), persistent=False)
            self.register_buffer("ybus_rows", torch.as_tensor(rows, device=dev, dtype=torch.long), persistent=False)
            self.register_buffer("ybus_cols", torch.as_tensor(cols, device=dev, dtype=torch.long), persistent=False)
            shp = torch.as_tensor(shape, device=dev) if not torch.is_tensor(shape) else shape.to(dev)
            self.register_buffer("_ybus_shape_tensor", shp, persistent=False)
            self._ybus_kind = "tuple"
            self._ybus_shape = tuple(int(x) for x in shp.tolist())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _ybus(self):
        if self._ybus_kind == "sparse":
            return self.y_bus_sparse
        return (
            self.ybus_vals,
            self.ybus_rows,
            self.ybus_cols,
            self._ybus_shape if isinstance(self._ybus_shape, tuple)
            else tuple(int(x) for x in self._ybus_shape_tensor.tolist()),
        )

    def _sizes(self) -> Tuple[int, int]:
        ng = int(self.p_min.size(1))
        nb = int(self.v_min.size(1))
        return ng, nb

    def _split_pred(self, y: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Assumes output layout [PG | QG | Va | Vm] -> returns (pg,qg,va,vm)."""
        ng, nb = self._sizes()
        pg = y[:, :ng]
        qg = y[:, ng: 2*ng]
        va = y[:, 2*ng: 2*ng + nb]
        vm = y[:, 2*ng + nb: 2*ng + 2*nb]
        return pg, qg, va, vm

    def _ineq_violation(self, pg: Tensor, qg: Tensor, vm: Tensor) -> Tensor:
        """Sum of squared distances to box bounds, per sample."""
        def sqrelu(a): return F.relu(a) ** 2
        v = 0.0
        v = v + sqrelu(self.p_min - pg).sum(dim=1) + sqrelu(pg - self.p_max).sum(dim=1)
        v = v + sqrelu(self.q_min - qg).sum(dim=1) + sqrelu(qg - self.q_max).sum(dim=1)
        v = v + sqrelu(self.v_min - vm).sum(dim=1) + sqrelu(vm - self.v_max).sum(dim=1)
        return v

    # ------------------------------------------------------------------
    # Forward (inherit DEN, add optional test-only clipping)
    # ------------------------------------------------------------------
    def forward(self, x: Tensor) -> Tensor:
        """DEN forward with gating (super) + optional test-only clipping.

        IMPORTANT: To ensure *test-only* clipping, set:

            model.clip_test = True

            model.eval(); model.eval_stage = "test"

        Validation should be run with `model.eval_stage = "val"`.

        """
        y = super().forward(x)  # uses DEN gating and single head
        if self.clip_test and (not self.training) and (self.eval_stage == "test") and self.p_min.numel() > 0:
            pg, qg, va, vm = self._split_pred(y)
            pg = torch.clamp(pg, self.p_min, self.p_max)
            qg = torch.clamp(qg, self.q_min, self.q_max)
            vm = torch.clamp(vm, self.v_min, self.v_max)
            y = torch.cat([pg, qg, va, vm], dim=1)
        return y

    # ------------------------------------------------------------------
    # Composite penalty loss
    # ------------------------------------------------------------------
    def loss_fn(
        self,
        x: Tensor,
        y_true: Tensor,
        y_pred: Tensor,
        *,
        metadata: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Return (loss, scalars) for logging."""
        ng, nb = self._sizes()

        # split predictions/targets
        pg, qg, va, vm = self._split_pred(y_pred)
        pg_t, qg_t, va_t, vm_t = self._split_pred(y_true)

        # PD/QD
        if metadata is not None and ("pd" in metadata and "qd" in metadata):
            pd = metadata["pd"]
            qd = metadata["qd"]
        else:
            pd = x[:, :nb]
            qd = x[:, nb: 2*nb]

        # terms
        mse  = F.mse_loss(y_pred, y_true, reduction="mean")
        r_re, r_im = power_balance_residuals(
            pg=pg, qg=qg, pd=pd, qd=qd, vm=vm, va=va,
            y_bus=self._ybus(), gen_bus_idx=self.gen_bus_idx, load_bus_idx=self.load_bus_idx, n_bus=nb
        )
        eq   = (r_re.pow(2).sum(dim=1) + r_im.pow(2).sum(dim=1)).mean()
        ineq = self._ineq_violation(pg, qg, vm).mean()

        total = self.lambda_mse * mse + self.lambda_eq * eq + self.lambda_ineq * ineq
        scalars = dict(mse=float(mse.detach()), eq=float(eq.detach()),
                       ineq=float(ineq.detach()), total=float(total.detach()))
        return total, scalars

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate_mse(self, loader) -> float:
        """Pure MSE over a DataLoader (sample-weighted)."""
        device = next(self.parameters()).device
        self.eval()
        prev_stage = self.eval_stage
        # make sure validation doesn't apply test-only clipping
        self.eval_stage = "val"
        s, n = 0.0, 0
        for batch in loader:
            xb, yb, *rest = batch
            xb, yb = xb.to(device), yb.to(device)
            yp = self(xb)
            s += F.mse_loss(yp, yb, reduction="sum").item()
            n += yb.numel()
        self.eval_stage = prev_stage
        return s / max(1, n)

    # ------------------------------------------------------------------
    # Optional training loop (composite-loss training, MSE validation)
    # ------------------------------------------------------------------
    def fit_task(
        self,
        train_loader,
        val_loader,
        test_loader=None,
        *,
        max_epochs: Optional[int] = None,
        delta: Optional[float] = None,
    ) -> float:
        device = next(self.parameters()).device
        opt = torch.optim.Adam((p for p in self.parameters() if p.requires_grad), lr=float(self.lr))

        max_epochs = int(max_epochs if max_epochs is not None else self.max_epochs)
        best_val = float("inf")
        best_state = {k: v.detach().clone() for k, v in self.state_dict().items()}
        patience_left = int(self.patience)
        min_delta = float(delta if delta is not None else self.loss_thr)

        # optional warmup
        for _ in range(int(getattr(self, "warmup_epochs", 0))):
            self.train()
            for xb, yb, *rest in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                yp = self(xb)
                loss, _ = self.loss_fn(xb, yb, yp)
                opt.zero_grad()
                loss.backward()
                opt.step()

        # main loop
        while max_epochs > 0:
            max_epochs -= 1
            # train 1 epoch
            self.train()
            for xb, yb, *rest in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                yp = self(xb)
                loss, _ = self.loss_fn(xb, yb, yp)
                opt.zero_grad()
                loss.backward()
                opt.step()

            # validate with pure MSE (no clipping)
            val_mse = self.evaluate_mse(val_loader)

            if val_mse + min_delta < best_val:
                best_val = val_mse
                best_state = {k: v.detach().clone() for k, v in self.state_dict().items()}
                patience_left = int(self.patience)
            else:
                patience_left -= 1
                if patience_left <= 0 and best_val > float(self.loss_thr):
                    # expand last hidden layer (matching DEN semantics)
                    self.expand_layer(self.depth - 1, self.ex_k)
                    # rebuild optimizer after topology change
                    opt = torch.optim.Adam((p for p in self.parameters() if p.requires_grad), lr=float(self.lr))
                    patience_left = int(self.patience)
                    # continue training after expansion

        # restore best
        self.load_state_dict(best_state)
        return best_val
