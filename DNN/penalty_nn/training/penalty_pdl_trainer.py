"""
A thin wrapper around Dyn_DNN4OPF.training.pdl_trainer.PDLTrainer.

We override **one** internal helper – `_primal_loss` – so that,
*when* the model supplies a custom `loss_fn`, the trainer delegates
to it, passing along the current penalty coefficient ρ.  
All remaining logic (dual updates, schedulers, logging, etc.)
is inherited unchanged.
"""
from Dyn_DNN4OPF.training.pdl_trainer import PDLTrainer


class PenaltyPDLTrainer(PDLTrainer):
    """Trainer that understands `PenaltyPDL.loss_fn()`."""

    # ------------------------------------------------------------------ #
    #  Only the primal loss is different – everything else we reuse.
    # ------------------------------------------------------------------ #
    def _primal_loss(self, x, batch):
        # If the model exposes a custom loss_fn(x, batch, rho) we use it.
        if hasattr(self.model, "loss_fn"):
            try:
                return self.model.loss_fn(x, batch, self.rho)
            except TypeError:
                # allow for legacy signature loss_fn(x, batch)
                return self.model.loss_fn(x, batch)

        # otherwise fall back to the vanilla implementation
        return super()._primal_loss(x, batch)
