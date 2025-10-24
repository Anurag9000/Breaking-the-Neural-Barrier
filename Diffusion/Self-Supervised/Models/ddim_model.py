import torch
from models.schedules import make_beta_schedule, DiffusionCoeffs


class DDIMSampler:
    """
    DDIM Sampler for deterministic or stochastic diffusion model sampling.
    Reference:
        "Denoising Diffusion Implicit Models" – Song et al. (2021)
    """

    def __init__(self, eps_model, T=1000, schedule="linear"):
        """
        Args:
            eps_model: Model predicting ε given (x_t, t)
            T (int): Total diffusion steps used in training
            schedule (str): Beta schedule type ("linear", "cosine", etc.)
        """
        self.model = eps_model  # expects net(x_t, t) -> eps
        self.T = T

        betas = make_beta_schedule(T, schedule)
        self.coeff = DiffusionCoeffs(betas)

    # ----------------------------------------------------------
    # DDIM sampling process
    # ----------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        B: int,
        steps: int = 50,
        eta: float = 0.0,
        device: str = "cuda",
        size: tuple = (3, 32, 32),
    ) -> torch.Tensor:
        """
        Generates samples using DDIM (deterministic if eta=0).

        Args:
            B (int): Batch size.
            steps (int): Number of sampling steps (≤ T).
            eta (float): Controls stochasticity; eta=0 → deterministic DDIM.
            device (str): Device to run on.
            size (tuple): Output image size (C, H, W).

        Returns:
            torch.Tensor: Generated samples in range [-1, 1].
        """
        # Uniform time stride
        ts = torch.linspace(self.T - 1, 0, steps, dtype=torch.long, device=device)
        x = torch.randn(B, *size, device=device)
        c = self.coeff

        for i in range(steps):
            t = ts[i].repeat(B)
            eps = self.model.net(x, t)

            alpha_bar_t = c.alpha_bars[t][:, None, None, None]
            x0 = (x - torch.sqrt(1 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)

            # Final step: return predicted x₀
            if i == steps - 1:
                x = x0
                break

            t_prev = ts[i + 1].repeat(B)
            alpha_bar_prev = c.alpha_bars[t_prev][:, None, None, None]

            # Compute variance and direction terms
            sigma_t = (
                eta
                * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t))
                * torch.sqrt(1 - alpha_bar_t / alpha_bar_prev)
            )

            dir_xt = torch.sqrt(1 - alpha_bar_prev) * eps
            x = torch.sqrt(alpha_bar_prev) * x0 + dir_xt

            if eta > 0:
                x = x + sigma_t * torch.randn_like(x)

        return x.clamp(-1, 1)
