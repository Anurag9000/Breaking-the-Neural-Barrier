from .dae_vae_conv_stl import DAEVAEConv, dae_vae_total_neurons


class DAELadderVAEConv(DAEVAEConv):
    """
    Ladder VAE-style Conv DAE.

    For now this reuses the same backbone as DAEVAEConv, but is separated so
    that ADP and runners can evolve independently (e.g. adding auxiliary
    denoising paths at intermediate layers).
    """

    pass


def ladder_vae_total_neurons(width: int, depth: int, latent_dim: int) -> int:
    # Alias to the same capacity proxy as the plain VAE.
    return dae_vae_total_neurons(width, depth, latent_dim)

