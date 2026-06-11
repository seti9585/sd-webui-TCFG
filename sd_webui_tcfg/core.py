"""
sd_webui_tcfg/core.py
=====================
TCFG — Tangential Damping Classifier-Free Guidance
Paper: arXiv:2503.18137

Algorithm:
    Before CFG is computed, the uncond noise score is projected onto the
    principal direction of the [uncond, cond] noise score matrix via SVD.
    This removes the tangential component of uncond relative to cond,
    reducing unwanted directional drift in the guidance.

    1. Convert denoised predictions → noise scores (epsilon = x_t − x0)
    2. Stack [uncond_noise, cond_noise] → (B, 2, H*W) matrix
    3. SVD → take first right singular vector v1
    4. uncond_td = project(uncond_noise, v1)   (principal direction only)
    5. Convert back → denoised space
    6. Standard CFG proceeds with the damped uncond

Hook type: set_model_sampler_pre_cfg_function  (same tier as SkimmedCFG)

Original implementation:
    Shiba-2-shiba/TCFG-APG-Mahiro-for-ForgeClassic (forge_tcfg.py)
"""

import torch

MARKER = "sd_webui_tcfg_v1"


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_tangential_damping(
    cond_score: torch.Tensor,
    uncond_score: torch.Tensor,
) -> torch.Tensor:
    """
    Project uncond noise score onto the principal SVD direction of
    the stacked [uncond, cond] noise score matrix.

    Args:
        cond_score:   (B, C, H, W) — noise prediction for conditioned pass
        uncond_score: (B, C, H, W) — noise prediction for unconditional pass

    Returns:
        uncond_td: (B, C, H, W) — tangentially damped uncond noise score
    """
    B = cond_score.shape[0]
    cond_flat   = cond_score.reshape(B, 1, -1).float()   # (B, 1, N)
    uncond_flat = uncond_score.reshape(B, 1, -1).float()  # (B, 1, N)

    # Stack into (B, 2, N) matrix; SVD is computed per-sample in the batch
    score_matrix = torch.cat([uncond_flat, cond_flat], dim=1)

    try:
        _, _, Vh = torch.linalg.svd(score_matrix, full_matrices=False)
    except RuntimeError:
        # CUDA SVD can fail on some hardware/drivers; fall back to CPU
        _, _, Vh = torch.linalg.svd(score_matrix.cpu(), full_matrices=False)
        Vh = Vh.to(uncond_flat.device)

    v1 = Vh[:, 0:1, :]  # (B, 1, N) — first right singular vector

    # Project uncond onto v1 (retain only the principal direction)
    uncond_td = (uncond_flat @ v1.transpose(-2, -1)) * v1  # (B, 1, N)

    return uncond_td.reshape_as(uncond_score).to(uncond_score.dtype)


# ---------------------------------------------------------------------------
# Pre-CFG hook
# ---------------------------------------------------------------------------

@torch.no_grad()
def _tcfg_pre_cfg_fn(args: dict) -> list:
    """
    Pre-CFG function registered via set_model_sampler_pre_cfg_function.

    Receives denoised predictions; converts to noise space for SVD damping,
    then converts the damped uncond back to denoised space.
    Standard CFG proceeds unchanged after this modification.
    """
    conds_out = args["conds_out"]   # [cond_denoised, uncond_denoised]
    x_orig    = args["input"]       # x_t (noisy latent)

    # conds_out[0] = positive (cond), conds_out[1] = negative (uncond)
    # If uncond is zeroed (CFG scale = 1 or uncond disabled), skip
    if not torch.any(conds_out[1]):
        return conds_out

    cond_denoised   = conds_out[0]
    uncond_denoised = conds_out[1]

    # Denoised → noise (epsilon) space
    cond_noise   = x_orig - cond_denoised
    uncond_noise = x_orig - uncond_denoised

    # Tangential damping on uncond noise
    uncond_td_noise = score_tangential_damping(cond_noise, uncond_noise)

    # Noise → denoised space
    conds_out[1] = x_orig - uncond_td_noise

    return conds_out


# Marker for identification / deduplication
_tcfg_pre_cfg_fn._sd_webui_tcfg_marker = MARKER


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def _is_tcfg_fn(fn) -> bool:
    return getattr(fn, "_sd_webui_tcfg_marker", None) == MARKER


def remove_tcfg_patches(unet) -> None:
    """Remove all TCFG pre_cfg_function patches from unet.model_options."""
    key = "sampler_pre_cfg_function"
    existing = unet.model_options.get(key, [])
    if isinstance(existing, list):
        unet.model_options[key] = [fn for fn in existing if not _is_tcfg_fn(fn)]


def apply_tcfg(unet):
    """Apply TCFG pre_cfg_function to unet exclusively (deduplicates)."""
    remove_tcfg_patches(unet)
    unet.set_model_sampler_pre_cfg_function(_tcfg_pre_cfg_fn)
    return unet
