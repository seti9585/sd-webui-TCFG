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

    1. Convert denoised predictions -> noise scores (epsilon = x_t - x0)
    2. Stack [uncond_noise, cond_noise] -> (B, 2, H*W) matrix
    3. SVD -> take first right singular vector v1
    4. uncond_td = project(uncond_noise, v1)   (principal direction only)
    5. Convert back -> denoised space
    6. Standard CFG proceeds with the damped uncond

Hook type:
    reForge / Forge Classic -> Pre-CFG  (set_model_sampler_pre_cfg_function,
                                dict args, "conds_out" style; runs AFTER
                                model evaluation there)
    Forge Neo               -> Post-CFG (sampler_post_cfg_function; Forge
                                Neo's pre-cfg list runs BEFORE model
                                evaluation, so cond/uncond predictions are
                                not available there -- confirmed by reading
                                backend/sampling/sampling_function.py
                                directly: sampler_pre_cfg_function fires
                                before calc_cond_uncond_batch())

Composition with SkimmedCFG on Forge Neo:
    Both TCFG and SkimmedCFG live in the same sampler_post_cfg_function list
    on Forge Neo -- there is no separate pre-CFG slot with denoised values
    available on that backend. Forge Neo rebuilds the args dict fresh on
    every hook call with the SAME raw cond_denoised / uncond_denoised
    tensors each time (only "denoised" chains forward), so TCFG cannot hand
    its damped uncond to SkimmedCFG through the "denoised" chain alone.
    Instead TCFG publishes its damped uncond into
    model_options["_tcfg_damped_uncond"] (model_options is the one dict
    object shared across the whole post-cfg loop for a given sampling call).
    SkimmedCFG's Forge Neo path reads this key if present and falls back to
    the raw uncond_denoised otherwise. See sd_webui_skimmed_cfg/core.py for
    the consuming side.

    List order matters: TCFG (priority 13.0) must run before SkimmedCFG
    (14.0) so the stash exists by the time SkimmedCFG runs. Both extensions
    insert themselves via _priority_insert_post_cfg(), which keeps
    SETI-suite hooks (identified by a _sd_webui_priority attribute on the
    marker function) in ascending priority order regardless of registration
    order, while leaving unrelated third-party post-cfg hooks wherever they
    already sit in the list.

Original implementation:
    Shiba-2-shiba/TCFG-APG-Mahiro-for-ForgeClassic (forge_tcfg.py)
"""

import logging
import os
import sys

import torch

logger = logging.getLogger(__name__)

MARKER = "sd_webui_tcfg_v1"

# Mirrors TCFGScript.sorting_priority in scripts/sd_webui_tcfg.py. Kept in
# sync manually; used only to order this extension's hook within Forge
# Neo's sampler_post_cfg_function list relative to other SETI extensions.
_PRIORITY = 13.0

# Suite-wide debug convention: 0 = off, 1 = apply-time settings + chain
# dump, 2 = per-step tracing.
DEBUG_ENV_VAR = "SD_WEBUI_SETI_DEBUG"

# One chain dump per sampling pass. Reset by apply_tcfg().
_CHAIN_DUMPED = False


def _debug_level():
    try:
        return int(os.environ.get(DEBUG_ENV_VAR, "0"))
    except Exception:
        return 0


def _emit(level, fmt, *args):
    """Emit to both logging and stderr. Some forks suppress module level
    loggers, so the stderr print is required for cross-backend visibility."""
    if _debug_level() < level:
        return
    try:
        msg = (fmt % args) if args else fmt
    except Exception:
        msg = str(fmt)
    text = "[TCFG] " + msg
    logger.warning(text)
    try:
        print(text, file=sys.stderr, flush=True)
    except Exception:
        pass


def _describe_chain(fns):
    """Render a hook list as 'name(priority)' in actual execution order."""
    parts = []
    for fn in fns or []:
        name = getattr(fn, "__name__", None) or type(fn).__name__
        prio = getattr(fn, "_sd_webui_priority", None)
        parts.append("%s(%s)" % (name, "-" if prio is None else prio))
    return " -> ".join(parts) if parts else "(empty)"


# ---------------------------------------------------------------------------
# Backend detection (identical logic to sd-webui-SkimmedCFG, duplicated
# intentionally -- each extension in this suite stays independent)
# ---------------------------------------------------------------------------

_BACKEND_IS_NEO = None  # cached


def _is_forge_neo_backend() -> bool:
    """
    Return True if the active backend is Forge Neo.

    Forge Neo's sampler_pre_cfg_function is called BEFORE model evaluation as
    fn(model, cond, uncond_, x, timestep, model_options) -- denoised
    predictions are not available there. On reForge / Forge Classic the hook
    receives a single dict whose "conds_out" already holds the predictions.

    Detection: Forge Neo ships backend.sampling.sampling_function with
    sampling_function_inner and calc_cond_uncond_batch; reForge / Classic use
    ldm_patched.modules.samplers instead.
    """
    global _BACKEND_IS_NEO
    if _BACKEND_IS_NEO is not None:
        return _BACKEND_IS_NEO

    is_neo = False
    try:
        from backend.sampling import sampling_function as _sf
        is_neo = (
            hasattr(_sf, "sampling_function_inner")
            and hasattr(_sf, "calc_cond_uncond_batch")
        )
    except Exception:
        is_neo = False

    _BACKEND_IS_NEO = is_neo
    logger.debug("[TCFG] backend detected: %s", "Forge Neo" if is_neo else "reForge / Forge Classic")
    return is_neo


# ---------------------------------------------------------------------------
# Priority-ordered insertion for Forge Neo's post-cfg list
# ---------------------------------------------------------------------------

def _priority_insert_post_cfg(unet, fn) -> None:
    """
    Insert fn into unet.model_options["sampler_post_cfg_function"] at the
    position that keeps SETI-suite hooks (those carrying a _sd_webui_priority
    attribute) in ascending priority order -- e.g. TCFG (13.0) before
    SkimmedCFG (14.0) before MaHiRo (15.5) -- regardless of the order in
    which their apply_*() functions happened to run this call. Third-party
    hooks without that attribute are left exactly where they already are;
    only the new fn's position relative to them is decided (inserted before
    the first tracked hook with a strictly greater priority, otherwise
    appended at the end).
    """
    key = "sampler_post_cfg_function"
    existing = unet.model_options.get(key, [])
    priority = fn._sd_webui_priority

    insert_at = len(existing)
    for i, other in enumerate(existing):
        other_priority = getattr(other, "_sd_webui_priority", None)
        if other_priority is not None and other_priority > priority:
            insert_at = i
            break

    unet.model_options[key] = existing[:insert_at] + [fn] + existing[insert_at:]


# ---------------------------------------------------------------------------
# Priority-ordered insertion for the reForge / Forge Classic pre-cfg list
# ---------------------------------------------------------------------------

def _priority_insert_pre_cfg(unet, fn, disable_cfg1_optimization: bool = False) -> None:
    """
    Twin of _priority_insert_post_cfg for the pre-CFG list. Identical
    semantics, different key.

    Replaces the plain append that set_model_sampler_pre_cfg_function
    performs. That append made execution order depend on extension load
    order, which put TCFG (13.0) at the END of the reForge pre-cfg chain --
    the opposite of its design intent, since tangential damping is meant to
    clean the raw uncond BEFORE the other guidance extensions reshape it.
    Forge Neo was already correct because that path used
    _priority_insert_post_cfg; only the reForge path was affected.

    disable_cfg1_optimization mirrors the flag that
    set_model_sampler_pre_cfg_function sets, so callers that rely on it keep
    working. TCFG does not need it (it returns conds_out untouched when
    uncond is all zeros), but the parameter is kept so this helper can be
    copied verbatim into sibling extensions that do.

    A new list is built rather than mutating in place, matching the
    semantics of the backend helper, so a cloned unet never leaks the
    change into its source.

    Duplicated deliberately: each extension in this suite carries its own
    copy so no cross-extension import dependency exists.
    """
    key = "sampler_pre_cfg_function"
    existing = unet.model_options.get(key, [])
    priority = fn._sd_webui_priority

    insert_at = len(existing)
    for i, other in enumerate(existing):
        other_priority = getattr(other, "_sd_webui_priority", None)
        if other_priority is not None and other_priority > priority:
            insert_at = i
            break

    unet.model_options[key] = existing[:insert_at] + [fn] + existing[insert_at:]

    if disable_cfg1_optimization:
        unet.model_options["disable_cfg1_optimization"] = True


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
# Pre-CFG hook  (reForge / Forge Classic)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _tcfg_pre_cfg_fn(args: dict) -> list:
    """
    Pre-CFG function registered via set_model_sampler_pre_cfg_function.

    Receives denoised predictions; converts to noise space for SVD damping,
    then converts the damped uncond back to denoised space.
    Standard CFG proceeds unchanged after this modification.
    """
    global _CHAIN_DUMPED

    conds_out = args["conds_out"]   # [cond_denoised, uncond_denoised]
    x_orig    = args["input"]       # x_t (noisy latent)

    # Ground truth for pre-CFG ordering. The post-CFG chain dump lives in
    # sd-webui-FreSca and reads sampler_post_cfg_function; it cannot see
    # this list. Emitted once per pass, from inside the hook, so what is
    # printed is the list as the sampler actually holds it at call time.
    if not _CHAIN_DUMPED and _debug_level() >= 1:
        _CHAIN_DUMPED = True
        try:
            opts = args.get("model_options") or {}
            _emit(1, "pre-CFG chain: %s",
                  _describe_chain(opts.get("sampler_pre_cfg_function")))
        except Exception as exc:
            _emit(1, "pre-CFG chain dump failed: %r", exc)

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
_tcfg_pre_cfg_fn._sd_webui_priority = _PRIORITY


# ---------------------------------------------------------------------------
# Post-CFG hook  (Forge Neo)
# ---------------------------------------------------------------------------
# Forge Neo post-CFG args dict keys (identical shape to sd-webui-SkimmedCFG):
#   "denoised"        -- current CFG result, chained from earlier post-cfg
#                         hooks. NOT used as an input here -- TCFG always
#                         damps starting from the raw predictions, matching
#                         the reForge pre-cfg behaviour of operating on
#                         conds_out before CFG math runs.
#   "cond_denoised"    -- positive prediction (raw, same tensor every call)
#   "uncond_denoised"  -- negative prediction (raw, same tensor every call;
#                         None when CFG=1 / uncond disabled)
#   "cond_scale"       -- CFG scale
#   "input"            -- x_t (noisy latent)
#   "model_options"    -- the actual model_options dict (shared reference
#                         across the whole post-cfg loop this call) -- used
#                         as the stash SkimmedCFG reads from.
# ---------------------------------------------------------------------------

def _make_tcfg_post_fn():
    """TCFG — Post-CFG (Forge Neo)."""
    @torch.no_grad()
    def _fn(args):
        uncond_denoised = args.get("uncond_denoised")
        if uncond_denoised is None or not torch.any(uncond_denoised):
            return args["denoised"]

        x_orig     = args["input"]
        cond_scale = args["cond_scale"]
        cond       = args["cond_denoised"]
        uncond     = uncond_denoised

        cond_noise   = x_orig - cond
        uncond_noise = x_orig - uncond
        uncond_td_noise = score_tangential_damping(cond_noise, uncond_noise)
        uncond_td = x_orig - uncond_td_noise

        # Publish the damped uncond for downstream post-cfg hooks (e.g.
        # SkimmedCFG) to build on -- mirrors the reForge pipeline where TCFG
        # runs before SkimmedCFG in the same pre-cfg list. model_options is
        # the same dict object across this call's post-cfg loop, so a hook
        # later in the list sees this key.
        model_options = args.get("model_options")
        if isinstance(model_options, dict):
            model_options["_tcfg_damped_uncond"] = uncond_td

        # Self-contained result in case TCFG is the only guidance-shaping
        # post-cfg hook active (e.g. SkimmedCFG disabled this run).
        return uncond_td + cond_scale * (cond - uncond_td)

    _fn._sd_webui_tcfg_marker = MARKER
    _fn._sd_webui_priority = _PRIORITY
    return _fn


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def _is_tcfg_fn(fn) -> bool:
    return getattr(fn, "_sd_webui_tcfg_marker", None) == MARKER


def remove_tcfg_patches(unet) -> None:
    """Remove all TCFG patches from both pre- and post-CFG lists."""
    for key in ("sampler_pre_cfg_function", "sampler_post_cfg_function"):
        existing = unet.model_options.get(key)
        if isinstance(existing, list):
            unet.model_options[key] = [fn for fn in existing if not _is_tcfg_fn(fn)]


def apply_tcfg(unet):
    """
    Apply TCFG to unet exclusively (deduplicates), choosing the correct hook
    for the active backend.

      * Forge Neo               -> Post-CFG, priority-ordered so it runs
                                    before SkimmedCFG (and stashes its
                                    damped uncond for SkimmedCFG to use).
      * reForge / Forge Classic -> Pre-CFG (original behaviour, unchanged).
    """
    global _CHAIN_DUMPED
    _CHAIN_DUMPED = False   # one chain dump per sampling pass

    remove_tcfg_patches(unet)

    if _is_forge_neo_backend():
        post_fn = _make_tcfg_post_fn()
        _priority_insert_post_cfg(unet, post_fn)
        _emit(1, "registered post-CFG hook (Forge Neo), priority=%s", _PRIORITY)
    else:
        # v1.1: priority-ordered insertion replaces the plain append that
        # set_model_sampler_pre_cfg_function performs. See
        # _priority_insert_pre_cfg for why.
        _priority_insert_pre_cfg(unet, _tcfg_pre_cfg_fn)
        _emit(1, "registered pre-CFG hook (reForge / Forge Classic), "
                 "priority=%s", _PRIORITY)

    return unet
