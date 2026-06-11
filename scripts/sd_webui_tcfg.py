"""
sd-webui-TCFG — Tangential Damping CFG for Forge-derived WebUIs
================================================================
Location: extensions/sd-webui-TCFG/scripts/sd_webui_tcfg.py

Paper: arXiv:2503.18137

Hook: set_model_sampler_pre_cfg_function  (Pre-CFG, same tier as SkimmedCFG)

Compatibility:
    ✅  reForge / Forge Classic / Forge (lllyasviel) / Forge Neo
    ❌  A1111 — no Forge backend

Sorting priority: 13.0
    Runs before SkimmedCFG (14) so TCFG-damped uncond feeds into SkimmedCFG.

    Processing order when all three are active:
        TCFG (Pre-CFG, 13.0) → SkimmedCFG (Pre-CFG, 14) → CFG → MaHiRo (Post-CFG, 15.5)
"""

import logging
import os
import sys
import traceback
from functools import partial
from typing import Any

import gradio as gr
from modules import scripts, script_callbacks

# ---------------------------------------------------------------------------
# sys.path — ensure the extension root is importable
# ---------------------------------------------------------------------------
_EXT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _EXT_ROOT not in sys.path:
    sys.path.insert(0, _EXT_ROOT)
# ---------------------------------------------------------------------------

from sd_webui_tcfg import apply_tcfg, remove_tcfg_patches

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def _has_forge_backend(p) -> bool:
    return hasattr(p, "sd_model") and hasattr(p.sd_model, "forge_objects")


def _warn_no_forge() -> None:
    msg = (
        "[sd-webui-TCFG] Requires Forge backend "
        "(reForge / Forge Classic / Forge Neo / Forge). "
        "A1111 is not supported."
    )
    logger.warning(msg)
    print(msg)


# ---------------------------------------------------------------------------
# Script
# ---------------------------------------------------------------------------

class TCFGScript(scripts.Script):
    """
    TCFG — Tangential Damping CFG.

    Sorting priority 13.0 ensures this runs BEFORE SkimmedCFG (14),
    so SkimmedCFG receives the TCFG-damped uncond as its input.
    """

    sorting_priority = 13.0

    def __init__(self):
        self.enabled = False

    def title(self) -> str:
        return "TCFG"

    def show(self, is_img2img: bool):
        return scripts.AlwaysVisible

    def ui(self, is_img2img: bool):
        with gr.Accordion(open=False, label=self.title()):
            gr.HTML(
                "<p><i>"
                "<b>Pre-CFG</b>: Damps the tangential component of the unconditional "
                "score via SVD, reducing directional drift in guidance."
                "</i></p>"
            )
            enabled = gr.Checkbox(label="Enable TCFG", value=False)

        enabled.change(fn=lambda x: self._update_enabled(x), inputs=[enabled])
        return [enabled]

    def _update_enabled(self, value: bool) -> None:
        self.enabled = value

    def process_before_every_sampling(self, p, *args, **kwargs):
        if len(args) >= 1:
            self.enabled = bool(args[0])
        else:
            logger.warning("[TCFG] process_before_every_sampling: missing args")
            return

        # XYZ Grid override
        xyz = getattr(p, "_tcfg_xyz", {})
        if "enabled" in xyz:
            self.enabled = (xyz["enabled"] == "True")

        if not self.enabled:
            return

        if not _has_forge_backend(p):
            _warn_no_forge()
            return

        unet = p.sd_model.forge_objects.unet.clone()
        apply_tcfg(unet)
        p.sd_model.forge_objects.unet = unet

        p.extra_generation_params.update({"tcfg": "enabled"})
        logger.debug("[TCFG] applied")


# ---------------------------------------------------------------------------
# XYZ Grid support
# ---------------------------------------------------------------------------

def _set_xyz_value(p, x: Any, xs: Any, *, field: str) -> None:
    if not hasattr(p, "_tcfg_xyz"):
        p._tcfg_xyz = {}
    p._tcfg_xyz[field] = x


def _register_xyz_axes() -> None:
    xyz_grid = None
    for script in scripts.scripts_data:
        if script.script_class.__module__ == "xyz_grid.py":
            xyz_grid = script.module
            break

    if xyz_grid is None:
        return

    new_axes = [
        xyz_grid.AxisOption(
            "(TCFG) Enabled",
            str,
            partial(_set_xyz_value, field="enabled"),
            choices=lambda: ["True", "False"],
        ),
    ]

    if not any(x.label.startswith("(TCFG)") for x in xyz_grid.axis_options):
        xyz_grid.axis_options.extend(new_axes)


def _on_before_ui() -> None:
    try:
        _register_xyz_axes()
    except Exception:
        print(
            f"[sd-webui-TCFG] XYZ Grid registration failed:\n{traceback.format_exc()}"
        )


script_callbacks.on_before_ui(_on_before_ui)
