# Copyright (C) 2026 seti9585
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# Derived from ComfyUI (comfy_extras/nodes_tcfg.py), Copyright (C)
# comfyanonymous and ComfyUI contributors, licensed under GPL-3.0.

"""
sd-webui-TCFG - Tangential Damping CFG for Forge-derived WebUIs
================================================================
Location: extensions/sd-webui-TCFG/scripts/sd_webui_tcfg.py

Paper: arXiv:2503.18137

Hook: set_model_sampler_pre_cfg_function (reForge / Forge Classic)
      sampler_post_cfg_function (Forge Neo -- that backend calls the pre-CFG
      hook before model evaluation, so denoised predictions are not yet
      available there; see core.py for how the post-CFG path reconstructs
      the equivalent result and stashes it for SkimmedCFG / DifferenceCFG /
      APG to read)

Compatibility:
    [OK]  reForge / Forge Classic / Forge (lllyasviel) / Forge Neo
    [NO]  A1111 - no Forge backend

_sd_webui_priority: 13.0
    Execution order in the chain is decided by priority insertion
    (_priority_insert_pre_cfg / _priority_insert_post_cfg in core.py), not
    by sorting_priority below, which only controls where this extension's
    accordion is drawn in the UI. TCFG registers at the lowest priority in
    the suite so it runs first and receives the raw unconditional
    prediction:
        TCFG (13.0) -> SkimmedCFG (14.0) -> DifferenceCFG (14.2)
        -> APG (14.5) -> CFG -> CFGZeroStar (15.0) -> FreSca (15.2)
        -> MaHiRo (15.5) -> CFGNorm (16.0) -> CFGRegulator (16.5)

    Set SD_WEBUI_SETI_DEBUG=1 to have the assembled chain printed at
    sampling time.
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
# sys.path - ensure the extension root is importable
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
    TCFG - Tangential Damping CFG.

    sorting_priority below only controls where this accordion is drawn in
    the UI. Execution order (TCFG running before SkimmedCFG) is decided by
    _sd_webui_priority = 13.0 and priority insertion in core.py -- see the
    module docstring above.
    """

    # UI accordion position only. Execution order is decided separately by
    # _sd_webui_priority = 13.0 and priority insertion in core.py.
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

        # Infotext round-trip (PNG Info -> Send to txt2img / img2img).
        # The Enable checkbox is bound through a callable because infotext paste
        # leaves a component untouched when its key is absent (Forge Neo ->
        # gr.skip(), reForge -> gr.update() no-op); a bare key could never turn
        # TCFG off. The callable resolves a missing "tcfg" key to False, forcing
        # OFF when an image generated without TCFG is sent. The metadata itself
        # is written in process() (see below).
        #
        # NOTE: no enabled.change() listener is registered here. A previous
        # version synced an instance flag via change(), but that value was always
        # overwritten by the UI args read in process_before_every_sampling(), so
        # it was dead code; removing it also keeps the checkbox clear of any event
        # listener that could interfere with the paste update.
        self.infotext_fields = [
            (enabled, lambda d: d.get("tcfg", "") == "enabled"),
        ]

        return [enabled]

    # ------------------------------------------------------------------
    # Effective enable state (UI checkbox + XYZ Grid override)
    # ------------------------------------------------------------------

    def _effective_enabled(self, p, args) -> bool:
        enabled = bool(args[0]) if len(args) >= 1 else False
        xyz = getattr(p, "_tcfg_xyz", {})
        if "enabled" in xyz:
            enabled = (xyz["enabled"] == "True")
        return enabled

    # ------------------------------------------------------------------
    # Metadata write (runs once before sampling, like a normal extension)
    # ------------------------------------------------------------------

    def process(self, p, *args):
        # Write the infotext key here, not in process_before_every_sampling().
        # process() runs once before the batch loop and the resulting
        # p.extra_generation_params is captured by create_infotext() for every
        # saved image, which is required for the PNG Info round-trip to work.
        # The XYZ override is read so the recorded value matches the run.
        if self._effective_enabled(p, args):
            p.extra_generation_params["tcfg"] = "enabled"

    # ------------------------------------------------------------------
    # Hook application (correct timing for forge_objects.unet)
    # ------------------------------------------------------------------

    def process_before_every_sampling(self, p, *args, **kwargs):
        self.enabled = self._effective_enabled(p, args)
        if not self.enabled:
            return

        if not _has_forge_backend(p):
            _warn_no_forge()
            return

        unet = p.sd_model.forge_objects.unet.clone()
        apply_tcfg(unet)
        p.sd_model.forge_objects.unet = unet
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
