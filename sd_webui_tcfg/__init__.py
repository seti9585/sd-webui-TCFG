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

"""sd_webui_tcfg — Tangential Damping CFG for Forge-derived WebUIs"""
from .core import apply_tcfg, remove_tcfg_patches, MARKER

__all__ = ["apply_tcfg", "remove_tcfg_patches", "MARKER"]
