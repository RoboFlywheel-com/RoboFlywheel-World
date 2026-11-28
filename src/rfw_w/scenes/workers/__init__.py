# SPDX-License-Identifier: Apache-2.0
"""Worker processes: engine-specific execution behind a process boundary.

Nothing here is imported eagerly. A worker runs under its engine's interpreter -- Isaac, MuJoCo,
Blender -- as a subprocess, so the orchestrator never loads a simulator into its own process.
"""
