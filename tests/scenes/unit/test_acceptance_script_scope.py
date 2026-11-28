# SPDX-License-Identifier: Apache-2.0
"""No function in the acceptance script may use a derived id it does not define.

This is a source-level check, which this repository otherwise avoids -- it is here because the bug it
catches bit **twice**, and both times the symptom was nothing like the cause.

`support_id` and `target_id` used to be the literals `"desk"` and `"cup"` at roughly fifty places.
Replacing them with derived ids left two functions that *used* the names without binding them, and
the resulting `NameError` does not surface as a NameError. A stage handler that raises is caught by
the orchestrator and turned into a `StageOutcome` with `PRECONDITION_UNMET` and the exception text as
its detail; that signature maps to no `FailureEvent`, so `Driver._needs_semantic_change` returns True
and the loop spends a **model request** replanning a scene that was never broken. The run reports a
scene problem, and the actual line number is three layers down in a truncated detail string.

The first occurrence took an instrumented run to find. The second took another. A check that reads
the AST is cheaper than either.
"""

from __future__ import annotations
from rfw_w.scenes.resources import module_path, data_path

import ast
from pathlib import Path

import pytest

SOURCES = tuple(module_path(name) for name in ("rfw_w.scenes.application", "rfw_w.scenes.assembly.placement", "rfw_w.scenes.assembly.geometry", "rfw_w.scenes.delivery.compose"))

# The ids that were literals before the plan was derived from the spec, and that are therefore the
# ones a rename can leave dangling.
DERIVED_IDS = ("target_id", "support_id")


def _bound_and_used(function: ast.AST, module_names: set[str]) -> tuple[set[str], set[str]]:
    """Every name the function binds, and every name it reads."""
    bound: set[str] = {a.arg for a in function.args.args + function.args.kwonlyargs}  # type: ignore[attr-defined]
    if function.args.vararg:  # type: ignore[attr-defined]
        bound.add(function.args.vararg.arg)  # type: ignore[attr-defined]
    if function.args.kwarg:  # type: ignore[attr-defined]
        bound.add(function.args.kwarg.arg)  # type: ignore[attr-defined]
    used: set[str] = set()
    for node in ast.walk(function):
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                bound.add(node.id)
            else:
                used.add(node.id)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            bound |= {(alias.asname or alias.name).split(".")[0] for alias in node.names}
        elif isinstance(node, (ast.comprehension,)):
            bound |= {t.id for t in ast.walk(node.target) if isinstance(t, ast.Name)}
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node is not function:
            # A nested function's body sees the enclosing scope, so its assignments are not
            # evidence of anything -- but its parameters are bound for its own body only.
            bound.add(node.name)
    return bound, used


@pytest.fixture(scope="module")
def script() -> ast.Module:
    return ast.Module(body=[node for path in SOURCES for node in ast.parse(path.read_text()).body], type_ignores=[])


def _module_level_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for statement in tree.body:
        for node in ast.walk(statement):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                names.add(node.id)
    return names


def _offenders(tree: ast.Module) -> list[str]:
    module_names = _module_level_names(tree)
    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        bound, used = _bound_and_used(node, module_names)
        for name in sorted(set(DERIVED_IDS) & (used - bound - module_names)):
            found.append(f"{node.name}() at line {node.lineno} reads {name!r} without binding it")
    return found


def test_no_function_reads_a_derived_id_it_does_not_bind(script) -> None:
    offenders = _offenders(script)
    assert not offenders, (
        "a NameError here becomes `PRECONDITION_UNMET`, which the driver treats as a semantic "
        "failure and answers with a model request:\n  " + "\n  ".join(offenders)
    )


def test_the_check_would_notice_a_dangling_read(script) -> None:
    """The check must be able to fail, or it is another gate that cannot.

    A synthetic module with the same shape as the bug: a function using `target_id` that neither
    takes nor assigns it.
    """
    synthetic = ast.parse(
        "def stage(scene):\n"
        "    return scene['instances'][target_id]\n"
    )
    assert _offenders(synthetic), "the checker passed a function that reads a free `target_id`"


def test_a_bound_id_is_not_flagged(script) -> None:
    """And it must not cry wolf: a parameter, an assignment and a module constant are all fine."""
    synthetic = ast.parse(
        "TARGET_ID = 'cup'\n"
        "def from_param(target_id):\n"
        "    return target_id\n"
        "def from_local(scene):\n"
        "    target_id = scene['target_id']\n"
        "    return target_id\n"
        "def from_module():\n"
        "    return TARGET_ID\n"
    )
    assert not _offenders(synthetic)
