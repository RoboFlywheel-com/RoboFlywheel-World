# SPDX-License-Identifier: Apache-2.0
"""A real delivery manifest must satisfy the schema emitted from its own contract.

The manifest was a dict assembled at the end of a run with no schema and nothing validating it.
That is how it lost `zones` and `assets_used` in a rewrite: no document declared which keys were
supposed to be there, so their absence was not a failure anywhere -- it surfaced later as a KeyError
in a downstream tool.

Two checks, because they catch different things:

* **The key set** is checked with the stdlib, so it runs everywhere. A manifest carrying a key the
  schema does not describe, or missing one it requires, is a drift between the producer and the
  contract -- and the schema's `additionalProperties: false` is what makes the first case an error
  rather than a note.
* **Full validation** runs when `jsonschema` is importable. The core is stdlib-only by contract, so
  this cannot be a hard dependency; it is a stronger check where the library is available, and the
  key-set check still holds where it is not.

Both skip when there is no delivery to check. A test that invented a manifest would only be testing
itself.
"""

from __future__ import annotations
from rfw_w.scenes.resources import data_path, module_path

import json
from pathlib import Path

import pytest

# These checks inspect generated deliveries, which are intentionally absent in a
# clean checkout. Keep the existence guard strict in the explicit delivery tier.
pytestmark = pytest.mark.delivery

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA = data_path('schemas') / 'delivery-manifest.schema.json'


@pytest.fixture(scope="module")
def schema() -> dict:
    if not SCHEMA.is_file():
        pytest.skip("no emitted manifest schema; run scripts/ops/emit_schemas.py")
    return json.loads(SCHEMA.read_text())


@pytest.fixture(scope="module")
def deliveries(schema) -> list[Path]:
    """Scene runs whose manifest is the version the schema describes.

    Earlier runs on disk were produced by earlier code and declare v1. They are history rather than
    evidence about the current producer, and holding them to a schema written after them would mean
    either weakening the schema or deleting the comparison -- so they are excluded here, and
    `test_a_current_delivery_exists` fails if the exclusion ever leaves nothing to check.
    """
    root = REPO_ROOT / "outputs"
    if not root.is_dir():
        pytest.skip("no outputs/ directory; nothing has been run here")
    pinned = schema["properties"]["schema_version"]["const"]
    found = [
        path for path in root.glob("*/*/delivery_manifest.json")
        if json.loads(path.read_text()).get("schema_version") == pinned
    ]
    if not found:
        pytest.skip(f"no delivery declaring {pinned}; run an experiment first")
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


def test_a_current_delivery_exists(schema) -> None:
    """Guards the exclusion above: a suite that checks nothing must not pass.

    Delete every delivery, or stop running experiments, and the four checks in this file would all
    skip -- which reads as green.
    """
    root = REPO_ROOT / "outputs"
    pinned = schema["properties"]["schema_version"]["const"]
    declaring = [
        path for path in root.glob("*/*/delivery_manifest.json")
        if json.loads(path.read_text()).get("schema_version") == pinned
    ] if root.is_dir() else []
    assert declaring, (
        f"no delivery on disk declares {pinned}, so nothing in this file is being checked; run "
        f"scripts/run_m1_acceptance.py --experiment <id> --run <name>"
    )


class TestKeySet:
    """The stdlib half. No delivery on disk means nothing to check, which is a skip, not a pass."""

    def test_every_delivery_carries_only_described_keys(self, schema, deliveries) -> None:
        described = set(schema["properties"])
        offenders = {}
        for path in deliveries:
            manifest = json.loads(path.read_text())
            extra = sorted(set(manifest) - described)
            if extra:
                offenders[str(path.relative_to(REPO_ROOT))] = extra
        assert not offenders, (
            f"the manifest carries keys the schema does not describe: {offenders}. "
            f"Either the producer grew a field the contract did not, or the schema is stale"
        )

    def test_every_delivery_carries_the_required_keys(self, schema, deliveries) -> None:
        required = set(schema["required"])
        offenders = {}
        for path in deliveries:
            manifest = json.loads(path.read_text())
            missing = sorted(required - set(manifest))
            if missing:
                offenders[str(path.relative_to(REPO_ROOT))] = missing
        assert not offenders, f"a delivery is missing required keys: {offenders}"

    def test_the_declared_version_is_the_one_the_schema_pins(self, schema, deliveries) -> None:
        """A manifest claiming a version the schema does not describe should not validate."""
        pinned = schema["properties"]["schema_version"]["const"]
        for path in deliveries:
            manifest = json.loads(path.read_text())
            assert manifest.get("schema_version") == pinned, (
                f"{path.name} declares {manifest.get('schema_version')!r}, schema pins {pinned!r}"
            )


class TestFullValidation:
    def test_a_delivery_validates_against_the_emitted_schema(self, schema, deliveries) -> None:
        jsonschema = pytest.importorskip("jsonschema")
        validator = jsonschema.Draft202012Validator(schema)
        failures = {}
        for path in deliveries:
            errors = sorted(validator.iter_errors(json.loads(path.read_text())), key=str)
            if errors:
                failures[str(path.relative_to(REPO_ROOT))] = [
                    f"{'/'.join(str(part) for part in error.absolute_path)}: {error.message}"
                    for error in errors[:8]
                ]
        assert not failures, failures

    def test_the_schema_rejects_a_manifest_that_lost_a_key(self, schema, deliveries) -> None:
        """The failure that motivated the schema, written as a test.

        `zones` and `assets_used` were dropped from the manifest once by a rewrite and nobody
        noticed until a tool raised a KeyError. Under this schema that edit fails immediately.
        """
        jsonschema = pytest.importorskip("jsonschema")
        jsonschema.Draft202012Validator.check_schema(schema)
        validator = jsonschema.Draft202012Validator(schema)
        stripped = json.loads(deliveries[0].read_text())
        del stripped["gates"]
        assert list(validator.iter_errors(stripped)), (
            "a manifest missing `gates` validated; the required list is not doing anything"
        )
