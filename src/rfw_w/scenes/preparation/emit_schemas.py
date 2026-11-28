#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Emit JSON Schema documents for the v3 contracts.

The contracts are Python types with fail-closed validation; this makes the same shapes
available to anything outside this process -- a worker, a reviewer, or the independent
delivery verifier reading a JSON artefact.

The output is deterministic: fields are emitted in declaration order, keys are sorted, and no
timestamp or path is embedded. Emitting twice must produce byte-identical files, which is
what lets a schema change be reviewed as a diff rather than discovered at runtime.

Derivation is introspective rather than hand-written so that a schema cannot drift away from
the type it describes. Anything the emitter cannot express is recorded in the document's
``x-pvscene-unexpressed`` list instead of being silently omitted -- an incomplete schema that
looks complete is worse than one that admits its gaps.
"""

from __future__ import annotations
from rfw_w.scenes.resources import module_path, data_path

import argparse
import collections.abc as cabc
import json
import sys
import types
import typing
from dataclasses import MISSING, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Mapping, Union, get_args, get_origin


import rfw_w.scenes.contracts as contracts

SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"

# The serialised (to_dict) contracts, which are what cross a process boundary. Internal types
# such as Provenance appear nested rather than as top-level documents.
REGISTRY: tuple[tuple[str, type], ...] = (
    ("scene-spec", contracts.SceneSpec),
    ("asset-record", contracts.AssetRecordV3),
    ("scene-ir", contracts.SceneIR),
    ("scene-hashes", contracts.SceneHashes),
    ("tool-request", contracts.ToolRequest),
    ("tool-result", contracts.ToolResult),
    ("evidence", contracts.Evidence),
    ("readiness-resolution", contracts.ReadinessResolution),
    ("delivery-manifest", contracts.DeliveryManifest),
)


def _json_type(annotation: Any, unexpressed: list[str], where: str) -> dict[str, Any]:
    """Map a Python annotation onto a JSON Schema fragment."""
    if annotation is type(None):
        return {"type": "null"}
    if annotation in (str,):
        return {"type": "string"}
    if annotation in (bool,):
        return {"type": "boolean"}
    if annotation in (int,):
        return {"type": "integer"}
    if annotation in (float,):
        return {"type": "number"}
    if annotation is Any:
        return {}
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return {"type": "string", "enum": [member.value for member in annotation]}
    if is_dataclass(annotation):
        # Nested dataclasses are described by their own field set; a $ref would require a
        # bundled document per type, which the artefact layout does not have.
        return _object_schema(annotation, unexpressed, where)

    origin = get_origin(annotation)
    # get_origin returns the collections.abc class for typing aliases, not the typing name.
    if origin in (list, tuple, set, frozenset, cabc.Sequence, cabc.Set):
        args = [arg for arg in get_args(annotation) if arg is not Ellipsis]
        if not args:
            return {"type": "array"}
        if origin is tuple and len(args) > 1 and Ellipsis not in get_args(annotation):
            # A fixed-length tuple like (float, float, float) is a positional triple.
            return {"type": "array", "items": [_json_type(arg, unexpressed, where) for arg in args]}
        if len(set(map(str, args))) == 1:
            return {"type": "array", "items": _json_type(args[0], unexpressed, where)}
        return {"type": "array", "items": [str(arg) for arg in args]}
    if origin in (dict, cabc.Mapping, cabc.MutableMapping):
        args = get_args(annotation)
        if len(args) == 2:
            return {"type": "object", "additionalProperties": _json_type(args[1], unexpressed, where)}
        return {"type": "object"}
    if origin is Union or origin is types.UnionType:
        options = [_json_type(arg, unexpressed, where) for arg in get_args(annotation)]
        options = [option for option in options if option]
        if not options:
            return {}
        if len(options) == 1:
            return options[0]
        return {"anyOf": options}

    unexpressed.append(f"{where}: {annotation!r}")
    return {}


def _object_schema(cls: type, unexpressed: list[str], where: str) -> dict[str, Any]:
    # Fields are read through get_type_hints rather than field_info.type: every module here
    # uses `from __future__ import annotations`, so the raw annotation is the *string*
    # "'str'" and introspecting it directly would classify every field as unexpressible.
    try:
        hints = typing.get_type_hints(cls)
    except NameError as error:  # a forward reference the module never defined
        unexpressed.append(f"{where}: unresolved annotations ({error})")
        hints = {}

    properties: dict[str, Any] = {}
    required: list[str] = []
    for field_info in fields(cls):
        annotation = hints.get(field_info.name, field_info.type)
        fragment = _json_type(annotation, unexpressed, f"{where}.{field_info.name}")
        properties[field_info.name] = fragment
        if field_info.default is MISSING and field_info.default_factory is MISSING:
            required.append(field_info.name)
    # Keys `to_dict` writes that are not stored fields. A class declares them in
    # SERIALISED_DERIVED so the schema describes the document that is actually produced -- a
    # property that appears in every manifest and in no schema is how `meets_request` came to be
    # rejected by a contract test against a real delivery. Required, because `to_dict` always
    # emits them.
    for name, fragment in (getattr(cls, "SERIALISED_DERIVED", None) or {}).items():
        properties[name] = fragment
        required.append(name)

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = sorted(required)
    # Constructors reject unexpected keys, so the schema says so too.
    schema["additionalProperties"] = False
    return schema


def build_document(name: str, cls: type) -> dict[str, Any]:
    unexpressed: list[str] = []
    schema = _object_schema(cls, unexpressed, name)
    document: dict[str, Any] = {
        "$schema": SCHEMA_DIALECT,
        "$id": f"pvscene.v3/{name}",
        "title": cls.__name__,
        "description": (cls.__doc__ or "").strip().split("\n\n")[0].replace("\n", " "),
        **schema,
    }
    # Read the field default through dataclasses.fields, not getattr: on a slots dataclass
    # the class attribute is a member descriptor rather than the default value, so getattr
    # would silently return a descriptor and the pin would never be emitted.
    for field_info in fields(cls):
        if field_info.name == "schema_version" and isinstance(field_info.default, str):
            document["properties"]["schema_version"] = {
                "type": "string",
                "const": field_info.default,
            }
    if unexpressed:
        document["x-pvscene-unexpressed"] = sorted(set(unexpressed))
    # Annotating a field as `Mapping[str, Any]` emits `{"type": "object"}` -- a valid schema that
    # constrains nothing. That is a silent hole: it does not reach `x-pvscene-unexpressed`, so a
    # field added carelessly looks exactly like one deliberately left open. Listing them makes the
    # difference visible, and a test asserts the list is what it is expected to be.
    open_paths = sorted(_open_object_paths(schema))
    if open_paths:
        document["x-pvscene-untyped"] = open_paths
    return document


def _open_object_paths(node: Any, prefix: str = "") -> Iterator[str]:
    """Every path in a schema that resolves to an object constraining nothing.

    `{"type": "object"}` with no `properties` and no `additionalProperties` is what an
    ``Any``-valued mapping produces: any object at all validates against it. Walked rather than
    supplied by hand, because a hand-written list is the thing that drifts.
    """
    if not isinstance(node, Mapping):
        return

    def is_open(value: Any) -> bool:
        """An object that accepts anything.

        `Mapping[str, Any]` emits `additionalProperties: {}` -- any value at all is valid -- and a
        bare `{"type": "object"}` (a mapping with no declared value type) is the same hole written
        another way. `additionalProperties: false` with properties is a closed object and not a hole.
        """
        if not isinstance(value, Mapping) or value.get("type") != "object":
            return False
        if value.get("properties"):
            return False
        return value.get("additionalProperties") in (None, {})

    for key, value in (node.get("properties") or {}).items():
        path = f"{prefix}{key}"
        if is_open(value):
            yield path
        yield from _open_object_paths(value, f"{path}.")

    items = node.get("items")
    if isinstance(items, list):
        for option in items:
            if is_open(option):
                yield f"{prefix.rstrip('.')}[]"
            yield from _open_object_paths(option, prefix)
    elif isinstance(items, Mapping):
        if is_open(items):
            yield f"{prefix.rstrip('.')}[]"
        yield from _open_object_paths(items, prefix)

    for combinator in ("anyOf", "oneOf", "allOf"):
        for option in node.get(combinator) or []:
            if is_open(option):
                yield prefix.rstrip(".") or "(root)"
            yield from _open_object_paths(option, prefix)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    for name, cls in REGISTRY:
        document = build_document(name, cls)
        path = out / f"{name}.schema.json"
        path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
        written.append(path.name)

    print(json.dumps({"schema_count": len(written), "output_dir": str(out), "files": written}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
