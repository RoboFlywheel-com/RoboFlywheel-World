# SPDX-License-Identifier: Apache-2.0
"""VLM filter configuration.

Environment variables for server deployment:
  DASHSCOPE_API_KEY or ANTHROPIC_API_KEY : API key for the endpoint
  VLM_ENDPOINT_BASE                       : override endpoint base URL
  VLM_MODEL_ID                            : override model id
"""

import os

ENDPOINT_BASE = os.getenv(
    "VLM_ENDPOINT_BASE", "https://dashscope.aliyuncs.com/apps/anthropic"
)
MODEL_ID = os.getenv("VLM_MODEL_ID", "qwen3.7-plus")

IMG_SIDE = 448
JPEG_Q = 85
MAX_TOKENS = 1800
VIEWS = ("front", "back", "left", "right", "bottom")
CONCURRENCY = 4

SYSTEM_PROMPT = (
    "You are a 3D asset validator for rigid-body physics simulation. "
    "Judge each mesh by applying the checks in strict order. "
    "Output STRICT JSON only, no markdown fences, no extra commentary."
)

INPUT_DESC_5VIEW = (
    "Below are 5 orthographic views (front, back, left, right, bottom) of the SAME mesh. "
    "All five views use the SAME camera distance; the camera rotates 90° between "
    "horizontal views around the vertical axis, and the bottom view looks up from below. "
    "The background is white."
)


def build_task(domain: str, class_name: str, include_step6: bool = True) -> str:
    """Build the VLM judge task prompt with domain/class labels filled in.

    The whole prompt is a single f-string, so JSON braces must be doubled
    ({{ -> { after f-string interpolation). No .format() call afterwards.
    """
    if include_step6:
        step6_block = f"""
Step 6. matches_domain_class (bool)
   PASS if the observed object is visually consistent with the labeled domain
        "{domain}" AND class "{class_name}". A close functional/visual match is
        enough (e.g., "wrench" labeled as "Oil Filter Wrench" passes).
   FAIL only if the object is clearly something completely different
        (e.g., labeled "Breadboard" but looks like an animal or fruit).
"""
        matches_domain_field = '"matches_domain": true|false,'
        matches_class_field = '"matches_class": true|false,'
        matches_domain_class_field = '"matches_domain_class": true|false,'
        failure_step_values = "null | 1 | 2 | 3 | 4 | 5 | 6"
        sim_ready_expr = "(is_single_object AND is_rigid_body AND is_complete AND no_floating_fragments AND is_recognizable AND matches_domain_class)"
    else:
        step6_block = ""
        matches_domain_field = ""
        matches_class_field = ""
        matches_domain_class_field = ""
        failure_step_values = "null | 1 | 2 | 3 | 4 | 5"
        sim_ready_expr = "(is_single_object AND is_rigid_body AND is_complete AND no_floating_fragments AND is_recognizable)"

    return f"""You are an expert 3D asset validator. The labeled metadata for this asset is:
  Domain: {domain}
  Class:  {class_name}

Examine the five orthographic views carefully, then provide a controlled description and answer the checks below in order. Each check is an independent boolean. Output STRICT JSON only.

PHASE 1 — CONTROLLED OBJECT DESCRIPTION
Provide a structured description in the exact JSON sub-object below. Keep each field concise.
  observed_category: what the object visually appears to be (1-3 words)
  material_guess:    likely primary material (plastic/metal/wood/glass/ceramic/fabric/etc.)
  shape_summary:     brief overall shape (e.g., "spherical body with cylindrical lens")
  surface_quality:   hard / soft / taut-fabric / porous / liquid / strand-like
  key_features:      array of 1-3 distinguishing visual features
  {matches_domain_field}
  {matches_class_field}

PHASE 2 — RUBRIC (apply each check in order; cite visual evidence in reasoning)

Step 1. is_single_object (bool)
   PASS if the mesh depicts ONE connected object or connected assembly.
        Connected parts (handle, lid on container, base, legs, frame) count as ONE.
   FAIL if there are 2+ COMPLETELY SEPARATE pieces with NO physical connection
        and a CLEAR GAP between them. Two parallel rods/blades that align in
        front/back but show a clear gap from left/right are separate pieces — FAIL.
   RULE: A lid, cap, or cover positioned directly above/on its container or
        bottle (even with a small gap for opening) is ONE object, not separate.

Step 2. is_rigid_body (bool)
   PASS if the mesh surface looks HARD, FLAT, SMOOTH, or TAUT as a static solid.
        Hinges, springs, mechanical joints, and moving parts DO NOT disqualify.
        A folding chair with a fabric seat is a static assembly — PASS.
        A flat rectangular slab (book, cutting board, tray, tablet) ALWAYS passes
        if its surface is smooth/hard, regardless of what material you might guess.
   FAIL only if the mesh itself simulates flowing/draped soft material:
        cloth with VISIBLE DRAPING FOLDS, rope in a loose coil, plush toy,
        liquid, hair, human/animal body scan. A folded blanket with soft rounded
        edges and fabric texture FAILS; a hard flat slab PASSES.

Step 3. is_complete (bool)
   PASS if the object silhouette is whole. Open tops, cutaways, seams between
        lid and body, slots, vents, and sensor holes are normal design features.
   FAIL if you see torn-out chunks, half-cut geometry, or a WIDE BAND of
        scrambled/jagged/noisy geometry crossing the object in MULTIPLE views.
        A single-view artifact is not enough; it must be corroborated.

Step 4. no_floating_fragments (bool)
   PASS if no stray pieces are detached from the main body.
   FAIL only if a piece is clearly floating with a LARGE, OBVIOUS gap (not a
        normal seam). A lid, cap, or cover positioned directly above/on its
        container or bottle is NOT a floating fragment — it is in normal open state.

Step 5. is_recognizable (bool)
   PASS if the object is a recognizable household / kitchen / desk / hand-held /
        tool / toy / small-furniture / electronics / personal-care / sports /
        automotive / laboratory / garden item, OR a natural object.
   FAIL if it is terrain, a building, an abstract sculpture, or unidentifiable.
{step6_block}
FINAL CONSISTENCY CHECK
   Compute sim_ready = {sim_ready_expr}.
   If sim_ready=false, failure_step must be the FIRST false boolean.
   Your reasoning must state which visual evidence justifies the first failure,
   or confirm all checks passed.

Return STRICT JSON:
{{
  "object_description": {{
    "observed_category": "<string>",
    "material_guess": "<string>",
    "shape_summary": "<string>",
    "surface_quality": "<string>",
    "key_features": ["<feature1>", "<feature2>"],
    {matches_domain_field}
    {matches_class_field}
  }},
  "is_single_object":      true|false,
  "is_rigid_body":         true|false,
  "is_complete":           true|false,
  "no_floating_fragments": true|false,
  "is_recognizable":       true|false,
  {matches_domain_class_field}
  "object_guess":          "<class name>",
  "failure_step":          {failure_step_values},
  "reasoning":             "<visual evidence + verdict>",
  "sim_ready":             true|false
}}
"""
