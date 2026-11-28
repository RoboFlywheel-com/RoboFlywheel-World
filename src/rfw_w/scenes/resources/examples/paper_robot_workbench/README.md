# Curated workbench visualization

This optional configuration selects source assets explicitly. It is a visualization input,
not a production SceneSpec or a robot task certificate. Source packages and Blender/Isaac
runtimes must be provided outside the installed package.

Use `python -m rfw_w.scenes.visualization.collect_case_assets --help`,
`python -m rfw_w.scenes.visualization.build_paper_case --help` and the corresponding
render/audit commands to inspect their explicit input options. Each run needs a fresh directory.
The constructed workbench is labelled separately from source assets. Reusing a shell requires
identical room geometry and openings; placement and validation are recomputed.

Packaging excludes local logs. Materials, poses, cameras and source fingerprints remain in
the local evidence; static rendering does not validate manipulation or soft-body dynamics.
