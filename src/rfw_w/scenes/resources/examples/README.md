# Bundled scene inputs

Each named example contains `task.json` and optional supporting inputs. Load them through
`rfw_w.scenes.resources.data_path("examples/<name>/<file>")`; installed wheels include these files.
Runtime outputs belong in a fresh directory outside the package.

```bash
rfw-w scene generate --experiment office_pick_cup --output-dir /tmp/rfw-office-preview --compile-only
```

The three basic examples are `office_pick_cup`, `kitchen_fetch_jar` and
`bedroom_pick_alarm_clock`. The preview uses supplied JSON and makes no model requests.
Full generation additionally needs the explicitly configured asset providers and worker environments.

`task.targets` and hard `ON` relations determine target and support; entity roles, categories,
size bounds and hard-required flags drive binding and placement. Measured geometry admits assets.
Room kind is recorded but does not yet select a distinct zoning algorithm. Paper inputs are
optional visualization configurations and do not certify robot task execution.
