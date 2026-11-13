"""
Load and merge the two config files a diagnostics run is built from.

A run is described by two YAML files with separate concerns:

* **Simulation config** (``config/sim/*.yaml``) -- *what was run*: the
  ``experiment`` identity and the ``data`` location of its output. It may
  also carry an optional ``diagnostics`` block of run-specific overrides
  (e.g. the ``time`` window worth analysing, or ``coastlines`` for a run
  with real Earth topography), so one plot config can serve every run.
* **Plot config** (``config/plot/*.yaml``) -- *how to plot it*: ``grids``,
  ``diagnostics`` and ``output``. It must not describe a simulation.

:func:`load_config` merges them into the single dict the rest of the
package consumes, with the same top-level layout as before the split
(``experiment``/``data``/``grids``/``diagnostics``/``output``), so no other
module needs to know there are two files. The simulation's ``diagnostics``
overrides are deep-merged over the plot config's: mappings merge key by
key, while any other value (scalar or list) replaces the plot config's
value wholesale -- a list is never concatenated.
"""

import copy
from pathlib import Path

import yaml

SIM_REQUIRED_SECTIONS = ("experiment", "data")
SIM_OPTIONAL_SECTIONS = ("diagnostics",)
PLOT_REQUIRED_SECTIONS = ("diagnostics", "output")
PLOT_OPTIONAL_SECTIONS = ("grids",)


def _read_yaml_mapping(path: Path, kind: str) -> dict:
    """Read *path* as a YAML mapping; *kind* names the file in errors."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{kind} config file not found: {path}")
    with path.open() as fh:
        content = yaml.safe_load(fh)
    if content is None:
        content = {}
    if not isinstance(content, dict):
        raise ValueError(
            f"{kind} config {path} must be a YAML mapping, "
            f"got {type(content).__name__}"
        )
    return content


def _check_sections(content: dict, path: Path, kind: str, required, optional) -> None:
    """Reject missing required sections and any section outside the schema.

    Strict on purpose: a ``data`` block left in a plot config (or a
    ``grids`` block in a simulation config) would otherwise be silently
    dropped or silently win, which is exactly the mixing the split exists
    to prevent.
    """
    unknown = sorted(set(content) - set(required) - set(optional))
    if unknown:
        raise ValueError(
            f"{kind} config {path} has unexpected section(s) {unknown}; "
            f"allowed: {sorted([*required, *optional])}"
        )
    missing = [name for name in required if name not in content]
    if missing:
        raise ValueError(f"{kind} config {path} is missing section(s) {missing}")
    for name in content:
        if not isinstance(content[name], dict):
            raise ValueError(
                f"{kind} config {path}: section `{name}` must be a mapping, "
                f"got {type(content[name]).__name__}"
            )


def deep_merge(base: dict, override: dict) -> dict:
    """Return a new dict: *override* merged recursively over *base*.

    Nested mappings merge key by key; any other value in *override*
    (including a list) replaces *base*'s. Neither input is modified.
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(sim_path, plot_path) -> dict:
    """Load a simulation config and a plot config into one merged config.

    See this module's docstring for the schema of each file and the merge
    rules. Raises ``FileNotFoundError`` for a missing file and
    ``ValueError`` for a file that doesn't follow its schema.
    """
    sim_path, plot_path = Path(sim_path), Path(plot_path)
    sim = _read_yaml_mapping(sim_path, "Simulation")
    plot = _read_yaml_mapping(plot_path, "Plot")
    _check_sections(sim, sim_path, "Simulation", SIM_REQUIRED_SECTIONS, SIM_OPTIONAL_SECTIONS)
    _check_sections(plot, plot_path, "Plot", PLOT_REQUIRED_SECTIONS, PLOT_OPTIONAL_SECTIONS)

    for section, key in (("experiment", "name"), ("data", "run_dir")):
        if not sim[section].get(key):
            raise ValueError(
                f"Simulation config {sim_path} must set `{section}.{key}`"
            )

    config = copy.deepcopy(plot)
    config.setdefault("grids", {})
    config["diagnostics"] = deep_merge(plot["diagnostics"], sim.get("diagnostics", {}))
    config["experiment"] = copy.deepcopy(sim["experiment"])
    config["data"] = copy.deepcopy(sim["data"])
    return config
