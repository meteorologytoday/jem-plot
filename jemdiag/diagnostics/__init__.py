"""Per-component diagnostic plot modules.

Each module exposes a ``HANDLERS`` dict mapping ``diag_type`` name to a
handler function, and a ``run(diag_type, ds, config, output_dir)``
dispatcher. Components silently skip diag_types they don't support.
"""
