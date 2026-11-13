"""Land (``lnd``) diagnostic plots -- see :mod:`.surface` for the diag_types."""

from jemdiag.diagnostics.surface import make_component

run, HANDLERS = make_component("lnd", "Land")
