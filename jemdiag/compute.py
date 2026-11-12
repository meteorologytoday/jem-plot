"""Area-weighted reductions and zonally-integrated ocean/atmosphere transports."""

import numpy as np
import xarray as xr

from . import grid

EARTH_RADIUS_M = 6_371_000.0  # mean Earth radius (m)

# Reference seawater properties used to convert v*T transport into watts for
# meridional heat transport. jem does not report these as run metadata, so
# fixed standard oceanographic values are used (consistent with common
# ocean-model defaults, e.g. MOM/NEMO); real seawater density/heat capacity
# vary with temperature and salinity, so this is an approximation.
RHO_SEAWATER = 1025.0  # kg/m^3
CP_SEAWATER = 3990.0  # J/(kg K)

# jem's atm component is SPEEDY physics (jcm.physics.speedy): a fixed, 7- or
# 8-layer sigma coordinate. Values below match jcm.physics.speedy.
# physical_constants.SIGMA_LAYER_BOUNDARIES and jcm.constants (p0/grav/cpd).
# Layer thickness (dsigma) is NOT recoverable from consecutive full-level
# sigma values alone -- verified against data/long_run's 8-level `level`
# coordinate, whose spacing does not match the half-level table's diffs --
# so the half-level boundaries are reproduced here rather than approximated.
SPEEDY_SIGMA_HALF_LEVELS = {
    7: np.array([0.0, 0.14, 0.26, 0.42, 0.6, 0.77, 0.9, 1.0]),
    8: np.array([0.0, 0.05, 0.14, 0.26, 0.42, 0.6, 0.77, 0.9, 1.0]),
}
P0_PA = 1.0e5  # reference pressure (Pa), for normalized_surface_pressure -> Pa
GRAV = 9.81  # m/s^2
CPD = 1004.64  # dry-air specific heat at constant pressure, J/(kg K)


def latitude_weights(lat_values):
    """Return ``cos(latitude)`` weights (latitude in degrees)."""
    return np.cos(np.deg2rad(np.asarray(lat_values, dtype=float)))


def weighted_mean(da, dims):
    """Mean of ``da`` over ``dims``.

    When the meridional dimension is reduced and has real (non-synthetic)
    coordinate values, the reduction is weighted by ``cos(latitude)`` to
    approximate an area-weighted average. Otherwise a plain mean is used.
    """
    _, y_dim = grid.horizontal_dims(da)
    if y_dim in dims and not da[y_dim].attrs.get("jemdiag_synthetic", False):
        weights = xr.DataArray(
            latitude_weights(da[y_dim].values),
            dims=[y_dim],
            coords={y_dim: da[y_dim]},
        )
        return da.weighted(weights).mean(dim=dims, keep_attrs=True)
    return da.mean(dim=dims, keep_attrs=True)


def _uniform_spacing(coord_values, dim_name):
    """Return the (constant) spacing of a 1D coordinate, or raise if irregular.

    Zonal-integral diagnostics below assume a regular longitude grid so that
    a single ``dx(lat)`` can be used for every column; jem's ocn grid is
    regular in longitude (verified against ``data/long_run``), but this is
    checked explicitly rather than assumed silently.
    """
    values = np.asarray(coord_values, dtype=float)
    if values.size < 2:
        raise ValueError(f"Need at least two '{dim_name}' points to infer grid spacing")
    diffs = np.diff(values)
    spacing = diffs[0]
    if not np.allclose(diffs, spacing, rtol=1e-6):
        raise ValueError(
            f"'{dim_name}' coordinate is not uniformly spaced; zonal-integral "
            "diagnostics (overturning streamfunction, heat transport) require "
            "a regular grid"
        )
    return spacing


def zonal_cell_width(lon_values, lat_values):
    """Physical zonal cell width in meters at each latitude: ``R*cos(lat)*dlon``.

    Assumes a regular (uniformly-spaced) longitude grid in degrees.
    """
    dlon_rad = np.deg2rad(_uniform_spacing(lon_values, "lon"))
    lat_rad = np.deg2rad(np.asarray(lat_values, dtype=float))
    return EARTH_RADIUS_M * np.cos(lat_rad) * dlon_rad


def overturning_streamfunction(v, dzt, mask, x_dim, y_dim, z_dim):
    """Meridional overturning streamfunction psi(depth, lat), in Sverdrups.

    ``v`` is meridional velocity (m/s) with dims including *x_dim*, *y_dim*,
    *z_dim* (already reduced over time). ``dzt`` is layer thickness (m)
    along *z_dim* only; ``mask`` is the land-sea mask (1=ocean, 0=land)
    sharing *v*'s dims, applied so land cells never contribute regardless of
    their raw velocity value.

    jem's ocn depth index runs from the seafloor (index 0, thickest layers)
    to the surface (last index, thinnest layers) — verified against
    ``data/long_run`` by matching the index-14 temperature profile to sea
    surface temperature.

    By the standard MOC sign convention (e.g. RAPID/AMOC diagnostics), psi
    is defined by integrating downward from the surface, so a
    poleward-flowing surface branch plus an equatorward-flowing deep branch
    (the classic thermohaline "conveyor belt" cell) comes out positive.
    Since the full water column's net transport is ~0 by mass conservation
    (verified against ``data/long_run``, holding at every latitude to
    within floating-point noise), integrating downward from the surface is
    equivalent to the negative of a cumulative sum from the seafloor (index
    0) upward — that negation is what's used here, since it's simplest and
    preserves the ~0 surface value exactly.
    """
    dx = zonal_cell_width(v[x_dim].values, v[y_dim].values)
    dx_da = xr.DataArray(dx, dims=[y_dim], coords={y_dim: v[y_dim]})
    transport = (v * mask * dx_da * dzt).sum(dim=x_dim)  # m^3/s per layer, dims (y_dim, z_dim)
    psi = -transport.cumsum(dim=z_dim) / 1.0e6  # Sverdrups
    return psi.transpose(z_dim, y_dim)


def meridional_heat_transport(v, temp, dzt, mask, x_dim, y_dim, z_dim):
    """Meridional ocean heat transport MHT(lat), in petawatts.

    MHT = rho_seawater * cp_seawater * (zonal & vertical integral of v*T).
    ``temp`` is potential temperature in degC (jem convention); using degC
    rather than Kelvin is standard practice for this diagnostic since the
    net volume transport across a latitude circle is ~0, so the constant
    Kelvin offset does not affect the result.
    """
    dx = zonal_cell_width(v[x_dim].values, v[y_dim].values)
    dx_da = xr.DataArray(dx, dims=[y_dim], coords={y_dim: v[y_dim]})
    vt_flux = (v * temp * mask * dx_da * dzt).sum(dim=[x_dim, z_dim])  # degC * m^3/s
    mht_watts = RHO_SEAWATER * CP_SEAWATER * vt_flux
    return mht_watts / 1.0e15  # petawatts


def sigma_layer_thickness(n_levels, z_dim, z_coord):
    """Return SPEEDY's sigma layer thickness (dsigma) as an ``xr.DataArray``.

    Ordered surface-first (index 0 = near-surface, last = TOA), matching
    jem's ``level`` coordinate — see ``SPEEDY_SIGMA_HALF_LEVELS`` (defined
    TOA-first, the order jcm itself uses).
    """
    if n_levels not in SPEEDY_SIGMA_HALF_LEVELS:
        raise ValueError(
            f"No known SPEEDY sigma half-levels for {n_levels} atm levels; "
            f"expected one of {sorted(SPEEDY_SIGMA_HALF_LEVELS)}"
        )
    half_levels = SPEEDY_SIGMA_HALF_LEVELS[n_levels]
    dsigma = np.diff(half_levels)[::-1]  # TOA-first -> surface-first
    return xr.DataArray(dsigma, dims=[z_dim], coords={z_dim: z_coord})


def atmosphere_heat_transport(v, temp, normalized_surface_pressure, x_dim, y_dim, z_dim):
    """Meridional atmospheric (sensible) heat transport AHT(lat), in petawatts.

    AHT = (c_pd / g) * (zonal & pressure-weighted-vertical integral of v*T).
    Layer pressure thickness ``dp = dsigma * ps`` follows from the
    hydrostatic relation (dp/g = mass per unit area of that layer), with
    ``ps = normalized_surface_pressure * P0_PA``. Unlike
    :func:`meridional_heat_transport`, no land-sea mask applies -- the
    atmosphere covers every column. As there, absolute temperature (K,
    jem's native units) is used directly: the net column-integrated mass
    flux across a latitude circle is ~0 (mass conservation), so a constant
    offset in temperature would not bias the result.
    """
    dx = zonal_cell_width(v[x_dim].values, v[y_dim].values)
    dx_da = xr.DataArray(dx, dims=[y_dim], coords={y_dim: v[y_dim]})
    dsigma_da = sigma_layer_thickness(v.sizes[z_dim], z_dim, v[z_dim])
    dp = dsigma_da * normalized_surface_pressure * P0_PA  # Pa, dims (z_dim, x_dim, y_dim)
    vt_flux = (v * temp * dx_da * dp).sum(dim=[x_dim, z_dim])  # K * Pa * m^2/s
    aht_watts = (CPD / GRAV) * vt_flux
    return aht_watts / 1.0e15  # petawatts


def gaussian_latitude_weights(n_lat):
    """Return Gauss-Legendre quadrature ``(latitude_degrees, weights)``.

    jem's atm component is a spectral model (SPEEDY dynamics via
    ``dinosaur``) whose nodal latitude grid is an exact Gauss-Legendre
    quadrature grid in sin(latitude) -- verified against
    ``data/long_run``'s 48-point ``lat`` coordinate, which matches
    ``numpy.polynomial.legendre.leggauss(48)`` to within 1e-14 degrees.
    Weights sum to 2 over the [-1, 1] domain; each latitude row's zonal
    band area is therefore exactly ``2*pi*R^2*weight`` (summing to the
    full sphere's ``4*pi*R^2`` over all rows) -- not an approximation, and
    measurably different from using consecutive latitude differences,
    since this grid's spacing is non-uniform (denser toward the poles).
    """
    nodes, weights = np.polynomial.legendre.leggauss(n_lat)
    return np.degrees(np.arcsin(nodes)), weights


def atmosphere_heat_transport_radiative(toa_net, surface_net, x_dim, y_dim):
    """Meridional atmospheric heat transport AHT(lat), in petawatts, via the
    classical top-of-atmosphere/surface radiative-residual ("indirect")
    energy-balance method (e.g. Trenberth & Caron 2001).

    ``toa_net``/``surface_net`` are both net-downward fluxes (W/m^2) at the
    top of atmosphere and the surface, respectively. Their difference is
    the atmosphere's own net radiative+turbulent energy source at each
    latitude (whatever arrives at the TOA either stays in the atmosphere
    or continues down to the surface, so TOA-minus-surface isolates what
    the atmosphere itself retains); in a steady state that must be carried
    away as meridional transport, so cumulatively integrating it from the
    south pole northward gives AHT(lat) directly. AHT is exactly 0 at the
    south pole (an empty integral) and should also come out ~0 at the
    north pole if the global-mean TOA and surface fluxes balance -- the
    same style of boundary check as :func:`overturning_streamfunction`'s
    seafloor/surface, just in latitude rather than depth.

    Unlike :func:`atmosphere_heat_transport` (time-averaged v*T, capturing
    only the mean-circulation transport), this needs no wind field at all
    and captures the *total* implied transport -- mean circulation plus
    transient eddies -- which is the traditional reason this method is
    used for observationally-based heat transport estimates.
    """
    net_source = (toa_net - surface_net).mean(dim=x_dim)  # zonal mean, W/m^2, dims (y_dim,)
    _, weights = gaussian_latitude_weights(net_source.sizes[y_dim])
    band_area = 2.0 * np.pi * EARTH_RADIUS_M**2 * xr.DataArray(
        weights, dims=[y_dim], coords={y_dim: net_source[y_dim]}
    )  # m^2, exact zonal-band area for this latitude row
    aht_watts = (net_source * band_area).cumsum(dim=y_dim)
    return aht_watts / 1.0e15  # petawatts
