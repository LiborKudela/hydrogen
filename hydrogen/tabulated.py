"""Tabulated (spline-surrogate) medium: a fast, vectorised, backend-agnostic
property layer built on top of any source medium.

`TabulatedMedium` samples a SOURCE medium (`CoolPropMedium`, `FeosMedium`,
or anything implementing the small sampling protocol below) once over a
user-chosen `(p, h)` window and replaces every property lookup with smooth
spline evaluations:

* **fast**: ~2 us/point batched (vs ~50-100 us per HEOS flash) -- and truly
  vectorised: one call evaluates all pipe cells at once (every evaluator
  accepts scalars OR numpy arrays and carries the ``_hydrogen_vectorised``
  marker consumed by the model's per-template batch path).  On the LN2
  hammer benchmark the end-to-end speedup vs HEOS is ~7x per time step
  (the remainder is python/template overhead, not property cost).
* **Analytic, exactly consistent derivatives**: first AND second partials
  are the derivatives *of the interpolant itself*, so values and Jacobian
  never disagree (no finite-difference noise) -- which is precisely what
  an implicit Newton solve wants.
* **Dome-safe by construction**: the two-phase region is NOT spanned by a
  2-D surface (the thing that breaks CoolProp's own BICUBIC tables).
  Instead the saturation line is represented by 1-D splines and the
  single-phase surfaces are fitted on dome-conforming mapped coordinates:

      liquid:  sigma = (h - h_lo) / (h_liq(p) - h_lo)      in [0, 1]
      vapor:   sigma = (h - h_vap(p)) / (h_hi - h_vap(p))  in [0, 1]

  Inside the dome the properties follow the exact homogeneous-equilibrium
  mixture rules driven by the saturation splines (rho from mass-weighted
  specific volume, T = T_sat(p), linear-in-quality transport).  VALUES are
  piecewise-exact across the saturation lines (both sides sample the same
  source, so they are continuous); the one-sided FIRST derivatives are
  blended with a C^2 quintic smoothstep over a ``blend_width`` band in
  enthalpy -- the analytic analogue of the finite-difference HEM smoothing
  in `CoolPropMedium`.  The ``*_ph_hem`` functions are therefore the SAME
  functions as the single-phase ones (the tabulated surface *is* the
  smooth HEM surface).

Backend protocol (what a source medium must provide):

* ``medium`` (fluid name, used for the lambdify module names -- keeping the
  source's name makes the tabulated medium a drop-in replacement whose
  equation templates and disk-cached lambdas are reused unchanged);
* scalar value evaluators ``eval_rho_ph``, ``eval_T_ph``, ``eval_mu_ph``,
  ``eval_k_ph``, ``eval_s_ph`` (batch variants are used when present);
* ``eval_h_pT`` + partials (kept DELEGATED to the source -- boundary-node
  only, never hot);
* ``sample_saturation(p_array)`` -- only needed when the window intersects
  the two-phase dome; returns the saturation-line arrays (see
  `CoolPropMedium.sample_saturation`).

Outside the window, evaluation clamps to the window edge and extends
linearly (C^1), so a stray Newton iterate gets a finite, monotone pull
back instead of polynomial-extrapolation garbage.

The sampled tables are cached on disk (`~/.cache/hydrogen`, same location
and env overrides as the lambdify cache) keyed by fluid/backend/window/
resolution, so the one-time sampling cost (~n_p*n_h source flashes) is
paid once per configuration.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import sympy as sp
from scipy.interpolate import CubicSpline, RectBivariateSpline

from .caching import lambda_cache_default_dir
from .medium import get_symbolic_property_function

__all__ = ["TabulatedMedium"]

#: Properties tabulated as 2-D surfaces.  `order` is the highest derivative
#: exposed through the lambdify modules (rho needs consistent SECOND partials
#: because the primitive dynamic levels differentiate `drho/dp` again in the
#: Jacobian; the others only ever appear under a single derivative).
_PROPS = {"rho": 2, "T": 1, "mu": 1, "k": 1, "s": 1}

#: Saturation-line quantities required from `sample_saturation` (all arrays
#: over the pressure grid).
_SAT_KEYS = ("h_l", "h_v", "T_sat", "rho_l", "rho_v", "mu_l", "mu_v",
             "k_l", "k_v", "s_l", "s_v")

_TABLE_CACHE_VERSION = 1


def _quintic_smoothstep(t):
    """C^2 smoothstep q(t) with q(0)=0, q(1)=1 and zero first/second
    derivatives at both ends; returns (q, q', q'') with t clamped to [0,1]."""
    tc = np.clip(t, 0.0, 1.0)
    q = tc ** 3 * (10.0 + tc * (-15.0 + 6.0 * tc))
    qp = 30.0 * tc ** 2 * (1.0 + tc * (-2.0 + tc))
    qpp = 60.0 * tc * (1.0 + tc * (-3.0 + 2.0 * tc))
    return q, qp, qpp


def _join(A, B, S, use_B):
    """Join two region evaluations (6-tuples ``(f, fp, fh, fpp, fph, fhh)``)
    at a saturation edge.

    * VALUE is piecewise-exact: ``B`` where `use_B`, else ``A``.  Property
      values are continuous across the edge (both regions are sampled from
      the same source up to the saturation line), only their slope kinks.
    * FIRST derivatives are S-blended across the band, which is precisely
      the analytic analogue of `CoolPropMedium`'s finite-difference HEM
      smoothing: a central difference of a continuous, kinked value equals a
      smeared blend of the one-sided slopes.  This keeps values EXACT
      everywhere while giving Newton a continuous Jacobian through the dome
      edge.
    * SECOND derivatives are the exact derivatives OF the blended first
      derivatives (product rule in S), so the second-order entries stay
      consistent with the first-order ones.
    """
    a, ap, ah, app, aph, ahh = A
    b, bp, bh, bpp, bph, bhh = B
    s, sp_, sh, spp, sph, shh = S
    f = np.where(use_B, b, a)
    fp = ap + s * (bp - ap)
    fh = ah + s * (bh - ah)
    fpp = app + s * (bpp - app) + sp_ * (bp - ap)
    # `ddrho_ph_dp_dh` and `ddrho_ph_dh_dp` bind to the same evaluator, so
    # use the symmetrised cross term.
    fph = (aph + s * (bph - aph)
           + 0.5 * (sh * (bp - ap) + sp_ * (bh - ah)))
    fhh = ahh + s * (bhh - ahh) + sh * (bh - ah)
    return f, fp, fh, fpp, fph, fhh


class _Sat1D:
    """Saturation-line quantity as a cubic spline of pressure, evaluated
    together with its first and second derivatives in a single pass (one
    shared `searchsorted` + Horner instead of three `PPoly` calls -- these
    evaluations sit on the hot path of every dome-window property lookup)."""

    def __init__(self, p_grid, values):
        cs = CubicSpline(p_grid, values, extrapolate=True)
        self._x = cs.x
        self._c = np.ascontiguousarray(cs.c)     # (4, n-1), highest power first

    def __call__(self, p):
        p = np.asarray(p, dtype=float)
        i = np.clip(np.searchsorted(self._x, p) - 1, 0, self._x.size - 2)
        t = p - self._x[i]
        c3, c2, c1, c0 = (self._c[0, i], self._c[1, i],
                          self._c[2, i], self._c[3, i])
        f = ((c3 * t + c2) * t + c1) * t + c0
        f1 = (3.0 * c3 * t + 2.0 * c2) * t + c1
        f2 = 6.0 * c3 * t + 2.0 * c2
        return f, f1, f2


class _Bicubic2D:
    """Tensor-product cubic spline stored as per-cell bicubic patch
    coefficients.

    Fitting still uses FITPACK (`RectBivariateSpline`, s=0), but the fitted
    surface is converted EXACTLY to monomial patches: within every data cell
    the C^2 spline is a single bicubic polynomial, which is uniquely
    determined by (f, fx, fy, fxy) at the cell corners -- all four read off
    the fitted spline on the grid.  Evaluation is then one `searchsorted` +
    a handful of einsums returning the value and ALL five derivatives at
    once, ~6x faster than six separate FITPACK `parder` calls (the hot path
    of every Newton iteration).
    """

    def __init__(self, x, y, Z):
        kx = min(3, len(x) - 1)
        ky = min(3, len(y) - 1)
        spl = RectBivariateSpline(x, y, Z, kx=kx, ky=ky, s=0)
        self._x = np.asarray(x, dtype=float)
        self._y = np.asarray(y, dtype=float)
        if kx < 3 or ky < 3:
            # Tiny grids: keep the FITPACK object (cold path, tests only).
            self._spl = spl
            self._A = None
            return
        self._spl = None
        f = spl(x, y)
        fx = spl(x, y, dx=1)
        fy = spl(x, y, dy=1)
        fxy = spl(x, y, dx=1, dy=1)
        nx1, ny1 = len(x) - 1, len(y) - 1
        HX = np.diff(self._x)[:, None]                  # (nx-1, 1)
        HY = np.diff(self._y)[None, :]                  # (1, ny-1)
        # Per-cell Hermite data on the unit square (derivatives scaled by the
        # cell widths): D[i,j] = [[f00, f01, fy00, fy01],
        #                         [f10, f11, fy10, fy11],
        #                         [fx00, fx01, fxy00, fxy01],
        #                         [fx10, fx11, fxy10, fxy11]]
        D = np.empty((nx1, ny1, 4, 4))
        D[..., 0, 0] = f[:-1, :-1];   D[..., 0, 1] = f[:-1, 1:]
        D[..., 1, 0] = f[1:, :-1];    D[..., 1, 1] = f[1:, 1:]
        D[..., 0, 2] = fy[:-1, :-1] * HY;  D[..., 0, 3] = fy[:-1, 1:] * HY
        D[..., 1, 2] = fy[1:, :-1] * HY;   D[..., 1, 3] = fy[1:, 1:] * HY
        D[..., 2, 0] = fx[:-1, :-1] * HX;  D[..., 2, 1] = fx[:-1, 1:] * HX
        D[..., 3, 0] = fx[1:, :-1] * HX;   D[..., 3, 1] = fx[1:, 1:] * HX
        D[..., 2, 2] = fxy[:-1, :-1] * HX * HY
        D[..., 2, 3] = fxy[:-1, 1:] * HX * HY
        D[..., 3, 2] = fxy[1:, :-1] * HX * HY
        D[..., 3, 3] = fxy[1:, 1:] * HX * HY
        # Hermite -> monomial: p(u) = [1,u,u^2,u^3] @ H @ (f0,f1,d0,d1)^T
        H = np.array([[1.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0],
                      [-3.0, 3.0, -2.0, -1.0],
                      [2.0, -2.0, 1.0, 1.0]])
        self._A = np.einsum("mk,ijkl,nl->ijmn", H, D, H)
        self._A = np.ascontiguousarray(self._A)

    def ev_all(self, X, Y, order=2):
        """Value and derivatives (f, fx, fy, fxx, fxy, fyy) at (X, Y)."""
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        if self._A is None:                              # tiny-grid fallback
            ev = self._spl.ev
            f = ev(X, Y)
            fx = ev(X, Y, dx=1)
            fy = ev(X, Y, dy=1)
            if order < 2:
                z = np.zeros_like(f)
                return f, fx, fy, z, z, z
            return (f, fx, fy, ev(X, Y, dx=2), ev(X, Y, dx=1, dy=1),
                    ev(X, Y, dy=2))
        x, y = self._x, self._y
        i = np.clip(np.searchsorted(x, X) - 1, 0, x.size - 2)
        j = np.clip(np.searchsorted(y, Y) - 1, 0, y.size - 2)
        hx = x[i + 1] - x[i]
        hy = y[j + 1] - y[j]
        u = (X - x[i]) / hx
        v = (Y - y[j]) / hy
        A = self._A[i, j]                                # (...,4,4)
        one = np.ones_like(u)
        zero = np.zeros_like(u)
        U = np.stack([one, u, u * u, u ** 3], axis=-1)
        V = np.stack([one, v, v * v, v ** 3], axis=-1)
        Du = np.stack([zero, one, 2.0 * u, 3.0 * u * u], axis=-1)
        Dv = np.stack([zero, one, 2.0 * v, 3.0 * v * v], axis=-1)
        AV = np.einsum("...mn,...n->...m", A, V)
        f = np.einsum("...m,...m->...", U, AV)
        fx = np.einsum("...m,...m->...", Du, AV) / hx
        ADv = np.einsum("...mn,...n->...m", A, Dv)
        fy = np.einsum("...m,...m->...", U, ADv) / hy
        if order < 2:
            return f, fx, fy, zero, zero, zero
        Duu = np.stack([zero, zero, 2.0 * one, 6.0 * u], axis=-1)
        Dvv = np.stack([zero, zero, 2.0 * one, 6.0 * v], axis=-1)
        fxx = np.einsum("...m,...m->...", Duu, AV) / hx ** 2
        fxy = np.einsum("...m,...m->...", Du, ADv) / (hx * hy)
        AVv = np.einsum("...mn,...n->...m", A, Dvv)
        fyy = np.einsum("...m,...m->...", U, AVv) / hy ** 2
        return f, fx, fy, fxx, fxy, fyy


class _MappedSurface:
    """Single-phase property surface on dome-conforming mapped coordinates.

    ``kind="plain"``:  F(p, h) directly on the rectangular window.
    ``kind="liquid"``: F(p, sigma), sigma = (h - h_lo) / (h_l(p) - h_lo).
    ``kind="vapor"``:  F(p, sigma), sigma = (h - h_v(p)) / (h_hi - h_v(p)).

    Evaluation returns the 6-tuple (f, fp, fh, fpp, fph, fhh) with the full
    chain rule through the coordinate mapping, so the partials are exact
    derivatives of the interpolant.
    """

    def __init__(self, p_grid, sigma_grid, values, kind, h_lo, h_hi,
                 sat_a=None, sat_b=None):
        self._spl = _Bicubic2D(p_grid, sigma_grid, values)
        self._kind = kind
        self._h_lo = float(h_lo)
        self._h_hi = float(h_hi)
        # `sat_a`: the h-offset spline (h_l for liquid upper edge, h_v for
        # vapor lower edge); only the relevant one is used per kind.
        self._sat = sat_a if kind == "liquid" else sat_b

    def _sigma(self, p, h, sat_t=None):
        """sigma and its mapping derivatives; returns
        (sigma, s_p, s_h, s_pp, s_ph) -- s_hh is identically zero.
        `sat_t` optionally passes a precomputed saturation-edge tuple."""
        if self._kind == "plain":
            one = np.ones_like(p)
            zero = np.zeros_like(p)
            return h, zero, one, zero, zero
        if self._kind == "liquid":
            hl, hl1, hl2 = sat_t if sat_t is not None else self._sat(p)
            a, a1, a2 = self._h_lo, 0.0, 0.0
            d = hl - self._h_lo
            d1, d2 = hl1, hl2
        else:                                   # vapor
            hv, hv1, hv2 = sat_t if sat_t is not None else self._sat(p)
            a, a1, a2 = hv, hv1, hv2
            d = self._h_hi - hv
            d1, d2 = -hv1, -hv2
        sig = (h - a) / d
        s_h = 1.0 / d
        s_p = -(a1 + sig * d1) / d
        s_ph = -d1 / d ** 2
        s_pp = -(a2 + 2.0 * s_p * d1 + sig * d2) / d
        return sig, s_p, s_h, s_pp, s_ph

    def eval(self, p, h, order=2, sat_t=None):
        sig, s_p, s_h, s_pp, s_ph = self._sigma(p, h, sat_t)
        # FITPACK CLAMPS out-of-range arguments, so the spline value would go
        # flat past sigma = 0/1 while the chain-rule derivatives kept their
        # interior slope -- inconsistent.  Evaluate at the clamped sigma and
        # extend linearly in the overshoot `e` ourselves (the overshoot only
        # occurs inside the dome-blend band, where the surface's weight is
        # already fading).
        if self._kind != "plain":
            sc = np.clip(sig, 0.0, 1.0)
            e = sig - sc
            extended = bool(np.any(e != 0.0))
        else:
            sc, e, extended = sig, 0.0, False
        F, Fp, Fs, Fpp, Fps, Fss = self._spl.ev_all(
            p, sc, order=2 if (order >= 2 or extended) else 1)
        if extended:
            F = F + Fs * e
            Fp = Fp + Fps * e
            Fs = Fs + Fss * e          # dF/dsigma of the extended surface
        fp = Fp + Fs * s_p
        fh = Fs * s_h
        if order < 2:
            zero = np.zeros_like(F)
            return F, fp, fh, zero, zero, zero
        # Second derivatives use the clamped-point curvature (exact where
        # e == 0; approximate in the small extension strip).
        fpp = Fpp + 2.0 * Fps * s_p + Fss * s_p ** 2 + Fs * s_pp
        fph = Fps * s_h + Fss * s_p * s_h + Fs * s_ph
        fhh = Fss * s_h ** 2
        return F, fp, fh, fpp, fph, fhh


class TabulatedMedium:
    """Spline-surrogate medium wrapping a slow source backend.

    Drop-in replacement for `CoolPropMedium` / `FeosMedium`: exposes the
    identical public surface (sympy-able property functions, scalar and
    batch ``eval_*`` evaluators, ``modules`` / ``batch_modules`` for
    lambdify, ``default_vars``) with all `(p, h)` properties served from
    smooth spline tables.  See the module docstring for the construction.

    Parameters
    ----------
    source :
        The medium to sample (kept for `h(p, T)` delegation, which is
        boundary-node-only and never hot).
    p_range, h_range :
        The `(min, max)` window the table covers.  Choose it to enclose the
        full transient envelope (peaks included) with some margin;
        evaluation outside is clamped + linearly extended (C^1).
    n_p, n_h :
        Grid resolution (default 160x160; bicubic error scales ~ h^4).
    two_phase :
        ``"auto"`` (default) samples the saturation line when the source
        provides ``sample_saturation`` and the dome intersects the window;
        ``True`` / ``False`` force the choice.
    blend_width :
        Enthalpy band [J/kg] over which the one-sided property SLOPES are
        smoothed across the saturation lines (values stay exact; same role
        as `CoolPropMedium.hem_fd_dh`).
    cache :
        Cache the sampled tables on disk (default True; location and
        disable/override via the ``HYDROGEN_LAMBDA_CACHE`` env var, shared
        with the lambdify cache).
    validate :
        After building, compare the surrogate against the source on an
        offset probe grid and warn if the relative error exceeds ~1e-3.
    """

    def __init__(self, source, p_range, h_range, n_p=160, n_h=160,
                 two_phase="auto", blend_width=2000.0, cache=True,
                 validate=True, disable_warnings=None):
        self.source = source
        self.medium = source.medium
        self.backend = f"TAB({getattr(source, 'backend', type(source).__name__)})"
        self.disable_warnings = (getattr(source, "disable_warnings", False)
                                 if disable_warnings is None
                                 else disable_warnings)
        self.p_min, self.p_max = (float(p_range[0]), float(p_range[1]))
        self.h_min, self.h_max = (float(h_range[0]), float(h_range[1]))
        if not (self.p_max > self.p_min and self.h_max > self.h_min):
            raise ValueError("TabulatedMedium: empty (p, h) window")
        self.n_p = int(n_p)
        self.n_h = int(n_h)
        self.blend_width = float(blend_width)

        # --- decide two-phase handling -------------------------------------
        can_sat = hasattr(source, "sample_saturation")
        if two_phase == "auto":
            self._two_phase = can_sat and self._dome_intersects_window()
        else:
            self._two_phase = bool(two_phase)
        if self._two_phase and not can_sat:
            raise ValueError(
                "TabulatedMedium(two_phase=True): the source medium does not "
                "implement `sample_saturation(p_array)`; implement it or "
                "restrict the window to a single-phase region "
                "(two_phase=False).")

        # --- build or load the tables ---------------------------------------
        data = self._load_cached_tables() if cache else None
        if data is None:
            data = self._sample_tables()
            if cache:
                self._store_cached_tables(data)
        self._build_splines(data)

        # Small LRU so the value + up-to-5 partial lookups a single template
        # makes at the SAME (p, h) pay the surface evaluation once.
        self._eval_cache = OrderedDict()
        self._eval_cache_size = 64

        if validate:
            self._validate()

        # --- symbolic property functions + lambdify modules -----------------
        self._init_symbolic_interface()

    # =====================================================================
    # construction helpers
    # =====================================================================

    @staticmethod
    def _chebyshev(n):
        """Chebyshev-extrema abscissae on [0, 1]: node density ~1/sqrt(x(1-x))
        concentrates resolution at both interval ends, where property
        curvature peaks (saturation line, near-critical corner, window
        edges).  Cuts the worst-corner interpolation error by ~2 orders of
        magnitude vs a uniform grid at equal n."""
        return 0.5 * (1.0 - np.cos(np.pi * np.arange(n) / (n - 1)))

    def _p_grid(self):
        return self.p_min + (self.p_max - self.p_min) * self._chebyshev(self.n_p)

    def _sigma_grid(self):
        return self._chebyshev(self.n_h)

    def _dome_intersects_window(self):
        """Probe the saturation line at a few pressures to see whether the
        dome overlaps the (p, h) window."""
        try:
            sat = self.source.sample_saturation(
                np.linspace(self.p_min, self.p_max, 12))
        except Exception:
            return False
        h_l, h_v = np.asarray(sat["h_l"]), np.asarray(sat["h_v"])
        ok = np.isfinite(h_l) & np.isfinite(h_v)
        if not ok.any():
            return False
        return bool(np.any((h_v[ok] > self.h_min) & (h_l[ok] < self.h_max)))

    #: Max points per batch source call while sampling the tables.  A source
    #: batch evaluator (e.g. `CoolPropMedium`) may allocate ONE heavyweight
    #: EoS state object per array element and keep them pooled -- fine for the
    #: model's per-template calls (a handful of pipe cells) but catastrophic
    #: for a full n_p*n_h sampling grid (32768 HEOS states ~ 22 GB, enough to
    #: freeze a machine).  Chunking bounds the TRANSIENT state pool to ~this
    #: many objects.  32 sits just above the memory floor (~300 MB, dominated
    #: by library imports): dropping it further saves little (the HEOS flash
    #: cost dominates the build, not the pool) but keeps a little batch width
    #: for sources whose batch path genuinely vectorises.
    _SAMPLE_BATCH_CHUNK = 32

    def _source_batch(self, name, p, h):
        """Value sampling through the source's batch evaluator when present,
        else a scalar loop; per-point failures become NaN (filled later).

        The batch path is chunked (`_SAMPLE_BATCH_CHUNK`) so a large sampling
        grid never forces the source to materialise one EoS state per grid
        point at once."""
        out = np.empty(p.shape, dtype=float)
        flat_p, flat_h, flat_o = p.ravel(), h.ravel(), out.ravel()
        n = flat_p.size
        batch = getattr(self.source, f"eval_{name}_ph_batch", None)
        if batch is not None:
            ok = True
            for s in range(0, n, self._SAMPLE_BATCH_CHUNK):
                e = min(s + self._SAMPLE_BATCH_CHUNK, n)
                try:
                    flat_o[s:e] = np.asarray(
                        batch(flat_p[s:e], flat_h[s:e]), dtype=float).ravel()
                except Exception:
                    ok = False
                    break
            if ok:
                return out
        scalar = getattr(self.source, f"eval_{name}_ph")
        for i in range(n):
            try:
                flat_o[i] = float(scalar(float(flat_p[i]), float(flat_h[i])))
            except Exception:
                flat_o[i] = np.nan
        return out

    @staticmethod
    def _fill_nans_along_h(vals):
        """Replace NaN samples by the nearest valid neighbour along the
        h-axis (per pressure row); rows with no valid sample raise."""
        bad_total = 0
        for row in vals:
            bad = ~np.isfinite(row)
            if not bad.any():
                continue
            if bad.all():
                raise RuntimeError(
                    "TabulatedMedium: an entire constant-pressure sample row "
                    "failed; shrink the (p, h) window to the EoS's valid "
                    "domain")
            idx = np.arange(row.size)
            row[bad] = np.interp(idx[bad], idx[~bad], row[~bad])
            bad_total += int(bad.sum())
        return bad_total

    def _sample_tables(self):
        """Sample the source over the mapped grids; returns a plain dict of
        arrays (picklable / npz-able).

        Sampling touches every grid point exactly once, so the source's
        batch-evaluator state pool (which keeps ``batch_state_pool_size``
        distinct chunks alive -- heavyweight EoS objects) offers no reuse and
        only inflates peak memory.  We pin it to a single chunk for the
        duration and release the pool afterwards."""
        prev_pool = getattr(self.source, "batch_state_pool_size", None)
        try:
            if prev_pool is not None:
                self.source.batch_state_pool_size = 1
            return self._sample_tables_impl()
        finally:
            if prev_pool is not None:
                self.source.batch_state_pool_size = prev_pool
            clear = getattr(self.source, "clear_cache", None)
            if callable(clear):
                clear()
            # `clear_cache()` empties the (p,h) batch cache dict but leaves the
            # FREE pool of reusable EoS state objects parked (one per chunk
            # element ~ 0.7 MB each for HEOS).  Sampling is one-shot, so drop
            # it too -- otherwise ~`_SAMPLE_BATCH_CHUNK` heavyweight states
            # stay resident for the life of the medium.
            free = getattr(self.source, "_batch_state_free_ph", None)
            if isinstance(free, list):
                free.clear()

    def _sample_tables_impl(self):
        pg = self._p_grid()
        sg = self._sigma_grid()
        data = {"p_grid": pg, "sigma_grid": sg,
                "two_phase": np.array(self._two_phase)}

        if self._two_phase:
            sat = self.source.sample_saturation(pg)
            h_l_raw = np.asarray(sat["h_l"], dtype=float)
            n_bad = int((~np.isfinite(h_l_raw)).sum())
            if n_bad > max(2, 0.02 * h_l_raw.size):
                raise ValueError(
                    f"TabulatedMedium({self.medium}): the saturation line "
                    f"could not be sampled at {n_bad}/{h_l_raw.size} grid "
                    f"pressures -- the window likely crosses the critical "
                    f"pressure.  Cap p_range below p_crit (or pass "
                    f"two_phase=False for a supercritical window).")
            for key in _SAT_KEYS:
                arr = np.asarray(sat[key], dtype=float)
                if arr.shape != pg.shape:
                    raise ValueError(
                        f"sample_saturation: '{key}' has shape {arr.shape}, "
                        f"expected {pg.shape}")
                bad = ~np.isfinite(arr)
                if bad.any():
                    idx = np.arange(arr.size)
                    arr = arr.copy()
                    arr[bad] = np.interp(idx[bad], idx[~bad], arr[~bad])
                data[f"sat_{key}"] = arr
            h_l, h_v = data["sat_h_l"], data["sat_h_v"]
            regions = []
            # A single-phase surface only exists where its mapped width stays
            # positive over the WHOLE pressure grid; a saturation edge that
            # crosses a window edge would make the sigma mapping degenerate.
            if np.all(h_l > self.h_min):
                regions.append(("liq", h_l))
            elif np.any(h_l > self.h_min):
                raise ValueError(
                    f"TabulatedMedium({self.medium}): the saturated-liquid "
                    f"line crosses h_range[0] inside the pressure window "
                    f"(h_l spans [{h_l.min():.4g}, {h_l.max():.4g}] J/kg, "
                    f"h_range[0]={self.h_min:.4g}).  Lower h_range[0] below "
                    f"min(h_l) so the liquid surface has positive width.")
            if np.all(h_v < self.h_max):
                regions.append(("vap", h_v))
            elif np.any(h_v < self.h_max):
                raise ValueError(
                    f"TabulatedMedium({self.medium}): the saturated-vapor "
                    f"line crosses h_range[1] inside the pressure window "
                    f"(h_v spans [{h_v.min():.4g}, {h_v.max():.4g}] J/kg, "
                    f"h_range[1]={self.h_max:.4g}).  Raise h_range[1] above "
                    f"max(h_v) (or cap it below min(h_v) for a liquid/dome "
                    f"window).")
            if not regions:
                raise ValueError(
                    f"TabulatedMedium({self.medium}): the whole (p, h) window "
                    f"lies inside the two-phase dome; enclose at least one "
                    f"single-phase region.")
        else:
            regions = [("all", None)]

        n_filled = 0
        for tag, edge in regions:
            if tag == "liq":
                H = self.h_min + sg[None, :] * (edge[:, None] - self.h_min)
            elif tag == "vap":
                H = edge[:, None] + sg[None, :] * (self.h_max - edge[:, None])
            else:
                H = np.broadcast_to(
                    self.h_min + sg[None, :] * (self.h_max - self.h_min),
                    (pg.size, sg.size)).copy()
            P = np.broadcast_to(pg[:, None], H.shape).copy()
            for prop in _PROPS:
                vals = self._source_batch(prop, P, H)
                n_filled += self._fill_nans_along_h(vals)
                data[f"{tag}_{prop}"] = vals
        if n_filled and not self.disable_warnings:
            print(f"TabulatedMedium({self.medium}): {n_filled} failed source "
                  f"samples filled by neighbour interpolation")
        return data

    def _build_splines(self, data):
        pg = np.asarray(data["p_grid"])
        sg = np.asarray(data["sigma_grid"])
        self._two_phase = bool(np.asarray(data["two_phase"]))
        self._sat = {}
        self._surf = {}
        if self._two_phase:
            for key in _SAT_KEYS:
                self._sat[key] = _Sat1D(pg, np.asarray(data[f"sat_{key}"]))
            sat_hl = self._sat["h_l"]
            sat_hv = self._sat["h_v"]
            self._has_liq = f"liq_{next(iter(_PROPS))}" in data
            self._has_vap = f"vap_{next(iter(_PROPS))}" in data
            for prop in _PROPS:
                if self._has_liq:
                    self._surf[f"liq_{prop}"] = _MappedSurface(
                        pg, sg, np.asarray(data[f"liq_{prop}"]), "liquid",
                        self.h_min, self.h_max, sat_a=sat_hl)
                if self._has_vap:
                    self._surf[f"vap_{prop}"] = _MappedSurface(
                        pg, sg, np.asarray(data[f"vap_{prop}"]), "vapor",
                        self.h_min, self.h_max, sat_b=sat_hv)
        else:
            self._has_liq = self._has_vap = False
            h_grid = self.h_min + sg * (self.h_max - self.h_min)
            for prop in _PROPS:
                self._surf[f"all_{prop}"] = _MappedSurface(
                    pg, h_grid, np.asarray(data[f"all_{prop}"]), "plain",
                    self.h_min, self.h_max)

    # --- disk cache ---------------------------------------------------------

    def _cache_key(self):
        payload = json.dumps({
            "v": _TABLE_CACHE_VERSION, "fluid": self.medium,
            "src": f"{type(self.source).__name__}:{getattr(self.source, 'backend', '')}",
            "p": [self.p_min, self.p_max], "h": [self.h_min, self.h_max],
            "n": [self.n_p, self.n_h], "two_phase": self._two_phase,
        }, sort_keys=True)
        return hashlib.md5(payload.encode()).hexdigest()

    def _cache_path(self):
        base = lambda_cache_default_dir()
        if base is None:
            return None
        return Path(base) / f"proptab_{self._cache_key()}.npz"

    def _load_cached_tables(self):
        path = self._cache_path()
        if path is None or not path.exists():
            return None
        try:
            with np.load(path) as z:
                return {k: z[k] for k in z.files}
        except Exception:
            return None

    def _store_cached_tables(self, data):
        path = self._cache_path()
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **data)
        except Exception:
            pass  # cache is best-effort (e.g. read-only sandbox)

    # --- validation -----------------------------------------------------------

    def _validate(self, n_probe=200, tol=2e-3):
        """Probe the surrogate against the source on random in-window points
        (biased away from the blend bands, where the surrogate INTENTIONALLY
        deviates to smooth the phase kink)."""
        rng = np.random.default_rng(1234)
        p = rng.uniform(self.p_min, self.p_max, n_probe)
        h = rng.uniform(self.h_min, self.h_max, n_probe)
        if self._two_phase:
            hl = self._sat["h_l"](p)[0]
            hv = self._sat["h_v"](p)[0]
            keep = (np.abs(h - hl) > 2 * self.blend_width) & \
                   (np.abs(h - hv) > 2 * self.blend_width)
            p, h = p[keep], h[keep]
        worst = {}
        for prop in ("rho", "T"):
            ref = self._source_batch(prop, p[:, None], h[:, None]).ravel()
            ok = np.isfinite(ref)
            if not ok.any():
                continue
            got = self._eval_prop(prop, p[ok], h[ok], order=0)[0]
            rel = np.abs(got - ref[ok]) / np.maximum(np.abs(ref[ok]), 1e-30)
            worst[prop] = float(rel.max())
        self.validation_max_rel_err = worst
        bad = {k: v for k, v in worst.items() if v > tol}
        if bad and not self.disable_warnings:
            print(f"TabulatedMedium({self.medium}) WARNING: surrogate error "
                  f"exceeds {tol:g}: " +
                  ", ".join(f"{k}: {v:.2e}" for k, v in bad.items()) +
                  "  (raise n_p/n_h or shrink the window)")

    # =====================================================================
    # evaluation core
    # =====================================================================

    def _blend_S(self, p, h, edge_t):
        """Smoothstep tuple (S, Sp, Sh, Spp, Sph, Shh) rising 0 -> 1 across a
        band of width `blend_width` CENTRED on the saturation edge (passed as
        a precomputed `(e, e', e'')` tuple); used to blend the one-sided
        FIRST derivatives (see `_join`)."""
        w = self.blend_width
        e, e1, e2 = edge_t
        t = (h - e) / w + 0.5
        q, qp, qpp = _quintic_smoothstep(t)
        t_h = 1.0 / w
        t_p = -e1 / w
        t_pp = -e2 / w
        S = q
        Sp = qp * t_p
        Sh = qp * t_h
        Spp = qpp * t_p ** 2 + qp * t_pp
        Sph = qpp * t_p * t_h
        Shh = qpp * t_h ** 2
        return S, Sp, Sh, Spp, Sph, Shh

    def _dome_eval(self, prop, p, h, hl_t, hv_t, order=2):
        """Homogeneous-equilibrium mixture rules driven by the saturation
        splines; exact analytic derivatives.  `hl_t` / `hv_t` are the
        precomputed `(h, h', h'')` saturation-edge tuples."""
        hl, hl1, hl2 = hl_t
        hv, hv1, hv2 = hv_t
        W = hv - hl
        W1 = hv1 - hl1
        W2 = hv2 - hl2
        # Clip the quality with a wide margin: points far outside the dome
        # still evaluate this branch (their contribution is masked in
        # `_join`), and an unclipped x << 0 could drive the mixture specific
        # volume through zero (1/v -> inf -> NaN after masking).  The clip
        # never activates inside the dome or the blend bands.
        x = np.clip((h - hl) / W, -0.1, 1.1)
        x_h = 1.0 / W
        x_p = -(hl1 + x * W1) / W
        x_ph = -W1 / W ** 2
        x_pp = -(hl2 + 2.0 * x_p * W1 + x * W2) / W

        if prop == "T":
            T, T1, T2 = self._sat["T_sat"](p)
            zero = np.zeros_like(T)
            return T, T1, zero, T2, zero, zero

        if prop == "rho":
            rl, rl1, rl2 = self._sat["rho_l"](p)
            rv, rv1, rv2 = self._sat["rho_v"](p)
            vl, vv = 1.0 / rl, 1.0 / rv
            vl1 = -rl1 / rl ** 2
            vv1 = -rv1 / rv ** 2
            vl2 = (2.0 * rl1 ** 2 / rl - rl2) / rl ** 2
            vv2 = (2.0 * rv1 ** 2 / rv - rv2) / rv ** 2
            dv = vv - vl
            dv1 = vv1 - vl1
            dv2 = vv2 - vl2
            v = vl + x * dv
            v_p = vl1 + x_p * dv + x * dv1
            v_h = x_h * dv
            v_pp = vl2 + x_pp * dv + 2.0 * x_p * dv1 + x * dv2
            v_ph = x_ph * dv + x_h * dv1
            v_hh = np.zeros_like(v)
            r = 1.0 / v
            r2 = r * r
            r3 = r2 * r
            f = r
            fp = -v_p * r2
            fh = -v_h * r2
            fpp = 2.0 * v_p ** 2 * r3 - v_pp * r2
            fph = 2.0 * v_p * v_h * r3 - v_ph * r2
            fhh = 2.0 * v_h ** 2 * r3 - v_hh * r2
            return f, fp, fh, fpp, fph, fhh

        # mu / k / s: linear-in-quality (mass-weighted) HEM convention.
        gl, gl1, gl2 = self._sat[f"{prop}_l"](p)
        gv, gv1, gv2 = self._sat[f"{prop}_v"](p)
        dg = gv - gl
        dg1 = gv1 - gl1
        dg2 = gv2 - gl2
        f = gl + x * dg
        fp = gl1 + x_p * dg + x * dg1
        fh = x_h * dg
        fpp = gl2 + x_pp * dg + 2.0 * x_p * dg1 + x * dg2
        fph = x_ph * dg + x_h * dg1
        fhh = np.zeros_like(f)
        return f, fp, fh, fpp, fph, fhh

    def _eval_prop_core(self, prop, p, h, order):
        """Blended in-window evaluation (p, h already clipped)."""
        if not self._two_phase:
            return self._surf[f"all_{prop}"].eval(p, h, order)

        # Saturation-edge tuples computed ONCE and shared by the sigma
        # mapping, the dome mixture rules, and the blend smoothstep.
        hl_t = self._sat["h_l"](p)
        hv_t = self._sat["h_v"](p)
        dome = self._dome_eval(prop, p, h, hl_t, hv_t, order)
        if self._has_liq:
            liq = self._surf[f"liq_{prop}"].eval(p, h, order, sat_t=hl_t)
            S = self._blend_S(p, h, hl_t)
            F = _join(liq, dome, S, use_B=(h > hl_t[0]))
        else:
            F = dome
        if self._has_vap:
            vap = self._surf[f"vap_{prop}"].eval(p, h, order, sat_t=hv_t)
            S = self._blend_S(p, h, hv_t)
            F = _join(F, vap, S, use_B=(h > hv_t[0]))
        return F

    #: Positive-definite properties get a LOG-linear (exponential) window
    #: extension: `f_edge * exp((fp*dp + fh*dh)/f_edge)` matches value and
    #: first derivatives at the window edge (C^1) but can never cross zero --
    #: essential because unseeded boundary components may sit VERY far outside
    #: the window (e.g. ambient gas vs a cryogenic-liquid table), where a
    #: linear extension would hand Newton a negative viscosity/density and a
    #: NaN friction factor.  Entropy stays linear (sign-indefinite).
    _POSITIVE_PROPS = frozenset({"rho", "T", "mu", "k"})

    def _eval_prop(self, prop, p, h, order=2):
        """Public evaluation with window clamping + C^1 extension."""
        p, h = np.broadcast_arrays(np.asarray(p, dtype=float),
                                   np.asarray(h, dtype=float))
        pc = np.clip(p, self.p_min, self.p_max)
        hc = np.clip(h, self.h_min, self.h_max)
        f, fp, fh, fpp, fph, fhh = self._eval_prop_core(prop, pc, hc,
                                                        max(order, 1))
        dp = p - pc
        dh = h - hc
        if np.any(dp != 0.0) or np.any(dh != 0.0):
            if prop in self._POSITIVE_PROPS:
                u = np.clip((fp * dp + fh * dh) / f, -40.0, 40.0)
                g = np.exp(u)
                f = f * g
                fp = fp * g
                fh = fh * g
                fpp = fpp * g
                fph = fph * g
                fhh = fhh * g
            else:
                f = f + fp * dp + fh * dh
        return f, fp, fh, fpp, fph, fhh

    def _eval_cached(self, prop, p, h):
        """LRU over the full 6-tuple so value + partial calls at the same
        (p, h) (typical within one template) evaluate the surfaces once."""
        p_arr = np.asarray(p, dtype=float)
        h_arr = np.asarray(h, dtype=float)
        key = (prop, p_arr.tobytes(), h_arr.tobytes())
        hit = self._eval_cache.get(key)
        if hit is not None:
            self._eval_cache.move_to_end(key)
            return hit
        res = self._eval_prop(prop, p_arr, h_arr, order=_PROPS[prop])
        self._eval_cache[key] = res
        if len(self._eval_cache) > self._eval_cache_size:
            self._eval_cache.popitem(last=False)
        return res

    # =====================================================================
    # public evaluator surface (mirrors CoolPropMedium)
    # =====================================================================

    def _make_eval(self, prop, comp):
        """Build a scalar-or-array evaluator returning component `comp` of
        the derivative tuple (0=f, 1=fp, 2=fh, 3=fpp, 4=fph, 5=fhh)."""
        def _eval(p, h):
            scalar = not (isinstance(p, np.ndarray) and np.ndim(p) > 0) and \
                     not (isinstance(h, np.ndarray) and np.ndim(h) > 0)
            res = self._eval_cached(prop, p, h)[comp]
            if scalar:
                return float(res.reshape(-1)[0]) if res.size else float(res)
            shape = np.broadcast_shapes(np.shape(p), np.shape(h))
            return np.broadcast_to(res, shape).copy() if res.shape != shape \
                else res
        _eval._hydrogen_vectorised = True
        return _eval

    def _init_symbolic_interface(self):
        med = self.medium
        src = self.source

        # (p, h) property evaluators from the tables.
        ev = {}
        for prop, order in _PROPS.items():
            ev[f"{prop}_ph"] = self._make_eval(prop, 0)
            ev[f"d{prop}_ph_dp"] = self._make_eval(prop, 1)
            ev[f"d{prop}_ph_dh"] = self._make_eval(prop, 2)
            if order >= 2:
                ev[f"dd{prop}_ph_dp_dp"] = self._make_eval(prop, 3)
                ev[f"dd{prop}_ph_dp_dh"] = self._make_eval(prop, 4)
                ev[f"dd{prop}_ph_dh_dh"] = self._make_eval(prop, 5)

        # Expose the standard `eval_*` names (used by seeding / calibration
        # code and by the symbolic-function plumbing below).
        self.eval_rho_ph = ev["rho_ph"]
        self.eval_drho_ph_dp = ev["drho_ph_dp"]
        self.eval_drho_ph_dh = ev["drho_ph_dh"]
        self.eval_d2rho_ph_dp2 = ev["ddrho_ph_dp_dp"]
        self.eval_d2rho_ph_dpdh = ev["ddrho_ph_dp_dh"]
        self.eval_d2rho_ph_dh2 = ev["ddrho_ph_dh_dh"]
        self.eval_T_ph = ev["T_ph"]
        self.eval_dT_ph_dp = ev["dT_ph_dp"]
        self.eval_dT_ph_dh = ev["dT_ph_dh"]
        self.eval_mu_ph = ev["mu_ph"]
        self.eval_dmu_ph_dp = ev["dmu_ph_dp"]
        self.eval_dmu_ph_dh = ev["dmu_ph_dh"]
        self.eval_k_ph = ev["k_ph"]
        self.eval_dk_ph_dp = ev["dk_ph_dp"]
        self.eval_dk_ph_dh = ev["dk_ph_dh"]
        self.eval_s_ph = ev["s_ph"]
        self.eval_ds_ph_dp = ev["ds_ph_dp"]
        self.eval_ds_ph_dh = ev["ds_ph_dh"]
        # HEM aliases: the blended surface IS the smooth HEM surface.
        self.eval_drho_ph_hem_dp = ev["drho_ph_dp"]
        self.eval_drho_ph_hem_dh = ev["drho_ph_dh"]
        self.eval_dT_ph_hem_dp = ev["dT_ph_dp"]
        self.eval_dT_ph_hem_dh = ev["dT_ph_dh"]
        self.eval_dmu_ph_hem_dp = ev["dmu_ph_dp"]
        self.eval_dmu_ph_hem_dh = ev["dmu_ph_dh"]
        self.eval_dk_ph_hem_dp = ev["dk_ph_dp"]
        self.eval_dk_ph_hem_dh = ev["dk_ph_dh"]
        self.eval_d2rho_ph_hem_dp2 = ev["ddrho_ph_dp_dp"]
        self.eval_d2rho_ph_hem_dpdh = ev["ddrho_ph_dp_dh"]
        self.eval_d2rho_ph_hem_dh2 = ev["ddrho_ph_dh_dh"]

        # h(p, T) family: delegated to the source (boundary-node only).
        self.eval_h_pT = src.eval_h_pT
        self.eval_dh_pT_dp = src.eval_dh_pT_dp
        self.eval_dh_pT_dT = src.eval_dh_pT_dT

        # --- sympy function objects (same names/derivative wiring as
        # CoolPropMedium so components are agnostic) ------------------------
        self.h, self.p, self.T = sp.symbols('h p T', real=True)
        gspf = get_symbolic_property_function
        self.h_pT = gspf(self.eval_h_pT, {1: self.eval_dh_pT_dp, 2: self.eval_dh_pT_dT}, ["p", "T"], med, "h_pT")
        self.rho_ph = gspf(self.eval_rho_ph, {1: self.eval_drho_ph_dp, 2: self.eval_drho_ph_dh}, ["p", "h"], med, "rho_ph")
        self.mu_ph = gspf(self.eval_mu_ph, {1: self.eval_dmu_ph_dp, 2: self.eval_dmu_ph_dh}, ["p", "h"], med, "mu_ph")
        self.T_ph = gspf(self.eval_T_ph, {1: self.eval_dT_ph_dp, 2: self.eval_dT_ph_dh}, ["p", "h"], med, "T_ph")
        self.s_ph = gspf(self.eval_s_ph, {1: self.eval_ds_ph_dp, 2: self.eval_ds_ph_dh}, ["p", "h"], med, "s_ph")
        self.k_ph = gspf(self.eval_k_ph, {1: self.eval_dk_ph_dp, 2: self.eval_dk_ph_dh}, ["p", "h"], med, "k_ph")
        self.rho_ph_hem = gspf(self.eval_rho_ph, {1: self.eval_drho_ph_dp, 2: self.eval_drho_ph_dh}, ["p", "h"], med, "rho_ph_hem")
        self.T_ph_hem = gspf(self.eval_T_ph, {1: self.eval_dT_ph_dp, 2: self.eval_dT_ph_dh}, ["p", "h"], med, "T_ph_hem")
        self.mu_ph_hem = gspf(self.eval_mu_ph, {1: self.eval_dmu_ph_dp, 2: self.eval_dmu_ph_dh}, ["p", "h"], med, "mu_ph_hem")
        self.k_ph_hem = gspf(self.eval_k_ph, {1: self.eval_dk_ph_dp, 2: self.eval_dk_ph_dh}, ["p", "h"], med, "k_ph_hem")
        self.drho_ph_dp = gspf(self.eval_drho_ph_dp, {1: self.eval_d2rho_ph_dp2, 2: self.eval_d2rho_ph_dpdh}, ["p", "h"], med, "drho_ph_dp")
        self.drho_ph_dh = gspf(self.eval_drho_ph_dh, {1: self.eval_d2rho_ph_dpdh, 2: self.eval_d2rho_ph_dh2}, ["p", "h"], med, "drho_ph_dh")
        self.drho_ph_hem_dp = gspf(self.eval_drho_ph_dp, {1: self.eval_d2rho_ph_dp2, 2: self.eval_d2rho_ph_dpdh}, ["p", "h"], med, "drho_ph_hem_dp")
        self.drho_ph_hem_dh = gspf(self.eval_drho_ph_dh, {1: self.eval_d2rho_ph_dpdh, 2: self.eval_d2rho_ph_dh2}, ["p", "h"], med, "drho_ph_hem_dh")

        self.default_vars = dict(getattr(src, "default_vars", {}))

        # --- lambdify modules: SAME names as the source medium, so cached
        # equation templates bind to the tabulated callables unchanged -------
        self.modules = [
            {f"{med}_h_pT": self.eval_h_pT}, {f"{med}_dh_pT_dp": self.eval_dh_pT_dp}, {f"{med}_dh_pT_dT": self.eval_dh_pT_dT},
            {f"{med}_rho_ph": ev["rho_ph"]}, {f"{med}_drho_ph_dp": ev["drho_ph_dp"]}, {f"{med}_drho_ph_dh": ev["drho_ph_dh"]},
            {f"{med}_mu_ph": ev["mu_ph"]}, {f"{med}_dmu_ph_dp": ev["dmu_ph_dp"]}, {f"{med}_dmu_ph_dh": ev["dmu_ph_dh"]},
            {f"{med}_T_ph": ev["T_ph"]}, {f"{med}_dT_ph_dp": ev["dT_ph_dp"]}, {f"{med}_dT_ph_dh": ev["dT_ph_dh"]},
            {f"{med}_s_ph": ev["s_ph"]}, {f"{med}_ds_ph_dp": ev["ds_ph_dp"]}, {f"{med}_ds_ph_dh": ev["ds_ph_dh"]},
            {f"{med}_k_ph": ev["k_ph"]}, {f"{med}_dk_ph_dp": ev["dk_ph_dp"]}, {f"{med}_dk_ph_dh": ev["dk_ph_dh"]},
            {f"{med}_ddrho_ph_dp_dp": ev["ddrho_ph_dp_dp"]}, {f"{med}_ddrho_ph_dp_dh": ev["ddrho_ph_dp_dh"]},
            {f"{med}_ddrho_ph_dh_dp": ev["ddrho_ph_dp_dh"]}, {f"{med}_ddrho_ph_dh_dh": ev["ddrho_ph_dh_dh"]},
            # HEM names bind to the same (already smooth) tabulated surface.
            {f"{med}_rho_ph_hem": ev["rho_ph"]}, {f"{med}_drho_ph_hem_dp": ev["drho_ph_dp"]}, {f"{med}_drho_ph_hem_dh": ev["drho_ph_dh"]},
            {f"{med}_T_ph_hem": ev["T_ph"]}, {f"{med}_dT_ph_hem_dp": ev["dT_ph_dp"]}, {f"{med}_dT_ph_hem_dh": ev["dT_ph_dh"]},
            {f"{med}_mu_ph_hem": ev["mu_ph"]}, {f"{med}_dmu_ph_hem_dp": ev["dmu_ph_dp"]}, {f"{med}_dmu_ph_hem_dh": ev["dmu_ph_dh"]},
            {f"{med}_k_ph_hem": ev["k_ph"]}, {f"{med}_dk_ph_hem_dp": ev["dk_ph_dp"]}, {f"{med}_dk_ph_hem_dh": ev["dk_ph_dh"]},
            {f"{med}_ddrho_ph_hem_dp_dp": ev["ddrho_ph_dp_dp"]}, {f"{med}_ddrho_ph_hem_dp_dh": ev["ddrho_ph_dp_dh"]},
            {f"{med}_ddrho_ph_hem_dh_dp": ev["ddrho_ph_dp_dh"]}, {f"{med}_ddrho_ph_hem_dh_dh": ev["ddrho_ph_dh_dh"]},
        ]
        # Every tabulated evaluator is natively array-capable.
        self.batch_modules = self.modules

        # Attach numba (nopython) twins to the evaluators so the model can
        # JIT-compile whole equation templates that call them (see
        # `Model.instantiate(numba=True)`).  Only available for single-phase
        # windows (the dome/blend path is python-level); failure to build
        # them is never fatal -- the model falls back to the numpy path.
        try:
            self._attach_numba_twins(ev)
        except Exception:
            pass

    # =====================================================================
    # numba twins (single-phase windows only)
    # =====================================================================

    def _attach_numba_twins(self, ev):
        """Build `@njit` twins of the `(p, h)` evaluators and attach them as
        ``_hydrogen_numba`` attributes so `Model.instantiate(numba=True)` can
        compile whole equation templates in nopython mode.

        Only supported for SINGLE-PHASE windows (``kind="plain"`` surfaces):
        the twin is a scalar loop over the bicubic patch tensor with window
        clamping and the same C^1 log-linear/linear extension as the numpy
        path, so both paths return bit-comparable values.  Dome windows keep
        the numpy path (their blend logic is python-level).
        """
        self.numba_available = False
        if self._two_phase:
            return
        try:
            from numba import njit
        except ImportError:
            return
        surfs = {prop: self._surf[f"all_{prop}"] for prop in _PROPS}
        if any(s._spl._A is None for s in surfs.values()):
            return                       # tiny-grid FITPACK fallback in play

        p_min, p_max = self.p_min, self.p_max
        h_min, h_max = self.h_min, self.h_max

        def _make_twin(prop, comp):
            surf = surfs[prop]
            xg = surf._spl._x
            yg = surf._spl._y
            A = surf._spl._A                     # (nx-1, ny-1, 4, 4)
            positive = prop in self._POSITIVE_PROPS
            nx2 = xg.size - 2
            ny2 = yg.size - 2

            @njit(cache=False, fastmath=False)
            def _twin(p, h):
                n = p.shape[0]
                out = np.empty(n)
                for k in range(n):
                    pk = p[k]
                    hk = h[k]
                    pc = min(max(pk, p_min), p_max)
                    hc = min(max(hk, h_min), h_max)
                    i = np.searchsorted(xg, pc) - 1
                    if i < 0:
                        i = 0
                    elif i > nx2:
                        i = nx2
                    j = np.searchsorted(yg, hc) - 1
                    if j < 0:
                        j = 0
                    elif j > ny2:
                        j = ny2
                    hx = xg[i + 1] - xg[i]
                    hy = yg[j + 1] - yg[j]
                    u = (pc - xg[i]) / hx
                    v = (hc - yg[j]) / hy
                    # Bicubic patch f = U(u) . A[i,j] . V(v)^T.  Collapse the
                    # v-axis first (Horner in v for each row m of A), then
                    # Horner in u for the value and its u-derivatives.
                    s0 = 0.0
                    s1 = 0.0
                    s2 = 0.0
                    s3 = 0.0
                    d0 = 0.0
                    d1 = 0.0
                    d2 = 0.0
                    d3 = 0.0
                    e0 = 0.0
                    e1 = 0.0
                    e2 = 0.0
                    e3 = 0.0
                    for m in range(4):
                        a0 = A[i, j, m, 0]
                        a1 = A[i, j, m, 1]
                        a2 = A[i, j, m, 2]
                        a3 = A[i, j, m, 3]
                        sv = ((a3 * v + a2) * v + a1) * v + a0
                        sdv = (3.0 * a3 * v + 2.0 * a2) * v + a1
                        sddv = 6.0 * a3 * v + 2.0 * a2
                        if m == 0:
                            s0 = sv
                            d0 = sdv
                            e0 = sddv
                        elif m == 1:
                            s1 = sv
                            d1 = sdv
                            e1 = sddv
                        elif m == 2:
                            s2 = sv
                            d2 = sdv
                            e2 = sddv
                        else:
                            s3 = sv
                            d3 = sdv
                            e3 = sddv
                    f = ((s3 * u + s2) * u + s1) * u + s0
                    fu = (3.0 * s3 * u + 2.0 * s2) * u + s1
                    fuu = 6.0 * s3 * u + 2.0 * s2
                    fv = ((d3 * u + d2) * u + d1) * u + d0
                    fuv = (3.0 * d3 * u + 2.0 * d2) * u + d1
                    fvv = ((e3 * u + e2) * u + e1) * u + e0
                    fx = fu / hx
                    fy = fv / hy
                    fxx = fuu / (hx * hx)
                    fxy = fuv / (hx * hy)
                    fyy = fvv / (hy * hy)
                    dp = pk - pc
                    dh = hk - hc
                    if dp != 0.0 or dh != 0.0:
                        if positive:
                            uarg = (fx * dp + fy * dh) / f
                            if uarg > 40.0:
                                uarg = 40.0
                            elif uarg < -40.0:
                                uarg = -40.0
                            g = np.exp(uarg)
                            f *= g
                            fx *= g
                            fy *= g
                            fxx *= g
                            fxy *= g
                            fyy *= g
                        else:
                            f = f + fx * dp + fy * dh
                    if comp == 0:
                        out[k] = f
                    elif comp == 1:
                        out[k] = fx
                    elif comp == 2:
                        out[k] = fy
                    elif comp == 3:
                        out[k] = fxx
                    elif comp == 4:
                        out[k] = fxy
                    else:
                        out[k] = fyy
                return out

            return _twin

        for prop, order in _PROPS.items():
            names = [(f"{prop}_ph", 0), (f"d{prop}_ph_dp", 1),
                     (f"d{prop}_ph_dh", 2)]
            if order >= 2:
                names += [(f"dd{prop}_ph_dp_dp", 3), (f"dd{prop}_ph_dp_dh", 4),
                          (f"dd{prop}_ph_dh_dh", 5)]
            for key, comp in names:
                ev[key]._hydrogen_numba = _make_twin(prop, comp)
        self.numba_available = True

    # --- introspection / cache management (interface parity) ----------------

    def get_default_vars(self):
        return self.default_vars

    def clear_cache(self):
        self._eval_cache.clear()

    def print_cache_info(self):
        print(f"TabulatedMedium({self.medium}): eval LRU holds "
              f"{len(self._eval_cache)} entries; "
              f"validation max rel err: "
              f"{getattr(self, 'validation_max_rel_err', {})}")
