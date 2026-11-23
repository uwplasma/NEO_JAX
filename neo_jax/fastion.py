r"""Fast-ion proxies :math:`\Gamma_c` and :math:`\Gamma_\alpha` from the Boozer spectrum of ``|B|``.

Both are built from the bounce-averaged drifts of trapped particles, which are derivatives of the
second adiabatic invariant :math:`J(s,\alpha,\lambda)=\oint v_\parallel\,\mathrm dl`
(Nemov et al., Phys. Plasmas 15, 052501 (2008); Velasco et al., Nucl. Fusion 61, 116059 (2021),
eqs. 14-16 and 25).  In Boozer coordinates :math:`\mathrm dl=(G+\iota I)\,\mathrm d\varphi/B` and a
field line is :math:`\theta=\alpha+\iota\varphi`, so with :math:`w=(1-\lambda B/2)/\sqrt{1-\lambda B}`
and :math:`\hat G=G+\iota I`

.. math::

   \partial_\alpha J=-\hat G\int w\,\frac{\partial_\theta B}{B^2}\,\mathrm d\varphi,\qquad
   \partial_s J=\hat G'\int\frac{\sqrt{1-\lambda B}}{B}\,\mathrm d\varphi
   -\hat G\int w\,\frac{\partial_sB+\iota'(\varphi-\varphi_c)\,\partial_\theta B}{B^2}\,\mathrm d\varphi ,

and :math:`\gamma_c^*=(2/\pi)\arctan(\partial_\alpha J/|\partial_sJ|)`.  Only the Boozer spectrum of
``|B|`` on five neighbouring surfaces (a quartic fit gives :math:`\partial_sB`, :math:`\iota'`,
:math:`\hat G'`), ``iota``, ``G``, ``I`` and ``nfp`` enter: no :math:`\lambda`-angle, metric or pointwise
tangency factor.  The shear term :math:`\iota'\varphi\,\partial_\theta B` is secular along a line; it is
referenced to the midpoint :math:`\varphi_c` of each well, i.e. the derivative is taken at fixed
position of the well on the surface, which keeps it bounded and independent of the line length.

The bounce machinery is that of :mod:`neo_jax.ripple` (Hermite cubics along Boozer lines, exact
monotone-piece roots, pitch panels between the critical values of ``|B|``, sine-mapped well Gauss rule,
smooth window weights); only the integrands differ.  ``Gamma_alpha`` follows Velasco et al.
section 4.2 (ripple wells ignored): an orbit is lost when its precession meets an outward
superbanana (``gamma* > 0.2``) before an inward one; each well takes the label ``alpha + 2 pi iota k / nfp``
of its field period ``k``, and the signed ``gamma*`` and precession direction ``-sign(dJ/ds)`` of the
wells are binned on a periodic ``alpha`` grid with cubic B-spline weights (C2); thresholds are smoothsteps of half-width
``threshold_width`` (0.05 gave 5 % value jumps on QA, 0.2 about 3 %; above 0.2 an omnigenous field loses particles).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .data_models import BoozerData
from .ripple import _cubic, _gl, _peval, _root_v, _series, _smooth_step, _stationary

__all__ = ["FastIonBudget", "fast_ion_proxies"]

_GAMMA_TH = 0.2


@dataclass(frozen=True)
class FastIonBudget:
    """Resolution: ``n_lines`` lines of ``n_transits`` transits (+ ``n_extension`` to close wells),
    ``nodes_per_period`` Hermite nodes per field period, ``n_pitch`` Gauss nodes on each of the
    critical-value panels and of ``n_panel`` extra equispaced panels, ``n_well`` nodes per well,
    ``n_level`` shared pitch levels and ``threshold_width`` for ``Gamma_alpha``."""

    n_lines: int = 16
    n_transits: int = 4
    n_extension: int = 3
    nodes_per_period: int = 24
    n_pitch: int = 4
    n_panel: int = 48
    n_well: int = 64               # Gauss nodes on each piece of a well
    n_split: int = 2               # interior stationary points at which a well is cut (n_split + 1 pieces)
    n_level: int = 32              # uniform pitch panels of Gamma_alpha (besides the critical levels)
    n_pitch_alpha: int = 2
    n_bin: int = 32                # alpha bins of the Gamma_alpha maps
    n_surface: int = 64
    n_dense: int = 128
    n_fit: int = 5                 # rows of the radial polynomial fit (3: central differences)
    max_wells: int | None = None
    max_critical: int | None = None
    threshold_width: float = 0.2   # half-width in gamma* of the smoothstep around 0.2; wider gives Gamma_alpha > 0 at gamma* = 0
    rho_units: bool = False        # ratio in (rho, alpha) instead of Velasco's (s, alpha); for cross-checks


def _wells(cs, pcs, h, b, iota, dG, ip, nstart, ramp, nwell, nquad, sign, scale, nsplit=2):
    """Per-well ``(window weight, v tau, N, D, midpoint)`` of one line at pitch ``b``; ``N, D`` are
    ``d_alpha J`` and ``d_s J`` over ``|Ghat|`` and ``tan(pi gamma*/2) = N / |D|``."""
    cB, cT, cS = cs
    n = cB.shape[0]
    iv, t0, t1, v0 = pcs
    m = iv.shape[0]
    pos = jnp.concatenate([v0, _peval(cB[-1], 1.0)[None]]) >= b
    down = pos[:-1] & ~pos[1:] & (iv < nstart)
    up = ~pos[:-1] & pos[1:]
    d = jnp.nonzero(down, size=nwell, fill_value=m)[0]
    ups = jnp.nonzero(up, size=m // 2 + 1, fill_value=m)[0]
    u = ups[jnp.minimum(jnp.searchsorted(ups, d), m // 2)]
    valid = (d < m) & (u < m)
    d, u = jnp.minimum(d, m - 1), jnp.minimum(u, m - 1)
    r1 = (iv[d] + _root_v(cB[iv[d]], b, t0[d], t1[d])) * h
    r2 = (iv[u] + _root_v(cB[iv[u]], b, t0[u], t1[u])) * h
    x, w = _gl(nquad, -1.0, 1.0)
    ang = 0.5 * jnp.pi * x
    # a well is cut at its interior stationary points (humps of |B| just below the level make 1/sqrt(1 - B/b)
    # almost singular inside it) and every piece takes the sine-mapped Gauss rule, which resolves the
    # square-root behaviour at both of its ends
    stat = jax.lax.stop_gradient(jnp.sort(jnp.where((t1 < 1) & (jnp.arange(m) % 3 < 2), (iv + t1) * h, (n + 1.0) * h)))   # cut points: the integral does not depend on them
    first = jnp.searchsorted(stat, r1, side="right")
    inner = stat[jnp.minimum(first[:, None] + jnp.arange(nsplit), m - 1)]
    edges = jnp.concatenate([r1[:, None], jnp.minimum(inner, r2[:, None]), r2[:, None]], 1)
    mid = jnp.where(valid, 0.5 * (r1 + r2), 0.0)
    pm, ph = 0.5 * (edges[:, 1:] + edges[:, :-1]), 0.5 * (edges[:, 1:] - edges[:, :-1])
    pm, ph = jnp.where(valid[:, None], pm, 0.0), jnp.where(valid[:, None], ph, 0.0)
    phi = (pm[:, :, None] + ph[:, :, None] * jnp.sin(ang)).reshape(nwell, -1)
    wq = (ph[:, :, None] * 0.5 * jnp.pi * jnp.cos(ang) * w).reshape(nwell, -1)
    phi = jnp.where(wq > 0, phi, mid[:, None])               # empty pieces sample an interior point, not a bounce point
    idx = jnp.clip(jnp.floor(phi / h).astype(int), 0, n - 1)
    t = phi / h - idx
    Bq, Tq, Sq = _peval(cB[idx], t), _peval(cT[idx], t), _peval(cS[idx], t)
    x2 = 1 - Bq / b
    live = x2 > 0
    rad = jnp.where(live, jnp.sqrt(jnp.where(live, x2, 1.0)), 0.0)
    inv = jnp.where(live, 1.0 / jnp.where(live, rad, 1.0), 0.0)
    wd = (1 - 0.5 * Bq / b) * inv
    vt = 2 * (wq * inv / Bq).sum(-1)
    Nn = -(wq * wd * Tq / Bq**2).sum(-1)                          # d_alpha J / Ghat
    Dn = dG * (wq * rad / Bq).sum(-1) - (wq * wd * (Sq + ip * (phi - mid[:, None]) * Tq) / Bq**2).sum(-1)
    ok = valid & (vt > 1e-30)
    Nn, Dn = jnp.where(ok, Nn, 1.0), jnp.where(ok, Dn, 1.0)
    Nn = sign * scale * Nn                                         # d_alpha J / |Ghat|, sign = sign(Ghat)
    win = (_smooth_step(r1 / ramp) * _smooth_step((nstart * h - r1) / ramp)
           * _smooth_step((n * h - r2) / (0.5 * (n - nstart) * h)))
    return (jnp.where(ok, win, 0.0), jnp.where(ok, vt, 0.0), jnp.where(ok, Nn, 0.0),
            jnp.where(ok, Dn, 0.0), mid), down.sum() > nwell


def _lost_fraction(gamma, direction, width):
    """Probability that precession meets an outward superbanana before an inward one (eq. 25)."""
    n = gamma.shape[0]
    def step(g):
        u = jnp.clip((g - _GAMMA_TH) / (2 * width) + 0.5, 0.0, 1.0)
        return u * u * (3 - 2 * u)

    out, inw = step(gamma), step(-gamma)
    free = jnp.clip(1.0 - out - inw, 0.0, 1.0)
    ar = jnp.arange(n)

    def sweep(sg):
        k = (ar[:, None] + sg * ar[None, :]) % n
        alive = jnp.cumprod(free[k], axis=1)
        alive = jnp.concatenate([jnp.ones_like(alive[:, :1]), alive[:, :-1]], 1)
        return jnp.sum(out[k] * alive, axis=1)

    plus = 0.5 * (1.0 + direction)
    return (plus * sweep(1) + (1.0 - plus) * sweep(-1)) * (1.0 - jnp.prod(1.0 - inw, axis=0))


def _fit_derivative(s, f, rows=5):
    """Derivative at ``s[2]`` of the polynomial through the ``rows`` (3 or 5) central rows of ``f`` (rows first)."""
    k = (5 - rows) // 2
    s, f = s[k:5 - k], f[k:5 - k]
    x = s - s[rows // 2]
    inv = jnp.linalg.inv(x[:, None] ** jnp.arange(rows)[None, :])
    return jnp.tensordot(inv[1], f, axes=(0, 0))


def _surface(amp, damp, xm, xn, iota, diota, Ghat, dGhat, nfp, bud, srow, ramp=0.25):
    nturn, nper, nquad = bud.n_transits, bud.nodes_per_period, bud.n_well
    n = nfp * (nturn + bud.n_extension) * nper
    nstart = nfp * nturn * nper
    L, h = 2 * jnp.pi * nturn, 2 * jnp.pi * (nturn + bud.n_extension) / n
    nwell = bud.max_wells or 2 * nfp * nturn + 4
    ncrit = bud.max_critical or 4 * nfp * (nturn + bud.n_extension)
    th = jnp.arange(bud.n_surface) * 2 * jnp.pi / bud.n_surface
    TH, PH = jnp.meshgrid(th, th / nfp, indexing="ij")
    Bg = _series(amp, xm, xn, TH, PH)[0]
    inv_b2 = (1.0 / Bg**2).mean()
    norm = inv_b2 * L * (1 - ramp)
    sign, dG = jnp.sign(Ghat), dGhat / Ghat
    scale = 1 / (2 * jnp.sqrt(srow)) if bud.rho_units else 1.0   # ratio in (rho, alpha)
    phi = jnp.arange(n + 1) * h
    pitch, _ = _gl(bud.n_pitch)
    sg = pitch * pitch * (3 - 2 * pitch)
    unif = (jnp.arange(1, bud.n_panel + 1) / (bud.n_panel + 1))

    def line_data(alpha):
        Bn, Bt, Bp, Btt, Btp, _ = _series(amp, xm, xn, alpha + iota * phi, phi, order=2)
        Sn, St, Sp = _series(damp, xm, xn, alpha + iota * phi, phi)
        cB = _cubic(Bn, Bp + iota * Bt, h)
        cT = _cubic(Bt, Btp + iota * Btt, h)
        cS = _cubic(Sn, Sp + iota * St, h)
        ta, tb = _stationary(cB)
        brk = jnp.stack([jnp.zeros(n), ta, tb, jnp.ones(n)], -1)
        iv = jnp.repeat(jnp.arange(n), 3)
        t0, t1 = brk[:, :3].ravel(), brk[:, 1:].ravel()
        return (cB, cT, cS), (iv, t0, t1, _peval(cB[iv], t0)), Bn, ta, tb

    def levels(cB, Bn, ta, tb, uniform=True):
        cand = jnp.concatenate([jnp.where(ta < 1, _peval(cB, jax.lax.stop_gradient(ta)), jnp.nan),
                                jnp.where(tb < 1, _peval(cB, jax.lax.stop_gradient(tb)), jnp.nan)])
        top, ncand = Bn.max(), (~jnp.isnan(cand)).sum()
        cs = jnp.sort(jnp.where(jnp.isnan(cand), top, cand))
        pick = jnp.where(ncand > ncrit, jnp.round(jnp.linspace(0.0, 1.0, ncrit) * (ncand - 1)),
                         jnp.arange(ncrit)).astype(int)
        low = jnp.minimum(Bn.min(), cs[0])
        crit = jnp.concatenate([cs[jnp.minimum(pick, cand.size - 1)], Bn[:1], Bn[-1:], top[None]]
                               + ([low + (top - low) * unif] if uniform else []))
        crit = jnp.sort(crit)
        return crit, crit[1:] - crit[:-1] > 1e-9 * (top - low)

    # Lagrange map from the panel nodes to a dense Gauss rule: the well integrals are evaluated on the
    # panel nodes, and gamma_c^2 = (2/pi atan(N/|D|))^2, which has narrow resonances where the poloidal
    # drift D vanishes, on the dense rule
    tq = pitch
    td, wd = _gl(bud.n_dense)
    P = np.ones((bud.n_dense, bud.n_pitch))
    for q in range(bud.n_pitch):
        for j in range(bud.n_pitch):
            if j != q:
                P[:, q] *= (td - tq[j]) / (tq[q] - tq[j])
    P = jnp.asarray(P)
    sd, dsd = td * td * (3 - 2 * td), 6 * td * (1 - td)      # the map of the panel nodes

    def gamma_c_line(alpha):
        cs, pcs, Bn, ta, tb = line_data(alpha)
        edges, live = levels(cs[0], Bn, ta, tb)
        lo, hi = edges[:-1], edges[1:]
        bq = jnp.where(live[:, None], lo[:, None] + (hi - lo)[:, None] * sg, jax.lax.stop_gradient(0.5 * (edges[0] + edges[-1])))
        (win, vt, Nn, Dn, _), over = jax.vmap(lambda b: _wells(cs, pcs, h, b, iota, dG, diota, nstart, ramp * L,
                                                                nwell, nquad, sign, scale, bud.n_split))(bq.ravel())
        npanel = lo.shape[0]
        sh = lambda a: a.reshape(npanel, bud.n_pitch, nwell)
        okp = jnp.all(sh(vt) > 0, axis=1)                          # well present at every node of the panel
        dense = lambda a: jnp.einsum("mq,pqw->pmw", P, sh(a)) * okp[:, None, :]
        wv, Nd, Dd = dense(win * vt), dense(Nn), dense(Dn)
        gam = (2 / jnp.pi) * jnp.arctan2(jnp.where(okp[:, None, :], Nd, 1.0), jnp.where(okp[:, None, :], jnp.abs(Dd), 1.0))
        bd = lo[:, None] + (hi - lo)[:, None] * sd                   # (panel, dense)
        wpd = jnp.where(live, hi - lo, 0.0)[:, None] * dsd * wd / bd**2
        return jnp.sum(wpd * (wv * gam**2).sum(-1)), over.any()

    alpha = (jnp.arange(bud.n_lines) + 0.5) * 2 * jnp.pi / bud.n_lines
    S, over_c = jax.lax.map(jax.checkpoint(gamma_c_line), alpha)
    gamma_c = jnp.pi / (8 * jnp.sqrt(2.0)) * S.mean() / norm

    # Gamma_alpha: the panels are bounded by the critical levels of every line (so the well topology of all
    # lines is constant on each panel and the b integrand is smooth) and a global uniform grid
    nb, nl = bud.n_bin, bud.n_level
    def line_levels(a):
        cs, _, Bn, ta, tb = line_data(a)
        return levels(cs[0], Bn, ta, tb, False)[0]

    crit = jax.lax.map(line_levels, alpha)
    bmin, bmax = Bg.min(), Bg.max()
    edges = jnp.sort(jnp.concatenate([crit.ravel(), bmin + (bmax - bmin) * jnp.arange(nl + 1) / nl]))
    lo, hi = edges[:-1, None], edges[1:, None]
    live = hi - lo > 1e-9 * (bmax - bmin)
    xa, wa = _gl(bud.n_pitch_alpha)
    sa, dsa = xa * xa * (3 - 2 * xa), 6 * xa * (1 - xa)
    bl = jnp.where(live, lo + (hi - lo) * sa, jax.lax.stop_gradient(0.5 * (bmin + bmax))).ravel()
    wl = (jnp.where(live, hi - lo, 0.0) * dsa * wa).ravel() / bl**2
    nlev = bl.shape[0]

    def alpha_line(a):
        cs, pcs, *_ = line_data(a)
        (win, vt, Nn, Dn, mid), over = jax.vmap(lambda b: _wells(cs, pcs, h, b, iota, dG, diota, nstart,
                                                                  ramp * L, nwell, nquad, sign, scale, bud.n_split))(bl)
        k = jnp.floor(mid * nfp / (2 * jnp.pi))
        return win * vt, Nn, Dn, a + 2 * jnp.pi * iota * k / nfp, over.any()

    wv, Nn, Dn, lab, over_a = jax.lax.map(jax.checkpoint(alpha_line), alpha)      # (line, level, well)
    on = wv > 0
    gam = jnp.where(on, (2 / jnp.pi) * jnp.arctan2(jnp.where(on, Nn, 1.0), jnp.abs(jnp.where(on, Dn, 1.0))), 0.0)
    # smooth sign of the precession: a hard sign would drop the jump at D = 0 from the gradient
    direction = jnp.where(wv > 0, -sign * Dn / jnp.sqrt(jnp.where(wv > 0, Dn**2 + Nn**2, 1.0)), 0.0)
    u = jnp.mod(lab / (2 * jnp.pi) * nb, nb)
    i0 = jnp.floor(u).astype(int)
    t = u - i0
    cub = (1 - t) ** 3 / 6, (3 * t**3 - 6 * t**2 + 4) / 6, (-3 * t**3 + 3 * t**2 + 3 * t + 1) / 6, t**3 / 6
    ij = [(i0 + j - 1) % nb for j in range(4)]                  # cubic B-spline (C2) weights on the periodic alpha grid
    lv = jnp.broadcast_to(jnp.arange(nlev)[None, :, None], i0.shape)

    def binned(v):          # summed over lines and wells
        return sum(jnp.zeros((nlev, nb)).at[lv, i].add(v * c) for i, c in zip(ij, cub))

    mass = binned(wv)
    ref = 0.05 * mass.mean(1, keepdims=True) + 1e-30        # a bin holding a sliver of a well must not carry its full gamma
    maps = [binned(wv * v) / (mass + ref) for v in (gam, direction)]
    lost = _lost_fraction(maps[0].T, maps[1].T, bud.threshold_width)                # (bin, level)
    per_well = sum(lost[i, lv] * c for i, c in zip(ij, cub))
    scale_a = 0.25 / (bud.n_lines * norm)
    gamma_alpha = scale_a * jnp.sum(wl * jnp.sum(wv * per_well, (0, 2)))
    f_trapped = scale_a * jnp.sum(wl * jnp.sum(wv, (0, 2)))
    bad = over_c.any() | over_a.any()
    return (jnp.where(bad, jnp.nan, gamma_c), jnp.where(bad, jnp.nan, gamma_alpha), f_trapped)


def fast_ion_proxies(booz: BoozerData, budget: FastIonBudget | None = None):
    r"""``(Gamma_c, Gamma_alpha, f_trapped)`` on the interior rows ``2 .. n-3`` of ``booz`` (traceable).

    ``booz`` holds consecutive surfaces; each target row uses itself and its two neighbours on each side
    for the radial derivatives.  ``ixm`` and ``ixn`` must be concrete.  A row that overflows the well
    slots returns NaN.  See the module docstring for the definitions.
    """
    bud = budget or FastIonBudget()
    bmnc = jnp.asarray(booz.bmnc)
    bmns = 0 * bmnc if booz.bmns is None else jnp.asarray(booz.bmns)
    amp = bmnc - 1j * bmns
    s = jnp.asarray(booz.es)
    iota = jnp.asarray(booz.iota)
    Ghat = jnp.asarray(booz.curr_pol) + iota * jnp.asarray(booz.curr_tor)
    xm, xn = np.asarray(booz.ixm, float), np.asarray(booz.ixn, float)
    nfp, nrow = int(booz.nfp), bmnc.shape[0]
    out = []
    for r in range(2, nrow - 2):
        sl = slice(r - 2, r + 3)
        out.append(_surface(amp[r], _fit_derivative(s[sl], amp[sl], bud.n_fit), xm, xn, iota[r], _fit_derivative(s[sl], iota[sl], bud.n_fit),
                            Ghat[r], _fit_derivative(s[sl], Ghat[sl], bud.n_fit), nfp, bud, s[r]))
    return tuple(jnp.stack(v) for v in zip(*out))
