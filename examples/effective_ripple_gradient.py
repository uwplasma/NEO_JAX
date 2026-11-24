"""Effective ripple from bounce integrals and its gradient with respect to the Boozer spectrum.

Evaluates ``epsilon_eff**(3/2)`` on three surfaces of a quasi-isodynamic and a rotating-ellipse
equilibrium, compares with stored Fortran ``xneo`` values, and checks the reverse-mode gradient
with respect to the ``|B|`` harmonics against central differences. Writes
``effective_ripple_gradient.png`` (the figure of ``docs/applications.rst``).

    python examples/effective_ripple_gradient.py
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from neo_jax import BoozerData, booz_xform_to_boozerdata_jax, effective_ripple_3_2

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "ripple"
# Fortran xneo, 128 x 128 spline, 96 particles, 80 steps per period, acc_req 1e-3 (64 x 64 where that run is NaN)
XNEO = {"QI, nfp 4, finite beta": ("boozer_nfp4_QI_finite_beta.npz", [5.1933e-4, 4.7918e-4, 4.5405e-4]),
        "rotating ellipse, nfp 2": ("boozer_ellipse_nfp2.npz", [1.1828e-2, 1.6702e-2, 2.1461e-2])}
BLUE, ORANGE, GREY = "#2f6f9f", "#d9822b", "#4a4a4a"


def load(name):
    with np.load(FIXTURES / name) as data:
        return booz_xform_to_boozerdata_jax({**{k: data[k] for k in data.files}, "mode_first": True}, mode_first=True)


def one_surface(booz, row):
    """The single surface ``row`` of ``booz``."""
    keep = {k: v[row:row + 1] for k, v in booz.__dict__.items() if getattr(v, "ndim", 0) >= 1 and k not in ("ixm", "ixn")}
    return BoozerData(**{**booz.__dict__, **keep})


plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, (ax_line, ax_value, ax_grad) = plt.subplots(1, 3, figsize=(13.5, 3.9), constrained_layout=True)

# (a) |B| along one Boozer field line of the QI surface, with the wells trapped at one pitch
qi = load(XNEO["QI, nfp 4, finite beta"][0])
surf = one_surface(qi, 1)
iota, nfp = float(surf.iota[0]), int(surf.nfp)
phi = np.linspace(0.0, 3 * 2 * np.pi / nfp, 1500)
xm, xn = np.asarray(surf.ixm, float), np.asarray(surf.ixn, float)
field = (np.asarray(surf.bmnc[0]) * np.cos(xm * (iota * phi[:, None]) - xn * phi[:, None])).sum(-1)
level = field.min() + 0.55 * (field.max() - field.min())
ax_line.plot(phi * nfp / (2 * np.pi), field, color=GREY, lw=1.6)
ax_line.fill_between(phi * nfp / (2 * np.pi), field, level, where=field < level, color=BLUE, alpha=0.25, lw=0)
ax_line.axhline(level, color=BLUE, lw=1.0, ls="--")
cross = np.nonzero(np.diff(np.sign(field - level)))[0]
ax_line.plot(phi[cross] * nfp / (2 * np.pi), np.full(cross.size, level), "o", color=BLUE, ms=5)
ax_line.text(0.03, level, r"$b = 1/\lambda$", color=BLUE, va="bottom", transform=ax_line.get_yaxis_transform())
ax_line.set(xlabel="toroidal angle [field periods]", ylabel="|B| [T]", title="(a) wells and bounce points on a field line")

# (b) value against Fortran xneo
width, ticks = 0.38, []
for k, (label, (name, reference)) in enumerate(XNEO.items()):
    booz = load(name)
    values = np.asarray(effective_ripple_3_2(booz))
    x = np.arange(3) + 4 * k
    ax_value.bar(x - width / 2, reference, width, color=GREY, label="Fortran xneo" if k == 0 else None)
    ax_value.bar(x + width / 2, values, width, color=BLUE, label="bounce integrals" if k == 0 else None)
    for xi, v, r in zip(x, values, reference):
        ax_value.text(xi, 1.25 * max(v, r), f"{100 * (v / r - 1):+.1f}%", ha="center", fontsize=8)
    ticks += [(xi, f"s={s:.2f}") for xi, s in zip(x, np.asarray(booz.es))]
    ax_value.text(x[1], -0.17, label, ha="center", va="top", fontsize=9, transform=ax_value.get_xaxis_transform())
ax_value.set_yscale("log")
ax_value.set_ylim(5e-5, 1.5e-1)
ax_value.set_xticks([t[0] for t in ticks], [t[1] for t in ticks], fontsize=8)
ax_value.set(ylabel=r"$\epsilon_\mathrm{eff}^{3/2}$", title="(b) value on three surfaces")
ax_value.legend(frameon=False, loc="upper left")

# (c) reverse-mode gradient with respect to the |B| harmonics against central differences
amplitudes = jnp.asarray(surf.bmnc[0])


def ripple(a):
    return effective_ripple_3_2(BoozerData(**{**surf.__dict__, "bmnc": a[None]}))[0]


gradient = np.asarray(jax.jit(jax.grad(ripple))(amplitudes))
largest = np.argsort(-np.abs(np.asarray(amplitudes)))[1:13]         # skip B00
value = jax.jit(ripple)
step = 1e-6 * float(amplitudes[0])
central = np.array([(float(value(amplitudes.at[i].add(step))) - float(value(amplitudes.at[i].add(-step)))) / (2 * step)
                    for i in largest])
span = [0.5 * np.abs(central).min(), 2 * np.abs(central).max()]
ax_grad.plot(span, span, color=GREY, lw=1.0)
ax_grad.plot(np.abs(central), np.abs(gradient[largest]), "o", color=ORANGE, ms=6)
ax_grad.set(xscale="log", yscale="log", xlabel="central differences [1/T]", ylabel="reverse mode [1/T]",
            title=r"(c) $|\partial\epsilon_\mathrm{eff}^{3/2}/\partial B_{mn}|$, 12 largest harmonics")
worst = np.abs(gradient[largest] / central - 1).max()
ax_grad.text(0.05, 0.9, f"largest difference {100 * worst:.2g}%", transform=ax_grad.transAxes)

fig.savefig("effective_ripple_gradient.png", dpi=150)
print(f"largest gradient difference against central differences: {worst:.2e}")
