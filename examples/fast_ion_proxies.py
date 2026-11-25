"""Fast-ion proxies Gamma_c and Gamma_alpha and their derivatives on an analytic Boozer field.

A quasi-helically symmetric field ``B = 1 + 0.12 cos(theta - N phi)`` confines all trapped orbits.
Adding a toroidal and a second helical harmonic of relative size ``delta`` breaks the symmetry;
both proxies rise from zero. The script scans ``delta``, checks the forward-mode derivative against
central differences and writes ``fast_ion_proxies.png`` (the figure of ``docs/applications.rst``).

    python examples/fast_ion_proxies.py
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from neo_jax import BoozerData, FastIonBudget, fast_ion_proxies

NFP = 4
S = np.linspace(0.3, 0.7, 5)                    # five surfaces: the kernel differentiates in s around the middle one
XM, XN = np.array([0, 1, 1, 2]), np.array([0, NFP, 0, NFP])
BUDGET = FastIonBudget(n_transits=2, n_level=16)             # two knobs below the defaults, for speed
BLUE, ORANGE, GREY = "#2f6f9f", "#d9822b", "#4a4a4a"


def boozer(delta):
    """|B| = 1 + 0.12 cos(theta - N phi) + delta (0.6 cos theta + 0.4 cos(2 theta - N phi)), growing with s."""
    shape = jnp.sqrt(jnp.asarray(S) / 0.5)[:, None]
    b = jnp.stack([1.0, 0.12, 0.6 * delta, 0.4 * delta]) * jnp.concatenate([jnp.ones((5, 1)), shape * jnp.ones((5, 3))], 1)
    zero = jnp.zeros_like(b)
    return BoozerData(rmnc=zero, zmns=zero, lmns=zero, bmnc=b, ixm=XM, ixn=XN, es=jnp.asarray(S),
                      iota=1.1 + 0.2 * jnp.asarray(S), curr_pol=jnp.ones(5), curr_tor=jnp.zeros(5), nfp=NFP)


def proxies(delta):
    gamma_c, gamma_alpha, trapped = fast_ion_proxies(boozer(delta), BUDGET)
    return jnp.stack([gamma_c[0], gamma_alpha[0], trapped[0]])


value = jax.jit(proxies)
tangent = jax.jit(lambda d: jax.jvp(proxies, (d,), (1.0,))[1])
deltas = np.linspace(0.0, 0.06, 13)
values = np.array([value(d) for d in deltas])
forward = np.array([tangent(d) for d in deltas[1:]])
h = 2e-4
central = np.array([(value(d + h) - value(d - h)) / (2 * h) for d in deltas[1:]])

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, (ax_field, ax_value, ax_grad) = plt.subplots(1, 3, figsize=(13.5, 3.9), constrained_layout=True)

theta, phi = np.meshgrid(np.linspace(0, 2 * np.pi, 121), np.linspace(0, 2 * np.pi / NFP, 121), indexing="ij")
amp = np.asarray(boozer(0.04).bmnc[2])
field = sum(a * np.cos(m * theta - n * phi) for a, m, n in zip(amp, XM, XN))
mesh = ax_field.contourf(phi * NFP / (2 * np.pi), theta / (2 * np.pi), field, 24, cmap="viridis")
ax_field.contour(phi * NFP / (2 * np.pi), theta / (2 * np.pi), field, 8, colors="w", linewidths=0.5)
fig.colorbar(mesh, ax=ax_field, label="|B| / B$_{00}$")
ax_field.set(xlabel="toroidal Boozer angle [field periods]", ylabel=r"poloidal Boozer angle [$2\pi$]",
             title=r"(a) field at $\delta$ = 0.04, s = 0.5")

ax_value.plot(deltas, values[:, 0], "o-", color=BLUE, label=r"$\Gamma_c$")
ax_value.plot(deltas, values[:, 1], "s-", color=ORANGE, label=r"$\Gamma_\alpha$")
ax_value.plot(deltas, values[:, 2], "--", color=GREY, label="trapped fraction")
ax_value.set(xlabel=r"symmetry-breaking amplitude $\delta$", ylabel="proxy", title="(b) value: zero for the symmetric field")
ax_value.legend(frameon=False)

for k, (color, marker, label) in enumerate(((BLUE, "o", r"$\Gamma_c$"), (ORANGE, "s", r"$\Gamma_\alpha$"))):
    ax_grad.plot(deltas[1:], central[:, k], "-", color=color, lw=1.2)
    ax_grad.plot(deltas[1:], forward[:, k], marker, color=color, ms=6, mfc="none", label=f"d{label}/d$\\delta$")
ax_grad.plot([], [], "-", color=GREY, lw=1.2, label="central differences")
ax_grad.plot([], [], "o", color=GREY, mfc="none", label="forward mode")
ax_grad.set(xlabel=r"symmetry-breaking amplitude $\delta$", ylabel="derivative", title="(c) derivative")
ax_grad.legend(frameon=False)

fig.savefig("fast_ion_proxies.png", dpi=150)
scale = np.abs(central[:, :2]).max(0)
print("largest forward-mode error over the scan, relative to the largest derivative (Gamma_c, Gamma_alpha):",
      np.abs(forward[:, :2] - central[:, :2]).max(0) / scale)
