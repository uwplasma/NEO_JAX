"""First-order reverse differentiation of dynamic loops with logarithmic storage."""

from functools import partial

import jax
import jax.numpy as jnp

_BLOCK_SIZE = 8


def _run(body, lower, upper, value, constants):
    def step(carry):
        i, value = carry
        return i + 1, body(i, value, *constants)

    return jax.lax.while_loop(lambda c: c[0] < upper, step, (lower, value))[1]


def _zero_cotangent(x):
    dtype = jax.dtypes.float0 if not jnp.issubdtype(jnp.asarray(x).dtype, jnp.inexact) else x.dtype
    return jnp.zeros(jnp.shape(x), dtype=dtype)


def _add_cotangent(x, y):
    return x if x.dtype == jax.dtypes.float0 else x + y


@partial(jax.custom_vjp, nondiff_argnums=(0,))
def _checkpointed(body, lower, upper, value, constants):
    return _run(body, lower, upper, value, constants)


def _forward(body, lower, upper, value, constants):
    return _run(body, lower, upper, value, constants), (lower, upper, value, constants)


def _backward(body, residual, cotangent):
    lower, upper, initial, constants = residual
    # Depth-first right-to-left traversal: each frame stores an interval's
    # starting state. Replaying its first half creates the right checkpoint.
    depth = jnp.iinfo(lower.dtype).bits + 1
    starts = jnp.zeros(depth, dtype=lower.dtype).at[0].set(lower)
    ends = jnp.zeros(depth, dtype=lower.dtype).at[0].set(upper)
    states = jax.tree.map(
        lambda x: jnp.zeros((depth,) + jnp.shape(x), jnp.asarray(x).dtype).at[0].set(x), initial
    )
    constants_bar = jax.tree.map(_zero_cotangent, constants)

    def visit(carry):
        top, starts, ends, states, value_bar, constants_bar = carry
        lo, hi = starts[top], ends[top]
        value = jax.tree.map(lambda x: x[top], states)

        def split(_):
            mid = lo + (hi - lo) // 2
            middle = _run(body, lo, mid, value, constants)
            new_states = jax.tree.map(lambda stack, x: stack.at[top + 1].set(x), states, middle)
            return (
                top + 1,
                starts.at[top + 1].set(mid),
                ends.at[top].set(mid).at[top + 1].set(hi),
                new_states,
                value_bar,
                constants_bar,
            )

        def leaf(_):
            def block(value, constants):
                def step(i, value):
                    return jax.lax.cond(
                        lo + i < hi, lambda x: body(lo + i, x, *constants), lambda x: x, value
                    )

                return jax.lax.fori_loop(0, _BLOCK_SIZE, step, value)

            _, pullback = jax.vjp(block, value, constants)
            initial_bar, added_bar = pullback(value_bar)
            total_bar = jax.tree.map(_add_cotangent, constants_bar, added_bar)
            return top - 1, starts, ends, states, initial_bar, total_bar

        return jax.lax.cond(hi - lo > _BLOCK_SIZE, split, leaf, operand=None)

    result = jax.lax.while_loop(
        lambda c: c[0] >= 0, visit, (jnp.int32(0), starts, ends, states, cotangent, constants_bar)
    )
    return None, None, result[4], result[5]


_checkpointed.defvjp(_forward, _backward)


def checkpointed_fori_loop(lower, upper, body, value):
    """Exact dynamic loop and first-order VJP using O(log N) checkpoint states.

    The body may close over differentiable values: closure conversion makes
    their cotangents explicit. Backward replay costs O(N log N) body work and
    stores only eight local steps, independently of the dynamic trip count.
    This reverse-mode primitive intentionally does not implement forward JVP.
    """
    dtype = jnp.result_type(lower, upper)
    if not jnp.issubdtype(dtype, jnp.integer):
        raise TypeError("Checkpointed loop bounds must have integer dtype")
    lower, upper = jnp.asarray(lower, dtype=dtype), jnp.asarray(upper, dtype=dtype)
    value = jax.tree.map(jnp.asarray, value)
    # Unlike jax.closure_convert, hoist integer/batched constants too. Keeping
    # a BatchTracer in a function closure would leak it when lax.map/vmap
    # transposes the custom VJP (rational branch decisions are integer-valued).
    closed, output_shapes = jax.make_jaxpr(body, return_shape=True)(lower, value)
    output_tree = jax.tree.structure(output_shapes)
    jaxpr = closed.jaxpr

    def converted(index, carry, *constants):
        arguments = jax.tree.leaves((index, carry))
        outputs = jax.core.eval_jaxpr(jaxpr, constants, *arguments)
        return jax.tree.unflatten(output_tree, outputs)

    return _checkpointed(converted, lower, upper, value, closed.consts)
