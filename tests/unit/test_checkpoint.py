import jax
import jax.numpy as jnp
import numpy as np
import pytest

from neo_jax._checkpoint import checkpointed_fori_loop


@pytest.mark.parametrize("count", [0, 1, 8, 9, 37, 100])
def test_checkpointed_loop_value_and_reverse(count):
    def evaluate(x, checkpointed):
        def body(i, carry):
            value, counter = carry
            return jnp.sin(value * x[0]) + x[1] * (i + 1) / 100, counter + 1

        if checkpointed:
            return checkpointed_fori_loop(0, jnp.int32(count), body, (x[2], jnp.int32(0)))[0]
        return jax.lax.fori_loop(0, count, body, (x[2], jnp.int32(0)))[0]

    x = jnp.array([0.9, 0.2, 0.1])
    expected = jax.value_and_grad(lambda x: evaluate(x, False))(x)
    actual = jax.jit(jax.value_and_grad(lambda x: evaluate(x, True)))(x)
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-12)


def test_nested_checkpointed_loop_under_vmap():
    def objective(x, replay):
        def outer(i, value):
            count = jnp.asarray(2 * i + 1, jnp.int32)

            def inner(j, value):
                return jnp.sin(value * x) + jnp.asarray(j + 1, dtype=value.dtype) / 100

            return (
                checkpointed_fori_loop(0, count, inner, value)
                if replay
                else jax.lax.fori_loop(0, 2 * i + 1, inner, value)
            )

        if replay:
            return checkpointed_fori_loop(0, jnp.int32(4), outer, x)
        value = x
        for i in range(4):
            for j in range(2 * i + 1):
                value = jnp.sin(value * x) + jnp.asarray(j + 1, dtype=value.dtype) / 100
        return value

    x = jnp.array([0.2, 0.7])
    actual = jax.jit(jax.grad(lambda x: jax.vmap(lambda a: objective(a, True))(x).sum()))(x)
    expected = jax.grad(lambda x: jax.vmap(lambda a: objective(a, False))(x).sum())(x)
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)


def test_checkpointed_loop_preserves_int64_bounds():
    lower, upper = jnp.int64(2**31 + 1), jnp.int64(2**31 + 10)

    def objective(x):
        return checkpointed_fori_loop(
            lower, upper, lambda i, y: y + x * i.astype(x.dtype) / 2**31, jnp.array(0.0)
        )

    value, derivative = jax.jit(jax.value_and_grad(objective))(jnp.array(0.5))
    expected = np.arange(int(lower), int(upper), dtype=np.int64).sum() / 2**31
    np.testing.assert_allclose(value, 0.5 * expected, rtol=1e-14)
    np.testing.assert_allclose(derivative, expected, rtol=1e-14)
