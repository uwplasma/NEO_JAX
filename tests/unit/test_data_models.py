import jax

from neo_jax.data_models import BoozerData, NeoInputs, NeoOutputs, VmecData


def test_dataclasses_are_pytrees():
    vmec = VmecData(state=jax.numpy.array([1.0]))
    booz = BoozerData(data=jax.numpy.array([2.0]))
    inputs = NeoInputs(config={"theta_n": 3}, surfaces=jax.numpy.array([0.5]))
    outputs = NeoOutputs(
        eps_eff=jax.numpy.array([1.0]),
        eps_par=jax.numpy.array([2.0]),
        eps_tot=jax.numpy.array([3.0]),
        ctr_one=jax.numpy.array([4.0]),
        ctr_tot=jax.numpy.array([5.0]),
        diagnostics={"foo": jax.numpy.array([6.0])},
    )

    for obj in (vmec, booz, inputs, outputs):
        leaves, treedef = jax.tree_util.tree_flatten(obj)
        rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
        assert rebuilt == obj
