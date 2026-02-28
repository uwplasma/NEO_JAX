import numpy as np

from neo_jax import NeoOutputs, neo_outputs_to_results


def test_neo_outputs_to_results():
    outputs = NeoOutputs(
        eps_eff=np.array([1.0, 2.0]),
        eps_par=np.array([[0.1, 0.2], [0.3, 0.4]]),
        eps_tot=np.array([1.0, 2.0]),
        ctr_one=np.array([0.5, 0.6]),
        ctr_tot=np.array([0.7, 0.8]),
        diagnostics={
            "s": np.array([0.2, 0.8]),
            "r_eff": np.array([0.1, 0.3]),
            "iota": np.array([0.9, 1.1]),
            "b_ref": np.array([1.2, 1.3]),
            "r_ref": np.array([1.4, 1.5]),
            "bareph": np.array([0.01, 0.02]),
            "barept": np.array([0.03, 0.04]),
            "yps": np.array([0.05, 0.06]),
            "flux_index": np.array([5, 7]),
        },
    )

    results = neo_outputs_to_results(outputs)
    assert len(results) == 2
    assert np.allclose(results.epsilon_effective, [1.0, 2.0])
    assert results[0].flux_index == 5
    assert np.isclose(results[1].iota, 1.1)
