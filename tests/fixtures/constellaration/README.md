Constellaration Boozer fixtures reported by Misha Padidar on April 10, 2026.

These two `boozmn` files trigger extremely small `|iota|` on one or more
surfaces. In the legacy NEO rational-surface correction, that implies
`nfp_rat ~= ceil(1 / acc_req / |iota|)`, which can reach millions of field
periods. Without a preflight guard, the run can appear to hang indefinitely.

NEO_JAX regression tests use these fixtures to verify that the solver now fails
fast with a detailed diagnostic instead of silently running for an unbounded
amount of time.
