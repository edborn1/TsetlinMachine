"""Minimal HVTM example using the existing CUDA backend."""

import numpy as np

from PyCoalescedTsetlinMachineCUDA.hypervector import HypervectorTsetlinMachine


# Already-Booleanized input. In a trading pipeline this can be the existing
# thermometer/threshold literal matrix.
X = np.array(
    [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ],
    dtype=np.uint8,
)
y = np.array([0, 1, 1, 0], dtype=np.uint32)  # XOR

model = HypervectorTsetlinMachine(
    number_of_clauses=200,
    T=15,
    s=1.0,       # Reasoning by Elimination, as highlighted in the HVTM paper
    hv_size=1024,
    n_bits=4,
    seed=42,
)
model.fit(X, y, epochs=100)

print("Predictions:", model.predict(X))
print("Projection diagnostics:", model.projection_diagnostics())
