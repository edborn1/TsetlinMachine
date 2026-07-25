# Hypervector Tsetlin Machine extension

This branch adds a sparse Hypervector Tsetlin Machine (HVTM) front end to the
existing CUDA Coalesced Tsetlin Machine. The implementation follows the core
construction in *Exploring Effects of Hyperdimensional Vectors for Tsetlin
Machines*:

- each source concept receives a random-looking, deterministic sparse binary
  hypervector;
- **binding** is implemented by cyclic permutation;
- **bundling** is implemented by bitwise OR;
- the resulting Boolean hypervector is learned by the existing TM without
  changing its CUDA kernels;
- `s=1.0` is the default, exposing the paper's Reasoning by Elimination (RbE)
  setting for sparse hyperspaces.

## Existing Boolean feature matrices

The most direct path for the current trading TM is to keep the frozen
Booleanization/thermometer features and project each active source literal into
hyperspace:

```python
from PyCoalescedTsetlinMachineCUDA.hypervector import (
    HypervectorMultiClassTsetlinMachine,
)

model = HypervectorMultiClassTsetlinMachine(
    number_of_clauses=2000,
    T=15,
    s=1.0,
    hv_size=8192,
    n_bits=4,
    seed=42,
    feature_names=feature_names,
)
model.fit(X_binary_train, y_train, epochs=20)
prediction = model.predict(X_binary_test)
```

Every source feature position gets a sparse token. A sample is the OR-bundle of
all active feature tokens, mirroring the paper's ECFP experiment. The backend
may append negated hyperliterals, so absent source evidence remains available to
clauses.

## Generic hypervector construction

```python
from PyCoalescedTsetlinMachineCUDA.hypervector import SparseHypervectorSpace

space = SparseHypervectorSpace(hv_size=8192, n_bits=4, seed=42)

set_hv = space.encode_set(["MACD_up", "spread_low", "RTH"])
sequence_hv = space.encode_sequence(["up", "up", "down"])
record_hv = space.encode_record({"symbol": "AAPL", "session": "RTH"})
```

`encode_sequence` and `encode_record` bind tokens to positions/roles with a
deterministic cyclic shift before OR-bundling.

## State and explainability

The token dictionary is part of model state and is saved together with the TM
state:

```python
model.save("hvtm.pkl")
model.load("hvtm.pkl")
```

For collision inspection and interpretation:

```python
print(model.projection_diagnostics())
print(model.encoder.bit_sources(123))
```

Because sparse projections can overlap, one hyperliteral may decode to more than
one original feature. This is reported rather than hidden.

## Tests

The encoder and wrapper tests do not require CUDA because they inject a fake TM
backend:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

The actual model still requires the repository's PyCUDA/CUDA environment.
