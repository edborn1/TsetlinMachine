"""End-to-end CUDA certification for the Hypervector Tsetlin Machine.

The script exits with status 0 only when:
  1. PyCUDA initializes an NVIDIA device.
  2. Sparse hypervectors are accepted by the existing CUDA TM backend.
  3. The backend trains and predicts XOR perfectly.
  4. Model/encoder state survives a save/load round trip.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np


def main() -> None:
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401  # creates the CUDA context
    except Exception as exc:  # pragma: no cover - requires a CUDA host
        raise SystemExit(f"CUDA CERTIFICATION FAILED: PyCUDA initialization failed: {exc}")

    from PyCoalescedTsetlinMachineCUDA.hypervector import HypervectorTsetlinMachine

    device = cuda.Context.get_device()
    print(f"CUDA device: {device.name()}")
    print(f"CUDA compute capability: {device.compute_capability()}")

    X = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ],
        dtype=np.uint8,
    )
    y = np.array([0, 1, 1, 0], dtype=np.uint32)

    model = HypervectorTsetlinMachine(
        number_of_clauses=1000,
        T=50,
        s=1.0,
        hv_size=1024,
        n_bits=4,
        seed=42,
    )

    # Train in bounded incremental blocks. This avoids declaring failure merely
    # because a stochastic TM initialization takes a few extra epochs to settle.
    predictions = None
    epochs_used = 0
    for block in range(10):
        model.fit(X, y, epochs=100, incremental=(block > 0))
        epochs_used += 100
        predictions = model.predict(X).astype(np.uint32)
        accuracy = float(np.mean(predictions == y))
        print(
            f"epochs={epochs_used:4d} predictions={predictions.tolist()} "
            f"accuracy={accuracy:.3f}"
        )
        if np.array_equal(predictions, y):
            break

    if predictions is None or not np.array_equal(predictions, y):
        raise SystemExit(
            "CUDA CERTIFICATION FAILED: CUDA backend trained, but XOR did not "
            f"converge after {epochs_used} epochs. Last predictions: "
            f"{None if predictions is None else predictions.tolist()}"
        )

    diagnostics = model.projection_diagnostics()
    assert diagnostics.token_count == X.shape[1]
    assert diagnostics.hv_size == 1024
    assert diagnostics.n_bits == 4
    print(f"Projection diagnostics: {diagnostics}")

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "hvtm_state.pkl"
        model.save(path)

        restored = HypervectorTsetlinMachine(
            number_of_clauses=1000,
            T=50,
            s=1.0,
            hv_size=1024,
            n_bits=4,
            seed=42,
        )
        restored.load(path)
        restored_predictions = restored.predict(X).astype(np.uint32)

    if not np.array_equal(restored_predictions, y):
        raise SystemExit(
            "CUDA CERTIFICATION FAILED: save/load round trip changed predictions: "
            f"{restored_predictions.tolist()}"
        )

    print("State round-trip predictions:", restored_predictions.tolist())
    print("HVTM CUDA CERTIFICATION: PASS")


if __name__ == "__main__":
    main()
