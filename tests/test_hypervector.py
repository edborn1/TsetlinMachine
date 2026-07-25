import tempfile
import unittest

import numpy as np

from PyCoalescedTsetlinMachineCUDA.hypervector import (
    BinaryFeatureHypervectorEncoder,
    HypervectorMultiClassTsetlinMachine,
    SparseHypervectorSpace,
)


class FakeBackend:
    def __init__(self, number_of_clauses, T, s, **kwargs):
        self.number_of_clauses = number_of_clauses
        self.T = T
        self.s = s
        self.kwargs = kwargs
        self.fit_X = None
        self.fit_y = None
        self.state = None

    def fit(self, X, y, epochs=100, incremental=False):
        self.fit_X = np.asarray(X).copy()
        self.fit_y = np.asarray(y).copy()
        self.epochs = epochs
        self.incremental = incremental

    def score(self, X):
        X = np.asarray(X)
        return np.vstack((X.sum(axis=1), -X.sum(axis=1)))

    def predict(self, X):
        return np.argmax(self.score(X), axis=0)

    def transform(self, X):
        return np.asarray(X)

    def get_state(self):
        return {"marker": 7, "fit_y": self.fit_y}

    def set_state(self, state):
        self.state = state


class SparseHypervectorSpaceTests(unittest.TestCase):
    def test_token_is_deterministic_and_sparse(self):
        a = SparseHypervectorSpace(hv_size=128, n_bits=4, seed=11)
        b = SparseHypervectorSpace(hv_size=128, n_bits=4, seed=11)
        va = a.token("price")
        vb = b.token("price")
        np.testing.assert_array_equal(va, vb)
        self.assertEqual(int(va.sum()), 4)

    def test_binding_is_cyclic_shift(self):
        space = SparseHypervectorSpace(hv_size=32, n_bits=3, seed=3)
        token = space.token("x")
        np.testing.assert_array_equal(space.bind(token, shift=5), np.roll(token, 5))

    def test_bundling_is_or(self):
        space = SparseHypervectorSpace(hv_size=64, n_bits=4, seed=3)
        a = space.token("a")
        b = space.token("b")
        expected = np.bitwise_or(a, b)
        np.testing.assert_array_equal(space.bundle([a, b]), expected)

    def test_sequence_binding_changes_position(self):
        space = SparseHypervectorSpace(hv_size=128, n_bits=4, seed=4)
        first = space.encode_sequence(["up", "down"])
        second = space.encode_sequence(["down", "up"])
        self.assertFalse(np.array_equal(first, second))

    def test_state_round_trip(self):
        space = SparseHypervectorSpace(hv_size=64, n_bits=4, seed=9)
        space.token("alpha")
        space.token("beta", namespace="role")
        restored = SparseHypervectorSpace.from_state_dict(space.state_dict())
        np.testing.assert_array_equal(space.token("alpha"), restored.token("alpha"))
        np.testing.assert_array_equal(
            space.token("beta", namespace="role"),
            restored.token("beta", namespace="role"),
        )


class BinaryFeatureEncoderTests(unittest.TestCase):
    def test_transform_bundles_active_features(self):
        X = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.uint8)
        encoder = BinaryFeatureHypervectorEncoder(
            hv_size=128,
            n_bits=4,
            seed=5,
            feature_names=["a", "b", "c"],
        )
        H = encoder.fit_transform(X)
        expected_0 = np.bitwise_or(
            encoder.feature_projection("a"), encoder.feature_projection("c")
        )
        expected_1 = encoder.feature_projection("b")
        np.testing.assert_array_equal(H[0], expected_0)
        np.testing.assert_array_equal(H[1], expected_1)

    def test_non_binary_input_rejected(self):
        encoder = BinaryFeatureHypervectorEncoder()
        with self.assertRaises(ValueError):
            encoder.fit(np.array([[0.0, 0.5]]))

    def test_encoder_state_round_trip(self):
        X = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        encoder = BinaryFeatureHypervectorEncoder(hv_size=64, n_bits=3, seed=7)
        original = encoder.fit_transform(X)
        restored = BinaryFeatureHypervectorEncoder.from_state_dict(encoder.state_dict())
        np.testing.assert_array_equal(original, restored.transform(X))


class HypervectorTMWrapperTests(unittest.TestCase):
    def test_wrapper_uses_rbe_and_existing_backend_interface(self):
        X = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.uint8)
        y = np.array([0, 1], dtype=np.uint32)
        model = HypervectorMultiClassTsetlinMachine(
            100,
            15,
            hv_size=64,
            n_bits=4,
            backend_factory=FakeBackend,
        )
        model.fit(X, y, epochs=2)
        self.assertTrue(model.reasoning_by_elimination)
        self.assertEqual(model.backend.fit_X.shape, (2, 64))
        self.assertEqual(model.backend.s, 1.0)
        self.assertEqual(model.predict(X).shape, (2,))

    def test_model_state_can_be_saved_and_loaded(self):
        X = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        y = np.array([0, 1], dtype=np.uint32)
        model = HypervectorMultiClassTsetlinMachine(
            20,
            10,
            hv_size=32,
            n_bits=2,
            backend_factory=FakeBackend,
        ).fit(X, y, epochs=1)
        with tempfile.NamedTemporaryFile(suffix=".pkl") as handle:
            model.save(handle.name)
            restored = HypervectorMultiClassTsetlinMachine(
                20,
                10,
                hv_size=32,
                n_bits=2,
                backend_factory=FakeBackend,
            ).load(handle.name)
        np.testing.assert_array_equal(
            model.encoder.transform(X), restored.encoder.transform(X)
        )
        self.assertEqual(restored.backend.state["marker"], 7)


if __name__ == "__main__":
    unittest.main()
