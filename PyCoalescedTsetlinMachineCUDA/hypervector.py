"""Sparse hypervector support for the Coalesced Tsetlin Machine.

This module implements the sparse binary hypervector construction described in
"Exploring Effects of Hyperdimensional Vectors for Tsetlin Machines".

The existing CUDA Tsetlin Machine remains unchanged. Inputs are projected into a
fixed-size sparse Boolean hyperspace before being passed to the existing TM.
Each source token receives a deterministic sparse hypervector containing exactly
``n_bits`` active positions. More complex samples are represented by:

* binding: cyclic permutation of a token hypervector;
* bundling: bitwise OR of bound/token hypervectors.

For already-Booleanized tabular data, each original feature position is treated
as a token, matching the paper's ECFP-style construction: active source bits are
projected and bundled into one fixed-length hypervector.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


ArrayLike = Any
BackendFactory = Callable[..., Any]


def _stable_bytes(value: Hashable) -> bytes:
    """Return a stable byte representation for deterministic token projection."""
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    return repr(value).encode("utf-8")


def _seed_for_token(seed: int, namespace: str, token: Hashable) -> int:
    payload = (
        int(seed).to_bytes(8, byteorder="little", signed=True)
        + namespace.encode("utf-8")
        + b"\x00"
        + _stable_bytes(token)
    )
    digest = hashlib.blake2b(payload, digest_size=16).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


@dataclass(frozen=True)
class ProjectionDiagnostics:
    """Summary of token projection occupancy and collisions."""

    token_count: int
    hv_size: int
    n_bits: int
    occupied_bits: int
    occupancy: float
    collision_count: int
    mean_sources_per_occupied_bit: float
    max_sources_per_bit: int


class SparseHypervectorSpace:
    """Deterministic sparse binary hypervector token space.

    Parameters
    ----------
    hv_size:
        Number of Boolean positions in each hypervector.
    n_bits:
        Number of active projection bits assigned to each token.
    seed:
        Global seed. Token vectors are derived from ``seed + namespace + token``
        rather than token insertion order, so projections are reproducible.
    dtype:
        Boolean-like NumPy dtype used for returned vectors.
    """

    STATE_VERSION = 1

    def __init__(
        self,
        hv_size: int = 8192,
        n_bits: int = 4,
        seed: int = 42,
        dtype: np.dtype = np.uint8,
    ) -> None:
        if hv_size <= 0:
            raise ValueError("hv_size must be positive")
        if n_bits <= 0:
            raise ValueError("n_bits must be positive")
        if n_bits > hv_size:
            raise ValueError("n_bits cannot exceed hv_size")

        self.hv_size = int(hv_size)
        self.n_bits = int(n_bits)
        self.seed = int(seed)
        self.dtype = np.dtype(dtype)
        self._token_indices: Dict[Tuple[str, Hashable], np.ndarray] = {}

    def token_indices(self, token: Hashable, namespace: str = "token") -> np.ndarray:
        """Return active positions for ``token``, creating them if necessary."""
        key = (str(namespace), token)
        cached = self._token_indices.get(key)
        if cached is not None:
            return cached.copy()

        rng = np.random.default_rng(_seed_for_token(self.seed, key[0], token))
        indices = np.sort(
            rng.choice(self.hv_size, size=self.n_bits, replace=False).astype(np.int32)
        )
        self._token_indices[key] = indices
        return indices.copy()

    def token(self, token: Hashable, namespace: str = "token") -> np.ndarray:
        """Return the sparse hypervector assigned to ``token``."""
        hv = np.zeros(self.hv_size, dtype=self.dtype)
        hv[self.token_indices(token, namespace=namespace)] = 1
        return hv

    def permute(self, hypervector: ArrayLike, shift: int) -> np.ndarray:
        """Bind by cyclically shifting a hypervector."""
        hv = self._validate_hypervector(hypervector)
        return np.roll(hv, int(shift) % self.hv_size)

    def role_shift(self, role: Hashable, namespace: str = "role") -> int:
        """Derive a deterministic cyclic-shift amount for a semantic role."""
        return _seed_for_token(self.seed, namespace, role) % self.hv_size

    def bind(
        self,
        hypervector: ArrayLike,
        role: Optional[Hashable] = None,
        *,
        shift: Optional[int] = None,
        role_namespace: str = "role",
    ) -> np.ndarray:
        """Bind a hypervector to a role using a cyclic permutation.

        Exactly one of ``role`` or ``shift`` may be supplied. If neither is
        supplied, a one-position cyclic shift is used.
        """
        if role is not None and shift is not None:
            raise ValueError("provide role or shift, not both")
        if shift is None:
            shift = 1 if role is None else self.role_shift(role, role_namespace)
        return self.permute(hypervector, shift)

    def bundle(self, hypervectors: Iterable[ArrayLike]) -> np.ndarray:
        """Bundle hypervectors using logical OR."""
        output = np.zeros(self.hv_size, dtype=self.dtype)
        count = 0
        for hypervector in hypervectors:
            output |= self._validate_hypervector(hypervector).astype(self.dtype, copy=False)
            count += 1
        if count == 0:
            return output
        return output

    def encode_set(
        self,
        tokens: Iterable[Hashable],
        namespace: str = "token",
    ) -> np.ndarray:
        """Bundle an unordered set/multiset of concept tokens."""
        return self.bundle(self.token(token, namespace=namespace) for token in tokens)

    def encode_sequence(
        self,
        tokens: Sequence[Hashable],
        namespace: str = "token",
        position_namespace: str = "position",
    ) -> np.ndarray:
        """Encode a sequence by binding each token to its position, then OR-bundling."""
        return self.bundle(
            self.bind(
                self.token(token, namespace=namespace),
                role=position,
                role_namespace=position_namespace,
            )
            for position, token in enumerate(tokens)
        )

    def encode_record(
        self,
        record: Mapping[Hashable, Hashable],
        value_namespace: str = "value",
        role_namespace: str = "field",
    ) -> np.ndarray:
        """Encode field/value pairs through role binding and bundling."""
        return self.bundle(
            self.bind(
                self.token((field, value), namespace=value_namespace),
                role=field,
                role_namespace=role_namespace,
            )
            for field, value in record.items()
        )

    def bit_sources(self, bit: int) -> List[Tuple[str, Hashable]]:
        """Return all registered source tokens projecting onto a hypervector bit."""
        bit = int(bit)
        if bit < 0 or bit >= self.hv_size:
            raise IndexError("bit outside hypervector")
        return [key for key, indices in self._token_indices.items() if bit in indices]

    def diagnostics(self) -> ProjectionDiagnostics:
        counts = np.zeros(self.hv_size, dtype=np.int32)
        for indices in self._token_indices.values():
            counts[indices] += 1
        occupied = counts > 0
        occupied_bits = int(occupied.sum())
        total_projections = len(self._token_indices) * self.n_bits
        collision_count = int(total_projections - occupied_bits)
        mean_sources = float(counts[occupied].mean()) if occupied_bits else 0.0
        max_sources = int(counts.max()) if occupied_bits else 0
        return ProjectionDiagnostics(
            token_count=len(self._token_indices),
            hv_size=self.hv_size,
            n_bits=self.n_bits,
            occupied_bits=occupied_bits,
            occupancy=occupied_bits / self.hv_size,
            collision_count=collision_count,
            mean_sources_per_occupied_bit=mean_sources,
            max_sources_per_bit=max_sources,
        )

    def state_dict(self) -> Dict[str, Any]:
        """Return a serializable state dictionary."""
        return {
            "state_version": self.STATE_VERSION,
            "hv_size": self.hv_size,
            "n_bits": self.n_bits,
            "seed": self.seed,
            "dtype": self.dtype.str,
            "tokens": [
                {
                    "namespace": namespace,
                    "token": token,
                    "indices": indices.copy(),
                }
                for (namespace, token), indices in self._token_indices.items()
            ],
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "SparseHypervectorSpace":
        version = int(state.get("state_version", 0))
        if version != cls.STATE_VERSION:
            raise ValueError(f"unsupported hypervector state version: {version}")
        space = cls(
            hv_size=int(state["hv_size"]),
            n_bits=int(state["n_bits"]),
            seed=int(state["seed"]),
            dtype=np.dtype(state["dtype"]),
        )
        for item in state.get("tokens", []):
            key = (str(item["namespace"]), item["token"])
            indices = np.asarray(item["indices"], dtype=np.int32)
            if indices.shape != (space.n_bits,):
                raise ValueError("invalid token projection shape in state")
            if np.any(indices < 0) or np.any(indices >= space.hv_size):
                raise ValueError("token projection outside hypervector in state")
            space._token_indices[key] = np.sort(indices)
        return space

    def _validate_hypervector(self, hypervector: ArrayLike) -> np.ndarray:
        hv = np.asarray(hypervector)
        if hv.ndim != 1 or hv.shape[0] != self.hv_size:
            raise ValueError(f"expected one-dimensional hypervector of size {self.hv_size}")
        if not np.all((hv == 0) | (hv == 1)):
            raise ValueError("hypervectors must be Boolean (0/1)")
        return hv.astype(self.dtype, copy=False)


class BinaryFeatureHypervectorEncoder:
    """Project Boolean feature matrices into sparse hypervectors.

    Every original Boolean feature position is assigned a sparse token HV. For a
    sample, tokens corresponding to active source features are OR-bundled. The
    existing TM backend may append negated hyperliterals, allowing absence to be
    used directly and supporting Reasoning by Elimination when ``s=1``.
    """

    STATE_VERSION = 1

    def __init__(
        self,
        hv_size: int = 8192,
        n_bits: int = 4,
        seed: int = 42,
        feature_names: Optional[Sequence[Hashable]] = None,
        space: Optional[SparseHypervectorSpace] = None,
    ) -> None:
        self.space = space or SparseHypervectorSpace(hv_size, n_bits, seed)
        if space is not None and (
            space.hv_size != hv_size or space.n_bits != n_bits or space.seed != seed
        ):
            # Supplying a space is authoritative; explicit defaults should not
            # make loading an existing encoder fail.
            hv_size, n_bits, seed = space.hv_size, space.n_bits, space.seed
        self.hv_size = int(hv_size)
        self.n_bits = int(n_bits)
        self.seed = int(seed)
        self.feature_names = list(feature_names) if feature_names is not None else None
        self.n_features_in_: Optional[int] = None
        self._feature_indices: Optional[List[np.ndarray]] = None

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> "BinaryFeatureHypervectorEncoder":
        Xb = self._validate_binary_matrix(X)
        self.n_features_in_ = int(Xb.shape[1])
        if self.feature_names is None:
            self.feature_names = list(range(self.n_features_in_))
        if len(self.feature_names) != self.n_features_in_:
            raise ValueError("feature_names length must match X.shape[1]")
        self._feature_indices = [
            self.space.token_indices(name, namespace="feature")
            for name in self.feature_names
        ]
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        Xb = self._validate_binary_matrix(X)
        if self.n_features_in_ is None or self._feature_indices is None:
            raise RuntimeError("encoder must be fitted before transform")
        if Xb.shape[1] != self.n_features_in_:
            raise ValueError("X feature count differs from fitted encoder")

        output = np.zeros((Xb.shape[0], self.hv_size), dtype=np.uint8)
        # Sparse-by-row accumulation avoids constructing an n_samples x n_features
        # x hv_size tensor.
        for row_index, active_features in enumerate(Xb.astype(bool)):
            for feature_index in np.flatnonzero(active_features):
                output[row_index, self._feature_indices[int(feature_index)]] = 1
        return output

    def fit_transform(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> np.ndarray:
        return self.fit(X, y=y).transform(X)

    def feature_projection(self, feature: Union[int, Hashable]) -> np.ndarray:
        """Return the sparse token HV for a fitted source feature."""
        if self.n_features_in_ is None or self.feature_names is None:
            raise RuntimeError("encoder must be fitted first")
        if isinstance(feature, (int, np.integer)) and 0 <= int(feature) < self.n_features_in_:
            name = self.feature_names[int(feature)]
        else:
            name = feature
            if name not in self.feature_names:
                raise KeyError(f"unknown feature: {feature!r}")
        return self.space.token(name, namespace="feature")

    def bit_sources(self, bit: int) -> List[Hashable]:
        """Decode a hyperliteral bit to the source features that project onto it."""
        return [token for namespace, token in self.space.bit_sources(bit) if namespace == "feature"]

    def state_dict(self) -> Dict[str, Any]:
        if self.n_features_in_ is None:
            raise RuntimeError("cannot serialize an unfitted encoder")
        return {
            "state_version": self.STATE_VERSION,
            "space": self.space.state_dict(),
            "feature_names": self.feature_names,
            "n_features_in": self.n_features_in_,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "BinaryFeatureHypervectorEncoder":
        version = int(state.get("state_version", 0))
        if version != cls.STATE_VERSION:
            raise ValueError(f"unsupported encoder state version: {version}")
        space = SparseHypervectorSpace.from_state_dict(state["space"])
        encoder = cls(
            hv_size=space.hv_size,
            n_bits=space.n_bits,
            seed=space.seed,
            feature_names=state["feature_names"],
            space=space,
        )
        encoder.n_features_in_ = int(state["n_features_in"])
        encoder._feature_indices = [
            encoder.space.token_indices(name, namespace="feature")
            for name in encoder.feature_names or []
        ]
        if len(encoder._feature_indices) != encoder.n_features_in_:
            raise ValueError("encoder state has inconsistent feature count")
        return encoder

    @staticmethod
    def _validate_binary_matrix(X: ArrayLike) -> np.ndarray:
        Xb = np.asarray(X)
        if Xb.ndim != 2:
            raise ValueError("X must be a two-dimensional feature matrix")
        if not np.all((Xb == 0) | (Xb == 1)):
            raise ValueError("X must contain only Boolean 0/1 values")
        return Xb.astype(np.uint8, copy=False)


class _HypervectorTMBase:
    """Shared adapter logic around an existing Tsetlin Machine backend."""

    STATE_VERSION = 1

    def __init__(
        self,
        number_of_clauses: int,
        T: int,
        s: float = 1.0,
        *,
        hv_size: int = 8192,
        n_bits: int = 4,
        seed: int = 42,
        feature_names: Optional[Sequence[Hashable]] = None,
        encoder: Optional[BinaryFeatureHypervectorEncoder] = None,
        backend_factory: Optional[BackendFactory] = None,
        backend_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if s <= 0:
            raise ValueError("s must be positive")
        self.number_of_clauses = int(number_of_clauses)
        self.T = int(T)
        self.s = float(s)
        self.encoder = encoder or BinaryFeatureHypervectorEncoder(
            hv_size=hv_size,
            n_bits=n_bits,
            seed=seed,
            feature_names=feature_names,
        )
        self.backend_factory = backend_factory
        self.backend_kwargs = dict(backend_kwargs or {})
        self.backend: Optional[Any] = None

    @property
    def reasoning_by_elimination(self) -> bool:
        """Whether the paper's RbE configuration (specificity s=1) is active."""
        return bool(np.isclose(self.s, 1.0))

    def _ensure_backend(self) -> Any:
        if self.backend is None:
            factory = self.backend_factory or self._default_backend_factory()
            self.backend = factory(
                self.number_of_clauses,
                self.T,
                self.s,
                **self.backend_kwargs,
            )
        return self.backend

    def fit(self, X: ArrayLike, y: ArrayLike, epochs: int = 100, incremental: bool = False):
        if incremental and self.encoder.n_features_in_ is None:
            raise ValueError("incremental fitting requires an already-fitted encoder")
        X_hv = self.encoder.transform(X) if incremental else self.encoder.fit_transform(X, y)
        backend = self._ensure_backend()
        backend.fit(X_hv, np.asarray(y), epochs=epochs, incremental=incremental)
        return self

    def score(self, X: ArrayLike) -> np.ndarray:
        return np.asarray(self._ensure_backend().score(self.encoder.transform(X)))

    def predict(self, X: ArrayLike) -> np.ndarray:
        return np.asarray(self._ensure_backend().predict(self.encoder.transform(X)))

    def transform(self, X: ArrayLike) -> np.ndarray:
        backend = self._ensure_backend()
        if not hasattr(backend, "transform"):
            raise AttributeError("backend does not expose transform")
        return np.asarray(backend.transform(self.encoder.transform(X)))

    def get_state(self) -> Dict[str, Any]:
        backend = self._ensure_backend()
        if not hasattr(backend, "get_state"):
            raise AttributeError("backend does not expose get_state")
        return {
            "state_version": self.STATE_VERSION,
            "wrapper_class": type(self).__name__,
            "number_of_clauses": self.number_of_clauses,
            "T": self.T,
            "s": self.s,
            "backend_kwargs": self.backend_kwargs,
            "encoder": self.encoder.state_dict(),
            "backend_state": backend.get_state(),
        }

    def set_state(self, state: Mapping[str, Any]) -> "_HypervectorTMBase":
        version = int(state.get("state_version", 0))
        if version != self.STATE_VERSION:
            raise ValueError(f"unsupported HVTM state version: {version}")
        self.number_of_clauses = int(state["number_of_clauses"])
        self.T = int(state["T"])
        self.s = float(state["s"])
        self.backend_kwargs = dict(state.get("backend_kwargs", {}))
        self.encoder = BinaryFeatureHypervectorEncoder.from_state_dict(state["encoder"])
        backend = self._ensure_backend()
        backend.set_state(state["backend_state"])
        return self

    def save(self, path: Union[str, Path]) -> None:
        """Persist encoder projections and backend state to a pickle file."""
        path = Path(path)
        with path.open("wb") as handle:
            pickle.dump(self.get_state(), handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: Union[str, Path]) -> "_HypervectorTMBase":
        path = Path(path)
        with path.open("rb") as handle:
            state = pickle.load(handle)
        return self.set_state(state)

    def projection_diagnostics(self) -> ProjectionDiagnostics:
        return self.encoder.space.diagnostics()

    def _default_backend_factory(self) -> BackendFactory:
        raise NotImplementedError


class HypervectorTsetlinMachine(_HypervectorTMBase):
    """Binary HVTM backed by the existing CUDA ``TsetlinMachine``."""

    def predict(self, X: ArrayLike) -> np.ndarray:
        # Use the score directly. This also avoids scalar conversion in older
        # versions of the repository's binary backend predict method.
        return (self.score(X) >= 0).astype(np.uint32)

    def _default_backend_factory(self) -> BackendFactory:
        from .tm import TsetlinMachine

        return TsetlinMachine


class HypervectorMultiClassTsetlinMachine(_HypervectorTMBase):
    """Multiclass HVTM backed by the existing CUDA ``MultiClassTsetlinMachine``."""

    def _default_backend_factory(self) -> BackendFactory:
        from .tm import MultiClassTsetlinMachine

        return MultiClassTsetlinMachine


class HypervectorMultiOutputTsetlinMachine(_HypervectorTMBase):
    """Multi-output HVTM backed by the existing CUDA ``MultiOutputTsetlinMachine``."""

    def _default_backend_factory(self) -> BackendFactory:
        from .tm import MultiOutputTsetlinMachine

        return MultiOutputTsetlinMachine
