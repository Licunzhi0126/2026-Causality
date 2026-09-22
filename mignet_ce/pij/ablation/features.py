from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mignet_ce.config import TemporalRunConfig
from mignet_ce.networks.base import NetworkContext
from mignet_ce.pij.base import TimePair
from mignet_ce.pij.compare._shared.features import CompareFeatureSet, build_compare_feature_set


@dataclass(frozen=True)
class PairFeatureBlocks:
    cci_source: np.ndarray | None
    cci_target: np.ndarray | None
    grn_source: np.ndarray | None
    grn_target: np.ndarray | None
    cci_representation: str | None
    pairwise_cci_used: bool
    metadata: dict[str, object]


class AblationFeatureProvider:
    """Lazy adapter around the production feature extractors.

    No NMF, Laplacian, or dual-end GRN logic is reimplemented here.  N and L are
    obtained from the existing compare feature builder; G is taken from the exact
    pairwise GRN block already used by the production N+G path.
    """

    def __init__(self, context: NetworkContext, cfg: TemporalRunConfig) -> None:
        self.context = context
        self.cfg = cfg
        self._cache: dict[str, CompareFeatureSet] = {}

    def _feature_set(self, key: str) -> CompareFeatureSet:
        if key not in self._cache:
            self._cache[key] = build_compare_feature_set(
                self.context,
                self.cfg,
                (key,),
                apply_feature_weights=False,
            )
        return self._cache[key]

    @staticmethod
    def _select_pair(feature_set: CompareFeatureSet, side: str, pair: TimePair) -> tuple[np.ndarray, np.ndarray, bool]:
        if side == "lower":
            timewise = feature_set.lower_features
            pairwise = feature_set.pairwise_lower_features
        elif side == "upper":
            timewise = feature_set.upper_features
            pairwise = feature_set.pairwise_upper_features
        else:
            raise ValueError("side must be 'lower' or 'upper'.")
        if pairwise is not None and pair in pairwise:
            source, target = pairwise[pair]
            return np.asarray(source, float), np.asarray(target, float), True
        return (
            np.asarray(timewise[pair[0]], float),
            np.asarray(timewise[pair[1]], float),
            False,
        )

    def _select_grn(self, side: str, pair: TimePair) -> tuple[np.ndarray, np.ndarray]:
        # The production builder exposes the expression-gated GRN block only on
        # the N/light_cci_grn path.  Reuse it verbatim rather than duplicating GRN code.
        feature_set = self._feature_set("N")
        pairwise = (
            feature_set.pairwise_lower_grn_features
            if side == "lower"
            else feature_set.pairwise_upper_grn_features
        )
        if pairwise is None or pair not in pairwise:
            raise ValueError(
                "Controlled GRN ablations require network_method='light_cci_grn' "
                "(or the compatible PGR variant) so the production dual-end GRN block exists."
            )
        source, target = pairwise[pair]
        return np.asarray(source, float), np.asarray(target, float)

    def pair_blocks(
        self,
        *,
        side: str,
        pair: TimePair,
        cci_representation: str | None,
        use_grn: bool,
    ) -> PairFeatureBlocks:
        cci_source = cci_target = None
        pairwise_used = False
        feature_metadata: dict[str, object] = {}
        if cci_representation is not None:
            if cci_representation not in {"N", "L"}:
                raise ValueError(f"Unsupported CCI representation {cci_representation!r}.")
            feature_set = self._feature_set(cci_representation)
            cci_source, cci_target, pairwise_used = self._select_pair(feature_set, side, pair)
            feature_metadata["cci_feature_metadata"] = feature_set.metadata
        grn_source = grn_target = None
        if use_grn:
            grn_source, grn_target = self._select_grn(side, pair)
            feature_metadata["grn_source"] = "production_pairwise_double_end_expression_gated_state"
        return PairFeatureBlocks(
            cci_source=cci_source,
            cci_target=cci_target,
            grn_source=grn_source,
            grn_target=grn_target,
            cci_representation=cci_representation,
            pairwise_cci_used=pairwise_used,
            metadata=feature_metadata,
        )
