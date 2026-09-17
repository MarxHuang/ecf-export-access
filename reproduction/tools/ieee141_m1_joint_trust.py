"""Q/method/side/rho-band joint applicability-domain helpers for EQ067.

The old trust gate compared a direction library and an exact active-set
library independently.  That permits a cross-product of combinations that
was never observed and rejects legitimate nearby combinations.  This module
stores one joint cluster per preregistered (Q, method, side, rho-band) group.
Membership requires all of the following against the same cluster:

* normalized-allocation direction distance <= 0.025;
* per-family active-set Jaccard distance <= 0.10;
* core/halo containment;
* cluster-local scalar feature ranges.

The helper is proxy-only.  It never reads AC truth, pass/fail, failure IDs or
mitigation output.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from r4r.serialization import canonical_hash
from tools.ieee141_m1_rho_policy import (
    LEGACY_RHO_BANDS,
    LEGACY_RHO_POLICY,
    RhoBandPolicy,
    band_json,
    policy_from_payload,
)

SIGNATURE_FAMILIES = ("branch_from_mva", "branch_to_mva", "voltage_lower_pu", "voltage_upper_pu")
FEATURE_NAMES = (
    "total_export_normalized",
    "branch_loading_ratio",
    "voltage_lower_slack_pu",
    "voltage_upper_slack_pu",
    "proxy_effect_norm",
    "proxy_effect_gain_per_rho",
)
DEFAULT_DIRECTION_MODE = "ALLOCATION_CAPACITY"
PROXY_ELECTRICAL_EFFECT_DIRECTION_MODE = "PROXY_ELECTRICAL_EFFECT"
PROXY_VOLTAGE_NORMALIZATION_PU = 0.05
# Public compatibility alias used by the historical V7--V11 artifacts.  New
# callers pass an explicit ``RhoBandPolicy`` instead of mutating this value.
RHO_BANDS = LEGACY_RHO_BANDS
JOINT_DIRECTION_DISTANCE_CAP = 0.025
JOINT_JACCARD_DISTANCE_CAP = 0.10
MIN_EVIDENCE_SOURCE_ROWS = 2
MIN_SOURCE_DIVERSITY_PROFILES = 2
MIN_SOURCE_DIVERSITY_FAMILIES = 2


def rho_band(rho: float, *, q_mode: str = "Q0", rho_policy: RhoBandPolicy | None = None) -> str | None:
    """Return a band under the supplied policy, or legacy B4=3.5 by default."""
    value = float(rho)
    policy = rho_policy or LEGACY_RHO_POLICY
    bands = policy.bands_for(q_mode)
    for index, (name, lower, upper) in enumerate(bands):
        if lower <= value < upper or (index == len(bands) - 1 and lower <= value <= upper):
            return name
    return None


def _library_rho_policy(library: Mapping[str, Any]) -> tuple[RhoBandPolicy | None, str | None]:
    """Resolve a serialized explicit policy without upgrading legacy libraries.

    ``None, None`` means an unmodified legacy library.  A malformed explicit
    policy is reported separately so screening can fail closed rather than
    falling back to the historical bands.
    """
    raw_policy = library.get("rho_policy")
    declared_hash = library.get("rho_policy_hash")
    if raw_policy is None and declared_hash is None:
        return None, None
    if not isinstance(raw_policy, Mapping) or not isinstance(declared_hash, str):
        return None, "JOINT_CLUSTER_RHO_POLICY_INVALID"
    try:
        # A joint-trust library only binds the registered PAR068 domain.  Do
        # not allow a correctly hashed but unrelated parameter object to be
        # substituted as a look-alike rho policy.
        policy = policy_from_payload(raw_policy, parameter_id="PAR068")
    except ValueError:
        return None, "JOINT_CLUSTER_RHO_POLICY_INVALID"
    if policy.policy_hash != declared_hash:
        return None, "JOINT_CLUSTER_RHO_POLICY_MISMATCH"
    return policy, None


def normalized_direction(allocation: Sequence[float], capacity: Sequence[float], rho: float | None = None) -> list[float]:
    values = [float(x) for x in allocation]
    caps = [float(c) for c in capacity]
    radial = math.sqrt(math.fsum((x / c) ** 2 for x, c in zip(values, caps))) if rho is None else float(rho)
    return [0.0] * len(values) if radial <= 1.0e-12 else [x / c / radial for x, c in zip(values, caps)]


def electrical_effect_direction(
    *,
    predicted: Mapping[str, Any],
    baseline: Mapping[str, Any] | None,
    rates: Sequence[float],
    voltage_normalization_pu: float = PROXY_VOLTAGE_NORMALIZATION_PU,
) -> tuple[list[float], float]:
    """Return a normalized proxy electrical-effect direction.

    The direction is formed from *proxy deltas* relative to a preregistered
    operating-point baseline.  It is deliberately not an AC sensitivity or a
    physical Jacobian: the source is the frozen affine proxy only.  Missing
    baselines fail closed so a caller cannot silently fall back to a raw
    allocation direction and claim electrical coverage.
    """
    if not isinstance(baseline, Mapping):
        raise ValueError("PROXY_ELECTRICAL_EFFECT requires an immutable proxy baseline")
    try:
        n = len(rates)
        pf = [float(v) for v in predicted["branch_p_from_mw"]]
        qf = [float(v) for v in predicted["branch_q_from_mvar"]]
        pt = [float(v) for v in predicted["branch_p_to_mw"]]
        qt = [float(v) for v in predicted["branch_q_to_mvar"]]
        vv = [float(v) for v in predicted["voltage_pu"]]
        bpf = [float(v) for v in baseline["branch_p_from_mw"]]
        bqf = [float(v) for v in baseline["branch_q_from_mvar"]]
        bpt = [float(v) for v in baseline["branch_p_to_mw"]]
        bqt = [float(v) for v in baseline["branch_q_to_mvar"]]
        bvv = [float(v) for v in baseline["voltage_pu"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("proxy electrical-effect baseline/predicted vectors are malformed") from exc
    if len(pf) != n or len(qf) != n or len(pt) != n or len(qt) != n or len(bpf) != n or len(bqf) != n or len(bpt) != n or len(bqt) != n or len(vv) != len(bvv):
        raise ValueError("proxy electrical-effect vectors are length-inconsistent")
    if any(float(rate) <= 0.0 or not math.isfinite(float(rate)) for rate in rates):
        raise ValueError("PROXY_ELECTRICAL_EFFECT requires positive finite RATE values")
    if not math.isfinite(float(voltage_normalization_pu)) or float(voltage_normalization_pu) <= 0.0:
        raise ValueError("proxy voltage normalization must be positive and finite")
    vector = [
        *[(a - b) / float(rate) for a, b, rate in zip(pf, bpf, rates)],
        *[(a - b) / float(rate) for a, b, rate in zip(qf, bqf, rates)],
        *[(a - b) / float(rate) for a, b, rate in zip(pt, bpt, rates)],
        *[(a - b) / float(rate) for a, b, rate in zip(qt, bqt, rates)],
        *[(a - b) / float(voltage_normalization_pu) for a, b in zip(vv, bvv)],
    ]
    if any(not math.isfinite(value) for value in vector):
        raise ValueError("proxy electrical-effect direction contains non-finite values")
    norm = math.sqrt(math.fsum(value * value for value in vector))
    return ([0.0] * len(vector) if norm <= 1.0e-12 else [value / norm for value in vector]), norm


def signature_from_proxy(*, predicted: Mapping[str, Any], rates: Sequence[float], lower: float, upper: float) -> dict[str, list[int]]:
    branch_from = [math.hypot(float(predicted["branch_p_from_mw"][i]), float(predicted["branch_q_from_mvar"][i])) for i in range(len(rates))]
    branch_to = [math.hypot(float(predicted["branch_p_to_mw"][i]), float(predicted["branch_q_to_mvar"][i])) for i in range(len(rates))]
    voltage = [float(value) for value in predicted["voltage_pu"]]
    return {
        "branch_from_mva": [i + 1 for i, value in enumerate(branch_from) if value / float(rates[i]) >= 0.95],
        "branch_to_mva": [i + 1 for i, value in enumerate(branch_to) if value / float(rates[i]) >= 0.95],
        "voltage_lower_pu": [i + 1 for i, value in enumerate(voltage) if value - float(lower) <= 0.005],
        "voltage_upper_pu": [i + 1 for i, value in enumerate(voltage) if float(upper) - value <= 0.005],
    }


def proxy_features(
    *,
    predicted: Mapping[str, Any],
    allocation: Sequence[float],
    capacity: Sequence[float],
    rates: Sequence[float],
    lower: float,
    upper: float,
    direction_mode: str = DEFAULT_DIRECTION_MODE,
    proxy_baseline: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    values = [float(x) for x in allocation]
    caps = [float(c) for c in capacity]
    radial = math.sqrt(math.fsum((x / c) ** 2 for x, c in zip(values, caps)))
    branch_from = [math.hypot(float(predicted["branch_p_from_mw"][i]), float(predicted["branch_q_from_mvar"][i])) for i in range(len(rates))]
    branch_to = [math.hypot(float(predicted["branch_p_to_mw"][i]), float(predicted["branch_q_to_mvar"][i])) for i in range(len(rates))]
    voltage = [float(value) for value in predicted["voltage_pu"]]
    mode = str(direction_mode)
    if mode == DEFAULT_DIRECTION_MODE:
        direction = normalized_direction(values, caps, radial)
        direction_norm = radial
    elif mode == PROXY_ELECTRICAL_EFFECT_DIRECTION_MODE:
        direction, direction_norm = electrical_effect_direction(predicted=predicted, baseline=proxy_baseline, rates=rates)
    else:
        raise ValueError(f"unsupported proxy trust direction mode: {mode}")
    return {
        "rho": radial,
        "direction": direction,
        "direction_mode": mode,
        "direction_norm": direction_norm,
        "direction_defined": direction_norm > 1.0e-12,
        "proxy_effect_norm": float(direction_norm) if mode == PROXY_ELECTRICAL_EFFECT_DIRECTION_MODE else 0.0,
        "proxy_effect_gain_per_rho": (float(direction_norm) / radial) if mode == PROXY_ELECTRICAL_EFFECT_DIRECTION_MODE and radial > 1.0e-12 else 0.0,
        "total_export_normalized": math.fsum(values) / max(math.fsum(caps), 1.0e-12),
        "branch_loading_ratio": max(max((v / r for v, r in zip(branch_from, rates)), default=0.0), max((v / r for v, r in zip(branch_to, rates)), default=0.0)),
        "voltage_lower_slack_pu": min((v - float(lower) for v in voltage), default=float("inf")),
        "voltage_upper_slack_pu": min((float(upper) - v for v in voltage), default=float("inf")),
        "active_constraint_signature": signature_from_proxy(predicted=predicted, rates=rates, lower=lower, upper=upper),
    }


def _direction_distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(math.fsum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))


def _jaccard_distance(left: Sequence[int], right: Sequence[int]) -> float:
    a, b = set(int(v) for v in left), set(int(v) for v in right)
    union = a | b
    return 0.0 if not union else 1.0 - len(a & b) / len(union)


def signature_jaccard_distances(left: Mapping[str, Sequence[int]], right: Mapping[str, Sequence[int]]) -> dict[str, float]:
    return {family: _jaccard_distance(left.get(family, ()), right.get(family, ())) for family in SIGNATURE_FAMILIES}


def _compatible(record: Mapping[str, Any], center: Mapping[str, Any]) -> bool:
    if _direction_distance(record["direction"], center["direction"]) > JOINT_DIRECTION_DISTANCE_CAP + 1.0e-12:
        return False
    distances = signature_jaccard_distances(record["signature"], center["signature"])
    return max(distances.values(), default=0.0) <= JOINT_JACCARD_DISTANCE_CAP + 1.0e-12


def _core_halo(records: Sequence[Mapping[str, Any]]) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    core: dict[str, set[int]] = {family: set(records[0]["signature"].get(family, ())) for family in SIGNATURE_FAMILIES}
    halo: dict[str, set[int]] = {family: set() for family in SIGNATURE_FAMILIES}
    for record in records:
        for family in SIGNATURE_FAMILIES:
            current = set(record["signature"].get(family, ()))
            core[family].intersection_update(current)
            halo[family].update(current)
    return ({family: sorted(values) for family, values in core.items()}, {family: sorted(values) for family, values in halo.items()})


def _feature_ranges(records: Sequence[Mapping[str, Any]]) -> dict[str, list[float]]:
    # The two electrical-effect coordinates were added in the V8 contract.
    # Keep the cluster builder replayable for legacy allocation-direction unit
    # fixtures by treating absent legacy coordinates as zero; V8 records always
    # carry the explicit values from ``proxy_features``.
    return {
        name: [
            min(float(record.get("features", {}).get(name, 0.0)) for record in records),
            max(float(record.get("features", {}).get(name, 0.0)) for record in records),
        ]
        for name in FEATURE_NAMES
    }


def build_joint_clusters(
    records: Sequence[Mapping[str, Any]],
    *,
    source_policy: str = "PREREGISTERED_FARTHEST_FIRST_Q_METHOD_SIDE_RHO_BAND",
    rho_policy: RhoBandPolicy | None = None,
) -> dict[str, Any]:
    """Build a joint library under legacy bands or an explicit registered policy.

    Omitting ``rho_policy`` preserves the historical output shape and B4=3.5
    semantics.  Supplying one serializes its identity and hash into the
    library; subsequent screening must provide the same policy explicitly.
    """
    if rho_policy is not None:
        # Reject a manually constructed/stale in-memory policy before any
        # source row can be grouped under it.  Otherwise a policy hash could
        # describe one interval while the runtime bands describe another.
        rho_policy.assert_integrity()
    active_policy = rho_policy or LEGACY_RHO_POLICY
    groups: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = {}
    excluded: list[dict[str, Any]] = []
    for record in records:
        q_mode = str(record.get("q_mode"))
        try:
            band = rho_band(float(record["features"]["rho"]), q_mode=q_mode, rho_policy=active_policy)
        except (KeyError, TypeError, ValueError):
            band = None
        if band is None:
            excluded.append({"row_id": str(record.get("row_id")), "reason": "RHO_OUTSIDE_PREREGISTERED_BANDS"})
            continue
        key = (q_mode, str(record["method_id"]), str(record["side"]), band)
        groups.setdefault(key, []).append(record)
    clusters: list[dict[str, Any]] = []
    for key in sorted(groups):
        remaining = sorted(groups[key], key=lambda row: str(row["row_id"]))
        ordinal = 0
        while remaining:
            center = remaining[0]
            members = [row for row in remaining if _compatible(row, center)]
            member_ids = {id(row) for row in members}
            remaining = [row for row in remaining if id(row) not in member_ids]
            core, halo = _core_halo(members)
            direction_radius = max((_direction_distance(row["direction"], center["direction"]) for row in members), default=0.0)
            nominal_jaccard = max((max(signature_jaccard_distances(row["signature"], center["signature"]).values(), default=0.0) for row in members), default=0.0)
            body = {
                "cluster_id": f"{key[0]}::{key[1]}::{key[2]}::{key[3]}::C{ordinal:03d}",
                "q_mode": key[0], "method_id": key[1], "side": key[2], "rho_band": key[3],
                "direction_distance_cap": JOINT_DIRECTION_DISTANCE_CAP,
                "nominal_jaccard_distance_cap": JOINT_JACCARD_DISTANCE_CAP,
                "direction_radius_observed": direction_radius,
                "nominal_jaccard_distance_observed": nominal_jaccard,
                "nominal_direction": [float(v) for v in center["direction"]],
                "nominal_signature": {family: sorted(int(v) for v in center["signature"].get(family, ())) for family in SIGNATURE_FAMILIES},
                "core_signature": core,
                "halo_signature": halo,
                "feature_ranges": _feature_ranges(members),
                "source_row_ids": sorted(str(row["row_id"]) for row in members),
                "source_reporter_ids": sorted({str(row["source_reporter_id"]) for row in members if isinstance(row.get("source_reporter_id"), str) and row.get("source_reporter_id")}),
                "source_template_ids": sorted({str(row["source_template_id"]) for row in members if isinstance(row.get("source_template_id"), str) and row.get("source_template_id")}),
                "source_fold_ids": sorted({str(row["source_fold_id"]) for row in members if isinstance(row.get("source_fold_id"), str) and row.get("source_fold_id")}),
                "source_profile_ids": sorted({str(row["source_profile_id"]) for row in members if isinstance(row.get("source_profile_id"), str) and row.get("source_profile_id")}),
                "source_family_ids": sorted({str(row["source_family_id"]) for row in members if isinstance(row.get("source_family_id"), str) and row.get("source_family_id")}),
                "source_ancestry_root_ids": sorted({str(row["source_ancestry_root_id"]) for row in members if isinstance(row.get("source_ancestry_root_id"), str) and row.get("source_ancestry_root_id")}),
                "family_generator_ids": sorted({str(row["family_generator_id"]) for row in members if isinstance(row.get("family_generator_id"), str) and row.get("family_generator_id")}),
                "independent_geometry_ids": sorted({str(row["independent_geometry_id"]) for row in members if isinstance(row.get("independent_geometry_id"), str) and row.get("independent_geometry_id")}),
                "topology_groups": sorted({str(row["topology_group"]) for row in members if isinstance(row.get("topology_group"), str) and row.get("topology_group")}),
                "source_transform_ids": sorted({str(row["source_transform"]) for row in members if isinstance(row.get("source_transform"), str) and row.get("source_transform")}),
                "row_count": len(members),
            }
            body["cluster_hash"] = canonical_hash(body)
            clusters.append(body)
            ordinal += 1
    result: dict[str, Any] = {
        "serialization_id": "proxy_joint_trust_cluster_library.v1",
        "source_policy": source_policy,
        "rho_bands": band_json(LEGACY_RHO_POLICY, "Q0") if rho_policy is None else band_json(active_policy, "Q0"),
        "direction_distance_cap": JOINT_DIRECTION_DISTANCE_CAP,
        "nominal_jaccard_distance_cap": JOINT_JACCARD_DISTANCE_CAP,
        "signature_families": list(SIGNATURE_FAMILIES),
        "feature_names": list(FEATURE_NAMES),
        "clusters": clusters,
        "excluded_rows": excluded,
        "cluster_count": len(clusters),
    }
    if rho_policy is not None:
        result["rho_policy"] = active_policy.to_json()
        result["rho_policy_hash"] = active_policy.policy_hash
        result["rho_bands_by_q"] = {q_mode: band_json(active_policy, q_mode) for q_mode in ("Q0", "Q95")}
    return result


def _contains(core: Mapping[str, Sequence[int]], observed: Mapping[str, Sequence[int]], halo: Mapping[str, Sequence[int]]) -> bool:
    for family in SIGNATURE_FAMILIES:
        c, o, h = set(core.get(family, ())), set(observed.get(family, ())), set(halo.get(family, ()))
        if not c.issubset(o) or not o.issubset(h):
            return False
    return True


def screen_joint_clusters(
    library: Mapping[str, Any] | None,
    *,
    q_mode: str,
    method_id: str,
    side: str,
    features: Mapping[str, Any],
    min_source_rows: int = 1,
    require_source_diversity: bool = False,
    require_strict_source_diversity: bool = False,
    rho_policy: RhoBandPolicy | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {"status": "JOINT_CLUSTER_NOT_EVALUATED", "pass": False, "failure_reasons": [], "cluster_id": None}
    if not isinstance(library, Mapping) or not isinstance(library.get("clusters"), list):
        result.update(status="JOINT_CLUSTER_LIBRARY_MISSING", failure_reasons=["JOINT_CLUSTER_LIBRARY_MISSING"])
        return result
    if rho_policy is not None:
        try:
            rho_policy.assert_integrity()
        except ValueError:
            result.update(status="JOINT_CLUSTER_RHO_POLICY_INVALID", failure_reasons=["JOINT_CLUSTER_RHO_POLICY_INVALID"])
            return result
    library_policy, library_error = _library_rho_policy(library)
    if library_error is not None:
        result.update(status=library_error, failure_reasons=[library_error])
        return result
    if library_policy is not None:
        if rho_policy is None:
            result.update(status="JOINT_CLUSTER_RHO_POLICY_REQUIRED", failure_reasons=["JOINT_CLUSTER_RHO_POLICY_REQUIRED"])
            return result
        if rho_policy.policy_hash != library_policy.policy_hash:
            result.update(status="JOINT_CLUSTER_RHO_POLICY_MISMATCH", failure_reasons=["JOINT_CLUSTER_RHO_POLICY_MISMATCH"])
            return result
        active_policy = rho_policy
    elif rho_policy is not None and not rho_policy.is_legacy:
        # A V2 policy cannot be retrofitted to an unbound legacy library:
        # rebuilding is required so excluded rows and clusters use one domain.
        result.update(status="JOINT_CLUSTER_RHO_POLICY_MISMATCH", failure_reasons=["JOINT_CLUSTER_RHO_POLICY_MISMATCH"])
        return result
    else:
        active_policy = rho_policy or LEGACY_RHO_POLICY
    try:
        band = rho_band(float(features.get("rho", float("nan"))), q_mode=q_mode, rho_policy=active_policy)
    except (TypeError, ValueError):
        band = None
    observed = features.get("active_constraint_signature")
    direction = features.get("direction")
    if band is None or not isinstance(observed, Mapping) or not isinstance(direction, list):
        result.update(status="JOINT_CLUSTER_FEATURE_INVALID", failure_reasons=["JOINT_CLUSTER_FEATURE_INVALID"])
        return result
    candidates = []
    for cluster in library["clusters"]:
        if not isinstance(cluster, Mapping) or cluster.get("q_mode") != q_mode or cluster.get("method_id") != method_id or cluster.get("side") != side or cluster.get("rho_band") != band:
            continue
        feature_ranges = cluster.get("feature_ranges", {})
        if not isinstance(feature_ranges, Mapping):
            continue
        # Legacy allocation-direction clusters predate the two V8 electrical
        # effect coordinates.  They remain replayable as historical
        # diagnostics; current V8 clusters carry the full FEATURE_NAMES set.
        declared_names = tuple(feature_ranges.keys())
        if set(declared_names) == {"total_export_normalized", "branch_loading_ratio", "voltage_lower_slack_pu", "voltage_upper_slack_pu"}:
            names_for_screen = declared_names
        else:
            names_for_screen = FEATURE_NAMES
        if any(name not in feature_ranges for name in names_for_screen):
            continue
        ranges_ok = all(float(feature_ranges[name][0]) - 1.0e-9 <= float(features.get(name, 0.0)) <= float(feature_ranges[name][1]) + 1.0e-9 for name in names_for_screen)
        direction_distance = _direction_distance(direction, cluster.get("nominal_direction", ()))
        jaccard = signature_jaccard_distances(observed, cluster.get("nominal_signature", {}))
        containment = _contains(cluster.get("core_signature", {}), observed, cluster.get("halo_signature", {}))
        if ranges_ok and direction_distance <= JOINT_DIRECTION_DISTANCE_CAP + 1.0e-9 and max(jaccard.values(), default=0.0) <= JOINT_JACCARD_DISTANCE_CAP + 1.0e-9 and containment:
            candidates.append((direction_distance, max(jaccard.values(), default=0.0), str(cluster.get("cluster_id")), cluster))
    if not candidates:
        result.update(status="JOINT_CLUSTER_OUTSIDE_TRUST_REGION", failure_reasons=["JOINT_CLUSTER_OUTSIDE_TRUST_REGION"], rho_band=band)
        return result
    _, _, cluster_id, cluster = min(candidates, key=lambda item: (item[0], item[1], item[2]))
    source_row_count = int(cluster.get("row_count", len(cluster.get("source_row_ids", []))))
    result["source_row_count"] = source_row_count
    result["evidence_minimum_source_rows"] = int(min_source_rows)
    if source_row_count < int(min_source_rows):
        result.update(
            {
                "status": "SINGLETON_CLUSTER_DIAGNOSTIC_ONLY",
                "pass": False,
                "failure_reasons": ["SINGLETON_CLUSTER_DIAGNOSTIC_ONLY"],
                "cluster_id": cluster_id,
                "rho_band": band,
                "feature_ranges": dict(cluster.get("feature_ranges", {})),
            }
        )
        return result
    if require_source_diversity or require_strict_source_diversity:
        profile_ids = {str(value) for value in cluster.get("source_profile_ids", cluster.get("source_fold_ids", ())) if value}
        family_ids = {str(value) for value in cluster.get("source_family_ids", ()) if value}
        ancestry_root_ids = {str(value) for value in cluster.get("source_ancestry_root_ids", ()) if value}
        generator_ids = {str(value) for value in cluster.get("family_generator_ids", ()) if value}
        geometry_ids = {str(value) for value in cluster.get("independent_geometry_ids", ()) if value}
        topology_groups = {str(value) for value in cluster.get("topology_groups", ()) if value}
        strict_diversity_ok = (
            source_row_count >= int(min_source_rows)
            and len(family_ids) >= MIN_SOURCE_DIVERSITY_FAMILIES
            and len(ancestry_root_ids) >= MIN_SOURCE_DIVERSITY_FAMILIES
            and len(geometry_ids) >= MIN_SOURCE_DIVERSITY_PROFILES
        )
        permissive_diversity_ok = (
            len(profile_ids) >= MIN_SOURCE_DIVERSITY_PROFILES
            or len(family_ids) >= MIN_SOURCE_DIVERSITY_FAMILIES
            or len(generator_ids) >= 2
            or len(geometry_ids) >= 2
            or len(topology_groups) >= 2
        )
        if (require_strict_source_diversity and not strict_diversity_ok) or (require_source_diversity and not require_strict_source_diversity and not permissive_diversity_ok):
            result.update(
                {
                    "status": "INSUFFICIENT_SOURCE_DIVERSITY",
                    "pass": False,
                    "failure_reasons": ["INSUFFICIENT_SOURCE_DIVERSITY"],
                    "cluster_id": cluster_id,
                    "rho_band": band,
                    "feature_ranges": dict(cluster.get("feature_ranges", {})),
                    "source_profile_count": len(profile_ids),
                    "source_family_count": len(family_ids),
                    "source_ancestry_root_count": len(ancestry_root_ids),
                    "source_generator_count": len(generator_ids),
                    "independent_geometry_count": len(geometry_ids),
                    "topology_group_count": len(topology_groups),
                }
            )
            return result
    # Return the selected cluster-local scalar ranges with the decision.  The
    # executable trust gate must evaluate all scalar features against the same
    # joint cluster that supplied the direction/signature decision; falling
    # back to a global range would reintroduce an inconsistent second gate and
    # can reject a boundary point that is explicitly admitted by this cluster.
    result.update({
        "status": "JOINT_CLUSTER_PASS",
        "pass": True,
        "cluster_id": cluster_id,
        "rho_band": band,
        "feature_ranges": dict(cluster.get("feature_ranges", {})),
    })
    return result


__all__ = [
    "DEFAULT_DIRECTION_MODE", "FEATURE_NAMES", "JOINT_DIRECTION_DISTANCE_CAP", "JOINT_JACCARD_DISTANCE_CAP", "MIN_EVIDENCE_SOURCE_ROWS", "MIN_SOURCE_DIVERSITY_PROFILES", "MIN_SOURCE_DIVERSITY_FAMILIES", "PROXY_ELECTRICAL_EFFECT_DIRECTION_MODE", "PROXY_VOLTAGE_NORMALIZATION_PU", "RHO_BANDS", "SIGNATURE_FAMILIES",
    "LEGACY_RHO_POLICY", "RhoBandPolicy", "build_joint_clusters", "electrical_effect_direction", "proxy_features", "rho_band", "screen_joint_clusters", "signature_from_proxy", "signature_jaccard_distances",
]
