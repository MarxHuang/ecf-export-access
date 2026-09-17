"""Diagnostic reduction of signed proxy rows for fixed-regime algebraic rules.

The scalar algebraic rules are defined only on a common monotone domain.  AC
finite-difference voltage rows are nevertheless retained in their original
signed form for screening.  This module can remove a signed row only when a
simple box certificate proves that its maximum over the declared allocation
box is already below its right-hand side.  Otherwise it fails closed.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from r4r.allocation_domain import FeasibleAllocationDomain, LinearConstraintRow
from r4r.errors import ValidationError
from r4r.serialization import canonical_hash
from r4r.types import FiniteFloat, Sha256


@dataclass(frozen=True, slots=True)
class SignedRowRedundancyCertificate:
    constraint_id: Any
    max_lhs_over_box: FiniteFloat
    rhs: FiniteFloat
    tolerance: FiniteFloat
    status: str
    domain_hash: Sha256
    serialization_id = "signed_row_redundancy_certificate.v1"

    def __post_init__(self) -> None:
        if not hasattr(self.constraint_id, "value"):
            raise ValidationError("redundancy certificate requires a typed constraint ID")
        if self.status not in {"REDUNDANT", "NON_REDUNDANT"}:
            raise ValidationError("redundancy certificate status is not registered")

    def to_json(self) -> dict[str, Any]:
        return {
            "serialization_id": self.serialization_id,
            "constraint_id": self.constraint_id.to_json(),
            "max_lhs_over_box": self.max_lhs_over_box.to_json(),
            "rhs": self.rhs.to_json(),
            "tolerance": self.tolerance.to_json(),
            "status": self.status,
            "domain_hash": self.domain_hash.to_json(),
        }


@dataclass(frozen=True, slots=True)
class SignedDomainReductionResult:
    full_domain_hash: Sha256
    reduced_domain: FeasibleAllocationDomain | None
    certificates: tuple[SignedRowRedundancyCertificate, ...]
    status: str
    full_domain: FeasibleAllocationDomain
    serialization_id = "signed_domain_reduction_result.v1"

    def __post_init__(self) -> None:
        if self.status not in {"REDUCED_MONOTONE", "ALGEBRAIC_FIXED_REGIME_UNSUPPORTED_ON_SIGNED_DOMAIN"}:
            raise ValidationError("signed-domain reduction status is not registered")
        if self.status == "REDUCED_MONOTONE" and self.reduced_domain is None:
            raise ValidationError("successful signed-domain reduction requires a reduced domain")
        if self.status != "REDUCED_MONOTONE" and self.reduced_domain is not None:
            raise ValidationError("unsupported signed domain cannot expose a reduced domain")
        if not isinstance(self.full_domain, FeasibleAllocationDomain):
            raise ValidationError("signed-domain reduction requires the full source domain")
        if self.full_domain.domain_hash != self.full_domain_hash:
            raise ValidationError("reduction full-domain hash does not match its source domain")
        if any(certificate.domain_hash != self.full_domain_hash for certificate in self.certificates):
            raise ValidationError("reduction certificate is bound to a different full domain")
        if self.status == "REDUCED_MONOTONE":
            if any(certificate.status != "REDUNDANT" for certificate in self.certificates):
                raise ValidationError("successful reduction cannot contain non-redundant certificates")
            full_ids = {row.constraint_id.value for row in self.full_domain.constraints}
            reduced_ids = {row.constraint_id.value for row in self.reduced_domain.constraints}
            removed_ids = {certificate.constraint_id.value for certificate in self.certificates}
            if reduced_ids & removed_ids or reduced_ids | removed_ids != full_ids:
                raise ValidationError("reduction certificate does not partition the full-domain rows")

    @property
    def certificate_hash(self) -> Sha256:
        return Sha256(canonical_hash({"certificates": [item.to_json() for item in self.certificates]}))

    def to_json(self) -> dict[str, Any]:
        return {
            "serialization_id": self.serialization_id,
            "full_domain_hash": self.full_domain_hash.to_json(),
            "reduced_domain_hash": self.reduced_domain.domain_hash.to_json() if self.reduced_domain else None,
            "certificates": [item.to_json() for item in self.certificates],
            "certificate_hash": self.certificate_hash.to_json(),
            "status": self.status,
        }


def _box_maximum(row: LinearConstraintRow, domain: FeasibleAllocationDomain) -> float:
    return sum(
        coefficient * (upper if coefficient >= 0.0 else lower)
        for coefficient, lower, upper in zip(
            row.coefficients.values,
            domain.lower_bounds_mw.values,
            domain.upper_bounds_mw.values,
        )
    )


def reduce_signed_rows_for_fixed_regime(
    domain: FeasibleAllocationDomain,
    *,
    tolerance: float | None = None,
) -> SignedDomainReductionResult:
    """Return a monotone reduced domain or an explicit fail-closed status.

    Full signed rows remain available through the original domain hash for AC
    screening.  Only rows whose box maximum is certified below their RHS are
    omitted from the algebraic allocation domain.
    """
    if not isinstance(domain, FeasibleAllocationDomain):
        raise ValidationError("domain must be FeasibleAllocationDomain")
    # The reduction certificate is part of the domain contract.  Callers may
    # not silently choose a more permissive tolerance and thereby turn a
    # non-redundant signed row into a redundant one.  An explicitly supplied
    # value is accepted only as an exact restatement of the domain tolerance.
    canonical_tolerance = domain.feasibility_tolerance.value
    if tolerance is not None and tolerance != canonical_tolerance:
        raise ValidationError(
            "signed-domain reduction tolerance must equal domain.feasibility_tolerance"
        )
    tol = domain.feasibility_tolerance
    certificates: list[SignedRowRedundancyCertificate] = []
    keep: list[LinearConstraintRow] = []
    for row in domain.constraints:
        has_negative = any(value < 0.0 for value in row.coefficients.values)
        effective = any(value > 0.0 for value in row.coefficients.values)
        if not has_negative and effective and row.sense == "LESS_EQUAL" and row.rhs.value >= 0.0:
            keep.append(row)
            continue
        if row.sense != "LESS_EQUAL":
            return SignedDomainReductionResult(domain.domain_hash, None, tuple(certificates), "ALGEBRAIC_FIXED_REGIME_UNSUPPORTED_ON_SIGNED_DOMAIN", domain)
        maximum = _box_maximum(row, domain)
        status = "REDUNDANT" if maximum <= row.rhs.value + canonical_tolerance else "NON_REDUNDANT"
        certificate = SignedRowRedundancyCertificate(
            constraint_id=row.constraint_id,
            max_lhs_over_box=FiniteFloat(maximum),
            rhs=row.rhs,
            tolerance=tol,
            status=status,
            domain_hash=domain.domain_hash,
        )
        certificates.append(certificate)
        if status != "REDUNDANT":
            return SignedDomainReductionResult(domain.domain_hash, None, tuple(certificates), "ALGEBRAIC_FIXED_REGIME_UNSUPPORTED_ON_SIGNED_DOMAIN", domain)
    if not keep:
        return SignedDomainReductionResult(domain.domain_hash, None, tuple(certificates), "ALGEBRAIC_FIXED_REGIME_UNSUPPORTED_ON_SIGNED_DOMAIN", domain)
    reduced = replace(domain, constraints=tuple(keep), unresolved_fields=tuple(domain.unresolved_fields) + ("signed-row redundancy certificate retained",))
    return SignedDomainReductionResult(domain.domain_hash, reduced, tuple(certificates), "REDUCED_MONOTONE", domain)


__all__ = ["SignedRowRedundancyCertificate", "SignedDomainReductionResult", "reduce_signed_rows_for_fixed_regime"]
