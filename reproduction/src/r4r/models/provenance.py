"""Run-level provenance metadata; no absolute paths or execution side effects."""
from __future__ import annotations

import re
from dataclasses import dataclass

from r4r._version import CONTRACT_VERSION
from r4r.errors import ValidationError
from r4r.models.base import ContractModel
from r4r.models.validation import require_identifier
from r4r.types import Identifier, Sha256


@dataclass(frozen=True, slots=True)
class RunMetadata(ContractModel):
    source_tree_hash: Sha256
    scientific_payload_hash: Sha256
    manuscript_source_hash: Sha256
    variable_registry_hash: Sha256
    data_type_registry_hash: Sha256
    entity_registry_hash: Sha256
    equation_registry_hash: Sha256
    status_gate_hash: Sha256
    claim_map_hash: Sha256
    module_readiness_hash: Sha256
    contract_hash: Sha256
    config_hash: Sha256
    network_hash: Sha256
    baseline_hash: Sha256
    participant_hash: Sha256
    profile_hash: Sha256
    selection_hash: Sha256
    environment_hash: Sha256
    solver_environment_hash: Sha256
    package_attestation_hash: Sha256
    git_commit: Identifier
    run_id: Identifier
    mode: Identifier
    contract_version: str
    serialization_id = "run_metadata.v1"

    def __post_init__(self) -> None:
        require_identifier(self.git_commit, "git_commit")
        require_identifier(self.run_id, "run_id")
        require_identifier(self.mode, "mode")
        if not re.fullmatch(r"[0-9a-fA-F]{40}", self.git_commit.value):
            raise ValidationError("git_commit must be a 40-character commit identifier")
        if self.contract_version != CONTRACT_VERSION:
            raise ValidationError(f"RunMetadata contract_version must be {CONTRACT_VERSION}")
