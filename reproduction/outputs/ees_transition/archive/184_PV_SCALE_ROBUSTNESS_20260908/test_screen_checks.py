"""Supplementary boundary tests; no modifications to the frozen experiment.

Consistent failed screens must return False, not disappear from the sample.
Contradictory numerical values and recorded pass flags must be rejected.
These tests do not validate OpenDSS itself or missing line ampacities.
"""
from pathlib import Path
import copy
import hashlib
import json
import math

import run_scale as run
from audit_replay_ac import screen_predicate

A = Path(__file__).resolve().parent


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    run.verify_sources()
    frozen = {name: digest(A / name) for name in [
        'PROTOCOL.json', 'PROTOCOL.md', 'prepare_scale.py', 'run_scale.py',
        'scale_inputs.py', 'test_scale.py', 'audit_replay_ac.py',
        'verify_scale.py', 'pipeline.py',
    ]}
    good = dict(converged=True, customer_undervoltage_count=0,
                customer_overvoltage_count=0, transformer_overload_count=0,
                physical_pair_pass=True, customer_voltage_min_pu=.9,
                customer_voltage_max_pu=1.1, transformer_max_loading_pu=1.)
    accepted = []
    rejected = []

    def accept(name, expected, **changes):
        state = dict(good, **changes)
        snapshot = copy.deepcopy(state)
        assert screen_predicate(state) is expected, name
        assert state == snapshot, name
        accepted.append(dict(name=name, physical_pass=expected))

    def reject(name, **changes):
        try:
            screen_predicate(dict(good, **changes))
        except AssertionError:
            rejected.append(name)
        else:
            raise AssertionError('Contradiction was accepted: ' + name)

    accept('inclusive_voltage_boundaries', True)
    accept('transformer_numerical_tolerance', True,
           transformer_max_loading_pu=1. + 5e-10)
    accept('undervoltage_retained', False,
           customer_voltage_min_pu=.8999, customer_undervoltage_count=1,
           physical_pair_pass=False)
    accept('overvoltage_retained', False,
           customer_voltage_max_pu=1.1001, customer_overvoltage_count=1,
           physical_pair_pass=False)
    accept('transformer_overload_retained', False,
           transformer_max_loading_pu=1.0001, transformer_overload_count=1,
           physical_pair_pass=False)
    accept('nonconvergence_retained', False,
           converged=False, physical_pair_pass=False)
    accept('multiple_violations_retained', False,
           customer_voltage_min_pu=.89, customer_voltage_max_pu=1.11,
           customer_undervoltage_count=2, customer_overvoltage_count=3,
           transformer_max_loading_pu=1.1, transformer_overload_count=1,
           physical_pair_pass=False)
    reject('pass_flag_contradicts_clean_counts', physical_pair_pass=False)
    reject('pass_flag_contradicts_nonconvergence', converged=False)
    reject('voltage_below_lower_limit_unreported', customer_voltage_min_pu=.8999)
    reject('voltage_above_upper_limit_unreported', customer_voltage_max_pu=1.1001)
    reject('transformer_overload_unreported', transformer_max_loading_pu=1.0001)
    reject('reversed_voltage_extrema', customer_voltage_min_pu=1.05,
           customer_voltage_max_pu=1.04)
    reject('nan_voltage_with_pass_flag', customer_voltage_min_pu=math.nan)
    reject('nan_transformer_with_pass_flag', transformer_max_loading_pu=math.nan)
    reject('infinite_voltage_with_pass_flag', customer_voltage_max_pu=math.inf)
    reject('false_undervoltage_count', customer_undervoltage_count=1,
           physical_pair_pass=False)
    reject('false_overvoltage_count', customer_overvoltage_count=1,
           physical_pair_pass=False)
    reject('false_overload_count', transformer_overload_count=1,
           physical_pair_pass=False)
    assert frozen == {name: digest(A / name) for name in frozen}
    run.verify_sources()
    report = dict(passed=True, consistent_screens=accepted,
                  rejected_contradictions=rejected, frozen_hashes=frozen,
                  test_sha256=digest(Path(__file__)),
                  new_power_flows=0,
                  scope='Screen predicate boundaries and internally contradictory reports; not a physical-model validation.')
    (A / 'SCREEN_BOUNDARY_TESTS.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
