"""Independent wire reconstruction for the mandatory seed42 experiment.

The legacy auditor reconstructs text and original image bytes without importing
any API transport. Add the required seed explicitly; validate responses against
both independent wire hashes and the persisted actual request seed.
"""
from scripts.analysis.audit_memeye_deepseek_flash_alias import (
    wire_request as unseeded_wire_request, validate_response as legacy_validate,
)


def wire_request(payload, model, token_limit, image_detail):
    return {**unseeded_wire_request(payload, model, token_limit, image_detail), 'seed': 42}


def validate_response(answer, request, protocol, stage):
    assert protocol['contract']['seed'] == 42 and answer['seed'] == 42
    return legacy_validate(answer, request, protocol, stage)
