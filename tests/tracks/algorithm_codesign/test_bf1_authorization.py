"""Authorization/source gates are tested without any scientific input."""
import pytest
from trottertracks.algorithm_codesign.freeze import check_text_path, verify_launch


def test_draft_rejects_before_any_source_or_input_read(tmp_path):
    # Empty plan and malformed domain are intentional: neither is touched
    # until the independent execution authorization is present.
    with pytest.raises(PermissionError, match='NOT_AUTHORIZED'):
        verify_launch(tmp_path, {}, {'science_execution_authorized': False}, b'not JSON', b'not JSON')


@pytest.mark.parametrize('path', ['../source.py', '/outside.py', 'artifacts/input.npz'])
def test_source_inventory_cannot_turn_into_molecular_IO(path):
    with pytest.raises(ValueError, match='text paths'):
        check_text_path(path)


def test_boolean_flip_does_not_authorize_unbound_source(tmp_path):
    with pytest.raises(PermissionError, match='bind'):
        verify_launch(tmp_path, {}, {'science_execution_authorized': True}, b'{}', b'{}')
