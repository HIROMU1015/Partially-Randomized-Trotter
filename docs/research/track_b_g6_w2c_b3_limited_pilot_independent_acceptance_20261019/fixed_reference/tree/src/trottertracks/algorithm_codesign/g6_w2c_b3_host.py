"""Pinned R1/F01 host connection, with no CLI or authority bootstrap.

This technical baseline is not a science release. A future reviewed caller must
supply a separately verified real user approval and existing authorized store.
The host never initializes a ledger, issues an anchor/claim, or changes schedule.
"""
from pathlib import Path

from . import g6_w2c_b3_admission as A
from . import g6_w2c_b3_driver as D


def check_host(contract, root):
    """Reject unpinned/old entrypoints before admission or any science imports."""
    A.require(contract.get('schema') == A.HOST_SCHEMA, 'HOST_V2_CONTRACT_REQUIRED')
    entry = contract.get('entrypoint', {})
    pin = contract.get('source_pins', {}).get(A.HOST)
    A.require(entry == {'path': A.HOST, 'sha256': pin, 'api': 'execute_approved_v1'} and
              type(pin) is str and A.HEX.fullmatch(pin), 'PINNED_HOST_REQUIRED')
    path = A.exact_path(root, A.HOST)
    A.require(Path(__file__).resolve() == path, 'LOADED_HOST_ROOT_MISMATCH')
    A.require(A.digest(A.read_regular(path)) == pin, 'HOST_SOURCE_SHA_MISMATCH')


def execute_approved(contract, *, root, ledger, authority=None):
    """Connect one externally authorized bundle to driver, always exiting port.

    No artificial bypass, candidate-only approval, env flag, or automatic setup.
    Tests replace the factory and scientific dependencies in their own process;
    the published production entrypoint has no mock/executor argument.
    """
    root = Path(root)
    port = A.ProductionAdmission(contract, root, ledger, authority)
    with port:
        check_host(port.contract, root)
        # Full source/input/approval checks before entering driver. Its Boundary
        # rechecks, atomically reserves once, rechecks, and consumes one receipt.
        port._check(D.identity(port.contract['candidate']))
        return D.execute(port.contract['candidate'], root=root,
                         output=A.exact_path(root, port.contract['candidate']['output_relative']),
                         approval_port=port)
