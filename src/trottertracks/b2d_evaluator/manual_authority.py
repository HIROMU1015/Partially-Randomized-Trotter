"""Manual ordinary-user publication, not proof of chat identity.

The user saves actual chat approval and issues the packet from their terminal.
Codex must never manufacture approval or issue a real grant. Ownership cannot
distinguish Codex from the user on the same OS account: that is an operating
rule, not an independently enforced identity boundary. Existing packet/claim
conventions are reused; private permissions exclude other OS users only.
"""
import json
import os
from pathlib import Path
import re

from trottertracks.pr_dictionary_validation.authorization import _canonical, file_sha

AUTHORITY_ROOT = Path.home()/".prt-b2d-stage1-approvals"
SCHEMA = "b2d_manual_user_receipt_v2"


def approval_policy():
    return dict(mode="MANUAL_SAME_USER_OPERATIONAL_APPROVAL_V1",
        record_root=str(AUTHORITY_ROOT),owner_uid=os.geteuid(),
        user_publication_required=True,codex_issuance_forbidden=True,
        identity_limit="SAME_OS_ACCOUNT_NOT_DISTINGUISHABLE")


def _private(path, *, directory=False):
    """User-owned private leaf; do not follow aliases or writable parents."""
    if path.is_symlink() or path.resolve()!=path or not (path.is_dir() if directory else path.is_file()):
        raise PermissionError("Private manual approval path missing or aliased.")
    st=path.stat()
    if st.st_uid!=os.geteuid() or st.st_mode & 0o077:
        raise PermissionError("Manual approval path must be user-owned and private.")
    for parent in path.parents:
        st=parent.stat()
        # Sticky /tmp protects each user's private child from other users.
        if parent.is_symlink() or (st.st_mode & 0o022 and not st.st_mode & 0o1000):
            raise PermissionError("Writable or aliased manual approval parent.")


def _write_new(path,data):
    with os.fdopen(os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),"wb") as file:
        file.write(data);file.flush();os.fsync(file.fileno())


def protected_receipt(path):
    """Private user record, with no claim of independent user authentication."""
    if path.parent!=AUTHORITY_ROOT:
        raise PermissionError("Manual approval record location differs.")
    _private(AUTHORITY_ROOT,directory=True)
    _private(path)
    return path.read_bytes()


def receipt_fields(packet, approval, grant_bytes, contract_bytes):
    grant,contract=json.loads(grant_bytes),json.loads(contract_bytes)
    return dict(schema=SCHEMA,approval_policy=contract["approval_policy"],
        packet_dir=str(Path(packet).resolve()), grant_id=grant["grant_id"],
        approval_reference=grant["approval_reference"], approval_sha256=file_sha(approval),
        authorization_sha256=file_sha(grant_bytes),contract_sha256=file_sha(contract_bytes),
        source_commit=contract["source_commit"],recipe_fingerprint=contract["recipe_fingerprint"],
        graph_layout=contract["graph_layout"],scope_sha256=file_sha(_canonical(contract["call_scope"])),
        environment_sha256=file_sha(_canonical(contract["environment_binding"])),
        fresh_output_namespace=grant["fresh_output_namespace"],
        retry_allowed=False,mandatory_stop=True)


class ManualAuthority:
    is_mock=False

    def verify_approval(self,packet,approval,authorization_digest,contract_digest):
        packet=Path(packet).resolve()
        gb=(packet/"authorization.json").read_bytes()
        cb=(packet/"contract.json").read_bytes()
        grant=json.loads(gb)
        token=grant.get("grant_id","")
        if re.fullmatch(r"b2d_[A-Za-z0-9_-]+",token) is None:
            raise PermissionError("New manual B2-D grant identity required.")
        if file_sha(gb)!=authorization_digest or file_sha(cb)!=contract_digest:
            raise PermissionError("Manual approval packet changed.")
        if json.loads(cb).get("approval_policy")!=approval_policy():
            raise PermissionError("Manual user approval policy binding mismatch.")
        receipt=json.loads(protected_receipt(AUTHORITY_ROOT/(token+".json")))
        if receipt!=receipt_fields(packet,approval,gb,cb):
            raise PermissionError("Manual user approval receipt binding mismatch.")
        return True


MANUAL_AUTHORITY=ManualAuthority()


def adoption_bytes(draft):
    contract=dict(draft)
    if contract.pop("proposal_status",None)!="UNAPPROVED_TEMPLATE":
        raise PermissionError("An unapproved contract template is required.")
    contract["status"]="USER_ADOPTED_STAGE1"
    return _canonical(contract)


def issue_manual_packet(draft_dir,packet_dir,approval_file,approval_reference,confirmed_contract):
    """USER ONLY, after actual chat approval. Never invoked on real data in preparation.

    Requires explicit digest confirmation. This operation issues a fresh grant;
    it must be run by the user, never by Codex or a scientific worker.
    """
    draft_dir,packet=Path(draft_dir).resolve(),Path(packet_dir).resolve()
    cb=adoption_bytes(json.loads((draft_dir/"contract.proposal.json").read_bytes()))
    if file_sha(cb)!=confirmed_contract:
        raise PermissionError("User must confirm the exact adopted-contract digest.")
    approval_path=Path(approval_file).absolute()
    _private(approval_path.parent,directory=True)
    _private(approval_path)
    approval=approval_path.read_bytes()
    if not approval.strip() or not approval_reference.strip() or any(
            marker in approval for marker in (b"UNAPPROVED_TEMPLATE",b"SAMPLE-NOT-APPROVAL")):
        raise PermissionError("Copy the actual separate user approval, not the sample wording.")
    contract=json.loads(cb)
    if contract["stage"]!="B2D_STAGE1" or contract["retry_allowed"] is not False or contract["mandatory_stop"] is not True:
        raise PermissionError("One-shot contract policy differs.")
    if contract.get("approval_policy")!=approval_policy():
        raise PermissionError("Manual user approval policy binding mismatch.")
    from .preflight import _check_source, validate_scope, graph_layout
    from .input_package import InputRecipe
    from .environment import ENVIRONMENT_PATH, verify_environment
    from .scope import candidate_graph
    root=Path(contract["source_root"])
    _check_source(root,contract)
    graph=candidate_graph(contract["source_commit"])
    validate_scope(contract["call_scope"],graph)
    if (contract["recipe_fingerprint"]!=InputRecipe().fingerprint or
            contract["graph_layout"]!=graph_layout(graph) or
            contract["numerical_atol"]!=graph.settings.numerical_atol or
            contract["input_resolution"]!="REGISTERED_RECIPE_THEN_BACKEND_CORRECTNESS"):
        raise PermissionError("Science recipe or graph binding differs.")
    frozen=json.loads((root/ENVIRONMENT_PATH).read_bytes())
    if (contract["environment_binding"]!=frozen or contract["dependency_versions"]!=frozen["dependency_versions"] or
            contract["python_runtime"]!=frozen["python_runtime"]):
        raise PermissionError("Contract environment binding differs from frozen SOURCE.")
    verify_environment(frozen)
    out=Path(contract["fresh_output_namespace"])
    permitted=root/"artifacts/research_b_b2d_stage1"
    if not out.is_absolute() or out.resolve()!=out or not out.is_relative_to(permitted) or out==permitted or out.exists() or packet.exists():
        raise PermissionError("Fresh packet and output required.")
    _private(packet.parent,directory=True)
    token="b2d_"+file_sha(cb)[:24]
    gb=_canonical(dict(stage="B2D_STAGE1",authority="USER_EXPLICIT_B2D_STAGE1",
        status="APPROVED_BOUND_ONE_SHOT",approval_sha256=file_sha(approval),
        contract_sha256=file_sha(cb),source_commit=contract["source_commit"],
        source_closure_sha256=file_sha(_canonical(contract["source_files"])),
        fresh_output_namespace=str(out),grant_id=token,approval_reference=approval_reference,
        retry_allowed=False,mandatory_stop=True))
    AUTHORITY_ROOT.mkdir(exist_ok=True,mode=0o700)
    _private(AUTHORITY_ROOT,directory=True)
    receipt_path=AUTHORITY_ROOT/(token+".json")
    if receipt_path.exists():
        raise PermissionError("Grant identity already issued; no replacement or resume.")
    packet.mkdir(mode=0o700)
    for name,data in (("contract.json",cb),("authorization.json",gb),("user_approval.txt",approval)):
        _write_new(packet/name,data)
    receipt=receipt_fields(packet,approval,gb,cb)
    _write_new(receipt_path,_canonical(receipt))
    return dict(status="USER_ISSUED_NOT_CLAIMED",grant_id=token,packet_dir=str(packet),
                receipt=str(receipt_path),contract_sha256=file_sha(cb))
