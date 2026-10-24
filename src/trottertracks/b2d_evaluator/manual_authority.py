"""Manual user/admin publication, not cryptographic proof of chat identity.

Only a user-operated admin publication may create the protected receipt.
Codex prepares drafts; this task never calls issue_manual_packet. The receipt
must be outside the agent's writable roots and root-owned/non-writable by it.
Existing authorization.json/contract.json/marker/claim conventions are reused.
"""
import json
import os
from pathlib import Path
import re

from trottertracks.pr_dictionary_validation.authorization import _canonical, file_sha

AUTHORITY_ROOT = Path("/var/lib/prt-b2d-stage1/approvals")
SCHEMA = "b2d_manual_user_receipt_v1"


def protected_receipt(path):
    """OS ownership boundary. Hashes bind content; they do not identify a user."""
    if path.is_symlink() or not path.is_file() or path.resolve()!=path:
        raise PermissionError("Protected manual approval receipt missing.")
    for p in (path, *path.parents):
        st=p.stat()
        if st.st_uid!=0 or st.st_mode & 0o022:
            raise PermissionError("Manual approval receipt is agent-writable.")
    if os.geteuid()==0:
        raise PermissionError("Science launcher must not run as approval administrator.")
    return path.read_bytes()


def receipt_fields(packet, approval, grant_bytes, contract_bytes):
    grant,contract=json.loads(grant_bytes),json.loads(contract_bytes)
    return dict(schema=SCHEMA,mode="MANUAL_USER_ADMIN_PUBLICATION_NOT_CRYPTOGRAPHIC_CHAT_PROOF",
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
    """USER/ADMIN ONLY, after actual chat approval. Never invoked in preparation.

    Requires explicit digest confirmation. This operation issues a fresh grant;
    it must be run by the user, never by Codex or a scientific worker.
    """
    if os.geteuid()!=0:
        raise PermissionError("User/admin publication outside agent writable roots required.")
    draft_dir,packet=Path(draft_dir).resolve(),Path(packet_dir).resolve()
    cb=adoption_bytes(json.loads((draft_dir/"contract.proposal.json").read_bytes()))
    if file_sha(cb)!=confirmed_contract:
        raise PermissionError("User must confirm the exact adopted-contract digest.")
    approval=Path(approval_file).read_bytes()
    if not approval.strip() or not approval_reference.strip() or b"UNAPPROVED_TEMPLATE" in approval:
        raise PermissionError("Copy the actual separate user approval, not the sample wording.")
    contract=json.loads(cb)
    if contract["status"]!="USER_ADOPTED_STAGE1" or contract["retry_allowed"] is not False or contract["mandatory_stop"] is not True:
        raise PermissionError("One-shot contract policy differs.")
    from .preflight import _check_source, validate_scope
    from .scope import candidate_graph
    _check_source(Path(contract["source_root"]),contract)
    validate_scope(contract["call_scope"],candidate_graph(contract["source_commit"]))
    out=Path(contract["fresh_output_namespace"])
    if not out.is_absolute() or out.exists() or packet.exists():
        raise PermissionError("Fresh packet and output required.")
    token="b2d_"+file_sha(cb)[:24]
    gb=_canonical(dict(stage="B2D_STAGE1",authority="USER_EXPLICIT_B2D_STAGE1",
        status="APPROVED_BOUND_ONE_SHOT",approval_sha256=file_sha(approval),
        contract_sha256=file_sha(cb),source_commit=contract["source_commit"],
        source_closure_sha256=file_sha(_canonical(contract["source_files"])),
        fresh_output_namespace=str(out),grant_id=token,approval_reference=approval_reference,
        retry_allowed=False,mandatory_stop=True))
    # No key generation, signature service or broad grant database. Root owns
    # this independent publication; normal worker only writes the old marker.
    if AUTHORITY_ROOT.exists():
        st=AUTHORITY_ROOT.stat()
        if AUTHORITY_ROOT.is_symlink() or st.st_uid!=0 or st.st_mode & 0o022:
            raise PermissionError("Unsafe manual authority directory.")
    AUTHORITY_ROOT.mkdir(parents=True,exist_ok=True,mode=0o755)
    for p in (AUTHORITY_ROOT,*AUTHORITY_ROOT.parents):
        if p.is_symlink() or p.stat().st_uid!=0 or p.stat().st_mode & 0o022:
            raise PermissionError("Unsafe manual authority parent.")
    receipt_path=AUTHORITY_ROOT/(token+".json")
    if receipt_path.exists():
        raise PermissionError("Grant identity already issued; no replacement or resume.")
    uid=int(os.environ.get("SUDO_UID","-1"));gid=int(os.environ.get("SUDO_GID","-1"))
    if uid<=0 or gid<0:
        raise PermissionError("Issue from the approving user's sudo session.")
    packet.mkdir(mode=0o700)
    os.chown(packet,uid,gid)
    for name,data in (("contract.json",cb),("authorization.json",gb),("user_approval.txt",approval)):
        with (packet/name).open("xb") as file:
            file.write(data);file.flush();os.fsync(file.fileno())
        os.chown(packet/name,uid,gid)
        (packet/name).chmod(0o400)
    receipt=receipt_fields(packet,approval,gb,cb)
    with receipt_path.open("xb") as file:
        file.write(_canonical(receipt));file.flush();os.fsync(file.fileno())
    receipt_path.chmod(0o444)
    return dict(status="USER_ISSUED_NOT_CLAIMED",grant_id=token,packet_dir=str(packet),
                receipt=str(receipt_path),contract_sha256=file_sha(cb))
