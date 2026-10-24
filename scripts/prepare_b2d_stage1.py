#!/usr/bin/env python3
"""Prepare an UNAPPROVED packet; no grant, claim, input or science imports."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root",required=True)
    p.add_argument("--draft",required=True)
    p.add_argument("--fresh-output",required=True)
    a=p.parse_args();root=Path(a.source_root).resolve();draft=Path(a.draft).resolve()
    if root!=Path(__file__).resolve().parents[1]:
        p.error("Use the script from the frozen SOURCE checkout")
    sys.path.insert(0,str(root/"src"))
    from trottertracks.pr_dictionary_validation.authorization import _canonical,file_sha
    from trottertracks.b2d_evaluator.environment import ENVIRONMENT_PATH,verify_environment
    from trottertracks.b2d_evaluator.preflight import source_inventory,graph_layout,_check_source
    from trottertracks.b2d_evaluator.scope import candidate_graph,registry
    from trottertracks.b2d_evaluator.input_package import InputRecipe
    from trottertracks.b2d_evaluator.manual_authority import adoption_bytes
    head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()
    graph=candidate_graph(head);env=json.loads((root/ENVIRONMENT_PATH).read_bytes())
    verify_environment(env)
    out=Path(a.fresh_output).resolve();permitted=root/"artifacts/research_b_b2d_stage1"
    if not out.is_relative_to(permitted) or out==permitted or out.exists():
        p.error("A fresh dedicated scientific output is required")
    contract=dict(stage="B2D_STAGE1",proposal_status="UNAPPROVED_TEMPLATE",
        source_commit=head,source_root=str(root),source_files=source_inventory(root),
        recipe_fingerprint=InputRecipe().fingerprint,graph_layout=graph_layout(graph),
        input_resolution="REGISTERED_RECIPE_THEN_BACKEND_CORRECTNESS",call_scope=registry(graph),
        numerical_atol=1e-12,dependency_versions=env["dependency_versions"],
        python_runtime=env["python_runtime"],environment_binding=env,
        fresh_output_namespace=str(out),retry_allowed=False,mandatory_stop=True)
    _check_source(root,contract)
    data=_canonical(contract);digest=file_sha(adoption_bytes(contract))
    draft.mkdir(mode=0o700)
    (draft/"contract.proposal.json").write_bytes(data)
    sample=(f"科学SOURCE {head}、採択後contract SHA-256 {digest} のB2-D Stage 1条件を採択し、"
        "固定scope・環境・fresh outputで一回だけの限定科学検証と新grantの発行を承認します。"
        "retry／resume・条件変更・追加gridは認めず、完了・失敗ともmandatory STOPとします。\n")
    (draft/"approval-wording-SAMPLE-NOT-APPROVAL.txt").write_text("UNAPPROVED_TEMPLATE\n"+sample)
    print(json.dumps(dict(status="UNAPPROVED_TEMPLATE_NO_GRANT",source_commit=head,
        adopted_contract_sha256=digest,draft=str(draft),fresh_output=str(out)),indent=2))


if __name__=="__main__": main()
