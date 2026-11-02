#!/usr/bin/env python3
"""Check local display packet integrity and optionally record its mandatory STOP."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import re
import subprocess
from zoneinfo import ZoneInfo

P = Path(__file__).resolve().parent
ROOT = P.parents[4]
ADDITION = '\n- [全方式のH-chain総費用図：探索A・精度別7系列（2026-10-22）](figures/all_methods_hchain_20261022/README.md) — 主3精度と追加2精度、相対費用補助図、出典・検証・STOP。ローカル保存、未commit・未公開。\n'


def sha(b):
    return hashlib.sha256(b).hexdigest()


def load(name):
    return json.loads((P/name).read_text())


def dump(name, obj):
    (P/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seal-stop', action='store_true')
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()
    if args.check:
        prov = load('provenance.json')
        for x in prov['outputs']:
            assert sha((P/x['path']).read_bytes()) == x['sha256'], x['path']
        stop = load('STOP.json')
        assert stop['provenance_sha256'] == sha((P/'provenance.json').read_bytes())
        assert stop['status'] == 'COMPLETE_MANDATORY_STOP'
        for x in load('preservation_check.json')['files']:
            assert sha((ROOT/x['path']).read_bytes()) == x['after_sha256'], x['path']
        print(json.dumps(dict(status='PASS', checked_outputs=len(prov['outputs']), mandatory_STOP=True)))
        return

    # The pre-task digest inventory was taken before any artifact was written.
    baseline = json.loads(Path('/tmp/pr-all-methods-preservation-before-20261022.json').read_text())
    preservation = []
    for path, before in baseline.items():
        b = (ROOT/path).read_bytes()
        after = sha(b)
        if path == 'docs/progress/track_a/README.md':
            assert b.endswith(ADDITION.encode())
            assert sha(b[:-len(ADDITION.encode())]) == before
            result = 'ORIGINAL_BYTES_PRESERVED_WITH_ONE_AUTHORIZED_INDEX_LINK_APPENDED'
        else:
            assert before == after, path
            result = 'BYTE_IDENTICAL'
        preservation.append(dict(path=path, before_sha256=before, after_sha256=after, status=result))
    dump('preservation_check.json', dict(status='PASS', files_checked=len(preservation),
        byte_identical=sum(x['before_sha256']==x['after_sha256'] for x in preservation),
        index_additions_only=1, original_figure_source_main_table_changes=0, files=preservation))

    sources = load('input_sources.json')
    for x in sources:
        snapshot = (P/x['local_path']).read_bytes()
        assert sha(snapshot) == x['sha256']
        assert snapshot == git('show', x['commit']+':'+x['source_path'])
    data = load('figure_data.json')['rows']
    assert len(data)==140 and sum(r['cost_exact'] is not None for r in data)==124
    assert load('verification.json')['status'] == 'PASS'
    assert load('display_layout_check.json')['status'] == 'PASS'
    assert load('display_check.json')['status'] == 'PASS'
    assert load('display_checks/pdf_render_check.json')['status'] == 'PASS'

    links = []
    for name in ('README.md','captions_ja.md','figure_notes_ja.md','figure_mapping.md'):
        for target in re.findall(r'\]\(([^)]+)\)', (P/name).read_text()):
            if target.startswith(('https://','http://','#')):
                continue
            links.append(dict(document=name, target=target))
    dump('packet_verification.json', dict(status='PASS', local_links=len(links),
        link_records=links, all_source_copies_equal_git_blobs=True,
        protected_preexisting_files=len(preservation),
        exported_figures=10, png_svg_pdf_files=30, display_data_numeric_points=124,
        explicit_missing_slots=16, P_observations_used=0,
        scientific_new_calls=dict(signal=0,sampling=0,compile=0,fit=0,bootstrap=0,
                                  shot_generation=0,total_cost_accounting=0,
                                  candidate_scientific_ranking=0,reoptimization=0),
        display_operations=['copy saved A selection','copy saved PF ordinals',
                            'requested minimum of unchanged H2/H4 scalar totals',
                            'format fractions','normalize displayed costs to full PF'],
        independent_scientific_review=False, final_evidence_freeze=False))

    outputs = []
    for f in sorted(P.rglob('*')):
        if not f.is_file() or '.mpl-cache' in f.parts or '__pycache__' in f.parts:
            continue
        if f.name in ('provenance.json','STOP.json'):
            continue
        outputs.append(dict(path=str(f.relative_to(P)), bytes=f.stat().st_size,
                            sha256=sha(f.read_bytes())))
    now = datetime.datetime.now(ZoneInfo('Asia/Tokyo')).isoformat()
    dump('provenance.json', dict(schema='ALL_METHODS_HCHAIN_DISPLAY_PROVENANCE_V1',
        created_at_jst=now, repository='HIROMU1015/Partially-Randomized-Trotter',
        checkout_path=str(ROOT), checkout_HEAD=git('rev-parse','HEAD').decode().strip(),
        checkout_branch=git('branch','--show-current').decode().strip(),
        preexisting_dirty_checkout=True, inputs=sources,
        scientific_SOURCEs={s: sorted({r['scientific_SOURCE'] for r in data if r['system']==s})
                            for s in ('H2','H4','H6','H8')},
        PF_policy='H2 D70 / H4 D190 / H6-H8 saved ALL_SAVED458 replacement; not a new accounting pass',
        PR_policy='Only saved EXPLORATION_N8 A selection; no P or old confirmation mixing',
        display_version_diff='See historical_A_PF_comparison.csv; same global PF IDs, some stored costs updated',
        figure_data_sha256=sha((P/'figure_data.json').read_bytes()),
        renderer_sha256=sha((P/'build_figures.py').read_bytes()),
        source_copy_policy='Unmodified pinned Git blobs, original relative Markdown links via original GitHub URLs',
        output_hash_exclusions=['provenance.json (self-reference)','STOP.json (binds provenance hash)'],
        outputs=outputs, verification='Local Codex technical checks and visual inspection; no independent science review',
        artifact_commit=None, new_commit_created=False, pushed=False, newly_published=False,
        current_artifact_state='LOCAL_UNCOMMITTED', mandatory_stop_required=True,
        next_stage_authorized=False))
    if args.seal_stop:
        dump('STOP.json', dict(status='COMPLETE_MANDATORY_STOP', date_jst='2026-10-22',
            completed_at_jst=now, scope='Saved-result display packet only',
            primary_epsilons=['0.05','0.005','0.0005'], additional_epsilons=['0.01','0.001'],
            absolute_figures=5, relative_figures=5, exports=30, series_per_figure=7,
            numeric_points=124, missing_slots=16, source_snapshots=len(sources),
            verification_status='PASS', display_check_status='PASS', preservation_status='PASS',
            new_scientific_calculations=0, science_sampling_compile_fit_bootstrap_reranking_reoptimization=0,
            new_shots_or_total_cost_accounting=0, independent_P_observations_used=0,
            commit_created=False, pushed=False, newly_published=False,
            provenance_sha256=sha((P/'provenance.json').read_bytes()),
            retry=False, automatic_continuation=False, next_stage_authorized=False,
            outstanding_authorized_work=[],
            next_action=dict(owner='user', status='OPTIONAL_PROPOSAL_NOT_AUTHORIZED',
                             action='Select figures for slides/manuscript and specify publication scope if desired'),
            Codex_action='STOP; do not start another scientific stage or publication'))
    for link in links:
        target = P/link['target'].split('#')[0]
        if target.name == 'STOP.json' and not args.seal_stop:
            continue
        assert target.exists(), link
    print(json.dumps(dict(status='PASS', sources=len(sources), preexisting_files=len(preservation),
                          local_links=len(links), outputs=len(outputs), STOP_sealed=args.seal_stop)))


if __name__=='__main__':
    main()
