"""Copy immutable Git blobs only; no scientific runner is imported or executed."""
from pathlib import Path
import hashlib
import json
import subprocess

HERE = Path(__file__).resolve().parent
OLD = '5d53698534a1f325c8b7ed7be1bca6fa5192a77d'
RESULT = '6e2b7eb6a0afc4ee2c5f57d3b7baf89b4ed237e1'
PRICES = '3983fe4f1ede003953735e757d2c098c12b4644f'
OLD_PATH = 'docs/progress/track_a/figures/all_methods_hchain_20261022/'
STAGE3 = 'artifacts/resource_applicability/track_a_h46_s4_counter_native_compatibility/2026-10-18/stage3/'
PRICE_PATH = 'artifacts/resource_applicability/track_a_h6_noncancellation_stage_b_rerun/2026-10-20/'


def acquire():
    entries = []

    def copy(label, commit, path, extension=None):
        data = subprocess.check_output(['git', 'show', f'{commit}:{path}'], cwd=HERE)
        name = f'sources/{label}{extension or Path(path).suffix}'
        target = HERE / name
        target.parent.mkdir(exist_ok=True)
        if target.exists():
            assert target.read_bytes() == data, name
        else:
            target.write_bytes(data)
        entries.append(dict(label=label, commit=commit, source_path=path, local_path=name,
                            git_blob=subprocess.check_output(['git', 'rev-parse', f'{commit}:{path}'], cwd=HERE, text=True).strip(),
                            bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                            url=f'https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/{commit}/{path}'))
        return data

    for name in ['figure_data.json', 'group_selection_audit.json', 'historical_A_PF_comparison.csv',
                 'input_sources.json', 'figure_notes_ja.md', 'provenance.json']:
        copy('legacy_' + Path(name).stem, OLD, OLD_PATH + name)
    previous = json.loads((HERE / 'sources/legacy_input_sources.json').read_text())
    for entry in previous:
        copy(entry['label'], entry['commit'], entry['source_path'])
    for name in ['shot_weighted_totals.json', 'pf_pr_comparisons.json', 'preserved_shots.json',
                 'direction_audit.json', 'theory_rows.json', 'result_summary.json', 'terminal.json', 'audit_projection.json']:
        copy('stage3_' + Path(name).stem, RESULT, STAGE3 + name)
    copy('h68_a_readme', '64c85f7a833b5950ef2ccae623708b73081edcd6',
         'artifacts/resource_applicability/track_a_pr_new_a_total_cost_results/2026-10-18/README.md')
    copy('theory_report', 'f15752625613956d087bab221e48f1ce7dd59fef',
         'docs/research/track_a_h46_theory_compiled_total_plots_20261019.md')
    copy('slide', '12efc52b8387182df3063a3523c7099e6bc1bfec',
         'docs/progress/track_a/PR資源評価_詳細スライド構成案_14枚基本・12枚圧縮_2026-10-21_説明意図明確化版.md')
    for name in ['candidate_ranks.csv', 'candidate_totals.csv', 'combined_selection100.json',
                 'used_price_evidence.json', 'sealed_contract.json', 'summary.json', 'README.md', 'STOP.json']:
        copy('stage_b_' + Path(name).stem, PRICES, PRICE_PATH + name)
    (HERE / 'input_sources.json').write_text(json.dumps(entries, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'saved_Git_blob_copies': len(entries), 'bytes': sum(e['bytes'] for e in entries)}))


if __name__ == '__main__':
    acquire()
