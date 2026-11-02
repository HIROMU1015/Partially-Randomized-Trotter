"""Finalize saved evidence, reports and mandatory STOP; no scientific execution."""
from fractions import Fraction as F
from datetime import datetime, timezone
import csv
import math
import re
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/resource_applicability/pr_pf_s2_recomputed/2026-10-22'
OUT=ROOT/'docs/progress/track_a/figures/pr_pf_s2_recomputed_20261022'
IDENTIFIER='pr_pf_s2_recomputed_20261022'


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,indent=2,sort_keys=True,allow_nan=False)+'\n')


def main():
    assert not (BASE/'STOP.json').exists(),'Overall mandatory STOP already issued; do not finalize again'
    for system in ['H2','H4','H6','H8']:assert (BASE/(system+'_terminal.json')).exists(),system+' still running'
    for p in (BASE/'units').rglob('marker.json'):assert p.with_name('STOP.json').exists(),str(p)+' still active'
    subprocess.run([sys.executable,str(ROOT/'scripts/build_recomputed_s2_figures.py')],check=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/verify_recomputed_s2_packet.py')],check=True)
    data=read(OUT/'figure_data.json');check=read(OUT/'numerical_verification.json');rows=data['rows']
    assert not check['active_units']
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    numeric=data['numerical_pairs'];qualified=data['conditional_pairs'];complete=data['complete_pairs']
    theory_pairs=sum(all('G_'+role+'_theory_lower' in row and 'G_'+role+'_theory_upper' in row for role in ['PR','PF']) for row in rows)
    MC=[]
    for p in (BASE/'units').glob('*/*/MC_main/result.json'):
        d=read(p);MC.append(dict(candidate=str(p.parent.parent.relative_to(BASE/'units')),n=d['n'],
            target_met=all(x['target_met'] for x in d['statistics'].values()),
            relative_halfwidth_max=max(x['relative_halfwidth'] for x in d['statistics'].values())))
    write(OUT/'MC_summary.json',dict(main_groups=MC,pilot_n_per_candidate=128,pilot_pooled=False,
        accepted_new_trajectories=check['completed_pilot_and_main_trajectories'],
        actual_occurrence_seeds_verified=check['unique_actual_occurrence_seeds'],automatic_extension=False))
    rare_rows=[]
    for p in sorted((BASE/'units').glob('*/*/MC_main/result.json')):
        d=read(p);law=d['rare_event_diagnostics']['law'];system,candidate=p.parts[-4:-2]
        for epsilon,statistics in d['statistics'].items():
            shot=read(p.parent.parent/('shot_accounting_'+epsilon)/'result.json')
            costs=[v['cosine']*shot['N_cos']+v['sine']*shot['N_sin'] for v in d['values']]
            for order in [2,4]:
                mass=law['order_probabilities'].get(str(order),0.)
                if mass==0:continue
                any_order=-math.expm1(law['events_per_trajectory']*math.log1p(-mass))
                observed=[cost for cost,value in zip(costs,d['values']) if order in value['orders']]
                rare_rows.append(dict(system=system,candidate=candidate,epsilon=epsilon,order=order,n=d['n'],
                    event_probability=mass,events_per_trajectory=law['events_per_trajectory'],
                    probability_any_this_order=any_order,expected_observed_trajectories=d['n']*any_order,
                    probability_absent_in_main=math.exp(d['n']*math.log1p(-any_order)),
                    observed_trajectories=len(observed),
                    observed_unconditional_shot_weighted_cost=sum(observed)/d['n'],
                    observed_fraction_of_sample_total_cost=sum(observed)/sum(costs),
                    unobserved_population_cost_contribution='UNKNOWN_NO_COMPILED_POPULATION_BOUND',
                    stratum_note='Trajectories containing order k; overlapping strata, not incremental event costs'))
    with (OUT/'rare_event_summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rare_rows[0]));writer.writeheader();writer.writerows(rare_rows)
    unobserved={(r['system'],r['candidate'],r['order']) for r in rare_rows if r['observed_trajectories']==0}
    write(OUT/'rare_event_summary.json',dict(rows=rare_rows,unobserved_positive_order_candidate_strata=len(unobserved),ordinary_MC_is_population_guarantee=False,
        data_source='Saved MC_main trajectories and finite-law probabilities only; no resampling',
        strata_overlap=True,unobserved_compiled_population_cost_bound_established=False))
    lines=['# 固定S2候補の理論／compiled比較 — 2026-10-22','',
        f'理論GをPR/PF両方式で取得できた条件は{theory_pairs}/20。', '',
        f'**理論・compiledの数値対応は{numeric}/20条件。元H2が5条件、新H4/H6/H8が{data["new_cohort_pairs"]}条件。**',
        f'実施した技術検査をすべて通過した対応は{complete}条件、追加unweighted-G診断の不合格を残す条件付き対応は{qualified}条件。数値取得と科学的検証完了を区別する。','',
        'H2は元Hamiltonian/state/preparation hashと法則SOURCE・保存normalization・元shot・実測RZを照合した回収版で、保存n=8を保持する。新H2 MC/compileの失敗単位を再実行していない。H4/H6/H8の新MCとはpoolしない。', '',
        '## 図A/B','', '|ε|cohort境界を明示した一覧|新cohortのみ|回収した元H2のみ|','|---|---|---|---|']
    for eps in ['0.05','0.01','0.005','0.001','0.0005']:
        name=eps.replace('.','p');cells=[]
        for directory in ['overview','recomputed_only','recovered_H2']:
            cells.append(' / '.join(f'{kind}: [PNG]({directory}/{kind}_eps_{name}.png)・[SVG]({directory}/{kind}_eps_{name}.svg)・[PDF]({directory}/{kind}_eps_{name}.pdf)' for kind in ['A','B']))
        lines.append('|'+eps+'|'+'|'.join(cells)+'|')
    lines+=['','基本の図A/B各5枚に加え、cohort別表示を保存した。計30図、PNG/SVG/PDF計90出力。`*`を付けたH6/H8は補助診断不合格の留保付き。','',
        '## 固定条件','',
        'DF電子Hamiltonian、linear H2/H4/H6/H8、隣接1.00 Å、STO-3G、signed tol-only DF1e-8、rank3/10/19/28、generation order、cutoff0。T=.8、usual signed bias、経験的numerical allowance倍率1、P=0、cosine/sine両軸のfull Hadamard wrapper総RZ費用。native DF、symmetric-directional、original_plus_tail_once、reference_rotated=false。Qiskit1.3.0・Qiskit Nature0.7.2、rz/sx/x/cx、opt1、seed17、topologyなし。','',
        '外側S2、旧最良候補を固定。理論側の再最適化、新候補、補助的な候補再選択は行っていない。新cohortで旧候補が最良であるとは主張しない。各候補のL_D/q/r/K、δ=.8/q、axis別shotと単価は[JSON](figure_data.json)・[CSV](figure_data.csv)・[coverage](coverage_matrix.md)で確認できる。検査した範囲に限定されたlocal evidenceであり、全RPE最終総費用・一般優位性・immutable CI・外部再現ではない。','',
        '## 仮説A/Bを直接確認する数値','',
        'ρはPR/PF総費用比。1未満はPRが低費用。Rは100(1−compiled/theory)。R_PF−R_PRが正なら、今回のnative理論規約からcompiledへ移る相対的な低下がPFで大きい。PRについては法則期待値と有限MC平均の差も含むため、低下全体をcompilerの因果効果とは呼ばない。','',
        '|系|ε|L_D(PR/PF)|q(PR/PF), δ(PR/PF)|r,K(PR)|ρ theory|ρ compiled ± 名目95% MC半幅|R_PR / R_PF [%]|検証範囲|',
        '|---|---|---|---|---|---:|---:|---:|---|']
    for r in rows:
        qt=f'{r["PR_q"]}/{r["PF_q"]}, {0.8/r["PR_q"]:g}/{0.8/r["PF_q"]:g}'
        if 'rho_compiled' in r:
            def interval(lower,upper,digits):
                return f'{lower:.{digits}f}' if lower==upper else f'[{lower:.{digits}f}, {upper:.{digits}f}]'
            desc=[interval(r['rho_theory_lower'],r['rho_theory_upper'],6),
                  f'{r["rho_compiled"]:.6f} ± {r["rho_compiled_half95"]:.6f}',
                  ' / '.join(interval(r[role+'_reduction_lower'],r[role+'_reduction_upper'],3) for role in ['PR','PF'])]
        else:desc=['欠測','欠測','欠測']
        scope='元H2 n8' if r['system']=='H2' else '条件付き・補助G不合格' if r['status'].startswith('CONDITIONAL') else '実施済み技術検査PASS'
        lines.append('|'+ '|'.join([r['system'],r['epsilon'],f'{r["PR_L_D"]}/{r["PF_L_D"]}',qt,f'{r["PR_r"]},{r["PR_K"]}',*desc,scope])+'|')
    lines+=['','この表は固定候補の直接的な会計であり、方式全体の理論最適性や有意差検定ではない。条件付き点を確定した支持／反証として扱わない。','',
        '## MCと未解決事項','',
        f'新PR本評価は{len(MC)}候補、候補ごとに独立pilot128本。本評価nは{min(x["n"] for x in MC)}–{max(x["n"] for x in MC)}本で事前固定し、{sum(x["target_met"] for x in MC)}/{len(MC)}候補が対象全精度で名目95%相対半幅1%以下を満たした。pilotを本評価へ混ぜず、自動延長していない。詳細は[MC台帳](MC_summary.json)・[希少event診断](rare_event_summary.csv)。', '',
        f'正の確率を持つcandidate×Taylor次数のうち、今回の本評価で未観測だった層は{len(unobserved)}。表の層はその次数を含むtrajectory全体であり、層同士は重なり得る。観測費用寄与は追加gateだけの増分ではない。', '',
        'MC区間は古典的な期待回路費用の推定誤差で、量子shotの信号誤差とは別。希少Taylor eventの正の確率質量は保持した。未観測eventのcompiled費用寄与はUNKNOWNとし、標本分散や通常区間を母平均の保証にしない。元H2のn=8には新MCの1%目標を適用していない。元探索Aの候補選択効果も補正していない。','',
        'H6/H8の補助unweighted-G作用は固定1e-9閾値を一部で超えた。実際のHamiltonian項lambda*G^2は全項が閾値内だったが、補助検証をPASSへ変更せず、正規化finite-RTE信号への影響は未解決として残す。[物理検証監査](physical_validation_audit.json)・[検証文書](../../../../recomputed_s2_theory_compiled_validation.md)を参照。','',
        'H2の新cohort MC/compile失敗3単位と、H6/H8の追加診断失敗2単位は、grant/marker/STOPを保全し再実行していない。追加の詳細科学レビュー、モデル変更、精度目標の緩和は行っていない。','',
        '## 回収・再計算・差異・保存','',
        'H2/H4は既存の再構築済み科学配列が元hashと一致することを確認して再利用し、H6は元snapshotを回収した。今回新しくSCF・MO・積分・tol-only DF・weighted Hermitian projection・固定状態を構築したのはH8。H8のHamiltonian/state hashは元と異なるため、元compiled結果と組み合わせていない。','',
        '新compiled単価は実full-wrapper compileで取得し、旧H6/H8のALL_SAVED458価格モデルを理論単価や新compiled単価に流用していない。元結果との違いにはMC標本・shot再会計・H8入力/DF gauge・位相bookkeepingが関係し得る。差の全量を単一原因へ帰属しない。','',
        '[元H2同一性監査](original_H2_identity_audit.json) / [数値検算](numerical_verification.json) / [実行単位監査](execution_unit_audit.json) / [provenance](provenance.json)。原入力・SOURCE・grant/marker/STOP・全trajectory費用は `artifacts/resource_applicability/pr_pf_s2_recomputed/2026-10-22/`。大きいcontextは可逆gzip版も保存し、復元後のSHA-256を台帳化した。','',
        '人工技術テストは49件と36 subtestがPASS。物理入力を用いた実行単位の失敗は、それと分けて保存した。主担当自身のlocal技術検証であり、独立科学レビューではない。','',
        'SOURCE初版 `5e3503d8247d9a8788668c04818b665693501292`、主実行修正 `0cd0adfb947f29d563ce65cd6c86afcad83c7696`、追加物理検証 `4ec95cd461804cf6513efd83608445f06c0fadb4`。元counterは `f01fa5fbd9c508e468dac3cff8405f57370dbb51`。旧結果commit `2d4ff27cb630939f87e08ab5881404c72ad2fa19` と旧raw/STOPを変更していない。','',
        f'branch `{branch}`。結果はローカルcommitへ保存し、push・merge・PR作成は行わない。既存研究原本の保全指示に従い、研究概要・原稿・スライドは変更せず、検証文書・索引・証拠台帳・新しい日付ノートへ追加した。','',
        '**MANDATORY STOP。追加の科学実行・retry・再選択・詳細科学レビューは開始しない。**','']
    (OUT/'README.md').write_text('\n'.join(lines))
    captions=['# 日本語caption','', '条件はREADMEおよびfigure_dataの各rowを参照。全図は同じ候補を理論・compiledで対応させる。','']
    for eps in ['0.05','0.01','0.005','0.001','0.0005']:
        captions += [f'## ε={eps}','',
            '図A：固定S2候補におけるPR/PFのshot込み総RZ費用比。Theoryは既存native明示分解の法則期待費用、Compiledはfull Hadamard wrapperの有限MC平均（PFは決定論単価）を同じaxis shotで重み付けした値。基準線1未満はPRが低費用。誤差棒は古典MC平均の名目95%区間。', '',
            '図B：同じ候補の理論→compiled費用低下率100(1−G_compiled/G_theory)。PRの差には有限標本効果を含む。H2は回収した元n=8、H4/H6/H8は新cohortで、一覧図は背景・境界・見出しで区別し、別々のcohort図も保存した。H6/H8の*は補助G診断不合格を残す条件付き表示。','']
        for r in [x for x in rows if x['epsilon']==eps]:
            captions.append(f'- {r["system"]} DF rank{r["DF_rank"]}、PR L_D={r["PR_L_D"]}, q={r["PR_q"]}, δ={.8/r["PR_q"]:g}, r={r["PR_r"]}, K={r["PR_K"]}；PF {r["PF_candidate_id"]}, L_D={r["PF_L_D"]}, δ={.8/r["PF_q"]:g}。')
        captions.append('')
    (OUT/'captions_ja.md').write_text('\n'.join(captions)+'\n')
    (OUT/'figure_notes_ja.md').write_text('\n'.join(lines[lines.index('## 固定条件'):])+'\n')
    record=dict(id=IDENTIFIER,status='LOCAL_COST_PAIRS_WITH_FAILED_AUXILIARY_DIAGNOSTICS_MANDATORY_STOP',
        summary='Fixed-candidate native S2 theory/compiled accounting; original H2 and recomputed H4/H6/H8 kept distinct, with failed auxiliary G diagnostics retained.',
        numerical_pairs=numeric,theory_cost_pairs=theory_pairs,executed_all_technical_checks_pass_pairs=complete,conditional_pairs=qualified,
        original_H2_pairs=5,new_cohort_pairs=data['new_cohort_pairs'],new_main_candidates=len(MC),
        new_main_goal_met=sum(x['target_met'] for x in MC),failed_one_shot_units=len(check['failed_units']),
        SOURCE=['5e3503d8247d9a8788668c04818b665693501292','0cd0adfb947f29d563ce65cd6c86afcad83c7696','4ec95cd461804cf6513efd83608445f06c0fadb4'],
        report='docs/recomputed_s2_theory_compiled_validation.md',figure_packet=str(OUT.relative_to(ROOT)),
        raw_namespace=str(BASE.relative_to(ROOT)),library='src/trotterlib/recomputed_s2_physics_validation.py',
        runner='scripts/run_recomputed_s2_comparison_v2.py',test='tests/test_recomputed_s2_physics_validation.py',
        scientific_validation_complete=False,independent_scientific_review=False,failed_units_retried=False,
        final_full_RPE_total_cost=False,immutable_CI=False,push_executed=False)
    manifest=read(ROOT/'artifacts/validation_manifest.json')
    manifest[IDENTIFIER]=record
    manifest['result_sets']=[x for x in manifest['result_sets'] if x.get('id')!=IDENTIFIER]+[record]
    present=manifest['artifact_inventory']['present']
    if not any(x.get('path')==str(BASE.relative_to(ROOT)) for x in present):
        present.append(dict(path=str(BASE.relative_to(ROOT)),kind='directory',supports=[IDENTIFIER],
            reason='Authorized one-shot inputs, SOURCE, grants/markers/results/STOP, paired MC prices and explicit failed diagnostics.'))
    write(ROOT/'artifacts/validation_manifest.json',manifest)
    section='\n## 固定S2理論／compiled再計算（2026-10-22）\n\n'+f'数値対応{numeric}/20条件（元H2 5、新cohort {data["new_cohort_pairs"]}）。実施済み技術検査PASS {complete}、補助G診断不合格を残す条件付き対応{qualified}。新MC本評価{len(MC)}候補のうち{sum(x["target_met"] for x in MC)}が名目95%相対半幅1%設計目標内。失敗one-shot {len(check["failed_units"])}単位は再試行なし。\n\n[検証文書](docs/recomputed_s2_theory_compiled_validation.md)・[図A/Bと数値](docs/progress/track_a/figures/pr_pf_s2_recomputed_20261022/README.md)。全RPE最終総費用、厳密な母平均保証、独立科学レビュー、immutable CIではない。旧結果・STOPを保全し、今回の承認範囲を終えてmandatory STOP。\n'
    p=ROOT/'VALIDATION_STATUS.md';p.write_text(p.read_text()+section)
    note=ROOT/'docs/research/研究ノート/2026-10-22_S2理論compiled再計算.md'
    assert not note.exists(),'Preserve any pre-existing dated note'
    note.write_text('# 固定S2理論／compiled再計算 — 2026-10-22\n\n'+section.replace('(docs/','(../../')+'\n元環境へのファイル移送を前提にせず、回収可能な科学配列・SOURCEを照合し、不足したH8だけ物理入力を構築した。旧最良候補を固定した。H2の新MC/compile失敗と追加G診断不合格は再実行せず保全した。H6/H8の数値は条件付きとし、診断不合格の科学的影響を未解決として残した。独立詳細科学レビューは開始していない。既存研究概要・原稿・スライドは依頼に従い変更しない。\n')
    (BASE/'README.md').write_text('# 固定S2比較の実行記録\n\n'+f'branch `{branch}`。実行単位は `units/system/candidate/stage`。全単位one-shot、失敗後retryなし。SOURCE/入力/契約/承認receipt、各grant/marker/result/STOPを保存した。\n\n'+
        '結果の入口は `docs/progress/track_a/figures/pr_pf_s2_recomputed_20261022/README.md`。context.pklは元bytesを保持したままcontext.pkl.gzへ可逆圧縮した。Gitには圧縮版を保存し、`context_compression_manifest.json`の元SHA-256で復元後を照合できる。`portable_input_map.json`に入力の相対パスを記録した。\n\n'+
        'SOURCE runnerを消費済みgrantで再実行しない。図の再生成は保存JSONだけを読む `scripts/build_recomputed_s2_figures.py`、検算は `scripts/verify_recomputed_s2_packet.py`。原科学結果を再評価しない。失敗した追加G診断をPASSへ変更せず、詳細科学レビューへ自動進行しない。\n')
    link_files=[OUT/'README.md',OUT/'captions_ja.md',OUT/'figure_notes_ja.md',
                ROOT/'docs/recomputed_s2_theory_compiled_validation.md',note]
    checked_links=[]
    for document in link_files:
        for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)',document.read_text()):
            if target.startswith(('http:','https:','mailto:','#')):continue
            local=target.split('#')[0]
            assert (document.parent/local).exists(),f'Broken report link: {document}: {target}'
            checked_links.append(dict(document=str(document.relative_to(ROOT)),target=target))
    write(OUT/'document_link_verification.json',dict(status='PASS',checked_local_links=len(checked_links),links=checked_links))
    provenance=read(OUT/'provenance.json')
    provenance.update(report_source_sha256=sha(__file__),numerical_verifier_source_sha256=sha(ROOT/'scripts/verify_recomputed_s2_packet.py'),
        technical_dispatcher_sha256=sha(ROOT/'scripts/run_recomputed_s2_candidates.py'),
        technical_dispatcher_scope='H6/H8 order-only dispatcher calls frozen v2 scientific units against completed shared input; no numerical implementation',
        output_inventory={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ['provenance.json','STOP.json']})
    write(OUT/'provenance.json',provenance)
    stop=dict(stopped_at_utc=datetime.now(timezone.utc).isoformat(),status='MANDATORY_STOP_COST_DATA_COMPLETE_WITH_VALIDATION_FAILURES' if numeric==20 else 'MANDATORY_STOP_PARTIAL',
        numerical_pairs=numeric,theory_cost_pairs=theory_pairs,technical_checks_complete_pairs=complete,conditional_pairs=qualified,
        failed_units=len(check['failed_units']),active_scientific_units=[],new_main_groups=len(MC),
        new_main_groups_meeting_design_goal=sum(x['target_met'] for x in MC),
        figure_provenance_sha256=sha(OUT/'provenance.json'),numerical_verification_sha256=sha(OUT/'numerical_verification.json'),
        failed_scientific_units_retried=False,next_stage_authorized=False,independent_scientific_review=False,
        branch=branch,local_commit_authorized=True,push_executed=False,merge_or_PR_created=False,
        unresolved=['H6/H8 auxiliary unweighted-G diagnostics','New H2 MC/compile failed; original cohort retained','Rare-event compiled population cost guarantee not established'])
    write(BASE/'STOP.json',stop);write(OUT/'STOP.json',stop)
    print(json.dumps(stop,indent=2))


if __name__=='__main__':main()
