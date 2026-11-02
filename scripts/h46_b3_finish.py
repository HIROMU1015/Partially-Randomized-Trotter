"""Minimal stage continuation and standalone exports for the B3 benchmark."""
import json
import math
from pathlib import Path
import time
import h46_b3 as b
import h46_b3_report as accounting


def rare_events():
    native=b.setup();_,csv=accounting.setup();out=[];metrics=[]
    for cohort in ('b3_exploration','b3_confirmation'):
        prices=b.read(b.PUBLIC/(cohort+'_costs.json.gz'))
        for identity,g in prices.items():
            r=g['candidate'];first=b.read(g['trial_files'][0]);laws=first['laws']
            for order in (0,2,4,6):
                support=order<=r['K']
                ideal=support and any(p>0 for l in laws for k,p in zip(l['orders'],l['probabilities']) if k==order)
                grid=support and any(p>0 for l in laws for k,p in zip(l['orders'],native.cdf_law(l['probabilities'])['probabilities']) if k==order)
                count=g['rarity']['observed_order_counts'].get(str(order),0)
                out.append(dict(system=r['system'],T=r['T'],formula=r['formula'],q=r['q'],R_outer=r['R_outer'],K=r['K'],
                    id=identity,cohort=cohort,n=g['n'],order=order,in_finite_support=support,
                    ideal_law_positive=bool(ideal),effective_float_CDF_positive=bool(grid),observed_events=count,
                    state='ABSENT_FROM_SUPPORT' if not support else 'OBSERVED' if count else 'POSITIVE_UNOBSERVED' if ideal else 'ZERO_PROBABILITY',
                    probability_at_least_order=g['rarity']['probability_at_least_one_order_or_higher'].get(str(order)),
                    conditional_compiled_price_acquired=False,conditional_compiled_price=None,
                    population_tail_bound=None,unobserved_cost_imputed=False))
            for axis,m in g['mean_metrics'].items():
                metrics.append(dict(id=identity,cohort=cohort,system=r['system'],T=r['T'],formula=r['formula'],
                    q=r['q'],R_outer=r['R_outer'],K=r['K'],s=r['s'],axis=axis,**m))
    csv(b.PUBLIC/'rare_event_cost_status.csv',out);b.save(b.PUBLIC/'rare_event_cost_status.json.gz',out)
    csv(b.PUBLIC/'secondary_native_metrics.csv',metrics)


def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap,BoundaryNorm
    from matplotlib.patches import Patch
    comps=b.read(b.PUBLIC/'comparisons.json');best=b.read(b.PUBLIC/'best_configurations.json')
    def save(fig,name):
        fig.tight_layout()
        for suffix in ('png','svg','pdf'): fig.savefig(b.PUBLIC/(name+'.'+suffix),dpi=180,bbox_inches='tight')
        plt.close(fig)
    # Matching T and epsilon axes; each family tunes its own parameters.
    fig,axes=plt.subplots(2,5,figsize=(18,7),sharex=True)
    colors={'Full_PF':'#777777','B0':'#bd8800','B2':'#0072b2','B3':'#d55e00'}
    for si,s in enumerate(b.signal.PREFIX):
        for ei,e in enumerate(b.signal.EPS):
            ax=axes[si,ei];rows=sorted((v for v in comps if v['system']==s and v['epsilon']==e),key=lambda v:v['T'])
            for name,color in colors.items(): ax.plot([v['T'] for v in rows],[v[name+'_G'] for v in rows],'.-',color=color,label=name)
            ax.set(yscale='log',title=f'{s}, epsilon={e:g}',xlabel='T')
            if ei==0: ax.set_ylabel('Estimated RZ x analytic shots')
    axes[0,0].legend(fontsize=8);save(fig,'cost_vs_time')
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
    for si,s in enumerate(b.signal.PREFIX):
        for e in b.signal.EPS:
            rows=sorted((v['best'] for v in best if v['system']==s and v['epsilon']==e and v['formula_group']=='ALL' and v['best']),key=lambda v:v['T'])
            axes[si,0].plot([v['T'] for v in rows],[v['q'] for v in rows],'.-',label=f'{e:g}')
            axes[si,1].plot([v['T'] for v in rows],[v['delta'] for v in rows],'.-',label=f'{e:g}')
        axes[si,0].set(title=s+' B3 selected q',ylabel='q',xlabel='T',yscale='log')
        axes[si,1].set(title=s+' B3 selected delta',ylabel='delta=T/q',xlabel='T',yscale='log')
        axes[si,0].legend(title='epsilon',fontsize=8)
    save(fig,'selected_q_delta')
    data=b.read(b.PUBLIC/'accounting.json.gz')
    fig,axes=plt.subplots(2,4,figsize=(15,7))
    for si,s in enumerate(b.signal.PREFIX):
        nominee=next(v['best'] for v in best if v['system']==s and v['T']==2. and v['epsilon']==.0005 and v['formula_group']=='ALL')
        rows=sorted((r for r in data if r['system']==s and r['T']==2. and r['epsilon']==.0005
            and r['q']==nominee['q'] and r['formula']==nominee['formula'] and r['K']==nominee['K']),key=lambda v:v['R_outer'])
        for j,(key,label) in enumerate([('B','B'),('N_cos','N cosine'),('C_cos','Mean C cosine'),('G','G')]):
            ax=axes[si,j];available=[v for v in rows if v.get(key) is not None]
            ax.plot([v['R_outer'] for v in available],[v[key] for v in available],'.-')
            ax.set(xscale='log',yscale='log',xlabel='R_outer',ylabel=label,
                title=f"{s} {nominee['formula']} q{nominee['q']} K{nominee['K']}")
        # Missing circuit prices remain missing, rather than proxy estimates.
    save(fig,'R_outer_B_N_C_G')
    names=list(colors);cmap=ListedColormap(list(colors.values()))
    fig,axes=plt.subplots(1,2,figsize=(11,5))
    for ax,s in zip(axes,b.signal.PREFIX):
        z=np.array([[names.index(next(v['expanded_best_family'] for v in comps if v['system']==s and v['T']==T and v['epsilon']==e)) for T in b.signal.Q] for e in b.signal.EPS])
        ax.imshow(z,cmap=cmap,norm=BoundaryNorm(np.arange(-.5,4.5,1),4),aspect='auto')
        ax.set(xticks=range(4),xticklabels=list(b.signal.Q),yticks=range(5),yticklabels=b.signal.EPS,xlabel='T',ylabel='epsilon',title=s+' acquired-set point minimum')
        for i in range(5):
            for j in range(4): ax.text(j,i,names[z[i,j]],ha='center',va='center',color='white')
    fig.legend(handles=[Patch(facecolor=c,label=n) for n,c in colors.items()],loc='lower center',ncol=4)
    save(fig,'expanded_best_method_map')


def report():
    summary=b.read(b.PUBLIC/'accounting_summary.json');comps=b.read(b.PUBLIC/'comparisons.json')
    best=b.read(b.PUBLIC/'best_configurations.json');confirm=b.read(b.PUBLIC/'independent_comparisons.json')
    formula=b.read(b.PUBLIC/'formula_confirmation.json');ext=b.read(b.PUBLIC/'grid_extension.json')
    exp=b.read(b.PUBLIC/'b3_exploration_cost_summary.json');con=b.read(b.PUBLIC/'b3_confirmation_cost_summary.json')
    correct=b.read(b.PUBLIC/'B3_correctness.json')
    def find(s,T,e,f='ALL'): return next(v['best'] for v in best if v['system']==s and v['T']==T and v['epsilon']==e and v['formula_group']==f)
    m5wins=sum(v['high_over_S2']<1 for v in formula if v['formula']=='4th(m5_best)')
    reconstruction=max(v['tail_action_error'] for v in correct['reconstruction'])
    text=['# H4/H6 長時間DF-native B3資源評価','',
        '対象はlinear H4/H6、1.0 Å、STO-3G、tol-only DF threshold=1e-8、rank10/19、T=.8/1.2/1.6/2.0、epsilon=.05/.01/.005/.001/.0005、P=0。',
        '保存Hamiltonian・state・全Tのfull-H referenceを再利用した。H6保存weighted-Hermitian completionを維持し、SCF/DF/state/referenceの再生成は0。',
        '**以下は取得済み適格候補の有限標本費用と経験的精度会計であり、全域最適性・母平均優位・厳密bias保証ではない。**','',
        '## 主結果','',
        f"1. B3の探索点費用がB2より低い条件は **{summary['B3_below_B2']}/40**。固定B3 nomineeの独立確認で低い条件は **{sum(v['B3_lower'] for v in confirm if v['family']=='B2')}/40**。B2側の独立標本がない比較は明示している。",
        f"2. B3が最良Full PFより低い条件は **{summary['B3_below_Full_PF']}/40**、B0/B0+Iも含む決定論PF envelopeより低い条件は **{summary['B3_below_PF_envelope']}/40**。",
        f"3. epsilon=.0005の各公式が自身のq/R_outer/Kを選んだ事前固定nomineeの独立確認では、m5_bestがS2より低い条件は **{m5wins}/8**。以下にformula比較を示す。",
        '4. B3の取得域最良q/delta/R_outer/Kは次表。T依存性はsignalの最小適格q、探索点最小費用q、独立確認費用を区別する。',
        f"5. B3を加えた四方式の拡張取得集合でB3が探索点最良なのは **{summary['B3_expanded_best']}/40**。旧研究の33/40は旧集合内のB2対PF結果として保持し、上書きしない。",'',
        '## epsilon=.0005 のB3探索nominee','',
        '|系|T|公式|q|delta|R_outer|K|s|B|N cos / sin|C cos|G ± SE|B3/B2|B3/PF envelope|',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---|---:|---:|']
    for s in b.signal.PREFIX:
        for T in b.signal.Q:
            v=find(s,T,.0005);c=next(v for v in comps if v['system']==s and v['T']==T and v['epsilon']==.0005)
            text.append(f"|{s}|{T:g}|{v['formula']}|{v['q']}|{v['delta']:.5g}|{v['R_outer']}|{v['K']}|{v['s']}|{v['B']:.5g}|{v['N_cos']} / {v['N_sin']}|{v['C_cos']:.5g}|{v['G']:.6g} ± {v['G_SE']:.3g}|{c['B3_over_B2']:.4g}|{c['B3_over_PF_envelope']:.4g}|")
    text+=['','## 高次公式の独立確認','',
        '各公式のepsilon=.0005探索nomineeを事前固定し、別cohortで比較。確認後minへの選び直しや探索とのpoolingは行っていない。','',
        '|系|T|公式|高次/S2 G|差のSE|','|---|---:|---|---:|---:|']
    for v in formula: text.append(f"|{v['system']}|{v['T']}|{v['formula']}|{v['high_over_S2']:.6g}|{v['difference_SE']:.4g}|")
    text+=['','## 実装・科学入力・数値正しさ','',
        '基準はcommit 155529e20a8fac39e00786c40b57830b71203c4f。h46_long_time_signal.evolve_batch/qualify、h46_new2_coreのspectral・paired polynomial・matrix_only_tail_coefficients、既存signed PF scheduleとABS_TIME_PER_OUTER整数配分をそのまま利用。',
        'nativeはh46_long_time_nativeのsampling、compose、cost_trial、fingerprint/compiler cache、summarize_trialsとOccupationOracleを再利用。B3ではone-bodyだけが決定論側にあるため、oracleの不要な保持DF spectrum生成を省いた。新規の独立RTE実装は作成していない。',
        'B3はL_D=0で全DF二体fragmentをrandom tailへ置く。one-body correctionは決定論Gaussian blockとして残り、Hamiltonian constantとtailから抽出したidentityをscalarとして一度適用する。全Hamiltonian純RTEとは呼ばない。',
        f'occupation演算によるtail再構成の最大作用vector差は{reconstruction:.4g}。H4/H6・正負時間・K2/K4/K6の条件付きeventとactual adjointを確認した。条件付き確認は費用標本へ混ぜていない。',
        'symmetric-directional control、scalar一回、actual adjoint、測定付きcosine/sine full Hadamard wrapperを使用。Qiskit1.3.0、basis rz/sx/x/cx、opt1、seed17、backend/coupling/layout/routing追加指定なし。代表native/compiled作用vectorを独立occupation oracleと照合。',
        'B、bias、Nは同じ保存input・lawで評価。direct finite-time corrected complex-signal biasを主判定に用い、rawとのnormalization closureを確認。normalizationの二重適用なし。u floor1e-9、alpha_axis=.025、旧共通log_B margin・headroom/u政策を維持。厳密reference/compiled残差保証とは区別する。','',
        '## 探索範囲・候補選抜','',
        f"signal {summary['signal_candidates']}構成、精度endpoint {summary['endpoints']}。基本1584構成に対し、R64境界・基本不適格のq/K sliceへR128/256/512を追加し、必要なsystem/T/formulaだけq32、続いてq64まで追加。詳細理由は[grid_extension.json](grid_extension.json)。T>2や別分子には拡張していない。",
        f"native価格は{summary['priced_candidates']}候補、探索{exp['complete']} trajectory、独立確認{con['complete']} trajectory。top2のshot×work proxy、K別、低normalization・低単回価格、delta=.2対照を残した。proxyはcompiled下界でなく、未計測候補の劣位は認定しない。",
        '探索格子の未評価域と価格未取得域はcandidate_inventory、grid_extension、candidate_precision_accountingに明示。q64/R512の境界に達する候補もあり、その外の最適性は未確認。s=q R_outerが同じでもone-body/PF scheduleやcompiled費用が同じとは仮定していない。',
        '旧B3 H4 rank12・1.30 Åの価格とPauli純RTEのs512は今回へ転用していない。旧コード・計算方法だけを条件照合して再利用した。','',
        '## 独立確認・rare event・不確かさ','',
        '探索9 trajectory/候補。探索最良の各T×epsilon nomineeと厳しい精度の各公式nomineeを固定し、探索分散×4、名目相対MCSE2%、最低33で独立確認nを取得前に固定。別seed namespace/cohort、確認後top-upなし。cos/sinは同じtrajectoryでpaired、異なる候補間はpairingしない。',
        '独立B3対保存baselineは[independent_comparisons.csv](independent_comparisons.csv)。B2側は保存独立cohortがある場合に限って再利用し、ない場合は探索baselineと明記。正規近似95%区間・Bonferroni感度は有限標本診断であり、希少eventを含む母平均の厳密保証ではない。',
        'Kは最大偶数event次数、polynomial最高次数はK+1。supportにないevent、数学的確率正だが未観測、実効float-CDFで到達不能、観測event、条件付き価格未取得を[rare_event_cost_status.csv](rare_event_cost_status.csv)で区別。未観測費用を0とみなさず、母平均tail上界はnull。条件付きgate正しさ確認は、条件付きcompiled価格測定を意味しない。',
        '今回の取得済み順位は希少event全lawの母平均優位を確定しない。有限標本のゼロ分散も母分散ゼロとは解釈しない。','',
        '## 保存・図表','',
        '[全40比較](comparisons.csv) / [最良構成](best_configurations.csv) / [q依存](best_by_q.csv) / [R/K依存とB/N/C/G](R_K_sensitivity.csv) / [同一q/R/K対照](matched_B2_B3_controls.csv) / [delta=.2](fixed_delta.csv) / [全6 native指標](secondary_native_metrics.csv)。',
        '[T対費用](cost_vs_time.png) / [最適q/delta](selected_q_delta.png) / [R増加のB/N/C/G](R_outer_B_N_C_G.png) / [四方式map](expanded_best_method_map.png)。各図はSVG/PDFも保存し、旧図を変更していない。',
        'private event/seed、trial/checkpoint、fingerprint cache、代表QPYは /home/AbeHiromu/projects/h46-long-time-b3/20261026/ 。原研究・統合論文worktree・旧grant/ledger/marker/STOPは編集していない。',
        '結果の主要claim正式採択・詳細科学レビューは実施していない。今回の科学実行・図表・公開終了後mandatory STOP。次の作業なし。']
    (b.PUBLIC/'REPORT.md').write_text('\n'.join(text)+'\n')
    root=['# H4/H6 長時間DF-native B3資源評価','',
          '[研究REPORT](artifacts/resource_applicability/track_a_h46_long_time_b3/2026-10-26/REPORT.md)',
          '',*text[8:13],'',
          '旧長時間研究のSOURCE・数値・33/40の結論は保持。旧REPORTは[固定commit](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/REPORT.md)。',
          '', '有限取得集合・経験的allowance・有限標本の結果。全域最適性や母平均優位の保証ではない。',
          '', '今回の取得・公開終了後mandatory STOP。次の作業なし。']
    (b.ROOT/'REPORT.md').write_text('\n'.join(root)+'\n')


def pipeline():
    # Wait only for this study's own stage output. A failed stage remains STOP.
    while not (b.PUBLIC/'b3_exploration_cost_summary.json').exists():
        time.sleep(10)
    if b.read(b.PUBLIC/'b3_exploration_cost_summary.json')['errors']: raise ValueError('EXPLORATION_STOP')
    print(json.dumps(accounting.account()),flush=True)
    print(json.dumps(accounting.plan()),flush=True)
    print(json.dumps(b.run_cost('B3_CONFIRMATION','confirmation_selection.json')),flush=True)
    print(json.dumps(accounting.confirm_account()),flush=True)
    rare_events();figures();report()
    b.save(b.PUBLIC/'stages_complete.json',dict(scientific_acquisition=True,accounting=True,figures=True,
        report=True,commit_push_pending=True,automatic_follow_on_science=False))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('command',choices=['pipeline','export'])
    args=p.parse_args()
    if args.command=='pipeline': pipeline()
    else: rare_events();figures();report()
