"""Presentation exports from completed saved accounting; never invokes science."""
import csv
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/progress/track_a/figures/pr_pf_analytic_vs_circuit_CNG_20261022'
OLD=OUT.parent/'pr_pf_s2_recomputed_20261022'
RAW=ROOT/'artifacts/resource_applicability/pr_pf_analytic_vs_circuit_CNG/2026-10-22'
EPS=['0.05','0.01','0.005','0.001','0.0005']
SYSTEMS=['H2','H4','H6','H8']
COLORS={'PR':'#0072b2','PF':'#d55e00'}


def read(p):return json.loads(p.read_text())
def write(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def draw(rows,scope):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
        'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
    systems=SYSTEMS if scope=='all_conditional' else SYSTEMS[:2]
    directory=OUT/scope;directory.mkdir(exist_ok=True)
    for eps in EPS:
        selected={(r['system'],r['role']):r for r in rows if r['epsilon']==eps}
        for kind in ['C','N','G']:
            fig,axes=plt.subplots(1,2,figsize=(10.4,5.6),sharey=True)
            fig.subplots_adjust(left=.09,right=.98,bottom=.24,top=.69,wspace=.17 if kind!='N' else .26)
            title={'C':'C | circuit unit cost','N':'N | saved sufficient shots','G':'G | total cost with N fixed'}[kind]
            fig.suptitle(title+'    epsilon = '+eps,x=.09,y=.99,ha='left',fontsize=17,fontweight='bold')
            subtitle={'C':'Compiled / native theory; each measurement axis shown separately',
                'N':'Independent circuit N is unavailable; no circuit/theory ratio is plotted',
                'G':'G00: theory; G01: compiled C with saved N; independent G11 unavailable'}[kind]
            fig.text(.09,.91,subtitle,fontsize=11,color='#444444')
            fig.text(.09,.85,'H2: recovered original n=8  |  H4/H6/H8: new cohort' if len(systems)==4
                else 'H2: recovered original n=8  |  H4: new cohort',fontsize=10,color='#666666')
            for ax in axes:
                ax.axvspan(-.45,.45,color='#fff1d3',zorder=-5)
                ax.axvline(.5,color='#bbbbbb',lw=.8,ls=':')
                ax.set_xlim(-.5,len(systems)-.5)
                ax.set_xticks(range(len(systems)),[s+('*' if s in ['H6','H8'] else '') for s in systems])
                ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
            if kind=='C':
                for ax,axis in zip(axes,['cosine','sine']):
                    ax.set_title(axis.capitalize()+' measurement',fontsize=12)
                    for i,s in enumerate(systems):
                        for role,offset,marker in [('PR',-.07,'o'),('PF',.07,'s')]:
                            r=selected[s,role];v=r['axes'][axis]['C_ratio'];mid=(v['lower']+v['upper'])/2
                            ax.errorbar(i+offset,mid,yerr=[[mid-v['nominal95_lower']],[v['nominal95_upper']-mid]],
                                color=COLORS[role],marker=marker,ls='none',capsize=4,markersize=7,
                                markerfacecolor='white' if s in ['H6','H8'] else COLORS[role])
                    ax.axhline(1,color='#666666',ls='--',lw=1);ax.set_ylim(0,1.15)
                axes[0].set_ylabel('Circuit C / theory C')
                handles=[Line2D([],[],color=COLORS[r],marker=m,ls='none',label=l) for r,m,l in [('PR','o','PR S2'),('PF','s','Fixed PF S2')]]
                foot='Nominal 95% MC intervals for PR; theory endpoints are separate model bounds'
            elif kind=='N':
                for ax,role in zip(axes,['PR','PF']):
                    ax.set_title('PR S2' if role=='PR' else 'Fixed PF S2',fontsize=12)
                    for i,s in enumerate(systems):
                        for axis,offset,color,marker in [('cosine',-.07,'#0072b2','o'),('sine',.07,'#d55e00','s')]:
                            value=selected[s,role]['axes'][axis]['N_theory']
                            ax.plot(i+offset,value,marker=marker,ls='none',color=color,markersize=7,
                                markerfacecolor='white' if s in ['H6','H8'] else color)
                    ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0),useMathText=True)
                    largest=max(selected[s,r]['axes'][a]['N_theory'] for s in systems for r in ['PR','PF'] for a in ['cosine','sine'])
                    ax.set_ylim(0,largest*1.15)
                axes[0].set_ylabel('Saved N theory [shots / axis]')
                handles=[Line2D([],[],color=c,marker=m,ls='none',label=a.capitalize()) for a,c,m in [('cosine','#0072b2','o'),('sine','#d55e00','s')]]
                foot='N is a sufficient-condition allocation with empirical numerical inputs; not measured shots'
            else:
                for ax,role in zip(axes,['PR','PF']):
                    ax.set_title('PR S2' if role=='PR' else 'Fixed PF S2',fontsize=12)
                    for i,s in enumerate(systems):
                        r=selected[s,role];v=r['G01_over_G00'];mid=(v['lower']+v['upper'])/2
                        ax.plot(i-.075,1,marker='D',markerfacecolor='white',color='#555555',ls='none',markersize=6)
                        ax.errorbar(i+.075,mid,yerr=[[mid-v['nominal95_lower']],[v['nominal95_upper']-mid]],
                            color=COLORS[role],marker='o',ls='none',capsize=4,markersize=7,
                            markerfacecolor='white' if s in ['H6','H8'] else COLORS[role])
                    ax.axhline(1,color='#666666',ls='--',lw=1);ax.set_ylim(0,1.18)
                axes[0].set_ylabel('Total RZ cost / G00')
                handles=[Line2D([],[],color='#555555',marker='D',mfc='white',ls='none',label='G00: theory'),
                    Line2D([],[],color=COLORS['PR'],marker='o',ls='none',label='G01: PR'),
                    Line2D([],[],color=COLORS['PF'],marker='o',ls='none',label='G01: fixed PF')]
                foot='G01 sums N x C over both axes; PR nominal MC intervals retain paired-axis covariance'
            fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.54,.81),ncol=len(handles),frameon=False,fontsize=10)
            fig.text(.09,.13,foot,fontsize=9,color='#444444')
            fig.text(.09,.075,'* H6/H8: conditional only; auxiliary G-operator validation failed (threshold unchanged)' if len(systems)==4
                else 'H2/H4: passed the executed technical checks; no independent scientific review',fontsize=9,color='#8d4b16')
            fig.text(.09,.025,'Fixed candidates | linear chains, 1 A, STO-3G | T=0.8 | native RZ, not paper Toffoli resources',fontsize=8.5,color='#666666')
            name=kind+'_eps_'+eps.replace('.','p')
            for suffix in ['png','svg','pdf']:fig.savefig(directory/(name+'.'+suffix),dpi=180)
            plt.close(fig)


def tables(rows):
    flat=[]
    for r in rows:
        for a,v in r['axes'].items():
            flat.append({k:r[k] for k in ['system','epsilon','role','candidate','L_D','q','delta','r','K','DF_rank','cohort','physical_status','MC_n']}|dict(
                axis=a,C_theory_lower=v['C_theory']['lower'],C_theory_upper=v['C_theory']['upper'],
                C_circuit=v['C_circuit']['mean_exact'],C_circuit_SE=v['C_circuit']['SE'],C_circuit_half95=v['C_circuit']['half95'],
                C_ratio_lower=v['C_ratio']['lower'],C_ratio_upper=v['C_ratio']['upper'],N_theory=v['N_theory'],
                N_circuit=None,N_ratio=None,G00_lower=r['G00']['lower'],G00_upper=r['G00']['upper'],
                G01=r['G01']['mean_exact'],G01_SE=r['G01']['SE'],G01_half95=r['G01']['half95'],G11=None,delta_G_N=None))
    with (OUT/'figure_data.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]),lineterminator='\n');w.writeheader();w.writerows(flat)
    audit=dict(status='NO_INDEPENDENT_CIRCUIT_N',independent_conditions=0,required_conditions=20,
        estimator='For each axis, B times average of fresh-IID Hadamard outcomes in {-1,+1}; each shot uses fresh RTE trajectory for PR',
        inequality='Two-sided Hoeffding: 2 exp(-N h^2/(2 B_upper^2)) <= alpha_axis',
        formula='h=epsilon/sqrt(2)-abs(bias)-u; N=ceil(2 B_upper^2 log(2/alpha_axis)/h^2)',
        alpha_axis=.025,alpha_total=.05,allowance_factor=1,
        bias_origin='Classical native/CSR/spin finite-polynomial expectation minus full-H reference; original H2 saved signal/shot',
        normalization_origin='Finite-RTE analytical law with saved binary64 normalization and empirical log_B margin',
        numerical_guarantee='Hoeffding is conditional on valid bias/normalization bounds, IID sampling, and correct implementation; saved allowance is EMPIRICAL, not certified',
        circuit_evidence='Representative statevector checks at 1e-7, plus primitive checks; no independently computed compiled ensemble mean and combined numerical error budget for all PR draws',
        ideal_equivalence='If every sampled wrapper has exactly the same measurement law, same estimator, normalization, bias, allowance and alpha, the same Hoeffding expression gives equal sufficient N. It is not an independent empirical result.',
        zero_N_change_observed=False,N_change_identified=False,N_change_cause=None,N_circuit=None,N_ratio=None,
        source_code_refs=['scripts/recomputed_s2_runtime_v2.py:signal,shots,compile_one',
            'artifacts/resource_applicability/pr_pf_s2_recomputed/2026-10-22/fixed_provider/src/trottertracks/resource_applicability/ax2b_numerical_accounting.py:u_aware_shots',
            'docs/research/研究方法・解析手順.md: independent bounded outcomes and fresh IID trajectories'],
        missing_information=['Independent full compiled-ensemble corrected signal or justified equivalence/error transfer for the entire law',
            'Combined candidate/reference/normalization numerical error budget under the same sufficient-condition guarantee',
            'Exact probability law, bias and estimator assumptions under the same fixed task'],
        unadopted_alternatives=['Variance-adaptive shot estimators or exact-binomial minimum shots would change the statistical contract; not computed/adopted'],
        scope='N sensitivity not identified. G00 to G01 is C substitution at fixed saved N, not proof that true N has no change.',
        physical_status_ref='../pr_pf_s2_recomputed_20261022/physical_validation_audit.json')
    write(OUT/'N_comparability_audit.json',audit)
    write(OUT/'MC_uncertainty.json',dict(rows=[{k:r[k] for k in ['system','epsilon','role','candidate','MC_n','MC_kind']}|dict(
        axis_statistics={a:v['C_circuit'] for a,v in r['axes'].items()},paired_total=r['G01']) for r in rows],
        original_H2_not_pooled=True,pilot_not_pooled=True,no_resampling=True,nominal_not_population_guarantee=True,
        rare_event_diagnostics_ref='../pr_pf_s2_recomputed_20261022/rare_event_summary.json'))
    coverage=['# C/N/G coverage','',
        'CとG00/G01の数値対応は20条件。H2/H4の10条件と補助G診断不合格のH6/H8の10条件を区別する。独立N比較は0/20。','',
        '|系|ε|PR / PF候補|cohort|C・G00/G01|N circuit・G11|','|---|---|---|---|---|---|']
    for system in SYSTEMS:
        for eps in EPS:
            group=[r for r in rows if r['system']==system and r['epsilon']==eps];p,f=group
            coverage.append('|'+ '|'.join([system,eps,p['candidate']+' / '+f['candidate'],p['cohort'],
                '条件付き・補助G不合格' if system in ['H6','H8'] else '実施済み技術検査PASS','null（独立評価なし）'])+'|')
    (OUT/'coverage_matrix.md').write_text('\n'.join(coverage)+'\n')


def documents(rows):
    lines=['# PR/PF：解析的評価と実回路評価のC・N・G比較','',
        '**CとN固定のGは20/20条件を作図した。独立したN_circuitは0/20条件で、N比・G10・G11・N変更寄与はnull。**', '',
        '同じNを使ったG00→G01の差は、会計上すべてCの置換に対応する。独立にNの変化が0と確認できたわけではない。PRのC差には法則期待値と有限MC平均の差も含まれ、全量をcompile最適化だけに帰属しない。','',
        '## 図への入口','',
        '各図のPNG/SVG/PDFを保存。Cは両測定軸別のcompiled/theory単価比、Nは取得済みの十分shot数（独立比較値は未取得）、GはG00基準のG01。主図の横軸はH2/H4/H6/H8。空抜きのH6/H8は条件付き参考値。','',
        '|ε|4系・条件付き参考を含む|H2/H4の実施済み検査PASS範囲|','|---|---|---|']
    for eps in EPS:
        cells=[]
        for scope in ['all_conditional','confirmed_H2_H4']:
            cells.append(' / '.join(f'{k}: [PNG]({scope}/{k}_eps_{eps.replace(".","p")}.png)・[SVG]({scope}/{k}_eps_{eps.replace(".","p")}.svg)・[PDF]({scope}/{k}_eps_{eps.replace(".","p")}.pdf)' for k in ['C','N','G']))
        lines.append('|'+eps+'|'+'|'.join(cells)+'|')
    lines += ['', '計30図・90ファイル。既存スライド6・詳細A.9と元図は変更していない。スライド6への採用は未判断。','',
        '## 定義とN監査','',
        '|量|解析側|回路側|今回の扱い|','|---|---|---|---|',
        '|C_a|NATIVE_MIXED / explicit_rz_sx_x_cx_v1の法則期待RZ数|PRの保存compiled平均／PFの決定論計数|cos/sin別に比較。理論lower/upperを保持|',
        '|N_a|保存bias・normalization・経験的allowanceをHoeffding十分条件へ代入|同じ推定器・精度・失敗率のもとでcompiled信号から独立評価する量|未取得null。実機の実測必要shot数ではない|',
        '|G00|Σ N_theory,a C_theory,a|—|native費用モデル＋数値信号入力によるhybrid会計|',
        '|G01|—|Σ N_theory,a C_circuit,a|**N固定・Cのみ置換した総費用比較**|',
        '|G10 / G11|Σ N_circuit,a C_theory,a|Σ N_circuit,a C_circuit,a|独立N未取得のため両方null|','',
        'N_theoryは純粋に解析式だけから得た値ではない。corrected平均信号とfull-H参照との差を古典数値評価し、経験的u、finite-RTE正規化、α_cos=α_sin=.025、各軸ε/√2を使う。Nは必要最小回数でなく十分条件に基づく配分。補正付き独立有界±1測定平均にHoeffdingを適用する。PRでは量子shotごとにfresh IID RTE抽選を仮定する。','',
        'すべての抽選回路で同じ理想測定法則を実装し、推定器・bias・u・B・αを同じにすれば、同じ式は同じ十分Nを返す。これは条件付きの数学的帰結で、保存compiled費用だけからNを独立取得した証拠ではない。代表状態ベクトルの1e-7検査だけを、全抽選列のbias・正規化・数値誤差を保証する証明へ格上げしない。独立Nを得るにはcompiled平均信号または全法則に対する同値性／誤差移送と、同じ保証強度の数値誤差予算が必要。分散適応や別の統計保証を採用する計算は行っていない。','',
        '[N比較可能性監査](N_comparability_audit.json)に6項目のpreflightと不足情報を記録した。[追加比較の契約案](N_comparison_contract_DRAFT.md)は未採用・未実行。新推定器の科学的採用は未承認。今回のN図は取得済みshotの情報図であり、独立比較図ではない。','',
        '## 固定条件・cohort・留保','',
        'linear H2/H4/H6/H8、隣接1.00 Å、STO-3G、signed tol-only DF1e-8、rank3/10/19/28、generation順、T=.8、通常signed bias、allowance倍率1、P=0。full cos/sin Hadamard wrapper、native DF、symmetric-directional、original_plus_tail_once、reference_rotated=false。Qiskit1.3.0・Qiskit Nature0.7.2、rz/sx/x/cx、opt1、seed17、topologyなし。外側S2・旧compiled最良候補を固定し、理論側の再選択なし。各L_D/q/r/K・δ=.8/qと入力hashは[データ](figure_data.json)・[coverage](coverage_matrix.md)・[provenance](provenance.json)に対応する。','',
        'H2は元cohort n8の回収5条件。H4/H6/H8は新cohort15条件で、H2とpoolしない。H6/H8の10条件はunweighted-G補助診断不合格の留保付きで、固定1e-9閾値を変更せず、検証済み主結果へ格上げしない。[元の物理検証](../pr_pf_s2_recomputed_20261022/physical_validation_audit.json)を参照。失敗科学単位・grant・STOPは再利用／retryなし。','',
        'PRの誤差棒は保存標本から得る平均費用の名目Student-t 95%区間。Gの両軸はpairedで共分散を含める。Nの量子測定統計誤差とは別。元H2 n8に新1%目標を適用しない。新本評価11候補の設計目標達成を継承するが、5つの未観測positive-order層のcompiled母平均寄与は不明で、通常区間は厳密保証ではない。[MC記録](MC_uncertainty.json)・[元の希少event診断](../pr_pf_s2_recomputed_20261022/rare_event_summary.json)。','',
        '原PR論文のToffoli資源式の直接再現検証ではなく、native DFのRZ費用モデルとcompiled回路の比較。FT T、実機noise、全RPE最終総費用、一般的優位性、外部再現、immutable CIを主張しない。','',
        '## 総費用とPR/PFの傾向','',
        '表のR=100(1−G01/G00)はNを固定した低下率。正のRはcompiled費用が低い。R_PF−R_PR>0はPFの相対低下が大きいという点比較。N寄与は未取得で、0と代入しない。','',
        '|系|ε|L_D PR/PF|δ PR/PF|R_PR [%]|R_PF [%]|R_PF−R_PR [pp]|','|---|---|---|---|---:|---:|---:|']
    trends=[]
    for system in SYSTEMS:
        for eps in EPS:
            p,f=[next(r for r in rows if r['system']==system and r['epsilon']==eps and r['role']==role) for role in ['PR','PF']]
            rp=100*(1-p['G01_over_G00']['lower']);rf=100*(1-f['G01_over_G00']['lower'])
            trends.append(dict(system=system,epsilon=eps,PR_reduction_percent=rp,PF_reduction_percent=rf,
                PF_minus_PR_pp=rf-rp,physical_status=p['physical_status'],N_effect=None))
            lines.append(f'|{system}{"*" if system in ["H6","H8"] else ""}|{eps}|{p["L_D"]}/{f["L_D"]}|{p["delta"]:g}/{f["delta"]:g}|{rp:.3f}|{rf:.3f}|{rf-rp:+.3f}|')
    write(OUT/'trend_summary.json',dict(rows=trends,causal_compiler_attribution=False,N_effect_identified=False))
    lines+=['','元H2の5条件ではPFの低下率がPRより大きい。H4は粗い2精度でPF、残る3精度でPRの低下率が大きい。H6/H8の条件付き点比較では全精度でPRの低下率がわずかに大きい。普遍的な方式差や有意差、compilerだけの因果効果とはしない。','',
        '[C/N/G分解の検算表](CNG_decomposition_verification.json)を保存した。集約C比×集約N比をG比として使っていない。G01−G00=ΣN_theory,a(C_circuit,a−C_theory,a)を有理数で検算した。G01→G11の寄与と置換順序依存性はN未取得のため評価不能。','',
        '## 保存・検証・停止','',
        '[数値と形式の検証](verification.json)、[C/N/G CSV](figure_data.csv)、[実行receipt](../../../../../artifacts/resource_applicability/pr_pf_analytic_vs_circuit_CNG/2026-10-22/execution_receipt.json)、[SOURCE](../../../../../artifacts/resource_applicability/pr_pf_analytic_vs_circuit_CNG/2026-10-22/SOURCE.json)。新しい保存値会計は40単位をone-shot実行し、新signal・shot評価・compile・samplingは0。既存SOURCE/raw/grant/ledger/marker/STOP/図/スライドを保全した。','',
        'branch `codex/pr-pf-analytic-vs-circuit-cng-20261022`。SOURCE `bb5251fc9780ac7ac3b6c3c540aa3c865879eb4f`。完成物はローカルcommitへ保存し、push・公開・merge・PRは実行しない。既存研究原本の保全指示を優先し、研究概要・目的・スライドは変更せず、検証文書・新日付ノート・索引・証拠台帳へ追加した。主担当自身の技術検証で、GPT独立詳細科学レビューは未開始。**mandatory STOP**。','']
    (OUT/'README.md').write_text('\n'.join(lines))
    (OUT/'definitions.md').write_text('\n'.join(lines[lines.index('## 定義とN監査'):lines.index('## 固定条件・cohort・留保')])+'\n')
    captions=['# 日本語caption・notes','', '固定条件・cohort・MC限界は[README](README.md)を参照。','']
    for eps in EPS:
        captions += [f'## ε={eps}','',
            '図C：PR S2と固定PF S2のcompiled/native理論回路単価比。左右はcosine/sine、基準1。shot重みを混ぜず、各軸のRZ単価を比較する。PRの誤差棒は名目95% MC区間。','',
            '図N：保存済みHoeffding十分shot数を軸別に示す。独立N_circuitは未取得で、比は描かない。実機で必要shot数を測定した結果ではない。','',
            '図G：G00=ΣN_theory C_theoryを1とし、G01=ΣN_theory C_circuitを示す。N固定・Cのみ置換した総費用比較で、G11は未取得。誤差棒はpaired軸の共分散を含む古典MC平均の名目95%区間。','']
        for r in [x for x in rows if x['epsilon']==eps]:
            captions.append(f'- {r["system"]} {r["role"]}、rank{r["DF_rank"]}、L_D={r["L_D"]}、q={r["q"]}、δ={r["delta"]:g}、r={r["r"]}、K={r["K"]}、MC n={r["MC_n"]}。'+('補助G不合格の条件付き。' if r['system'] in ['H6','H8'] else '実施済み技術検査PASS範囲。'))
        captions.append('')
    (OUT/'captions_ja.md').write_text('\n'.join(captions).rstrip()+'\n')


def main():
    assert not (OUT/'STOP.json').exists(),'Packet is stopped; do not overwrite it'
    rows=read(OUT/'figure_data.json')['rows']
    tables(rows);documents(rows)
    for scope in ['all_conditional','confirmed_H2_H4']:draw(rows,scope)
    print(json.dumps(dict(figures=30,files=90,N_independent=0)))


if __name__=='__main__':main()
