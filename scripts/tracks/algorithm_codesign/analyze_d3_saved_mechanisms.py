#!/usr/bin/env python3
"""Saved scalars/aggregated incidence only; no science modules or budget solver."""
import argparse
import csv
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import platform
import resource
import time

RESULT_COMMIT = 'f0df9a3b36bc3540f0c83941871fa0d3110ddbf7'
D3 = Path('artifacts/track_b_d3_result/2026-10-14/v2')
D2 = Path('artifacts/track_b_d2_key_mechanisms_preparation/2026-10-13/v1')
OLD = Path('artifacts/track_b_rte_development_d1_result/2026-10-13/v1/synthesis_cache.json')
OUT = Path('artifacts/track_b_d3_final_mechanism_analysis/2026-10-14/v1')
LEDGERS = ('G0_P0', 'G1_P0', 'G0_P1', 'G1_P1')
METRICS = ('T', 'CX', '1Q')


def dumps(value):
    def encode(v):
        if isinstance(v, F):
            return str(v)
        raise TypeError(type(v).__name__)
    return json.dumps(value, ensure_ascii=False, indent=2, default=encode) + '\n'


def whole_incidence(mu, steps, acceptance):
    """mu is conditional on a positive step; whole zero trials cost zero."""
    return steps * F(acceptance) ** steps * F(mu)


def contributions(incidences, old, new, shots):
    return {k: 2 * shots * F(mu) * (F(new[k]) - F(old[k]))
            for k, mu in incidences.items()}


def symmetric_product_gap(n_a, c_a, n_b, c_b):
    """Signed B-A, descriptive symmetric allocation, not a causal attribution."""
    shot = (n_b - n_a) * (F(c_a) + F(c_b))
    unit = (n_a + n_b) * (F(c_b) - F(c_a))
    assert shot + unit == 2*n_b*F(c_b) - 2*n_a*F(c_a)
    return dict(shot_term=shot, unit_term=unit, total=shot+unit)


def affine_relation(a, b):
    """B-A = intercept + slope*h. No ranking or global intersection search."""
    da, db = F(b[0])-F(a[0]), F(b[1])-F(a[1])
    return dict(B_minus_A_intercept=da, B_minus_A_slope=db,
                tie_h=None if db == 0 else -da/db,
                condition='B lower iff intercept+slope*h<0, h>=0')


def single_key_tie(margin, slope, current_price):
    if slope == 0:
        return dict(tie_price=None, displacement=None)
    delta = -F(margin)/F(slope)
    return dict(tie_price=F(current_price)+delta, displacement=delta)


def write_report(output, rows, focus, repeat, selection, prices, comparisons, checks):
    """Display existing exact analysis fields and saved envelopes; no new scoring."""
    report=Path('docs/tracks/algorithm_codesign/d3_final_saved_mechanisms_and_gpt_handoff_20261014.md')
    artifact='../../../'+str(OUT)+'/'
    alias={x['ratio']:x['key_alias'] for x in prices}
    short={'ordinary':'ordinary','partial_return_tail':'partial','closed_P3_tail':'P3+tail',
           'full_return':'local full','closed_P5_full':'P5','matched_CTS':'CTS'}
    def label(v):
        r,a=v.split(':',1);return r+' '+short[a]
    def num(v,digits=3):return f'{float(F(v)):,.{digits}f}'
    def pct(v):return f'{float(F(v))*100:.6f}%'
    def table(head,lines):
        return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+
                         ['| '+' | '.join(map(str,line))+' |' for line in lines])+'\n'
    sections=[f'''# D3 v2保存結果：最終機構分析とGPT引き継ぎ

2026-10-14 JST。保存scalar・既存key incidenceへのexact rational演算による技術分析。
科学分類を変更せず、新しい科学取得・合成・shot予算評価・pair評価は0。**mandatory STOP**。

P1でP3+tailがT最小となった主因は、先頭3 eventのroot keyの値上がりがP5で大きかったこと。
G1_P1のr1 P3+tailのT切片はr1 P5より約59.9万低いが、P5を分母に約0.170%の差にとどまる。
保証変更はr1の統計残余幅を大きく回復し、r2より大きく十分shot数を減らすため、最良rを変える。
これは今回の保存会計の説明であり、一般的優位・新規性・研究採択の判断ではない。

## 固定入力・証拠と分析の境界

正本result commit **`{RESULT_COMMIT}`**、science S2 **`dceccaa774229527a46a1386131583a4f618f8ad`**。
[原結果/GPT引き継ぎ](d3_v2_results_and_gpt_handoff_20261014.md)、
[既存exact48行](../../../{D3}/saved_output_audit/resource_rows_exact_v2.csv)、
[既存comparison projection](../../../{D3}/saved_output_audit/saved_comparison_projection_v2.json)、
[既存factorial effects](../../../{D3}/factorial_effects.json)を正本とする。
44入力をGitHubの上記固定commitから取得し、local bytes・fixed git blob・既存manifest（記載対象）と照合した。
[入力identity記録]({artifact}input_remote_identity_v1.json)に各URL・bytes・SHA256を保存。
resource_mapは531,617 bytes、SHA256
`a4bc655cdeda9a5a438f3034a70dce0d0b7bf647029c115b4d3696292f60a43a`であり、原2 chunksの結合一致も確認した。

対象は既知developmentの3-system-qubit synthetic involution、p=(1/5,3/10,1/2)、tau=5/7、sigma=+1、state=000。
Q0=Z0、V1=R_XX01(pi/4)、V2=R_XX12(pi/4)R_ZZ01(pi/4)、Qi=Vi†ZiVi。
finite P5(-i tau R/r)^r、r=1,2、同I1六方式。分子geometry、軌道basis、DF rank、split L_Dは適用外。
`closed_P3_tail`も同じfinite P5 targetを実現する表現で、Taylor targetを3次へ変更した比較ではない。
旧2579監査・36合成検証・D1/D2 event監査・全ランキング・全交点は再実行していない。
旧source、contract、authorization、消費済みmarker、STOP、原結果は変更しない。

## 価格寄与の接続

D2の保存positive-step条件付きproposal pi=q/Zから得た係数muをそのまま使う。
条件付き単価d、全trial単価C、全二axis切片Aは

\\[
d_{{i,T}}=\\ell_i+\\sum_k \\mu_{{ik}}t_k,\\quad
C_{{i,r,T}}=r Z_i^r d_{{i,T}}-S_{{i,r,T}},\\quad
A_{{i,r,T}}=2N_{{i,r}} C_{{i,r,T}}.
\\]

ここでSは保存済みliteral-only境界節約額（r1は0）。D3計画のno-R-cancellation静的認証を参照し、
219,303 pairを再評価しない。unconditional step費用はZ*dなので、単純なr倍やacceptanceの欠落は避ける。
全trialのkey期待出現回数はnu=r Z^r mu、二axis全体は2N nu。
同Gの価格変化は `Delta A=2N sum nu Delta t`。N・law・moment・range・biasはP0/P1で固定。
native1Qも同じnuと保存1Q価格を用いる。Tのellは既存独立係数と一致し、1Qのliteral部分は
旧保存step scalarから既存incidence*旧価格を引いた残差として明示する。
native CXのprice-only差は0。Tはnative1Qに含まれ、重複加算しない。

[36 key価格・用途・event index]({artifact}key_prices_and_incidence_v1.json)、
[小さい価格CSV]({artifact}key_prices_display_v1.csv)、
`rows/row_00.json`〜`row_11.json`のexact寄与表で、角度ratio、old/new identity、価格差、mu/nu、
各方式/rのT・1Q増価を結ぶ。signed逆回転は保存sequenceのadjointで同価格。
RZ角度はsigned atan(ratio)、scale1のラジアン。Wはnative1Q数に含めず、controlled relative phaseの既存strict guardは保持。
全12構成×2保証×T/1Qのkey寄与合計は、既存factorialのprice-only effectとexact一致した。
''']
    # Deliberately use saved fields rather than re-evaluating any shot formula.
    a,b=rows['G1_P1'][2],rows['G1_P1'][4]
    sections.append('## G1_P1 r1：P3+tailとP5の小さい差\n\n'+table(
        ['保存量','P3+tail','P5'],[
            ['十分N/axis',a['budget']['N_per_axis'],b['budget']['N_per_axis']],
            ['全trial T単価 P0',num(rows['G1_P0'][2]['per_whole_trial_native']['T'],9),num(rows['G1_P0'][4]['per_whole_trial_native']['T'],9)],
            ['全trial T単価 P1',num(a['per_whole_trial_native']['T'],9),num(b['per_whole_trial_native']['T'],9)],
            ['P0 T切片',num(rows['G1_P0'][2]['expected_native_intercept']['T']),num(rows['G1_P0'][4]['expected_native_intercept']['T'])],
            ['P1 T切片',num(a['expected_native_intercept']['T']),num(b['expected_native_intercept']['T'])],
            ['P1−P0 T切片',num(F(a['expected_native_intercept']['T'])-F(rows['G1_P0'][2]['expected_native_intercept']['T'])),num(F(b['expected_native_intercept']['T'])-F(rows['G1_P0'][4]['expected_native_intercept']['T']))],
            ['保存M2 upper',num(a['budget']['M2_upper'],9),num(b['budget']['M2_upper'],9)],
            ['保存range upper',num(a['budget']['range_upper'],9),num(b['budget']['range_upper'],9)]
        ])+'''
両方式のacceptanceは1、同rなのでbiasとremainingは同じ。P3+tailは保存moment/rangeが大きく、Nが19,836多い。
P0ではP3+tailの単価も高い。P1ではP3+tailの単価が約3.200 T低くなり、shot面の不利を小さく上回る。
保存Nは実測量子shotsでも必要最小shotsでもなく、固定common Bernstein会計の十分forecastである。
''')
    f=next(v for v in focus if v['guarantee']=='G1' and v['steps']==1)
    alloc=f['shot_unit_symmetric_allocation']
    sections.append(f'''
符号をP5−P3+tailとして、旧価格の差は{num(f['old_price_B_minus_A'])} T、
価格差によるmargin移動は+{num(f['price_margin_shift'])} T、
新価格の差は+{num(f['new_price_B_minus_A'])} T（P5分母で{pct(f['margin_fraction_of_B'])}）。
対称恒等式による記述的配分では、shot差項{num(alloc['shot_term'])} T、単価差項+{num(alloc['unit_term'])} T。
二項の和が新価格marginへexact一致する。一意の因果割合とは扱わない。

以下のkey寄与は「P5の増価−P3+tailの増価」。N変更の寄与を混ぜない。
'''+table(['key alias / exact ratio','old→new T','marginへの価格差寄与T'],[
        [alias[v['ratio']]+' / '+v['ratio'],f"{v['old_T']}→{v['new_T']}",num(v['observed_price_change_margin_contribution'])]
        for v in f['key_price_sensitivity'][:8]]))
    sections.append('''
K32=2189372/2921709はP5/local-fullの先頭3 event（index0,1,2）に使い、条件付き出現回数約1.75624。
K33=310/413はP3+tailの同じindex0,1,2の3 event、約1.73690。K32は68→80、K33は70→78で、
この二keyだけのmargin移動は約+1,500.933万T。他keyのnet移動約−51.797万Tを合わせ、全移動約+1,449.136万Tとなる。
多数の既存non-root eventを対応のない一対一因果比較にはしていない。

### 記号的価格感度

他価格・N・lawをP1のまま固定すると、margin Mは `M + a_k (t_k−t_k,P1)`。
K32の1T単位slopeは約+3,806,306.826 T、tie価格79.842638188。
K33のslopeは約−3,833,294.232 T、tie価格78.156253943。
すなわち連続記号変数としては約0.157T程度の一key変位でtieとなる。
これらは整数・parity・精度・strict phase条件を満たす合成sequenceの取得可能性を示さない。
全keyのexact tieと符号は[focused対照JSON](../../../artifacts/track_b_d3_final_mechanism_analysis/2026-10-14/v1/P3_vs_P5_price_mechanisms_v1.json)に保存。
新精度・seed・合成器への頑健性は追加取得なしには判定できない。
''')
    sections.append('## r別のP3+tail/P5 marginと保証によるr選択\n\n'+table(
        ['G','r','P0: P5−P3 T','P1: P5−P3 T','P1差/P5'],[
            [f['guarantee'],f['steps'],num(f['old_price_B_minus_A']),num(f['new_price_B_minus_A']),pct(f['margin_fraction_of_B'])] for f in focus])+
        '\nP1/r2でもroot key K27=18205486/50387421（68→80）とK28=43/119（70→78）の差が主要因。\n'
        'P3+tailとP5の価格機構はr1/r2で対応するが、総費用には各rのNと反復単価を戻す。\n')
    budget_lines=[]
    for g in ['G0','G1']:
        for r,i in [(1,0),(2,6)]:
            q=rows[g+'_P0'][i]['budget']
            budget_lines.append([g,r,num(q['Taylor_product_upper'],12),num(q['implementation_product_upper'],12),num(q['axis_bias_upper'],12),num(q['remaining'],12)])
    sections.append('\n'+table(['保証','r','Taylor積上界','native実装積上界','axis bias上界','統計残余幅'],budget_lines)+'''
同rのこれらの予算は六方式共通。native primitive誤差会計は全4ケース1e-6固定。
G1でnative積上界が微小に変わるのはTaylor上界との積の伝播によるもので、新1e-7精度へのcreditではない。
G1はr1のremainingを0.003861→0.004599へ回復する一方、r2は0.004901→0.004924の小変化。
Nの主項は保存moment/remaining²、range補助項はrange/remainingであることを固定sourceの静的式から確認した。
今回この式を使ったNの再評価・最適化は行わず、以下は保存整数の表示である。
'''+table(['方式','r1 N G0→G1','r2 N G0→G1'],[
        [short[v['arm']],f"{v['by_r']['1']['N_G0']:,}→{v['by_r']['1']['N_G1']:,}",f"{v['by_r']['2']['N_G0']:,}→{v['by_r']['2']['N_G1']:,}"]
        for v in repeat if v['price']=='P0']))
    sections.append('''
r1は約29.5%減、r2は約0.93%減。同GのNはP0/P1で同じ。
`Delta_G A = 2 Delta N * C(P)`なので、同価格の保証効果に回路単価の変化を含めない。
factorial interactionは `2 (N_G1−N_G0) Delta_P C`で、Nへの価格creditではない。
[保証/r分析]('''+artifact+'''guarantee_and_repeat_mechanisms_v1.json)は各方式/rのsaved bias、remaining、N比、T差、factorial接続を保存。
'''+table(['方式','P0: r2−r1 T G0 / G1','P1: r2−r1 T G0 / G1'],[
        [short[repeat[i]['arm']],num(repeat[i]['r2_minus_r1_T']['G0'])+' / '+num(repeat[i]['r2_minus_r1_T']['G1']),
         num(repeat[i+1]['r2_minus_r1_T']['G0'])+' / '+num(repeat[i+1]['r2_minus_r1_T']['G1'])] for i in range(0,12,2)])+'''
符号負はr2が低費用。G0/P0のP5ではr2の利益は約86.0万Tにすぎず、P1ではr2の追加増価がr1より大きく約88.3万Tの不利へ反転する。
G1ではr1の大きなshot削減がr2の反復費用節約を上回り、全六方式の各価格でr1が低いh=0 Tとなる。
保証と価格を同時変更した一つの効果として扱わず、既存factorialを参照する。
''')
    sections.append('## primary partial/P5と全六方式\n\nh=0 Tについて、primaryは両方式へ同じr=1,2自由度を与える。\n\n'+table(
        ['ケース','partialの選択','P5の選択','P5−partial T','partial分母のP5削減'],[
            [v['ledger'],label(v['metrics']['T']['primary_each_arm_equal_r_freedom_h0']['selected']['partial_return_tail']),
             label(v['metrics']['T']['primary_each_arm_equal_r_freedom_h0']['selected']['closed_P5_full']),
             num(v['metrics']['T']['primary_each_arm_equal_r_freedom_h0']['P5_minus_partial']),
             pct(v['metrics']['T']['primary_each_arm_equal_r_freedom_h0']['P5_reduction_fraction_of_partial'])] for v in selection]))
    sections.append('\n全12構成の勝者・次点は既存rankingから直接読む。次点が同方式の別rの場合も明記する。\n\n'+table(
        ['ケース / 指標','最小','次点','次点−最小','差/次点'],[
            [v['ledger']+' / '+m,label(v['metrics'][m]['subsets']['all12']['winner'][0]),
             label(v['metrics'][m]['subsets']['all12']['runner_up'][0]),
             num(v['metrics'][m]['subsets']['all12']['runner_up_minus_winner']),
             pct(v['metrics'][m]['subsets']['all12']['gap_fraction_of_runner_up'])] for v in selection for m in METRICS]))
    sections.append('''
G0_P1のT次点は同じP3+tailのr1。異なる方式で最も近い対照はr1 P5、差約916.323万T。
G1_P1のT次点はr1 P5で前述約59.897万T。
primaryのP5 T優位と、全六方式でP3+tailがT最小になることは矛盾しない。
G1_P1のnative1Q最小はpartial、CX最小はCTSで、T最小方式に全指標優位を付与しない。
各方式のr選択を許した保存順序、primary各指標のbest-r対照、exact winner/runner-up差は
[選択した比較のexact記録]('''+artifact+'''selected_saved_comparisons_v1.json)。新しい48行rankingやParetoは生成しない。

## 準備・読出し費用h>=0の領域

保存affineは各指標別 `G(h)=A+Bh`、B=2N Z^r。
hはaccepted whole quantum call当たりの準備・読出し費用、三指標を独立表示するformal domain。
physical条件h_1Q>=h_Tは保持し、未知classical online費用はNOT_ACQUIREDで0としない。
h=0はmodelの断面で、実装の準備・読出しが無料と測定された意味ではない。
以下の領域は既存comparisonのminimum_open_regionsを直接表示した。端点のtieは原JSONにある。
''')
    lines=[]
    for g in LEDGERS:
        for section in ['all12','primary_equal_r_choice']:
            for m in METRICS:
                segs=[]
                for v in comparisons[g][section][m]['minimum_open_regions']:
                    hi='∞' if v['open_hi'] is None else num(v['open_hi'],6)
                    segs.append('('+num(v['open_lo'],6)+', '+hi+'): '+', '.join(label(a) for a in v['minimum_arms']))
                lines.append([g,section,m,'; '.join(segs)])
    sections.append('\n'+table(['ケース','候補集合','指標','保存open領域: 最小構成'],lines))
    sections.append('''
全hの既存[projection](../../../artifacts/track_b_d3_result/2026-10-14/v2/saved_output_audit/saved_comparison_projection_v2.json)と
`comparisons/<ledger>.json`のexact endpointを参照。丸め境界で順位判定しない。
G1_P1のP3+tail/r1利益はTのhが約15.098未満の領域で成立する。
primaryだけを見る場合はr1 P5が約226.932まで最小だが、全方式ではP3+tail/r2が先に登場するためprimaryの領域を全方式へ移転しない。

fixed-rのP5−partialのaffineは選択記録へ保存。Tでは全ケース/rでP5が低く、slope差も負なのでh>=0で逆転しない。
CX/r1はP5が低いが、CX/r2はh=0でpartialが低く、P5へ変わる交点はG0で約6.637185、G1で約6.642139。
native1Q/P1/r1はh=0でpartialが低く、P5へ変わる交点はG0で約223.857638、G1で約223.985033。
これら二方式の交点は全六方式の最小領域と同義ではない。

## 何を説明でき、何を取得していないか

保存価格差から全方式/rのT・1Q増価、P3/P5逆転の主要root key、N変化から保証効果とinteractionをexactに説明した。
全法のh領域は既存資料を再利用。脆弱な比較は、G0_P0 T勝者/次点の約0.096%、G1_P1 P3/P5の約0.170%、
P1/r1 key tieの小変位、P0→P1で反転するP5/partial native1Q、P5自身のG0/P0からG0/P1へのr選択である。
固定値の数学的誤差と、合成価格を別取得した場合の不確かさを混同しない。

未取得は、別seed/provider/precisionでのsequence価格、実装可能なtie価格、whole-circuit最適化、
古典online生成費用、実量子shot数、独立validation、DF/分子/一般Hamiltonianへの成立範囲。
追加取得は今回行わない。各ledgerのconfidenceは個別12行familywise0.05で、四ledgerを一つの96axis同時保証と呼ばない。

## 検証・公開とGPTへ返す論点

追加の保存算術同値checks '''+str(checks)+'''件PASS、FAIL0。
9人工unit tests（acceptance込み出現回数、signed価格寄与、shot/単価恒等式、affine/tie、bytes改ざん、未宣言入力、固定commit拒否、有理数保存）がPASS。
これはlocal技術検証で、旧監査の再実行、CI/外部独立再現、新しい科学結果取得ではない。
[検証・scope記録]('''+artifact+'''validation_and_scope_v1.json)、[証拠manifest]('''+artifact+'''evidence_manifest_v1.json)。
新しい分析sourceはstdlib保存値専用でscience libraryをimportせず、登録solver/shot budget/native価格kernel/runnerを呼ばない。
公開後に新報告・evidenceのremote bytes/SHA、branch HEAD、worktree cleanと親既存変更保護を照合する。

GPTには、今回の価格依存と小marginからどこまで一般化できるか、誤差保証と表現選択の成立条件をどう主張するか、
追加の独立検証に研究上の情報価値があるか、論文の着地点を判断する材料として返す。
Codexは科学的GO/STOP、アルゴリズム採択、新規性、論文化の十分性を決めない。
GPTの新しい詳細科学レビューはユーザーの明示開始承認後のみ。
**技術分析完了後mandatory STOP。追加の合成・再会計・r/provider探索・compile・pair評価・SP/DF/分子へ進まない。**
''')
    report.write_text('\n'.join(sections))


class Inputs:
    def __init__(self, directory, receipt):
        self.directory = Path(directory)
        self.receipt = receipt
        if receipt['result_commit'] != RESULT_COMMIT or not receipt['all_identities_PASS']:
            raise ValueError('fixed remote identity required')
        self.identities = {x['path']: x for x in receipt['files']}
        self.opened = set()

    def read(self, path):
        path = str(path)
        record = self.identities[path]  # undeclared inputs refused
        data = (self.directory/path).read_bytes()
        if len(data) != record['bytes'] or hashlib.sha256(data).hexdigest() != record['sha256']:
            raise ValueError('input identity changed: '+path)
        self.opened.add(path)
        return json.loads(data)


def analyze(inputs, output):
    start_wall, start_cpu = time.monotonic(), time.process_time()
    output = Path(output)
    rows = inputs.read(D3/'resource_map.json')['rows']
    inventory = inputs.read(D3/'frozen_inventory.json')
    old = inputs.read(OLD)
    new = inputs.read(D3/'synthesis_cache_new.json')
    effects = inputs.read(D3/'factorial_effects.json')
    comparisons = {g: inputs.read(D3/'comparisons'/f'{g}.json') for g in LEDGERS}
    projection = inputs.read(D3/'saved_output_audit/saved_comparison_projection_v2.json')
    boundary_cert = inputs.read('artifacts/track_b_d3_preparation/2026-10-13/v1/static_boundary_price_certificate_v1.json')
    assert boundary_cert['all_rows_no_R_cancellation'] is True
    coefficients = [inputs.read(D2/'rows'/f'row_{i:02d}.json') for i in range(12)]
    assert set(new) == {k['ratio'] for k in inventory['keys']} and len(new) == 36
    checks = []

    def check(label, value):
        checks.append({'check': label, 'PASS': bool(value)})
        if not value:
            raise AssertionError(label)

    for view in projection['views']:
        g = view['ledger']
        check(g+':saved ranking projection', view['all12_h0_rankings'] ==
              {m: comparisons[g]['all12'][m]['h0_ranking'] for m in METRICS})
    check('saved no-R certificate', all(r['no_R_can_be_cancelled'] for r in boundary_cert['rows']))
    price_rows = []
    for ki, entry in enumerate(inventory['keys']):
        k = entry['ratio']
        # All recorded words have zero lowercase-t; do not substitute a price convention.
        check(k+':saved price convention', old[k]['Tdagger_count'] == new[k]['Tdagger_count'] == 0)
        usage = []
        for i in entry['rows']:
            c = coefficients[i]
            match = next(v for v in c['keys'] if v['ratio'] == k)
            usage.append(dict(row_index=i, arm=c['arm'], steps=c['steps'],
                              conditional_step_occurrences=c['key_multiplicity_conditional'][k],
                              event_indices=match['event_indices'],
                              saved_event_index_source=str(D2/'rows'/f'row_{i:02d}.json')))
        price_rows.append(dict(key_alias=f'K{ki:02d}', ratio=k, angle_convention='signed atan(ratio), native RZ(theta) radians; scale=1',
                               old_T=old[k]['T_count'], new_T=new[k]['T_count'],
                               delta_T=new[k]['T_count']-old[k]['T_count'],
                               old_1Q=old[k]['one_qubit_count'], new_1Q=new[k]['one_qubit_count'],
                               delta_1Q=new[k]['one_qubit_count']-old[k]['one_qubit_count'],
                               old_sequence_sha256=old[k]['sequence_sha256'],
                               new_sequence_sha256=new[k]['sequence_sha256'],
                               old_inventory_key=entry['old_key'], new_inventory_key=entry['new_key'],
                               uses=usage))
    (output/'key_prices_and_incidence_v1.json').write_text(dumps(dict(
        kind='SAVED_PRICE_METADATA_AND_EXISTING_CONDITIONAL_INCIDENCE', keys=price_rows,
        negative_rotation='actual adjoint, same stored cost', global_W='excluded from 1Q, strict phase already guarded',
        new_synthesis=0)))
    with (output/'key_prices_display_v1.csv').open('w', newline='') as f:
        w=csv.writer(f, lineterminator='\n')
        w.writerow(['key_alias','ratio','old_T','new_T','delta_T','old_1Q','new_1Q','delta_1Q','used_arm_r'])
        for p in price_rows:
            w.writerow([p[k] for k in ['key_alias','ratio','old_T','new_T','delta_T','old_1Q','new_1Q','delta_1Q']]+[
                ';'.join(f"r{u['steps']}:{u['arm']}" for u in p['uses'])])
    row_changes = []
    per_key = {}
    for i, c in enumerate(coefficients):
        base = rows['G0_P0'][i]; r=base['steps']; z=F(base['step_acceptance'])
        mu = {k:F(v) for k,v in c['key_multiplicity_conditional'].items()}
        check(f'row{i}:incidence context', r==c['steps'] and base['arm']==c['arm'] and z==F(c['acceptance']))
        nu = {k:whole_incidence(v,r,z) for k,v in mu.items()}
        data = dict(row_index=i, arm=c['arm'], steps=r, acceptance_step=z, acceptance_path=z**r,
                    whole_trial_key_occurrences=nu, metrics={}, effects_by_guarantee={})
        for m, field in [('T','T_count'),('1Q','one_qubit_count')]:
            p0={k:old[k][field] for k in mu}; p1={k:new[k][field] for k in mu}
            # 1Q literal residual is inferred from saved scalar; T has independent saved literal coefficient.
            literal = F(base['per_step_trial_native'][m])/z - sum(mu[k]*p0[k] for k in mu)
            if m=='T':
                check(f'row{i}:independent literal T coefficient',literal==F(c['literal_T_conditional']))
            step_delta = z*sum(mu[k]*(p1[k]-p0[k]) for k in mu)
            whole_delta = sum(nu[k]*(p1[k]-p0[k]) for k in nu)
            check(f'row{i}:{m}:step price attribution',step_delta ==
                  F(rows['G0_P1'][i]['per_step_trial_native'][m])-F(base['per_step_trial_native'][m]))
            check(f'row{i}:{m}:whole price attribution',whole_delta ==
                  F(rows['G0_P1'][i]['per_whole_trial_native'][m])-F(base['per_whole_trial_native'][m]))
            data['metrics'][m] = dict(literal_conditional=literal,
                literal_source='independent saved D2 coefficient' if m=='T' else 'saved P0 scalar minus saved incidence*prices',
                step_price_delta=step_delta, whole_trial_price_delta=whole_delta)
            for g in ['G0','G1']:
                a,b=rows[g+'_P0'][i],rows[g+'_P1'][i]; n=a['budget']['N_per_axis']
                check(f'row{i}:{g}:unchanged price shot budget',a['budget']==b['budget'])
                values=contributions(nu,p0,p1,n); per_key[(g,i,m)]=values
                ef=next(e for e in effects if e['row_index']==i and e['coordinate']==m)
                check(f'row{i}:{g}:{m}:sum=factorial price-only',sum(values.values(),F(0))==F(ef['price_only_'+g]))
                check(f'row{i}:{g}:{m}:literal saving fixed',a['literal_boundary_saving_per_whole_trial']==b['literal_boundary_saving_per_whole_trial'])
                data['effects_by_guarantee'].setdefault(g,dict(N_per_axis=n,metrics={}))['metrics'][m]=dict(
                    expected_occurrences_both_axes={k:2*n*v for k,v in nu.items()},
                    price_delta_by_key=values,total=sum(values.values(),F(0)),
                    saved_factorial_source=f'factorial_effects.json row={i} coordinate={m}')
        data['metrics']['CX']=dict(price_delta=F(0),reason='saved literal-only CX prices independent of R synthesis')
        (output/'rows'/f'row_{i:02d}.json').parent.mkdir(exist_ok=True)
        (output/'rows'/f'row_{i:02d}.json').write_text(dumps(data))
        row_changes.append(data)
    focus=[]
    for g in ['G0','G1']:
        for r,ia,ib in [(1,2,4),(2,8,10)]:
            # A=P3, B=P5; positive margin means P3 cheaper.
            a0,b0=rows[g+'_P0'][ia],rows[g+'_P0'][ib]
            a1,b1=rows[g+'_P1'][ia],rows[g+'_P1'][ib]
            margin0=F(b0['expected_native_intercept']['T'])-F(a0['expected_native_intercept']['T'])
            margin1=F(b1['expected_native_intercept']['T'])-F(a1['expected_native_intercept']['T'])
            diff={k:per_key[(g,ib,'T')].get(k,F(0))-per_key[(g,ia,'T')].get(k,F(0))
                  for k in set(per_key[(g,ia,'T')])|set(per_key[(g,ib,'T')])}
            check(f'{g}:r{r}:P3-P5 margin price reconciliation',sum(diff.values(),F(0))==margin1-margin0)
            na,nb=a1['budget']['N_per_axis'],b1['budget']['N_per_axis']
            slopes={k:2*nb*row_changes[ib]['whole_trial_key_occurrences'].get(k,F(0))-
                        2*na*row_changes[ia]['whole_trial_key_occurrences'].get(k,F(0)) for k in diff}
            sensitivity=[]
            for k,v in slopes.items():
                tie=single_key_tie(margin1,v,new[k]['T_count'])
                if tie['displacement'] is not None:
                    check(f'{g}:r{r}:{k}:symbolic tie',margin1+v*tie['displacement']==0)
                sensitivity.append(dict(ratio=k,old_T=old[k]['T_count'],new_T=new[k]['T_count'],
                    observed_price_change_margin_contribution=diff[k],margin_per_T_price_unit=v,**tie))
            # Key ordering here is descriptive magnitude, never method/candidate ranking.
            sensitivity.sort(key=lambda v:abs(v['observed_price_change_margin_contribution']),reverse=True)
            item=dict(guarantee=g,steps=r,A='closed_P3_tail',B='closed_P5_full',
                saved_rows={'A':ia,'B':ib},old_price_B_minus_A=margin0,new_price_B_minus_A=margin1,
                margin_fraction_of_B=margin1/F(b1['expected_native_intercept']['T']),
                price_margin_shift=margin1-margin0,N_A=na,N_B=nb,
                whole_trial_T_A=F(a1['per_whole_trial_native']['T']),whole_trial_T_B=F(b1['per_whole_trial_native']['T']),
                old_whole_trial_T_A=F(a0['per_whole_trial_native']['T']),old_whole_trial_T_B=F(b0['per_whole_trial_native']['T']),
                shot_unit_symmetric_allocation=symmetric_product_gap(na,a1['per_whole_trial_native']['T'],nb,b1['per_whole_trial_native']['T']),
                key_price_sensitivity=sensitivity,
                prep_relation=affine_relation((a1['expected_native_intercept']['T'],a1['prep_readout_slope']),
                                             (b1['expected_native_intercept']['T'],b1['prep_readout_slope'])),
                imaginary_prices_are_not_acquired_sequences=True)
            focus.append(item)
    (output/'P3_vs_P5_price_mechanisms_v1.json').write_text(dumps(dict(comparisons=focus,
        same_finite_P5_target=True,single_key_ties='symbolic other prices/N/law fixed; no sequence feasibility claim')))
    repeat=[]
    for i in range(6):
        for p in ['P0','P1']:
            entry=dict(arm=rows['G0_P0'][i]['arm'],price=p,by_r={})
            for r,j in [(1,i),(2,i+6)]:
                a,b=rows['G0_'+p][j],rows['G1_'+p][j]
                saved=next(e for e in effects if e['row_index']==j and e['coordinate']=='T')
                dt=F(b['expected_native_intercept']['T'])-F(a['expected_native_intercept']['T'])
                check(f'{p}:row{j}:guarantee scalar explanation',dt ==
                      2*(b['budget']['N_per_axis']-a['budget']['N_per_axis'])*F(a['per_whole_trial_native']['T']))
                check(f'{p}:row{j}:saved guarantee/interaction',dt==F(saved['guarantee_only'])+
                      (F(saved['interaction']) if p=='P1' else 0))
                entry['by_r'][str(r)]=dict(row=j,N_G0=a['budget']['N_per_axis'],N_G1=b['budget']['N_per_axis'],
                    N_ratio=F(b['budget']['N_per_axis'],a['budget']['N_per_axis']),
                    Taylor_product_G0=a['budget']['Taylor_product_upper'],Taylor_product_G1=b['budget']['Taylor_product_upper'],
                    remaining_G0=a['budget']['remaining'],remaining_G1=b['budget']['remaining'],
                    whole_trial_T=a['per_whole_trial_native']['T'],T_change=dt,
                    saved_factorial_guarantee=saved['guarantee_only'],saved_factorial_interaction=saved['interaction'])
            # These two-value contrasts explain r, not a regenerated all-candidate ranking.
            entry['r2_minus_r1_T']={g:F(rows[g+'_'+p][i+6]['expected_native_intercept']['T'])-
                                    F(rows[g+'_'+p][i]['expected_native_intercept']['T']) for g in ['G0','G1']}
            entry['best_r_h0_from_saved_ranking']={g:next(
                label for group in comparisons[g+'_'+p]['all12']['T']['h0_ranking'] for label in group
                if label.split(':',1)[1]==entry['arm']) for g in ['G0','G1']}
            repeat.append(entry)
    (output/'guarantee_and_repeat_mechanisms_v1.json').write_text(dumps(dict(
        entries=repeat,sufficient_forecast_not_measured_or_minimum_shots=True,
        new_N_evaluations=0,native_error_account_fixed='1/1000000')))
    selection=[]
    for g in LEDGERS:
        bylabel={f"r{r['steps']}:{r['arm']}":r for r in rows[g]}
        view=dict(ledger=g,metrics={},fixed_r_primary={})
        for m in METRICS:
            v=dict(saved_comparison_path=str(D3/'comparisons'/f'{g}.json'),subsets={})
            for section in ['all12','primary_equal_r_choice']:
                rank=comparisons[g][section][m]['h0_ranking']
                w,nxt=rank[0],rank[1]
                wr,nr=bylabel[w[0]],bylabel[nxt[0]]
                gap=F(nr['expected_native_intercept'][m])-F(wr['expected_native_intercept'][m])
                v['subsets'][section]=dict(winner=w,runner_up=nxt,
                    winner_cost=wr['expected_native_intercept'][m],runner_up_cost=nr['expected_native_intercept'][m],
                    runner_up_minus_winner=gap,gap_fraction_of_runner_up=gap/F(nr['expected_native_intercept'][m]),
                    h_regions_source_field=section+'.'+m+'.minimum_open_regions',
                    h_endpoints_source_field=section+'.'+m+'.minimum_switch_endpoint_ties')
                distinct=next(label for group in rank for label in group
                              if label.split(':',1)[1]!=w[0].split(':',1)[1])
                v['subsets'][section]['nearest_distinct_arm']=distinct
                v['subsets'][section]['distinct_arm_minus_winner']=F(bylabel[distinct]['expected_native_intercept'][m])-F(wr['expected_native_intercept'][m])
            # One selected r per arm projected in existing saved order.
            seen=set();choices=[]
            for group in comparisons[g]['all12'][m]['h0_ranking']:
                for label in group:
                    arm=label.split(':',1)[1]
                    if arm not in seen:seen.add(arm);choices.append(label)
            v['each_arm_first_saved_label_h0']=choices
            pair={arm:next(label for group in comparisons[g]['primary_equal_r_choice'][m]['h0_ranking']
                          for label in group if label.split(':',1)[1]==arm)
                  for arm in ['partial_return_tail','closed_P5_full']}
            a,b=bylabel[pair['partial_return_tail']],bylabel[pair['closed_P5_full']]
            v['primary_each_arm_equal_r_freedom_h0']=dict(selected=pair,
                partial_cost=a['expected_native_intercept'][m],P5_cost=b['expected_native_intercept'][m],
                P5_minus_partial=F(b['expected_native_intercept'][m])-F(a['expected_native_intercept'][m]),
                P5_reduction_fraction_of_partial=1-F(b['expected_native_intercept'][m])/F(a['expected_native_intercept'][m]))
            view['metrics'][m]=v
        for r,ia,ib in [(1,1,4),(2,7,10)]:
            a,b=rows[g][ia],rows[g][ib]
            view['fixed_r_primary'][str(r)]={m:affine_relation(
                (a['expected_native_intercept'][m],a['prep_readout_slope']),
                (b['expected_native_intercept'][m],b['prep_readout_slope'])) for m in METRICS}
        selection.append(view)
    (output/'selected_saved_comparisons_v1.json').write_text(dumps(dict(views=selection,
        new_ranking_or_global_intersections=False,primary='partial_return_tail vs closed_P5_full',
        h_units='per accepted whole quantum call, separately T/CX/1Q',classical_online='NOT_ACQUIRED; not zero')))
    (output/'validation_and_scope_v1.json').write_text(dumps(dict(
        kind='ADDITIONAL_SAVED_MECHANISM_IDENTITIES_ONLY',checks=checks,PASS=len(checks),FAIL=0,
        remote_input_files_opened=sorted(inputs.opened),python=platform.python_version(),
        wall_seconds=time.monotonic()-start_wall,CPU_seconds=time.process_time()-start_cpu,
        peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        old_2579_audit_reexecuted=False,new_synthesis=0,new_N_evaluation=0,event_or_pair_evaluations=0,
        new_ranking=0,new_science=0,circuit=0,compile=0,matrix=0,sampling=0,LP=0,DF=0,molecule=0,GPU=0,
        scientific_classification_changed=False,method_adoption=False,mandatory_STOP=True)))
    write_report(output,rows,focus,repeat,selection,price_rows,comparisons,len(checks))
    return dict(checks=len(checks),keys=36,row_coefficients=12)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output)
    receipt=json.loads((out/'input_remote_identity_v1.json').read_text())
    print(analyze(Inputs(a.inputs,receipt),out))
