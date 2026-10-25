#!/usr/bin/env python3
"""Render completed saved comparisons; no new target, sampling or ranking."""
import csv,json
from fractions import Fraction as Q
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/pauli_rte_event_extension/2026-10-26'
def read(p):return json.loads(p.read_text())
def num(x):return '—' if x is None or x=='' else format(float(Q(str(x))),'.7g')
def main():
    a=read(OUT/'comparison_details.json');v=read(OUT/'verification.json');summary=read(OUT/'result_summary.json')
    assert read(OUT/'STUDY_STOP.json')['mandatory_STOP'] and read(OUT/'archive_restore_verification.json')['status']=='PASS'
    source=read(OUT/read(OUT/'CURRENT_SOURCE.json')['path'])['commit']
    lines=['# H4/H6 Pauli-JW：純RTEの総event数拡張と保存Hybridとの比較','',
        '2026-10-26。branch `codex/pauli-rte-event-extension-execution-20261026`。',
        f'準備commit `c6d42d925b258939768a8c6a34ee3f48db27b964`、科学SOURCE `{source}`。','',
        '## 結果と適用範囲','',
        '総event数s=q*rを64より広げ、固定q=8の新候補を評価した。以下は固定分子・仮想ρ=0・有限価格標本による条件付きfamily比較である。',
        'RTEの追加パラメータ探索に対してHybridは保存済み探索・独立確認を固定した。現行保存family間の比較であり、両familyの大域最適値の比較ではない。',
        '実compiledρ、母平均価格tail、全パラメータ大域最適性、実機/FT/QPE/RPE費用は保証しない。','',
        '**H6では調査grid上の最小がs=512（q8/r64/K2）へ移り、s=1024でGが上昇した。独立確認48本でも全5精度でRTEが保存Hybridより低く、Hybrid/RTEは約1.814–5.938となった。旧s64までのHybrid優位は、この拡張RTE familyとの条件付き比較では逆転する。**',
        '**H4ではs=128・256でGが上がり、RTE最良は旧s64に留まった。方式間の結果はε=.05でHybrid低G、他4精度でRTE低Gのままである。**','',
        '| 分子 | ε_sig | 確認集合RTE最小の q/r/K (s) | G RTE | G Hybrid | Hybrid/RTE | RTE/旧RTE | 名目比較 |',
        '|---|---|---|---:|---:|---:|---:|---|']
    for x in a:
        c=x['comparison'];r=x['RTE']
        lines.append('| '+' | '.join([c['molecule'],c['epsilon_sig'],f"{r['q']}/{r['r']}/{r['K']} ({r['s']})",num(r['G_exact']),num(x['Hybrid']['G_exact']),num(c['Hybrid_over_RTE_exact']),num(c['new_RTE_over_old_exact']),c['classification']])+' |')
    lines+=['','Gの単位はcompiled RZ×shot。H/R>1はRTEの標本平均条件付きGが低いことを示す。表の最小は有限の確認済み集合からの事後選抜であり、全探索候補の独立確認や母平均winner保証ではない。',
        '探索前固定・探索選抜・独立確認・確認後集合minを区別する。探索と確認は混合しない。5精度は同じ価格標本を共有する。','',
        '## sの増加とB/N/Cのトレードオフ','',
        '新候補の探索価格n=8、確認は48から固定条件で必要時のみ96/192/384へ延長した。全new targetは候補固有のlaw/B/bias/uで会計した。',
        '次表はε=.0005のK2候補。保存旧s64と新探索/確認を別行で示す。B低下によるshot節約と、event数増大による単回C増加を区別する。','',
        '| 分子 | s | cohort (n) | B | N_cos+N_sin | C_cos | C_sin | G | G SE |',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    with (OUT/'event_cost_trends.csv').open() as f:trends=list(csv.DictReader(f))
    for r in sorted([r for r in trends if r['epsilon_sig']=='0.0005' and r['q']=='8' and r['K']=='2' and int(r['s'])>=64],key=lambda r:(r['molecule'],int(r['s']),r['cohort'])):
        n=None if not r.get('N_cos') or not r.get('N_sin') else str(int(r['N_cos'])+int(r['N_sin']))
        lines.append('| '+' | '.join([r['molecule'],r['s'],r['cohort']+' ('+str(r['price_n'])+')',num(r.get('B_exact') or r.get('B_saved_decimal')),n or '—',num(r['C_cos_exact']),num(r['C_sin_exact']),num(r['G_exact']),num(r['G_SE'])])+' |')
    lines+=['','全5精度・両K・cohort別のbias/u/B/N/C/G/SE/比はCSVに収録する。Gの下降から上昇は調査grid上の経験的な最小近傍を示し、連続域の単峰性や大域最適性の証明ではない。',
        'H6のK2ではs64→512でB約20.6408→1.47567、最厳精度のN_total約6.0852e10→3.0536e8、確認C_cos約424.10→3114.98となる。shot節約が単回価格増加を上回った。s1024ではB約1.21485へさらに下がる一方、C_cos約6239.5へ増え、Gは上昇した。',
        '探索上限に最小が残った場合は最適域未確認である。未確認の隣接候補は探索値のまま表示し、確認済み価格へ格上げしない。','',
        '## 選抜と統計的不確かさ','',
        '| 分子 | ε | 探索nominee | nominee確認G | 確認集合min | n | G SE | 上限sでminか |',
        '|---|---|---|---:|---|---:|---:|---|']
    for x in a:
        c=x['comparison'];lines.append('| '+' | '.join([c['molecule'],c['epsilon_sig'],str(c['frozen_nominee_id']),num(c['nominee_confirmation_G_exact']),str(c['RTE_id']),str(c['RTE_n']),num(c['RTE_G_SE']),str(c['confirmed_min_on_upper_boundary'])])+' |')
    lines+=['','名目比較は差が2 combined-SE以内ならNEAR_TIE_OR_UNRESOLVEDとする記述的規則であり、多重選抜や逐次観察を補正した母平均順位保証ではない。価格のcosine/sineは同一trajectoryのpaired標本として共分散を含める。',
        '延長は固定Hybridまたは隣接sのRTEとの近接、名目t95半幅>1%G、候補側SEが比較側floorを超える場合だけ行う。各nでの判断はconfirmations archiveのdecisions.jsonに保存した。',
        'K2のorder4はlawのsupport外である。K4で未観測のorder4やevent componentは費用寄与0としない。標本価格・SEは母平均価格tailの厳密上界ではない。','',
        '## 科学条件・数値包含・再利用','',
        'Linear H4/H6、隣接1.00Å、STO-3G、中性singlet。保存Pauli-JWの非identity項H4=184/H6=918とconstant、同じraw state/full-H reference・共通exact normを使用した。再生成・再正規化なし。',
        'S2/T=.8、L_D=0、q8。Kは最大偶数event次数で、多項式次数K+1、PF次数S2とは異なる。sだけのq/r完全同値は仮定せずq8を固定した。',
        '既存RTE sampler/AuditedPauliSignalAdapter、D2 central-rotation control、full Hadamard wrapperのcosine/+Im sine、constant/global/relative phaseと演算順序を維持した。',
        'Qiskit1.3.0、rz/sx/x/cx、optimization_level1、seed17、topology/routing追加指定なし。状態準備費用を除くcompiled RZを使用した。',
        'H4は固定d3fdfadのFLINT mean-matrix/vector接続、64→96→128bitの条件付きschedule。H6は既存matrix-free tight L2、Arb scalar128bit/vector中心53bitを使用した。新しいcompiled信号の状態作用は行わない。',
        '理想law meanの包含はcompiled実装残差ρの包含ではない。B_savedを信号補正へ一回適用し、shotのB²rangeとは区別する。保存referenceの区間と同じraw normを使う。',
        'α_axis=.025、ε_axis=ε_sig/√2、h=ε_axis−|bias|−u、u/h<=.01、N=ceil(A_upper/h²)の元十分条件と任意精度整数会計を保持した。実ρは未保証なのでN/Gは仮想ρ=0である。',
        '旧h4_conditional_cng/h6_rte_fairnessのscale/shot会計・paired統計を再利用。rのvalidatorだけを私有namespaceで拡張し、旧kernel/CLI/output/STOPは変更しなかった。',
        'H4/H6それぞれの元SeedSequenceを維持し、探索101/確認202、candidate IDとindexを固定した。計画全4704 seedの衝突0、実使用seedを照合した。PRNG streamの数学的独立性証明ではない。','',
        '## 追加候補・終了と保全','',
        '初期H6 r16/32/64 K2、H4 r16 K2/K4。固定適応条件に該当した候補だけをfresh batchで追加し、decisions/とstage planに先行固定した。H4追加理由は改善だけでなく、保存Hybridとの順位変化の条件を含む。',
        'H6 K4はmaterial bias/ambiguity/Hybrid proximity条件に該当した場合のみ許可され、無条件gridは実施していない。非実行候補は0費用として扱わない。',
        f"新候補{v['new_candidates']}、target状態作用{v['target_state_actions']}、新trajectory{v['trajectories']}、両軸compile{v['axis_compiles']}。新Hamiltonian/state/reference/H2/Hybrid scienceはいずれも0。",
        '探索64本（8候補×8）と独立確認96本（2候補×48）。全10確認精度行で延長条件に該当せず、96/192/384本への延長0。H4の4 targetは64bitだけ、H6の4 targetは128bit scalar/53bit vectorの一作用だけで完了した。新8候補×5精度の40行はすべて条件付きguard適格で、新規null補完はない。',
        f"Native child {v['native_children']}件すべてexit0/reap。全新QPY {v['all_new_QPY_byte_SHA_checked']}個のgzip/raw SHAを照合し、代表compiled QPY {v['representative_compiled_QPY_decoded']}個のwire/basis/RZ/CXをdecode照合した。代表構造検査を全回路の数学的compiler証明とは呼ばない。",
        '準備人工12件（新7＋再利用5）と実行規則人工7件PASS。これはprimary Codexの技術検証であり、独立科学レビューや科学的費用優位の証明ではない。',
        '独立taskを実際の空きCPUに応じたprocess workerで並列化し、対象processだけnice10で実行した。小さなmatrix/vector処理はBLAS1threadとtask並列を組み合わせ、oversubscriptionを避けた。共有設定・他者process/fileへの変更なし。',
        '固定SOURCE/input/runtime、native exit、raw event/QPY、target、stage/confirmation STOPを保存した。既存の可逆QPY-aware圧縮とgeneric restorerで元bytesのSHA復元を確認した。元rawは削除・変更していない。',
        '元s64以下の成功runは再実行せず保存値を引用した。H6最厳精度の旧guard未確定2行などはnullを保持し、0/∞で順位付けしない。旧SOURCE/raw/grant/ledger/marker/STOPの変更なし。',
        '新しい科学grant・承認証明基盤は作成していない。旧disabled planはそのまま保全し、新active scopeを別出力に固定した。active scope内の候補descriptor初期statusは計画時の値であり、実際の取得状態は最終CSV/target/native記録が示す。','',
        '## データ入口と次の担当','',
        '- [全新候補・精度・cohort](../artifacts/pauli_rte_event_extension/2026-10-26/candidate_precision_table.csv)',
        '- [保存旧値を含むs別比較](../artifacts/pauli_rte_event_extension/2026-10-26/event_cost_trends.csv)',
        '- [精度別確認比較](../artifacts/pauli_rte_event_extension/2026-10-26/best_by_precision.csv)／[詳細JSON](../artifacts/pauli_rte_event_extension/2026-10-26/comparison_details.json)',
        '- [図と証拠の利用方法](../artifacts/pauli_rte_event_extension/2026-10-26/README.md)／[技術照合](../artifacts/pauli_rte_event_extension/2026-10-26/verification.json)',
        '- [H6旧s64比較](h6_rte_fairness_20261025.md)／[H4/H2旧比較](pauli_hybrid_rte_comparison_20261025.md)／[準備](pauli_rte_event_extension_preparation_20261026.md)','',
        '新r>8のmatched Hybridは未取得/null。旧同一q/r/K対照は旧報告に保存されており、今回のfamily差を同一条件のprefixだけの因果効果とは扱わない。',
        '科学計算はSTUDY_STOP.jsonで終了。commit/publishとGitHub remote SHA照合後もmandatory STOPを維持する。',
        '次の担当：ユーザー/GPTが、この有限候補・条件付き結果を研究上どう扱うか判断する。Codexは追加科学計算、主要claim正式採択、詳細レビュー、論文変更、母平均保証の新証明へ自動移行しない。必要なら対象を限定した別承認後に進める。','']
    with (ROOT/'docs/pauli_rte_event_extension_20261026.md').open('x') as f:f.write('\n'.join(lines))
if __name__=='__main__':main()
