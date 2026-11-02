#!/usr/bin/env python3
"""Render completed H2 saved results, no new scientific accounting or ranking."""
import csv,json
from fractions import Fraction as Q
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/h2_pauli_rte_validation/2026-10-26'
def read(p):return json.loads(p.read_text())
def num(x):return '—' if x is None or x=='' else format(float(Q(str(x))),'.8g')
def main():
    data=read(OUT/'comparison_details.json');v=read(OUT/'verification.json');source=read(OUT/'SOURCE.json')['commit']
    assert read(OUT/'STUDY_STOP.json')['mandatory_STOP'] and read(OUT/'archive_restore_verification.json')['status']=='PASS'
    rows=list(csv.DictReader((OUT/'candidate_precision_table.csv').open()));mapping=list(csv.DictReader((OUT/'old_candidate_confirmation_map.csv').open()))
    text=['# H2 Pauli-JW：純RTEの追加独立確認とevent数拡張','',
      '2026-10-26。branch `codex/h2-pauli-rte-confirmation-extension-20261026`。',
      f'基点 `10cb3e5cca7834ebf75301bd91bd093ca51d4a6b`、科学SOURCE `{source}`。','',
      '## 得られた結果と範囲','',
      '**未確認の有望純RTE候補を独立確認すると、旧確認集合でのHybrid低Gから、全5精度でRTEの標本平均Gが低い結果へ変わった。**',
      '更新後の確認集合minはε=.05/.01でq1/r4/K2、他3精度でq2/r2/K4（いずれもs=4、n384）。保存HybridとのG比H/Rは約1.086–1.111。',
      '**s64・128への拡張では新たな改善が現れなかった。** Bは1へ近づくが、必要shotの節約は小さく、compiled RZ価格増加によりGが高くなる。追加s128K4/s256・Phase B確認は固定条件に該当せず実施しなかった。',
      'これは固定H2・有限候補・仮想ρ=0・有限標本価格による条件付き比較である。確認後にminを選んだ値を含み、母平均winner・連続域の大域最適性・実compiled精度の保証ではない。','',
      '| ε_sig | 旧確認RTE G (s2/n96) | 保存Hybrid G (n96) | 更新RTE ID | 更新RTE G | G SE RTE | H/R | 名目比較 |',
      '|---|---:|---:|---|---:|---:|---:|---|']
    for x in data:
      c=x['comparison'];text.append('| '+' | '.join([c['epsilon_sig'],num(c['old_RTE_G_exact']),num(c['Hybrid_G_exact']),c['updated_RTE_id'],num(c['updated_RTE_G_exact']),num(c['updated_RTE_G_SE']),num(c['Hybrid_over_RTE_exact']),c['classification']])+' |')
    text+=['','Gの単位はcompiled RZ×shot。H/R>1はRTE標本平均Gが低いことを示す。旧確認RTEは `PR_p0000_q1_r2_K4`、保存Hybridは `PR_p0001_q1_r4_K4`。どちらも元n96を再利用し、再samplingしていない。',
      '主表の差は記述的な2 combined-SE規則では近接に該当しなかった。ただし多重選抜・逐次観察を補正した母平均順位保証ではない。RTE候補どうしの近接も残り、上記IDを唯一の母平均最適候補とは呼ばない。','',
      '## Phase A：13既存RTE構成の確認coverage','',
      'ユーザー指定3候補に加え、いずれかの精度で適格な探索Gが保存Hybrid以下、または2 combined-SE以内の未確認候補を追加し、8候補を実行前に固定した。',
      '追加5候補はq1/r2/K2、q1/r4/K2、q1/r8/K2、q1/r8/K4、q2/r4/K4。探索8本の不確かさを考慮すると、指定3候補だけでは競合候補の欠落を残すためである。',
      'q4/r4およびq8/r4のK2/K4計4候補は、全適格精度で探索GがHybridより2 combined-SEを超えて高く、今回確認しなかった。探索価格を母平均価格の厳密下界にして除外したのではない。',
      '全8候補について保存target bytes、B/bias/u、元のNと会計が一致することを確認した。Nを再計算せず、同じNへ新confirmation価格を掛けた。Phase Aの新しい状態作用・law再生成は0。','',
      '| 既存candidate ID | q/r/K (s) | 旧探索n | 旧確認n | 新確認n | 今回の状態 |',
      '|---|---|---:|---:|---:|---|']
    for r in mapping:text.append('| '+' | '.join([r['candidate_id'],f"{r['q']}/{r['r']}/{r['K']} ({r['s']})",r['old_exploration_n'],r['old_confirmation_n'] or '—',r['new_confirmation_n'] or '—',r['Phase_A_status']])+' |')
    text+=['','確認は96本から開始し、有限Gが保存Hybrid/旧RTE/他の新確認RTEと2 combined-SE以内、名目t95半幅>1%G、かつ候補自身のSEが比較側SEを超える場合だけ192→384へ延長した。',
      '6候補がn384、q2/r4/K2はn96、q2/r4/K4はn192で終了。計2592 paired trajectory。上限に達した近接状態を保証された優劣へ変えない。',
      '5精度は同じ価格標本を共有し、独立な5実験とは数えない。同一trajectoryのcosine/sine価格をpaired会計し、共分散を含める。探索標本と確認標本は混合しない。','',
      '## Phase B：s64・128の限定拡張','',
      '| candidate | s | B | N_cos+N_sin at ε=.05 | C_cos | G at ε=.05 | G SE | n |',
      '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
      if r['phase']=='B' and r['epsilon_sig']=='0.05':text.append('| '+' | '.join([r['candidate_id'],r['s'],num(r['B_exact']),str(int(r['N_cos'])+int(r['N_sin'])),num(r['C_cos_exact']),num(r['G_exact']),num(r['G_SE']),r['price_n']])+' |')
    text+=['','新3候補は各1回の128bit scalar/53bit vector tight L2作用と、各8本の探索価格で評価した。s64のB約1.0251→s128約1.0125でNはわずかに下がるが、C_cosは約219→420.75 RZとなる。',
      'Phase B内の最小s64/K2でも、新しい確認済みRTE最小のGより約5.6–5.7倍高い。全5精度で近接条件に該当せず、確認候補なしを `Phase_B_confirmation_nomination.json` に固定した。',
      'この無確認のPhase B結果は探索標本による傾向であり、確認済み費用優位と取り違えない。条件付きs128K4/s256は未取得、0や推定値で補完しない。',
      's2〜128の保存B/N/C/G一覧は `event_cost_trends.csv`、source/q/r/K/cohortを保持する。同じsのq/r分解が同一のcompiled費用を持つとは仮定しない。','',
      '## 科学条件・SOURCE・再利用','',
      'Linear H2、隣接1.00Å、STO-3G、中性singlet、保存Pauli-JW14非identity項＋constant。保存raw ray/full-H reference・共通exact normを再利用し、Hamiltonian/state/reference生成・再正規化はしない。',
      'S2/T=.8、純RTE L_D=0、central-rotation control、full Hadamard cosine/+Im sine wrapper、状態準備を除くcompiled RZ。Qiskit1.3.0/rz,sx,x,cx/opt1/seed17、topology/routing指定なし。',
      '有限Taylor paired RTE法則・CDF-gridのeffective mass・constant/global/relative phase・演算順序を維持。Kは最大偶数event次数で、多項式次数K+1とPF次数S2とは異なる。',
      'α_axis=.025、ε_axis=元のbinary64 ε_sig/√2、h=ε_axis−|bias|−u−ρ、u/h<=.01、N=ceil(2B_up²ln(2/α_axis)/h²)、G=N_cos C_cos+N_sin C_sinの同じ条件付き会計。実ρは未保証のため主評価は仮想ρ=0。',
      '新B targetは既存tight L2、Arb scalar128bit＋binary64 vector中心53bit。初期state/係数をexact保存値として扱い、共通normで補正しBを一回適用。保存Duhamel referenceはexact eigenstateを仮定しない。',
      '元H2の `price_job`・seed・law_descriptor・shot/paired統計、D2制御、H4/H6 event-extensionのfree-CPU dispatch、既存QPY-aware archive/restorerを再利用した。旧科学kernelは変更せず、私有law validatorだけr16/32へ狭く拡張した。',
      'SeedSequenceは元H2の[20261025,2,0,q,r,K,cohort_tag,index]（探索101/確認202）。全5424計画addressの衝突と旧cohort重複がないことを確認し、実seed/event数を保存検査した。PRNG streamの数学的独立性を証明したとは扱わない。',
      '新6接続/規則人工テスト＋新1 mean-oracleテスト＋既存12 controlテストの19件PASS。2qubit人工extended lawが既存Arb oracleを包含、raw norm/B一回、符号/phase/controlを確認した。最初の人工fixture JSON metadata不備は実科学前に修正し失敗logを保全。',
      '固定GitHub資料3件を実取得し指定commitのGit objectと内容照合した。旧13候補のlaw/target/cohort/seedと実行記録を既存archiveから読取専用で復元し、rootの旧SOURCE・STOPも指定parentのGit bytesと照合した。新科学grant・承認証明機構は作成しない。','',
      '## 実行・保全・残る限界','',
      f"新状態作用{v['new_target_state_actions']}、新trajectory{v['new_trajectory_count']}（Phase A確認2592＋Phase B探索24）、両軸compile{v['new_axis_compiles']}。新H4/H6/Hybrid科学実行、Hamiltonian/state/reference生成は0。",
      f"Native child {v['native_child_count']}件すべてexit0/reaped。全QPY{v['QPY_compressed_and_raw_SHA_checked']}個のcompressed/raw SHA、全compiled QPY{v['all_compiled_QPY_structure_decoded']}個の5wire/basis/RZ/CX、exact G・shot十分不等式・共通norm・B一回を照合。",
      '実際の空きCPUに応じて独立processを並列化し、対象processだけnice10、BLAS1threadで入れ子の過剰並列を避けた。共有設定・他ユーザーprocess/fileは変更しない。',
      '旧SOURCE/raw/grant/ledger/marker/STOPは無変更。旧H2結果を上書きせず、新表で比較の変化を示した。新raw/QPY/nativeと旧選択復元証拠は可逆archiveへ保存し、元bytesの全SHA復元検査に通過した。',
      'K2のorder4はsupport外、K4の希少event未観測は費用寄与0ではない。実ρ/全law精度、母平均費用tail、確認後選抜補正、未確認候補の母平均min、大域最適性、実機ノイズ/FT/QPE/RPE費用は未保証。',
      '母平均tail未保証やcompiler全体の形式的誤差証明を、この有限・条件付き比較の報告を妨げる新必須条件にはしない。技術照合はprimary Codexによるもので、独立科学レビュー・主要claim採択・正式certificate受理ではない。','',
      '## データと次の担当','',
      '- [旧13候補との対応表](../artifacts/h2_pauli_rte_validation/2026-10-26/old_candidate_confirmation_map.csv)',
      '- [全新候補の精度/cohort/bias/B/u/N/C/G/SE](../artifacts/h2_pauli_rte_validation/2026-10-26/candidate_precision_table.csv)／[s2〜128の保存比較](../artifacts/h2_pauli_rte_validation/2026-10-26/event_cost_trends.csv)',
      '- [5精度・旧/Phase A/Phase B/更新後比較](../artifacts/h2_pauli_rte_validation/2026-10-26/comparison_by_precision.csv)／[詳細JSON](../artifacts/h2_pauli_rte_validation/2026-10-26/comparison_details.json)',
      '- [図・SOURCE・復元の入口](../artifacts/h2_pauli_rte_validation/2026-10-26/README.md)／[技術検査](../artifacts/h2_pauli_rte_validation/2026-10-26/verification.json)','',
      '結果保存・commit/push・local/remote SHA一致確認後mandatory STOP。H2/H4/H6全体の最新結論の正式採択・GPT詳細レビューは開始しない。',
      '次の担当：ユーザー/GPTが、有限候補・条件付き順位変化を研究としてどう扱うか判断する。Codexは追加科学計算や研究方針採択へ自動移行しない。','']
    with (ROOT/'docs/h2_pauli_rte_validation_20261026.md').open('x') as f:f.write('\n'.join(text))
if __name__=='__main__':main()
