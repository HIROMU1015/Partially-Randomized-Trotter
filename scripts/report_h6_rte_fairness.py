#!/usr/bin/env python3
"""Render already completed H6 results; no new ranking or science kernels."""
import csv,json
from fractions import Fraction as Q
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/h6_rte_fairness/2026-10-25'
def read(p):return json.loads(p.read_text())
def num(x):return '—' if x is None or x=='' else format(float(Q(str(x))),'.6g')
def params(x):return '/'.join(str(x[k]) for k in ['L_D','q','r','K'])
def main():
    data=read(OUT/'comparison_by_precision.json');summary=read(OUT/'result_summary.json');v=read(OUT/'verification.json')
    assert read(OUT/'archive_restore_verification.json')['status']=='PASS'
    text=['# H6 Pauli-JW：r8を含むHybrid PR対全ランダム化RTEの追加検証','',
    '2026-10-25。branch `codex/h6-rte-fairness-20261025`。',
    '基点 `90c1c60820d52ca7ddcd4fb5315e9d21bb4a86c3`、科学SOURCE `54cd027431da92736e5f8b9a14c1a0a88760288a`。','',
    '## 得られた比較と適用範囲','',
    '**r8を含むRTE探索へ広げても、算定可能な有限候補・確認標本の条件付きGでは5精度すべてHybridが低かった。**',
    'ただし、旧r4限定比較の大きな倍率は縮まる。拡張RTE最良は全5精度でL0/q8/r8/K2となり、Hybrid/RTEは約0.0681–0.2189（RTE/Hybridで約4.57–14.69倍）。',
    'これは固定H6・仮想ρ=0・状態準備を除くRZ×shotの結果である。実compiledρ、母平均費用tail、大域最適性・無条件のend-to-end優位を保証しない。',
    'ε=.0005では数値guard未確定のRTE2行が残る。nullを0/∞として順位付けせず、以下のfamily minは算定可能な部分集合の範囲とする。','',
    '## 同一q/r/Kのmatched比較とfamily比較','',
    '各確認cohortは48 paired trajectory。探索標本とは混合しない。H/R比はG_Hybrid/G_RTE、Gの単位はcompiled RZ×shot。',
    'FamilyのHybrid側は保存探索・確認family、RTE側は拡張24構成の探索と確認済み集合を分ける。確認集合minは事後選抜であり、全24構成の母平均minではない。','',
    '| ε | Hybrid L/q/r/K | matched RTE L/q/r/K | G Hybrid | G matched RTE | G family RTE | matched H/R | family H/R |',
    '|---|---|---|---|---|---|---|---|']
    for x in data:
        a=x['comparison'];h=x['matched_Hybrid'];m=x['matched_RTE']
        text.append('| '+' | '.join([x['epsilon_sig'],params(h),params(m),num(h['G_exact']),num(m['G_exact']),num(a['expanded_RTE_G_exact']),num(a['matched_Hybrid_over_RTE_exact']),num(a['family_Hybrid_over_RTE_exact'])])+' |')
    text+=['','ε=.05のmatched対照はL684対L0のq1/r8/K4、他4精度はL78対L0のq8/r8/K2である。後者はfamily最良RTEとも一致した。',
    '前者のmatched RTEはq1を固定しているため、q8まで探索したfamily RTEとは大きく異なる。matched比とfamily比を混同しない。',
    '全10比較は名目2 combined-SEで近接と判定されなかった。ただし、名目SE・選抜・逐次観察を厳密な母平均winnerの信頼保証にしない。','',
    '| ε | G SE Hybrid / family RTE | 旧r4 family H/R | 新r8 family H/R | 新RTE G / 旧r4 RTE G | 保存最良決定論的PF/B0 G |',
    '|---|---|---|---|---|---|']
    for x in data:
        a=x['comparison'];text.append('| '+' | '.join([x['epsilon_sig'],num(a['Hybrid_G_SE'])+' / '+num(a['expanded_RTE_G_SE']),num(a['old_r4_Hybrid_over_RTE_exact']),num(a['family_Hybrid_over_RTE_exact']),num(a['expanded_RTE_over_old_r4_RTE_exact']),num(a['best_deterministic_G_exact'])])+' |')
    text+=['','旧RTE比は約.000698–.002111、新RTE比は約.0681–.2189。拡張RTEのGは旧r4最良の約1%へ下がり、Hybridの相対優位は約100分の1へ縮まった。',
    '旧r4の最厳精度はK4、新r8最良はK2でもあるため、このfamily差をrだけの純粋な因果効果とは呼ばない。旧確認標本は再samplingしていない。',
    '保存最良決定論的PF/B0のIDは順にB0_p0724_q1、B0_p0830_q2、B0_p0830_q4、B0_p0864_q8、B0_p0864_q8。今回再探索していない。','',
    '## Normalization・bias・shot・単回価格の内訳','',
    'Bは保存binary64値をexactに固定する。Biasはfull-H referenceに対するsigned cosine/sine、uは数値包含の半幅。Cは軸別compiled RZ平均、SEは価格の標準誤差（母平均tail上界ではない）。','',
    '| ε | 対象 L/q/r/K | B | bias cos / sin | u cos / sin | N cos / sin | C cos / sin | C SE cos / sin |',
    '|---|---|---|---|---|---|---|---|']
    for x in data:
        choices=[x['matched_Hybrid'],x['matched_RTE'],x['confirmation_RTE']['selected']];seen=set()
        for a in choices:
            if a['candidate_id'] in seen:continue
            seen.add(a['candidate_id'])
            text.append('| '+' | '.join([x['epsilon_sig'],params(a),num(a['B_exact']),num(a['bias_cos_exact'])+' / '+num(a['bias_sin_exact']),num(a['u_cos_exact'])+' / '+num(a['u_sin_exact']),a['N_cos']+' / '+a['N_sin'],num(a['C_cos_exact'])+' / '+num(a['C_sin_exact']),num(a['C_cos_SE'])+' / '+num(a['C_sin_SE'])])+' |')
    text+=['','Hybridの単回CはRTEより高い一方、tailの縮小に伴ってBとNが小さく、今回のGではその効果が単回費用差を上回る。',
    'ε=.05のmatched RTE Bは約1.0918e5で、Hybridの約1.027とは大きく違う。他4精度のmatchedではRTE B約20.6408、Hybrid約2.63064である。',
    '保持prefix変更は分割、tailのcomponent/確率、normalization、有限Taylor bias、回路bodyを同時に変える。この観測から決定論的部分の追加だけの純粋な効果とは主張しない。','',
    '## 候補選択・実行と再利用','',
    '固定linear H6、隣接1.00 Å、STO-3G、中性singlet、保存Pauli-JW918非identity項＋constant、保存raw ray/full-H reference・共通exact norm。Hamiltonian/state/referenceを生成し直していない。',
    'S2/T=.8、Pauli central-rotation control、Qiskit1.3.0、basis rz/sx/x/cx、opt1、seed17。Topology/layout/routingの追加指定なし。full Hadamard wrapperのcosine/+Im sineを同一trajectoryから価格化し、状態準備費用は除外。',
    'まずmatched2構成を実行し、r8で旧r4最良より費用が大きく下がることを確認して残りr8の6構成を追加した。未探索q/rの価格下界で除外する根拠がなかったため、許可された残り9構成も追加した。',
    'RTE領域L0、q={1,2,4,8}、r={2,4,8}、K={2,4}の24構成を網羅し、旧7構成のtarget/価格は再利用、新17構成だけtargetを計算した。各新batchはSOURCE・入力・候補・call・停止条件を実行前に固定した。',
    '新探索は16構成×8=128 paired trajectory。新q1/r2/K2は5精度すべてbias不適格で、価格compileは省略した（価格0とはしない）。',
    '探索内の算定可能minは5精度ともq8/r8/K2で、確認前にnomineeを固定した。matched q1/r8/K4と重複を除き2構成を各48本で独立確認した。固定の48→96→192→384規則を満たす精度行がなく、延長0。',
    'Hybridの確認48本・旧RTE確認48本を再利用した。独立確認の候補選抜と、確認後のfamily minを別fieldに保存する。5精度は同じ価格標本を共有し、独立な5実験とは数えない。',
    'SeedSequenceは元H6の[20261025,6,L_D,q,r,K,cohort_tag,index]、探索101/確認202。新候補IDの旧cohortとの重複を避け、保存seedを照合した。PRNG streamの数学的独立性の証明ではない。',
    'pauli_signal_adapter/rte/D2制御、元price_job、law_descriptor、tight L2法、paired/cost_summary/min helperを再利用した。旧出力固定CLIを再launchせず、新OUTのchildで呼び出した。',
    'Targetは既存の128bit Arb scalar＋binary64 vector中心/L2包含。Constant/native scalar、CDF-gridのeffective law、order0/2/4、演算順序、raw state/exact normを保持し、Bを一回だけ適用する。保存referenceはArbitrary-raw-state Duhamel包含で、exact eigenstateを仮定しない。',
    '旧targetからの誤差条件や1%guardを緩めず、新候補固有のbiasとuで同じ会計式を適用した。科学SOURCE/旧結果/旧STOPは保全する。','',
    '## SAFE_INTEGER_LIMITと未確定値','',
    '`h=epsilon_axis−|bias|−u`、`u/h<=.01`、`N=ceil(A_upper/h^2)`という元のshot十分条件は変更しない。',
    '元h4_conditional_cng.shotはFractionで上界を計算した後、N>=2^52でSAFE_INTEGER_LIMITを返していた。新実装はこの分岐だけPython任意精度整数のexact ceilへ接続し、`N*h^2>=A_upper`を分数で検証する。Small Nやguard不成立の判定は元処理をそのまま返す。',
    '旧SAFE_INTEGER_LIMITによるnullを8精度行で新たに有限会計できた。旧nullと旧SOURCEは上書きしない。旧q1/r4/K2の高精度4行はbias不適格で、整数上限とは区別する。',
    'ε=.0005のq1/r4/K4と新q2/r2/K4はWIDTH_AMBIGUITY/FIXED_GUARD_FAILEDのまま。Biasの不適格と断定せず、N/Gもnullのまま保持する。',
    'この2行についてはh<=epsilon_axisより、同じ固定観測価格を使う将来の有限会計Gの粗い下界を示せる。その下界はいずれも現在の有限最良Gを大きく超える。これは探索標本平均を母平均価格の下界にしたものではなく、母平均tail保証や欠測Gの補完でもない。',
    '[保留2行と固定価格条件付き下界](../artifacts/h6_rte_fairness/2026-10-25/unresolved_candidates.csv)に前提を記載した。完全なN/G一覧が必要な場合にだけ追加precision診断を検討し、今回は未確定を消さない。',
    'CSVのNはexactな十進整数文字列である。巨大整数をJavaScriptの通常Numberへ変換して会計しない。JSONの整数tokenもPython等の任意精度対応readerで読む。','',
    '## 保存証拠・限界・終了','',
    f"新target状態作用{summary['new_target_state_actions']}、新trajectory{summary['new_trajectories']}（探索128＋確認96）、両軸compile{summary['new_axis_compiles']}。新Hamiltonian/state/reference生成0。",
    f"人工6件PASS、native child{v['native_child_count']}件すべてexit0/reap、元engine4件のexit0も直接回収。全新QPYのSHA/RZ/CX/wire、seed、N十分不等式、exact G、raw norm/B一回を照合した。",
    '実際の空き資源に応じたprocess並列・自分の低優先度workerを使用し、共有設定・他者のprocess/fileは変更しなかった。',
    '確認q1/r8/K4ではorder0/2/4を179/157/48 event、q8/r8/K2では2991/81/0 event観測した。これはevent数である。K2のorder4はsupport外、K4の未観測eventとは意味が異なる。いずれも母平均費用tailの保証ではない。',
    '新raw/QPY/入力コピー/native記録は可逆tar.xzへ保存し、既存restorerによる元bytesのSHA復元を確認した。元rawは削除・変更していない。',
    '結果はempirical/conditional比較で、実ρ・PR全law残差・母平均費用tail・全連続パラメータの最適性・正式certificateは未保証。現時点の限定的問いに答えるための追加科学計算は不要。',
    '今回の技術照合はprimary Codexによるもので、独立科学レビューや主要claimの正式採択ではない。','',
    '- [精度別比較CSV](../artifacts/h6_rte_fairness/2026-10-25/comparison_by_precision.csv)／[matched CSV](../artifacts/h6_rte_fairness/2026-10-25/matched_comparison.csv)',
    '- [RTE24候補一覧](../artifacts/h6_rte_fairness/2026-10-25/registered_RTE_candidates.csv)／[全精度・cohortのRTE一覧](../artifacts/h6_rte_fairness/2026-10-25/expanded_RTE_inventory.csv)',
    '- [選択候補の内訳](../artifacts/h6_rte_fairness/2026-10-25/selected_cost_breakdown.csv)／[JSONと探索・確認・nominee](../artifacts/h6_rte_fairness/2026-10-25/comparison_by_precision.json)',
    '- [旧整数上限調査](../artifacts/h6_rte_fairness/2026-10-25/integer_limit_investigation.json)／[照合](../artifacts/h6_rte_fairness/2026-10-25/verification.json)／[データ入口](../artifacts/h6_rte_fairness/2026-10-25/README.md)','',
    '科学計算はSTUDY_STOP.jsonで終了した。commit/push・local/remote SHA一致の確認後もmandatory STOPを維持する。',
    '次の担当はユーザー/GPT：この有限・条件付き比較の研究上の扱いを判断する。必要と判断した場合だけ、tail/実ρ保証や残存precision診断の具体的範囲を別承認する。Codexは追加科学計算、詳細科学レビュー、論文執筆、正式claim採択へ自動移行しない。','']
    p=ROOT/'docs/h6_rte_fairness_20261025.md'
    with p.open('x') as f:f.write('\n'.join(text))
    print(p)
if __name__=='__main__':main()
