# 候補A：一回の限定科学実行計画（準備済み・未承認・未実行）

2026-11-03 JST。**今回の依頼は準備の許可であり、以下の科学実行を許可していない。**
固定SOURCEは本計画・adapter・runner・testsを含む専用branchの公開commit。
最終チャットのfull SHA／固定URLを用い、その後の別版へ無断で移さない。
承認証明書、grant、署名、承認原文保存は要求しない。チャットでこの具体的範囲を許可すれば足りる。

## 科学的な問いと固定入力

fresh-IID trajectoryと条件付き±1測定の実装について、保存physical attenuated event-meanと
理想state-actionモデルの周辺平均が整合するかを調べる。Bの条件付き局所照合を再実行しない。

linear H4/隣接1.00Å/STO-3G/legacy DF rank12、8 system qubit、N_e=4/dimension70。
元Hのphysical ground state、Partial L_D=3、δ=.02、r1/K2。
m0/q1とm3/q8だけを対象とし、Full・他round・r≥2・q>8・新PF/energy目標を追加しない。

| m/q/axis | 保存N | 保存μ（直接転記） | 保存α | 保存coordinate ε |
|---|---:|---:|---:|---:|
| 0/1/cosine | 510 | .9989255707627144 | 2.8897922699779505e-08 | .266146579872968 |
| 0/1/sine | 510 | .04330819375415813 | 2.8897922699779505e-08 | .266146579872968 |
| 3/8/cosine | 454 | .939502685390124 | 2.2180178685403194e-07 | .26589319587097715 |
| 3/8/sine | 454 | .3393522369363202 | 2.2180178685403194e-07 | .26589319587097715 |

全行の保存β_PF=.014050012886294481、β_RTE=9.899584668737594e-08、
β_stat=.38594988811785885を保持する。診断閾値をこれらの配分へ代入しない。
targetはV2 commit78c49d0b5f1d71e63b5959e78813237923bbc648の
`docs/research/pr_rpe_v2_branch_success_20261103/V2_ROUND_RESULTS.csv`から
ld3/m0,m3、`saved_signal_kind=attenuated_event_mean_signal`、`saved_complex_json`のRe/Imを取る。
保存α/β/εは同じ行から転記し、新しいbinomial確率やRPE成功率を計算しない。
元signalの所在はcarrier4591b8d75a2a6a55c74a51225f0c61648023c8eeの
`artifacts/research_direction_ablation/2026-09-21/wp04_ld3_full_schedule_signal_grid_v1.json`、
該当q/r/Kの`state_results[state_label=physical_df_ground_state]/attenuated_event_mean_signal`。

| 入力 | 固定identity |
|---|---|
| H snapshot | `artifacts/rte_connected_cluster_cost_validation/h4_sto3g_d100_rank12_ld3_dt0p1_ref4_k2_connected_pilot30_max1500_hold1500_rare375_v1.hamiltonian.npz`、carrier4591b8d75a2a6a55c74a51225f0c61648023c8ee |
| snapshot SHA256 / H hash | `13e4b10d2347ed900fe4aa4b2238128ebe735a48988e321b9e4fabac4d092a3b` / `56e4df83655aa2f2f8132126f2635996516dc2cfb1bf4cb8d95490184ad631e5`。別の対象のhash |
| B state | e22d7718c47358ff650dbc04e8e72f6ab1c8bf41の`artifacts/rpe_fixed_request_measurement_b/2026-11-03/run01/physical_state.npz` |
| state NPZ SHA256 | `793300c86417f52569289ce86980c5f9a87292e130199355a0f4f7056225c9c4` |
| physical fingerprint | `a07d6c7fdbbb23a035f9fc54b9956ee79e557a2185e29b1870748b771958be90` |
| B RESULT | 同runのRESULT.json、SHA256 `b9b4615ed862a7a85dc6790692f8632d0997d53a1ad36ad59788572a64a51283` |
| DF preparation | 保存hash `89145d8324bf139eaec67265f3134a5c1a3cebb7ed97792d54ed3221d31f8919`、218 retained component、deterministic fragment0,1,2 / tail3..11、extract_identity_phase |

Hとstateのbyte、H/sector/state fingerprint、samplerと必要な旧numerical SOURCEの版を確認する。
stateは保存物を読み、再対角化・ground-state solve・state準備回路を作らない。
入力が違う場合に新SCF/DF/stateを作って合わせることはせずSTOPする。

## 確率法則・RNG・shot順序

既存有限RTE分布はτ=λ_R δ/r、偶数order0/2と正規化B_Kを使う。
componentの確率・係数符号、products/rotation順序、(-1)^(order/2)、atan(τ/(order+1))を保持する。
独立eventについてE[V_e]=T_K/B_Kなので、同じhalf作用の独立積は
E[V_ω]=B_K^(-rq) A_corr。したがって保存z_att=〈ψ|E[V_ω]|ψ〉が理論marginalに対応する。
この等式のSOURCE/定義対応と、実際に有限標本を生成した結果は分けて報告する。

新しい**公開再現性recipe**を以下へ固定する。旧非公開seedを使用・公開しない。

1. namespace文字列は`rpe_fresh_iid_measurement_a_20261103_v1_public`。
2. roleは`trajectory`または`measurement`。
   masterはSHA256(UTF8(namespace+":"+role))の先頭8byteをbig-endian整数へ変換し、63bit mask。
3. 旧_seed_for_shotを変更せず使い、SHA256(UTF8(f"{master}:{q}:{axis}:{shot}"))の
   先頭8byte・big-endian・63bit maskを各seedとする。axisは`cosine`/`sine`、shotは0始まり。
4. trajectory seedで既存sample_rte_eventsを一度だけ呼び、q*r eventを生成する。
   sampler内部のNumPy Generator(PCG64)とorder→componentの抽選順は元実装のまま。
5. 別roleのseedで新しいGenerator(PCG64)を作り、1回のuniform drawから±1を得る。
   cosine/sineでtrajectoryやmeasurement RNG objectを共有しない。
6. 論理順はq1/cosine→q1/sine→q8/cosine→q8/sine、各shot昇順。
   並列実行順は任意だがkey/seedは不変。集計時はshot index順に並べる。

計3,856個のrole付きkey/seedの重複を検査する。衝突ならSTOPし、seedを選び直さない。
**異なるseedだけで数学的独立性を証明しない。** IIDはPRNGモデルの仮定として残る。
同じevent列の自然な再出現は拒否しない。価格取得のpaired cos/sinとは区別する。

各z_ωは条件付きunitary signal。Re/Imをμとし、P(+1|ω)=(1+μ)/2。
Bで確認した負時間符号・scalar/identity/event位相を保ち、B_K補正や追加phaseを重ねない。
raw complexが非有限またはunit diskから1e-10超逸脱すればSTOP。
対応座標が[-1,1]から1e-12超逸脱してもSTOP。
1e-12以内だけ境界へ戻し、raw/used/p/補正flagを保存する。

## 数値準備と科学call範囲

Qiskit1.3.0、complex128/float64。必要なDF分割/basis metadataと218 signed sector operatorを
一度だけ構成し、最大13個の基底をcacheする。deterministic最大4blockから既存primitiveで
forward/reverseの2halfを組む。旧prepare内部のGaussian gate定義構築もこの入力準備に含む。
これはcontrolled/measurement回路の新buildではなく、既存gate演算子の数値合成である。
今回準備ではこのH4数値処理を実行していない。新経路は人工小回路だけで確認した。

| 新科学処理（別承認対象） | 一回の範囲 |
|---|---:|
| 保存H/保存B stateの数値load・identity確認 | 各1 |
| L_D3のDF/basis preparation・finite-RTE config | 各1 |
| retained signed sector operator構成 | 218、同じ入力から一度だけ |
| reference half matrix構成 | forward/reverse各1、同一δ/rを全shotで共有 |
| fresh trajectory生成 | 1,928回 |
| event位置 | 8,284個（q1の1,020＋q8の7,264） |
| 既存_trajectory_signalの条件付きstate-action | 1,928回 |
| 条件付き仮想±1測定 | 1,928回 |
| 統計集計 | 4axis、事前固定12診断のみ |
| ground-state solve / 新SCF・DF / controlled・Hadamard回路build / compile / 実backend shot / 費用再解析 | 全て0 |

Full、他round、全18round decoder・energy推定、PF係数、半径fit、新μ reference、
旧V1/V2/V3/R1/B runnerの再実行は追加しない。

## 統計判定（結果を見る前に固定）

各axis jに保存target μ_j、条件付きmean列u_i、測定値Y_i∈{±1}がある。
理論target、trajectory標本平均 \bar u、測定標本平均 \bar Yを別欄にする。
\bar Y−\bar uは条件付きBernoulliの揺らぎとして別に記録する。
有限trajectory平均を分布平均、有限shot平均を厳密期待値と呼ばない。

**診断family η_diag=.05、J=12**を固定する。
4axisそれぞれについて、(1)|\bar u−μ_j|、(2)|\bar Y−μ_j|、(3)|\bar Y−\bar u|の3診断。
各axisの幅は

\[
h_j=\sqrt{\frac{2}{N_j}\log\frac{2J}{\eta_{diag}}}.
\]

(1)/(2)はh_j+1e-12、(3)はh_jを超えた場合に不一致flagを立てる。
これは範囲長2のHoeffdingと12項union boundによる保守的診断である。
(3)はtrajectoryを条件付けた測定の独立性を仮定する。
同じaxis内の3診断同士の独立性を仮定していない。
保存μを正しい真の平均とする条件、PRNGを独立標本モデルとして扱う条件も残る。
保存μ自体の浮動小数点誤差を厳密区間で伝播した保証ではない。

- 一つでもflag：`DISCREPANCY_DETECTED`／`INCONSISTENT`。確率的な反証診断であり、
  η_diagのfalse alarm可能性を残す。閾値・seed/Nを変えて再試行しない。
- 全flagなし：`NO_DISCREPANCY_DETECTED`だが科学的な同等性は`UNDETERMINED`。
  同等性marginを新しいRPE精度目標として追加しないため、同等性PASSは出さない。
- 欠測・非有限・不正確率・中断：`UNDETERMINED`としてSTOP。部分標本で判定しない。

固定Nではh_jより細かい差の識別力は保証しない。真の差が2h_j+2×1e-12を超えることは、
この保守的モデルで高確率検出へ至る十分な分離の目安であり、最小検出差の最適値ではない。
未検出を一般的同等性、稀な失敗率、RPE成功確率の保証に読み替えない。
K2 eventが未観測でも、追加sampling/強制event/seed変更を行わない。

## 資源・出力・STOP・承認

実行前に空きCPU/RAM、quota、共有負荷を確認する。
独立shotを固定index chunkへ分け、immutable contextをworkerへ一度だけ渡す。
worker数は使用可能CPUと実memoryに従い、small matrix用のjob内BLAS1threadで
oversubscriptionを避ける。人工CPU/RAM/wall-time上限は置かない。
人工fixtureではserial/processとchunk順序の一致を確認済み。実H4の速度比較は未実施。

承認後の出力候補はfresh `artifacts/rpe_fresh_iid_measurement_a/2026-11-03/run01/`。
既存directoryには再実行せずresumeしない。RESULT.jsonにSOURCE commit、固定入力・配分、
公開seed recipe、context/versions、集計・statusを残す。
shots.jsonlにkey、条件付きcomplex、raw/used μ、p、±1、event order/component/phase/probabilityを残す。
秘密seedの数字列や内部host/pathを保存・公開する必要はない。
rawの並列完了順による並びの違いは許すが、key別の数値と集計順は固定する。
公開は実行後の機密・整合確認対象であり、runnerは外部送信しない。

metadataだけの確認は`python docs/research/rpe_fresh_iid_measurement_a_preparation_20261103/run_fresh_iid_a.py --describe`。
**別承認後のみ**、実行専用worktreeで公開commitを固定して次を1回使う。
`--run`は処理選択にすぎず、承認を代行・証明しない。

```bash
python docs/research/rpe_fresh_iid_measurement_a_preparation_20261103/run_fresh_iid_a.py --run --output-dir artifacts/rpe_fresh_iid_measurement_a/2026-11-03/run01
```

成功した実行・不一致・未判定の全てでmandatory STOP。
中断時はqueued chunkをcancelし、既に走っているchunkは安全に回収する。
得られた部分rawは残すが部分推論・自動再開を行わない。
追加標本が必要なら新しい検証案と別承認へ戻す。

**次はユーザーが本固定範囲の新sampling/state-action/仮想測定の1回実行を明示許可する。
Codexは許可後に固定commitを使って実行・記録し、停止する。**
今回の準備、方針への同意、過去Bの許可はAの科学実行承認ではない。
独立詳細科学レビューにも別許可が必要。Aの実行後もC/D・全18round・費用再解析を自動開始しない。
