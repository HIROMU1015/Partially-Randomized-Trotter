# 次の最小科学検証案

2026-11-03 JST。**提案・未承認・未実行。第一候補はB。**
今回の技術準備の許可では以下のstate solve / build / signal評価を開始しない。
V1/V2/V3/R1と既存M06-FのSTOP・契約は変更しない。

## 候補比較

共通条件：linear H4/隣接1.00Å/STO-3G/legacy DF12、8qubit、4電子sector、
物理H基底状態、L_D3/12、δ=.02、保存18roundのq/r/K/N/α/β。
新energy目標・候補・配分・PF公式を追加しない。以下のprobe subsetは新しい最適化候補ではない。

| 候補 | 未保証条件・既存証拠 | 初めて判別する情報と再利用 | 必要call・最小範囲 | 判定・V2への影響・負荷 |
|---|---|---|---|---|
| A fresh-IID→marginal | SOURCEの期待値恒等式とδ=.1/r4/q≤8の旧fresh-IID evidenceはある。最終scheduleの実sampler marginalは未確認 | 最終request分布のstate-actionと保存attenuated μの整合性。sample_rte_events、_seed_for_shot、_trajectory_signalを最小拡張しrをround別に渡す。旧_explicit_fresh_iid_validation全体は固定r・再対角化/buildを含むため直呼びしない | B後の別案。まず保存m0/q1/r1/K2とm3/q8/r1/K2の固定N、両軸各1 batch、fresh trajectory/shot。新state-action、sampling、仮想Bernoulli。compile0 | 元Nを変えず、事前固定の二項区間とtrajectory平均不確かさを別に報告。範囲内でも全18roundのIID・稀なorder2を保証しない。μ不一致ならV2の実装への適用条件が崩れ得る。小qでも有限Nでは小biasの識別力が弱い |
| **B 実回路の条件付き平均（推奨）** | small synthetic event/control/wrapper testsと短qのbasis-policy同値性がある。最終H4 physical inputで同一requestの独立意味論照合は未実施 | 与えたtrajectoryの理論complex値と実controlled/Hadamard平均の相違。非可換積順序、order2符号、scalar位相、sector/endian、selected policyを直接切り分ける。既存step/repeated/wrapper、sector action再利用 | 下記固定6request、12axis wrapperのideal mean。保存Hからstate solve1、state-action、実回路build。RNG/shot sampling0、compile0、価格0 | 1e-10の数値整合性と明示identityを事前判定。合格はそのrequestだけ。失敗は保存μと実回路marginalの接続を再検討する理由。有限個・9qubit理想作用で負荷を限定でき、統計誤差なしで符号の問題を判別 |
| C 入力状態・半径 | V1/V2の保存physical平均、Full7点不一致、高overlapは既知。Full δ=.02 control/診断直接fingerprintは欠測 | 同δのPF有効固有状態とphysical入力の差、数値誤差寄与。finite_rte_signal_validation、deterministic_physical_signal_diagnosticの必要部分を再利用 | 必要と判断された場合のみ、同一snapshotのFull δ=.02 controlとphysical、該当7roundへ限定する案。PF演算子生成・eig/state-action、新半径評価。sampling/compile0 | 数値state identityと厳密下界を分ける。通常floatの半径再評価だけでrigorous下界は得られない。V2は既にphysical μを使うので、単位半径モデルを再確認するだけでは測定接続の主要な空白を閉じにくい |
| D 18round end-to-end | 既存decoder、V2保存平均・固定N解析確率、今回人工18round接続はある | 生成した量子測定模擬結果→branch→energyの結合動作。新adapterと元decoderを再利用 | A/B後に別承認。保存全N、全18round、両axisのfresh trajectoryと測定生成。回路実行を要求するならbuild/ideal simulationも別に固定。長q compileは既定にしない | E₀は評価のみ。branch/energy失敗件数と試行数に応じた不確かさを記録。1成功batchは成功率保証ではない。長qでq*r event/shotが支配し、現段階の費用対情報が低い |

A/C/Dの具体的な追加seed・反復数・新sourceは今回の実行対象ではない。
Bを通過しても自動で開始しない。Aはサンプリング分布の確認、Bは条件付き回路意味論、
Cはstate/radius、Dは全体分布という異なる問いであり、一つのPASSで他を代用しない。

## 第一候補Bの固定入力とSOURCE

科学的な問い：**同じ物理入力ψと同じ有限RTE trajectory requestについて、
理論〈ψ|V_T|ψ〉のRe/Imと、実際に構築したordinary-controlled/Hadamard回路の平均は一致するか。**
保存平均を再びdecoderへ渡してV2の成功を繰り返す検証にはしない。

旧SOURCEの参照版は全てcarrier
`4591b8d75a2a6a55c74a51225f0c61648023c8ee`。新adapterは本branchの公開commitに固定する。
科学runnerは今回付属せず、次の実行許可後に下表のbindingだけを加えた別SOURCEへ固定する。
runner完成後、固定した同じB範囲の許可を毎工程で取り直す必要はない。

| 入力・処理 | 固定内容 |
|---|---|
| H snapshot | `artifacts/rte_connected_cluster_cost_validation/h4_sto3g_d100_rank12_ld3_dt0p1_ref4_k2_connected_pilot30_max1500_hold1500_rare375_v1.hamiltonian.npz` |
| NPZ identity | SHA256 `13e4b10d2347ed900fe4aa4b2238128ebe735a48988e321b9e4fabac4d092a3b`。この名前のδ=.1/r4は保存時の由来であり、今回probeのδ/rではない |
| H identity | `56e4df83655aa2f2f8132126f2635996516dc2cfb1bf4cb8d95490184ad631e5`。NPZ byte hashとは別 |
| physical入力 | 保存Hの同じ4電子sector最低固有状態。snapshot writerにはstate vector欄がないため、保存vectorが別に利用可能ならそれを優先。なければ同じdense_df_operator_in_sector / eigh経路で**1回だけ再構成するstate solveをBの承認範囲に含める**。SCF/新DF生成なし |
| state照合 | ψ norm、Hψ−Eψ residual、保存E₀との差を既存matrix allowance1e-10で確認。保存fingerprint a07d6c7fdbbb23a035f9fc54b9956ee79e557a2185e29b1870748b771958be90との一致／不一致も記録。fingerprint不一致だけでglobal phase等を決めつけず、保存byte同一性は未確認と表示。数値state条件不一致ならSTOP |
| Full入力 | 新probeでは同じψをL_D12にも明示的に渡し新recordにfingerprintを保存。旧Full18点のmissing fingerprintを補完しない |
| finite-RTE | prepare_df_partial_s2 / split_df_hamiltonian_by_ld、identity_policy=extract_identity_phase。make_rte_configでK=2を明示、保存_final設定を読む。cutoff自動選択なし |
| trajectory | df_partial_s2_repeatedの明示requestとdf_rte_circuitのevent request。下記conditional probeはsampler母分布の標本ではない |
| 回路 | QiskitDFPartialS2RepeatedCircuitBuilder、boundary_optimized、support_run_le_1。register_support_restricted_basesとapply_run_threshold_policyを既存training fingerprintで利用。wrapperは既存benchmark builder、maximum=q8、include_measurement=True。期待値評価用に終端測定を除くcopyを使用 |
| 理論側 | _trajectory_signal / _apply_event_to_stateの順序を再利用。各signed componentをHからsectorで構成する。deterministic halvesは同じ仕様だが、比較対象のcontrolled回路からsignalを逆算してoracleにしない |
| statevector側 | system stateを既存のQiskit↔OpenFermion permutationで8qubitへ埋め、ancilla=8を|0〉に置く。実wrapperのideal Z expectationを取る。新state-preparation回路は作らない |
| 環境 | Qiskit1.3.0、complex128/float64。compileは0なのでopt2/rz,sx,x,cx/seed17の価格設定は保持するmetadataであり、新価格ではない |

## Bの最小request・call範囲

保存最終scheduleに存在するm0/q1とm3/q8を使う。
Partialはいずれもr1/K2、Fullはいずれもr0/K0。N/α/βはmetadataのまま固定し、
ideal mean照合で新Nを作らない。

1. Partialのsymbolic retained tail componentsをcomponent_id順に並べ、c0を最初のcomponent、
   c1をc0とbasis_idが異なる最初のcomponentとする。係数符号を含むcomponentをそのまま使う。
   該当c1がなければcoverage不足でSTOPし、別selectorへ自動変更しない。
2. 各qに対して、全occurrenceがorder0（rotation c0）のprofileを一つ作る。
3. 各qに対して、最初のoccurrenceだけorder2（rotation c0、products c1,c0）、
   他occurrenceは同じorder0のprofileを一つ作る。_make_eventの既存paired分布・phase・角度を使う。
   basisが異なるだけで非可換と認定せず、conditional probeの実際の作用順を照合する。
4. Fullは各qに一つのdeterministic request。合計Partial4＋Full2＝6 request。
   各requestでcosine/sineの2wrapper、計12wrapperをbuildする。

各requestの理論complex signalは1回、各wrapperのideal Z expectationは1回評価する。
constant/extracted identity、event order2の−1、|0〉branchのI、|1〉branchのV、
両branchへのaction、sector leakageを同じrequestから検査する。
controlを二重適用しない。controlled作用はglobal-phase alignmentをして比較しない。
同じrequestのcos/sin照合は意味論のpaired診断であり、独立quantum shotsではない。

許可を求める科学callは、H snapshot読込、H-sector構築とground state solve1、
L_D3/12のpreparation、必要なcomponent/half作用の構築、6trajectory理論signal、
6controlled evolution build、12wrapper buildとideal expectation、対応residual評価に限定する。
seed/RNG draws、確率sampling、SCF/新DF、長q build、transpile/compile、価格取得・再ランキングは0。
旧V1/V2/V3/R1 runnerを起動しない。強制order profileをfresh-IID標本へ数えない。

## 判定・情報の限界

- **局所整合**：同じψ/requestのRe/Im残差、control action、leakageが1e-10以内。
  state norm/residual/保存energy条件も同allowance以内。これは既存matrix numerical toleranceを
  用いる局所数値検証で、厳密certificate・全18roundへの一様保証ではない。
- **不一致**：上の数値条件を外す、basis/permutation/scalar/controlが対応しない。
  失敗profileを保存してSTOP。自動再sampling、閾値緩和、別profileへの置換、再実行を行わない。
- **未判定**：固定snapshot/SOURCEが取得できない、state条件を確立できない、c1 coverageがない、
  必須環境が未用意、実際の資源不足。欠測を0/PASSで埋めない。
  保存state byteの直接一致が確認できない場合は、他条件を満たしても
  `SOURCE_CONSISTENT_PHYSICAL_RECONSTRUCTION; SAVED_STATE_BYTES_UNVERIFIED`と表示する。

合格なら、実回路の条件付き平均という空白を当該6requestで狭める。保存marginalへの分布平均、
r≥2、q>8、全18roundの成功率、PF control identity、rigorous半径下界、実準備・backend/noiseは残る。
不一致でも既存V2の数学的な「保存μを真の平均と仮定した」判定そのものを改変せず、
その実回路への適用を保留する。

## 資源・次候補・承認とSTOP

実行前に実際のCPU/空きRAM/GPUと既存timingを確認する。
9qubit ideal actionは大きなH4長round samplingとは負荷が異なる。immutable sector/componentを共有し、
12wrapperの独立actionをbatch/並列化できる。process数とBLAS threadを調整しoversubscriptionを避ける。
scopeを変えない人工reference比較を先に行う。人工CPU/RAM/worker/wall-time上限を置かず、
共有設定・他jobへ干渉しない。旧M06-F workerのCPU-only/1thread固定契約を流用・変更しない。

Bの局所整合後にユーザーが次を選ぶならAを提案する。B不一致なら原因の技術調査と
必要な科学再実行の範囲を分けて戻す。state/radiusが支配問題と分かればCへ変更提案する。
DはB/Aの条件が整理された後にのみ再検討する。

**次の担当はユーザー：Bの上記state solve/build/signal範囲の明示実行許可。
Codex：許可後に限定backend/runnerを実装してSOURCEを固定し、同じ許可範囲を1回実行する。
独立詳細科学レビューが必要と判断された場合はその別許可も要る。承認証明書・grant発行は新設しない。**

**mandatory STOP：Bの局所結果・不一致・未判定のいずれでも記録後に停止。
A/C/D、科学的retry、18round sampling、長q価格、費用再解析、独立詳細レビューを自動開始しない。**
