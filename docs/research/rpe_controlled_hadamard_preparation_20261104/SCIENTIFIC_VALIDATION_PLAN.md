# 次の限定科学検証案：局所回路の同値性

2026-11-04 JST。**未実行・未承認の提案**。
これは全18roundを直接buildする計画ではない。新しい実H4 science runnerも今回は作成・起動していない。
本計画とproduction関数の版を固定し、将来の最小runnerを人工検証して実行SOURCE commitへ固定した後、
そのSOURCE・12 probe・以下のcall範囲へのユーザーの別途明示実行承認が必要。
本書を実行許可や、未完成runnerの実行可能宣言として使わない。

## 科学対象と最小request集合

linear H4／1.00 Å／STO-3G／保存legacy DF12、Partial L_D3、δ=.02、
8 system qubit＋ancilla8、保存physical state・4電子sector70、K2、
extract_identity_phase、coefficient_atol=1e-12を固定する。

12 probeは「6種類のrそれぞれについて位相の基準になるq_probe1と、
boundary・複数eventを通すq_probe2」の最小の二profile設計。
全成分／全trajectoryを網羅する数学的最小集合という意味ではない。

| 親request（保存m/q/r/K） | 診断subrequest | 選定理由 |
|---|---|---|
| m0 / 1 / 1 / 2 | r1：q_probe1/all0、q_probe2/stress | Bのcutoff0から固定1e-12への接続、2 occurrenceで複数order2 |
| m8 / 256 / 2 / 2 | r2：同上2本 | r≥2の最初、1 occurrence内の複数event |
| m10 / 1024 / 4 / 2 | r4：同上2本 | 保存r4を他rの結果から推測しない |
| m11 / 2048 / 8 / 2 | r8：同上2本 | 保存r8のθとphase加算 |
| m13 / 8192 / 16 / 2 | r16：同上2本 | 保存r16のθとphase加算 |
| m17 / 131072 / 32 / 2 | r32：同上2本 | 最長r、既存m17 categorical入力への対応 |

親の保存q/N/α/β/εは全てmetadataとしてそのまま保持する。
`q_probe`は別fieldの局所診断長であり、親qの置換・新schedule・新Nの採用ではない。
q_probe2の結果をm8等の実際のqでの直接回路PASSと表示しない。
実際の18roundのqは[f490固定入力](../rpe_18round_ideal_enclosure_preparation_20261104/fixed_inputs.json)に従う。
この診断条件を受け入れるかも、別途実行承認の範囲に含める。

### profileの結果前固定

固定selectorはBと同じID順の `c0=df-fragment-10:Z0`（sign−）、
`c1=df-fragment-11:Z0`（sign＋）を使い、保存exportのbasis/support/signを先に照合する。
違えば代替を自動選択せずSTOP。異なるbasisだけで非可換と認定しない。

- `Z0` event：rotation c0、productsなし、n=0。
- `Z1` event：rotation c1、productsなし、n=0。
- `T−` event：rotation c0、products **c1,c0**、n=2、event.phase=−1。
- `T+` event：rotation c1、products **c1,c1**、n=2、event.phase=−1。
- all0：q_probe1のr位置でZ0,Z1を交互に置く。
- stress/r1：q_probe2にT−,T+の順で置く。
- stress/r≥2：まず2r位置をZ0,Z1の交互列にし、step0の先頭をT−、
  step0の末尾をT+、step1の末尾をT+へ置換する。

rotation firstの`_make_event`入力と、products firstのapplication metadataを別々に記録する。
θは親の**signal側**τ64に対する`atan(τ/(n+1))`を用い、hexを結果前に固定する。
固定された条件付きprobeであり、確率lawからのsampleではない。sampling、fresh-IIDの確認は0。

## SOURCE・入力・hash

production source treeは `f4905fff592351d33ba07b927892fb9532c99711`。
個々の参照module、必須GitHub報告、B実行SOURCEとの対応は
[SOURCE_BINDINGS.json](SOURCE_BINDINGS.json)に固定した。
今回追加したSOURCEは人工testのみであり、科学runnerのSHAはまだ存在しない。
将来runnerは次の既存処理を最小利用する。

| 役割 | 固定再利用先 |
|---|---|
| event/occurrence/repeated request | rte._make_event、df_partial_s2_cost._request_for_events、DFPartialS2RepeatedRequest.from_step_requests |
| circuit | QiskitDFRTEEventCircuitBuilder、QiskitDFPartialS2CircuitBuilder、QiskitDFPartialS2RepeatedCircuitBuilder |
| policy | register_support_restricted_bases、apply_run_threshold_policy、保存training fingerprint |
| wrapper | RPEHadamardInterrogationRequest、QiskitRPEHadamardBenchmarkCircuitBuilder（probe maximum=2） |
| 独立参照補助 | m17 exportのgate descriptors/primitive parameters、mean_adapterのanalytic gate規約、B embed_sector/trajectory_state |

| 入力 | 固定commit・path／SHA256 |
|---|---|
| H snapshot | carrier `4591b8d75a2a6a55c74a51225f0c61648023c8ee`、`artifacts/rte_connected_cluster_cost_validation/h4_sto3g_d100_rank12_ld3_dt0p1_ref4_k2_connected_pilot30_max1500_hold1500_rare375_v1.hamiltonian.npz`；`13e4b10d2347ed900fe4aa4b2238128ebe735a48988e321b9e4fabac4d092a3b` |
| physical state | B結果 `e22d7718c47358ff650dbc04e8e72f6ab1c8bf41`、`artifacts/rpe_fixed_request_measurement_b/2026-11-03/run01/physical_state.npz`；`793300c86417f52569289ce86980c5f9a87292e130199355a0f4f7056225c9c4` |
| component/basis/primitive export | m17結果 `34840ce9bf81d985b32764d9bc9d77645ed52e59`、`artifacts/rpe_m17_rigorous_mean_enclosure/2026-11-03/run01/source_replay.json`；`ad7fd3617a524207fbc86a77de471ed6222b010046822a11bd9675edcace1f08` |
| 18保存tupleとN/α/β/ε | V2 CSV `78c49d0b5f1d71e63b5959e78813237923bbc648`；`e830057532c0d7b5700b4d723ddfb77e6b0734d714177d18706331996d1b42b0` |
| tuple・τ・selector binding | f490 `fixed_inputs.json`。自身のhashと内部入力hashもSOURCE_BINDINGSに保存 |
| m17 law・包含 | 固定m17 `categorical_laws.json`／`enclosure_operators.zip`／RESULT。hash一覧はf490入力一覧。今回再評価しない |

保存H hash `56e4df83655aa2f2f8132126f2635996516dc2cfb1bf4cb8d95490184ad631e5`、
state fingerprint `a07d6c7fdbbb23a035f9fc54b9956ee79e557a2185e29b1870748b771958be90`、
preparation hash `89145d8324bf139eaec67265f3134a5c1a3cebb7ed97792d54ed3221d31f8919`、
tail hash `65b31bcf432b703c523753ee4607236a2e9530d93401dcec9255e686cc3f04af`を照合する。
H/state solve、SCF、DF再因子化、保存vectorの差替えはしない。

`make_rte_config/_finite_inputs`をそのまま呼んでmodel側τへ戻さない。
runner準備ではsignal側τ・固定θをexplicit config/distributionへ渡す最小adapterを人工検証する。
既存validatorや準備hashの規約を緩和して受理させない。新configのscience parameter replayは将来の6群callに計上する。
他5群の過去CDF payload復元や分布平均再計算をこの計画に含めない。
別Codexの全18包含結果を使う場合は、その完了後の公開commit/hashを別途固定してから接続する。

## 回路と独立参照

各probeは同じ強制event列を使ってuncontrolled raw、ordinary-controlled raw、
ordinary-controlled boundary_optimizedを構築する。
controlled2経路には保存`support_run_le_1`、training fingerprintを明示する。
無計画full-basis経路で黙って代用しない。
Hadamardはboundary_optimizedにcosine/sine各1本、測定付きで構築する。

参照はcontrolledのcontrol1 blockを読み出して作らない。
保存exportのRZ/XXPlusYYパラメータからanalytic basisを構成し、
occupation energy（one-body／λ n_eta²）からdeterministic halfを作る。
constant/IをFに1回だけ含め、signed component行列と閉形式
`cos θ I-i sin θ P`のeventを列順に作用させる。
productionのdiagonal primitive appenderを参照側で呼ばず、Bの共通依存を一段減らす。
保存export・DFの意味は共通依存であり、Hamiltonian定義全体の独立証明とは呼ばない。

両control branchを**位相alignmentなし**で比較する。
control0の期待値は元ψ、control1は理論V_Tψ。
測定付きwrapperのcopyから終端測定だけを外し、ancilla Zのideal meanを得る。
実状態準備回路、量子backendは含めない。

## 必要な科学callと判定

以下は将来の1回の限定batchに対する上限であり、今回の実施数は全て0。

| 操作 | 提案数 |
|---|---:|
| 保存H/stateのload | 各1（再solve0） |
| 固定DF preparation / support basis登録 | 各1 |
| signal側τによるfinite config/distribution parameter replay | 6群、CDF構築0 |
| 参照F/R（2half一組）／選定signed component参照 | 1組／2 component |
| 固定trajectory参照vector＋complex signal | 12 |
| uncontrolled raw evolution build / state-action | 各12 |
| controlled raw / boundary_optimized evolution build | 各12、計24 |
| controlled branch state-action | 各回路2branch、計48 |
| measured wrapper build / ideal mean-action | 各24 |
| real H4 new q≥4 circuit build、q-sized長回路、compile、sampling、quantum shot | 全て0 |
| 新categorical mean包含、成功率/費用再解析、状態準備/noise | 全て0 |

Qiskit1.3.0、NumPy1.26.4、SciPy1.14.1、complex128/float64を基準にする。
参照analytic factorは96-bit Acbで構成する案とし、追加160/224 bitで自動再試行しない。
branch vector、uncontrolled vector、各axis meanの絶対残差を既存Bと同じ1e-10で比較する。
参照の包含半径を併記し、参照不確かさを含む上端で判定できないときはUNKNOWN/STOP。
global-phase alignment、複素signalの共役、半径へのprojectionでPASSへ変更しない。
component commutator、sector leakage、norm、basis preserved-column residualを併記するが、
浮動小数点0を厳密な0の証明にはしない。

局所atolのPASSは長q誤差保証ではない。局所残差を単純にq倍して勝手に新保証を付けない。
全supportの理想同値性とπ64 phase transferを対応付ける作業が、長q接続には別に残る。
特にboundary scalar再結合とblock融合の転送は既存event π64 boundに含めず、
局所結果の記録へ分離する。この計画で長qの新しい誤差上界を計算・認定しない。

## 効率と共有環境

r群の独立probeを並列化し、immutable basis/half参照をcacheする。
q_probe≤2なので最大64eventであり、q131072×r32の実回路を展開しない。
起動前にCPU affinity・空きRAM・GPU可用性・既存timingを確認し、
小行列のBLAS threadとprocess数の積を調整する。CPU/RAM/worker・phase/total wallの人工上限は設けない。
出力数・science call数・precision・対象scopeは上表のまま。
共有環境や他ユーザー・SERVER2 H8のjob/filesを変更しない。
他jobへ支障がある場合は自jobだけを調整／停止する。速度比較なしにspeedup倍率は主張しない。

## 得られること・残ること・mandatory STOP

通過すれば、固定preparation・signal側θ・全6rについて、選定12条件の回路control/phase/order/測定平均を局所確認できる。
全218成分、任意event列、任意support-run、保存全18qの直接回路、compiled correctness、
実IID/virtual guard/PCG64/noise/state prep、RPE全体のtolerance保証は得られない。

必須入力/hash/schema/sector/tuple不一致、selector不成立、保存signalθに結合不能、
SOURCEの予期しない変更、参照半径不足、nonfinite、1e-10不合格、共有環境への支障でSTOP。
失敗後にprofile/θ/cutoff/toleranceを変更して科学retryしない。
科学call上限到達またはbatch完了後もmandatory STOP。
未評価を0やPASSとして埋めず、旧grant/ledger/marker/STOPを消費・更新・再発行しない。
技術的修正は別versionへ分離し、再科学実行には適用される別許可を得る。

全18categorical結果とこの局所回路検証が両方揃っても自動certifyしない。
全体 `RPE_tolerance_certified=false`、`scientific_status=UNDETERMINED`を保持する。
