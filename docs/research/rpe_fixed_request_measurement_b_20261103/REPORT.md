# RPE 固定request B：物理入力・controlled/Hadamard平均の局所照合

2026-11-03 JST / LOCAL。**固定6 request・12測定軸は全て既存matrix許容差1e-10以内で一致した。**
判定は`LOCAL_FIXED_REQUEST_MEANS_CONSISTENT_MANDATORY_STOP`。
科学実行は1回で終了し、sampling・compile・費用再解析・独立詳細科学レビューは行っていない。

## 問い・固定条件・SOURCE

同じ物理入力ψと同じ有限RTE trajectoryについて、理論〈ψ|V_T|ψ〉のRe/Imと、
実際にbuildしたordinary-controlled/Hadamard回路のideal測定平均が一致するかを調べた。
前段[計画B](../rpe_18round_measurement_connection_20261103/MINIMAL_SCIENTIFIC_VALIDATION_PLAN.md)
と[今回の実行範囲](EXECUTION_SCOPE.md)に限定する。

linear H4、隣接1.00Å、STO-3G、保存legacy DF rank12、8 system qubit、
4電子sector dimension70。Partial L_D3 / Full L_D12、δ=.02、
保存18round schedule内のm0/q1とm3/q8だけを使う。
Partialはr1/K2、Fullはtail-free r0/K0。PF公式とN/α/β/εを変えていない。
Nはmetadataとして保持するだけで、今回有限shotを生成していない。

実行SOURCEは`4d7c455db3ed7d3e4bc9e2e1fd4dbeba9f46b272`。
[runner](run_fixed_request_b.py)、[人工tests](test_fixed_request_b.py)を実行前commitで固定した。
旧16主要moduleはcarrier `4591b8d75a2a6a55c74a51225f0c61648023c8ee`とbyte一致。
carrierを旧M06-Fのclean実行SOURCEと呼ばず、旧dirty provenanceと局所STOPを保全する。
H4の新SCF/DF生成はせず、既存snapshotをloadした。

| 入力 | identity |
|---|---|
| 保存H snapshot | `artifacts/rte_connected_cluster_cost_validation/h4_sto3g_d100_rank12_ld3_dt0p1_ref4_k2_connected_pilot30_max1500_hold1500_rare375_v1.hamiltonian.npz` |
| snapshot SHA256 | `13e4b10d2347ed900fe4aa4b2238128ebe735a48988e321b9e4fabac4d092a3b` |
| H hash | `56e4df83655aa2f2f8132126f2635996516dc2cfb1bf4cb8d95490184ad631e5` |
| 共通physical state fingerprint | `a07d6c7fdbbb23a035f9fc54b9956ee79e557a2185e29b1870748b771958be90` |

## 実施した照合

Partialのcomponent_id順selectorはc0=`df-fragment-10:Z0`（係数符号−1）、
c1=`df-fragment-11:Z0`（符号+1）を選んだ。両者のbasis_idは異なる。
各qで全order0 profileと、先頭occurrenceだけorder2のprofileを用いた。
order2はrotation c0、products c1,c0、event.phase=−1である。
この強制profileは**trajectory分布の標本ではない**。

Partial4、Full2の計6 controlled evolutionと、cosine/sineの12wrapperをbuildした。
boundary_optimized、support_run_le_1、保存training fingerprintを維持し、
Partialのsupport applicationはq1で各1、q8で各8だった。
constant/extracted identity、event符号はancilla-relative phaseに保持する。
H→controlled V→H→ZでRe、H→controlled V→S†→H→ZでImを読む。
bit0→+1、bit1→−1、U=exp(-itH)の規約を変更していない。

理論側ではuncontrolled deterministic halvesをsectorで構成し、signed component作用と
既存_apply_event_to_stateを使った。元_trajectory_signalはvectorを捨てるため、
同じ順序のvector-returning companionを最小追加し、人工非可換toyで元scalarと照合した。
理論signalを比較対象controlled回路の出力から作っていない。
一方、half参照にも既存deterministic builderを使うため、両経路に共通するモデル誤りを
独立に検出する証明ではない。

実回路側では同じphysical stateをsector/endian permutationで8qubitへ埋め、
ancilla |0〉と|1〉の両branchへの作用を比較した。
測定付きwrapperのcopyから終端測定だけを除いてideal Z expectationを計算した。
実state-preparation回路、量子backend、noiseは含めない。
control作用をglobal-phase alignmentして合格にする処理は用いない。

## 結果

以下は上記H4/DF12/δ=.02/物理入力、指定q/r/Kの固定requestだけのlocal numerical evidence。

| L_D / q / profile | 両軸mean残差の最大 | 両control branch作用の最大振幅残差 |
|---|---:|---:|
| 3 / 1 / all order0 | 1.7764e-15 | 2.3108e-15 |
| 3 / 1 / first order2 | 1.4433e-15 | 1.2985e-14 |
| 3 / 8 / all order0 | 7.4385e-15 | 1.8385e-14 |
| 3 / 8 / first order2 | 8.6597e-15 | 1.3418e-14 |
| 12 / 1 / deterministic | 1.1102e-15 | 1.2018e-15 |
| 12 / 8 / deterministic | 4.7184e-15 | 9.5880e-15 |

全12axisが1e-10内。control branchのsector leakageは今回の評価で全て数値0、
reference half leakageとsupport completionのpreserved-column残差も数値0だった。
これらの浮動小数点0を厳密な数学的0の証明へ昇格しない。

保存Hからphysical stateを1回再構成した。energyは保存値−2.1663874486347616Haと一致し、
ground-state residualは1.397e-15、norm誤差は1.11e-16。
再構成vectorのfingerprintは保存physical fingerprintと一致した。
これは保存されたhashとの一致であり、未取得の元state vectorそのものとの直接byte比較ではない。
今回のFull q1/q8は同じψを明示的に受け取り、新記録にfingerprintを保持する。
**原Full18round診断自身のmissing fingerprintを今回の記録で補完しない。**

## 実行・技術検査と原本保全

人工testsは11件PASS。component selector、強制profile、endian埋込み、元scalarとの
非可換toy比較、負sine、relative phaseの誤り検出、serial/process経路を確認した。
主エージェント自身の技術検査であり、独立詳細科学レビューではない。

Python3.11.0rc1 / NumPy1.26.4 / SciPy1.14.1 / Qiskit1.3.0、complex128/float64。
CPU32論理コア・利用可能RAM約51GiBを確認し、6 requestのideal actionを6 spawn processで実施。
512振幅の小行列に対するBLAS overheadを避けるため、job内BLASは1thread。
必要な2componentのbasis cacheを使い、全tail componentの重複dense構築を避けた。
GPU driverは利用不可でCPU経路を使用。共有環境・他jobの設定は変更していない。
観測wall timeは約44.93秒。peak RSSやspeedup factorは測定していない。

| 科学操作 | 実施数 |
|---|---:|
| physical ground-state solve | 1 |
| uncontrolled reference half構築（2halfを一組） | 2 |
| 理論trajectory signal | 6 |
| controlled evolution build | 6 |
| measurement wrapper build | 12 |
| control branch state-action | 12 |
| Hadamard ideal mean-action | 12 |
| RTE/RNG sampling・quantum shots・compile・価格評価・旧保存signal再解析 | 全て0 |

fresh output `run01`に1回だけ実行し、exit0で完了。SOURCE変更・科学的retry・別profileへの置換なし。
旧V1/V2/V3/R1、H/state/reference/raw/QPY、grant/ledger/marker/STOP、他研究worktreeは変更していない。
旧6種rの全sampling lawや長q価格を実行したとはしない。

## 証拠と解釈の限界

- [RESULT.json](../../../artifacts/rpe_fixed_request_measurement_b/2026-11-03/run01/RESULT.json)：各requestの理論complex、12axis平均、control残差、設定、SOURCE、call数。
- [physical_state.npz](../../../artifacts/rpe_fixed_request_measurement_b/2026-11-03/run01/physical_state.npz)：今回再構成したphysical vectorとsector/basis/hash。元snapshotを変更していない。
- RESULT SHA256：`b9b4615ed862a7a85dc6790692f8632d0997d53a1ad36ad59788572a64a51283`。
- state NPZ SHA256：`793300c86417f52569289ce86980c5f9a87292e130199355a0f4f7056225c9c4`。

本結果は同一state/requestの**条件付き平均**という接続の空白を6requestで狭める。
保存attenuated event-meanへの分布平均、fresh-IID測定、r≥2/q>8、全18round成功率、
PF有効固有状態入力、厳密半径下界、実準備・compiled correctness・backend/noiseは未検証である。
componentのbasis_idが違うことだけから非可換を証明したとはしない。

V1のFull7点半径不一致と同δのPF control未発見、V2両候補CONDITIONAL_SUCCESS、
V3長q価格外挿、R1 Partial約4.858%点選好と総合UNDETERMINEDは変えない。
今回の合格を新しいRPE成功率や費用優位のGOへ読み替えない。

## 次の担当とmandatory STOP

次の提案は候補A：今回保存したphysical stateを再利用し、Partial L_D3/δ=.02/r1/K2の
保存q1（各軸N510）、q8（各軸N454）でfresh trajectory/shot→条件付き±1測定→marginalの
接続を検証する。N/α/βを変更せず、固定seedと判定を結果前に定める必要がある。
一つの有限batchで稀な失敗率や全18roundのIIDを保証できるとはしない。

**ユーザー**がAの新sampling・state-action・仮想測定の別実行許可を判断する。
**Codex**はその許可後にAの限定SOURCE・seed・判定を固定し実行する。
今回はAのbackend作成・samplingを開始していない。
科学記録・commit/push・remote照合を終えたらmandatory STOP。
詳細レビュー、C/D、全18round実測、compile、費用再解析へ自動移行しない。
