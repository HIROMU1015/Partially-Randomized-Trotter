# m17 ideal categorical平均の包含：固定実行計画（未実行）

2026-11-03 JST。**実行SOURCEは本folderを含む公開commitで固定し、ユーザーの別の明示実行許可後だけ起動する。**
今回の依頼は技術準備まで。本計画のH4 replay/CDF/包含平均は未実行。
数式と保証の対象は[ERROR_BOUND_SCOPE.md](ERROR_BOUND_SCOPE.md)。

## 固定条件・SOURCE

linear H4、隣接1.00 Å、STO-3G、保存legacy DF12、Bのphysical ground-stateベクトルのray、
8qubit・4電子sector70、Partial L_D3、単一δ=.02、m17/q131072/r32/K2。
extract_identity_phase、coefficient_atol=1e-12、保存N411/axis、α=.016957089147912167、
ε=.15244325851620388、保存β_PF/β_RTE/β_statを保持する。
旧RESULTのfixed_literals.saved_meanをV2保存μ0と同じbinary64として読み、再推定・再最適化しない。
α/εは元m17の残余評価と同じ保存decimal文字列をexact rationalで解釈する。

新科学SOURCE：[export_adapter.py](export_adapter.py)、[mean_adapter.py](mean_adapter.py)、
[ball_adapter.py](ball_adapter.py)、[run_mean_enclosure.py](run_mean_enclosure.py)、[requirements.txt](requirements.txt)。
これらをHEADの固定commitへ一致させ、SOURCE/hashとpackage versionsを原RESULTへ記録する。
承認証明書やgrant等は不要。SOURCE検査は数値再現性のためで、許可の証明ではない。

主な再利用元：carrier `4591b8d75a2a6a55c74a51225f0c61648023c8ee`、
元m17 SOURCE `128e9a5bf2630cd0333ec5f69b81b25def78eb79`、
候補A修正SOURCE `d0b1da1b04c266a1f7abf8e43966ca6e289a8695`。
Task A/B-1aは別の固定helperと原記録として参照する。carrier・実行SOURCE・結果公開commitの役割を混同しない。

## 固定入力

| 結果commit / folder | File | SHA256 |
|---|---|---|
| carrier / `artifacts/rte_connected_cluster_cost_validation/` | `h4_sto3g_d100_rank12_ld3_dt0p1_ref4_k2_connected_pilot30_max1500_hold1500_rare375_v1.hamiltonian.npz` | `13e4b10d2347ed900fe4aa4b2238128ebe735a48988e321b9e4fabac4d092a3b` |
| cd749066… / `artifacts/rpe_m17_mean_validation/2026-11-03/run01/` | RESULT.json | `79fdf7e84f90c48230857b13f39f9becb30f19892e2e9c09d80805d346432a4a` |
| 同上 | operators.npz | `e1a65238e9d9b61a6d0aa542ca7f509827b0a761fc751615f1f2018a0e7020df` |
| e22d7718… / `artifacts/rpe_fixed_request_measurement_b/2026-11-03/run01/` | physical_state.npz | `793300c86417f52569289ce86980c5f9a87292e130199355a0f4f7056225c9c4` |
| a5304864… / `artifacts/rpe_m17_precision_isolation/2026-11-03/run01/` | RESULT.json | `e5c3cb1cd455d102c8a1e74d7556ec95b26ab79e72f0727e172f4483162f362e` |
| 同上 | frozen_steps.npz | `2250e10bcae320c2fe1ec30662eb9ce6e14df50798f3db2d78f60c28dcfd7c30` |
| f766a1f7… / `artifacts/rpe_m17_b1_construction_precision/2026-11-03/run01/` | RESULT.json | `d392c2430882ca239e1e4ed5f8095dc912b5ef3597313b98cf0bf51e5ea7415e` |

full commitはrunnerと[先行入力調査](../rpe_m17_b1b_b2_assessment_20261103/INPUT_AVAILABILITY.md)で固定。
元operators/frozen_stepsはbyte/header保全照合のみ。保存A/F/R/HP operatorを真のideal回路として流用しない。
B-1a scalar_traceのθ64は元m17の保存値ではなく、その実行でSOURCE replayして保存された新入力である。

## 実行順序と最大範囲

1. pinned version・新SOURCE・既存input/helperを照合し、fresh出力directoryを一度作る。
2. 保存H snapshotをloaderで読み、H hash、8qubit/DF12を照合。保存ψ/sector/hash/fingerprintを照合する。
3. 既存split_df_hamiltonian_by_ld/prepare_df_partial_s2を一度呼び、L_D3の218成分・4 deterministic blockを取得。
   係数/確率・Gaussian basis descriptor/local matrix・diagonal primitiveの生成を許可対象のSOURCE replayとする。
   preparation hash `89145d8324bf139eaec67265f3134a5c1a3cebb7ed97792d54ed3221d31f8919`、
   tail hash `65b31bcf432b703c523753ee4607236a2e9530d93401dcec9255e686cc3f04af`、保存λを照合。
   成分順/符号/係数/確率、basis ID・hash・descriptor・局所matrix、cutoff/identity、prefix/tail、phase角をexport。
   新生成物をSOURCE_REPLAY_GENERATEDと表示する。未対応gateやhash/順序不整合はSTOP。
4. component probability vectorと保存order vectorにNumPy1.26.4のCDF処理を各1回だけ適用。
   CDF64境界とexact 53-bit grid massを保存。全体のCDF和を再生成してPASSを狙う反復はしない。
5. 固定grid law・ideal gateでν_logicalを包含し、builderのπ64 summation transferも加えてν_catのRe/Imを包含。
   ψは保存rayのexactなnorm²で期待値を割り、rawを変更しない。sector不変性は許されたgate式で確認する。
6. μ0からのU_catと固定N/αによる統計半幅sを外向きに評価し、両軸のstrict十分条件を判定する。
   η_exec←catはnull。categorical結果と実実行保証を分離して原結果へ記録する。
7. 正常/不整合/中断/未判定のいずれでも結果を保全してmandatory STOP。

最大新科学call：component/basis preparation/export1回、component CDF1回、order CDF1回、
1 complex平均の包含を最大3回（96/160/224 bit）。一つのcomplex平均からcosine=Re/sine=+Imを取得する。
各精度ではbasis/component/event/half/r/q/期待値・phase bound・予算比較をその精度で一度評価する。
norm/radius計算もこの包含callの内部処理として許可範囲に含める。
成分replayとCDFを精度ごとに再生成しない。SOURCE replay内の対角化は保存DF因子からのbasis準備で、SCF/DF因子化とは区別する。

## 方法・精度・判定

python-flint0.9.0/FLINT3.6.0、NumPy1.26.4、SciPy1.14.1、Qiskit1.3.0、qiskit-nature0.7.2を固定。
exact dyadic/rational入力、解析的RZ/XXPlusYY、Acb演算、global spectral radius、binary powerを使用。
人工8次元dense unitary/q131072で直接entrywise powerとの比較を行い、この方法を科学実行前に選定した。
実行結果に合わせて別算法へ切り替えない。

まず96 bitで1回。両軸の条件が確定するか、少なくとも一軸で十分条件の不成立を下界から確定したら終了。
区間幅により判定が確定しない場合だけ160 bit、必要なら224 bitへ進む。
224 bitでも未判定なら追加精度・再試行なしでSTOP。
phaseの一様上界等が幅の原因の場合、高精度化しても解消するとは約束しない。

| 結果 | 規則 |
|---|---|
| CONDITIONAL_IDEAL_CATEGORICAL_SUFFICIENT_CONDITION_CONFIRMED | 両軸でupper(U_cat+s)<lower(ε) |
| SUFFICIENT_CONDITION_EXCLUDED | 包含から得る誤差の下界とs下端の和がε上端以上となる軸あり。「この十分条件では認定不可」 |
| UNDETERMINED | 区間が境界をまたぐ、幅不足、必要な入力/bridgeがない |
| INTERRUPTED_OR_INVALID_MANDATORY_STOP | 非有限、norm²下端≤0、不正CDF、unsupported gate、hash不一致、exception/interruption |

原INCONSISTENT・旧u_num=nullを保持。η_exec←catが未確立なので、categorical条件が確認されても
RPE_tolerance_certified=false、scientific_status=UNDETERMINEDを維持する。
1e-10は過去の診断用閾値であり今回の合格条件ではない。保存統計残余約8.60e-5は過去結果の引用で、準備時には再計算しない。
finite-RTE/PF予算・β_stat/εの二重会計、他roundや成功確率の新評価はしない。

## 資源と実行前条件

実行前にCPU/RAM/GPUと共有負荷をread-onlyで確認する。
dominant処理は70次元matrix積とqのbinary power。basis共有、componentのbasis単位集約でevent列挙を避ける。
FLINTの内部threadを利用可能CPUへ合わせ、process/BLASとの多重化を調整する。独立でないmatrix chainを重複して並列評価しない。
人工試験のFLINT threadは1、将来の科学runnerはCPU affinityから設定する。人工worker/CPU/RAM/wall capは設けない。
他job/共有環境を変更せず、自分のjobだけ調整する。性能向上率や実H4の包含幅は未測定。

実行前条件は、公開された本SOURCE、requirements環境、固定input照合、人工テストPASS、
上記のstate ray・phase/CDF・ideal scopeへのユーザーの明示的実行許可。
追加の承認証明ファイル、grant、署名は不要。

別承認後のコマンド：

```bash
python docs/research/rpe_m17_rigorous_mean_enclosure_preparation_20261103/run_mean_enclosure.py --run --output-dir artifacts/rpe_m17_rigorous_mean_enclosure/2026-11-03/run01
```

## 出力とSTOP

新出力folderだけに、RESULT.json（SOURCE/input/条件/各bitの包含endpoint・U_cat/s/判定・null bridge・資源/終了）、
source_replay.json（成分/primitive/basis/local matrix）、categorical_laws.json（CDF64 hex/exact mass）、
enclosure_operators.zip（各bit・中間operatorのexact center/global radius/norm上端）を保存する。
エラー時も部分RESULT・取得済みexport/operatorを保持し、既存folderへ上書き・resumeしない。
正常な未判定・十分条件除外を技術中断の再試行対象にしない。
技術中断の修正は既存の許可範囲に限り、失敗原本・新修正版SOURCEを保全して扱う。

今回の準備中は上記artifact folderを作らず、実H4のSOURCE replay/CDF/包含/数値再解析を開始しない。
科学実行終了後もB-1b/B-2、R2/R3、他round、sampling/量子shot、controlled回路build/compile、
価格/成功確率再評価、独立詳細科学レビューへ自動移行せずmandatory STOP。
