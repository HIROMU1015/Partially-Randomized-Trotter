# 研究C：Reduced Uの有限Armijo backtracking・実H4実行結果

2026-11-03 JST。ユーザーの明示実行許可に基づき、固定科学SOURCE [`025d495166311a58ec9c24c43f07ed6e40f65f42`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/025d495166311a58ec9c24c43f07ed6e40f65f42)と[契約](research_c_u_step_control_contract_unapproved_20261103.json)を無変更でfresh run01へ一度実行した。**64/64 U更新を数値・物理資格付きで受理。1042/1042 logical calls COMPLETE、correctness FAIL/retry/resume 0。停止理由は`MAXIMUM_ACCEPTED_UPDATES`であり、J/U同時一次収束は未達。公開照合後mandatory STOP。**

[全結果](../../artifacts/research_c_u_step_control/2026-11-03/run01/result.json)／[実行集計](../../artifacts/research_c_u_step_control/2026-11-03/execution_summary.json)／[Calls](../../artifacts/research_c_u_step_control/2026-11-03/run01/science_calls.jsonl)／[STOP](../../artifacts/research_c_u_step_control/2026-11-03/run01/STOP.json)／[保全確認](../../artifacts/research_c_u_step_control/2026-11-03/execution_postflight.json)。歴史的な契約名・`UNAPPROVED_SCIENTIFIC_PROPOSAL` metadataは変更せず、許可の証明書・保存原文・新grantを作成していない。

## 1. 固定条件と実行範囲

保存直鎖H4・隣接1.00 Å・STO-3G、number CDF rank8、元frame順LD2、一般signed実対称J・実直交U、元g補正κ・核scalar一回、Nα=Nβ=2/sector36。目的は元の`Fγ=f/f*+γ(Mε/M*)²`、γ=.1、ε=1e-8 Ha、f*=2.1839632053132894e-9 Ha²、M*=2.2093719370722074 Haを維持した。元codec/gauge・数値閾値・拡張精度評価も不変である。

今回の対象は既存PR reduced objectiveの数値最適化と独立DF物理評価だけ。PFのδ窓・r/Kは今回の評価に非適用で、PF/RTE・ground-state性能・shots・native・総資源を実行していない。DF予算CA/8=.00019875 HaのPASSを、全CA達成や量子資源優位へ読み替えない。

初期U/Jは[旧統合実験](research_c_pr_integrated_execution_20261102.md)の`update005`、candidate SHA256 `938340b8e09c85f5f4021582501ba6d81fc79d5382c37c1824863939bdf6e56e`。保存点の来歴・STOP・NPZ hashを照合し、独立Phase Aの物理予算を再評価した。さらに元initial.npzのcold Jから同じUを新たにsolve＋2 refinementし、そのrefined初期点も数値・物理資格を通した。新初期candidate hashは保存selectedと一致した。[input identity](../../artifacts/research_c_u_step_control/2026-11-03/run01/input_identity.json)／[Phase A](../../artifacts/research_c_u_step_control/2026-11-03/run01/phase_A.json)

旧543-callのresume・残枠流用ではない。今回の初期評価、64更新、全trialは新契約の独立fresh実験である。準備時の[194人工・回帰PASS](research_c_u_step_control_preparation_20261103.md)は不変SOURCEの技術証拠として参照し、今回科学実行時に再テストしたとは記載しない。

## 2. 固定したstep制御と実績

各recentered η=0で数値資格付きphysical tangentからd=-gを作る。初回α0=min(1,1.01/‖g‖)、後続中心はさらに`2×前回受理α`とのmin、α_i=α0·.5^i（i=0..31）。元の値比較尺度τvを保持し、`Ftrial+τv≤F0+1e-4 α gᵀd`かつ`Ftrial+τv<F0`を要求する。Provisional pass後に独立終端、次いで独立DF物理評価を通してから受理・recenterした。[固定SOURCE](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/025d495166311a58ec9c24c43f07ed6e40f65f42/src/trottertracks/c_u_step_control.py)

| 評価・判定 | 実績 |
|---|---:|
| 新U点（refined初期を含む） | 151 |
| Trial | 150 |
| Provisional Armijo PASS→独立終端PASS→物理PASS→受理 | 64 |
| 数値適格だがArmijo／値尺度で棄却 | 81 |
| Reduced-gradient数値未資格で棄却 | 4 |
| J数値未資格で棄却 | 1 |
| 元手順によるinner J solve | 453（各新U点3回） |
| うちrefinement | 302（各新U点2回） |
| 独立終端評価 | 64 |
| 物理評価 | 66（保存selected＋refined初期＋64受理候補）、すべて予算PASS |

出典は[全trace](../../artifacts/research_c_u_step_control/2026-11-03/run01/trajectory.jsonl)の`backtracking_trial/envelope/inner_summary/independent_terminal_assessment/accepted`と[集計](../../artifacts/research_c_u_step_control/2026-11-03/execution_summary.json)。146点が`NUMERICAL_PILOT_ADMITTED`、5点が未資格であり、全151点がadmission PASSだったという主張ではない。

最初の方向は16 trial目にα=3.0517578125e-5でprovisional passした。最後の方向は1 trial、α=1.1920928955078125e-7。すべての64方向で登録32trial以内に受理候補が得られた。今回の終端は有限step集合の不成立ではなく、登録最大64更新への到達による。[result](../../artifacts/research_c_u_step_control/2026-11-03/run01/result.json)の`trajectory.trials`

数値未資格trialは以下のとおり。Decision内のobjectiveはnull、`gradient_used=false`であり、未資格点の勾配を次方向や受理判定へ使っていない。

| Trace label | α | 棄却status |
|---|---:|---|
| k000_0 | 1 | REDUCED_GRADIENT_UNQUALIFIED |
| k000_7 | .0078125 | REDUCED_GRADIENT_UNQUALIFIED |
| k000_14 | .00006103515625 | REDUCED_GRADIENT_UNQUALIFIED |
| k001_2 | .0000152587890625 | J_UNQUALIFIED |
| k047_2 | .00000005960464477539063 | REDUCED_GRADIENT_UNQUALIFIED |

## 3. F・exact λ・J/U一次条件・物理精度

次表は保存された`trajectory.history`の初期・8更新ごとの値を転記した。全65履歴は[全結果](../../artifacts/research_c_u_step_control/2026-11-03/run01/result.json)に保持し、checkpointを新しく計算していない。J∞・physical U₂は元のstable/binary64両経路を使うstationarity判定値である。Fは自己目的でありenergy誤差ではない。

| 点 | Fγ | Exact tail λ [Ha] | J∞ | Physical U₂ |
|---|---:|---:|---:|---:|
| initial | .006617509794531903 | .5683498546862645 | 9.5932671593e-8 | .658935814277 |
| update008 | .006600399525176942 | .5676143515941514 | 1.2130055662e-7 | 1.69235387517 |
| update016 | .006560816871882698 | .5659100607514386 | 7.9632759558e-8 | .655701638598 |
| update024 | .006167183072472236 | .5486707896215567 | 1.0562370227e-7 | .630265429431 |
| update032 | .006071661682254264 | .5444048787114728 | 9.4566306052e-8 | 1.69688093329 |
| update040 | .006034151122706684 | .5427208491145672 | 1.2365854442e-7 | .618041765106 |
| update048 | .006033780263975387 | .5427041695617647 | 5.6751667033e-8 | .612778267283 |
| update056 | .006033701772850164 | .5427006199699683 | 8.8258993324e-8 | .653691997360 |
| update064 | .006033479843764215 | .5426906587519222 | 8.9799029404e-8 | .608660383670 |

全受理点でFは低下し、終端のexact λも初期より小さい。JとUの残差がすべて単調改善したわけではない。U₂は途中で1を超えており、終端でも元一次閾値1e-5に対して`NOT_MET`。Jは初期・全受理点で`MET`だが、J/U同時一次条件は未達である。

終端のstable J∞=4.94210978980265e-8、binary64 J∞=8.979902940426526e-8を別保存した。Stable physical U₂=.6086603836696319、binary64 U₂=.6086603724084128、直交性残差=4.884981308350689e-15。元の値・gradient allowance、latest-pair U差1e-6、rank/nullspace/tail injectivity、独立終端規則を変更していない。[result](../../artifacts/research_c_u_step_control/2026-11-03/run01/result.json)の`terminal_state`

| 物理精度 | 新refined初期 | update064 |
|---|---:|---:|
| Sector operator誤差 [Ha] | 5.233373518021879e-9 | 5.673413458637335e-9 |
| Engineering guard [Ha] | 3.227392702910976e-9 | 3.2244465214668673e-9 |
| Sector誤差＋guard≤CA/8=.00019875 Ha | PASS | PASS |

終端tensor Frobenius=3.358086260639999e-9、general-K/raw CAR max-entry=4.135580766728708e-15（元1e-10以内）。新候補は`terminal.npz`、SHA256 `53d8f5027ce9e61bc72f6943c0268666bfcf87515154850a9a99326687b26ed3`で、物理評価candidate hashと一致した。[終端物理評価](../../artifacts/research_c_u_step_control/2026-11-03/run01/physical/update064.json)

物理残差そのものを初期より改善した結果ではなく、共通予算を保ちながらFとλを低下させた数値観測である。Engineering guardによるnumerical pilotであり、厳密DF enclosureではない。

## 4. Raw optimizer・数値資格・保証を分ける

453 inner Newton-CGのraw statusは0が388件、2（precision loss）が64件、1（maxiter）が1件。各solveの登録J基準はMET449件／NOT_MET4件だった。**すべてのinnerが正常optimizer収束した、あるいはJ基準を通ったとは記載しない。** 初回solveは観測し、登録2 refinement後の最終inner・独立Jやlatest-pair等がadmissionを決める元規則を維持した。

元のSciPy pre-step jac OR accepted/endpoint照合規約、same-decoded-pointと独立J/physical-U評価、raw x/jac/traceを保全した。Correctness FAIL0は、未資格trialやraw precision lossが0という意味ではない。

Inner nit合計60737、nfev118211、njev118149、nhev60801。終点・照合を含む実VG119570／Hessian60801も保存記録から集計した。Logical unitはこれらの1 iteration/nfevと同じ単位ではない。[集計](../../artifacts/research_c_u_step_control/2026-11-03/execution_summary.json)の`raw_solver_status_counts/inner_J_stationarity_counts/inner_counters`

有限inner解に対するadmissionと数値Armijoを通したという結果であり、厳密な`min_J F`、真のreduced-gradient誤差上界、厳密降下・収束保証ではない。`true_reduced_gradient_qualification`・`certified_descent`・`actual_CA_achieved`はnullを保持する。新しいffsim joint fit、外部outer BFGS、RC2継続、追加refinementは0だが、inner Newton-CGの実反復は上記のとおり行っている。

## 5. Calls・資源・保全

| Logical unit | COMPLETE |
|---|---:|
| saved_load / initial_identity / physical_context / phase_A_assessment / objective_setup / optimization_driver / export | 各1＝7 |
| inner_J_solve | 453 |
| precision_state | 302 |
| reduced_admission | 151 |
| independent_terminal_assessment | 64 |
| physical_assessment | 65 |
| **合計** | **1042/1042 COMPLETE、FAIL0（契約最大12430）** |

最大64更新は消費済みであり、残るcall枠を使って65回目を試したり、別candidate・namespaceへ継続したりしない。Retry/resume0を保持する。

[実行前確認](../../artifacts/research_c_u_step_control/2026-11-03/execution_preflight.json)で固定SOURCE212（新5＋旧207）・入力/来歴9、fit/evaluation両環境、fresh出力を照合した。実行前CPU affinity32、MemAvailable約52 GB、memory pressure avg10=0。実行中も共有負荷・memory pressureを確認した。GPU照会はdriver通信失敗だったため、固定されたCPU経路を維持し、GPU/共有設定を変更していない。

主workloadは逐次依存するcold J solve・2 refinement・U/trialである。既存のvectorized Newton-CG/Hessian、拡張精度residual/JAX、persistent physical workerを使用し、プロセス内BLAS/OpenMP1で小行列処理のthread競合を避けた。別方向・未来trialの先行実行や科学条件を変える高速化を追加していない。共有環境・他ユーザーjobへの変更はない。

[資源記録](../../artifacts/research_c_u_step_control/2026-11-03/run01/resources.json)：runner wall394.254940631秒、parent process CPU379.704082643秒、parent peak RSS461880 KiB。CPU/RSSは親processの値であり、子physical workerを含む総CPU/最大メモリではない。CPU値はprocess累積値で、wallとの差からspeedupを主張しない。

[Postflight](../../artifacts/research_c_u_step_control/2026-11-03/execution_postflight.json)でSOURCE212/input9と契約・SOURCE copyのbyte一致、旧tracked SOURCE・raw結果・科学契約・STOP等の差分0、Calls journalとSTOPの一致、各call上限を確認した。旧raw全件のhash棚卸しを繰り返したとは主張しない。新run01原本1394ファイル／69,776,718 bytes、最大単一file24,777,359 bytesを保全する。

科学停止後は保存JSON/JSONLの報告集計・byte hash・Git公開だけで、新たな実H4数値query・NPZ/QPY数値decodeは0。これはローカル実行後にcommitする証拠であり、immutable CI・外部再現ではない。準備時の人工PASSも科学的収束や資源優位へ昇格しない。

## 6. 今回分かったこと・未解決事項・次の担当

同じ保存update005を出発点とするfresh評価で、有限Armijo backtracking・α引継ぎ・recenterを使い、既存の数値/物理資格を維持して64回の更新を進められた。旧`FINITE_STEP_GRID_NO_ACCEPT`地点を、資格を削除せずに数値的に前進できたことを示す。

一方、停止理由は上限到達であり、U一次条件は未達。今回の結果だけでは、旧停止の唯一の原因がF/‖g‖²尺度だったこと、真のreduced gradientの精度、十分な収束、最適化済みRC2との公平な勝敗は認定できない。F/λの低下を新J/UのPF/RTE・shots・native価格・量子総資源の改善へ推定しない。これらはすべて未評価/nullである。

次は**ユーザー／研究C**が、同じ目的での継続・最適化方法の再検討・下流評価の必要性を判断する。追加実H4計算や重要GPT科学レビューはそれぞれ別の明示許可が必要で、今回自動開始しない。**統合担当Codex**へは本報告・machine-readable集計を引き渡し、Codex-Aのr系列との接続と共有研究概要・manifest・索引・dated noteの更新を依頼する。前の指示に従い、本branchでは共有ファイルを同時編集しない。

Branch `codex/research-c-u-step-control-execution-20261103`。科学SOURCE/契約は固定し、新結果/raw/Calls/STOPと本報告をcommit・push後、remote SHA/tree・公開本文・成果物の整合を照合する。**実行範囲を完了しmandatory STOP。**
