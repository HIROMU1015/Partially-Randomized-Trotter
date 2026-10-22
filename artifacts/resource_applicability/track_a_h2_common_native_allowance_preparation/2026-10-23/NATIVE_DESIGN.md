# 共通nativeの提案と新旧差

提案policy：`H2_COMMON_ADJACENT_QR_SIGNED_PIECES_V1_PROPOSED`。正式採択前である。
固定kernelのfile/hash/ASTは[kernel_bindings.json](kernel_bindings.json)、実装用引数の静的recipeは[deferred_native_recipes_4.json](deferred_native_recipes_4.json)。recipeは呼出しを実行しない。

## 共通化の境界

| 項目 | 共通にする条件 | 保持する方式差 |
|---|---|---|
| preparation | new2 adjacent complex QR Gaussian、既存_prepare/DF fragment | Full prefix3、PR prefix1。tailの有無 |
| schedule | 固定h46_m5_plan.scheduleと浮動小数演算順 | S2/Y4、q4/q2、負時間piece、PR outer32 |
| native lowering | composeでpieceごと既存S2を使用、scalar=0 | Full deterministic、PR controlled RTE tail |
| scalar | 全pieceの後、ancillaに一回 | Full ham.constant、PR ham.constant＋extracted tail identity |
| control/wrapper | symmetric_directional、system4/ancilla4、cos/sin測定wrapper | method固有の中身のみ |
| compiler | Qiskit1.3.0、rz/sx/x/cx、level1、seed17、phase guard | 実回路・native fingerprintは候補ごと |

「new2」はGaussian adapterの名前であり、PF次数を変更する意味ではない。新PR経路を第一候補にする理由は、SOURCEを追跡できるadjacent操作列、明示的な再構成check、既存PRとFullで使えるprepare/compose境界、再利用可能なreceiptである。保存費用順位に基づく選択ではない。

## Gaussian操作列

旧provider `217c971d…` の `df_trotter/ops.py::U_to_qiskit_ops_jw(U)` は、通常 `BogoliubovTransform(U.T)` をimportし、失敗時にはdense Gaussian fallbackを選ぶ。旧SOURCEを「常にdense」と断定しない。旧実行時に選ばれた操作列を、preparation hashだけから逆算できない。

新SOURCE `4ec26921…:scripts/h46_new2_native.py::gaussian_operations` は、列ごとに下から隣接rowをcomplex QRで消去する。対角位相の非zero PhaseGateをmode順に置き、消去操作を逆順に戻す。各pairはXXPlusYY(theta,beta)、upperへのPhase(alpha)、lowerへのPhase(-alpha)の順である。小さい角を独自に刈り込まず、adjacent wire順を保持する。元kernelにはunitarity/QR residual/pair reconstruction/full reconstructionの1e-12 checkがあるが、今回はそのkernel自体を実行していない。

`prepare(ham,prefix)` は私有process内で `ops.U_to_qiskit_ops_jw` を差し替え、元 `_prepare` を呼び、finallyで戻す。将来並列化するときは同一Python processで異なるGaussian政策を同時にpatchしない。分離workerごとのSOURCE固定を使う。元providerファイルを書き換えない。

保存hashは旧Full prefix3 `8e32e931047087a4a9c1f613d406b6fe353c0aa4601b7e5328750ab6ab2c9a42`、新full prefix3 `98c1beb120c22077b409550a4d61c31638b9a52637246fae4cb1ac55a776f1d9`、新cut prefix1 `85b8243d752d3390e214b429a80e3899f0cc8b0166e0958fc1d73a930a77597c`。方式・prefixを無視したhash一致を求めない。将来の確認単位は同一rotation/basisに対する操作列、各DF block、同一prefix全prep、最終controlled作用に分ける。

## signed時間・phase・wrapper

composeは元scheduleのpiece順で既存S2を繰り返す。Y4係数の再評価、時間の均一化、積順変更、位相fitをしない。Fullはdeterministic block全体、PRはprefixと該当requestのtailを使用する。各pieceへのscalarは0で、最後だけ `qc.p(-scalar*.8,n)` を置く。PRのtail scalarをFullへ加えない。負時間とRTE allocationは元SOURCEのまま保持する。

Fullにも `PER_PIECE, r_value=1` でraw scheduleを作る元compose仕様がある。そのrte_stepsは未使用placeholderで、RNG/requestを起動しない。mock schemaのFull rte_steps=0は「RTE作用なし」を表す意味用metadataであり、raw SOURCE scheduleを改変する仕様ではない。

PRはK4、controlled request、ancilla4、preparation hash、signed time、piece順、unconditional lawを一致させる。条件付きfixtureを費用sampleへ混ぜない。保存eventの再構成で足りるか、追加samplingが必要かはidentity witnessで決める。今回はどちらも実行していない。

最終cos/sin full wrapperにはmeasurementを含む。P=0は状態準備費用の会計条件であり、保存stateを別stateに替える条件ではない。

## compiler/guardとcorrectness

新共通contextではphase guardを**FullとPR両方**のcost transpileへ私有process内で適用する。historical PR用 `h2_outer32_runtime.context` はprefix1を固定しているため、そのcontext全体をFullへ流用しない。prepare/composeと個別kernelを再利用する。実行条件の新規結合とloaded runtime封印は未接続である。

提案gateはnative low/leakage/occupation-vector/wrapper誤差がstrict <1e-9、保存stateでのcompiled-wrapper vector差がstrict <1e-7。phase alignment/state fitなし。結果identityはimplementation、Hamiltonian、state、preparation、basis、operation stream、schedule、wrapper、compiler SOURCE/settings、runtime closure、event lawのSHA256を含む。各gateには誤差とevidence hashを要求する。mock validatorは実科学receiptを受け入れず、人工PASSからscience_GOを発行しない。

これらは提案閾値であり、shot allowanceや全state operator証明ではない。nativeの操作等価性、保存state上の作用一致、compiled回路の同一性は別の判定にする。

旧Fullにはcontrol/measurement/単価receiptがあるが、受領projectionと今回調べた原本workerにはcompiled-vector受入記録・保存QPYを確認できていない。旧nativeのcompiled correctnessと新共通nativeのcorrectnessを混同しない。

## 費用差の解釈

Gaussian分解、scalar処理、guard/pass、wrapper、signed-piece境界、実行runtimeの差は単価へ影響し得る。一方、Full全DF因子対PR tail sample、K4/outer32、q、Y4負時間は残すべきアルゴリズム差である。共通nativeにした際のC・signal・費用順位は今回評価していない。実装差を逆転の原因と断定しない。
