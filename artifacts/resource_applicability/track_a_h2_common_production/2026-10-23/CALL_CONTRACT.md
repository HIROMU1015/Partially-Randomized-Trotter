# 科学callの単位・exact分岐・STOP

[call_recipes.json](call_recipes.json)はrunnerの `counts()` から生成する。全3分岐のcontrol-flow traceがmock backendで予算を過不足なく消費することを人工検証した。現時点で実grantはない。

| kind | 計数単位・内部との対応 |
|---|---|
| bootstrap | 科学backend import集合へ入る1回。最初のimport失敗でも1消費 |
| setup | 保存snapshot restore、H/state/basis/occupation representation構築1回。新SCF/DF生成なし |
| prepare | prefixごと共通prepare1回。prefix3/1の2回。Gaussian内部gate数はoperation receiptで記録 |
| diagnose | candidateごとの診断join1回。内部path_signal2とreference_rebind1を別記 |
| path_signal | candidate/pathごとのpropagation全体1回。各expm/spectral/Taylor matvecを独立実験としては数えず、内部work/traceへ実測記録 |
| reference_rebind | candidateのcorrected complex mean−旧FP scalarを計算する1回。新reference生成なし |
| native_validation | candidateのnative/control/compiled gates全体1回。以下の内部予算も消費 |
| compose | evolution compose1回。Full2候補のpieceは4+6、PRも4+6 |
| lowering | compose内のS2 native builder1回。両候補合計10 |
| wrapper | measurement付きaxis wrapper build1回。2候補×2軸=4 |
| compile | `transpile_and_measure_cost` 1回。wrapperごとに1回 |
| transpile | cost API内のguarded Qiskit transpile1回。compileごとに1回 |
| simulate | evolution low/high各1＋各axis before/after各1=6/candidate、計12 |
| request_factory | finite-RTE request生成factory entry1回。PR任意分岐の4+6=10 |
| rte_event | requestへ登録するevent数。PR任意分岐のみ128+64=192。乱数variatesの数ではない |
| occupation_trajectory | 同じPR request列に対する独立occupation作用1回。任意分岐2回 |
| allowance | candidateの診断からu/marginを生成する1回 |
| accounting | candidateのshot/N/Gを計算する1回。ineligibleならNoneを記録 |
| ranking | 固定4結果の条件付きA/B/逆転候補比較1回。科学GO・因果認定なし |

compilerのBasisTranslator pass数、Gaussian gate数、BLAS内部operation、乱数generator内部variate数は別の科学callとして予約していない。これらは1つの固定kernel内の処理で、操作列/guard/work receiptに保存する。1高レベルentryへ内部呼出しを隠す意味ではなく、上表で契約の計数境界を固定する。kindごとの枠は失敗時も消費し、どの階層もretryしない。

## 分岐

- **DIAGNOSTICS**：4候補の両path診断とreference再結合、共通prep receiptを生成。sampling/compile/shot算術なし。共通入力・reference政策・runtime seal・明示承認で実行する候補。
- **COMPARE_REUSE_PR**：上記にFull2候補のnative/compiled correctnessとRZ計測を追加。PRは33/34 paired task単価を証拠照合して再利用。4 allowance、4 accounting、1 rankingまで一括候補。完全PR C witnessがなければ開始前にBLOCKED。
- **PR_TWO_NATIVE_CONTROLS**：別の任意契約。2固定PRの保存seed列からrequestを再構成し、代表2軌道だけnative/compile gateを評価。診断や費用再会計なし。全cohortのC同一性をこの2軌道で証明しない。

新reference分岐と影響PR軸の再compile分岐は未承認入力不足でBLOCKEDである。新reference再利用不可という科学判断、または具体的なaffected event/axis列が揃った後に別contractを作る。件数nullを許すlive budget、未知の追加call用reserve、全67 task自動再取得は実装しない。

## 実行前contract

`contract_TEMPLATE_BLOCKED.json` は見本であり実行用ではない。source_commitは固定するが、使用file list/canonical root/runtime/identity proof/run_rootは未充足である。preflightはこれらが揃わなければ拒否する。

使用fileにはSHA256/role、SOURCE fileにはrunner/acquisition/providerのroot・固定commit・相対pathを記録する。snapshot/H/stateのSHAは保存H2に固定。old_reference入力は前段固定excerpt schemaに結合する。参考値のuncertaintyを0で補完しない。PR C witnessは16 identity field、各taskのaccepted receiptとpaired単価、cohort/correctness/eventsの封印proofを要求する。

old_referenceファイルはcc7404ed…の保存excerpt全byteとの一致も検査する。PR単価は同commitのtask_acceptance_67.jsonからcandidate/sample/axisを対応させ、witnessのtask列と厳密一致を要求する。このschemaのreceipt_sha256は受領result projectionのcanonical JSON digestであり、元実行が当時封印した個別receipt hashへの格上げではない。preflightでproofの存在・封印・保存単価対応を確認し、実行ではprepare直後に新prep/runtime等のidentityを確認してから診断・Full compileへ進む。

明示採択と科学実行承認が外部receiptとして揃ってからのみexecuteを使う。今回このCLIを実保存値でexecuteしていない。mock testのgrant/ledger/markerは人工temp内だけで、本番grantを発行・消費したものではない。
