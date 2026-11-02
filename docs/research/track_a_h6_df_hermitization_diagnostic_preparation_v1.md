# Track A H6：DF Hermitization STOP後の診断準備 v1

2026-10-10 JST。**既存証拠の監査と、保存integrals限定の診断source・plan・sealを準備した。実DF decompositionと診断runnerは起動していない。**
`H6_DF_DIAGNOSTIC_NOT_AUTHORIZED` / `H6_NOT_AUTHORIZED` / `DRAFT_NOT_AUTHORIZATION`。
sealは条件固定だけで、別schemaの新grantを作成していない。mandatory STOP、科学的判断はGPTへ戻す。

## 既存証拠と回収範囲

元runは[H6入力生成STOP](track_a_ax2b_h6_input_generation_stop_v1.md)に記録。
linear H6 / 1.00 Å / STO-3G、12 spin orbitals、DF tol-only1e-8、Hermitization許容1e-10。
input-generation source `67312f3195aede26e8ba4f5727d89c236772f82e`、結果 `df3b1f694ceb72a198ab6e3e89706b239e56e1da`。
SCF/integralsは完了、DF adapterはfragment_15で例外。DF receipt/stateは未生成、PF prefix/delta/signal比較なし。

[検索記録](../../artifacts/resource_applicability/track_a_ax2b_h6_df_diagnostic_preparation/2026-10-10/existing_evidence_search_v1.json)では登録worktreeのartifact 48794 path、配列ファイル116件をpaths/bytes/NPZ member名で検査した。
他のDF snapshot候補112件を区別したが、元run/inputの分解rawとして同定できるものはない。
同じintegralsのfile SHA一致は元runの1件だけ。worker/parent terminal・原inventory以外に同じSTOPの数値を保存した資料は見つからない。
ローカル登録worktreeのartifact範囲の確認であり、別server・外部storage・過去process memoryの全検索ではない。
元sourceでは全fragment検査後にmetadataを作るため、例外でreturned raw/lambda/rank/切断値/補正が未保存のまま終わる。
**旧runの失敗fragmentの数値は回収できていない。新しい診断出力を旧raw bytesの復元と呼ばない。**

[入力照合](../../artifacts/resource_applicability/track_a_ax2b_h6_df_diagnostic_preparation/2026-10-10/input_identity_audit_v1.json)は元raw20件と元source183件を、結果/実行source commit・local bytesで確認。
[保存integrals](../../artifacts/resource_applicability/track_a_ax2b_h6_input_generation/2026-10-10/launch_v1/integrals.npz)のSHA-256は`edd0a618f86011757cacae481eff44dc637c11a3f64c55b0cbfb7ffbe637e51d`。
6-array NPZのfile/data SHA・dtype・shapeと、receipt・元manifest・worker STOPを結合した。NumPyで実入力をdecodeしていない。
根本原因、actual rank、lambda_15、g_15、差、他fragmentの違反、DF representation整合性は未確定。

## 診断の固定内容

同じ保存integralsからone/two-bodyだけをdecodeし、`two.copy()`を既存OpenFermionへ一回渡す。
**明示kwargsは`{"truncation_threshold":1e-8}`のみ**。final_rankを渡さず、分子config・rank fallback・再SCFを使わない。
implicit defaults final_rank=None / spin_basis=Trueは別欄に記録し、kwargsに追加したと表現しない。
returned orderを保持し、係数cutoff0、fragment削除なし。失敗fragmentだけでなく全fragmentを記録する。

| 保存ファイル | 配列・記録と役割 |
|---|---|
| raw_decomposition.npz | lambdas_raw[L]、g_matrices_raw[L,12,12]、one_body_correction_raw[12,12]、truncation_value_raw scalar。returned dtype/valueを保持し、元runのrawと同一とは主張しない |
| raw_decomposition_receipt.json | file/data hashes・dtype・shape・完全な明示kwargs。Hermitizationや数値受理判定より先にrawを保存 |
| hypothetical_hermitization.npz | gH=(g+g†)/2、one+correctionとそのHermitian projection、chemist tensorとplain-square再構成、normal-ordered係数。**診断専用でH6入力ではない** |
| diagnostic_summary.json | 全fragmentのlambda/abs_lambda、raw norm、Hermiticity defect、元検査の変更量・relative値、前後hash、違反index一覧、index15の全行、correction/切断値・整合性量 |
| runtime_environment.json / decomposition_call.json | Python/packages・installed helper/config hash、NumPy BLAS config、thread env・CPU affinity、呼出したmodule/function・kwargs・call前tensor hash |
| input_decode_receipt.json / frozen_diagnostic.json / grant copies | 入力からdecodeされた配列hash、新source/plan/input/env/CPU sealと新execution identity、original grant bytes |
| progress / log / worker/parent terminal | raw saved/summary saved/decomposer returned/attempted countersを分離。部分失敗・欠測も保存 |

rawが保存できる形式で返った後、要約・layout・finite・rank上限検査に失敗してもrawを保持する。
decomposer自体の例外では返されていない配列を補わない。partial bytesを残し、retry/resumeしない。
rank36は6 spatial orbitalからのdiagnostic配列/行数capであり、rank固定・調整・削除の政策ではない。超過時はraw保存後STOP。
Hermitization許容1e-10を保持して全違反を記録する。診断の完了は記録完了であり、元政策のPASSやH6 GOではない。

## lambda・非Hermiticity・表現整合性を分ける

各rowでlambda、||g||F、||g-g†||F、||gH-g||Fを別に記録する。
さらに|lambda|*(sum|g|)^2、weighted one/two-body係数norm、g→gHによるweighted係数差、weighted係数Hermiticity defectを別欄へ置く。
lambdaが小さい/ゼロでもgの偏差rowを削除しない。これらの係数normはHamiltonian作用素norm・signal error・厳密uのcertificateではない。

診断用のnormal-order identityは
`C_raw = correction + sum_l(lambda_l * g_l @ g_l)`、
`T_raw[p,q,r,s] = -sum_l(lambda_l * g_l[p,r] * g_l[q,s])`。
inputの`sum T[p,q,r,s] a†p a†q ar as`と比べる際は、両creation/annihilation indexをantisymmetrizeした
`A(T)=(T-T_pq-T_rs+T_pq_rs)/4`同士の差を記録する。**平方の積にconjugationを追加しない。**
one-body残差とquartic残差を分離し、raw gとhypothetical gHの係数差も分離する。
global projection差はgをprojectionした影響で、one+correctionのprojection偏差は別欄に記録する。

installed helperと同じchemist転置/spin抽出により36×36 interaction tensorを定義し、
transpose asymmetry/imaginary l1、plain-square再構成差、source再orderingとのcorrection差、
各returned eigenpairのresidual、cross-spin/alpha-beta block差も保存する。追加のeigendecompositionやFock/sector Hamiltonianを作らない。
normal-orderの符号とantisymmetrizationは独立2-mode Fockの**合成代数fixture**で検査済みだが、実H6での照合ではない。

[公式OpenFermion API](https://quantumai.google/reference/python/openfermion/circuits/low_rank_two_body_decomposition)を参照した。
実際の固定対象は[current installed 1.6.1 helper bytes](../../artifacts/resource_applicability/track_a_ax2b_h6_df_diagnostic_preparation/2026-10-10/installed_dependency_sources/openfermion_low_rank_1_6_1.txt)と[来歴/license](../../artifacts/resource_applicability/track_a_ax2b_h6_df_diagnostic_preparation/2026-10-10/installed_dependency_source_inventory_v1.json)。
現APIの文章をinstalled版の厳密な数値実装の代用にしない。
元runはhelper bytesの独立hashを記録していないため、今回観測した外部dependency hashを過去へ遡って認定しない。
新診断ではこのhelper/config SHAとversionを結果前にsealする。新旧rawの再現性は観測後に別に判断する。

## Source・seal・予算・合成検証

新診断source commit：`ff24de4bc410234472a416186b773fc7875ae373`。
準備資料commit：`905ee89eb814ca6b2be2c24727101ec991589f54`。
[GitHub再取得・bytes照合記録](../../artifacts/resource_applicability/track_a_ax2b_h6_df_diagnostic_preparation/2026-10-10/remote_verification_v1.json)は、この資料commitの187 science / 2 validation sources、旧source183件・旧raw20件、旧baseline4051件の保全とrepository相対リンクを確認した。
本記録とこの索引追記だけを後続commitで公開する。rootの別作業による新しい未追跡review文書1件を観測したが、本作業では変更・stageしていない。
[source freeze](../../artifacts/resource_applicability/track_a_ax2b_h6_df_diagnostic_preparation/2026-10-10/source_freeze_v1.json)はscience187件・validation2件。
[sealed preparation](../../artifacts/resource_applicability/track_a_ax2b_h6_df_diagnostic_preparation/2026-10-10/sealed_diagnostic_preparation_v1.json)と[preflight](../../artifacts/resource_applicability/track_a_ax2b_h6_df_diagnostic_preparation/2026-10-10/prelaunch_seal_audit_v1.json)で旧入力・新source・plan・環境・CPU2を固定した。
source/input/環境の変更は新sealを要し、sealとdraftだけでは実行できない。

| 古典資源cap | 固定値 |
|---|---:|
| saved input / decomposition / diagnostics | 60 / 300 / 120 秒 |
| total（startup/import含む） | 480秒＝8分 |
| AS / aggregate output / log | 8GiB / 32MiB / 64KiB |
| worker / BLAS / assigned logical CPU | 1 / 1 / 2 |
| saved input expanded / raw NPZ expanded | 16MiB / 4MiB |
| decomposer calls / fragment rows | 1 / 36 |
| progress / diagnostics records | 128 / 128 |
| molecular build / state solver / signal / sampling / compile | 全て0 |

予算は結果前の工学的capで、実DF診断の完走時間/RSS予測やhost専有保証ではない。
[環境/resource seal](../../artifacts/resource_applicability/track_a_ax2b_h6_df_diagnostic_preparation/2026-10-10/environment_seal_v1.json)は時点観測。GPU/再fit/最適化/DF政策変更なし。
[43 local synthetic tests](../../artifacts/resource_applicability/track_a_ax2b_h6_df_diagnostic_preparation/2026-10-10/synthetic_test_audit_v1.json) / [JUnit](../../artifacts/resource_applicability/track_a_ax2b_h6_df_diagnostic_preparation/2026-10-10/synthetic_test_results_v1.xml)を保存。
fake decomposer・temporary配列・dummy watchdogだけを用い、real low-rank/MolecularData/eighと実入力np.loadを禁止した。
診断CLIはASTでgate順序を確認しただけで、help/defaultも含めて起動していない。実DF call0。

実装入口：[runner](../../scripts/resource_applicability/run_track_a_h6_df_diagnostic_v1.py)、
[contract/gate](../../src/trottertracks/resource_applicability/ax2b_h6_df_diagnostic_contract_v1.py)、
[raw/stats port](../../src/trottertracks/resource_applicability/ax2b_h6_df_diagnostic_port_v1.py)、
[watchdog](../../src/trottertracks/resource_applicability/ax2b_h6_df_diagnostic_watchdog_v1.py)、
[saved bytes audit](../../src/trottertracks/resource_applicability/ax2b_h6_df_diagnostic_audit_v1.py) / [audit CLI](../../scripts/resource_applicability/audit_track_a_h6_df_diagnostic_v1.py)。
旧source/STOP/実integralsは変更しない。今回の[別inventory](../../artifacts/resource_applicability/track_a_ax2b_h6_df_diagnostic_preparation/2026-10-10/preparation_inventory_v1.json)を正本にする。

## 実行認可対象とGPT判断

認可対象は**このsealの保存integralsだけからDF診断一回（480秒/AS8GiB/output32MiB/CPU2）**。
`track_a_h6_df_diagnostic_authorization_v1`の別grantへ、ユーザーの明示指示・manifest digest・CPU・exclusive output・retry/resume falseを結合し、original bytes SHAをpinする。
元入力生成grantを再利用しない。future outputは`artifacts/resource_applicability/track_a_ax2b_h6_df_diagnostic/2026-10-10/launch_v1/`で、現在未作成。
認可後は新grant/source/planを公開・remote確認→一回診断→raw/全summary/partial監査/source identityを公開→mandatory STOP。
今回新grantの作成・診断runner起動・DF再実行を行っていない。

GPTが判断する論点は、raw分解の表現整合性、非Hermiticityとlambda-weighted影響の区別、
原Hermitization政策とDF targetを維持できるか、追加の証拠が必要か、である。
失敗の根本原因・DF表現の変更・閾値変更・fragment削除・H6 pilot GOをCodexは決定しない。
N/Gnull・u未認定・UNDETERMINED、H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATIONを維持して実行認可待ちでSTOP。
