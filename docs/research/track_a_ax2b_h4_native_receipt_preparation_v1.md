# Track A AX-2B：H4-P専用runner・実行前固定 v1

2026-10-10 JST。利用者の「作業を進めて」は、直前に提示したH4-Pの**実装・合成テスト・実行前固定**に適用する。
[GPT独立レビュー](track_a_ax2b_h4_post_independent_scientific_review_2026-10-10.md) §21と
[metadata preflight v3](track_a_ax2b_h4_prelaunch_contract_v3.md)の次の準備である。
今回、実保存H4入力の数値load・native分子準備・signal・sampling・wrapper build・compileを開始していない。
`H4_NATIVE_RECEIPT_NOT_AUTHORIZED` / `H4_LIMITED_NOT_AUTHORIZED` / `H6_NOT_AUTHORIZED` / `DRAFT_NOT_AUTHORIZATION`、mandatory STOP。

## 実装した範囲

| 入口 | 役割 |
|---|---|
| [専用runner](../../scripts/resource_applicability/run_track_a_ax2b_h4_native_receipt_v1.py) | defaultは未認可draftの出力。metadata bindingはGit blob・ファイルhash・Unicode metadataのみ。実行は別のH4-P grant必須 |
| [contract/controller](../../src/trottertracks/resource_applicability/ax2b_h4_native_receipt_v1.py) | 保存入力1 load・登録8 native preparation call、exact coverage・structural boundsのreceipt。実numerical importは認可後のProductionPort内のみ |
| [親watchdog](../../src/trottertracks/resource_applicability/ax2b_h4_native_watchdog_v1.py) | exec worker一つ、全cellとstartup/imports共通wall、log/output監視、process group停止、terminal/receipt全体照合 |
| [専用tests](../../tests/tracks/resource_applicability/test_ax2b_h4_native_receipt_v1.py) | injected metadata/native ports・dummy processのみ。実分子I/Oと数値backendを使わない |

既存sourceは編集せず、`_load_snapshot_once`、`_prepare` / `_prepare_discard`、`actual_bounds` / `check_bounds`、
`block_instruction_bound`、既存budget/writer/worker limitsを再利用する。
`MolecularPort.setup()`はreference計算を含むため呼ばない。signal、MP、matvec probe、sampling、wrapper builder、transpiler、solverの呼出しを実装経路へ入れない。
native preparationはorbital decomposition・basis operation/Gate object・symbolic tail specを作る。これは未来のH4-P grantの対象であり、今回実行したmetadata照合とは区別する。

## 固定scopeと上限

入力は[凍結snapshot](../../artifacts/pr2_s0_s1_validation/2026-09-28/h4_1p00_rank12_development_v1.npz)：linear H4、1.00 Å、STO-3G、DF rank12、generation-prefix、8 qubits、Nα=Nβ=2・sector36。
SHA-256 `3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a`、T=0.8。
targetは保存binary64 Hamiltonianと保存vectorを数学的に正規化した指定state。Hamiltonian/state生成や真の基底状態への差替えなし。

| cell | method / PF | prefix | q | R / K |
|---|---|---:|---:|---|
| H4_B1_S2_q1 | B1 / S2 | 12 | 1 | — |
| H4_B1_S2_q4 | B1 / S2 | 12 | 4 | — |
| H4_B0_q4 | B0 / S2 | 6 | 4 | — |
| H4_B2_K2 | B2 / S2 | 6 | 4 | 8 / 2 |
| H4_B2_K4 | B2 / S2 | 6 | 4 | 8 / 4 |
| H4_B3_K6 | B3 / S2 | 0 | 4 | 8 / 6 |
| H4_B1_S4_q1 | B1 / global S4 | 12 | 1 | — |
| H4_B1_S4_q4 | B1 / global S4 | 12 | 4 | — |

同じprefixでも登録cellごとにprepareする。8 prepared objectを同時保持せず、一cellのreceiptを保存して次へ進む。
各blockのprimitive ID、basis ID/hash、basis operation metadata、runtime operation数、diagonal/λのhexを記録する。
tailの完全なcomponent specとdigest、preparation/partition/Hamiltonian hash、identity extraction・λRを記録する。
各cellで凍結`actual_bounds`を用い、そのrowを結合する。登録primitive ID・binary64 timeの**集合全体**とscheduleを旧v3 JSONに照合する。
179組・537 probeは未来のH4限定検証の予定coverageであり、H4-P中のprobe作用数は0。

| 制約 | H4-P固定値 |
|---|---|
| CPU / BLAS / worker | CPU3 / 1 thread / 1 worker。manifestへの指定であり、CPU予約や実行ではない |
| wall / AS | 900秒（startup/imports込み）/ 8 GiB address space。RSS測定や物理メモリ予約ではない |
| output / log / diagnostics | 16 MiB / 64 KiB / 128。小さいterminal用領域を予約 |
| 呼出し | snapshot load 1、native preparation 8。counterを呼出し前に消費 |
| 後続H4の構造上界gate | primitive 2000、untranspiled instructions 1,000,000。今回実probe/buildなし |
| その他 | signal/reference/MP/probe/sampling/wrapper build/compile/solver/input generation/H6/H8/GPU 0。retry/resumeなし |
| 未来の専用output | `artifacts/resource_applicability/track_a_ax2b_h4_native_receipt/2026-10-10/launch_v1`。今回は作成しない |

block instruction式・tail最大Taylor order式は既存定義を変更しない。
この上界は構造上界で、実コンパイル費用、expanded gate数、compiler RAM、物理資源、精度の認定ではない。

## 認可・失敗・終了

専用grant schemaは `track_a_ax2b_h4_native_authorization_v1`、kindは `H4_NATIVE_RECEIPT`。
H4_LIMITED/H6のgrantとは相互代用不可。別の明示的user認可、manifest semantic digestとexact file SHA-256、grant file SHA-256、CPU、exclusive outputを必要とする。
今回approved grantを作成していない。grant不在・誤scopeはsource/input/environment I/Oとnumerical importより前に拒否する。
workerは親のlaunch binding・一回claimを確認し、affinity/AS/file/thread上限設定後に初めてnumerical dependencyをimportする。

source/input/env/static receipt・cell/time集合・同じprimitiveのbasis metadata不一致、非finite/JSON不正、instruction/probe cap超過、wall/log/output/AS超過でSTOP。
source/input/envは成功時に再確認する。親はworker exit 0だけで成功扱いにせず、1 load/8 preparation・8 cell bytes/digest・receipt binding・全禁止call0・終了flagsまで検査する。
失敗・部分成果を保存し、retry・resume・cap自動増量・結果依存の間引きを行わない。
親は正常終了でもworkerのprocess groupを終了させる。terminalが欠落・不正でも親のSTOP記録を残す。

H4-P acquisition planだけをmetadataでsealできる。native instruction値はまだ未知であり、**H4 science manifestは未seal・未認可**のままである。
将来の成功receiptのcoverage.sealed=trueもreceipt固定を表す。H4 science manifestへの組込み・seal・launchは別作業と別認可で、runnerが自動的に行う処理ではない。
N/G=null、accuracy UNDETERMINED、numerical allowance未認定、H6未認可、mandatory STOPを全成功receiptでも維持する。

## 検査と来歴

[48 tests receipt](../../artifacts/resource_applicability/track_a_ax2b_h4_native_receipt_preparation/2026-10-10/synthetic_tests_v1.junit.xml)はlocal engineering evidence。
未認可・別scope・未seal・cell/cap改変・CPU/output/retry gate、依存関係変更、one-shot claim、呼出しcounter、部分保存、非finite・coverage・構造上界超過を合成入力で検査した。
dummy childによるwall/log超過・異常終了・terminal欠落、親の全receipt照合、CLIの認可前numerical import禁止、file pin、worker limits/claim/import順も検査した。
CI、実分子correctness、上界の実値取得、独立再現の証拠ではない。

新sourceを既存branchへcommitした後、そのexact Git blobを別の
[H4-P準備manifest](../../artifacts/resource_applicability/track_a_ax2b_h4_native_receipt_preparation/2026-10-10/native_preparation_manifest_v1.json)と
[preparation freeze](../../artifacts/resource_applicability/track_a_ax2b_h4_native_receipt_preparation/2026-10-10/preparation_source_freeze_v1.json)へ結合する。
freezeは**未実行の準備source**であり、旧H4 v5 execution freezeを置換しない。
新module追加でlibrary closureが広がるため、旧167件のdraftを現在のsourceで実行することも認可しない。将来のscience seal時にsource bindingの更新が必要。
[準備inventory](../../artifacts/resource_applicability/track_a_ax2b_h4_native_receipt_preparation/2026-10-10/preparation_inventory_v1.json)は科学manifestとは別にsource/tests/metadata/docsを登録する。
旧178 preparation source、旧実行freeze・結果・科学manifest、既存dirty/untracked、Track Bを保全する。

次は、この固定manifestに対する**別の明示H4-P実行認可**がある場合に限り、receipt取得一回を実行してSTOPする。
今回の作業は実装・実行前固定で終了する。科学的GO/STOPやH6/H8への進行は行わない。

## 固定完了の記録

準備source commitは `7877131e764f5ce8b296cbbfa9ff3e859d04a8f2`。H4-P runner closure170件、継承178件＋v3 metadata source/tests2件＋今回4件のpreparation freeze184件をlocal/Git blobまで照合した。
manifestはCPU3、Python3.11.0rc1・numpy1.26.4/scipy1.14.1/mpmath1.3.0/qiskit1.3.0/openfermion1.6.1、保存入力hash、旧v3 static receiptを固定した。
[metadata監査](../../artifacts/resource_applicability/track_a_ax2b_h4_native_receipt_preparation/2026-10-10/metadata_binding_audit_v1.json)に全照合を記録した。数値importを禁止した状態でmetadata bindingのみ実行した。
[保全記録](../../artifacts/resource_applicability/track_a_ax2b_h4_native_receipt_preparation/2026-10-10/preservation_audit_v1.json)のとおり既存2,605ファイルとroot側の4 review文書を保全する。
H4-P実行output・approved grantは不存在。instruction上界の実値は未取得。次の実行認可前でSTOPする。
