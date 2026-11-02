# Track A AX-2B：H4限定科学検証の一回実行・coverage interface STOP

2026-10-10 JST。利用者の「作業を進めて」を、直前に示した**固定H4限定検証の別認可→一回実行→証拠公開→mandatory STOP**へ適用した。
[seal契約](track_a_ax2b_h4_limited_seal_v1.md)のsource/input/plan/capsを変更せず、独立した[別認可](../../artifacts/resource_applicability/track_a_ax2b_h4_limited_execution/2026-10-10/authorization_v1.json)を先にcommit/pushした。
実行は`H4_LIMITED_STOP`。worker理由は`ValueError:ACTUAL_COVERAGE_CHANGED`、親理由は`WORKER_FAILED_OR_INCOMPLETE`。
登録数値検証は未到達で、H4 scientific PASS、総u、精度適格性、shot/総費用、H6 GOは得られていない。

## 固定条件と来歴

linear H4 1.00 Å、STO-3G、legacy DF rank12・generation-prefix L_D=0/6/12、8 system qubits、Nα=Nβ=2・sector36、T=0.8。
q1/q4・δ0.8/0.2、B2/B3 R8・r2・K2/4/6、B1 global S2/S4の旧8 cellを維持。
targetは保存binary64 DF Hamiltonianと数学的に正規化した指定保存state。新Hamiltonian/state生成・再最適化はしない。

| 対象 | 固定ref・保存証拠 |
|---|---|
| 科学source | `61091c2cb00eb871d7a692b125219d34d99cc923`、170件closure。凍結[backend](../../src/trottertracks/resource_applicability/ax2b_molecular_ports_v2.py)、[runner](../../scripts/resource_applicability/run_track_a_ax2b_bound_v2.py)を変更しない |
| 準備freeze | [189件source freeze](../../artifacts/resource_applicability/track_a_ax2b_h4_limited_seal/2026-10-10/preparation_source_freeze_v1.json)、実行前後のlocal/Git bytes一致 |
| sealed manifest | 保存commit `c56ebc433b7ae14df3f50fec3d0b95e04c318208`。[正本](../../artifacts/resource_applicability/track_a_ax2b_h4_limited_seal/2026-10-10/sealed_preparation_manifest_v1.json)とrun内frozen copy一致、SHA-256 `679511b859630992870be50e2ec4d15fa24c4794ddec2bc8b6d127f3e65a58c2` |
| 別認可・実行base | `2f4f536ffff62d9da4218210de56d7042785f2ff`。認可SHA-256 `6c6f66ead40e835b1fe30dc662cefdc8d052d262d8b03ae2669f4f8ec8c63324` |
| exact command | [launch command](../../artifacts/resource_applicability/track_a_ax2b_h4_limited_execution/2026-10-10/launch_command_v1.json)。manifest digest・authorization file SHA・専用output・CPU3を結合 |
| 実行前・後監査 | [prelaunch audit](../../artifacts/resource_applicability/track_a_ax2b_h4_limited_execution/2026-10-10/prelaunch_audit_v1.json)、[execution audit](../../artifacts/resource_applicability/track_a_ax2b_h4_limited_execution/2026-10-10/execution_audit_v1.json) |
| 原terminal | [親](../../artifacts/resource_applicability/track_a_ax2b_h4_limited_validation/2026-10-10/launch_v1/terminal_status.json)、[worker](../../artifacts/resource_applicability/track_a_ax2b_h4_limited_validation/2026-10-10/launch_v1/worker_terminal.json)。失敗を成功へ変更しない |
| 保存監査 | [saved STOP audit](../../artifacts/resource_applicability/track_a_ax2b_h4_limited_execution/2026-10-10/saved_stop_audit_v1.json)、[stdlib-only監査helper](../../artifacts/resource_applicability/track_a_ax2b_h4_limited_execution/2026-10-10/audit_saved_stop_v1.py) |
| evidence inventory | [実行inventory](../../artifacts/resource_applicability/track_a_ax2b_h4_limited_execution/2026-10-10/execution_inventory_v1.json)。旧scientific manifest/statusと別に登録 |

CPU3・science worker1・BLAS1、Python3.11.0rc1、numpy1.26.4/scipy1.14.1/mpmath1.3.0/qiskit1.3.0/openfermion1.6.1。
phase wall900/1800/300秒、total3000秒、AS8GiB、output512MiB、log64KiB、diagnostics1024など旧capsを維持。
ASはaddress-space上限でありRSSではない。独立CPU/RLIMIT telemetry・peak RSSは保存されておらずnullとする。
旧H4 v5 Python3.11.1、H4-P原親STOP/新再監査PASSの来歴は保存し、今回実行と混同しない。

## 実行で記録されたことと欠測

| 項目 | 原記録 |
|---|---|
| launch数 | 一回。別認可は消費済み。retry/resume0 |
| 終了 | parent/workerともH4_LIMITED_STOP、worker exit code1 |
| 親wall | 2.045328374952078秒 |
| output | 8 file・126,556 bytes、worker.log345 bytes |
| phase | input_referenceに入った。correctness/wrapper_cost phaseへ未到達 |
| correctness | 0/8完了。MP80/120桁記録0、4代表event記録0 |
| 観測call | reference_matvec/primitive/control_probe/trajectory/occurrence/compileすべて0 |
| load/native準備 | coverage比較行へ到達した凍結sourceの制御フローから、保存snapshot load1・8 cell準備の完了を推論。独立counterは未保存 |
| 科学output | actual_coverage/input_reference/primitive_validation、8 correctness、16 MP、4 event記録は欠測 |

worker logはMatplotlibの書込不可config directoryに対する/tmp fallback警告であり、terminalに記録された失敗理由とは別である。
workerはinput load・basis/sector bridge・native準備・bounds計算/cap検査の後、reference構築より前のcoverage直接比較で例外を返した。
科学計算を一切しなかったstageとは記述しない。実行時のload/native準備は上記推論で、保存されたcall0と区別する。
actual_coverageは比較に通った後にwriterが保存するため、失敗したactual bounds全体は残らなかった。

## 保存metadataによる静的interface診断

[凍結schedule生成](../../src/trottertracks/resource_applicability/ax2b_h6_contract.py)はtupleを含むPythonリストを返す。
`MolecularPort.setup`は`manifest['coverage_binding']['expected_bounds'] != bounds`という直接比較を行う。
JSON保存/読込によりtupleはlistへ変わるため、同じprimitive index/timeを持つscheduleでもPython比較は不一致になる。

保存expected boundsに、凍結stdlib scheduleとASTから切り出した純粋な`canonical_cell`/`validation_times`だけを結合したmetadata診断では、925箇所のtuple/list型差があった。
Python比較は不一致、canonical JSON semantic digestは一致した。これは新native準備・signal/probe・数値backendの再実行ではない。
例：`$.cells[0].schedule.ordinary_one_outer_step[0]`はruntime scheduleではtuple、保存値ではlist。
wrapper上界は保存値を保持し、新しいprepared inputから再計算していない。

**この型差は現在のinterfaceを阻害するが、失敗時の全runtime値が保存expected boundsと一致した証明ではない。**
actual boundsが未保存のため、それ以外の差分の有無はこのrunから確定できない。`SAVED_STOP_AUDIT_PASS`は保存失敗記録と来歴の一致だけを表す。
seal時の35 synthetic testsは未認可gate/flags/合成boundsの検査で、実scheduleをJSON往復してsetupの直接比較へ渡す境界検査が欠けていた。この不足を記録する。
今回はsource/manifest/認可/失敗recordを修正せず、テストや科学runも再実行しない。

## 保全・停止と後続の候補

[保全監査](../../artifacts/resource_applicability/track_a_ax2b_h4_limited_execution/2026-10-10/preservation_audit_v1.json)で旧2,668 file、root側5 review、dirty/未追跡/Track B、旧科学source/freeze/結果/STOPを確認した。
raw8 file、別認可3記録、stdlib保存監査、報告・dated note・索引だけを公開する。`git add .`/force pushなし。
旧validation_manifest.jsonやseal状態を書き換えず、今回stageの別inventoryでH4_LIMITED_STOPを追跡する。

次の保全的な候補は、**coverageのserialization境界修正と合成round-trip検査**。
値・順序・index/time・instruction上界・全集合・capsを維持した正規化比較を新source versionで実装し、float差やcoverage欠落を許容する比較へ緩めない。
型以外の差分を失敗前に小さいbounded recordへ保存する診断も検討対象。実装・新source/manifest固定は今回未実施。
新しい一回実行には別scope/manifest/source/outputと明示認可が必要。今回grantやlaunch_v1をretry/resumeに使わない。
純粋なpath/schema修正なら[独立レビュー](track_a_ax2b_h4_post_independent_scientific_review_2026-10-10.md) §21のCodex範囲。意味論・target/state・独立性の変更や正しさの矛盾ならGPTへ戻す。

現在はmandatory STOP。N/G=null、accuracy UNDETERMINED、総数値allowance未認定。
`H6_NOT_AUTHORIZED` / `DRAFT_NOT_AUTHORIZATION` / `next_stage_authorized=false`を維持する。
H6入力生成・pilot、本検証、H8/GPUや科学GO/STOPへ進まない。
