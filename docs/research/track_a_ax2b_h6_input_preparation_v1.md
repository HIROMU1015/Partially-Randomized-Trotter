# Track A AX-2B：H6入力生成準備 v1

2026-10-10 JST。H4補完後の次工程として、**入力生成専用のsource・合成検証・結果前計画を固定した**。
`H6_INPUT_GENERATION_NOT_AUTHORIZED` / `H6_NOT_AUTHORIZED` / `DRAFT_NOT_AUTHORIZATION`。
CPU未割当、別grantなし、`execution_plan_sealed=false`。**実H6入力・状態・signalを生成せず、pilotも開始していない。**
本資料は実装・準備の報告であり、GPTの追加科学レビューやH6の実行認可ではない。mandatory STOP。

## 採用済み判断・既存資産との対応

[GPT独立レビュー](track_a_ax2b_h4_limited_stop_independent_review_2026-10-10.md) §§16,19の順序を維持する。
[H4補完結果](track_a_ax2b_h4_supplement_execution_v1.md)は旧6 correctness/12 MP＋新2 correctness/4 MP＋代表event4群を出典付きで保存済み。
旧STOPと未認定flagsを遡って変更しない。
[H6準備契約v2](track_a_ax2b_h6_pilot_preparation_contract_v2.md)のtol-only、7 cell/36 wrapper、H6 developmentを維持する。

現在のsourceにはH6 snapshot loader、400-sector reference、4096-vector native action、sampling/cost経路が既にある。
旧準備文書やdocstringの「molecular backend未実装」は当時の記録であり、現在の機能一覧ではない。
今回の新規部分は、分子入力取得→既存tol-only adapter→bounded状態生成→snapshot保存への接続、
入力生成専用のgrant/seal/CPU/source/environment gate、watchdog、保存bytes監査である。
既存H4/H6の凍結source・契約・結果を編集せず、新versionを追加した。

| 既存資産 | 再利用の範囲 |
|---|---|
| [tol-only DF adapter](../../src/trottertracks/resource_applicability/ax2b_h6_input.py) | `truncation_threshold=1e-8`のみ、rank fallbackなし、Hermitization前後の記録 |
| [bounded solver](../../src/trottertracks/resource_applicability/ax2b_h6_controller.py) | solver/rmatvecと事後residualで共通before-call counter |
| [DF sector/matrix-free library](../../src/trotterlib/df_hamiltonian.py) | spin-sector、HF初期occupation、numba/1 thread/chunk1作用 |
| [primitive sector検査](../../src/trottertracks/resource_applicability/ax2b_h4_science_v5.py) | 完全sector順序、Hermiticity、厳密cross-spin zeroを作用前に検査 |
| [state/sector hashとphase規約](../../src/trotterlib/pr2_s0_s1_validation.py) | largest-sector-amplitude real-positive規約を実際に適用して保存 |
| [既存H6 loader](../../src/trottertracks/resource_applicability/ax2b_molecular_ports_v3.py) | variable-rank NPZ layout、内部hash、sector/orderのroundtrip検査 |
| [atomic writer](../../src/trottertracks/resource_applicability/ax2b_supplement_records_v1.py) | exclusive JSON、診断/output上限、原子的公開 |

既存[bound launch v3](../../src/trottertracks/resource_applicability/ax2b_bound_launch_v3.py)は現在もH4-onlyで、H6 pilotを拒否する。
今回の入力生成runnerをH6 pilot gateの代用にしない。

## 結果前の入力・状態契約

linear H6 / 1.00 Å / STO-3G、6 Hをz=0,1,2,3,4,5 Åに置く。charge0、multiplicity1。
12 spin orbitals、even alpha / odd beta、Nα=Nβ=3、sector400。
HF molecular orbital basisのone/two-body integralsから指定binary64 H_DFと指定stateを保存する。
有限時間signal計算やenergy estimationの精度目標は、入力生成の完成条件に追加しない。

OpenFermion-PySCFの既存prepare/SCF/integral helperを再利用し、RHF conv_tol1e-9/max_cycle50、thread1を固定する。
SCF収束・6 orbital/6 electron・Angstromを確認し、不成立はSTOP。
`spinorb_from_spatial`後のtwo-bodyに1/2を掛けるInteractionOperator規約を維持する。
`run_pyscf`に伴う暗黙の`MolecularData.save`は呼ばず、scratchは新exclusive output内へ隔離する。
SCF checkpoint・temp/JIT/cacheもaggregate outputに含める。既存分子builder/configやsession cacheは変更しない。

decomposerへ渡すkwargsは`{"truncation_threshold":1e-8}`だけで、`final_rank`・config rankを渡さない。
actual rank L、truncation value、元integral hashes、one-body correction、coefficient generation order、
各Hermitization前後hash/差を保存する。後段cutoff0、Hermitization差の上限1e-10。
既存H6 loader/gateが受け入れるL=2..144を結果前のlayout scopeとし、外れた場合はrankを調整せずSTOPする。
Lを結果から恣意的に選ばず、H6 pilotのprefixは後段で既定のL / (L+1)//2 / 0に展開する。

完全spin-sectorと全one-body/fragmentのcross-spin係数の厳密zeroを確認してからsolver作用へ進む。
near-zeroをdiscardしてsectorに合わせたり、別sectorへ変更したりしない。
HF occupation初期vector、eigsh k1/whichSA/tol1e-12/maxiter1000/ncv40を固定する。
numba backend/thread1/chunk1を使い、利用不能・未収束ではfallbackせずSTOP。
returned sector vectorをbinary64で一回正規化し、既存phase規約を適用する。
同じbounded operatorでresidualを求め、`1e-9 + 1e-10*max(1,abs(E))`のengineering gateとsaved norm1e-12を確認する。
この小residualやSA solver完了をground-state証明・chemical accuracyと認定しない。
`ground_state_certified=false`、N/Gnull、UNDETERMINED、u未認定を維持する。

## 古典計算予算・保存形式

| 対象 | 固定提案上限 |
|---|---:|
| integrals phase（startup/import含む） | 900秒 |
| DF decomposition phase | 300秒 |
| state/snapshot phase（numba初期化・JIT含む） | 900秒 |
| total | 2100秒 |
| worker / BLAS / CPU affinity | 1 / 1 / 後段で明示割当 |
| AS / aggregate output / log | 8GiB / 128MiB / 64KiB |
| integral build / DF decomposition | 1 / 1 |
| solver matvec/rmatvec/residual共通 | 10000（per-actionも10000） |
| snapshot expanded / progress / diagnostics | 16MiB / 256 / 512 |
| trajectory / occurrence / circuit compile | 0 / 0 / 0 |

これは今回固定した入力生成用の提案予算で、旧H6 pilotの予算とは別。
CPU利用可能性・memory観測・host専有・実H6の完走時間は未確認。未測定のspeedupやRSSを報告しない。
static array layoutはfull vector65536 bytes、sector vector6400 bytes、最大rank144のg stack331776 bytes、
two-body tensor331776 bytes、solver400×40のcomplex workspace256000 bytes。
これらは配列の算術値であり、SCF/JIT/library一時領域やRSSの上界ではない。
4096×4096 fragment行列/固有vector cacheを入力生成で構築しない。GPUは使わない。

未来の科学outputは`artifacts/resource_applicability/track_a_ax2b_h6_input_generation/2026-10-10/launch_v1/`。
**現在は未作成**。保存予定は、exact integral arraysの`integrals.npz`、integral/DF receipt、
8-arrayの`h6_input_snapshot.npz`とsnapshot receipt、original grant bytes・canonical JSON copy、
source-bound manifest・claim、phase/progress、worker/parent terminal・log/scratch。
NPZはuncompressed・pickleなし・C-layout・固定dtype。既存H6 loaderのroundtripを保存前工程内で確認する。
失敗時のpending bytesやrawは残し、既存ファイルを上書きしない。

進捗はphase/record境界、最初と100回ごとのmatvec開始/完了で保存する。静的上界220件で256 cap内。
一回のSCF/decomposition/matvec内部の定周期heartbeatを保証しない。
強制終了後は最後のsnapshotのcounter・観測RSSだけを引用し、未観測の残りを補わない。
AS/壁時計/log/output/call上限、非finite、SCF/solver未収束、rank/sector/receipt/schema不成立はSTOP。
retry/resumeなし。結果後のtol/rank/initial vector/gate/予算変更による救済をしない。

## Source・検証・監査索引

source commit：`67312f3195aede26e8ba4f5727d89c236772f82e`。
準備資料保存commit：`0201dc84bce60a86a1796f9c143105162f9915be`。
[source freeze](../../artifacts/resource_applicability/track_a_ax2b_h6_input_preparation_v1/2026-10-10/source_freeze_v1.json)は
今回のexecution closure183件とvalidation3件をlocal/Git blob SHA-256で照合した。
旧H4 science180件と旧全結果/freezesのbytesを保全した。新closureを旧grant/manifestへ流用しない。

| 新source / artifact | 役割 |
|---|---|
| [input generation契約/gate](../../src/trottertracks/resource_applicability/ax2b_h6_input_generation_contract_v1.py) | stdlib-only、固定plan、独立schema、source/env/CPU/output/grant結合 |
| [port](../../src/trottertracks/resource_applicability/ax2b_h6_input_generation_port_v1.py) | 既存adapter/solver/loaderの接続、bounded snapshot |
| [watchdog/progress](../../src/trottertracks/resource_applicability/ax2b_h6_input_generation_watchdog_v1.py) | startup込みphase/total、process group STOP、原保存保持 |
| [runner](../../scripts/resource_applicability/run_track_a_ax2b_h6_input_generation_v1.py) | default metadata-only、別grant前にnumerical importしない |
| [saved-byte audit](../../src/trottertracks/resource_applicability/ax2b_h6_input_generation_audit_v1.py) / [audit CLI](../../scripts/resource_applicability/audit_track_a_ax2b_h6_input_generation_v1.py) | stdlib NPZ header/raw SHA、occupation indices、grant/state/sector来歴。signal/residual再計算なし |
| [合成tests](../../tests/tracks/resource_applicability/test_ax2b_h6_input_generation_v1.py) | 入力/gate/serializer/STOP/監査のfixture検証 |
| [prepared manifest](../../artifacts/resource_applicability/track_a_ax2b_h6_input_preparation_v1/2026-10-10/prepared_input_generation_v1.json) | source/environmentを結合、CPU null・unsealed・grantなし |
| [static audit](../../artifacts/resource_applicability/track_a_ax2b_h6_input_preparation_v1/2026-10-10/preparation_static_audit_v1.json) | 再利用一覧、配列規模・進捗上界、未確認事項 |
| [synthetic audit](../../artifacts/resource_applicability/track_a_ax2b_h6_input_preparation_v1/2026-10-10/synthetic_test_audit_v1.json) / [JUnit](../../artifacts/resource_applicability/track_a_ax2b_h6_input_preparation_v1/2026-10-10/synthetic_test_results_v1.xml) | 87 local tests pass（新38・既存49） |
| [preparation inventory](../../artifacts/resource_applicability/track_a_ax2b_h6_input_preparation_v1/2026-10-10/preparation_inventory_v1.json) | 今回の証拠・未実行/未認可状態 |
| [保全監査](../../artifacts/resource_applicability/track_a_ax2b_h6_input_preparation_v1/2026-10-10/preservation_audit_v1.json) / [remote照合](../../artifacts/resource_applicability/track_a_ax2b_h6_input_preparation_v1/2026-10-10/remote_verification_v1.json) | source183/validation3、新公開26 path、旧H4 raw1212＋30、旧prep195と既存3993ファイルの保全確認 |

合成testsはfake chemistry/decomposer/solver/operatorとdummy processだけを使った。
SCF provider wiring・two-body1/2・DF kwargs、shared matvec capとresidual、state phase規約、
serialization/既存loader/stdlib監査、source/env/schema/JSON型/CPU/output拒否、log/wall/phase STOPを検査した。
defaultと不完全execute CLIは`python -S`でも確認し、numerical imports前のgateを検査した。
real molecule/eigsh/sampling/circuitおよび実科学artifact I/Oはfixtureで禁止した。
合成snapshotはtemporary fixtureであり、実H6入力・分子検証結果・immutable CI・外部再現として扱わない。
実source/Git/環境は別のmetadata監査で照合し、real-array gate性能は未測定。

## 次の認可境界・GPTレビュー

今回指示は準備・契約固定まで。新grantを作成していない。
次は、ユーザーの入力生成に対する明示指示を受けて、使用CPU/resource観測とmetadata seal、
この入力生成専用schemaの新grant/digest/原grant SHAを固定・公開する。
その後にだけ一回入力生成→snapshot/監査/hash固定→STOPとする。
現在のprepared manifest内部のfalse flagsや文書はgrantの代用にならない。

入力が揃った後にactual L/sector/DF provenanceを使い、7 cell/36 wrapperのactual primitive/instruction bounds、
pilot source/env/CPU・独立grantを結果前に固定する。今回そのnative準備・sampling・compileは実施していない。
H6 pilot gateの接続もその別工程で扱い、H4-only gateや入力grantを再利用しない。
次の主要GPT科学レビューはH4補完＋別認可されたH6技術pilot後に、H6本検証GO/STOPを判断する節目。
DF政策の実decomposition不一致、sector/独立性の矛盾、科学的条件変更が必要な場合は前倒しでGPTへ戻す。
H8は独立評価用に保護する。Track Bと別系列server run10のsource/job/比較条件には触れていない。
`H6_NOT_AUTHORIZED` / `DRAFT_NOT_AUTHORIZATION`、mandatory STOPを維持して停止する。
