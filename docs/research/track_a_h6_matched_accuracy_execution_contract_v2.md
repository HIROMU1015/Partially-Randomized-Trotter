# Track A H6精度一致資源比較 v2 資源limit修正と実行準備

2026-10-11。H6精度一致比較v1の起動時STOPを修正した別versionである。利用者が認可した範囲はsource修正、実子processによるsynthetic検査、manifestの固定・公開まで。**科学実行は未認可、新grantは未発行、mandatory STOP。** 旧v1のsource・manifest・消費済みgrant・原STOP出力を変更しない。

## 固定sourceと保存入力

新source commitは `54986a9f54515c8c2289e38fa16fd81e2b6008c2`。[v2 contract](../../src/trottertracks/resource_applicability/h6_matched_contract_v2.py)、[v2 supervisor/worker](../../src/trottertracks/resource_applicability/h6_matched_execution_v2.py)、[runner](../../scripts/resource_applicability/run_track_a_h6_matched_v2.py)、[metadata preparer](../../scripts/resource_applicability/prepare_track_a_h6_matched_v2.py)、[static auditor](../../scripts/resource_applicability/audit_track_a_h6_matched_v2.py)、[synthetic tests](../../tests/tracks/resource_applicability/test_h6_matched_v2.py)を追加した。

[新sealed manifest](../../artifacts/resource_applicability/track_a_h6_matched_preparation_v2/2026-10-11/sealed_manifest_v2.json)はruntime source216 path、保存入力33 path、環境、CPU配置、全plan、別の未使用出力を固定する。manifest SHA-256は `99cd2055283d2bb35ec5b373d934b3107fac6997a013e5f523914df3216014b9`、canonical digestは `7693137a582dc80d2f221dc2ed8a51161ebd79f6f183f6a4bb9dbe4e0f20b19e`。

入力は同じ[保存snapshot](../../artifacts/resource_applicability/track_a_h6_saved_df_completion_parallel_v2/2026-10-10/launch_v1/h6_input_snapshot.npz)。入力source `f37005f01b2be38c5993d6e82df91abe9c643d21`、保存結果 `554fc52add39c2c1b45b765a3135df76fda6f15a`、[監査](../../artifacts/resource_applicability/track_a_h6_saved_df_completion_parallel_execution_v2/2026-10-10/execution_evidence_inventory_v2.json) `f52787a22542b31bd39fd004a8d3d71325bc56b0`。snapshot SHA-256 `99440a59d903dfd6330e786d84a956f1dd8a687a592295a1cd771c839769b005`、親33 fileのbytesと元commitを照合する。SCF・DF分解・state生成は含まない。

## 科学条件と費用取得はv1のまま

[旧結果前契約](track_a_h6_matched_accuracy_execution_contract_v1.md)の科学plan・accounting・numerical portを直接再利用する。旧採用契約commit `12183db780bef14c25c93ba6523c44c52edd7828` のplanと、新manifestのplanはcanonical digest `801e1eea5d09b6fbe60fe66102bc21edb1bd50a0d9aa94ca05a2b0659d1f8cf3`で一致。[科学条件・実装同一性](../../artifacts/resource_applicability/track_a_h6_matched_preparation_v2/2026-10-11/science_plan_identity_v2.json)。v2のouter identityだけを分離し、内部plan名とseed namespaceのv1表記も保持する。

対象はlinear H6、等間隔1.00 Å、STO-3G、12 modes、α3/β3、sector400、tol-only `1e-8`のsigned generation-order DF rank19、cutoff0、採用済みweighted Hermitian projection・同じ保存state、T=0.8。B0 generation-prefix5/10/15の12候補、B1 full19 S2/S4の8候補、B2 prefix5/10/15の72候補、計92候補。q1/2/4/8、B2 r1/2/4・R=qr・K2/4、delta0.8/0.4/0.2/0.1、ε_sig=0.05/0.01/0.005/0.001を維持する。

bias・empirical numerical allowance・normalization・軸別Hoeffding shot会計、全適格候補のexploration n8、精度別上位2のunion最大8候補を別seedでconfirmation n32、paired X/Y・ordinary anchor、最大1,708 measured wrapper、compile guard1,712、全call/instruction guardを変更しない。B3は168組のnormalization-only診断で、新B3 signal/sampling/compileは含まない。state-preparation除外と線形感度、population winner未認定、ground-state未認定も維持する。

## coordinatorとworkerのアドレス空間制限

| process | AS soft limit | AS hard ceiling | 役割 |
|---|---:|---:|---|
| coordinator | 2 GiB | 8 GiB | 割当上限2 GiBをsoftで強制し、子へ8 GiBのceilingを継承 |
| signal worker | 8 GiB | 8 GiB | 数値port import前に設定 |
| cost worker | 8 GiB | 8 GiB | 各独立taskの数値port import前に設定 |

coordinatorの実効割当予算は2 GiBのまま。hard ceilingは子の8 GiB設定を可能にするための区別で、coordinator自身のsoft limitを後から引き上げる経路はない。元のcoordinator soft/hard両方2 GiBを子が継承していた問題を解消する。`install_coordinator_limits()`と`verify_worker()`が実際のsoft/hard値を読んでreceiptへ保存する。上位環境のAS hard ceilingが8 GiB未満、またはFSIZE ceilingが2 GiB未満なら、launch前に明示的に拒否する。

signal CPU IDs [0,2,5,6]・Numba/OMP4/BLAS1。costは[0,2]/[5,6]の2 disposable process・各Numba/OMP/BLAS/Qiskit/Rayon1、Qiskit parallel=false。aggregate RSS18 GiB、output2 GiB・log1 MiB、signal/private output64 MiB、cost/private output32 MiB、core dump0を維持。phase/total wall capはnull。OSのASとlive RSSの上限は別で、物理RAMの予約を示さない。

compact signal、sector400、primitive/spectral/prefix cache、signal先行の適格性screen、2 processの費用取得を再利用する。既存pilotの保存timingと[静的性能確認](../../artifacts/resource_applicability/track_a_h6_matched_preparation_v2/2026-10-11/performance_review_v2.json)ではwrapper費用取得が主な古典workloadと判断する。[host inventory](../../artifacts/resource_applicability/track_a_h6_matched_preparation_v2/2026-10-11/host_inventory_v2.json)にCPU/GPU/RAMと配置を記録した。科学backendはCPUのままで、追加benchmarkやspeedup倍率の認定はしない。

## 検証範囲

[ローカル検査報告](../../artifacts/resource_applicability/track_a_h6_matched_preparation_v2/2026-10-11/synthetic_test_report_v2.json)、[JUnit](../../artifacts/resource_applicability/track_a_h6_matched_preparation_v2/2026-10-11/synthetic_tests_v2.xml)：新規17・既存159、計176 PASS。CIや外部再現、実H6 signalの科学的証拠ではない。[検査履歴](../../artifacts/resource_applicability/track_a_h6_matched_preparation_v2/2026-10-11/test_history_v2.json)も保存した。

実coordinator processがlimitを設定した後、production supervisorを通して実子processをspawnし、子がproduction `verify_worker()`を通過する順序で検査した。signal1・探索cost2・確認cost2の[実測receipt](../../artifacts/resource_applicability/track_a_h6_matched_preparation_v2/2026-10-11/synthetic_process_receipts/)で、継承時AS2/8 GiB、子の設定後AS8/8 GiB、CPU affinity・thread環境・FSIZE/coreを確認した。synthetic検査のCPUは利用可能affinityの先頭4、科学sealのCPUは上記固定4 IDである。

coordinatorでは2 GiBの追加仮想mappingが拒否され、子では許可されることも検査した。mappingはpageをtouchせず直ちに解放する。旧coordinator hard2 GiBでの失敗を実子で再現し、上位hard ceiling不足、旧v1 grant、新plan/limit改変、未認可launchの拒否を検査した。子の本数・env・private writerは実supervisor経路を使い、子のnumerical portはimportしない。実分子archiveやtrajectory、compileは使わない。

## 保全と実行認可待ち

[旧v1 STOP索引](track_a_h6_matched_accuracy_result_v1.md)、保存結果 `274b984971f9afc3425f3b3d1ce36e1c478150ea`、公開確認 `ffe9414d3bd4d70466e85583426565918ac61bd3`、旧source `057ba97ba4db66775cf713c65cb6955e9dcb0d85`、旧grant消費済みを保持する。追加moduleを含む新treeのclosureはv2 manifestへ固定し、旧v1の監査は旧保存commitのtreeで追跡する。

[準備監査](../../artifacts/resource_applicability/track_a_h6_matched_preparation_v2/2026-10-11/preparation_audit_v2.json)、[保全監査](../../artifacts/resource_applicability/track_a_h6_matched_preparation_v2/2026-10-11/preservation_audit_v2.json)。[認可template](../../artifacts/resource_applicability/track_a_h6_matched_preparation_v2/2026-10-11/authorization_template_v2.json)は`approved_by_user=false`で、実行grantではない。[具体的な次の認可対象](../../artifacts/resource_applicability/track_a_h6_matched_preparation_v2/2026-10-11/launch_request_v2.json)は、このv2 source/manifest・未使用出力 `artifacts/resource_applicability/track_a_h6_matched_accuracy_v2/2026-10-11/launch_v1`に従う一回実行だけ。

科学runnerを起動せずSTOPする。別の明示認可後に、新しいv2 grantを発行・pinして一回実行し、原結果・監査・source/input事後identityを公開・remote照合してmandatory STOPする設計である。retry/resumeなし、H8・追加探索・入力再生成は対象外。`H6_NOT_AUTHORIZED` / `DRAFT_NOT_AUTHORIZATION`、`next_stage_authorized=false`を維持する。
