# Track A AX-2B H4後：GPT独立科学レビュー資料索引 v1

**source公開追記（2026-10-10）**：v5必須24件と旧監査source16件を追加公開した。現在の対応は[第7節](#source-completion)を参照する。以下の「未公開24件／46件」は初回公開commit時点の履歴で、最新の公開状態ではない。H6未採用・未認可と科学的認定状態は変わらない。

2026-10-10 JST。今回の作業は既存資料の公開・リンク補修・静的照合だけである。
**独立科学レビューは未実施。3つの中心文書はCodex作成の案であり、GPTの判断・承認を表さない。H6案は未採用・未認可。**

Repository：`HIROMU1015/Partially-Randomized-Trotter`。公開branch：`track-a-ax2b-h4-post-review-20261010`。
保存証拠commit：[`aa9b4768819680600d99ceb962b22aec16b99fb0`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/aa9b4768819680600d99ceb962b22aec16b99fb0)。
中心文書・本索引はその直後の資料公開commitに含む。最終報告のfull SHAで文書を固定できる。

## 1. レビュー対象と読む順序

| 文書 | 役割 | 証拠区分 |
|---|---|---|
| [H4後科学レビュー案](track_a_ax2b_h4_post_scientific_review_v1.md) | H4 pilotの解釈、成果範囲、追加検証と比較案 | Codex提案。独立レビュー結果ではない |
| [不足検証11項目](track_a_ax2b_h4_validation_gaps_v1.md) | U-N1〜U-I1、既存能力と必要な追加の分解 | 静的監査に基づく案。未公開sourceあり |
| [H6 7 cell・36 wrapper契約案](track_a_ax2b_h6_pilot_contract_draft_v1.md) | 入力・sector・DF・候補・capsの具体案 | DRAFT_NOT_AUTHORIZATION、未採用 |

先に[H4 v5結果報告](track_a_ax2b_h4_pilot_v5_result.md)と下記一次JSONを読む。
「技術gate通過」「保存値の一致」「科学的精度認定」「測定込み方式選択」を区別する。

## 2. H4一次結果・実行条件・監査

対象はlinear H4、隣接1.00 Å、STO-3G、legacy DF rank12、generation-prefix、8 system qubits、T=0.8。
DF prefix0/6/12、q1/q4、δ=T/q=0.8/0.2。保存normalized state、Nα=Nβ=2、sector dimension36。
B2/B3はcanonical finite-RTE、R=8、r=2、K=2/4/6。分子・stateは今回再生成していない。

| 一次資料 | 確認できる内容 |
|---|---|
| [raw run_v1](../../artifacts/resource_applicability/track_a_ax2b_h4_pilot_v5/2026-10-10/launch_v1/run_v1) | 8 correctness、28 wrapper cost、7 trajectory records、391 diagnostics等の全447ファイル |
| [input_reference.json](../../artifacts/resource_applicability/track_a_ax2b_h4_pilot_v5/2026-10-10/launch_v1/run_v1/input_reference.json) | 保存state、target metadata、sector全列、expm/eigh、残差と参照signal |
| [primitive_lowering.json](../../artifacts/resource_applicability/track_a_ax2b_h4_pilot_v5/2026-10-10/launch_v1/run_v1/primitive_lowering.json) | native・basis・phase・primitive検査のscope |
| [wrapper_cost_summary.json](../../artifacts/resource_applicability/track_a_ax2b_h4_pilot_v5/2026-10-10/launch_v1/run_v1/wrapper_cost_summary.json) | 両axis・ordinary/directional・replica別の費用集計と元records |
| [parent terminal](../../artifacts/resource_applicability/track_a_ax2b_h4_pilot_v5/2026-10-10/launch_v1/run_v1/terminal_status.json) / [worker terminal](../../artifacts/resource_applicability/track_a_ax2b_h4_pilot_v5/2026-10-10/launch_v1/run_v1/worker_terminal.json) | COMPLETE、counter、wall、mandatory STOP、次段未認可 |
| [保存監査](../../artifacts/resource_applicability/track_a_ax2b_h4_pilot_v5/2026-10-10/launch_v1/saved_evidence_audit_v5.json) | H4_SAVED_PILOT_EVIDENCE_VERIFIED、8/28/7、source/input/認可binding |
| [保存解析・raw全hash](../../artifacts/resource_applicability/track_a_ax2b_h4_pilot_v5/2026-10-10/launch_v1/saved_result_analysis_v5.json) | 旧v3/v4との一致、stage wall/RSS、447ファイルの同一性 |
| [execution inventory](../../artifacts/resource_applicability/track_a_ax2b_h4_pilot_v5/2026-10-10/launch_v1/pilot_evidence_inventory_v5.json) | 実行時保存inventory。絶対pathは当時のorigin記録 |
| [H4 v5認可](../../artifacts/resource_applicability/track_a_ax2b_h4_pilot_v5/2026-10-10/launch_v1/authorization_v5.json) / [実割当CPU](../../artifacts/resource_applicability/track_a_ax2b_h4_pilot_v5/2026-10-10/launch_v1/cpu_selection_v5.json) / [実行前監査](../../artifacts/resource_applicability/track_a_ax2b_h4_pilot_v5/2026-10-10/launch_v1/prelaunch_audit_v5.json) | H4一回限定のscope、environment、source/plan identity。H6認可ではない |
| [実行直前167 tests JUnit](../../artifacts/resource_applicability/track_a_ax2b_h4_pilot_v5/2026-10-10/launch_v1/prelaunch_167.junit.xml) | 保存済みlocal tests。今回は再実行していない |
| [manifest](../../artifacts/resource_applicability/track_a_ax2b_h4_fingerprint_preparation/2026-10-09/h4_preparation_manifest_v5.json) / [source freeze](../../artifacts/resource_applicability/track_a_ax2b_h4_fingerprint_preparation/2026-10-09/source_freeze_v5.json) | 登録task/capと157 science・6 validation source hashes |

costはB0 prefix6/q4、B2 prefix6/q4/R8/K2、B3 prefix0/q4/R8/K6、B1 global四次prefix12/q1,q4の5種類に限る。
B1二次とB2 K4はcorrectnessのみ。全8 cellの費用を測ったとは扱わない。
random cost用trajectoryは計4、outer occurrence16、2 trajectory/random cell。両axis/controlへ共有した工程標本であり、量子shotではない。

measured full Hadamard wrapperはstate preparationを除く。compilerはrz/sx/x/cx、optimization1、seed17、backend/coupling指定なし。
実行環境はPython3.11.1、NumPy1.26.4、SciPy1.14.1、Qiskit1.3.0、OpenFermion1.6.1。
snapshot metadataの生成時environment（Python3.11.0rc1等）は旧入力の来歴で、v5実行環境とは別である。
logical CPU3、worker1、BLAS/OMP/MKL thread1、RLIMIT_AS8 GiB、phase900秒、total2700秒、output512 MiB、log64 KiB、diagnostics1024が固定上限。
保存parent wallは787.9848190899938秒。RSSは古典評価側の記録であり、量子回路費用に加算しない。詳細・限界は結果報告を参照する。

## 3. 失敗・準備・欠測を含む経緯

| 段階 | 保存資料 | 残す意味 |
|---|---|---|
| AX-2A v1 | [技術準備](track_a_ax2a_technical_preparation_v1.md) / [57-test・再利用inventory](../../artifacts/resource_applicability/track_a_ax2a_preparation/2026-10-09) | finite state-action、sector等の接続。分子scopeでの認定ではない |
| AX-2A native v2 | [native DF準備](track_a_ax2a_native_df_preparation_v2.md) / [117-test evidence](../../artifacts/resource_applicability/track_a_ax2a_native_preparation/2026-10-09) | 二次・global四次・対称control・scalar phaseのtoy検査 |
| v3 runner準備 | [準備v3](track_a_ax2b_h4_runner_preparation_v3.md) / [174-test・freeze](../../artifacts/resource_applicability/track_a_ax2b_h4_runner_preparation/2026-10-09) | bounded H4 runnerの履歴 |
| v3実行STOP | [結果v3](track_a_ax2b_h4_pilot_result_v1.md) / [全保存記録](../../artifacts/resource_applicability/track_a_ax2b_h4_pilot/2026-10-09/launch_v1) | 24/28 wrapper後MemoryError。traceback・partial・認可・STOPを保存 |
| v4準備・STOP | [memory準備v4](track_a_ax2b_h4_memory_preparation_v4.md) / [結果v4](track_a_ax2b_h4_pilot_v4_result.md) / [準備記録](../../artifacts/resource_applicability/track_a_ax2b_h4_memory_preparation/2026-10-09) / [実行記録](../../artifacts/resource_applicability/track_a_ax2b_h4_pilot_v4/2026-10-09/launch_v1) | 16/28後PHASE_WALL_CAP。worker terminal欠測は欠測のまま |
| v5準備 | [fingerprint準備v5](track_a_ax2b_h4_fingerprint_preparation_v5.md) / [保存tests・profile・freeze](../../artifacts/resource_applicability/track_a_ax2b_h4_fingerprint_preparation/2026-10-09) | 初回synthetic 165 pass/2 failと最終167 passを両方保存。toy値をH4/H6性能に外挿しない |
| H4後文書・静的監査 | [post_review artifacts](../../artifacts/resource_applicability/track_a_ax2b_h4_post_review/2026-10-10) | 不足11項目、H6草案、文献確認記録、保全結果。新scienceなし |

異なる完了coverageのv3/v4/v5のwall/RSS比を改善倍率にしない。
現在も総数値allowanceは未認定、accuracyはUNDETERMINED。shot/total cost/winner、chemical energy accuracyは未評価。
primitive ±0.2やexpm/eigh差は総u_boundではない。未解決の全stage時間・非unitary誤差伝播・sampled estimator接続等は不足一覧へ残す。

## 4. commit・source・snapshotの来歴

- v5実行時base HEAD：`b2e1bf65e21893b6c617223b42313623d3186f12`。当時branchは`track-a-ax2b-h4-v5-execution-20261010`。
- 実際の実行sourceはbase HEADと未commitのlocal source bytesの組合せ。source一式を表す公開Git commitはない。
- 新しい結果保存commit：`aa9b4768819680600d99ceb962b22aec16b99fb0`。科学結果と旧停止記録のbytesをそのまま保存したcommitであり、実行時source commitではない。
- 157 science sourceのうち139のexact bytesは既公開branch `pr2-v4-s2-parallelization-20260928`、commit `b2e1bf65e21893b6c617223b42313623d3186f12`に一致。今回は再commitしていない。
- 18 science sourceと6 validation sourceは未公開。[source freeze](../../artifacts/resource_applicability/track_a_ax2b_h4_fingerprint_preparation/2026-10-09/source_freeze_v5.json)のhashを正本とする。
- 全sourceのpath・SHA256・公開branch/commit/URLは[dependency_registry_v1.json](../../artifacts/resource_applicability/track_a_ax2b_gpt_review_publication/2026-10-10/dependency_registry_v1.json)。[remote 68 heads確認](../../artifacts/resource_applicability/track_a_ax2b_gpt_review_publication/2026-10-10/remote_heads_before_v1.json)と[履歴到達性検査](../../artifacts/resource_applicability/track_a_ax2b_gpt_review_publication/2026-10-10/source_reachability_check_v1.json)を併置する。

| 既公開source（全てbase commit上のexact bytes） | 役割 |
|---|---|
| [df_hamiltonian.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b2e1bf65e21893b6c617223b42313623d3186f12/src/trotterlib/df_hamiltonian.py) | DF・sector・matrix-free |
| [pf_c_system_size_validation.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b2e1bf65e21893b6c617223b42313623d3186f12/src/trotterlib/pf_c_system_size_validation.py) | PF state/sector既存経路 |
| [df_gpu_statevector.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b2e1bf65e21893b6c617223b42313623d3186f12/src/trotterlib/df_gpu_statevector.py) | GPU既存経路（今回未実行） |
| [rte.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b2e1bf65e21893b6c617223b42313623d3186f12/src/trotterlib/rte.py) | finite distribution・normalization・sampling |
| [df_rpe_hadamard_compiled_cost.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b2e1bf65e21893b6c617223b42313623d3186f12/src/trotterlib/df_rpe_hadamard_compiled_cost.py) | measured Hadamard wrapper |
| [rte_compiled_cost.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b2e1bf65e21893b6c617223b42313623d3186f12/src/trotterlib/rte_compiled_cost.py) | compiler費用測定 |
| [ax1b_evaluation.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b2e1bf65e21893b6c617223b42313623d3186f12/src/trottertracks/resource_applicability/ax1b_evaluation.py) | 旧conditional-oracle会計 |

保存H4入力：[h4_1p00_rank12_development_v1.npz](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b2e1bf65e21893b6c617223b42313623d3186f12/artifacts/pr2_s0_s1_validation/2026-09-28/h4_1p00_rank12_development_v1.npz)。
snapshot SHA256：`3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a`。
入力は既公開の同一bytesを再利用し、再commit・再生成していない。snapshot内部の生成source hashesとv5実行freezeは別の来歴である。

旧AX-1b結果・FEW：[model_fits.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b2e1bf65e21893b6c617223b42313623d3186f12/artifacts/resource_applicability/track_a_ax1b_saved_model_audit/2026-10-09/run_v1/model_fits.json)、
[execution audit](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b2e1bf65e21893b6c617223b42313623d3186f12/artifacts/resource_applicability/track_a_ax1b_saved_model_audit/2026-10-09/execution_audit_v1/result_review_report.md)。
AX-1b source commitは`fc297cd9ab840018c4f35b2764d0b8be07c57285`、結果保存commitはbase HEAD。旧FEW・契約を変更／再fitしない。

<a id="research-objectives"></a>
### 採用済みRQの記録と今回の案

[共有されたAX-1b後GPTレビューの写し](track_a_ax1b_post_scientific_review_2026-10-09.md)と
[AX-2A追補案](track_a_ax2a_research_amendment_v1.md)にRQ-R主／RQ-P1補助の来歴がある。
既存dirty `docs/research/研究目的・研究課題.md`はstageせず、[exact text snapshot](../../artifacts/resource_applicability/track_a_ax2b_gpt_review_publication/2026-10-10/local_research_objectives_snapshot_v1.json)に原文・hashを保存した。
旧日付の認可／未認可記述は当時の対象に限る。現在のH6認可として扱わない。
3つの新レビュー文書には科学的提案・解釈が含まれる。保存結果との区別を維持し、今回の公開で採用・認定へ変更しない。

<a id="unpublished-source"></a>
### 未公開sourceとレビューの限界

**v5の必須依存24ファイルは未公開。sourceを無条件commitしないという今回の指示に従い、変更もstageもしていない。**
既公開68 branchの全到達履歴を調べても当該exact blobsはなかった。hashだけから実装内容を復元することはできない。
従って保存claim・数値・実行条件はGitHubで追跡できるが、native実装のコード監査と完全な再現には未公開sourceが必要である。
「source込みで外部再現可能」「immutable CIで検証済み」とは報告しない。文書の旧sourceリンクは本節へ付け替え、元pathを補修記録に残した。

| 未公開v5依存（repository path） | freeze区分 | SHA256先頭16桁（全桁はregistry） |
|---|---|---|
| `scripts/resource_applicability/profile_track_a_ax2b_fingerprint_v5.py` | preparation_validation_source | `c80a8f5a6fd8a6bb` |
| `scripts/resource_applicability/run_track_a_ax2b_h4_v5.py` | science_source | `59ccf9351db50bcc` |
| `scripts/resource_applicability/verify_track_a_ax2b_h4_pilot_v5.py` | preparation_validation_source | `102fb874abb8ac5a` |
| `src/trottertracks/resource_applicability/ax2a_control_plan.py` | science_source | `19046fc005a6be92` |
| `src/trottertracks/resource_applicability/ax2a_native_df.py` | science_source | `b67cc43993fa23b6` |
| `src/trottertracks/resource_applicability/ax2a_preparation.py` | science_source | `5478a99b1acb5568` |
| `src/trottertracks/resource_applicability/ax2a_state_action.py` | science_source | `57a5e5affa762000` |
| `src/trottertracks/resource_applicability/ax2b_diagnostics_v4.py` | science_source | `927cfed5fabccbc1` |
| `src/trottertracks/resource_applicability/ax2b_h4_contract.py` | science_source | `d1af2549cd34d13d` |
| `src/trottertracks/resource_applicability/ax2b_h4_contract_v4.py` | science_source | `ba93b61c85b3412e` |
| `src/trottertracks/resource_applicability/ax2b_h4_contract_v5.py` | science_source | `94a92415307458da` |
| `src/trottertracks/resource_applicability/ax2b_h4_science.py` | science_source | `80b26eb984d32fe6` |
| `src/trottertracks/resource_applicability/ax2b_h4_science_v4.py` | science_source | `3d776ed186bfffc3` |
| `src/trottertracks/resource_applicability/ax2b_h4_science_v5.py` | science_source | `64cea16f29bd2a6d` |
| `src/trottertracks/resource_applicability/ax2b_limits.py` | science_source | `21510feed20ecf73` |
| `src/trottertracks/resource_applicability/ax2b_native_df_v4.py` | science_source | `5ee68a5515952ee4` |
| `src/trottertracks/resource_applicability/ax2b_native_df_v5.py` | science_source | `7b8b13483066bb39` |
| `src/trottertracks/resource_applicability/ax2b_preflight.py` | science_source | `530efb7b76c4451c` |
| `src/trottertracks/resource_applicability/ax2b_stream_fingerprint_v4.py` | science_source | `6dd7a637762b45da` |
| `src/trottertracks/resource_applicability/ax2b_stream_fingerprint_v5.py` | science_source | `2e7c78f93eb65354` |
| `tests/tracks/resource_applicability/test_ax2b_h4_runner_v5.py` | preparation_validation_source | `473b2cb9c62d7d21` |
| `tests/tracks/resource_applicability/test_ax2b_h4_saved_audit_v5.py` | preparation_validation_source | `1b4799d2c9034b0e` |
| `tests/tracks/resource_applicability/test_ax2b_native_df_v5.py` | preparation_validation_source | `ff2fc9f6b3d5a64a` |
| `tests/tracks/resource_applicability/test_ax2b_stream_fingerprint_v5.py` | preparation_validation_source | `3fee4f9ffb1c3f0e` |

旧段階のrunner/testsと保存監査helperの別22ファイルも未公開。全46ファイルのpath・hash・理由は[registry](../../artifacts/resource_applicability/track_a_ax2b_gpt_review_publication/2026-10-10/dependency_registry_v1.json)。
元sourceやartifact内helperのコピーを別pathへ公開することも行っていない。監査結果JSONとhelper sourceを区別する。

## 5. H6未認可とGPTが独立に判断する論点

[H6機械可読草案](../../artifacts/resource_applicability/track_a_ax2b_h4_post_review/2026-10-10/h6_pilot_contract_draft_v1.json)は`DRAFT_NOT_AUTHORIZATION`。
`science_authorized=false`、`source_implementation_authorized=false`、`input_generation_authorized=false`、
`launch_allowed=false`、`next_stage_authorized=false`、`launcher_implemented=false`、`mandatory_stop=true`、`assigned_resources=null`。
actual rank、source/input identity等は未確定。`H6_NOT_AUTHORIZED`を維持し、7 cell/36 wrapper・tol・capsは提案のままである。

独立GPTへの判断依頼は既存レビュー案の四点を維持する。Codexは今回これらへ科学的なGO/STOPを出さない。

1. H4 technical acceptanceの範囲と、u_bound/u_empiricalの扱いは妥当か。
2. 追加H4検証は参照・実時間集合・非unitary誤差会計のどこまで必要か。
3. H6の7 cell/36 wrapper案、DF政策、state、参照と上限は情報価値に見合うか。
4. 最小成果とH8独立検証の条件を満たす比較候補・標本規則は何か。

未公開実装への依存が独立レビューに与える制限も明示して判断する。新しい実装・scienceの認可は別のユーザー指示を要する。

## 6. 公開作業の保全と追跡

[公開対象の全path/hash](../../artifacts/resource_applicability/track_a_ax2b_gpt_review_publication/2026-10-10/evidence_package_manifest_v1.json)、[公開前保全baseline](../../artifacts/resource_applicability/track_a_ax2b_gpt_review_publication/2026-10-10/preservation_before_publication_v1.json)、
[文書リンク補修・原文archive](../../artifacts/resource_applicability/track_a_ax2b_gpt_review_publication/2026-10-10/document_link_repairs_v1.json)、[公開前静的照合](../../artifacts/resource_applicability/track_a_ax2b_gpt_review_publication/2026-10-10/publication_static_check_v1.json)を参照する。
旧科学artifact・source・旧inventory・validation manifestのbytesは変えず、指定branchに新しい資料だけを追加した。
旧inventoryの一部文書hashは公開前の原文を指す。リンク補修後の文書hashへ黙って書き換えず、原文archiveと補修記録で対応を残す。
初回synthetic失敗logにある元のtrailing whitespaceも、保存bytes保全のため修正しない。
他branch/worktreeの内容は取り込まず、8つの既存tracked dirty文書はそのままunstagedに残す。
新しい分子計算、入力生成、signal評価、trajectory sampling、回路build/transpile/compile、benchmark、再fit、GPU実行、tests再実行はない。
公開後はSTOP。H6/H8や次の科学段階へ自動進行しない。

<a id="source-completion"></a>
## 7. v5 exact source公開の追記（2026-10-10）

必須24件のsource公開commit：[`31cdff47f3282d2898c153a1af22f90e500be4c6`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/31cdff47f3282d2898c153a1af22f90e500be4c6)。
旧監査source16件の公開commit：[`3e9358421c072af970a9a37025e7fdc9c51ce8ea`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/3e9358421c072af970a9a37025e7fdc9c51ce8ea)。
実行時base HEADは`b2e1bf65e21893b6c617223b42313623d3186f12`のままである。新しい公開commitを実行時HEADに置き換えない。
v5は当時のbase HEADと未commitのfrozen bytesで実行された。今回のcommitはそのexact bytesの公開来歴であり、新しい実行・再現試験・科学的認定ではない。

[全163 frozen sourceのpath・SHA256・commit・URL](../../artifacts/resource_applicability/track_a_ax2b_gpt_review_publication/2026-10-10/source_completion_v1/source_publication_manifest_v1.json)、
[実行／準備／レビューworktreeでのbytes照合](../../artifacts/resource_applicability/track_a_ax2b_gpt_review_publication/2026-10-10/source_completion_v1/source_bytes_check_v1.json)、
[公開内容の確認](../../artifacts/resource_applicability/track_a_ax2b_gpt_review_publication/2026-10-10/source_completion_v1/source_publication_content_check_v1.json)、
[保全・公開前静的照合](../../artifacts/resource_applicability/track_a_ax2b_gpt_review_publication/2026-10-10/source_completion_v1/source_publication_static_audit_v1.json)を正本とする。
既存dependency_registry_v1.jsonとsource_freeze_v5.json、過去の実行監査・inventoryは改変していない。

### 新たに公開したv5必須24件

science18件・validation6件は、レビューworktree／v5実行worktree／v5準備worktreeの3箇所でfreezeのSHA256に一致した。
Gitのstaged blobとcommit blobも同じbytesである。sourceの再実装・修正・代用はない。

| sourceへの固定commitリンク | 区分 |
|---|---|
| [scripts/resource_applicability/profile_track_a_ax2b_fingerprint_v5.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/31cdff47f3282d2898c153a1af22f90e500be4c6/scripts/resource_applicability/profile_track_a_ax2b_fingerprint_v5.py) | validation |
| [scripts/resource_applicability/run_track_a_ax2b_h4_v5.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/31cdff47f3282d2898c153a1af22f90e500be4c6/scripts/resource_applicability/run_track_a_ax2b_h4_v5.py) | science |
| [scripts/resource_applicability/verify_track_a_ax2b_h4_pilot_v5.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/31cdff47f3282d2898c153a1af22f90e500be4c6/scripts/resource_applicability/verify_track_a_ax2b_h4_pilot_v5.py) | validation |
| [src/trottertracks/resource_applicability/ax2a_control_plan.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/31cdff47f3282d2898c153a1af22f90e500be4c6/src/trottertracks/resource_applicability/ax2a_control_plan.py) | science |
| [src/trottertracks/resource_applicability/ax2a_native_df.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/31cdff47f3282d2898c153a1af22f90e500be4c6/src/trottertracks/resource_applicability/ax2a_native_df.py) | science |
| [src/trottertracks/resource_applicability/ax2a_preparation.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/31cdff47f3282d2898c153a1af22f90e500be4c6/src/trottertracks/resource_applicability/ax2a_preparation.py) | science |
| [src/trottertracks/resource_applicability/ax2a_state_action.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/31cdff47f3282d2898c153a1af22f90e500be4c6/src/trottertracks/resource_applicability/ax2a_state_action.py) | science |
| [src/trottertracks/resource_applicability/ax2b_diagnostics_v4.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/31cdff47f3282d2898c153a1af22f90e500be4c6/src/trottertracks/resource_applicability/ax2b_diagnostics_v4.py) | science |
| [src/trottertracks/resource_applicability/ax2b_h4_contract.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/31cdff47f3282d2898c153a1af22f90e500be4c6/src/trottertracks/resource_applicability/ax2b_h4_contract.py) | science |
| [src/trottertracks/resource_applicability/ax2b_h4_contract_v4.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/31cdff47f3282d2898c153a1af22f90e500be4c6/src/trottertracks/resource_applicability/ax2b_h4_contract_v4.py) | science |
| [src/trottertracks/resource_applicability/ax2b_h4_contract_v5.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/31cdff47f3282d2898c153a1af22f90e500be4c6/src/trottertracks/resource_applicability/ax2b_h4_contract_v5.py) | science |
| [src/trottertracks/resource_applicability/ax2b_h4_science.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/31cdff47f3282d2898c153a1af22f90e500be4c6/src/trottertracks/resource_applicability/ax2b_h4_science.py) | science |
| [src/trottertracks/resource_applicability/ax2b_h4_science_v4.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/31cdff47f3282d2898c153a1af22f90e500be4c6/src/trottertracks/resource_applicability/ax2b_h4_science_v4.py) | science |
| [src/trottertracks/resource_applicability/ax2b_h4_science_v5.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/31cdff47f3282d2898c153a1af22f90e500be4c6/src/trottertracks/resource_applicability/ax2b_h4_science_v5.py) | science |
| [src/trottertracks/resource_applicability/ax2b_limits.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/31cdff47f3282d2898c153a1af22f90e500be4c6/src/trottertracks/resource_applicability/ax2b_limits.py) | science |
| [src/trottertracks/resource_applicability/ax2b_native_df_v4.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/31cdff47f3282d2898c153a1af22f90e500be4c6/src/trottertracks/resource_applicability/ax2b_native_df_v4.py) | science |
| [src/trottertracks/resource_applicability/ax2b_native_df_v5.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/31cdff47f3282d2898c153a1af22f90e500be4c6/src/trottertracks/resource_applicability/ax2b_native_df_v5.py) | science |
| [src/trottertracks/resource_applicability/ax2b_preflight.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/31cdff47f3282d2898c153a1af22f90e500be4c6/src/trottertracks/resource_applicability/ax2b_preflight.py) | science |
| [src/trottertracks/resource_applicability/ax2b_stream_fingerprint_v4.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/31cdff47f3282d2898c153a1af22f90e500be4c6/src/trottertracks/resource_applicability/ax2b_stream_fingerprint_v4.py) | science |
| [src/trottertracks/resource_applicability/ax2b_stream_fingerprint_v5.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/31cdff47f3282d2898c153a1af22f90e500be4c6/src/trottertracks/resource_applicability/ax2b_stream_fingerprint_v5.py) | science |
| [tests/tracks/resource_applicability/test_ax2b_h4_runner_v5.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/31cdff47f3282d2898c153a1af22f90e500be4c6/tests/tracks/resource_applicability/test_ax2b_h4_runner_v5.py) | validation |
| [tests/tracks/resource_applicability/test_ax2b_h4_saved_audit_v5.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/31cdff47f3282d2898c153a1af22f90e500be4c6/tests/tracks/resource_applicability/test_ax2b_h4_saved_audit_v5.py) | validation |
| [tests/tracks/resource_applicability/test_ax2b_native_df_v5.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/31cdff47f3282d2898c153a1af22f90e500be4c6/tests/tracks/resource_applicability/test_ax2b_native_df_v5.py) | validation |
| [tests/tracks/resource_applicability/test_ax2b_stream_fingerprint_v5.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/31cdff47f3282d2898c153a1af22f90e500be4c6/tests/tracks/resource_applicability/test_ax2b_stream_fingerprint_v5.py) | validation |

既公開science139件はbase commit上のbytesを再利用し、再commitしていない。
旧139件と新science18件、validation6件の全freeze項目が追加source公開後のtreeに一致する。
実行時環境・入力・結果の来歴は第2〜4節と保存artifactを参照する。新しい公開HEADに合わせてlaunch契約を変更していない。

### 旧runner・tests・監査helper 22件の扱い

16件を公開した。旧v3/v4 entrypoint・保存verifier4件、v4/v5保存解析とpost-review静的監査helper3件、
過去のsynthetic検証の前提・probe coverage・failure/cap処理を読むためのtests9件である。
各ファイルは初回dependency registryのhashと公開済み歴史inventory等のhashに一致する。科学的判断は行わず、いずれのrunner・tests・helperも起動していない。

| 公開した監査source |
|---|
| [artifacts/resource_applicability/track_a_ax2b_h4_pilot_v4/2026-10-09/launch_v1/analyze_saved_records_v4.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3e9358421c072af970a9a37025e7fdc9c51ce8ea/artifacts/resource_applicability/track_a_ax2b_h4_pilot_v4/2026-10-09/launch_v1/analyze_saved_records_v4.py) |
| [artifacts/resource_applicability/track_a_ax2b_h4_pilot_v5/2026-10-10/launch_v1/analyze_saved_records_v5.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3e9358421c072af970a9a37025e7fdc9c51ce8ea/artifacts/resource_applicability/track_a_ax2b_h4_pilot_v5/2026-10-10/launch_v1/analyze_saved_records_v5.py) |
| [artifacts/resource_applicability/track_a_ax2b_h4_post_review/2026-10-10/audit_saved_review_inputs_v1.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3e9358421c072af970a9a37025e7fdc9c51ce8ea/artifacts/resource_applicability/track_a_ax2b_h4_post_review/2026-10-10/audit_saved_review_inputs_v1.py) |
| [scripts/resource_applicability/run_track_a_ax2b_h4.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3e9358421c072af970a9a37025e7fdc9c51ce8ea/scripts/resource_applicability/run_track_a_ax2b_h4.py) |
| [scripts/resource_applicability/run_track_a_ax2b_h4_v4.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3e9358421c072af970a9a37025e7fdc9c51ce8ea/scripts/resource_applicability/run_track_a_ax2b_h4_v4.py) |
| [scripts/resource_applicability/verify_track_a_ax2b_h4_pilot.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3e9358421c072af970a9a37025e7fdc9c51ce8ea/scripts/resource_applicability/verify_track_a_ax2b_h4_pilot.py) |
| [scripts/resource_applicability/verify_track_a_ax2b_h4_pilot_v4.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3e9358421c072af970a9a37025e7fdc9c51ce8ea/scripts/resource_applicability/verify_track_a_ax2b_h4_pilot_v4.py) |
| [tests/tracks/resource_applicability/test_ax2a_native_df.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3e9358421c072af970a9a37025e7fdc9c51ce8ea/tests/tracks/resource_applicability/test_ax2a_native_df.py) |
| [tests/tracks/resource_applicability/test_ax2a_preparation.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3e9358421c072af970a9a37025e7fdc9c51ce8ea/tests/tracks/resource_applicability/test_ax2a_preparation.py) |
| [tests/tracks/resource_applicability/test_ax2b_h4_runner.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3e9358421c072af970a9a37025e7fdc9c51ce8ea/tests/tracks/resource_applicability/test_ax2b_h4_runner.py) |
| [tests/tracks/resource_applicability/test_ax2b_h4_runner_v4.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3e9358421c072af970a9a37025e7fdc9c51ce8ea/tests/tracks/resource_applicability/test_ax2b_h4_runner_v4.py) |
| [tests/tracks/resource_applicability/test_ax2b_h4_saved_audit.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3e9358421c072af970a9a37025e7fdc9c51ce8ea/tests/tracks/resource_applicability/test_ax2b_h4_saved_audit.py) |
| [tests/tracks/resource_applicability/test_ax2b_h4_saved_audit_v4.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3e9358421c072af970a9a37025e7fdc9c51ce8ea/tests/tracks/resource_applicability/test_ax2b_h4_saved_audit_v4.py) |
| [tests/tracks/resource_applicability/test_ax2b_native_df_v4.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3e9358421c072af970a9a37025e7fdc9c51ce8ea/tests/tracks/resource_applicability/test_ax2b_native_df_v4.py) |
| [tests/tracks/resource_applicability/test_ax2b_preflight.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3e9358421c072af970a9a37025e7fdc9c51ce8ea/tests/tracks/resource_applicability/test_ax2b_preflight.py) |
| [tests/tracks/resource_applicability/test_ax2b_stream_fingerprint_v4.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3e9358421c072af970a9a37025e7fdc9c51ce8ea/tests/tracks/resource_applicability/test_ax2b_stream_fingerprint_v4.py) |

残る補助6件は未公開のまま。全path・hash・個別理由は上記manifestに保存した。

| 残すsource | 理由 |
|---|---|
| `artifacts/resource_applicability/track_a_ax2b_h4_fingerprint_preparation/2026-10-09/toy_profile_before_v5.py` | 変更前toy profile補助。H4実行／停止理由の監査経路ではない。 |
| `artifacts/resource_applicability/track_a_ax2b_h4_fingerprint_preparation/2026-10-09/verify_preservation_v5.py` | 旧準備worktreeの保全操作。保存済みreport/hashを参照する。 |
| `artifacts/resource_applicability/track_a_ax2b_h4_pilot_v4/2026-10-09/launch_v1/verify_preservation_v4.py` | 旧v4 worktreeの保全操作。STOPを判定するrunner/verifierは公開した。 |
| `artifacts/resource_applicability/track_a_ax2b_h4_pilot_v5/2026-10-10/launch_v1/verify_preservation_execution_v5.py` | 旧実行worktreeの保全操作。今回のbytes照合は独立した静的操作で行う。 |
| `scripts/resource_applicability/prepare_track_a_ax2a.py` | metadata CLI。実装moduleと保存manifestは公開済み。 |
| `scripts/resource_applicability/prepare_track_a_ax2b_preflight.py` | preflight metadata CLI。実装module、旧tests、保存reportは公開済み。 |

補助6件の過去writer／保全手順まで完全に再実行できるとするものではない。source公開だけで外部再現・CI認定を主張しない。
総数値allowance未認定、accuracy UNDETERMINED、shot／総費用／winner未評価という科学的限定は変わらない。
`H6_NOT_AUTHORIZED`、`DRAFT_NOT_AUTHORIZATION`、`mandatory_stop=true`を維持する。H6入力作成・pilot・H8独立評価を認可しない。
push後のremote取得・hash照合結果は作業完了報告で確認する。独立科学判断はGPTへ戻す。
