# Track A ローカル側inventory

2026-10-14 JST。資料照合・Git metadataの棚卸し。科学レビュー、共通正本の採択、実験実行認可ではない。
サーバー正本候補は branch `track-a-cross-environment-sync-server-20261014`、固定commit `2298c4cdbc81d81c6268fdb7bd80f0b29d5b4e8c`。
利用者が共有したpushコマンドのbranchをGitHubで実取得し、その時点のremote HEADを固定した。チャット上の完了報告SHAを推測したものではない。
[差分報告](track_a_environment_sync_differences.md)、[統合候補](track_a_shared_research_status_candidate.md)、[参照一覧](track_a_local_sync_reference_index.md)を併読する。

## 調査範囲と再現可能性

既存rootで151登録worktree・154 local branchをmetadata棚卸し。Track A、明示的に関連するrepresentation履歴、旧環境exportの既知保存先から15 worktreeを選択した。
研究資料265ファイル実体、内容93種類を確認。全worktree・全GitHub履歴・home全域の網羅的監査ではない。
小さい文書/JSON/既知ZIPのbytes hashとGit treeを照合した。NPZ/NPY/pickleをdecodeせず、runtimeや科学sourceを起動しない。
サーバー主要証拠19件と入口・索引等を合わせ37件をGitHubから取得し、bytes・Git blob・SHA-256を照合した。
[公開metadata](../../artifacts/resource_applicability/track_a_cross_environment_sync_local/2026-10-14/local_inventory_v1.json)にはpath、HEAD、dirty/untracked、照合先commit/blobを保存し、[取得照合](../../artifacts/resource_applicability/track_a_cross_environment_sync_local/2026-10-14/github_reference_verification_v1.json)で固定GitHub内容へ遡れる。
非公開baselineには既存dirtyファイルのhashとGit状態を保存。公開版は絶対ローカルパス、資格情報、private設定を含めない。

## Repository・branch・作業状態

本研究repositoryのoriginは `HIROMU1015/Partially-Randomized-Trotter`。remote設定の追加・変更は行わない。
新しい資料専用branch `track-a-cross-environment-sync-local-20261014` はサーバー固定commitを親にする。既存root/研究branchをcheckout・pull・merge・rebaseしない。
科学source/rawをcheckoutしない独立Git metadata＋docs専用worktreeで作業し、既存rootのbranch ref・remote設定を維持する。
表の未追跡値は `git status` のpath group数であり、配下全ファイル数ではない。比較時点を固定し、branch名の不在をcommit未公開と解釈しない。

|ローカル識別子|HEAD短縮|fresh remoteとの関係|tracked差分 / untracked path group|
|---|---|---|---|
| `LOCAL_ROOT` | `e098c54c78f5` | DIFFERENT_REMOTE_HEAD | 27 / 38 |
| `pr2-v4-s2-parallelization-20260928` | `b2e1bf65e218` | REMOTE_HEAD_MATCH | 21 / 19 |
| `representation-attribution-pair-20261010` | `f98050e9402e` | REMOTE_HEAD_MATCH | 0 / 0 |
| `representation-attribution-review-intake-20261011` | `f98050e9402e` | NO_SAME_BRANCH_BUT_HEAD_ON_OTHER_REMOTE | 7 / 3 |
| `representation-construction-comparison-20261010` | `c24dcede10ed` | REMOTE_HEAD_MATCH | 0 / 0 |
| `representation-exploration-20261010` | `39345830ddfe` | REMOTE_HEAD_MATCH | 0 / 0 |
| `rte-basis-cost-attribution-diagnostic-20261013` | `403204edf22e` | REMOTE_HEAD_MATCH | 0 / 0 |
| `track-a-ax2a-preparation-20261009` | `b2e1bf65e218` | NO_SAME_BRANCH_BUT_HEAD_ON_OTHER_REMOTE | 8 / 31 |
| `track-a-ax2b-h4-fingerprint-v5-20261009` | `b2e1bf65e218` | NO_SAME_BRANCH_BUT_HEAD_ON_OTHER_REMOTE | 8 / 62 |
| `track-a-ax2b-h4-memory-v4-20261009` | `b2e1bf65e218` | NO_SAME_BRANCH_BUT_HEAD_ON_OTHER_REMOTE | 8 / 45 |
| `track-a-ax2b-h4-post-review-20261010` | `054bb7b2b2ca` | REMOTE_HEAD_MATCH | 9 / 16 |
| `track-a-ax2b-h4-v4-execution-20261009` | `b2e1bf65e218` | NO_SAME_BRANCH_BUT_HEAD_ON_OTHER_REMOTE | 8 / 48 |
| `track-a-ax2b-h4-v5-execution-20261010` | `b2e1bf65e218` | NO_SAME_BRANCH_BUT_HEAD_ON_OTHER_REMOTE | 8 / 65 |
| `track-a-h6-scientific-review-publication-20261013` | `0dd5c3d204da` | REMOTE_HEAD_MATCH | 0 / 0 |
| `h6-frozen-basis-source-20261012` | `054bb7b2b2ca` | NO_SAME_BRANCH_BUT_HEAD_ON_OTHER_REMOTE | 0 / 0 |

rootは `all-r-coherent-opt2-reoptimization`、HEAD `e098c54c78f589055082f9cfc2b13de50c90ca94`、同名remote `f29eafbf4412193e8e8660d9f051889c2cba684b` に2 commit ahead / 0 behind。
ただしその2 commit（e098c54…、33a1c0d…）はサーバー固定commitの祖先として既に公開されている。未pushという判定は同名branch差分に限定される。
古いAX-2作業branchのb2e1bf…も別公開branchに存在する。ローカルdirty差分は固定sourceとして採用せず、未追跡というだけで未公開科学結果と呼ばない。
representation系列は関係を確認するためのbackgroundのみ。本taskでTrack Aの比較結果へ統合しない。

## 保存資料と重要なローカル履歴

M1/M2、PM-0/1/2、原稿v0.1、AX-0/AX-1、H4 correctness、H6 STOPの固定資料が残る。
その多くはGitHub公開archiveと同じbytesであり、今回再commitしない。[履歴・一次artifact索引](track_a_local_sync_reference_index.md)から固定commitへ参照する。
ローカルにあるGPT reviewの同一公開copyと、未採択のworking noteを分けた。M2の結果前固定5構成transferと、PM-2のPOSTHOC精度会計を新しい独立検証に合算しない。
古いH4 developmentのB2点最小主張は元比較集合の履歴であり、B0+I追加後の新集合へ無条件に移さない。
AX-1b後のGPT判断はRQ-R主、RQ-P1補助、RQ-P2未達成として分離する。古い拡張planを現行の科学認可へ昇格させない。

## LOCAL_ONLYの確認範囲と扱い

44ファイル実体・16種類の内容は、今回調べた公開tree/主要証拠/公開archive/選定remote HEADではbytes一致先が見つからなかった。
サーバーfilesystemや全GitHub履歴に存在しないことの証明ではない。詳細hashと扱いは[一覧JSON](../../artifacts/resource_applicability/track_a_cross_environment_sync_local/2026-10-14/local_only_materials_v1.json)。
研究ノート・planは原本を保全し、今回は摘要とhash索引だけ公開する。大容量/ZIP/private/runtimeの一括公開は行わない。

|ローカル識別子|SHA-256先頭|保全・公開判断|
|---|---|---|
| `LOCAL_ROOT/docs/research/planning/track_a/pr2_post_m2_research_redesign_20261004.md` | `0819a0f358cdbb20…` | historical governance/planning input; no automatic scientific adoption or authorization |
| `LOCAL_ROOT/docs/research/planning/track_a/pr2_post_pm1_research_redesign_20261005.md` | `d66db1a1671fb29f…` | historical governance/planning input; no automatic scientific adoption or authorization |
| `LOCAL_ROOT/docs/research/planning/track_a/pr2_post_pm2_research_redesign_20261005.md` | `815e84d32fa44c5b…` | historical governance/planning input; no automatic scientific adoption or authorization |
| `LOCAL_ROOT/docs/research/planning/track_a/track_a_extension_detailed_plan_v1_20261008.md` | `1f33939815cd705b…` | historical governance/planning input; no automatic scientific adoption or authorization |
| `LOCAL_ROOT/docs/progress/overall/2026-10-13_PR研究の横断的進捗調査.md` | `3274a5ac922344e0…` | cross-track progress history, Track A synopsis only; later server evidence is separately indexed |
| `LOCAL_ROOT/AI_AGENT/research_project_instructions_gpt_codex_review_approval_2026-10-12.md` | `b00598003bf87711…` | historical governance/planning input; no automatic scientific adoption or authorization |
| `LOCAL_ROOT/h6-old-environment-exporter-v1.zip` | `74736b61df9828a8…` | archive retained locally; exact source packet is already public at b53271f59609f9b633f1771a85c7ca7181fdba8a |
| `track-a-ax2b-h4-v5-execution-20261010/docs/research/研究ノート/2026-10-09_track_a_ax2a_preparation.md` | `67ec8483abd15cc3…` | uncommitted dated working note; source/result identity remains in fixed public records |
| `track-a-ax2b-h4-v5-execution-20261010/docs/research/研究ノート/2026-10-09_track_a_ax2b_h4_pilot.md` | `5cf8481f2c4788c7…` | uncommitted dated working note; source/result identity remains in fixed public records |
| `track-a-ax2b-h4-v5-execution-20261010/docs/research/研究ノート/2026-10-09_track_a_ax2a_native_df.md` | `6bd0eefdb151aeae…` | uncommitted dated working note; source/result identity remains in fixed public records |
| `track-a-ax2b-h4-v5-execution-20261010/docs/research/研究ノート/2026-10-09_track_a_ax2b_h4_runner.md` | `af4d22c699d4d928…` | uncommitted dated working note; source/result identity remains in fixed public records |
| `track-a-ax2b-h4-v5-execution-20261010/docs/research/研究ノート/2026-10-09_track_a_ax2b_h4_pilot_v4.md` | `c6ffa53e6beea8d5…` | uncommitted dated working note; source/result identity remains in fixed public records |
| `track-a-ax2b-h4-v5-execution-20261010/docs/research/研究ノート/2026-10-09_track_a_ax2b_h4_memory_v4.md` | `7e520088545b32ae…` | uncommitted dated working note; source/result identity remains in fixed public records |
| `track-a-ax2b-h4-post-review-20261010/docs/research/研究ノート/2026-10-10_track_a_ax2b_h4_post_review.md` | `91f3f0fdc12edbcb…` | uncommitted dated working note; source/result identity remains in fixed public records |
| `track-a-ax2b-h4-v5-execution-20261010/docs/research/研究ノート/2026-10-10_track_a_ax2b_h4_pilot_v5.md` | `faa62735db7b9bdf…` | uncommitted dated working note; source/result identity remains in fixed public records |
| `OLD_ENV_EXPORT/export_terminal.json` | `88e07438a7dcccc4…` | exact terminal copy published; other runtime/bundle remains environment specific |

旧export terminalの330 bytesだけは、機密/絶対パスがないことを確認し[完全同一copy](../../artifacts/resource_applicability/track_a_cross_environment_sync_local/2026-10-14/old_environment_export_terminal_original.json)を追加した。
原本SHA-256 `88e07438a7dcccc4688338e6bd48d7548a8847f1942720ae5c38ef97cdfbd713`、EXPORT_STOP、FROZEN_PARTIAL_450_REPRODUCTION_MISMATCH。
source packet b53271f…と入力snapshot054bb7…の来歴は[provenance](../../artifacts/resource_applicability/track_a_cross_environment_sync_local/2026-10-14/old_environment_export_provenance_v1.json)。再生成eventや実際のdigestは未保存、実呼出し数はunknownでありreserved trajectory1/occurrence8をactualへ置換しない。
7準備/codec gate通過という旧報告はsource制御順からの推論であり、保存raw配列で再現した事実ではない。

## 意図的に統合しないもの

サーバーruntime/checkpoint/ledger、H8 live process、venv・環境固有設定、秘密情報、大きい入力/結果arrayは同期しない。
兄弟repositoryはremote identityのread-only確認まで。関連性が未確定のHIROMU1015配下3件は除外し、非HIROMU1015/quration等は内容を統合せずcommit/pushしない。
Track B/N/Qの独立結果も対象外。既存scientific manifestやVALIDATION_STATUSは変更しない。
新科学計算0、H8変更0、新grant0。資料整備後mandatory STOP。
