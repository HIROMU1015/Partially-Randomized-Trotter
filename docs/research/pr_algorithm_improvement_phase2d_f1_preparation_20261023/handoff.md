# F1-PRE引継ぎ・mandatory STOP

状態は `READY_FOR_F1_EXECUTION_APPROVAL`。専用環境、保存B lawを復元するadapter、固定件数の遅延case計画、出力／STOP仕様は準備済み。追加された正式共通指示も取得して照合した。科学的PASS、grant、実行権限は存在しない。

公開先：`HIROMU1015/Partially-Randomized-Trotter`。
branch：`research/pr-algorithm-improvement-phase2d-f1-preparation-20261023`。
起点：`32c76bc6392ba277ce9fdd8d4bc54ffb5339b767`。
公開payload／読み戻し結果は、公開後に本書末尾へ記録する。

次はユーザーが、[技術記録](TECHNICAL_PREPARATION.md)の価格binding移管・未資格interval fallbackを含む実行範囲を承認するか判断する。未資格のまま認証価格選択やfull P0 qualificationを完了扱いしない。明示承認を受けたCodexが下記の最終bindingとpreflightを行い、その承認範囲だけを実行する。

**F1を実行する場合に固定するもの**（今回発行していない）：

1. 本branchの最終公開SOURCE commitとmanifestの全SOURCE hash。独立したfrozen checkoutを作り、旧B SOURCEは使い回さない。
2. 既存private handoffのB rawをread-only照合。physical/catalog、4arm、5law、F1 bundleを別のprivate packetへbyte-identicalにcopyし、manifestのhashを照合する。Bの係数・lawを再生成しない。
3. F1専用interpreter、lock、全dependency version、private environment_file_hashes。別machineで再構築した場合はlocal path／native identityを再bindingする。これは科学grantの代わりにならない。
4. privateのordinal-only planをcase_inventory.jsonとしてbyte固定。SOURCE内iter_cases()と一致し、B件数と同じ16familyであることを旧gateで照合する。event payload解決は認可消費後。
5. 最終contract：stage F1、context T=1/5,q2,r2,K2、adapter F1_SAVED_B_V1、price_interval_policy UNQUALIFIED_EXPLICIT_FALLBACK_NO_CERTIFIED_SELECTION、interval_provider_qualification空、manifestのlogical_case_counts、fixture_bundle_path、input_files、source_files、source_closure_sha256、environment_lock_sha256、source_commit、interpreterをbindingする。今回のmanifest／private contract.planはDRAFT_NOT_AUTHORIZATIONである。
6. **実際のユーザー明示承認の原文**を保存し、旧authorization.pyが要求するauthorization.jsonのapproval／contract／inventory／source digest、fresh output namespace、no-retry、mandatory STOPを一つの新規one-shot grantへ固定する。旧B grant／marker／STOPを参照許可や再利用許可に変えない。
7. 実行直前にCPU／RAM／disk／他jobとの非干渉を再確認し、新しいpreflightで全identityを確認する。技術的な性能設定を実機資源へ合わせる。科学的条件やcase数を変えない。
8. 承認後だけ、専用Pythonを `-I -B` 付きで新runnerへ渡す。`--packet`、`--source-root`、`--output` はすべて新しい固定private pathを指定する。成功・失敗を問わずSTOPし、結果と実績call数を報告する。F2・追加科学test・レビューは別承認。

private handoffには独立clone、F1 venv、wheelhouse、lock、ID plan、環境file identity、B原本参照、原workspace保全snapshotを保存した。非公開raw、ユーザー承認原文、grant／marker、内部absolute pathはGitHubに載せない。

今回は研究runner・旧科学20test・追加science adapterを実行していない。通常import／signature／stdlib管理testを科学的PASSとして使わない。公開・照合後は **MANDATORY_STOP_AFTER_F1_PREPARATION**。
