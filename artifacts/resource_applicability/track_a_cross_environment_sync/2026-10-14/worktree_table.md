# 登録worktreeの読取snapshot

2026-10-14。絶対保存先は公開しない。branchは可動参照、HEADは固定commit。branch remoteなしとcommit未公開を区別する。

|識別名|branch|HEAD（短縮）|remote HEAD（短縮）|branch公開状態|commit remote到達性|tracked変更/untracked|
|---|---|---|---|---|---|---:|
|partially-randomized-trotter|main|0babed07006c|0babed07006c|一致|確認|958/0|
|cross-environment-sync-server-20261014|track-a-cross-environment-sync-server-20261014|47dc17a83e2f|—|同名remoteなし|確認|0/0|
|h4-approved-relaunch-20261009|track-a-h4-approved-relaunch-20261009|ab86c31155c6|—|同名remoteなし|確認|0/0|
|h4-authorized-launch-20261009|track-a-h4-authorized-launch-20261009|0bfeb3e6da83|—|同名remoteなし|確認|0/0|
|h4-b0i-compile-preparation-20261013|track-a-h4-b0i-compile-preparation-20261013|34ad305d55a7|34ad305d55a7|一致|確認|0/0|
|h4-b0i-manuscript-v04-20261014|track-a-h4-b0i-manuscript-v04-20261014|47dc17a83e2f|47dc17a83e2f|一致|確認|0/0|
|h4-b0i-map-parallel-run01-20261014|track-a-h4-b0i-map-parallel-run01-20261014|4bc56ea6eecf|—|同名remoteなし|確認|0/0|
|h4-b0i-map-parallel-run02-20261014|track-a-h4-b0i-map-parallel-run02-20261014|98cdb35cd084|98cdb35cd084|一致|確認|0/0|
|h4-b0i-map-preparation-20261014|track-a-h4-b0i-map-preparation-20261014|eff4a73f5153|eff4a73f5153|一致|確認|0/0|
|h4-b0i-pilot6-20261014|track-a-h4-b0i-pilot6-20261014|c377b074ab6c|c377b074ab6c|一致|確認|0/0|
|h4-cleanup-esrch-fix-20261009|track-a-h4-cleanup-esrch-fix-20261009|6f8e95ad0f69|6f8e95ad0f69|一致|確認|0/0|
|h4-common-panel-h8-preflight-20261013|track-a-h4-common-panel-h8-preflight-20261013|217c971dd537|217c971dd537|一致|確認|0/6|
|h4-encoding-speedup-20261010|track-a-h4-encoding-speedup-20261010|5706ab29e0ed|—|同名remoteなし|確認|0/0|
|h4-entrypoint-cache-fix-20261009|track-a-h4-entrypoint-cache-fix-20261009|461ddb2c2d61|—|同名remoteなし|確認|0/0|
|h4-gaussian-structure-proposal-20261010|track-a-h4-gaussian-structure-proposal-20261010|56c07727743b|—|同名remoteなし|確認|0/0|
|h4-geometry-analysis-20261013|track-a-h4-geometry-analysis-20261013|c81f7326dde3|c81f7326dde3|一致|確認|0/0|
|h4-geometry-results-20261013|track-a-h4-geometry-results-20261013|7c4483378047|7c4483378047|一致|確認|0/0|
|h4-geometry-technical-integration-20261013|track-a-h4-geometry-technical-integration-20261013|f6b0cb5e2dda|f6b0cb5e2dda|一致|確認|0/0|
|h4-host-pressure-grace-fix-20261010|track-a-h4-host-pressure-grace-fix-20261010|fa66bfa01ecb|—|同名remoteなし|確認|0/0|
|h4-host-pressure-policy-fix-20261010|track-a-h4-host-pressure-policy-fix-20261010|08beb875ef23|—|同名remoteなし|確認|0/0|
|h4-inverse-instruction-speedup-20261011|track-a-h4-inverse-instruction-speedup-20261011|28866c82ed5b|—|同名remoteなし|確認|0/0|
|h4-library-cache-fix-20261009|track-a-h4-library-cache-fix-20261009|f25284607fd5|—|同名remoteなし|確認|0/0|
|h4-lightweight-speedup-20261009|track-a-h4-lightweight-speedup-20261009|5bb241125717|—|同名remoteなし|確認|0/0|
|h4-monitor-fix-a-20261007|track-a-h4-new-server-monitor-fix-a-20261007|b2a5ad89e8b3|—|同名remoteなし|確認|11/18|
|h4-monitor-fix-a-20261007-02|track-a-h4-new-server-monitor-fix-a-20261007-02|3e0494e1e8a0|3e0494e1e8a0|一致|確認|0/0|
|h4-monitor-preparation-20261007|track-a-h4-new-server-monitor-preparation-20261007|814212cb1c25|814212cb1c25|一致|確認|0/0|
|h4-native-proof-seal-20261009|track-a-h4-native-proof-seal-20261009|5de745199748|5de745199748|一致|確認|0/0|
|h4-per-attempt-budget-20261009|track-a-h4-per-attempt-budget-20261009|91112b6cac2f|—|同名remoteなし|確認|0/0|
|h4-prelaunch-preparation-20261007|track-a-h4-prelaunch-preparation-20261007|798270161d85|798270161d85|一致|確認|0/0|
|h4-production-resume01-20261011|track-a-h4-production-resume01-20261011|d0b4e1734b06|—|同名remoteなし|確認|0/0|
|h4-production-resume02-20261011|track-a-h4-production-resume02-20261011|81c78b366d2c|81c78b366d2c|一致|確認|0/0|
|h4-production-run03-20261009|track-a-h4-production-run03-20261009|a3248055ec8e|—|同名remoteなし|未確立|0/0|
|h4-production-run04-20261009|track-a-h4-production-run04-20261009|cde398580dda|—|同名remoteなし|確認|0/0|
|h4-production-run05-20261009|track-a-h4-production-run05-20261009|34c3fd46cd5d|—|同名remoteなし|確認|0/0|
|h4-production-run06-20261010|track-a-h4-production-run06-20261010|00cc443636be|—|同名remoteなし|確認|0/0|
|h4-production-run07-20261010|track-a-h4-production-run07-20261010|84cea869cfb0|—|同名remoteなし|確認|0/0|
|h4-production-run08-20261010|track-a-h4-production-run08-20261010|8ab82ef6f600|—|同名remoteなし|確認|0/0|
|h4-production-run09-20261010|track-a-h4-production-run09-20261010|82d1f43e97ee|—|同名remoteなし|確認|0/0|
|h4-production-run10-20261010|track-a-h4-production-run10-20261010|95c2269263c9|95c2269263c9|一致|確認|0/0|
|h4-receipt-final-review-20261008|track-a-h4-receipt-final-review-20261008|992c09d60c80|992c09d60c80|一致|確認|0/0|
|h4-receipt-reseal-20261009|track-a-h4-receipt-reseal-20261009|c81144bb6678|c81144bb6678|一致|確認|0/0|
|h4-resume-preparation-20261011|track-a-h4-resume-preparation-20261011|99249f5c032a|—|同名remoteなし|確認|0/0|
|h4-resume02-preparation-20261011|track-a-h4-resume02-preparation-20261011|37e8895b9e2f|—|同名remoteなし|確認|0/0|
|h4-run07-stop-cause-audit-20261010|track-a-h4-run07-stop-cause-audit-20261010|b1c2e746eee9|—|同名remoteなし|確認|0/0|
|h4-worker-error-fix-20261009|track-a-h4-worker-error-fix-20261009|00c6ad5fa7c7|—|同名remoteなし|確認|0/0|
|h4-worker-memory32-fix-20261010|track-a-h4-worker-memory32-fix-20261010|4fa8f9c48a14|—|同名remoteなし|確認|0/0|
|h6-conditional-s2-pilot-preparation-20261013|track-a-h6-conditional-s2-pilot-preparation-20261013|e0e6473d7047|e0e6473d7047|一致|確認|0/404|
|h6-confirmation-preparation-20261012|track-a-h6-confirmation-preparation-20261012|3b6d733a1072|—|同名remoteなし|確認|0/7707|
|h6-exit-observation-fix-20261012|track-a-h6-exit-observation-fix-20261012|ce95a13902ed|—|同名remoteなし|確認|0/0|
|h6-four-worker-preparation-20261012|track-a-h6-four-worker-preparation-20261012|8eaea37d6ab2|—|同名remoteなし|確認|0/40|
|h6-hash-reuse-eight-worker-20261012|track-a-h6-hash-reuse-eight-worker-20261012|321cd8c30661|321cd8c30661|一致|確認|0/4767|
|h6-manuscript-h8-plan-20261013|track-a-h6-manuscript-h8-plan-20261013|653d64bedba7|653d64bedba7|一致|確認|0/0|
|h6-matched-server-preparation-20261011|track-a-h6-server-migration-preparation-20261011|76a8ac62bd08|e51f19b649a7|HEAD差分|確認|0/3|
|h6-native-dependency-fix-20261012|track-a-h6-native-dependency-fix-20261012|169d83c91b05|—|同名remoteなし|確認|0/67|
|h6-native-reproduction-fix-20261012|track-a-h6-native-reproduction-fix-20261012|1293109e2c27|1293109e2c27|一致|確認|0/2|
|h6-parallel-preparation-20261012|track-a-h6-parallel-preparation-20261012|6f7bdbc54122|—|同名remoteなし|確認|0/0|
|h6-rare-order-cost-analysis-20261013|track-a-h6-rare-order-cost-analysis-20261013|eee1343f6e99|eee1343f6e99|一致|確認|0/0|
|h6-roundoff-execution-20261012|track-a-h6-roundoff-execution-20261012|19da6a6de2a3|—|同名remoteなし|確認|0/392|
|h6-roundoff-resume01-20261012|track-a-h6-roundoff-resume01-20261012|df99d6ee2894|df99d6ee2894|一致|確認|0/6575|
|h8-duplicate-work-optimization-20261013|track-a-h8-duplicate-work-optimization-20261013|f8c503943925|f8c503943925|一致|確認|0/0|
|h8-fast-retry21-20261013|track-a-h8-fast-retry21-20261013|a55da5c2b012|a55da5c2b012|一致|確認|0/0|
|h8-remaining-fast-handoff-20261013|track-a-h8-remaining-fast-handoff-20261013|adbd67661ed3|adbd67661ed3|一致|確認|0/0|
|h8-resource-map-production-20261013|track-a-h8-resource-map-production-20261013|eaeeefc38be4|eaeeefc38be4|一致|確認|0/0|
|h8-resource-map-resume01-20261013|track-a-h8-resource-map-resume01-20261013|96b9560e6bc0|96b9560e6bc0|一致|確認|0/0|
|h8-resource-map-resume02-20261014|track-a-h8-resource-map-resume02-20261014|38d386c87322|38d386c87322|一致|確認|0/0|
|h8-saved-df-state-preflight-20261013|track-a-h8-saved-df-state-preflight-20261013|0f96a2e45ba5|0f96a2e45ba5|一致|確認|0/0|
|h8-signal-compile-preflight-20261013|track-a-h8-signal-compile-preflight-20261013|22197f9901c3|22197f9901c3|一致|確認|0/0|

full40文字SHAと少数fileのblob/SHAは`worktree_inventory.json`。新しい同期worktree自身のHEADは作成時の起点47dc17a8であり、成果commitは最終公開報告に従う。

## Dirty worktreeの分類

|識別名|status件数|path family件数|
|---|---|---|
|partially-randomized-trotter|{"D ": 958}|{".gitattributes": 1, ".github": 1, ".gitignore": 1, ".vscode": 1, "AGENTS.md": 1, "BentoSlide構成案.md": 1, "BentoSlide構成案_回路コスト検証.md": 1, "Evaluation of gate numbers for ground state energy calculations using higher-order.pdf": 1, "Greatly-improved-higher-order-product-formulae-for-quantum-simulation.pdf": 1, "PROJECT_MAP.md": 1, "Partial Randomized Study Protocol.md": 1, "Phase estimation with partially randomized time evolution.pdf": 1, "README.md": 1, "README_partial_randomized_pf.md": 1, "VALIDATION_STATUS.md": 1, "abe_trotter_project.ipynb": 1, "artifacts": 671, "codex-inst.md": 1, "current-bentoslide.pdf": 1, "docs": 58, "notes": 3, "pyproject.toml": 1, "requirements-gpu.txt": 1, "requirements.txt": 1, "scripts": 54, "src": 101, "tests": 50}|
|h4-common-panel-h8-preflight-20261013|{"??": 6}|{"artifacts": 6}|
|h4-monitor-fix-a-20261007|{" M": 11, "??": 18}|{"PROJECT_MAP.md": 1, "VALIDATION_STATUS.md": 1, "artifacts": 18, "docs": 6, "scripts": 1, "src": 2}|
|h6-conditional-s2-pilot-preparation-20261013|{"??": 404}|{".server-preparation": 1, "artifacts": 403}|
|h6-confirmation-preparation-20261012|{"??": 7707}|{".server-preparation": 1, "artifacts": 7706}|
|h6-four-worker-preparation-20261012|{"??": 40}|{".server-preparation": 1, "artifacts": 39}|
|h6-hash-reuse-eight-worker-20261012|{"??": 4767}|{".server-preparation": 1, "artifacts": 4766}|
|h6-matched-server-preparation-20261011|{"??": 3}|{".server-preparation": 2, "artifacts": 1}|
|h6-native-dependency-fix-20261012|{"??": 67}|{".server-preparation": 1, "artifacts": 66}|
|h6-native-reproduction-fix-20261012|{"??": 2}|{"artifacts": 2}|
|h6-roundoff-execution-20261012|{"??": 392}|{".server-preparation": 1, "artifacts": 391}|
|h6-roundoff-resume01-20261012|{"??": 6575}|{".server-preparation": 1, "artifacts": 6574}|

全差分と多数のruntime file metadataはprivate auditとして保持し、このtableへ原本を転載しない。既存dirty/indexは変更していない。
