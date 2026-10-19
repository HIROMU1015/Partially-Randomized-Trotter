# 静的preflightと閉鎖gate

contract.candidateは固定科学/Adapter bytes、AST signatures、input/precision/angle、SOURCE artifact、Python executable/library hashes、worker runtime、monitoringを結合。
preflightはcandidateとmanifestを正規再生成比較し、scope/claim、seed、call/bytes、storage destination、authority、retry/resumeを一致確認する。
SOURCE固定後verify_source_commitが完全Git SHAの全pin bytesを検査する二段階seal。後続docs-only commitでも測定SOURCEを保持する。
他worktreeのHEAD/statusは観測だけで、無関係な変更を実行拒否条件にしない。

content_adopted=false / run_ready=false / contract_grants_execution=false / science_release=null /
scientific_adoption=null / artificial_only=true / mandatory_STOP=true / retry0 / resume=false。
wrong SOURCE/API/input/precision/runtime/worker/seed/budgets/contract/approval/retry/resume/storageは人工拒否検証。
production_entryとFutureScientificResolver.resolveは無条件拒否。実moduleはloader未実装のためboolean変更でも呼び出せない。
parent/spawn両方にload前guardを導入しcore load一覧・試行数を保存。driver/helperの人工importだけが許可範囲。

**未採択・実科学未実行。mandatory STOP。次の担当：ユーザー・GPT。**
