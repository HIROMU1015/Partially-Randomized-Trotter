# Track B W2c-B3: 正式採択・一回correctness実行

ユーザーの明示承認に基づく最大512件correctness-only。採択条件と原文は [adoption.md](adoption.md)、[user_approval.json](user_approval.json)。実行SOURCEは基点bf86からactivationと必要最小限の接続・検査差分を加えて固定する。科学Coreは固定commitと完全一致し、旧adapter/driver/Global Bundle/F01/Storage V2も変更しない。差分は [activation_source_delta.json](activation_source_delta.json)。

変更影響の人工検証は56件PASS、正常/P1 angle失敗/receipt close失敗の3 integrationが期待分類、semantic mutation20件を全拒否。主148件の全面再実行は行わない。記録は `focused_artificial/`。`raw_artificial.tar.gz` が人工rawの公開用完全archiveであり、人工値を実科学値とは扱わない。

実行用のSOURCE inventory、approved manifest、preflight、および実行後のEvidence Packetはこのdirectoryに追記する。再実行コマンドの推奨ではない。`authorized_runner.py` の execute は新scopeを一回だけ消費し、retry/resumeしない。科学実行後は結果の成否にかかわらずmandatory STOP。次の担当はユーザー・GPT。

実行後：**512/512技術的完了、全cell coverage、retry0、mandatory STOP**。詳細は [Q1–Q10結果・引き継ぎ](handoff.md)、[保存値検査](saved_evidence_check.json)、[raw inventory](raw_inventory.json)、[metadata残件](observed_metadata_issues.json)。`artificial_only` の集約report既定値の矛盾は原本のまま保持する。
