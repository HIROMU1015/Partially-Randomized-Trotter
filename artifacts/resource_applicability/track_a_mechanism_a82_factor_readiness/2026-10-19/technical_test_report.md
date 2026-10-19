# 技術確認・人工テスト

最終人工38件PASS（失敗/skip0）。対象はinvented scalars・fake32 archive・mock gateのみ。式blockのAST同一性と人工の既存定義結果へのexact一致、入力非変更、side/core/他比較呼出し禁止、旧欠測保全、追加target/axis拒否、不足・非有限/表現不能factor拒否、無承認/旧scope/retry拒否、SOURCE/input/契約/runtime/control SHA改変拒否を確認した。

人工native E2Eでは専用fake gateとARTIFICIAL permitを使い、親→workerの1回生成・hash受入・wait/reap、失敗時STOP・retry拒否を確認した。有効production grantや実保存科学値は人工workerへ渡していない。初回32件PASS後、最終SOURCEの受入条件と6つの改変拒否検査を含め38件を再実行した。重複分を合計しない。

固定SOURCE `06c94745f50345b2d93714760608d9d11b83e784` の5 fileをGit blobと照合。うち既存3 fileは定義SOURCE b6716396…とbyte同一。実保存入力は14 fileのSHAと既存JSONのID・status・PF exact copy・native証跡を静的照合しただけで、factor関数を呼んでいない。Python/runtime68 fileを照合。旧保全対象363 fileは準備前後にSHA一致。

無効grant案を実CLIへ渡すとNUMERICAL_AUTHORIZATION_REQUIREDでnative exit1となり、契約・科学入力・outputを開く前に拒否された。実factor式0、side再評価0、core/他31比較再評価0、新sampling/compile/signal/統計0、有効grant0、科学output未作成。

最終の保全・schema・資料対応21群PASSを[技術受入](technical_acceptance.json)に保存した。初回checkerはGitの日本語path引用表示を通常pathとして比較して停止したため、NUL区切りに修正した。初回checkerと停止記録を保全し、SOURCE・入力・科学結果には変更していない。

これらは主担当自身のlocal technical checks。独立review・immutable CI・詳細科学レビューではない。共有環境や他jobの変更、AI delegationは0。
