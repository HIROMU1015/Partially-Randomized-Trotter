# A2 Phase B契約固定：引継ぎ

[正式条件](formal_contract.md)／[machine C](execution_contract_C.json)／[採択根拠](adoption_record.json)／[固定identity](identity_manifest.json)／[静的検査](static_inspection.json)／[保全](preservation_audit.json)／[公開照合](publication_receipt.json)。

現在は `CONTRACT_C_ADOPTED_PHASE_B_NOT_AUTHORIZED`。ユーザー指定の全科学・算術・保存条件を固定し、UNSETなし。SOURCE S、旧Phase A草案・人工test・入力原本と他研究は変更していない。SOURCE開発・追加科学レビュー・契約の再設計は不要。数値correctness・実入力の証拠取得費用は未検証で、初回の成功を保証しない。

次は**ユーザー**が、公開された契約Cの固定commit／raw SHA256、SOURCE S `8d483f9bfa5983bb52fef55887e2f9f73a790511`、input manifest IDを対象に、Phase Bの一回実行を別途明示承認するか判断する。今回の採択・公開を実行承認に転用してはいけない。

その承認を受けた場合だけ、**Codex**が専用authorizationと外部pinsを固定し、canonical checkout `/tmp/prt-qc-a2-pilot-phase-a-20261018` のorigin・SOURCE bytes・入力原bytes・canonical ledger・既存claim非衝突・実空き資源と非干渉を実行直前に再確認して、既存のguard付きone-shot pipelineへ進む。管理用checkoutへledgerを移したり、Cを再serializeして消費済みclaimを回避しない。新しい準備フェーズや永続領域への移動を場所だけを理由に要求しない。

guard拒否はinput IO/claim前にSTOP。claim後は入力のfinite/Hermiticity不合格、証拠不合格、資源不足、中断、保存失敗でも自動retryなし。取得できた証拠・partial・failure・hashを保全し、公開または巨大原本の安全な保全とmanifest公開後、必ずmandatory STOP。256 bisectionsで幅未達でも設定を変えない。

ground accessは条件付き、trialはU専用。元分子保証、state preparation/native/総資源費用、研究GO/NO-GOは対象外。A2 `NOT_ADOPTED`、A1 `CLOSED`、K1-F未承認・未実行、SPEC-08未承認を維持する。今回の科学計算・係数decode・certificate・authorization・grant・marker・claim消費は全て0件。
