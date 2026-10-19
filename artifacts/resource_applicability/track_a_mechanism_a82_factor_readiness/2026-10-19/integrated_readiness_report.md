# A82比較説明因子のSOURCE・契約準備

**実行準備まで完了。新しい実科学factor計算0、有効grant0、mandatory STOP。説明side64/64・比較因子31/32の現状を保持。**

前段 `182d7044fb3e61d68d8c7a5dd86d84c0e961ed6c` と検証SOURCE `a8961c45d7fd0310c1e34c50ffd004dd58898247` を継承し、残るA82_OLD_PR一比較だけを扱う新SOURCE `06c94745f50345b2d93714760608d9d11b83e784` を固定した。既存31比較と同じ因子式blockを再利用し、side/core式の再呼出しを切り離した。人工38件PASS、固定旧3科学fileのbyte一致、式blockのAST・人工数値exact一致を確認した。

|成果物|内容|
|---|---|
|[契約案](factor_one_shot_contract_DRAFT.md)|一比較2軸の7因子/7logのみ、one-shot・retryなし|
|[SOURCE・入力manifest](source_input_manifest.json)|SOURCE5、保存入力14、runtime68と固定SHA|
|[output schema](output_schema.json)・[無効grant投影](one_shot_grant_INVALID_DRAFT_projection.json)|新scope専用、旧grant再使用拒否、実行未承認|
|[受入・STOP](acceptance_and_STOP.md)・[人工テスト](technical_test_report.md)|native人工E2E、欠測/改変/非有限/未承認拒否|
|[PASS後の統合計画](integration_plan_AFTER_PASS.md)|将来32比較の別記録、旧core・図・cohort保全|
|[readiness](paper_evidence_readiness.json)・[次の承認](next_approval_request.md)|現在31/32を保持、実行と公開の承認境界|

実保存PR/PFはA82の旧n32、H8 linear1Å/STO-3G/tol-only DF28/T=.8/P=0、通常signed/factor1、ε=.05、PR B2_p7_q2_2nd_r4_K2対PF B0I_p11_q1_2nd。normalization/headroom/Nstar/kappa等を再計算せず、検証済み保存結果から比較因子だけを将来算出する。H8独立n48やH4 Pを混合しない。

全保存入力のSHA、SOURCE、元core/説明/delta、保存status・PF exact copy・native終了証跡を静的照合した。旧保全363 fileは不変。新binding・runtime・正本契約は専用local auditに置き、private raw/絶対path/hash一覧はGitに入れていない。科学output先は未使用、無効grantの実CLIは入力読込前に拒否された。

今回の変更は新adapter/runner/testsと準備資料だけ。既存core32、他31比較、PR/PF side、Figure 2全数値・画像・cohort、H4 P別表、旧SOURCE/grant/ledger/marker/STOPを変更していない。H4/H6並行検証・共通protocol v1・他job/共有設定/他ユーザー/AI delegationへの操作0。利用可能資源を観測し、恣意的な固定capは追加しない。

独立branchへlocal commitする。新資料の公開承認範囲が不明のためpushなし、remote SHA照合は未実施（push対象なし）。最終evidence freeze・詳細科学レビューなし。次担当はユーザー：この固定SOURCE/契約への実行承認と公開範囲を判断する。別途実行承認後のみCodexが一回実行し、PASS後保存結果を統合して再びSTOPする。
