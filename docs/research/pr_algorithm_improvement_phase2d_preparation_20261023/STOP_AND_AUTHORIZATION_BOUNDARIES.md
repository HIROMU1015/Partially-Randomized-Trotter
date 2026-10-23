# 承認境界とmandatory STOP

## 現在：P0だけ承認済み

静的SOURCE読解、公開固定資料取得、文書・管理JSON・verifier、専用branchへの公開・bytes/hash/blob/link照合まで。今回の文書は `DRAFT_NOT_AUTHORIZATION`。科学grant、ledger、consumed markerを作成しない。旧grant／STOP／失敗runを再利用しない。

新規実装、production SOURCE変更、SOURCE import、fixture／数値行列／状態／Pauli係数／科学input生成、sampling／event列挙の実行、trajectory生成、回路build／compile、simulation、数値再解析、新しい非自明な証明は今回0件。GPT/Claude詳細レビュー、旧C1／RQ-04／他trackを開始しない。

## 次の第1承認を分ける必要

ユーザー指示§7はF1承認前のSOURCE/input/call固定を要求し、同時にP0での実装・fixture生成を禁止している。新SOURCEが未実装のため、このP0だけで実行SOURCE hashを確定することはできない。指示で許可された「実装準備と科学実行の異なる承認境界」を次のように具体化する。

| 提案段階 | 必要な明示承認／対象 | 終了点 |
|---|---|---|
| I：実装のみ | 新prototype／runner／test定義／索引接続。科学module import・数値test実行・input生成なし | 新SOURCE commit／closure候補を提示してSTOP |
| B：限定input binding | IのSOURCEを固定後、physical fixture＋catalogの2input case、4arm catalog/law/recipe identity、正確件数・保存計画の生成だけ。build/compile／motif／full validationなし | exact input hashes／L／case件数／最終契約を提示してSTOP |
| F1：登録検証 | SOURCE/input/contract/call inventory確定後、ユーザーが登録F1検証を明示承認 | 成否にかかわらずmandatory STOP |
| F2：別承認 | F1結果をユーザー・GPTが確認後、別SOURCE/input/context/call/authorizationを確定して明示承認 | 成否にかかわらずmandatory STOP |

IとBをまとめた承認をユーザーが選ぶこともできるが、Bの科学生成範囲を具体的に含める必要があり、「準備を続ける」だけをscience許可に読み替えない。Bで得た入力caseをF1台帳へbinding済みとして参照し、同じ生成をF1で重複実行しない。I/Bも**今回未承認**。これは新たなgrantではなく、依存関係と不足を示す提案である。

## 将来のone-shot境界

実行契約が承認されてから、承認文、SOURCE closure、input、case inventory、output namespace、fresh authorization／ledger／markerを固定する。F1とF2は別物。markerやledgerのschema設計を記載することと実物を発行することを区別し、今回実物を発行しない。

各logical caseとunique workload keyを登録し、claimed/in-progress/completed/failedの遷移をatomicに行う。失敗attemptを未実行へ戻さず、同じ科学runを自動retry／resumeしない。通信・文書取得の再取得と科学callの再実行を混同しない。compiler/buildの失敗もattemptとして記録する。障害回収は失敗記録と安全停止までで、再開は新たな判断へ返す。

## 即STOP条件

- H identity、finite corrected mean、control0/1／ancilla phase、order、sign、actual adjointの不一致。
- 中間sector投影、identity/scalar policy移動、同じHを破るscreening／split変更。
- SOURCE／input／env／compiler identity違反、未binding hash、未登録call、件数超過、未承認のprecision／r/K／threshold／seed変更。
- 原本、旧結果・レビュー・SOURCE・grant・ledger・marker・STOP・他worktree/jobの保全違反。
- coverage不足、NaN/Inf、資格条件不成立、layoutや価格providerの不明、資源不足時の無断MC化。

既存のgrant／契約の資源条件を変更しない。CPU等に新たな恣意的capを作らず、利用可能資源と非干渉で構成を決める。共有環境の変更はしない。

P0公開後、I/B後、F1後、F2後はそれぞれSTOP。次の段階への許可を前段階の成功・文書公開から推定しない。研究上の性能GO、新規性GO、投稿GOは本契約の成果ではない。
