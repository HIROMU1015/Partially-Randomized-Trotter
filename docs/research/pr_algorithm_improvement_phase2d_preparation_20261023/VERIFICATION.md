# P0文書・公開管理検査

この検査は**文書管理だけ**。科学preflight、SOURCE import、数値検証、test実行、実装、回路build/compileではない。科学的正しさ・数値閾値の妥当性・性能・新規性を認証しない。

専用の独立checkoutでbase `ef2eb7d55ea37bf93e0b85773df20b63811d1f9b` を起点にし、新namespaceの17文書だけを追加する。SOURCE、Hamiltonian/state/DF input、raw、既存研究記録／レビュー／契約／grant／ledger／marker／STOP、他worktree/job、既存原稿を変更しない。privateレビュー原本や内部絶対path等を公開しない。

## ローカル検査範囲

standard-libraryだけの管理verifierで、必須11文書、全JSON parse、[台帳schema](CALL_INVENTORY_DRAFT.schema.json)の構造・型・必須field・false authorization、29call行の一意ID／入出力／件数式／失敗・dedup条件、未binding値null、4arm、8motif／32trajectory／96controlled build／64wrapper compile／68catalog event／136catalog buildの文書間対応を検査する。数値を実際に生成・検証する処理は含まない。

36固定資料recordのbytes／SHA256／Git blobを固定Git objectと照合。うち必須parent13文書はGitHubから再取得した内容／blobとも照合済み。参照の確認方法が固定checkout、今回のremote再取得、既存公開cacheのhash再照合のどれかを出典台帳で区別する。後続recipeのtransitive SOURCE closureが揃ったとはしない。

Markdownの相対file linkは全て存在を確認。固定外部URLは公開repo／公式tagged SOURCEを用いる。private path／credential patternを走査。baseの全tracked pathをbyte比較し、Git差分が新namespaceのA追加だけであることを検査する。元workspaceのbranch/ref/worktree/statusはprivate監査snapshotと比較し、他作業による変化を本作業でreset／cleanしない。

## 公開照合

[PUBLICATION.json](PUBLICATION.json)の `verified_payload_commit` は公開後に全fileを再取得して照合したpayloadを指す。receipt自身を更新した最終commitと、receiptが記録するpayload commitを区別し、自己参照hashを捏造しない。

公開時は専用branchのremote HEAD、baseからの変更file集合、各fileのUTF-8 bytes／SHA256／Git blob、全相対文書リンクを照合する。可能な範囲で同じ公開branchを独立Git fetchし、最終treeとbase file非変更も確認する。GitHub APIだけの照合をローカルCI成功と呼ばない。最終照合のfile数／digestはreceiptと最終報告に記録する。

今回の新規実装・科学call・生成・sampling・build・compile・証明・科学grant発行は0件。検査完了後mandatory STOP。
