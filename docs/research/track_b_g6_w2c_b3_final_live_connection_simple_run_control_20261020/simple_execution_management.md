# 簡易実行管理

承認者はユーザー本人。準備了承と実行承認を分離し、今回有効な実行許可は0。
従来の外部issuer/keys/revocation/authorization server等は新scopeの必須工程から外した。
旧SOURCE・claim・STOPの契約や記録は変更していない。変更は今回§5の方針に基づく候補であり、正式な512実行契約採択はユーザー・GPTへ戻す。

run_control.SimpleStoreは所有する専用rootだけを使う。現在作成できるのは人工専用prefixのstoreとMockApproval。
実用rootではpermit gateが先に拒否する。実storeへの接続・許可生成は今回0。
同じRun IDとscopeの両方をO_EXCL intentで予約し、SQLite UNIQUE/BEGIN IMMEDIATE/FULL/WALで記録する。
一方のintentだけ、空/partial intent、commit/ACK不明、process死でもspent扱いで、削除・refundしない。
NOT_STARTED→START_RESERVED/SPENT→COMPLETED/STOPPED/UNKNOWN。STOP/UNKNOWN/COMPLETED後ENTER禁止。
COMPLETEDは512完成、storage close、receipt/delivery/caller証拠確認後だけ。coverage不足でも固定Nを完了した実行状態と科学的判定は分ける。
STOP markerのclaimは常にspentで、COMPLETEDへの最終記録も再実行権限にならない。

12process競合one winner、同Run別scope/同scope別Run拒否、3crash点、partial intent、STOP/UNKNOWN、receipt一回、symlink拒否を人工試験した。
file/directory fsyncとSQLite FULLを要求した事実を記録し、物理power-loss完全耐久性や悪意あるownerの改ざん防止は主張しない。
専用storeとinput/provenanceの固定管理費はresult/log output capの外。科学Evidenceをstoreへ逃がす経路はなく、全trial保存はglobal cap内。


**実科学未実行、実行未承認。mandatory STOP。次の担当：ユーザー・GPT。**
