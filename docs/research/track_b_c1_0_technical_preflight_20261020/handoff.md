# C1-0：GPT・ユーザー向けhandoff

**実装前の静的preflightを完了。Gate 1/2はCONDITIONAL、Gate 3はIMPLEMENTATION_REQUIRED。C1の科学的GO・新規性・投稿GOは判定しない。**

- [A：Gate 1〜3結果](README.md)
- [B：adapter・Evidence仕様草案](adapter_evidence_draft.md)
- [C：誤差・confidence・費用の接続表](error_cost_connection.md)

指定snapshotは固定Git/GitHub/既知ローカルcopyの**8,652 bytes・指定SHA256が一致**した。NPZを開かず、state・DF factorの既存digest記録と生成順prefix3のcomponent記録も特定できた。主要PR-2 SOURCEの実行時pinと監査pinのhash一致を確認した。前回の入力本体未取得という不足は、opaque bytesについて解消した。

不足は、実行可能なbasis操作/registry保存物、新exact p/λと元H_DFとの差certificate、H4のPF/provider/native/state/readout証拠、全path estimatorとfamilywise予算、非列挙の認証期待費用である。保存snapshotの一致から、basis再対角化後のgate列や物理準備費用まで認証したとはしない。

**既存技術だけで閉じ得る条件**：保存registryを取得でき、正有理入力・short step・F1/F3を満たし、全label CQ/actual-adjoint・RZ strict誤差を既存G8契約へ結べ、D1/D3の積誤差・moment/rangeを実scheduleへ適用できること。P5少数群のangleとDF instruction上界も既存部品であり、一律に新証明が必要とはしない。

**主要障害**：registryがなければbasis再構成に数値処理が必要。既存D1は参照event・native key・価格/平均/matrix取得側で全列挙する。G8 on-demandは観測keyのstrict取得を示すが全support termination/期待価格を保証しない。P5群angleやraw instruction上界だけでは、word別basis共有・strict native T・総期待費用が閉じない。

## 次の担当・必要な承認

| 担当 | 次の作業 | 承認境界 |
|---|---|---|
| ユーザー | 既存serialized registry/preparation artifactを提供するか、限定再解析を許すかを判断。費用評価を既存構造上界に限定するか期待nativeまで求めるかをGPTと決める | C1の科学的限定継続という既存方針と、新実行の許可は別 |
| Codex | 別指示後、許可された既存artifactの所在/bytes確認、または承認範囲でのbasis/入力certificate取得。次に独立sidecar・measurement bindingの実装仕様を固定 | 再解析、adapter実装/人工科学test、native取得、科学計算はそれぞれ別の明示scope承認。今回実施0 |
| GPT・ユーザー | 全誤差/対照の適用条件、必要な費用保証と追加作業の範囲を判断 | 新しい非自明な保証が不可避になった場合だけPROOF_REQUIREDとして証明作業を別承認。今回新しい定理/算法を作っていない |

提供済みの科学判断を反復する新レビューを自動開始しない。任意添付C1レビュー原本は既知の場所で未取得で、読了済み・公開済みとしない。案A/B、Full-N、追加Pilot、全round RPE、旧grant/marker/STOPへ進まない。

## 公開・保全

repository：`HIROMU1015/Partially-Randomized-Trotter`。branch：`track-b-c1-0-technical-preflight-20261020`。基点：`291e21218c1d1fde062c001a1586ba7c1bb9c988`。

新規専用directoryの4文書とprovenance/管理検証/公開receiptだけを公開する。[evidence inventory](evidence_inventory.json)は固定path/commit/bytes/SHA256・既存identityのJSON pointer・SOURCE対応を記録し、[publication receipt](publication_receipt.json)はpayload commitと公開照合を記録する。receipt自身を含む最終HEAD/全公開fileは公開後に照合し、チャットとローカルsidecarへ最終SHAを提示する。

既存方法ノート/A/B/C1調査、科学SOURCE/raw/結果/grant/claim/ledger/marker/STOPは変更しない。私的依頼原本を公開しない。snapshotは既存固定URLの参照のみで、本packetへ複製しない。他worktree・他process・共有設定は操作せず、既存契約の資源条件も変更しない。

**新科学計算・保存数値再解析・NPZ開封・科学SOURCE実行/import/変更・新実装・科学test・回路/compile/synthesis・新非自明証明はすべて0。公開・保全確認後mandatory STOP。次の判断はユーザー・GPT、技術作業は別指示後のCodex。**
