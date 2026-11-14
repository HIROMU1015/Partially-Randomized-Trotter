# 統合ロードマップへの影響（本体の更新は統合担当へ）

指定正本 `fa84c0d6bce0c8e8b8accbe6009069074b466d32:docs/research/RESEARCH_ROADMAP.md` と公開A c212/B999・最新9c0/C855/Daceaを照合した。roadmap branchを変更していない。

| 箇所・論点 | 更新すべき技術的状態 | 解決していない科学課題 |
|---|---|---|
| §4〜5 Full PFの信号 | A c212が保存meanのround別原本を特定。shortlistに全13正式packetはないが、S2/S2L/S3の分類で再利用・不足範囲が明確 | S3の22（＋2）mean/方式の科学承認、旧mean受入方針、packet化、Bのrisk/N |
| §5 Truncatedの候補不一致 | p7 S2/.1の13rawと、p7価格shortlistは別schedule。shortlistにもA調査の保存meanがあり、D CELL_PLANとIDが一致 | 比較scheduleの採択、signal/価格を同候補に揃える科学範囲 |
| §5「p7 S2/.1の価格なし」 | **現行guard受入済み価格はない**。旧価格carrierには同物理点m3/m4のlocatorがある。元wrapper/axis/実装証拠の受入を今回行っていない | 旧metadataを正式full-wrapper receiptと誤認せず、必要なら受入根拠を補う |
| Dのp7保存M5受入 | C855で保存4receipt、Dacea ADMISSION_UPDATEで保存4＋新cal20＋holdout12接続が公開済み | 比較policy/2%許容差の採択、科学取得・fit・独立holdout、長q外挿資格 |
| PR-DF | 52軸coverageとA physical IDが対応。**B9c0で版付きtotal bridgeは技術解消**。LD1 E[S]26軸、P1/P2は別cohort。LD2同条件価格なし | scalar/source価格受入、全law残差/平均compiled資格、Bのtotal MSE/正式N。旧partial MSEを移送しない |
| PR-Pauli | B元→total橋渡し、条件付きmodel MSEとUSER_SUPPLIED vectorあり。B0はLD1m0単一path X/Y | B1未承認、辞書別平均価格、law/rare/CI/正式N/比較契約 |
| Full-H | κ2 signal全13とP-1m3の元identity/law対応。R448もB9c0で独立metadata接続済み。P-1元STOPと別revalidationを保全 | P-2識別可能なcal/holdoutとconsumer接続、全law価格、B MSE/N。R448の正式候補採択なし |
| §7〜8 final integration | 候補×round×axis×cohortの技術対応表を公開。単なるroute/canonical一致で結合しない | 全G・winnerはnull。metadata一致は科学資格ではない |

価格実装・N solverの再開発は不要。次の優先順位はユーザーによる比較candidate/claim選択に依存する。保存結果受入、metadata bridge、新signal、直接compile、独立model検証、MSE/N評価を別項目としてHANDOFFした。旧m12巨大T2を必要条件へ追加していない。

**mandatory STOP**。新しい科学計算や比較採択は本報告から自動開始しない。
