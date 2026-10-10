# 2026-10-10 独立Hamiltonian表現探索の判断履歴

この記録は独立系列の当日履歴。現行結果は[初期検証報告](../representation_exploration_initial_validation_20261010.md)、
規範範囲は[scope](../representation_exploration_scope.md)。既存Track A/Bの履歴は書き換えない。

1. 利用者依頼を、PR内部改善の別Trackへ統合せず、A/B/Cの初期探索・小規模検証として受けた。
   当初欠けていたGPTレビューは利用者がrootに追加した原本を読み、独立worktreeの入力namespaceへ保存した。
2. mainを最新結果とは仮定せず、b2e1bf6を既存DF/RTE helperの基点に選び、後続Track A/Bの別commitを監査した。
   元rootのdirty差分と1312 pathsの初期状態を保護し、専用branch/worktreeを作成した。
3. factor恒等式・Gram proxy・block平均を先に確認し、分子計算に進まず判別力のあるtoyと対照を固定した。
   source451bfa5を固定したrun1はcontrolled primitiveのscalar phase mismatchでtechnical STOP。
   failure/auditを保持した。利用者依頼には技術的対処が含まれ、旧Trackのone-shot契約を流用しない。
4. scalarだけの差を密行列認証できる場合に限りglobal phase補償し、追加controlの同値性も検査した。
   入力・角度・比較・資源上限は不変、source25d7135で別run2を実行。
5. run2はA20/B9/C3行と19 compilesを完了。Aのproxyと実lambda/costの不一致、Bの平均とtrajectoryの違い、
   Cのmixed oracle代用による理論利益の喪失が比較可能になったため、科学計算をここで停止した。
6. 保存結果・commit source・イベント集計・Taylor行列を別stdlib verifierで監査し、旧結果を上書きしなかった。
   bounded-support可解core＋正確残差は新しい検討候補として記述するだけで、追加実装や採択はしない。
7. 追跡可能なsource/入力/raw結果/失敗/log/環境/監査を公開し、別取得でGitHubの実branch・commit・blobを確認する。
   中心仮説、新規性、研究方針はGPTへ戻す。next-stage=false、central-hypothesis=null、mandatory STOP。
