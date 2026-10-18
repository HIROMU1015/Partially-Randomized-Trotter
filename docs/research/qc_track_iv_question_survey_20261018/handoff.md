# GPT・ユーザー向け引継ぎ

[入口・全成果物](README.md)、[提案](candidate_comparison.md)、[既存状態](existing_research.md)、[最小検証案](minimal_verification.md)、[固定証拠](evidence_index.json)。

## 技術的な結論

**順位付きで残すのは1件だけ：固定粒子数・DF構造を保つ全エネルギーshadows共分散縮約。** 新しい有利な生成則・定理・量子資源改善を確認したわけではない。係数の低rankと共分散の選択則を、展開前の縮約で利用する余地を問う。現在の最強直接対照は2606.30601v2 S8と著者公開SOURCEであり、既知式への通常の代入・tensor contractionへ還元される可能性が高い。新規性の確信度は低い。

共分散処理の古典費用が主目的である。全energy分散の評価が速くなっても測定法が同じなら物理的なshotsは変わらない。分散boundの改善、実shots削減、T/CX/depth、ground preparationを含む総資源を分離する。現時点で量子化学アルゴリズムの新しい優位を支持する結果は0件である。

## 科学的判断として残すこと

1. **研究目的との適合**：古典共分散取得費を主な改善指標にする狭い問いを、④の新課題として検討する価値があるか。量子gate削減が必須なら現候補を無理に採択しない。
2. **独立差の必要水準**：既知classの計算量改善が必要か、同等既知法のengineering応用評価でも目的を満たすか。後者を新アルゴリズムと呼ばない。
3. **入力モデル**：exact ground copiesの条件付き測定問題を最初のtaskとするか、近似state品質・準備を含む元ground-energy問題を最初から求めるか。異なる情報契約の対照を不公平にしない。
4. **紙上停止条件**：強対照も同じfactor/correlation/cacheで同等に処理できる場合、追加実装へ投資せず区切るか。未取得の高次相関が支配する場合も同様。

これらは判断事項の列挙であり、GPT重要科学レビューを今開始したものではない。ユーザーのテーマ採択・論文化GOをCodexが代行しない。

## 維持する境界

- D1とD2を維持。qubitization/block-encoding等を新規対象へ戻すなら別途ユーザーの方針判断が必要だが、今回その変更は提案しない。
- A2現実用化経路への追加投資見送り、A2 `NOT_ADOPTED`、A1 `CLOSED`、K1-F未承認、N1 STOP/N2 HOLD/N3 CLOSED/THRIFT現template CLOSED、PF/Exact Encoding/H1〜H4等の既存状態を保持する。
- ①②③および別repositoryは変更しない。M1-S等の旧結果・消費済みgrant/claimを新課題へ流用しない。
- 既存SOURCE・raw・certificate・契約・レビュー・authorization・grant・ledger・markerと親worktreeを変更しない。今回作成するJSONは**調査・来歴・静的監査metadata**であり、新しい科学契約ではない。

## 調査の限界・未取得事項

一次文献と公開SOURCEへの有界な調査であり、網羅的priority searchではない。arXivの取得version、HTML bytes、本文で確認した箇所を記録した。全文取得と全文の独立数学検証は別であり、出版版・全参考文献・全compilerを読了したとはしない。S8係数と著者SOURCEのcorrectnessも実行していない。

検索connectorに認証エラーがあり、Google結果はJavaScript誘導で科学的証拠として使えなかった。arXiv公式API・本文とGitHub公式API/rawへの直接読取へ切り替えた。一時、追加取得の自動承認審査も認証更新エラーで停止し、そのcallは未実行。ユーザーの再開指示後、同じ承認経路で読取が成功した。未実行callを取得済みとして数えない。

文献の全更新履歴、未公開研究、著者の別branch全体、実際の化学相関stateの高次RDM取得費・DF全energy専用の最強実装の不存在は未確認。候補の独立差を確定する追加資料・比較は未取得であり、存在しないと断定しない。指定必須A2文書・共通運用指示は取得・確認済み。過去の科学レビューは固定原本への帰属として参照し、同じ科学レビューを再実施していない。

## 次の担当

**ユーザー**：本候補を限定的に検討するか、候補なしとして別の方針を考えるか判断する。採択前に科学的価値の重要判断を必要とする場合、GPTレビューの開始を別途明示する。

**Codex**：ユーザーが別途依頼した場合だけ、[最初の紙上方法差確認](minimal_verification.md)を行う。実装・人工テスト、新科学契約、科学実行、native/state-preparation評価へはそれぞれ必要な別途承認後に進む。現在の依頼からは自動移行しない。

調査資料のcommit・push・公開同一性・保全確認を終えた時点で **mandatory STOP**。新規科学計算・保存数値再解析・SOURCE実装・compile・実験契約・authorization/grant/marker/claim発行・重要科学レビュー・テーマ採択は全て0件。
