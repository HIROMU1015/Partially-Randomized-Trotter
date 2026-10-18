# 研究④：A2後の新規課題選定に向けた技術調査

2026-10-18 JST。基点 `f7e103bf3a0ffcee6162bd4b8d244f9a2f4502c5`。**候補は条件付きで1件。独立した新アルゴリズムを確認した件数は0件**である。3件に水増しせず、既知法との差が残るかを先に問う。

| 資料 | 内容 |
|---|---|
| [既存研究の整理](existing_research.md) | 成果、還元、負の結果、正式停止と残る問題 |
| [候補比較・提案](candidate_comparison.md) | 唯一残した限定課題、改善機構、強い既知対照、入力・費用、新規性の不確実性 |
| [最小検証の提案](minimal_verification.md) | まず紙上の方法差確認。その後の実装・実行は別承認 |
| [GPT・ユーザーへの引継ぎ](handoff.md) | 判断事項、採択境界、STOP、取得範囲の限界 |
| [証拠索引](evidence_index.json) | 固定commit/path/bytes/SHA256、一次論文版、公開SOURCE identity |
| [調査記録](literature_scope.json) | 検索式、取得失敗、本文の確認箇所、未確認事項 |
| [静的・保全確認](verification.json) | 新規科学評価を伴わないファイル・リンク・保全確認 |
| [公開照合](publication_receipt.json) | payload公開bytes/SHA256、後続receipt commitの方針 |

提案する問いは、**固定粒子数のfermionic orbital-rotation測定に対し、DFの低rank構造を保ったまま全エネルギー推定量の共分散を処理できるか**である。個別RDM推定、粒子数対称性による改善、共分散の解析そのものは既知である。新たに問うのは、係数と共分散を別々の巨大な表へ展開せず、同じ情報を与えた最強既知実装より少ない取得・縮約費用で、全エネルギーの精度を管理できるかという狭い差分である。

現時点では、その差分を実現する新しい縮約規則・一様分散保証は未構成であり、改善を確認していない。単なる既知式のtensor contractionや既知SOURCEの高速化へ還元される可能性が大きい。この不確実性を受け入れて**限定的な方法差確認を検討する候補**として提示し、実装・分子pilotの即時開始は推奨しない。

本調査はユーザーが依頼した技術的事前調査であり、重要科学レビューの実施・新数学証明・研究テーマ採択ではない。現行A2の追加投資見送り、A2 `NOT_ADOPTED`、A1 `CLOSED`、K1-F未承認、その他のCLOSED/STOP/PAUSED/HOLDを維持する。既存D1（block-encoding/qubitization等を新規対象から除外）とD2（元電子Hamiltonianの絶対基底energy）も解除しない。

新規科学計算、保存値再解析、SOURCE実装、simulation/compile、実験契約、authorization/grant/marker/claim、重要科学レビューは全て0件。**公開・保全確認後mandatory STOP。次はユーザーが、この限定課題をさらに検討するかを判断する。**
