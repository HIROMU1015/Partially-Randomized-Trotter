# 元Track A アルゴリズム改善の静的調査（2026-11-09）

Claude Codeが2026-11-09に行った2件の静的調査の報告です。2026-11-10にユーザーの指示でrepositoryへ保存しました。

- 対象：元Track A（2026-11-06〜08のアルゴリズム改善系列。O1-M1、O1-M2、B2-P1、C-S V0）。
- 調査方法：固定commitの資料・SOURCE（テキストとAST）・保存CSV/JSONの該当行を読んだだけです。
- 行っていないこと：新しい科学計算、数値再解析、build/compile、sampling、テスト実行、module import。
- 状態：**mandatory STOP**。どちらの報告の技術的意見も採択ではなく、科学的なGOでもありません。次の判断はGPTとユーザーが行います。

| 文書 | 内容 |
|---|---|
| [TRACK_A_RESEARCH_SURVEY.md](TRACK_A_RESEARCH_SURVEY.md) | 研究の経緯、O1・B2-P1の到達点、改善候補一覧、他Trackとの重複と再利用資産、次の判断に必要な事項 |
| [K3_A3_TECHNICAL_SURVEY.md](K3_A3_TECHNICAL_SURVEY.md) | K3（D/R/drop三分割）とA3（round別D/R分割）の技術調査、B2-P1とTrack C C-H4の条件対応、旧opt2とTrack Cの相違、再利用可能な実装、最小検証計画A3-1・K3-1（未実行・未承認） |

## 他の文書と異なる点・訂正

- 横断調査 `2e79d80d`（`docs/research/pr_rpe_cross_track_20261109/`）は、強PFの旧5 STOP armを「未再資格」としています。実際には、`3e316e07`・`e162ef70` のHP 80/120桁SOURCEで再資格されており、全8 armが条件付きbranch/shot PASS（各18,540 shots）です。旧5 STOPは、旧SOURCEの記録として残っています。
- 横断調査のK3初手案（D=LD3＋Truncated coeff_L1_2の除去集合）は、λ_Rを約.05%しか変えません。判別力が低いと判断しました（K3_A3_TECHNICAL_SURVEY.md §A.6）。

## 未決の判断（K3_A3_TECHNICAL_SURVEY.md §12）

- GPTの判断：
  - A3-1（H4 m17、LD4–7）とK3-1（H6 m17、評価B）のどちらを行うか。
  - 比較基準を評価A（combined）とB（個別上界）のどちらにするか。
  - Track AがC-H4条件・Track C pipelineで、自身の契約として検証してよいか。
- ユーザーの判断：選んだ計画の科学計算・build/compile・価格MCの実行承認。
