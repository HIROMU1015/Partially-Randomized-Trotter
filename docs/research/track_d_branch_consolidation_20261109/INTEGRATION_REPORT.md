# Track D成果統合の技術報告

2026-11-09 JST。統合先 `codex/track-d-consolidated-20261109`。新しい科学実行SOURCEではない。

## 実際に収録した内容

公開済みC1実H4結果 `0ef53fa9a65a70a251a7d6e5dbeb91141a1a67e9` を起点とし、38本分のTrack D履歴と既存資産を継承した。独立C7-4とB3〜B5を、内容を取り込む通常merge 2回で追加した。計43本のTrack D tipが現在の履歴の祖先である。

| 内容 | 統合元 | 今後読む実装・資料 |
|---|---|---|
| Strong・Reference gadget/χ/actual inverse・X/Y checker | H4/Stage A〜C2 | `src/trottertracks/c1_h4_strong_baseline.py`、`tests/track_d_h4_reference_checks.py` |
| whole ordered path・step融合・共通価格request | C3/C5/C6a/C6b | `track_d_c3_whole_path.py`、`track_d_whole_path_price.py`、`track_d_c6a_pilot.py` |
| q16/q32独立paired価格・CX非加法性 | C7-1/C7-2/C7-3 | 各固定RESULT/REPORTとartifacts、`track_d_c7_2_toy.py` |
| C7-4モデル・数値安定性・失敗/修正 | `64072ba81bb70db1df1b34e94adb08e78926f24b` | `track_d_c7_4_models.py`、`track_d_c7_4_stability.py`、`docs/research/track_d_c7_4_model_stability_20261107/` |
| 元H統計adapter・B4-2十分N・B5不足保証 | `0a8b953aea1be20b110784960763d7f36ebccac8` | `track_d_original_h_accuracy.py`、`c1_stage3_accuracy.py`、B3/B4/B5資料 |
| B6b測定同値性・条件付き13-round既存会計 | B6b/B-C total | `track_d_b6b_bc.py`、`docs/research/track_d_bc_total_20261108/REPORT.md` |
| O5静的選別・人工cut・保存q4 cohort構造 | O5 design/cut/metadata | `track_d_cut_selection.py`、対応固定REPORT |
| C1 head-tail人工・実H4限定移送 | C1人工と実H4 | `track_d_head_tail_artificial.py`、`track_d_head_tail_h4.py`、対応固定REPORT |

上表のmodule名は `src/trottertracks/` 相対。旧結果が統合先で再実行されたという意味ではない。

## 通常mergeと衝突

1. C7-4実merge commit `12b60b92e3097907b81071be12d76d5e479c3717`、tree `264177aae5b7342aec061e0b83ee03526af3dcee`。元親はC1 H4 tipとC7-4 tip。46 pathを取り込み、衝突8 pathは索引/README/manifestのみ。
2. B3〜B5実merge commit `a7557032854e8e7473284c02ee610616e7e912d8`、tree `c95241d4b5f8c2aaeefb689bd5c903ad99fcceab`。元親は上記mergeとB5 tip。33 pathを取り込み、衝突7 pathは索引/README/manifestのみ。

共通Markdownはbase/ours/theirsの追加節を両方残した。manifestはdict keyとresult_setsの固有idで三者比較し、異なる科学結果を混ぜなかった。Strong/gadget/whole-path/C1を再実装していない。B系の既存共有accuracy helper変更を元bytesのまま取り込んだ。`merge -s ours`、rebase、force overwrite、main mergeは使用していない。

## O2移管資産は独立contextを保持

保持ref：`codex/o2-vacuum-h4-preparation-20261108`、tip `781e155d0907a82c962e41ab8e89f985735bc49b`。このtipはCSWAP元tip `0edc0fa5ad4656fdd492ca6c96ade4f77cadf412` を含む。

- [真空参照CSWAP・SE-QPE対応・人工反例/負結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0edc0fa5ad4656fdd492ca6c96ade4f77cadf412/docs/research/o2_vacuum_reference_cswap_20261108/REPORT.md)
- [State-free H4 adapter・phase ledger・POVM誤差・人工検証](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/781e155d0907a82c962e41ab8e89f985735bc49b/docs/research/o2_vacuum_reference_h4_preparation_20261108/REPORT.md)
- 同tipの `src/trottertracks/algorithm_codesign/vacuum_reference_cswap.py`、`vacuum_h4_adapter.py`、`vacuum_h4_execution.py`、`vacuum_h4_numeric.py` と既存tests。

O2 tipを丸ごとmergeすると共通祖先から1304 pathの差分があり、大部分は対象外のSampling/Return Aggregation等の独立資産となる。移管moduleはA1 engine connection、metadata execution等に依存し、それらはTrack D現在treeにない。互換性未確認のまま一括importや同gadgetの新規複製を行わず、元の1 branchを正式な実装保持先とした。O2の成果を収録済みコードと偽ってはいない。O1/O2/O3との共同研究に必要な出典は辿れるが、互換統一APIの科学的資格付けは今回行っていない。

## 技術確認と実行境界

既存tests `check_track_d_original_h_accuracy.py`、`check_track_d_head_tail_h4.py`、`check_track_d_c7_4.py` のうち人工fitテスト1件を除いた64件がPASS。実H4原本・保存価格を読まず、synthetic scalar/8-mode span/phase/拒否条件/native adapterを検査した。新しい実H4 simulation、sampling、P0 compile、モデルfit、RPE再算定、価格再解析、詳細科学レビューは0。

C7-4の既存SVGにあるtrailing whitespaceは、元blob同一性を守るため変更しなかった。新規整理文書は別途whitespace/JSON/syntaxを確認する。技術test PASSは新しい科学結果ではない。

既存のq64 worktree/SOURCE/output/STOP、private archive/QPY/NPZ、他worktree、main、共有環境は変更しなかった。GitHub接続で作成したtreeはローカルで実mergeしたindex treeと一致する。
