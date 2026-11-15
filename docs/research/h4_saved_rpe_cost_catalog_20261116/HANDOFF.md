# H4 保存RPE結果の自動比較：B・B2・D・D2・統合担当への引渡し

同じコマンドでINPUTS.jsonの固定commit/path一覧を読み、公開済みQ3判定と保存条件付き費用を照合して方式別最安点を返す。対象は **指定した保存候補集合内の最安適合点**。新しい信号・価格・N・Q3や全配分の探索は実行しない。

## 再実行するコマンド

作業branchは `codex/h4-saved-rpe-cost-catalog-20261116`。旧C価格branchの固定6ad23def0から分離した専用worktreeで、他担当branchは変更していない。

```bash
cd /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/c-h4-saved-cost-catalog-20261116
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /home/AbeHiromu/projects/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/venv/bin/python -B scripts/collect_saved_rpe_costs.py --inputs docs/research/h4_saved_rpe_cost_catalog_20261116/INPUTS.json --output docs/research/h4_saved_rpe_cost_catalog_20261116/run01
```

`INPUTS.json.sources` に `{"commit":"固定SHA","path":"repo内のSUMMARY/C_RESULT_JOIN/HANDOFF.json"}` を追加して同じコマンドを再実行する。既存SUMMARY群と同じproducer形式ならcode変更は不要。臨時に追加する場合は `--source 固定SHA:repo内path` を繰り返してもよい。Git objectが未取得なら先に通常のfetchで取得する。runnerはネットワークや未公開worktreeを探索しない。入力略SHAは40桁に解決して出力へ保存する。

出力は指定directoryの `CATALOG.json`（候補・方式別最安・完全locator）、`REPORT.md`（比較表）、`INPUTS.json`（実際の入力一覧）。この生成directoryを同じコマンドで更新する。元研究の固定Git blob、原API、SOURCE、価格/配分・旧STOPは変更しない。過去版はGit履歴へ保持する。

## 初回固定入力と追加指示

初期6成果（b4296b110、cbc471c91、738df2262、cdbce383f、8ec7c8828、6ad23def0）に、作業中の明示指示によるa006b8e15、2de33a4f4、0defa286c、7349f6f9bを追加した。完全SHA/pathとH4 snapshot・H/state・精度条件はINPUTS.json。開始後の新成果を自動的に追跡せず、この固定10入力を今回の集合にする。

- Full PF/S2・m12各1shot・531533828 RZは保存のMODEL_BOUND_EXCEEDS_ADOPTED_THRESHOLD。数値失敗に読み替えずモデル不適合として残す。m12=2の559457386 RZを最安適合点として選ぶ。
- p7/S2はm12各1shot・376203472 RZをモデル不適合として残し、2shot・395962678 RZを4点の最安適合点として選ぶ。
- Full PFの「2shotが最安」というB2結果の範囲は、m0〜m11固定、m12のX/Y同数・各軸1shot以上。全配分の最適性へ拡張しない。catalogの選択範囲は引き続き保存集合内。
- PR-Pauli LD3共通VP、PR-DF LD7共通V0はいずれもQ3適合・価格待ち。費用nullのままで順位に入れない。LD3の旧信号のみ状態はhistoryへ残す。
- PR-Pauliの事前固定予算・比較は旧615304502 RZの基準と元SOURCEをhistorical_*へ転記し、新Full PF最安値へ付け替えない。
- Full-Hは同じconfiguration・26軸・固定1.5倍配分のQ3をB6c5569c3から照合したうえで、C6ad23def0のモデル参照総額を表示する。m0実測1pathと残24モデルの資格は保持する。

## 再利用・照合と選択規則

既存 `baseline_signal_pipeline.blob` の固定Git読取と、B/B2/DのSUMMARY・C_RESULT_JOIN・INPUTS・候補/価格対応表を再利用する。既存summarizerの確定済み最安値をそのまま集めるだけでなく、全保存行を共通形式へ写し、今回の入力集合に対して最小値を比較する。新しいoptimizer/decoder/N solver/価格計算器はない。

比較条件はsnapshotだけでなくH/state、basis/geometry/tol policy、h/M、energy prior、元full DF-H solver proxy、epsilon/採択閾値、total decoder/loss、P0、compiler/control/wrapperを固定する。方式固有のideal finite-K law・split・K/κは属性として保持し、方式間で同じlawを強制しない。候補内ではcase/configurationのlawを照合する。

各点で、原configuration・total decoder view・canonical candidate・physical ID・round/axis・shot vectorを照合し、26価格の元sourceへ戻せるようにする。新配分の価格termに記録されたNはQ3配分と一致させる。1shot価格の元receiptに付いた過去のNは価格自体の属性ではないため、物理条件が同じ別配分への価格再利用は拒否しない。今回N×Cを新しく計算せず、保存済み条件付き総額とその配分のsourceを照合する。

選択対象は、同条件・保存Q3適合・全26軸価格対応あり・保存総額ありの点だけ。費用比較はexact有理数で行い、表示の丸め値を使わない。同率最安はすべて返す。Q3待ち/価格待ち/モデル不適合/判定保留/metadata不一致を区別する。metadata不一致を科学的不適合にしない。

HANDOFF→SUMMARYの重複参照は同一documentとして読み、同一configuration・配分・評価契約・価格locatorの点も一つに統合する。alias labelと元sourceは全て残す。保存判定/費用が矛盾する重複は選択せず不一致を表示する。

現在対応するのは今回のPF6、決定論PF m12比較、Pauli配分比較、PR-DF split比較、信号packet、価格のないQ3軸索引、Full-H混合参照総額の公開schema。未対応schemaは明示的に停止し、価格や資格を推測しない。将来のD2等は、同じ形式の保存総額＋同じ配分のQ3結果を追加すれば同じコマンドで比較できる。26軸価格だけからの新規model予測・配分生成・G計算は自動では開始しない。

## 検証・§10.5・性能

関連の旧集計 `summarize_pf6_cost_budget_q3_fixed.py`、Pauli summarize.py、`summarize_pr_df_ld_split_q3_resume.py` とCの保存会計を調査した。方式固有の集計とreceipt読取は存在したが、snapshot/精度をqueryにして全方式を横断し、入力追加だけで更新するconsumerは確認範囲で未実装だった。H2やnon-DF旧catalogは別対象のため転用しない。同等成果/同名C jobは確認範囲で0、他担当private rawやprocessには介入していない。

人工・保存結果の技術検収はTESTS.txt。実行入口の同じcommandを2回使い、INPUTSのsource追加だけでp7最安値が435481090→395962678へ更新されることを確認する。snapshot/精度/law/compiler/axis/N/physical ID不一致、欠測、判定保留、exact比較と重複も検査する。保存値の読取と比較であり、新規科学callはない。

`Reader.document` は固定commit/pathをcacheし、複数HANDOFFから同じ大きなJSONを反復読取しない。既存B2と同じprocess-local整数桁上限の解除で長いexact MSE分子・分母を読む。OSや他環境の設定は変更しない。性能方法の短い索引はdocs/PERFORMANCE_METHODS.md。今回追加のworkerや資源上限は不要。

ロードマップ本体は変更せず、保存集合の最安点と未解決点を今回の成果へまとめる。正式E[C]/G/十分N/全空間最適性・方式winnerは認定しない。commit/push・remote一致確認後mandatory STOP。
