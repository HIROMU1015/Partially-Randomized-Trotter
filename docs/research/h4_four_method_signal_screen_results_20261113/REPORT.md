# H4・4方式signal screen：入力context構築時のSTOP

2026-11-13 JST。**承認された固定20セルbatchを1回起動したが、入力contextのimport失敗でSTOPした。新mean 0/20、Full-H normalizer 0/6、新reference 0。無断retryは行っていない。候補の科学的FAILや優劣を示す結果ではない。**

科学SOURCE：`256caad3414a80de02fb68eca89c13c957ed6354`。既存A branch `codex/pr-r-dependent-n-model-20261111`を継続使用した。[固定計画](../h4_four_method_signal_screen_preparation_20261113/SCREEN_PLAN.md)、[実行PLAN](run01/PLAN.json)、[元のSTOP結果](run01/result.json)、[実行記録](EXECUTION.json)、[ログ](EXECUTION.log)、[20セルの未評価一覧](STOP_SUMMARY.json)、[技術確認](TECHNICAL_CHECKS.txt)、[B向けHANDOFF](HANDOFF.md)を保存した。

## 固定条件と実行範囲

Linear H4/1.00 Å/STO-3G、保存tol-only DF10、weighted-Hermitian completion、sector(2,2)/36、同じH/state/snapshot。T=.1×2^m、M12、prior/decoder/誤差budget・P0は元条件のまま。Full PFは全10fragment＋one-body、B0Iはprefix7/Yoshida4/Δ.4、Full-H RTEはone-bodyを含む全非scalar・決定論block0・K2/K4・κなしの明示q/R。signed weights・fragment順・scalar規則・数値閾値は変更していない。

全20 physical IDは元PLANと一致する。Full-Hのq/Rはm3で4/180、m6で26/1434、m12で1639/91751。DF B3への代替はしていない。今回の起動は2026-11-13 15:42 JST、終了まで約3.28秒、CPU affinity64・batch設定20 workers。セルworkerの開始前に停止したため、この設定は並列科学計算の実績ではない。

## 停止原因と数値検証の到達範囲

元のエラーは`No module named 'h46_new2_core'`。呼出し経路は`h4_four_method_screen.execute` → `h4_pr_rmse_completion.load_context` → 保存H4 loader → `h6_df19_rpe_d2.make_context`の冒頭`from h46_new2_core import occupation_one_body`。

起動側の直接関数呼出しでは`src/`のみを`sys.path`へ追加し、`scripts/`を追加していなかった。既存の`python scripts/<runner>.py`形式はscript directoryをimport経路に持つため、同じ前提が直接呼出しでは満たされなかった。準備時の17人工・回帰testは実context loaderをmockしており、この起動条件を検証できていなかった。

固定screen・数値leaf・既存loader/actionのSOURCEは不変、実行PLANは固定PLANと同一。停止は保存snapshot loaderを経由したcontext構築冒頭であり、screen用spectra・macro-step・各セル・normalizerには到達していない。reference pairの数値gate、closure、power norm/unitarity、bias/phase/radius/attenuationの結果も**未取得**で、PASSにしていない。保存H/state arrayの読取を開始したことと、新signalを評価したことを区別する。

停止後の5技術確認はSOURCE/PLAN/未評価状態の保全、import探索、停止位置の静的確認だけである。`scripts/`を探索pathに加えると当該moduleを発見できることを、moduleを実行せず確認した。これはimport以降の科学的成立性や全dependencyの起動成功を保証しない。

## 各候補の状態

| 候補 | 対象round | 新meanの取得数／計画数 | 今回の結果 |
|---|---|---:|---|
| Full PF S2/Δ.1 | m6・m12 | 0/2 | 未評価 |
| Full PF S2/Δ.4 | m6・m12 | 0/2 | 未評価 |
| Full PF標準Yoshida4/Δ.1 | m6・m12 | 0/2 | 未評価 |
| Full PF標準Yoshida4/Δ.4 | m6・m12 | 0/2 | 未評価 |
| Full PF M5 best/Δ.1 | m6・m12 | 0/2 | 未評価 |
| Full PF M5 best/Δ.4 | m6・m12 | 0/2 | 未評価 |
| Truncated PF B0I/p7/Y4/Δ.4 | m6・m12 | 0/2 | 未評価 |
| Full-H pure RTE K2 | m3・m6・m12 | 0/3 | 未評価 |
| Full-H pure RTE K4 | m3・m6・m12 | 0/3 | 未評価 |
| PR-KEEP | 全13round | 0/0 | 元SOURCE・保存locatorを保全 |
| Truncated PF B0/p7/S2/Δ.1 | 全13round | 0/0 | 元SOURCE・保存locatorを保全 |

20セルのraw/corrected mean・bias・logB・attenuationは`STOP_SUMMARY.json`でnull。0という科学値で代用していない。十分N・global RMSE・winnerもnull。20セルに対する数値安定性比較や、全13roundへ進める候補の選定材料は今回取得できていない。いずれの候補もこの起動失敗を理由に除外しない。

## 保存40セルとの対応と保全

Full PFのm3/m4計12、B0Iのm3/m4計2、PRとB0の各13、合計40は`run01/PLAN.json:reuse`に元locatorを保持した。Full PFは`155529e20a8fac39e00786c40b57830b71203c4f`、B0I m3は`0b1886f53f7c3a5e99e0c13cb5f7090581c5e38c`、B0は`89c91c692e1d35de94d2e7afbe890aabe6d80b7e`等、PRは`169d3316dd008eb98dd9be56fe4ecefc5c1a7d60`と物理契約`bba150834ca30bdb186fbfd58c55a8d51177b855`。保存値の新しい比較・再計算・旧Nの流用はない。

新referenceは取得せず、m3/m6/m12の保存独立pairへの参照をそのまま保持した。既存SOURCE/input/raw/結果/grant/ledger/marker/KEEP/STOPは変更せず、B/C/Dのbranch・worktree・processにも変更していない。

## 次に必要な有限作業

**ユーザー**：今回の「無断retry禁止」に従い、`scripts/`をimport経路に含めた同一20セル・6 normalizer・新reference0の再実行を明示承認する。これは新候補や科学条件の採択ではなく、停止済みbatchの再実行範囲の指定である。

**Codex A**：承認後だけ、元run01を保全し、同じ固定科学SOURCEとPLAN・fresh出力で1回実行する。SOURCE、閾値、formula/q/R/Kを変えず、以降の失敗でも部分保存・STOPを維持する。**B**：現時点ではSTOPと未評価状態を受け取り、全13round候補の採択を保留する。

十分N/global RMSE・C/G・winnerの認定、全13round追加、最適化・samplingは未実施。本STOP結果をcommit/pushしremote SHA/tree/主要ファイルを確認した後、mandatory STOP。

## 追補：ユーザー承認後のrun02

import経路を補正し同じ固定SOURCEで再実行した。[run02 REPORT](run02/REPORT.md)／[HANDOFF](run02/HANDOFF.md)を参照。4セルを保存し数値gateでSTOP、6失敗・9未開始・1空例外未解決。run01の原STOP・本記録・旧STOP_SUMMARYを保持し、run02の部分結果で過去を上書きしない。
