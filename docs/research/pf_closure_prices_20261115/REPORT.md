# 決定論PF：採択済み閉包条件による60軸の価格確定

**60/60軸のcompiled RZを元凍結式からexactに算出・保存した。保留0、欠落0、重複0。既存直接価格96軸との交差0。全156軸は「直接96＋条件付き導出60」であり、直接観測が156軸へ増えたわけではない。**

正本は[run01/PRICES.json](run01/PRICES.json)の`rows[0:60]`、簡易表は[PRICES.csv](run01/PRICES.csv)。候補・軸から元モデル/監査へ戻れる。固定shot配分との積算やG/winnerは今回実行していない。

結果固定commitは**`e4458cd7548977e6e11a5a7b0c8dca0d049535d4`**。[HANDOFF_B](HANDOFF_B.md)に結合キー・固定locator・利用条件を示す。公開固定JSONを既存保存readerから60行/整数値として読取確認した（再予測・会計なし）。

## 採択条件・SOURCE

利用者はCの推奨 `c3f761f0af5b21207e64051c9c619858064450fe:docs/research/pf_closure_price_recommendation_20261115/` を採択し、同じ物理候補・固定shot配分の全13 round比較に限って60価格を使用することを指示した。本結果ではその有限domainを`PF_FIXED_COMPILER_CLOSURE_CONDITIONAL_60_AXES_V1`として表示する。旧推奨書の未採択時点の記述は履歴として保全した。

- 実行SOURCE：**`0ecd30f3b955d90d9898d0b09ff8bc6b15d93ce6`**、`scripts/freeze_pf_closure_prices.py`。
- 元12モデル：`12372fa6b53c878dc5dd124e036572ba0cd82061:docs/research/pf_stage1_model_freeze_20261115/run01/FROZEN_MODELS.json`。
- 保存QPY監査：`606881b042ed69f9a1e7f00311417f2b65855d7b:docs/research/d_pf_qpy_closure_audit_20261115/`。
- 対応元：採択推奨の`RECOMMENDATION.json#rows`及びその固定alignment/model/audit locator。
- exact evaluator：既存`src/trottertracks/price_model_validation.py:predict`。元凍結SOURCEの同moduleとbyte一致を確認して再利用。fit/validateは今回呼び出していない。

Linear H4/1Å/STO-3G・保存tol-only DF10、元H/state/snapshot、Full10又はB0I p7、S2/標準Yoshida4/M5の固定幅とsigned piece、fragment/basis順、scalarの総T一回適用を維持。symmetric-directional、測定付きfull controlled Hadamard wrapper、P=0、Qiskit1.3.0/rz,sx,x,cx/opt1/seed17、coupling mapなし。入力binding・split・scalar・native/profileを各行へ保存した。

固定compiler、監査が示す障壁・反復構造・終端runの挙動が対象qまで持続するという**条件付き**価格である。installed Rustと参照sourceの対応、未確認の規模依存分岐等の前提は採択推奨のまま。m6/m9の2%規則を未観測点の保証にしていない。誤差上界/CIはnull、元モデルの資格flagはfalseのまま保全した。

## 価格結果

以下は保存結果の転記で、X/Yが一致していてもJSON/CSVでは別行・別groupのまま保持する。各行の`compiled_RZ`は整数、`exact_RZ`は分母1のexact表現、`mean.rz_count`は同じ値。丸めを用いていない。

| 方式・公式 | m7 | m8 | m10 | m11 | m12 |
|---|---:|---:|---:|---:|---:|
| Full S2（X/Y各別） | 436603 | 872899 | 3490675 | 6981043 | 13961779 |
| Full Yoshida4（同） | 565147 | 1129987 | 4519027 | 9037747 | 18075187 |
| Full M5（同） | 685531 | 1370755 | 5482099 | 10963891 | 21927475 |
| B0I p7 S2（同） | 309035 | 617763 | 2470131 | 4939955 | 9879603 |
| B0I p7 Yoshida4（同） | 397243 | 794179 | 3175795 | 6351283 | 12702259 |
| B0I p7 M5（同） | 485451 | 970595 | 3881459 | 7762611 | 15524915 |

S2のqは56/112/448/896/1792、Yoshida4は24/48/192/384/768、M5は8/16/64/128/256。実幅は16h/7、16h/3、16hで、元alignmentのbinary64値を使用した。

## §10.5・重複照合

[PREFLIGHT.json](PREFLIGHT.json)をSOURCE固定前に保存し、実行直前にも同じ入口で差分確認した（[run01/CHECKS.json](run01/CHECKS.json)）。Cの開始公開HEADはc3f761f0a、Dの公開branchは606881b04のまま。同一60価格の既存freezeは関連C履歴/結果pathに見当たらず、同一ユーザーの関連Python runner名も該当0だった。他担当のworktree/private QPYへ変更・読込をしていない。

DのFull/p7各5 stageの公開PRICES、Cのm3受入、保存M5/m4の元C受入locatorから既存直接96軸をmetadataで索引化した。保存4件のEXPLORATION cohort・元price_source/record_sourceを保持。初回metadata preflightではこの4件がDの新規PRICESに含まれず92軸となったため、元凍結input locatorから補い、その経路の小さい回帰testを追加した。科学計算前の修正で、元receiptや科学式は変更していない。

対象60軸と既知直接96軸は結合キー`(configuration_id, physical_id, m, axis)`で互いに素。直接価格を再平均/再集計していない。未列挙archive、別host、別名jobを網羅した不存在の主張ではない。公開D更新や新しい同一直接価格が見つかる場合には、その元receiptを優先して差分を照合する。

## 実行・検証

SOURCEとpreflightをcommit後、fresh `run01`で60件を一度だけ評価した。`prediction_calls=60`、fit/compile/QPY読取=0、保留0。前の起動要求1回は自動承認レビューの利用上限でprocess未起動となり、出力directoryも作られていなかった。再開後、SOURCE/branch・出力不存在を確認して初回科学処理を実行した。科学retryや部分値の上書きはない。

exact評価段階は約0.019秒（metadata確認・公開時間は除外）。小さな有理数評価60件であり、既存reader cacheを利用した。回路構築や大規模並列処理は必要としない。

[TESTS.txt](TESTS.txt)：新人工/metadata6件＋既存exact model回帰3件、計9件PASS。人工係数で別X/Y応答、整数exact結果、欠落/重複/scope拒否、負値/非整数の丸め禁止を確認。実入力はmetadata照合のみをテストし、実60価格は別の一回の実行で取得した。保存後はJSON/CSV転記一致、12 group×5 round、96＋60の156 unique axis、元推奨・凍結JSON・保存receipt・guard不変を確認。再予測して検証値を作っていない。

## Bへの意味と終了点

価格は`MODEL_PREDICTION`、`FIXED_COMPILER_CLOSURE_DERIVATION_CONDITIONAL`。`direct_observation=false`、`unobserved_circuit=true`、`conditional_use_adopted=true`を保持する。既存の汎用`qualified_for_candidate=false`は変更せず、今回の限定policyを別metadataにした。Bはこの採択済みscopeを明示して利用し、直接価格や無条件の`QUALIFIED_MODEL`へ付け替えない。

ロードマップ差分：**6候補の全156軸について、直接96軸＋条件付き導出60軸の価格入力が揃った。C側の60価格待ちは解消。同じ物理候補の固定shot配分による全13 round会計はBの後続範囲で未実施。** 原価格のconsumer受入・N/decoder側の資格はそれぞれの担当成果を維持する。ロードマップ本体は統合担当へ委ねる。

**mandatory STOP：固定価格表・HANDOFF_B公開とremote照合で終了。新compile、QPY再監査、再fit、N/総費用/winnerへ進まない。**
