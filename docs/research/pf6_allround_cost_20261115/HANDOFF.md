# 統合担当・B/C/Dへ：PF6の全13round総compiled RZ数

[REPORT](REPORT.md)／[RESULT.json](RESULT.json)／[RESULT.csv](RESULT.csv)／[AXIS_TERMS.json](AXIS_TERMS.json)／[INPUTS.json](INPUTS.json)。集計SOURCE・入力の固定commitは`43f87397aedc720ff1b8243038f728a3199f4e6b`（`allround.py`と`INPUTS.json`）。

## 結果

| 順位 | 候補 | K（既知8round） | U（追加5round） | K+U |
|---:|---|---:|---:|---:|
| 1 | B0I p7 S2（16h/7） | 81,621,086 | 511,934,266 | 593,555,352 |
| 2 | B0I p7 Yoshida4（16h/3） | 107,719,628 | 658,183,130 | 765,902,758 |
| 3 | Full PF S2（16h/7） | 115,251,570 | 723,442,010 | 838,693,580 |
| 4 | B0I p7 M5（16h） | 146,850,902 | 804,431,994 | 951,282,896 |
| 5 | Full PF Yoshida4（16h/3） | 153,106,880 | 936,567,962 | 1,089,674,842 |
| 6 | Full PF M5（16h） | 207,232,138 | 1,136,164,634 | 1,343,396,772 |

最小と次点の差は**172,347,406 RZ**。既知8round時点からの順位変化は無い。

## 使用した入力

| 用途 | 固定SHA | path |
|---|---|---|
| C価格表（60軸） | `e4458cd7548977e6e11a5a7b0c8dca0d049535d4` | `docs/research/pf_closure_prices_20261115/run01/PRICES.json`、selector `rows` |
| CのHANDOFF_B | `c383105b16db7004db39b1ae270b220627d0f3b4` | `docs/research/pf_closure_prices_20261115/HANDOFF_B.md` |
| B小計K・96直接軸 | `540b54ab2aa01a164f913d5b18497e711ae4a29c` | `docs/research/pf6_eight_round_cost_20261115/PARTIAL_SUMS.json`、`INPUTS.json` |
| Q3固定shot配分 | `8dba096dc39788b8948af9eafe394f03d386c520` | `docs/research/pf_shortlist_q3_fixed_results_20261115/INPUTS.json` |
| 閉包の推奨（推奨時点） | `c3f761f0af5b21207e64051c9c619858064450fe` | `docs/research/pf_closure_price_recommendation_20261115/` |

## 閉包の適用条件

- `adoption_policy=PF_FIXED_COMPILER_CLOSURE_CONDITIONAL_60_AXES_V1`。用途は「同じ物理候補・固定shot配分の全13round条件付き比較」に限る。
- 固定H4/DF10、Full10又はB0I p7、各公式の固定実幅、元compiler経路（Qiskit 1.3.0、opt1、seed17）。障壁・反復構造・終端runの持続、installed Rustと参照sourceの対応、未調査の規模依存分岐が意味を変えないことを前提とする。
- 60軸は`MODEL_PREDICTION`・`direct_observation=false`のまま。直接価格96軸とは区別して記録した（AXIS_TERMSの`price_class`）。
- 誤差上界・CIはnull、2%保証は無い。前提が崩れた場合は、影響する候補・roundの総額を保留する。

## 主張できること・できないこと

- 主張できる：上の条件と既評価のQ3配分のもとでの、6候補の総compiled RZ数と順位。最小はB0I p7 S2。
- 主張できない：未観測60軸の価格誤差の保証、十分N・最適N、物理的global RMSE、5方式の正式G・winner、他の配分での順位。
- Q3は部分保存（最終API returnとCPU・peak RAMが欠測）のままで、本件はそれを補っていない。

## ロードマップへの影響

ロードマップ本体は統合担当が更新する。影響する箇所の案：

- §5（物理候補ごとの進捗）・§10（管理欄）：決定論PF 6候補（Full PF・B0I p7 × S2 16h/7・Yoshida4 16h/3・M5 16h）について、**採択済み閉包条件・Q3固定shot配分での全13round総compiled RZ数が確定**した（直接96＋条件付き閉包60＝156軸）。最小はB0I p7 S2、次点との差は172,347,406 RZ。
- 第2水準のGとしては扱わない（配分は十分Nではない）。5方式比較のGとwinnerはnullのまま。
- 残る課題：十分Nの契約と検証、未観測60軸の価格の資格（条件付きのまま）、他方式との同条件比較。

**mandatory STOP**。
