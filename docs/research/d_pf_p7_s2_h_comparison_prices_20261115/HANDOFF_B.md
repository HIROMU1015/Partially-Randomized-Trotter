# Bへ：p7/S2・要求幅hの26軸比較価格と固定配分の費用

**この配分を保って16h/7からh=.1へ細かくすると、保存Q3モデルMSEは約1.507e−6→1.113e−6 Ha²へ改善する一方、費用は435,481,090→993,110,014 RZ（＋557,628,924、＋128.048941%）となる。hの費用は採用規則を用いたモデル総費用であり、その98.388806%が未観測長qへの外挿部分である。**

価格表 [run01/COMPARISON_PRICES.json](run01/COMPARISON_PRICES.json)／[CSV](run01/COMPARISON_PRICES.csv) は26/26軸を持つ。**直接10軸＋外挿16軸**で、直接coverageは10/26のまま。固定配分の積算は [run01/FIXED_ALLOCATION_COST.json](run01/FIXED_ALLOCATION_COST.json)。詳細 [REPORT](REPORT.md)。

## 結合と価格の利用

- 元configuration：`e6754564a42fc59a7f7088708da23ee861ec193e18f50641eeafd1096dfef7bc`。
- 元precision側：`bec032fc26093177f529754060df06d5375944b4:docs/research/p7_s2_d01_same_shots_q3_results_20261115/C_RESULT_JOIN.json`。
- 結合キー：`(configuration_id, physical_id, m, axis)`。total view `b8ceb77562c4648f2ae51d7beebdc4069edb0f273d316dfcaee500d1d33015fb`、binding `8e1e602dfc3969251499d516ad03f4a9fccfca907db7d1ee84c9e51a971d94db` は精度側IDとして費用JSONに保持し、元候補を付け替えない。
- m0〜m4の `origin=DIRECT_COMPILED` では `unit_price_RZ` は受入済み直接値。元receipt/sidecarは `direct_receipt` / `direct_sidecar`、入力行は `input_row` から固定SOURCEへ戻る。
- m5〜m12は `origin=ADOPTED_RULE_EXTRAPOLATION`、`unit_price_RZ=5513*q+307`。`direct_receipt=null` / `direct_compiled_RZ=null` を維持し、実装guard PASSや実測価格へ変換しない。

| m | q | X単価 RZ | Y単価 RZ | origin |
|---|---:|---:|---:|---|
| 0 | 1 | 5,820 | 5,820 | 直接 |
| 1 | 2 | 11,333 | 11,333 | 直接 |
| 2 | 4 | 22,359 | 22,359 | 直接 |
| 3 | 8 | 44,411 | 44,411 | 直接 |
| 4 | 16 | 88,515 | 88,515 | 直接 |
| 5 | 32 | 176,723 | 176,723 | 外挿 |
| 6 | 64 | 353,139 | 353,139 | 外挿 |
| 7 | 128 | 705,971 | 705,971 | 外挿 |
| 8 | 256 | 1,411,635 | 1,411,635 | 外挿 |
| 9 | 512 | 2,822,963 | 2,822,963 | 外挿 |
| 10 | 1024 | 5,645,619 | 5,645,619 | 外挿 |
| 11 | 2048 | 11,290,931 | 11,290,931 | 外挿 |
| 12 | 4096 | 22,581,555 | 22,581,555 | 外挿 |

対象quantityは保存linear H4 / 1Å / STO-3G / tol-only DF10、一体項＋fragments0..6保存順、B0I p7/S2、実幅binary64 .1、scalar c_H＋tail identityをtotal Tへ一回、P0除外の**測定付きfull controlled Hadamard wrapper1 shotのcompiled全RZ数**。symmetric_directional、Qiskit1.3.0 / rz,sx,x,cx / opt1 / seed17、元native policy `H46_LONG_TIME_NATIVE_COMPOSE`、RTEなしを保つ。他幅・公式・入力・control・compilerへの移送は対象外。

別のshot配分では、同じcandidate/schedule/quantityを保つ限り、この表の各 `unit_price_RZ` に対応軸のNを掛けて合計できる。価格取得のやり直しは不要。直接部分・外挿部分を分け、モデル不確かさUNKNOWNを引き継ぐ。別配分のQ3適合性はB側で別途評価が必要であり、今回生成・評価していない。

## この固定配分の総費用と保存精度

X/Yとも `N=[59,55,51,47,43,39,35,31,27,23,19,13,4]`、計892 shot。

| 指標 | 値 |
|---|---:|
| 直接10軸の部分額 | 16,000,932 RZ |
| 外挿16軸の部分額 | 977,109,082 RZ |
| 幅hのモデル総費用 | **993,110,014 RZ** |
| 外挿割合 | **98.3888056938%**（exact分数はJSON） |
| 幅16h/7の保存達成費用 | 435,481,090 RZ |
| 差 h−16h/7 | **＋557,628,924 RZ** |

Q3は同じ `bec032fc…/SUMMARY.json` の保存引用。hのMSE区間 `[1.113188114132502061033934e−6,1.113188114132502061033945e−6] Ha²`、16h/7は `[1.506660616251764083585358e−6,1.506660616251764083585362e−6] Ha²`、両者 `MODEL_BOUND_MEETS_ADOPTED_THRESHOLD`。保存solver proxy・Z1＋T1 total decoder・IID二項・exact τの条件付き結果で、十分Nや物理的保証の新認定ではない。

**この同一配分では細幅のモデル費用が高い。精度余裕を使ってshotを減らせば費用も下がるか、粗幅より有利になるかは今回未評価。** 435,481,090 RZをhの達成費用へ転用しない。

## 規則・証拠・公開

5点は規則の選択に使った観測で、X/Y各5行が5513q+307へ整数完全一致。固定実幅で同じS2 stepを反復する構造が採用の根拠になる。ただし長qのcompiler最適化、境界相殺、total T scalarとX/Y終端の数値依存を独立に検証していない。追加のQPY監査・compileは0、長qモデル誤差上界はnullである。任意qへの厳密閉包、2%精度、価格上下界は主張しない。

実行SOURCE：`33d75cfbf0e891a53184e2973286f6cb9d6ef82b`、[caller](build_comparison.py)。価格入力 `e04a14797a56a586dcb72065a55f4d97bfa1479b` のdirect表と、C/B原結果は不変。各出力に固定input locator/SHAを保存。人工3 test、26軸join・CSV・source・整数積算の検算PASS。関連D/B branchで同候補・同規則・同配分の後続成果は調査範囲で未確認（[§10.5照合](SCOPE_AND_PREFLIGHT.md)）。

統合担当への差分：**hの比較用価格26/26軸と、この保存Q3配分のモデル費用が利用可能になった。新しい直接証拠は0軸、直接10/26のまま。全費用の98.389%は外挿。** ロードマップ本体は統合担当に任せる。global optimum、winner、別配分の十分N、全方式比較は未判定。

commit/push・remote照合後 **mandatory STOP**。追加compile/QPY/fit/Q3/配分生成へ進まない。
