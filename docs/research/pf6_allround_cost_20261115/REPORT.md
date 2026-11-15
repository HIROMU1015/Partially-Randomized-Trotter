# 決定論PF6：固定shot配分での全13round総compiled RZ数

2026-11-15 JST、B2（Claude Code、主agentのみ。独立レビューではない）。

**結論の範囲**：採択済みの閉包条件と、既評価のQ3固定shot配分のもとでの、決定論PF 6候補の総compiled RZ数と順位。未観測60軸の2%保証、十分N・最適N、物理的global RMSE、5方式の正式G・winnerへは拡張しない。

## 主結果

固定shot配分は、全6候補・m0〜m12・X/Y各軸で `N = [59,55,51,47,43,39,35,31,27,23,19,15,11]`。価格Cは測定付きfull controlled Hadamard wrapper 1回分のcompiled RZ数。X/Yはそれぞれのshot数を掛けて別々に計上した。

| 候補 | 既知8round小計K | 追加5round小計U | 全13round合計K+U | 順位 |
|---|---:|---:|---:|---:|
| B0I p7 S2（16h/7） | 81,621,086 | 511,934,266 | **593,555,352** | 1 |
| B0I p7 Yoshida4（16h/3） | 107,719,628 | 658,183,130 | **765,902,758** | 2 |
| Full PF S2（16h/7） | 115,251,570 | 723,442,010 | **838,693,580** | 3 |
| B0I p7 M5（16h） | 146,850,902 | 804,431,994 | **951,282,896** | 4 |
| Full PF Yoshida4（16h/3） | 153,106,880 | 936,567,962 | **1,089,674,842** | 5 |
| Full PF M5（16h） | 207,232,138 | 1,136,164,634 | **1,343,396,772** | 6 |

- **最小はB0I p7 S2、次点はB0I p7 Yoshida4。差は 172,347,406 RZ**（次点は最小の約1.29倍）。
- 同額の候補は無い。
- **既知8round時点から順位が変わった候補組は無い**（15組すべてで順序が保たれた）。
- 隣接順位の差：2位→3位 72,790,822、3位→4位 112,589,316、4位→5位 138,391,946、5位→6位 253,721,930。
- 総額のうち追加5round（U）が約85〜86%を占める。

**Bの逆転境界との照合**：既知差が最も小さい2組でも順序は変わらなかった。
- B0I p7 M5 対 Full PF Yoshida4：境界Δ=6,255,978 に対し、U差は −132,135,968（B0I p7 M5の方がUも小さい）。
- B0I p7 Yoshida4 対 Full PF S2：境界Δ=7,531,942 に対し、U差は −65,258,880。

機械可読：[RESULT.json](RESULT.json)（順位、round別のU、最小と次点の差、順位変化の一覧）、[RESULT.csv](RESULT.csv)、[AXIS_TERMS.json](AXIS_TERMS.json)（156軸のN・C・N×Cと価格の区分・出所）。

## 価格の区分（156軸）

| 区分 | 軸数 | round | 出所 |
|---|---:|---|---|
| 直接価格 `DIRECT_COMPILED_DETERMINISTIC` | 96（各候補16） | m0〜m6、m9 | Bの8round集計が受け入れた元receipt |
| 条件付き閉包価格 `CONDITIONAL_FIXED_COMPILER_CLOSURE_DERIVATION` | 60（各候補10） | m7、m8、m10、m11、m12 | Cの採択済み60価格（`MODEL_PREDICTION`、`direct_observation=false`、`conditional_use_adopted=true`） |

各候補で既知16軸と追加10軸が1回ずつ入り、6候補で156軸になることを確認した。欠落・重複・物理候補の不一致は0件で、保留した候補は無い。AXIS_TERMSの各行に区分を残しており、直接価格と閉包価格を結果から区別できる。閉包価格は、同じ候補・roundでX/Yが同じ値だった（別行として計上している）。

## 入力（固定SHA）

| 入力 | 固定SHA・path |
|---|---|
| Bの既知8round小計K・96直接軸・60未観測軸 | `540b54ab2aa01a164f913d5b18497e711ae4a29c`、`docs/research/pf6_eight_round_cost_20261115/`の`PARTIAL_SUMS.json`・`INPUTS.json`（集計SOURCEは`732c703b0`） |
| Q3固定shot配分 | `8dba096dc39788b8948af9eafe394f03d386c520`、`docs/research/pf_shortlist_q3_fixed_results_20261115/INPUTS.json` |
| Cの60価格表 | `e4458cd7548977e6e11a5a7b0c8dca0d049535d4`、`docs/research/pf_closure_prices_20261115/run01/PRICES.json`、selector `rows` |
| CのHANDOFF_B（上の表を指定） | `c383105b16db7004db39b1ae270b220627d0f3b4`、`docs/research/pf_closure_prices_20261115/HANDOFF_B.md` |
| 閉包の適用条件（推奨時点の記録） | `c3f761f0af5b21207e64051c9c619858064450fe`、`docs/research/pf_closure_price_recommendation_20261115/` |

**入力の照合**
- Kの6値は、指示書の値、Bの`PARTIAL_SUMS.json`、96直接軸の`N×C`の和の3つで一致した。
- Q3の6候補の配分は、上のshot数と一致した。
- Cの60行は、Bの未観測60行と`(configuration_id, physical_id, m, axis)`で1対1に結合した。各行で、方式・公式・幅・q・T・model group（CのRECOMMENDATION行と一致）、canonical ID（Bのoriginal candidate IDと一致）、`direct_observation=false`、`conditional_use_adopted=true`、価格3 field（`compiled_RZ`、`exact_RZ`、`mean.rz_count`）の一致を確認した。

**閉包の適用条件**（CのHANDOFF_Bのとおり、`adoption_policy=PF_FIXED_COMPILER_CLOSURE_CONDITIONAL_60_AXES_V1`）
- 固定H4/1Å/STO-3G/tol-only DF10、元H/state/snapshot、Full10又はB0I p7、各公式の固定実幅とsigned piece、scalarは総Tで一回。
- symmetric-directional、測定付きfull controlled Hadamard wrapper、P=0。Qiskit 1.3.0、rz/sx/x/cx、opt1、seed17、coupling mapなし。
- 監査で見た障壁・反復構造・終端runが対象qまで持続するという閉包前提、installed Rustと参照sourceの対応、未調査の規模依存分岐に関する前提を置く。
- 誤差上界・CIはnull。2%保証は無い。

**Q3の部分保存の扱い**：6候補の保存DP端点はモデル閾値に適合しているが、親driverの不具合で最終API returnとCPU・peak RAMが欠測した部分保存のままである。本件はこの記録を変更・再評価していない。配分はUSER_SUPPLIEDのcontrolで、十分N・最適N・原論文nominal Nのいずれでもない。

## 実行と検証

- 手順：`allround.py prepare`で固定Git blobを結合して`INPUTS.json`を作成（issue 0件）→ scriptとINPUTSをcommit `43f87397aedc720ff1b8243038f728a3199f4e6b`で固定・push → `allround.py analyze`を一度だけ実行（exit 0、wall 0.09秒）。[EXECUTION.txt](EXECUTION.txt)。
- 計算はPythonの任意精度整数だけで行い、浮動小数点への変換は無い。60軸の整数演算なので並列化は不要と判断した。
- 実行後の検算：AXIS_TERMSの156行について`N×C`を確認し、候補別の和がRESULTのK・合計と一致した。手計算でも1候補（B0I p7 S2のU）を確認した。
- 人工test 6件PASS（[TESTS.txt](TESTS.txt)）：156軸の分割、整数和、X/Yの個別計上、順位変化の検出、同額の同順位、欠落・重複・不一致・直接価格扱い・未採択・canonical ID不一致・価格field不一致・非整数の拒否、Kの検算、巨大整数。C価格は架空の値を使った。
- 重複照合（共通運用指示§10.5）：[PREFLIGHT](PREFLIGHT.md)。同じ6候補・同じ配分・同じ価格表による全13round集計の既存結果・進行中作業は、確認した範囲に無かった。集計直前にも再確認した。Bの8round集計は入力として再利用した。

**実施していないこと**：Q3の再評価、shot探索、回路のbuild・compile、価格の再fit・再予測、96直接価格の読み直し、N・G・winnerの判定。

## 保全とSTOP

- 作業branch `codex/b2-pf6-allround-cost-20261115`（起点 `540b54ab2`）。追加したのはこのdirectoryだけである。
- B・C・Dの他branch、固定SOURCE・raw・旧結果・Q3の部分保存記録・STOPは変更していない。
- **mandatory STOP**：この結果を根拠に、Q3再評価、shot探索、追加compile、再fit、5方式のG比較へは進まない。
