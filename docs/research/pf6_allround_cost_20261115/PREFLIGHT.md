# PF6全13round集計：固定入力の確認と重複照合

2026-11-15 JST、B2（Claude Code、主agentのみ。独立レビューではない）。

ユーザーは、未観測60軸に対するCの閉包価格の条件付き採用と、今回の集計を承認済みである。集計の入力となるCのHANDOFF_Bと60軸の機械可読価格表は、**この文書の作成時点では未公開**だった。そのため、ここでは固定入力の確認と集計コードの準備だけを行った。60価格の算出はCの担当であり、B2は行わない。

## 1. 固定入力の確認

| 入力 | 固定SHA・path | 確認した内容 |
|---|---|---|
| Bの既知8round小計K | `540b54ab2aa01a164f913d5b18497e711ae4a29c`、`docs/research/pf6_eight_round_cost_20261115/`の`PARTIAL_SUMS.json`・`INPUTS.json`・`REPORT.md`・`HANDOFF.md` | 6候補のKが指示書の値と一致した（下表）。`INPUTS.json`は156軸を持つ：直接価格96軸（`ACCEPTED_DIRECT`）、未観測60軸（`UNOBSERVED_ROUND`、`C=null`）。issueは0件 |
| Q3固定shot配分 | `8dba096dc39788b8948af9eafe394f03d386c520`、`docs/research/pf_shortlist_q3_fixed_results_20261115/` | 6候補ともX/Yで`[59,55,51,47,43,39,35,31,27,23,19,15,11]`。保存DP端点は全件モデル閾値に適合。親driverの不具合で、最終API returnとCPU・peak RAMが欠測した部分保存のまま（再評価しない） |
| 閉包価格の適用条件 | `c3f761f0af5b21207e64051c9c619858064450fe`、`docs/research/pf_closure_price_recommendation_20261115/` | 推奨時点の記録。60行（6 configuration × m7/m8/m10/m11/m12 × X/Y）、各行`price=null`、`user_adopted=false`。その後、ユーザーが固定shot・全13round比較への条件付き採用を承認した |

| 候補 | 指示書のK | Bの`PARTIAL_SUMS.json` |
|---|---:|---:|
| Full PF S2（16h/7） | 115,251,570 | 115,251,570 |
| Full PF Yoshida4（16h/3） | 153,106,880 | 153,106,880 |
| Full PF M5（16h） | 207,232,138 | 207,232,138 |
| B0I p7 S2（16h/7） | 81,621,086 | 81,621,086 |
| B0I p7 Yoshida4（16h/3） | 107,719,628 | 107,719,628 |
| B0I p7 M5（16h） | 146,850,902 | 146,850,902 |

未観測60軸の対応：Bの`INPUTS.json`の60行と、Cの`RECOMMENDATION.json`の60行は、(original configuration ID、physical ID、m、axis)で1対1に対応する。Bのlabelから読める方式・公式・幅（例：`B0I_P7_M5_16h` → `TRUNCATED_PF`・`4th(m5_best)`・`16`）も、Cの行と一致する（人工testで全60行の結合を確認）。

## 2. 重複照合（共通運用指示§10.5）

- 2026-11-14以降の全remote branchのcommitとpathを確認した。同じ6候補・同じshot配分・同じ価格表による全13roundの集計結果は、確認した範囲に無かった。
- 進行中の同種作業も確認できなかった。Bのbranch head `47ccd1f37`はFull-H κ2のQ3 call準備であり、本件とは別である。
- Bの8round集計（96直接価格の軸別積とK）は、今回の入力としてそのまま再利用する。96軸の価格を読み直したり、Kを作り直したりはしない（Kは96行の`N×C`の和との一致だけを検算する）。

## 3. 準備した集計

- `allround.py prepare --c-table COMMIT:PATH[:selector]`：BとCの固定Git blobを読み、156軸を結合して`INPUTS.json`を作る。
  - 直接価格96軸は`DIRECT_COMPILED_DETERMINISTIC`、閉包価格60軸は`CONDITIONAL_FIXED_COMPILER_CLOSURE_DERIVATION`として区別する。
  - Cの行が、BのUNOBSERVED行・CのRECOMMENDATION行と、ID・方式・公式・幅・q・Tで一致することを確認する。
  - 欠落・重複・不一致は`issues`に残す。Cの行が直接価格として登録されていれば拒否する。
- `allround.py analyze`：scriptと`INPUTS.json`のcommit後に一度だけ実行する。`U=Σ_{m∈{7,8,10,11,12}}(N_X[m]C[m,X]+N_Y[m]C[m,Y])`と`K+U`を整数で計算し、順位（同額は同順位）、最小と次点の差、既知8round時点からの順位変化を出力する。issueが残る場合は総額を出さない。
- 人工test 6件PASS（[test_allround.py](test_allround.py)）：156軸の分割、整数和、X/Yの個別計上、順位変化の検出、同額の同順位、欠落・重複・不一致・直接価格扱い・非整数の拒否、Kの検算、巨大整数。C価格は架空の値を使った。
- 計算は60軸の整数演算であり、並列化・高速化は不要と判断した。

## 4. 残っている開始条件

Cが、HANDOFF_Bと60軸の機械可読価格表を固定SHAで公開すること。公開後は、追加承認なしで`prepare`→commit→`analyze`へ進む。Cの表の形式が想定と違う場合は、読取部分だけを最小限直してから進む。

## 5. 追補：Cの公開後（同日20:48 JST）

- CがHANDOFF_B（`c383105b16db7004db39b1ae270b220627d0f3b4`）を公開し、価格表 `e4458cd7548977e6e11a5a7b0c8dca0d049535d4:docs/research/pf_closure_prices_20261115/run01/PRICES.json#rows`を指定した。開始条件が成立したので、追加承認なしで集計へ進んだ。
- 表の価格fieldは`compiled_RZ`・`exact_RZ`・`mean.rz_count`だったので、読取部分をこれに合わせ、採択policy・非直接・canonical ID・model groupの検査を加えた（人工testを更新、6件PASS）。
- 集計直前に全remote branchを再確認し、同じ全13round集計が無いことを確認した。
