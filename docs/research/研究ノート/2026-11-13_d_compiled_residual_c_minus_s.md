# 2026-11-13：compiled価格残差C−Sの評価方法（Codex D）

## 確認済み（技術）

- B `3216d27a`から新branch `codex/d-compiled-residual-c-minus-s-20261113`を作成した。
- 次を固定Gitで読み、metadataだけをinventoryにした。
  - C1の保存paired S/C（`2e6a0cdf5`）
  - S0a
  - Cの価格検証計画（`d458b5e4`）
  - CのK0/K1（`7aca17fe`）
- 保存の`paired_residual_covariance`は、cell内のCov(D)と代数的に一致する。一方、Dの平均は保存summaryになく、今回は計算していない。
- S0の13roundには、保存されたSもCもない。

## SOURCE読解での発見（未検証）

- compiled経路の制御ZZ回転は、`RZZGate.control(1)`→汎用add_controlで**CCX·CRZ·CCX**になる。
- C1 `Accounting`の`crzz`は**CX·CRZ·CX**として数える。
- この展開差がC−Sの主成分である可能性があるが、単位compileでは確認していない。
- Qiskit 1.3.0 opt1の1q置換はrun長を基準にするため、RZ数は増えることも減ることもある。

## 判断

- **最も直接な経路**：C K1のE[S]に、target roundの同一path paired D̄を足す方式。
- **m8以上**：残差modelが必要。次の順で検証する。いずれも未承認・未実行。
  1. P0：機序の単位compile
  2. P1：既存8 wrapper案の再利用（同pathでSも取得）
  3. P2：経路上ladder
- **識別可能性**：m0/m6の2点では3係数を識別できない。R≈2qの経路上では、qとRを分離できない。
- **旧係数**：C1係数・旧cluster係数は移植しない。

## 検証と範囲

- [REPORT](../d_compiled_residual_c_minus_s_20261113/REPORT.md)。
- 新13件と回帰を合わせて140 PASS。既存C byte固定test 1件は以前からのFAILで、今回の変更によるものではない。
- 新しいsampling・compile・fit・保存値の再解析・N/Gは0件。mandatory STOP。

## 追補：技術高速化と最小P2計画（2026-11-13）

[新REPORT](../d_compiled_price_pipeline_20261113/REPORT.md)と[最小P2提案](../d_compiled_price_pipeline_20261113/P2_MINIMAL_PLAN.md)。P1完了54b64ef7を基点に、元AccountingのRZ-only projection・X/Y共通本体と既存fast canonical writerをC1へ接続。人工55項目PASS・人工wall/RSS比較のみで、実H4倍率は未測定。平均価格への直接paired法を優先し、旧n=8全ladderやm8以上のmodel必須という上記提案を今回の実行前提にしない。実H4科学実行・保存値数値再解析・fit・平均C認定/N/RMSE/Gは0、P2は別途明示許可待ち。旧計画・結果・STOPは保全。
