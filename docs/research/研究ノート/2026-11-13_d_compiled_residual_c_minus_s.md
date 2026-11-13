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

## 追補：P2直接paired pilot完了（2026-11-13）

[REPORT](../d_p2_direct_pilot_20261113/REPORT.md)／[HANDOFF](../d_p2_direct_pilot_20261113/HANDOFF.md)。ユーザーが有限20 compileを明示承認し、SOURCE ea0706ab0で保存replay2 pathの全照合後にfresh8 pathを取得した。m6履歴対replay全path時間比3.56、fresh D平均はm0=187.5/m6=74,388.75。P1/replayはfresh統計に加えず、全8 freshでorder2未観測を寄与0と解釈しない。条件付きn/負荷の候補を報告したが、2%/95%・全law C・他round・N/RMSE/Gは未資格、fit/外挿・追加path・retry0。次の本格平均価格評価へ進まず公開後mandatory STOP。

## 追補：全13roundへ向けた価格評価方法の設計（2026-11-13）

[REPORT](../d_compiled_price_methods_20261113/REPORT.md)／[C_HANDOFF](../d_compiled_price_methods_20261113/C_HANDOFF.md)／[最小科学案](../d_compiled_price_methods_20261113/MINIMAL_SCIENCE_PLAN.md)。P2完了39ddd59edを基点に同branchを再利用。旧ladder/model必須案を実行前提にせず、保存E[S]＋target-R短outer-windowのconnected残差を検証候補にした。恒等式は厳密でも短windowから長qへのtail条件は未証明で、m12価格はUNKNOWN。既存C kernelのS出口閉包はcompiler閉包ではなく、符号付き残差を非負reward区間へ直接渡せない。C向けfeatureと最大24 local compileの診断案を定義したが、科学実行は未承認・未実行。人工・回帰42項目PASS、新実H4/保存P1/P2数値再解析/fit/E[C]/G/8次PF0。旧SOURCE・raw・STOPを保全し、公開照合後mandatory STOP。

## 追補：RPE候補探索の価格区間・取得優先度の設計（2026-11-13）

[REPORT](../d_rpe_price_search_design_20261113/REPORT.md)／[HANDOFF](../d_rpe_price_search_design_20261113/HANDOFF.md)／[有限科学案](../d_rpe_price_search_design_20261113/MINIMAL_SCIENCE_PLAN.md)。ユーザーの目的に合わせ、固定L_D1の全26軸高精度化や旧24 local compileを先行要件とせず、whole-schedule候補の同時価格区間・部分下界・共通価格差分による選別を設計した。C1 contextがH4 L_D1/S2/δ=.1に固定されるため新候補には薄いadapterが必要。既存C/B/DのSOURCE/APIを再利用し、候補生成/N solver/第二のcontrollerを作らない。54人工・回帰PASS、新実H4/保存値再解析/fit/E[C]/G0。外部menu/N/γ又はτが未供給で科学scopeはUNRESOLVED、必要時だけ最大16 compileの診断又は必要cellの直接取得を提案する。m12 UNKNOWN、旧SOURCE/raw/STOPを保全し、公開照合後mandatory STOP。
