# H4 DF10 RPE費用会計：保存m=3の実データ数値検証

2026-11-10 JST。ユーザーが明示許可した**T=.8/m=3の3価格ケースだけを一括評価し、全ケースで接続検証を完了**した。[result.json](result.json)が数値の正本。新しいN/C推定や全RPE費用・費用比率・新順位は計算していない。

## 条件と結果

Linear H4、1.00 Å、STO-3G、tol-only 1e−8の保存DF10、Nα=Nβ=2。targetは保存DF-H同sectorのground energy、保存数値state/ideal lane条件付き、準備費P=0。εE=.00159360010199040 Ha、αtotal=.05/方式、βtotal/stat=.4/.2、h=.1、全schedule m0…12を変更しない。今回の計算はm3のみ。

| case | candidate / 構成 | 保存N_X=N_Y | 保存C_RZ,X=C_RZ,Y | 古典n | G_RZ,m3=N_X C_X+N_Y C_Y |
|---|---|---:|---:|---:|---:|
| PF決定論 | 49709717a6c3d405c60d / B0/S2/p7/qPF8 | 705 | 44411 | 1 | **62619510.0** |
| PR探索 | 14bd81a375400c9b0305 / S2/p1/qPF8/R_outer4/K2 | 977 | 16857.11111111111 | 9 | **32938795.11111111** |
| PR独立確認 | 同PR candidate | 977 | 16947.78787878788 | 33 | **33115977.515151516** |

PRの2行は別cohortによる平均単価の点会計であり、poolしていない。小数の費用は保存平均単価による積和であり、実際に実行した整数gate数や母平均費用の保証ではない。PF B0の切捨てはD1のsignal diskに含まれ、Full PFへ読み替えていない。

## 入力・SOURCEの対応

| 固定版・path | 用途 |
|---|---|
| [費用準備305439c01][PREP]のPREPARED_INPUTS.json | 指定3ケースの保存N/C、cohort、candidate/binding/compiler/law_context |
| [D1 carrier ffe8a71da][D1]のresult.json | ideal_numerical_pilotのNと条件。元D1科学SOURCEは311255aa |
| e91021ae2準備のSAVED_POINTS.json | 各candidate/cohortの元compiled単価とmetrics_source。固定D1/費用準備内の同一blobを照合 |
| 価格carrier155529e2、artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/ | PF/PR探索はexploration_costs.json.gz、PR独立確認はconfirmation_costs.json.gz。元科学来歴は当該provenanceを維持 |
| **67fe43b9e4c61ca53ae5d2de57877b29e0b00af1** | 今回の数値検証SOURCE。結果公開commitとは区別 |

runnerはD1保存planをlaw_context追加分以外で完全照合し、保存Nを一切solveしない。価格は各cohortのX=cosine、Y=sineの保存RZとmetrics_source/nを照合した。controlはsymmetric_directional、完成Hadamard wrapper/測定込み、compiler hash `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。入力と会計SOURCEのhashはresult.input_sha256/source_sha256へ保存した。reference-only profileを新しいcompiler資格に昇格しない。

## 会計検証とcoverage

既存[account_subset](../../../src/trottertracks/h4_df10_rpe_cost_connection.py)を変更せず、各caseにmetrics=('rz_count',)、selected_rounds=[3]、expected_rounds=range(13)を渡して各1回呼んだ。X/Yの量子Nに対応する保存Cを乗じることを、保存binary64単価のexact-rational積和で独立照合し、全ケースで差が1 ULP以内となった。新しい価格平均や統計推定ではない。

- covered_rounds=[3]。
- missing_rounds=[0,1,2,4,5,6,7,8,9,10,11,12]：未評価12roundは0で補わない。
- full_rpe_resources=null、全RPE energy success=false、actual success qualified=false。
- total covariance / confidence interval / actual N / 保存幅はnull。

量子Nと古典nを分離し、n/Bを価格へ再乗算しない。Bは保存D1のraw測定用Nへ既に反映されている。PR有限cohort平均、ideal law対finite-bit samplerの距離、compiled実装誤差、数値stateの厳密ground同定、u/referenceの厳密資格は未保証のまま。これらを0へ置換しない。本結果は保存入力に対する条件付き費用算術の確認であり、全law期待費用・actual energy成功・全13roundのPF/PR優位を示さない。

## 技術検証・実施範囲・STOP

[薄いrunner](../../../scripts/run_h4_df10_rpe_cost_m3.py)と[固定scope PLAN](PLAN.md)を科学実行前にcommitした。runner構文を確認し、既存14人工・回帰の[保存PASS記録](../h4_df10_rpe_cost_connection_20261110/ARTIFICIAL_TEST_RESULTS.txt)を再利用した。旧会計器とD1コード/結果は無変更で、同じ人工検証やD1科学計算は再実行していない。今回の実データ算術確認は著者自身の確認であり、独立科学レビュー/immutable CIとは呼ばない。

account_subset calls=3、exact-axis-sum checks=3。新N/C推定、prior/horizon、signal、sampling、回路build/compile、model fit/bootstrap、追加round、感度/モデル残差/費用比率/新順位、全RPE/H6の評価はすべて0。1回の一括実行で完了し、数値再試行や旧run再開はない。

既存resource-eval-rpe-shot-error-connection branchを継続し、旧SOURCE/input/raw/result/grant/ledger/marker/KEEP/STOPを保全した。別担当の長時間価格モデルbranch/worktreeへ介入せず、新grantやapproval証明機構も作らない。GitHub公開後、remote SHA/tree/公開成果物を照合して**mandatory STOP**。追加round/signal/価格fit/全Gを自動開始しない。

次は**ユーザー**が本結果と並行価格調査を踏まえ、追加roundのN資格と採択候補/価格coverageのどこを次に評価するか判断する。**Codex**は新しい具体的実行許可まで追加科学計算を開始しない。

[PREP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/305439c01b6e9d8ec264045585ad77b36630275c/docs/research/h4_df10_rpe_cost_connection_20261110/REPORT.md
[D1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ffe8a71da98bec330ce1af109a9606f14e1fc598/docs/research/h4_df10_rpe_d1_20261110/REPORT.md
