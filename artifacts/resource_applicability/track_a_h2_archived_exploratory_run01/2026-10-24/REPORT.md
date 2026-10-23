# H2 PR/PF 4方式：保存値による探索的総費用比較 run01

**承認された探索的会計を一回実行し、4方式すべてが既存empirical allowance下で適格となった。点推定ではFullはS2、PRはYoshida4が低費用で、次数の順位逆転候補を確認した。4方式の点推定最小はFull S2。正式な共通native同等性認定や母平均優位性の保証ではない。**

2026-10-24 JST、GPUA100-gleap02。専用branch `codex/h2-archived-exploratory-ready-20261023`。利用者の「計算を実行して　予定外のところで中断された場合は修正を行って再計算して」に基づく実行。主担当Codex自身の実行・技術監査であり、別agentや独立科学レビューではない。

## 結果（保存出力の引用）

| 方法 | 費用trajectory数 | C平均/RZ・軸 | analytic N合計 | G合計/RZ | paired G標準誤差 |
|---|---:|---:|---:|---:|---:|
| Full S2 | 1 | 1655 | 223,624,782 | 370099014210 | — |
| Full Yoshida4 | 1 | 2463 | 152,268,200 | 375036576600 | — |
| PR S2 | 33 | 3914.5152 | 223,754,163 | 875889061278 | 7.3380809e+09 |
| PR Yoshida4 | 34 | 2931.0294 | 158,014,863 | 463146210949 | 4.1314535e+09 |

全方式でcosine/sineの保存C平均は等しい。PRのCは元33/34 trajectoryの標本平均、Fullは各候補の保存1組の単価。Gは元COMPAREのmultiply-then-divide演算順を維持した主値である。表は表示を丸めており、正確な保存値・軸別C/SE/N/Gは[CSV](four_method_saved_output_table.csv)、全u/headroom/normalization/shot診断と不確かさは[comparison.json](comparison.json)。CSVとこの表は出力fieldの転記で、会計kernelの再実行ではない。

元ranking出力は `full_Y4_cheaper=false`、`PR_Y4_cheaper=true`、`reversal_candidate=true`、`causal_attribution=null`、`science_GO=false`。選択したFull S2/q4・Y4/q2とPR S2/q4・Y4/q2の4点についての探索的順位であり、逆転原因の帰属は行っていない。新しい費用比・削減率・区間・追加rankingは計算していない。

## 固定条件と実際の処理

Linear H2、隣接1.0 Å、STO-3G、tol-only DF1e-8/rank3、T=.8、ε_signal=.0005、α_axis=.025、P=0、system4/ancilla4、usual signed bias、元symmetric-directional/scalar/full Hadamard仕様。Fullはprefix3のS2/q4とY4/q2。PRはprefix1/K4/outer32のS2/q4 `H2O32_c7ec590a32a859bc` とY4/q2 `H2O32_8a2ecdd0294ffb74`。元cohort・67 trajectory/134軸は維持した。

共通Full保存単価は1655/2463 RZ/軸で、旧Full1367/2033 RZは使用しない。PR/Full費用は固定B/B-2のPython3.12系保存結果、bias/reference/log_B/allowance入力は受入済みDIAGNOSTICSから継承。計算用runtimeは封印済みPython3.11のstdlib算術で、保存回路の再生成や再compileではない。

既存 `Backend.allowance/account/rank`、`empirical_allowance/u_aware_shots/axis_headroom`、`resource_row/metric_summary` をSOURCEどおり再利用。[事前準備・再利用一覧](../../track_a_h2_archived_exploratory_ready/2026-10-23/REPORT.md)。

## 承認・SOURCE・call・終了

- 契約：`H2_FOUR_METHOD_ARCHIVED_COST_EXPLORATORY_V1`。
- SOURCE：`aca689060d302b4e1f37d32d7075c9266b637f56`。
- 契約canonical SHA256：`a71201e23738f5b7117213fb94dfe265d307ca4f194de2ef0defc55a950d6e23`。
- input manifest file SHA256：`e30c4de7ef9a48fd04696367bb2dda68ba8829cc4ba8026598af0d337775fffe`。
- 元比較出力SHA256：`71c87b547d0ba963b26f613bb3533cbd06620e5c80a261d18f6336b1ea0f78ad`。

[実行承認receipt](authorization_receipt.json)と[直前preflight](fresh_preflight.json)。新しいUNUSED grantを1件発行し、開始時に新run_rootのCONSUMED markerへ記録した。承認・grant・内部絶対pathを含む原本はprivate領域に保持し、Git管理資料へコピーしていない。

[process receipt](process_receipt.json)：native exit0、親のwait完了・child回収、wall約1.206秒。**失敗0、中断0、再計算0、SOURCE修正0。** 技術中断がなかったため、承認済みの修正/再計算は実施していない。将来の修正時も旧marker/partial/消費を上書きせず、別SOURCE/契約/attempt/grantへ分離する方針で受領した。

[exact ledger](execution_ledger_receipt.json)：外側bootstrap1/bundle_load1/allowance4/accounting4/cost_statistics4/ranking1、内部empirical_allowance4/u_aware_shots4/axis_headroom8/resource_row4/metric_summary8、合計43 event。sampling/circuit_build/lowering/compile/transpile/simulate/signal/reference_recompute/setup/prepare/primitive評価はすべて0。追加のallowance/会計/統計/順位kernelの呼出しも0。

[元completion](native_completion.json)と[元STOP](native_STOP.json)に、9結果fileのSHA、正確な消費、SOURCE、marker SHA、mandatory STOPが保存された。結果file全文SHAと43 ledger、契約、authority/markerの対応を終了後に再照合した。[出力引用監査](output_quote_audit.json)。有効未使用grantは残さず、同grant/同run_rootの再実行は拒否する。

## 科学的範囲と保全

PRの標準誤差はcosine/sineをtask単位で対にした、保存費用標本の有限標本変動である。母平均保証、未観測eventの費用保証、選択後coverage、全誤差源を含む統計区間ではない。新しい名目区間の規則は導入せずnullを維持した。未観測event・欠測費用を0で埋めていない。

Full全parameter streamの共通native照合、完全historical load、provider差の完全な動的非干渉、3.12/3.11出力同値性、Aの独立diagonal証拠は未解決のまま。本探索ではユーザーの指示どおり開始条件から除外し、結果の限界として保持した。single-state correctnessをall-state/operator保証へ、empirical allowanceを数学的certificateへ昇格しない。

旧strict COMPAREは**BLOCKED**のまま保全。探索的結果と正式採択結果を混同しない。詳細科学レビュー・原因分解・最終evidence freezeは開始していない。

[保全監査](preservation_audit.json)は既知の旧保護対象・今回の996封印file・準備済み成果物の計16,853 fileの前後SHA/bytesを確認した。旧SOURCE/input/結果/grant/ledger/marker/STOPを変更せず、共有venv/OS/scheduler/他ユーザーprocessへの変更0。人工CPU/RAM上限を追加せず、今回の小さな算術と非返却ledger順序に従う単一processで実行した。

SOURCE不変更のため、準備段階の人工108件＋封印runtime人工smokeを再利用し、同じ人工suiteは再実行していない。今回新たに行ったのは承認済み実会計と、その出力/ledger/SHA終了検査。[最終監査](final_verification.json)。

担当ユーザー：この4点の探索結果と限界を確認する。後続の科学レビュー・因果分析・正式比較に進む場合は別の指示/承認として扱う。担当Codex：今回の計算・結果保存・終了監査を完了し、共通運用指示のコミットに含まれるプッシュ許可に従って、公開可能な成果物だけをcommit・pushする。private原本・authorityは公開しない。追加計算を始めない。**mandatory STOP。**
