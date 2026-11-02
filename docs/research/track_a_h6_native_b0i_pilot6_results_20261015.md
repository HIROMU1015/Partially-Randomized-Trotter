# H6 native B0+I：代表6回路の実測結果

2026-10-15 JST。**3構成×cosine/sineの6回路が正常完了した。今回測定したB0+Iの3構成との比較では、保存済み独立32標本のPR費用点推定が全5精度で低かった。** B0+I family全体や母平均での最良方式を確定する結果ではない。

[結果CSV・監査入口](../../artifacts/resource_applicability/track_a_h6_native_b0i_pilot6_result_v1/2026-10-15/README.md)／[正式共通ノート](track_a_shared_research_status.md)。本資料は既存計算の結果整理であり、新科学計算・compile・grant・原稿改訂・GPT科学reviewは0。pushは利用者が行う。

## 実行条件と固定identity

- run ID：`h6-native-b0i-pilot6-20261015-run04-v3`。
- execution SOURCE：`2e733d8a683d415792ce92a56c7d149b78f42716`。科学provider SOURCE：`87b6b9f46b7e8ba9abe132ee21478c114b6f46f7`。
- linear H6/1.00Å/STO-3G、12 spin modes、alpha3/beta3・sector400。保存tol-only DF1e-8/rank19/cutoff0/signed generation order/weighted Hermitian projection、保存H/stateを使用。
- T=.8、S2、prefix/q＝5/1・10/2・15/8、δ＝.8/.4/.1。B0のdiscarded-tail full-Fock identityを保持するB0+I。symmetric_directional full Hadamard wrapper、native structured Gaussian lowering。
- Qiskit1.3.0、basis rz/sx/x/cx、optimization1、seed17、topologyなし。compiler hash `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。旧環境とのoutput bit完全同一性は未確認。
- grant/seal commit：`d8cdc1a7141d65658308de0fb66a4ed67233023c`。[認可](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d8cdc1a7141d65658308de0fb66a4ed67233023c/artifacts/resource_applicability/track_a_h6_proc_exit_fix_v1/2026-10-15/rebinding_v3/authorization_v1.json)／[seal](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d8cdc1a7141d65658308de0fb66a4ed67233023c/artifacts/resource_applicability/track_a_h6_proc_exit_fix_v1/2026-10-15/rebinding_v3/seal_v1.json)。契約SHA `1cf99b426aab08f738e6d71e64e89e001941180f2648e37e08307c54997f9db4`。
- 既存launch branch/記録HEAD：`track-a-h6-pilot-proc-exit-fix-20261015` / `d1fa0974064ae4feeb1c8168a88b43f89fc58340`。[修正・起動履歴](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d1fa0974064ae4feeb1c8168a88b43f89fc58340/docs/research/track_a_h6_proc_exit_fix_20261015.md)を保全し、今回SOURCEや契約を変更していない。

H4 geometryのlegacy/ordinary系列、H6 PR確認、今回の決定論的I対照を別scopeで管理する。εは有限時間complex signalの精度で、energyのHaや最終QPE/RPE精度ではない。

## 回路資源と正しさ

cosine/sineは各構成で全6指標が一致した。下表は**1軸・1wrapper**当たりで、両軸合計ではない。

|prefix / q|RZ|CX|RZ depth|CX depth|total depth|size|保存compile wall／軸|
|---|---:|---:|---:|---:|---:|---:|---|
|5 / 1|10,670|4,764|1,475|2,013|4,184|22,069|6.38秒|
|10 / 2|39,787|18,204|5,555|7,838|15,993|83,102|45.15–45.59秒|
|15 / 8|233,661|107,860|32,831|46,848|94,833|488,752|236.93–240.05秒|

[6軸の保存指標](../../artifacts/resource_applicability/track_a_h6_native_b0i_pilot6_result_v1/2026-10-15/axis_compiled_metrics_v1.csv)にfingerprint/compiler/source、native exit、wall/CPU/RSSを保存した。compile wallは記録されたcompile区間であり、前後のfingerprint・正しさ検査・結果保存を含む全stage wallとは異なる。

独立に保存されたB0+I forward-overlapとの検査は3構成すべてPASS。最大差はp5q1＝1.07418e-14、p10q2＝1.03584e-14、p15q8＝8.94287e-14だった。全状態ベクトルの新oracleやstrict numerical certificateではない。[正しさ・旧B0参照](../../artifacts/resource_applicability/track_a_h6_native_b0i_pilot6_result_v1/2026-10-15/correctness_and_B0_reference_v1.json)。

## 要求精度への適格性

保存signal/reference・経験的allowance・既定のaxis headroom/shot会計に従った結果である。実装の数値一致PASSと、full-H referenceに対する近似精度の適格性を区別する。

|B0+I構成|ε=.05|.01|.005|.001|.0005|
|---|---|---|---|---|---|
|p5q1|不適格|不適格|不適格|不適格|不適格|
|p10q2|適格|不適格|不適格|不適格|不適格|
|p15q8|適格|適格|適格|適格|適格|

factor1/10で適格性は同じ。費用は厳しい精度で整数shotの差により僅かに変わるため、factor10を同値として省略せず保存した。不適格は費用0ではなくnull。[全30 precision行](../../artifacts/resource_applicability/track_a_h6_native_b0i_pilot6_result_v1/2026-10-15/precision_accounting_v1.csv)が正本。

## 保存PRとの総費用点比較

`G_RZ = N_cos C_RZ,cos + N_sin C_RZ,sin`、P=0。Nは既定の解析的な十分shot数であり、hardware shotsを実行した値ではない。B0+Iは今回の3構成内の適格な最小費用、PRは[固定H6 D0結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8dbdb68293c988a999645e21a82b53966f9ef6cd/docs/research/track_a_d0_saved_value_reanalysis_result_20261014.md)の確認union8×独立32標本におけるRZ点最良を使う。探索8本とpoolしない。factor1を主表示する。

|ε|今回のB0+I選択|B0+I G_RZ|保存PR G_RZ|PRの点費用低減率|
|---:|---|---:|---:|---:|
|.05|p10q2|2,034,508,245|403,423,200|80.17%|
|.01|p15q8|84,634,350,810|19,549,158,739|76.90%|
|.005|p15q8|350,007,588,069|101,836,339,182.65625|70.90%|
|.001|p15q8|11,733,932,954,004|約5.14673e12|56.14%|
|.0005|p15q8|74,457,269,554,761|約3.26277e13|56.18%|

低減率は`1−G_PR/G_B0I`の保存値算術。PRの候補ID/SE/元CSV表現と比は[5行の比較CSV](../../artifacts/resource_applicability/track_a_h6_native_b0i_pilot6_result_v1/2026-10-15/comparison_to_saved_PR_v1.csv)へ保存した。元CSVの表示精度を超えた母平均のexact値を復元したとはしない。確認集合内でも結果を見た点選択であり、SEは記述的な標本SEで同時winner保証ではない。

**これはB0+Iの代表3構成との比較であり、強化した決定論family全体に対する低減率ではない。** 同じ系でp5/p10/p15×q1/2/4/8の12構成24軸を取得した場合の最小は未確定。今回6軸を全面比較の完成や、PRの一般的優位性の確立としない。B0/B0+I/full S2/S4を合わせた決定論分母の全面再最小化も今回実施していない。

旧B0の保存metadata参照はp10q2・p15q8に存在するが、各行のprofile/full numeric identityを今回のmatched runとして確認していない。p5q1はMISSING_OR_NONUNIQUE。同一条件でのmatched B0再測定は0、全構成の`deltas=null`を維持する。旧指標との単純差をscalar追加だけの純効果と報告しない。H4の「追加費用0」をH6へ移送しない。

## 資源・終了・監査

- 全stageのlauncher wallは2,262.029280498624秒（約37分42秒）。最大worker peak RSSは4,192,169,984 bytes（約3.90GiB）。fork後の共有pageを含むprocess別値で、合算して独立物理memory量や予約量としない。
- SOURCE/input/profile/carryは元契約のbindingを保持。snapshot load1、overlay3、native evolution build3、measured wrapper6、probe copy6、control probe12、compile6。今回の資料作業ではこれらを再実行していない。
- 最大6worker/内部thread1、driver CPU62・workers61/60/59/56/49/47。数値各AS32GiB/RSS9GiB、admission240.5GiB、既存metadata launcherの独立観測を維持。保存observer最大間隔1.050923秒、最大観測所要.013573秒で、5秒監視のSTOPなし。
- driverの同じ親によるwait0、recorder0、全6worker exit0/reaped、cleanup errors[]。first STOP null、`COMPLETE_MANDATORY_STOP`。terminal内部のnative_exit_code=nullと外部wait receiptの0は別に保持し、正確な終了時刻nullを後から補完しない。
- 元terminalに列挙された結果/cleanup/6cost/3correctnessの11fileのbytes/SHAを照合。原runtime/checkpoint/ledger・native identity・入力NPZ・venvを一括公開せず、公開用projection/CSVだけを作った。旧STOP、消費済みgrant、known/held/unknown carryを保全し返却/reset0。

[結果summary・資源・来歴](../../artifacts/resource_applicability/track_a_h6_native_b0i_pilot6_result_v1/2026-10-15/result_summary_v1.json)、[既存完了metadata受領](../../artifacts/resource_applicability/track_a_shared_status_server_update/2026-10-15/h6_completion_metadata_receipt_v1.json)、[今回の限定文書検証](../../artifacts/resource_applicability/track_a_h6_native_b0i_pilot6_result_v1/2026-10-15/documentation_checks_v1.json)を参照する。科学計算の外部再現・独立GPT科学review・CIではない。

既定wallなし政策と現契約4hの明示例外未確認は[先行の来歴記録](track_a_server_information_update_20261015.md)を維持する。今回の終了を理由に契約変更・方針差の解決としない。

## 保存・公開と残る範囲

直前の共通ノート追補commit `6b78e5f39917d92d6c60694d3e0b0d06394d7eb5` から結果専用branch `track-a-h6-native-b0i-pilot6-results-20261015` を作成。科学SOURCE/実行worktreeは変更しない。前commitの資料も本branchの祖先として保持する。

今回の利用者指示は「まとめた結果のcommitとpush command提示」。結果資料・CSV・軽量監査・索引を明示stageしてcommitし、push自体は行わない。未pushをGitHub公開済みとは表示しない。

未測定native対照、同一条件matched B0、仮想Pを超えたstate preparation、経験的allowanceの証明、未観測rare-event費用・母平均winner、energy/QPE/RPE最終費用は別課題。次の科学的な採否や全面24軸への実行承認を本資料で代行しない。

本runは完了済み。automatic_retry=false、next_stage_authorized=false、mandatory STOP。新SCF/DF/state/signal/trajectory/compile、既存job・共有環境・venv・H8変更、新grant・原稿改訂は今回すべて0。commitとpush command提示後に停止する。
