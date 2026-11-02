# H6 native B0+I 全12構成：取得結果と保存PRとの比較

**登録12構成の適格性と必要な費用取得を完了した。新規12軸＋旧pilot6軸を取得済みとし、残る6軸は全条件で不適格のため未測定。**
P=0のshot-weighted RZ点費用では、B0/B0+I/full S2/full S4から選んだ決定論的最良に対して、保存PR確認集合の点最良が全5精度で29.35〜56.14%低かった。
取得済み集合と既定の経験的allowanceによる比較で、母平均winner・全探索最適性・最終energy/QPE/RPE費用の証明ではない。

[結果・CSV・監査入口](../../artifacts/resource_applicability/track_a_h6_native_b0i_full_family_result_v1/2026-10-15/README.md)。
今回は完了済み記録の整理・保存値算術・文書commitのみ。新科学計算・追加compile・新grant・原稿改訂・GPT科学reviewは0。push自体は利用者が行う。

## 条件と来歴

- linear H6、1.00Å、STO-3G、12 spin modes、保存tol-only DF rank19/state、signed generation order。新SCF/DF/state/trajectory/signal exponential actionはない。
- T=.8、S2、B0+I prefix L_D=5/10/15 × q=1/2/4/8、δ=T/q=.8/.4/.2/.1。full-Fock tail identity、symmetric_directional full Hadamard wrapper、既存native structured Gaussian lowering。
- compiler basis rz/sx/x/cx、optimization1、seed17、compiler hash `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。固定private profileを無変更で使用。旧compiler outputの完全bit同一性は未検証のまま。
- 今回のexecution SOURCE `ac6d28e35cf01398f169ce59577182a38da9b959`、launch review `e8e182aa5ff89691414b21d6ad30cc8a52a7d221`、provider `87b6b9f46b7e8ba9abe132ee21478c114b6f46f7`。run ID `h6-native-b0i-full-family-run04-20261015`。
- 旧pilot p5q1/p10q2/p15q8の6軸は元SOURCE `2e733d8a683d415792ce92a56c7d149b78f42716` の実測としてREUSE_WITH_DECLARED_LIMITS。新SOURCEで再compileしたとはしない。[旧pilot結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ae3b992f22aead88515ef594cde7e5f867aefa87/docs/research/track_a_h6_native_b0i_pilot6_results_20261015.md)。
- PRと旧決定論的対照は[固定D0結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8dbdb68293c988a999645e21a82b53966f9ef6cd/docs/research/track_a_d0_saved_value_reanalysis_result_20261014.md)のH6_CONFIRMATION_UNION8（28候補×10条件）を使用。PRは確認union8/n32だけで、探索n8を混合しない。

## 適格性と取得範囲

εは有限時間complex signalの要求精度で、Haや最終energy精度ではない。factor1/10で適格性は一致したが、整数shot・費用の微差は両方を保存した。

|prefix / q|費用取得|ε=.05|.01|.005|.001|.0005|
|---|---|---|---|---|---|---|
|5 / 1|旧2軸を再利用|不適格|不適格|不適格|不適格|不適格|
|5 / 2|未測定・取得省略|不適格|不適格|不適格|不適格|不適格|
|5 / 4|未測定・取得省略|不適格|不適格|不適格|不適格|不適格|
|5 / 8|未測定・取得省略|不適格|不適格|不適格|不適格|不適格|
|10 / 1|新規2軸|適格|不適格|不適格|不適格|不適格|
|10 / 2|旧2軸を再利用|適格|不適格|不適格|不適格|不適格|
|10 / 4|新規2軸|適格|不適格|不適格|不適格|不適格|
|10 / 8|新規2軸|適格|不適格|不適格|不適格|不適格|
|15 / 1|新規2軸|適格|不適格|不適格|不適格|不適格|
|15 / 2|新規2軸|適格|適格|適格|不適格|不適格|
|15 / 4|新規2軸|適格|適格|適格|適格|不適格|
|15 / 8|旧2軸を再利用|適格|適格|適格|適格|適格|

12構成×5精度×factor1/10の120行は、適格34行・不適格86行。うち取得済み費用に対応する不適格56行、未測定の不適格30行。
全条件で不適格な新p5q2/q4/q8の6軸はbuild/compileを省略し、費用を0や成功実測と扱わない。旧p5q1は歴史的実測を保全する。
`registered_domain_cost_coverage_complete=true`は登録内の適格な費用欠測がないという意味で、全24軸をcompileした意味ではない。

## 回路費用とnative正しさ

表は1軸・1wrapper当たり。各構成でcosine/sineの6指標が一致した。総shot費用は次節の軸別会計であり、この表の両軸合計ではない。

|prefix / q|来歴|RZ|CX|RZ depth|CX depth|depth|size|
|---|---|---:|---:|---:|---:|---:|---:|
|5 / 1|旧再利用|10,670|4,764|1,475|2,013|4,184|22,069|
|10 / 1|新規|20,292|9,208|2,825|3,943|8,101|42,311|
|10 / 2|旧再利用|39,787|18,204|5,555|7,838|15,993|83,102|
|10 / 4|新規|78,777|36,196|11,015|15,628|31,777|164,684|
|10 / 8|新規|156,757|72,180|21,935|31,208|63,345|327,848|
|15 / 1|新規|29,905|13,668|4,187|5,898|12,037|62,424|
|15 / 2|新規|59,013|27,124|8,279|11,748|23,865|123,328|
|15 / 4|新規|117,229|54,036|16,463|23,448|47,521|245,136|
|15 / 8|旧再利用|233,661|107,860|32,831|46,848|94,833|488,752|

新6構成のcontrol/X-Y・保存独立forward overlap参照は全PASS、最大差5.55433e-14以下（契約許容1e-8）。
測定wrapperの実入力はcompile前後SHA一致・compiler一致・worker native exit0を確認した。
[軸別CSV](../../artifacts/resource_applicability/track_a_h6_native_b0i_full_family_result_v1/2026-10-15/axis_compiled_metrics_v1.csv)と[正しさ記録](../../artifacts/resource_applicability/track_a_h6_native_b0i_full_family_result_v1/2026-10-15/new_case_correctness_v1.json)を参照。
保存参照とのnative数値一致は近似誤差のcertificateではない。全状態ベクトルの新oracleは作成していない。

## 主比較：P=0、RZ点費用

`G_RZ = N_cos*C_RZ,cos + N_sin*C_RZ,sin`。Nは保存signal/headroomと既定規則から得た解析的十分shot数で、hardware shots実行数ではない。
分母は登録B0/B0+I/full S2/full S4内の適格な最良。PRも確認済み集合内で費用を見て選ぶ点最良であり、同時winner保証ではない。主表示はfactor1。

|ε|決定論的点最良|決定論的 G_RZ|PR G_RZ点推定|PRの点費用低減率|
|---:|---|---:|---:|---:|
|0.05|B0I_p10_q1_2nd|571,037,172|403,423,200.0|29.35%|
|0.01|B0I_p15_q2_2nd|38,854,513,278|19,549,158,739.0|49.69%|
|0.005|B0I_p15_q4_2nd|217,936,448,114|101,836,339,182.65625|53.27%|
|0.001|B0I_p15_q8_2nd|11,733,932,954,004|5,146,731,715,972.781|56.14%|
|0.0005|B1_p19_q4_4th|65,737,570,406,690|32,627,743,438,156.03|50.37%|

ε=.05ではB0+I p10q1が旧B0 family点最良より約18.56%低く、新しい決定論的最良となった。
ε=.01/.005/.001でもB0+I p15q2/q4/q8が分母になり、ε=.0005ではfull S4 p19q4が最良のまま。
旧pilot3構成だけとの大きな低減率（ε=.05で80.17%等）を今回の強化した分母へ流用しない。

[全60比較CSV](../../artifacts/resource_applicability/track_a_h6_native_b0i_full_family_result_v1/2026-10-15/comparison_P0_v1.csv)では、5精度×factor1/10×RZ/CX/RZ depth/CX depth/depth/sizeの全60条件でPR/決定論的点費用比が1未満。
他5指標は補助解析。全120行のshot/eligibilityと全family minimaもJSONに保存した。母平均・未観測rare-event・全候補の最良は未証明で、経験的uはcertificateではない。

同一条件matched B0のprofile/numeric同一性は取得していない。全`matched_B0_delta=null`を維持し、単純な旧gate数との差をI追加だけの純効果としない。
新SOURCEで旧pilotのcompiled bytes完全同一性も主張しない。

## 仮想状態準備費用Pの感度

`G(P)=G(0)+N_total*P`という全方式共通の仮想per-shot加算モデル。RZならPの単位はRZ/shotで、実際のstate preparation costは未測定。
候補の再選択を含むlower envelopeを全60条件で保存した。factor1/RZで最初にPR以外の線が下包絡へ入る境界は次の通り（境界では同費用）。

|ε|P境界：RZ/shot|その直後の決定論的候補|
|---:|---:|---|
|0.05|約896,865.4|B0I_p15_q4_2nd|
|0.01|約1,882,868.0|B0I_p15_q8_2nd|
|0.005|約1,884,282.5|B0I_p15_q8_2nd|
|0.001|約614,430.2|B1_p19_q4_4th|
|0.0005|約172,936.2|B1_p19_q4_4th|

P増加時にはPRの点優位が続くとは限らない。厳密有理数の境界・全294segmentは保存表にあり、上表の丸めを再計算の正本にしない。

## 実行・終了監査

- 全stage launcher wallは2326.545872329秒（約38分47秒）。2worker/各thread1、driver CPU63・worker CPU62/61、既存metadata launcherが独立観測した。
- 新規snapshot load1、tail scalar9、analytic overlay9、native build6、measured wrapper12、probe copy12、control probe24、compile12。旧6軸を再取得せず、元run03予約6とactual到達/完了unknownを保持し、refund/reset0。
- driver peak RSS観測5.88316GiB、worker wait4 peak最大6.19914GiB。forkの共有pageを含むprocess別値で、足し合わせて独立物理使用量としない。
- observer2306回、最大interval1.020200秒、最大観測所要0.016400秒、first STOPなし。終了直前のhost full PSIは0・OOM増分0。全期間の各PSI値をcompact JSONLから復元したとはしない。
- driverの同じ親によるnative wait0、recorder0、全12worker exit0/reaped、cleanup errors[]。terminal COMPLETE_MANDATORY_STOP。正確な終了時刻nullを後から補完しない。
- 元terminalの116file receiptをbytes/SHA照合。final source/profile/input gateを完了した固定runnerの成功terminalと元bindingを保持する。公開作業でvenv全体やNPZの大量再監査は繰り返していない。

[終了監査](../../artifacts/resource_applicability/track_a_h6_native_b0i_full_family_result_v1/2026-10-15/completion_audit_v1.json)。実runtime/checkpoint/ledger/NPZ/profile実体・認証・内部接続情報はcommitしない。
親の重複fingerprint削減とcleanup修正を適用して完走した事実であり、未完走の旧runと同一条件のspeedup測定をしたとはしない。

## 保存・検証と停止

結果専用branch `track-a-h6-native-b0i-full-family-results-20261015` をlaunch reviewから作成し、実行worktree/旧結果/STOP/grantを保全した。
保存されたscalar tableだけの検査で、120会計行、18取得済み軸、60比較、60lower envelope/294segmentの2,607条件PASS。
[再照合用stdlib script](../../artifacts/resource_applicability/track_a_h6_native_b0i_full_family_result_v1/2026-10-15/verify_public_tables_v1.py)は固定D0 Git blobをread-onlyで参照し、新科学API・compileを呼ばない。CIや別担当による外部再現ではなく主担当の照合で、subagents0。

研究上の採否は次のGPT科学reviewへ戻す。q16、matched B0、実状態準備費用、H8/H5/H7、別T/PF、次stageの計算認可は追加しない。
pushは未実行。軽量結果と文書をcommitし、実行可能なpush commandを提示してmandatory STOPする。
