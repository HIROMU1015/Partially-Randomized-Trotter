# H8 native B0+I 全登録候補：実測結果と保存PRとの比較

**全12構成の適格性判定と、必要な8構成・16軸の実compileを正常完了した。** 残るprefix7の4構成・8軸は全条件で不適格のため未測定で、費用0ではない。
P=0のshot-weighted RZ点費用では、B0/B0+I/full S2/full S4の登録内適格最良に対し、保存PR確認union6の点最良が5精度すべてで23.83〜56.12%低かった。
これは固定DFモデル・登録範囲・compiler・経験的数値allowance・有限標本に条件付けた比較であり、母平均winner、厳密数値certificate、最終energy/QPE/RPE費用の認定ではない。

[全表・監査・公開資料だけの検証入口](../../artifacts/resource_applicability/track_a_h8_native_b0i_full_family_result_v1/2026-10-16/README.md)。[H4/H6/H8を区別した整理と次の判断](track_a_h4_h6_h8_strengthened_baseline_status_20261016.md)。
本書は完了済み結果の監査・保存算術・文書化であり、新科学計算・追加compile・新grant・原稿改訂・新GPT科学reviewは0。

## 条件と固定来歴

- linear H8、1.00 Å、STO-3G、16 spin modes、Na=Nb=4、sector dimension4900。保存tol-only DF rank28、tol1e-8、cutoff0、signed generation order、weighted Hermitian projection、固定Hamiltonian/state。
- T=.8、S2、B0+I prefix L_D=7/14/21 × q=1/2/4/8、δ=T/q=.8/.4/.2/.1、native symmetric-directional full Hadamard wrappers、既存structured Gaussian lowering。
- full-Fock tail identity `C=Σtail λ/4*((Tr G)^2+Tr(G^2))`を保持する。native最終scalarへ一度だけ加え、保存B0 signalを`exp(-iTC)`で回転する。full-H referenceを回転しない。固定sector traceへの置換はしていない。
- compiler basis rz/sx/x/cx、optimization1、seed17、no topology。compiler hash `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。既存private Python3.11.0rc1/Qiskit1.3.0を無変更で使用。profile digest `622ce4a8391002558b877240b48d831d9bb0a72aeabba2e4d73bfec64b9c9f12`。旧host compiler出力の完全bit同一性は未検証。
- execution SOURCE `54ed98c5fa5b7190f77255271f4e3bec8db15630`、launch REVIEW `e009b368ed424dead82aeb3a934edbaee5ed1626`、科学provider SOURCE `c31ef424175c02379eafebe8fc6c7548c1d9aabe`。run ID `h8-native-b0i-full-family-run01-20261015`。[起動認可と履歴](track_a_h8_native_b0i_authorized_launch_20261015.md)。
- 比較用の旧B0/full S2/full S4とPRは[固定H8 D0結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3d910ee1de4597b8895e8506500553dbad5a5085/docs/research/track_a_h8_d0_saved_value_result_20261015.md)のH8_CONFIRMATION_UNION6、26候補×10条件=260行。PRは固定union6の独立32 trajectoryだけで、探索n8・H6のunion8を混合しない。PR費用を再compileしていない。

## 適格性と取得範囲

εは有限時間complex signalの精度であり、Haやenergy精度ではない。経験的allowance factor1/10の全120条件で適格32行・不適格88行。factorによる適格判定の変化はなかった。下表はfactor1。

|prefix / q|費用取得|ε=.05|.01|.005|.001|.0005|
|---|---|---|---|---|---|---|
|7 / 1|未測定・省略|不適格|不適格|不適格|不適格|不適格|
|7 / 2|未測定・省略|不適格|不適格|不適格|不適格|不適格|
|7 / 4|未測定・省略|不適格|不適格|不適格|不適格|不適格|
|7 / 8|未測定・省略|不適格|不適格|不適格|不適格|不適格|
|14 / 1|新規2軸|適格|不適格|不適格|不適格|不適格|
|14 / 2|新規2軸|適格|不適格|不適格|不適格|不適格|
|14 / 4|新規2軸|適格|不適格|不適格|不適格|不適格|
|14 / 8|新規2軸|適格|不適格|不適格|不適格|不適格|
|21 / 1|新規2軸|適格|不適格|不適格|不適格|不適格|
|21 / 2|新規2軸|適格|適格|適格|不適格|不適格|
|21 / 4|新規2軸|適格|適格|適格|不適格|不適格|
|21 / 8|新規2軸|適格|適格|適格|適格|適格|

`registered_domain_cost_coverage_complete=true`は登録内の適格な候補に費用欠測がないという意味。全24軸をcompileしたこと、q16以上や他prefixも含む最適性とは区別する。[全120条件CSV](../../artifacts/resource_applicability/track_a_h8_native_b0i_full_family_result_v1/2026-10-16/case_eligibility_v1.csv)。

## 主比較：P=0、RZ点費用

`G_RZ=N_cos*C_RZ,cos+N_sin*C_RZ,sin`。Nは保存signalと既定headroom/shot規則から導出した解析的十分shot数で、量子hardwareで実行したshotsではない。主表示はfactor1、PRも固定確認集合内の費用による点選択である。

|ε|登録内決定論的点最良|決定論的 G_RZ|PR G_RZ点推定|PRの点費用低減率|
|---:|---|---:|---:|---:|
|0.05|B0I_p14_q1_2nd|1,921,883,184|1,368,998,259.84375|28.77%|
|0.01|B0_p21_q4_2nd|128,573,351,900|60,792,079,233.40625|52.72%|
|0.005|B0_p21_q4_2nd|627,154,922,800|296,433,476,425.75|52.73%|
|0.001|B0_p21_q8_2nd|35,005,009,439,312|15,360,006,659,421.375|56.12%|
|0.0005|B1_p28_q4_4th|186,504,990,295,360|142,054,819,336,237.5|23.83%|

ε=.05の分母は、旧B0 p21/q1からB0+I p14/q1へ変わった。PRの点低減率は旧B0/B1比較の約39.48%から28.77%へ小さくなった。ε=.01/.005/.001は旧B0、.0005はfull S4が登録内最良のままである。「B0+I追加で全精度の分母が改善した」「精度が厳しいほどPRが有利」とはしない。

5精度×factor1/10×6指標の全60条件でPR/決定論的点費用比は1未満だった。[全比較CSVと選択PRの保存paired SE](../../artifacts/resource_applicability/track_a_h8_native_b0i_full_family_result_v1/2026-10-16/comparison_P0_v1.csv)。SEは標本SEで、同時CIや選択後の母平均winner保証ではない。状態準備P=0での結果を一般的な決定論的方法全体の最適費用へ外挿しない。

## 回路費用とnative correctness

各行は1軸・1wrapper当たり。全8構成でcosine/sineの6指標が一致した。軸ごとのwall/CPU/RSSとnumeric fingerprintは[16軸CSV](../../artifacts/resource_applicability/track_a_h8_native_b0i_full_family_result_v1/2026-10-16/axis_compiled_metrics_v1.csv)へ保存した。

|prefix / q|RZ|CX|RZ depth|CX depth|depth|size|
|---|---:|---:|---:|---:|---:|---:|
|14 / 1|53,448|23,540|5,755|9,169|17,439|110,919|
|14 / 2|105,304|46,664|11,383|18,274|34,596|218,823|
|14 / 4|209,016|92,912|22,639|36,484|68,910|434,631|
|14 / 8|416,440|185,408|45,151|72,904|137,538|866,247|
|21 / 1|78,719|34,952|8,574|13,737|26,028|163,650|
|21 / 2|155,846|69,488|17,021|27,410|51,774|324,285|
|21 / 4|310,100|138,560|33,915|54,756|103,266|645,555|
|21 / 8|618,608|276,704|67,703|109,448|206,250|1,288,095|

全8構成のcontrol-left/right・X/Yと保存独立forward overlap参照はPASS。最大数値差1.14530e-13（契約許容1e-8）。全16軸の入力fingerprintは検証時・compile前後で一致し、固定compiler一致・worker native exit0だった。保存参照との数値一致を全状態のoracleや厳密近似誤差certificateとはしない。

同一環境・同一numeric条件のmatched旧B0再compileは行っていない。全`matched_B0_delta=null`を保持し、旧B0との単純gate数差をidentity追加だけの純効果とはしない。

## 仮想状態準備費用P

全方式共通の仮想per-shot加算`G(P)=G(0)+N_total*P`。Pは未実測で、RZならRZ/shotという単位を持つ。候補再選択lower envelopeを全60条件・284区間で保存した。factor1/RZで最初に決定論的な線が下包絡へ入る境界（境界では同費用）は次の通り。

|ε|P境界：RZ/shot|その直後の決定論的候補|
|---:|---:|---|
|.05|約581,954.8|B0I_p21_q2 / B0_p21_q2（同費用）|
|.01|約4,736,142.5|B0_p21_q8|
|.005|約4,738,357.2|B0_p21_q8|
|.001|約1,212,916.6|full S4 p28/q4|
|.0005|約108,369.8|full S4 p28/q4|

厳密有理数の端点・全tie・切替候補を[JSON](../../artifacts/resource_applicability/track_a_h8_native_b0i_full_family_result_v1/2026-10-16/P_sensitivity_v1.json)と[区間CSV](../../artifacts/resource_applicability/track_a_h8_native_b0i_full_family_result_v1/2026-10-16/P_envelope_segments_v1.csv)へ保存した。確認union6の外のB2候補を混ぜていない。P増加後もPRの点優位が続くとは限らず、表示丸めを正本にしない。

## 終了・資源・監査

- launcher wall 8,381.611483秒（約2時間19分42秒）。driver CPU3、worker CPUs4/5/6/8、内部thread1、2構成ずつ最大4軸並列。最初の2回はCPU busy基準で起動前に拒否され、科学呼出・one-shot消費0の証拠を保全した。科学runは1回。
- snapshot load1、prefix preparation3、tail scalar12、analytic overlay12、native build8、measured wrapper16、probe copy16、control probe32、compile16。旧PR再取得、新SCF/DF/state/PR trajectory/reference exponential action生成は0。
- driver peak RSS観測10.94667GiB、worker wait4 peak最大11.13149GiB。COW共有page・親からの継承を含み得るprocess別peakで、足して固有の物理RAM消費量とはしない。AS64GiB/RSS32GiBの上限を実消費と混同しない。
- worker CPU合計5,525.684436秒、driver最終観測CPU6,723.18秒。observer8,309回、最大interval1.021465秒、最大観測所要0.016722秒、first STOPなし。最終host/cgroup PSI0、OOM増加0。compact JSONLは全時点のPSIを保存していないため、全観測でPSI0だったとは主張しない。
- driver native wait0、recorder0、全16worker exit0・4wave全reap・cleanup error0。元driver/launcher/worker identityは残存0。正確なexit timestampはnullを保持する。
- terminalが結合した158出力のbytes/SHAを照合。source13ファイルのdisk/Git SOURCEを照合し、contract/auth/seal/input/profile/carryのdigestを確認した。profile34,273ファイルの再inventoryは公開作業で繰り返さず、固定runnerの最終profile検査とCOMPLETEを根拠にする。
- 保存shot×実費用・最小候補選択612項目と、全P区間/端点の11,396厳密有理数検査がPASS。元の科学scalar・signal・shotを再生成しない。[終了監査](../../artifacts/resource_applicability/track_a_h8_native_b0i_full_family_result_v1/2026-10-16/completion_audit_v1.json)。

元JSONのcomparison欄名`confirmed_PR_union8`はH6由来の表示名残りで、実入力・source guard・候補集合はすべてH8のunion6/独立32を使用していた。公開projectionでは`confirmed_PR_union6`へ名前だけ訂正した。元結果・source・hashは不変で、数値変更や再実行をしていない。

経験的allowance、保存binary64係数の算術、有限標本と未観測希少event、点選択、旧compiler bit再現性、state preparationの実測、energy/RPE総費用の限界は残る。新科学的採択をCodexで追加せず、詳細科学reviewを自動開始しない。H5/H7・T比較の新計算は別途承認とし、今回は結果公開の準備後mandatory STOP。
