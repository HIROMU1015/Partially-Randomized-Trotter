# M1-S Gate 0 精度予算の保存データ原因調査（2026-10-16）

原因は、全Hamiltonian共通の絶対係数誤差上界を小さいtailのlambda_Rで正規化する組合せである。
G3/G6は係数誤差の伝播がほぼ全予算を占め、G9は分母感度から二次項まで大きくなる。
観測した演算誤差そのものが大きいと判定したものではない。
同じ条件付きgammaモデルで局所化を検討する技術的余地はあるが、固定v4式をそのまま変えずに下げられる値ではない。
旧TECHNICAL_STOP、W/v/rho=null、正式契約・精度17項目・残差品質・coverage・SOURCE・markerを保全した。

## 対象・承認・今回行った計算

対象はlinear H4、1.00 Å、STO-3G、8 spin orbitals、DF rank12。
HDは保存DF順prefix G3/G6/G9、HRは残りの非冗長signed Pauli tail、identityはHDへ移動。
delta window・finite-step評価なし、保存数値sector oracleのみ。真の固有状態距離・gap・非縮退・総資源は扱わない。

基点はc937ed37e73bb9b0bb39f2fa7116d255a3215269、科学SOURCEは606036db52da2f080cf846e48824e5e418430762、
正式契約は5e57955baa79e42f37aa233ac011c60539e431fd。
ユーザーが保存データの項別原因調査を指定し、「必要なら追加計算し得もよい」と追記した。
この承認内で、保存NPZにあるrawスケールの再集計と、既存誤差式のscalar診断を一回行った。
新しいGate 0のauthorization、grant、markerは作成していない。

[事前に固定した診断範囲](../../../artifacts/track_b_m1s_gate0_precision_cause/2026-10-16/v1/diagnostic_contract.json)と
[診断SOURCE](../../../scripts/tracks/algorithm_codesign/analyze_m1s_gate0_precision_saved.py)はcommit d34e2fad5935348282bc43bed8c17073ac471aba。
3行×8ケースを固定し、採択済みeps_W=eps_v=2^-24を変えていない。今回のケースは結果後の原因調査であり、独立な事前登録科学実験ではない。

JSONに保存されていなかったS_Rを得るため、固定NPZのconstant/one_body/lambdas/g_matricesだけをstdlibでdecodeした。
float64 bit値をFractionへexact変換し、abs/L1/sumと既存式を有理数で評価。stateはdecodeしていない。
Pauli再生成・行列演算・W/v/rho評価・Gate0 runner起動はすべて0、保存データscalar再解析runは1。
CPU1、GPU/NumPy/BLASなし、入力は一回読んで尺度と各行Lをcache。4 GiB memory、256 MiB output、wall capなし。
新規の大きな科学workloadやbenchmarkは不要だった。

[診断結果](../../../artifacts/track_b_m1s_gate0_precision_cause/2026-10-16/v1/diagnostic_result.json)に全項・全ケースを収録。
full S・alphaと、各行5種類の保存scalar予算を上向きbinary64値まで一致させ、W/vの展開式と元式の有理数恒等性も確認した。
ファイル内decimal文字列は表示用24桁丸めであり、丸めなしのcertificateそのものではない。
旧raw result SHA256は83bf30919202b045101fba6aed9e7aec9e149581c4696c8c9e7212c70ae593c0、
NPZ SHA256は3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a。

## 1. 係数上界からW/vへ流れる項

u=2^-53、gamma_k=ku/(1-ku)、C=1000000、M=65536として、固定SOURCEは

```
S = |constant| + ||h||_(entry L1) + sum_l |lambda_l| ||g_l||_(entry L1)^2
alpha = 2 gamma_(24 C+64) S (1+gamma_(M+32))
L = sum_P |c_hat_P|, lambda_hat = saved fsum result
s = gamma_(M+32) L, delta_lambda = alpha+s
lambda_lo = lambda_hat-delta_lambda
ec = alpha/lambda_lo + delta_lambda L/(lambda_lo lambda_hat)
```

を使う。S=57.6648142603812737 Ha、alpha=3.0730068189987043e-7 Haは全split共通。
Sは|constant|=2.29310124732、one-body L1=15.91160189453と全DF blockの絶対尺度の和。
gammaの24 C+64は24000064であり、全処理のresource capを使う。特定HRの実際の丸め経路長ではない。

|split|lambda_R [Ha]|alpha/lambda_R|lambda_lo [Ha]|ec|
|---|---:|---:|---:|---:|
|G3|0.8137790394|3.77622e-7|0.8137787321|7.55251e-7|
|G6|0.02629445765|1.16869e-5|0.02629415034|2.33741e-5|
|G9|5.473824347e-7|0.5614003344|2.400817528e-7|2.559966997|

全3行でlambda_lo>0。停止理由はlambda区間の零包含ではなく、その後のW/v予算超過。
lambda sum由来sはG3/G6/G9で約5.924e-12／1.914e-13／3.985e-18 Haと、alphaより小さい。
L/lambda_hatはほぼ1で、sを無視できる説明用極限ではx=alpha/lambda_hatに対しec≈2x/(1-x)。
G9ではx≈0.5614となり、1/lambdaだけでなく1/(1-x)の増幅も生じている。

W予算はy=ec+assembly+matvec+divisionとして2y+y²+dot/quotient+subtraction。
v予算はlaw直接項4ep、meanへ伝播するlaw項4(1+defect)ep、quotient/summation、二次項、最後のvariance算術へ展開した。

|split|W予算|主なW寄与|v予算|主なv寄与|
|---|---:|---|---:|---|
|G3|1.512373424e-6|2ec: 99.8763%|6.042595235e-6|直接law＋center-law: 99.9903%|
|G6|4.675057563e-5|2ec: 99.9948%|1.869937665e-4|直接law＋center-law: 99.9994%|
|G9|11.67336503|2ec: 43.8600%、y²: 56.1400%|27.03316700|直接law＋center-law: 75.7578%、center二次: 24.2422%|

G9の大きな二次項も、出発点はalpha由来のecである。独立に実測した不安定性ではない。
assembly線形項は全行約1.863e-9、matvec線形項は約7.390e-12。
残差B_res、full reference B_ref、採択1e-8 Haのstate品質成分はlaw_limitsの入力ではない。
残差gateだけを変えても、今回の精度停止原因は取り除けない。

## 2. 小さいlambdaと全H共通上界を切り分ける

HRはDF tail blockだけから作られ、constant/one-body/HD prefixを含まない。
そこで新しい誤差式の入力候補として、S_R=sum_(l in tail)|lambda_l| G_l²を保存NPZから求めた。
これはc_hatの相殺後L1を、相殺前の安全な入力尺度へ流用した値ではない。

|split|S_R [Ha]|S/S_R|S_R/lambda_R|
|---|---:|---:|---:|
|G3|4.147449063|13.9037|5.09653|
|G6|0.1215080442|474.576|4.62105|
|G9|2.922602190e-6|1.97306409e7|5.33923|

全HのSを固定するとalpha/lambdaが増える。一方、tailスケールもlambdaとともに縮小しており、S_R/lambdaは3行とも約4.6–5.4。
したがって「lambdaが小さいだけで不可避に同じ精度悪化が起きる」とは言えない。
共通の大きな絶対上界を微小tailへ割り当てる設計の影響が大きい、という保存データ上の切り分けである。
入力丸め・DF近似・状態誤差へこの結論を拡張しない。

## 3. 固定した感度ケース

以下の数字はW/vそのものではなく、同じ誤差式に候補scale/countを入れたscalar予算。
採択限界はともに5.960464477539063e-8。全ケースの厳密な数値と比率はmachine結果にある。

|case|G3 eW / ev|G6 eW / ev|G9 eW / ev|
|---|---|---|---|
|v4そのまま|1.512e-6 / 6.043e-6|4.675e-5 / 1.870e-4|11.673 / 27.033|
|law段のMだけ360|1.511e-6 / 6.042e-6|4.675e-5 / 1.870e-4|11.673 / 27.033|
|全H S、Cを静的706837|1.070e-6 / 4.271e-6|3.305e-5 / 1.322e-4|4.363 / 12.257|
|全H S、C=保存counter27781・law M360|4.199e-8 / 1.679e-7|1.299e-6 / 5.195e-6|0.06438 / 0.25452|
|tail S_Rだけ、C/M cap維持|1.105e-7 / 4.352e-7|1.004e-7 / 3.947e-7|1.157e-7 / 4.559e-7|
|tail S_R、C706837・law M360|7.681e-8 / 3.072e-7|6.964e-8 / 2.785e-7|8.047e-8 / 3.218e-7|
|tail S_R、C27781・law M360（未証明）|3.037e-9 / 1.208e-8|2.755e-9 / 1.095e-8|3.180e-9 / 1.266e-8|
|alpha=0の説明用極限|1.885e-9 / 6.441e-10|1.885e-9 / 6.441e-10|1.885e-9 / 6.441e-10|

Mの置換だけ、静的global Cの置換だけ、tail scaleだけ、では解決しない。
最後から2行目は局所scaleとcountの組合せの可能性を示すが、27781を新しい保証値に採択してよいという結果ではない。
これは_combineの全実行counterで、real-operation DAG／per-block／fsum長の完全な台帳ではない。
未来の実行へそのまま固定することもできない。coefficient DAGの証明とruntime resource capを分離する必要がある。
alpha=0は実装案ではなく、他の算術項だけで同じ停止が不可避かを見る反実仮想。
rho分母条件・probability・期待値・W/v correctnessが将来通ることは、どの感度ケースも保証しない。

## 4. 修正案の保証範囲・判断・計算

詳細は[選択肢一覧](../../../artifacts/track_b_m1s_gate0_precision_cause/2026-10-16/v1/improvement_options.json)。候補は未採択。

|案|保証できる／今わかる範囲|未解決の判断・証明|追加計算の必要性|
|---|---|---|---|
|law-stage count360|実際の表のsum/assembly長を使う改善候補。同じgamma仮定を保てる|pre-aggregation DAGに360を流用しない。単独では不足|保存表＋静的SOURCEで準備可。metricsには新実行|
|HRのS_Rと演算グラフ別予算|同じbinary64/gamma/BLAS/fsum条件のまま精緻化する候補|real projection、identity、消えた係数を含め、全依存経路のweighted absolute massと演算誤差を証明。正式v4の式変更はユーザー採択が必要|候補証明は保存SOURCE/inputで可能。必要なら計数instrumentation。新metricsは別実行|
|exact dyadic係数の独立照合|入力bit値targetに対するcoefficient層のexact検証が可能な設計|独立JW実装の正しさ、exactと保存supportのunion、zero/sign/identity、sumを検証。後続BLASは条件付きのまま|新しい係数検証計算が必要。今回は未実施|
|validated interval／exact W/v|全算術を包めれば数値certificateを目指せる|現行engineering区分より強い保証が必要か、算術実装の検証範囲を決める|新SOURCE・数値検証・metrics計算が必要。今回は未実施|
|高精度との一致・実測差から閾値設定|有限な経験的比較としてのみ使える|一致だけで厳密保証にできない。ceiling緩和は精度要求の変更|今回は不要。自動fallbackに使わない|

現行の条件付きモデルを保つ技術準備としては、HR raw scaleと演算グラフ別の上界を先に具体化する案がある。
ただし固定v4の式・SOURCEを変えないまま結果をPASSへ読み替えることはできない。
新契約の正式採択は行わない。追加計算許可を受けた今回の診断に必要な処理は完了しており、Gate 0再実行は不要だった。

## 5. coverage・一次資料・保全

今回証明したのは既存式の代数分解と保存値の再現。全演算のerror coverageを新たに認証したものではない。
CONDITIONAL_ENGINEERING_NOT_CERTIFIEDを維持する。
gamma_kと通常の演算モデルの出典は[Higham–Mary, §1, (1.1)–(1.2)](https://eprints.maths.manchester.ac.uk/2673/1/paper.pdf)。
同論文の確率的な誤差縮小へ勝手に切り替えない。
[NumPy matmul](https://numpy.org/doc/1.26/reference/generated/numpy.matmul.html)はBLAS依存、
[Python math.fsum](https://docs.python.org/3.11/library/math.html#math.fsum)にはrounding条件とdouble-roundingの注意がある。
build hashや今回のscalar Fraction計算で、元runのBLAS/fsum/underflow仮定を証明したことにはならない。

[監査・evidence](../../../artifacts/track_b_m1s_gate0_precision_cause/2026-10-16/v1/evidence_index.json)と
[公開receipt](../../../artifacts/track_b_m1s_gate0_precision_cause/2026-10-16/v1/publication_receipt.json)を参照。
旧source・raw結果・採択・authorization・消費済みmarker・親worktreeを保全する。
新しい精度方針の採否や研究GO/no-goを判断せず、mandatory STOP。
