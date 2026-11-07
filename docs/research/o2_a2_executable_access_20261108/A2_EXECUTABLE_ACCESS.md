# O2-A2：具体的なLCU測定アクセス

2026-11-08 JST。**`EXECUTABLE_WITH_MINIMAL_DELTA` ＋ `COST_ADVANTAGE_UNSUPPORTED`**。既存の単位操作から物理回路を指定できた。実H4のbinding・新engine誤差認証・資源削減を資格化したという意味ではない。

前回[4e610afのA2資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4e610af3c0a0b9745cff63b39b57ee1f97657cf6/docs/research/o2_a2_j1_mechanism_20261108/A2_MULTILEVEL_FEASIBILITY.md)のHadamard状態Ψ、差分e、χ、observable Ôの定義を継承し、代数的target liftを再導出しない。今回はPREP/SELECT、失敗block、測定基底を具体化した。

## 1. 利用できるアクセスと未許可のoracle

既存PRは明示的なsystem gate列とordinary controlled evolutionを持つ。`df_partial_s2.py` のrequest/builderはcontrolledの有無を指定できる。`rte.py` のeventはapplication順・符号・phase・角度を保持し、既存Hadamard builderはX/Yとbit0→+1、bit1→−1を定める。既知gate列に追加controlを付けることは物理的な標準操作であり、未知black-box unitaryのglobal phaseを制御できるという仮定とは異なる。

今回のadapterは **system-onlyの明示的fine/coarse unitary回路** を受け取り、同じwire順でcontrolled `diag(I,U)` を作る。入力にstate preparation、測定、reset、未束縛parameter、任意の未知Gateを認めない。元state ψを毎attempt利用できるという既存仮定だけを使う。逆state準備・ground reflection・coherent sampler・振幅/norm oracleは使わない。Gate列の合法なcontrol生成と1-qubit rotationのみが追加アクセスである。

実N1からこのsystem-only列を復元するscience bindingはまだ行っていない。既存uncontrolled builderの存在は、selected policyの一致を実H4で確認した証拠ではない。旧SavedEngineを改変したり、保存stateを読み出して差分を作ったりしていない。

## 2. 二level候補：既存K2からn=0へのpushforward

参照fineはN1 `744e1a09661ebba6841fe5faf456f88c41c47780`、stateはN2 `5e9bd08068738cdf1070bdc43bee8194b113ae33`。保存linear H4／1.00 Å／STO-3G／legacy DF12／L_D3／8system／全4電子sector／m0,q1,δ=.02,r1,K2／coefficient_atol=0.0、N1 actual Q・selected fullを維持する。実payloadは今回評価していない。

次は **未採択の階層候補** で、r/time/geometryを変更しない。既存eventのapplication順metadataを `(n; j₀,…,jₙ)`（最後がrotation）と書く。fineをそのまま残し、coarseを **同じselected policyのn=0 full event、label=jₙ** に写す。`order0_projection` はこのmetadata写像だけを実装し、event・law・乱数を生成しない。

N1のfactorized actual lawが `Q(n,labels)=q_n∏q_j` のままなら、jointはfineを引いてcoarseを決定するpushforward。coarseのlabel marginalはexactにq_jとなる。有限wordは元のorder/label wordを使い、coarse用の追加wordを消費しない。これは共有seedだけの主張ではなくmarginalの指定である。人工dyadic lawでのみ確認した。

actual finite meanは、保存full operationを `F_(n,labels)`、n=0を `C_j=F_(0,j)` として

\[
 M_f=\sum Q(n,labels)F_{n,labels},\quad M_c=\sum_jq_jC_j,
 \quad M_f=M_c+E_\Pi[F-C].
\]

coarse平均も未知の測定対象であり、fineと同平均とは仮定しない。ideal実数lawなら中央はそれぞれ既存のP3/B_K2、P1/B_K0に対応するが、**actual Qと保存gateで定義した平均をideal多項式で置き換えない**。新normalizerを実入力で再計算していない。一次targetは元のattenuated raw mean、共通weightは1。補正targetを扱うなら共通normalizationを双方に戻す。

n=0では両者のfull operationが同じ。n=2ではfineのproduct・Taylor phase・角度を全て残し、coarseはn=0自身の角度とphaseを使う。この差分は希少なorderでのみ非zeroになり得るが、条件付きstate差が小さいとは仮定しない。metadataで同一と分かるeventを省略するなら、baselineにも同じ省略を認める。追加levelのr/K値やqDRIFTの二次/四次moment則は採択しない。

## 3. 実際のLCU block

ζ>0を外部から明示する設計parameterとし、科学的な値は採択しない。`s=√(1+ζ²)`、`β=ζ+s` とする。元signal ancilla a、追加block b、追加herald hを使う。systemは既存順、Qiskit indexは `a=n,b=n+1,h=n+2`、初期状態は `|0_h0_b0_a⟩|ψ⟩`。

`V_f=C_a(U_f)H_a`、`V_c=C_a(U_c)H_a`、

\[
 A_c=\frac1s\begin{bmatrix}-\zeta&-1\\1&-\zeta\end{bmatrix}
     =R_y(2\operatorname{atan2}(1,-\zeta))
\]

とする。PREP_h は `|0⟩→√(ζ/β)|0⟩+√(s/β)|1⟩`。

回路の時間順は次のとおり。

1. aにHを作用。hにPREPのRYを作用。
2. `(h=0,a=1)` でfine system gate列を制御実行。
3. `(h=1,a=1)` でcoarse system gate列を制御実行。
4. h=1でbにA_cのRYを作用。
5. hにPREP†を作用。h=0を成功flagとする。

full unitary Wのh=0 input/output blockは

\[
 K/\beta=(\zeta I_b\otimes V_f+sA_c\otimes V_c)/\beta.
\]

b=0入力への成功非正規化状態は

\[
 \frac1\beta\left[|0_b\rangle\zeta(\Psi_f-\Psi_c)
                    +|1_b\rangle\Psi_c\right]=\chi/\beta.
\]

失敗blockは `√(ζs)/β (A_c⊗V_c−I_b⊗V_f)`。失敗状態を捨てても、そのattemptの実行費は捨てない。`p_success=S/β²`、`S=1+ζ²||Ψ_f−Ψ_c||²` であり、理想構成では `1/β²≤p_success≤1`。S/pのpath別の値をalgorithmへ渡す必要はない。

この構成は既知の2-unitary LCU。根拠は[Childs–Wiebe, §II Lemma 2/Fig.1](https://arxiv.org/pdf/1202.5822)のancilla rotation、controlled unitary、逆rotation、flag測定である。係数とb操作を上記のように選んだ適用で、新しいdilation原理ではない。amplitude amplification・repeat-until-successは追加しない。U_f†/U_c†もこの経路には不要。

## 4. 測定基底とbitからの再構成

前回の `Ô_b=A_ζ⊗O_b` の2×2部分は

\[
 A_\zeta=\begin{bmatrix}\zeta^{-2}&\zeta^{-1}\\\zeta^{-1}&0\end{bmatrix}
 =R_y(\theta)\operatorname{diag}(\lambda_+,\lambda_-)R_y(-\theta),
\]

\[
 \theta=\operatorname{atan2}(2\zeta,1),\qquad
 \lambda_\pm=(\zeta^{-2}\pm\sqrt{\zeta^{-4}+4\zeta^{-2}})/2.
\]

bにRY(−θ)、aにX軸ならH、Y軸ならSdg→Hを作用し、b/aをZ測定する。成功時の測定値は `z=λ_(b_bit)(−1)^(a_bit)`。hも測定し、**全attemptの** outcomeを `T=β²z`（h=0）、`T=0`（h=1）とする。`E[T|path]=f_f−f_c`。同じcommon exact real weight wならwTとする。異なるlevel weightを無条件に共通weightへ潰す接口は提供しない。

adapterはflagを最後に測るdeferred measurementであり、失敗にもreadout gateが作用する。中間flag測定＋成功時readoutだけという別経路の費用を、この回路の費用に流用しない。classical register順は `(h,b,a)`。success数で分母を置き換えない。

## 5. 保証と未資格範囲

上式は理想unitary/real-angle操作の既知恒等式。実装はbinary64のgate parameterとdecoderで、人工correctnessの許容差を検査した。理想式への一様な数値誤差は **UNKNOWN**。N1 binding、追加control、RY、readout eigenvalue/scale、joint measurement gridの資格も未取得。旧engine認証はこの追加gate群・測定器には適用しない。

従って今回の完成は「明示操作だけでsuccess/failureと測定を定義できる」という技術的到達点。資源改善の判定と今後の扱いは[A2_MEASUREMENT_AND_COST.md](A2_MEASUREMENT_AND_COST.md)、人工結果は[A2_IMPLEMENTATION_AND_ARTIFICIAL_TESTS.md](A2_IMPLEMENTATION_AND_ARTIFICIAL_TESTS.md)。
