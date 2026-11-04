# 全18round categorical平均とControlled Hadamard回路の接続

2026-11-04 JST / LOCAL、技術調査・人工検証のみ。
**任意trajectoryに必要な演算子恒等式と、長qへ一様に転送する誤差上界の構成法を整理した。
全18roundのproduction回路への数値接続保証は未完成である。**
実H4の新しいsignal・trajectory・平均・上界・回路は評価していない。

条件はlinear H4／隣接1.00 Å／STO-3G／保存legacy DF12／Partial L_D=3／単一δ=.02、
8 system qubit・4電子sector70、B保存physical stateのnormalized ray、
extract_identity_phase、coefficient_atol=1e-12、K=2、m=0..17・q=2^m。
保存N/α/β/ε、component/basis/sign/support policy、signal側τを維持する。
RPE_tolerance_certified=false、scientific_status=UNDETERMINED、
U_virtual_IID_upper=null、U_exec_upper=null。以下の分類は正式statusを変更しない。

## 1. 既存証拠とSOURCEの固定

| 証拠 | 固定commit | 今回の扱い |
|---|---|---|
| 既存SOURCE対応・人工回路 | bdffe73a6a8af0988e62ff90c9dd344880e8b8b5 | [対応表](../rpe_controlled_hadamard_preparation_20261104/CIRCUIT_SOURCE_MAP.md)・テスト再利用 |
| Cの科学SOURCE | 7279f700a58921209b851c8d9c363d38ff2f1d98 | production構成を静的確認 |
| C run02結果 | 82b2df6af80eb923302d35e369c4572ed9fa0a9f | [12 probe/24軸](../rpe_controlled_hadamard_local_probes_v2_execution_20261104/REPORT.md)をそのまま引用 |
| 全18round categorical SOURCE | f4905fff592351d33ba07b927892fb9532c99711 | law/mean定義を静的確認 |
| 全18round結果 | d2e7a7db683e116243eae061c0e4c87865054e9e | [公開報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d2e7a7db683e116243eae061c0e4c87865054e9e/docs/research/rpe_18round_ideal_enclosure_execution_20261104/REPORT.md)・保存literalのみ参照 |
| m17再利用結果 | 34840ce9bf81d985b32764d9bc9d77645ed52e59 | 全18結果のraw groupsにはm17がないため、元の96-bit結果を参照 |

[FIXED_BINDINGS.json](FIXED_BINDINGS.json)にcommit/path/blob/SHA256、共通入力、
18 tupleのselector、6rのτ/θ hex対応を記録した。[static_binding.py](static_binding.py)は
git blobの読出し・hash・保存literal比較だけを行い、law・信号・包含を再計算しない。
productionのsrcファイルはC SOURCE、categorical SOURCE、本branchのbaseでbyte一致した。
全36軸の保存判定はSUFFICIENT_CONDITION_CONFIRMED。今回新たにその不等式を評価していない。

Cで既に確認されたのは、この固定H4条件における6r×q_probe=1/2・固定2成分の12 trajectoryだけ。
最大axis残差上端約4.03e-15は局所照合値である。全wordの上界ではなく、q倍して保証には使わない。
候補Bの6 request/12軸も同様に限定的な証拠である。
Dのα・decoder・E0、費用・状態準備・noiseは今回の対象外。

## 2. 固定trajectoryの演算子とSOURCE

列vectorへ右から作用させる。BはSOURCEに記録されたbasis回路の理想unitary、
D_j(t)はそのbasis内の対角発展とする。blockは
H_j(t)=B_j D_j(t) B_j†。append順はB_j†、D_j、B_jである。
半stepのblock数をb、constant/identityのscalarをs=exp(i(a_0+a_I))とすると、

\[
 F=s H_{b-1}(\delta/2)\cdots H_0(\delta/2),\qquad
 R=H_0(\delta/2)\cdots H_{b-1}(\delta/2).
\]

Rは正時間の逆順でありF†ではない。scalarはF側で一度だけ入る。
trajectoryのstep kは
\(U_k=R E_{k,r-1}\cdots E_{k,0}F\)、
\(V_T=U_{q-1}\cdots U_0\)、
\(z_T=\psi^\dagger V_T\psi/(\psi^\dagger\psi)\)。

| 構成 | 理論とSOURCEの対応 |
|---|---|
| deterministic F/R | [df_partial_s2.py](../../../src/trotterlib/df_partial_s2.py)の_append_basis/_append_block/_build_parts。one-body/squared primitivesは[ops.py](../../../src/trotterlib/df_trotter/ops.py)。append順と行列積順を反転して読む |
| event | [rte.py](../../../src/trotterlib/rte.py)の_make_event/event_unitary。保存indicesはrotation先頭だがapplication_sequenceはproduct群→rotation |
| signed component | \(P_\ell=s_\ell B_\ell Z_{S_\ell}B_\ell^\dagger\)、\(s_\ell=\pm1\)。偶数order nのeventは \((-1)^{n/2}e^{-i\theta_nP_{\rm rot}}P_n\cdots P_1\)。非可換productを並べ替えない |
| rotation | [df_rte_qiskit.py](../../../src/trotterlib/df_rte_qiskit.py)の_append_central_application。変数名unsigned_rotation_angleには実際にはsign×θが入る。RZ/RZZ(2 sign θ)はexp(-i sign θ Z_S) |
| product sign / event phase | controlled productは各supportへCZ。負成分1個につきπ64を加算し、eventの−1もπ64。tail全体で最後にancilla PhaseGateへ集約。rotation符号をproduct位相として重ねない |
| identity/scalar | extract_identity_phaseでtailからidentityを分離し、constantとextractedを別のstep PhaseGateに入れる。event signとHamiltonianのconstant/extractedは別項 |
| q/rと境界 | [df_partial_s2_repeated.py](../../../src/trotterlib/df_partial_s2_repeated.py)。r eventを各stepに配置しq stepを順に合成。同じstep_time/preparationが必要。実builderはstep_resultsも構築しO(qr)へ展開するため、巨大qの直接使用は効率的な検証方法ではない |
| support policy | [sequence_policy.py](../../../src/trotterlib/research_direction_sequence_policy.py)のregister_support_restricted_bases/make_run_threshold_basis_plan。保存policyはsupport_for_source_basis_run_le_1、training fingerprintはrun02/source_inputsに固定。neighborに依存してbasisを選ぶ |
| wrapper | [rpe_hadamard_interrogation.py](../../../src/trotterlib/rpe_hadamard_interrogation.py)、benchmark builder。H→controlled evolution→sineならSdg→H→測定。bit0=+1、bit1=−1 |

理想的なcontrolled写像を
\(C(U)=|0\rangle\langle0|\otimes I+|1\rangle\langle1|\otimes U\)
とすると、\(C(A)C(B)=C(AB)\)。
unconditional basis sandwichも
\((I\otimes B)C(D)(I\otimes B^\dagger)=C(BDB^\dagger)\)。
したがってcontrol=0はI、control=1はV_Tになる。
basisの逆回路は逆順gate.inverse()で、有限parameterの符号反転が正確なら理想inverse。
隣接basisの相殺は同じ登録basis/keyのB†Bに限り、異basisや非可換eventは交換しない。

scalar exp(iφ)をsystem全体のglobal phaseとして捨てるとC(U)は変わる。
controlled経路のPhaseGate(φ)はancilla-relativeであり保持が必要。
uncontrolledのQuantumCircuit.global_phaseにはbinary64加算・modulo処理があり、
その回路全体を後からcontrolする経路とordinary-controlled builderとのbitwise同値性はここでは証明しない。

正規化した入力についてcontrolled後は
\((|0\rangle\psi+|1\rangle V_T\psi)/\sqrt2\)。
X平均はRe z_T、Y平均は+Im z_T。Sdg→H→ZはY測定になるためsine=+Im。
この証明は状態準備の方法を仮定せず、指定normalized rayに対する理想測定を対象とする。

Qiskitはqubit jを整数bit j（little endian）に対応させる。
全回路ではancilla8が最上位なのでcontrol branchは前半/後半。
gate.control(1)の局所行列ではcontrolが局所bit0なので偶数/奇数indexになる。
sectorはpopcount=4の保存順であり、OpenFermionへのbit順変換は既存permutationを使う。
run02でこの保存state/sectorの対応を確認済み。任意の状態準備回路への保証へ広げない。

### Qiskit controlと抽象gate／定義／数値評価の区別

固定runtimeはQiskit1.3.0。[RZ](https://quantum.cloud.ibm.com/docs/en/api/qiskit/1.3/qiskit.circuit.library.RZGate)と
[CRZ](https://quantum.cloud.ibm.com/docs/en/api/qiskit/1.3/qiskit.circuit.library.CRZGate)の理想行列、
実際にimportしたrz.py/rzz.py/xx_plus_yy.py/add_control.pyを照合した。関連4ファイルのhashはbindingsに記録。

numeric RZ.control(1)はCRZ。定義はtarget RZ(t/2)、CX、RZ(−t/2)、CX。
RZZ.control(1)はCCX→CRZ(t)→CCX、global_phase=0。
add_controlのRZ分岐はcontinueするため、RZのU1定義のglobal_phaseを二重計上しない。
この有限templateは人工parameterで確認した。
定義に現れるt/2のbinary64生成まで同一operatorとするには
Q(fl(t/2))=Q(t)/2が必要。subnormalでは成立しない例を人工テストで検出した。
eventの2θのexact doublingはcategorical SOURCEの既存guardが対象とする。
実H4の全deterministic parameterとsupport gate classについての有限auditは未実施。
抽象理想gate、definitionへの展開、Statevectorの浮動小数点評価、transpile後の近似は別の保証対象である。

## 3. Boundary再結合の任意qへの上界

以下は有限parameterで定義された理想unitaryのoperator normに対する証明であり、
実H4 parameterを代入した数値上界は今回計算していない。
Q(x)は生成済みbinary64値xをそのままexact rationalへ写す操作とする。

### 3.1 scalar

rawでは各stepにa=fl(−δc0)、b=fl(−δcI)の2 gateがある。
optimizedではSOURCEの評価順で
g=fl(fl(−δq) fl(c0+cI))の1 gateを生成する。
Pythonのint→float変換も含めて実際に生成したgを使う。

\[
 d_s=\left|Q(g)-q(Q(a)+Q(b))\right|,\qquad
 \|C(e^{ig}U)-C(e^{iq(a+b)}U)\|\le\min(2,d_s).
\]

理由は|exp(ix)−exp(iy)|≤min(2,|x−y|)。
相殺・subnormal・qのint変換誤差を含む生成済みparameter差をexactに取るため、
相対丸め誤差モデルやgeneric γ_nの仮定は要らない。
任意の正整数qでparameterが有限なら適用できる。overflow/nonfiniteならUNRESOLVED/STOP。
必要な処理は定数個のparameter生成と整数・有理数演算で、q個のgateは作らない。
s=exp(i(a+b))を先にbinary64で丸めた値をq倍する方法ではない。

### 3.2 最初のblockの融合

隣接するreverse block0半stepと次のforward block0半stepだけを融合する。
同じbasisについて対角primitiveを
\(D=e^{i\phi}\prod_j e^{-ia_jZ_j/2}\prod_{jk}e^{-ib_{jk}Z_jZ_k/2}\)
と書く。full/halfの実際の生成parameterから

\[
 d_b=|\phi_f-2\phi_h|
 +\tfrac12\sum_j|a_{f,j}-2a_{h,j}|
 +\tfrac12\sum_{jk}|b_{f,jk}-2b_{h,jk}|
\]

をexact rationalで計算すれば、\(\|D_f-D_h^2\|\le\min(2,d_b)\)。
SOURCEのzero除去はatol=0のexact zeroであり、欠けた項は0として比較できる。
自動的にd_b=0とはしない。人工subnormalで生成されたfullと2halfが異なる例を確認した。
basisが同じexact unitaryならconjugationはnormを変えない。

q−1個の境界をunitary積のtelescopingで置換すると、

\[
 \Delta_{\rm boundary}(q)\le\min(2,d_s+(q-1)d_b).
\]

他のblockやevent間の可換性は不要。basis相殺に数値matrix inverseを使わないこと、
raw/referenceのhalf primitive parameterが一致することが前提である。
[connection_adapter.py](connection_adapter.py)のscalar_gap/block_gap/boundary_boundがこの構成を実装する。
過去の観測残差をq倍する証明ではない。

### 3.3 既存π64上界の正確な範囲

既存mean_adapter.builder_phase_bound(q,r)はK2/extracted identityのtailごとに
高々3r回の非負π64加算を扱う。
π64と数学的πの差、および加算γ_(3r)をq stepへunitary telescopingで転送する。
全18roundの保存I_cat/U_catにはこの項が既に含まれる。

**この関数はboundaryのd_s/d_b、support basisの近似、Qiskit lowering、
uncontrolled global_phaseのmodulo、Statevector丸め、compiled/noisy誤差を含まない。**
既存関数・結果は変更していない。新しいtransferへπ64項をもう一度加えない。

## 4. support basisの有限な追加証拠

support completionは浮動小数点orbital matrixの指定列をcopyし、残りを直交補完して再合成する。
保存preserved_columns_max_abs_residual=0やbasis plan内の固定0.0だけでは、
合成後の理想gate列の完全一致は証明できない。run02にはselected basis hash/countはあるが、
全selected gate parameter列は保存されていない。

ただしnumber-preserving Gaussian回路Bは、vacuum phaseを除けば一粒子unitary Gの第二量子化である。
CARより、column g_iに対して
\(BZ_iB^\dagger=I-2a^\dagger(g_i)a(g_i)\)。
したがってsupport Sの列だけが一致すればBZ_S B†も一致する。
未使用列の一致は不要。||a(u)||=||u||、unit columnについて

\[
 \|n(u)-n(v)\|\le2\|u-v\|,\quad
 \|BZ_SB^\dagger-B'Z_SB'^\dagger\|
 \le\Delta_S=\min(2,4\sum_{i\in S}\|g_i-g_i'\|).
\]

ZZは2個のZの積に対するtelescopingを使う。signを付けても上界は同じ。
rotationは同じθなら差が|sinθ|Δ_S≤Δ_S、productはΔ_S。
K2のeventはrotation1＋product高々2なので、任意wordについて

\[
 \Delta_{\rm support}(q,r)\le\min(2,3qr\max_\ell\Delta_{S_\ell}).
\]

これは各componentに対する保証付き一様差から導くoperator boundで、
12 probeの残差拡大ではない。neighbor依存のsupport/full選択でも各出現に同じ一様差を適用できる。
fullを選ぶ出現は追加差0。

適用にはgate列がRZと**JW順で隣接する**XXPlusYYだけであることが必要。
非隣接の通常qubit XXPlusYYをGaussianと扱わない。
保存full11 basisのgate名/wire/finite parameter metadataはこの条件を満たした。
selected support列についてはまだ確認していない。classのruntime照合も追加audit対象。

one_particleは既存mean_adapter.gate_matrixとball_adapterのMatrixBallを再利用する。
RZのvacuum-relative一粒子作用diag(...,exp(it),...)だけを補足し、
XXPlusYYの一粒子blockをwire順に埋め、時系列で左乗算する。
二つのcertified column差には両MatrixBallのglobal radiusも加算し、丸め幅を捨てない。
8×8行列で済み、70×70/256×256のH4 evolutionやq個の回路を要しない。
CARは全Fock空間で成立するためsector70にも適用できる。
この新しい経路は人工系で検証した段階であり、H4のΔ_Sは未評価。

## 5. Categorical平均への接続条件

保存SOURCEのcomponent lawは、metadata probabilityそのものではなく、
fsum正規化→NumPy cumsum→CDF末尾による正規化→searchsorted(side=right)の
uniform53格子のmass p_lである。CDF境界C_iに対する格子countは
ceil(2^53 C_i)−ceil(2^53 C_(i−1))。CDF=1は最終境界。
order lawも保存binary64 CDFから同様に定める。
今回CDFやmassを作り直していない。

signed involution平均A=Σ_l p_l P_l、
order mass w_n、保存θ_n=atan64(signal τ/(n+1))をexact parameterとすると、
rotation drawとn個のproduct drawがそれぞれ独立で同じcomponent lawの場合、

\[
 M=\sum_{n=0,2}w_n(-1)^{n/2}
      (\cos\theta_n I-i\sin\theta_n A)A^n.
\]

有限和と行列積の分配で導出できる。個々のP_lは可換でなくてもよい。
さらに各r位置・各q stepのdrawが独立で同じlawなら、

\[
 \mathbb E_T[V_T]=(RM^rF)^q,\qquad
 \mathbb E_T[z_T]=\psi^\dagger(RM^rF)^q\psi/(\psi^\dagger\psi).
\]

追加のB除算・attenuation除算はない。ψはTと独立な固定rayである。
support policyがword依存なら実装そのものの平均を同じMへ因数分解するとは限らない。
その場合はまずfull ideal referenceへwordごとにtransferし、
|E z_impl−E z_ref|≤sup_T |z_impl−z_ref|を使う。
この順序ならpolicyがneighborを参照していてもよい。

| 対象 | 言えること |
|---|---|
| 固定T | z_Tの理想wrapper平均。lawによる平均とは別 |
| 位置ごとの独立uniform53 | 上記因数分解。保存CDF/order/component/θが一致することが必要 |
| 同じeventを全位置で再利用 | 一般にE[E^k]≠(E[E])^k。人工非可換例で差を確認 |
| 固定seed PCG64 | iter_sample_rte_eventsはPCG64、複数step helperはrandom.Randomからstep seedを生成。固定streamの数学的IID性や、全seedを平均したlawの同一性は未証明 |
| Float64仮想測定 | state-action/normalize/guard/clip/Bernoulli等の別bridgeが必要。本作業では再実行しない |
| ideal quantum shot | 指定ray・ideal gates・新しい独立draw・shot独立性を仮定すれば条件付きmeanは接続可能。実装全体の成功確率は別 |
| compiled/noisy回路 | compiler error/noise/状態準備が未評価。接続結果を転用しない |

6rのC run02 parameters.jsonと全18 SOURCEのθ hexが一致した。
r32のlawは保存m17由来であり、元結果を再計算していない。
model側τはsignal側τと近くても置き換えない。

| r | 保存m | q |
|---:|---|---|
| 1 | 0..7 | 1..128の2冪 |
| 2 | 8,9 | 256,512 |
| 4 | 10 | 1024 |
| 8 | 11,12 | 2048,4096 |
| 16 | 13,14 | 8192,16384 |
| 32 | 15..17 | 32768,65536,131072 |

## 6. 現時点の分類と判定

| 分類 | 確認できた範囲 | 条件・残り |
|---|---|---|
| ALGEBRAICALLY_ESTABLISHED | 任意q/r/wordのcontrolled積・F/R/event順序・Re/+Im、独立lawの因数分解 | exact unitary/inverse、相対位相、同一law/independenceが前提。数学的導出でありformal proof checkerの認定ではない |
| ALGEBRAICALLY_ESTABLISHED | parameter差によるboundary一様上界、Gaussian/CARによるsupport一様上界 | 実H4の有限parameter・descriptorをまだ代入していない |
| SOURCE_CONSISTENT | productionの構成順・符号・phase・basis cancellation、6rのτ/θ literal、18 tupleと既存判定の対応 | SRC byte一致と静的構成の確認。全runtime gate descriptorの完全auditではない |
| NUMERICALLY_TESTED | 既存H4の12 local probe/24軸、今回の人工test | 固定probe以外のH4 trajectory/長q/分布平均は未評価 |
| UNRESOLVED | 全18のproduction ideal controlled回路と保存I_catの保証付き数値接続 | H4のd_s/d_b、全support column差、raw half primitive一致、local lowering prerequisiteが不足 |
| UNRESOLVED | PCG64 IID、Float64仮想測定、compiled/noisy・実測定全体、RPE/energy成功保証 | 今回の範囲外。旧status/両U nullを維持 |

将来必要な追加transferを
Δ_extra≤min(2,Δ_boundary+3qr max Δ_S+Δ_lowering)
とする。raw half primitiveが保存exportと一致しなければ別のparameter差証明が必要であり、今回は自動で進めない。
Δ_lowering=0を使えるのは全local gateの構造・parameter前提が証明できた場合だけ。
欠けた項はNone/UNRESOLVEDで、0にはしない。

保存I_catをΔ_extraだけ拡張し、保存U_cat_upper+Δ_extraと保存shot_half_width上端が
保存εより厳密に小さいかを将来確認すれば、指定ideal circuit/IIDモデルの局所十分条件を転送できる。
I_catに含まれるπ64を再加算しない。
これだけでU_virtual_IID_upper/U_exec_upperを埋めたり全体RPEを認定したりしない。

## 7. 人工検証と今回の実施範囲

既存test_circuit_semantics.pyの30件を再利用した。
追加26件はscalar exact rational/Acb、相殺、subnormal、primitive phase/rotation変異、
production artificial非可換q=3/5・r=1/3・K=2/4、local RZ/RZZ control template、
3/4 modeのGaussian/CAR（JW符号・逆wire順・Z/ZZ）、未選択列、非隣接gate拒否、
independent6位置の全列挙とevent再利用の反例を対象とした。
既存mean enclosure/local probeの人工58件もPASS。計114件PASS。
科学的な新H4数値上界を示すテストではない。
Float matrix比較の許容差はtest用で、certificateのoperator上界には加えていない。

実行command（既存の専用venvを使用、install/共有環境変更なし）：

~~~sh
/tmp/prt-rpe-m17-enclosure-venv-20261103/bin/python -m pytest -q \
  docs/research/rpe_categorical_circuit_connection_20261104/test_connection_adapter.py \
  docs/research/rpe_controlled_hadamard_preparation_20261104/test_circuit_semantics.py
# 56 passed in 2.93s
/tmp/prt-rpe-m17-enclosure-venv-20261103/bin/python -m pytest -q \
  docs/research/rpe_m17_rigorous_mean_enclosure_preparation_20261103/test_mean_enclosure.py \
  docs/research/rpe_controlled_hadamard_runner_v2_20261104/test_local_probes.py
# 58 passed in 5.64s
/tmp/prt-rpe-m17-enclosure-venv-20261103/bin/python \
  docs/research/rpe_categorical_circuit_connection_20261104/static_binding.py
# Static byte/literal join saved; H4 scientific evaluations=0
~~~

既存builder/event/wrapper/ball/categorical testを再利用した。
新規helperが必要だったのは、旧builder_phase_boundがboundary再結合を対象にせず、
既存sector basis evaluatorにはvacuum-relative一粒子certificateがなかったためである。
production SOURCE・旧テスト・raw・grant・ledger・marker・STOPは変更していない。
別branchの全18包含結果はgit showで参照し、mergeせず、進行中作業に触れていない。
正式共通運用指示2026-10-25の§8.1、§8.4、§10を確認し、承認証明機構は追加していない。

次段階の固定案は[SCIENTIFIC_VALIDATION_PLAN.md](SCIENTIFIC_VALIDATION_PLAN.md)。
公開照合後はmandatory STOP。今回の許可で次の科学実行へ進まない。
