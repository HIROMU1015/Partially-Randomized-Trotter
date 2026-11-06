# Track D B5：idealからactualへのRPE統計保証接続・静的調査

2026-11-06 JST。**B4-2のideal条件付き十分shot数をactual実装へ移すためには、implemented whole-path law、各軸内のshot列の統計条件、全使用pathに適用する測定実装誤差の三つが不足している。** 保存fixtureのcorrectness PASSとseed分離だけでは、この不足を埋められない。本書は既存SOURCEの静的調査と技術設計であり、新しいlaw誤差、実装誤差、N、成功率、総費用を算定していない。未知項はnull／未資格を維持する。

## 1. 対象・作業範囲・証拠の読み方

固定対象はlinear H4、隣接1.00 Å、STO-3G、Nα=Nβ=2、number-CDF8、元frame順LD2、ordered partial S2/s1/r32/K0、h=δ=.1。初期状態は元Hamiltonian Hの厳密基底状態を仮定し、状態準備費を除外する。

| B4-2から維持する条件 | 値 |
|---|---|
| energy精度／failure上限 | .00159 Ha／.01 |
| βtotal／βstat | π/6／π/12 |
| αaxis | .01/26 |
| schedule／prior | m0..12、q=2^m=1..4096／[−10,10] Ha |
| 元H packet SHA256 | `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db` |
| CDF endpoint SHA256 | `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a` |
| 保存law JSON SHA256 | `c347d7f2ec2ed40adb6f2490960dbebf693862cddcce96418c1027c48a68d6a5`。このhashはlaw記述のidentityであり、joint-law正確性の証明ではない |

[B4-2保存結果][B42JSON]の引用は、X/Y合計のHoeffding十分shot数14,738、energy幅.0012783173232380342 Ha、failure配分和.01、全13 round条件付きPASSである。B1数値operator norm＋engineering guard、ideal finite-RTE IID／exact native、元H厳密groundと数値的prior資格を前提とする。actual成功保証へ読み替えない。[B4-2報告][B42]

専用branchは `track-d-b5-actual-rpe-static-survey-20261106`、基点はB4-2公開 `07d18bfe1d5145d6f8de026c0d994d5cb1c783c8`。共通／repository AGENTS.md、PROJECT_MAP.md、研究概要、正式共通指示2026-10-25を確認した。SOURCE・コード・JSON・契約・STOPの読取りだけを行い、分子配列の展開・再解析、importによる研究処理、test実行、sampling、回路生成／compile／postcheck、新N算定、B2／B4-2再実行、詳細科学レビューは行っていない。実装SOURCE、保存入力・raw・結果・grant／ledger／marker／STOPは変更しない。

本書では、**保存証拠**は既存commitの記録、**SOURCE確認**は静的に読める実装、**条件付き恒等式／標準不等式**は明示した数学的前提で成り立つ関係、**提案**は未実施の取得方法、**未資格**は根拠が不足する事項を意味する。提案を新しい定理・検証済みの性質として扱わない。

## 2. 取得・確認した固定資料と適用範囲

以下は全てローカルGit objectから取得した固定版。C3/C5のSOURCEをB4-2 branchへ取り込んだり実行したりせず、指定commitの資料として読んだ。

| 固定commit・資料 | 確認した内容と適用範囲 |
|---|---|
| B4-2 `07d18bfe1d5145d6f8de026c0d994d5cb1c783c8`：[報告][B42]／[PLAN][B42PLAN]／[SOURCE][B42SOURCE]／[結果][B42JSON]／[STOP][B42STOP] | 固定target、13-round ideal評価、actual null、数値的alias資格。実行local SOURCE `8e472b5001278e51b9919a8302d0a4bd3fcf9209`と公開SOURCE `6da5afba79895c6b3c092f41e332d444dedc8ed3`は同一treeの別commit。既存の対応記録を保持 |
| B4-1 `31cd67c91abda3c9c5d4b643c434c040b8ceaeca`：[報告][B41]／[adapter][ADAPTER] | 元H wrapper、raw誤差幾何、Hoeffding／horizon／decoder接続、unknown拒否。人工52＋既存21 PASSは過去記録で、今回再実行していない |
| B3 `89a13fe6841c1017f0feba724ed71dfc79f9e81b`：[静的報告][B3] | 誤差単位、lawと依存の区別、Referenceの共通N・会計接続条件。B4-1/B4-2以前の未実装・未評価表記は履歴として読む |
| B1/C1 `d3913866e6ade4ab97ac02645fed91b2d5fcfa75`：[報告][B1]／[SOURCE][B1SOURCE]／[STOP][B1STOP] | 実行SOURCE `8c6eeb02e89f6cf14870789719493e75482f62f8`。同sector元H operator輸送と保存00→01のq2 raw correctness。新しい全law boundではない |
| Stage A `3dc9a61ee6defa3ffa8cda8601f87253cdd7f8b0`：[報告][A]／[SOURCE][ASOURCE]／[STOP][ASTOP] | 実行SOURCE `2dd012dc77dcc877d5918c7d212819eb79b97557`。q1、保存32 path、event16後cut、32 precompile／128 wrapper postcheckの有限cohort |
| C3 `cc554cf6daea08df6c20ea8e1e92405259ce49b5`：[完了報告][C3]／[補完契約][C3PLAN]／[SOURCE][C3SOURCE]／[STOP][C3STOP] | q4=00..03/cut2、q8=00..07/cut4。構築・8 compile SOURCE `52b3cab50d5c5edcbd7e3249637168e9978be535`、補完SOURCE `88f0496349c056d0736f08487889f942c1aa4d37`。全8 postcheckと最終paired X/Y・χ PASS |
| C5 `9a6920b8355643701ab7436a4fd3b34862e9e3bd`：[報告][C5]／[契約][C5PLAN]／[SOURCE][C5SOURCE]／[STOP][C5STOP]／[共通入口][C5CODE] | 実行SOURCE `976c898fbdb79477c1db3178b5d32df1fe5aeb5c`。固定q2=00→01/cut1、4 P0 wrapper・paired X/Y・vacuum χ PASS。保存列用入口と有限fixture価格資格 |
| CDF8/r32生成SOURCE `4df8eca29c59ffd55e2dd898f05da16bb246ddbe`：[sampling／seed][R32]／[native生成経路][NATIVE] | 保存32 pathの生成元。master/cohort/candidate/SOURCEをseedへ渡すprice-only cohortで、RPE量子shot列ではない |
| C側再利用SOURCE `ba0e2bf2d1d6f0dc78616d3392f3ff5b6ba7d5b7`：[rte][RTE]／[connection][CONNECTION]／[native bridge][NATIVEBRIDGE] | 保存eventのsampling、分布・符号規約、corrected signed-event回路入口。B4-1のcheckoutの同名fileを無条件に代用しない |
| carrier `3649b75af191916daf44abb91e14d34c43b83e8f`：[保存law][LAW] | 216 atoms、λ_R、identity、ideal lawラベル、`implemented_finite_bit_law_error=null`。carrierは科学実行SOURCEと区別 |

**C3の注意：** 同commit内の旧 `RESULT.md` はrun01の技術失敗STOPである。今回の完了根拠は `COMPLETION_RESULT.md` と `run03_completion/result.json`／STOP。run02の中断と、別許可による補完resume=1を消さず、初回失敗だけを最終状態と誤読しない。

SOURCE版照合では、`rte.py`のblobがC側ba0e／CDF生成4df8で `03124795ad26076a858724fb2336ae01b2dec123`、B4-1で `53601b7c524653f5e1966b3955b9f51373e54002` と異なることを確認した。差は `event_unitary` の任意arithmetic引数経路で、今回読んだ分布・sampler部分は同じ。`c1_stage3_connection.py`は3版ともblob `08997ddfd9fbc176502f1d453d10e8471a4878c9`。これは該当fileの版照合であり、closureの互換性や新規数値再現を認定するものではない。

## 3. B5-1：ideal lawとimplemented sampler

### 3.1 分布と生成経路

[rte.py][RTE]は非負の有限paired-even Taylor重みと、signを吸収したHermitian involution分解を使う。`finite_rte_distribution`が次数重みとB_Kを作り、`_make_event`が次数確率×各component確率の積をevent metadataへ記録する。componentの選択確率は係数の絶対値に比例し、符号を確率の符号にしない。

今回K0では次数は0のみ、各eventは一つのsigned involutionのrotation。τ=λ_R h/r、θ=atan(τ)、B_0=sqrt(1+τ²)。event単位ではrotation componentを選び、1 h-stepに32 event、roundにはq個のh-stepがある。ideal product lawならその全順序列のlawは、固定component lawの32q回の積となる。一般Kの次数／product因子の議論をK0の対象へ勝手に拡張しない。

静的に追跡した保存cohortの生成経路は次のとおり。

`c_near_practical_r32.sampling/execute` → `c_update064_r32_downstream.calibrate` → `c_pr_integrated_performance.native` → `c_pr_aware_df_stage_b.independent_occurrence` → `c1_stage3_connection.trajectory_seed/sample_occurrence` → `rte.iter_sample_rte_events`。

[生成元][NATIVE]はsampleごとに32 eventを生成し、events／sampling policy／SOURCE／JU等を保存する。master=2026110304、cohort=`near_practical_PR_r32_fixed32_v1`、axis=`NEAR_PRACTICAL_PR_PRICE_ONLY`、round_index=−1、shot=sample ID、occurrence=0である。保存欄 `coherent_occurrences_independent=true` は生成者の宣言であり、ここではq>1のRPE shot列を検査した記録ではない。

[trajectory_seed][CONNECTION]は `(master,arm,candidate,round_index,axis,shot,occurrence)` のreprをSHA256へ渡し先頭8 bytesをseedにする。masterには元SOURCE digestとcohortも含む。[r32 SOURCE][R32]自身がseed分離を「hash-separated finite-bit seed streams; not proof of ideal IID law or cost transfer」と資格付けしている。keyの分離・有限collision検査はprovenance／再現性に有用だが、独立乱数や全joint-law距離を証明しない。

### 3.2 三段階のlawを分ける

| 対象 | SOURCEから確認できること | 未取得の証拠 |
|---|---|---|
| ideal finite-RTE p | 固定signed係数、次数重み、component product、K0/r32、独立eventを仮定した解析law | 実装PRNG列がこの積分布を生成する保証 |
| 浮動小数点の確率配列 | `make_tail`はFraction比からfloat確率へ変換し、samplerは`math.fsum`で再正規化する。`RTEFiniteDistribution`は重み／確率の整合性を許容差で確認する | 宣言したideal確率と実際のbinary64配列／累積thresholdの距離上界。整合性許容差はlaw error boundではない |
| 実装samplerの出力law | `Generator(PCG64(seed))`。各eventで次数を`choice`し、続いてcomponentを`choice(size=order+1)`。K0でも次数選択callを通る | 使用NumPy版のchoice・累積加算・有限uniform格子・境界規則・seed分布／PRNG状態を含む出力lawと依存の保証 |
| 保存32本／固定fixture | 保存eventsは同一bytesで再利用。Stage Aは32本、C1/C2/C5は00→01、C3は00..03/00..07の固定列 | fresh whole-path lawの網羅、全13-round quantum shot実行、全law bound |

`make_tail`はtiny nonzero係数を自動thresholdせず、floatへのloss／overflowを拒否する。それでもFraction比→float、再正規化、choiceの分布差が0と認定されたわけではない。K0の次数分布が形式上退化していても、保存streamとの対応では乱数call順序を保持する必要がある。B5ではsamplerもcall順序も変更していない。

### 3.3 独立性の四つの単位

1. **event内／h-step内：** K0では一eventのcomponent一個、32 eventは同じPCG64 streamから順に生成される。APIの逐次drawは確認できるが、数学的独立bitの証拠ではない。
2. **q occurrence間：** intended APIにはoccurrence keyがある。B1のideal平均operatorの積には対象product lawが必要。固定sampleの昇順連結や同じ短pathのq乗は、そのproduct lawの検証ではない。
3. **同じround・axis内の量子shot間：** 既存Hoeffding helperには独立なbounded±1試行が必要。各shotでfreshなsystem状態、reference真空、ancilla、trajectoryを対象規約どおりに用意する。既存price-only cohortはこの実行記録ではない。
4. **round／axis／方式間：** 全26軸のunion bound自体は独立を要しない。X/Yや方式間でpaired random pathを共有する設計でも、各軸内の統計条件と各方式の周辺測定lawを別に確認する。paired価格標本を量子shot独立性の証明にしない。

固定trajectoryをN回測れば、そのpathに条件付けたBorn測定の独立性は仮定できても、平均先はlaw平均でなくそのpathの信号になる。保存32本の巡回／bootstrapもideal whole-path expectationをそのまま実現しない。seed非重複、histogram一致、有限テストPASSだけでこの差を埋めない。

### 3.4 law errorの定義と最小取得案

round mのideal whole-path lawをP_m、implemented whole-path lawをQ_m、ideal path信号をz_m(ω)、|z_m(ω)|≤1と定める。mean-bias用の量は

\[
e^{raw}_{law,m}\ \ge\
\left|\mathbb E_{Q_m}z_m-\mathbb E_{P_m}z_m\right|.
\]

無次元の**raw complex-signal**単位であり、TVやoperator norm、Ha、radとは別の量。標準的なTV定義 `TV(P,Q)=sup_S|P(S)-Q(S)|=(1/2)||P-Q||_1` を採用すれば、上記差≤2 TV(P_m,Q_m)。これは既知のbounded期待値不等式を使う設計候補で、今回TV値を評価していない。

最小案はK0の固定component集合に限定し、理想係数比→浮動確率→choiceの実効categorical lawの対応を解析すること。使用NumPy／bit-generatorのSOURCEとseedの確率モデルを固定する。独立uniform bitsを仮定したcategorical rounding boundと、PCG64の固定seed列の保証は別の資格で記録する。固定masterなら古典生成列自体は決定的であり、「seedは非重複だから各drawはIID」としない。

もし各conditional drawの対象categorical lawからのTVを履歴一様にδ_jで抑えられる、または独立drawのproduct lawが別に正当化できるなら、whole-path TVをΣ_jδ_jで保守的に接続する候補がある。K0ではevent数32qがその適用単位。one-eventの周辺誤差だけからこの積／conditional条件を推定しない。q／shot列の依存を含むモデルが必要なら、そのjoint lawを別に扱う。

**mean biasのboundはconcentrationの保証ではない。** 全量子測定transcriptのTV≤ηを使う候補では、成功事象のfailureはideal failure＋ηで移送する。B4-2のαaxis和は既に.01なので、正のηを無断で足す余地として扱えない。別の統計モデル、既存Nでのより強い条件の検査、または配分変更が必要かをユーザー／GPTへ戻す。今回はその選定・算定をしない。

## 4. B5-2：Referenceの測定分布保存条件

### 4.1 Whole ordered pathと真空ray

時間順に適用するU_0,…,U_{q−1}に対してV=U_{q−1}…U_0=A Bとし、Bをcut以前、Aをcut以後とする。必要なのは**各halfが同じreference rayを保存**すること：

\[
A|v\rangle=e^{ia}|v\rangle,\quad
B|v\rangle=e^{ib}|v\rangle,\quad \chi=a+b\pmod{2\pi}.
\]

Vだけのray保存からA/B個別の保存を推定しない。既存[Reference gadget][GADGET]は二つのCSWAP cascadeの間でsystemへactual `A.inverse()`、referenceへBを適用し、最後にancillaへP(+χ)を置く。rawのAを正しく逆順・逆角度・逆global phaseにするactual inverseが必要で、「負の時間なら同じ並びでよい」としない。

既存恒等式の前提の下で、H後の両branchは

\[
u_0=e^{ib}A^\dagger|\psi\rangle\otimes|v\rangle/\sqrt2,
\quad
u_1=e^{i(\chi-a)}B|\psi\rangle\otimes|v\rangle/\sqrt2,
\]

したがって `2<u0|u1>=<ψ|AB|ψ>`。χの符号・scalar／identity一回・A†がこの一致に必要である。この条件付き関係は[B3][B3]と既存[correctness harness][CHECKS]が扱うもので、B5が全pathを新しく検証した結果ではない。

### 4.2 signed event、scalar、normalization

固定number-preserving Gaussian basis操作は真空rayを保存する構造候補であり、Z/ZZ対角primitiveは真空が固有rayになる。SOURCEの最適化されたbasis遷移・global phase補正がこの構造と宣言位相を保つことまで確認対象にする。任意の数値Gaussian合成・compilerへ無条件に拡張しない。

[path_phase][PHASE]はK0について `scalar_phase + Σ(arg(event.phase)−event.unsigned_rotation_angle)` を使う。`rotation_angle=atan(τ/(n+1))`と、係数signを反映する`unsigned_rotation_angle`を区別する。[emitter][EVENTBUILD]はsigned productのπ位相とrotation signを別に扱う。K0にproductはないが、正負atomを同じ符号にしてよいわけではない。

各h-stepのscalar phaseは `−(constant+extracted_identity)h`。q全体とA/Bの位相へ各一回だけ会計し、whole-path χを合成する。保存1 pathのχを別pathへ転用せず、観測値へphase fitしない。B_K補正／attenuationはunitaryや±1 outcomeへ掛けない。raw平均のattenuation A_mはB4-2と同じ定義で扱う。

再利用上の注意として、旧 `c1_stage3_connection.partial_kernel` にはsigned-angle検査の旧問題がある。[修正済みnative bridge][NATIVEBRIDGE]はunsignedの保存angleを検査し、`c_pr_aware_df_stage_b.build_kernel`はこのbridgeを呼ぶ。将来も新しい独自emitterを作らず、この実際のcorrected経路と`build_event_sequence`を候補にする。旧同名入口を無条件に再採用しない。

### 4.3 complex coherenceからX/Yの二値分布へ

既存shellはH→core→YのみSdg→H→ancilla測定、bit0→+1／bit1→−1。z=<ψ|V|ψ>なら

\[
\mu_X=\Re z,\quad\mu_Y=\Im z,\qquad
\Pr(Y_b=y\mid\omega,m)=(1+y\mu_b(\omega,m))/2.
\]

従って、同じpathに対するcomplex coherence／X/Y observable一致は、明示したreference／ancilla初期化の下で各axisの条件付き二値分布一致につながる。同じP_mで平均し、同じfresh shot joint lawを用いたときにのみ、全測定列の分布も共通化できる。1 shot周辺分布が同じだけでは、依存する全shot列の分布まで同じとは限らない。

短いReference gadgetをq回反復するとsystem-referenceの途中状態・entanglementを含む別channelになる。既存C1/C3/C5は**whole ordered pathに一つのgadget**であり、その成果を短gadget反復の証拠にしない。途中reset、postselection、weightsも追加していない。

### 4.4 保存correctnessのcoverage

| 証拠 | 確認済みの有限範囲 | その証拠だけでは未資格 |
|---|---|---|
| Stage A | 保存32個のq1 path、cut event16、全256 system入力列、32 raw／128 postcheck、paired X/Y、χ欠落negative control | 全216-atom列の組合せ、fresh全law、q>1、統計的IID |
| C1／C2 | 保存00→01、q2/cut1、一回gadget、raw complex coherence／AB／actual adjoint／χ、C2の4 P0 wrapper | 他pair/cut、全law、長q、実装誤差の一様上界 |
| C3完了 | 保存00..03のq4/cut2、00..07のq8/cut4、各22 raw check＋paired、全8 P0 postcheck／最終paired・vacuum χ | 未使用path、q16..4096、全13-round合成／合成精度保証 |
| C5 | 新共通保存列入口でq2=00→01/cut1、4 P0 wrapper、paired X/Yとvacuum χ。強いsymmetricの合法なstep融合とReferenceのcut損失を保持 | fresh法則generator、任意requestの検証済み価格、全law／long-q資格 |

検査はreference／ancillaを0として全256 system入力に対応する誘導observableを扱うため、保存own-DF ground一点の検査より広い。ただし、**system入力のcoverageとsampling pathのcoverageは別**。component abs1e−10、observable数値Frobenius／signal2e−10、rtol0、global phase fitなしは有限fixtureの判定規約であり、全law boundでも丸め誤差の厳密包囲でもない。既存最大観測残差・PASS閾値をそのままe_implへ入れない。

構造的に整理できる候補は、number-preserving primitiveごとのvacuum位相、A/Bの任意ordered列でのray保存、exact adjoint、合法な融合・global scalar規則、Hadamard測定恒等式である。数値実装で追加確認すべき部分は、Gaussian分解・relative transitionの位相、finite precisionのχ、full-wrapper compiler、実際に用いる合成とreference leakage、全使用domainへのboundである。B5では構造証明の完成・全domain数値検証を宣言しない。

## 5. B5-3：raw誤差と統計保証への接続

### 5.1 誤差の分解・単位・重複防止

方式gのimplemented平均信号と基準 `A_m exp(−it_m E0)` を、次の順序で比較する。

\[
\mathbb E_{Q_m}z_{g,impl}
\longrightarrow\mathbb E_{Q_m}z_{ideal}
\longrightarrow\mathbb E_{P_m}z_{ideal}
\longrightarrow A_m e^{-it_mE_0}.
\]

この分解に対するboundを整合して得た場合、既存adapterへ渡す候補は

\[
d_{g,m}=A_m\epsilon_m^{B1}
+e^{raw}_{impl,g,m}+e^{raw}_{law,m}.
\]

e_implはQ_m平均下の実装差を抑える量、e_lawは同じideal path信号のP_m対Q_m平均差を抑える量である。P_m下でしか得ていない実装誤差をQ_mへ無確認で移さない。一様path boundならこの違いを避けられるが、取得済みとはしない。

| 誤差項 | 意味と単位 | 接続時の条件 |
|---|---|---|
| B1のPF/RTE/DF | 同sectorのnormalization-corrected operator norm、無次元。DF norm自体はHaで、Duhamelの時間を経てoperator誤差へ輸送済み | B1固定13行を使う。rawへ一回だけA_mを掛ける。DF reserve／Weylをenergyへ再加算しない |
| 確率丸め・finite-bit law | P_mとQ_mの差によるraw平均bias | TVなどを定義どおりrawへ換算する。B1の有限Taylor truncationと同じ項ではない |
| shot間依存 | 濃度不等式／joint-law成功事象の前提 | e_lawへ足すだけでIIDを認定しない。別のfailure移送なら同じ誤差を二重計上しない |
| native角度・compile／synthesis | 対象pathのunitary／isometry／observable／signal差 | 対象input subspace、norm、global／relative phase、適用domain、確率モデルを明示してrawへ換算 |
| Reference χ／actual inverse／leakage | 同じreference初期化での測定差 | full-wrapper observable boundがこれらを包含するなら、その内訳を追加で重ねない |
| state・reference初期化・noise | 元H厳密ground、reference真空、ancillaを外れるときの測定差 | 今回はoracle／model前提。準備費除外は準備誤差0の実証ではない。実機noise保証へ拡張しない |

native exactモデルの誤差0と、実装SOURCEがbinary64値を使うことも区別する。normalization A_m自体の数値評価差、τ／θやscalarの表現差を評価する場合は、基準のどちらへ帰属させたかを固定する。

### 5.2 既存関数へ戻す条件

[B4-1 adapter][ADAPTER]は、binding、元H state／operator単位、law lane、alias、IIDとprovenance、追加raw誤差を要求する。未確認のe_impl/e_lawはNone、未確認のIIDは未資格のままにし、`UNRESOLVED_SIGNAL_OR_SAMPLING_PREMISES`／shots=Noneを維持する。boolやprovenance文字列を埋めるだけで外部premiseが正しくなるわけではない。

条件が資格付けされた後だけ、既存[raw geometry][ACCURACY]が

\[
\rho_{g,m}=A_m-d_{g,m}>0,\quad
\beta_{det,g,m}=\arcsin(d_{g,m}/A_m),\quad
\beta_{det,g,m}+\beta_{stat}\le\beta_{total},
\]

\[
\epsilon_{coord,g,m}=\rho_{g,m}\sin(\beta_{stat})/\sqrt2,\quad
N_{g,m,b}=\left\lceil2\log(2/\alpha_{axis})/\epsilon_{coord,g,m}^2\right\rceil
\]

を使う。新しいshot式を実装する必要はない。追加raw誤差へA_mを再度掛けない。正半径・phase不通過は十分条件の不成立で、物理的不可能性とは区別する。

B4-2のphase slackは保存された**余裕の診断**であり、B5がlaw／χ／compileへ新しい数値budgetを割り当てる根拠ではない。追加誤差がphase内でも半径が縮めばcoordinate許容幅が狭くなり、必要Nが変わり得る。phase条件だけで旧N利用を認定しない。

### 5.3 全q直接compileを避ける誤差接続候補

以下は標準的なnorm不等式に基づく**未実施の設計候補**であり、分子入力への数値適用・新bound評価はしていない。

- **局所合成の和：** 対応するideal／implemented unitary primitiveの一様operator誤差η_jが既知なら、同じordered productの差はtelescopingによりΣ_jη_j以下。qと各pathのprimitive出現回数を保守的に会計し、全13-roundを式で覆う候補となる。構造保存を含む局所証拠が先に必要であり、最適化後full-wrapperの誤差を未検証kernelの単純反復へ置き換えない。
- **isometry→observable：** reference初期化を含む入力isometry Wに対し、正規化を保つWとW̃の差≤ηなら、||M_b||≤1の誘導observable差は≤2η。axis別observable boundをκ_X,κ_Yとして得た場合、raw complex差は≤sqrt(κ_X²+κ_Y²)。max-entry residualをoperator normへそのまま同一視しない。既存数値Frobeniusは検査の候補だが、丸め包囲と全law domainが別途必要。
- **χとadjoint：** 位相誤差δχだけなら信号回転の大きさは≤|exp(iδχ)−1|≤|δχ|。actual adjointのunitary差、reference leakageを独立項として扱うなら、同じisometry／observableの分解で重複を避ける。既存full-wrapper boundが全て包含する場合は追加しない。
- **有限local dictionary：** 固定K0のfinite atom／basis／合法なtransition／fusion規則に局所証拠を持たせ、任意ordered列の合成へ接続する候補。全216-atom長列の直接列挙を最初から要求しない。一方、Qiskit opt1がfull-wrapperで新たな書換えを行う場合、局所辞書だけではcoverageにならない。compiler変換の正当化または別の適用限定が必要。

bit生成／PRNG依存はunitary telescopingでは解消しない。logical native回路とfault-tolerant RZ合成、routing／noiseは別のmodel。小q postcheckは接続実装を検査する補助であり、一様boundの代用ではない。

## 6. B5-4：Reference／symmetricのN共通化

### 6.1 共通化の必要条件と現在地

| 条件 | 現在の確認状況 |
|---|---|
| 同じV・whole ordered path・cutの意味 | C1/C3/C5の固定fixtureで確認。C5の同request入口は取り違え防止に再利用可能。fresh全law生成器ではない |
| 同じP_m/Q_m、normalization、scalar・signed event | 固定metadataと保存pathで確認。implemented全joint-law／finite-bit誤差は未資格 |
| 同じX/Y条件付き分布 | raw恒等式の前提と固定fixtureの数値observableで確認。全使用path・compile／合成後の一様誤差は未資格 |
| 各軸内のfresh shot条件／concentration | idealモデルで仮定。actual shot列の根拠は未取得 |
| 追加実装誤差と共通targetへの適合 | 全13-roundのe_impl/e_lawはnull。B4-2のphase余裕を割当済みとはしない |
| energy／decoder／alias／failure | 既存符号φ=−hE、連続dyadic、principal prior、βtotal<π/3、26軸unionの条件を維持。B4-2のprior資格は保存数値groundに条件付きで、厳密spectral enclosureではない |

idealの条件付き測定lawとfresh joint lawが両方式で同じなら、同じN列・decoderに対する成功分布を共通に扱える。一方、actual誤差の大きさが違っても同じ整数Nとなる可能性はあるが、検査前に認定できない。「同じN」は方式間で同じround・axis別列という意味で、全roundへ同じ整数を使う意味ではない。

### 6.2 将来の最も保守的な共通評価

同じモデル・law・source domainで両方式の有効raw誤差が得られた場合、`d_common,m=max(d_Reference,m,d_symmetric,m)` を使い、既存adapterで共通の最悪半径／phase／Nを評価する候補が最小である。lawや統計条件が異なるなら、数値をmaxにするだけで同じ資格にはならない。

**B4-2の既存Nをそのまま使えるか**は、新Nの評価とは区別して検査できる。既存の各軸整数N0_mに対し、独立±1試行を前提とする同じHoeffding十分条件から

\[
\sqrt{2\log(2/\alpha_{axis})/N0_m}
\le \rho_{common,m}\sin(\beta_{stat})/\sqrt2
\]

とphase・alias・全round条件を満たすかを見る案がある。これは旧Nのceil余裕まで含めた適否検査の仕様で、B5では数値代入していない。B4-2のphase slackだけではこの判定を代用できない。

片方式だけが旧N条件を外れる、方式間でlaw／noise／合成誤差が異なる、共通最悪評価が研究比較に不要なほど過大な場合は、同じtargetとfailure配分で方式別Nを評価する候補とする。その科学実行は別許可が必要。異なる統計方式・failure配分・targetを導入する変更は事前にユーザー／GPTへ戻す。方式間共通化のために既存精度／failureを事後緩和しない。

## 7. 再利用するコード・テスト・pipeline

以下は**再利用候補の静的確認**。B5でtest／runnerを起動した意味ではない。C3/C5の別commit部品は固定リンクから選び、必要な新SOURCE revisionで限定的に接続する。

| 部品 | 最小の再利用用途／限界 |
|---|---|
| [rte.py][RTE]：`finite_rte_distribution`、`iter_sample_rte_events`、`_make_event`、`RTEEvent` | 分布記述・既存sampler・signed metadata。samplerを書き直さず実効lawの調査対象にする。新samplingは別許可 |
| [prototype.make_tail][TAIL]、[connection.trajectory_seed/sample_occurrence][CONNECTION] | 固定係数比、float化、seed key、occurrence生成。seedの分離と数学的独立を混同しない |
| [native bridge][NATIVEBRIDGE]／[event emitter][EVENTBUILD] | 修正済みsigned-angle検査、元順・full-frame・phase保持。旧partial_kernelの検査へ逆戻りしない |
| [Stage A adapter][PHASE]：`saved_record`、`prepare_common`、`path_phase` | 保存law／endpoint確認、K0位相、head／tail共有。保存32本を全law generatorにしない |
| [C5共通入口][C5CODE]／[C3 body][C3CODE] | 同requestのwhole-path／method／axis、A/Bとactual inverse、strong fusion。C5価格recordは固定q2資格に限定 |
| [Reference harness][CHECKS]、`run_track_d_h4_reference.verify_one` | AB、vacuum、χ、actual adjoint、complex coherence、X/Y、leakage、postcheck。限定科学callであり再実行許可が必要 |
| [元H adapter][ADAPTER]、[共有accuracy][ACCURACY] | `OriginalHRoundEvidence.from_b1_row`、`assess_round/assess_energy/decode_energy`。raw誤差・qualified前提を入力し旧shot式を再利用。joint-law／martingale証明器ではない |
| [Hoeffding helper][HOEFFDING]、既存horizon／dyadic decoder | 同targetのN・energy／failure接続。sampling・成功率simulationを不要に追加しない |
| [rte tests][RTETEST] | `test_finite_taylor_distribution_is_normalized`、`test_all_finite_event_probabilities_sum_to_one`、`test_enumerated_event_mean_matches_dense_paired_taylor_reference`、`test_fixed_seed_reproduces_event_sequence`、signed product／heterogeneous normalization。toy整合性・再現性であり、finite-bit／IIDの証明ではない |
| [元H人工tests][ADAPTERTEST] | unknown／state／単位／DF一回／ideal-actual／failure／horizon／decoder。外部boundの科学的正当性は別 |
| [C5 tests][C5TEST]／[C3 tests][C3TEST]／Stage A人工tests | order／cut／SOURCE／χ／scalar／negative atoms／fusion／価格資格の接続回帰。test内の回路構築やsamplingまで無断実行しない |

既存 `rpe_hadamard_failure_validation` のfresh trajectory／Born samplingや旧短round Monte Carloは、モデル検査の過去経路である。今回は起動しない。将来も全law保証を得る最初の手段として大量Monte Carloを要求しない。

## 8. B5-5：優先順位と依存関係を持つ最小追加検証案

以下は未実施。ユーザー／GPTが対象claimと実行範囲を採択し、Codexが許可された部分をまとめて実施する。追加grant発行・承認証明システムは不要。既存の固定契約・STOPはそのまま保護する。

| 優先・依存 | 最小作業／成果 | 担当と必要な許可 | GO／STOP |
|---|---|---|---|
| 0：起点 | actualの意味を固定：論理native＋finite-bit samplerか、RZ合成／routing／実機noiseまで含むか。PRNGをどの確率モデルで扱うか、B1数値pilotを前提とするclaimかを明示 | ユーザー／GPTが判断。詳細科学レビューの開始を自動要求・実施しない | 不明なmodelを誤差0で埋めない。fixed target／state／lawを変える場合は具体的承認へ戻す |
| 1：0後 | 固定K0 component／確率配列／choice SOURCEの対応を解析し、one-draw law rounding boundの取得方法を確定。seed・stream／shot／occurrenceのjoint条件を別項目で定義 | Codex。静的追加設計を越える実装・人工検証はその許可を得る。分子確率から新law誤差を数値評価する場合はSOURCE・入力・精度・対象13-roundを明記した科学実行許可が必要 | 独立uniformモデルの条件付き結果と固定PCG64列の保証を分離。単なる周辺一致ならactual全統計は未資格 |
| 2：0後、1と独立 | 固定primitive・frame・合法なfusionからvacuum-ray／χ／actual inverse／X/Yの構造対応を明文化。全pathへ移せる範囲とcompilerで失う範囲を特定 | Codexの技術検証。新コード／toy circuit構築は明示scopeで許可。分子回路・postcheckは別科学許可 | proof obligationが残れば「構造候補」で止める。小fixture PASSを全law証明にしない |
| 3：2後 | 対象Gaussian／対角／phase／合成／compilerの局所operatorまたはrestricted-isometry誤差を取得し、誤差の和→observable→rawへ接続。回路を全q直接compileせず長qを覆える式を優先 | Codex。新数値誤差計算・合成計算・分子compile／postcheckは使用SOURCE・primitive集合・精度・call範囲・出力・STOPを限定して別許可 | global compile書換えやuncovered transitionがあれば該当domain未資格。固定nativeモデルからrouting等へ自動拡張しない |
| 4：1＋2＋3後 | one-path law→whole-path law、各軸内のshot統計、全13-roundのe_impl/e_law・alias・horizonを一つの入力仕様へ接続 | Codex。数値代入を含むなら保存B1／B4-2＋新誤差入力の科学再解析許可が必要。targetとαを固定 | mean biasだけでshot依存を解消しない。全transcriptのfailure移送が必要で固定αに収まる根拠がなければSTOPし、判断へ戻す |
| 5：4後 | 共通最悪raw評価で旧N列の十分性を検査。必要な場合だけ同じhelperで共通または方式別N・energy／failureを評価 | Codex。旧N適否の新数値解析・N再算定を科学実行範囲へ明記。sampling／Born shots／ground solve／B2は含めない最小案 | radius／phase／failure／alias／horizon／coverage／数値範囲不通過なら不成立を保存。事後配分変更や無断retryなし |
| 6：5後、別段階 | 資格付き方式別NとStage Cの同law・同scope期待価格を将来接続 | ユーザー／GPTが価格資格と接続scopeを判断。Codexは全RPE総費用の明示許可後 | C6aの有限paired価格だけでは全law・long-q・統計保証を認定しない。今回の実行対象外 |

1〜3は必要な技術作業を一つの限定タスクにまとめられる。細かな関数ごとの中間承認・handoffを新設しない。重い数値検証を後日許可する場合は、既存cache・共通context・vector化・独立task並列を適合する範囲で使い、実空き資源に応じて配分する。恣意的なCPU/RAM/worker上限は設けず、共有環境・他jobは変更しない。

この順序でlaw／構造／誤差接続を先に解決すれば、最大qの17-qubit full-wrapper直接compileや大量Monte Carloを最初から要求せずに済む可能性がある。局所boundが過大で予算を通せない場合、その十分条件の限界を記録して次の最小検証を判断する。B2はlaw依存や全path compile保証の不足を自動的に解消しない。

## 9. Stage C6aとの将来の情報受渡し

C6aは別担当のq2／q4 fresh whole-path paired価格pilotとしてユーザーから通知されている。**B5ではC6aのSOURCE／新sample／結果を取得・読了・検証済みとして扱っていない。** C6aのSOURCE・入力・sample・結果・STOPの内容を調査対象にせず、branch・作業領域・processにも変更・操作を加えていない。以下は公開完了後、利用許可された情報を受け取る項目である。

| 受け取る情報 | B5側での用途と限界 |
|---|---|
| 公開commit、実行SOURCE、契約、STOP、input／endpoint／law binding | B5固定対象と同じ条件か識別する。C6aの実行許可やSTOPをB側へ移さない |
| round/q、sample単位、ordered全event、cut、seed key／stream・occurrence割当、生成版 | fresh whole-pathの実装経路を特定する補助。seed分離やfinite sampleだけでIID／全joint-lawを証明しない |
| 同requestの両方式X/Y、scalar／identity、A/B／actual adjoint、χ、normalization | 同じpath信号・測定規約か確認する資料。固定path以外のuniform errorを自動付与しない |
| raw／post／paired correctness、失敗・欠測・選択規則 | 技術接続とcoverageの把握。最大残差やPASS閾値を全law boundへ転用しない |
| full-wrapper P0／合成／最適化／topology、価格・paired差・不確かさとdomain | 将来の同scope価格入力候補。law平均・CI・long-q coverageの資格を個別確認。価格標本数をNへ掛けない |
| 取得できない項目と明示null | 未取得のまま引き継ぐ。q2/q4からq16..4096や全13-roundへ外挿しない |

内部CPU/RAM観測、内部絶対path、資格情報、非公開rawは受渡し・公開文書へ転記しない。C6aの新sampling／compileをB5の未許可科学実行や全law保証の代わりに使わない。情報が後日公開されても、B5のmandatory STOPを自動解除しない。

## 10. 未解決事項・STOP・次担当

| 未解決事項 | 現在の値／状態 |
|---|---|
| implemented finite-bit whole-path law error | null／未資格 |
| actual shot／occurrence joint-law・concentrationの根拠 | 未資格。ideal IIDはモデル仮定 |
| 全path・全13-roundのcompile／synthesis／Reference誤差 | null／未資格 |
| B4-2の既存Nを両方式actualへ共通使用 | 未認定。新N評価なし |
| actual RPE成功確率／保証 | 未評価／未資格。B1 normとpriorも数値pilotの資格を保持 |
| 全RPE総費用 | 未評価。価格との乗算なし |

後続実行では、SOURCE／input／sector／law／signed phaseの不一致、DF二重計上、誤差単位不一致、actual未知誤差の0埋め、同じNを支えるshot統計の不足、radius／phase／failure／alias／horizonの不成立、未検証domainへの外挿、範囲外のsampling／compile／再解析の必要性をSTOP条件とする。不通過時はpartialと理由を保存し、閾値緩和・目標変更・別fixtureへの置換・無断retryで進めない。旧STOPを上書きしない。

本B5はMarkdown報告書と入口索引2件だけを専用branchでcommit／公開し、remote SHA・同一tree・3ファイルbytesを照合する。科学実行・実装変更は0。公開commitがGitHub接続によりlocal commitと異なる場合は、最終チャットで両SHAとtree／bytes一致を報告する。追加の承認証明や管理台帳は作らない。

**公開・照合後mandatory STOP。次担当：ユーザー／GPT。** §8の0でclaimと確率モデルを選び、優先1〜3の対象SOURCE・入力・実装／数値call範囲・精度・STOPを明示して次の作業を許可するか判断する。**Codexはその明示許可後**、既存sampler／corrected bridge／Reference harness／元H adapterを再利用して必要な最小接続と検証を実施する。新sampling、実装誤差数値計算、追加Reference回路検証、旧N適否の数値評価／N再算定、全費用、詳細科学レビューへ自動進行しない。

[B42]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/07d18bfe1d5145d6f8de026c0d994d5cb1c783c8/docs/research/track_d_b4_2_rpe_shots_20261106/RESULT.md
[B42PLAN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/07d18bfe1d5145d6f8de026c0d994d5cb1c783c8/docs/research/track_d_b4_2_rpe_shots_20261106/PLAN.md
[B42JSON]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/07d18bfe1d5145d6f8de026c0d994d5cb1c783c8/artifacts/track_d_b4_2_rpe_shots/2026-11-06/run01/result.json
[B42SOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/07d18bfe1d5145d6f8de026c0d994d5cb1c783c8/artifacts/track_d_b4_2_rpe_shots/2026-11-06/run01/SOURCE.json
[B42STOP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/07d18bfe1d5145d6f8de026c0d994d5cb1c783c8/artifacts/track_d_b4_2_rpe_shots/2026-11-06/run01/STOP.json
[B41]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/31cd67c91abda3c9c5d4b643c434c040b8ceaeca/docs/research/track_d_b4_1_rpe_statistics_implementation_20261106/REPORT.md
[B3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/89a13fe6841c1017f0feba724ed71dfc79f9e81b/docs/research/track_d_b3_rpe_statistics_preparation_20261106/REPORT.md
[B1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d3913866e6ade4ab97ac02645fed91b2d5fcfa75/docs/research/track_d_b1_c1_validation_20261106/RESULT.md
[B1SOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d3913866e6ade4ab97ac02645fed91b2d5fcfa75/artifacts/track_d_b1_c1_validation/2026-11-06/run01/SOURCE.json
[B1STOP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d3913866e6ade4ab97ac02645fed91b2d5fcfa75/artifacts/track_d_b1_c1_validation/2026-11-06/run01/STOP.json
[A]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3dc9a61ee6defa3ffa8cda8601f87253cdd7f8b0/docs/research/track_d_h4_paired32_reference_20261106/RESULT.md
[ASOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3dc9a61ee6defa3ffa8cda8601f87253cdd7f8b0/artifacts/track_d_h4_paired32_reference/2026-11-06/run01/SOURCE.json
[ASTOP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3dc9a61ee6defa3ffa8cda8601f87253cdd7f8b0/artifacts/track_d_h4_paired32_reference/2026-11-06/run01/STOP.json
[C3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cc554cf6daea08df6c20ea8e1e92405259ce49b5/docs/research/track_d_c3_q4_q8_20261106/COMPLETION_RESULT.md
[C3PLAN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cc554cf6daea08df6c20ea8e1e92405259ce49b5/docs/research/track_d_c3_q4_q8_20261106/COMPLETION_PLAN.md
[C3SOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cc554cf6daea08df6c20ea8e1e92405259ce49b5/artifacts/track_d_c3_q4_q8/2026-11-06/run03_completion/SOURCE.json
[C3STOP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cc554cf6daea08df6c20ea8e1e92405259ce49b5/artifacts/track_d_c3_q4_q8/2026-11-06/run03_completion/STOP.json
[C5]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a6920b8355643701ab7436a4fd3b34862e9e3bd/docs/research/track_d_c5_q2_20261106/RESULT.md
[C5PLAN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a6920b8355643701ab7436a4fd3b34862e9e3bd/docs/research/track_d_c5_q2_20261106/PLAN.md
[C5SOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a6920b8355643701ab7436a4fd3b34862e9e3bd/artifacts/track_d_c5_q2/2026-11-06/run01/SOURCE.json
[C5STOP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a6920b8355643701ab7436a4fd3b34862e9e3bd/artifacts/track_d_c5_q2/2026-11-06/run01/STOP.json
[C5CODE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a6920b8355643701ab7436a4fd3b34862e9e3bd/src/trottertracks/track_d_whole_path_price.py
[C3CODE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cc554cf6daea08df6c20ea8e1e92405259ce49b5/src/trottertracks/track_d_c3_whole_path.py
[R32]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4df8eca29c59ffd55e2dd898f05da16bb246ddbe/src/trottertracks/c_near_practical_r32.py
[NATIVE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4df8eca29c59ffd55e2dd898f05da16bb246ddbe/src/trottertracks/c_pr_integrated_performance.py
[RTE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ba0e2bf2d1d6f0dc78616d3392f3ff5b6ba7d5b7/src/trotterlib/rte.py
[CONNECTION]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ba0e2bf2d1d6f0dc78616d3392f3ff5b6ba7d5b7/src/trottertracks/c1_stage3_connection.py
[NATIVEBRIDGE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ba0e2bf2d1d6f0dc78616d3392f3ff5b6ba7d5b7/src/trottertracks/c1_stage3_native_bridge.py
[TAIL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ba0e2bf2d1d6f0dc78616d3392f3ff5b6ba7d5b7/src/trottertracks/pr_dictionary_validation/prototype.py
[EVENTBUILD]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ba0e2bf2d1d6f0dc78616d3392f3ff5b6ba7d5b7/src/trottertracks/pr_dictionary_validation/circuits.py
[LAW]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3649b75af191916daf44abb91e14d34c43b83e8f/artifacts/research_c_near_practical_r32/2026-11-03/run01/PR/A/law.json
[PHASE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3dc9a61ee6defa3ffa8cda8601f87253cdd7f8b0/src/trottertracks/track_d_h4_paired_reference.py
[GADGET]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3dc9a61ee6defa3ffa8cda8601f87253cdd7f8b0/src/trottertracks/track_d_h4_reference.py
[CHECKS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3dc9a61ee6defa3ffa8cda8601f87253cdd7f8b0/tests/track_d_h4_reference_checks.py
[ADAPTER]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/31cd67c91abda3c9c5d4b643c434c040b8ceaeca/src/trottertracks/track_d_original_h_accuracy.py
[ACCURACY]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/31cd67c91abda3c9c5d4b643c434c040b8ceaeca/src/trottertracks/c1_stage3_accuracy.py
[HOEFFDING]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/31cd67c91abda3c9c5d4b643c434c040b8ceaeca/src/trotterlib/rpe_resource_accounting.py
[RTETEST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ba0e2bf2d1d6f0dc78616d3392f3ff5b6ba7d5b7/tests/test_rte.py
[ADAPTERTEST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/31cd67c91abda3c9c5d4b643c434c040b8ceaeca/tests/check_track_d_original_h_accuracy.py
[C5TEST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a6920b8355643701ab7436a4fd3b34862e9e3bd/tests/check_track_d_c5.py
[C3TEST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cc554cf6daea08df6c20ea8e1e92405259ce49b5/tests/check_track_d_c3.py
