# M1-S：RPE適用条件と総回路費用の成立条件 — Decision Packet

2026-10-17 JST。**`SYMBOLIC_CONDITIONS_PREPARED_NOT_ADOPTED / NO_SCIENTIFIC_EXECUTION / MANDATORY_STOP`**。

本書はユーザーが依頼したCodexの資料調査・数式と費用会計の整理である。同じGPT詳細科学レビューを再実施せず、研究継続を正式採択しない。新しいshot数、cost、biasの数値評価・ランキングは作成していない。

## 1. 判断に必要なこと

M1-Sのvisibility改善がRPEで有利になるには、少なくとも、**同じ推定taskで位相誤差予算を満たすこと、測定法と平均信号が対応すること、必要shot数を1 shot費用と掛けた総額で強い対照を上回ること**が必要である。visibilityの大小だけでは決まらない。

現時点では、有限stepのG3/G6 visibility positiveとphase trade-offの証拠はある。一方、RPEの目的位相・精度・confidence・時間horizon・実装費用・状態準備費用が未固定で、総資源の成立判定はできない。既存資料で解決できる技術上の対応は本書で整理し、残る科学的選択は[次期契約候補][candidate]とmachine候補の未決fieldへ明示する。

|区分|本packetの到達点|意味|
|---|---|---|
|レビュー原本|ローカル原本と公開commitのbytes/SHA256を確認|読解対象を固定。レビューの再開催ではない|
|位相・shot・費用の式|変数・仮定・十分条件・proxyを明示|数値代入・forecastなし|
|既存SOURCEとの対応|流用可能な会計式と流用できない仮定を特定|adapter実装・実行なし|
|次の契約|未採択候補、未決入力をnullで保持|実行契約の完成やRUN_READYを意味しない|
|科学的な進行|ユーザーによる方針・条件の採否待ち|新科学実施にはさらに別承認が必要|

## 2. 固定資料と引継ぎ

原本：`/home/abe/Project/Partially Randomized Trotter/m1s_finite_step_pilot_results_gpt_scientific_review_20261017.md`。
37,756 bytes、SHA256 `a67ce8921a698fee844acffc3e3bfe4b60ffbd1a9a9a8a8bea3b2feda77d7c18`。
公開レビューcommit：`e5ed0db6b8f50ed02ee9bc17cc6d0e2d4ed0898b`、[原本][review]。
結果固定commit：`025c5a90888b02b73d2815fbb10d6bd1d2823602`、[完了報告][result]。
SOURCE S2：`2eac35ff33f100fa96e296b737752addde5af5de`、契約C2：`5afd57ed3feb900b8f97317c38cc62f52959981b`、authorization A2：`3dfd4470f8755937be1818c24580e8cbb76d3171`。

raw resultのSHA256は`ff6c5192d8cca769cd61a1478cea01671d058a85a8f6a4cdcdc9baa672a56fe0`。今回raw/CSVはbytes identityの対象としてのみ読み、科学的な数値再集計をしない。[入力・資料identity][identity]に固定commitとpathを列挙する。

適用指示は2026-10-16版共通研究指示。旧v4のTECHNICAL_STOP/null、v5／独立one-shot、pilot試行01/02、消費済marker・SOURCE・契約・結果は不変である。

### 保存済み証拠のscope

linear H4 1.00 Å、STO-3G、8 spin orbitals、保存DF rank12、生成順prefix G3/G6/G9、保存stateの元norm²、全256次元。δ=1/8・1/16 Ha^-1、R=2・4、T=1・4 Ha^-1、zero込み108行。非冗長nonidentity Pauli-tail、shared ±／ordinary／independent ±の同一概念label数比較である。

レビューは、G3/G6でvisibility改善とphase bias増加のtrade-offがあること、G9の絶対contrastがunresolvedであることを支持する。T1bのscalar rate適用域、厳密固有状態、RPE shots、native費用、総資源GOは認定していない。**本書では新しい勝敗件数や最良分割を作らない。**

## 3. 記号・task・証拠水準

|記号|意味・単位|
|---|---|
|j, m, k, a|分割、方式、RPE round、測定軸X/Y|
|T_k, t_0|相対時間、基準時間 [Ha^-1]。候補はT_k=2^k t_0|
|z^0_{mjk}|数値pilotが対象とする理想平均信号。物理実装後の信号とは分ける|
|z^p_{mjk}|実際のwrapper・状態準備・測定が与えるcoherent平均信号|
|A^-_{mjk}|対象とする真の平均信号の正のvisibility下界。中心値そのものではない|
|φ_k^*|目的位相。raw信号のArgか、E_* T_kかは未採択|
|b^+_{mjk}|目的位相との円周距離の上界 [rad]。数値uncertaintyのcoverageも必要|
|γ_k, q_{mjk}|許容round角誤差、統計誤差へ残す正の角予算 [rad]|
|α_{k,a}, N_{mjk,a}|測定軸別失敗確率と整数shot数|
|C_{mjk,a}^{(g)}, P_{mjk,a}^{(g)}|全wrapper費用と状態準備費用。metric gごとに定義|
|F_m^{(g)}|一回の前処理・calibration等の固定費用。同じ単位に換算できる場合のみ足す|

「必要条件」はこのpacketで選ぶ無補正RPE/会計routeを成立させる条件であり、任意の推定器に対する下界・不可能性ではない。「十分条件」は仮定付きで失敗確率や費用の順序を保証する式。「連続proxy」は整数丸め等を省略したscreeningであり、実測勝敗や最適資源の定理ではない。

目的を二つに分ける。

- **A：保存raw信号に対する単一時刻の位相task。** 既存T=1,4の診断を再利用できるが、raw位相が一つのE_* Tに線形に従う保証がないため、そのままRPE energy taskと呼べない。
- **B：特定エネルギーE_*に対する多round RPE task。** 事前energy interval、t_0、全round、guiding-state/phase contamination、最終精度・confidenceを必要とする。今回のT=1,4だけでは中間roundを含む完全なdyadic列を検証していない。

AをBの代替PASSに使わない。最初にどちらを検討するかはユーザーの採択事項である。

## 4. 位相誤差台帳：何を足し、何を二重計上しないか

### 4.1 保存数値の意味

pilotの`phase_bias_rad`は`TOTAL_PRINCIPAL_PHASE_DIFFERENCE_TO_RAW_H`。`phase_error_radius_rad`はその計算の不確かさであり、bias許容値ではない。従来のsignal 2^-24、phase 2^-18 rad、floor 1/4は**保存pilotの精度仕様**として保全し、新RPEの精度要求へ黙って転用しない。

将来、保存中心\(\widehat b\)とその条件付き角半径r_bを採用するなら、branch条件の下で

\[
b^+_{\mathrm{saved}\to\mathrm{raw}}:=\min\{\pi,|\widehat b|+r_b\}.
\tag{P1}
\]

これは数値coverageを仮定した上界の構成規則であり、今回は数値代入しない。小さい半径は小さいbiasを意味しない。

### 4.2 台帳routeを一つ選ぶ

|route|足す項|二重計上を避ける規則|
|---|---|---|
|raw直接route|保存総bias上界＋raw参照から目的位相への差＋実装／SPAMによる差|保存総biasにH_tilde−H_fullモデル差は既に含まれる。ηをもう一度加えない|
|split分離route|H_tilde基準アルゴリズム位相上界＋model phase上界＋raw参照から目的位相への差＋実装／SPAM差|H_tilde理想信号は未評価。現在は純アルゴリズムbiasの点値を持たず、必要な下界・branch条件付きのenclosureが必要|

raw routeの保存角半径はmethodとreference双方の数値誤差を既に含む。別のphase誤差を重ねて加えたり、逆にその半径を消して中心だけを使ったりしない。

Hermitianな両Hamiltonianに対する\(\eta_j(T)=|T|B_j\)は、参照とsplit idealの複素信号差を覆う。位相への変換には、参照の正のvisibility下界L、η<L、必要floor、zero/branchからの分離が必要で、成立時のみarcsin(η/L)型上界を使える。**η自体をradの位相上界として足さない。** 現契約のモデル差をHDへ吸収したり、保存Pauli係数を修正したりしない。

### 4.3 Oracle/referenceからenergyへの橋

保存stateのnorm²による正規化は定義上の操作であり、E_*固有状態の認定ではない。絶対残差受理基準だけからoverlapやground energyへの位相保証は得られない。対象固有値の識別やgap等の追加情報が必要になる。

例として、正確なH_fullの対象固有stateへの**証明された重み**g>1/2があれば、理想参照信号について

\[
z_{\rm raw}(T)=g e^{iE_*T}+r(T),\quad |r(T)|\le1-g,
\quad |z_{\rm raw}(T)|\ge2g-1,
\quad d_{\mathbb S^1}(\arg z_{\rm raw},E_*T)
\le\arcsin((1-g)/g).
\tag{P2}
\]

これは既知のdisk幾何による一つの十分条件の例であり、gを新規算出したり、現在のstateにその条件を認定したりしない。g>1/2だけで希望するRPE角予算を満たすわけでもない。guiding stateの条件を明示しないenergy claimは未解決のままにする。

### 4.4 物理実装とSPAM

exact HD exponentialを実際のDF sweep/PFへ置く誤差、rotation synthesis、control/identity phase、読み出し・状態準備による信号差を別台帳に置く。真の平均signalに対する複素実装誤差d_implが正当化でき、理想methodのvisibility下界A_0に対してd_impl<A_0なら、

\[
A_p^-\ge A_0-d_{\rm impl},\qquad
d_{\mathbb S^1}(\arg z^p,\arg z^0)\le\arcsin(d_{\rm impl}/A_0).
\tag{P3}
\]

実装誤差が適用前のnormで与えられる場合、unitary積のtelescopingと期待値への伝播を先に証明する。SPAMは自動で0としない。費用としての状態準備と、phase contaminationを生むstate品質は別fieldにする。

補正を選ぶなら、目的biasの残差区間と補正値の不確かさ・取得費用を定める。dense真値の無料差引きは禁止。現時点では補正方式は未採択である。

## 5. RPEの条件付き適用route

一つのenergyを読む候補routeでは、時間・符号・identity基準を固定し、\(T_k=2^k t_0\)、位相\(E_*T_k\)を用いる。energy intervalは既知のphase周期aliasを排除する必要があり、幅<2π/t_0という条件と、初期branchの位置・decoder初期化を組にして定める。幅だけで初期化を済ませない。

標準のnearest-consistent-branch decoderを採用する場合の保守的候補は、各roundの総角誤差を\(\gamma_k<\pi/3\)とすることである。最終roundには別途\(\gamma_K/T_K\le\epsilon_E\)を課す。中間roundだけ小さくてもenergy精度の証明にはならない。

\[
b^+_{mjk}+q_{mjk}\le\gamma_k<\pi/3,\qquad
0<q_{mjk}<\pi/2,\qquad
\sum_{k,a}\alpha_{k,a}\le\alpha_{\rm tot}.
\tag{R1}
\]

採択後に全roundでこの条件・signal下界・初期化・統計モデルが保証されれば、union boundで全round角条件を同時に満たす確率を管理できる。主値差は円周距離で扱い、energyへ戻す際にalias-free branchの保証を必要とする。π/2をRPE branch成功限界として使わない。[2021訂正][erratum]および[PR論文Appendix B][paper]。

本routeのR1は、decoderと予算方式を選んだ場合の十分条件であり、一般のRPE必要条件や最適性ではない。b^+が予算以上、またはb^+自体が未取得なら、このrouteの保証は作れない。**保守的上界の不通過を実際のbias超過や一般no-goと分類しない。**

`confidence付き絶対energy誤差`とRMSEは別taskである。成功時ε_Eの保証と総失敗確率だけで、無条件RMSEを認定しない。RMSEが必要ならfailure時損失の上限、energy interval、失敗確率の配分を別途会計する。

## 6. Visibilityとshots：既存式を適用できる条件

各軸のHadamard結果が±1、期待値が対応するRe/Im z^pで、軸内のshotが独立な場合、Hoeffdingと2軸diskから

\[
t_{mjk}=A^-_{mjk}\sin(q_{mjk})/\sqrt2,
\qquad
N_{mjk,a}\ge\left\lceil2\log(2/\alpha_{k,a})/t_{mjk}^2\right\rceil.
\tag{S1}
\]

両軸のαをround αの半分にする特殊例は、各軸\(\lceil4\log(4/\alpha_k)/(A^-\sin q)^2\rceil\)となり、最新レビューの式と同じである。visibilityを上げることに加え、位相biasを払った残余qが必要である。**この式を保存値へ代入していない。**

条件は、lawのfresh-IID per shot、branch内の共有label規則、測定符号、bounded outcomes、state/measurementの安定性である。coherent平均を求めたpilotは、finite-shotの分布・phase復元を実行した証拠ではない。同じ乱択回路を複数shotで使う場合は、このIID保証をそのまま使わず、回路乱択の集中誤差を含む別契約を要する。

このNは選んだHoeffding設計の十分shot数で、最小必要shot数や下界ではない。sharedの設計上界が対照の上界より低いことだけでは、最適shot数の優位性を証明しない。比較する場合は、同じ統計policyの実行可能なschedule同士と明示する。整数丸めでshot数が同じになる可能性も保持する。

## 7. 回路費用の会計

### 7.1 label、branch指数演算、native価格

保存pilotでは全方式の概念draw数はsR。sharedはbranch指数演算2sR、ordinary/independentはsRという形式的会計である。これをnative費用比へ直結させない。shared位置の二枝は

\[
\operatorname{diag}(e^{-i\theta P},e^{i\theta P})=e^{-i\theta Z_a\otimes P},
\qquad
\operatorname{diag}(I,e^{iaP})
=e^{ia I_a\otimes P/2}e^{-iaZ_a\otimes P/2}.
\tag{C1}
\]

正負共有のPauli rotationをまとめられることは既知の構造である。ただしClifford・parity network、routing、相殺・融合、native gate set、rotation精度を含む費用は未取得。因子数を最小費用の定理としない。[PR論文Appendix A.4][paper]。

全wordの負枝は同順sign reversalであり、逆順adjointではない。共有mapのΦ/DXDをCPTP conjugationへ変えず、independentのVXVをV†XVにしない。

### 7.2 誤差と費用を同じwrapperへ結ぶ

|費用要素|今回分かること|未取得／決めること|
|---|---|---|
|tail Pauli回転|law、角、label共有・branch意味論|native synthesis・Clifford／routing費用|
|deterministic HD|pilotはexact-Hermitian exponential target|DF sweep/PFの実装、追加bias、精度配分|
|controlled/directional操作|C1の代数|全wrapperでの最適化、control位相、workspace|
|identity/scalar phase|保存契約の移動を維持|相対phaseとして実装される箇所の費用と誤差|
|readout／ancilla|X/Yの測定意味論|測定を含む採用metric、native profile|
|state preparation|候補間でshot数が違えば相殺しない|Pの範囲、包含／除外、state品質|
|classical前処理・calibration|存在を0としない|固定費、shotごとの費用、通貨／metricの変換|
|費用の不確かさ|数値なし|sampling、compiler、移送、量子shot誤差を別管理|

費用vectorはT/CX/RZ/depth/workspace等を混ぜて単一数にしない。scalar primary metricを選ぶかPareto比較を採択する必要がある。深さのshot総和はserial累積depth-workであり、並列wall-timeではない。固定費Fを加えるには同じ資源単位へ換算する。換算不能なclassical費用は別列のまま保持する。

## 8. 総資源の成立条件

固定した同一task・同じ成功基準を満たすscheduleについて、metric gごとに

\[
G_m^{(g)}=F_m^{(g)}+
\sum_{k,a}N_{mjk,a}\bigl(C_{mjk,a}^{(g)}+P_{mjk,a}^{(g)}\bigr).
\tag{G1}
\]

状態準備を除いた研究scopeも選べるが、その場合は別のno-prep量として表示し、Pが相殺されたと言わない。同一Pでもshot数が違えば差が残る。

実用的な改善claimには、両scheduleの精度・統計適格性、同じmetric・費用scope、coverageと不確かさ、強いbaselineを確認した上で、総費用を比較する必要がある。候補となる頑健な十分条件は\(G^+_{\rm sh}<(1-\mu)G^-_{\rm cmp}\)、0≤μ<1。μと上下界の意味・coverageは未採択である。μ=0や5%等を暗黙の既定にしない。二つの単なるupperを比べたことを、この頑健分離と取り違えない。

固定費なし・一時刻・同じaxis費用・同じconfidence・整数丸めなしのproxyに限れば、

\[
\frac{C_{\rm sh}+P_{\rm sh}}{C_{\rm cmp}+P_{\rm cmp}}
<\left(\frac{A^-_{\rm sh}\sin q_{\rm sh}}
{A^-_{\rm cmp}\sin q_{\rm cmp}}\right)^2
\tag{G2}
\]

がこの連続proxyでのbreak-even条件になる。reviewの式をP込みに表しただけで、現H4で評価していない。G2を全RPE・異なるconfidence・整数shot・固定費込みの必要十分条件へ格上げしない。各roundで勝つことは総額の十分な経路になり得るが、全roundでの勝利は総額改善の必要条件ではない。

強い対照には、同じPauli lawのordinary/independentのほか、既知の部分乱択RTEとdeterministic endpointを候補に含めるか判断する。既存pilotの対照を遡及変更しない。baselineにも同じ情報・最適化・精度配分・controlの機会を与える。ρやλ²単独の資源STOP規則は採用しない。

## 9. SOURCEの接続可能性：静的な対応表

|既存SOURCE／文書|再利用可能な考え方|そのまま移せない箇所|
|---|---|---|
|`m1s_finite_signal_v2/pilot.py`|phase球、保存bias、model差別台帳、label／時間の定義|物理wrapper・RPE decoder・shots・native費用は未実装|
|`m1s_finite_signal_v2/numerics.py`|条件付きsignal/phase半径、exact targetと中心の区別|platform certificate・physical synthesis誤差の保証ではない|
|`rpe_resource_accounting.py::_hoeffding_shots`|S1の有界shot式、軸別α、費用和、厳しいbranch limit|既存evaluate関数はeffective eigenstate/unit radius/finite-RTE等を仮定。M1-Sをそのまま渡さない|
|同`RPERoundSpecification`|T_k=2^k t_0の帳簿|δ_timeとpilot δ_relを同一の時間変数として自動接続しない|
|`rpe_hadamard_interrogation.py`|bit0→+1、X/Y expectation、ancilla/readout provenance|旧wrapperはordinary-controlled、既存短round限定。shared/independentのbuilderではない|
|`df_rpe_hadamard_compiled_cost.py`／文書|全wrapperをcompileし、axis別期待費用へつなぐ設計|旧DF/RTE/trajectoryとM1-SのPauli指数lawの価格は同一と未証明|

**符号の技術事項**：pilotは+ i T H。旧Hadamard文書は− i T Hに対するIm z=−sin(ET)を記載する。X/Yの測定器定義そのものはRe/Imだが、energyへの変換符号が違う。future adapterでenergy sign・relative time・ancilla branch label・identity phaseを明示照合する必要がある。現在そのadapterを書いたり実行したりしていない。

旧resource moduleの`certified`分類は同moduleの仮定scopeの下に限る。M1-Sをその分類へ変更しない。Monte Carloによるcompiled費用の標本数S_MCは量子shot Nではなく、G1にもう一度倍率として掛けない。

## 10. Decision Packet：未決事項と最小検証案

### ユーザーの科学的採否が必要な項目

|ID|決める内容|現在の状態|
|---|---|---|
|D1|限定継続するか、理論・信号ノートで区切るか|未採択|
|D2|raw信号の単一時刻taskか、E_*の多round RPEか|未決|
|D3|誤差の意味（高確率絶対誤差／RMSE）、精度、confidence|値・policy未決|
|D4|t_0、round/horizon、energy prior、decoder初期化|未決。T=1,4から補完しない|
|D5|bias route、補正の可否、state/参照の保証水準|未決。無料oracle補正なし|
|D6|候補cell・baseline集合、事後選択と将来holdoutの扱い|未決。新分割/gridなし|
|D7|primary metric/Pareto、native profile、HD実装と精度|未決|
|D8|state準備費用と品質、classical固定費のscope|未決|
|D9|IID/reuse・shot配分・統計policy|候補式のみ、未採択|
|D10|materiality μ、費用不確かさ・coverage、failure分類|未決|
|D11|次期検証の厳密scope、resource/call/output上限、one-shot|未固定。新規契約の時間capは既定なし|

科学的判断を要しない技術事項は、原本・hash照合、旧SOURCE仮定の対応、符号・時間・費用単位の明示、誤差二重計上防止、参照の整理である。本packetで対応し、これだけを理由に同じGPT詳細レビューへ戻さない。

### 追加検証候補：依存順に必要なものだけ選ぶ

|候補|目的|今回の状態／別途承認|
|---|---|---|
|V0 非数値のtask/契約固定|D1–D11を埋め、何が判明すれば方針が変わるか決める|本packetは候補の提示まで。ユーザー採択待ち|
|V1 保存signalからのphase/shot screening|既存gridで位相予算を支払えるかと設計shotを照合|未実施。数値再解析なので別のSOURCE固定・実行承認が必要|
|V2 限定wrapper比較|共通精度でdirectional/control/HD費用が利益条件を満たすか|未実装・未実施。V1等から必要と判断された場合だけ別契約・compile承認|
|V3 最小不足signal／独立照合|採択したtaskの欠測や数値信頼性を埋める|現在未選択。新Hamiltonian/signal/RPE計算を自動要求しない|

V1は「保存結果に対する事後screening」であり、全RPE成功率やnative総優位を認定しない。上界が不通過なら、実際の失敗・一般no-go・科学的NO-GOを自動出力しない。全候補を順に実施する計画でもない。

### STOP・分類

- 入力/SOURCE/contract identity不一致、壊れたschema、未承認の評価要求：`TECHNICAL_STOP`、証拠保全、実行しない。
- 目的位相・精度・alias・bias/visibility保証・費用等の欠落：`UNDETERMINED_MISSING_TASK_OR_EVIDENCE`。推測で補わない。
- 採択した十分条件・資源上界が不通過：将来は`UNRESOLVED_UNDER_SELECTED_POLICY`等として保存し、一般性能negativeと区別する。
- すべての技術条件が揃っても、正式採択・source-bound別実行承認までは`EXECUTION_NOT_AUTHORIZED`。
- 不一致や失敗時にgrid・精度・thresholdを緩和して再評価しない。新契約・旧成果保全・明示承認なしにretryしない。

**本packet作成後はmandatory STOP。研究の限定継続・新しい科学条件・実験契約は未採択、新しい科学計算とGPT詳細レビューは開始しない。**

## 11. 成果物と静的確認

[契約候補][candidate]、[machine候補][machine]、[入力identity][identity]、[非数値静的checker][checker]、[静的結果][static]、[証拠索引][index]を一組として読む。static PASSは、closed guard・null未決値・参照/hash・旧成果保全の確認に限り、式のplatform coverageやRPE性能の証拠ではない。

非数値静的検査は156 PASS／0 FAIL。初回の3件のmetadata検査不通過（自分自身の出力fileの作成順、LaTeX文字表現、Gitの日本語path quotingの扱い）は`static_validation_initial.json`へ保全した。checkerの修正後に再確認し、科学SOURCE・算術条件・保存科学結果を変えていない。[保全監査][preservation]と公開照合receiptを別に保存する。

先行資料の関連箇所だけを確認した。[RPE訂正][erratum]はround誤り条件の修正、[PR論文][paper]はAppendix A.3/A.4・Bの平均、control、branch条件を参照する。網羅的新規性調査や新しいGPT科学レビューではない。

[review]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e5ed0db6b8f50ed02ee9bc17cc6d0e2d4ed0898b/docs/research/m1s_finite_step_pilot_results_gpt_scientific_review_20261017.md
[result]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/025c5a90888b02b73d2815fbb10d6bd1d2823602/docs/tracks/algorithm_codesign/m1s_finite_signal_v2_completed_results_20261016.md
[candidate]: m1s_rpe_next_verification_contract_candidate_20261017.md
[machine]: ../../../artifacts/track_b_m1s_rpe_conditions/2026-10-17/v1/next_verification_contract_candidate.json
[identity]: ../../../artifacts/track_b_m1s_rpe_conditions/2026-10-17/v1/input_identity.json
[checker]: ../../../scripts/tracks/algorithm_codesign/check_m1s_rpe_conditions_static.py
[static]: ../../../artifacts/track_b_m1s_rpe_conditions/2026-10-17/v1/static_validation.json
[index]: ../../../artifacts/track_b_m1s_rpe_conditions/2026-10-17/v1/evidence_index.json
[preservation]: ../../../artifacts/track_b_m1s_rpe_conditions/2026-10-17/v1/preservation_audit.json
[erratum]: https://journals.aps.org/pra/pdf/10.1103/PhysRevA.104.069901
[paper]: https://arxiv.org/pdf/2503.05647v2
