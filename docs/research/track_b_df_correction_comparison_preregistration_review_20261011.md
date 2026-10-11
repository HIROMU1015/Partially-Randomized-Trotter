# Track B：DF三次補正 A/B 比較の実行前契約レビュー

- 作成日：2026-10-11 JST
- レビュー開始承認：利用者の「レビューを開始して」
- Repository：`HIROMU1015/Partially-Randomized-Trotter`
- 固定対象commit：`73b2102a305f018d22348418adc7939cae63f0b2`
- Branch：`track-b-df-correction-comparison-preregistration-20261011`
- 対象：whole-commutator方式Aとcoin-first局所Pauli pairing方式Bの結果前契約v1・実行計画・関連source。
- 位置付け：科学的な実験設計のレビュー。比較結果のレビュー、source実行認可、新規アルゴリズム採択ではない。

## 0. 判断

**比較の目的・二方式・固定入力の維持は支持する。ただし、v1を無修正で本番実行することは採択しない。以下の会計・認証・判定範囲の修正を契約v2へ反映し、source・focused tests・実行guardの準備を一つのCodex作業として進める。**

提案する判断ラベルは `REVISE_CONTRACT_AND_PREPARE_SOURCE / EXECUTION_NOT_AUTHORIZED`。これは本レビューの判断であり、リポジトリの保存statusを書き換えたものではない。

比較方式、fixture、seed、t・δ・γ・s、P5、最終誤差・失敗確率、compile/T呼出し上限を増やさない。今回の修正の主眼は、同じ作業から何を科学的に結論できるか、実際に実装する有限bit分布と会計が一致するかを明確にすることにある。

現在の計画は、CDにおける条件付き精度・十分shot予算・損益分岐と、別入力Eにおける局所実装screeningを行うもの。**PR全体の実測総T費用の勝敗やwhole方式一般の終了を、この計画だけで確定するものではない。**

前回の「限定比較で差が残らなければ候補を区切る」という表現は、この識別可能性を条件にする必要がある。未取得の同入力費用や不完全なT coverageによる判定不能は、資源的非優位の証拠ではない。

## 1. 証拠・取得状況・今回実施したこと

### 1.1 正本

比較契約[R1]、実行計画[R2]、機械可読contract[R3]、handoff[R4]、未決事項[R5]、reference source manifest[R6]を固定commitで取得した。Remote branch HEADはこのcommitと一致していた[R11]。

参照sourceのうち、数値guard[R7]、R1 strict synthesis guard[R8]、SP economics/sequence evaluator[R9]を本文まで確認した。runtime identity[R10]も取得した。

資料の不足によりレビューが停止している状態ではない。一方、比較runner、angle adapter、hard guardが未実装であることは契約に記録された作業状態であり、未pushの欠損とは区別する。

### 1.2 実施・非実施

実施したのは、取得した契約・sourceの読解、平均と誤差伝播の一般的な導出、有限bit・資源会計の論理確認、公式ソフトウェア資料の限定照合である。補助的に、状態の規格化、12区間の失敗確率配分、誤差予算の残余、gate誤差配分、CPU上限の算術を有理数で確認した。

登録CD/Eの乱数生成、NPZ load、Hamiltonian/信号行列生成、6 taskのshot予算計算、compile、Clifford+T合成、量子shot、旧監査の再実行は行っていない。新しいA/B性能結果を得たとは主張しない。リポジトリへの書込みも行っていない。

この補助算術は独立監査の77検査・15テスト、旧レビューの18自己検算、準備段階の63メタデータ検査に加算しない。

## 2. 比較が答える問いと、答えない問い

### 2.1 支持する比較

Aはwhole nested commutatorの回転を正確率・weight 1で使う。Bは同じ局所TをPauli展開し、coin-first identity–Pauli pairingと測定weightを使う。BをweightなしのPauli qDRIFTへ置き換えてはならない。

有限stepの平均演算子は異なる。共通の理想信号へのalgorithmic bias、digital/native誤差、統計誤差を戻して比較する設計は正しい。同じ係数質量や一回路単価を直接比べて方式の優劣としない。[R1–R3]

### 2.2 二domainの識別可能性

| domain | 本計画で取得するもの | 未取得のもの | 許される結論 |
|---|---|---|---|
| CD | 固定3-mode入力の平均・保証式・moment/range・十分shot予算 | 同じCD入力の全PR回路費用、補正回路単価、prep/readout費用と実装誤差認証 | 保証予算の比較、未取得費用に依存する損益分岐、実装条件付きの精度envelope |
| E | 固定r=3/r=6の局所center回路、CX/RZ/depth、限定T合成、conditional weight/bias | CD入力の回路単価、全系Gaussian embedding、PR trajectoryの全資源 | 指定された局所実装の費用・合成可能性。PR総費用の勝敗ではない |

CDとEが別fixtureであることは欠陥ではなく、混同を避けるための重要な区別である。ただし、二表を併記しても同入力の全体比較にはならない。

CDの期待補正頻度2.5をEの回路単価に掛けない。r=3というサイズ一致だけでも同入力とは言えない。EのT角度をCDへ流用しない。

### 2.3 generic Aの位置付け

EのAはoptimization level 0のgeneric unitary decompositionに固定されている。これは再現可能な参照実装のscreeningとして有効だが、構造を利用する最善のwhole回転の下界ではない。[R2]

前回レビューで認めた構造的な実装改善の可能性は残る。ただし、本契約の範囲を広げて新compiler探索を今追加することは要求しない。今回の結果で棄却できるのは、十分な共通会計が成立した範囲での**登録実装・登録予算policy**である。

## 3. そのまま採用できる結果前条件

### 3.1 CDの固定入力と時刻

以下を維持する。[R1–R3]

- 3-mode、3-block、係数9/20・7/20・1/4、primitive rank/sign=(1,+1),(2,−1),(1,−1)。
- NumPy1.26.4/PCG64、seed6121107、既存recipeの生成順序。
- t=1、δ∈{1/8,1/16,1/32}、n∈{8,16,32}、γ=5/4、κ=4。
- s=1、paired cutoff4=P5。
- CDはsynthetic development入力であり、分子・held-out・一般DFの代表性を認定しない。

この固定集合を変更したり、良い結果を得るために別seedやγを探したりする必要はない。

### 3.2 初期状態案

\[
|\psi\rangle=\frac{|001\rangle+i|010\rangle+|100\rangle+|011\rangle-i|101\rangle+|110\rangle}{\sqrt6}
\]

は、6つの直交基底への振幅の絶対値が等しく、規格化される。非vacuumの診断状態として採用できる。配列index1,2,4,3,5,6とlittle-endianの対応を仕様・テストで固定する。

この状態は1粒子・2粒子sectorの重ね合わせであり、固定電子数の分子基底状態ではない。今回の複素信号比較に使うことはできるが、ground-state/QPE性能の証拠にはしない。状態を結果後に変更しない。

primary biasとmomentはstate-independentな上界なので、この状態のexact signalを見てshot予算を縮めない。状態準備費用をhとして残すことは正しい。将来の実装保証には、その準備誤差もnative台帳に含めるか、理想準備という条件を明示する。

### 3.3 失敗確率の解釈

A/B×3δ=6 task、X/Y各axis/taskに1/240なら、

\[
12\times\frac1{240}=\frac1{20}=0.05.
\]

X/Yそれぞれのfamily予算1/40を保ち、6 taskへ等分する案を採用できる。Union bound自体にtask間の独立性は不要。各estimatorについて必要な独立shot/drawの条件とは別である。

これは将来の推定量についての同時成功条件であり、決定論的な数値enclosureの失敗確率や新規性のp値ではない。今回は量子shotを実行しない。

### 3.4 複素精度とaxis精度

各axisの誤差を1/20000とすれば、複素誤差は高々√2/20000で、指定1/10000より厳しい。これは安全だが、1e−4の複素許容域を全部使う最小費用設計ではない。

両方式に共通なので、この保守的規則は保持してよい。ただし費用結果は、この固定axis規則の十分予算であり、最小shot数ではない。

## 4. Bのweighted平均の検算

\[
T_\tau=\sum_j\beta_{\tau j}P_j,\quad
\ell_\tau=\sum_j|\beta_{\tau j}|,\quad
\alpha_\tau=\arctan(u\ell_\tau),\quad
w_\tau=\sqrt{1+(u\ell_\tau)^2}.
\]

PauliがHermitian involutionなら、

\[
w_\tau\sum_j\frac{|\beta_{\tau j}|}{\ell_\tau}
 e^{i\operatorname{sgn}(\beta_{\tau j})\alpha_\tau P_j}
=I+iuT_\tau.
\]

理由は、w cosα=1、w sinα=uℓである。外側coinとqτ=cτ/A_massを入れ、ηu=A_mass δ³/2を用いると、

\[
\bar C_B=I+\frac{i\delta^3K_3}{2}.
\]

T=0ならidentity・weight1の分岐へ戻す。ℓ=0で除算しない。

Aのmeanは有限stepでこの式と同じではない。A/Bに同じoperator平均を要求する検査ではなく、それぞれの登録meanと共通targetへの誤差を確認する検査にする。

## 5. 有限step上界は構造的に整合している

### 5.1 使う比較演算子

厳密な対称積をS、厳密correctorをC=exp(iδ³K3/2)、中央P5を用いたPR平均をMと書く。S,Cはunitary、\|M−S\|≤e_Rとする。

既存のformal cancellationとmajorantから\|CSC−U\|≤b_symを使用する。b_symは、次数5以上の二つの指数tail、corrector一次とSの次数2以上の積、corrector二次以上をそれぞれ上界にしたものとして整合する。[R1、R7]

これはscalar式の妥当性の確認であり、QR assembly・native回路・有限bitのcertificateが既にあるという意味ではない。

### 5.2 A

Aの平均correctorはunitaryの凸平均なので\|Cbar_A\|≤1。\|Cbar_A−C\|≤e_Aから、

\[
\|\bar C_A M\bar C_A-CSC\|
\le e_R+2e_A.
\]

従ってb_A=b_sym+2e_A+e_Rは妥当である。

### 5.3 B

Bのweighted平均をcontractionとは仮定できない。ただし\|Cbar_B−C\|≤e_Bなら\|Cbar_B\|≤1+e_Bである。従って、

\[
\|\bar C_B S\bar C_B-CSC\|
\le e_B(1+e_B)+e_B=2e_B+e_B^2,
\]
\[
\|\bar C_B(M-S)\bar C_B\|\le e_R(1+e_B)^2.
\]

契約の

\[
b_B=b_{sym}+2e_B+e_B^2+e_R(1+e_B)^2
\]

はこの区別を正しく戻している。ここをb_sym+2e_B+e_Rへ省略しない。

### 5.4 全時間と誤差予算

局所誤差b、理想Uがunitaryなら、telescopingにより

\[
\|\bar M^n-U^n\|\le(1+b)^n-1.
\]

全algorithmic capは2.5e−5、digital/native/numericを5e−6/5e−6/1e−6とすれば、最悪でもaxis統計幅は

\[
5\times10^{-5}-2.5\times10^{-5}-5\times10^{-6}-5\times10^{-6}-10^{-6}
=1.4\times10^{-5}>0.
\]

algorithmic項だけを登録されたcertified上界に置き換え、他三項は割当最大値を控除する規則も一貫している。observed errorやexact signalを見て予算を追加で戻さない。

ただし**予算を予約したことは、その誤差を実装が満たした証拠ではない**。CDのnative実装がない段階では、実装誤差が割当内に収まることを条件としたenvelopeと記す。

## 6. moment・range・Bernstein会計

### 6.1 ℓ≤32の根拠

d=2³=8、β_P=Tr(PT)/dのPauli規約なら、

\[
\sum_P|\beta_P|
\le d\left(\sum_P\beta_P^2\right)^{1/2}
=d\sqrt{\operatorname{Tr}(T^2)/d}
\le d\|T\|\le8\times4=32.
\]

従って理想CDでEℓ²≤1024とするprimary上界は安全である。Eのr=6には32を流用しない。

この上界はtightnessを保証しない。詳細momentをsecondaryに置く方針は維持できるが、primaryの結果を最適sampling・最適confidence・最小資源へ一般化しない。Aはweight1のため特にtightなmomentを持ち、Bは粗い一般上界を使う、という非対称性がある。これは事前固定したsufficient-budget比較として許されるが、アルゴリズムの内在的な最適性比較ではない。

### 6.2 理想momentと十分shot数

元RTE倍率Γを保ち、Aではν_A=Γ²、W_A=Γ。Bでは独立slotのもとで

\[
\nu_B\le\Gamma^2(1+\eta u^2\,1024)^{2n},\qquad
W_B\le\Gamma(1+u^2\,1024)^n.
\]

\(|Z|\le W\)、Var(Z)≤E[Z²]≤νなら、中心化変数の絶対値は≤2Wなので、契約の

\[
N=\left\lceil\log(2/\alpha_{axis,task})
\left(\frac{2\nu}{s_{stat}^2}+\frac{4W}{3s_{stat}}\right)\right\rceil
\]

は十分条件として整合する。各不等式を上向きに評価する。

### 6.3 必須の追記：実装weightでenclosureを作る

finite-bit probabilityを使っても、外側ηを正確に維持し、各受理weightが同じpointwise上界以下なら上式は維持できる。

しかし、weight、ℓ、Γを丸めた実装について、理想上界だけを示して終えてはいけない。実装された\(\widetilde W\)を覆うrange、\(E_{\widetilde p}\widetilde W^2\)の上界を保存する。必要なら認証済みroundoff radiusを保守的に加える。

numeric biasの予算内であることは、理想Wを実装rangeとして使える証明ではない。meanの誤差とconcentrationに必要なrange/momentは別の保証である。

ℓ≤32は理想入力の命題として保持する。数値assemblyやweight評価の誤差を無断でゼロにせず、実行量への移行を証明する。証明できない場合はtechnical INCONCLUSIVE。

## 7. 必須の追記：有限bit lawと費用の一致

### 7.1 現在の記法の不足

v1は実装分布を48-bit largest-remainderへ丸める一方、費用式は理想qτ、pτjで書かれている。[R1–R3]

これは理想費用の式としては正しい。しかし、実装lawの期待費用と同一とは限らない。信号biasが小さいからといって、費用の差が自動的にゼロになるわけではない。

### 7.2 v2での修正

有限bit complete lawを\(\widetilde p_{i,e}\)とし、実装費用を

\[
\bar C_i^{exec}=\sum_e\widetilde p_{i,e}C_{i,e}
\]

で評価する。costが未取得ならこの重みを持つ記号式を保存する。理想lawの費用も参考として別fieldに残せる。

理想と実装の差をboundで扱う場合には、例えば

\[
|E_{\widetilde p}C-E_pC|
\le C_{max}\|\widetilde p-p\|_1
\]

を使える。C_maxや費用変数も未取得なら、その事実を式へ残す。

補正頻度の2nηは外側coinの機会数に関するもので、zero-T identityを実回転へ数えない。identityの量子costを0とする場合も、PR本体は実行される。

### 7.3 Gの範囲

\[
G_{i,x}=2N_i(h_x+C_{base,x}+c_{i,x})
\]

は、固定shot予算と加法的なblock会計に対する期待総gate数である。weightとcostが相関していても、Nが結果非依存に固定される限り、期待総cost=N E[C]という関係は成立する。そこをE[W²C]へ無断変更しない。

一方、native biasの\(E[|W|\sum\epsilon_{gate}]\)はweightとgate列の相関を保持し、無根拠にE|W|とEcostへ因数分解しない。

block単価の和は、whole-wrapperを一度compileした費用ではない。cross-boundary cancellationを測っていない場合は、加法的assembly模型／実装方針として表示する。depthをT/CX同様に単純加算した最適depthと呼ばない。

## 8. 数値認証は「高精度で計算した」だけでは閉じない

### 8.1 理想入力の定義

保存QR出力はbinary64であり、そのまま厳密unitaryとみなさない方針は正しい。ただし、「どの理想frameからの距離を認証するか」を結果前に一意にする必要がある。

既存seed・draw順・符号を変えず、例えば固定乱数行列の厳密QR規約、または保存float frameに対する規約固定のorthonormal化との対応を明示する。どちらを使うかをCodexが技術的に具体化し、対象とassembly errorの定義を固定する。元の保存recipe自体を差し替えない。

この定義と多体primitiveへの誤差伝播がないまま、係数が実数である、Q²=I、\|T\|≤4を数値行列へ直接適用してはならない。

### 8.2 参照guardの範囲

`numerics.py`[R7]は、IEEE binary64と標準BLAS forward-error仮定、与えられたbinary64行列に条件付きのguardであると明記している。物理DF誤差や今回のPauli/QR組立誤差まで自動的に保証するものではない。

この条件付きguardを使用すること自体は排除しない。定理で保証する理想量、計算モデル依存のnumerical guard、単なるdouble residualを区別して出力する。多倍長で繰り返しただけの数値をcertificateに昇格させない。

### 8.3 zero・tie・LRMの停止

厳密なtrace-zero、number/parity symmetryで消えるPauli係数、同じremainderのtieなどは起こり得る。区間が0やfloor境界をまたぐだけでは、係数が非zeroともexact zeroとも証明できない。

v1の「lexicographic tie」と「判定不能ならSTOP」は安全だが、正しい恒等式のゼロで必ず停止するような実装を作っても有益な比較にならない。off-domain controlsで、exact zero/tieを証明した場合と単なる未分離intervalを区別する。

thresholdによる係数のdropは禁止のまま。largest-remainderで最終digital確率が0になることと、係数を都合よく削除することは別であり、前者はL1誤差に課金する。

有限precisionで理想LRMの順位が決まらない問題への小さい代替案として、事前に固定した有理数の計算用lawに対してLRMを正確に定義し、そのlawと理想lawの距離を認証する方法がある。これはv1の理想LRMとの同一性ではないため、使用するなら結果前の明示的amendmentとする。登録結果後のprecision retry、tie変更、近似dropによる救済は行わない。

### 8.4 complete-lawのL1距離

外側coin、tuple、Pauli、RTE次数を階層生成する場合でも、biasは一つのcomplete drawのlawへ戻して評価する。\(\|\otimes p_d-\otimes\widetilde p_d\|_1\le\sum_d\|p_d-\widetilde p_d\|_1\)から、Wmax×ΣΔ_L1という安全なboundが得られる。

条件付きlawの距離には親branchの確率を戻すか、安全な上界にする。3n complete drawsと、内部coin/node数を混同しない。ΓやB側weightのroundoffは別途覆う。

## 9. native構築と22 compileの判断

固定all-to-all、Qiskit1.3.0、rz/sx/x/cx、optimization0、seed20261011を維持した代表compileは、限定screeningとして妥当である。[R2]

Bはsupport weight wについて、parity ladderからCX=2(w−1)（uncontrolled）、CX=2w（controlled）、任意角RZ=1/2個が得られる。ただし、basis-changeに伴うClifford回転もcompilerのtotal RZ/1Qへ含まれる。

**任意角RZ数とtotal RZ数を分ける。** また、同じwでもX/Y/Zの内訳により1Q数は違う。各wの代表compile一つの1Q数を、そのclass全eventへ掛けない。全Pauliラベルについて固定IRから解析的に数え、代表compileでそのloweringを検査する。これは4096 compileを要求するものではない。

identityやzero係数の構造的除外は証明付きで行う。符号・control-zero branch・全体/相対位相のcontrolsは登録performance用compileとは別のoff-domain testとして準備する。

## 10. T合成：同じgate精度は採用できるが、最適資源比較ではない

### 10.1 固定精度の意味

ε_gate=2.5e−11と最大32000 occurrenceから、

\[
32000\epsilon_{gate}=8\times10^{-7}
\]

が成立する。matrix/angle/compiler誤差2e−7を加えるlocal1e−6の算術は整合している。

ただし、これは全必要gateが合成され、位相を含む各誤差が認証された場合の上界である。48個のA keyしか取れていない段階でwhole回路の1e−6保証を完成したとは言えない。

Bの任意角RZは1/2個なので、共通ε_gateはBにとって大幅な過剰精度になり得る。それでもbackend較正の固定policyとして採用してよい。ただし、同じ最終errorで各方式のprecisionを最適配分した費用ではない。結果後のprecision緩和は禁止を維持する。

### 10.2 64 keysで全量が取れる保証はない

Aの位置抽出48、Bの8、phase reserve8という上限は維持できる。ただしAの全unique角がcoveredであるかは未確認であり、位置抽出は統計的な無作為sampleでもない。平均T×RZでtotalを埋めない。[R2–R4]

Bの4angles/rというcoverageも、実際のIRの符号・half-angle・basis Clifford・phaseを含めて確認する。Bが完全であると、angle表の名前だけで決めない。

### 10.3 改善案：全量不明でも使える固定実装の片側bound

kをcanonical key、m_kをその出現回数、t_kを保存されたstrict-valid sequenceのT+T†数とする。既知Clifford部分のT数を別に正確会計し、合成済み集合Sから

\[
L_A=T_{known}+\sum_{k\in S}m_k t_k
\]

を計算する。未合成keyの費用が非負であることから、これは**既得sequenceを使い、各RZを別々に置換してそのT数を加算する固定実装計画のcompletion下界**になる。

B全体のcertified費用上界U_Bが得られ、L_A>U_Bなら、A totalの具体値が不明でも、その限定実装計画の局所T分離は示せる。出現multiplicityを必ず掛け、deduplicateされたkey数だけを数えない。

重要な限定：

- t_kは最短合成T数ではない。
- この下界は任意compiler、別gate decomposition、合成後のcross-gate cancellation、whole方式一般に対する下界ではない。
- 未合成keyがある場合、Aの実行可能な全回路とそのerrorが認証されたとは言わない。
- Eでの局所分離からCD/PR総資源の分離は出ない。
- 以上の条件を満たせない場合は片側boundを出さず、既定T_INCONCLUSIVEを保つ。

`A_TOTAL_T_UNKNOWN`と`FIXED_PLAN_LOCAL_T_SEPARATION`は別fieldにする。この判定規則を結果前のv2へ加えることを推奨する。これはsample外挿ではなく、既知部分の非負加算に基づく限定boundである。

### 10.4 合成順序の情報価値

有限budgetをAの位置sampleで先に使い切ると、Bの全量すら得られず終了する恐れがある。v2ではinventory全体をまず固定し、その後の実行順をBの必要key→必要phase→Aの登録位置keyとすることを推奨する。

key集合・精度・seed・最大64は増やさず、空き枠補充もしない。順序変更は結果前に記録する。A/Bのwinnerを見たadaptive selectionではない。resource/数学的失敗時のSTOPは維持する。

全Tが不明でもCD・CX/RZ・既得Tの監査結果は保持し、どの出力が完成したかを分ける。coverage不足を数理反例や科学的非優位へ分類しない。

## 11. phase・angle adapter：参照コードの具体的な注意

### 11.1 R1のstrict guardは再利用候補

`rte_reallocation/numeric.py`[R8]の`strict_guard`は、位相最小化を行わずFrobenius上界を計算する。`configure`もmpとivの両contextを設定している。この部分は今回の方針と合う。

一方、同ファイルの`angle_value`は、kindがpiでない場合をatan2へ送る。raw dyadic radiansを新kindとして渡すだけでは、例えば1 radがatan(1)=π/4に置き換わる危険がある。

閉じたangle種別でpi/atan/raw-radを明示分岐し、未知kindは拒否する小adapterを別namespaceに作る。旧sourceを変更して再利用したと偽らない。

### 11.2 SPのprojective guardを代用しない

`synthesis_placement/economics.py`[R9]の`error_guard`は16個のscalar phaseから最良witnessを選び、`exact_sequence`もphase-freeなpi/4 fast pathを含む。このguardを今回のstrict証明にそのまま使えない。

同ファイルの`sequence_matrix`は、Wを含むwritten-productの行列評価器として再利用候補になり得る。再利用できる関数と、意味論が違うguardを区別する。

これは比較runnerに既にこのバグがあるという指摘ではない。runnerは未実装であり、source準備時に防ぐべき流用誤りである。

### 11.3 angle identityと数値精度

公開pygridsynth仕様はRZ(φ)=exp(−iφZ/2)、radians、位相を無視するoptionを明示している[W1]。今回のup_to_phase=Falseは保持する。

semanticなatan角度区間と、native circuitのbinary64 dyadic角度を区別する。同一keyのreuseには、実際の対象角とstrict errorの対応を認証する。無条件2π moduloを使わず、符号・half-angle・4π周期・control相対位相のtestを置く。

100dpsでpoint計算しただけでは誤差enclosureにならない。mp/iv contextとdirected serializationを別々に扱う[W2–W3]。mpmath ivの未対応関数をpoint関数へ落としてcertificateと称さない。

## 12. 実行上限は維持できるが、完走保証ではない

native最大22、T最大64、全体wall2400s/CPU1800s/aggregate RSS3072MiBという上限案は、限定実行の停止条件として維持できる。[R2–R3]

ただしT per-key CPU20s×64=1280sで、phase CPU900sを超える。これは契約の数学的矛盾ではなく、全keyが個別上限を使うなら途中停止するという設計である。**最大64とは64件完了保証ではない。** 2workerでもCPU時間の総和は減らない。

hard guardでは、親・現在の子・終了済み子のCPU会計、同時RSSの集計、process group停止、prefix保存、queue打切りを明示する。RSSの周期監視でkillする方式は、瞬間的な超過を一切許さないmemory上限とは違う。検出間隔とovershootの扱いを報告する。

post-checkだけでhard containmentと称さない。逆に、このためだけに無制限の運用基盤を作らない。既存環境で実現可能なguardをoff-domainでテストし、限界を固定する。

一つのone-shot内部でnativeからliteral anglesを得てinventoryをfreezeし、その後T合成する順序は、選択規則が結果前固定なら成立する。literal keyが今ないことだけを理由に、この境界で新しい科学レビューをもう一つ追加する必要はない。

## 13. v2へ反映する事項

| ID | 反映事項 | 範囲 |
|---|---|---|
| M1 | CD条件付きenvelope／損益分岐、E局所screening、PR総勝敗を明示分離 | 必須 |
| M2 | 実装finite-bit lawでcost・moment・rangeを評価。理想値は別field | 必須 |
| M3 | 理想frame、assembly誤差、Pauli real/zero/tie、LRM判定の規約を具体化 | 必須 |
| M4 | strict guardとraw-rad adapter、angle inventory coverage、symbolic phase会計 | 必須 |
| M5 | partial Tの限定completion下界・B先行実行順・未coverage statusを結果前固定 | 採用を推奨。無理に全A T取得しない |
| M6 | 加法費用とcompiled全体費用、任意角RZとtotal RZ、共通gate精度の範囲を区別 | 必須 |
| M7 | resource cap・完了済みtask・停止原因の会計と、source-bound testsを具体化 | 必須 |

U1初期状態、U2confidence解釈については、本レビューで科学的に採用可と判断した。別の状態・失敗率へ変更する必要はない。U3以降は、ここで示した式・適用範囲に従う実装上の証拠が必要である。[R5]

## 14. 次のCodex作業とSTOP

次は、v1を履歴として保持し、今回の修正を契約v2・machine-readable JSON・実行計画へ同期する。その上で、runner、最小angle adapter、finite-bit accounting、証明用scalar計算、hard guard、off-domain focused tests、保存schemaを**一つのsource準備作業**として実装する。

登録fixtureの比較計算、CD6 taskの予算評価、E22 compile、最大64 synthを準備段階で先行実行しない。testsには別の小さいcontrolを使い、登録結果を覗いて契約を調整しない。

Codexへ求める証拠は、数学的specと実装の対応、precision/phase/digital分布の意味論、resource guardの動作、入力・source/runtime・contractの固定、何を実行していないかの記録である。各小修正でGPTの研究方針レビューを繰り返さない。

source Sへfreezeし、commit・push・remote再取得後にSTOP。新しい比較実行には、source/contractの整合確認と、別の明示的one-shot実行承認が必要である。今回のレビュー承認をexecution authorizationとしない。

数学的な反例、比較targetの変更、予算上限の引き上げ、別fixture、全PR費用計測へのscope拡張が必要になった場合は、科学的な判断へ戻す。単なるmissing raw-angle adapterの修正はCodexの技術作業としてまとめてよい。

## 15. 結果を受け取った後の判断規則

- **B mean/proofが不成立**：比較対照として無効。Aの科学的優位を意味しない。
- **数値認証・guard失敗**：technical INCONCLUSIVE。未完結果をpositive/negativeへ昇格しない。
- **Eの登録A実装が局所費用で不利**：その実装のscreening結果。whole方式一般の不可能性やPR総敗北ではない。
- **CDのNやenvelopeに差**：そのconfidence/bias policyによる十分予算の差。最小shot数の定理ではない。
- **全費用が記号変数のまま**：損益分岐条件を成果とし、勝敗を作らない。
- **同入力・同target精度・実装誤差・必要費用が閉じたとき**：初めてその範囲の総資源比較を判断する。
- **独立性や新規性**：今回の契約レビューでは再判定していない。技術比較のPASSを論文化GOへ自動変換しない。

現時点では、既存G10終了判断、旧source・contract・marker・STOPを保持する。

## 16. 最終要約

比較契約の骨格、Bのweighted mean、A/B別の誤差伝播、初期状態案、1/240のfamilywise配分、共通bias予算は支持する。

修正が必要なのは、(1)この計画が答えられる範囲、(2)finite-bit実装と会計の一致、(3)数値認証の境界、(4)strictなangle/phase処理、(5)部分Tデータの用途である。

**結論は「研究案を白紙に戻す」ではなく、「比較の問いを守ったまま契約を修正し、source準備へ進む」である。比較本番はまだ認可しない。**

## 参考資料

以下のrepository資料はすべて固定commit `73b2102a305f018d22348418adc7939cae63f0b2` を基準とする。外部資料はソフトウェア仕様の限定確認であり、新規性の追加調査ではない。

- [R1 比較契約v1](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/73b2102a305f018d22348418adc7939cae63f0b2/docs/tracks/algorithm_codesign/df_correction_comparison_preregistration_20261011.md)
- [R2 実行計画v1](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/73b2102a305f018d22348418adc7939cae63f0b2/docs/tracks/algorithm_codesign/df_correction_comparison_execution_plan_20261011.md)
- [R3 機械可読contract v1](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/73b2102a305f018d22348418adc7939cae63f0b2/artifacts/track_b_df_correction_comparison_preparation/2026-10-11/contract_v1.json)
- [R4 GPT handoff](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/73b2102a305f018d22348418adc7939cae63f0b2/docs/tracks/algorithm_codesign/df_correction_comparison_gpt_handoff_20261011.md)
- [R5 open items](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/73b2102a305f018d22348418adc7939cae63f0b2/artifacts/track_b_df_correction_comparison_preparation/2026-10-11/open_items.json)
- [R6 reference source manifest](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/73b2102a305f018d22348418adc7939cae63f0b2/artifacts/track_b_df_correction_comparison_preparation/2026-10-11/reference_source_manifest.json)
- [R7 spectral/roundoff numerics.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/73b2102a305f018d22348418adc7939cae63f0b2/src/trottertracks/algorithm_codesign/numerics.py)
- [R8 R1 strict numeric.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/73b2102a305f018d22348418adc7939cae63f0b2/src/trottertracks/algorithm_codesign/rte_reallocation/numeric.py)
- [R9 SP economics.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/73b2102a305f018d22348418adc7939cae63f0b2/src/trottertracks/algorithm_codesign/synthesis_placement/economics.py)
- [R10 runtime identity](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/73b2102a305f018d22348418adc7939cae63f0b2/artifacts/track_b_df_correction_comparison_preparation/2026-10-11/runtime_identity_preparation.json)
- [R11 branch確認](https://api.github.com/repos/HIROMU1015/Partially-Randomized-Trotter/branches/track-b-df-correction-comparison-preregistration-20261011)
- [W1 pygridsynth 2.0.0（公式配布・仕様）](https://pypi.org/project/pygridsynth/2.0.0/)
- [W2 mpmath 1.3.0 contexts](https://mpmath.org/doc/current/contexts.html)
- [W3 mpmath 1.3.0 basic usage / precision](https://www.mpmath.org/doc/current/basics.html)

## 付録：補助算術の記録

有理数演算で確認した事項：状態norm²=1、12/240=1/20、2(1/20000)²<(1/10000)²、最悪統計幅7/500000、32000/(4×10¹⁰)=1/1250000、64×20s=1280s>900s。

登録入力とは無関係な例として、p=(1/3,2/3)、2-bit LRM=(1/4,3/4)、cost=(0,4)なら、理想期待費用8/3と実装期待費用3は異なることを確認した。これは有限bit costを理想lawの式で代用できないことの説明例で、CD/Eの結果ではない。

補助JSON：`df_comparison_contract_review_20261011/review_arithmetic.json`。PythonのFractionを使用し、比較runner、実験コード、package import、科学計算を実行していない。
