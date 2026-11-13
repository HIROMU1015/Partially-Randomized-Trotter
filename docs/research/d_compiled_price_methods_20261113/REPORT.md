# 全13roundのcompiled RZ価格：評価方法と成立条件

2026-11-13 JST。**推奨は、保存26 E[S]の再利用と、target Rを保った短いouter-windowの
connected残差補正を組み合わせる方法の検証である。** 短windowから長qへのcompiler影響の
上界が必要で、現在その上界はない。したがってm12のE[C]はUNKNOWNのままとする。
N_ZZ比例式、既存P1/P2へのfit、未検証の有限局所性は採用しない。

[C向け特徴量・HANDOFF](C_HANDOFF.md)／[最小科学検証案](MINIMAL_SCIENCE_PLAN.md)／
[SOURCE](SOURCES.md)／[人工29 PASS](TESTS.txt)／[残差回帰13 PASS](TEST_RESIDUAL_REGRESSION.txt)。

## 1. 範囲・既存成果との差分

固定対象はLinear H4/1.00 Å/STO-3G・保存tol-only DF10、S2/L_D1/ΔPF=.1/K2/h=.1/M12/P0、
元exactκ/H/state/snapshot、symmetric-directional full Hadamard X/Y。
新lowering `CONTROLLED_ZZ_CX_CRZ_CX_V1`、Qiskit1.3.0/rz,sx,x,cx/opt1/seed17を維持する。

| 固定SOURCE | 確認・再利用した内容 |
|---|---|
| `39ddd59edf54a3a901e5302207b2765ddbd03fbb` | P2 REPORT/HANDOFFの既存記述、専用runnerとfresh-only adapter。数値JSON/CSV/rawの新解析はしていない |
| `5a67cea7705e80ff295306606611d774724ae2e2` | compiled_price_fast、S共通prefix・RZ projection、既存fast canonical writer、人工テスト |
| `54b64ef72bf507be25970f2d3362d3be091321c3` | P1 REPORT、実行SOURCE20fe1194とpinned C1 loader。旧controlled-ZZ係数へ戻さない |
| `9d95d3ea544e1fa1d0e4710c1ca24d8c5776469f` | Cのh4_structural_kernel/Model/Wire、schedule adapter、MeanKernel、全26 E[S]の再利用条件 |
| C1 `1a87e533`、carrier `155529e2`、provider `c31ef424` | 同pathのAccounting S、tail/PF compose、scalar一回、wrapperとcompiler設定 |
| 既存Track D emitter/1q/CX stream bridge | 公開ZSXX合成・frontier・cache再利用。別CDF8系列の限定資格と最悪O(L²)/O(L)の制約も確認 |

旧D残差設計にはcompiler文脈依存、局所cluster、直接paired法とladder案が既にある。
今回の追加は、**outer単位の厳密なconnected恒等式とtail条件**、C既存状態に沿う具体的な
特徴量仕様、長時間domainを直接問う有限案である。旧n=8 ladderや順次m7→m8→m9は実行前提にしない。
既存stream bridgeやcompiler全体を再実装しない。

## 2. Dの発生構造：SOURCEから分かること

### Z/ZZ rotation

controlled ZはCRZ、controlled ZZはCX·CRZ·CXとなり、元Accountingのcrz/crzz展開と一致する。
この宣言native展開では両方とも中央rotation由来RZは2個で、ZZの外側CXが境界文脈を変える。
旧generic controlled RZZのCCX経由の固定差は、新loweringに重ねて加算しない。
同じZ/ZZ数でも、support・符号・角度・前後のbasis/1q runが違えばcompile結果は変わり得る。
P2 REPORTには、m0の全4 pathでN_ZZ=2でもDが異なったという既存観測がある。
この記述を引用するだけで、今回新しい相関・係数を計算していない。

### basis変換と相殺

Sは同basisの共有、literalに一致するinverse basis対、wire別pending-Zの順次binary64和を扱う。
Qiskitは標準gateへのtranslationと、各wireの1q行列積の再合成を行う。
Accountingのbasis比較は元operations表現の比較であり、basis ID一致だけと同義ではない。
headのlist表現とtailのtuple/asdict表現・matrix_sha256等を正規化してSの意味を変えない。

### head/tail・outer境界・wrapper

headのforward→tail→reverse-directional headと、隣のouterとの境界が1q run/CX相殺に関わる。
Cの「outer出口が固定」は**Sのpending-Z/遅延basis状態**の証明であり、compilerのCX frontierや
1q行列の全状態がリセットされる証明ではない。
scalar p(−scalar T)は全qの後に一回、YはSdg、両軸ともHと測定を持つ。
境界だけを別compileした価格を無条件に加算しない。X/Yは個別に保持する。

### 固定compilerの文脈依存性

実際の設定でpass managerだけを生成し、回路をrunせず構成を確認した。
initのInverseCancellation、translationのUnitarySynthesis/HighLevelSynthesis/BasisTranslator、
optimizationのOptimize1qGatesDecomposition→InverseCancellationとdepth/size固定点loopがある。
layout/routingはなく、1q decomposerはtarget=None・ZSXX。
opt1ではCommutativeCancellationや測定前diagonal除去を前提にしない。

Qiskit1.3のRust SOURCEではtargetなしの1q再合成採否はgate数を基準にし、RZ数を目的にしない。
角度簡約には数値toleranceも使う。Sのliteral zeroとは同じではない。
**人工1q例では7 gate/2 RZが5 gate/3 RZになった。** これは小さい技術反例で、H4価格の観測ではない。

別の人工例ではRZ(a)→長いCX列→その逆列→RZ(−a)を使い、Sの2 RZに対しopt1出力は0だった。
CXの消去が離れた1q runを接続する。長さ2/8/32の人工例がPASSした。
SOURCE上は同一qargsの偶数CX runを一度に消去でき、任意長の同種runで両端の1qを接続し得る。
混在mirrorのtestは有限長だけで、固定点loopのmax_iterationも維持する。
したがってgate局所性だけから固定haloの十分性を結論しない。
H4の合法lawでこの反例がどの確率で起きるかは評価していない。

## 3. 四つの方法の比較

Lは全native gate数、sは閉じた状態数、hはwindow outer数、Jは特徴量数とする。

| 方法 | 成立条件・必要な実測 | 負荷 | 誤差と長時間の限界 |
|---|---|---|---|
| ①Dを構造から直接計数 | compilerと同じtranslation・数値簡約・全fixed-point規則、または同値性証明。既存native bridgeは部品として再利用可 | path streamは少なくともO(L)。既存部分bridgeは最悪O(L²)時間/O(L)記憶。閉包が証明できればkernel化可能 | 現bridgeは一般P0未資格。Sのwire閉包だけでは不足。m12に現在の厳密D counterはない |
| ②解析的・局所補正 | 固定sourceの局所単位とcapを定義し、connected tail上界を持つ。少数windowの実測が必要 | window build/compileはO(hR)入力規模。期待値代数はqにO(1)、kernel rewardならO(s³ log R)等 | 単位価格の和は相殺を二重計上。qで増幅されるtailを抑えられない点はUNKNOWN |
| ③少数構造量モデル | Cの計算しやすいfeatureとdomain、独立校正/holdout、同定可能性・bias評価 | feature期待値はqR列不要。固定有限kernelで概ねO(Js³ log R)、単純countは更に軽い | 平均featureだけでは非線形compiler効果を決められない。跨R/小角・長q/境界/rareのmodel誤差が必要 |
| ④直接paired sampling | 同path/axisのfull-wrapper S/C、law条件と十分な有限標本誤差評価 | n path×両軸compile、O(nqR)以上のword処理 | domain内の最も直接な参照。小qからm12の母平均を認定しない。P2標本を今回fit/再解析しない |

推奨は②を主候補、④を可能な範囲の参照、③を必要時の限定補助とする。
①の新しいfull compiler再実装は費用対効果が低く、既存部分bridgeを新P0 oracleとして扱わない。

## 4. 推奨案：target Rでのouter-window connected残差

各target mを固定する。元回路の1 outerをV_iとし、その内部のR_m個のevent・角度・head順を維持する。
F(i:j)はV_i…V_jだけを、**target mと同じwrapper/測定とscalar p(−scalar T_m)一回**で囲った
局所回路のC−Sとする。F_emptyも同じcapsで定義する。局所量は本来のphysical候補価格recordとは区別する。

単一outerではΔ(i:i)=F(i:i)−F_empty、長さ2以上では

\[
\Delta(i:j)=F(i:j)-F(i+1:j)-F(i:j-1)+F(i+1:j-1).
\]

これは任意のcost functionalに対する二重差分であり、pathごとに

\[
D_q=F_{empty}+\sum_{1\le i\le j\le q}\Delta(i:j)
\]

が厳密に成り立つ。**この恒等式自体は局所性を証明しない。**
同じR・capsでwindow lawが長さに関して定常な場合に限り、μ_n=E[F_n]として
δ_1=μ_1−μ_0、δ_k=μ_k−2μ_(k−1)+μ_(k−2)で

\[
E[D_q]=\mu_0+\sum_{k=1}^q(q-k+1)\delta_k.
\]

hで打ち切ると

\[
d_q^{(h)}=(q-h+1)\mu_h-(q-h)\mu_{h-1},\qquad
|E[D_q]-d_q^{(h)}|\le\sum_{k=h+1}^q(q-k+1)|\delta_k|.
\]

この形は同じRでの有限差分であり、N_ZZ係数や全round共通のqR比例fitではない。
h=2でも境界capsは重みの和1により一回だけ残る。全体scalarを各outerに分配しない。
μ_hとμ_(h−1)は同じdrawのprefixでcoupleし、標本共分散を保持する。
重みa=q−h+1、b=−(q−h)に対し名目標本平均分散は(a²s_h²+b²s_(h−1)²+2ab s_cross)/n。
X/Yも同じpathの対応応答であり独立標本数を増やさない。

小回路化だけで必要総計算量が減ったとは認定しない。重みはqとともに大きくなり、
couplingしてもstencil分散v_Zが大きければ必要nが増える。
同じMC誤差予算b_MCでの名目仕事量はv_Z×t_Z/b_MC²（t_Zは両prefix・必要軸を含む1 drawのwall）で、
直接paired参照のv_D×t_D/b_MC²と比較する。局所案はtail誤差予算も消費するため、
それを含めて安い場合だけ採用する。今回これらの実H4分散・wallを新しく推定していない。

### m12への意味

既存scheduleのm12はq4096/R8160。全trajectoryは33,423,360 event位置である。
h≤3なら局所回路は最大24,480 event位置で、targetの小角・R・内部basis連鎖を直接含められる。
これは入力規模の比較であり、局所compileのwall/RSSを保証しない。
全13roundをこの方法で扱う場合の列規模はO(h ΣR_m)で、全pathのO(Σq_m R_m)とは異なる。
跨R補間をしなければ各Rのlocal値が要るが、各roundの巨大full q回路は不要になる可能性がある。

ただし短windowでδ_3等が小さくても、残りの全δ_kを0とはできない。
固定halo内でcompiler依存が必ず閉じる証明、または合法lawで影響spanのtailを抑える証拠が必要である。
例えば全k>hについて|δ_k|≤Aρ^(k−h−1)が成立するなら、右辺は有限幾何和で評価できる。
今回のhelperは**与えられた上界の和を計算するだけ**で、A/ρを実データからfitせず、premiseも認定しない。
Sの固定outer出口、短window一致、任意長CX反例の有限不出現は、その証拠の代わりにならない。

## 5. m12までcompileなしに扱える範囲

| 量 | 現在の扱い |
|---|---|
| 全26 E[S]と元数値enclosure | 固定C成果をそのまま再利用。再評価不要 |
| q/R/T/境界数 | 元schedule metadataから定義可能 |
| Z/ZZ/order/product/basis-transitionの期待count | 同じlawの確率を用いる式またはC既存stateのrewardでqR列なしに評価可能。**新しい数値評価は今回しない** |
| Sのlocal/terminal reward、literal fold・basis境界の派生feature | 既存CのModel/Wireを利用できる候補。新rewardのparity/数値enclosureは別作業 |
| local connected係数 | 未取得。新しい科学測定が必要 |
| 全qのcompiler influence tail、跨R model error | UNKNOWN。Cのkernelをcompiler閉包へ自動昇格しない |
| 全13 E[C] | 今回未推定。上記の残差・law・cohort/rare・tail誤差を充足した点だけ将来評価できる |

## 6. 精度と最小検証

目標はD自体の完全説明ではなく、E[S]+E[D]の価格誤差である。旧P2の2%/95%は候補のまま。
使用点ごとにS数値幅、local測定誤差、window tail、law差、rare/tail、跨R model誤差を分けて配分する。
未知項を0にしない。統計誤差をn増加で減らしてもlaw biasやwindow tailは消えない。

CのE[S]がideal law、paired Dが実sampler lawである場合、targetをidealとすれば
E_ideal[C]=E_ideal[S]+E_sampler[D]+(E_ideal[D]−E_sampler[D])である。
actual samplerのE[C]をtargetにするなら、最後のlaw補正はE_sampler[S]−E_ideal[S]となる。
どちらを評価するか先に固定し、異なるlawの期待値を無補正で接続しない。
finite cohortの標準誤差はこのlaw差ではない。rareを含む母分散の妥当なCIが既にあれば
rare標本誤差を重複計上せず、未観測rareに対する保証がなければ別の未知項として残す。

最初の有限案は、既存m6の保存2 pathを用いた短window診断と、m12 lawでの短window診断を
各12 compile、合計最大24 local-wrapper compileとする案。詳細は[最小計画](MINIMAL_SCIENCE_PLAN.md)。
既存m6比較は事後development診断であり、独立holdoutと表示しない。
m12のRを保つだけではq4096の検証にならないため、tail条件が満たせない場合はそこでSTOP/UNKNOWN。
解析的閉包等が得られないまま認定を要求するなら、最大使用qの適格referenceが別途必要になる。
既存native bridgeの未資格出力をreferenceの代用にしない。

## 7. 技術検証・保全・次の担当

追加helperは`outer_window_residual_design.py`のみ。既存compiler/emitter/kernelは変更しない。
二重差分恒等式、collapsed stencil、共分散、条件付きtail和、既存MeanKernel rewardを人工値で検証。
Qiskitの小さい人工反例を含む新14＋既存mean15＋既存残差13＝42 PASS。
作者自身の技術確認であり、独立科学レビューや実H4検証ではない。

新実H4 sampling/build/compile、保存P1/P2数値のfit/再解析、新E[C]/G、8次PFは0。
旧SOURCE/input/raw/KEEP/STOPを変更していない。

- **ユーザー**：必要精度とtarget lawを選び、必要なら最小科学案の対象・SOURCE・有限予算を別途承認する。
- **Codex D**：今回の設計・人工結果をcommit/push・remote SHA/tree/主要成果物照合後、mandatory STOP。
- **Codex C**：現在は実行不要。[HANDOFF](C_HANDOFF.md)の定義を参照し、将来明示されたfeature/rewardだけを別SOURCEで評価する。
- **Codex B**：構造S、local residual、full-wrapper meanのquantityと未知誤差を分け、未資格価格を供給しない。
