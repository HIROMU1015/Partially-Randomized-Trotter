# B6b：Reference測定同値性と理想共通Nの条件

2026-11-08 JST。[固定PLAN](PLAN.md)。本書は許可された代数導出・SOURCE対応・人工検証の報告であり、詳細科学レビューではない。価格や新Nを計算しない。

## 一つのwhole-path gadget

時間順U0,…,Uq−1に対しV=Uq−1…U0=AB、Bは先に作用するprefix、Aは後のsuffix。system初期状態ψ、reference初期状態v（本SOURCEでは真空）、ancilla0を使う。既存gadgetはH後にCSWAP cascade、A†をsystem/Bをreference、同じCSWAP cascade、ancilla P(+χ)を適用する。XはH、YはSdg→Hでreadoutし、bit0を+1/bit1を−1とする。

```text
ancilla: H ──●──────────────●── P(+χ) ── [Y:Sdg] ─ H ─ measure
system:  ψ ─ SWAP ─ A† ─── SWAP
reference:v ─ SWAP ─ B ─── SWAP
```

A/Bはunitary、A†はactual adjoint（逆順/逆角度/逆global phase）とする。readout直前の非正規化branch isometryは

\[
K_0\psi=\frac{A^\dagger\psi\otimes Bv}{\sqrt2},\qquad
K_1\psi=\frac{e^{i\chi}B\psi\otimes A^\dagger v}{\sqrt2}.
\]

従ってcomplex coherenceの**演算子**は

\[
T_{Ref}=2K_0^\dagger K_1
=e^{i\chi}\langle Bv|A^\dagger v\rangle AB
=e^{i\chi}\overline{\langle v|V|v\rangle}\,V.
\]

係数c=e^{iχ}〈v|V†|v〉とすれば、任意ψに対してzRef=c〈ψ|V|ψ〉。混合状態ρでもzRef=c Tr(ρV)。この一般式自体は個々のA/Bのvacuum-ray保存を仮定しない。任意A/Bで無条件にzRef=zSymと主張する式ではない。

直接制御diag(I,V)はTSym=V。強いsymmetric-directionalがdiag(W0,W1)という形なら、必要なのはW0†W1=Vという位相付きcoherence関係であり、W0=Iであることは必須ではない。これを既存C3/C5 checkerがdirect-V対応として照合している。

\[
O_X=(T+T^\dagger)/2,\qquad O_Y=(T-T^\dagger)/(2i).
\]

従ってTRef=TSymなら全入力のX/Y誘導observableと±1の条件付き分布が一致する。単にfull unitaryのglobal phase同値を比較しただけの主張ではない。

## 信号一致と最終reference真空を分ける

十分条件は**whole Vのreference ray保存**、Vv=e^{iχ}v。これならc=1となり、任意system入力について同値。逆に全入力で同値を要求するなら(c−1)Vの全quadratic formが0でなければならず、unitary V≠0よりc=1。Cauchy–Schwarzの等号条件からVv=e^{iχ}vが必要となる（位相はmod2π）。

このときBv=e^{iχ}A†vだから、両branchのreferenceは同じrayへ分離する。ただしそのrayはA†vであって、必ずしも初期真空vとは限らない。

既存DF SOURCEが使う、より強い条件は

\[
Av=e^{ia}v,\quad Bv=e^{ib}v,\quad \chi=a+b\pmod{2\pi}.
\]

この条件なら最終referenceも真空へ戻り、branch0=e^{ib}A†ψ⊗v/√2、branch1=e^{i(χ−a)}Bψ⊗v/√2。既存`reference_column_check`のvacuum branch/zero-leakage検査に一致する。

**B5のhalf-ray条件はこのDF構成・最終真空復帰に対して十分である。一般の信号一致だけに必要と読むのは強過ぎる。** 旧B5・科学結果は書き換えず、本B6bでこの区別を明示する。whole Vだけがrayを保存しhalfは保存しない人工例は信号同値の代数caseとして検査するが、既存DF checkerのeligibilityを緩めない。全Vがreference rayを保存しなければ一般に|c|<1または位相が異なり、P(χ)だけで無条件に正しい信号へできない。

## SOURCEの構築規則との対応

|対象|ideal構造での確認|literal実装に残る資格|
|---|---|---|
|Gaussian basis G|number-preserving Gaussian/XXPlusYY・RZ/global phaseは真空rayを保存。G† D Gのbasis真空位相は相殺|数値分解/relative transitionが対象frameを再現する精度・global gauge|
|signed event|固定K0ではsigned Hermitian involutionのunitary rotation。Z/Z-product diagonalの真空固有値は±1で、符号を回転へ保持|有限bitのangle/phase、全supportの数値誤差包絡|
|scalar/identity|global e^{is}をVへ一回含め、χにも同じ一回を含める。制御相対位相として消去しない|SOURCEのfsum/MPFR160宣言位相とbinary64 gate適用|
|ordered列|各primitiveが真空rayを保つなら任意の合法ordered積/prefix/suffixも保つ|SOURCE/input/順序のbindingと実装全pathの誤差|
|strong baseline|uncontrolled Gaussian、directional中心、head relative fusion、tail同basis共有、合法step境界融合・scalar集約を維持し、W0†W1=Vを要求|数値Gaussian融合/全P0変換のuniform fidelity|
|Reference|actual A.inverse、B、二CSWAP、P(+χ)を既存関数のまま使用|compiled gate列/χ/adjoint/全列observableとleakage|
|normalization|pathはunitary。attenuation A_mはlaw平均を基準へ結ぶ量で、回路unitaryや価格に再乗算しない|同じideal P/implemented Qとraw誤差のdomain|

従って、**ideal数学的PR blueprintがSOURCEの正しいsigned/phase/ordered/fusion規約を実現する**という条件下では、固定DF familyの全適格path・任意system入力へ上の恒等式を適用できる。全pathを有限標本で列挙した結果ではなく、構成要素の閉包と代数に基づく条件付きの一般化である。literal floating SOURCEの全support/全round compiled誤差やPRNG lawはこの証明に含めない。

q1では同一h-step内のcomplete event後cut（Stage Aはevent16）でAB=Vを作る。q≥2のcomplete half-step cutとは価格上の境界数/融合が異なるが、一つのgadgetの代数は同じ。任意cutでの理想信号の議論を、結果依存cut選択や価格モデル移送の許可にしない。短いgadgetをq回連ねた別構成の同値性はこの証明から推定せず、whole path一つを維持する。

主なSOURCE：[gadget](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/acea43987edae0d5eb0c186a89aa8cd7139fbd7b/src/trottertracks/track_d_h4_reference.py)、[whole-path](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/acea43987edae0d5eb0c186a89aa8cd7139fbd7b/src/trottertracks/track_d_c3_whole_path.py)、[checker](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/acea43987edae0d5eb0c186a89aa8cd7139fbd7b/tests/track_d_h4_reference_checks.py)、[B5](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0a8b953aea1be20b110784960763d7f36ebccac8/docs/research/track_d_b5_actual_rpe_static_survey_20261106/REPORT.md)。

## 条件付き測定分布・path law・共通N

各軸±1の理想測定はP(+1|ω)=(1+μ_b(ω))/2。全適格pathでOX/OYが同じなら、同じψ・同じωの条件付き分布が同じ。任意の共通whole-path marginal law P_mまたはQ_mで積分しても、方式間の一測定の周辺lawは同じ。

同じ**joint path law**と、各shotでfresh system/reference/ancillaおよび同じ条件付きBorn試行を用いるなら、依存のあるpath列を含め方式間の測定transcript lawも等しくできる。一方、一測定のmarginal一致だけでjoint-law一致やdecoder全成功分布を推定しない。

B4-2の十分Nを適用するには別に、ideal finite-RTE product law（event/occurrenceの独立性）、各round・axis内のfresh独立±1試行、同じraw信号/元H ground/attenuation/誤差・prior/decoder/energy/failure条件が必要。この理想モデルの下では、Referenceとstrongに**同じround/axis別N**を使える。全roundへ同じ整数を使う意味ではない。26軸のunion bound自体はround/axis間の独立を要求しない。方式間で同じpathをpairedに共有しても、各軸内のconcentration前提とlawの対応が成立すればよい。

B4-2の保存合計14,738はこの条件付きHoeffding十分値の引用で、新Nや最小Nの算定ではない。同じfinite-bit Qを使って方式間の分布が一致しても、QとB4-2のideal Pとの差やshot列のconcentrationが未取得なら、この数値の元H成功保証への転用は未資格。seed非重複/固定cohort巡回/ヒストグラム一致をIID証明としない。

compiled誤差等が方式別に異なる場合、両方式を同じ保守的なraw半径/phase/coordinate予算へ含められ、保存整数Nのceil余裕まで資格付けできれば共通Nの候補になる。さもなければ方式別Nを同じtarget/failure条件で評価する必要があるが、今回はどちらも再算定しない。actual誤差未取得を理想モデルの資源会計の絶対的な開始禁止条件とはせず、明示したideal/conditional laneとactual保証を分けて引き渡す。
