# Track B：DF少数軌道・希薄三次補正付きPRの独立監査後・科学的レビュー

- 作成日：2026-10-11 JST
- 開始承認：利用者の「レビューを開始して」
- 対象repository：`HIROMU1015/Partially-Randomized-Trotter`
- 固定commit：`8a5e21c8f6ecf98ee285ec1fad49079ffb5ed97f`
- branch：`track-b-df-sparse-correction-independent-audit-20261011`
- 旧研究参照：`22a0d032717b4139cdfa806850643e93ba25624f`
- 位置付け：GPTの科学的判断。Codexの監査報告とは別。既存source・結果・契約・STOPの変更、production実行、分子入力の生成、Clifford+T合成は行わない。

## 0. 結論

**理想数学の成立は支持する。しかし、現候補を独立した新規PRアルゴリズムとして採択し、そのままproduction実装を完成させる段階には進めない。継続する作業を「同一のDF局所補正対象に対する、whole-commutator回転と具体的な局所Pauli対照の比較」に絞る。**

その理由は次の通りである。

1. ordered partial-S2の三次BCH係数、左右補正、理想samplerの一次平均、P5以上での四次精度は、独立導出・source照合と整合する。数学的反例が見つかったための終了判断ではない。
2. 三次corrector、係数比例のランダム回転、逆確率の角度拡大、identityによる希薄化は、既知法との具体的な対応を持つ。「正の確率・追加weight 1・稀な補正」を主新規性にはできない。
3. r≤6という支持の縮約は有用な構成材料だが、同じ局所表現を利用するPauli補正も全系Pauli表なしで構成できる。全系Pauli展開を強制された弱い対照だけで比較してはいけない。
4. 保存されたCX5,299／RZ27,153は一つの汎用分解であり、最小費用・下界ではない。一方、その費用を支払う合理的な機構もまだ実証されていない。
5. 今回の追加解析では、正の希薄samplerが一般に五次の収縮方向の項を持つこと、安価なPauli回転と追加weightの交換を行う具体的対照、元RTEのnormalizationによるstep幅制約を明示した。これらを同じ誤差とconfidenceで比較する情報価値はある。

研究の中心候補は、**「希薄補正という一般原理」ではなく、「DF局所表現を使うと、既知補正法にも同じ情報を与えた比較で何が改善するか」**へ限定する。これは全面的な新テーマ再設計でも、現候補のscientific GOでもない。

次の実務担当はCodex。既存A–Dを丸ごと再検証するのではなく、今回具体化した対照と同じ局所入力・誤差会計の比較を一つの限定作業として準備・検証する。production DF importer、汎用compiler開発、全parameter探索、分子実行を先に完備しない。

## 1. 証拠区分と今回の確認範囲

### 1.1 使用した既存証拠

固定commitから、独立監査報告、先行研究比較、A/CD/Eを実装するsource、CD/Eの保存値、manifest、原設計のnormalization節を取得した。branch HEADが固定commitと一致することも確認した。[R1–R8]

原設計のローカル添付は41,234 bytes、SHA256 `e6cec20eb2e4abe4f00d0e62fd05d28180132888857cd110edb368b2d4ba1c03`。原ZIPは34,941 bytes、SHA256 `75851dff8fb1f7851ad48b0dcd31853cb650bde148b59d54825b798193dca29b`。今回のローカルhashはmanifestに記録された値と一致した。これは全31資料を今回独立に再取得してbyte照合した、という意味ではない。

Codexの77 checksと15 focused testsは保存された監査結果であり、今回GPTがその全てを再実行したものではない。全Fock参照は小さいsynthetic条件、compilerは限定5呼出しである。

### 1.2 今回新たに行ったこと

- 既存のK3・sampler・有限時間boundの論理を読み直した。
- qDRIFTとの代数的対応を確認し、CPF、NCC、SparSto、HNCCの関連一次本文を参照した。
- 希薄samplerの五次項、RTE normalizationの小step展開、局所Pauliのcoin-first対照、rank-one primitiveのGaussian閉包を導出した。
- `review_checks.py`で18件の限定自己検算を行った。有理的な記号展開と小行列照合、公開native sourceの乱数生成手順による二つのfixtureの再構成を含む。

この18件は新しい分子実験や登録済み科学runではない。Codexの77件と合計して新しい「全監査PASS件数」にしない。局所Pauli normは浮動小数の診断値であり、運用用のcertified boundではない。

### 1.3 外部文献の確認限界

Randomized THRIFTを含むarXiv:2608.07720は、著者・題名・abstract相当の一次検索結果を確認できたが、今回のweb接続ではPDF／HTML本文を再取得できなかった。Codexが記載したSM S63–S96、S91–S92等の具体的対応は、今回の独立全文確認ではなく、固定repository内の先行研究監査に依拠する。[R2, P5]

NCC、SparStoのPDF本文は抽出できたが、当該ページのscreenshotは取得に失敗した。qDRIFTは本文とp.2の画像を確認した。NCCの式(111)等は抽出本文から読んでおり、図表の新しい数値を採用していない。

この制約の下でも、qDRIFT型の対応と本候補の数式は独立に追跡できる。ただし、全先行研究に対する優先性確定・完全同値性を宣言しない。

## 2. 監査の到達点を科学的主張へ翻訳する

| 項目 | 支持されること | 支持されないこと |
|---|---|---|
| A | 固定ordered-S2の三次係数、符号、理想correctorによる相殺 | 新しいBCH理論、任意に並べ替えた回路への自動適用 |
| B | number-conserving Gaussian基底の高々2軌道primitiveならr≤6へ閉じる | 任意registryがGaussian、数値rank切捨てが無誤差、production access完成 |
| C | 正時間・理想確率下の平均と有限step上界 | finite-bit samplerが実装済み、負時間対応が検証済み |
| D | 固定入力・固定γ・独立draw・P5以上でlocal O(δ⁵)、global O(tδ⁴) | 各sampleが四次PF、一般channel保証、energy精度の保証 |
| E | 指定synthetic回転を汎用分解で構築できる | 最小CX/T費用、分子上の平均費用、T合成済み |
| F | 具体的な既知部分と差候補が整理された | 独立新規性、全protocol同値性、資源優位の確定 |

条件付きPASSをFAILに読み替えない。同時に、未実装項目を単に埋めれば新規性が増えるわけでもない。[R1–R6]

## 3. ordered-S2の数学は保持する

H=Σ_{j=0}^{L-1}H_j+R、H_L=R、B_j=Σ_{ℓ>j}H_ℓとする。実際のordered half-sweepに対応する理想中心の対称積は

\[
S_2(\delta)=\prod_{j=0}^{L-1}e^{-i\delta H_j/2}\;e^{-i\delta R}\;\prod_{j=L-1}^{0}e^{-i\delta H_j/2}.
\]

\[
\log S_2(\delta)=-i\delta H-i\delta^3K_3+O(\delta^5),
\]
\[
K_3=\sum_j\left\{\frac{[H_j,[H_j,B_j]]}{24}+\frac{[B_j,[H_j,B_j]]}{12}\right\}.
\]

この式は二成分のsymmetric BCHから内側→外側の帰納で得る。D全体をexactに発展できるという仮定は不要。R=0でもD内部の非可換性に由来するK3は残る。

C=exp(+iδ³K3/2)を左右に置くと、三次項を相殺し、理想的なCSCはlocal O(δ⁵)になる。これはsymmetric correctorの既知原理に対応する。[R1,R5,P1]

中央の有限RTEがP3なら−δ⁴R⁴/24が残る。cutoff4はP5に対応し、本候補が主張する次数にはP5以上が必要である。P3からP5への費用増を隠さない。[R1,R5]

## 4. 最大6軌道という結果の意味

Qa=σa∏_{v∈Ua}(I−2n_v)、|Ua|≤2のとき、Tτ=[Qa,[Qb,Qc]]は三primitiveの軌道列のspan Sτ上のeven CAR代数に入る。dim Sτ≤6。異なるprimitive間の軌道は直交していなくてもよい。

そのspanを先頭r個のmodeへ移すnumber-conserving Gaussian Gτで、Tτを局所行列へ縮約できる。little-endianかどうかによりtensor表記の左右は変わるが、spectatorにidentityを残す演算子恒等式である。

これは電子数≤6や状態の局所化を仮定しない。global qubit k-localと同じ概念でもない。

有用性は、全分子Fock行列ではなく最大64×64の局所行列を作れる点にある。しかし同じ局所行列はPauli分解、既知corrector、sector別合成にも利用できる。**局所表現の価値とwhole exp(Tτ)の価値は分ける。**

productionの軌道列・基底作用・rank処理は未完成だが、これは実装課題であってアクセス不可能性の証明ではない。sourceにはsquare-U経路とsyntheticなGaussian復元の証拠がある。interval rankなどを完成させる前に、どのbackendを採用する価値があるかを判別すべきである。[R1,R4]

## 5. 希薄samplerのqDRIFT型還元

既存規約を保持する。

\[
K_3=\sum_\tau c_\tau T_\tau,\quad c_\tau\ge0,\quad A=\sum_\tau c_\tau,\quad \mathcal B=4A,
\]
\[
\|T_\tau\|\le\kappa_\tau\le4,\quad
\Lambda_3=\sum_\tau c_\tau\kappa_\tau,
\quad \eta=\min(1,\gamma\delta),\quad \theta=\frac{\mathcal B\delta^3}{2\eta}.
\]

非zero項について
\[
\bar C=(1-\eta_{eff})I+\eta_{eff}\sum_\tau q_\tau'
  e^{i(\Lambda_3\delta^3/(2\eta_{eff}))H_\tau'},
\]
\[
q_\tau'=c_\tau\kappa_\tau/\Lambda_3,\quad
H_\tau'=T_\tau/\kappa_\tau,\quad
\eta_{eff}=\eta\Lambda_3/\mathcal B.
\]

これはone-drawの係数比例回転をBernoulliで間引いて角度を拡大した分布である。qDRIFTのchannel保証をそのままoperator平均へ引用せず、この対応は分布の代数として確認する。SparStoにもinverse-probabilityの生成子とidentity化がある。[R2,P2,P3]

したがって、希薄化の基本原理を独立新規性の中心に置かない。一方、qDRIFTに形式的に対応するだけでは、DF局所回転を与える入力処理・native実装まで既知ということにはならない。

## 6. 今回の追加解析：weight 1で残る五次項

以下は既存監査の再引用ではなく、固定samplerから今回導出した解釈である。正時間、固定γ、unsaturated η=γδを考え、κ=0項はT=0として省く。

Taylor展開で
\[
\boxed{\bar C=I+\frac{i\delta^3K_3}{2}-\delta^5D_5+O(\delta^7)},
\]
\[
\boxed{D_5=\frac{\mathcal B}{8\gamma}\sum_\tau\frac{c_\tau T_\tau^2}{\kappa_\tau}\succeq0}.
\]

確率がδ、回転角がδ²なので、一次がδ³、二次がδ⁵となる。一方、ideal correctorは
\[
e^{i\delta^3K_3/2}=I+i\delta^3K_3/2-\delta^6K_3^2/8+O(\delta^9).
\]

従って両者の差は−δ⁵D5+δ⁶K3²/8+O(δ⁷)。元のO(δ⁵)保証に反するものではなく、そのleading部分の意味を明確にした。

さらに、unitary ensembleの一般恒等式
\[
I-\bar C^\dagger\bar C
=\mathbb E[(V-\bar C)^\dagger(V-\bar C)]\succeq0
\]
から、確率化は一次平均の非unitary性を生む。今回のleading defectは収縮方向であり、単なるcoherent phaseの五次係数ではない。左右の補正を入れたstepの五次項には−2D5が現れる。ただし、任意のρと時刻で観測信号の絶対値が単調減少するという別主張はしていない。

K3の項間相殺が強くてもΣcT²/κが小さいとは限らない。例としてTと−Tを等確率で選ぶとK3=0だが、平均は(1−η)I+ηcos(uT)であり、一般にIではない。この例はsamplerの性質の診断であって、実際の分子でそうなるとの観測ではない。

**意味：追加normalizationが1なのは、補正が無料だからではない。代価の一部をbiasへ回している。** 新規性・資源比較では、そのbiasとweightの交換を同じ物理taskで評価する必要がある。

## 7. 「一定の期待補正回数」の限定

既存samplerで、2n slotsの要求数Jは独立な場合
\[
J\sim\mathrm{Binomial}(2n,\eta\Lambda_3/\mathcal B),
\qquad \mathbb E[J]=2n\eta\Lambda_3/\mathcal B.
\]

n=t/δ、η=γδならE[J]=2tγΛ3/B。固定t・γでstep細分化に比例して増えない、という性質は正しい。

しかし、γを小さくすればD5が1/γで大きくなる。γを精度とともに変更する戦略では「γ固定」の定理を無条件に流用できない。Jのhard capは2nであり、期待値が小さいことは固定depthを保証しない。

またκ=4はzero Tの要求も含み得る。要求数と、非identity回転を実際に出力する数は異なる。閾値でzeroとみなして落とす場合には別のbiasが生じる。

## 8. 既存native値をどう読むか

Phase Eはall-to-all論理接続、basis `rz,sx,x,cx`、optimization_level=0、Qiskit1.3.0、seed20261011である。r=3、r=6に対してcontrol有無を一度ずつ、追加で小さいGaussian embeddingを一度compileした。[R3,R4]

| 中央回転 | CX | RZ | depth |
|---|---:|---:|---:|
| r=3 uncontrolled | 20 | 120 | 125 |
| r=3 controlled | 90 | 453 | 549 |
| r=6 uncontrolled | 1,868 | 11,256 | 13,981 |
| r=6 controlled | 5,299 | 27,153 | 35,310 |

r=6 controlled値は中央unitaryの値であり、分子全体のGaussian embedding、状態準備、元PR/RTEの費用を含めていない。別の4-mode embedding例の399CXをこのr=6例へ加算しない。CDの期待要求数2.5とr=6の費用を掛けて「保存済み総費用」にすることもしない。異なるfixtureだからである。

T合成は0件。RZ countをT countへ読み替えない。角度と許容誤差、strict controlled phaseに基づく実際のClifford+T合成が必要。

このcompileは存在する実装経路の測定であり、最小費用・下界ではない。particle-number blocks、外側involutionとの反交換、Gaussian特殊化を使っていない。これを理由に新方式全体を数理的に不可能と結論しない。

同時に、6modeだから軽いという期待も支持されない。現時点の中心的な懸念は「このwhole回転を使う必要があるのか」である。

## 9. 安い構造が既に分かる特殊ケース

全三primitiveがrank-one parity
\[
Q_a=I-2\,d\Gamma(P_a),\qquad P_a=v_av_a^\dagger
\]
なら、二次fermion演算子の交換子は
\[
[d\Gamma(X),d\Gamma(Y)]=d\Gamma([X,Y])
\]
を満たすので
\[
\boxed{[Q_a,[Q_b,Q_c]]=-8\,d\Gamma([P_a,[P_b,P_c]])}.
\]

符号σはその積を掛ければよい。この場合、中心のexponentialはGaussianである。これは今回の構成から直ちに得られる特殊化であり、一般のrank-two parity三重項にGaussian性を延長しない。

Phase Eの`full_embedding` fixtureは三つともrank-oneの列から作られている。[R4] したがって、その汎用unitary compile値は、構造を使う代表的最良値ではない。今回の4mode合成検査でquadratic閉包の残差は約3.09e−16だった。

ただし、この安い種類が実際の係数分布の何割を占めるかは未評価である。安いケースだけを選び、一般DFの結果と称することはしない。

また任意のinvolution Qaについて{Qa,[Qa,X]}=0であるため、Tτは外側Qaと反交換する。この構造も汎用compileは利用していない。これだけで安価な全合成が完成したとはしない。

## 10. 具体的な強い対照：局所Pauli pairingをcoin-firstで使う

ここが次の比較で最も重要である。r≤6の局所表現をwhole回転だけの専有条件にしない。

以下では監査で実際に使ったκτ=4の場合を考える。元samplerは、outer coin確率η、tuple qτ=cτ/A、受理時unitary
\[
V_\tau=e^{iuT_\tau},\qquad u=\frac{\delta^3A}{2\eta}
\]
である。

同じ局所行列を
\[
T_\tau^{(r)}=\sum_j\beta_{\tau j}P_j,
\qquad \ell_\tau=\sum_j|\beta_{\tau j}|
\]
とPauli展開する。最大4^6=4,096係数であり、global n-qubitのPauli表ではない。T=0ならidentityとする。局所行列の生成とPauli係数計算の古典費用は課金する。

### 10.1 実行手順

1. 元と同じ確率ηでouter coinを引く。失敗時はidentity、測定weight1。
2. 成功時だけtuple τをcτ/Aで選び、局所Tとβを構成する。
3. jを|βτj|/ℓτで選ぶ。
4. ατ=arctan(uℓτ)として、局所unitary exp(i sgn(βτj) ατ Pj)を実行する。
5. この受理branchだけ、測定weight wτ=√(1+(uℓτ)²)を掛ける。

Gaussian embeddingは元のwhole回転と同じものを利用する。この手順はNCC等のidentity–Pauli pairingの既知恒等式を、現在の有限局所入力とouter coinへ具体化したものとして提案する。[P4]

### 10.2 平均

Pauli恒等式から、tuple条件付きのweighted meanは
\[
w_\tau\sum_j\frac{|\beta_{\tau j}|}{\ell_\tau}
 e^{i\operatorname{sgn}(\beta_{\tau j})\alpha_\tau P_j}
 =I+iuT_\tau.
\]

従って半補正全体は
\[
\boxed{\mathbb E[WV]=I+\frac{i\delta^3K_3}{2}}.
\]

ideal exp(iδ³K3/2)との差はδ⁶||K3||²/8以下。この半補正だけを見れば、元のpositive samplerが持つδ⁵D5項はない。ただしweightは1より大きくなる。両者は有限stepで同じ平均ではないため、それぞれのbiasを戻して比較する。

### 10.3 第二moment、range、頻度

outer coinを先に引くので、局所行列を構成する期待回数と補正回転数は2nη=2tγ。identityの大多数についてTを先に作る必要はない。

\[
\boxed{\mathbb E[W_{half}^2]
=1+\eta u^2\mathbb E_{q}[\ell_\tau^2]
=1+\frac{\delta^5 A^2}{4\gamma}\mathbb E_q[\ell_\tau^2]}
\]
（unsaturated領域）。独立2n slotsでは
\[
\mathbb E[W_{corr}^2]
=\left(1+\frac{\delta^5 A^2}{4\gamma}\mathbb E_q[\ell_\tau^2]\right)^{2n}.
\]

小δでそのlogはtδ⁴ A² Eℓ²/(2γ)+高次。これは無料ではないが、必ず巨大になるとも言えない。

有限branchでのrange上界は、ℓmaxの証明済み上界から
\[
W_{corr,max}\le(1+u^2\ell_{max}^2)^n
\]
を得られる。Pauli係数のCauchy–Schwarzよりℓτ≤2^r||Tτ||≤256がr≤6の粗い普遍上界。これはglobal tableを要求しないが、保守性がある。浮動小数の診断normをcertified ℓmaxとして採用しない。

### 10.4 回路単価

weight wのPauli stringのcontrolled回転は、基底Clifford、parity計算、controlled RZ、uncomputeで構成できる。論理all-to-all、追加workspaceなしの通常のparity ladderならCX≤2w≤12、中心の任意角RZは2個でよい（identity stringはancilla phase）。Gaussian embeddingとClifford、合成精度・相対位相は別に課金する。

**CX5,299とCX≤12の比を、そのまま改善率にしてはいけない。** 前者はwhole exp(iuT)、後者はweighted平均I+iuTを作る一つのPauli回転で、weight・bias・抽出対象が異なる。必要なのは同じhalf-correction targetおよび最終complex-signal精度を満たす総費用比較である。

### 10.5 この対照が意味すること

「全系Pauli表が要るから既知NCCは使えない」は、現在の局所表現を両者へ与える比較では成立しない。また「稀にしか実行しないこと」をwhole回転だけの利点にできない。

ただし、この対照の数学を既知原理から構成できたことは、production実装・finite-bit sampler・Clifford+T優位まで確定したことではない。次の独立確認に渡す明示的な比較仕様であって、新しい実用的GOではない。

## 11. 保存sourceのfixtureに対する追加診断

Phase E sourceのseedと乱数生成手順を再構成し、同じ仕様のr=3／r=6中央行列について局所Pauli係数を計算した。既存のコンパイルは再実行していない。元のtranspiled fingerprintと独立に一致させたという主張もしない。

| 診断量 | r=3 | r=6 |
|---|---:|---:|
| ||T||（float診断） | 3.3494590732 | 2.9673431114 |
| ℓτ（float診断） | 7.8248389720 | 29.9271086938 |
| Pauli再構成のoperator残差 | 4.03e−16 | 7.37e−16 |
| 1e−12超の係数数（診断のみ、削除なし） | 15 | 990 |
| 保存角度θ=1/32、κ=4に対応するu | 1/128 | 1/128 |
| 受理branchのweight √(1+(uℓ)²) | 1.0018667910 | 1.0269688549 |

r=6の1.02697は一つの受理branchのweightであって、trajectory全体やshot数の倍率ではない。頻度・step数・second momentとrange・元RTEΓが必要になる。

この診断からwhole回転の敗北は確定しないが、**数千CXのwhole回転を正当化する前に、具体的で安いPauli対照を調べる価値が高い**と判断する。全系Pauli展開や全tuple表を要求しないため、アクセス条件を不当に強めた仮想対照でもない。

## 12. PR全体のstep幅とshot負担を戻す

元設計にあるordinary paired P5の正規化は
\[
B_5(x)=\sum_{k=0,2,4}\frac{x^k}{k!}\sqrt{1+(x/(k+1))^2}.
\]
RTE microsteps数をs、n=t/δとするとΓtot=B5(λRδ/s)^{ns}。[R7]

今回の展開は
\[
\log B_5(x)=x^2-\frac59x^4+O(x^6),
\]
\[
\boxed{\log\Gamma_{tot}=\frac{\lambda_R^2t\delta}{s}
+O\left(\frac{\lambda_R^4t\delta^3}{s^3}\right)}.
\]

すなわち、δを大きくしてdeterministic sweepsを減らしても、s固定ならΓtotとsampling負担が増える。補正のorder改善が最終費用へ効くのは、以前のstep幅がPF biasに強く制約されていた場合などである。sampling制約が支配していれば、correctorでbiasを減らしてもstep数を十分減らせない可能性がある。

これは新しい分子結果ではなく、設計書のB5からの解析である。Rを小さくすれば必ず有利という一般claimにもしていない。R=0ならこのnormalization負担は消えるが、D内部corrector費用は残り得る。

## 13. 費用を比較する式

各axisの残余統計幅s_stat>0、二次moment上界ν、range Wmaxに対し、同じBernstein型の十分予算を使う。
\[
N_{req}=\left\lceil\log(2/\alpha_a)
 \left(\frac{2\nu}{s_{stat}^2}+\frac{4W_{max}}{3s_{stat}}\right)\right\rceil.
\]

positive whole方式では元RTEΓを含めてν=Γtot²、Wmax=Γtot。局所Pauli対照では、correction weightのsecond momentとrangeを掛ける。biasは同じ物理targetに対して先に引く。

資源x（T、CX等）を別々に
\[
G_x=2N_{req}\,[h_x+\mathbb E C_{base,x}+\mathbb E C_{corr,x}]
\]
とする。hはstate preparation/readoutなどの共通費用。classical取得・local decompositionも別の帳簿に残す。

whole方式の補正期待費用は
\[
\mathbb E C_{corr,x}
=\frac{2t\gamma}{\mathcal B}\sum_\tau c_\tau\kappa_\tau C_{\tau,x}
\]
（unsaturated、1回の費用CτにGaussian embedding等を含める）。元方式対照のsweep削減分よりこれが小さいことだけでなく、Nreqの変化まで必要になる。

次の比較ではCτの代表値一つで実際の分布平均を置換しない。未評価の部分は未知変数としてbreak-even条件を示す。存在しない分子費用を補って数値勝敗を作らない。

## 14. finite-bitと合成精度

数学が成立しても、digital/native errorを一定に固定したままδ→0で四次になるとは言えない。

unsaturated領域でV−I=O(δ²)だから、確率誤差のL1距離Δpに対する一slotのbiasはO(Δpδ²)。local O(δ⁵)を保つ十分条件の一例はΔp=O(δ³)。受理確率O(δ)でconditional unitary誤差がεUなら、local O(δ⁵)を保つ一例はεU=O(δ⁴)。いずれも必要条件・最適配分を主張するものではない。

Gaussian GとG†を同じ近似回路の厳密inverseとして使うなら、identityのときの相殺を利用できる。例えば||G−G̃||≤εGでは、conjugation誤差を2εG||V−I||で押さえられる。その構造を保つ実装と、単純なgate-by-gate triangle boundは区別する。

これは合成費用をゼロにするものではなく、厳密なerror budgetを将来作る際の注意である。現監査のT未評価は明確に維持する。

## 15. 先行研究と主張の置き場所

| 項目 | 結論 |
|---|---|
| 三次BCH補償・四次化 | CPF等に先行。新規主張にしない |
| coefficient-weighted random exponential | qDRIFT型へ明示的対応。一般原理の新規性なし |
| identity混合・inverse-probability角度 | 疎化の既知原理。randomized THRIFTの近接性もCodexが報告 |
| coherent first-momentという形式 | それだけでNCC等からの独立差にならない |
| maximum6の局所構成 | 具体的に残す価値。ただし支持閉包だけで優先性・費用優位は確定しない |
| whole nested-commutator exponential | ここに実装上の差が残り得るが、局所Pauli対照と比べた追加利益は未評価 |

CPFは二成分のcorrectorとexponential列による実装を与える。[P1] NCCはidentity–Pauli pairingと一般Hamiltonianの交換子補償を扱う。[P4] HNCCはPauli channelを使う別のtask・input classを持つため、そのoracleや精度scalingをDFへ無料転用しない。逆にchannel方式という理由だけで比較対象から除外もしない。[P6]

「general LCUに含まれる」だけで新規性を否定しない。しかし入力・生成・実装・保証まで同じなら別名を付けた新手法にはならない。同じ局所backendを共有して比較することは、whole回転固有の価値を分離するためのablationである。局所表現それ自体に独立した新規性があるかは、別途その具体的な構成と先行研究を対応させる。

## 16. 新規性について以前の判断から何を更新するか

前回の判断は「数式と生成法が具体化したので独立監査する価値がある」だった。その判断自体を撤回する必要はない。実際に数学的構成が支持され、実装費用と先行法への対応という新しい判断材料が得られた。

今回の更新は、(i) 希薄samplerの基本原理への新規性の期待を下げる、(ii) whole回転の大きな汎用費用を重く見る、(iii) ただし最小費用とはみなさない、(iv) 同じ局所入力から安いPauli対照が作れることを主比較へ上げる、の四点である。

従って「数学PASS→production完成」とは進めない。単に未実装項目を全て埋める研究にはしない。

## 17. 代替方向の比較

| 選択肢 | 評価 | 判断 |
|---|---|---|
| 現whole方式をそのまま分子productionへ | method差と費用機構が未判定 | 不採択 |
| 汎用r=6費用だけで全候補終了 | 最小費用でなく、構造未利用 | 根拠不足 |
| 大規模な専用compilerを先に開発 | 改善の必要量・強い対照の費用が未知 | 不採択 |
| rank-oneなど安いケースのみへ即縮小 | Gaussian既知特殊化だけになる危険、実分布未評価 | 主線にはしない。controlにする |
| production rank/digital APIを先に完成 | 技術的には可能でも研究価値の問いを解かない | 後回し |
| 同じ局所Tのwhole対Pauli比較 | 具体的な差を小範囲で判別できる | **限定採択** |
| 現成果を保持し実験休止 | 有効な選択肢。比較で差が残らなければ妥当 | 終了選択肢として保持 |

## 18. 次の一つの作業単位

**目的：DF局所表現の下で、whole exp(Tτ)を使うことに、既知Pauli-pairingを具体化した対照を超える理由があるかを判別する。**

対象は既存r=3/r=6中央fixtureと、既存CDの有限ブロック条件を利用する。これらはdevelopmentであり、分子transferでも新しい独立test setでもない。二種類のfixtureの費用と確率を無根拠に混ぜない。

Codexへ委ねる関連作業は、今回の追加恒等式の独立確認、coin-first局所Pauli対照の実装、同じactive-frame/control/phase/precisionでの限定費用比較、weightとbiasを戻したresource式の算術確認である。技術的な実装順序、安定な小行列処理、focused tests、結果保存はまとめて任せる。

比較する際は、元generic compileを履歴として残す。既存の標準的な構造利用を一回の限定対照に使うことはよいが、勝つまで最適化を繰り返す汎用compiler研究へは拡張しない。Tが重要なら共通precisionに基づく小さい合成調査を計画し、RZをT換算しただけの値をactual synthesisとしない。

このレビューは次の科学的比較範囲を選択したもので、未作成sourceを即時実行した、旧markerを解除した、という意味ではない。実行契約・scopeは元監査と別に固定する。

## 19. 比較の判定条件

**継続の根拠になり得るもの**：同じ目標平均または同じ最終誤差を満たす比較で、whole方式に費用・sampling・取得の具体的な差が残り、その原因が説明できる。数値的小差はuncertainty・precision・compiler依存と分離する。新しい任意の5%/10%閾値を結果後に追加しない。

**whole方式を終了／休止する根拠**：比較範囲で局所Pauli方式に実質的に優位な資源点を作れず、別の取得上の差もない。これは任意whole合成の不可能性定理ではなく、現候補へ研究資源を投入する理由がないという判断である。

**結論保留**：費用errorの区間が重なる、primitive精度が比較不能、原inputの復元条件が成立しない場合。その技術的未解決を科学的negativeにしない。必要な一箇所の修正に限定し、別input探索へは逃げない。

局所表現が両方式を等しく助ける場合、whole方式専有の利益とは主張しない。その共通frontendの独立method deltaが具体的に残るかは、保存された証拠から判断する。薄い実装・対象変更を新しい候補名にして自動継続しない。

## 20. 研究成果と着地点

現時点で保持すべき成果は、ordered DF-S2のcorrector、bounded active-mode構成、理想sparse meanの保証、独立監査、明確なnative障害である。これらは有用な研究材料であり、無かったことにしない。

ただし、現時点で独立原著の主要claimを確定しない。将来の中心claim候補は、「DFの局所構造が、同じ情報を与えた強い補償法比較でもアクセスまたは総費用を改善する」である。単なる四次化、希薄化、r≤6という観察の列挙だけを中心にはしない。

局所対照比較がnegativeなら、現在の設計と監査を既存記録へ統合してこの候補を区切ればよい。その度に新しい技術ノートや同じ論文化レビューを追加し続ける必要はない。G10 v1.0の終了判断は変更しない。

## 21. 最終判断表

| 判断項目 | 今回の結論 |
|---|---|
| 理想数学 | 支持。監査の条件を保持 |
| 新規性 | sparse原理では主張しない。DF局所実装の差は未確定 |
| 現generic native費用 | 高い。下界・最小値とはしない |
| native/PR総資源優位 | 未実証 |
| production実装・分子検証 | 不採択・未認可 |
| 次の研究作業 | 局所whole対Pauliの費用・bias・weight比較に限定 |
| 次の実務担当 | Codex（新しい限定scope、既存証拠不変） |
| 次のGPT判断 | その比較が返ったとき。資料取得確認と開始承認を分離 |
| 既存STOP | 全て維持 |

## 22. 再現資料と未実施事項

同梱の`review_checks.py`、`review_checks.json`、局所再構成NPZは、本レビューの自己検算である。initial→finalの変更はcoin-first weight第二momentの明示的確認を追加したもので、失敗を隠した結果差替えではない。初期scriptも保存する。

18 checks PASS。実行環境はPython3.13.5、NumPy2.3.5、SciPy1.17.0、SymPy1.14.0。登録本番、分子生成、quantum shots、native compile、Clifford+T synthesisは0。外部repositoryへの変更・pushは0。

全31資料の独立byte再照合、77+15件の再実行、全先行文献の網羅的優先性調査、production rank/digital proof、分子資源予測は未実施。Randomized THRIFT本文の再取得不能は外部接続上の制約として残す。repository資料の未pushとは扱わない。

## 参考資料

### 固定repository

- [R1：独立監査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8a5e21c8f6ecf98ee285ec1fad49079ffb5ed97f/docs/tracks/algorithm_codesign/df_sparse_correction_independent_audit_20261011.md)
- [R2：先行研究比較](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8a5e21c8f6ecf98ee285ec1fad49079ffb5ed97f/docs/tracks/algorithm_codesign/df_sparse_correction_prior_art_comparison_20261011.md)
- [R3：Phase E結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8a5e21c8f6ecf98ee285ec1fad49079ffb5ed97f/artifacts/track_b_df_sparse_correction_audit/2026-10-11/phase_E_checks.json)
- [R4：native監査source](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8a5e21c8f6ecf98ee285ec1fad49079ffb5ed97f/scripts/tracks/algorithm_codesign/audit_df_sparse_correction_native.py)
- [R5：数学監査source](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8a5e21c8f6ecf98ee285ec1fad49079ffb5ed97f/scripts/tracks/algorithm_codesign/audit_df_sparse_correction.py)
- [R6：Phase CD結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8a5e21c8f6ecf98ee285ec1fad49079ffb5ed97f/artifacts/track_b_df_sparse_correction_audit/2026-10-11/phase_CD_checks.json)
- [R7：原設計](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8a5e21c8f6ecf98ee285ec1fad49079ffb5ed97f/docs/research/track_b_df_sparse_correction_design_review_20261011.md)
- [R8：証拠manifest](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8a5e21c8f6ecf98ee285ec1fad49079ffb5ed97f/artifacts/track_b_df_sparse_correction_audit/2026-10-11/evidence_manifest.json)

### 一次文献と今回の確認箇所

- [P1] Bagherimehrab et al., *Faster Algorithmic Quantum and Classical Simulations by Corrected Product Formulas*, [arXiv:2409.08265v3](https://arxiv.org/html/2409.08265v3). Sec.3.2、symmetric/symplectic corrector、Sec.5 compilation。本文を確認。
- [P2] Campbell, *A random compiler for fast Hamiltonian simulation*, [arXiv:1811.08017v2](https://arxiv.org/pdf/1811.08017). Eq.(1)、sampling recipe、Eqs.(4)–(11)。本文・p.2画像を確認。channel保証の自動転用はしない。
- [P3] Ouyang, White, Campbell, *Compilation by stochastic Hamiltonian sparsification*, [arXiv:1910.06255v2](https://arxiv.org/pdf/1910.06255v2). Sec.3、inverse-probability Hamiltonian term、expected gate budget。本文抽出を確認。
- [P4] Zeng et al., *Simple and high-precision Hamiltonian simulation by compensating Trotter error with linear combination of unitary operations*, [arXiv:2212.04566v2](https://arxiv.org/pdf/2212.04566). Eq.(111)のidentity–Pauli pairing、Appendix Eの一般Hamiltonian。本文抽出を確認。
- [P5] Kim, García-Pintos, *Randomized product formulas beyond optimal deterministic scaling*, [arXiv:2608.07720](https://arxiv.org/abs/2608.07720). 今回は一次検索結果の書誌・abstractまで。詳細SMの対応はR2の監査に依拠し、独立全文確認と区別する。
- [P6] Wang et al., *Trotter error compensation with polylogarithmic precision and nested-commutator scaling without ancillas*, [arXiv:2607.11856v2](https://arxiv.org/html/2607.11856v2). Pauli-channel construction、input/observable task、Theorem1の条件を確認。lattice理論を一般DFへ無料転用しない。

本書は、独立新規性または全既知法に対する最適性を確定する文書ではない。現在の固定証拠から、次に解くべき比較を具体化して研究投資の範囲を選んだ科学的レビューである。
