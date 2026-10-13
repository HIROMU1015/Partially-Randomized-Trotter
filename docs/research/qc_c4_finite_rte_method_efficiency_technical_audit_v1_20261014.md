# C4 finite-RTE proposal：方法差・効率的生成の限定技術監査

2026-10-14 JST。Codex技術監査。Phase A/C完了、Phase B・production実装・科学計算は未開始。状態は **REDUCES_TO_KNOWN（現構成）／CONDITIONAL（限定classの実装可能性）／UNDETERMINED（独立した圧縮法・実用資源利益）**。

現C4の最適確率則、whole-pathの費用を用いること、平均を保存するweight、有限状態backward DPそのものには、独立した算法上の差を確認できなかった。強い対照として、同じ費用情報を使う既知のresource-optimal IS、rejection/zero-fill、weighted-automataのweight pushingを置くと、現案は既知手順へ還元できる。
一般DF/native IRの費用状態を多項式サイズへ圧縮する具体的な構成は提示されていない。本文未取得を差の根拠にはしていない。したがって、指示されたPhase A停止条件に従い、詳細実装・新しい候補生成へ進まない。C4の正式採否、研究価値、C1の終了・再設計はGPTの科学判断に残す。

## 1. 固定資料、取得範囲、非実行

- Repository：HIROMU1015/Partially-Randomized-Trotter。独立branch：qc-c4-finite-rte-technical-audit-20261014。
- 文書base/C1結果：5ff55e749ceb33e54da0b3c70ac6ac08599ebb82。C1文書・結果・認可・STOPはbyte保存し、新たな集計や数学調査をしていない。
- [C4の元具体案](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fc063696026b2500d9608abe447849fd76274ed5/docs/research/qc_algorithm_candidates_technical_prevalidation_20261013.md)：fc063696026b2500d9608abe447849fd76274ed5、§7。
- D1 read-only：2dcd56737be35526f6436a52835f59d817adf30c。[固定参照identity](../../artifacts/qc_c4_finite_rte_technical_audit_v1/2026-10-14/reference_identity.json)、[作業scope](../../artifacts/qc_c4_finite_rte_technical_audit_v1/2026-10-14/scope.json)、[状態分類](../../artifacts/qc_c4_finite_rte_technical_audit_v1/2026-10-14/method_status.json)。別branchのsourceをimport・実行・変更していない。
- 一次本文の版・URL・関連節・取得SHA：[literature registry](../../artifacts/qc_c4_finite_rte_technical_audit_v1/2026-10-14/literature_registry.json)。必須三論文の本文は取得済み。Mohriは著者PDFをweb経由で確認し、local直接取得のDNS失敗と本文取得を区別した。文献全文はrepositoryへ再配布しない。
- 新H、分子、固有値、PF、compile/T合成、sampling、D1/C1の事後数値集計、係数代入、production実装、新GPT科学レビューはいずれも0。以下の式は係数を代入しない技術的な照合で、数値性能結果や独立新規定理の認定ではない。

## 2. 先行本文との直接対応

### 2.1 一次資料と重要な相違

[Resource-Optimal IS（2026、v1）](https://arxiv.org/html/2603.13495v1)のTheorem 1、II.1、Theorem 2、IV/Theorem 3/Corollary 1、V.5、Appendix Aを確認した。一般費用のnet-cost最小化に加え、合成操作、実装biasの不変性、error detection、signed PECも扱う。対象がwhole circuitやsigned列であるだけではC4の差にならない。一方、最適lawを記述する定理と、指数supportから有限bitで安くdrawする生成器の計算量保証は別であり、後者の一般保証は確認できない。

[Importance sampling（Quantum 2023）](https://quantum-journal.org/papers/q-2023-04-13-977/pdf/)のAlgorithm 1、III、IV.1、Theorem 4–5を確認した。ここではproposalに応じて各termの時間も変更し、qDrift/composite channelのbiasと集中を会計する。費用lawは係数/C_j型であり、C4の係数/√g型と異なる目的・意味論を持つ。異なる確率則を単に新規性としない。

[PRHS（2026、v1）](https://arxiv.org/html/2608.29756v1)のEq.(4)–(10)、Appendix A–C、結論を確認した。correlated slicesのsigned係数をrun decompositionで与える。結論はresource-optimal ISとの接続を既に明示する。閉じた係数式、(N,M)選択、絶対値lawの正規化・drawの計算量は区別する。一般のarchitecture費用を入れた正規化済みサンプラーの効率は本文から未確定。signed係数をそのまま確率とする表記は合法なlawではなく、絶対値proposalとsign/phase保持が必要である。

[PR v2](https://arxiv.org/pdf/2503.05647v2)のIV/V、Appendix A.2を確認した。even orderとIID labels、回転pair、phase、LCU normalizationを既に与える。[SCU/CTS v2](https://arxiv.org/html/2407.21095v2)のII.1/II.3、IV.2、Supplementary Note 3はPauli collectionと層samplingを扱う。CTSの明示展開費用と層samplingのnormalization負担を区別し、全CTSが全path列挙を必須とする対照は作らない。v2と[出版版](https://www.nature.com/articles/s41534-025-01168-w)を同一版と扱っていない。

### 2.2 確率・目的・生成法の対応表

| 方法 | 対象分布・列 | 統計/資源目的 | 確率則 | 具体的な生成手順と費用情報 |
|---|---|---|---|---|
| Resource-optimal 2026 | 固定protocolのθ。θをwhole pathやsigned PEC patternとしてよい | 第二moment上界×平均実行費用 | p(θ)/√c(θ)。error detectionでは後述のk,fを含む | lawの解析式、応用別の集約。cは一般の取得済み費用。一般指数列への有限bit生成器・normalizer計算量を定理が無条件に供給するわけではない |
| Quantum 2023 | term law q(j)から独立qDrift列、deterministic/stochastic composite | channel bias、集中、term数、実験数、native費用 | q_c(j)∝h_j/C_j；角度t h_j/(Nq_j)も変更 | Algorithm 1のterm drawと時間更新。L個の係数・term価格が必要 |
| PRHS 2026 | Γ^M個のcorrelated signed slice列 | channel近似、query NMとsigned overhead | run/Stirling式によるsigned係数。正proposalはその絶対値正規化 | 選んだpath係数の閉形式と(N,M)探索。全絶対massの効率的生成は別。native相殺価格は主会計に入っていない |
| PR finite-RTE | Taylor order、ordered labels、子回転、繰返しstep | 平均Hadamard signal、打切り・normalization、phase estimation | order law×IID coefficient labelsが基本 | order→labels→回転/phaseを生成。係数とfinite truncationの情報。word全列挙を要しない基本対照 |
| SCU/CTS | Pauli-collection ensemble、coherent cross terms、層列 | LCU/SCU normalization、observable precision、depth/shots | normalized coefficient/group masses；層samplingも合法 | Pauli algebraによるcollection、real/imag grouping、層ごとのdraw。collectionアクセスとnorm取得はclass依存 |
| Track B D1 | 同じ固定finite P5の六表現、digital event列とwhole-trial zeros | 有理bias、M2/range、Bernstein十分shots、T/CX/1Qとprep/readout | canonical/group/local rejection/dyadic law。保存reference表はoperational oracleでない | g7_generator/return_aggregation、g9_nativeのliteral IR。小providerの取得済みstrict primitive価格。proposal_search=false |
| 現C4 | 既存finite-Fのpath lawを変更 | まずM2×平均費用；D1費用は未保証 | b_γ/√g_γ、b=|a| | finite-state tilt+positive quadrature+backward DP案。小状態・取得可能な価格・係数因子化は前提であり、generic DFには未構成 |

| 方法 | 古典時間・メモリ | 有限precision・統計保証 | C4に固有の差があるか |
|---|---|---|---|
| Resource-optimal 2026 | 費用関数/normalizer/drawアクセスの一般bit complexityは未確定。合成の漸近解析とsampler費用は別 | bias不変性、error-detection重み、第二moment net-cost。D1の有限confidence Bernstein契約そのものではない | 最適則、whole-path、signed再重み付けはREDUCES_TO_KNOWN |
| Quantum 2023 | L-term tableなら通常の有限カテゴリ生成。price取得・bit長を別に要する | qDrift/channel固有のbiasと集中保証；C4のFをそのまま保証しない | 対象平均が異なる。cost-aware sampling一般をC4の差に数えない |
| PRHS 2026 | 係数評価/有限候補(N,M)の探索と、Γ^M supportの正規化・drawを別扱い。後者の一様効率はUNDETERMINED | signed normalizationと近似誤差。native/finite-bit/D1 range保証を無料で得ない | correlated path自体は既知。Fとchannelの相違は新しい生成器の証拠ではない |
| PR/SCU/CTS | 基本order/IID lawはstreaming。CTSにはcollectionと層samplingの選択があり、後者でも取得・norm負担が残る | phase/normalization/打切り/observable誤差の契約を保つ | 「列挙しない」だけでは差なし |
| D1 | local query、有限bit law、保存価格へのlookup。native stackはpath単体を線形に処理できるが、全prefixの同時DPは別 | 実law、zero-fill、range、bias、accepted-tail/hardを会計。online sampling費用は未取得 | source再利用候補はあるが、新proposalへの証拠はない |
| C4 | J、L、状態S、遷移E、bit長を含む必要がある。元O(JrS²)の表記だけではalphabet・精度・初期化を欠く | 実際のqとweightの一致、full support、位相、有限bit、rare tailsを新契約で保証する必要 | 新しい圧縮法・同情報下の保証差は未特定。効率はCONDITIONAL |

## 3. Resource-optimal理論への還元と適用限界

### 3.1 固定finite-Fの推定意味論

F=Σ_γ a_γU_γ、a_γ=b_γe^{iφ_γ}、b_γ>0とする。係数0のラベルは最初に除く。controlled operatorはV_γ=e^{iφ_γ}U_γ。各axisのHadamard出力Y_γ∈{−1,+1}はReまたはIm〈ψ|V_γ|ψ〉を平均に持つ。
q_γ>0からdrawしX=b_γY_γ/q_γとすれば、平均は同じaxisの〈ψ|F|ψ〉。複素aを、Re〈U〉しか測らない単一実出力へ単に掛けても目的signalにならない。controlled phaseか、両axisの正しい線形会計が必要。

取得済みの正費用g_γについて

\[
V(q)=\sum_\gamma b_\gamma^2/q_\gamma,\quad
G(q)=\sum_\gamma q_\gamma g_\gamma,\quad
V(q)G(q)\ge(\sum_\gamma b_\gamma\sqrt{g_\gamma})^2.
\]

B0=Σb、p=b/B0でTheorem 1へ対応する。signed/phaseを保つ拡張も基本原理を変えない。真のsecond momentがpathごとにv_γ<1と分かる場合は別目的になり得るが、未知measurement/ground stateを無料にしてb√v/√gを使わない。
qの変更は同じfinite-FのTaylor/PF biasを改善しない。実装Ṽ_γと元係数が固定なら、平均実装差Σb(Ṽ−V)もqに依存しない。primitive精度や許す回路を変えることは別契約である。

費用が0のpathを持つT-only会計にはb/√gを直接使えない。正のprep/readout等を同じ資源価格で含めるか、zero-cost classを別に扱う必要がある。都合のよい正offsetを無断に加えると目的が変わる。小さいqへ質量を押し込むinfimumは、有限range・古典費用・実行可能性を保証しない。

### 3.2 Error detectionの定理をD1と混ぜない

2026論文IVの設定は、完全に信頼できるerror flag、成功確率f(γ)>0、同pathで最大L_det回の物理実行、k_γ=1−(1−f_γ)^{L_det}である。ZeroFillのweightはp/(qk)、Discardは受理normalizer E_q[k]をさらに含む。Corollary 1のlawは

\[
q^*_{L_{\rm det}}(\gamma)\propto
\frac{p_\gamma}{k_\gamma}\sqrt{\frac{f_\gamma}{c_\gamma}}.
\]

これはknown resultで、C4やD1への新しい検証ではない。f=1なら通常lawへ戻る。論文の反復/error-detection費用、success条件下の結果分布、normalizerの取得をすべて必要とする。
D1のzero-fillは量子実行前のclassical acceptanceであり、物理error detectionではない。D1へ上式やerror-detection由来の性能改善をそのまま移さない。また、net-cost/CLT scalingをfinite-confidenceの十分shot数と同一視しない。

## 4. 「指数列挙なし」を達成する強い既知対照

### 4.1 通常のrejection samplingで足りるclass

係数law p0=b/B0を安く生成・評価でき、sampled pathの価格gも時間T_g(L)で評価でき、取得可能な正界0<g_min≤g≤g_maxがあると仮定する。[Devroye, Chapter II.3](https://luc.devroye.org/chapter_two.pdf)の標準rejectionを特殊化すると、p0からpathをdrawし、t_γ=√(g_min/g_γ)でacceptすれば、accepted lawはq*になる。

\[
A=\mathbb E_{p0}[t],\quad
q_{\rm acc}(\gamma)=p0_\gamma t_\gamma/A,\quad
\mathbb E[\text{proposals}]=1/A\le\sqrt{g_{\max}/g_{\min}}.
\]

R=g_max/g_minが入力bit長に対して多項式で、p0/g/coinが効率的なら、期待古典時間はO(√R(T_p0+T_g+T_coin))。sampler memoryはpath/IR/stackを保持するO(L)等で足り、全stack状態のDPや全path列挙は不要。Rが指数的、p0が高価、価格が未compileのoracle、あるいはcoin精度が未保証なら効率性は得られない。期待timeとhard tailも区別する。

accepted qの**正規化された確率**をweightに使うにはAが要る。rejectionでdrawできることと、Aを安く計算できることは同じではない。次のzero-fill形式ならそのnormalizerを避けられる。

### 4.2 Normalizer不要のwhole-trial zero-fill

各p0 drawを一試行と数え、tでacceptしたときだけ量子回路を実行し、X=bY/(p0 t)とし、rejectはX=0と記録する。すると

\[
\mathbb EX=\text{axis signal}(F),\quad
V_{\rm zero}=\frac{B0}{\sqrt{g_{\min}}}\sum b_\gamma\sqrt{g_\gamma},\quad
G_{\rm zero}=\frac{\sqrt{g_{\min}}}{B0}\sum b_\gamma\sqrt{g_\gamma}.
\]

よってV_zero G_zeroは同じ既知下界に一致する。列挙・Aの計算なしでも**理想の量子費用×second-moment envelope**に到達できる場合がある。これは既知IS/rejection/zero-fillの直接代数であり、新算法の認定ではない。最大weightはB0√R、全試行の古典費用は非zeroで、D1のrange項・shot ceiling・accepted-tail・準備費用を戻した最適性は示していない。

これは理想p0/tの場合の式である。実際に生成するp0_hat,t_hatが正で、実係数と実確率からb/(p0_hat t_hat)を評価すれば平均を保つが、上記最適productへの一致は一般には失う。固定bit coinの失敗をretry・clippingで救済しない。概念上の強いbaselineとして置いたのであり、今回sampleを一つも生成していない。

## 5. 既存C4の有限状態案を技術的に展開したときの条件

この節は元C4案の成立条件の監査であり、差がないままPhase Bの新constructor設計を起動したものではない。[Mohri–Pereira–Riley §3.4, Proposition 1](https://cs.nyu.edu/~mohri/pub/hbka.pdf)のweighted-automata weight pushingがbackward massによる確率生成の強い既知対照である。

### 5.1 必要な表現

固定された最大長Lのevent列を、停止/identity paddingを含む有限層へ埋める。状態s、event e、決定的遷移T_t(s,e)、非負係数weight w_t(s,e)、終端w_end(s)を取得でき、

\[
b_\gamma=w_{\rm end}(s_L)\prod_t w_t(s_t,e_t),\qquad
g_\gamma=g_0+g_{\rm end}(s_L)+\sum_t c_t(s_t,e_t)>0
\]

と表す必要がある。単にunitaryが有限alphabetであるだけでは不十分。wordをcollectした後の|Σa|や平方根group normは、元label-productの因子化を一般には保たない。新たな係数アクセス・collectionを無料化しない。

### 5.2 既知backward DPへ還元する手順

正のquadratureの各node u_jに対し、tilted transitionがw_t exp(−u_j c_t)となる。これを事前に正の有理weight W_{j,t}として固定した場合、終端H_{j,L}から

\[
H_{j,t}(s)=\sum_e W_{j,t}(s,e)H_{j,t+1}(T_t(s,e))
\]

を計算し、transition確率W H_next/Hを用いてdrawする。dead stateは零massとして除外する。全mass Z_jを使い、quadrature係数β_j Z_jに比例してnodeを選ぶ。joint(node,path)を作るだけでなく、目的がmarginal path q_γなら全nodeの質量和を用いて実q_γを評価する必要がある。選んだ一nodeの確率だけをmarginal qと呼ばない。
この正規化・遷移法は既知weight pushing/forward-backwardの直接適用である。Laplace恒等式とpositive quadratureの組合せだけをC4固有の算法とはしない。

### 5.3 計算量を閉じるために不足している情報

J nodes、各層S states、eventを区別した遷移数E_t、b-bit係数のとき、DPはO(JΣ_t E_t)個の算術演算、mass保存O(JLS)。deterministic next stateでlabel数dならE_t≤Sd、dense pairならS²にevent multiplicityを含める必要がある。元O(JrS²)だけでは、d、各primitive word長、bit演算費用、price-table取得を隠す。
drawは条件付きCDFをその場で読むならO(Ld)、alias等の前処理を与えれば別time/memory tradeoff。選んだpathのmarginal law評価は、状態を決定的に再現できるclassでO(JL)算術。状態遷移がambiguousならlatent列のsumを追加しなければならない。

有理massのbit長と演算時間を算術回数と分ける。共通denominatorを固定する等の実装が要る。Jを所定の費用window/誤差で安く選べる証明、exp近似のbit長、underflow回避、支持を失わない丸め、init costを現案は具体化していない。J/Sが多項式という名前だけでは完成した効率算法ではない。

quadratureを定数精度で使っても、実qとweightが一致すれば平均Fは保てる。q*近似の性能は別問題である。例えばtilt近似の相対幅ξを**全path**で証明できた場合だけ、正規化比とM2×Gの幅をそこから計算できる。今回その幅やJを取得したことにはしていない。

## 6. Native IRの費用状態：成立classと増大するcase

### 6.1 限定classの境界

| class | 具体的な状態/価格 | 成立条件と費用 | 技術状態 |
|---|---|---|---|
| 完全加法primitive価格 | coefficient生成状態と累積整数費用 | 取得済み価格、no fusion。cost-sum DPは最大累積価格Cに依存しpseudo-polynomial。rejection対照も可能 | CONDITIONAL、既知構成 |
| 直前basisだけの共有 | last basis label、係数群、既定pairwise変換価格C_ij | circuitをbasis–diagonal–uncomputeの固定grammarに限定。全pair価格取得O(d²)、grammar内の費用だけをexactと呼ぶ | CONDITIONAL、generic compilerとの一致は未証明 |
| 相殺長が固定bのlocal pass | 末尾b tokens、coefficient state、確定済み費用 | 将来がpopしないprefixをcommitできる証明。alphabet Aに対しstate O(A^b)、bが増えると効率性を失う | CONDITIONAL、境界証明が必要 |
| 非収集Pauli-product label | Pauli frameとphase、係数state | n-qubitの全frameは4^n級。n固定/特殊対称部分群なら小さくなるが一般分子へ多項式とは言えない | CONDITIONAL、restricted only |
| 任意inverse stack/global synthesis | 全未確定stack、angle/frame履歴 | 個別pathをO(L)memoryで処理可能でもprefix集合のexact有限状態DPは指数的になり得る | 一般効率はUNDETERMINED |

b-buffer内で負のcost更新を用いる表現は、最終g>0を保証しても、tiltの大きい中間weightやbit長を別に制御する必要がある。確定gateにのみ非負価格を課す固定grammarの方が条件を明示しやすいが、それは既存全IR passと同一という証明を意味しない。

### 6.2 D1のstackに関する具体的確認

[g9_native.simplify](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2dcd56737be35526f6436a52835f59d817adf30c/src/trottertracks/algorithm_codesign/g9_native.py)は末尾gateのinverseならpopし、そうでなければpushする。文字列の末尾一個だけに置換してよいpassではない。複数popの後にさらに以前のgateへ到達する。
[return_aggregation.reduce_word](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2dcd56737be35526f6436a52835f59d817adf30c/src/trottertracks/algorithm_codesign/return_aggregation.py)も同種のstackを使う。係数側のreturn queryとnative価格stackを同じ状態だと仮定しない。

最大長2ℓを許す、inverse対を持つ二generatorのfree-word grammarでは、長ℓのreduced prefixesは4·3^(ℓ−1)個。異なる同長prefix u,vに長ℓのsuffix u^−1を加えると、uは空になりvはならない。残るgateに正価格を付けるclassでは費用も区別される。よって将来のexact reduced costを区別する有限状態機械は、これらprefixを一律にmergeできない。三involutionのgrammarでも3·2^(ℓ−1)個の同種の区別が起こる。これは総horizon 2ℓに対して指数的なstate必要性のcaseである。
これは unrestricted stack の有限状態圧縮への静的反例であり、一般DF lower boundや固定P5/D1の計算不能性の証明ではない。固定P5の有限supportは小さいが、次のsize/degree/長さへ移す効率保証は別である。

generic DFで任意Gaussian frameを維持すると、frameの行列表示をpath単体で多項式memoryに保存できても、異なるprefixのframeは指数個あり得る。global synthesis/routing/angle-dependent T価格をlast-label pair tableで正しく表せるとは未確認。取得可能な代理費用ĝならDPはĝに対して最適化するだけで、実gへの利益は別のenvelopeや費用取得が必要。equalityを確認せず「exact compiler cost」としない。

## 7. Finite-bit、実確率、defensive mixture

q=(1−ζ)q_tilt+ζq0をwhole-pathで一回mixし、0<ζ≤1とする。q≥ζq0ならW≤W0/ζ、W0=sup b/q0。q0=b/B0をexact生成できる場合だけW0=B0。D1の丸め済みq0へB0をそのまま代入しない。
stepごとにmixするとwhole-path下界は一般にζ^L q0であり、同じζのrange保護ではない。ζの最適化を今回行っていない。

実装に必要なのは、有限bitのgroup/transition/node/acceptance countsが作る**実際のlaw**と、そのpath確率の評価である。dyadic conditional probabilitiesの積とnode全和からqを算出し、同じqをweightに使う。支持を丸めで0へ落とすと平均保証を失う。相対区間を満たせなければfail-closedとするD1の方針を保持する。
positive rational lawを生成することと、ideal exp/√g lawをexact生成することは違う。coef/phase/回転精度とweight算術誤差は別のbias budgetへ入れる。正規化質量やweightの高精度計算、random-bit消費もonline classical費用に含める。finite-bit rejectionがrare probabilityを失う場合に無断のprecision rescueやretryを行わない。

## 8. D1会計との接続条件

### 8.1 正本の式と現在の禁止境界

[D1 accounting.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2dcd56737be35526f6436a52835f59d817adf30c/src/trottertracks/algorithm_codesign/rte_development/accounting.py)をread-onlyで確認した。現在の契約はproposal_search=false、cross_step_fusion=false、SP_enabled=false。固定六arm、P5、r=1/2の一回認可をC4に流用しない。
D1は明示3-system-qubit synthetic involution、p=(1/5,3/10,1/2)、総時間5/7、state000、sign+、P5、r=1/2、axis ε=1/200、family α=1/20、strict primitive1e−6、proposal160bit/root256bit。geometry/basis/DF rank/L_Dは非該当。新たな数値値は算出していない。

statistical余白s=ε_axis−bias>0に対し、元D1の十分shot数は

\[
N=\left\lceil \overline{\log(2/\alpha_{axis})}
\left(\frac{2\overline M_2}{s^2}+\frac{4\overline W}{3s}\right)\right\rceil .
\]

独立step proposalの場合に限り、元sourceはstep moment/rangeのr乗、acceptanceのr乗、native平均r Z^(r−1)cを使う。C4のwhole-path lawが非因子化なら、このべき乗とfactorをそのまま再利用してはいけない。whole-path coefficientsとsubprobabilityを使った新しい会計が必要である。

### 8.2 Whole-trialで閉じる必要量

非zero quantum path γの無条件subprobabilityをr_γ、零massを1−Σrとし、Fの係数magnitudeをb_γとする。rはここではsubprobability記号で、D1のstep数とは異なる。phase込みaxis出力で

\[
\bar M_2\ge\sum b_\gamma^2/r_\gamma,\quad
\bar W\ge\max b_\gamma/r_\gamma,\quad
A=\sum r_\gamma,\quad
G_k=\sum r_\gamma g_{k,\gamma}
\]

を、運用時に取得可能な量から保証する必要がある。新proposalに不変なのはFと物理/近似biasの契約であり、M2/W/A/Gではない。accepted-onlyの平均へ無断に切り替えない。sampleしたpathのweightは知れても、全supportの上界を安く証明できるとは限らない。

2axisを使うD1型会計なら、Nを十分条件から取得し、価格モデルごとに

\[
C_k=Q_{setup,k}+2N\{G_{k,\rm circuit}+A H_{k,\rm prep/readout}+C_{k,\rm classical\ per\ trial}\}
\]

を比較する。異質な古典時間とT/CXを足す価格が未固定なら別資源として報告する。新state情報取得がある場合もQ_setupへ入れる。Tはnative1Qに含まれるので重複加算しない。

| 必要量 | 減少だけで足りない理由／取得課題 |
|---|---|
| 第二moment | exact momentでなく使える上界が必要。cheap-normalizer envelopeがoracle値へ置換されないこと |
| 最大weight/range | rare pathのq低下で増大し、Bernsteinの線形項やshot capを悪化させ得る |
| whole-trial acceptance | pre-quantum zeroでも試行を記録し、古典生成/評価は費用0にしない |
| rare-event tail | 未観測の高価wordがなくなる証明ではない。accepted countのtailとpath価格のtail/worstを別に扱う |
| controlled価格 | scalar phase、actual-adjoint uncompute、basis/parity、primitive precisionを共通に含める |
| online sampling | q/weight評価、random bits、価格算出、拒否を含む。保存constructor時間やserialization時間をonline時間としない |
| setup/normalization | DP/table価格取得、J選択、bit長・root/exp評価、norm証明を一回費用としても償却条件が必要 |
| prep/measurement | accepted量子試行ごとの費用。Hadamard信号精度をground-energy精度へ直接読み替えない |

各whole trialが独立でacceptance上界を得られる場合だけ、元D1と同種のbinomial/Bernstein accepted-tailを作れる。再weightで必要Nが増えること、hard attempts、最大native価格、記録failureを同時に戻す。範囲外correlationや適応proposalには別の統計証明が必要。

### 8.3 公平な強い比較

PR ordinary、canonical/group/local full、closed P5、CTSの各合法表現に、同じ費用情報、finite-state表現、rejection/zero-fill、支持保護、有限bit選択を許す。同じFを保持するproposal比較と、Fを異なるensembleへ再表現する比較を区別する。CTSのPauli情報をC4だけに無料付与せず、full enumeration oracleは診断専用で運用生成器には使わない。
費用のPareto/価格は共通に固定し、最良のknown optimized proposalを対照とする。M2×Gが改善しても、上式のNや最終C_kが改善するとは認定しない。
finite-RTEの対象は平均振幅Fであり、Σq UρU†のrandom-unitary channelではない。別channelのdiamond-norm guaranteeをこのsignalやground-energy taskの証明に転用しない。

## 9. 到達点と投資境界

| 問い | 技術的結論 |
|---|---|
| 現C4に独立した算法上の差が残ったか | 現案の最適則、whole-path、bias保存、DPはREDUCES_TO_KNOWN。新しいnative-cost圧縮・生成手順の差は未特定 |
| 理論・データ構造・手順・費用保証のどこに残り得るか | 一般でない具体的IR classの多項式状態圧縮、cheapなM2/W/tail certificate、同情報known samplingとの差。現在はUNDETERMINED |
| D1への具体的接続はあるか | actual digital law/weight、pre-quantum zeros、native価格、Bernstein/accepted-tailという接続点はSUPPORTED。ただし新whole-path会計はCONDITIONAL、実装/結果なし |
| 現時点で追加投資する技術的根拠はあるか | production/科学benchmarkへ進む根拠は未成立。一般的なcost-aware ISの実装候補としての成立と、独立算法への投資は区別 |
| GPT判断を変える情報 | 明示IR grammarと圧縮証明、取得可能価格、bit/J/S/range証明、同情報rejection/DP対照との差、setup/online/prepを含む費用接続。現在それらの証拠はない |
| 差が残らない部分 | Theorem1への係数law置換、signed/phase再weight、standard rejection/zero-fill、weighted-automata normalizationの直接適用 |

ここで「未特定」は網羅的な新規性不存在証明ではない。未知の構成が存在し得ることだけを理由に、同型候補探索・詳細実装・数値scanを継続しない。
今回新しいpilot実行計画・科学認可は発行しない。[GPT引継ぎ](qc_c4_finite_rte_gpt_handoff_v1_20261014.md)に科学判断を戻す。C1の既存結果・STOPは不変で、C4のtechnical reductionをC1終了の代行理由にしない。

## 10. 保全、公開、Mandatory STOP

361既存証拠/source/準備files、旧142manifest records/non-result fields、既存global description欠落FAIL、元worktreeの132paths、他branchのsource/結果/認可/STOPを維持する。N1 STOP・N2 HOLD・N3 CLOSED・THRIFT現template終了、C1 one-shot消費済みSTOPは変えない。
[静的保全とリンク検査](../../artifacts/qc_c4_finite_rte_technical_audit_v1/2026-10-14/static_integrity.json)は文書/identity検査で、数学の外部再証明や新しいCI/科学結果ではない。commit・push後にGitHubの主要文書、source参照、保護証拠を再取得してbyte照合する。
公開後Mandatory STOP。同じC4の候補探索を反復せず、GPT重要科学判断の別開始承認を待つ。新計算・benchmark・production実装には、別の固定scopeと明示実行認可が必要。

## GitHub公開確認と最終STOP

[再取得receipt](../../artifacts/qc_c4_finite_rte_technical_audit_v1/2026-10-14/publication/receipt.json)：監査資料payload `60da5738d48e0faaedd3de125b0350301935dc3a` の21変更blobs、361保護files、C4/D1固定参照10filesをGitHubからbyte/SHA照合した。独立cacheにlocal alternatesはなく、公開は指定repositoryの独立branchのみ。receiptは後続metadataとして保存し、最終remote取得で全リンクを照合する。新しい数値・数学プログラム・science source実行やGPTレビューは追加せずMandatory STOP。
