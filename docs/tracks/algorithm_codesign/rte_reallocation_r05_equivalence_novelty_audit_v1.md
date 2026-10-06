# RTE reallocation R0.5: result-prior equivalence / novelty closure audit v1

2026-10-06 JST. Base `672d6bc667eaa7b9ca4979b012f1530499d701b8`。
利用者の[固定指示](inputs/rte_reallocation_r05_user_instruction_20261006.txt)に従うdocs/symbolic-only監査。
既存Aの数学は[R0独立証明](rte_reallocation_r0_independent_proof_v1.md)を使用し、再探索しない。

**限定classification: `METHOD_DELTA_CANDIDATE`。result-prior gate: `CONDITIONAL-R1`。**
これは指定P0–P3の具体構成との比較で残った候補差分であり、世界初・algorithm成立・
研究GOではない。R1の必要性・claim・実装contractはGPT判断へ返す。R1実行は未認可。

## 1. 同じtargetと二つの情報model

\(Q_\ell^\dagger=Q_\ell\)、\(Q_\ell^2=I\)、\(p_\ell\ge0\)、\(\sum p_\ell=1\)、
\(\widehat R=\sum p_\ell Q_\ell\)。有限targetは
\(M=P_{2d+1}(-i\sigma x\widehat R)\)。A-family、末端条件、a/b/c/phiは利用者指示とR0のまま固定。
\(E=\sum_{j=0}^{d}x^{2j}/(2j)!\)、\(O=\sum_{j=0}^{d}x^{2j+1}/(2j+1)!\)、
\(B_A=\sqrt{E^2+O^2}\)。以下のnormalizationは**一つのlinear operator mean**に対するもの。
P2/P3のforward/backward observableやchannelのmomentを同じ一乗の資源として転記しない。

| Model | 利用情報 | 本監査での意味 |
|---|---|---|
| I0 | x/m/sigma、p、Q sampling/implementation access、Q²=I | A/P0/PR/ゼロ次PTSCのevent description生成。全word列挙・Pauli closure・集約・dense Mなし |
| I1 | さらにPauli multiplicationと全係数collection/cancellation | CTS collected。full collectionを省くper-word closureもI0に含めず、I1情報の一部として表示 |

**event description生成と物理実装oracleは分ける。** 一般black-box Qを作用させられることだけで、
任意角exp(-i phi Q)、controlled-Q、そのrelative phaseのnative実装が無料で得られるとはしない。
A/P0/ゼロ次PTSCで共通の実装前提を固定する必要がある。I0での係数・order・word生成という差と、
controlled/native資源差を混同しない。実際のRTEEventのodd拒否というR0所見も維持する。

## 2. P0/P1: ordinary endpointのみの対応

通常端点はa_(2j)=t_(2j)、b_(2j)=t_(2j+1)、odd coefficient=0。
そのc、order確率c/B、IID word law、phaseと角度は既知paired Taylor構成である。
P0の表示は+timeかつwordの右にrotation、Aは-timeかつ左にrotation。
P0の+time **event全体のadjointを取り**、wordを逆順に記述すると

\[
[(i\sigma)^n Q_1\cdots Q_n e^{+i\sigma\phi Q_0}]^\dagger
=(-i\sigma)^n e^{-i\sigma\phi Q_0}Q_n\cdots Q_1
\]

となり、通常端点のweights・分布・unitariesまで一致する。
rotationを非可換word内で移動したのではない。単純なtime-sign置換だけの右/左対応はfirst meanのみ。
P1 A.2は直接のleft-rotation paired-event baseline。符号/分母はR0と現repoの
明示規約exp(-i sigma phi Q)、phi=atan[x/(n+1)]に合わせる。
抜粋notationの曖昧さを新しいerratumやalgorithm差とはしない。

P0/P1の通常端点はこの規約変換・finite cutoffのscopeで`EQUIVALENT`。
Aの非通常点、all-odd最適性、new RTE一般のclaimはこの同値から得られない。

## 3. P2/PTSC: YES/NO/INCOMPARABLE監査

具体箇所・authors/versionは[一次資料表](rte_reallocation_r05_primary_source_locator_table_v1.md)を参照。
`NO`は読んだ構成・定理からの直接parameter/corollary命題に対する判定である。
未読論文を含む不存在証明、一般化の不可能性証明ではない。

| 命題 | 答え | 対応・反対理由 |
|---|---|---|
| Z1: 全adjacent-order A-familyをPTSCのparameter choiceで直接取得 | NO（m≥3の一般family） | K=0はidentityとorder1だけをpairし、残りはpure words。K>0はremainder leading groups。各kのa_k+b_(k-1)=t_kという重なりを持つ自由係数列がない。m=1通常点は特殊な一致 |
| Z2: identity配分/common etaSigmaからall-odd attaining解を直接取得 | NO | 既知のEuler/common-angle原理は使える。しかし偶奇を交換したvectorの共線条件と有限末端を同時に解き、非負性を示す構成は別途必要。PTSCの固定leading-group配分を代入したものではない |
| Z3: A-class lower bound/attainmentはPTSC定理の直接corollary | NO | Props.3/4/10はそのLCUのnormalization上界・remainder誤差。Aのrestricted feasible classの最小値・全奇数次数の達成を与える命題ではない。似たEuclidean式だけではcorollaryとしない |
| Z4: K>0 compensationと固定P_mを同target扱い | INCOMPARABLE | V_K=U S_K†と、そのtruncated compensation times S_Kは一般にP_m(U generator)そのものではない。remainderの小さいnormalizationをAの優劣に使わない |
| Z4: K=0、lambdaをxへ吸収、s_c=mへ固定したsame-target比較 | YES | S0=IなのでEq.48を有限化した平均はP_m。同じtargetで以下の生成法を比較できる |
| Z5: ゼロ次PTSCのPauli条件をgeneral involutionへ拡張 | YES | I-i x QのEuler化と残るunitary wordsだけ。Q²=Iで同じ証明がそのまま通る。Aだけが一般involutionへ対応するとは主張しない |
| Z5: higher PTSCのatom分類/rephasingをI0へそのまま移植 | NO | Pauli wordはphase付きPauliに閉じ、Hermitian/anti-Hermitian分類ができる。一般wordはそのいずれでもない。remainderのHermiticityだけでは個々のwordを同じrotation atomにできない |

同targetのゼロ次PTSCは

\[
M=\sum_\ell p_\ell(I-i\sigma xQ_\ell)
+\sum_{n=2}^{m}(-i\sigma)^n t_n\,\mathbb E[Q_n\cdots Q_1],\qquad
\mu_{Z0}=\sqrt{1+x^2}+\sum_{n=2}^{m}t_n.
\]

最初のgroupはc0=√(1+x²)、rotation角atan(x)を持つ。残りはcoefficient t_nとphase付きpure word。
canonical確率はgroup coefficient/μ、indicesはIID p。前処理はO(m)、一eventのword生成はO(m)以下
（pのsampler準備と有限precisionは別）。全Taylor wordの列挙は必要ない。
通常pairingはtailでもadjacent pairをEuler化するため、m≥3,x>0で

\[
B_A<B_{\rm ordinary}<\mu_{Z0}.
\]

これは同I0の**logical normalization**差であり、PTSC higher-orderの性能に勝ったとはしない。
両方がO(m)生成なので、ゼロ次PTSCに対する漸近classical取得cost優位は主張しない。
PTSCのcommon parameter、zero-sum rephasing、sampling frameworkは既知部品として残す。

## 4. P3/CTS: C1の式の修正と解析比較

CTSの有限specializationは、zero-degree identityを分離したまま

\[
M=C+I+iS,\quad
C=\sum_{j=1}^{d}(-1)^jt_{2j}\widehat R^{2j},\quad
S=-\sigma\sum_{j=0}^{d}(-1)^jt_{2j+1}\widehat R^{2j+1}.
\]

I1で各Hermitian operatorのPauli係数を集約し、Lc=Σ|C_P|、Ls=Σ|S_P|とすると
μ_coll=Lc+√(1+Ls²)。real atomsはsigned Pauli、imaginary atomsはcommon angleのsigned-Pauli rotation。
zero-degree Iまでreal groupへ吸収し直す追加最適化は、このliteral CTS比較へ入れていない。

### C1: 提示式はgeneral I0 CTSの実装値ではない

単にwordの長さがoddであることからword自体がHermitian involutionだとはいえない。
W=Q0 Q1 Q2は自由involution代数でW†≠±Wであり、

\[
(I-i\beta W)^\dagger(I-i\beta W)
=(1+\beta^2)I+i\beta(W^\dagger-W)
\]

は一般にscalar Iでない。従ってI-i O Wを√(1+O²)倍のrotationとするshortcutは不可。
**Lc=E-1、Ls=OはCTS literal collected coefficientsの一般的な等式ではない。**
これらはraw degree masses/upper boundsであり、式だけをI0の実行可能CTS comparatorとしない。

ただしI1のper-word closureがあれば、次の**明示したpadded CTS型構成**は可能である。
Pauli word W=ζP、ζ∈{±1,±i}について、HermitianならW、anti-Hermitianなら-iWをW'とする。
各degreeでRhat^nはHermitianなのでanti-Hermitian subensembleのweighted sumは0。
従ってE[W']=Rhat^nであり、全word列挙/係数collectionなしで、sampled wordだけを分類できる。
even degree n≥2のsigned W'をmass t_nでpure atomとし、odd degreesをmass Oへまとめ、
Iと共通角atan(O)でpairする。この構成の**指定されたpadded masses**は

\[
L_c^{\rm pad}=E-1,\quad L_s^{\rm pad}=O,\quad
\mu_{\rm pad}=E-1+\sqrt{1+O^2}.
\]

既知PTSCのzero-sum rephasingとCTSのidentity pairingを組み合わせた解析上のspecializationであり、
P3がこのexact samplerをそのまま記述していると主張しない。Pauli closureを必要とするのでI0ではない。
有限targetとμを保つこの構成を、result-prior protocolで登録したI1 padded comparatorとして検査した。

E>1、O>0なら

\[
\mu_{\rm pad}-B_A
=(E-1)\left[1-\frac{E+1}{\sqrt{E^2+O^2}+\sqrt{1+O^2}}\right]>0.
\]

m=1またはx=0なら等号。これはpadded構成との解析比較であり、CTS最良normalizationの下界ではない。

### C2–C5

| 命題 | 答え | 範囲 |
|---|---|---|
| C2: collected CTSがAより小さくなり得る | YES | 固定X/Y/Z fixtureの9条件すべてでμ_coll<B_A。I1の係数cancellationを利用した値。AのI0-class定理の反証ではなく、Pauli域では必要な強い対照 |
| C3: Note3はfull collectionを省けるか | YES | 部分Pauli expansionを独立にsampleして積にするlayering。exact block inputsなら同じfinite meanを保てる |
| C3: I0のみで同じLc/Ls/common-angle分布/μ_collを保証するか | NO（記載手順からは） | 部分Pauli表現とnorm取得が残り、paperはcollectionを逃すとnormが増え得ると明記。meanの一致からnormalizationやaccessの一致は出ない |
| C4: Aの全atomがCTS具体constructionの特殊atomか | NO | 一般Aはrotation times word。固定odd X/Z branchはidentity係数0でY/Zの両係数が非零。signed Pauliにもsingle-Pauli rotationにも該当しない。word=I等の個別特殊一致はある |
| C5: 一般involution/異basis DFへCTSをそのまま拡張 | NO（unchanged construction） | wordのPauli closure/Hermiticity分類とcommon-axis rotation生成がない。新Hermitianization、block encoding、別workspaceを導入すれば別構成・費用であり、無料のequivalenceにしない |

これらのNOは任意の未来のI0 algorithmを排除するものではない。
CTS Markov手順から得られる資源/normalizationを不当に弱く扱わず、部分collectionの取得costも含める。
異basisQ=V†ZVのknown circuitを合成することと、そのwordがglobal Pauliに閉じることは別である。

## 5. Classificationとresult-prior gate

| Claim scope | Classification | 残る差・制約 |
|---|---|---|
| ordinary端点 vs P0/P1 | EQUIVALENT | finite cutoff/sign/adjointを明示。new RTE一般は既知 |
| 共通角でidentityを配分する原理 | DIRECT_COROLLARY | 既知Euler/PTSC/CTSの枠組み。新原理として主張しない |
| A restricted class theorem/closed form vs 読んだP2/P3の具体定理 | MATHEMATICALLY_DISTINCT_BUT_FRAMEWORK_KNOWN | 明示構成/下界対象が別。広いalgorithm noveltyは弱く、publication priorityは未証明 |
| 同target・同I0のA最適端点 vs ordinary / ゼロ次PTSC | METHOD_DELTA_CANDIDATE | normalizationとlogical event familyが具体的に異なる。native cost/momentの交換を調べる仮説はある |
| A I0 generator vs literal CTS collected / per-word padded | METHOD_DELTA_CANDIDATE | Pauli algebraなしの生成、異なるatom class。これはordinary/PTSCより弱いoracleを使うというclaimではない |
| higher PTSC remainder / enhanced PF vs 固定M | INCOMPARABLE_TARGET | targetを合わせない優劣は無効 |
| CTS collected superiorityのI1 fixture | SPECIAL_CASE | 既知CTSの有限specialization。AのグローバルLCU最適性は主張しない |

STOP-Aの完全同値/特殊点/直接corollaryかつ差分なし、という条件は本scopeでは満たされない。
STOP-Bの「restricted theoremだけ・実装差を調べる合理的仮説なし」も、同I0でのnorm差と
追加odd native actionのtrade-offが具体的なので本scopeでは満たされない。
従って利用者が結果前に指定したgateは**CONDITIONAL-R1**。
これはR1必要性をGPTへ返す分類であり、R1 authorizationを与えない。

Blocking/pending: 指定corpusを越える優先性、native rotation/controlled-oracle前提、odd builder、
取得cost・有限precision・強いIS/Pauli対照・実資源materialityの結果前contract。
P3 exact v2 PDFは今回は取得できた。publisher PDFは書誌/headerのみ確認し、
journal本文・supplementとの全文対照やbyte-level論文監査はしていない。
Citation networkの不存在証明はしない。B/DF一般の実益・世界初・論文化はGPTが判断する。

## 6. Fixed symbolic evidenceとSTOP

[事前protocol](../../../artifacts/track_b_rte_reallocation_r05/2026-10-06/result_prior_protocol_v1.json)
にm={3,5,7}、x={1/8,1/4,1}、sigma=±1、p=(1/2,1/3,1/6)と全comparison/semantic witnessを固定。
自由wordとI1の一qubit X/Y/Z fixtureを分離した。結果後の条件/比較追加は0。
[結果とsupport表](rte_reallocation_r05_symbolic_comparison_readout_v1.md)、
[comparison JSON](../../../artifacts/track_b_rte_reallocation_r05/2026-10-06/comparison_v1.json)、
[exact artifact](../../../artifacts/track_b_rte_reallocation_r05/2026-10-06/exact_symbolic_comparison_v1.json)
にmean identity、outward rational norm interval、phase-preserving support、source identityを保存。

技術checker一回、9 norm条件/18 sign比較、約5.968秒、peak RSS 16,640 KiB。
local technical evidenceでありCI・外部再現ではない。Aの一般定理は有限fixtureの外挿ではない。
旧R0 checker/科学runは再実行しない。旧科学結果・marker・authorization・Track Aは変更しない。
R1/sampling/synthesis/compile/matrix/solver/分子/DF/NPZ/GPU=0、共通API変更0。
必要資料をcommit/push後mandatory STOP。次判断は[GPT handoff](rte_reallocation_r05_gpt_handoff_20261006.md)。
