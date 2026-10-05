# 研究B：部分ランダム化時間発展のアルゴリズム設計・最適化

作成日：2026-10-04  
文書の位置付け：研究計画の提案。性能改善・新規性を実証した報告ではない。  
研究証拠の基準：`HIROMU1015/Partially-Randomized-Trotter`、commit `b6e65c6123475add5e620ec1064f361378bead95`。  
現行研究AのM1/M2、過去P-D等の正式status・契約・データは変更しない。

## 0. 方針の結論

研究Bは、研究Aの「既存方式の資源・適用条件を理解する研究」と分け、**時間発展の構成または標本生成法を実際に改善するアルゴリズム研究**とする。

第一候補は、**finite-RTE呼出し負担を含む、native Product Formulaの係数・stage配分設計**（以下B-F）。第二候補として、**平均信号を保持する、回路列costに基づくRTE sampler**（以下B-S）を残す。ただし両者を最初から同時に実装・共同最適化しない。

直近の推奨は、B-Fの次数条件・比較対照・低次元候補familyを確定して小規模な設計可能性を調べること。B-Sは並行して、保存済み情報と既知の最適importance-sampling限界から改善余地を診断する段階までにとどめる。

THRIFT/interaction-pictureとTrotter-error compensationは、別の構成へ進む選択肢として評価するが、B-F/B-Sが不調だから自動で順番に実装するfallbackにはしない。

### 前の計画から変更する点

1. **最適splitが変わることは、成功の必要条件ではない。** 同じsplitでも、正当な新構成が同一精度の資源を改善すれば成果になり得る。逆にsplitの変化だけでは新規性を示さない。
2. **既知アルゴリズムの組合せを一律に弱いと扱わない。** 新しい誤差・費用の両立、適用条件、実行可能な合成手順を実証すれば構成的な貢献になる。
3. **研究Bを汎用algorithm selectorの開発へ限定しない。** 主成果はPF family、sampler、構成法、または独立に有用な限界・設計則でもよい。
4. **研究Aの完成を研究Bの開始条件にしない。** A/Bは並列に進め、共通基盤の変更を採用する時だけ整合性を確認する。
5. **過去P-DのSTOPを全PF最適化の否定に拡張しない。** P-Dは既存候補上でfinite-model refinementが選択を改善するかを調べた。新たな係数familyを構成するB-Fとは検証対象が違う。

---

## 1. やりたいことと研究Aとの境界

### 研究A

固定したDF表現と既知方式のclassで、residualのdiscard、deterministic保持、finite-RTE補完を比較し、accuracy・shot・回路費用の競合を説明する。既知の適切なbaselineは強化してよいが、新しいアルゴリズムの完成をAの完了条件にしない。

### 研究B

決定論側とランダム側が接続する構造を設計対象とする。例えば、PF係数を変えると近似誤差だけでなく、tailの出現回数・符号付き時間・細分化数・normalization・回路費用が変わる。この結合を使って、既存の最適化済み方式より良い時間発展または標本生成手順を作れるかを問う。

主目的は「多くのアルゴリズム名からwinnerを選ぶ」ことではない。最終的には、他の入力にも適用できる設計手順と、その有効範囲を残す。

Bの改善がAの固定baselineへ後から混入してはいけない。Aの旧結果は保存し、Bの手法をAへ導入する場合は別version・別比較にする。

---

## 2. 既存証拠から確実に引き継ぐもの

### 2.1 M1/M2

M2は固定5構成のtransferで`TRANSFER_SUPPORTED`。これは研究Aの証拠であって、まだBの新PF・新samplerの有効性を示していない。[R1]

H4 1.00 Åと1.30 Åは、Bでは既知のdevelopment/exploratory情報として再利用できる。Bの手法をそこで設計した後、同じ条件を新しいblind evidenceと呼ばない。

### 2.2 P-D

保存済みS1の事後解析では、内部workを数えるB1b、leading model B2、finite model B4の選択が一致し、各scopeのB4 regretは0だった。一方、全候補のfeasibilityが一致したわけではない。停止理由は「既知の候補選択にfinite補正を追加する独立した利益が出なかった」ことである。[R2]

したがってB-Fでは、同じ既存PF一覧をもう一度並べてB2/B4差を探さない。**候補familyそのものを生成・最適化する**。さらに、内部workを無料とする対照、過剰精度の対照、nested/nativeの異なる受理規則による見かけの差を避ける。

### 2.3 P-A、R3、FR

P-Aのinterval-DP、R3のcheap selector、FRのcertificateは、過去の停止理由を保持する。[R3]

B-Sは既存回路列の順序を変えるcompilerではなく、**列を生成する確率分布と補正weightを変更する**候補である。ただし、これが新しいというだけで採用せず、一般importance sampling・逐次samplingの既知手法との比較が必要。

過去pilotのSTOPはその仮説・対象classに対する判断であり、関連する研究領域全体の不可能性証明ではない。

---

## 3. 比較する技術方向

| 方向 | 何を変えるか | 魅力 | 最大の問題 | 今回の配置 |
|---|---|---|---|---|
| B-F：PF係数・tail配分 | 誤差次数条件を守る係数、stage配置、RTE配分 | 既存PF/RTE基盤から低次元の設計研究へ進める | PRの絶対時間則、既存最適化PF、SPRINTとの近接 | 暫定主線 |
| B-S：回路列sampler | 軌道分布、既知weight、有限shot評価 | 平均信号を保ったまま費用を変えられる | 理想分布は既知、normalizer・weight・sampling費用が難しい | 第二候補。先に改善余地だけ診断 |
| B-I：THRIFT/IP | 参照Hamiltonianとinteraction-picture構成 | scale separationを別の形で利用 | 安く実装できるH0とH0+αh_jが必要 | primitive成立性が見えた場合の別案 |
| B-C：PF error compensation | 物理residualではなくPFの欠陥を補正 | 近似器とrandom補正の役割を変更できる | LCUによるTrotter補償が既知、補償演算子の複雑化 | 先行法を超える具体差分がある場合だけ |

qDRIFT・randomized PF・multiproductを無条件に除外するわけではない。ただし今回の主線にすべてのalternativeを入れない。coherent-signal taskとの対応を先に固定し、必要な対照として採用する。

---

## 4. 最接近研究と、新規性として使わないもの

### PR/RTE

Güntherらは部分ランダム化をcoherent Hadamard信号として解析し、高次PFのtail絶対時間とRTE細分化配分も扱っている。qDRIFTを位相推定に使う解析もある。[W1]

従って、以下を新規性にしない。

- 高次PFとRTEを接続したこと。
- 負時間係数に対して絶対値が費用へ入ること。
- `sum |b_j|`または比例RTE配分の導出。
- qDRIFTではなくRTEを使うだけの変更。

### 最適化PF・QPE用PF

Moralesらは係数探索、異なる長さ・次数の公平比較、processed/non-processed formulasを扱う。[W2]
HejaziらはQPEのenergy誤差に特化したPF解析と公式を与える。[W3]

従って、数値最適化した係数、BCH条件、energy-aware objectiveだけでは差分にならない。

### SPRINT

SPRINTは高次をdominant groupへ、低次を小さいgroupへ、さらに小さいremainderへqDRIFT/RTE等を使う構成を明示する。誤差・係数・実装を組み合わせる一般思想は既知である。[W4]

従って「algorithm allocation frameworkを提案する」と大きく名付けるだけでは弱い。B-Fでは、限定familyの具体的設計と、有限RTEを含む実測task費用で残る差分を示す。

### importance sampling

Cuginiらは任意の回路costに対するnet-cost最適分布を与える。この一般結果はcostをevent単位の和に限定しない。[W5]

従って「DFのcostは非加法的なのでcost-aware samplingを考えた」だけでは差分にならない。B-Sの差分候補は、全列挙・全compileなしに生成可能で、weightと平均信号を正しく扱う具体的sampler、計算量、有限shot保証である。

### THRIFT/IPとerror compensation

THRIFTではH0とH0+αh_jの時間発展を要求精度で効率的に実装できることが重要で、DFの複数fragmentの和だから自動的に満たされるわけではない。[W6]
interaction-pictureのhybridization自体にも先行研究がある。[W7]

ZengらはLCUによるTrotter error compensationを提案している。[W8]
2026年のHNCCはchannel-levelの補償であり、そのままcoherent first momentの補償になるとは仮定しない。[W9]

### 本調査の限界

重点文献を確認したが、同一の目的関数・family・samplerが文献全体に存在しないことは証明していない。新規性候補は、B-Fの係数・手順が具体化した段階で式とalgorithm単位で再照合する。文献調査が無限に続くことを避けるため、明白な重複がなければ最小の反証可能なpilotへ進める。

---

## 5. 共通の数学的契約：同じ何を計算するか

### 5.1 研究Bの初期task

提案する初期taskは、研究Aと比較可能な有限時間のcoherent signal

\[
z_H(T)=\langle\psi|e^{-iHT}|\psi\rangle
\]

を、複素絶対精度ε_sig、失敗確率αで求めること。

これは研究の開始scopeの提案であり、全RPE/QPEや化学精度のenergy推定を達成したという意味ではない。Bで新しい構成が成立してから、必要ならその別taskへ接続する。

state、Hamiltonianのscalar phase、T、target、要求精度は対照と一致させる。各方法のq、r、K、許される内部精度は、それぞれに選ばせる。

### 5.2 ランダム手法の共通interface

ランダム回路U_ω、分布P_x(ω)、既知補正weight W_x(ω)について

\[
M_x=\mathbb E_{P_x}[W_x(\omega)U_\omega]
\]

を明示する。physical targetとの系統誤差はM_xとe^{-iHT}の比較で扱う。

Hadamard測定Y_a∈{−1,+1}を用いる場合、軸別の補正した標本X_aの期待値・二次モーメント・最大値を定める。複素weightを使う手法では測定phaseまたは実虚部の混合を含めて定義し、実weightの式をそのまま流用しない。

finite-RTEで「unbiased」と書く場合は、**固定した有限Taylor平均信号に対して不偏**なのか、exact evolutionに対して不偏なのかを明記する。有限cutoffのphysical biasは通常残る。

### 5.3 channel一致だけでは足りない

\[
\mathbb E[U_\omega\rho U_\omega^\dagger]
\quad\text{と}\quad
\mathbb E[U_\omega]
\]

は別の対象である。例としてUと−Uの50/50混合は、system channelではUと同じだが平均operatorは0。

これは基本的な代数例で、qDRIFTが使用不可能という意味ではない。qDRIFTは実際にcoherent QPE解析がある。[W1] 必要なのは、ancillaも含めたcontrolled実装とfirst momentの対応である。

### 5.4 同じ科学的精度、同じ保証水準

比較の主資源は

\[
G_x=\sum_aN_{x,a}\,\mathbb E[C_{x,a}]
\]

とする。回路count/depth、最大回路長、ancilla、古典preprocessingは別項目にする。

RZ-countとFT T-count、総depth-workと最大single-circuit depthを混同しない。追加ancillaの使用を一律禁止せず、使った資源として数える。初期に現行gate setを使うのは比較接続のためであり、一般ハードウェア最適性を仮定するためではない。

---

## 6. 主線B-Fの研究計画

### 仮題

**有限ランダム時間発展の実行負担を考慮した積公式係数の設計**

英語案：*Product-formula coefficient design for partially randomized coherent simulation*

### 主RQ

> 同じ次数条件を満たすnative PF familyの自由度を、PF誤差だけでなくfinite-RTEの呼出し数・符号付き時間・normalization・実装費用まで含めて選ぶと、既存の最適化済み公式を再調整したものより少ない資源で同じcoherent signalを推定できるか。

### 副RQ

- どの誤差係数を下げることが、どのRTE負担増加と競合するか。
- 既知のleading absolute-time modelで十分か。十分ならそれを設計に使い、有限補正が不要な結果を失敗扱いしない。
- 開発データで設計した係数または選択手順が、未使用のtask/instanceでも有効か。

**split変更はsecondary。主評価は同一taskの資源と、再利用可能な構成的差分。**

### 6.1 最初から全次数を探索しない

最初は4次familyの5段・7段対称compositionに限定する案がよい。これは8次が悪いという結論ではない。自由度を1〜2次元へ落とし、なぜ設計が変わったかを追うための出発点である。8次の既存有力公式は性能対照として残す。

研究成果に4次公式を必須とするわけではない。明確な限界が見えた場合の別familyへの移行は、その理由を記して別実験として判断する。

### 6.2 実際に実装するprimitiveから組む

\[
H=\sum_{\ell=0}^{L_D}D_\ell+R,
\]

とし、D_0をone-body、他D_ℓを保持DF fragment、Rを残差とする。各D_ℓの実装可能な時間発展を使い、理想的なnative二次kernelを

\[
S_2(h)=
\left(\prod_{\ell=0}^{L_D}e^{-ihD_\ell/2}\right)
e^{-ihR}
\left(\prod_{\ell=L_D}^{0}e^{-ihD_\ell/2}\right)
\]

と置く。HD全体のexact exponentialを無料oracleとしない。

その後

\[
S_{\mathbf w}(h)=S_2(w_1h)\cdots S_2(w_mh)
\]

を作る。real対称係数、\(w_j=w_{m+1-j}\)について

\[
\sum_jw_j=1,\qquad \sum_jw_j^3=0
\]

を満たせば、self-adjointな二次kernelのcompositionとして、理想的なtail exponentialを用いる場合に4次になる。これは標準composition理論であり、新規性ではない。[W2,W6]

5段例：

\[
(w_1,w_2,w_0,w_2,w_1),\quad
w_0=1-2(w_1+w_2),\quad
w_0^3+2w_1^3+2w_2^3=0.
\]

一般に1自由度。7段では独立な自由度が2つとなる。退化点・Jacobian rank低下は別扱いにし、全領域を一つの滑らかなchartと仮定しない。

既知Suzuki/Yoshida構成を初期点・対照として含める。符号・stage数・zero stageの扱いを明記する。

### 6.3 formal orderとfinite-RTE誤差を分ける

理想kernelの中央e^{-iw_jhR}を有限Taylor平均へ置換したものは、もとのunitary compositionと同じではない。\(P_{K+1}\)による平均を使うなら

\[
\nu_x=\langle\psi|M_x|\psi\rangle
\]

を直接定義し、\(\nu_x-z_H\)でtaskを評価する。

fixed r_jの下で1 microstepのTaylor残差はO(s^{K+2})。固定T、q反復の通常の小step評価では、累積残差は概ね

\[
O\!\left(\frac{T^{K+2}}{q^{K+1}}
\sum_j\frac{|w_j|^{K+2}}{r_j^{K+1}}\right)
\]

という次数を持つ（Hamiltonian norm等を定数へ含めた説明用次数評価）。K=2のTaylor三次を固定して使えば、理想PFが4次でも全体の誤差が4次とは限らない。

初期にK∈{2,4}を許す案とし、K=2が要求signal精度を満たせば使用できる。ただしその全体を無条件に4次algorithmとは呼ばない。

### 6.4 係数がRTEへ及ぼす作用

T=qh、stageごとの細分化をr_j、cutoffをK_jとすると、独立samplingのもとで

\[
\log\mathcal B_x
=q\sum_jr_j\log B_{K_j}
\left(\frac{\lambda_R|w_j|T}{qr_j}\right).
\]

完全に同じgeneratorの隣接因子を正確に融合できる場合は、**融合後のtail occurrence list**から評価する。特にHD=0やzero stageなどの退化で、不要なtail往復を残した対照を作らない。

\(\Gamma_R=\sum_j|w_j|\)と比例r_j配分は既知。[W1] これを再導出するだけでなく、誤差と回路費用を同時に動かす係数を実際に構成する。

小さい局所時間の領域では

\[
\log\mathcal B_x\approx
\frac{c_K\lambda_R^2T^2}{q}\sum_j\frac{w_j^2}{r_j}.
\]

c_Kは使用するnormalizationの展開から確認する。有限Kで常に同じ定数とは置かない。

### 6.5 最小化する量

補正信号の各軸biasを\(b_{x,a}\)、\(s_{x,a}=\epsilon/\sqrt2-b_{x,a}>0\)とすると、canonical finite-RTEで現行Hoeffding規則を採用する場合

\[
N_{x,a}=\left\lceil
\frac{2\mathcal B_x^2}{s_{x,a}^2}\log\frac{2}{\alpha_a}
\right\rceil.
\]

設計目的は

\[
\min_{\mathbf w,q,\{r_j,K_j\}}G_x,
\quad \text{subject to order conditions and signal accuracy}.
\]

誤差最小、Γ最小、stage数最小だけではない。固定次数familyでの小さいcontinuous探索と、q/r/Kの有限探索を組み合わせる。毎iterationでQiskitを呼ばず、固定した合成規則のcost構造から候補を絞り、最終候補でdirect compileする。

### 6.6 設計の具体的方法案

1. 順序条件を制約として保持し、5段の1次元曲線または7段の2次元面を生成する。多項式残差を単にpenaltyにして次数を犠牲にしない。
2. native kernelのBCH誤差を係数関数とHamiltonian作用に分ける。leading誤差をscreeningに使い、有限Tで直接検査する。
3. 既知の比例配分をr_jの基点にし、総microstep R_totと局所cutoffの必要性を評価する。
4. 係数・integer allocation・costを交互に調整する場合も、同じ計算budgetで対照の再最適化を行う。
5. 少数の非劣位係数を保存し、同じfinite-RTE平均でbiasとNを確認する。
6. 合成policyを固定しdirect compile。proposalと対照に同じcompiler改善を認める。
7. 係数または選択手順を凍結し、未使用条件で確認する。

これは設計案であり、この最適化が既存公式を改善すると確認した結果ではない。

---

## 7. B-Fの新規性を判定する基準

### 新規性にならないもの

- 二次を既存四次/八次へ変更したことだけ。
- energy bias最小の過剰精度候補に対して低costとなること。
- finiteモデルの方が詳しいこと。
- 「tailの負時間が高くつく」という既知説明。
- generic optimizerを呼び、H4一点で係数が変わったことだけ。

### 成果候補1：再利用可能な係数family/設計手順

同じ次数・比較可能なstage budgetのもとで、強い既存公式＋各自再最適化に対し資源上の改善があり、その構成を他条件でも再利用できる。係数の数値一覧だけではなく、入力、生成手順、適用条件、cost・errorの交換関係を公開する。

### 成果候補2：設計原理と、その必要性を示す対照

同じfamilyをPF誤差・通常costで最適化した場合と、RTEの負担を戻して最適化した場合を比較し、何を変えたことが利益を生んだか示す。

もしleadingモデルとfiniteモデルが同じ係数を出しても、**leadingモデルで新しい良い係数を作れたなら、設計研究は成立し得る**。finite補正の優位性を必須にしない。

### 成果候補3：限定classの非自明な設計限界

familyとresource modelを明示し、既存公式で十分な領域や、stage追加の利益を得られない理由を新しく定量化できれば限界研究を検討する。単なるoptimizerの不成功や、有限gridで良いものがなかっただけでは下界・最適性としない。

### 新規性の判定状況

現段階ではいずれも**仮説**。PRの既知normalization則と既存composition理論から出発するが、それだけで独立したmethod noveltyが確定したわけではない。

---

## 8. B-Fの対照設計

| 対照 | 固定/最適化すること | 目的 |
|---|---|---|
| 現行native S2＋finite RTE | q/r/Kを独自最適化 | 元構成に対する追加価値 |
| 既存の有力四次/八次＋RTE | 各公式でq/r/Kを最適化 | 既知formula選択だけで十分か |
| 同じ5/7段familyの通常PF設計 | 同じ自由度・探索budget、PF精度と回路costを最適化 | 追加自由度の利益とRTE-aware設計の利益を分離 |
| leading-RTE-aware設計 | 既知Γ・比例配分を利用 | finite補正が本当に必要か |
| fully evaluated候補 | 同じfinite平均・actual costで採点 | 共通参照 |
| full deterministicとdiscard | 必要な範囲で同様に最適化 | partial自体の価値を論じる時の対照 |
| 適用可能なnear-integrable構成 | その構成に適した設定を許す | SPRINT等の既知設計を無視しない |

すべてを最初の直積gridにはしない。中心claimに必要な対照を選ぶ。

「既存partial実装を改善した」と「強いdeterministicにも勝った」は別claim。前者だけでも独立成果になり得る。

### 必須ablation

- 係数だけを変更、sampling allocationは固定。
- r/Kだけを変更、係数は固定。
- 係数＋allocationを変更。
- compiler policyは共通。

qも最適化する最終比較と、寄与を見る固定q比較を別表示する。どちらも同じ物理Tを使う。

---

## 9. B-Fの最小pilot

### 段階BF-0：研究設計・代数検査

入力は既存source、既存JSON、文献。新しい分子NPZを読まず、次を作る。

- native kernelのsymbolicな作用列。
- 5段/7段familyと次数条件。
- 既知対照と許す自由度の表。
- finite平均operator、normalization、q/r/K配分の定義。
- floating-point係数の次数残差と、誤差予算への算入法。
- 小行列で検査すべきnoncommutative semantics一覧。

必要な代数sanity testは行ってよい。分子signal/trajectory/compileへ自動進行しない。

### BF-1：小さい設計可能性試験

新しい科学計算を始める場合の提案：H4 1.00 Åの既存snapshot、初めはLD=3固定、native構成。5段familyを主探索、7段は5段で見えた制限を調べる場合に限る。

T=0.8はAとの接続点として残す。精度の例としてε_sig∈{0.05,0.01,0.002}、q∈{1,2,4,8,16}を**初期候補案**とする。これらは実測に基づく有望点ではなく、実行前に採用・予算確認するproposal。

q=1で既存法が十分な条件では、高次数による反復数削減余地がないことを記録する。そこを除外して改善点だけを残さない。同時に、Aの緩い精度だけでB全域を棄却しない。

有限RTE平均を行列/state-actionで評価し、まだtrajectory/compileを多数作らない。candidate budget、初期seed、上限到達時の扱いを固定するが、exploratory optimizerの各iterateを新しい事前登録にしない。

### BF-1の主出力

- 実行可能な係数曲線とPF誤差・Γ・normalization・action costの関係。
- 対照再最適化後にも残る資源改善候補。
- 誤差とcostのどの項が変わったか。
- q=1で効果が小さい領域、finite-RTE誤差が支配する領域。
- oracle情報を使った部分の表示。

### BF-1後の判断

- 有効な係数候補と説明が残る：BF-2へ。
- 既知公式を同じ条件へ調整するだけで全効果が消える：B-Fの当該familyを閉じる。
- 意味論・次数条件が壊れる：修正し、性能結果と分離。
- 探索境界のため不明：その境界が主claimに効く場合のみ一段の追加案をreview。

「最適splitが同じ」「finiteモデルがleadingモデルと同じ」は単独ではSTOP条件にしない。

### BF-2：実回路による反証

最終候補と強い対照だけをfull controlled Hadamard wrapperへ接続する。新policyは双方へ適用する。

32 trajectoryは候補比較の開始案にできるが、旧M2の196 wrapper枠をそのまま再使用しない。必要cell×trajectory×axisで新しい予算を明示する。必要な不確かさが分かる前に追加96を既定路線にしない。

異なる係数で分布が変わる場合、単にseedが同じだから同一trajectoryとはしない。couplingを使うなら対応関係を定義し、候補内の各occurrence独立性を保つ。

### BF-3：未使用条件

BF-1/BF-2で手法を固定した後に、未使用のgeometryまたは小さい別系を一件以上選ぶ。対象は既存snapshot inventoryと古典計算量から決める。H6は候補であって採択済みではない。

二種類を区別する。

- **係数transfer**：係数を固定して移す。
- **設計手順transfer**：係数は入力に応じて変わるが、規則・入力情報・探索budget・選択法は固定。

後者がheld-outのexact targetを参照して調整するなら、それはoracle benchmarkであり実用手順のtransferではない。

---

## 10. 第二候補B-S：平均信号を保存するRTE sampler

### 主RQ

> 指定した有限RTEの平均信号を変えずに、DF basis/supportの列構造を使ってsampling分布を生成し、weightによるshot増加と古典生成費用を含めても必要資源を減らせるか。

### 10.1 変えてよいもの・変えないもの

固定candidateのcanonical trajectory分布をP、回路をU_ωとする。Q(ω)>0 whenever P(ω)>0を守り、

\[
X_a=\mathcal B\frac{P(\omega)}{Q(\omega)}Y_a
\]

を用いれば

\[
\mathbb E_Q[X_a]=\mathbb E_P[\mathcal BY_a]
\]

で平均信号は保存される。これは標準importance sampling。[W5]

新しいQのもとで回路内eventが相関しても、**元の全trajectory確率との正確な比**を使えば同じ平均を回復できる。marginal確率だけを合わせる、weightを付けずに同basis eventを増やす、noncommuting eventを勝手に並べ替える方法とは違う。

この変更でfinite-Taylor physical biasは消えない。

### 10.2 先に既知の改善上限を見る

補正前Y_a∈{−1,+1}、C(ω)>0のもとで、worst-case second-moment型のnet-costは

\[
J(Q)=\mathcal B^2\mathbb E_Q[C]\sum_\omega\frac{P(\omega)^2}{Q(\omega)}.
\]

既知の最適分布・最小値は

\[
Q^*(\omega)\propto\frac{P(\omega)}{\sqrt{C(\omega)}},
\qquad
J^*=\mathcal B^2(\mathbb E_P\sqrt C)^2.
\]

よってcanonicalに対する比は

\[
\frac{J^*}{J(P)}=
\frac{(\mathbb E_P\sqrt C)^2}{\mathbb E_PC}
=1-\frac{\mathrm{Var}_P(\sqrt C)}{\mathbb E_PC}.
\]

これらはCuginiらの結果であり、新規性ではない。[W5]

保存trajectoryのcostから見積もれるが、32件の推定を全分布の厳密下界・上界にはしない。希少eventは既知分布と有限supportの情報で確認する。raw記録がremoteに無ければ、Codexがローカルで集計した軽量summaryを使い、新規compileで埋めない。

この最適値でも改善余地が小さいなら、同じ回路family内のsamplingだけへ大規模投資する理由は弱い。ただしこれはJのoracle値であって、状態依存varianceや他の推定器を含む全手法の普遍下界ではない。

### 10.3 実行可能なsamplerの具体案：有限状態のcost-tilt

まず一般Qiskitのglobal costを有限Markov状態で正確に表せるとは仮定しない。明示したstreaming synthesis規則に限定し、state sに現在のbasis/frame、必要support、pending operation等を含める。

有限長event列のcanonical生成を

\[
P(\omega)=\prod_j p_j(e_j|s_{j-1}),
\quad s_j=F(s_{j-1},e_j)
\]

とする。有限cutoffの可変長列はterminal/padding stateを使って表現する。random effectへ関係するstateを省略してPを誤って再現しない。

cost surrogateを

\[
\widehat C(\omega)=c_0+\sum_jc_j(s_{j-1},e_j)
\]

と定め、

\[
Q_\beta(\omega)=
\frac{P(\omega)e^{-\beta\widehat C(\omega)}}{Z(\beta)}
\]

を考える。

backward message

\[
h_{j-1}(s)=\sum_e p_j(e|s)e^{-\beta c_j(s,e)}h_j(F(s,e))
\]

から正規化定数とconditional proposalを計算できる。terminal h_J=1（終了costを別途含む）。各stateのoutgoing edge総数をEとすると、固定Jについて概ねO(JE)のpreprocessingとなる。ただしstate数そのものが巨大化すれば利点は失われる。

Qの生成に使ったsurrogateとactual compiled costが異なっても、**Qの確率とP/Qが正しければ平均信号は保存される**。変わるのは資源改善の保証である。

このtilting/DPは既知の逐次samplingに近く、形式だけで新方法とはしない。[W10] DF固有の実用state、controlled semantics、normalizer、weight、有限shot、古典費用まで閉じることが差分候補となる。

### 10.4 この案で検算できる簡単な関係

surrogateが実際のCと一致し、固定有限supportなら

\[
Z(\beta)=\mathbb E_Pe^{-\beta C},\quad
J(Q_\beta)=-\mathcal B^2Z'(\beta)Z(-\beta).
\]

従って

\[
\left.\frac{dJ(Q_\beta)}{d\beta}\right|_{\beta=0}
=-\mathcal B^2\mathrm{Var}_P(C).
\]

非定数costなら、小さい正のtiltでこのJは局所的に改善する。これは本計画での初等的導出であり、新定理・実測効果ではない。古典費用、finite-confidenceのrange項、実costとの不一致まで含む改善を保証しない。

### 10.5 weight安定性と有限shot

独立stepごとにimportance samplingすると、likelihood ratioが列長に対して積になる。boundedな一step weightでも全列で巨大化し得る。

対策案は、全trajectory分布にcanonicalを混ぜる

\[
Q_\kappa=(1-\kappa)Q_\beta+\kappa P,\quad 0<\kappa\le1
\]

で、P/Q_κ≤1/κを確保する。これは既知のdefensive mixture型の考えであり、式自体を新規性にしない。

通常の自己正規化

\[
\sum_iw_iY_i/\sum_iw_i
\]

へ黙って置き換えない。これを使う場合は有限標本biasと保証を別途導出する。

variance×costの減少だけで、現行Hoeffdingの必要Nが下がるとは限らない。rangeとsecond momentを使う保証を採用する場合は、同じ保証方式をcanonicalにも適用する。

例えば既知のV≥Var(X)、|X−EX|≤Mなら、Bernsteinから

\[
N\ge \frac{2V+2Ms/3}{s^2}\log\frac{2}{\alpha}
\]

が一つの十分条件となる。proposalでは|X|≤B/κよりM=2B/κを使える。Vの計算または上界が重すぎれば、その費用も判定に含める。

### 10.6 B-Sの最初の判別

1. 保存costからJの理想改善余地を診断する。
2. 小さいsynthetic有限event空間で列挙し、Q、normalizer、P/Q、first moment、second momentを確認する。
3. 同じ小空間で理想Q*、simple eventwise、rejection-based baseline、cost-tiltを比較する。
4. 有限shotの資源と古典costまで改善余地が残る場合だけDF trajectoryへ進む。

B-Fの結果を待たなくても1は可能だが、B-FとB-Sの同時大規模実装は当面しない。

---

## 11. B-I / B-Cを今すぐ主線にしない理由

### THRIFT/IP

THRIFTは「HDのsolver名」ではなく、H0と各perturbationを組み替える構成として扱うべきである。HDが複数の異なるDF basisのfragment和なら、そのexact evolutionを安価に使えるとは限らない。

one-body H0のGaussian evolutionは候補になるが、H0+αh_jの構成費用、interaction-picture下での作用の複雑化、時間積分・制御位相まで戻して評価する必要がある。[W6,W7]

従って、最初のtaskは「THRIFTを全部実装する」ではなく、一つの実際のDF primitiveに対して、必要な演算と費用が閉じるかを確認することである。

### Trotter error compensation

\[
V_{\theta}(h)=e^{-ihH}S_{\theta}(h)^\dagger
\]

をrandom/LCUで補正する思想は既知。[W8] これを使えば物理tail RそのものではなくPFの欠陥を補償できるが、commutatorがDF classに閉じるとは限らない。

小さいnormの補正でも、term数、作用support、LCU normalization、controlled回路、古典生成costが増える可能性がある。channel-level HNCCを用いる場合も、signal first momentへの接続を別に示す。[W9]

この方向は新しいDF-native補償の構成が見えた場合に独立提案として扱う。既知補償法の付加だけでは新規性としない。

---

## 12. 研究全体の進行計画

| milestone | 作業 | 科学的な完了条件 | 不調時 |
|---|---|---|---|
| B0 | B-Fのfamily/意味論/対照を固定、B-Sの既知headroom確認 | 何を作り何と比べるかを一文と式で特定 | 明白な重複なら対象を閉じる |
| B1 | B-Fの低次元探索と有限平均検証 | 有効な係数と資源改善候補、寄与の説明 | 当該familyの限界として記録 |
| B2 | 少数候補をactual wrapperへ接続 | 対照再最適化後にも説明可能な利益 | compiler由来ならclaimを修正 |
| B3 | 手法を凍結し独立条件で確認 | 固定係数または固定手順の有効性 | transfer失敗の原因とscopeを特定 |
| B4 | 原稿・再現資料へ統合 | method、根拠、benchmark、限界が一貫 | 不足claimを削り完結範囲を選ぶ |

研究AはPM-0〜PM-2を別に進める。A/Bの週数・人員配分はここでは仮定しない。Bの各milestoneに必要な追加計算量が分かった段階で配分する。

すべてのmilestoneで監査書を何枚も作ることを目的にしない。探索はexploratoryとしてまとめ、確認実験へ移る点でsource・手法・データ・判定をfreezeする。

---

## 13. GO / STOP / INCONCLUSIVEの扱い

### GOに必要なもの

- 平均operatorまたは信号の意味論が正しい。
- 比較対照の精度、回路scope、最適化自由度、利用情報が適切。
- 明示した新しい構成/手順、または独立した設計原理がある。
- 誤差やcostだけを片側で見るのではなく、同一taskの資源または評価負荷へ利益がつながる。
- 開発条件での偶然だけではないことを確認する計画がある。

### GOの必要条件ではないもの

- 最適splitが変わること。
- 四次/八次であること。
- finiteモデルがleadingモデルより良いこと。
- 新しい量子primitiveの発明。
- すべてのsimulation algorithmに勝つこと。
- operator-normの大域最適性。

### STOPにするもの

既知の対照へ同じ自由度を戻すと利益が消え、新しい説明・構成・限界も残らない場合。あるいは、実行不能なoracle、過大なclassical preprocessing、weightの爆発に依存する場合。

新規性が未確定というだけで永久にpilotを禁止しない。一方、条件を増やしてpositive resultだけを集める進行もしない。

### INCONCLUSIVE

sampling誤差、optimizerの局所性、探索端、実装不一致、データ不足を分ける。追加計算を行う場合は「何の判断を変えるためか」と上限を明記する。r4/r8の細かい順位のためだけに高統計化しない。

### 改善率閾値

旧M2の10%はそのtransfer契約の判定であり、Bの新規性閾値ではない。Bでは改善の大きさ、適用範囲、理論/手法の内容、不確かさを合わせて判断する。実務上のmaterialityを数値化する場合は、pilot前または確認実験前に対象metricとともに固定する。

---

## 14. 最小着地点・目標着地点・発展先

### B-Fの最小着地点

明示した有限family内で、次数とfinite-RTE信号を正しく扱う設計法を与え、同じ自由度を持つ強い対照に対して、その有効条件と改善または非自明な限界を示す。少数の新しい係数だけでなく、生成・選択手順が残ること。

### B-Fの目標着地点

RTEを含む資源に対して設計した少数PFまたはparameterized familyが、複数の未使用条件で同一signal精度のcostを改善する。係数、tail配分、誤差、basis costのどの結合を使ったかを説明し、利用できる入力情報と古典設計費用を示す。

### B-Sの最小着地点

定義した有限状態回路classで、列挙なしに生成可能なQと正確なweightを構成し、coherent signal保存、weight安定性、計算量、有限shotのtrade-offを示す。一般IS式の再導出だけで終わらない。

### B-Sの目標着地点

固定したsamplerを新しいDF列へ移し、強いcanonical/既知IS対照に対して、classical overhead込みで必要資源を削減する。cost surrogateが外れても平均の正しさと効率の問題を切り分ける。

### 発展先

PF係数とsamplerの同時最適化、splitの共同最適化、近似入力状態、複数時間のenergy推定、他factorization、FT resource。いずれも第一成果の完成条件に一括で入れない。

特に「B-Fができたら自動でB-Sを合体する」とはしない。片方だけで独立して閉じられる設計にする。

---

## 15. 想定原稿と主図

### B-F本文

1. finite-RTEを含むPF設計の問題設定と既知範囲。
2. native family、次数条件、平均信号とresource model。
3. 係数設計法と対照・ablation。
4. 開発条件のerror/resource frontier。
5. actual compiled costと独立検証。
6. 失敗条件、利用可能情報、energy-taskへの限界。

主図案：

- 同じ4次family上のPF error、Γ、B、costの非一致。
- error/costのみ設計した係数とRTE-aware設計した係数の対照。
- q/r/K再最適化後のG対ε_sig。
- 係数のみ、allocationのみ、共同設計のablation。
- frozen手法の独立条件でのresource ratioと失敗点。

### B-S本文

既知の理想IS限界を出発点に、実際に生成可能な列分布、weight/first-momentの正当性、有限shot、classical complexity、actual costを順に示す。B-Fと一論文に統合する必要はない。

---

## 16. リポジトリと実装の境界

本計画は既に分離作業が完了しているとは仮定しない。必要なら次の論理構造へ合わせる。

```text
docs/tracks/algorithm_codesign/
    README.md
    research_plan.md
    prior_art_matrix.md
    bf_kernel_contract.md
    evidence_index.md

src/trotterlib/experimental/algorithm_codesign/
    pf_family.py
    stage_ir.py
    finite_mean.py
    objective.py
    # B-S採択後のみ sequence_sampler.py

scripts/algorithm_codesign/
artifacts/algorithm_codesign/
tests/tracks/algorithm_codesign/
```

具体pathは現repoの構造を見て決める。既存共通RTE/PF関数はread-only再利用を基本とし、APIの一般化が必要なら小さな独立commitで提案する。

研究Bのcandidate identityには、少なくともtrack、Hamiltonian snapshot、state、task、family、係数precision、stage IR、r/K、sampling law、weight law、compiler policy、seedを含める。

同じrowにAのresultとBのresultを混ぜない。旧M2 authorizationは消費済みで、Bには引き継がない。

---

## 17. Codexへ渡す直近の作業範囲案

この節は次の実装依頼の下書きであり、今回の計画書作成だけでscience実行を認可しない。

```text
Track BのB0を実施してください。研究Aの進行とM1/M2証拠を変更しないでください。

1. 既存native PF、finite-RTE、negative-time、controlled-phaseのAPIをread-onlyで確認する。
2. 研究Bの主線を「RTE実行負担を含むPF係数設計（B-F）」として記載する。
3. 5段/7段対称composition、次数条件、native stage IRを定義する。
4. ideal PFとfinite-RTE平均を分け、T=q*h、normalization、bias、shot objectiveを明示する。
5. 既存最適化4次/8次、同じfamilyの通常cost-aware設計、leading-RTE-aware設計を対照にする。
6. q/r/Kを各方式に選ばせるが、巨大な全組合せgridは作らない。
7. BF-1の初期domain、計算予算、未解決事項と、実行前に必要な承認を一枚にまとめる。
8. B-Sは保存済みcostから既知IS最適値のheadroomを集計できるかだけ確認する。
   raw記録が無ければmissingとし、新しいtrajectoryやcompileで埋めない。
9. 必要な代数・synthetic correctness testsと、科学計算を伴うtestsを分ける。
10. 新規性は仮説と記載する。最適split変更をGOの必須条件にしない。

禁止：新しい分子計算、NPZからのscience評価、trajectory sampling、circuit compile、
H4/H6での係数探索、M2再実行、旧artifact書換え、Track Aのmethod自動変更。

出力：研究B README、B-F数学契約、先行研究との差分表、BF-1最小pilot案、変更一覧。
完了後停止し、science execution authorizationを自動作成しない。
```

---

## 18. 今回の確認範囲と未実施事項

今回行ったもの：基準commitの研究概要・P-D事後解析の読取り、提供チャットの停止理由の確認、一次文献の重点照合、計画と初等的な代数整理。

行っていないもの：新しい係数の数値探索、分子NPZ読込、Hamiltonian simulation、trajectory sampling、Qiskit compile、research tests、source hash全件監査、repositoryの変更・push。

特にB-Fの新係数、B-Sの実装可能なDF state、資源改善率、一般化性能は未確定。計画中の初期domainやbudgetは提案として扱う。

---

## 19. 参照資料

### リポジトリ・提供資料

[R1] `docs/research/研究概要・現状.md`、commit `b6e65c6123475add5e620ec1064f361378bead95`。M2固定5構成のtransferと限定scope。

[R2] `docs/research_direction_pd_s1_posthoc.md`、同commit。S1 v2の主対照一致、P-D停止、nested/nativeの解釈。

[R3] 提供された「研究チャット引継ぎ」および過去チャット全文。P-A、P-B、P-C、P-D、R3、FRの停止理由。過去時点の未実行記述を現在のstatusへ流用しない。

### 一次文献（2026-10-04に重点確認）

[W1] J. Günther et al., *Phase Estimation with Partially Randomized Time Evolution*. PRX Quantum 7, 020332 (2026). DOI: 10.1103/ynxb-p2xq. arXiv:2503.05647（v2）。主にcoherent Hadamard解析、Appendix AのRTE/partial PF/絶対tail時間。PDFの関連式・図を確認。

[W2] M. E. S. Morales et al., *Selection and improvement of product formulae for best performance of quantum simulation*. arXiv:2210.15817v3、Quantum Information & Computation 25 (2025). DOI: 10.2478/qic-2025-0001。係数探索と公平比較の既知範囲。今回の確認はabstract/書誌中心で、全係数catalogの照合はBF-0に残る。

[W3] K. Hejazi et al., *Better product formulas for quantum phase estimation*. arXiv:2412.16811v1。energy-specific PF解析。abstractとHTML本文を確認。

[W4] P. A. M. Casares et al., *Theory and practice of Trotter product formulas for quantum chemistry*. arXiv:2606.30741v1。SPRINT、near-integrable設計、remainderへのRTE等、係数設計。PDFのFig.1と関連本文を確認。

[W5] D. Cugini, T. A. Atif, Y. Subaşı, *Resource-Optimal Importance Sampling for Randomized Quantum Algorithms*. arXiv:2603.13495v1。Sec. II、Theorem 1、Eq. (7)–(13)、composite channelとbias保存。HTML本文を確認。

[W6] *Efficient and practical Hamiltonian simulation from time-dependent product formulas*. Nature Communications (2025). DOI: 10.1038/s41467-025-57580-5。THRIFTの実装条件、Theorem 1/Proposition 1と式(6)–(13)。本文を確認。

[W7] A. Rajput, A. Roggero, N. Wiebe, *Hybridized Methods for Quantum Simulation in the Interaction Picture*. Quantum 6, 780 (2022). DOI: 10.22331/q-2022-08-17-780. arXiv:2109.03308。今回はabstract/論文情報を確認。DF実装への詳細還元は未検討。

[W8] P. Zeng, J. Sun, L. Jiang, Q. Zhao, *Simple and High-Precision Hamiltonian Simulation by Compensating Trotter Error with Linear Combination of Unitary Operations*. PRX Quantum 6, 010359 (2025). DOI: 10.1103/PRXQuantum.6.010359. arXiv:2212.04566。abstract/著者説明により、補償の一般思想とランダム実装が既知であることを確認。全証明の独立監査は未実施。

[W9] X. Wang et al., *Trotter error compensation with polylogarithmic precision and nested-commutator scaling without ancillas*. arXiv:2607.11856v2。abstractとtask定義を確認。channel-level補償であり、本計画のcoherent first-momentへそのまま転用できると主張しない。

[W10] J. Heng, A. N. Bishop, G. Deligiannidis, A. Doucet, *Controlled Sequential Monte Carlo*. Annals of Statistics 48(5), 2904–2929 (2020). DOI: 10.1214/19-AOS1914. arXiv:1708.08396。逐次proposal制御・normalizerという一般分野の先行研究。今回はabstractと書誌を確認し、B-Sと同一algorithmと断定していない。

本書のB-F/B-S構成案・停止条件・研究運用は、これらの資料を踏まえた提案であり、引用文献が同じ提案を実証したという意味ではない。
