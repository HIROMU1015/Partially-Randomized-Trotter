# Track B：DF-RTE Gaussian積生成候補の監査後科学レビュー

- 作成日：2026年10月13日（日本時間）
- レビュー開始承認：利用者の「レビューを開始して」
- 対象repository：`HIROMU1015/Partially-Randomized-Trotter`
- 監査branch：`track-b-df-rte-product-generation-audit-20261013`
- 監査公開commit：`e6530c055c505712d5f1d1af604f1accf85cebba`
- 対応実装の読取りcommit：`321cd8c3066194fe98741cbebbcdba1515d1deaa`
- 前提とする方針レビュー：`track_b_post_closure_research_direction_review_20261013.md`
- 適用する共通指示：`research_project_instructions_gpt_codex_review_approval_2026-10-12.md`
- レビュー状態：**完了。現在記述されたGaussian積生成手順を、独立した新規PRアルゴリズムの主線としては採択せず、この候補監査を区切る。**
- 性能の判断：**期待資源の改善・非改善は未確定。性能実験で敗れた候補として扱わない。**
- 応用研究：既知法のDF-RTEへの適用としての可能性は保持するが、今回自動的に研究目的を切り替えない。
- 実行：新しい行列生成、回路生成、compile、T合成、sampling、repository tests、remote writeは行っていない。

---

## 0. 最終判断

**今回の採択は、提示された選択肢のうち「現在の候補を区切る」である。** 対象は、active span・反射積・partial Gaussian合成を組み合わせた、現在の具体的な手順である。Gaussian圧縮一般、DF-RTE改善一般、Track B全体の可能性についての不可能性判断ではない。

前回は、独立した手順の差があるかを最初に確認する候補監査を選んだ。今回の監査は、その問いに対し、既知の反射積表現とpartial Gaussian合成への具体的な対応を示した。一方、候補固有の生成法、情報取得の節約、誤差保証、資源改善は、現在の仕様として定義されていない。[R1, R3]

したがって、新規アルゴリズムの本番pilotへは進まない。方法の独自性を判定する関門が閉じていないまま、先にorbital adapter、compiler、条件付き費用実験を大きく作ることは推奨しない。

ただし、**既知の部品を使っていることだけで研究価値がなくなるわけではない。** 組合せによる新しい保証、適用条件の拡張、取得手順の削減、明確な資源上の効果は研究上の寄与になり得る。今回は、その追加部分が現手順から特定されておらず、性能の証拠もまだない、という判断である。

| 論点 | 本レビューの判断 |
|---|---|
| 指定primitiveでのGaussian積構成 | 数学的に支持する |
| 現手順の独立した方法差 | 未定義。新規アルゴリズムとして採択しない |
| 既知compilerと出力回路・費用が完全一致するか | 未証明。そうとは主張しない |
| 元PR分布での期待資源改善 | 未確定 |
| 費用改善が小さい／ないという性能negative | 認定しない |
| 応用研究への自動的な変更 | 行わない |
| 次の新規数値pilot | 今回は進めない |
| 旧generic whole補正系列のclosure | 維持 |
| Track A／N3の方針・実行 | 変更・介入しない |
| 次の実務 | 必要最小限の研究判断・証拠索引への反映のみ |
| mandatory STOP | 維持 |

## 1. 使用資料と確認範囲

### 1.1 正本と取得確認

監査報告[R1]、保存費用の確認範囲JSON[R2]、前回レビューのGate 1–4と着地点[R3]、metadata検査の件数・区分[R4]を、固定commitから取得した。branchのremote HEADが指定された`e6530c...`と一致することも確認した。[R8]

監査報告は長いため複数の行範囲で読み、冒頭の分類だけでなく、M1–M6、実装との対応、compact WYへの還元、費用モデル、保存値の不足、実行境界、文献確認の限界まで確認した。

実装については、RTE eventを作る`_make_event`、DF builderのbasis reuse・位相の処理、basis registryの型と登録処理、`basis_unitary`の用途を重点的に読んだ。[R5–R7] 全repository、全branch、全ての関連関数を再監査したものではない。

### 1.2 125 PASSの意味

保存された125件は、Git/ASTと保存JSON/CSVを対象とするmetadata bookkeepingである。候補回路のoperator一致、T費用、数学的証明の機械認証、母平均の信頼区間の件数ではない。[R1, R4]

私は125件を再実行していない。数学的な式については以下で独立に展開し、sourceとの整合を読み取った。これはCodexの検査件数に加算する新しいテスト件数ではない。

### 1.3 原資料・外部確認・本レビューの判断を区別する

- **原資料にある事実**：監査分類、対象source、保存確率、未保存field、未実行項目。
- **本レビューの数学的検討**：反射積の再帰、位相、支持、期待費用の分解などの式の確認と解釈。
- **外部一次資料で確認した事項**：compact reflector表現、partial basis rotation、制御付きfree-fermion圧縮の既知性。
- **研究運営上の判断**：現在の候補を新規アルゴリズム系列として延長しないこと。性能の非優位を証明したこととは別である。

## 2. 研究全体における位置付けと、前回判断との関係

Track Bの目的は、PRの具体的な内部処理に変更を加え、研究として区別できる資源改善・生成手法・保証を得ることであった。過去の系列では、normalizationの改善、局所性、primitive単価だけでは、強い対照を戻した全費用の改善にならないことが問題になった。

直前のgeneric whole補正は、三重交換子の指数回転という一般に非Gaussianな対象を実装する方式だった。現在の候補は既存RTEのparity演算子の積を対象にするため、両者は別である。旧方式の局所T費用の結果を、今回の圧縮の性能negativeへ流用しない。[R9]

前回レビューは、Gaussian閉包やcompact WYを新発見として認定せず、まず手順の差を監査する方針を選んだ。Gate 1には「全処理が既知の組合せと実質的に同じであれば、新規アルゴリズムの本番pilotへ進まない」と記している。Gate 3も、Gate 1–2に具体的な余地が残る場合に限る。[R3]

**今回は、その監査結果を受けた判断であり、同じ証拠のまま担当や方針を反転させたものではない。** 前回の候補提案は新規性の認定ではなく、少ない追加作業で独自性と実装余地を調べるための仮説だった。

今回の結果を受けて、候補の名前だけを変えたり、終端回転やsampling lawの変更を追加したりして、そのまま次の実験へ進むことはしない。

## 3. 理想構成についての評価

### 3.1 parity primitiveとGaussian表現

真空での位相を1に固定したnumber-conservingの第二量子化を

\[
\Gamma(M)=\bigoplus_{\nu=0}^{N}\wedge^\nu M
\]

とする。正規直交列を持つ\(V_j\in\mathbb C^{N\times r_j}\)、\(r_j\in\{1,2\}\)に対して、

\[
R_j=I-2V_jV_j^\dagger,
\qquad
Q_j=\sigma_j\Gamma(R_j)
=\sigma_j\prod_{\ell=1}^{r_j}(1-2n(v_{j\ell}))
\]

が成り立つ。選択軌道の占有ごとに符号が反転し、その直交補空間では恒等となることから確認できる。[R1]

\(\sigma_j\)はmany-body演算子のscalar signである。これを一粒子行列の\(-R_j\)に吸収してはならない。実際、\(\nu\)粒子sectorでは

\[
\wedge^\nu(-R_j)=(-1)^\nu\wedge^\nu R_j
\]

であり、全Fock空間で一様な符号反転ではない。

この構成を支持するのは、指定されたGaussian basisで共役したZ/ZZ primitiveに対してである。DFという名称、registryへ登録されているという事実だけで、任意のoperation列へ一般化しない。

### 3.2 積の順序とeventの保持

時間順に\(Q_1,\ldots,Q_k\)を作用させると、

\[
P=Q_k\cdots Q_1
=\chi\Gamma(M),
\quad M=R_k\cdots R_1,
\quad\chi=\prod_j\sigma_j.
\]

これは\(\Gamma\)の積に関する性質であり、因子の可換性や並べ替えを仮定しない。

対応sourceは、Taylor order個のproduct applicationの後に、終端rotationを置く。抽選された最初のcomponentを終端rotationに用い、残りをproductとする処理と、回路で作用させる順序は区別されている。event phaseは\((-1)^{k/2}\)である。[R5]

監査は`unsigned_rotation_angle`という名前と、符号を含めた実装上の意味が異なることも指摘している。この指摘を新しい符号規約へ変更する理由にはせず、元の終端rotationとphase処理をそのまま保持する必要がある。[R1]

**候補の対象はevent内のproduct部分だけである。** 終端rotation、外側PF、sampling確率、normalization、weightを変更する方式へ、ここで対象を広げない。

### 3.3 active spanの評価

\(\mathcal S=\operatorname{span}\{V_1,\ldots,V_k\}\)とし、直交basisを\(W\)、その次元を\(d\)とする。各\(R_j\)は\(\mathcal S\)を保ち、\(\mathcal S^\perp\)で恒等なので、

\[
M=I+W(M_S-I_d)W^\dagger,
\qquad
d\le\min\left(N,\sum_jr_j\right)\le\min(N,2k).
\]

この支持の主張は妥当である。[R1] ただし、前のwhole補正の「最大6軌道」とは別であり、K2では最大4、K4では最大8という語長に応じた上限になる。実際のspan次元が上限より小さいかは、公開された保存資料からは確認されていない。[R2]

### 3.4 制御と終端rotation

\(M_S=Z\operatorname{diag}(e^{i\phi_\ell})Z^\dagger\)を使えば、basis変換、occupation phase、逆basis変換で構成できる。制御する中央作用は

\[
e^{i\phi n}=\operatorname{PhaseGate}(\phi)
=e^{i\phi/2}RZ(\phi)
\]

である。scalarを落としたRZだけでは、制御枝の相対位相を一般に保持しない。\(\chi\)とTaylor phaseも別に戻す必要がある。[R1, R6]

同じbasis circuitと厳密なinverseを両側に使うと、そのbasisのscalar phaseは相殺する。しかし、中央作用やeventのscalarを捨ててよいことにはならない。また、control ancillaを含む全作用まで、無条件にnumber-conserving Gaussianと呼ぶわけではない。

rank2の終端\(e^{-i\theta Q}\)を一般にGaussianとしない監査も妥当である。\(Q=(1-2n_1)(1-2n_2)\)の回転から真空位相を除くと、単独占有は各\(e^{2i\theta}\)、二重占有は1となる。Gaussianの対角作用なら二重占有は単独二つの積\(e^{4i\theta}\)である必要があり、一般の角度では一致しない。[R1]

### 3.5 有限精度について

各eventの理想作用、確率、weightを保てれば、有限first operator momentは変わらない。実装誤差は

\[
\left\|\sum_ep_ew_e(\widehat U_e-U_e)\right\|
\le\sum_ep_e|w_e|\,\|\widehat U_e-U_e\|
\]

へ戻す必要がある。[R1, R3]

一粒子空間で正確にunitaryな\(M,M'\)に対する

\[
\|\wedge^\nu M-\wedge^\nu M'\|\le\nu\|M-M'\|
\]

は、tensor積のtelescopingと反対称部分空間への制限から確認できる。しかし、非unitaryな浮動小数配列や、Gaussian性を正確には持たない合成済みClifford+T回路へ、そのまま適用して全誤差を保証できる式ではない。

**理想構成が正しいことを支持する一方、新しい候補の有限精度実装が完成・検証済みとは判断しない。**

## 4. 方法差について：何が既知処理へ対応するか

### 4.1 compact反射積への対応を式で確認する

primitive内の列が正規直交なら、

\[
I-2VV^\dagger=\prod_{\ell=1}^{r}(I-2v_\ell v_\ell^\dagger).
\]

各primitiveをrank1反射へ展開し、異なるprimitive間の順序は保持できる。\(H_s=I-\tau_sv_sv_s^\dagger\)、\(\tau_s=2\)とし、

\[
M_{s-1}=I-X_{s-1}T_{s-1}X_{s-1}^\dagger
\]

を左から\(H_s\)で更新すると、

\[
H_sM_{s-1}
=I-XTX^\dagger-\tau_svv^\dagger
+\tau_svv^\dagger XTX^\dagger.
\]

よって\(X_s=[X_{s-1},v_s]\)と置き、

\[
T_s=
\begin{pmatrix}
T_{s-1}&0\\
-\tau_s v_s^\dagger X_{s-1}T_{s-1}&\tau_s
\end{pmatrix}
\]

とすれば、\(M_s=I-X_sT_sX_s^\dagger\)となる。この再帰は、列間の独立性を要求しない。\(X_s=WC\)と因数分解すれば

\[
M_s=I-W(CT_sC^\dagger)W^\dagger,
\quad (M_s)_S=I_d-CT_sC^\dagger
\]

であり、active-span内の反射積と同じ対象が得られる。これは監査の式を本レビューでも展開して確認したものである。[R1]

反射積をcompactな三角因子で表す方法は既知である。[P1, P2] ただし、複素LAPACK routineのtrapezoidal storage、implicit diagonal、DIRECT/STOREVの約束は別に満たす必要がある。**任意のdense DF軌道列を、そのまま`ZLARFT`へ渡せるという結論ではない。**

### 4.2 既知性の判断の範囲

| 現候補の内容 | 対応する既知処理・根拠 | 本レビューの判断 |
|---|---|---|
| parity積を一粒子行列で扱う | number-conserving Gaussian表現 | 出発点として妥当だが、それ自体を新規としない |
| 順序付き低rank反射積 | compact WY型表現 | 現仕様に固有の新しい積再帰は示されていない |
| 選択spanだけを座標へ運ぶ | partial Givens QR/basis rotation | 既知手順を対照側にも認める |
| 小unitaryとoccupation phase | 通常のGaussian合成 | 独自の角度取得・precision節約は未定義 |
| controlled圧縮 | 既知のcontrolled free-fermion構成 | control対応の欠如を既知法との差にしない |
| sourceに専用hookがない | 現repositoryの実装不足 | 文献上の独自性とは別 |
| event lawとweightを不変にする | 同じ演算子の実装置換 | 公平な比較に有用だが、それだけで新規なPR法ではない |

Campsらの一般的なfree-fermion回路圧縮[P4]、Kökcüらのcontrolled free-fermion構成[P3]も近接する対照である。Kökcüらのv2ではSec. IV.2にcontrolled quadratic HamiltonianとQ-block mappingがあり、Hadamard-test型の使用も明示されている。

ただし、その論文が現在のDF-RTE生成手順、任意角T合成、同じfull wrapperの費用を全て評価済みだとは述べない。**既知の枠組みへの対応と、具体的なcompilerの出力・費用が同じである証明は異なる。**

### 4.3 「部品が既知だから新規性なし」という短絡は採らない

既知部品の組合せでも、従来の手順では得られない情報アクセス、厳密な生成保証、少ない取得計算量、有限精度での安定性、対象分布での意味のある資源差を具体化すれば、研究上の寄与になり得る。

今回、その可能性を否定し切ったわけではない。一方、現在記述された手順は「反射積を小空間で作り、通常のGaussian合成へ渡す」までである。上記の追加寄与を生む処理は示されていない。

同じ理想演算子を実装するという事実だけで、二つのアルゴリズムが全く同じとも言えない。しかし、現在の候補については、標準的な構成手順に何を付け加えたのかが不明なままである。**現段階の技術分類を支持し、新しいアルゴリズムとしての採択は行わない。**

## 5. 資源モデルの評価と、候補を過度に不利に扱わないための留意点

### 5.1 強いthin-basis対照を残す

部分basisをGivens QRで構成するrotation-count modelとして、

\[
g(N,d)=Nd-\frac{d(d+1)}2
\]

を用いることは、Mottaらのpartial-basisの議論に対応する。[P5] これはeliminationの回数であり、RZ数、CX数、T数を同一の整数で与えるものではない。複素位相やcontrol、native lowering、合成精度は別である。

\(s=\sum_jr_j\)とすると、逐次thin実装の粗いbasis費用は\(2\sum_jg(N,r_j)\)。active-span方式は\(2g(N,d)+O(d^2)\)などと記述できる。\(d\approx s\approx2k\)の短い語では、両方とも概ね\(O(Nk)\)となり、漸近次数の改善は自動的にはない。[R1]

逆に、繰り返しや共有spanによって\(d\ll s\)なら、まとめることに利益の余地はある。その条件を既知の構造的compilerも利用できる点と、元のRTE分布でどれだけ現れるかを確認する必要がある。

### 5.2 本レビューによるモデル解釈上の補足

監査の\(O(d^2)\)を、必ず別に支払わなければならない量子回路費用の下界と読まない。\(M_S=ZDZ^\dagger\)を得た後、\(WZ\)を直接partial basis compilerへ渡せる実装なら、\(W\)と\(Z\)を量子回路として別々に生成する必要はない。

これは既知のbasis合成をどの段階でまとめるかという実装選択であり、今回新しいcompilerを作った結果ではない。**候補へ不要な内部gateを二重計上して、性能不利を作ることもしない。**

一方、この整理だけでは、対照に与えられない新しい処理が得られたことにはならない。小行列計算、軌道情報の取得、phase、有限精度認証の費用も残る。

長いGaussian語が有限サイズの変換へ圧縮できることは、一般Gaussian圧縮の主要な既知特性とも整合する。[P3, P4] 今回のK2/K4という短い語の設定に、長時間・長語極限の改善率をそのまま移さない。

### 5.3 RZ/T費用は減るとは限らない

元の中央productはZ/CZ系のCliffordである一方、積を対角化したoccupation phaseは一般角度になり得る。[R1, R6] basis開閉回数が減ることと、合成後のT費用が減ることは同じではない。

これは実際の悪化を測った結果ではない。支持の縮約だけでは最終資源の優位を決められない理由である。

## 6. 期待費用の未確定部分を、どう解釈するか

### 6.1 保存確率から分かる範囲

保存資料の8構成について、少なくとも一つnonzero orderを含むtrajectoryの確率と、product slot数の期待値が記述的に整理されている。[R2]

| q/r | K | nonzero orderを含む確率（約） | product slots/trajectory（約） |
|---|---:|---:|---:|
| 1/4 | 2 | 0.063793 | 0.130757 |
| 1/4 | 4 | 0.063965 | 0.131485 |
| 2/4 | 2 | 0.032973 | 0.066918 |
| 2/4 | 4 | 0.032996 | 0.067012 |
| 4/4 | 2 | 0.016697 | 0.033658 |
| 4/4 | 4 | 0.016700 | 0.033670 |
| 8/2 | 2 | 0.016697 | 0.033658 |
| 8/2 | 4 | 0.016700 | 0.033670 |

表はCodexの保存済み値を丸めて転記した。新しいdistributionやtrajectoryを生成していない。これらはH6の既知/developmentデータであり、Track Bの独立移送試験ではない。

nonzero orderを含むことは、圧縮によって利益が出ることの十分条件ではない。identity product、重複因子、basisの境界、compilerの挙動が残る。各span次元と条件付き費用はunknownである。

### 6.2 確率が1.7–6.4%だから、改善もその程度とは言えない

同じlaw・shot条件で、変更対象trajectory集合を\(\Omega\)とする。候補が\(\Omega\)の外で元wrapperと同じで、費用が非負なら、

\[
\Delta G_x
=\sum_an_a\mathbb E[C^0_{a,x}-C^1_{a,x}]
\le\sum_an_a\mathbb E[1_\Omega C^0_{a,x}].
\]

\(p_\Omega=\Pr(\Omega)\)だけが分かっても、右辺には\(\mathbb E[C^0\mid\Omega]\)が必要である。[R1–R3]

一つのaxis・指標で、\(\mu_1=\mathbb E[C^0\mid\Omega]\)、\(\mu_0=\mathbb E[C^0\mid\Omega^c]\)と書くと、変更対象が占める元費用の比率は

\[
\frac{p_\Omega\mu_1}{(1-p_\Omega)\mu_0+p_\Omega\mu_1}.
\]

これは本レビューの代数的な整理であり、保存H6へ新しい数値を当てはめた結果ではない。同じ小さい\(p_\Omega\)でも、条件付き費用の比によって元費用の寄与率は大きく変わる。従って、低頻度だけで低価値と判定しない。

**今回のGate 2は未確定であり、性能negativeではない。** 「改善上限が5%未満」などの閾値も、結果後に導入しない。

### 6.3 データ不足の種類を分ける

公開wrapper CSVにはaxis-paired費用とevent digestがあるが、actual event labels/order列/span dがない。公開registryにはrawへのpath/hashはあるが、その内容は今回の公開summaryには含まれていない。[R1, R2]

既存rawの公開によって対応を確認できる部分と、そもそも未観測で新しい条件付きcompileを必要とする部分は別である。hashだけからtrajectoryの中身を推測してはならない。

実行STOP capも、未生成の全trajectoryについての母集団費用上界ではない。局所gate費用の和を、全体fusion後のwrapper費用の厳密上界とするにも対応の証明が必要である。[R1]

### 6.4 この不足は、今回の採択判断を保留する必須理由ではない

これらの費用が分かれば、既知法の応用としての有用性をよりよく判断できる可能性がある。しかし、費用を測ったことだけで現在の生成手順に新しい方法差が生まれるわけではない。

現在の独立新手法pilotを見送る判断は、期待利益がゼロと分かったことに依存しない。方法差が未定義であり、前回のGate 1を通していないことを基礎にする。**新規性の判断と、性能の未確定を混同しない。**

## 7. 入力accessの不足は、研究上どの程度重要か

sourceのbasis registryは、operation metadataとruntime gatesを保持する。一粒子の\(N\times r\)軌道列や、各registered basisがnumber-conserving Gaussianであることを保証する型情報が、そのまま提供されているわけではない。[R7]

`basis_unitary`も、小系のdense referenceへ進む関数であり、名前だけで\(N\times N\)軌道unitaryを返すAPIだと解釈してはいけない。[R7]

これは「必要な軌道情報は作れない」という意味ではない。今後実装するなら、保存DF情報・runtime basis・列の規約・sign・precisionを結ぶ入力経路を定義する必要があるという意味である。

そのadapterを用意することは、正しい実装に必要な技術作業にはなる。しかし、**実装不足を埋めたことだけを、独立した新アルゴリズムの寄与とは数えない。** 再対角化、dense Fock評価、未知のcacheを無料の入力アクセスと仮定することもしない。

このため、「小さいcompilerを一つ試すだけ」と想定していた作業でも、同じ入力・同じ精度の比較にするには周辺の構築が必要になる。費用を定量的に見積もったわけではないが、現候補に固有の研究差を定義しないまま、その準備を自動的に拡張することは避ける。

## 8. 三つの選択肢の比較

### A. 新しい方法差を具体化して継続する

この方向が意味を持つのは、例えば、同じ入力情報・精度・controlの下で、既知のpartial Gaussian手順にはない処理の省略、認証付きの生成保証、明確なcache非依存の取得節約などを具体的に示せる場合である。これは必要な差分の種類の例であり、新しい研究案をここで採択したものではない。

今回、active spanの利用、reflectorの集約、central-only control、既存basisとの結合を検討しても、現在の仕様に追加された独立手順は特定できていない。単なる実装順序や書き方の変更をmethod deltaとして扱わない。

結論：**現仕様のままAへ進まない。新機構の不存在を一般証明したわけではない。**

### B. 既知法の応用研究へ限定する

「既知のGaussian圧縮をDF-RTEへ適用したとき、元の分布とfull wrapperへ戻して有効か」という問いは、科学的に正当である。独立した新定理がなくても、実装上の適用範囲や反例を丁寧に評価する応用研究になり得る。

しかし、現在は変更対象の条件付き費用も候補の実装費用も未取得であり、応用価値を実証した状態ではない。高次数を強制した成功例だけを作る実験は、元PRの期待資源という問いに答えない。

応用研究へ進むには、それを新しい主要目的として選び、同じ入力・同じ精度の対照、元分布への戻し方、取得負担、終了条件を決める必要がある。元の新手法研究がGate 1を通らなかったからという理由だけで、目的を自動変更しない。[R3]

結論：**可能性は保持するが、今回新しい応用研究として採択・起動しない。** Track Aが独立に取得する将来の希少費用情報が利用可能になった場合、必要性を改めて判断することはできる。ただし、その作業をTrack Bのために自動的に開始・変更しない。

### C. 現候補を区切る

前回の独自性監査の条件と整合し、研究目的を後付けで変えずに判断できる。理想構成・既知法との対応・access gap・期待費用の欠測を残せば、今回の監査成果は失われない。

結論：**これを採択する。現在の新規アルゴリズム候補としての自動延長を止め、追加pilotには進まない。**

この選択は、期待資源での敗北や、Gaussian圧縮一般の無価値を意味しない。取得済みの情報と、現研究目的に対して、次の実装系列を開始するだけの具体的な差が残っていないという研究資源配分の判断である。

## 9. 保持する成果と、主張しない内容

### 保持する内容

- 指定されたZ/ZZ Gaussian primitiveに対する、順序付き反射積と第二量子化の対応。
- productのscalar sign、event phase、control、終端rotationの分離。
- active spanの上限と、compact reflector表現への具体的対応。
- 現basis APIと軌道情報アクセスの不足の特定。
- 期待費用を評価するには、頻度だけでなく条件付きfull-wrapper費用が必要という整理。
- 本格的な実装前に、新手法pilotへ進む条件が満たされていないと判別したこと。

### 主張しない内容

- 新しい一般PRアルゴリズムを確立した。
- Gaussian積圧縮は既存法より必ず高価、または必ず安価である。
- 全DF registryへそのまま適用できる。
- 全RTE、終端rank2回転、外側PFまでGaussianとして圧縮できる。
- H6では最大でも1.7–6.4%しか資源を減らせない。
- 同じ理想演算子を生成するので、既知compilerとT数も同じである。
- Track B全体の改善可能性を否定した。

現段階では、独立論文の主要な貢献が確立したとは判断しない。監査・設計上の知見として既存の研究記録へ残すのが適切である。

## 10. 次の作業

直近の実務担当はCodexでよいが、**新しいcompiler開発や数値取得ではない。** 今回のレビュー原本と、現候補を区切る判断、性能は未確定、応用研究は未採択という三点を、既存の証拠索引へ最小限反映すればよい。

新しい大型レポート、全repositoryの再監査、再現ZIP、同じ125件の反復検査を、この記録作業のためだけに要求しない。文書の保存・リンク・原本identityを必要な範囲で確認し、完了後STOPする。

今回の判断に含めない実行は、次のとおり。

- orbital sidecar、candidate circuit、比較runnerの新規実装・実行。
- 新しいmatrix/control fixture、native compile、T合成、sampling。
- H6 conditional cost取得、H8・分子拡張、DF再取得。
- 旧generic whole、G10、RA-D0の再開。
- 新しいsampling lawやHamiltonian表現への自動転換。

上記が将来必要になった場合は、その科学的目的を別に採択し、必要な計算実行承認を受ける。今回のレビュー開始承認は、実行許可でもrepository書換え許可でもない。

記録反映の完了だけを理由に、今回と同じ科学レビューへ戻る必要はない。独立した新手順が提示された場合、応用研究へ目的変更することを利用者が選んだ場合、または判断を変え得る具体的な費用証拠が得られた場合に、次のレビューの必要性を判定する。

## 11. Track B全体と他研究の扱い

本レビューは現在のGaussian積生成候補を扱う。全ての将来のTrack B候補を調査して終了したものではない。従って「Track Bに有望な研究は一つも残っていない」という判断はしない。

一方、この候補を止めた直後に、別の未定義案を自動採択して計算を回すこともしない。現時点でこの監査から直接認可される新しい科学実行はない。

Track Aの既定の希少費用評価やN3の研究判断は独立であり、ここで優先順位を変更・統合しない。他研究の情報は比較方法と不足項目の理解に用いたが、その成果を本候補の成果へ付け替えない。

## 12. 確認の限界

- 監査対象の全10 Pythonファイルを今回全量読み直したわけではない。event・builder・registryの重要な経路を重点確認した。
- metadata125件、20ファイルのremote再取得等のCodex確認を独立再実行していない。
- private trajectory、event列、軌道列、NPZへアクセスせず、未取得fieldを推測していない。
- 候補と既知compilerの同一入力・同一精度native比較はない。費用の優劣は不明。
- 外部文献に、まったく同じDF-RTE実装が存在しないという網羅的不在証明をしていない。
- compact WY論文は出版社abstract、複素reflectorの扱いはLAPACK公式仕様を確認した。出版社有料全文を読んだとは主張しない。
- Kökcüらはv2 HTMLのcontrolled構成を確認した。全数値図表・実機結果の再検証はしていない。
- Campsらは一次abstractと版情報を確認し、詳細な後退安定性証明を本レビューで再証明していない。
- Mottaらv2 PDFは取得テキストのPartial Basis Rotationを確認した。Web screenshotの該当ページ取得は内部エラー、別のPDFダウンロードも失敗したため、ページ画像・図表の視覚確認は根拠にしていない。rotation-count式は抽出テキストに明示されており、図からの数値読取りは行っていない。
- 数学式の展開と費用の代数的整理は行ったが、新しい数値性能実験、独立再現計算、実行authorizationは行っていない。

**結論を保留したのは期待資源の性能判定であり、現候補を新規アルゴリズム主線として延長しないという研究判断は完了した。**

---

## 参考資料

### リポジトリ資料

**[R1] Codex監査報告。** 固定commit `e6530c055c505712d5f1d1af604f1accf85cebba`。

https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e6530c055c505712d5f1d1af604f1accf85cebba/docs/tracks/algorithm_codesign/df_rte_product_generation_audit_v1_20261013.md

Git blob：`646ee6d2bec6aeabf5d45072d55dff8fe469ba41`。本レビューではM1–M6、方法差、source access、Gate 2、停止範囲、文献確認範囲を読んだ。

**[R2] 保存期待費用の確認範囲。** 同固定commit。

https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e6530c055c505712d5f1d1af604f1accf85cebba/artifacts/track_b_df_rte_product_generation_audit/2026-10-13/saved_cost_scope_v1.json

Git blob：`07de94f80cf2414cfa3d40b7e92bb431933a66bc`。8行とmissing fieldsを確認した。

**[R3] 前回GPT方針レビュー。** 同固定commitに保存された原本。特に第10–15節。

https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e6530c055c505712d5f1d1af604f1accf85cebba/docs/research/track_b_post_closure_research_direction_review_20261013.md

Git blob：`cc282369916c2dd99925f855e38751220783e3d2`。このconversationの同名原本も参照した。

**[R4] metadata validation。** 同固定commit。

https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e6530c055c505712d5f1d1af604f1accf85cebba/artifacts/track_b_df_rte_product_generation_audit/2026-10-13/metadata_validation_v1.json

Git blob：`a886d14160a52f9c5e26917e9e12a0947d20ea65`。125 PASSを保存記録として確認した。

**[R5] RTE event生成source。** commit `321cd8c3066194fe98741cbebbcdba1515d1deaa`、`_make_event`。

https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/321cd8c3066194fe98741cbebbcdba1515d1deaa/src/trotterlib/rte.py

Git blob：`53601b7c524653f5e1966b3955b9f51373e54002`。

**[R6] DF-RTE回路builder。** 同source commit、basis reuseと位相処理。

https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/321cd8c3066194fe98741cbebbcdba1515d1deaa/src/trotterlib/df_rte_qiskit.py

Git blob：`da3c7c68e6ee66e28a599a28fe087b4c272e8504`。

**[R7] DF basis registryとdense reference。** 同source commit。

https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/321cd8c3066194fe98741cbebbcdba1515d1deaa/src/trotterlib/df_rte_tail.py

Git blob：`499d6f9c70609891375cc232a29ad45ae5a29f05`。

**[R8] branch取得確認。** 本レビュー時点でHEADが`e6530c...`に一致。内容の引用は原則として固定commitを用いる。

https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/track-b-df-rte-product-generation-audit-20261013

**[R9] 旧generic whole系列の判断。** 本conversationで既に完了したレビュー・closureを継承し、今回費用を再検証していない。

https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4192077723770f4846cc391ea0da96c1e115f2fa/docs/tracks/algorithm_codesign/df_correction_et_series_closure_20261012.md

### 外部一次資料・公式仕様

**[P1] R. Schreiber and C. Van Loan (1989), A Storage-Efficient WY Representation for Products of Householder Transformations.** SIAM Journal on Scientific and Statistical Computing 10(1), 53–57。出版社abstractを確認。

https://epubs.siam.org/doi/abs/10.1137/0910005

**[P2] LAPACK 3.12.1, ZLARFT公式仕様。** complex block reflector、DIRECT/STOREV、三角因子、storage規約を確認。実routine未実行。

https://www.netlib.org/lapack/explore-html/d7/d0d/group__larft_ga99d53d027cc196a5e5f7e2bbacdf4979.html

**[P3] E. Kökcü et al., Algebraic Compression of Free Fermionic Quantum Circuits: Particle Creation, Arbitrary Lattices and Controlled Evolution.** arXiv:2303.09538v2（HTML表示：2025-05-15）、Sec. IV.2 Eq.(42)–(43)、Appendix Cのscopeを確認。

https://arxiv.org/html/2303.09538v2

**[P4] D. Camps et al., An Algebraic Quantum Circuit Compression Algorithm for Hamiltonian Simulation.** arXiv:2108.03283v2（2021-08-17改訂）。一次abstract・版情報を確認。

https://arxiv.org/abs/2108.03283v2

**[P5] M. Motta et al., Low rank representations for quantum simulation of electronic structure.** arXiv:1808.02625v2、Partial Basis Rotationの抽出テキスト、特にEq.(20)–(22)周辺とrotation-countの記述を確認。本文以外の数値性能は本候補へ流用しない。

https://arxiv.org/pdf/1808.02625v2

---

## 状態の要約

本レビューの意味を機械的なstatusへ整理する場合も、次の三つは分ける。

- **研究判断**：現Gaussian積生成手順を独立新手法として採択せず、現候補の自動延長を停止する。
- **性能判断**：期待資源の改善・非改善は未確定。
- **実行判断**：新しい科学実行は未承認、mandatory STOPを維持する。

ここに記した分類は本GPTレビューの判断であり、既存repositoryの監査statusやraw結果を書き換えたものではない。
