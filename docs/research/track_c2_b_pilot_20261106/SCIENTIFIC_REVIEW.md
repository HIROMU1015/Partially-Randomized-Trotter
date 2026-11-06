# Track C2 科学レビュー：PF・合成精度・RPE総費用の改善方向

**日付：2026-11-06（JST）**  
**対象：** `HIROMU1015/Partially-Randomized-Trotter`  
**C2調査固定版：** `6f6c084bdfaf04ba951fd5d8ea0c521313ceb0b3`  
**継承するTrack C結果：** `720416c5b88b227cf76ff7aa729d91a0a98ed58b`  
**状態：** 承認された研究方向レビューを完了。研究方向の正式採択、性能・新規性の確定、新しい科学計算の実行許可ではない。

## 0. 結論

**C2-AとC2-Bを、具体的な問いに絞った独立研究として並列に進めることを推奨する。C2-Cは、具体的なPF構成が提示されるまで新しい係数探索を保留する。**

- **C2-A：round依存PF**では、有限PFメニューの有利不利がroundによって入れ替わり、その効果が、round別にstep・RTE・誤差配分を調整した最良の固定PF族よりも全費用で有利になるかを問う。共通energy保証の構築だけを、独立した新研究の中心にしない。
- **C2-B：非一様合成精度**では、単に回転ごとにepsilonを変えるのではなく、**基底変換・中心演算・実際の逆回路を一つのblockとして誤差評価し、その構造に応じて決定論的合成精度を配分する**方向を具体化する。本書で成立条件と誤差式を示す。既知の共役・誤差合成の数学を使う提案であって、世界初の原理とは主張しない。
- **C2-C：基底energy向けPF**は、現在の資料では新しいPF列・構成規則が特定されていない。P-D/BF等に近い探索を名前だけ変えて再開しない。既知energy特化PFを対照・有限メニューに含めることは可能だが、その実装を新しいPF発明とは呼ばない。

Bを具体化した理由は、旧C-Sのrandomized residualとは異なり、新しい大きなnormalizationを導入せずに合成費を変えられる可能性があるためである。Aを並列に残す理由は、他の層を固定した比較ができ、Bの実装完了を待たずにPF切替の利益を判別できるためである。これは研究進行上の推奨であり、実際の改善率の順位ではない。

本レビューでは、新規数値計算、保存値の科学的再集計・再ランキング、simulation、sampling、回路build/compile、T合成、repository変更を実施していない。以下の解析は紙上の導出であり、新しい実測値ではない。

## 1. 前提・証拠水準・参照範囲

### 1.1 合意済みの研究目的

元の物理Hamiltonian Hの厳密基底状態を各shotで利用できるとする。

\[
H|E_0\rangle=E_0|E_0\rangle.
\]

状態準備コストだけを除き、同じenergy精度・成功確率でPR＋RPE全体の回路費用を削減する。

\[
G^{(a)}=\sum_{m,b}N_{m,b}\,\mathbb E[C^{(a)}_{m,b}],\qquad b\in\{X,Y\}.
\]

T、RZ、CX、depth等は別の資源指標aとして扱う。shotを掛けたdepthは実wall timeではない。古典的な情報取得・最適化・合成の費用と追加qubitも別記する。

厳密基底状態の供給から、数値E0、gap、古典的な完全状態記述、最適bias補正、制御付き状態準備oracleを無料で得るとは仮定しない。ただし小系benchmarkでE0を診断に用いることは可能であり、実行時の選択規則とは分けて表示する。

H4、旧精度、旧18round設定は再利用可能なbenchmarkだが、本レビューでC2の実行条件へ自動採択したわけではない。Bは総T数を主指標にすることを推奨する。AをRZ/CXで先に調べる場合、Bと異なる単位の改善率を統合して順位付けしない。

### 1.2 証拠の区分

**内部資料の事実**は固定報告・SOURCEから読み取った内容で、独立再実行ではない。**外部一次資料**は実際に取得した関連箇所に基づく。**本レビューの解析**は明示した仮定の下での導出で、新規性を主張しない。実入力での費用改善は**未検証仮説**、次作業の選択は**研究上の推奨**である。

### 1.3 読解範囲

C2のREPORTと候補比較、Track CのA/B報告、合成SOURCE `track_c_h4_synthesis.py`、保証SOURCE `track_c_source_guarantee.py` の関連部分を取得した。合成SOURCEは全文、保証SOURCEは冒頭190行のbasis/generator/誤差伝播等を静的に読んだ。旧raw全量、全143資産群、全価格標本を再監査したわけではない。moduleのimportやtest/runner実行はしていない。

外部文献はUWC、Managing Approximation Errors、固有値誤差評価、energy特化PF、variable-precision PEの関連部分を確認した。網羅的な世界初判定ではない。版・取得制約は末尾に記す。

## 2. Track Cの負結果が否定した範囲

| 過去結果 | 支持される解釈 | 拡張してはいけない解釈 |
|---|---|---|
| C-N：S2の2,304 paired標本でfullと強baselineの4資源が一致 | 指定compiler A・構成・取得範囲でfull化の追加利益がなかった | PF切替・T合成精度・全compiler最適化が無効 |
| C-S：固定coarse/residual policyで巨大shot負担 | 登録policyの価格低下だけでは総費用を改善できない | 決定論的な非一様精度配分も無効 |
| C-P：指定processorで信号改善なし | 選んだkernel/P/stepでは利益を観測できなかった | 非共役PFや全processed構成が無効 |
| P-D/BF/BMの限定負結果・既知同値 | 同じscore・family・情報条件の再実行を避ける必要がある | energyを考慮する研究すべてが不可能 |
| A保証：保存行列モデルで18round条件が成立 | 再利用可能なsignal/半径/alias/shot会計がある | 元Hの全表現誤差を含む無条件保証が完成 |

出典：[I1]–[I4]。S2の十分shot 8,598、Full Y4の新しい十分shot 6,914と、旧価格primaryの6,912を混同しない。これらは保存報告の引用であり、今回再計算した数値ではない。

過去結果から新研究を始めることは可能だが、「前の試行と違う名前」を付けるだけでは足りない。変更する科学的自由度、強い対照、最小の判断可能な差を特定する。

## 3. C2-A：round依存PFの判断

### 3.1 既知部分と残る問い

UWCはround別の表現最適化に言及し、最終round支配を使う近似と古典最適化負担を説明している [P1]。位相推定精度に応じてHamiltonian精度を変える考え方も既知である [P5]。したがって「roundごとに方式を変える」こと自体を独立した新原理とはしない。

残る実用的な問いは、**PF族の変更による有利不利の入れ替わりが、既知のround別パラメータ最適化後にも存在するか**である。固定PF対照にまで全round同一stepを強制すれば、不当に弱い対照になる。

適切な固定族対照は、PF族jだけを全roundで共通とし、step幅、r/K、許される誤差配分を候補と同様に調整できるものとする。round別にPF族も選べる方式との違いを測る。

### 3.2 共通energy保証に、PF固有基底の一致は必須ではない

以下は、本研究の独立shot型RPEに対する十分条件の整理である。各roundで同じ|E0〉を新しく供給するとする。

\[
z_m=\langle E_0|Q_m|E_0\rangle,\qquad
\theta_m=-t_mE_0\pmod{2\pi}.
\]

Q_mは当該roundのcoherent meanであり、unitaryとは限らない。各roundの系統誤差と統計誤差を含む位相推定が、同じtheta_mからbeta_m以内にあると保証できればよい。各Q_mの固有基底・effective energyが同じであることは、この十分条件に含まれない。

たとえば、既知の正スケールa_mに対して

\[
|z_m-a_m e^{-it_mE_0}|\le d_m<a_m
\]

を示せれば、幾何学から

\[
|z_m|\ge a_m-d_m,\qquad
|\arg z_m+t_mE_0|_{2\pi}\le\arcsin(d_m/a_m)
\]

が得られる。実際には既存の位相・半径別上界でもよく、この円盤条件を必須にする必要はない。a_mとd_mを知るための情報は、各方式で利用可能なものに限定する。

時刻がt_m=2t_(m-1)で、前roundの正しいliftから最も近い現在候補を選ぶ場合、

\[
2\beta_{m-1}+\beta_m<\pi
\]

は正しい分枝を選ぶ十分条件である。理由は、真の非折返し位相が前roundの2倍であり、予測と正しい候補の距離が最大2beta_(m-1)+beta_mになるからである。初回alias windowが正しいこと、各誤差事象が成立することを仮定する。一様betaなら3beta<pi。各失敗確率の和を全体alpha以下にする。

これは一般のcoherent QPEへ無条件に拡張する定理ではない。また、既存のRPEロバスト条件の適用・整理であり、新しい共通energy理論を発明したという結論でもない。現在の保存行列モデルの保証資産 [I3] を基礎に、変えた構成だけを再適合させればよい。

### 3.3 新しいoptimizerの前に、切替利益の有無を確認する

同じalpha/beta配分等を固定した一次診断で、g_mjを「PF族jをround mで使い、許されたstep/r/Kを調整した後の適格なshot×full価格」とする。有限メニューで、追加のround間結合制約がない場合、

\[
G_{\rm fixed}=\min_j\sum_m g_{mj},\qquad
G_{\rm switch}=\sum_m\min_j g_{mj}\le G_{\rm fixed}.
\]

一つのjが全roundで最小なら、同じ有限モデル内の切替利益は0である。費用交差があれば利益があり得るが、実際の差、古典選択費、誤差予算・価格の不確かさを確認する必要がある。

この式は固定した費用表に対する理想選択の診断であって、未知E0やexact価格を使う実行可能selectorの証明ではない。alpha/betaの共同最適化や他の結合条件があれば、その制約を含めた最適化へ戻る。ここでの0を全メニュー・全問題のno-goにしてはいけない。

短round集合Sのみを変え、残りのroundのNと価格が変わらないなら、極端にSを無料化しても削減率は

\[
\frac{\sum_{m\in S,b}N_{m,b}\bar C_{m,b}}{G_{\rm total}}
\]

を超えない。現在の保存データにこの値を適用して再集計したわけではない。最終round支配を全ケースに決めつけるのではなく、まず利用可能な費用情報から上限を確かめるべき、という条件付きの判断規則である。

### 3.4 推奨する最小研究

既存の有限PFメニュー、同じ元H/state、同じsplitと合成policyを使い、まずroundごとの位相・半径・Nと費用交差を調べる。新しいPF係数探索や大型schedule frameworkは不要。既存のr/K・失敗配分にPF族ラベルを追加する程度の差分から始める。

長q費用が未知なら、外挿点値だけで勝者を認定しない。探索の候補絞り込みとして明示的な条件付きモデルを使うことはできる。新しい価格証拠は、費用交差があり、結論を変え得るround・構成に限定する。

**継続材料：** 強い固定PF族対照後にも費用交差が残り、共通energy条件を戻しても全費用の差が残る。  
**縮小材料：** 全round同じPF族が勝つ、短roundだけの削減上限が小さい、または単なる既存r/K調整の言い換えに尽きる。

## 4. C2-B：非一様精度だけでは利益は保証されない

### 4.1 既知研究との重複

Häner・Roetteler・Svoreは、subroutine別の誤差配分と費用最適化をすでに扱う [P2]。そのTheorem II.2は、頻度を伴う費用和と誤差予算を明示している。従って「頻度の違う回転へ違う精度を配る」だけを新規原理としない。

SP、Parent、ALLOCATIONは近接するが、主な変数はplacement、残差の係数、proposal等であり、すべてが決定論的Clifford+T精度の選択と同じではない [I1,I2]。旧C-Sの巨大normalizationは、その特定の残差方式の負結果である。

### 4.2 頻度は費用側と誤差側で相殺することがある

以下は制限した連続モデルの解析であり、実T価格の代替ではない。回転class jの出現回数をn_j、一回の合成費用を

\[
c_j(\epsilon_j)=a_j\log(1/\epsilon_j)+b_j
\]

とし、誤差をsum_j n_j epsilon_j <= Eで足す。内点最適解のLagrange条件は

\[
-\frac{n_j a_j}{\epsilon_j}+\lambda n_j=0,
\qquad
\epsilon_j=\frac{a_j}{\lambda}.
\]

a_jが共通なら最適epsilonも共通である。**頻度が大きいという理由だけでは、非一様性による利益は生じない。** epsilonの上下限・離散T価格・Clifford近傍・合成器特有の角度依存がある場合は別問題である。

誤差がsum_j n_j s_j epsilon_j <= Eの形になれば、同じモデルでepsilon_j=a_j/(lambda s_j)となる。本質は、s_jの違いに正当な根拠があるかである。未知の正確なstateやbiasを無料oracleとして使ってs_jを決める方法は、今回の運用仮定には含めない。

## 5. C2-Bを具体化する案：基底変換blockに応じた合成精度

### 5.1 条件と誤差式

以下はunitaryの共役構造から直接得る不等式であり、新規性を主張しない。向きの規約を

\[
U=BDB^\dagger,\qquad D=e^{-i\tau\Lambda}
\]

とする。LambdaはHermitianな対角生成子、Bはunitaryな基底変換である。

Bをunitary回路B_tildeで近似し、帰り側を独立合成せず**その実際のadjoint**にする。中心DはD_tildeで近似する。

\[
\widetilde U=\widetilde B\widetilde D\widetilde B^\dagger,\quad
\|B-\widetilde B\|\le\epsilon_B,\quad
\|D-\widetilde D\|\le\epsilon_D.
\]

任意の複素scalar zについて、zIの共役は不変なので

\[
\begin{aligned}
BDB^\dagger-\widetilde B D\widetilde B^\dagger
={}&(B-\widetilde B)(D-zI)B^\dagger\\
&+\widetilde B(D-zI)(B^\dagger-\widetilde B^\dagger).
\end{aligned}
\]

従って、

\[
\|U-\widetilde U\|
\le 2\|D-zI\|\epsilon_B+\epsilon_D.
\]

z=0ならusualな2epsilon_B+epsilon_D。Lambdaのスペクトルを含む区間の幅をWとし、その中心をcとすれば、z=exp(-i tau c)により

\[
\|D-e^{-i\tau c}I\|\le |\tau|\|\Lambda-cI\|\le |\tau|W/2.
\]

両方のboundを併用すると

\[
\boxed{
\|U-\widetilde U\|
\le \min(2,|\tau|W)\epsilon_B+\epsilon_D.
}
\]

Wは真のE0やgapではなく、当該対角fragmentの値・係数から得る幅上界でよい。exactな幅を求めるために大規模対角化を必要条件にしない。Wが保守的に大きければ利益は減る。

### 5.2 改善の意味

中心Dがidentityに近い場合、BとB†を独立に評価して誤差を足すより、basis誤差の係数が小さくなる。極限D=IではB_tilde I B_tilde†=Iなので、基底変換の近似誤差は消える。

同じblockをn=t/|tau|回使う単純な比較で、|tau|W<2なら、basis部分の保守加算はn×|tau|W epsilon_B=tW epsilon_Bとなる。naiveな2n epsilon_Bとは異なり、この寄与だけを見ればstepを細かくしたときに1/|tau|で悪化しない。非可換な他blockが挟まっても、各unitary block誤差のtelescoping加算は利用できる。ただし高次PFの負係数・複数stageでは各|tau_j|を足す必要があり、負時間を無料の相殺にしない。

これは許容合成精度を緩める余地を与えるが、実T数の減少を保証しない。離散合成列の価格、basisとdiagonalの比率、全時間、既存fusion、必要Nに依存する。

### 5.3 Controlled回路・phase・sectorの注意

controlled blockは

\[
C(U)=|0\rangle\langle0|\otimes I+|1\rangle\langle1|\otimes U
\]

なので、同じ相対phase規約なら||C(U)-C(U_tilde)||=||U-U_tilde||である。zIを引く操作はboundの導出上の操作であり、実回路のcontrolled scalarを削除する許可ではない。

B_tildeの帰り側が正確な実adjointでないなら、上の相殺は成り立たず、追加の不一致誤差を足す必要がある。Bの共通global phaseはexact adjointとの対で消せるが、中心Dの位相をprojectiveに消すことはできない。

理想Bが数保存でも、gateごとにClifford+T近似したB_tildeが同じsectorを厳密に保つとは限らない。boundとcorrectnessはfull空間で扱い、native中間gateをsectorへ射影してleakageを隠さない。

compiler Aは既に制御を付けたjoint回路への適用条件を持つ。lower後に新しい量子controlを付ける手順へ無断変更しない。basis blockの同定とactual inverseの保持が現SOURCEで可能かを最小の接続課題として確認する。

### 5.4 PR全体・必要shot数への接続

初期検証ではsampling law、canonical weight、元PF、splitを変えず、決定論側の適合するbasis blockの合成だけを変える。これなら新しいresidual lawや追加normalizerを導入しない。ただし測定mean自体は合成誤差で変わるので、必要Nが自動的に不変になるわけではない。

path xのfull operator差をd_xで押さえ、既存のraw/corrected規約に沿って平均のsignal diskへ伝播する。rawの正確率lawならE[d_x]または一様上界を使える。weighted meanならE[|W_x|d_x]を扱う。norm/期待値/二軸の既存変換定数を勝手に変更せず、同じerror accountingへ渡す。

共通のround別合成disk上限を全armに与え、同じ保守rho/beta/NでT費用だけを比較する段階は可能である。その後、全armに同じ誤差再配分機会を与えてN×Tを再評価する。前者は誤差会計の差を分離するablation、後者が総費用の主比較になる。

RTEの高order pathには、中心がidentityに近いとは限らない積・符号演算が含まれる。上のsmall-tau利点をすべてのrandom eventへ一律適用しない。初期候補は数式が適合する決定論blockに限定することを推奨する。

### 5.5 既存コード・既知法との関係

保証SOURCEにはすでに `generator_from_basis`、`d_task` があり、B diag(E) B†の誤差を生成Hamiltonianへ接続している [I6]。従って「基底誤差を時間に依存する形で扱う」という数学自体をproject内で未発見としない。

一方、合成SOURCE `acquire(theta, exponent)`、`strict_guard`、`exact_word_matrix` はactual-angle/precision keyとstrict phaseを扱うが、現` t_metrics(qc, exponent)`は単一exponentを受け取る [I5]。basis blockとその逆回路を一組として精度選択する仕組みが、ここに完成済みであるとは確認していない。

必要なのは新しい合成器や保証基盤ではなく、**既存の構造誤差評価を、現物の合成精度選択・実T価格・RPEのNへつなぐ小さい接続**である。この接続が既知方法の応用に還元される可能性は残る。それでも全費用の有利条件・不利条件を示す研究にはなり得る。

### 5.6 強対照と最小検証

全方式に同じactual adjoint再利用、basis融合、Clifford処理、角度情報、合成器を与える。次の2×2を概念上分ける。

| 誤差会計／精度選択 | 最適化した一様精度 | 最適化した非一様精度 |
|---|---|---|
| 従来のgate/class加算 | 一様対照 | 既知非一様配分対照 |
| block構造を反映 | 構造評価だけの利益 | 構造＋配分の追加利益 |

全4armを大量scanする要求ではない。どの差を主張するかが分かる最小対照を残す。構造評価を対照にも与えると差が消える場合、非一様optimizer独自の利益とは呼ばない。純粋に保守boundを改善したことによる実装費低下なら、その貢献を正確に記す。

最初は既存固定PFと適合するDF blockを使い、kappa=min(2,|tau|W)が実際に2より小さいか、該当basisのT費が十分な割合か、actual inverseを共有できるかを判別する。次いで少数の実合成keyとfull signal/errorを接続する。最初から全primitiveの新精度gridを生成しない。

**継続材料：** 既知配分・構造aware対照後にも、実Tの低下がN増加を上回る。  
**縮小材料：** kappaがほぼ2、basis費が小さい、既知対照と同じ精度へ落ちる、compiler/fusion喪失で利得が消える、またはstrict phaseと成功条件が閉じない。

## 6. C2-C：位相biasと半径を同時に見る必要がある

### 6.1 既知研究と現段階の不足

Mehendaleらは固有値摂動を用いたPF・partition選択を扱い、Hejaziらはenergy推定に特化したPF誤差解析と9項の二次公式を示している [P3,P4]。従って「energy誤差を目的関数にする」という方向は既知である。

今回のC2資料には、新しい係数列・非共役構成・取得可能な情報を使う選択規則が特定されていない。一般名だけを理由にP-D/BFの探索へ戻るべきではない。

### 6.2 なぜ固有値biasだけでは不足するか

以下はideal unitary PFについての解析。K_h=exp(-ih H_eff)の固有状態を|phi_j〉、そのenergy branchをE_j^effとする。元H groundから見た重みをw_j=|〈phi_j|E0〉|²とすれば、t=nhで

\[
z(t)=\sum_j w_j e^{-itE_j^{\rm eff}}.
\]

対象branchの重みを1-w、energyをE_starとすると

\[
|z(t)-e^{-itE_\star}|\le 2w.
\]

従って2w<1なら

\[
|z(t)|\ge1-2w,\qquad
|\arg z(t)+tE_0|_{2\pi}
\le t|E_\star-E_0|+\arcsin(2w)
\]

という十分な評価が得られる。右辺が小さい範囲で位相保証へ使う。E_starのbranchやwを無料で利用できるという主張ではなく、何を評価すべきかを示す式である。

補正finite mean Qが||Q-K_h^n||<=etaなら、上の円盤半径は2w+etaに拡張できる。raw信号がcorrected/Bなら、その半径下界は(1-2w-eta)/Bになる。2w+eta<1の範囲が必要で、Bとweight負担を二重計上しない。

非縮退の孤立固有値、十分小さいepsilon、解析的摂動H_eff=H+epsilon V+O(epsilon²)を仮定すれば、

\[
E_\star-E_0=\epsilon V_{00}+O(\epsilon^2),\qquad
w=\epsilon^2\sum_{n\ne0}\frac{|V_{n0}|^2}{(E_n-E_0)^2}+O(\epsilon^3).
\]

対角要素V00の低減は、非対角成分による混入の低減を保証しない。gapが小さい条件では、この単純な展開を使えない。未知gapを仮定の追加で無料化もしない。

同じkernelへの純粋共役P†KPは固有値を変えず、重みを変える。旧C-Pの負結果はその指定処理の範囲であり、別kernelを否定しない。一方、内部H_Dだけにenergy高次性を持つ公式を適用しても、入力が全Hのgroundで、外側D/Rとfinite-RTEが残るため、同じenergy次数が全PRへ自動移るわけではない。

### 6.3 今回の推奨

C2-Cを永久に捨てる理由はないが、現段階で新しい係数探索を始める根拠は足りない。非共役の具体的構成と「biasの改善が半径・negative-stage費に潰されない理由」が示された時に再判定する。

既知energy特化PFをAの有限メニューまたは強baselineへ加えることは別の応用検証である。その場合も、既に他repositoryで扱っていないか、係数版・内部D/外側split・負時間を照合し、未実施と断定しない。

## 7. 並列実施と比較の公正性

| stream | 独立に変えるもの | 固定するもの | 共有可能な資産 |
|---|---|---|---|
| C2-A | roundごとのPF族・許された構成選択 | 元H/state、合成policy、compiler、比較の情報条件 | PF registry、signed RTE、RPE半径/branch会計、価格provider |
| C2-B | block誤差評価に基づく決定論的合成精度 | 元H/state、PF schedule、sampling law、split、compiler原則 | strict合成key/guard、actual adjoint、basis/generator評価、RPE会計 |
| C2-C | 具体的構成未提示のため数値探索保留 | 旧STOPと科学前提 | 位相/混入診断、BCH・P-D/BFの負材料 |

Aで安かったPFを事後にBへ移し、それを独立B結果としない。単独機構が支持された後、組合せを別の有限比較として調べる。改善率を掛け算して総改善率にしない。

「強い対照」とは、実在する既知方式に同じ利用可能情報、合成器、適用可能fusion、誤差配分機会を与えることである。候補だけにexact E0/priceを無料で与えない。一方、対照に未実装の万能optimizerや候補の全独自手順を無条件に与えて、定義上改善を不可能にする比較も行わない。

元H表現の厳密包含や全compilerの形式検証を、探索pilotの一律開始条件にしない。保存行列・理想gateモデルの結果であることを明記して、条件付きに研究を進められる。最終的な物理energyの保証付きclaimには、その外側の対応を閉じる必要がある。

## 8. 次の最小作業：追加の資料整理だけで終えない

今回は「もう一度広い文献調査・棚卸し」を勧めない。既存資料を再利用し、正式採択・対象範囲の実行許可後に、以下を小さい科学的判別として進める。

### A：切替の余地を調べる

既存有限PFメニューについて、同じ時刻・元H/state・成功条件で適格なg_mjを対応させる。保存signalや価格を使う科学的再会計も、新実行の許可範囲へ明示する。価格の未知幅を保持したまま、PF族の有利不利が入れ替わるか、短round改善の上限が十分かを確認する。利益が見えなければ新optimizer・大量compileへ進まない。

### B：block構造が実精度を変えるか調べる

既存DFの固定構成一つを対象に、basis/diagonal/actual inverseの対応とW・tauを使う誤差重みを確認する。旧コードの数値basis誤差と、今後選ぶ物理Clifford+T合成誤差は区別する。既知一様・非一様対照と同じfusion/adjoint条件で、少数keyの実T価格に差が出るか調べる。合成disk・rho・Nまで戻して判断する。

Bの単回blockが安くなっただけでは完了としない。全roundへ接続可能な誤差会計と、実現したfull回路の費用、必要Nの変化をセットで残す。

### 保存・終了

主な入力・SOURCE・変更した自由度・比較対象・成功条件・必要科学callを固定し、新しい出力へ保存する。旧runやgrant/marker/STOPは変更しない。新しい承認証明書やgrant発行基盤は作らない。必要な独立処理は利用可能資源の下で並列化し、共有環境・他ユーザーに影響する変更は行わない。

承認範囲が完了、同じ既知対照へ還元、成功条件不成立、価格不確かさで判断不能等の時点でSTOPし、有限条件の結論を報告する。負結果を救うための無断grid拡大はしない。

## 9. 主張・確信度・留保

| 判断 | 確信度 | 適用条件・留保 |
|---|---|---|
| 旧Cの結果はC2全体のno-goではない | 高 | 旧結果の有限条件と新変数が異なる |
| fresh-shot型RPEは共通energyに対する位相誤差条件で異PFを接続できる | 高 | 初回alias・統計条件・誤差上界が必要。一般coherent QPEへは拡張しない |
| 頻度だけでは非一様精度の利益は出ない場合がある | 高 | 共通log-slope・線形誤差・内点の連続モデル |
| actual adjointを共有した共役blockの誤差式 | 高 | unitary B/D、strict中心phase、全空間bound |
| Bの実T削減につながる可能性 | 中・未実証 | 小さいtau W、basis費、離散合成価格、N、既知対照に依存 |
| Aの費用切替利益 | 未確定 | 費用交差と全round寄与を数値評価していない |
| Bの世界的新規性 | 未確定 | 標準的な共役誤差の適用。限定検索で初出とは主張できない |
| C2-Cに具体的な新PF機構がある | 現資料では未特定 | 研究可能性を否定せず、数値探索開始の材料が不足 |

## 10. 参照資料と取得限界

### 内部一次資料

[I1] C2限定調査REPORT（固定6f6c084b）  
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6f6c084bdfaf04ba951fd5d8ea0c521313ceb0b3/docs/research/track_c2_limited_investigation_20261106/REPORT.md

[I2] C2候補比較（同固定版）  
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6f6c084bdfaf04ba951fd5d8ea0c521313ceb0b3/docs/research/track_c2_limited_investigation_20261106/CANDIDATE_COMPARISON.md

[I3] Track C A保証（固定720416c5）  
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/720416c5b88b227cf76ff7aa729d91a0a98ed58b/docs/research/track_c_guarantee_cn_price_20261106/A_REPORT.md

[I4] Track C B価格（同固定版）  
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/720416c5b88b227cf76ff7aa729d91a0a98ed58b/docs/research/track_c_guarantee_cn_price_20261106/B_REPORT.md

[I5] 合成SOURCE（同固定版）。全文の静的読解のみ。旧opt2を説明するdocstringは、新compiler Aの価格資格を意味しない。  
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/720416c5b88b227cf76ff7aa729d91a0a98ed58b/src/trottertracks/track_c_h4_synthesis.py

[I6] SOURCE/mean保証（同固定版）。今回直接確認したのは冒頭190行のbasis/generator/誤差伝播等。  
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/720416c5b88b227cf76ff7aa729d91a0a98ed58b/src/trottertracks/track_c_source_guarantee.py

### 外部一次資料

[P1] Kanasugi et al., *Enabling Chemically Accurate Quantum Phase Estimation in the Early Fault-Tolerant Regime*, arXiv:2603.22778v2. §III.C、Eqs.33–34周辺のround依存最適化と最終round近似をPDF本文・p.8画像で確認。HTML取得は途中までのため、後半はPDFを使用。  
https://arxiv.org/abs/2603.22778v2

[P2] Häner, Roetteler, Svore, *Managing Approximation Errors in Quantum Programs*, arXiv:1807.02336v1. §II Lemma II.1/Theorem II.2の合成誤差・頻度・費用をPDF抽出本文で確認。スクリーンショット取得はcache errorで不成功。図の細部を根拠とする主張は行っていない。  
https://arxiv.org/abs/1807.02336v1

[P3] Mehendale et al., *Estimating Trotter Approximation Errors to Optimize Hamiltonian Partitioning for Lower Eigenvalue Errors*, arXiv:2312.13282v3. HTMLの§I–IIIとApp.A.7の所在・関連記述を確認。摂動によるenergy評価は厳密上界と区別。  
https://arxiv.org/html/2312.13282v3

[P4] Hejazi et al., *Better product formulas for quantum phase estimation*, arXiv:2412.16811v1. 初回取得でv1関連本文、再取得でabstractと版情報を確認。v2 URLは取得不能であり、v2を読了と扱わない。9項公式の存在はabstractにも明記される。本レビューで係数表を独立検算していない。  
https://arxiv.org/abs/2412.16811v1

[P5] Kivlichan, Granade, Wiebe, *Phase estimation with randomized Hamiltonians*, arXiv:1907.10070v1. abstractを確認。可変Hamiltonian精度という先行着想の根拠として使用し、gap/variance定理を本研究へ適用したとはしない。  
https://arxiv.org/abs/1907.10070v1

既存PR、SPRINT、PAI等の全一次資料をこのC2レビューで再読したわけではない。C2資料の既知関係を継承した箇所はその帰属のまま保持する。限定調査で「類似論文がない」と確定した候補はない。

## 11. 完了と次の判断

科学レビューを完了し、本Markdownを保存した。新しい性能値・改善率は生成していない。旧SOURCE、入力、raw、結果、grant、ledger、marker、STOP、共有環境は不変更である。

次のユーザー判断は、**C2-Aの有限PF切替判別と、C2-Bのblock構造に応じた決定論的合成精度の検証を、並列の研究範囲として採るか**である。採択後のCodex作業は、この具体的な問いに対する最小の設計・接続・科学検証とし、再び一般的な資料整理へ戻す必要はない。

本レビューの推奨は、科学計算・保存値再解析・実装・compile・合成の実行許可を代行しない。実際の対象範囲への明示的許可後に開始する。C2-Cと旧C-N/C-S/C-P等の停止条件を、このレビューだけで解除しない。
