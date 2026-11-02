# 量子化学の基底エネルギー計算に向けた低コスト時間発展：研究方針レビュー

- 作成日：2026-10-13（日本時間）
- 開始承認：利用者の「レビューを開始して」
- 対象：量子化学Hamiltonianの基底エネルギー計算を低コスト化するためのPR、PF、分解・前処理、mixed propagator、位相推定との共同設計。
- 正式な運用指示：`research_project_instructions_gpt_codex_review_approval_2026-10-12.md`。研究全体の俯瞰、Codex先行、科学レビューと計算認可の分離を適用。
- 実施：固定GitHub資料の読取り、一次文献調査、解析的な構成・誤差評価、候補比較、研究記録作成。
- 非実施：分子積分の新規生成、固有値計算、最適化、科学入力生成、証拠探索、回路compile、sampling、既存実験再実行、Git書込み。
- 判定：次に構成可能性を調べる研究候補の選択。新規性、資源優位、論文化可能性の確定ではない。

## 0. 結論

**第一候補は、分子の対角エネルギー差を用いて、励起ごとの相対時間発展をコヒーレントに直接生成する方法とする。**

研究の対象は、一般的な `exp[-it(A+B)]` を速くすることではなく、実際の分子Hamiltonianから生成した `A+B_j` の構造と、使用する外側の積公式に現れる相殺を利用することである。THRIFTのmixed propagatorは有力な接続先だが、THRIFTやPRの採用自体を成功条件にしない。

具体的には、

\[
H=A+\sum_j B_j,\qquad C_j(t)=e^{itA}e^{-it(A+B_j)}
\]

を考える。`A` を占有数に対して対角な一体・二体部分、`B_j` を一つの励起とその随伴から作ると、各有効な占有対で `C_j` は二準位の相対回転になる。共通平均エネルギーは相殺し、残る占有依存エネルギー差は、対角Hamiltonianが二体までならspectator占有数のアフィン関数である。

この代数自体や条件付き二準位回転は、新規性として認定しない。**候補の寄与は、指数個の占有枝を列挙せず、相殺を保った有限精度回路を生成し、原Hamiltonianの基底エネルギーを同精度で求める全費用を下げられるか**にある。

**第二候補は、近似状態の情報誤差を含めて、基底固有位相の主要誤差を抑える固定ユニタリPFの構成。** ただし、軌道・順序変更によるenergy-error averaging/cancellationや摂動誤差による分割選択は既報である。単にその再実装を行う案は採択しない。真の基底状態なしでの設計、情報取得費用、有限stepでの位相の一貫性に独立した差が残るかが条件となる。

PRの分割・sampling・DFとの共同設計は予備候補として保持する。既知のPR、SPRINT、GRADE、重要度samplingを組み合わせただけで新規と扱わない。

今回、前回の「スペクトル応答保存PF」を最優先から外す。理由は、その理論を否定したからではなく、利用者が研究対象を「量子化学の基底エネルギー計算」に再指定したためである。

---

## 1. 目的を固定し、手法を固定しない

### 1.1 求める出力

目標は、明示した電子Hamiltonianに対して、所定の基底エネルギー精度と成功確率を満たす推定を得ることである。有限基底・active spaceの電子問題と、完全基底極限や化学モデル全体の精度は区別する。

PR、決定論PF、THRIFT、相互作用描像、Hamiltonian分解は、そのための手段である。PRとTHRIFTを組み合わせることを目的化しない。スペクトル強度・励起エネルギー差・時間依存相関へ目的を変更して、基底エネルギーの成果として報告しない。

### 1.2 資源の対象

単一時刻の一回路ではなく、採用するQPE/RPE/単一ancilla推定器に必要な時刻列を通した費用を評価する。例えば一つのゲート指標について、

\[
G_{\rm total}=\sum_k N_k\left(G_{\rm prep}+G_{\rm controlled\ evolution}(t_k)+G_{\rm readout}\right)
\]

を用いられる。これは評価枠組みの提案で、実測値ではない。`N_k` は推定器の統計保証と、必要ならnormalizationやstate overlapに応じて決める。

RZ、CX、logical T/Toffoli、深さ、logical qubit数、古典前処理時間を単位の違うまま加算しない。主指標と副指標、合成精度、補助qubit上限は実行前に固定する。状態準備の費用が未取得なら、共通条件付きの評価または明示したパラメータ依存として示し、無料とはしない。

### 1.3 誤差の役割

少なくとも、モデル・DF打切り、時間発展近似、有限精度の係数演算／回転合成、統計推定、基底branchの同定を分ける。新方式がHamiltonian自体を近似した場合、元Hamiltonianへの誤差を失わない。

回路ノルム誤差をenergy biasへ換算するには、その推定器とbranchの条件が必要である。短時間誤差の小ささを、そのまま長い時刻列での基底エネルギー保証としない。

---

## 2. 既存研究の到達点と、今回引き継ぐ範囲

### 2.1 THRIFTの旧toy検証

固定commit `c24dcede10ed9726e1e0b1030eca749ba7bc5cba` の結果第4節を取得した。[R1]

- 対象は対角Ising `A` と `alpha * sum X_i` の2・3qubit模型。
- graphのdegreeを2以下、分岐を4以下としたconditional SU(2)構成。
- 同じ信号精度・compiler・taskで通常一次PF、対称S2、THRIFTを比較。
- 登録8条件ではS2のRZ/CX仕事量がTHRIFTより小さかった。
- 実分子、denseな量子化学のspectator依存、可逆算術を用いる多分岐回避は検証していない。

従って、旧結果は「その実装と入力ではmixedアクセスの費用を回収できなかった」という限定結果である。量子化学での全mixed構成が無価値だという証拠ではない。

### 2.2 Track AのPR資源結果

固定H6報告では、同じ有限時間complex signalの精度についてPRの有利な点推定がある一方、状態準備費用の除外、希少orderの未観測、母平均費用の保証、最終RPE/ground-energy費用は未確定である。[R2]

今回、この結果を新しいenergy-task優位の証拠へ格上げしない。引き継ぐのは、full-wrapper、同じ精度、normalization、希少event、費用単位を分ける評価基盤である。

### 2.3 N3とTrack Bの停止を維持

N3のclosureは、理論と限定結果を保持したまま現追加benchmark・帰属系列を終了し、主研究として未採択としている。[R3]

Gaussian積生成候補のclosureは、性能が負けたためではなく、記述された手順に独立した新規操作が定義されていないための非採択であり、Track B全体の終了ではない。[R4]

今回の候補は、それらの名前を変えて再開するものではない。N1停止、N2保留、元のgate・STOP・raw結果は変更しない。過去の固定条件を新候補の普遍的制約にもしない。特に、本候補の対角係数 `J` にN3の「非負density」を引き継がない。

### 2.4 前回の広域レビューからの変更

前回のスペクトル応答案は、finite-windowの強度を対象にする研究候補だった。[R5] 今回は基底エネルギーを対象とするので、その順位を下げる。一般的なcost-aware word samplingも予備候補へ下げる。

これは利用者の研究目的の明確化に対応する変更であり、同じ目的・同じ証拠で恣意的に順位を変更するものではない。

---

## 3. 一次文献から確認した重複と未確定事項

| 文献 | 今回確認した範囲 | 今回への含意 |
|---|---|---|
| Bosseほか、THRIFT、Nature Communications 16, 2673 (2025) [L1] | 出版社全文、定理1・高次構成・mixedアクセス条件・Fermi–Hubbardでの費用上の制約 | `exp[-it(A+B_j)]` を安く実装できることは元手法の明示的な条件。THRIFTを適用したというだけで新規ではない |
| Patel–Yen–Izmaylov、対称性による可解Hamiltonianの拡張 [L2] | arXiv v2 PDFの定義、fermion例、projectorごとの対角化、付録のcontrolled unitary表示 | occupation条件付き可解性や二準位化の一般原理は既知。効率的測定と、全枝でcoherentなcontrolled evolutionの費用を区別 |
| Casaresほか、SPRINT/GRADE [L3] | v1 PDF、導入、near-integrable・processing・randomizationの関連箇所 | DF等の分解＋near-integrable PF＋乱択という一般的組合せは既に広く扱われている |
| Güntherほか、PR [L4] | arXiv現行書誌v2、取得された旧版PDFの分割・位相推定・表現の節 | PRをenergy estimationへ接続すること、tail係数を使うsplit最適化だけでは新規でない。v2全文の独立取得は未成功 |
| Mehendaleほか、Digital Discovery (2025) [L5] | 出版社公開本文・結論 | energy errorを摂動期待値で推定して分割を評価する方法は既報。近似stateと費用も議論される |
| Kronenberger–Erakovic–Reiher、arXiv:2602.18913v1 [L6] | 公開HTML第2.5・4.3・結論 | 軌道／順序を変えて基底energy errorを平均・相殺する方向も既報。真値を使わず符号を知る困難を明記 |
| Maxwellほか、arXiv:2606.30738v1 [L7] | 公開HTMLとabstract、BCHと固有基底の対角成分の位置付け | BCH期待値を使うこと、compact BCHやimportance samplingの一般論を新規としない |
| Yi–Crosson、arXiv:2102.12655 [L8] | abstract | QPEに適した固有値・固有vector摂動の考え方は既存。第二候補で一般的なspectral-error改善を初提案とはしない |

**文献があることは提案の全可能性を否定しないが、文献にないことを本レビューで網羅的に証明したわけでもない。** 特に分子mixed-blockの可逆算術・table/interpolationとの完全な既報性比較は、次のCodex構成監査の課題である。

THRIFTの文献を「分子Hamiltonianで既に速いと証明した論文」とは扱わない。今回確認した本文の比較模型はspin系・Fermi–Hubbardであり、mixed実装費用が不利になる場合も記述されている。[L1]

---

## 4. 第一候補：分子励起の相対時間発展を直接生成する

### 4.1 問い

> 分子の占有エネルギー差の構造を用いて、mixed propagatorの共通対角発展を相殺した量を、指数的な分岐表なしに、有限精度でcoherentに生成できるか。その生成費用を含めても、基底エネルギー推定が既存のnative PF/PRより安くなるか。

「二準位に落ちる」ことと「全分子のsimulationが安くなる」ことを分ける。最初から実用優位を仮定しない。

### 4.2 分解

実係数で粒子数を保存する有限基底の電子Hamiltonianを、検討の出発点として

\[
H=E_{\rm nuc}+A+\sum_j B_j,
\quad
A=\sum_p\epsilon_p n_p+\sum_{p<q}J_{pq}n_pn_q
\]

と書く。`A` は選んだ軌道基底での対角部分。`J` は一般に符号を持ち、非負性は必要ない。

`B_j` は一つの非対角励起monomialと随伴の組とする。例：

\[
B_{pq}=g_{pq}(a_p^\dagger a_q+a_q^\dagger a_p),
\]

\[
B_{pqrs}=g_{pqrs}(a_p^\dagger a_q^\dagger a_s a_r+\mathrm{h.c.}).
\]

添字の重なる二体項はdensity-assistedな一体励起へ整理できる。必要なら占有数因子を別のspectator制御として持つ。全二体Hamiltonianを一体Gaussian問題へ還元したという意味ではない。

数値係数・spin convention・fermionic sign・scalar shiftは、元の分子係数にbindする。モデルやDFを打ち切るならその誤差を別記する。

### 4.3 二準位構造

一つの励起の作用を受ける二つのoccupation patternを `chi_0, chi_1` とし、それ以外のoccupationを `s` とする。有効対で

\[
(A+B_j)|_s
=\bar d_j(s)I+\frac{\Delta_j(s)}2 Z+b_j(s)X
\]

と書ける。実係数のfermionic parity signは `b_j(s)` または明示的なparity計算へ含める。複素係数を扱う場合はX/Y成分へ拡張が必要であり、本レビューはその実装を完成したとはしない。

無効なoccupation patternでは `B_j=0` である。スペクテータoccupationを測定することはしない。量子superpositionのまま全枝で正しい作用を実装する。

`A` のdegreeが2なら、同じspectatorの対角二体項は差を取ると消えるため、

\[
\Delta_j(s)=\Delta_{j,0}+\sum_{k\notin\mathrm{supp}(B_j)}d_{jk}n_k.
\]

一般にO(N)個の係数を持つアフィン関数であり、2^(N-k)個のoccupation枝を列挙する必要は代数上ない。これは本レビューによる構成上の整理であり、新規性の認定ではない。

例えば一体励起でpを占有したpatternとqを占有したpatternを比べると、符号の定義を固定したうえで、

\[
\Delta_{pq}(s)=\epsilon_p-\epsilon_q+
\sum_{k\ne p,q}(J_{pk}-J_{qk})n_k.
\]

double excitationでも、支持内の定数とspectatorに対する結合差の和になる。density-assistedな振幅は、同じspectatorが励起の前後で不変であるため、別の既知関数 `b_j(s)` として扱える。

### 4.4 実装対象はmixed演算子を単独に分けず、相対積とする

THRIFTのseedには

\[
U_{\rm seed}(t)=e^{-itA}\prod_j C_j(t),\qquad
C_j(t)=e^{itA}e^{-it(A+B_j)}
\]

が現れる。これはTHRIFT定理1の構成を対応させたもので、積の順序は契約で固定する。[L1]

有効対では

\[
\boxed{C_j(t)|_s=
 e^{it\Delta_j(s)Z/2}
 e^{-it[\Delta_j(s)Z/2+b_j(s)X]}.}
\]

`bar d_j(s)` は相殺する。無効対では `C_j=I`。全systemの大きい `A` をmixed blockのたびに単独生成・逆生成する必要がない構成を狙う。外側に残る `exp(-itA)` と核反発等の既知位相は、正しく会計する。

これは数式上の相殺であり、コンパイラが同じ相殺を既に実現できるなら追加の新規性にはならない。既存THRIFTやfermionic gateの最適化済み実装を対照にする。

### 4.5 想定する回路の役割

1. 支持上のoccupationが有効な励起対かを、可逆に判別する。
2. 励起対を論理二準位へ写し、必要なfermionic parityを計算する。
3. spectatorから `Delta_j(s)` と必要なら `b_j(s)` をb-bitレジスタに計算する。
4. 相対SU(2)回転を、可変角回転・可逆関数計算・適切な圧縮table等で生成する。
5. 一時レジスタをuncomputeし、元のmode順序へ戻す。

b-bitの固定係数加算なら、単純な見積りはO(Nb)の可逆演算を含む。これは演算部分の提案上の次数で、完成回路のT-countではない。回転関数評価、精度、coherent control、table準備、補助qubit、routing、uncomputeの費用を含める必要がある。

2^b要素のtableを用意して「b-bitだから多項式」とはしない。全spectator枝の角度tableも不可。実際の圧縮・関数計算の費用を示せない場合は、効率的accessは未成立と扱う。

QPEのcontrolで必要な相対位相を消してはいけない。mid-circuitでspectatorを測定して回路を古典分岐させる手順は、通常のcoherent controlled-unitaryを代替しない。測定task向けの可解Hamiltonian実装を、そのままQPEへ流用しない。[L2]

---

## 5. 第一候補の解析：対角差の近似と相殺を一緒に設計する

以下は本レビューでの解析的導出であり、登録実験の結果でも新規定理の優先性認定でもない。

### 5.1 定義

一つの有効対について

\[
C_{\Delta,b}(t)=e^{it\Delta Z/2}e^{-it(\Delta Z/2+bX)}
\]

を考える。実数の `Delta,b`、実数時間tを仮定する。近似された `tilde Delta` を両指数に共通に用いて相対積を実装する。これは各指数の近似を無関係に選ぶ方法とは異なる。

### 5.2 誤差bound

interaction pictureでは

\[
C_{\Delta,b}(t)=\mathcal T\exp\left[-i\int_0^t V_\Delta(s)\,ds\right],
\quad V_\Delta(s)=e^{is\Delta Z/2}bX e^{-is\Delta Z/2}.
\]

任意の実Delta、tildeDeltaについて

\[
\|V_\Delta(s)-V_{\tilde\Delta}(s)\|
=2|b|\left|\sin\frac{(\Delta-\tilde\Delta)s}{2}\right|
\le |b|\,|\Delta-\tilde\Delta|\,|s|.
\]

時間依存unitaryのDuhamel差を積分すると

\[
\boxed{\|C_{\Delta,b}(t)-C_{\tilde\Delta,b}(t)\|
\le\frac12 |b|\,|\Delta-\tilde\Delta|\,t^2.}
\]

全spectatorで `|Delta-tildeDelta| <= eta_Delta`、`|b| <= b_max` なら、そのdirect-sum全体でも同じ最大値boundが使える。

振幅も `|b-tildeb|<=eta_b` で近似し、残りの回転合成等の誤差を `epsilon_impl` とすれば、triangle inequalityで

\[
\|C_{\Delta,b}-\widetilde C\|
\le \frac12 b_{\max}\eta_\Delta t^2+|t|\eta_b+\epsilon_{\rm impl}
\]

という十分boundが得られる。parity・有効対判別・controlの不具合はこのスカラー近似boundには含めず、別に正しさを検証する。

### 5.3 何が有用か

- `b=0` なら、同じ近似Deltaを両側へ使った理想相対積は厳密にIである。
- 対角差の誤差は、ここでは単に `O(t eta_Delta)` ではなく `O(b t^2 eta_Delta)` として入る。
- 1/Deltaを使う摂動展開ではないので、Delta=0近傍だけを理由にこのboundは発散しない。
- ただし、実際の角度計算・回転合成は零結合でのidentityを保存するよう設計する必要がある。誤差のある独立gateを並べれば、自動的にこの性質が成立するわけではない。

### 5.4 THRIFTの誤差を自動的に保持するわけではない

`B_j=alpha * B_j^0` なら、上の対角差の実装誤差は `O(alpha eta_Delta t^2)` である。alphaを小さくしてもeta_Deltaを固定すれば、理想THRIFTのalpha^2特性を支配する可能性がある。

従って、外側の目標誤差に合わせた精度配分が必要である。対称・高次のseed、負時間stageを含む構成では、各stageの時間と実装誤差を足し合わせる。高次THRIFTの局所誤差次数を、この二準位boundだけから認定しない。[L1]

また、`C(-t)` と `C(t)^dagger` は一般に同じ順序ではない。高次化で必要なadjointは正しい全seedから生成し、相対blockごとの符号変更だけで代用しない。

### 5.5 全系への累積

同じAを外側で正しく実装し、一つの小stepに各C_jが一回、合計r step・総時間T=rhを用いる単純な場合には、対角差近似のunitary誤差の十分boundは

\[
\epsilon_{\rm access}(T)
\le \frac{Th}{2}\sum_j b_{j,\max}\eta_{\Delta,j}
+T\sum_j\eta_{b,j}
+\sum_{\rm all\ occurrences}\epsilon_{{\rm impl},j}
\]

となる。外側のPF誤差、A自体の合成誤差、モデル誤差は別途である。この式は粗いtelescoping boundで、資源優位を証明したものではない。

---

## 6. 第一候補の新規性と、強い反論

### 6.1 新規と主張しない要素

- THRIFTの積そのもの。[L1]
- occupation対の二準位化や条件付き可解Hamiltonian。[L2]
- coherent arithmetic、可変角回転、QROMそのもの。
- Gaussian変換やfermionic excitation gateを使用すること。
- 単に分子で初めて試す、という対象拡張だけ。

### 6.2 差分候補

検討に値するのは、以下を一体として持つ構成が既存法と実質的に異なる場合である。

- molecular coefficientから励起・エネルギー差の圧縮表現を生成する。
- global Aの反復生成とspectator全列挙を回避する。
- 相対積の零結合相殺を保ち、係数・関数近似の費用を相殺後の誤差尺度へ対応させる。
- 支持と対角差構造から、まとめてもcoherentに扱えるgroupを構成する。
- 最終のground-energy精度・state overlap・QPE/RPE時刻列に対して原Hamiltonianからの資源利益を評価する。

全部を新規性の必須条件として足しているのではない。数学・構成・実装のどこに独立した寄与を置くかを明確にするための候補である。

### 6.3 励起数が最大の障壁になり得る

一般二体HamiltonianにはO(N^4)個の励起係数があり得る。単純に各励起へO(Nb)の算術を付けると、一stepはO(N^5 b)規模になり得る。これは素朴構成の演算見積りで、下界ではない。

したがって、「2x2を解けばよい」だけではDF-native PFに勝てない。group数削減、低ランク構造、共通計算の再利用、あるいはstep数の大幅な削減が必要になり得る。しかし、それらを実装前から存在するものとして費用を減らさない。

### 6.4 Aが対角ならnear-integrableとは限らない

実分子のある軌道基底で、非対角部分が小さいとは限らない。強相関や結合伸長では特に注意する。核反発などの大きなidentityをAへ含めて、見かけ上の比率alphaを小さくする操作は、非可換誤差改善の根拠にならない。

人工alpha-scanは機構試験であり、実際の分子係数での利益の証拠とは分ける。

### 6.5 同じ構造を通常PFへも与える

mixed blockだけに有利な分割・係数圧縮・control最適化を与え、通常PFを素朴Pauli実装に固定してはならない。対照には少なくとも同じ分子・同じ精度のnative S2と高次PF、適用可能なnear-integrable PF、既存PRを含める。PFのorderが違う場合は同orderだけを強制せず、最終精度でそれぞれを公平に最適化する。

SPRINT/GRADEは一般的な統合改善が既にあることの参照として使い、別のspectroscopy問題での倍率を基底エネルギーの倍率へ移さない。[L3]

### 6.6 THRIFT＋PRは後段の比較候補

まずdeterministicな相対blockを生成できるかを確認する。PRは、その後に残る高費用・小weight部分をどう扱うかという選択肢である。

ランダムにB_jを一つ選ぶだけでTHRIFT全体が不偏になるとは限らない。first operator moment、平均channel、QPEで必要な信号は別である。weight/normalization、複数stepの独立性、有限tail、全wrapper費用を定義せずPR hybridのGOを出さない。

---

## 7. 第一候補の最初の検証

### 7.1 最初に答える問い

最初から全分子simulationを走らせるのではなく、次の三点を一つの技術準備にまとめる。

1. **構成の独立性**：既知のconditional solvability、sparse simulation、既存fermionic回路の直接再記述ではないか。
2. **実装可能性**：全occupation表を列挙せず、controlとuncomputeを含む回路を定義できるか。
3. **情報価値**：mixed accessの費用を計上しても、step数削減が費用増を回収し得るか。

### 7.2 登録外syntheticで確認すべき科学的意味論

零結合、零detuning、近共鳴、負detuning、fermionic parity、無効occupation、number conservation、ancilla-relative phaseを検査する。small matrixのexact exponentiationは参考値としてのみ使い、その手順を大系primitiveの費用0oracleにしない。

上の誤差boundが、係数丸めと回転近似の区別を保って成立するかを独立に検証する。これらはCodexでまとめられる技術検証であり、一項目ごとにGPTへ戻さない。

### 7.3 分子pilotへ進める条件

実際のmolecular integralからA/B_jを生成し、励起数、差係数の複雑さ、使用qubit、controlを含む費用modelが閉じること。最初の分子・basis・geometry・精度は、本レビューでは固定しない。H2だけの特殊可解性を成功としないが、何を使うかは次のplanで明示する。

その後に、同じ原Hamiltonianのground-energy誤差を確認する。小系のFCIやexact unitaryは事後診断に使えるが、未知のground情報をgeneratorや候補選択へ無償入力しない。

### 7.4 GO/STOP

GOを検討できるのは、独立した生成手順または精度・費用上の具体的差があり、full-wrapper／全時刻列へ費用を接続できる場合。

STOP候補は、既知手順と実質的に同じ、枝／励起数が優位性を壊す、低いprimitive誤差のための算術・合成が高すぎる、同精度のnative PF/PRで利益が消える場合。

有限小系で負けたことを一般的不可能性にはしない。ただし、追加の構成機構がないままサイズ・精度を変えて成功を探し続けない。

---

## 8. 第二候補：基底固有位相を対象とした固定ユニタリPF生成

### 8.1 狙い

全operatorの誤差を消すより、基底energyに現れる誤差を優先的に抑え、短いPFを生成する。最終対象は同じ基底エネルギーで、スペクトル強度や別observableへ変更しない。

ただし、摂動energy errorによる分割評価[L5]、軌道と順序を変えるenergy-error cancellation[L6]、BCH対角成分を使う誤差推定[L7]が既にある。この分野は未開拓ではない。

### 8.2 構成案

実装可能な対称二次PFの辞書 `S_j` を、全て同じHamiltonian Hに対して作る。

\[
S_j(h)=\exp\{-ih[H+h^2 E_j+O(h^4)]\}.
\]

時間反転対称な固定macrocycleを

\[
S_\theta(h)=\prod_{\ell=1}^L S_{j_\ell}(a_\ell h),
\quad a_\ell\ge0,\quad\sum_\ell a_\ell=1
\]

とする。リスト `(j_l,a_l)` はpalindromeとする。BCHの主要項では、全kernelの一次生成子がHで同じであるため、

\[
E_\theta=\sum_\ell a_\ell^3 E_{j_\ell}
\]

となる。palindromeにより全macrocycleの時間反転対称性を保持する。これは有限次元・log branchが連続な小h領域での形式的主要項で、finite-h性能を保証する完全なboundではない。

基底が非縮退で摂動追跡できる領域なら、

\[
E_{0,\theta}(h)-E_0=h^2\langle\psi_0|E_\theta|\psi_0\rangle+O(h^4).
\]

係数生成では真のpsi0を使わず、明示した近似情報と不確かさを用いる。

### 8.3 なぜ固定macrocycleか

QPEのcontrolled powersを `S_theta(h)^(2^k)` から作れば、一つのunitaryの固有位相を追跡するという意味論を保ちやすい。時刻ごとに独立した近似unitaryを使い、全部が同じ固有値を表すと仮定することは避ける。

この選択自体は新規な原理ではない。ランダム回路の平均からenergyを求める方法が不可能という意味でもない。PRを使うなら、その方法の信号とnormalizationの保証を別に与える。[L4]

### 8.4 近似状態の誤差を目的関数へ入れる

`e_j=<psi0|E_j|psi0>`、近似値 `tilde e_j`、認証または明示した経験的幅 `eta_j` があれば、

\[
\left|e_\theta\right|\le
\left|\sum_\ell a_\ell^3\tilde e_{j_\ell}\right|
+\sum_\ell a_\ell^3\eta_{j_\ell}.
\]

純粋状態のfidelity下界Fが正しく得られている場合の粗い評価は

\[
|e_j-\tilde e_j|\le2\|E_j\|\sqrt{1-F}
\]

である。しかしFを無料で知っているとは仮定しない。残差やgapから評価する場合も、ground branchの同定・gap下界の取得・費用を明示する。証明された幅と、試行state間のばらつきを同じcertificateにしない。

近似期待値をゼロにしても真値がゼロとは限らない。固定のetaが残るなら誤差は一般に `O(h^2 eta)+O(h^4)` で、無条件の四次energy収束ではない。

### 8.5 自明な候補を除く

- `P^{-1}SP` という純粋processorはSと全く同じ固有値を持つ。state overlapや有限時間信号には影響し得るが、固有位相誤差の直接改善にはならない。
- 二項のStrang `A/2-B-A/2` と `B/2-A-B/2` は、同じstepでは積の巡回類似からisospectralである。反対符号のenergy-error辞書として当然使えると仮定しない。
- 全てのe_jが同符号なら、正の係数の和だけでは主要誤差をゼロにできない。単に多くの小stepへ分けて係数を小さくした利益は、追加回路数と比較する。

これらは代数的な制約であり、新しいnumerical no-goを実行したものではない。

### 8.6 既存法との差と順位

[L6]は異符号のenergy errorを持つbasis／orderingを組み合わせる平均化・相殺を既に扱っている。従って、上の案の新規性を「エネルギー誤差を打ち消す」ことに置けない。

残る候補は、真値なしの構成規則、安価な情報取得誤差の扱い、spectral-equivalenceを除いた辞書、finite-step branch保証、全資源最小化を接続する部分である。ただしこれらの全部を実装すれば自動的に新規とはならない。

**先行研究との重複と状態情報の取得が重いため、第二候補とする。** 第一候補が不成立でも自動採択しない。

---

## 9. 予備候補と、今回は優先しない方向

### 9.1 PR/DFの共同設計

実際のnoncommutativity、deterministic block費用、tail normalization、state-preparation込み統計費用に合わせて分解・samplingを設計する方向は残る。しかしPR自体にsplit選択・表現最適化の議論があり、SPRINT/GRADEも統合的な分解・乱択を扱う。[L3,L4]

「lambdaを下げる」「DF rankを小さくする」「q/r/Kを共同探索する」「cost-aware確率を使う」だけでは新しい研究主題にしない。正確に残差を保持する構成や、指数個の回路列を列挙しない新sampling手順等に、独立した差がある場合の予備候補とする。

### 9.2 高次PFの係数だけを増やす

過去に利用した高次PFを対照へ含めることは必要だが、追加stage・追加パラメータの探索自体を新規研究としない。kernelが変わらない純粋processorの固有位相不変性にも注意する。

### 9.3 N3の幅をさらに狭める

現在の主目的からは優先しない。N3のclosureと証拠は保持する。今回の第一候補はHamiltonianを切り捨てる研究ではなく、元Hamiltonianの時間発展block生成なので、N3の非負densityや一外部mode削減条件を流用しない。

### 9.4 スペクトル応答保存PF

候補の可能性は維持するが、利用者が示した主目的から離れるため今回は優先しない。ground-energyの固有位相とspectral weightの問題を混同しない。

### 9.5 エネルギー差の相殺

energy gapや相対energyでのTrotter相殺についても先行研究がある。[L9] 化学的には重要だが、今回はabsolute ground-energy taskから勝手に切り替えない。将来、利用者が出力taskとして採択した場合に比較する。

---

## 10. 比較計画：最終taskから逆算する

### 10.1 第一段階は方式の違いを明確化

第一候補の最初の対照は、既知THRIFTのmixed実装、同じ分子係数を使うnative PF、既知の条件付き可解／excitation回路である。二準位式の再現だけでGOにしない。

第二候補の最初の対照は、既存のenergy-error-based ordering/basis selection、固定S2・高次PF、単純なstep細分化である。既知の二basis平均化やMPFとの同値を除く。

### 10.2 同じ入力・情報・品質

原Hamiltonian、basis、core/frozen orbitals、DF打切り、粒子sector、参照状態の生成法、使用できる古典情報を一致させる。新方式だけに真のground stateや既知の正解誤差を与えない。

同じ精度は一時刻のunitary normだけでなく、選んだenergy estimatorの要求へ結び付ける。手法によって必要な付帯処理が違うなら、それを消さずに会計する。

### 10.3 有限精度回路

logical Tを議論するなら、同じ回転精度・phase convention・補助qubitとnative accessを使う。arbitrary-angle RZ数をT数と同一視しない。単なる「解析的に指数が書ける」ことを回路費用1にしない。

### 10.4 統計・branch

共通初期状態のground overlap、近接固有値、位相unwrap、最長時刻での近似biasを扱う。全methodの同一overlapを確認できないなら、条件付き評価として残す。

PRのfirst momentが小さいscalarで減衰すれば、補正weight・必要shotが増える。平均channelが近いという保証だけではcontrolled signalを置き換えない。QPEが必要とするunitaryと、unbiased signal estimatorを明確に区別する。

### 10.5 資源比較の段階

(1) 解析的な構成・演算量、(2) small exact referenceでの意味論、(3) molecular primitive費用、(4) energy-task全体、という依存順序は必要である。しかし関連する準備・テストを一回のCodex作業へまとめ、各小工程ごとにGPT詳細レビューを追加しない。

数値pilotはsource・入力・比較条件・予算を固定し、別の明示実行承認を得てから行う。

---

## 11. 次のCodex作業として推奨するまとまった範囲

**推奨：第一候補の方法差・構成・費用の準備を一つにまとめる。**

- 本レビューの相対block、detuning構造、有限精度boundを独立に検討する。
- THRIFT、対称性による可解Hamiltonian、coherent arithmetic等の先行手順との対応を確認する。
- 占有対encoding、parity、coherent detuning算術、回転、uncompute、controlを含むconstructor案を明示する。
- 次数や小行列の可解性だけでなく、全励起数と前処理・補助qubit・精度を含むresource modelを作る。
- 登録外syntheticで意味論・零結合相殺・係数近似bound・位相・adjoint等を確認する準備を行う。
- 実分子への最初の限定検証について、入力・baseline・output・判断条件・call/memory/output予算を結果前に固定する案をまとめる。
- 必要なsourceと報告を別branchへ公開し、固定資料を取得可能にする。旧Track source/結果/STOPは変更しない。

技術方式の選択、コード構成、キャッシュ、通常のfocused testはCodexの裁量とする。ただし、中心仮説・主比較・保証対象の変更は科学判断へ戻す。

**このレビュー承認は、上記の科学実験の起動承認ではない。** 新たな分子計算、登録入力での誤差・費用評価、solver、compile、samplingは別認可である。今回の回答ではその実行を行っていない。

---

## 12. 次にGPTへ戻る条件

- 第一候補が既知手順と実質的に同じと判明した。
- 分子係数で必要なcoherent accessや有限精度保証に重大な問題がある。
- 実装案のresource modelから、研究の目的を変更する必要が出た。
- 認可済み限定pilotで新しい有意義な科学的結果が得られた。
- 第二候補へ移る具体的な根拠、または別の新構成が明確になった。

source・test・commit・公開が完了しただけで同じ科学レビューを繰り返す必要はない。変更のない比較設計を実行する場合のpreflightとユーザー実行承認は、詳細科学レビューとは別に扱う。

---

## 13. 論文化の着地点と打切り条件

### 第一候補が成立する場合

論文の候補主題は「分子の対角励起エネルギー構造を利用した、相対propagatorのcoherent低コスト生成と、基底エネルギー推定への接続」である。

必要な寄与は、単にTHRIFTの使用や二準位対角化の式ではない。既存構成にない効率的な生成手順、有限精度の費用削減、classの拡張、あるいは構造的な資源保証のどこに差があるかを示す。理論貢献を目指す場合、全例で実分子の性能勝利を必須とはしない。ただし応用優位をclaimするなら最終energy-taskでの証拠が必要である。

### 第二候補が成立する場合

候補主題は「真のground情報を使わない、情報誤差と有限stepを考慮した固有位相向けPF構成」である。既存の摂動評価・basis averagingから独立した構成・保証・取得計算量の差が必要。

### 終了する場合

- 既知法の直接的適用に尽きる。
- 主要な構成上の利益が付帯費用で消える。
- 真のground情報または指数的tableを無料にしないと成立しない。
- finite-windowでだけ見える効果を、energy-taskの全時刻へ移せない。
- 成功例を作るために入力・精度・baselineを結果後に繰り返し変更する必要がある。

この場合は成果と限定的な否定結果を残し、現候補を区切る。研究分野全体の不可能性とは呼ばない。

---

## 14. 確認範囲と限界

### 実際に行ったこと

- 最新指定の共通指示Markdownを読んだ。
- 直前の広域レビュー記録の対象・結論を確認した。
- 下記固定GitHub資料を実際に取得した。
- THRIFT出版社全文、関連する可解Hamiltonian・SPRINTのPDF、energy-error関連の公開本文を確認した。
- 相対二準位blockの式、detuning誤差bound、固定PF macrocycleの主要項、isospectralな自明変換の制約を解析的に整理した。
- 研究候補の優先順位、反論、最初の判別、Codex作業範囲を判断した。

### 行っていないこと

- 新しい分子integral、行列の固有値、最適化、回路合成、数値例、費用benchmark。
- 既存科学結果の全再監査。
- 全branchの最新進捗の網羅確認。
- 先行性の網羅的確認、提案の未発表性・性能・論文化の確定。
- PR v2全文の独立再取得。現行書誌はv2だが、一般PDF経路で取得された表示はv1であり、v2全文として扱わない。

### 文献取得に関する扱い

一部arXiv HTMLはcache miss/404だったが、THRIFTは出版社全文、PatelらとSPRINTはPDFへ切り替えて確認した。Patelらの最初のweb screenshotは取得失敗だったため、画像・図表を根拠にした定量claimはしていない。SPRINTの表の性能数値は本レビューの比較計算に用いていない。

文献L5の古いarXiv版はfirst-order等の記述があり、出版版と内容・著者が異なる。今回は出版版のsecond-order・近似state・結論を根拠とし、古い版を黙って同一視しない。

詳細な証拠の欠落が新規性や実装判断を左右する場合は、次の準備で明示する。情報欠落を「研究余地がある」証拠にはしない。

---

## 15. 固定repository資料

### [R1] THRIFT toy／mixed-accessの既存結果
- Repository: `HIROMU1015/Partially-Randomized-Trotter`
- Commit: `c24dcede10ed9726e1e0b1030eca749ba7bc5cba`
- Path: `docs/research/representation_construction_comparison_results_20261010.md`
- 確認：第4節、今回fetchの元行144–192。
- URL: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c24dcede10ed9726e1e0b1030eca749ba7bc5cba/docs/research/representation_construction_comparison_results_20261010.md

### [R2] Track A H6の固定確認報告
- Commit: `0dd5c3d204dab9b731268a1aa2add621080f3d20`
- Path: `docs/research/track_a_h6_independent_confirmation_result_v1.md`
- 確認：元行1–66。有限信号task、P=0、希少orderと母平均費用の限定。
- URL: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0dd5c3d204dab9b731268a1aa2add621080f3d20/docs/research/track_a_h6_independent_confirmation_result_v1.md

### [R3] N3 closureの研究判断
- Commit: `ccc575ee339be091fd4031272c38bee13866f097`
- Path: `artifacts/n3_achievements_closure/2026-10-13/research_decision.json`
- URL: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ccc575ee339be091fd4031272c38bee13866f097/artifacts/n3_achievements_closure/2026-10-13/research_decision.json

### [R4] Track B Gaussian積生成候補closure
- Commit: `64bb39fdcc4928d0f026b63f874c8bf4be2e69ab`
- Path: `docs/tracks/algorithm_codesign/gaussian_product_generation_closure_20261013.md`
- URL: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/64bb39fdcc4928d0f026b63f874c8bf4be2e69ab/docs/tracks/algorithm_codesign/gaussian_product_generation_closure_20261013.md

### [R5] 直前の広域レビュー
- Conversation artifact: `pr_algorithm_research_gap_and_new_direction_review_2026-10-13.md`
- 作業containerで本文を確認。今回の目的変更前の順位を引き継ぐ履歴。
- 新しい科学結果として扱わない。

## 16. 一次文献と確認箇所

### [L1] THRIFT
J. L. Bosse et al., *Efficient and practical Hamiltonian simulation from time-dependent product formulas*, Nature Communications 16, 2673 (2025).
- DOI: https://doi.org/10.1038/s41467-025-57580-5
- 出版社全文: https://www.nature.com/articles/s41467-025-57580-5
- arXiv: https://arxiv.org/abs/2403.08729
- 確認：定理1、Eq. (6)–(13)、mixed実装のalpha依存誤差条件、Fermi–Hubbardの実装費用上の限界。

### [L2] 対称性による可解Hamiltonianの拡張
S. Patel, T.-C. Yen, A. F. Izmaylov, *Extension of exactly-solvable Hamiltonians using symmetries of Lie algebras*.
- arXiv v2: https://arxiv.org/abs/2305.18251v2
- PDF取得経路: https://arxiv.org/pdf/2305.18251
- 出版版DOI: https://doi.org/10.1021/acs.jpca.4c00993
- 確認：symmetry-sector依存係数、fermionic mean-field solvability、projector条件付きunitary、測定taskとcoherent回路の区別。

### [L3] SPRINT/GRADE
P. A. M. Casares et al., *Theory and practice of Trotter product formulas for quantum chemistry*, arXiv:2606.30741v1 (2026).
- https://arxiv.org/abs/2606.30741
- https://arxiv.org/pdf/2606.30741
- 確認：PDF先頭頁、III B–Eの関連箇所、processingの役割。表の資源倍率を本taskに外挿しない。

### [L4] PR
J. Günther et al., *Phase estimation with partially randomized time evolution*.
- https://arxiv.org/abs/2503.05647
- 現行書誌：v2、2026-07-10改訂、PRX Quantum 7, 020332 (2026)。
- 取得PDF https://arxiv.org/pdf/2503.05647 はページ画像上v1表示だった。v1の分割・表現・位相推定の議論を参照し、v2全文の監査完了とはしない。

### [L5] energy-error推定と分割評価
S. G. Mehendale, L. A. Martínez-Martínez, P. D. Kamath, A. F. Izmaylov, *Estimating Trotter approximation errors to optimize Hamiltonian partitioning for lower eigenvalue errors*, Digital Discovery 4, 3540–3551 (2025).
- https://doi.org/10.1039/D5DD00185D
- https://pubs.rsc.org/en/content/articlelanding/2025/dd/d5dd00185d
- 関連arXiv: https://arxiv.org/abs/2312.13282
- 確認：出版版の摂動推定、approximate state、資源会計、効率的な最適化とupper boundに関する結論。古いarXiv版と混同しない。

### [L6] 軌道・順序変更と基底energy-error cancellation
M. Kronenberger, M. Erakovic, M. Reiher, *Trotter Error and Orbital Transformations in Quantum Phase Estimation*, arXiv:2602.18913v1 (2026).
- https://arxiv.org/html/2602.18913v1
- https://arxiv.org/abs/2602.18913
- 確認：第2.5、4.3、結論。異符号errorの組合せと、真値なしで符号を選ぶ困難。

### [L7] BCH対角成分とpractical error estimation
W. Maxwell et al., *Practical Estimation of Trotter Error for Hamiltonian Simulation*, arXiv:2606.30738v1 (2026).
- https://arxiv.org/html/2606.30738v1
- https://arxiv.org/abs/2606.30738
- 確認：公開HTML、asymptotic statementとcompact BCHの位置付け。finite-h certificateへの無断変換をしない。

### [L8] Spectral Analysis
C. Yi, E. Crosson, *Spectral Analysis of Product Formulas for Quantum Simulation*, arXiv:2102.12655 (2021).
- https://arxiv.org/abs/2102.12655
- 確認：abstract。全定理の独立検証として扱わない。

### [L9] energy差の相殺（今回の主taskとは区別）
A. J. Bay-Smidt et al., *Quantum simulation of nanographenes and Trotter error cancellation*, arXiv:2605.00745 (2026).
- https://arxiv.org/abs/2605.00745
- 確認：abstract。energy-gap問題の成果をabsolute ground-energyの改善へ移さない。

---

## 最終判断

**第一候補の構成可能性・方法差・資源見積りをCodexでまとめて準備することを推奨する。**

研究の中心は「THRIFT＋PRを組み合わせること」ではなく、分子Hamiltonianの構造を用いて時間発展primitiveとその有限精度実装を改善し、基底エネルギー計算の全費用へつなぐことである。

本レビューは候補選定と構成案の判断であり、新しい科学計算の認可、既存STOP解除、主研究としての最終採択ではない。
