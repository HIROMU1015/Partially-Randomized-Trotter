# ④の研究課題再設計：有限stepの相対形式保証から絶対基底エネルギーの出力区間・step選択へ

- 作成日：2026-10-17 JST
- 文書区分：ユーザーが「レビューを開始して」と明示承認した重要科学レビュー（Level 1）
- 対象：④ PRに限定しないアルゴリズム改善の研究課題再設計
- 対象外：①PRリソース評価、H8 Phase Cの評価・追加作業。②PR内部改善・③M1-Sは重複確認と既存知見の参照だけとし、再開しない。
- 実施：固定GitHub文書・過去レビューの必要箇所・一次文献の読解、独立した紙上導出、候補比較、研究方針の提案。
- 非実施：Hamiltonian・固有値・RDMの新規計算、保存科学数値の再解析、数値/symbolic実験、simulation、回路build/compile、SOURCE実装、GitHub書換え、authorization/grant/marker発行。
- 状態：レビュー完了。以下は**提案および独立検証前の紙上導出**であり、新規性・実用優位の認定、研究テーマの正式採択、科学計算の実行承認ではない。
- 運用正本：`research_project_instructions_unified_2026-10-17.md` をローカル原本から確認。旧版を正本として併用しない。

---

## 0. 結論

**条件付き第一候補として、未採択のSOS-PF A2（保証理論）を「有限stepの相対形式保証 → gap不要のエネルギー出力区間 → 実行可能なstep選択」という一つの具体的な構成問題へ絞ることを提案する。**

これは新しい研究系列を発見したという主張ではない。既存A2の未解決部分を明示して扱うものであり、CLOSEDのA1（新しいPSD/PF分割生成）や、終了したfixed-N DF係数shift案を再開しない。未採択A2への研究資源配分の変更になるため、実際の着手にはユーザーの採否判断が必要である。

今回のレビューでは、候補名だけでなく次の数学的な基準を与えた。

1. 正半定値fragmentの対称二次PFについて、小さいが明示された有限step領域で、実効Hamiltonianの誤差を**元の正半定値Hamiltonianに対する相対形式**として抑える保守的な上界を紙上導出した（§5）。
2. 既存K0の `K_SOS` をそのまま再利用し、先導係数と高次剰余を共にエネルギーに比例する形式で会計する（§6）。K0の辞書・係数を新発見として再提出しない。
3. 単一PF固有枝への高overlapを先に証明する代わりに、QPEが測るPFスペクトルの**許容エネルギー区間内の確率質量**を保証する（§7）。励起gapも真のE0の入力も不要な条件付き経路になる。
4. 実際の分割に対応する認証済み `[L,U]` を入力にし、実行するstep幅を決める非実行アルゴリズム仕様を示した（§8）。

ただし、ここで与える定数は粗く、許されるstepは `O(1/Λ)` の範囲に限られる。低エネルギーPF保証・相対形式評価・スペクトル測定の集中を結び付けた**全statementの独立新規性は未認定**であり、既存の強い保証を超える実用的なstep削減も未確立である。

したがって推奨は「このまま実装・分子benchmarkへ進む」ではない。**今回の明示命題の検算と、適用域を揃えた既知定理との包含・差分確認を、一つの数学的判断対象として扱う**ことである。単なる既知系と判明し、より鋭い/安価な具体的保証の差も示せなければ、独立アルゴリズム候補へ格上げしない。

---

## 1. 固定された研究状態・ユーザー意図

### 1.1 最新の正式状態

[正式採択記録 S1](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/67576526c4e96ae7235fa645fdf744278d049f8d/docs/research/qc_exact_encoding_direction_adoption_20261017/decision.json)を直接取得した。

- 対象のfixed-N spin-free DF平方shift案は `REDUCES_TO_KNOWN_METHOD`。
- その一案について追加探索・実装・数値検証を行わず、④を `RESEARCH_QUESTION_SELECTION` に戻した。
- encoding一般の不可能性、④全体終了、新課題採択を意味しない。
- ①②③および高次PF第1・第2研究の状態は変更しない。
- N1 STOP、N2 HOLD、N3現系列CLOSED、THRIFT現templateCLOSED、H1–H4現探索PAUSED、一体再配分現構成CLOSED、H-E1現定義CLOSED、M-a/M-b STOP、PF小改変PAUSED、旧D HOLDを維持。
- SOS-PF現分割生成A1はCLOSED、**保証理論A2はNOT_ADOPTEDでありCLOSEDではない**。K1-Fは未認可・未実施。

S1内のレビュー未承認という記録はその事務作業時点の状態である。今回の明示的開始承認は、このレビューだけに適用する。既存の計算・SOURCE・契約・STOPの権限は変えない。

### 1.2 維持する科学的範囲

既存資料S2のD1/D2を維持する。

- D1：block-encoding、qubitization、それらを中核とするQSVT/量子walkを新研究対象にしない。
- D2：元の有限基底電子Hamiltonianの絶対基底エネルギー。scalar、核反発、対象sector、phase originを追跡する。
- exact ground copiesを条件付き入力として使う場合も、古典E0・固有vector・RDM・gap、準備unitary/逆を無料で追加しない。
- 論文化ではなく、後からCodexが理解できる研究結果を残す。

以下で正半定値分割やexact copiesを置くのは、**候補命題の適用域を明示するため**である。全化学Hamiltonianや全入力状態をその条件に変更したわけではない。

---

## 2. 既存成果から何を引き継ぎ、何を繰り返さないか

### 2.1 既知還元が意味する範囲

[encoding引継ぎS3](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/830be75335bb9b599488cb365213d555b0dc18bb/docs/research/qc_exact_encoding_controlled_construction_20261017/decision_handoff.md)によれば、検討した

`D_s = D - s(N_hat-N0 I)`

はPR原論文の式(C14)のパラメータ選択と通常の対角回路生成で得られる。これを再度新しい符号化や生成規則として提案しない。今回、その原論文全体の独立再審査は行っていない。

この一案の既知性から、全encodingや全回路生成の一般的な不可能性は導けない。一方、具体案がない状態の「まだ何かあるはず」を次の実装指示にしない。

### 2.2 過去の優先付けへの修正

前回は、encodingからcontrolled回路への接続記録が不足していることと、実装費だけを変えられる設計上の利点を第一候補の根拠とした。しかし、接続記録の不足は独立した算法差の存在を示さない。その後の構成が既知shiftへ帰着したことを踏まえ、今回は**既に明示できる数学的な差分と、それが実行手順を変える条件**から研究課題を組み立てる。

「既知部品を使うから算法改良でない」とはしない。ただし既知操作の同じ合成、名称変更、catalogへの追加だけを新しい方法としない。

### 2.3 A1とA2の区別

A1は「有利で独立したPF-native分割を生成する」探索である。A2は「与えられた分割に対して、安価に検証できる誤差保証から実行可能なstepを選ぶ」方向である。

同じPF kernelでも、合法な保証が鋭くなり実行するstep数が減れば、運用アルゴリズムの費用は変わり得る。この点は[既存SOS-PF GPT再評価S4](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dbe51f1870cd069b952cc92593833d355da10e28/docs/reviews/quantum_chemistry_pf/qc_sos_pf_post_k05_gpt_scientific_review_20261016.md)が既に指摘している。今回の独自発見としては扱わない。

本提案はA2の限定的な再設計であって、A2の名称を隠して新課題に数えるものではない。

---

## 3. 候補比較と一つに絞る理由

| 方向 | 既存検討との関係 | 今回の判断 |
|---|---|---|
| 新しいencoding・対称性回路生成 | 直前の一案は既知shift。広い接続条件はpreflight済み | 現時点で次の独立した具体規則を提示できていない。追加adapter/benchmarkを既定の次作業にしない |
| sampling・parent配分・平均信号の再設計 | ②C4/IS/return/Parent allocationと③M1-Sの既存範囲 | 保存成果を保持。名称やパラメータだけを変えた再探索はしない |
| copies・測定・制御削減 | ④H2/H4、既知測定法・reference accessとの重複がある | 新しい情報取得手順が未提示。simulationから測定研究へ自動転換しない |
| **有限step保証からstep選択** | **既存A2の未採択部分。K0の先導係数と過去のgap-free粗い比較点がある** | **今回、具体的な相対形式上界と出力確率の接続を提示できたため、条件付き第一候補** |

この表は数値倍率による順位表ではない。構成を現時点で記述できること、旧研究との継承が明確なこと、最終taskへ接続する判定条件を持つことを判断理由にした。

代替候補を二つ三つ埋めるために、未提示の算法を捏造しない。また、これが唯一可能な研究方向とも主張しない。

---

## 4. 提案する一つの研究問い

> **元の電子Hamiltonianと既存の実装可能な正半定値fragment分割から、励起gap・真のE0・古典ground vectorを使わず、固定有限step PFのQPE出力が許容エネルギー区間へ入る確率を認証し、その認証から実行stepを選択できるか。さらに、最も近い既知保証に対して、検証費用を含めた有用な差を得られるか。**

変えるのは、必ずしもPFの係数・順序・fragmentではない。変える対象は「何を保証するか」と「その保証から実行scheduleを決める手順」である。

従来の孤立branchのbias/overlap保証を必須の中間目標にせず、誤差ε以内のエネルギーを返すという最終出力条件から考える。

### 4.1 入力として必要なもの

有限次元の共通不変sector上で、

\[
H=L I+G,\qquad G=\sum_{j=1}^{M}F_j,\qquad 0\preceq F_j\preceq f_j I,
\qquad\Lambda=\sum_j f_j.
\tag{1}
\]

`L`は実際に使う分割に対応する既知scalarである。`H`の真のground energyを`E0`、`G`のground energyを

\[
e=E_0-L\ge0
\tag{2}
\]

とする。`e`は証明上の量であり、実行アルゴリズムには真値を入力しない。

別途認証された上界`E0≤U`から

\[
W=U-L\ge e
\tag{3}
\]

を用いる。良好なU/Lが必ず安く得られるとは仮定しない。証拠生成・検証の費用と失敗確率を記録する。DF近似モデルを使う場合は、元integral Hへの誤差台帳が別に必要である。

正係数DF平方と一体項のsector下限によるPSD化、signed項の既知上下界によるshiftは既存K0の構成であり、新しく分割を探索する必要はない。ただしPSD化でW/Λが大きくなれば、期待する低エネルギー利得は得られない。

---

## 5. 今回の紙上導出RF0：有限stepの相対形式上界

**以下は今回の独立した紙上検討で得た保守的な命題と証明である。数値・形式検証・他研究者の独立検算は行っていない。既存文献で同一命題が未発見であることを新規性の証明にはしない。**

### 5.1 定義

正時間係数の対称二次PFを

\[
S_2(h)=e^{-ihF_1/2}\cdots e^{-ihF_{M-1}/2}e^{-ihF_M}
 e^{-ihF_{M-1}/2}\cdots e^{-ihF_1/2}
\tag{4}
\]

とし、原点近傍の解析的なlogを用いて

\[
K(h)=\frac{i}{h}\log S_2(h),\qquad K(0)=G
\tag{5}
\]

と置く。実hでKはHermitianで、palindromeからhの偶関数である。

Λ=0なら全Fj=0でH=LI、問題は自明なので除く。ここではΛ>0。

### 5.2 命題RF0

\[
|h|\Lambda<\frac14
\]

において、

\[
-\eta_0(h)G\preceq K(h)-G\preceq\eta_0(h)G,
\qquad
\boxed{\eta_0(h)=\frac{32(h\Lambda)^2}{1-16(h\Lambda)^2}}.
\tag{6}
\]

特に`|h|Λ<1/sqrt(48)`ならη0<1になる。

`1/4`や`32`はこの証明の保守的な解析上の選択であり、実験の閾値・資源制限・最適な収束半径ではない。高次PFの負時間係数を含む場合へ、この命題を無変更で適用しない。

### 5.3 証明：正のstageに対する境界評価

S2をstage列に書く。

\[
S_2(z)=E_J(z)\cdots E_1(z),\quad E_\ell(z)=e^{-izA_\ell},
\quad A_\ell\succeq0,\quad\sum_\ell A_\ell=G.
\]

各stageのnorm上界をaℓとしてΣaℓ=Λとできる。任意の正規化vについてEv=〈v,Gv〉と書く。

正半定値性から

\[
\|A_\ell v\|^2\le a_\ell\langle v,A_\ell v\rangle,
\qquad
\sum_\ell\|A_\ell v\|\le\sqrt{\Lambda E_v}.
\tag{7}
\]

半径r=1/(4Λ)の複素円上でx=rΛ=1/4、X=S2(z)−I、q=exp(x)−1と置く。

\[
\|X\|\le q\le1/3.
\tag{8}
\]

また

\[
\|(E_\ell-I)v\|\le r e^{r a_\ell}\|A_\ell v\|,
\quad
|\langle v,(E_\ell-I)v\rangle|
\le r e^{r a_\ell}\langle v,A_\ell v\rangle.
\tag{9}
\]

前者と積のtelescopingから、

\[
\|Xv\|,\ \|X^\dagger v\|
\le r e^{x}\sqrt{\Lambda E_v}.
\tag{10}
\]

次に、scalar期待値には次の**両端に差を置く展開**を使う。

\[
X=\sum_j(E_j-I)+\sum_{i>j}(E_i-I)E_{i-1}\cdots E_{j+1}(E_j-I).
\tag{11}
\]

空積はIとする。これは一方の向きのtelescopingで得たprefixを、反対向きに再展開した恒等式である。prefixがvを回転させることを無視して個々の `||Aj v||` に置き換えるのではない。

(9)と(11)から、

\[
|\langle X\rangle_v|
\le r e^xE_v+r^2e^x\sum_{i>j}\|A_i v\|\|A_jv\|
\le r e^x(1+x/2)E_v.
\tag{12}
\]

k≥2なら、

\[
|\langle v,X^kv\rangle|
\le\|X^\dagger v\|\|X\|^{k-2}\|Xv\|
\le r^2e^{2x}\Lambda E_v q^{k-2}.
\tag{13}
\]

### 5.4 logの期待値とCauchy評価

q<1なのでnorm収束するlog級数を使える。円上で、

\[
|\langle v,K(z)v\rangle|
\le \left[e^x(1+x/2)+x e^{2x}
\sum_{k=2}^{\infty}\frac{q^{k-2}}k\right]E_v.
\tag{14}
\]

`e^(1/4)≤4/3`、`q≤1/3`および

\[
\sum_{k=2}^{\infty}\frac{q^{k-2}}k\le\frac1{2(1-q)}\le\frac34
\]

により、角括弧は`3/2+1/3=11/6<2`以下である。

S2(−z)=S2(z)^{-1}と原点近傍のlog枝から、K(z)は解析的偶関数である。したがって

\[
f_v(z)=\langle v,K(z)v\rangle=E_v+\sum_{k\ge1}c_{2k}(v)z^{2k},
\qquad |c_{2k}(v)|\le\frac{2E_v}{r^{2k}}.
\tag{15}
\]

実hについて級数を足すと、

\[
|\langle v,(K(h)-G)v\rangle|
\le 2E_v\frac{(h/r)^2}{1-(h/r)^2}=\eta_0(h)E_v.
\]

全vに一様で、K(h)−GがHermitianなのでLoewner形式(6)が従う。

Ev=0なら各Aℓv=0、S2(z)v=vでありK(z)v=0。Gの逆やground gapを用いる必要はない。∎

### 5.5 この命題が解決していないこと

- 局所性を使わない代わりにΛ依存が強い。
- 適用域は小h。`hΛ<1/4`が最大の許容stepという意味ではない。
- 全fragmentが可換な場合には誤差0だが、(6)だけではその構造を反映しない。
- 高次/負時間係数、近似fragment、一般初期状態、実機誤差には別の解析が必要。
- 相対形式と解析性を使う道具は既知であり、この粗い命題の独立priorityを認定していない。

---

## 6. 既存K0を再利用するRF1：先導係数と剰余の両方を相対形式にする

### 6.1 既存K0から使う結果

[技術監査S5 §4、式(7)–(9)](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/58d8672702284d73321cc67a26c4640adaddde5f/docs/research/qc_sos_pf_k0_20261015/technical_audit.md)には、対称二次PFの

\[
K(h)=G+h^2D+O(h^4)
\]

に対して、係数辞書とfragment normだけから計算可能な `K_SOS` がある。元の記述はgroundの先導bias `|d|≤K_SOS e` という形である。

ここで重要なのは、その§4の分散評価自体は固有状態方程式を使っていない点である。任意の正規化ψについて

\[
\phi_j=\langle F_j\rangle_\psi\ge0,\quad
\sigma_j^2\le f_j\phi_j,\quad
\sum_j\phi_j=\langle G\rangle_\psi
\]

を使えば、同じ非負行列の行和評価から

\[
|\langle\psi,D\psi\rangle|\le K_{\rm SOS}\langle\psi,G\psi\rangle,
\quad
\boxed{-K_{\rm SOS}G\preceq D\preceq K_{\rm SOS}G}.
\tag{16}
\]

**K_SOSの辞書・計算法は既存成果。全vectorに対する形式への読み替えは今回の紙上観察**であり、旧資料にそのまま記載されていたと主張しない。

二分割ground特有の非負係数やH交換子の期待値消去まで、任意状態へ拡張したわけではない。ここで再利用するのは多fragmentの絶対値・分散上界である。

### 6.2 有限剰余の相対形式

(15)のうち四次以降を足せば、

\[
-\rho_4(h)G\preceq K(h)-G-h^2D\preceq\rho_4(h)G,
\qquad
\rho_4(h)=\frac{512(h\Lambda)^4}{1-16(h\Lambda)^2}.
\tag{17}
\]

従って、

\[
\eta_1(h)=h^2 K_{\rm SOS}+\rho_4(h),\qquad
\eta(h)=\min\{\eta_0(h),\eta_1(h)\}
\tag{18}
\]

を(6)のηとして使える。

旧K0の有限剰余はscalar normの `r4(h)I` であり、SOS slackに比例しないことが限界として明示されていた。今回の剰余は `rho4(h)G` という**相対形式**なので、低エネルギー性が剰余項から消えない。ただし、より大きいhや実用的定数を得たわけではない。

### 6.3 旧研究との差分を正しく数える

| 要素 | 既存の結果 | 今回の接続 |
|---|---|---|
| PSD化・fragment生成 | K0で既知DFのPSD化、shift、S2定義を整理 | そのまま入力として使用。新しい分割生成はしない |
| 先導係数 | `|d|≤K_SOS e`、係数辞書・行和は計算可能 | 同じ評価を全vectorの相対形式(16)へ読む |
| 有限剰余 | scalar `r4 I`。gap＋Schurでbranch bias/overlapへ接続 | `rho4 G`を導出し、出力区間確率へ接続 |
| 推定対象 | 個別のground-connected PF固有値・overlap | 許容energy帯へ入るPFスペクトル確率 |
| 入力 | gap下界を要する経路と、粗いgap-free vector経路 | gap不要。実行時はmatching `[L,U]`と既存証拠 |
| 未確立 | 有用性・独立新規性・総資源 | **依然未確立**。本レビューだけで採択しない |

---

## 7. Gap不要のエネルギー出力区間保証

### 7.1 スペクトル測定に対する補題EW

Gのground energyがe≥0で、

\[
-\eta G\preceq K-G\preceq\eta G,\qquad0\le\eta<1
\tag{19}
\]

とする。Gのground状態上でKを理想的にスペクトル測定した値をYとする。ground縮退は許す。ground部分空間にsupportを持つ混合状態でもよい。

\[
K\succeq(1-\eta)G\succeq(1-\eta)eI,
\qquad
\mathbb E[Y]=\langle K\rangle_g\le(1+\eta)e.
\tag{20}
\]

よって非負の確率変数 `Z=Y-(1-η)e` の平均は高々2ηeであり、d>0に対しMarkovから

\[
\boxed{\Pr(Y>e+d)\le\frac{2\eta e}{d+\eta e}}.
\tag{21}
\]

さらにd≥ηeなら`Y<e-d`は起きないので、同じ右辺が二側区間から外れる確率の上界になる。

これは初等的なLoewner順序とMarkov不等式の接続であり、この補題だけの独立新規性を主張しない。重要なのは、実際のPFに対して(19)を、実行可能な入力から保証できるかである。

### 7.2 真のE0を使わない十分条件

e≤W=U−L、`0<alpha_spec<1` とする。

\[
\boxed{\eta(h)W\le\frac{\alpha_{\rm spec}}{2-\alpha_{\rm spec}}d}
\tag{22}
\]

なら、ηe<dおよび式(21)右辺≤alpha_specを保証する。従って

\[
\Pr(|Y-e|>d)\le\alpha_{\rm spec}.
\tag{23}
\]

真のE0は証明にしか現れず、step選択の入力はWである。励起gap・個別PF branchの追跡・PF固有vectorの古典的な取得を必要としない。

ただしこれは**Gのground状態が入力である条件**を除去しない。未知の励起混合や、元Hと違うモデルのgroundを無条件に同じ入力としない。

### 7.3 単一branchを必須にしない理由

元groundが複数のK固有stateへ広がっていても、それらの測定energyがすべて許容帯内にあれば、絶対energyの精度というtaskは満たされる。個別固有vectorへのoverlapを1に近付けることは、常に必要な中間目標ではない。

これは、既存のbranch-based手法が誤りという意味ではない。保証対象を最終出力に合わせる別経路である。

---

## 8. 非実行仕様：証拠からstepを選んで絶対energyを返す

以下は今回の数学的候補を具体化する仕様であり、実行契約・SOURCE・grantではない。

### 8.1 入力

- (1)の実装可能な分割と、各fragmentの検証可能な上界fj。
- 同じ分割のL、認証済みE0上界U、およびそれらの取得・検証費用。
- 対象sector内のexact ground copies、またはその条件に対する別のstate品質証拠。
- 全体の許容誤差・失敗確率を、spectral窓、phase読出し、係数/モデル/実装/状態誤差に分ける設計。**数値配分は今回採択しない。**

### 8.2 手順

1. W=U−LとΛを証拠から取得する。W≤2εなら `[L,U]` の中点だけで要求絶対誤差εを満たし、量子計算の必要性を別に判断する。Λ=0、W=0の自明caseを分離する。
2. 窓幅d>0とalpha_specを与え、(22)を満たすηtarget<1を選ぶ。
3. RF0だけを使うなら、次の解析的十分stepを使える。
   \[
   h\le\frac1\Lambda\sqrt{\frac{\eta_{\rm target}}{32+16\eta_{\rm target}}}.
   \tag{24}
   \]
   RF1がより小さい上界を与える場合は、検証済みの単調上界を使って(22)を満たすstepを選ぶ。真のbias、真のE0、経験fitのwinnerを入力にしない。
4. **固定した同一のS2(h)のpowers**を使うspectral-output QPEを行う。roundごとに別のhや別kernelへ置換しない。
5. 位相をKのenergy推定値Yhatへ変換し、`Ehat=L+Yhat`を返す。既知scalarをcontrol側で保持して測る等価実装も可能だが、scalarを二重計上しない。

### 8.3 QPEの読出しとalias

K≤(1+η)G≤(1+η)ΛI、K≥0であり、上の小step領域ならphaseの既知supportは2πより十分短い。既知の共通phase shiftまたは円周上のinterval decodingを使い、0/2π境界を跨ぐ推定を誤ったenergyに割り当てない。logの存在だけをalias解決と同一視しない。

理想QPEの測定統計はKの固有basisの混合として記述できる。任意の固有valueについて、QPE読出しがεread以内で失敗確率alpha_read以下なら、

\[
\Pr(|\widehat E-E_0|>d+\varepsilon_{\rm read})
\le\alpha_{\rm spec}+\alpha_{\rm read}.
\tag{25}
\]

これは単一固有位相の平均信号を仮定するRPEの成功保証ではない。③M1-SのTask Aやfresh-IID乱択signalへそのまま移すこともできない。

### 8.4 実装誤差・近似state

(25)は理想の固定unitaryに対する結論である。近似Givens、回転合成、fragment係数、DFモデル差、state preparationの誤差は別予算に戻す。

たとえばexact ground部分空間のstateからのtrace distanceが別途τ以下と保証されれば、任意測定の確率差へτを加える一般原理は使える。ただし、低い変分energyだけからgapなしでそのτが得られるとは仮定しない。

符号化やsector restrictionを使うときは各logical fragmentの不変spaceを確認する。全Hの保存量だけで全fragmentの閉包を仮定せず、各native gateが常にsector内に留まることまで過剰に要求することもしない。

---

## 9. 資源にどう接続するか：十分会計と比較条件

### 9.1 記号的な実行費用

ある資源座標gについて、

\[
C_{\rm total}^{(g)}
=C_{\rm certificate}^{(g)}+C_{\rm setup}^{(g)}
+\sum_a N_a\{Q_a(h,\varepsilon_{\rm read},\alpha_{\rm read})C_{S_2}^{(g)}
+C_{{\rm prep},a}^{(g)}+C_{{\rm read},a}^{(g)}\}
\tag{26}
\]

のように会計する。古典時間、T、CX、depth、memoryは別座標とし、無根拠に合算しない。

標準的なfixed-unitary QPEで、失敗確率を固定した場合、必要base-unitary query数は `O(1/(h εread))` である。精度・失敗確率依存の実装定数や反復法は別途仕様化する。RF0でW>0、Λ、失敗確率を固定し、dとεreadを総誤差εに比例させ、式(24)の等号のstepを採る十分設計では、小さいε/Wの領域で、

\[
h=\Theta\!\left(\Lambda^{-1}\sqrt{\varepsilon/W}\right),
\qquad
Q=O\!\left(\Lambda\sqrt W\,\varepsilon^{-3/2}\right)
\tag{27}
\]

という**保守的な十分query予算**が得られる。これは最適下界でも、実native費用でも、強い既知法を上回った倍率でもない。合成精度に応じてC_S2が変化することも隠さない。

### 9.2 保証取得費用を払う

同じtask・入力access・成功確率のbaselineに対して、

\[
\Delta C_{\rm certificate}+\Delta C_{\rm setup}
< C_{{\rm quantum},\rm baseline}-C_{{\rm quantum},\rm proposed}
\tag{28}
\]

となる根拠が必要である。保証式だけが鋭くても、それを得るためにground情報や大きなSDPが必要なら、全体では得にならない可能性がある。

### 9.3 既存gap-free残差経路との紙上比較

既存レビューS4の保守的比較点は

\[
r_{\rm vec}(h)=|h|^3\Lambda^{5/2}\sqrt e\,e^{|h|\Lambda}
\]

による一step vector残差上界である。この既存式を再利用する。

同じlog/aliasが閉じた領域で、Kのspectral測定に対して

\[
4\,\mathbb E\left[\sin^2\{h(Y-e)/2\}\right]
=\|(S_2(h)-e^{-ihe})|g\rangle\|^2
\]

だから、`|h(Y-e)|≤π`と正のdが満たされれば、

\[
\Pr(|Y-e|>d)
\le\frac{|h|^6\Lambda^5W e^{2|h|\Lambda}}{4\sin^2(|h|d/2)}.
\tag{29}
\]

小h・固定failureで比較すると、こちらの十分stepは概ね `d^(1/2)/(Λ^(5/4) W^(1/4))`、相対形式経路は `d^(1/2)/(Λ W^(1/2))` の依存を持つ。非常に小さいW/Λでは後者に有利な依存があり得るが、定数・α依存・半径制限を払う必要がある。**この粗い旧比較点に対する形式的依存を、最良既知保証に対する優位へ格上げしない。**

既存の「gapあり/なしで二乗根か四乗根か」という経験的な整理は、保証する量と方法を固定した説明であり、一般的限界ではない。この留保はS4にも既にある。

---

## 10. 一次文献との関係と新規性の境界

ここでは実際に取得・読解した範囲に限定して要約する。原論文の主張と本レビューの推論を分ける。

### P1：Mizuta–Kuwahara

[Trotterization is substantially efficient for low-energy states, arXiv:2504.20746v1](https://arxiv.org/html/2504.20746v1)。main setup、Theorem 1、step数への接続、Supplement S2のnested commutator境界を確認した。

局所PSD相互作用・extensiveness・groupingの条件の下で、初期低エネルギーに線形なTrotter誤差と費用評価を扱う。従って「低energyを使う」「energy比例の上界を作る」だけでは新規性はない。

本レビューが明示したものは、任意有限PSD fragmentの小stepに対する実効Kの相対形式と、QPE出力区間への接続である。P1との適用条件・定数・statementの完全包含は未認定。DFを単に「非局所だからP1と無関係」とせず、実際の局所分解・group数・scalar shiftがWと費用へ与える影響まで比較する必要がある。

### P2：Hejazi–Zini–Arrazola

[Better bounds for low-energy product formulas, arXiv:2402.10362v1](https://arxiv.org/html/2402.10362v1)。設定・主要誤差境界・低energy漏洩とnested commutatorの関係を確認した。

低エネルギーPFのprojected作用と漏洩を含む既知の強い対照である。分子・fermionic演算子に適用不能と一括しない。今回の候補がgap不要であるという一点だけでは、P2に対する新規性や優位は立たない。

### P3：Maxwellほか

[Practical Estimation of Trotter Error for Hamiltonian Simulation, arXiv:2606.30738v1](https://arxiv.org/html/2606.30738v1)。§III.1–III.2のBCH/誤差係数評価と計算手順を確認した。

先導誤差の推定・コンパクトな再帰・samplingという既知の進展がある。先導係数の安い計算だけを新しい研究にしない。一方、先導係数の推定があるだけで、(22)を満たす有限stepの確率保証が自動的に得られるわけではない。

### P4：相対形式・エネルギー制約付き解析

[van Luijk, Energy-Limited Quantum Dynamics, arXiv:2405.10259](https://arxiv.org/abs/2405.10259)および[出版ページ](https://doi.org/10.1007/s00220-025-05282-w)。abstract・導入とエネルギー制限条件の記述を確認した。無限次元全般やTrotter関連の全証明を精読したとはしない。

エネルギー制約norm・形式不等式を用いて誤差と費用を扱う枠組み自体は既知である。本レビューはその道具の発明を主張しない。

### 補助検索の地位

SE-QPE（arXiv:2604.14921）、VPE（arXiv:2010.02538）、PF spectral解析（arXiv:2102.12655）、randomized-Hamiltonian phase estimation（arXiv:1907.10070）、Lin–Tong energy estimation（arXiv:2102.11340）は公開abstractを参照した。これらについて全文・補遺・最新出版版との一致を検証したとはしない。今回の中心命題の正当性・非同値性を、このabstract確認だけに依存させていない。

### 判定

- 正半定値性、Cauchy/log、形式不等式、Markov、通常QPEは既知の道具。
- K_SOSの係数辞書は既存project成果。
- RF0/RF1/EWは今回の紙上導出として明示したが、独立した新しい定理と認定していない。
- 完全に既知定理の特殊化なら、その旨を記録し、新規性主張は行わない。
- 具体的な定数・適用クラス・証拠取得の改善が残る場合にだけ、研究テーマとしての追加価値を判断する。

---

## 11. 残る研究課題をさらに限定する

「有限stepの保証があるか」という存在問題は、今回の粗い命題で一つの答えが与えられた。これをもう一度同じ形式でCodexへ考えさせない。

残すべき問いは次の一つである。

> **実装可能な既存DF分割について、入力係数と安価な検証可能証拠だけから得られる相対剰余評価を、最も近い既知保証より有用なstep選択へ結び付けられるか。**

具体的には、Λだけで潰した四次以降の評価を、既存K_SOSのようにfragment間の構造を保持した形へ置き換えられるかが判断対象になる。しかし、単に交換子の同じnormを計算するだけなら既知の範囲である。実行h、誤差・failure配分、証拠bit長、生成/検証費を揃えた差が必要になる。

**新しいSOS分割、Gram回転、pivot選択、旧counterterm、S2の順序scanをここへ混ぜない。** それらは今回の保証方向とは異なり、既存の終了/保留判断に触れる。

---

## 12. 進める場合の最小作業とSTOP条件

これは提案であり、以下を自動実行しない。

### 12.1 次の一回の作業

ユーザーがA2のこの限定した保証・schedule方向を選ぶ場合、科学設計側/技術検証担当には、**本書§§5–8の命題と、P1/P2の指定定理の差分**を一つの対象として渡す。

確認するのは、(i) RF0の両端telescoping・複素log・偶対称性、(ii) K0 Eq7–9の全vector形式への適用、(iii) 有限相対剰余、(iv) energy-window確率とQPEの接続、(v) 同じ条件の既知保証による完全包含、である。

広い候補一覧、repository全体の棚卸し、全paperの再収集、M1-Sの再解析を行う必要はない。未実装adapter、分子生成、native計測、数値実験を先に作らせない。

### 12.2 結果による区切り

| 結果 | 扱い |
|---|---|
| 紙上導出に誤り | どのstatementが失われるか限定し、独断で新数値や別分割へ移らない |
| 全て既知定理の直接系、費用差の具体機構なし | 既知の有用な基準として保存。新算法としては採択しない |
| 成立するが定数・半径・Wが実用条件を潰す | その制限を記録。保証の存在だけを理由に分子benchmarkへ進まない |
| 具体的により鋭い/安価な相対証拠の設計があり、比較taskが閉じる | その設計・適用範囲に限定して科学的採否を判断。SOURCE/数値は別承認 |

独立priorityが不明なだけで必ず次の広いレビューを開催するわけではない。どの定理のどの仮定・定数・出力が差なのかを示せない場合、同じ調査の反復で埋めない。

### 12.3 権限

- 本レビューは新しい主題の正式採択ではない。
- A1/current encoding等のCLOSED・CONCLUDEDを解除しない。
- A2は現時点でNOT_ADOPTEDのまま。
- 新規科学計算、symbolic checker、SOURCE・実行契約作成、compile、grant発行は未承認。
- 後日のCodex指示では最新版§8.4を維持する。恣意的なCPU/RAM/worker上限を設けず、共有環境・他ユーザーへの変更は禁ずる。既存の固定条件は無断変更しない。

---

## 13. 支持できる主張・支持できない主張

### 今回支持するもの

- 研究課題を「実装接続の空白」だけから選ぶより、具体的な実行手順を変える数理的障害から選ぶ方が、本件では重複を避けやすいという研究判断。
- 与えたPSD/S2/小step条件の下でのRF0/RF1/EWの紙上導出。独立検証前であることを保持する。
- gapや単一branch高overlapは、すべてのenergy精度保証経路で必須ではない。
- 同じkernelでも認証済みstepの選択を変えれば、実行費用を変える可能性がある。
- A2の限定再設計は、旧A1分割探索とは区別できる。ただし新発見の別分野ではない。

### 今回支持しないもの

- 新しい算法・定理の独立新規性が確立した。
- 既知の低エネルギーPF保証より強い、または最適である。
- 化学分子で有用なW/Λ・K_SOSが得られる。
- 実native費用・準備費込み総資源が減る。
- 任意の近似state、DF誤差、実機ノイズ、多round RPEへ自動適用できる。
- 未取得原本や全過去レビューを今回独立に再証明した。
- ①PRリソース評価の結果を今回の候補のGOに用いた。

---

## 14. 取得資料・読解範囲・再現性

### 固定repository資料

| ID | 固定資料 | 今回の読解範囲 |
|---|---|---|
| S1 | `67576526...` / `docs/research/qc_exact_encoding_direction_adoption_20261017/decision.json` | 採択・状態・権限の全文 |
| S2 | `56f1cc55...` / `docs/research/qc_new_algorithm_research_scope_20261016/gpt_decision_handoff.md` | 現課題、既存調査、未採択事項、入力条件の全文 |
| S3 | `830be753...` / `docs/research/qc_exact_encoding_controlled_construction_20261017/decision_handoff.md` | 一案の既知還元、未検証、保全・次の境界の全文 |
| S4 | `dbe51f18...` / `docs/reviews/quantum_chemistry_pf/qc_sos_pf_post_k05_gpt_scientific_review_20261016.md` | 今回取得した冒頭〜155行、結論・A2留保・旧gap-free比較点・取得範囲。全文の再検証ではない |
| S5 | `58d86727...` / `docs/research/qc_sos_pf_k0_20261015/technical_audit.md` | 今回取得した65–326行。二分割、K_SOS、有限剰余・gap・overlap、PSD化とscalar不変性 |
| S6 | `/mnt/data/qc_algorithm_strategy_ii_iii_iv_scientific_review_20261017.md` | 前回戦略レビューの冒頭・主方針・encoding推奨の範囲。原本保存、上書きなし |
| S7 | `/mnt/data/research_project_instructions_unified_2026-10-17.md` | 正式共通運用指示を確認。特に担当・承認・§8.4・記録 |

S1–S5は本レビューでGitHub connectorから実際に取得した。過去会話に表示された他資料は経緯把握には用いたが、今回すべてを再取得・再認証したとは記さない。特に旧Claude原本・未取得出版版本文の欠落を、要約から埋めていない。

### 一次文献の取得範囲

P1〜P3は固定arXiv HTMLの上記関連箇所を実際に確認した。P4と補助検索は§10記載の範囲に限定する。改訂版・出版版・全式の同一性は認証していない。PDF本文・図表を今回独立に分析したとは主張しない。

### 作業の区分

紙上の式変形は実施した。科学的数値計算・保存値の再集計・数値またはsymbolic checkerは実行していない。ファイル保存・UTF-8確認・hashは文書管理であり、数式の数値検証やSOURCEテストのPASSではない。

---

## 15. 最終判断と不確かさ

**今回の推奨は、一つの具体的な保証・schedule問題へ絞ること。直ちに新算法採択・A2実行・分子benchmarkへ進むことではない。**

既存の停止済み構成を繰り返さない方針には高い確信がある。RF0/RF1/EWの提示した範囲での紙上導出には検討上の根拠があるが、独立検算前のため未認証として扱う。これを研究の第一候補にする判断の確信度は中程度である。独立priority、実用的定数、証拠取得費を含む優位性の確信度は、現資料では付与できない。

新しい回路生成だけを主題とする場合、この提案は保証・実行schedule方向への範囲変更を含む。その採否はユーザーが判断する。今回、具体的な別の新回路生成規則を得たとは報告しない。

レビュー完了後も、既存SOURCE、raw結果、契約、grant、ledger、marker、終了・保留記録は不変である。

**Mandatory STOP。研究方針の正式採択・後続検証の指示・科学計算の実行承認は、それぞれ別に扱う。**
