# EW2・交換子対照・認証step選択の独立数学的監査

2026-10-17 JST。紙上の技術検算。科学数値計算・symbolic checker・SOURCE実装・新しいGPT科学レビューではない。

**EW2、元レビューの逆log式・確率対照・step反転は、以下の条件内で成立する。重大な数学的欠陥は見つからなかった。** 比較の際には、実時間unitarityと既知scalar centeringから同じ情報で得られる、さらに強い交換子対照も許す必要がある（§5）。この強化を含めた有限stepの条件を[比較表](finite_step_and_prior_art_comparison.md)に示す。総費用優位・新規性・正式採択は別判断である。

[判断用ハンドオフ](decision_handoff.md)、[証拠索引](evidence_index.json)、[今回のレビュー原本](inputs/qc_relative_form_post_audit_novelty_value_review_20261017.md)、[ユーザー指示原文](inputs/user_instruction_ew2_step_audit_20261017.txt)を併読する。原本はbyte保存し、改訂していない。

## 1. 固定前提と前回監査からの依存

有限次元の共通不変sectorで

\[
H=LI+G,\quad G=\sum_jF_j,\quad 0\preceq F_j\preceq f_jI,
\quad\Lambda=\sum_jf_j,\quad e=\lambda_{\min}(G)=E_0-L,
\quad0\le e\le W=U-L.
\]

E0は対象sectorのground energy。Lは実際のPSD分割に対応し、Uは同じHの認証上界である。G≤ΛIも使えるので、同じ情報からWをmin(W,Λ)へ狭める選択は両方式に共通に許される。ただし以下では与えられた共通のWを表記する。真のeやground vectorを古典oracleにしない。

正stage、固定順序のpalindromic S2(h)と原点からの解析的logを用いる。前回の[固定監査 §1–3](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/051df4661eeffb6c6c4c978f70f43eeeaadfddd6/docs/research/qc_relative_form_finite_step_audit_20261017/mathematical_audit.md)の仮定と結果を再確認したが、全証明は反復しない。

\[
K(h)=\frac{i}{h}\operatorname{Log}S_2(h),\quad
-\eta G\preceq\Delta:=K-G\preceq\eta G,
\quad\eta=\min(\eta_0,\eta_1),
\]
\[
\eta_0=\frac{32(h\Lambda)^2}{1-16(h\Lambda)^2},\qquad
\eta_1=h^2K_{\rm SOS}+\frac{512(h\Lambda)^4}{1-16(h\Lambda)^2}.
\tag{1}
\]

相対形式には|h|Λ<1/4、確率議論にはさらに0≤η<1。K_SOSは前回と同じK0の行和定数である。負stage高次PFには適用しない。h=0はK=Gの可除延長だが、QPEでh除算する実行点にはしない。Λ=0はG=0として別扱いする。

## 2. EW2の独立導出

### 2.1 逆の存在と非可換な積順序

まずe>0。G≥eI>0、`K≥(1−η)G>0` よりG^{-1/2}とK^{-1}が存在する。congruenceにより

\[
R=G^{-1/2}\Delta G^{-1/2},\quad R=R^\dagger,\quad
-\eta I\preceq R\preceq\eta I,
\quad K=G^{1/2}(I+R)G^{1/2}.
\]

この分解の逆を**順序どおり**取ると

\[
K^{-1}=G^{-1/2}(I+R)^{-1}G^{-1/2},
\]
\[
\Delta K^{-1}\Delta
=G^{1/2}R(I+R)^{-1}RG^{1/2}
\preceq\frac{\eta^2}{1-\eta}G.
\tag{2}
\]

間のG^{1/2}G^{-1/2}だけを消している。GとR、GとKを交換していない。Rは(I+R)^{-1}と可換であり、そのspectral calculusで `r²/(1+r)≤η²/(1−η)`（−η≤r≤η）を使う。逆作用素は証明用であり、古典計算・量子回路・oracleの入力ではない。

### 2.2 ground上のweighted moment

Gg=eg、||g||=1ならΔg=(K−e)gである。従って、Kのspectral output Yについて

\[
\begin{aligned}
\mathbb E\frac{(Y-e)^2}{Y}
&=\langle g|(K-e)K^{-1}(K-e)|g\rangle\\
&=\langle g|\Delta K^{-1}\Delta|g\rangle
\le\frac{\eta^2e}{1-\eta}.
\end{aligned}
\tag{3}
\]

同じground空間内の各vectorに成り立つので、縮退ground上の混合stateにも線形性で成立する。単一K固有枝への高overlapは使わない。一般の近似stateをgで置き換えることはできない。

### 2.3 両側tailとWへの置換

e>0、y>0で `φ(y)=(y−e)²/y=y−2e+e²/y`。その導関数は `1−e²/y²`。上側y>e+dでは `φ(y)≥d²/(e+d)`。下側が存在するd<eでは `φ(y)≥d²/(e−d)>d²/(e+d)`、d≥eでは下側eventが空である。よってd>0に対し、(3)を非負変数のMarkov評価へ入れて

\[
\Pr(|Y-e|>d)\le
\min\left(1,\frac{\eta^2e(e+d)}{(1-\eta)d^2}\right)
\le\min\left(1,\frac{\eta^2W(W+d)}{(1-\eta)d^2}\right).
\tag{EW2}
\]

最後の置換は、t≥0におけるt(t+d)の単調性による。η、W、dの次元はそれぞれ無次元・energy・energyである。

e=0なら任意のground gを各PSD F_jが消すため、S2g=g、原点branchのKg=0、Y=0。逆を用いず処理する。W=0もe=0かつE0=L既知、Λ=0も自明である。η=0ならΔ=0でY=e。d=0、η≥1はこの保証の対象外であり、逆や分母をそのまま使わない。

## 3. 元EWとの比較：全域での優劣ではない

前回EWからの両側上界は、ηW≤dを満たすとき

\[
B_1=\frac{2\eta W}{d+\eta W}.
\]

EW2の非clippedな右辺をB2とし、w=W/d>0、η>0とすると

\[
\frac{B_2}{B_1}
=\frac{\eta(w+1)(1+\eta w)}{2(1-\eta)}.
\tag{4}
\]

従って元EWが両側で使える領域では、EW2が小さいための正確な条件は右辺≤1である。固定W,dでηが小さい極限ではB2は二次、B1は一次。ただし全域でEW2が小さいわけではない。η=0では両者0、clip後の1も忘れない。

EW2自体はηW≤dを追加仮定しない。一方、B2<1なら `B2≥(ηW/d)²/(1−η)` よりηW<dが自動的に必要である。「条件が不要」を、ηW>dでも非自明な成功保証が得られるという意味に拡大しない。両上界はそれぞれの適用条件で併用できる。

## 4. 真値なしのstep選択

W,d>0、0<α=α_spec<1として

\[
c=\frac{\alpha d^2}{W(W+d)}>0.
\]

0≤η<1で `η²/(1−η)≤c` は `η²+cη−c≤0` と同値。負の根を除くと

\[
\eta\le\eta_*:=\frac{\sqrt{c^2+4c}-c}{2},\qquad0<\eta_*<1.
\tag{5}
\]

多項式はη=0で負、η=1で正、正の根は一つであり、η<1条件も保たれる。将来有限精度で評価するなら、等価な `η_*=2c/(sqrt(c²+4c)+c)` 等と有向丸めを検討する必要があるが、今回その数値評価はしない。

RF0の反転は

\[
0<h\le h_0:=\frac1\Lambda
\sqrt{\frac{\eta_*}{32+16\eta_*}}.
\tag{6}
\]

Λ>0ならh0Λ<1/4であり、原レビューの十分step条件は正しい。RF1でもη1(h)≤η*を使える。η1は正のhに単調で、1/4領域の端で発散するので、一意なh1を `η1(h1)=η*` で定義できる。x=h²を使うだけなら、同じ条件は

\[
(512\Lambda^4-16K_{\rm SOS}\Lambda^2)x^2
+(K_{\rm SOS}+16\eta_*\Lambda^2)x-\eta_*\le0,
\quad0\le x<\frac1{16\Lambda^2}.
\tag{7}
\]

二次係数の符号を仮定せず、元の単調関数と指定領域で判断する。min保証からの最大の十分stepは `h_R=max(h0,h1)` である。

さらに、与えられた二式には正確な恒等式 `η0=32Λ²h²+ρ4` がある。従って

\[
\eta(h)=a h^2+\rho_4(h),\qquad
a=\min(32\Lambda^2,K_{\rm SOS})\ge0.
\]

これはleadingだけの近似ではない。このaを使い

\[
A=512\Lambda^4-16a\Lambda^2\ge0,\qquad
B=a+16\eta_*\Lambda^2>0
\]

とおけば、min保証のendpointを閉式で

\[
\boxed{h_R^2=\frac{2\eta_*}{B+\sqrt{B^2+4A\eta_*}}}
\tag{7a}
\]

と書ける。A=0でも同じ式がη*/Bを返す。これは正の分母で(7)を処理した一意な根で、h_RΛ<1/4を満たす。a=0の場合も有限剰余を落とさず扱える。この紙上の選択規則は実行契約・数値探索ではなく、同じL,U,f_j,K_SOSとd,αだけを用い、真のeを評価しない。

## 5. Childs交換子対照と逆logの独立検算

### 5.1 原文との係数・順序・実時間条件の対応

[Childsら arXiv:1912.08854v3 §5.1 Eqs.(146)–(152), Proposition 16](https://arxiv.org/pdf/1912.08854v3)をPDF本文と印刷p.39の画像で確認した。二分割ABAのEq.(150)でA=F_j、B=R_jを取り、内側S2へ再帰する。各outer exponentialは実時間unitaryなので内部誤差のnormは増えない。従って、元レビューと同じ固定順序について

\[
C_2=\sum_{j<M}\left\{\frac{\|[F_j,[F_j,R_j]]\|}{24}
+\frac{\|[R_j,[R_j,F_j]]\|}{12}\right\},\quad R_j=\sum_{k>j}F_k,
\]
\[
\|S_2(h)-e^{-ihG}\|\le |h|^3 C_2\le |h|^3\overline C_2.
\tag{8}
\]

係数1/24と1/12は入れ替わっていない。paperのindexed product表記だけに依存せず、outer ABAで順序を対応させた。原文のt≥0から負hへはpalindromeの `S2(−h)=S2(h)^†` とadjoint normで拡張できる。Hermitian fragmentが必要で、PSDは(8)自体に不要。実時間(8)はRFの1/4領域に制限されない。C2およびその認証上界はenergy³、exact many-body normの無料取得は仮定しない。

### 5.2 元レビューの非可換なlog差分

x=|h|Λ<1/4、A=S2−I、B=exp(−ihG)−Iとすると `||A||,||B||≤q=e^x−1<1`。正確な恒等式

\[
A^n-B^n=\sum_{j=0}^{n-1}A^{n-1-j}(A-B)B^j
\]

から `||A^n−B^n||≤n q^(n−1)||A−B||`。log級数を足して

\[
\|\operatorname{Log}(I+A)-\operatorname{Log}(I+B)\|
\le\frac{\|A-B\|}{1-q}.
\tag{9}
\]

原点からの同じbranchでLog(exp(−ihG))=−ihGなので

\[
\|K-G\|\le b_{\exp}(h)=\frac{h^2\overline C_2}{2-e^{|h|\Lambda}}
\le\frac32h^2\overline C_2.
\tag{10}
\]

最後はe^x≤1/(1−x)≤4/3による。分母は正である。元レビューの式は正しい。ただし同じ(9)のb_exp自体はx<log 2まで成立し、3/2への単純化だけが1/4領域のもの。この対照の全適用域をRFのために切り縮めない。

### 5.3 同じ情報で使える実時間・centering付きの強い対照

実hでは一端telescopingとunitarityから、より直接に
`||S2−I||≤|h|Λ`、`||exp(−ihG)−I||≤|h|Λ`。
従って(9)でq=|h|Λを使え、x<1で `||K−G||≤h² Cbar2/(1−x)` が成立する。これは(10)より強く、領域も広い。

さらに入力だけから、`c_j=f_j/2`、`A_j=F_j−c_jI` は `||A_j||≤f_j/2`。`c_0=Σc_j=Λ/2`、`β=Σf_j/2=Λ/2` とおく。**元のF_j、L、H、S2は変更しない。** 証明中のunitaryを既知scalarでcenterして

\[
U_c=e^{ihc_0}S_2(h),\qquad V_c=e^{-ih(G-c_0I)}.
\]

UcはA_jの同じstage積であり、`||Uc−I||,||Vc−I||≤|h|β`。scalarは交換子と `||Uc−Vc||` を変えない。(9)より

\[
K_c=c_0I+\frac{i}{h}\operatorname{Log}U_c,\qquad
\boxed{\|K_c-G\|\le b_C(h):=\frac{h^2\overline C_2}{1-|h|\beta}},
\quad |h|\beta<1.
\tag{11}
\]

元RF領域かつη<1では `0≤K≤(1+η)ΛI` なので、`||h(K−c0I)||≤(1/2+η)|h|Λ<3/8<π`。よってbranchが一致し **K_c=K**。RFとの比較点では、同一generator・同一energy-windowへの、より強い対照である。

RF領域の外でも、|h|β<1ならcentered unitaryがIから距離1未満にあり、そのlog級数と既知scalarを戻したK_cは定義できる。QPEにはこの既知center周りのunwrapを使う。元のunshifted principal logを無条件に流用しない。実時間(8)やその他の対照までhΛ<1/4へ制限しない。

(11)は指定されたlog差分と既知scalar処理の直接的な帰結であり、新しいPF・入力norm oracle・SOURCEではない。既知phaseの保持・復元・制御回路の費用は別途共通会計に残す。よりtightな利用可能なcenter/capや別の正当な保証があれば、対照側にも許す。βを最適帯域だとは主張しない。

## 6. 同じenergy-windowとQPEへの接続

gがGのgroundなら `(K−e)g=(K−G)g`。ゆえに

\[
\mathbb E(Y-e)^2=\|(K-G)g\|^2
\le b(h)^2,\quad
\Pr(|Y-e|>d)\le\min(1,b(h)^2/d^2).
\tag{12}
\]

bは(10)でも(11)でもよく、後者では同じbranchのK_cを用いる。混合groundにも成立する。η<1、PSD、励起gap、特定PF固有枝は(12)のmoment評価自体には不要であり、operator上界と正しいlog/energy原点が必要である。相対形式だけがgap-freeという比較は誤りである。

固定S2の整数powers、既知phase支持からのunwrap、各固有値に一様なreadout誤差・失敗率を共通条件とすれば、両方式とも

\[
\Pr(|\widehat E-E_0|>d+\varepsilon_{\rm read})
\le\alpha_{\rm spec}+\alpha_{\rm read},\qquad \widehat E=L+\widehat Y.
\]

centered値を読むならL+c0を一度戻す。同一試行内の独立性は不要。stateのtrace-distance、係数/モデル/合成・controlled powersの全実装誤差、証拠失敗を別予算にする。逆演算子の量子費用はないが、exact ground copiesの準備費はある。RPE/M1-Sの平均信号へ移さない。

対照には(8)からのQPE回路telescoping `TV≤Q |h|³ Cbar2` も許される。Qは全powers中のbase-S2適用数。理想のG-QPEの保証へこの確率差を加える方法はgeneratorの小log領域を必須にしないが、理想QPEのalias/readoutとQ依存を別途満たす必要がある。小さい一step誤差をQ倍の蓄積なしに使わない。

元のspectral residual経路も、`||[S2−exp(−ihe)]g||≤|h|³ Cbar2` と支持・閾値のsine条件があれば利用できる。低energy専用保証、既知のさらに鋭いinverse-log保証も、適用条件と取得費を満たす範囲で比較対象から除外しない。

## 7. 有限step・Xi・特異条件の所在

[有限step比較表 §2–4](finite_step_and_prior_art_comparison.md)に、(5)–(7)と(11)–(12)からの最大十分step、厳密な記号的優位条件、Xiの漸近係数を示す。数値を代入していない。

- Cbar2=0なら(8)は全実hでS2が厳密であることを証明する。可換fragmentはこのcase。RFの正の粗い上界を使って優位を主張しない。
- K_SOS=0または有効なa=0なら、RF1の四次剰余を保持する。二次leadingで定義したXiによるstep比は適用しない。
- hΛ→1/4、η→1ではRFの剰余・EW2分母を落とさない。
- 近似state、元Hと異なるDFモデル、不確かなL/U/capは別証拠なしに本保証へ入れない。

中心証明に撤回を要する箇所はなかった。数学的な未閉鎖部分をoracleで救う操作もしていない。全known-baseline envelopeへの優位、新規性、認証取得費込みの資源差は今回証明していない。

**Mandatory STOP。** A1 CLOSED、A2 NOT_ADOPTED、K1-F未承認・未実行を維持する。
