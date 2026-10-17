# 有限step相対形式保証：独立監査後の新規性・研究価値・限定継続判断

- 作成日：2026-10-17 JST
- 文書区分：ユーザーの「レビューを開始して　丁寧に考えて」に基づく重要科学レビュー
- 対象：④の未採択A2に関係するRF0／RF1／EW／固定powers QPE
- 中核固定資料：`HIROMU1015/Partially-Randomized-Trotter`、`051df4661eeffb6c6c4c978f70f43eeeaadfddd6`
- 状態：レビュー完了。以下の判断は推奨であり、ユーザーの研究方針採択、SOURCE変更、実行承認ではない。
- 実施：監査資料・既存K0/K0.5・元レビューの読解、一次論文の必要部分の照合、独立した紙上の推論・導出、直接重複する高次PF資料の確認。
- 未実施：新規科学数値計算、保存科学値の再集計、symbolic checker、Hamiltonian／固有値／RDM生成、simulation、native compile、T合成、科学SOURCE実装、GitHub書込み、契約・grant・markerの変更。
- 対象外：①PRリソース評価・H8 Phase C、②③の追加探索、M1-S再解析、他研究の採否。

## 0. 結論

**RF0・RF1・元のEW・固定powers QPEへの接続について、監査に明示された仮定の範囲で成立を支持する。中心証明を撤回すべき欠陥は確認しなかった。** 元レビュー補助式(29)のsine閾値の適用域補足も支持する。[R1–R3]

一方、数学的に成立したことから、新しいPF生成アルゴリズムや実用的な量子資源優位を認定しない。新規性判断では、Mizuta–KuwaharaのTheorem S1が任意の射影を扱うため、rank-one射影によって交換子の全vector評価へ接続できる点を重視する。「射影付き評価に対し全vectorだから新しい」という区別だけでは足りない。[P1]

**推奨は、A2のうち『入力から検証可能な有限step相対証拠を作り、既知保証より有用なstep選択へ接続する』部分に限った理論・比較研究の継続である。** 新アルゴリズムの主研究としての全面採択や、直ちに分子benchmarkへ進むことは推奨しない。全体の既知性がまだ未確定であることのみを継続理由にはしない。

今回、その判断に関係する追加材料を二つ紙上で得た。

1. 同じ相対形式保証から、元のMarkov型EWとは別に、相対誤差ηに二次で依存する両側の出力確率上界を導出した（§6、便宜名EW2）。逆作用素は証明だけに現れ、量子inverse oracleは要求しない。独立監査前の今回の導出であり、新規性は主張しない。
2. 既知の二次PF交換子上界を、同じ固定unitaryのlog・同じenergy-window出力へ接続した比較を記述した（§8）。この対照もgap-freeであり、S2のε依存の指数だけでは差にならない。

研究として残る中心は、非局所DF等での『有限相対log保証のstatementと証拠取得手順』の独立性・有用性である。低エネルギー性、gap-free、先導BCH係数、Markov、通常QPE、定数32/512の存在そのものには新規性を置かない。

| 判断対象 | 今回の結論 | 確信の範囲 |
|---|---|---|
| RF0/RF1の成立 | 記載仮定内で支持 | 紙上の中心論理は高。形式証明器による認証ではない |
| 元EW/QPE | 記載条件内で支持 | ground支持、同一powers、branch・別誤差会計を維持 |
| 全statementのpriority | 未確定 | 既知定理への完全な包含も、非包含も確定していない |
| 実用的な資源優位 | 未確立 | 分子入力・認証費・native費の新評価はない |
| 限定理論研究の継続 | 推奨 | 中程度。具体的な証拠・比較問題があることが理由 |
| A2の正式状態 | NOT_ADOPTEDのまま | このレビューで変更しない |

## 1. 資料と証拠の扱い

### 1.1 直接の資料

[R1] 数学的監査、[R2] 先行研究・実用条件、[R3] 判断用ハンドオフをGitHubの固定commitから取得した。長いAPI出力は該当行範囲を別取得して補った。元レビュー[O1]は会話に提供された38,606 bytesのMarkdownを利用した。元レビュー・Codex監査・今回の新導出は別文書・別の証拠層である。

K0 [R4]の係数辞書・分散上界・行和・有限剰余、K0.5 [R5]のgap-free対照・Childs交換子上界・証拠費用を確認した。現行正式状態は[R3]の参照する[S0]に従う。

高次PF側は[H1]のU1/U3/U7と重複整理だけを直接読んだ。高次PF repository全branch、全実験、全レビューの監査ではない。

### 1.2 一次論文と取得の限界

[P1]の固定v1 HTML、PDFから得た本文、主定理・Supplement S2 Theorem S1の関連証明を確認した。[P2]は固定v1 HTMLの設定と低エネルギー誤差・漏洩の関連節。[P3]は固定v1 HTMLと取得できたPDFテキストのIII.A/BおよびAppendix Aの関連部分。[P4]はarXiv v3のPDF §5.1、Eqs.(150)–(152)、Proposition 16を読み、PDFの印刷p.39を画像でも確認した。

[P1]・[P3]ではPDF画像取得が失敗したため、取得済みHTMLの式・PDF解析テキストを使用した。本文と数式を取得できなかったわけではない。コンテナへの原PDFダウンロードは失敗し、今回のローカルPDF byte/SHA認証は行っていない。Codexが以前保存したPDFのhashを、今回自分が再認証したとはしない。

arXivの登録ページでも版を確認したが、出版版本文とpreprintの全式同一性を確認したわけではない。[P5/P6]等の相対摂動・spectral解析は補助的な位置付けのみで、そこから本稿の式を引用・輸入していない。文献検索の非検出は新規性の証明ではない。

## 2. 固定する数学的問題

有限次元の共通不変sectorにおいて

\[
H=L I+G,\qquad G=\sum_{j=1}^M F_j,\qquad
0\preceq F_j\preceq f_j I,\qquad \Lambda=\sum_j f_j.
\tag{1}
\]

同じ実装分割に対応するscalar L、認証された上界Uを用いる。

\[
e=E_0-L=\lambda_{\min}(G)\ge0,\qquad W=U-L\ge e.
\tag{2}
\]

eは証明中の記号で、実行時に真値を要求しない。より粗いUを使う場合はG≤ΛIからWをmin(W,Λ)へ狭める自明な選択も可能だが、新機構ではない。一般SOSの下界を、その下界を実現しない別のDF分割へ流用しない。

対称二次PFは

\[
S_2(h)=e^{-ihF_1/2}\cdots e^{-ihF_{M-1}/2}e^{-ihF_M}
 e^{-ihF_{M-1}/2}\cdots e^{-ihF_1/2},
\qquad K(h)=\frac{i}{h}\operatorname{Log}S_2(h).
\tag{3}
\]

Logは原点から解析接続するbranchであり、|h|Λ<1/4を共通の十分領域とする。h=0ではK=Gを連続延長するが、QPEでh除算する実行点にはしない。Λ=0はG=0の自明case。

元のEWおよび今回のEW2が出力E0を保証するstateは、Gのground空間に支持された純粋・混合stateである。ground縮退は許す。未知の近似stateをそのままこの条件へ置き換えない。対象sectorは各logical fragmentが保存する必要があり、各native gateでの逐次保存を余分な条件にはしない。

これはpositive-stage S2の定理である。負stageを持つ4次・6次・8次PFへ同じηを転用することはできない。新しい分子、数値精度、成功確率、費用metricを今回固定しない。

## 3. 独立監査の評価：RF0・RF1は何を証明しているか

### 3.1 RF0の中心論理

[R1]の二端telescopingは、非可換な内部積を保持している。

\[
X=S_2(z)-I=\sum_jB_j+\sum_{i>j}B_iE_{i-1}\cdots E_{j+1}B_j,
\quad E_l=e^{-izA_l},\quad B_l=E_l-I.
\tag{4}
\]

stage A_lはPSD、ΣA_l=G、cap和Λ。任意の単位vector vに対し

\[
\sum_l\|A_lv\|\le\sqrt{\Lambda\langle G\rangle_v}.
\]

したがって、両端のBをvへ当て、内部積だけをoperator normで評価できる。途中の回転stateを元のstateへ置き換える不正な仮定はない。

r=1/(4Λ)の円上で、log級数とPSD期待値から

\[
|\langle v,K(z)v\rangle|\le\frac{11}{6}\langle v,Gv\rangle
<2\langle v,Gv\rangle.
\]

palindromeによるKの偶対称性とCauchy評価から

\[
-\eta_0(h)G\preceq K(h)-G\preceq\eta_0(h)G,\qquad
\eta_0(h)=\frac{32(h\Lambda)^2}{1-16(h\Lambda)^2}.
\tag{RF0}
\]

Gの逆を用いないので、Gの零空間も扱える。zero-energy vectorを各PSD stageが消す点が重要である。実hでK−GがHermitianであるため、全vectorの二次形式からLoewner順序へ移れる。

この論理と定数を支持する。ただし32は最適定数ではない。η0<1には|h|Λ<1/√48が十分で、単なる1/4領域と混同しない。

### 3.2 RF1の再利用範囲

K0 [R4]は

\[
D=\sum_{j<M}\{[F_j,[F_j,R_j]]/24-[R_j,[R_j,F_j]]/12\},
\quad R_j=\sum_{k>j}F_k
\]

のraw係数を用いる。PSD分散評価と非負成分対称行列の最大行和K_SOSには、ground固有方程式を必要としない。よって全vectorに対する

\[
-K_{\rm SOS}G\preceq D\preceq K_{\rm SOS}G
\tag{5}
\]

への読み替えは妥当である。K0の別節にあるground交換子消去や二分割bias符号を、この全vector評価へ混ぜない。

K−G−h²DのCauchy剰余も相対形式で抑えられ、

\[
\eta_1(h)=h^2K_{\rm SOS}
+\frac{512(h\Lambda)^4}{1-16(h\Lambda)^2}.
\tag{RF1}
\]

同じG/S2/cap/domainである限りη=min(η0,η1)を使える。K_SOSはenergy²、Dはenergy³で、ηは無次元。K0のO(M³) raw演算・O(M)行和保存量は既存成果であり、新アルゴリズムとして再提出しない。

## 4. 新規性の判断を狭める

### 4.1 全vectorという語だけでは区別できない

Mizuta–Kuwahara [P1] Supplement Theorem S1は、PSD局所項からなるk-local・g-extensiveなH_0,…,H_qと、各ΠH_jΠ≤ΔΠを満たす**任意の射影Π**を対象とする。nested commutatorの射影normをq!(2kg)^qΔで抑える。

ここからの本レビューの推論として、Π=|v><v|、H_jを選択したfragment、Δ=<v,Gv>と取れば、F_j≤Gにより仮定を満たし、個々の交換子の全vector期待値評価が得られる。H_jの局所性等の仮定は残す。

したがって、「先行研究は射影を使うがRF1は全vector」という比較は、先導交換子の新規性を単独で支えない。一方、これだけで無限log級数、明示した有限領域、RF0/RF1の全剰余まで同じ定数で得られることを示したわけでもない。

非局所DFを単一の全support項として[P1]へ当てはめる粗い方法は排除しない。その場合のk、g、群数と係数の悪化を含めて比較する。「DFは非局所だから先行定理は無関係」とは言えない。

### 4.2 その他の近接結果

[P2]は低エネルギー部分空間の保持成分と漏洩を組み合わせたPF誤差を扱う。PF次数、local support、degree、群構造の条件があり、低エネルギー性やgap-freeという特徴は既にある。射影発展誤差を、有限logの全空間相対順序へ無条件に置き換えることはできない。

[P3]の漸近・dephasing解析とcompact BCH表現も重要な対照である。読んだ証明の固有値差分母を、uniformなgap-free有限h定理と解釈しない。先導係数計算法と、有限相対剰余の証明は異なる問題である。

[P4]は可換性を残した二次PFの有限実時間operator上界を与える。これをenergy-windowへ戻すだけでもgapを使わない比較が作れる（§8）。

### 4.3 貢献候補と既知の構成要素

| 論点 | 今回の新規性上の位置付け |
|---|---|
| PSD分散・低energy交換子 | 既知。全vector化の一部も既知補題で説明可能 |
| K0の辞書・先導BCH | 既存project成果・近接既報あり |
| 明示有限hでの相対log remainder | 最も具体的な貢献候補。ただし完全priority未確定 |
| 元EWのMarkov | 相対形式からの初等的帰結。単独の主張の柱にしない |
| 今回のEW2 | 相対摂動からの短い導出。新規性未認定 |
| 固定powers QPE・scalar復元 | 既知の構成を適切に接続する部分 |
| 安価な認証データから有利なstep選択 | 算法研究としての中心候補。実現・優位は未確立 |

「同じstatementを見つけられなかった」だけで新規性を認定しない。同時に、道具が初等的・既知であることのみで、有限定理全体の独立性を否定もしない。現時点では`VALID_WITH_UNRESOLVED_NOVELTY`を越えるpriority判定はしない。

## 5. 元のEW・QPE接続を維持する

相対形式

\[
-\eta G\preceq K-G\preceq\eta G,\quad0\le\eta<1
\tag{6}
\]

とground入力から、Kのspectral output YはY≥(1−η)e、EY≤(1+η)e。よって

\[
\Pr(Y>e+d)\le\frac{2\eta e}{d+\eta e}.
\tag{EW}
\]

下側tailはd≥ηeの場合に0。0<α_spec<1と

\[
\eta W\le\frac{\alpha_{\rm spec}}{2-\alpha_{\rm spec}}d
\tag{7}
\]

が両側保証の十分条件となる。[R1]のこの点を支持する。

QPEは同じS2(h)のpowersを使う。既知supportとscalarによるcentering、branch/unwrap、固有値ごとに一様なreadout誤差を明示する。元のHのoffset Lを一度だけ復元する。固有枝への高いoverlapがなくても、許容energy帯内に出る確率があればよい。

元レビュー補助式(29)のsine閾値評価については、非自明eventでは0<|h|d≤πと支持差の範囲を要する。eventが空なら0、h=0も別扱い。これはRF0/RF1/EWの欠陥ではない。

## 6. 今回の追加導出：相対残差による二次のenergy-window保証

**この節は今回のGPT紙上導出であり、完了済みCodex監査の判定対象には含まれない。以下を新しい文献上の定理と認定していない。** 同じ相対形式(6)をより直接利用する。

### 6.1 proof-onlyな相対誤差

e>0ならG>0であり、Δ=K−G、

\[
R=G^{-1/2}\Delta G^{-1/2},\qquad -\eta I\preceq R\preceq\eta I,
\qquad K=G^{1/2}(I+R)G^{1/2}.
\tag{8}
\]

η<1なのでKも正定値。正しい積順序で

\[
\Delta K^{-1}\Delta
=G^{1/2}R(I+R)^{-1}R G^{1/2}
\preceq\frac{\eta^2}{1-\eta}G.
\tag{9}
\]

最後の評価は、実r∈[−η,η]でr²/(1+r)≤η²/(1−η)というspectral calculusの帰結。GとKの可換性は仮定しない。

G|g>=e|g>からΔ|g>=(K−e)|g>。YをK測定値とすると

\[
\mathbb E\frac{(Y-e)^2}{Y}
=\langle g|\Delta K^{-1}\Delta|g\rangle
\le\frac{\eta^2 e}{1-\eta}.
\tag{10}
\]

混合ground stateにも線形性で成立する。G^{-1/2}、K^{-1}はこの証明にのみ現れ、実行入力、数値計算、量子oracle、block-encodingを要求しない。真のeをstep選択へ入力することもない。

### 6.2 両側tail

y>0でφ(y)=(y−e)²/yと置く。φはy≥eで増加、0<y≤eで減少する。|y−e|>dの上側ではφ(y)≥d²/(e+d)。下側eventがあるd<eではφ(y)≥d²/(e−d)>d²/(e+d)。d≥eなら下側eventは空である。

従って、任意d>0について

\[
\boxed{
\Pr(|Y-e|>d)
\le\min\left\{1,\frac{\eta^2 e(e+d)}{(1-\eta)d^2}\right\}
\le\min\left\{1,\frac{\eta^2 W(W+d)}{(1-\eta)d^2}\right\}.
}
\tag{EW2}
\]

元のEWの両側化とは異なり、d≥ηeを別途要求しない。全域で元EWより小さいとは主張しない。ηが十分小さい固定e,dで、二次依存の上界になる点が改善である。使える上界を両方保持すればよい。

e=0なら全Fjがgroundを消し、S2は恒等作用、Kも0なのでY=0。逆を使わず自明caseで処理する。W=0、Λ=0も別に処理する。

### 6.3 真値なしの十分step条件

W,d>0、0<α_spec<1について

\[
c=\frac{\alpha_{\rm spec}d^2}{W(W+d)},\qquad
\eta^2/(1-\eta)\le c
\]

なら(EW2)の失敗確率はα_spec以下。等価な十分条件は

\[
\boxed{\eta\le\eta_*:=\frac{\sqrt{c^2+4c}-c}{2}<1.}
\tag{11}
\]

η0のみであれば

\[
0<h\le\frac1\Lambda\sqrt{\frac{\eta_*}{32+16\eta_*}}
\tag{12}
\]

と選べる。η1/min保証でも同じ(11)へ戻せる。これらは記号的な仕様であり、登録済みの数値精度や実行契約ではない。

### 6.4 成功確率の意味

d≪Wでの小ηでは、元EWの十分条件はηがおよそα_spec d/(2W)以下、EW2はおよそ√α_spec d/W以下となる。したがってsingle-run設計のα依存は改善し得るが、これをconfidence全体の最適な改善とは呼ばない。

独立したground copiesから一定の成功確率>1/2のenergy推定を繰り返し、中央値で増幅する経路は両方式に使える。反復数のlog(1/α)依存、準備・読出し価格を両対照に共通に許す必要がある。共通のcertificateが誤っているリスクや系統biasは、単なる反復で消えない。

## 7. QPEの出力保証と費用の接続

[R1]のbranch処理を保持する。h>0、η<1、hΛ<1/4なら0≤K≤(1+η)ΛI、h||K||<1/2。c0=(1+η)Λ/2を使ってK−c0を読む等の既知scalar処理で、0/2πの境界を避けられる。

各固有値に対して

\[
\Pr(|\widehat Y-Y|>\varepsilon_{\rm read})\le\alpha_{\rm read}
\]

を保証する同一unitaryのpowersを使えば、EWまたはEW2とのunion boundにより

\[
\Pr(|\widehat E-E_0|>d+\varepsilon_{\rm read})
\le\alpha_{\rm spec}+\alpha_{\rm read},\qquad\widehat E=L+\widehat Y.
\tag{13}
\]

同一試行内の独立性は不要。既知のstate trace-distance、全controlled powersの実装による出力分布誤差、certification failureは別項目として会計する。PFモデルと元integral Hの差も消さない。

これはQPEスペクトル出力についての命題で、平均複素信号やM1-S、各roundで別kernelを使うRPEへの適用ではない。状態準備方法の研究へ自動的にscopeを広げない。

一定の読出し成功率でのbase-S2適用数はO(1/(h ε_read))。controlled U^(2^k)を一回とだけ数えて、その中のS2反復を省略しない。回転合成精度のh・query依存と前処理はnative費へ別途戻す。

## 8. 強い既知対照を同じenergy-windowへ戻す

**この節も、既知operator上界[P4]を出発点にした今回の紙上接続である。先行論文が同じ表記で本節全体を述べたという引用ではない。**

### 8.1 S2交換子上界

[P4] §5.1 Eqs.(150)–(152)の順序を[R4]へ合わせると

\[
C_2=\sum_{j<M}\left\{
\frac{\|[F_j,[F_j,R_j]]\|}{24}
+\frac{\|[R_j,[R_j,F_j]]\|}{12}\right\},
\quad\|S_2(h)-e^{-ihG}\|\le |h|^3 C_2.
\tag{14}
\]

実際のalgorithmには取得可能な認証上界\(\overline C_2\ge C_2\)を与える。高価なexact many-body normの無料取得は仮定しない。この実時間operator不等式自体は本稿のCauchy半径を要しない。

### 8.2 同じlog領域でのgenerator上界

x=|h|Λ<1/4、A=S2(h)−I、B=e^{-ihG}−I、q=e^x−1<1とする。双方norm≤q。非可換なべきのtelescopingから

\[
\|A^n-B^n\|\le n q^{n-1}\|A-B\|.
\]

log級数を足して

\[
\|\operatorname{Log}(I+A)-\operatorname{Log}(I+B)\|
\le\frac{\|A-B\|}{1-q}.
\]

同じbranchでLog(e^{-ihG})=−ihGなので

\[
\boxed{\|K-G\|\le b(h):=
\frac{h^2\overline C_2}{2-e^{|h|\Lambda}}
\le\frac32h^2\overline C_2.}
\tag{15}
\]

1/4円を使ったe^x≤4/3による安全な定数であり、最適な逆log定数とは主張しない。

### 8.3 gap不要の出力確率対照

ground g上では(K−e)g=(K−G)g。従って

\[
\mathbb E(Y-e)^2=\|(K-G)g\|^2\le b(h)^2,
\quad
\Pr(|Y-e|>d)\le\min\{1,b(h)^2/d^2\}.
\tag{16}
\]

十分条件は

\[
\frac32h^2\overline C_2\le\sqrt{\alpha_{\rm spec}}\,d.
\tag{17}
\]

この対照も励起gap・単一PF固有枝の高overlapを要求しない。したがって、それらを不要にしたことだけでRFを優位と判定できない。

この比較は共通小h領域での一つの強い基準である。既知operator保証にはより大きいhの領域、直接のQPE回路安定性、state残差、低エネルギー専用のより鋭い経路もあり得る。それらを除外して勝利を主張しない。

## 9. 何が小さければ実用上の利点になり得るか

### 9.1 εの指数は差にならない

η(h)=a h²+O(h⁴)と近似できる領域を説明用に取る。RF0ではa=32Λ²、RF1ではa=K_SOS。ただし後者は四次剰余が従属でa>0の場合。a=0や境界近傍をこの漸近式で扱わない。

d, ε_readを総精度εに比例配分し、αを一定、Wと係数を固定した小ε/W領域でEW2を使うと

\[
h_{\rm rel}=\Theta(\sqrt{\varepsilon/(aW)}),\qquad
Q_{\rm rel}=O(\sqrt{aW}\,\varepsilon^{-3/2}).
\tag{18}
\]

(15)–(17)側も

\[
h_{\rm norm}=\Theta(\sqrt{\varepsilon/\overline C_2}),\qquad
Q_{\rm norm}=O(\sqrt{\overline C_2}\,\varepsilon^{-3/2}).
\tag{19}
\]

ゆえに**S2のε^{-3/2}だけではspeedupを主張できない。** 誤差幅・成功率の配分、位相読出し、合成を共通にして比較する。

### 9.2 記号的な比較量

\[
\Xi=\frac{aW}{\overline C_2}
\tag{20}
\]

は上記二つの十分設計を比較するための無次元量である。小h、d≪W、一定αの領域でXiが十分小さければ、RFの証拠がより大きいhを認める可能性がある。これは新しい物理observableでも採択済みのgate閾値でもなく、今回の条件付き比較から出る診断量である。

有限hの判断では(11)、η=min(η0,η1)、hΛ<1/4、(15)–(17)を直接使う必要がある。Xi<1だけを万能なGO基準にしない。\(\overline C_2\)をわざと粗く取り、比較を有利にしてはいけない。局所性が有用なclassでは[P1/P2]も含め、より高次の強いPFまで含めた主張をするならそれらにも同じ比較機会を与える。

**実分子でXiが小さいことは今回確認していない。** 新しい比較値・分子・閾値・native予測は一切計算していない。

### 9.3 量子費用と古典費用

量子資源座標gについては、例えば

\[
C_{\rm quantum}^{(g)}=F^{(g)}+
\sum_a n_a\{Q_aC_{S_2,a}^{(g)}+P_a^{(g)}+R_a^{(g)}\}
\tag{21}
\]

と書ける。aは実際のphase-estimation試行種類、n_aは反復、Q_aはbase-S2適用数、P/Rは準備・読出し。stateを同一試行内で再準備しない場合にQ_a回課金しない。

certificate作成・検証、古典回路生成の時間・memoryは別座標として報告する。換算モデルなしに古典秒とT数を加えない。証拠の償却回数が異なる対照の比較にも注意する。量子gateが減った場合でも、全体の利得をいうにはこれらの費用を説明しなければならない。

## 10. 実用性を制限する具体的な条件

### 10.1 可換limit

すべてのFjが可換ならS2は厳密でC2=0。一方、Λ>0のRF0はh≠0で正の上界を返す。従ってRF0は既知交換子保証を一様に支配しない。RF1のraw係数や四次剰余も、相殺を使わなければ過剰になり得る。

研究価値を高めるには、単に32を少し減らすより、可換性・fragment構造を保つ有限剰余と、その安い証拠生成が重要となる。ただし、既存compact BCHの再実装を新しい手法としない。

### 10.2 大域的な帯域とlog

現RFはΛに基づく小h領域での全vector・単一logを扱う。局所HamiltonianではΛが系サイズとともに増し、この保守性が[P1/P2]の局所性を使う保証より不利になる場合がある。

また、正しいprincipal logを全スペクトルで使うには帯域とphase wrappingの問題が残る。1/4は保守的な証明定数だが、そこから『全system sizeで帯域非依存のprincipal log保証が当然得られる』とは言えない。低エネルギーの別branch設計は別の数学的問題である。

### 10.3 PSD化のshiftに対する敏感さ

FjをFj+a_jI、LをL−Σa_jに置き換え、外側scalarを正しく戻すと、実際の元Hamiltonianとその制御付きPFの作用は不変にできる。一方、Λ、e、Wは変わり、RFの認証値は悪化し得る。

従って、Wの小ささは実際に使うPSD分割の情報でなければならない。無関係なSOS下界の良さをRFへ移さない。shiftだけで回路や真のPF誤差が改善したとも主張しない。

### 10.4 情報・状態条件

L/U/fjの証明、係数一致、roundoffは実際に取得・検証可能でなければならない。元DF誤差も含める。ground copiesが比較設定で利用可能でも、その準備単価や近似誤差は無料ではない。

本命題がexact groundを用いることは、近似state一般に算法が不可能という意味ではない。ただし同じ保証をそのまま移すことはできず、別に取得された品質情報を使う必要がある。gapなしに変分energyだけから任意に高いground fidelityを推定しない。

### 10.5 古典的に終了する場合

認証幅W≤2εなら、[L,U]の中点で目標誤差を満たせる。この領域は『相対保証がよく効く量子優位の例』として数えない。区間を認証する古典費用は依然として残る。

## 11. 高次PF repositoryとの直接重複

[H1]のU7には、許す情報・branch・費用を固定したtruth-freeな幅／保証が資源判断に有効か、という既存課題がある。U1/U3は有限時間剰余・応答・情報取得費にも触れる。したがって今回のA2を『このprojectで初めて誤差幅と資源判断を考える研究』とは呼ばない。

一方、今回読んだ[H1]にはRF0/RF1の有限全vector相対logの同一証明は示されていない。これは選択した資料の記載範囲に関する判断であり、全repositoryで未検討であることの証明ではない。

今回の差は、少数の保存値による点推定や経験的幅ではなく、PSD分割の係数・認証値から有限hの上界を出すこと。負stageの高次PFへ現証明を適用したり、既存のH05/FS/Direction Cを今回の結果で再開したりはしない。関連する将来の構成が重なる場合には既存担当と同じ証拠を再利用し、二重研究として扱わない。

## 12. 研究としての採否判断

### 12.1 限定継続を推奨する理由

今回の対象には、単なる着想でなく、具体的なRF0/RF1の証明、通常QPEまでの入出力条件、真のeを必要としないstep選択がある。更にEW2と同taskの既知対照を記述でき、何が改善しなければ実用差が出ないかを具体化できた。

このため、候補名を変えながら広い探索を続けるより、**『有限相対証拠の強さ・取得可能性・既知定理との正確な対応』に限定して研究する**方が妥当と判断する。

ただし、新アルゴリズムの強い独立性や化学での有利なclassはまだ示されていない。現段階のまま大規模benchmark・native実装へ進むこと、定理の正しさだけで主要テーマへ格上げすることは推奨しない。

### 12.2 主張してよいこと／まだ言えないこと

| 主張 | 判定 |
|---|---|
| 記載PSD/positive S2/small-log条件でRF0/RF1が成立する | 支持 |
| 原groundに対するenergy-windowをgapなしで保証できる | 支持。既知の構成要素を含む |
| 全vector性だけで近接先行研究と本質的に異なる | 不支持 |
| 定数32/512が最適または実用的である | 未確立 |
| 全statementが既知定理の直接系である | 未確定 |
| RFによりS2のε指数が改善する | 不支持。比較式は同じε^{-3/2} |
| 同じ情報でより大きい認証stepを得る余地がある | 条件付きで合理的。実例・強対照優位は未確認 |
| 新しい実用PF算法が完成した | 不支持 |
| 誤差保証は同じ回路だから資源を変えない | 一般には不適切。実行stepが変わり得る |

### 12.3 他の選択肢

成果保存だけで区切る選択も合理的であるが、現時点では具体的な比較問題があるため、ただちに価値なしと判断する根拠は弱い。一方、保証理論を研究範囲に含めないというユーザー判断なら、回路生成研究へ自動的に言い換えて継続しない。

## 13. 次に必要な一まとまりの作業

このレビューの推奨をユーザーが採る場合、次は以下を**一つの限定的な理論・技術比較作業**として扱う。各補助式ごとに新しい承認gateを量産しない。

1. 今回追加したEW2と逆log対照(15)–(17)を独立に紙上照合する。完了したRF0/RF1監査を理由なく全面反復しない。
2. [P1] Theorem S1のrank-one特殊化を含め、有限log remainderまでの定理対応を確認する。既知の道具の名称比較で済ませず、仮定と結論を対応させる。全分野の網羅的surveyは要求しない。
3. 同じ入力・同じ実装分割・同じ精度成功率で、RFと少なくともChilds型、適用可能な低energy型の認証stepを記号的に比較し、証拠取得費も含む具体的な有利条件が残るかを整理する。
4. 差がある場合のみ、既知係数計算法を超える『有限相対剰余／安価な証拠生成』の具体的未解決箇所を一つに絞る。実分子計算・benchmark・新SOURCEはここから自動開始しない。

数学的な誤りが出た場合、影響部分と残る部分を区別してSTOPする。完全な既知還元が判明し、取得・保証の具体差もないなら、既知の基準として保存し新算法としない。未解決という語だけで同じ探索を繰り返さない。

すべて同じ紙上命題の通常照合で済むなら、GPTの広い研究方針レビューを直ちに再開催する必要はない。主要claimを変える欠陥、新しい完全包含、新しい有用な構成が得られた場合に限り、必要な科学判断を特定する。

## 14. 正式状態・実施境界

本書の「限定継続」は推奨である。A2を正式採択したり、既存文書を採択済みへ書き換えたりしていない。A1の現分割生成CLOSED、A2 NOT_ADOPTED、K1-F未承認・未実行、他のSTOP/PAUSED/HOLDは維持する。[S0]

①PRリソース評価やH8 Phase Cを今回の根拠へ取り込んでいない。②/③、過去encoding、high-order PFの実験・契約・判定も変更していない。

新規科学数値計算0、科学SOURCE実装0、symbolic checker0、native compile0、GitHub書込み0、新しい実行authorization/grant/marker0。今回行った新しい数学的な紙上導出は§6と§8–9、および既知S1からのrank-one対応である。原本を改訂済み・Codex認証済みと偽っていない。

**レビュー完了・mandatory STOP。** 研究方針採択と、将来の新規科学計算の実行承認は別である。

## 15. 固定資料・参考文献

### Repository資料

- [R1] [数学的監査、051df466…](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/051df4661eeffb6c6c4c978f70f43eeeaadfddd6/docs/research/qc_relative_form_finite_step_audit_20261017/mathematical_audit.md)。RF0/RF1/EW/QPEの中心節を読解、長い出力は分割取得。
- [R2] [先行研究・実用条件、同commit](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/051df4661eeffb6c6c4c978f70f43eeeaadfddd6/docs/research/qc_relative_form_finite_step_audit_20261017/prior_art_and_practical_conditions.md)。既知性区分と条件・費用を確認。全引用原本の再監査ではない。
- [R3] [判断用ハンドオフ、同commit](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/051df4661eeffb6c6c4c978f70f43eeeaadfddd6/docs/research/qc_relative_form_finite_step_audit_20261017/decision_handoff.md)。全文確認。
- [R4] [K0 technical_audit §4–5、58d86727…](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/58d8672702284d73321cc67a26c4640adaddde5f/docs/research/qc_sos_pf_k0_20261015/technical_audit.md)。今回直接確認した主範囲は係数・行和・証拠条件。
- [R5] [K0.5 technical_report §3–4、01bdc56a…](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/01bdc56a2970dcb749390b0540846143f775d6f9/docs/research/qc_sos_pf_k05_20261015/technical_report.md)。逆log・QPE・残差・C2・low-energy条件。
- [O1] [元レビューの保存copy、051df466…](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/051df4661eeffb6c6c4c978f70f43eeeaadfddd6/docs/research/qc_relative_form_finite_step_audit_20261017/inputs/qc_track_iv_research_question_redesign_review_20261017.md)。会話で渡された同名Markdownも使用。Codex記録の原本は38,606 bytes、SHA256 `954bd40d1da7dce70a4b97336c71d1cd55031314df9c339a2de95b82fabf8527`。
- [H1] [高次PFの未解決問題・証拠map、b4a7a820…](https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/b4a7a8204b8d7fa0a2a1bc79bfbdd7cf63177ca1/docs/research_audits/pf_finite_time_completion_20261016/open_question_and_evidence_map.md)。U1/U3/U7と直接の重複。全repository最新性の監査ではない。
- [S0] [正式状態decision、67576526…](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/67576526c4e96ae7235fa645fdf744278d049f8d/docs/research/qc_exact_encoding_direction_adoption_20261017/decision.json)。状態は[R3]でも再掲されている。
- 共通運用指示：会話Project原本 `research_project_instructions_unified_2026-10-17.md`。実際に確認し、旧版ではなく本版を適用した。

### 外部一次資料

- [P1] Kaoru Mizuta, Tomotaka Kuwahara, *Trotterization is substantially efficient for low-energy states*, [arXiv:2504.20746v1](https://arxiv.org/html/2504.20746v1), Supplement S2 Theorem S1 Eq.(S5)と関連証明。登録ページはPhys. Rev. Lett. 135, 130602 (2025)を対応づけるが、出版版全文の同一性監査は行っていない。
- [P2] Kasra Hejazi, Modjtaba Shokrian Zini, Juan Miguel Arrazola, *Better bounds for low-energy product formulas*, [arXiv:2402.10362v1](https://arxiv.org/html/2402.10362v1)。設定§I、low-energy誤差§IIと関連補足。
- [P3] William Maxwellほか, *Practical Estimation of Trotter Error for Hamiltonian Simulation*, [arXiv:2606.30738v1](https://arxiv.org/html/2606.30738v1)。III.A/B・Appendix Aの必要部分。漸近定理をfinite uniform relative-log定理としない。
- [P4] Andrew M. Childs, Yuan Su, Minh C. Tran, Nathan Wiebe, Shuchen Zhu, *A Theory of Trotter Error*, [arXiv:1912.08854v3](https://arxiv.org/pdf/1912.08854v3)。§5.1 Eqs.(150)–(152)、Proposition 16。arXiv登録はPhys. Rev. X 11, 011020 (2021)へ対応。本文p.39を画像確認。
- [P5] Ren-Cang Li, *Relative Perturbation Theory: II. Eigenspace and Singular Subspace Variations*, [DOI:10.1137/S0895479896298506](https://doi.org/10.1137/S0895479896298506)。出版社の要旨により相対摂動の既存分野を確認した補助資料。本文定理を取得してEW2の直接出典としたものではない。
- [P6] Changhao Yi, Elizabeth Crosson, *Spectral Analysis of Product Formulas for Quantum Simulation*, [arXiv:2102.12655](https://arxiv.org/abs/2102.12655)。今回の直接取得は登録・要旨中心。詳細なgap/branchの引用は[R5]/[H1]の既存記録と区別し、本文全読了とはしない。

取得範囲外の論文や過去Claude原本を読了済みとしていない。掲載版・後続版の全同一性、全定理からの還元、EW2のpriorityは未確定である。独立した科学的研究記録として本書を新規保存し、原レビュー・監査を上書きしない。
