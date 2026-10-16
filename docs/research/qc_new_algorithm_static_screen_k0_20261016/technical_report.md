# 新アルゴリズム静的K0：交換子・混合相関・回路生成の限定技術監査

2026-10-16 JST。一回限りの静的技術調査。資料索引：[文献対応表](literature_correspondence.md)／[取得・入力identity](provenance.json)。ユーザー指定の完成GPTレビューの§4–9・11を検証対象として扱い、正しいとは仮定しなかった。数学は以下の紙上導出であり、数値行列・科学コード・symbolic checkerの実行による検査ではない。

## 1. 到達点と判定の適用範囲

**総合技術判定：`NO_GO_NO_INDEPENDENT_GENERATION_RULE`。今回のH1–H4の記述から、既知法と区別できる具体的な新回路生成規則と情報込み資源改善を同時に特定できなかった。** これは今回の有限なscreenの判定であり、D1・D2内の全算法の不能性、研究全体の終了、正式PAUSE/CLOSEDの追加判断ではない。

| 方向 | 今回の技術分類 | 確認したこと／残る限界 |
| --- | --- | --- |
| H1：低体数化とPF共同設計 | `NO_GO_NO_INDEPENDENT_GENERATION_RULE` | 二分割の式は成立し既存監査にも記録済み。固有状態のH交換子消去によるPF設計も既報。安い指数化を保つ独立生成則は今回未構成 |
| H2：混合スペクトル推定 | `UNDETERMINED_DATA_REQUIRED` | Hadamard accessは合法。一般filterにも成立する混合恒等式を整理したが、small denominator・測定・発展価格込みで全H QPEと直接測定の双方を上回る根拠は未取得。新規性・費用をunknownのまま停止 |
| H3：Gaussian時間依存DF | `NO_GO_NO_INDEPENDENT_GENERATION_RULE` | 回転した各平方の指数化は既知のGaussian共役で生成可能。THRIFT/CFQM/DF回路と異なる新primitiveは未構成 |
| H4：制御なし絶対energy | `NO_GO_NO_INDEPENDENT_GENERATION_RULE` | Clinton/Kujiの既知accessを確認。コピーだけでは供給されない追加信号・準備/逆準備・既知基準が必要。独立した安価な原点校正規則は未構成 |

H2の未確定は次の数値実行の認可ではない。他方向についても「未特定」を一般的不存在の証明へ格上げしない。`STATIC_MECHANISM_IDENTIFIED_REVIEW_REQUIRED`に該当する方向はない。

D1・D2を維持する。元の有限基底電子Hamiltonianの絶対E₀、精度ε・失敗確率αが最終task。厳密量子ground状態のコピーは利用可能だが、コピー・準備・測定を価格に戻し、古典E₀/vector/RDM/gapを無料入力にしない。block-encoding/qubitization/QSVT/量子walkを中心とする新構成は作らない。

## 2. K0-1：二分割の符号・順序・低体数化

有限次元の共通不変sectorでA,BをHermitian、H=A+Bとし、積を

\[
S_A(h)=e^{-ihA/2}e^{-ihB}e^{-ihA/2}
       =\exp[-ih(H+h^2D+O(h^4))]
\]

と定義する。抽象変数X,Yについて対称BCHの三次項は

\[
\log(e^{X/2}e^Ye^{X/2})
=X+Y-\tfrac1{24}[X,[X,Y]]-\tfrac1{12}[Y,[X,Y]]+O((X,Y)^5).
\]

X=−ihA,Y=−ihBを代入すると、三重積の係数は(−i)³=iであるため

\[
D=\tfrac1{24}[A,[A,B]]+\tfrac1{12}[B,[A,B]].\tag{1}
\]

H|0⟩=E₀|0⟩なら任意Xに対し⟨[H,X]⟩₀=0。C=[A,B]として

\[
D=-\tfrac1{24}[A,[A,B]]+\tfrac1{12}[H,[A,B]],
\qquad d=\langle D\rangle_0=-\tfrac1{24}\langle[A,[A,B]]\rangle_0.\tag{2}
\]

さらに[A,[A,H]]=A²H−2AHA+HA²より

\[
d=\tfrac1{12}\langle A(H-E_0)A\rangle_0
 =\tfrac1{12}\langle\delta A(H-E_0)\delta A\rangle_0\ge0,
\quad\delta A=A-\langle A\rangle_0I.\tag{3}
\]

不等号には|0⟩が当該sectorのgroundでありAもsectorを保存する条件が要る。E₀は導出中の記号で、設計入力ではない。式(2)の測定にE₀や⟨A⟩の事前取得は不要。

一体A=dΓ(a)と高々二体Bについて、CARから一体との交換子は体数を増やさない。従って[A,B]と[A,[A,B]]は高々二体で、dは一体/二体期待値の線形結合で取得可能。完全2-RDMの取得は必須ではなく、必要な収縮・対応observableだけを測る選択もある。ただし測定価格は消えない。

ここまでの恒等式は縮退・gap情報なしでも成立する。dを一つのPF固有値のh²係数と読むには、同じlog branchの孤立した非縮退固有値を追う摂動条件が別に要る。縮退groundではPD Pの固有値を用い、任意の一コピーの⟨D⟩を全branchのshiftとしない。ABAのA/Bを交換すれば式中のA/Bも交換する。非負dから一般PFの誤差相殺不能を結論しない。

**既存との差**：式(1)–(3)はSOS-PF K0の二分割評価、および一体再配分事前監査§2に既にある（provenance G3/G6）。今回追加したのは順序・符号・情報層の明示的な再確認で、新しい数学的原理ではない。

### 多fragmentで消せるもの

順序付きF₀,…,Fₗについて、外側F₀・中央Fₗの対称積S=e^(−ihF₀/2)…e^(−ihFₗ)…e^(−ihF₀/2)を取る。`R_i = sum_{j>i} F_j`、`L_i = sum_{j<i} F_j`と書き、対称BCHを外側から再帰すると

\[
D=\sum_{i=0}^{\ell-1}\left\{
\tfrac1{24}[F_i,[F_i,R_i]]+\tfrac1{12}[R_i,[F_i,R_i]]\right\}.
\]

内部PFの三次logはこの次数では加算され、その外側との交差はより高次になる。Rᵢ=H−Lᵢ−Fᵢを代入して外側H交換子を消すと

\[
d=-\sum_{i=0}^{\ell-1}\left\{
\tfrac1{24}\langle[F_i,[F_i,R_i]]\rangle_0
+\tfrac1{12}\langle[L_i,[F_i,R_i]]\rangle_0\right\}.\tag{4}
\]

例としてA（一体）,B,C（二体）の三fragmentでは

\[
d=-\tfrac1{24}\langle[A,[A,B+C]]\rangle_0
  -\tfrac1{24}\langle[B,[B,C]]\rangle_0
  -\tfrac1{12}\langle[A,[B,C]]\rangle_0.\tag{5}
\]

第一項は高々二体だが、後二項の形式的上限は四体・三体である。式(4)のH交換子消去だけで全項が低体数になるとは言えない。この表示に高体項が残ることだけを、追加CAR/対称性を含む全還元の不可能性や「4-RDM取得が必須」の証明にもしない。

| 低体数化の十分条件・既存構造 | 期待値と回路の境界 |
| --- | --- |
| 二分割で外側が一体、相互作用Bを一括指数化 | dは二体以下。ただし一般のB全体のexpは安くない。内部PFでBを実装すれば、その誤差項を再び加える |
| B=ΣBⱼで[Bⱼ,Bₖ]=0、各expが既知 | 内部積がexactなので同じ二分割還元が使える。共通density frameのGivens＋diagonal phaseは既知構成 |
| D=Q_{≤2}+[H,X]が安価に構成できる（sector内等式） | ⟨D⟩=⟨Q⟩は成立。Xの形式的存在だけでは生成・測定・指数化価格を与えない。state固有の相殺も別条件 |
| 一体再配分の先頭pairで移送方向Xが[X,B]=0を満たす | 特定の**変化分**が1-RDMへ落ちる既存G6 §3.1。多fragmentの絶対d全体が1-RDMで決まる意味ではない |

一般分子の非可換DF平方を全て共通frameに置ける、または式(4)の全残余を安く二体へ閉じられるという追加条件は、今回得ていない。

## 3. K0-2：H1のreduction testと費用

[Hejazi v1 §I Eq.9・§II Eqs.10–13](https://arxiv.org/html/2412.16811v1)は、外側H交換子の固有状態期待値が消えることを**PF係数生成条件に用いる**。両blockのfast-forwardabilityを仮定し、5指数kernelとその対称化による9指数のenergy誤差改善を与える。従って「H交換子同値関係で固有値誤差を設計する」という中核は既報である。これはH1の全てのnative分割共同設計まで既報だと証明したものではない。

[CPF v4 Eqs.4–5,26–27・§5](https://arxiv.org/html/2409.08265v4)では共役processorとkernelを変えるcorrectorを区別する。exact P S P⁻¹は固有値を変えないため、operator評価の改善をそのまま固有位相改善と数えない。kernel内correctorは別で、実装・合成誤差を会計する。既存H-E1に記録されたEq.28の同符号sandwich／逆符号共役の表示不整合は保持し、その式を単独の保証根拠にしない。今回新しい訂正文の発見やCPF全体の無効性を主張しない。

[SPRINT/GRADE v1 §II–III・V.3・Appendix E.3](https://arxiv.org/html/2606.30741v1)は分割表現、native費用、leading固有値誤差、near-integrability/processingを既に結び付ける。今回のD2での有利な有限step保証を供給するとの一般化はしない。D1に属する補助方式を新構成へ採用しない。

repository内ではG5 H-E1が情報取得＋PF共同設計、G6が低体数の局所応答、G8が構造保存counterterm／射影を既に監査し、G3/G4がSOS・保証・費用の境界を整理している。今回、係数変更・再配置・既知corrector・scalar後処理・外挿を越える入力→出力回路規則は得られなかった。この時点でH1の追加探索を止める。

費用について、同じtask・同じ資源座標で一般に必要なのは

\[
C_{\rm total}=C_{\rm info}+N_{\rm copy}C_{\rm prep}
+\sum_r N_r\lceil t_r/h\rceil C_{\rm step}(h,\epsilon_{\rm synth})
+C_{\rm read}+C_{\rm classical}.
\tag{6}
\]

例として実際に有効な有限h保証|bias(h)|≤|d|h²+K₄h⁴があれば、その不等式とbranch/overlap条件でh_maxを制限する。leading dだけでh_maxを定めない。固定t・他の会計が同じ場合に初めてC_step/h_maxが比較軸になる。G3/G4のgap・剰余の未確定を消してはいない。

Q_{≤2}=ΣqⱼQⱼをPauli標本平均で測る単純な十分会計は、λ_Q=Σ|qⱼ|としてN_info=O(λ_Q²η_d⁻² log(1/α_info))。scalar補正のh²η_dをε_infoに収めるにはη_d≤ε_info/h²が必要で、N_infoはこの実現ではO(λ_Q²h⁴ε_info⁻² log(1/α_info))になる。これは全測定法の最適下界ではない。2-RDM化・主要係数減少だけから式(6)の減少は導けず、同情報を使う強い既知法にも同じ測定・cachingを許す。

## 4. K0-3：H2の合法access・有限filter・情報価格

H=H_c+V、V=ΣvⱼQⱼ、λ_V=Σ|vⱼ|とする。各QⱼがPauli unitaryなら、与えられたgroundコピーとancillaに対するcontrolled(U_c(t)Qⱼ)のX/Y Hadamard testで⟨U_c(t)Qⱼ⟩を測れる。Qⱼを先に作用させ、その後U_cを作用させる順序と相対scalar phaseを保持する。Vをcoherent LCUへ埋め込む必要はない。Pauli単独がsectorを出る場合、sector内だけの古典参照APIをそのまま量子accessと見なさない。

H_c=ΣEₖᶜPₖ、pₖ=⟨Pₖ⟩、aₖ=⟨PₖV⟩とおくと

\[
a_k=(E_0-E_k^c)p_k,\qquad
C(t)=\sum_k e^{-itE_k^c}a_k,\quad P(t)=\sum_k e^{-itE_k^c}p_k.
\tag{7}
\]

式(7)の右作用H|0⟩=E₀|0⟩だけでaₖの関係は成立し、pₖ>0ならE₀=Eₖᶜ+aₖ/pₖ。**これは可測性の証明で、資源優位の証明ではない。**

### 有限filterをprojectorと同一視しない

実際の古典filterをF(H_c)=ΣₗwₗU_c(tₗ)、f(E)=Σₗwₗe^{-itₗE}、W=Σ|wₗ|とする。b=⟨F⟩、a=⟨FV⟩に対し、単一Eₖᶜを使う推定の決定論biasはexactに

\[
E_k^c+a/b-E_0
=\frac{\sum_j f(E_j^c)(E_k^c-E_j^c)p_j}{b}.\tag{8}
\]

projectorへの漏れ・energy近似・alias・PF実装を無視できない。T=max|tₗ|なら、紙上の三角不等式より|f(E)−f(E′)|≤WT|E−E′|。従って指定のfilter contrastには対応するWTとスペクトル分離の価格が必要。これは固定Fourier filter族の制約で、全スペクトル推定法への一律の時間下界ではない。

他方、**spectral分離が混合恒等式そのものに必須というわけでもない**。m=⟨FH_c⟩を同じFで測れば、任意のF（b≠0）について

\[
E_0=(a+m)/b,\qquad C(t)=E_0P(t)-iP'(t).\tag{9}
\]

前者は⟨F(H_c+V)⟩=⟨FH⟩=E₀⟨F⟩から、後者はP′=−i⟨U_cH_c⟩から従う。同じ実装Fを一貫して使い元Hを測る前者では、F自体の近似だけでは恒等式を壊さない。ただしmの測定又は微分推定価格、実際に測るHの誤差は戻す必要がある。F=Iでは単に⟨H⟩=E₀へ戻る。式(9)は一般のmixed-estimator恒等式であり、新算法とは分類しない。

### 誤差と価格

|b|=p>0、r=a/bとし、推定誤差が|â−a|≤η_a、|b̂−b|≤η_b<pなら

\[
|\hat a/\hat b-a/b|\le
\frac{\eta_a+|r|\eta_b}{p-\eta_b}.\tag{10}
\]

Pauli/timeを係数絶対値で抽出する単純Hadamard標本平均ではaの標本rangeはWλ_V、bはW以下にできる。従って十分コピー数はN_a=O(W²λ_V²η_a⁻² log(1/α_a))、N_b=O(W²η_b⁻² log(1/α_b))。通常のη_a=O(pε)配分ではN_aにp⁻²ε⁻²が現れる。これはこのsampling実現の十分会計であり、あらゆる量子法のε⁻²下界ではない。

各shotにgroundコピー/準備、controlled U_c(tₗ)、controlled Qⱼ、ancilla/読出しを計上し、古典filter・周波数/branch選択も価格に戻す。||Ũ_c(t)−U_c(t)||≤δ_U(t)ならa,bの追加biasはそれぞれλ_VΣ|wₗ|δ_U(tₗ)、Σ|wₗ|δ_U(tₗ)以下で、式(10)に入る。式(9)を使う場合はH_c又は元Hの測定価格も加える。

強い関連例として[Cohn–Motta–ParrishのQFD＋DF v1、Eqs.32–38・Fig.3](https://arxiv.org/pdf/2104.08957v1)には、time-propagated基底のoverlapと元Hのmatrix elementをHadamard回路で取得する枠組みがある。一般化固有値問題を解くQFDと、exact ground上の式(7)の単一比を同一算法だとは断定しない。ただし「安い近似発展＋高精度H情報の測定＋古典処理」という型だけを未踏と扱えない。

比較には同コピー価格・ε・αで、全Hを用いるPF/QPEの時間・step・overlap・合成費用と、⟨H⟩直接測定の分散/群価格を戻す。λ_Vが小さくH_cが安い等の条件で局所利益が出る可能性は否定しないが、p/filter/時間別費用・取得可能な保証を含むその条件は今回の資料で閉じなかった。新しい数値を要求して継続せず、未取得として停止する。

## 5. K0-4：H3/H4の最小確認

**H3**。H_c=dΓ(h)、Fₗ=λₗdΓ(gₗ)²なら、CARの一体閉包により

\[
F_{\ell,I}(t)=\lambda_\ell d\Gamma(e^{iht}g_\ell e^{-iht})^2,
\quad e^{-isF_{\ell,I}(t)}
=e^{itH_c}e^{-isF_\ell}e^{-itH_c}.\tag{11}
\]

gₗ(t)の固有値は保存され、既知の対角化とframe回転を使えるので、各時刻の再対角化は数学的に必須ではない。ただしGaussian変換・隣接fusion・control・合成価格は残る。この共役生成は既知interaction-picture原理である。[THRIFT出版本文 Results Eqs.2–3](https://www.nature.com/articles/s41467-025-57580-5)はまさに時間依存の回転相互作用を利用し、[CFQM v3 §2.2 Eq.18・§3.4](https://arxiv.org/html/2403.13889v3)は時刻ごとのHamiltonianの和の指数をさらにPFで実装する。異なるframeの平方の和は一般に一つの平方ではないので、その指数を一primitiveとして無料に数えられない。既存native DF sourceへの接続はG9、旧THRIFT終了はG2の保存範囲を維持。新primitiveが得られなかったため深掘りしない。

**H4**。[Clinton v2 §2.1・Appendix A/B](https://arxiv.org/pdf/2410.21517v2)は、複数信号・干渉信号又は追加dynamics、support条件、準備/逆準備を使う。Appendix AのTrivial Ambiguitiesにはlinear phaseによるスペクトルの定数shiftが明記されている。追加信号から位相を復元できることと、無償に絶対energy原点が決まることを同一視しない。v2と2026出版版の全文差分は今回未照合。

独立に、無条件channelはH→H+cIで変わらず、UρU†と全測定統計は同一である。これはchannel-only情報モデルの識別不能性であって、係数・既知reference・測定accessを持つD2全般のno-goではない。既知のidentity係数を加算するだけで、未校正の復元スペクトルのshiftが常に解決するわけでもない。

[Kuji v2 §II・Appendix B](https://arxiv.org/html/2412.19590v2)は既知energyのreferenceとgroundのcoherent superposition、ASP/逆ASPを用い、energy差へreference energyを加えて絶対値を求める。これは合法な既知方式であり、コピーだけの契約からその準備unitary・逆演算を無料に得るわけではない。化学への原点校正、準備時間、追加信号数、noise/failureの保証と全コピー価格を明示した独立規則は今回得ていない。H3/H4の候補追加・実装へ進まない。

## 6. 証拠・保全・GPTへ戻す条件

GitHub固定資料13件をContents APIから原byteで取得し、blob SHA・SHA-256とローカル固定objectがある場合の一致を記録した。A3 `dbe51f…`、統合 `78b50c…`、K0 `58d867…`、K0.5 `01bdc5…`、H-E1/小改変監査、両repositoryの資料を含む。具体的commit/pathと取得状況は[provenance](provenance.json)。外部一次文献は[別表](literature_correspondence.md)の版・関連節に限定し、全証明・全出版版の同一性を審査したとはしない。

完成GPTレビューはユーザー提供の40,606 bytes、SHA-256 `b84e0613577c5e8d385654092f49176b6ba1a68e4479018a234c5841043dff50`。本文を改変・再公開しない。同レビューが論じたClaude最新版`レビューoutter(3).md`原本は今回未取得で、逐語比較済みとは記さない。K0/K0.5当時のClaude原本未取得、ACM出版版403/差分未確認という履歴も変更しない。

未取得なのは、H1の新生成規則と有用な有限h/情報保証、H2の取得可能なp下界・filter/branch条件・全費用、H3の既知共役を越えるprimitive、H4の現copy契約に束縛された安価な校正・追加accessである。必要な将来検査を行うなら、まずこの入力・比較契約と目的を別途限定する必要がある。今回、分子・grid・計算条件・新実行計画は作らず、未知を埋める数値も実行しない。

新たなLevel 1判断が必要になるのは、独立した具体生成規則・強い既報との差・情報込み資源改善式が揃う、又は既存主要claimを変える重要な数学的修正が見つかる場合。本調査ではそこに達しておらず、調査完了だけを理由に新しいGPT/Claude詳細レビューを要求・開始しない。正式な研究採否と今後の方向はユーザー/GPT側に残す。

新規Hamiltonian、固有値/期待値/RDM/gap、保存値再集計/ランキング、PF simulation、回路生成/compile/T合成、sampling、最適化/SDP、科学コード/benchmark/数値テストは**全て0**。既存source/results/reviews/decision/method status/authorization/manifest/STOPを変更しない。新規文書だけを専用branchへ公開し、remote identityと変更pathを照合する。科学的再検証のPASSとは呼ばない。

**Mandatory STOP。** K0.5/K1、新計算・実装、停止済み系列の再開、新算法の正式採択へ自動移行しない。SOS-PF現探索CLOSED、K0 `UNDETERMINED_DATA_REQUIRED`、A2 `NOT_ADOPTED`、K1-F未認可/未実施、および他のCLOSED/STOP/PAUSED/HOLDを保持する。
