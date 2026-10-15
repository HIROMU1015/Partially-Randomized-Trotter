# Track B／M1-S：S0修正理論契約の正式採択に向けたGPT独立科学レビュー

- **レビュー日**：2026-10-16（JST）
- **担当**：GPT-6
- **レビュー対象**：M1-S／G0-T1／第三案A′のS0修正理論契約候補。固定版は `0d84c2181de128e496bdc1e6161264cdf84eeba1`。
- **レビュー開始権限**：ユーザーによる「レビューを開始して　丁寧に考えて」という明示承認。**新しいG0-T1の定理研究、数値検証、回路生成・compile、実験、研究契約のユーザーに代わる正式採択は含まれない**。
- **レビュー種別**：Level 2（重要な理論契約・保証意味論の科学的判断）。M1-S全体を別方向へ転換するLevel 1の再審議ではない。
- **状態**：`GPT_S0_SCIENTIFIC_REVIEW_COMPLETED_RECOMMEND_ADOPTION_WITH_EXPLICIT_SELECTIONS_USER_ADOPTION_PENDING_STOP`
- **既存状態**：Parent allocation D＋Eは採択済み。M1-S、G0-T1の正式契約、T1a/T1bの新証明、Gate 0科学計算は未採択・未承認。**mandatory STOPを維持する**。

---

## 0．総合判断

### 0.1 最終推奨

**結論：S0修正契約は科学的に妥当な候補であり、以下に明記する対象選択と受理条件を組み込んだ版の「限定的な正式採択」を推奨する。ただし、公開済みS0文書を未決定欄が残ったまま、無条件で最終契約とすることは推奨しない。**

推奨する研究の流れは、以前のClaude／GPTレビューと同じく **A′：T1a → Checkpoint 1 → T1b → Checkpoint 2 → 必要ならGate 0** である。T1aは、既存の平均化・摂動論から過度に広い一般定理を作ることではなく、**非可換な分岐相関coherent信号の、誤差付き・長時間一様な保証**を得て、T1bで**部分乱択へどこまで移せるか**を判断するための補題と位置付ける。

今回のレビューが提案する**採択時の具体的選択**は次のとおり。

| 科学的決定事項 | 今回の推奨 | 証拠の地位 |
|---|---|---|
| 初期研究範囲 | **G0-T1のみ**。T1a：Q1-core＋Q2-min、T1b：Q3-core | 研究上の推奨。ユーザー正式採択待ち |
| 理論順序 | **A′**、T1a後とT1b後に独立した判断点を置く | 以前の暫定承認を支持 |
| T1a主比較モデル | **C-sec：二次のsecular平均化モデル**。厳密な \(\Phi_\theta\) を比較正本にする | 科学的推奨。誤差定理は未証明 |
| C-logの位置付け | **補助的な証明・検算手法**。完全なexact-logそのものを研究成果の代わりにしない | 科学的推奨 |
| 最初の入力class | **固定Hamiltonianの孤立した対象固有状態 \(|k\rangle\)** を中心にする。一般guiding state・RPEは後段 | 科学的推奨 |
| T1a誤差目標 | 複素信号の**絶対誤差**。非共鳴・近共鳴・pair-sum・局所高次累積を分離。位相誤差はsignal floor付きの系 | 新定理ではなく証明義務 |
| T1b参照protocol | exact deterministic half＋中央の**Pauli/involution-tail**、同位置label共有、IIDの方向制御± | 指定順序での平均写像は既存の厳密代数 |
| T1b最初のjoint scaling | **固定 \(r\ge1\)、\(\delta_{\rm rel}\to0\)、\(s\delta_{\rm rel}^2\) を有限範囲で管理**する長時間・小step系列を主対象にする | 科学的推奨。移送保証は未証明 |
| \(\rho_\infty^{(H)}\) | T1bで**成立／修正／不適用**を判定する未証明候補。Gate 0のGO条件にはまだしない | 未検証仮説 |
| 初期scopeからの除外 | 一般DF event、PF内部実装、完全RPE保証（Q4）、native費用・forecast（Q5）、科学数値計算 | 推奨 |
| T1a実施 | **今回は承認されていない**。ユーザーの研究契約正式採択と**別途の理論実施承認**が必要 | 権限上の境界 |

**確信度**：A′とG0-T1の限定scopeが適切であることは**中〜高**。S0の数学的意味論が正しいことは**高**（下記の定義条件の下）。T1aの非自明な一様保証の成立は**中**、T1bの有用な移送の成立は**中〜低**、量子化学のend-to-end資源優位は**低〜未評価**。

**進行上の重要な区別**：これは「本レビューによりG0-T1が正式採択された」という報告ではない。ユーザーが本文の最終推奨を正式採択して初めて契約が確定する。S0の機械的反映をCodexへ依頼することと、T1aの新規研究を許可することも別である。

### 0.2 採択に必須の科学的選択が残る理由

[今回のS0候補][S0]は過去の重大な留保を十分に反映している。とくに、(i) \(O(D|\theta|)\) 級の局所誤差、(ii) \(\Psi\)のexact-half因数分解、(iii) dressingの絶対誤差と相対誤差の非同値、(iv) \(\theta,\delta,r,s\) のjoint scaling、(v) Checkpoint 1/2の非自動進行、の各項が、既に候補文書へ取り込まれたことは確認できる。

しかしS0候補の§4.2には**C-sec/C-logが未選択**と書かれ、§8には**どのjoint pathで保証を主張するかが未選択**と明記されている。これらは単なる文書の未記入ではなく、**何を証明すれば合格か、何が反例となるか**を変える実質的な科学的選択である。したがってCodexへ「どちらでもよい」と任せてT1aを実施するのは不適切である。一方、今回のLevel 2レビューで選択方針を具体化できるため、同じ全面レビューをもう一度繰り返す必要もない。

---

## 1．正本、固定版、直接取得できた証拠

### 1.1 GitHub資料

対象repository：[`HIROMU1015/Partially-Randomized-Trotter`](https://github.com/HIROMU1015/Partially-Randomized-Trotter)。以下は今回GitHub接続から**指定commitの本文を実際に取得して確認した**もの。

| ID | 固定commit | 資料と本レビューでの役割 | 取得状況 |
|---|---|---|---|
| S0 | `0d84c2181de128e496bdc1e6161264cdf84eeba1` | [S0修正理論契約候補][S0]（§§3–16、18）。本レビューの直接対象 | 取得・主要節全文照合、blob `bdd32327e5686d0744f5f862789790acf72321f8` |
| S0-INDEX | 同上 | [S0 evidence index][S0INDEX]。原本、authority、未実施カウンタ | 取得・照合、blob `a1a08604e18984a6107042f7d169102679b1153d` |
| GPT-T1 | `d1039afae54520f15545177a7c852407830a0927` | [直前GPT独立科学レビュー][GPTT1]。局所誤差、dressing、joint scalingの根拠 | 取得・主要節照合、blob `5b470fe248e1cc7c2c6ce3dc7e0832e5dda26a8d` |
| PRE | `0f0952614dcf5c06d3d63c21b5d49091b8ecce37` | [G0事前契約][PRE]。Q1–Q5と案A/Bの広い基点 | 取得・関連節照合 |
| G0-PRE | `8b7666d0270f7fac01e430e195bef7b58ac0116c` | [G0-pre][G0PRE]。先行8論文、controlled law、費用モデル | 取得・関連節照合 |
| P0.5 | `ed3ddeacc88e998e7bc1d8c0db8163e106312791` | [P0.5][P05]。PR/Gu/Liの仮定、H6のordinary tail、未取得費用 | 取得・関連節照合 |

S0公開branch `track-b-m1s-g0t1-s0-contract-repair-20261016` のremote HEADが上記 `0d84c218...` に一致することも確認した。S0索引は原本GPT-T1が **52,629 bytes、SHA256 `cb24103f22ddfb12afb82c923fee4210fabae4b7dfec2355896a0686fb18f633`** でbyte保存されたと記録している。本レビューが各ファイルの独立SHA256をすべて再計算したという意味ではない。Codexによる公開・文書validationのPASSは、科学命題の成立証拠とはみなさない。

### 1.2 外部先行研究の到達範囲

公開書誌・概要と、既存の一次論文監査を参照した。Günther et al. [PR原論文][PR] は部分乱択RTE、独立乱数による±branchと対称PFのhalvingの区別を与える。Campbell [qDRIFT][CAMP] は確率的Hamiltonian simulationとcontrolled構成の既知の出発点である。Kanasugi et al. [early-FT QPE][FUJ] は部分乱択を用いた位相推定と資源モデルの重要な対照である。公式の書誌・概要は本レビューでも確認したが、今回**これらのPDF全文の証明を新たに再現したわけではない**。付録の式に関する詳細な対応はP0.5・G0-pre・GPT-T1の固定監査を区別して参照する。

背景のKato摂動論、Davies型secular近似、Chen–Huang–Kueng–Troppのrandom product formula集中等は有用な理論道具の**候補**であって、今回の非CPTPなcoherent写像に対する完成済み保証を意味しない。関連定理の適用条件・誤差normの同定はT1aの証明義務に含まれる。

### 1.3 レビュー範囲と実施していない作業

**実施**：S0契約の科学的意味論と未決事項の比較、既存レビューの数式・適用条件の独立な整合性確認、最小scopeと受理条件の選択、研究としての情報価値・STOP条件の評価。

**未実施**：T1aの新しい一様定理の構築・証明、T1bの新しい漸近定理、Hamiltonian／行列の数値評価、保存データの再集計、H4/H6の新しい科学量算出、回路生成・compile・synthesis、新規資源forecast、GitHub編集、ユーザーに代わる正式採択。**mandatory STOPは維持**する。

---

## 2．研究全体との位置付けと主要な先行制約

M1-Sの目的は、Hadamard測定の二枝で同位置の乱択ラベルを共有するdirectionalなrandomized time evolutionが、**coherent信号の減衰・位相をどの条件で改善できるか**を調べることである。研究価値は、その機構が非可換な決定論blockを含む**部分乱択**に意味のある形で移るかに依存する。

元のM1における「RTEからqDRIFTへ変えると回転数が約半分になる」という説明は、完全qDRIFTの既知の係数差と富士通論文の\(\gamma\)モデルに近く、それだけでは新規アルゴリズムclaimにならない。共有乱数、回文word、directional control、\(Z_c\otimes H\)に対するqDRIFTの外形も、既知技術から自然に導かれる。このことを今回の正式採択候補の**前提制約**として維持する。

一方、通常のchannel誤差が十分小さくない領域でもcoherent signalに条件付きの明確な減衰・位相保証が得られ、さらに\(H_D\neq0\)へ移せるなら、既存法に対する有用な保証となる可能性がある。これは**研究仮説**であり、S0文書を整えたという理由で確認済みとはしない。T1aだけで止まる場合はtechnical note級、T1bが非自明な移送を与えて初めて限定的な新規性を検討できる。実分子での有用性・RPEのRMS保証・end-to-end費用優位はさらに後段の証拠を要する。

A′の順序は、数学的保証を必要最小限にしながら、この**決定的な移送可否**を早期に判定するためのものとして合理的である。理論作業量が小さいことを事前に実測・保証したという意味ではない。

---

## 3．S0の既知代数：ここは合格

### 3.1 完全乱択の正確な平均

Pauli/involutionの初期参照を

\[
P_j=P_j^\dagger,\quad P_j^2=I,\quad p_j\ge0,\quad \sum_jp_j=1,\quad
h=\sum_jp_jP_j,\quad A_j=e^{i\theta P_j}
\]

とする。同位置の2枝でlabelを共有し、異なる位置間ではIIDを仮定する。時系列で\(B_1=A_{j_N}\cdots A_{j_1}\)、\(B_0=A_{j_N}^\dagger\cdots A_{j_1}^\dagger\)とすれば、負枝は同時系列のgateごとの逆角であり、全wordのliteral adjointではない。干渉平均は

\[
\mathbb E[B_0^\dagger B_1]=\Phi_\theta^N(I),\qquad
\Phi_\theta(X)=\mathbb E_j[A_jXA_j].
\]

\(M=\cos\theta I+i\sin\theta h\)、\(\mathcal C(X)=\mathbb E[(P_j-h)X(P_j-h)]\)として

\[
\Phi_\theta(X)=MXM-\sin^2\theta\,\mathcal C(X)
\]

は正確な代数恒等式である。\(\Phi\)は通常の\(\mathbb E[A_j X A_j^\dagger]\)というCPTP channelではない。\(\|\Phi(X)\|\le\|X\|\)はunitaryによる二側乗算と平均から従うが、これだけで固有射影の条件数・phase bias・RPE成功は保証しない。

Shotごとのfresh drawは**無条件平均恒等式の必要条件ではない**。同一列を繰り返し測定するときの条件付き・無条件分散の違いは統計契約の問題であり、今回のT1aのcoherent meanとは区別する。S0§6の扱いは正しい。

### 3.2 部分乱択の正確な平均

T1bで\(H=H_D+H_R\)、\(H_R=\lambda_R\sum_jp_jP_j\)とし、exact halfの\(D=D'=e^{i\alpha H_D}\)を、中央の乱択word\(R=A_{j_r}\cdots A_{j_1}\)の前後へ配置する。両枝は同じ位置のlabelを共有し、すべてのgateの符号を反転させる。この**指定した**branch/time-orderでは

\[
\boxed{\Psi(X)=D'\,\Phi_\theta^r(DXD')\,D}
\]

が正確に成立する。さらにstep間のIIDと固定lawがあれば\(\mathbb E[B_0^\dagger B_1]=\Psi^s(I)\)。S0§5.2の表記は、この限定contractの下で整合する。\(H_D=0\)では\(\Psi=\Phi^r\)に戻る。

これは**coherent一次momentの写像が存在する**という主張であって、\(\Psi\)の固有modeが\(\Phi\)と同じ、減衰指数が同じ、\(\rho_\infty^{(H)}\)が成立する、という主張ではない。\(DXD\)は密度演算子に対する\(DXD^\dagger\)とは異なり、通常の確率channelの理論を無条件に持ち込めない。S0§§5、9はこの区別を正しく保持している。

---

## 4．T1aの比較モデル：C-secを主対象に選ぶ

### 4.1 C-secを選ぶ科学的理由

S0§4.2はC-secとC-logを併記し、未採択としている。今回の正式採択判断では、**C-secをT1aの主たるstatementとし、C-logは補助的な方法に限定する**ことを推奨する。

理由は三つある。

1. **主張対象が物理的に明確**：\(\mathcal L_1(X)=hX+Xh\)が担うcoherentな自由振動に対し、二次の散逸補正の**同一周波数block**を残す。共鳴・非共鳴の区別、項内／項間分散、位相・減衰の意味を明示しやすい。
2. **強い先行法と比較可能**：既知の平均化・secular議論と、今回の非CPTP二側作用への適用差分を正面から評価できる。\(\log\Phi\)を厳密に書けること自体は一種の再表示であり、新しい保証を自動的に与えない。
3. **反証可能な受理条件になる**：具体的な\(z_N^{\rm sec}\)を固定し、局所高次誤差、非共鳴項、近共鳴項が明示的に小さい条件を要求できる。

ただし、**C-secの元の2項誤差式が既に正しい、という採択ではない**。修正した誤差会計を含めて初めて候補statementとなる。

### 4.2 主対象の数学的定義

まず\(\mathcal L_1(X)=hX+Xh\)とし、Pauli参照での二次差分を

\[
\mathcal D(X)=-\tfrac12\{I-h^2,X\}-\mathcal C(X)
\]

と置く。\(\mathcal L_1\)の周波数\(\omega=e_a+e_b\)に対するスペクトル射影を\(\Pi_\omega\)とし、**厳密な同一周波数blockを残す**

\[
\mathcal D_{\rm sec}:=\sum_\omega\Pi_\omega\mathcal D\Pi_\omega
\]

を主参照とする（近共鳴clusterへ後から変える際は、比較対象そのものの変更として扱う）。主近似は

\[
z_N^{\rm sec}(\psi)=\left\langle\psi\left|
 e^{iN\theta\mathcal L_1}e^{N\theta^2\mathcal D_{\rm sec}}(I)
\right|\psi\right\rangle,
\quad z_N(\psi)=\langle\psi|\Phi_\theta^N(I)|\psi\rangle.
\]

周波数射影の対象は**Hilbert–Schmidt演算子空間**であり、Hamiltonianの通常の固有energy gapをそのまま代用しない。\(\mathcal L_1\)は二側作用の周波数を持ち、\(e_a+e_b\)のpair-sumも現れる。実際の写像で許されるnorm、\(\mathcal D_{\rm sec}\)の安定性、初期\(I\)から結合するmode、dimension依存性はまだ証明対象である。

### 4.3 C-logの利用条件

\(\Phi_\theta\)の正確な対数に基づく「C-log」は、**一stepの位相driftや三次以上の局所項を見落とさないための解析・独立検算手段**として許容する。小角で対数の分枝が存在すること、branchを\(\theta=0\)から連続に選ぶこと、\(\log\Phi\)の**どの次数**を近似に採るかは明記する。

しかし、単に\(\Phi_\theta^N=\exp(N\log\Phi_\theta)\)という厳密な再表示を提示してT1a成功としない。C-logを別の主要**保証対象**へ変更する場合には、主参照C-secとの関係、誤差項・位相・受理条件を同じ尺度で説明してCheckpoint 1で科学的に判断する。**新しいGPT詳細レビューを技術的な証明手法の変更ごとに強制する必要はない**が、主要claim自体を変えるなら既存契約の科学的改訂と区別する。

---

## 5．T1aの誤差意味論・適用域・受理条件

### 5.1 最初のstate class

**第一の主張対象を、固定された\(h\)の孤立した固有state\(|k\rangle\)（\(h|k\rangle=e_k|k\rangle\)）の信号**に限定することを推奨する。\(V_k=1-e_k^2>0\)は\(\rho\)を議論する場合のdomain条件である。近縮退・縮退・pair-sumの結合が無いことを一般に仮定するのではなく、それらが存在する場合の対象modeと結合重みを**明示した適用域**として書く。

このclassに限る理由は、初めから任意のguiding stateのcomplex weightと複数modeの干渉を含めると、T1aの研究目的が事実上Q4（RPE）へ膨らむためである。一般normalized\(\psi\)に対するcomplex-signal boundは将来の自然な拡張または補助評価として認めるが、初期の独立GO条件とはしない。T1bでは**全\(H\)** の対象固有stateとその有効modeへ移る。これは実機がexact ground stateを無料で知るという主張ではなく、理論のoracle参照stateである。

### 5.2 複素信号の絶対誤差を主評価にする

採用を推奨する誤差座標は

\[
\varepsilon_{\rm sig}(N,\theta,k):=
\left|\langle k|\Phi_\theta^N(I)|k\rangle
      -\langle k|e^{iN\theta\mathcal L_1}e^{N\theta^2\mathcal D_{\rm sec}}(I)|k\rangle\right|.
\]

主定理候補は\(N\in\mathbb N\)、\(|\theta|\le\theta_0\)、\(N\theta^2\le D_{\rm cap}\)の**明示した対象族の全て**について、局所高次と平均化誤差を含む有効上界を与えるものとする。定数の\(D_{\rm cap}\)、gap／重み、対象次元、sampling lawへの依存を明記し、定数が実質的に無限大なら有用としない。

位相を主張するには、**どの参照信号についてのlower boundか**を定義する。参照信号が小さすぎると複素絶対誤差から位相差への安全な変換はできない。固定状態の\(|z_N^{\rm sec}|\ge z_{\min}>0\)などを**証明域の条件**とし、実際の信号が近傍にあることを複素誤差から示す必要がある。\(z_{\min}\)を恣意的な数値閾値として予め固定する必要はないが、最終定理では条件・依存性を明示する。位相保証の成立を、RPEのRMS保証まで完了した意味にはしない。

### 5.3 必須の誤差会計：局所三次項を明示する

[S0§7][S0]が前回GPTの可換Pauli検算を正しく反映した点を支持する。典型的な誤差の論理的分解は

\[
\varepsilon_{\rm sig}\ \le\ E_{\rm local}
+E_{\rm nonres}+E_{\rm near}+E_{\rm mode},
\]

のような形になる（**これは確立した不等式の表示ではなく、将来の証明義務の分類**）。このうち\(E_{\rm local}\)は、一stepの三次以上の項・有限角度位相歪みを累積した量であり、例えば正則で一様な局所三次剰余から\(N|\theta|^3\le D_{\rm cap}|\theta|\)というスケールの候補が生じる。**具体的な定数とnorm、同時極限での妥当性はT1aで証明する。**

- **局所高次／有限角度**：二次secularを主比較に選んだ以上、省略できない。\(\log\Phi\)で位相を補正する場合は二重計上しない。
- **非共鳴寄与**：\(\omega\neq\omega'\)のcouplingと周波数差の分母。小分母・結合の実質ゼロを区別。
- **低周波／pair-sum**：厳密同周波数blockに保持したmodeと、近共鳴として残差に出すmodeを別に定義。cutoffを導入するなら、事後的な好都合選択を避ける。
- **state／mode**：初期\(I\)から実際に励起され、\(\langle k|\cdot|k\rangle\)へ寄与する部分だけを扱う。全space norm boundを使う場合はその強さと費用を説明。
- **signal floor／phase**：複素信号誤差からphaseを求める追加条件。

「非共鳴項・近共鳴項が小さい」とだけ述べる一様証明は**受理しない**。前回GPTレビューの可換例ではそれらの漏れが無くても\(O(D|\theta|)\)の位相差が残る。なお可換例は**平均化研究自体の反証ではない**ため、これを理由にT1aの研究を中止する必要はない。

### 5.4 T1aのGOに必要な成果と不要な過大目標

初期のT1aが受理可能となる**必要な構成要素**は次のとおり。

1. 固定したlawと主比較モデルについて、複素信号の誤差命題が正確な定義と仮定を伴う。
2. \(N\theta^2\le D_{\rm cap}\)の対象族全体で、局所高次・非共鳴・近共鳴が明示された（空でない）有効な評価がある。
3. 仮定・定数・適用不能例が公開され、粗いchannel normとの比較に意味がある。
4. 完全乱択の先導係数と\(\rho_\infty\)を、**有限\(N,\theta\)の比の保証と区別**して扱っている。
5. T1bのexact-half／Pauli-tailへの移送に必要な構造（周波数、平均化量、error budget）をCheckpoint 1に明示する。

当初から**全次元uniform、任意縮退、Jordan block、一般Hermitian event、guiding-state RPE**まで要求しない。\(\theta_0\)、近共鳴cutoff、重み汎関数の具体定数は、将来の証明の内容である。現時点で根拠なく数値固定しないことがむしろ正しい。

T1aが固定有限次元・孤立modeでしか閉じなくても、それだけで全M1-SをSTOPしない。ただし、既存の平均化理論の単なる転記でありT1bへ意味を持たない場合は、Checkpoint 1で研究価値を低く評価する。

---

## 6．T1bの初期joint scalingを具体化する

### 6.1 S0§8が正しく示した二つの異なる極限

S0の参照辞書を採る：

\[
T_{\rm rel}=s\delta_{\rm rel},\qquad
\theta=\frac{\lambda_R\delta_{\rm rel}}{2r},\qquad
\alpha=\frac{\delta_{\rm rel}}4,\qquad
D_{\rm damp}=sr\theta^2
=\frac{\lambda_R^2T_{\rm rel}\delta_{\rm rel}}{4r}.
\]

\(\lambda_R\)と対象Hamiltonianを固定する場合、\(r\to\infty\)で\(\delta_{\rm rel}\)を固定すると\(\theta\to0\)だが\(\alpha\)は小さくならない。この経路で\(\rho_\infty^{(H)}\)へ無条件に還元することはできず、**有限\(\alpha\)のdressingを正面から扱う別契約**が必要である。

逆に\(\delta_{\rm rel}\to0\)では\(\alpha\to0\)だが、時間やstep数をどう増やすかを決めずに長時間の定量主張はできない。また、各stepの有効固有位相のgapも\(O(\delta_{\rm rel}\Delta E)\)に縮むため、分母も同時に扱う必要がある。S0がこの危険を記録している点は適切である。

### 6.2 T1bの**第一の主対象**に推奨するjoint path

次の**一つの系列**を、最初の科学的契約として採用することを推奨する。

- 対象は固定有限次元、固定\(H=H_D+H_R\)、固定\(\lambda_R>0\)、Pauli-tail。
- 各stepの乱択位置数\(r\ge1\)を**固定の正整数**とする（任意\(r\)・任意サイズへのuniform化は後段）。
- \(\delta_{\rm rel}\to0\)、\(\theta=\lambda_R\delta_{\rm rel}/(2r)\to0\)、\(\alpha=\delta_{\rm rel}/4\to0\)。
- \(s\)は正整数で、\(D_{\rm damp}=sr\theta^2\le D_{\rm cap}\)を満たす**長時間の系列**を考える。主要な非自明な減衰機構は\(D_{\rm damp}\)が零でない有限値へ近づく系列で調べ、\(D\to0\)は整合限界として別扱いにする。
- 同じ対象時間\(T_{\rm rel}=s\delta_{\rm rel}\)にordinary／独立±／gate共有±を対応させる。\(\theta\)だけ同じにして比較しない。

この場合、\(D_{\rm damp}=O(1)\)かつ正の非自明な値を維持すると、**形式的には**\(s=O(\delta_{\rm rel}^{-2})\)、\(T_{\rm rel}=O(\delta_{\rm rel}^{-1})\)という長時間・小stepの関係になる（ここは辞書からの解析的な対応であり、数値実験ではない）。各stepでの乱択効果は小さく、総減衰は有限となる。

このpathは、\(\alpha\to0\)と長時間の有限減衰予算を同時に満たすため、Claudeの部分乱択への**先導次数の移送仮説を調べる最小参照**として合理的である。ただし成功を保証しない。特に固有位相の分離は\(O(\delta_{\rm rel}\Delta)\)で縮むため、\(\Psi\)のstep-level perturbationとの比を誤差項で制御する必要がある。固定step幅・\(r\to\infty\)の経路、\(r\)と系サイズの同時増加は**初期契約の対象外**とする。

### 6.3 外側の決定論分割誤差を消したことにしない

\(D=e^{i\alpha H_D}\)を**exact**にしたとしても、一stepの\(D\,e^{i\delta H_R/2}D\)は、非可換な\(H_D,H_R\)に対して真の\(e^{i\delta H/2}\)と一般に同じではない。したがって「exact deterministic halfだからouter PF biasがゼロ」とは主張しない。

対称stepの局所分割誤差は通常、適切な有界交換子条件の下で\(O(\delta_{\rm rel}^3)\)の候補となる。上記の長時間系列では総\(s\)stepで\(O(s\delta_{\rm rel}^3)\)という累積を検討する必要があり、\(s=O(\delta_{\rm rel}^{-2})\)なら先導とは区別すべき\(O(\delta_{\rm rel})\)級の寄与になり得る。**係数・誤差normの確証はT1bの証明義務であって、本レビューで一般定理として宣言しない。**

T1bの論理的な二段比較は、まず**同じ分割の決定論的対称step基準**に対するcoherent attenuationと乱択biasを評価し、その後、**真の\(H\)に対するouter splitting bias**を別の項として明示することである。これをQ4まで全て先送りして無視すると、\(\rho\)の先導率と物理energy biasを混同しやすい。T1bでは少なくとも**誤差の存在とスケールの分類**を必要な範囲で行い、RPEの総error budgetや数値的費用への伝播はQ4/Q5に残してよい。

---

## 7．Dressing、\(\rho\)および共鳴の意味論

### 7.1 状態の近さと減衰率の相対精度は異なる

前回GPTレビューは、\(H|k\rangle=E_k|k\rangle\)、\(a=\langle H_D\rangle_k\)、\(D=e^{i\alpha H_D}\)のとき

\[
\|D|k\rangle-e^{i\alpha a}|k\rangle\|
\le|\alpha|\,\sigma_k(H_R)
\]

という状態ベクトルの上界を確認した。S0§9がこれをdressingの**候補尺度**とし、\(\Psi\)の有効modeや減衰係数の相対誤差へ無条件に移さない扱いは正しい。

T1bの提出物では、(a) \(H\)対象stateと\(\Psi\)の左右有効mode／射影、(b) 散逸係数の**絶対**差、(c) 分母が正で十分離れた場合だけの**相対**差、(d) 有効step固有位相のaliasとpair-sum、(e) \(r,\delta,s\)への定数依存を別々に示すべきである。

\(DXD\)と\(DXD^\dagger\)は違うので、「半sweep後の密度行列の分散だけ見ればよい」という近道は受理できない。\(\alpha\sigma(H_R)\)や\(\alpha^2\|[H,H_R]\|\)の小ささを、そのまま\(\rho\)の**相対誤差が小さい十分条件**と呼ばない。

### 7.2 \(\rho_\infty\)は完全乱択の先導係数の比

固定した\(h\)固有stateに対し、\(V_k=1-e_k^2>0\)、\(v_{kk}=\mathrm{Var}_j\langle k|P_j|k\rangle\)とすると、

\[
\rho_\infty=\frac{V_k+v_{kk}}{2V_k}
=\frac12+\frac{v_{kk}}{2V_k}\in[1/2,1]
\]

という**係数の代数関係**は正しい。ただし、この比を実際の有限時間で必要なイベント数の比と認定するには、その比較基準で成り立つ一様な信号・位相・減衰保証が要る。\(V_k=0\)では比は未定義である。

T1bの\(\rho_\infty^{(H)}\)は、\(H\)固有stateを使った候補であって、\(\Phi\)の\(h_R\)固有stateの公式を直接置換した定理ではない。**T1bでは、成り立つ／dressing補正版が必要／適用外／未解決**を出力区分として固定することを推奨する。非共鳴・近共鳴やsector外結合の条件が破れた場合は、比の無理な補完ではなく**domain failure**として記録する。

### 7.3 Pair-sumとsector

\(\mathcal L_1\)の固有周波数は\(e_a+e_b\)である。対象modeが全到達可能空間の**非縮退な端点**なら、非自明なpair-sum共鳴は制限される。一方、粒子数を保存するDF eventであっても、対象の\(S_z\)、spin、空間対称性まで保存されていない場合、到達可能空間の端点性は自明ではない。

初期契約はPauli-tailであり、各Pauliが電子数やspinのsectorを越えて結合する可能性を残す。**全Fock最小gapの小ささだけで自動STOPせず、目的modeと実際に結合する重み・pair-sum・aliasを区別する**。一般DF eventの対称性・正規化・native価格は初期T1bの成果条件へ無断で追加しない。

---

## 8．比較対象、研究の新規性、費用座標

### 8.1 比較するのは「同じ理想task」であり「同じ観測mean」を仮定しない

[S0§11][S0]の「同じ対象信号」は、科学的に**「同じ理想Hamiltonian時間発展に由来する目標量」**と明記する方が安全である。Ordinary、独立±、gate共有±は、**バイアス・減衰のある実際の平均信号まで等しいわけではない**。実際の平均が同じと前提して比較すると、まさに研究しようとしている差を消してしまう。

T1a/T1bでは、同じ\(H\)、同じ\(T_{\rm rel}\)、同じ初期参照state、同じ分割・係数law（比較に必要な範囲）、同じ理想的なcoherent phase目標を先に固定する。その上で各方式の**実際の平均信号とbias・減衰を別々に解析**する。将来のQ4/Q5で初めて同じRMS energy error・confidenceのrequired shots/rotationsを比較する。

### 8.2 Baselineとnative費用

- **ordinary qDRIFT**：片枝のfull relative timeを実装。有限角度の平均と減衰・biasを、方向制御とは別に評価する。
- **独立± qDRIFT**：half-timeの二branchを独立draw。正確な負枝順序・位相とsampling lawを固定する。
- **既知PR-RTE**：通常のunbiased numeratorとnormalizer、RPE側の保証が異なることを区別し、T1a/T1bのcoherent attenuationのみによる資源優位を主張しない。

選んだ基準によりone-event controlled price、HWP＋catalyst、STAR、Clifford+Tの単価・shot overheadは異なり得る。\(\rho\)の数値だけでend-to-end費用優位を結論しない。\(S\simeq f_R(1-\rho)\)は固定加法価格・同じshot等の近似であり、\(f_R\)は評価対象**全体**で同じ費用座標を使って定義する必要がある。M1-Sと整合する\(f_R\)は現時点で`NOT_ESTABLISHED`であり、数値GO閾値を導入しない。

### 8.3 研究価値と論文化の着地点

T1aの結果は、初期条件が狭ければ既知の平均化理論の適用・technical note級になる可能性が高い。**T1bで自然な条件の下に部分乱択への移送と、既知channel保証以上に有用なcoherent signalの定量評価が得られるか**が、新しい理論研究としての核である。

- **T1aだけ成立**：限定した平均化補題・適用不能例・注意点をtechnical noteへ整理する余地。独立した大きな研究GOではない。
- **T1a＋T1b成立**：条件付きの部分乱択coherent信号保証を研究成果候補として扱う。文献との等価性を再確認し、実用性の判断を別に行う。
- **さらにGate 0とQ4/Q5を通過**：RPE成功保証・同accuracyでの総資源改善という強い論文claimに近づく。ただし既存のPR・富士通の係数差・既知のdirectional controlを新発明と称しない。
- **T1bが不成立／条件が不自然**：技術ノート化、縮小、M2等への方向転換、Track Bの新規探索の一時休止を比較。限定失敗を全Track Bの一般no-goへ拡張しない。

---

## 9．正式採択を提案する最小S0契約：そのまま意思決定に使える一覧

次表は**GPTの科学的推奨を具体化したものであり、まだユーザーによる正式採択ではない**。既存S0文書へ反映する場合、ここで選んだ科学内容をCodexが勝手に変更しない。

| 契約の欄 | **採択推奨内容** | 受理・禁止の境界 |
|---|---|---|
| 研究目的 | 分岐相関directional-qDRIFTのcoherent signalを平均化し、部分乱択への先導移送を判定 | 回路外形そのものの新規性は主張しない |
| 流れ | A′：T1a→Checkpoint 1→T1b→Checkpoint 2 | Checkpoint 1/2からの自動実行なし |
| T1a対象 | 固定有限次元・Pauli/involution・IID label・同位置共有±。第一主張は孤立した\(h\)固有state | 一般guiding state・全次元一様は必須外 |
| T1a comparator | **C-secを主比較モデル**。\(\mathcal D_{\rm sec}=\sum_\omega\Pi_\omega\mathcal D\Pi_\omega\) | C-logは手法／整合性検算。主target変更は科学契約の改訂 |
| T1a outcome | 複素coherent信号の絶対誤差と\(N\theta^2\le D_{\rm cap}\)での一様性 | 局所三次項・非共鳴・低周波／pair-sum・mode weightが必須 |
| 位相 | Signal floorがある場合のみ複素誤差から位相保証へ進む | phaseをRPE RMSへ自動昇格しない |
| T1b対象 | exact deterministic halfの中央Pauli-tail、\(\Psi=D\Phi^r(D\cdot D)D\)、線形角 | 一般DF eventと実装PFは初期対象外 |
| T1b主極限 | **固定\(r\ge1\)、\(\delta_{\rm rel}\to0\)、\(s\delta_{\rm rel}^2\)を有界に管理し\(D_{\rm damp}=sr\theta^2\le D_{\rm cap}\)** | \(r\to\infty\)固定\(\delta\)・size-uniformな別極限は初期対象外 |
| T1b誤差 | 有効mode、dressing・絶対／相対係数、alias・pair-sum、outer symmetric-step biasの有無と規模 | \(\rho_\infty^{(H)}\)を未証明のまま正式指標にしない |
| 最初の強baseline | Ordinary／独立± qDRIFT、既存PR-RTEを対象・保証別に区別 | 同じ実際のmeanが出るとは前提しない |
| Checkpoint 1 | T1a結果の数学的有効性・既知性・T1b可搬性を判断 | 技術PASSだけでT1b開始しない |
| Checkpoint 2 | T1bの移送、\(\rho\)正当性、適用条件、独立新規性を重要GPTレビュー | ユーザーがその時点でレビュー開始承認 |
| Gate 0 | Checkpoint 2以降、理論で正当化した情報の範囲で事前登録する候補 | 新しい科学数値計算は別途承認 |
| STOP | 定理・保証・定量優位の失敗を区別。縮小やtechnical noteも許容 | M1-SのSTOPをTrack B全体の一般no-goにしない |

**科学的な採択判断**：上表の選択を契約内容として採用することを**推奨**する。現行S0§4.2と§8などの「未選択」を、上表で選んだ範囲に限って**ユーザー採択後に**更新するのであれば、新しい全面的なGPT詳細レビューをもう一度行う必要はない。異なる数学的対象・費用・仮定へ変更する場合は、その**差分に科学的判断が必要か**を改めて判定する。

---

## 10．Checkpoint、未解決時の処理、STOP

### 10.1 T1a → Checkpoint 1

T1aで必要なのは、\(z_N(\psi)\)に関する、非自明な条件下で有効な複素信号の一様な評価と、T1bへ持ち運べる周波数・重み・phase／attenuationの意味論である。**定理の形式的な完成だけでは科学的GOにならない**。例えば、全ての非共鳴条件を強く排除したために量子化学で使えないboundになれば、補題として正しくても研究の優先度は下がる。

結果は次の状態を分ける：`PROVED_WITH_DOMAIN`、`PARTIAL_SUFFICIENT_CONDITION`、`COUNTEREXAMPLE_IN_DEFINED_CLASS`、`UNRESOLVED`、`NOT_APPLICABLE`。根拠のある反例を`INCONCLUSIVE`だけにまとめない。次段へ進むには**T1bに実際に役立つ構造**が存在することを判断し、ユーザーによる次の実行承認を別に得る。新しい重要な科学的判断が必要なら、その時点でGPTレビュー開始承認を求めるが、形式的な文書修正だけで詳細レビューを頻発させない。

### 10.2 T1b → Checkpoint 2

T1bは、\(\Psi\)の対象law、全\(H\)の有効mode、固定したjoint pathでの先導減衰・位相とその絶対誤差、相対比の分母、dressing/alias/pair-sum、outer-step biasを報告する。\(\rho_\infty^{(H)}\)は**成立／dressing補正版が必要／不適用／未解決**を分ける。

**Checkpoint 2は重要GPT科学レビューが必須**だが、T1bの資料が完成しただけで自動的にレビューを開始しない。ユーザーがレビュー開始を明示承認した後、元の科学SOURCE・証明・反例・監査資料に基づいて研究価値と次の方針を判断する。Gate 0の数値取得はさらに個別の計算実行承認を必要とする。

### 10.3 早期の縮小・STOP

- T1aの補題が既存理論の直接系に終わり、部分乱択に有効な新しい情報がない：technical note化または対象縮小を検討。
- T1aが数学的に閉じないが条件付きの有用な結果がある：その範囲と反例を記録し、判定後に契約改訂を検討。
- T1bで\(\rho\)が自然な条件では定義・制御できない：資源倍率claimを停止。ただし複素信号の絶対誤差保証の価値は分けて評価。
- T1bでpair-sum/sector/aliasの制約が強すぎる：適用域を限定するか、DF・BLISS・M2等の別研究方向を検討。**代替案もその時点で新たに科学判断する**。
- 適用域や費用優位が不明：未承認のGate 0/E0数値検証で即時に埋めない。必要な科学契約と承認を別に整理する。

---

## 11．採択権限・次の担当・作業

| 行為 | 担当・承認 | このレビューによる権限 |
|---|---|---|
| **S0／G0-T1／A′研究契約の正式採択** | **ユーザー**。本書§9を採択候補とする | **推奨のみ、未採択** |
| 採択済みS0の差分反映・文書固定 | Codex。旧文書・SHA・歴史的レビュー原本を保護 | 正式採択後の技術作業 |
| **T1aの新しい数学的証明の開始** | ユーザーの**別途明示的な実施承認**。承認範囲を固定してCodex技術補助／理論担当で実施 | **未承認** |
| Checkpoint 1でT1bへ進むか | ユーザー、重要な数学的判断は必要に応じGPT | **未実施** |
| **T1bの新しい数学的証明の開始** | ユーザーの別途承認 | **未承認** |
| Checkpoint 2重要GPTレビュー | GPT、ユーザーのその時点の開始承認が必要 | **未承認** |
| Gate 0/E0科学計算 | Codex、対象・資源・STOPを別途固定してユーザー承認 | **未承認** |
| Q4・Q5・DF拡張・RPE/資源 | 後段の科学判断・実行承認 | **未承認** |

**推奨する直後の段取り**：ユーザーが本書§9の**正式研究契約の採択**を明示した場合、Codexが既存のS0候補に科学的選択を忠実に反映して別版として固定し、必要な参照・未承認フラグを確認する。その後、ユーザーがT1aの**新規理論実施**を別途明示承認した場合のみ、証明・検証作業を開始する。**採択と実施の二つの承認を一つに含めない**。

**mandatory STOP**：このレビュー完了時点ではG0-T1研究契約の正式採択、T1a/T1bの理論実施、科学数値計算、回路compile、新規source/resultの書換えを行っていない。結果・既存classification・authorization・marker・parent worktreeは不変と扱い、事実としてGitHubリポジトリを更新していない。

---

## 12．参照文献・固定ソース

### 12.1 GitHub固定資料

[S0]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0d84c2181de128e496bdc1e6161264cdf84eeba1/docs/tracks/algorithm_codesign/m1s_g0t1_s0_repaired_theory_contract_candidate_20261016.md
[S0INDEX]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0d84c2181de128e496bdc1e6161264cdf84eeba1/artifacts/track_b_m1s_g0t1_s0_contract_repair/2026-10-16/v1/evidence_index.json
[GPTT1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d1039afae54520f15545177a7c852407830a0927/docs/research/track_b_m1s_g0t1_claude_gpt_independent_scientific_review_20261016.md
[PRE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0f0952614dcf5c06d3d63c21b5d49091b8ecce37/docs/tracks/algorithm_codesign/m1s_g0_proof_and_gate0_precontract_20261015.md
[G0PRE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8b7666d0270f7fac01e430e195bef7b58ac0116c/docs/tracks/algorithm_codesign/m1s_g0_pre_prior_art_protocol_and_cost_decision_packet_20261015.md
[P05]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ed3ddeacc88e998e7bc1d8c0db8163e106312791/docs/tracks/algorithm_codesign/m1_p05_theorem_protocol_and_resource_preflight_20261015.md

### 12.2 公開一次文献・背景資料

[PR]: https://arxiv.org/abs/2503.05647v2
[CAMP]: https://arxiv.org/abs/1811.08017
[FUJ]: https://arxiv.org/abs/2603.22778v2

- Günther et al., *Phase estimation with partially randomized time evolution*, PRX Quantum 7, 020332 (2026). 対称PFとpartial-RTEのcontrolled law：Appendix A.4。RPE：Appendix B。費用：Appendix E。文献の固定版・確認範囲はP0.5/G0-preも参照。
- Campbell, *A random compiler for fast Hamiltonian simulation*, Phys. Rev. Lett. 123, 070503 (2019). qDRIFTとcontrolled evolution：Appendix D。
- Kanasugi et al., *Enabling chemically accurate quantum phase estimation in the early fault-tolerant regime*, arXiv:2603.22778v2. 一次資料の設定とSTAR費用：Appendix B。
- Childs–Ostrander–Su, *Faster quantum simulation by randomization*, arXiv:1805.08385v2（近接する対称ランダムPF。詳細対応はG0-pre）。
- Kivlichan–Granade–Wiebe, *Phase estimation with randomized Hamiltonians*, arXiv:1907.10070（状態依存乱択位相推定。詳細対応はG0-pre）。
- Chen–Huang–Kueng–Tropp, *Concentration for random product formulas*, arXiv:2008.11751v4（単一実現と平均の区別。詳細対応はG0-pre）。
- Gu et al., *Noise-resilient phase estimation with randomized compiling*, arXiv:2208.04100v2（channelスペクトルと摂動。T1aへの直接適用は未証明）。
- Kato, *Perturbation Theory for Linear Operators*（一般摂動論の背景）。
- Davies, *Markovian master equations*, Commun. Math. Phys. 39 (1974)（平均化／secularの背景。今回の写像への適用は未証明）。

---

## 13．研究レビュー完了時の監査

- [x] ユーザーの明示的なレビュー開始承認がある。
- [x] S0、S0索引、GPT-T1、旧G0事前契約、G0-pre、P0.5の固定原資料を取得した。
- [x] S0公開branchのHEADが固定commit `0d84c218...` に一致することを確認した。
- [x] 固定原本・ユーザー承認・科学的推奨・未実施の研究を区別した。
- [x] C-sec／C-logを比較し、T1aの主参照をC-secに選ぶ科学的推奨を具体化した。
- [x] T1aのstate class・誤差座標・local高次項・位相floor・Checkpoint条件を明確化した。
- [x] T1bのexact-half因数分解、joint scalingとouter splitting biasの意味を確認した。
- [x] \(\rho\)、dressing、pair-sum／sector、baseline、native費用の適用限界を区別した。
- [x] 新定理の証明・科学数値評価・回路compile・既存結果の再集計・GitHub変更を実施していない。
- [x] M1-S／G0-T1の正式採択とT1a/Gate 0の実行承認を代行していない。
- [x] mandatory STOPとParent allocation D＋E、既存成果・marker・authorizationを維持した。
- [x] 完成版Markdownを作成し、ファイル存在・参照の検査を行う。
