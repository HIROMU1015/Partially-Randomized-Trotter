# N1・N2・N3 限定feasibility：独立科学レビューと次段方針

- レビュー日：2026-10-11（JST）
- 利用者の明示指示：「レビューを開始して」
- 対象：`HIROMU1015/Partially-Randomized-Trotter`
- Branch：`hamiltonian-construction-feasibility-20261011`
- 結果・取得確認commit：`ae7434452450e80ec007d65417335e18a063fa1e`
- 科学source：`ed335a007bee6966a37f3026ac188534e526b5b9`
- 保存復旧source：`a08d04c564931aa5d085be60af5f14eedc2a26d6`
- 公開結果の親commit：`10fa5f97cf7e58ea351772bb02bfe8afa82e3806`
- レビュー状態：完了。新しい中心テーマの採択・次scienceの実行認可ではない。

## 0. 判断の要旨

三案をまとめて失敗とする判断も、N3だけを成功として採択する判断もしない。確認された改善機構、現在の構成・選択の弱点、評価尺度の限界を分ける。

**N1：条件付き継続・選択法の再設計。** 同じ近似Hamiltonianを実行する比較では縮退群内frameによる回路削減が成立している。元Hamiltonianの精度を含めると現候補は不利。今回、真の基底状態を用いない鋭い係数boundを導出し、保存費用で事後感度計算した。大半の悪化は縮まるが、保存された最良q2候補は元のnative exactをなお上回る費用となった。『上界を鋭くすれば既に勝っている』とは結論しない。次は基底回路だけを目的にせず、誤差を使うことによる費用増加まで含めた有限構成・選択へ進む価値を判別する。

**N2：現小系native scanは保留。** 実装成立、表現生成、実費用を別に判断する。摂動例では採否のentry-L1と異なるFrobenius least squaresを使うため、同じSに存在する許容表現を見落としている。これは標準的なL1 fitで回復できるが、それだけで量子資源は改善しない。4 occupation bit・raw RZ/CX・逐次加算の結果から、Hamming-weight phasing等のfault-tolerantな利益を否定できない。一方、単に既知のadderへ交換するだけでも新規研究にはならない。候補生成の費用と、最適化済み既知HWP等との損益分岐を先に整理する。

**N3：構造を使う誤差評価の方法論を優先的に検討する候補。** 高gap例のbound取得は支持するが、資源優位・新規性は未成立。小gap例について同一入力を独立に診断すると、4 active modesでは誤差約0.008363で目標0.004を超えるが、5 modesでは約0.0002622で足りる。現selectorのfull-space復帰は『物理的に全軌道が必要』ではなく『現在の十分条件が5 modesを認められなかった』という結果である。さらに、5 modesで正確なactive最低値・discarded最低値・結合normを診断用に入れても、粗い2×2比較式は約0.006086にしかならない。単に外部下界を精密化するだけでなく、低エネルギー状態と結合方向の関係を保持する設計が必要になる。

研究全体として、同型toyを追加し続けたり、ただちにN4以降を増やしたりするより、**N1の誤差と回路削減の損益分岐、N3の証明が情報を失う箇所を、次のアルゴリズム上の問いにする**ことを勧める。N2は並行した軽い方法論確認までとする。N3を中心テーマに採択したわけではない。

## 1. 研究目的と役割分担

今回の目的は、PR外部のHamiltonian表現・構成の改善、または外部とPR内部の共同設計により、科学的意義と新規性を持つ算法を作ることである。PR内部単独のsampling・return改良はTrack Bに残す。外部のみで成立する方法も対象とする。

本レビューは旧A-core/B′-pair/Cの結果の再レビューではない。新N1/N2/N3 batchの結果、数学的な解釈、新規性と研究の着地点を評価する。旧候補のSTOP、Track A/Bの契約・結果は変更しない。特定の分子、basis、RZだけの単一指標、普遍的な定理、全指標での同時勝利を新しい必須条件として追加しない。

GPTが科学的に変更すべき問いと比較基準を示し、Codexが具体的な有限実装・数値実行・保存を担当する。次の作業案は一つの限定packageにまとめられるが、本レビュー完了自体を次scienceの実行指示として扱わない。

## 2. 使用資料・取得状態・独立性

[R1]の報告、固定science source [R2]、一次result、CSV [R3]、N3 checkpoint [R4]、復旧・保存・構成監査 [R6–R9] を確認した。大容量native IRは通常のContents取得では本文が空になるが、Git blob `570f5ffe8101123d2e9fa2ae9e4f554daef34670` 経由で取得できた。約38.25 MBのIRが未push・欠落していたわけではない。

確認の深さを以下のように分ける。

- sourceの主要ロジック：N1候補と選択、N2係数fitと可逆charge、N3係数下界とactive拡張を読んだ。
- 保存結果：表、CSV、N3履歴・計算値、技術失敗・復旧記録と独立保存検証の報告を参照した。
- GPT側の独立作業：同じN1スペクトルの有理数bound、保存PF/countによる再会計、同じN2の固定Sに対するL1問題、同じN3入力のactive段階ごとの小行列診断を行った。付録の独立scriptはrepository moduleをimportしない。
- 行っていないこと：元science runnerの再実行、280回路の全再compile、全177 source blobsの独立再hash、全IRを別の第三実装で再認証、分子生成、未知入力の性能検証、量子shot、GPU、GitHubへの書込み。

GitHubから取得した報告書や監査が、外部研究者による科学再現であるとは表現しない。GPT側の追加計算もbinary64診断を含み、区間演算による機械的certificateとは区別する。

## 3. batchが答えたことと、まだ答えていないこと

### 3.1 共通会計

N1/N2は今回、決定論回路と固定時間complex Hadamard信号を用いており、RTEを組み込んだPR性能比較ではない。normalizationは1、各軸epsilonは `0.04/sqrt(2)`、X/Y合計失敗率は0.05である。

\[
 N(b)=\left\lceil \frac{2\log 80}{(0.04/\sqrt2-b)^2}\right\rceil,
 \qquad G_m=N(b)\,(C_{X,m}+C_{Y,m}).
\]

N1の `b=T epsilon_H + PF_bias` では、モデル誤差に解析上界を使い、PF誤差は選択後の小sector診断を使う。これを大系で真値なしに実行できる完全な誤差保証付きselectorとは呼べない。N2は可換なdensity演算子なので、この実装にPF biasはない。[R1–R3]

RZは論理T countではない。CX、RZ、workspace、depthは別々に評価する。全体のコストを任意の換算係数で合算しない。一般的なエネルギー推定の全コストを、この固定時間信号式と同一視しない。

### 3.2 N3のtaskは異なる

N3は基底エネルギーのactive-space縮約に関する係数boundの取得を試した。量子回路やactive solver、状態準備、QPE成功率を比較していない。N1/N2のcomplex信号精度とN3のエネルギー幅0.004は異なる評価量である。したがって『N3だけ結果がpositiveなので資源上最良』という候補比較はできない。[R4]

## 4. N1：成立したframe機構と、成立していない総費用削減

### 4.1 固定条件

4 fermionic modes、2粒子sector dimension6、重み `(0.7,-0.4)` のsquare factors、one-body `diag(.08,-.04,.03,-.02)`、scalar `.11`。T=.6、q=1/2/4/8、モデル誤差予算.02。近縮退スペクトルはfactor0 `(1,1.002,-.4,-.397)`、factor1 `(.7,.703,-.2,-.196)`。[R1,R2]

planted-localとdense-groupの2例では、同じ近似Hamiltonianのungauged対照より、縮退群内frameを選んだ回路のRZ/CXがそれぞれ9.92%/3.85%、4.89%/1.89%小さかった。この比較は『近似によってできたexact gaugeを使い、同じ演算子を安く実行する』という機構を支持する。

しかし元のHの精度まで含めると、q2で費用は増えた。

| context | exact N | N1 N | exact XY RZ/CX | N1 XY RZ/CX | 元の会計でN1/exact仕事量 RZ/CX |
|---|---:|---:|---:|---:|---:|
| planted | 16201 | 24131 | 1250 / 832 | 1126 / 800 | 1.34172 / 1.43219 |
| dense | 16296 | 20513 | 1310 / 848 | 1246 / 832 | 1.19728 / 1.23502 |

細かな比は元CSVの整数workから再計算する。dense例でN1と有限shifted-cutoffが同じ候補に到達したことも保存されている。有限shifted比較をSCDF全体と同一視はしない。[R1,R3]

### 4.2 選択の問題

現selectorは基底回路のRZまたはCXを最小にし、モデルboundをcap以下にする。full wrapper、反復数、phase、biasによるshot増加は選択に入っていない。これはbasis削減のfeasibility検査として理解できるが、最終費用最小化を達成した実装ではない。[R2]

元の無変更候補が候補集合にあっても、代理目的だけで選ぶと最終費用が悪くなる。『少なくともexactより悪くしない』という性質を目的にする場合、採否に最終評価と対応する情報を使う必要がある。ただしdense truthによるoracle選択と、大系で使えるselectorは別である。

### 4.3 係数だけで鋭くできるモデルbound

N1では、近似後に縮退群内frameを変えても、近似gは元のUで対角なままである。したがってFとF-tildeの共通固有基底で、固定粒子数nuのsquare差は

\[
 e_\ell=|w_\ell|\max_{|S|=\nu}
 \left|\left(\sum_{i\in S}\eta_i\right)^2-
             \left(\sum_{i\in S}\widetilde\eta_i\right)^2\right|.
\]

複数factorでは `||H-Htilde||_nu <= sum_l e_l` とする。これは真の基底状態を使わない。今回nu=2なので6組の係数和を調べるだけでよい。一般にnuがnとともに増える場合、全部分集合列挙は安価とは限らず、この式をそのまま大系効率的algorithmと呼ばない。

今回の選択ではどちらも変更factorは一つなので、意図された有理数スペクトルについて

- planted：`0.002113125 = 3381/1600000`
- dense：`0.001268925 = 50757/40000000`

となる。元のproduct-norm bound `.007007/.0042042` より鋭い。保存されたactual-sector診断と数値は一致するが、その診断をコピーしてcertificateに置き換えたわけではない。独立にスペクトルから導出した。有限精度で生成されたeig/frameの誤差は、正式な新実装では別途囲う必要がある。

### 4.4 上界の修正だけで勝敗は変わるか

元の候補、回路、PF診断をそのまま使い、モデルboundだけを上述の式へ変えた事後感度計算を行った。元runを再分類せず、新しい性能実験とも呼ばない。

| context | sharp bound | q2で再会計したN | RZ work比 / exact | CX work比 / exact |
|---|---:|---:|---:|---:|
| planted | .002113125 | 18118 | 1.0073881 | 1.0753135 |
| dense | .001268925 | 17419 | 1.0166909 | 1.0487445 |

q1/2/4/8の同じ保存候補でも確認し、再会計後の最良点はこのq2であった。RZ悪化は0.74%/1.67%程度へ縮小するが、優位には転じない。『保守的boundだけが原因だった』とも、『frame削減に価値がない』とも結論しない。

この結果は候補と比較を固定した感度である。新しいcenter、partition、全候補の再選択、精度を変えた探索、fault-tolerant synthesisは行っていない。

### 4.5 損益分岐を構成に組み込む

ceilを除いた近似では、a=epsilon/sqrt2、回路比r=C1/C0とすると、G1<G0の条件は

\[
 b_1-b_0 < (a-b_0)(1-\sqrt r).
\]

これにPFとmodel biasを分けて入れると、回路削減が支払えるモデル誤差を候補ごとに評価できる。plantedのRZではモデルboundが約.0019783未満、denseでは約.0009584未満が同じq2の連続式での損益分岐であり、現在のsharp boundは上回る。正式採否はceil込みで行う。

この不等式自体は新規理論ではない。研究としては、どの縮退・frame・centerを生成すればこの条件を満たせるか、既知shift/cutoffやcompilerより良い構造を作れるかが本体である。

### 4.6 新規性と判断

人工的な対称性による合成、rotationとToffoliの交換は既知 [P1]。SCDFはtensorとsymmetry shiftの最適化を扱う [P2]。N1をそれらと同一と断定はしないが、群内基底の存在確認だけでは差分は弱い。

**条件付き継続。** 既存の回路削減機構を、元H精度込みの候補生成・採否へ接続する一段だけを検討する。basis proxyだけの同型scanは増やさない。鋭い係数boundと誤差費用を入れても既知対照との差が残らなければ、現N1は実装手法として整理し主テーマへ昇格させない。

## 5. N2：表現の有無、生成器、回路尺度を分ける

### 5.1 固定結果

4 occupation bits、J full-ij二次式、Sは{-1,0,1}、最大2 charges。820有限候補を生成してFrobenius least squaresでKを求め、entry-L1 bound .0021で採否を決めた。[R2]

signed-overlapではJだけから同じoperatorを表すchargesを生成した。しかしcompute–phase–uncomputeはXY RZ791/CX664、直接実装は47/64で、workspaceも6個加わった。uniform例の生成器は2 chargesを返し、既知unsigned1 chargeより悪かった。これらは実装の事実であって、新規性・一般効率性の証拠ではない。[R1]

### 5.2 見落とされた許容表現

同じ固定S

\[
S=\begin{pmatrix}1&0\\1&1\\0&1\\-1&0\end{pmatrix},\quad
K_0=\begin{pmatrix}.3&-.1\\-.1&.2\end{pmatrix}
\]

からJ0を作り、J03/J30だけ+.001した摂動例を考える。K0をそのまま使えばentry-L1誤差は.002でcap .0021内。least-squaresで得るKはこの尺度で.0036となり、生成器は棄却した。

同じSについて、entry-L1を直接最小化する標準LPを独立に解くと.002の解が得られた。これは有限Sの再探索や新しい構成入力の追加ではない。

さらに、この固定Sでは、要素00/33/03/30だけから

\[
2|.3-K_{00}|+2|K_{00}-.299|\ge .002
\]

となり、K0が全体で.002を達成するため、固定SのL1最適値は有理数として1/500と分かる。

したがって、現在のNO_ELIGIBLEは『そのfamilyに許容表現が存在しない』ではなく、『LSで返した各Kを評価したとき許容候補が見つからなかった』というstatusである。報告書はこの区別を適切に記録している。

### 5.3 修正方法と、修正しても解決しないこと

固定SのK fitは

\[
\min_{K,t}\sum_i t_{ii}+2\sum_{i<j}t_{ij},\quad
-t_{ij}\le J_{ij}-(SKS^T)_{ij}\le t_{ij},\quad t_{ij}\ge0
\]

としてよい。これは標準的な線形計画であり、新規アルゴリズムの主張にはしない。

また、K fitが許容になっても加算器・phase・workspaceの費用は自動的には安くならない。uniformでsigned generic規約が不要なsign情報を持つこと、proxyが2 chargeを選ぶこと、単純incrementが既知最適adderより高いことは、表現の誤差とは別の改善対象である。

### 5.4 今のnative指標でHWPの可能性を否定しない

HWPの主な利益は、同角度の多数の高精度回転を、加算と少数の回転に置き換える点にある。論理T/Toffoliと高精度RZ synthesis、ancillaのspace-timeを比較する [P3,P4]。raw RZ countはこれを表さず、Clifford・特定角・一般角を区別しない。

N2のdensity二次式は、同角度独立回転の標準HWPと完全同一ではない。Q_a Q_bのphase、負数、可逆計算、追加controlを含めて比較する。標準HWPや2025年のcatalyzed HWPを自動移植すればよいと断定もしない。catalyst準備・reuseと同じtask・合成精度での比較が必要である。

4-bitで不利なことから大系・FTで不可能とは言えない。しかし、勝てる大きさを無制限に探索する理由にもならない。まず、Jのどんな構造ならSを多項式費用で生成できるのか、S/Kの情報圧縮と回路圧縮を両立できるのか、既知低rank/HWPより何が新しいかを定義する。[P5]

### 5.5 判断

**現4-bit native scanは保留。** L1 fitの修正は通常の数理最適化修正として一括処理できる。追加の主要検証は、合成精度を含む回路費用の損益分岐と、指数的候補列挙をしないS構成の差分が具体化できた場合に限る。

現在のN2 taskで準備したoccupation状態はDの固有状態なので、単一のphase信号そのものは古典的に分かる。全入力作用・workspace復帰のprimitive検証としては有効だが、これを相関分子の非自明な状態・エネルギー計算の改善とはしない。

## 6. N3：物理的な縮約可能性と、証明の失敗を区別する

### 6.1 保存結果を支持する範囲

6 modes、2粒子、full sector15、initial active4・sector6。hoppingとdensityを係数で与え、外部e=(6,8)または(.4,.6)。selectorは試行occupation01のU=0、係数norm、first-external classesだけを用いた。true energy/gapは選択後に診断している。[R4]

高gap例で、g_lower=(4.86,6.86)、beta=(.08,.04)、delta=.00154963614698を得て4 activeを選んだ。実際のshiftは.000640824981で、数学的十分条件をこの小例で確認できた。外部e=6,8を人為的に設定した例であり、典型的分子での容易さや化学的精度の達成、量子cost削減を一般化しない。

### 6.2 小gapのfull-space復帰は必要性の証明ではない

同じ固定Hamiltonianの各途中active setについて、GPT側で独立に実energy差を診断した。運用selectorへ真値を与えたものではなく、現selectorが失うものを調べるためだけである。

| active modes | sector dimension | 実際のE(active)-E(full) | 元selector |
|---:|---:|---:|---|
| 4 | 6 | .00836277867455 | UNRESOLVED_LOWER_BOUND |
| 5 | 10 | .000262192277976 | UNRESOLVED_LOWER_BOUND |
| 6 | 15 | 0 | FULL_SPACE |

目標energy .004に対して、4 modesは不足だが5 modesは足りる。従って『小gapでは全軌道が物理的に必要だった』は強すぎる。正しくは『現行の係数十分条件と拡張規則では5 modesを受理できず6へ戻った』である。

これも5 modesが最良active setだという最適性証明ではない。調べたprefix内のenergy feasibilityの診断に限る。

### 6.3 外部下界を鋭くするだけでも不十分な例

active5で、診断用に正確な数値を取ると

- a=E0(A)=-.294087596218998
- c=lambda_min(C)=.373522461492650
- beta=||B||=.0640312423743285

である。元の粗い2×2比較構造に最も有利なU=a、正確なc/betaを入れても

\[
\delta_{\rm scalar}
=\frac{\sqrt{(c-a)^2+4\beta^2}-(c-a)}2
\simeq .00608583199430>.004.
\]

実際のshift .0002622に比べてかなり大きい。つまり、この例では『cをうまく見積もれなかった』だけでなく、**全active空間の最大結合normと最低エネルギーだけに潰す情報損失**が大きい。

この診断の真のa/c/betaを将来の証明量へ流用しない。また、全種類の細分化・operator boundでも失敗するという不可能性定理ではない。現在の一つの粗い比較式についての限界である。

### 6.4 数学的に確認した基本bound

ブロックH=[A B; B† C]、a=lambda_min(A)<=U、C>=cI、||B||<=beta、g=c-U>0なら

\[
 a-\delta\le E_0(H)\le a,\quad
\delta=\frac{\sqrt{g^2+4\beta^2}-g}{2}.
\]

外部classを分け、class間結合をblock Gershgorin等で安全にd_jへ吸収し、g_j=d_j-U>0を得た場合、delta=Σ beta_j²/(g_j+delta)の比較に進める。単なるone-body gapをmany-body cと同一視しない。

N3sourceの係数評価は数学的十分条件に対応するが、binary64 bisectionはoutward roundingではない。数値上の十分条件PASSと機械認証の違いを保持する。今回の非形式的な研究判断を妨げないが、将来『certificate付きalgorithm』として売り出すなら必要な誤差処理である。

### 6.5 次に問うべき方法論

候補となる問いは、**真の基底状態・真のgapを使わず、低エネルギーのactive状態と、捨てる空間への強い結合が同時に起こるかを識別できるか**である。

一つの数学的入口は、係数から取得できるL_Aと非負kappa_j, eta_jに対する

\[
 B_jB_j^\dagger\preceq \kappa_j(A-L_AI)+\eta_j I
\]

のような関係を使うことである。これは未実装の設計方向であり、今回この関係の有利な係数を取得したとは主張しない。局所factorizationや正値性から多項式費用で検査できるかが重要で、全Hilbert行列SDPを解いてしまう方法を効率的前処理と呼ばない。

例えば既知のd_jからz<=Uに対しC-zの安全な比較を作れれば、Feshbach–Schur補正 B(C-z)^(-1)B† をscalar beta²だけに潰さず評価する余地がある。Feshbach–Schur法と固有値boundは既知 [P7]。新規性は、量子化学の係数からその関係を安く作り、どのorbitalを戻すかの構成へ結び付ける部分に必要である。

他方、まずpositive densityをnormの負寄与として引かず、各classで実際に作用しない項を除く、という簡単なbound改善は通常の保守的評価の改善として有効である。しかし前節のoracle診断から、単一scalar化だけの改善で必ず解決するとは期待しない。

### 6.6 新規性と着地点

SW/downfoldingの誤差理論は既知 [P6]。SQuISHはHamiltonianと近似基底stateを利用して縮約する [P8]。『active spaceを小さくする』『beta²/g型の式を使う』だけでは独立新規性にならない。

**N3を次の方法論検討で優先する理由は、唯一positiveだったからではなく、同じ小gap入力に実際の縮約余地があり、何の情報を捨てたために現保証が失敗したかが具体化できたから**である。まだ主テーマの採択、論文化可能性の確定、実資源改善の証明ではない。

将来のenergy budgetにはactive solverの誤差・状態準備・基底stateとの重なり・古典certificate取得費用を含める。エネルギーの縮約幅deltaから、任意初期状態の時間発展誤差をT deltaで保証することはしない。

## 7. 技術的信頼性と来歴

run1は最初のbasis compileのscalar phaseで停止。run2はN2全logical入力の意味論とcompiler初期状態仮定の不一致によりnon-scalar差で停止。run3は全比較へ修正を適用し、演算子とworkspace境界を検査した。[R1]

run3は科学結果と全9 checkpointを保存した後、pretty-JSON全IR出力がper-file上限を超えてTECHNICAL_STOPとなった。その後、別source `a08d04c...` が既存checkpointをcompact JSONへ集約し、元の数値とgate順序を保持した。科学再実行は0。元の失敗やsourceを成功へ遡及変更したものではない。[R6]

保存監査は280 native IR、120 semantic rows、177 source/input blobsを照合し、位相/workspace/shots改変を拒否している。[R7] その結果を引用するが、GPTの再実行・外部再現とは区別する。

pure scalarの位相補正は小行列同値確認に限られ、一般の大規模compilerで自由に誤りを修復できる算法ではない。43本の補正履歴は保持する。relative phaseやworkspace漏れまでglobal phaseの違いとして無視しない。

global manifestの旧description欠測は基点から同じで、新86番目のresult entryと旧85件は別監査されている。全体manifestをPASSとは扱わないが、この局所結果が取得不能・未pushだとも扱わない。修復のために全scienceを再実行する必要はない。

## 8. 研究方針の更新と次に渡す課題

ここまでの複数batchを『表現変更は全部失敗』とまとめない。今回N1では、以前の小例とは反対にnative exactがwhole-Pauli対照より安かった。したがってPauliを常に基準最良と決め打ちすることも不適切である。[R1]

ただし、数学的に正しい表現を一つ作り、回路を試し、負けたら別表現へ移るだけでは、研究の中心が定まらない。今回は以下の三つへ論点を絞る。

| 方向 | 次の科学的な問い | 継続しない作業 |
|---|---|---|
| N1 | 誤差費用込みで成立する縮退/frameを入力から選べるか。既知shift/cutoffとの差が残るか | basis countだけの同型grid増加 |
| N3 | 低energyとcouplingの関係を失わず、係数から証明可能な縮約を生成できるか | high-gapをさらに人為的に大きくするだけの成功例追加 |
| N2 | 許容normに沿う構成と、既知最適化HWP等に対する合成込み損益分岐があるか | 現4-bit nativeの同型追加compile |

**次の技術担当はCodex。** 本レビューの数式・判断を用いて、以下を一つの限定packageとして具体化する案が適切である。細かなtest修正ごとにGPTを挟まない。現回答は研究上の方針提案であり、次scienceの明示指示ではない。

1. N1：固定候補を使い、係数bound・回路・精度を同時に見るselectorを実装する。今回GPTが行った再会計そのものを新しい性能成果として再報告せず、何が候補生成・選択上新しくなったかを示す。元のexact候補を保持する。小nu係数boundをそのまま一般指数探索へ拡張しない。
2. N3：係数だけで改善できるclass下界と、結合の構造を保持するboundを区別し、少なくとも一つの具体構成を実装・比較する。true energy/gapは評価側だけ。5 activeを『正解』としてハードコードせず、別の強結合・近接energy対照でも安全側に拒否できることを確認する。構造boundを全行列計算でしか作れない場合は、その古典費用と限界を報告する。
3. N2：fixed-S L1 fit不整合を修正し、原statusと修正後を別に保存する。ここまでの修正自体は新規性としない。追加scienceを広げる前に、Toffoli・非Clifford回転精度・workspace・既知adder/HWPのaccounting上の差を作る。差分がなければ保留を維持する。

N1とN3に同じ計算資源を配る必要はない。重要な障害や同値性が早く判別できた場合はそこで記録して止める。一般的な性能優位・新規性・中心仮説をCodex単独で確定しない。

次の結果でもN1の利益が既知cutoffへ還元され、N3が安価な情報ではcertificateを閉じられず、N2が既知HWPの適用に留まるなら、現族の実装を比較基盤として保存し、別の未解決構成問題へ戻る。それを前提に今回三案を全部捨てる判断はしない。

## 9. 研究としての主張の範囲

現在、論文の主要claimとして確認済みなのは新しい一般算法ではなく、局所mechanism、構成・採否の不一致、certificateの失う情報である。これだけで独立論文に十分だとは判断しない。

N1の着地点候補は『許容誤差を回路削減へ変える、誤差-awareな縮退/frame生成法』。N3は『外部空間との結合構造を使い、真のground情報なしにactive spaceを選ぶ方法』。N2は『回路で安いinteger-charge構造を生成する方法』だが、現時点では近接研究との方法論差分が最も弱い。

positive/negativeの数やtoyでの倍率ではなく、同じ既知対照に対し何を新しく構成でき、どの条件で意味があるかを主要判断にする。benchmark論文への無言の転換はしない。

## 10. Repository参照（固定commit）

- [R1] 結果報告：<https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ae7434452450e80ec007d65417335e18a063fa1e/docs/research/hamiltonian_algorithm_design_results_20261011.md>
- [R2] 固定science source：<https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ed335a007bee6966a37f3026ac188534e526b5b9/src/trottertracks/representation_exploration/algorithm_design.py>
- [R3] 保存された精度・費用CSV：<https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ae7434452450e80ec007d65417335e18a063fa1e/artifacts/hamiltonian_algorithm_design/2026-10-11/precision_resource_comparison.csv>
- [R4] N3元checkpointとselection履歴：<https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ae7434452450e80ec007d65417335e18a063fa1e/artifacts/hamiltonian_algorithm_design/2026-10-11/run3/checkpoint_N3_coefficient_feasibility.json>
- [R5] compact一次result：<https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ae7434452450e80ec007d65417335e18a063fa1e/artifacts/hamiltonian_algorithm_design/2026-10-11/run3_recovered/result.json>
- [R6] export recovery audit：<https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ae7434452450e80ec007d65417335e18a063fa1e/artifacts/hamiltonian_algorithm_design/2026-10-11/run3_recovered/export_recovery_audit.json>
- [R7] 独立保存検証の記録：<https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ae7434452450e80ec007d65417335e18a063fa1e/artifacts/hamiltonian_algorithm_design/2026-10-11/independent_verification.json>
- [R8] 構成・選択・export監査：<https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ae7434452450e80ec007d65417335e18a063fa1e/artifacts/hamiltonian_algorithm_design/2026-10-11/construction_selection_and_export_audit.json>
- [R9] 旧entry保全と新manifest entry：<https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ae7434452450e80ec007d65417335e18a063fa1e/artifacts/hamiltonian_algorithm_design/2026-10-11/manifest_new_entry_and_preservation.json>
- [R10] 全native IR、blob SHA 570f5ffe8101123d2e9fa2ae9e4f554daef34670：<https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ae7434452450e80ec007d65417335e18a063fa1e/artifacts/hamiltonian_algorithm_design/2026-10-11/run3_recovered/native_ir.json>
- [R11] perturbed-fitの保存診断：<https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ae7434452450e80ec007d65417335e18a063fa1e/artifacts/hamiltonian_algorithm_design/2026-10-11/perturbed_fit_diagnostic.json>
- [R12] 実行前scope：<https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ae7434452450e80ec007d65417335e18a063fa1e/docs/research/hamiltonian_algorithm_design_scope.md>

## 11. 公開一次文献と確認範囲

参照日はいずれも2026-10-11。以下は本レビューの直接論点に絞った一次資料の比較であり、全世界のpriority認定・網羅的レビューではない。HTML取得可能な文献は本文を参照し、abstractだけを確認したものはその範囲を越えた命題の帰属に用いていない。

- [P1] P. Mukhopadhyay, N. Wiebe, H. Zhang, *Synthesizing efficient circuits for Hamiltonian simulation* (2023), npj Quantum Information. <https://www.nature.com/articles/s41534-023-00697-6>。係数対称性と可換集合合成、rotation/Toffoliの交換、qDRIFT接続を確認。N1固有値cluster-frame算法が完全に同一という主張はしない。
- [P2] *Reducing the runtime of fault-tolerant quantum simulations in chemistry through symmetry-compressed double factorization*, arXiv:2403.03502。<https://arxiv.org/html/2403.03502>。特にSec.2.3–2.4のsymmetry shift、tensor fitting、正則化と1-normの関係。有限shifted cutoffをSCDF全体と同じ比較と呼ばない。
- [P3] I. D. Kivlichan et al., *Improved Fault-Tolerant Quantum Simulation of Condensed-Phase Correlated Electrons via Trotterization*, arXiv:1902.10673 / Quantum 4,296 (2020)。<https://arxiv.org/abs/1902.10673>。今回はabstractと書誌を確認。具体的HWP算術の根拠は[P4]本文も用いた。
- [P4] A. Kan and B. C. B. Symons, *Resource-optimized fault-tolerant simulation of the Fermi-Hubbard model and high-temperature superconductor models*, arXiv:2411.02160v2 (2025-08-11)。<https://arxiv.org/html/2411.02160v2>。Sec.IIのbaseline/catalyzed HWP、adder・回転・catalyst・workspaceの会計。物理的に異なるdensity二次式へそのまま費用を転記しない。
- [P5] G. H. Low, Y. Su, Y. Tong, M. C. Tran, *On the complexity of implementing Trotter steps*, arXiv:2211.09133。<https://arxiv.org/abs/2211.09133>。今回はabstract・書誌と本研究の近接位置を確認。一般Jの可換性だけから安い実装を主張しない。
- [P6] S. Bravyi, D. P. DiVincenzo, D. Loss, *Schrieffer-Wolff transformation for quantum many-body systems*, arXiv:1105.0675。<https://arxiv.org/abs/1105.0675>。今回はabstract・書誌を参照。N3はSW変換を新たに実装した研究ではない。
- [P7] G. Dusson, I. M. Sigal, B. Stamm, *The Feshbach-Schur map and perturbation theory*, arXiv:2105.02058 (2021)。<https://arxiv.org/abs/2105.02058>。rigorous eigenvalue/eigenfunction estimatesとfixed-point方式の既知性を確認。今回のblock比較の導出は本文中に独立に記述した。
- [P8] D. B. Chamaki et al., *Self-consistent Quantum Iteratively Sparsified Hamiltonian method (SQuISH): A new algorithm for efficient Hamiltonian simulation and compression*, arXiv:2211.16522v1 (2022)。<https://arxiv.org/abs/2211.16522>。abstractのHamiltonianと近似ground-state情報を併用する縮約を比較。全アルゴリズムとの厳密同値性の監査を完了したとは主張しない。

## 12. 再現用の独立チェック

以下は今回GPTが実行したコードと実際の出力である。原science sourceやnative compilerをimport・起動しない。出典付きの固定数値を明示的に転記し、N1の有理数スペクトル計算、N2の固定Sに対するLP、N3の同じ6-mode・2粒子入力の独立診断だけを行う。

N3の真の固有値・結合normは**診断専用**。次のconstructorへ入力してはならない。N1は保存済みPF診断とcountを使う事後感度であり、元結果の再分類・新しいnative性能結果ではない。N2のLPは最適化原理の変更を確認するための一回の固定S計算で、S探索や新しい量子回路ではない。

実行例：本付録のコードを `review_checks.py` として保存し、`python review_checks.py --output review_checks.json`。NumPy/SciPyを用いる。元batchと同じPython/packagesでのbyte再現を要求するものではない。

### Code SHA-256と結果SHA-256

```json
{
  "review_checks.py": "08b307150d49d8030c52bc09a592b0c22369160547c66e25c831707e03856351",
  "review_checks.json": "7908b036291d1682253130f7e7b7ce8fd5ca93ca34aeda9381d10dc240ccf827"
}
```

### 実行コード

```python
"""Review-only checks on frozen N1/N2/N3 inputs (2026-10-11).
No repository imports, circuit generation/compilation, trajectory sampling,
new molecular inputs, or repository writes. Exact eigenvalues in N3 are
post-hoc diagnostics only and MUST NOT be supplied to a deployable selector.
Usage: python review_checks.py --output review_checks.json
"""
from __future__ import annotations
import argparse
import itertools
import json
import math
import platform
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import scipy
from scipy.optimize import linprog

REPO = 'HIROMU1015/Partially-Randomized-Trotter'
COMMIT = 'ae7434452450e80ec007d65417335e18a063fa1e'
SOURCE = 'ed335a007bee6966a37f3026ac188534e526b5b9'
ROOT = f'https://github.com/{REPO}/blob/{COMMIT}/'
REFERENCES = {
    'csv': ROOT+'artifacts/hamiltonian_algorithm_design/2026-10-11/precision_resource_comparison.csv',
    'n3': ROOT+'artifacts/hamiltonian_algorithm_design/2026-10-11/run3/checkpoint_N3_coefficient_feasibility.json',
    'source': f'https://github.com/{REPO}/blob/{SOURCE}/src/trottertracks/representation_exploration/algorithm_design.py',
}

def shots(bias: float) -> int | None:
    margin = .04 / math.sqrt(2) - bias
    return None if margin <= 0 else math.ceil(2*math.log(80)/margin**2)

def n1_check() -> dict:
    eta = [F(1), F(1002,1000), F(-4,10), F(-397,1000)]
    approximations = {
        'planted_local_gauge': [F(1001,1000), F(1001,1000), F(-797,2000), F(-797,2000)],
        'dense_group_frame': [eta[0], eta[1], F(-797,2000), F(-797,2000)],
    }
    # Transcribed from frozen CSV, not generated by the original runner.
    entries = {
        'planted_local_gauge': {
            'old_model_bound': .007007000000000472,
            'exact_shots_q2': 16201, 'exact_counts_q2': [1250,832],
            'exact_bias_q2': .005025652524517219,
            'rows': [(1,.02511909087763216,566,400), (2,.009226637082866038,1126,800),
                     (4,.005447388442105133,2246,1600), (8,.004514228023895625,4486,3200)]},
        'dense_group_frame': {
            'old_model_bound': .004204200000000004,
            'exact_shots_q2': 16296, 'exact_counts_q2': [1310,848],
            'exact_bias_q2': .005093300912958767,
            'rows': [(1,.023987697504043688,626,416), (2,.007614374149821832,1246,832),
                     (4,.0037791918672919027,2486,1664), (8,.002835682645730598,4966,3328)]},
    }
    result = {}
    for name, approx in approximations.items():
        subset_errors = []
        for subset in itertools.combinations(range(4),2):
            error = F(7,10)*abs(sum(eta[i] for i in subset)**2 - sum(approx[i] for i in subset)**2)
            subset_errors.append({'occupied': list(subset), 'weighted_error': str(error)})
        sharp = max(F(x['weighted_error']) for x in subset_errors)
        e = entries[name]
        assert shots(e['exact_bias_q2']) == e['exact_shots_q2']
        ref_rz = e['exact_shots_q2']*e['exact_counts_q2'][0]
        ref_cx = e['exact_shots_q2']*e['exact_counts_q2'][1]
        rows=[]
        for q,bias,rz,cx in e['rows']:
            pf = bias - .6*e['old_model_bound']
            new_bias=pf+.6*float(sharp)
            n=shots(new_bias)
            assert n is not None
            rows.append({'q': q, 'saved_pf_diagnostic': pf, 'posthoc_bias': new_bias,
                         'posthoc_shots': n, 'saved_xy_rz': rz, 'saved_xy_cx': cx,
                         'posthoc_work_rz': n*rz, 'posthoc_work_cx': n*cx,
                         'ratio_rz_to_saved_best_exact': n*rz/ref_rz,
                         'ratio_cx_to_saved_best_exact': n*cx/ref_cx})
        result[name]={'rational_bound':str(sharp),'bound':float(sharp),
                      'six_subset_errors':subset_errors,'sensitivity_rows':rows}
    result['interpretation']=(
        'Coefficient-only exact ideal-spectrum factor bound; only one changed factor per selected case. '
        'Uses saved PF diagnostics and counts for a post-hoc accounting sensitivity, not a new science run '
        'or outward-rounded certification of the executed floating-point circuit.')
    return result

def n2_check() -> dict:
    s=np.array([[1,0],[1,1],[0,1],[-1,0]],float)
    k=np.array([[.3,-.1],[-.1,.2]])
    original=s@k@s.T
    j=original.copy();j[0,3]+=.001;j[3,0]+=.001
    matrices=[np.outer(s[:,0],s[:,0]),
              np.outer(s[:,0],s[:,1])+np.outer(s[:,1],s[:,0]),
              np.outer(s[:,1],s[:,1])]
    a=np.column_stack([m.ravel() for m in matrices])
    ls=np.linalg.lstsq(a,j.ravel(),rcond=None)[0]
    # Standard entrywise L1 LP for this SAME S, not a new S-search.
    objective=np.r_[np.zeros(3),np.ones(16)]
    aub=np.block([[a,-np.eye(16)],[-a,-np.eye(16)]])
    bub=np.r_[j.ravel(),-j.ravel()]
    lp=linprog(objective,A_ub=aub,b_ub=bub,
               bounds=[(None,None)]*3+[(0,None)]*16,method='highs')
    if not lp.success:
        raise RuntimeError(lp.message)
    ls_res=float(np.abs(a@ls-j.ravel()).sum())
    lp_res=float(np.abs(a@lp.x[:3]-j.ravel()).sum())
    # Exact lower bound: entries 00,33,03,30 alone give
    # 2|3/10-k00| + 2|k00-299/1000| >= 1/500.
    exact_lower=2*abs(F(3,10)-F(299,1000))
    assert exact_lower == F(1,500)
    assert abs(lp_res-float(exact_lower)) < 1e-12
    return {'S':s.astype(int).tolist(),'input_J':j.tolist(),
            'least_squares_coefficients':ls.tolist(),'least_squares_entry_l1':ls_res,
            'l1_fit_coefficients':lp.x[:3].tolist(),'l1_fit_entry_l1':lp_res,
            'exact_fixed_S_optimum':str(exact_lower),'budget':.0021,
            'interpretation':'Restores feasibility, not proof of circuit savings or novel optimization.'}

def annihilators(n: int) -> list[np.ndarray]:
    operators=[]
    for m in range(n):
        a=np.zeros((1<<n,1<<n))
        for b in range(1<<n):
            if b>>m&1:
                a[b^(1<<m),b]=(-1)**((b&((1<<m)-1)).bit_count())
        operators.append(a)
    return operators

def n3_check() -> dict:
    aa=annihilators(6)
    nn=[a.T@a for a in aa]
    hopping=[(1,2,.2),(0,3,.1),(2,4,.08),(3,5,.04),(4,5,.05)]
    sector=[b for b in range(64) if b.bit_count()==2]
    rows=[]
    for name,external in [('gapped',[6.,8.]),('small_gap',[.4,.6])]:
        es=[-.2,-.1,.1,.25]+external
        h=sum(e*n for e,n in zip(es,nn))+.3*nn[0]@nn[1]
        h+=sum(v*(aa[i].T@aa[j]+aa[j].T@aa[i]) for i,j,v in hopping)
        full=float(np.linalg.eigvalsh(h[np.ix_(sector,sector)])[0])
        stages=[]
        for active in (4,5,6):
            p=[b for b in sector if b<(1<<active)]
            q=[b for b in sector if b not in p]
            a=float(np.linalg.eigvalsh(h[np.ix_(p,p)])[0])
            c=beta=delta=None
            if q:
                c=float(np.linalg.eigvalsh(h[np.ix_(q,q)])[0])
                beta=float(np.linalg.norm(h[np.ix_(p,q)],2))
                gap=c-a
                delta=2*beta**2/(math.sqrt(gap**2+4*beta**2)+gap)
            stages.append({'active_modes':active,'sector_dimension':len(p),
                           'active_ground_diagnostic':a,'actual_energy_shift_diagnostic':a-full,
                           'exact_C_min_ORACLE_DIAGNOSTIC':c,'exact_B_norm_ORACLE_DIAGNOSTIC':beta,
                           'ideal_scalar_bound_ORACLE_DIAGNOSTIC':delta})
        rows.append({'context':name,'full_ground_diagnostic':full,'stages':stages})
    assert abs(rows[0]['full_ground_diagnostic']-(-.286627834803626))<1e-12
    assert abs(rows[1]['full_ground_diagnostic']-(-.29434978849697374))<1e-12
    assert rows[1]['stages'][0]['actual_energy_shift_diagnostic'] > .004
    assert rows[1]['stages'][1]['actual_energy_shift_diagnostic'] < .004
    assert rows[1]['stages'][1]['ideal_scalar_bound_ORACLE_DIAGNOSTIC'] > .004
    return {'rows':rows,'target_energy_tolerance':.004,
            'interpretation': 'New independent fixed-input diagnostics, NOT certified selector inputs. '
             'Five active modes suffice in actual energy for the small-gap toy, yet the original lower-bound '
             'selector returns six. Even exact a,c,beta do not close the coarse scalar bound at five modes.'}

def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    out={'review_date':'2026-10-11','repository':REPO,'result_commit':COMMIT,'science_source':SOURCE,
         'scope':'review-only fixed-input analysis; no production runner/recompilation/new contexts',
         'references':REFERENCES,'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},
         'N1':n1_check(),'N2':n2_check(),'N3':n3_check()}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('REVIEW_CHECKS_COMPLETE:',args.output)

if __name__=='__main__':
    main()
```

### 今回の出力

```json
{
  "review_date": "2026-10-11",
  "repository": "HIROMU1015/Partially-Randomized-Trotter",
  "result_commit": "ae7434452450e80ec007d65417335e18a063fa1e",
  "science_source": "ed335a007bee6966a37f3026ac188534e526b5b9",
  "scope": "review-only fixed-input analysis; no production runner/recompilation/new contexts",
  "references": {
    "csv": "https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ae7434452450e80ec007d65417335e18a063fa1e/artifacts/hamiltonian_algorithm_design/2026-10-11/precision_resource_comparison.csv",
    "n3": "https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ae7434452450e80ec007d65417335e18a063fa1e/artifacts/hamiltonian_algorithm_design/2026-10-11/run3/checkpoint_N3_coefficient_feasibility.json",
    "source": "https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ed335a007bee6966a37f3026ac188534e526b5b9/src/trottertracks/representation_exploration/algorithm_design.py"
  },
  "environment": {
    "python": "3.13.5",
    "numpy": "2.3.5",
    "scipy": "1.17.0"
  },
  "N1": {
    "planted_local_gauge": {
      "rational_bound": "3381/1600000",
      "bound": 0.002113125,
      "six_subset_errors": [
        {
          "occupied": [
            0,
            1
          ],
          "weighted_error": "0"
        },
        {
          "occupied": [
            0,
            2
          ],
          "weighted_error": "3367/1600000"
        },
        {
          "occupied": [
            0,
            3
          ],
          "weighted_error": "16877/40000000"
        },
        {
          "occupied": [
            1,
            2
          ],
          "weighted_error": "16863/40000000"
        },
        {
          "occupied": [
            1,
            3
          ],
          "weighted_error": "3381/1600000"
        },
        {
          "occupied": [
            2,
            3
          ],
          "weighted_error": "0"
        }
      ],
      "sensitivity_rows": [
        {
          "q": 1,
          "saved_pf_diagnostic": 0.02091489087763188,
          "posthoc_bias": 0.022182765877631877,
          "posthoc_shots": 235414,
          "saved_xy_rz": 566,
          "saved_xy_cx": 400,
          "posthoc_work_rz": 133244324,
          "posthoc_work_cx": 94165600,
          "ratio_rz_to_saved_best_exact": 6.579560471575829,
          "ratio_cx_to_saved_best_exact": 6.985976649114727
        },
        {
          "q": 2,
          "saved_pf_diagnostic": 0.005022437082865755,
          "posthoc_bias": 0.006290312082865755,
          "posthoc_shots": 18118,
          "saved_xy_rz": 1126,
          "saved_xy_cx": 800,
          "posthoc_work_rz": 20400868,
          "posthoc_work_cx": 14494400,
          "ratio_rz_to_saved_best_exact": 1.007388087155114,
          "ratio_cx_to_saved_best_exact": 1.075313489670628
        },
        {
          "q": 4,
          "saved_pf_diagnostic": 0.0012431884421048502,
          "posthoc_bias": 0.00251106344210485,
          "posthoc_shots": 13194,
          "saved_xy_rz": 2246,
          "saved_xy_cx": 1600,
          "posthoc_work_rz": 29633724,
          "posthoc_work_cx": 21110400,
          "ratio_rz_to_saved_best_exact": 1.463303450404296,
          "ratio_cx_to_saved_best_exact": 1.566142640767664
        },
        {
          "q": 8,
          "saved_pf_diagnostic": 0.0003100280238953427,
          "posthoc_bias": 0.0015779030238953426,
          "posthoc_shots": 12288,
          "saved_xy_rz": 4486,
          "saved_xy_cx": 3200,
          "posthoc_work_rz": 55123968,
          "posthoc_work_cx": 39321600,
          "ratio_rz_to_saved_best_exact": 2.7220032343682488,
          "ratio_cx_to_saved_best_exact": 2.91719884337624
        }
      ]
    },
    "dense_group_frame": {
      "rational_bound": "50757/40000000",
      "bound": 0.001268925,
      "six_subset_errors": [
        {
          "occupied": [
            0,
            1
          ],
          "weighted_error": "0"
        },
        {
          "occupied": [
            0,
            2
          ],
          "weighted_error": "50463/40000000"
        },
        {
          "occupied": [
            0,
            3
          ],
          "weighted_error": "50589/40000000"
        },
        {
          "occupied": [
            1,
            2
          ],
          "weighted_error": "50631/40000000"
        },
        {
          "occupied": [
            1,
            3
          ],
          "weighted_error": "50757/40000000"
        },
        {
          "occupied": [
            2,
            3
          ],
          "weighted_error": "0"
        }
      ],
      "sensitivity_rows": [
        {
          "q": 1,
          "saved_pf_diagnostic": 0.021465177504043684,
          "posthoc_bias": 0.022226532504043682,
          "posthoc_shots": 238828,
          "saved_xy_rz": 626,
          "saved_xy_cx": 416,
          "posthoc_work_rz": 149506328,
          "posthoc_work_cx": 99352448,
          "ratio_rz_to_saved_best_exact": 7.0033730939452195,
          "ratio_cx_to_saved_best_exact": 7.189549930067339
        },
        {
          "q": 2,
          "saved_pf_diagnostic": 0.00509185414982183,
          "posthoc_bias": 0.00585320914982183,
          "posthoc_shots": 17419,
          "saved_xy_rz": 1246,
          "saved_xy_cx": 832,
          "posthoc_work_rz": 21704074,
          "posthoc_work_cx": 14492608,
          "ratio_rz_to_saved_best_exact": 1.016690931507568,
          "ratio_cx_to_saved_best_exact": 1.048744454015802
        },
        {
          "q": 4,
          "saved_pf_diagnostic": 0.0012566718672919002,
          "posthoc_bias": 0.0020180268672919,
          "posthoc_shots": 12704,
          "saved_xy_rz": 2486,
          "saved_xy_cx": 1664,
          "posthoc_work_rz": 31582144,
          "posthoc_work_cx": 21139456,
          "ratio_rz_to_saved_best_exact": 1.479412547264912,
          "ratio_cx_to_saved_best_exact": 1.5297375904261725
        },
        {
          "q": 8,
          "saved_pf_diagnostic": 0.00031316264573059537,
          "posthoc_bias": 0.0010745176457305954,
          "posthoc_shots": 11838,
          "saved_xy_rz": 4966,
          "saved_xy_cx": 3328,
          "posthoc_work_rz": 58787508,
          "posthoc_work_cx": 39396864,
          "ratio_rz_to_saved_best_exact": 2.7538021787766023,
          "ratio_cx_to_saved_best_exact": 2.8509183871953763
        }
      ]
    },
    "interpretation": "Coefficient-only exact ideal-spectrum factor bound; only one changed factor per selected case. Uses saved PF diagnostics and counts for a post-hoc accounting sensitivity, not a new science run or outward-rounded certification of the executed floating-point circuit."
  },
  "N2": {
    "S": [
      [
        1,
        0
      ],
      [
        1,
        1
      ],
      [
        0,
        1
      ],
      [
        -1,
        0
      ]
    ],
    "input_J": [
      [
        0.3,
        0.19999999999999998,
        -0.1,
        -0.299
      ],
      [
        0.19999999999999998,
        0.3,
        0.1,
        -0.19999999999999998
      ],
      [
        -0.1,
        0.1,
        0.2,
        0.1
      ],
      [
        -0.299,
        -0.19999999999999998,
        0.1,
        0.3
      ]
    ],
    "least_squares_coefficients": [
      0.29968,
      -0.09984000000000001,
      0.19992
    ],
    "least_squares_entry_l1": 0.0035999999999999366,
    "l1_fit_coefficients": [
      0.3,
      -0.1,
      0.2
    ],
    "l1_fit_entry_l1": 0.0020000000000000018,
    "exact_fixed_S_optimum": "1/500",
    "budget": 0.0021,
    "interpretation": "Restores feasibility, not proof of circuit savings or novel optimization."
  },
  "N3": {
    "rows": [
      {
        "context": "gapped",
        "full_ground_diagnostic": -0.28662783480362597,
        "stages": [
          {
            "active_modes": 4,
            "sector_dimension": 6,
            "active_ground_diagnostic": -0.28598700982242525,
            "actual_energy_shift_diagnostic": 0.0006408249812007161,
            "exact_C_min_ORACLE_DIAGNOSTIC": 5.77514384778861,
            "exact_B_norm_ORACLE_DIAGNOSTIC": 0.08944271909999159,
            "ideal_scalar_bound_ORACLE_DIAGNOSTIC": 0.0013195984128910837
          },
          {
            "active_modes": 5,
            "sector_dimension": 10,
            "active_ground_diagnostic": -0.28661095938257414,
            "actual_energy_shift_diagnostic": 1.6875421051831818e-05,
            "exact_C_min_ORACLE_DIAGNOSTIC": 7.77610851498021,
            "exact_B_norm_ORACLE_DIAGNOSTIC": 0.06403124237432849,
            "ideal_scalar_bound_ORACLE_DIAGNOSTIC": 0.0005084812214890318
          },
          {
            "active_modes": 6,
            "sector_dimension": 15,
            "active_ground_diagnostic": -0.28662783480362597,
            "actual_energy_shift_diagnostic": 0.0,
            "exact_C_min_ORACLE_DIAGNOSTIC": null,
            "exact_B_norm_ORACLE_DIAGNOSTIC": null,
            "ideal_scalar_bound_ORACLE_DIAGNOSTIC": null
          }
        ]
      },
      {
        "context": "small_gap",
        "full_ground_diagnostic": -0.2943497884969735,
        "stages": [
          {
            "active_modes": 4,
            "sector_dimension": 6,
            "active_ground_diagnostic": -0.28598700982242525,
            "actual_energy_shift_diagnostic": 0.008362778674548266,
            "exact_C_min_ORACLE_DIAGNOSTIC": 0.16447187700291735,
            "exact_B_norm_ORACLE_DIAGNOSTIC": 0.08944271909999159,
            "ideal_scalar_bound_ORACLE_DIAGNOSTIC": 0.01710978613958609
          },
          {
            "active_modes": 5,
            "sector_dimension": 10,
            "active_ground_diagnostic": -0.294087596218998,
            "actual_energy_shift_diagnostic": 0.0002621922779755259,
            "exact_C_min_ORACLE_DIAGNOSTIC": 0.37352246149264984,
            "exact_B_norm_ORACLE_DIAGNOSTIC": 0.06403124237432849,
            "ideal_scalar_bound_ORACLE_DIAGNOSTIC": 0.006085831994298733
          },
          {
            "active_modes": 6,
            "sector_dimension": 15,
            "active_ground_diagnostic": -0.2943497884969735,
            "actual_energy_shift_diagnostic": 0.0,
            "exact_C_min_ORACLE_DIAGNOSTIC": null,
            "exact_B_norm_ORACLE_DIAGNOSTIC": null,
            "ideal_scalar_bound_ORACLE_DIAGNOSTIC": null
          }
        ]
      }
    ],
    "target_energy_tolerance": 0.004,
    "interpretation": "New independent fixed-input diagnostics, NOT certified selector inputs. Five active modes suffice in actual energy for the small-gap toy, yet the original lower-bound selector returns six. Even exact a,c,beta do not close the coarse scalar bound at five modes."
  }
}
```

## 13. 最終状態

レビュー完了。必要な主要資料は取得でき、現時点の不可欠な追加push依頼はない。中心テーマ未採択。新しいscience run・Track A/B統合・旧STOP変更はしていない。

推奨はN1の誤差-aware選択の限定再設計、N3の構造を残すboundの方法論検討、N2のnative小系scan保留。新規性と総資源改善の両方が成立する候補を、次の証拠で判断する。
