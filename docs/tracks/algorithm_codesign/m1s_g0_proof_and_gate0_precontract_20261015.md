# Track B／M1-S：G0証明義務・Gate 0評価の候補契約

2026-10-15 JST。Status: **`M1S_G0_PRECONTRACT_PREPARED_NOT_ADOPTED_MANDATORY_STOP`**。

これは既存レビューを実行前の確認項目へ対応させたDecision Packetである。**研究契約は未採択、M1-S正式採択・G0開始・Gate 0科学計算は未承認**。本書の「受理」は将来の契約候補の要件であり、今回の科学的GOではない。数式は出典付きの既存結果または未採択の定義候補として示す。新しい証明は行っていない。

## 1. 目的・承認範囲・用語

ユーザー提供の「Track B／M1-S G0実施前の研究契約整理」に基づき、原本保存、固定資料照合、数学的対象・証明義務・数値評価候補・未決定事項の整理、文書と索引のcommit/pushを行う。[共通指示][COMMON]のCodex先行の技術整理に該当し、新しいGPT独立科学レビューではない。

本書では **G0-T** を新しい理論検討、**Gate 0** を分子系等の数値診断、**G0本段階** を部分乱択・RPE保証への接続と呼ぶ。過去のG0-preは資料調査であり、これらの開始を意味しない。いずれも今回実行していない。

承認されたのは**契約候補をレビュー可能にする文書準備**までである。新定理の確定、仮説の採択、数値simulation・行列評価・対角化、sampling・trajectory・operator guard、回路生成・compile・synthesis、費用再集計・forecast・最適化、既存classificationやauthorization/markerの変更、mandatory STOP解除は含まれない。

## 2. 固定GitHub資料・取得状況

Repository: `HIROMU1015/Partially-Randomized-Trotter`。次の7ファイルはGitHub connectorからbase64で取得し、復号したbytesと各固定commitのGit blobを照合した。すべて一致。各ファイルのbytes・SHA256・blob SHAは[今回の参照索引][INDEX]に記録する。

| ID | 固定commit | 資料・今回の役割 |
|---|---|---|
| GPT-FINAL | `4f5114e137563bb871c9f881f9eadab0c180bdd4` | [最新GPTレビュー原本][GPT]。数学的主張の証拠水準・留保の主要参照 |
| G0-PRE | `8b7666d0270f7fac01e430e195bef7b58ac0116c` | [G0-pre Decision Packet][PRE]。近接先行研究・control・費用の既存調査 |
| G0-INDEX | 同上 | [既存evidence index][PREINDEX]。調査範囲、版、未取得、旧原本identity |
| G0-LIT | 同上 | [8文献locator][PRELIT]。版・ページ・式・PDF hash |
| GPT-1 | `41fa1eb480728cd0691b00bb41dd4e5dc4ba6b03` | [前回GPTレビュー][GPT1]。当時のC判断と長時間反復への留保 |
| P0 | `fa91ffb3d4880d767e55f0f661e8ade294b4c9fe` | [P0][P0]。既知qDRIFT・PR・controlの基礎照合 |
| P0.5 | `ed3ddeacc88e998e7bc1d8c0db8163e106312791` | [P0.5][P05]。既存定理の仮定、H6 v5境界、費用field不足 |

P0/P0.5/G0-preの文献検索・科学監査は繰り返さず、既存locatorと判断を参照した。本書の先行研究欄は**既存調査の対応索引**であり、今回一次論文全体を独立再読・再証明した記録ではない。CHENはarXiv v4と2021年出版PDFを区別する。近接文献が見つからないことを新規性の証明にしない。

Claude完成版 `レビューinner.md` 自体は今回提供されていない。§4のClaude側の主張は、GPT-FINALが直接読んだと記録する原本の**GPTによる引用・評価の範囲**に限定する。G0-preの別名 `レビュー_inner(1).md` と同一原本と断定しない。今回の必須GPT原本は取得できており、Claude全文の再審査は作業範囲外。Claude原本の新しい箇所が判断に必要になれば、その出所を確保してから扱う。

## 3. 最新GPTレビュー原本の同一性

- ユーザーが本文で指定した提供元: repository直下 `track_b_m1s_claude_final_gpt_independent_scientific_review_20261015.md`。
- 保存先: `docs/research/track_b_m1s_claude_final_gpt_independent_scientific_review_20261015.md`。
- **46,341 bytes**、SHA256 **`d8e40225b3b2ddbbe79ed31a5eabf397803689d36f4c5b3021081a990d2ebdb5`**。
- 原本保存commit: **`4f5114e137563bb871c9f881f9eadab0c180bdd4`**。Git blob: `91fb06e6de2a2c00289173c41be8f34aa0137016`。
- 保存前に当該公開pathの全branch履歴を確認し、既存保存がないことを確認。提供元からbyte copyし、push後GitHubから再取得してbyte一致を確認した。本文、改行、数式、原本に記された当時の未承認状態を変更していない。

原本内の「repository変更は未承認」は**レビュー作成時点の権限記録**である。今回の文書公開許可は今回のユーザー指示による。原本の科学判断を今回のCodex判断で上書きしない。旧GPT-1原本は41,818 bytes、別hash・別commitであり、今回の原本と区別する。

## 4. Claude／GPTの一致点・相違点

出典: GPT-FINAL §§0.1–0.3, 3–8。以下はレビュー記録の対応整理であり、再評価ではない。

| 論点 | Claude側としてGPT原本に記録された主張 | GPTの評価・留保 | 契約整理への反映 |
|---|---|---|---|
| 条件付きB | 固定超演算子・二準位例を根拠にM1-Sを条件付き推奨 | 狭い理論検証候補として条件付き支持。方法GO・正式採択ではない | 推奨の存在だけ記録。採択欄はfalse |
| 完全乱択恒等式 | 共有±の回文word、共分散、分散分解 | 指定lawでは支持 | §5の確認済み層へ限定 |
| 長時間反復 | 固有値を先に摂動しN乗する方針 | 合理的な道筋。一様剰余・射影・非正規性の保証は未完 | Q1/Q2として残す |
| 二準位の絶対値 | `abs(lambda)=abs(cos(theta))` | 根号が実数のsmall-angle域に限定して支持 | 任意角の恒等式にしない |
| shot freshness | 毎shot新列が平均恒等式にも必要と読める記述 | 無条件平均には不要。列再利用時の分散・集中は別 | 平均と統計の契約を分離 |
| 部分乱択のrho | full-H状態での候補量をGate 0の鍵量に提案 | 実際のordered partial mapから未導出 | proxy候補、未認証。科学的GO/STOPには使わない |
| Gate 0の順序・閾値 | 早期数値診断、0.7／0.85等の目安 | 理論契約先行を推奨、閾値未採択 | §12で両案を並列提示し、Codexは選ばない |
| 化学・native費用 | 有用性を検討する価値 | sector外結合、pair-sum、signal floor、同情報・同費用が未確立 | 数値候補と資源保証を別段階にする |

GPT-1のCとGPT-FINALの条件付きBは、追加資料に対する**時点の異なる判断**として保存する。Cを誤りとして消さず、Bを正式採択として記録しない。Parent allocation D＋E、M1-A補助／technical-note位置付け、M2代替、既存STOPは保持する。

## 5. 既存レビューで厳密に確認された数学的結果

以下はGPT-FINAL §§3.1–3.4, 4.4の引用整理。今回新たに証明したものではない。主に**完全乱択 `H_D=0`、共通角、Hermitian involution `P_j^2=I`、位置間IID**の範囲である。

右端先作用、`A_j=exp(+i theta P_j)`、位置列 `omega=(j_1,...,j_N)` とする。

\[
B_1=A_{j_N}\cdots A_{j_1},\qquad
B_0=A_{j_N}^{\dagger}\cdots A_{j_1}^{\dagger},\qquad
\mathbb E[B_0^\dagger B_1]=\Phi_\theta^N(I),\quad
\Phi_\theta(X)=\sum_jp_j A_j X A_j.
\]

同順sign reversalと**全wordのadjoint**は異なる。上式の右側は `A_j X A_j^dagger` でなく、Φ自身をCPTPと呼ばない。拡張系のchannelのoff-diagonal coherence blockとしての解釈と区別する。

\[
h=\sum_jp_jP_j,\quad M=\cos\theta I+i\sin\theta h,\quad
\Phi_\theta(X)=MXM-\sin^2\theta\,\mathcal C(X),\quad
\mathcal C(X)=\sum_jp_j(P_j-h)X(P_j-h).
\]

`h|k>=e_k|k>`、`v_km=E_j |<m|(P_j-h)|k>|^2` に対して、

\[
V_k=1-e_k^2=\sum_m v_{km}=v_{kk}+w_k,\qquad
w_k=\mathbb E_j[1-\langle k|P_j|k\rangle^2],\qquad
0\le v_{kk}\le V_k.
\]

`v_kk` は項選択に伴う期待値のばらつき、`w_k` は項内の量子分散の平均である。全Hの任意状態にこの**h固有状態用の記号**を無条件に移さない。

一列をdrawしてK shot再利用する場合も、列drawを含む無条件平均は `E_omega z(omega)` に等しい。ただし条件付き独立測定なら、GPT-FINAL §3.4の全分散式

\[
\operatorname{Var}(\widehat Y_K)
=\operatorname{Var}_\omega(\mathbb E[Y|\omega])
+K^{-1}\mathbb E_\omega\operatorname{Var}(Y|\omega)
\]

の初項が残る。fresh IIDは簡単な独立shot会計の十分条件であり、無条件平均等式の必要条件ではない。

二準位参照は `P in {Z,X}`、各確率1/2、`h=(X+Z)/2`。GPT-FINAL §4.4の表示固有値について、**`2 cos^2(theta)-sin^2(theta)/4 >= 0` の領域**で `abs(lambda_±)=abs(cos(theta))` が成立する。この存在例を化学・部分乱択・多round RPE・全角度の一般保証にしない。

## 6. 未証明の主張・追加証拠が必要な主張

| 証拠水準 | 対象 | 現時点の境界・出典 |
|---|---|---|
| 条件付きの先導漸近 | 完全乱択の減衰係数とrho_infinity | GPT-FINAL §4.2–4.3。有限時間誤差付き一様定理は未完成 |
| 証明方針 | 固有値摂動→N乗、`N theta^2 <= D` での剰余管理 | 射影・条件数・thetaに依存する隔離・次元依存を未制御 |
| 未証明 | 低周波weightからの位相bias、pair-sum共鳴・near-degeneracyの保証 | `e_m-e_k`のgap表だけでは一般に不足。GPT-FINAL §4.5 |
| 未証明 | 非可換Dを挟むΨの係数・rho候補 | full-Hとh_Rの固有basis差、outer PFとrandomの交差項。§5.1 |
| 未証明 | overlap・complex weights・減衰・signal floor付きRMS energy error | 固有値の位相だけで実信号・branch成功は保証されない。§5.2 |
| 新しい科学的証拠が必要 | 化学系のrho、v/w比、低周波weight、sector外結合 | 今回値を計算・転記していない。§5.3–5.4 |
| 未取得／未確立 | M1-Sに整合するD/R費用割合・end-to-end利得 | P0.5 §4、GPT-FINAL §6。NOT_ESTABLISHED維持 |

`O(theta/Delta)`を分子サイズに一様な保証と呼ばない。固定Nの一次反復式を、`N theta^2=O(1)` で高次項を捨てたまま保証へ格上げしない。既存channel boundから鋭いcoherent-signal係数が自動的に得られたとも扱わない。

## 7. Controlled protocolの候補仕様：固定が必要な欄

本節は**未採択の候補契約**。§5の参照モデル以外について一つの実験lawを選んでいない。以下の欄が未記入なら、G0-Tのstatementも数値実行contractも完成扱いにしない。

### 7.1 Hamiltonian・D/R・branch

| 固定する欄 | 候補・既存参照 | 未確定事項／適用禁止 |
|---|---|---|
| `H=H_D(L_D)+H_R(L_D)` | 固定DF fragment prefixとone-body correctionの既存分割を参照。完全乱択対照はH_D=0 | 実Hamiltonian hash、DF rank/policy、順序、L_Dは未採択。DF低rank modelと電子Hamiltonian真値を区別 |
| 乱択部 | `H_R=lambda_R sum_j p_j P_j`、実係数の符号をHermitian involutionへ吸収、p_j>=0、総和1、共通角が参照条件 | Pauli/JWと一般involution、重複項の併合、identityの扱いを固定。DF fragment/Gaussian eventが単一Pauliと同じとはしない |
| 決定論block | exact exponentialを理論参照にする候補／既存ordered symmetric PFを使う候補 | forward/reverse half sweep、内部PF次数・substep・outer配置、各枝の符号・角度・scalarを未採択。exact-Dで得た保証を実装Dへ直結させない |
| controlled operator | `W_omega=diag(B_0(omega),B_1(omega))`、干渉信号 `⟨ψ¦B_0^dagger B_1¦ψ⟩` | control polarity、準備ancilla、X/Y読出しのS/S†・Y符号、energy符号の対応を明記 |
| gate共有±参照 | §5のB0/B1。同位置jは共通、negative branchは同じchronological順で各gateをinverse | `B0=B1^dagger`へ置換しない。full-block共有は別候補law |
| 比較law | ordinary、独立±、full-block共有±を区別 | 全方式を同物理taskへ対応させてから比較。独立±の負枝は正しい逆時間lawと逆順の条件を明示 |
| phase | system global phaseもcontrolではrelative phase。identity項を別phase台帳に保持 | projective同値では受理しない。既存H6の決定論directional＋ordinary tailをM1-S実装済みとしない |

Pauli-tail候補の `p_j=|a_j|/sum|a_j|` は既存qDRIFT型の候補規則であり、今回の科学的選択ではない。importance lawを変えるなら、共通角・重み補正・平均対象が変わるため別候補contractとして扱う。`lambda_R=0`、空tail、identityのみの境界を例外欄へ残す。

### 7.2 乱数の共有・独立性

| 軸 | 単一Φ反復の参照条件／候補 | 決定すべきこと |
|---|---|---|
| 同一gate位置の二branch | 同じjを共有、角度は符号反転 | branch同期の粒度、D配置との対応 |
| 異なるgate位置 | 同分布IID、replacementあり | permutation／without-replacement／相関列なら単一Φを使わない |
| step間 | 同じfull-step lawのIIDを候補とする | round内の角度・r・Dが変わる場合は位置依存写像。step再利用は別law |
| shot間 | fresh IID候補／複数shotで列を再利用する候補 | 後者はcluster数、各列のshot数、条件付き測定、経路分散と測定分散を分ける。両者は統計保証が異なる |
| X/Y測定間 | 独立列候補／paired列候補 | Pauli X/Y測定自体は別shot。共有するなら共分散・confidence会計を指定 |
| RPE round間 | 独立stream候補／明示したcoupling候補 | 適応性、履歴に条件付けたlaw、停止・再利用の規則。独立性はseed文字列の相違だけで証明しない |
| baseline間の対照実験 | paired seed候補は将来の比較設計 | 各方式のmarginal lawを保存。baseline pairingを実回路の二branch共有と混同しない |

### 7.3 時間・角度・イベント数の辞書候補

GPT-FINAL §§2.1, 4.3, 5.1–5.2とG0-pre §4.2に対応する。**以下の線形角規約は定義候補であり、実際の平均信号が理想時間発展に等しいとの主張ではない。**

- `T_rel`: 比較する干渉信号の相対時間。理想±では各枝の時間絶対値は `T_rel/2`。ordinaryは片枝が `T_rel`。物理的な各枝の時間とsignalの相対時間を別記する。
- 完全乱択・線形角候補は `N theta=lambda T_rel/2`。同じN・thetaをordinaryへ与えて時間が同じとしない。ハードウェアが `R_P(phi)=exp(-i phi P/2)` 規約なら参照A_jの符号と係数は `phi=-2 theta` と対応付ける。
- 部分乱択の候補辞書は、相対step幅 `delta_rel`、step数 `s`、一stepの乱択位置数 `r`、`T_rel=s delta_rel`。線形角のhalf-branch候補なら `theta=lambda_R delta_rel/(2r)`。**この辞書だけではDのordered word・Ψ・biasは定義されない**。
- `arctan`角／paired-Taylor由来角を使う候補は別欄とし、τ、位相補正、normalizerを出典とともに指定する。線形角の公式と混在させない。一般固有値のarctan歪みを一つの時間rescalingで厳密に消したとしない。
- 完全乱択参照の共有controlled位置はN、干渉word中には2N個のAが現れる。後者を物理gate数と数えない。部分候補ではr×sと、各stepに複数R occurrenceがある場合の総数を区別する。既存finite-RTEのr/Kを新qDRIFTのイベント数へ無条件転記しない。
- RPE各roundの `T_m, delta_m, s_m, r_m, theta_m`、最大時間、phase branch、乱数lawを一組で固定する。数値・round scheduleは今回選ばない。

### 7.4 完全乱択から部分乱択へ移るときの追加定義

GPT-FINAL §5.1にある候補写像を、そのまま未採択の一般形として参照する。

\[
\Psi(X)=\mathbb E_\omega[G_-(\omega)^\dagger XG_+(\omega)].
\]

`G_±`は**scalar phase・D half sweep・全R eventを含む一stepのordered branch word**として先に定義する必要がある。右端先作用とstep lawを揃え、IIDなら反復表現を検討できるという形式的事実と、減衰係数が完全乱択と一致することは別。`rev`が逆順かadjointかは記号ごとに宣言する。ΨにΦの固有basis、共分散係数、rhoをそのまま代入しない。

## 8. G0の証明義務と受理条件の草案

以下はGPT-FINAL §§4–8、P0.5 §2の未完事項を整理したもの。**Q1–Q5を解いていない**。将来の受理には、statementの適用域・仮定・定数依存・適用不能例を明記し、各項目を個別に判定することが必要。Q1を通してもQ3/Q4/Q5が通ったとしない。

### Q1：完全乱択の長時間・有限減衰予算

| 必須欄 | 候補契約 |
|---|---|
| 1. 主張対象 | 固定lawのΦに対し `z_N=⟨ψ¦Phi^N(I)¦ψ⟩` を近似する信号モデルと、`N theta^2<=D` の長時間範囲に一様な誤差。振幅・位相の保証対象を別記 |
| 2. 現在の事実 | §5の恒等式。GPT-FINAL §4.2は固有値を先に扱う方針を支持、§4.4に限定二準位例。一般一様定理はない |
| 3. 追加仮定候補 | 固定有限次元、bounded involution、対象modeの隔離、許容angle、D、射影条件数、初期stateの条件。どこまで緩和するか未採択 |
| 4. 未証明・受理に必要 | eigenvalueとeigenprojectionの双方、残余mode、有限theta/N、N=0等の境界、theta・gap・D・次元への定数依存を明示。signalが小さい場合のphase適用域も必要 |
| 5. 近接先行 | PR Lemma A.1/full-qDRIFT、CHENの平均と集中、Guのspectral perturbation。P0.5 §2とG0-LITの対象・norm差を維持。既知摂動論の適用だけならその差分を明記 |
| 6. 反例・STOP候補 | 固定N一次展開だけでは有限D保証にならない。射影や残差の一様性が示せなければ一般長時間claimを受理しない。二準位の角領域外を無条件例にしない |
| 7. 担当・権限 | 理論担当が証明、GPT/Claudeが重要科学判断、Codexは承認内の技術補助。**新理論実施の別途承認が必要** |

### Q2：隔離・共鳴・近縮退・低周波

| 必須欄 | 候補契約 |
|---|---|
| 1. 主張対象 | Q1/Q3に必要なsuperoperator modeの隔離と、結合weightを含む有効なスペクトル条件。低周波から位相biasを管理する条件 |
| 2. 現在の事実 | Φ_0(X)=MXMのmode固有値はmu_a mu_b。GPT-FINAL §4.5は `2e_k=e_a+e_b` のpair-sumとcouplingを区別。全Fock最小gapだけで成否を判断しない |
| 3. 追加仮定候補 | 単純隔離／縮退clusterのどちらを対象にするか、結合が零となる対称性、finite-step alias距離、sectorを跨ぐsupport、射影・resolventの条件。cutoff値は未採択 |
| 4. 未証明・受理に必要 | theta→0の隔離縮小、near-degenerate block、non-normal/Jordanの扱い、低周波汎関数の定義・誤差への換算。単一gap・v_km表で十分かも証明対象 |
| 5. 近接先行 | Gu SI S7–S16のchannel isolation/left-right weights、KGW Thm1の生成Hamiltonian列のgap、CHEN。異なるgap/normを同一視しない |
| 6. 反例・STOP候補 | 結合するpair-sum共鳴やaliasで対象mode分離が失われ、代替cluster保証もなければその保証を停止。小gapの存在だけで全M1-Sを棄却しない。sector内groundをh_Rの大域groundとしない |
| 7. 担当・権限 | 理論担当＋GPTの仮定選択。新証明は別途承認。分子のgap/weight取得はさらに**独立の科学計算承認** |

### Q3：部分乱択のordered controlled law

| 必須欄 | 候補契約 |
|---|---|
| 1. 主張対象 | §7で固定するG_±のΨと、実際の枝干渉・物理時間に整合する反復信号。Dを含む減衰・位相の候補モデル |
| 2. 現在の事実 | GPT-FINAL §5.1の一般写像・IID反復の方針。H6 v5は決定論directional＋ordinary tailであり、M1-S実装済みの根拠ではない |
| 3. 追加仮定候補 | Pauli-tail／一般event、exact D／内部PF、full-step law、sign/order/scalar、独立性、delta・r・thetaのjoint limit。候補の選択自体が未採択 |
| 4. 未証明・受理に必要 | 完全なordered定義からのmean対応、H_D→0極限との整合、outer/inner PF bias・交差項、full-H固有stateとh_Rの違い。rho^(H)を導けるか／代替が必要か |
| 5. 近接先行 | PR A.3/A.4のpartial RTEと独立±、Hagan–Wiebe/Composite QDrift、KIMのrandom insertion。channel保証・access差はP0.5/G0-pre参照 |
| 6. 反例・STOP候補 | G_-をG_+全体のadjointと取り違える、DF eventをP^2=Iへ置換する、phaseや時間が不一致なら当該contractを戻す。自然な条件で移送できなければ完全乱択ノートへ縮小／停止を判断 |
| 7. 担当・権限 | GPT/ユーザーが科学的protocol採択、理論担当が証明、Codexが技術整備。**採択・新理論実施・後続実装は別の承認対象** |

### Q4：RPEのRMS energy error

| 必須欄 | 候補契約 |
|---|---|
| 1. 主張対象 | 各roundの実信号から最終energy estimatorのRMS errorへ接続する保証。高確率errorを併記するなら別statement |
| 2. 現在の事実 | PR Thm B.2にはoverlap付きRMS保証があるが、そのまま部分M1-Sの定理ではない。GPT-FINAL §5.2はcomplex spectral weights・signal floorも要求 |
| 3. 追加仮定候補 | guiding overlapの下界、energy interval/alias-free条件、全roundの時間対応、位相・減衰・PF/synthesis bias、信号下界、shot law、X/Yとroundのconfidence配分 |
| 4. 未証明・受理に必要 | 対象以外の複素weight、branch誤選択、near-zero/abstention、統計tailがRMSへ与える寄与。rhoの先導係数だけでshot数を決めない。PF係数CとQPE/RPE統計誤差は別会計 |
| 5. 近接先行 | PR B.2/B13–B29、富士通Appendix A/B、Li Thm4.1のregister-QPE、KGW。各測定方式・overlap・normの仮定対応を要する |
| 6. 反例・STOP候補 | 小さい減衰でも鏡像modeや複素weightの干渉でsignal floorが消える場合は固有位相だけのRPE保証を受理しない。RMSを高確率のみで代替しない |
| 7. 担当・権限 | 理論担当＋GPTの誤差意味論判断、ユーザーの研究契約承認。仮想測定やRPE simulationは**別途計算承認後** |

### Q5：同条件の資源改善

| 必須欄 | 候補契約 |
|---|---|
| 1. 主張対象 | 同Hamiltonian・state・relative time・精度・confidence・情報アクセスでのend-to-end費用。イベント比とnative費用比を分離 |
| 2. 現在の事実 | 方向制御・qDRIFT/PR-RTE・HWP等は既知。富士通gamma=1/2は既知の費用モデル差。M1-S用D/R費用割合はNOT_ESTABLISHED |
| 3. 追加仮定候補 | ordinary/独立± partial-qDRIFT、PR型RTE、必要な決定論PFを含む強baseline群、共通の再最適化機会、primitive座標、誤差予算、前処理・online・state-prep/readout・ancillaの扱い |
| 4. 未証明・受理に必要 | Q3/Q4と一致するrequired events/shots、strict controlled phase、全wrapper価格、whole-circuit融合と費用帰属、合成精度・native overhead・不確かさ。oracle診断情報の無料転用禁止 |
| 5. 近接先行 | Campbell controlled qDRIFT、PR Appendix E/HWP、富士通B3–B5/STAR、MLMC/SCUのshot会計、P0.5費用field監査 |
| 6. 反例・STOP候補 | 小rhoの利得がD・shots・単価・準備費用で消えれば資源優位claimを受理しない。gamma差×rho×native仮定で無根拠な2倍/4倍を宣言しない。既知法と同保証ならclaimを限定 |
| 7. 担当・権限 | GPT/ユーザーが比較とcost座標を採択、Codexが承認された取得・compile/会計。**新forecast・合成・資源計算はすべて別途承認** |

## 9. Gate 0：評価候補と保存データの所在

### 9.1 利用可能な入力・既存証拠

以下は**データ可用性の目録**であり、Gate 0対象の選定ではない。JSONのfield、Git object、NPZのZIP member名とhashだけを確認した。数値arrayをloadして科学量を再計算していない。既存held-out H4 1.30 ÅのNPZ内部は開いていない。

| 候補系列・モデル識別 | 保存済み／今回確認した情報 | 欠けている情報・再利用条件 |
|---|---|---|
| H4 development [固定snapshot][H4NPZ]、[V1–V3記録][H4V13]。linear 1.0 Å、STO-3G、8 spin orbitals、DF rank12、旧PR-2新系列。prefix3/6/9は既存分割候補 | constant/one_body/lambdas/g_matrices、sector_basis_indices、state_vector/sector_state_vector、metadata_jsonを保存。raw SHA256 `3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a`、Hamiltonian hash `de7a549238e3a21f15a84018bef28440c345b31030282c01cf874f3d1d212424`。既存V1はN=4、Nα=Nβ=2、Sz=0の状態検査記録 | 各Pauli期待値・M1-S分散表・結合励起スペクトル・rhoは直接保存fieldとして確認していない。rank3/6/9を今回L_Dとして採択しない。旧pilotの別hashと混ぜない。M1-S delta窓は未設定 |
| H6 [保存snapshot receipt][H6RECEIPT]／[NPZ][H6NPZ]、[確認summary][H6SUMMARY]。linear 1.0 Å、STO-3G、12 spin orbitals、tol-only DF19、weighted Hermitian projection由来。既存費用scopeはT=0.8、prefix5/10/15/full19、delta0.1–0.8 | 同じ8種のNPZ member。raw SHA256 `99440a59d903dfd6330e786d84a956f1dd8a687a592295a1cd771c839769b005`、Hamiltonian hash `223628b06c794f4a2a7a84db532e19ed60915f7e86898de66347129a5aeaffa9`。sector400／full4096成分のstateを保存 | receiptの `ground_state_certified=false` を保持。stateが保存されていることと数学的にexact groundが認証済みであることは別。既存T/deltaはRTE費用比較の範囲で、M1-S検証窓ではない。Pauli-tail表・rho・低周波weightは未取得 |
| H6旧PF係数系列 [H2–H6記録][OLDPF]。H6 rank11、L_D=5、delta0.250–0.256の過去local validation | PF係数・state-action検証の集約JSONがある | DF19 snapshotと異なる系列。PF係数CをM1-Sのrho、スペクトル、RPE保証へ移さない。C_useは実行窓上の経験的包絡で厳密上界ではない |
| H4/H6既存resource記録 | P0.5 §4と[追加索引][P05INDEX]にfull-wrapper metric、shot等の保存fieldと出所 | M1-Sのordered law・same accuracyに整合するD/R別費用割合はNOT_ESTABLISHED。旧screeningのg_det/g_randや長round割合で穴埋めしない |

H6 receiptの当時のauthorization/STOP表示は履歴として保存し、今回のTrack Bへ移植・解除しない。H4/H6の保存stateは、将来診断に用いる場合 **`exact-state/oracle diagnostic`（保存近似stateの残差・認証範囲を併記）** とし、本番アルゴリズムの無料入力や未知groundを推定する実用predictorにしない。sector外へ出るPauliがあれば、sector内のgapだけで評価を閉じない。

### 9.2 科学計算の候補量：直接保存と未取得の境界

| 評価候補 | 直接保存が確認できるもの | 新たに必要となり得る科学作業（今回未実施） | 将来の使途と限定 |
|---|---|---|---|
| 個々のPauli期待値 | DF factorとstateという入力。Pauli期待値table自体は確認していない | Pauli展開とcanonical label/p_j固定、state上の期待値 | 項内・項間分散の診断。DF event期待値を代用品にしない |
| `<H_R>`、`<H_R^2>` | Hとsplit再構成の既存証拠。M1-S用moment pairは未確認 | 採択splitでtail作用・二次moment評価 | 全H固有stateは一般にH_R固有stateではない |
| 項間・項内分散 | §5の恒等式は既知。分子の対応tableは未取得 | p_j・期待値・h_R momentの評価、丸め・state誤差会計 | 完全乱択のV_kと部分候補の分母を混同しない |
| 結合励起state・spectral measure | 基底候補state、sector情報。全Fock励起basisはNPZ memberにない | 必要なsupportの励起solver／spectral approximation、必要ならsector外情報 | 全対角化を唯一の必須手法としない。採用法と誤差を先に固定 |
| 低周波weight・pair-sum coupling | 未取得。個々のPauli期待値だけでは一般に不足 | 周波数帯・kernel・cutoffを理論と対応させた評価 | v_kmと二modeのcouplingを別recordにする |
| resonance／alias | 既存RTE/PFの記録。M1-S theta/deltaに対応する表はない | 有限stepのmode距離・branch alignment評価 | bare gapの小ささだけで自動STOPしない |
| rho候補 | §10の式と成立範囲だけ。H4/H6の値は未取得 | 定義・分母のdomain・誤差を固定して評価 | 探索proxyまたは理論認証後の量。単独科学GO/STOPは禁止 |

ここで「未取得」は上記固定資料・member/field確認範囲での状態であり、全repository・全cacheに不存在と断定するものではない。既存に見つかった場合も、Hamiltonian/state/分割/定義/誤差/hashのidentity確認なしに流用しない。値の新しい再集計も今回の承認外である。

## 10. 改善因子rho：適用域と未確立部分

GPT-FINAL §§4.3, 5.1の式を、証拠水準を分けて記録する。

\[
\rho_\infty=\frac{V_k+v_{kk}}{2V_k}
=\frac12+\frac{v_{kk}}{2V_k}\in[1/2,1].
\]

| 層 | 何が言えるか | 何をまだ言えないか |
|---|---|---|
| 完全乱択の代数 | h固有state、V_k>0なら§5の不等式から上の式の範囲が支持される | 部分乱択stateの同じ範囲や有限時間総費用比ではない |
| 長時間の先導解釈 | 非共鳴・small-angle・孤立mode等の条件下で、同relative time・同減衰予算のイベント数候補比としてGPTが限定支持 | 有限theta/N・有限Dの誤差付き比はQ1/Q2待ち。任意有限Tで常に[1/2,1]とはしない |
| 部分乱択 | 下記rho_infinity^(H)はClaude由来の候補診断量として記録できる | 実際のΨ固有modeから未導出。必要条件・十分条件・heuristicのどれかも未確定 |
| 探索proxy | 案Bを別途採択・実行承認した場合に限り、未認証ラベルと全失敗例を伴う観測に使う可能性 | 小さい値から理論成立・RPE成功・実費用利益へ自動移行しない |
| 科学的判定指標 | Q1–Q5のうち主張に必要な保証、誤差、same-information/native会計と接続して初めて検討 | 現在は正式閾値も科学的GO/STOP規則も未採択 |

\[
\rho_\infty^{(H)}\stackrel{\rm candidate}{=}
\frac12+\frac{v_{kk}^{(H)}}{2(1-\langle k|h_R^2|k\rangle)}.
\]

全Hの対象state、`h_R=H_R/lambda_R`、`v_kk^(H)`の中心化・sampling lawを明示してから候補量を定義する。完全乱択の `e_k` をfull-Hのenergyや `<h_R>`へ黙って置換しない。`lambda_R=0`、分母零・極小、state誤差／roundoffで不安定な場合は**未定義／未判定**として保存し、比を便宜的に補完しない。完全乱択でもV_k=0は式の適用外。

**rho値単独で継続・停止を自動判定するcontractは作らない。** qDRIFT/RTEの既知gamma差、gate共有による候補イベント比、native一回単価・shot・総費用は独立した層である。最終総cost評価は未実施。

## 11. Gate 0の受理条件を後日固定するための根拠

0.7／0.85、およびend-to-end一割等はGPT-FINAL §8.3に記録された**過去レビューの参考値**で、現行の研究閾値ではない。本書は新閾値を選ばない。

| 固定前に必要な根拠 | 将来contractに記入すべき内容 |
|---|---|
| 理論的な意味 | 指標がどの信号・係数・保証に対応するか、必要／十分／proxyの区分、適用域とdomain例外 |
| 検証対象 | Hamiltonian hash、geometry/basis、DF rank/policy・ordered split L_D、Pauli/event law、state/sector、T・delta・theta・r・N・round、baseline |
| 近似誤差 | 漸近剰余、outer/inner PF、angle歪み、solver/state residual、スペクトル打切り、丸め・合成誤差。今回C_useを厳密boundへ格上げしない |
| 精度・信頼度 | scalar診断のtoleranceと、最終RMS energy error／高確率保証を別定義。error/failure budgetの配分根拠 |
| 統計変動 | deterministic保存state診断の数値誤差と、sampling/measurementの標準誤差を区別。列再利用、X/Y/round相関、多候補探索、区間境界の扱い |
| 資源改善との対応 | 同情報・同精度のstrong baseline、native座標、全wrapperとD/R・shots・準備・online費用。イベントratioを総費用ratioへ読み替えない |
| 誤判定のリスク | proxyで偽GOする危険、誤った早期STOPで理論的可能性を捨てる危険、近境界・未定義・欠測をundeterminedにする規則 |

受理候補は、(i)入力・law同一性、(ii)数学的適用域、(iii)誤差を含む指標値、(iv)比較の公平性、(v)主張に対応する費用証拠を**別々に記録する形**である。データ処理成功や文書検査PASSは科学的有用性gateの通過を意味しない。探索なら探索statusのまま報告し、結果を見て閾値を選んだ場合に事前登録と呼ばない。

将来の実行契約には、sampling/cache契約、task-bound seeds、source/input freeze、retry/STOP、メモリ・出力・ログ・call/instruction予算を含める。性能方針に従い支配処理、CPU/GPU/メモリ、sector/matrix-free/batch/cacheとthread数の整合を実行前に調べる。**今回backendやworker数は選定していない**。新規契約の科学phase/total wall-time capは既定で設けず、tool polling timeoutと区別する。過去の凍結jobや許可を変更しない。

## 12. 理論先行と探索的数値先行の比較

二案は判断材料として並べる。GPT-FINALは案Aに近い限定理論先行を推奨するが、**CodexはA/Bの一方を採択していない**。両案の科学計算、新証明はそれぞれ別途承認が必要。

| 比較軸 | 案A：理論先行 | 案B：探索的数値評価先行 |
|---|---|---|
| 順序 | 完全乱択の一様条件→部分Ψへの移送→正当化された量の固定→Gate 0→RPE/資源 | 未認証proxyの定義→小規模探索→有望なら理論→理論不成立ならproxyの科学GOを認めない |
| 情報価値 | 指標が何を意味するかを先に限定でき、無関係な数値取得を減らせる | 化学stateでproxyが変動するかの初期情報。保証成立や実用利益を判定する情報とは区別 |
| 必要な計算 | 初期は証明・仕様整理が中心。分子値取得は必要量が決まった後 | Pauli期待値・moment・必要ならspectral近似。全対角化や新state生成を自動的に含めない |
| 理論的リスク | 証明が強い隔離条件に依存し、化学に適用できない可能性 | proxyがΨ/RPEと無関係、basis/sector差で好結果が無意味になる可能性 |
| 研究資源 | 理論担当の時間を先に消費。数値量・費用予算は未見積 | 入力整理・solver/matvec等の計算資源を先に消費。所要時間・速度比は未算定 |
| 誤判断の危険 | 十分条件が厳しすぎるだけで実際の可行性を否定する危険 | 小rhoを偽GOに使う、選択bias、悪いproxyから不当な全般STOPを出す危険 |
| STOP可能な段階 | Q1の一様性／Q2の仮定／Q3の移送で限定・停止を判断。Q4/Q5へ自動進行しない | 取得不能・domain違反・proxy不調で探索を区切れる。ただし有望でも理論認証まで科学的GOなし |
| 実施前に残る選択 | どのstatement・仮定までG0-Tで扱うか | proxyの定義、候補system/split・独立性・許容情報、探索結果の使途と停止規則 |

「案Bで有望でなければ理論を行わない」という選択自体もユーザー/GPTの研究方針判断であり、未認証proxyに基づくM1-S一般no-goを正当化しない。

## 13. 継続・方向転換・STOP条件の候補

GPT-FINAL §8.3の判断材料を、契約で確かめる項目へ対応させる。**下表による今回の採否判定は行っていない**。

| 将来確認する条件 | 判断へ戻す内容 |
|---|---|
| Q1/Q2が明示した自然な仮定で閉じ、Q3の対象lawが定まる | その範囲のGate 0に情報価値があるか。数値実行は自動GOではない |
| 完全乱択だけ成立し、部分Ψの物理時間・phase・PF biasが閉じない | 完全乱択の限定理論ノートへ縮小／M1-SのPR資源claim停止を検討 |
| rho^(H)が未定義／非正当化 | その指標の科学的GO/STOP利用を止める。別proxyや理論課題の価値は個別判断 |
| 強い共鳴・低周波結合でsignal floorやRPE条件を保持できない | 当該領域を適用外にする根拠、仮定修正、PR/RPE claim縮小・終了を判断 |
| 正当化されたイベント利益がsame-accuracy native/shot費用で消える | 実用資源改善claimを採択せず、理論のみの価値を評価 |
| 同等のcoherent-signal保証が既知文献へ還元される | 独立新規性claimの縮小、既知手法の応用・technical noteとして整理 |
| 主要障害が残る | M2等への転換・新規探索休止を検討。Track B全候補の一般no-goにしない |

反例が必要な欄は**将来の検証対象**であり、今回新しい反例を構築していない。失敗・未定義・未確立と、資源優位が否定された結果を区別する。

## 14. 次にユーザー・GPTが判断／承認する項目

1. **候補契約の採択**: M1-Sを狭い理論課題として扱うか、Q1–Q5のどこまでを次のscopeにするか。条件付きBの引用だけでは採択にならない。
2. **対象law**: Pauli/involutionか一般DF eventか、H_Dの参照と実装、G_±の完全ordered word、sign/phase/時間/角度、乱数共有・再利用の範囲。
3. **実施順序**: 案A／案B／縮小・保留。案Bの場合のproxyの意味と利用範囲も判断する。
4. **G0-Tの実施承認**: 証明対象と追加仮定、受理／不成立の扱い、理論担当の範囲。文書公開の承認とは別。
5. **Gate 0の実施承認**: 必要となった場合のみsystem・rank/policy・L_D・delta/angle/time域・state情報・誤差・baseline・探索/確認・予算・STOPを結果前に固定。閾値はその根拠とともに別途判断する。
6. **RPE/費用への拡張**: RMSか高確率か、same-information、native座標、準備・online費用、strong baselineと公平な最適化。forecast/compile等は独立に承認する。

既存レビュー済みの同一論点を反復するためではなく、**未採択protocol・指標の正当化・実施順序**という残る科学判断を戻す。新しい科学レビューの開始依頼と、研究契約の採択、理論実施、数値実行の各承認を分離する。

## 15. 公開・検査・mandatory STOP・未実施

専用branch `track-b-m1s-g0-contract-preparation-20261015` をG0-pre固定commitから作成した。原本を先にcommit/pushし、§3のremote再取得を完了した。Decision Packetと索引の公開後もremote SHA・GitHub bytes/SHA256を照合する。公開receiptは[今回のartifact directory][ARTDIR]を参照し、自己参照hashを避けるためpayload commitとreceipt commitを区別する。

文書検査はMarkdown参照・表構文・固定Git path、原本bytes/hash、manifest旧entry不変、親HEAD/index/既存dirty file保全に限定する。**science runner、科学実験のテスト、数値arrayの評価・再ランキングは実行しない**。文書検査は科学的validation・CI再現ではない。

新証明、G0-T、Gate 0、simulation、行列評価、対角化、sampling、trajectory、operator guard、回路生成、compile、synthesis、新資源forecast/最適化は**すべて0**。既存source/result/classification/freeze/authorization/markerを変更しない。研究概要・履歴・manifestへの追記は今回の資料所在・準備statusだけで、既存科学statusを改訂しない。

**本作業完了後はmandatory STOP。M1-S正式採択、G0-T、Gate 0、RPE保証研究、native評価は自動開始しない。** 必須原本の欠落・版不明・新証明/計算なしには整理できない問題・既存worktreeを保護できない状態が生じた場合は、その未完了範囲を明示して止める。未取得資料を推測で補わない。

## 16. 固定source・先行研究への参照

本書の数学的根拠はGPT-FINAL §§3–6、未証明・判断境界は同 §§4–8。G0-pre §3とG0-LITは近接8文献、P0/P0.5とそのlocatorはPR/Campbell/Composite/Gu/Li/富士通の対応箇所を記録する。一次文献の保証の移送・独立新規性は本書で認定していない。

| 証明義務 | 既存調査が特定した主な原文locator |
|---|---|
| Q1 | [PR v2][PR] Lemma A.1/A14–A17、[CHEN v4][CHEN] Algorithm0.1/Thm1/2。共通law・normの追加対応が必要 |
| Q2 | [Gu v2][GU] SI S7–S16、[KGW v1][KGW] Thm1/Eq4–5。gap対象が異なる |
| Q3 | PR A32–A46、[Hagan–Wiebe v3][HW] Def3/Thm4–5、[Composite v1][CQP] Def3、[KIM v1][KIM] Eq2–13/Thm1、[COS v2][COS] Eq6/7。平均対象・access差を維持 |
| Q4 | PR ThmB.2/B13–B29、[Li v3][LI] Sec4/Thm4.1、[富士通 v2][FUJ] Appendix A/B。RMS／register-QPE／資源modelを区別 |
| Q5 | [Campbell][CAMP] Appendix D、PR Appendix E、富士通B3–B5、[MLMC v2][MLMC] §§2.2/2.3、[SCU v2][SCU] Eq1–3/6–8。coherent共有やcross termの外形だけを新規性にしない |

[GPT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4f5114e137563bb871c9f881f9eadab0c180bdd4/docs/research/track_b_m1s_claude_final_gpt_independent_scientific_review_20261015.md
[GPT1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/41fa1eb480728cd0691b00bb41dd4e5dc4ba6b03/docs/research/track_b_m1_claude_review_gpt_independent_assessment_20261015.md
[PRE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8b7666d0270f7fac01e430e195bef7b58ac0116c/docs/tracks/algorithm_codesign/m1s_g0_pre_prior_art_protocol_and_cost_decision_packet_20261015.md
[PREINDEX]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8b7666d0270f7fac01e430e195bef7b58ac0116c/artifacts/track_b_m1s_g0_pre/2026-10-15/v1/evidence_index.json
[PRELIT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8b7666d0270f7fac01e430e195bef7b58ac0116c/artifacts/track_b_m1s_g0_pre/2026-10-15/v1/primary_literature_locator.csv
[P0]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fa91ffb3d4880d767e55f0f661e8ade294b4c9fe/docs/tracks/algorithm_codesign/m1_p0_prior_art_control_and_resource_decision_packet_20261015.md
[P05]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ed3ddeacc88e998e7bc1d8c0db8163e106312791/docs/tracks/algorithm_codesign/m1_p05_theorem_protocol_and_resource_preflight_20261015.md
[P05INDEX]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ed3ddeacc88e998e7bc1d8c0db8163e106312791/docs/tracks/algorithm_codesign/m1_p05_reference_index_20261015.json
[COMMON]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8b7666d0270f7fac01e430e195bef7b58ac0116c/docs/research/research_project_instructions_gpt_codex_review_approval_2026-10-12.md
[INDEX]: ../../../artifacts/track_b_m1s_g0_contract_preparation/2026-10-15/v1/evidence_index.json
[ARTDIR]: ../../../artifacts/track_b_m1s_g0_contract_preparation/2026-10-15/v1/
[H4NPZ]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8b7666d0270f7fac01e430e195bef7b58ac0116c/artifacts/pr2_s0_s1_validation/2026-09-28/h4_1p00_rank12_development_v1.npz
[H4V13]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8b7666d0270f7fac01e430e195bef7b58ac0116c/artifacts/pr2_new_series_validation/2026-09-28/pr2_v1_v3_result_v1.json
[H6RECEIPT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0dd5c3d204dab9b731268a1aa2add621080f3d20/artifacts/resource_applicability/track_a_h6_saved_df_completion_parallel_v2/2026-10-10/launch_v1/snapshot_receipt.json
[H6NPZ]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0dd5c3d204dab9b731268a1aa2add621080f3d20/artifacts/resource_applicability/track_a_h6_saved_df_completion_parallel_v2/2026-10-10/launch_v1/h6_input_snapshot.npz
[H6SUMMARY]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0dd5c3d204dab9b731268a1aa2add621080f3d20/artifacts/resource_applicability/track_a_h6_confirmation_summary_v1/2026-10-12/result_summary_v1.json
[OLDPF]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8b7666d0270f7fac01e430e195bef7b58ac0116c/artifacts/pf_c_system_size_validation/h2_h6_paper_d6_c_v1.json
[PR]: https://arxiv.org/pdf/2503.05647v2
[CAMP]: https://arxiv.org/pdf/1811.08017
[CHEN]: https://arxiv.org/pdf/2008.11751v4
[GU]: https://arxiv.org/pdf/2208.04100v2
[KGW]: https://arxiv.org/pdf/1907.10070v1
[HW]: https://arxiv.org/pdf/2206.06409v3
[CQP]: https://arxiv.org/pdf/2306.16572v1
[KIM]: https://arxiv.org/pdf/2608.07720v1
[COS]: https://arxiv.org/pdf/1805.08385v2
[LI]: https://arxiv.org/pdf/2111.10430v3
[FUJ]: https://arxiv.org/pdf/2603.22778v2
[MLMC]: https://arxiv.org/pdf/2604.26865v2
[SCU]: https://arxiv.org/pdf/2407.21095v2
