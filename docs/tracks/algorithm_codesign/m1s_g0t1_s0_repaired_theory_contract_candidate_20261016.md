# Track B／M1-S G0-T1：S0修正理論契約候補

2026-10-16 JST。**`M1S_G0T1_S0_REPAIRED_CANDIDATE_READY_FINAL_ADOPTION_PENDING_MANDATORY_STOP`**。

**G0-T1への縮小と順序A′はユーザーが暫定承認済み。本書はS0の最終確認用の候補契約であり、研究契約の正式採択・T1a/T1bの新規証明実施・科学計算は未承認。** 以下の数式は既存GPTレビューの確認済み結果の引用、または将来証明するstatementの定義候補である。今回、新しい数学的結論は出していない。

## 1. 今回承認された暫定方針と作業権限

ユーザー提供「Track B／M1-S G0-T1 S0研究契約の修正・固定準備」を今回の権限根拠とする。前回の「A/Bとも未選択」から、**A′：T1a→Checkpoint 1→T1b→Checkpoint 2→Gate 0候補の判断**へ暫定方針が進んだ。これはCodexが独自に方針を採択したものではない。

| 項目 | 今回の状態 |
|---|---|
| G0-T1への縮小、T1a先行、Checkpoint配置、A′ | ユーザーが暫定研究方針として承認 |
| 原本保存、S0候補の修正、参照・構文検査、専用branchのcommit/push | 今回の作業として承認 |
| S0候補契約の正式採択 | 未承認。文書の版固定と科学的採択を区別 |
| T1a/T1bの新規定理証明、数値反例探索 | 未承認・未実施 |
| Gate 0/E0、行列・simulation、回路生成/compile/synthesis、forecast | 未承認・未実施 |
| Q4/RPE保証・Q5/native資源への研究拡張 | 初期scope外、未承認 |
| 既存authorization/marker・mandatory STOPの解除 | 今回の承認対象外 |

[共通指示][COMMON]の技術的な契約準備に従う。最新GPTレビューの当時の「未採択」という権限記録は原本に保持し、今回の暫定承認を別の履歴として記録する。Parent allocation D＋E、M1-A補助候補、M2代替、旧科学分類・成果・停止点は変更しない。

## 2. 正本資料・固定commit・原本同一性

数学的な証拠水準と修正理由の主要参照は[最新GPTレビュー][GPT]である。[前回事前契約][OLD]は差分の基点であり、Q1–Q5を今回一括採択する正本ではない。固定資料の取得とbyte照合は[参照索引][INDEX]へ記録した。

| ID | 固定commit | 内容・今回の確認範囲 |
|---|---|---|
| GPT-T1 | `d1039afae54520f15545177a7c852407830a0927` | [2026-10-16 GPT原本][GPT]。全文を読み、§§4–10/12の修正事項を対応付け |
| G0-CONTRACT | `0f0952614dcf5c06d3d63c21b5d49091b8ecce37` | [G0事前契約][OLD]と[既存evidence index][OLDINDEX]。旧候補・未実施範囲を保持 |
| GPT-PREV | `4f5114e137563bb871c9f881f9eadab0c180bdd4` | [前回GPT原本][PREV]。完全乱択の恒等式、rhoとRPE/費用の留保 |
| G0-PRE | `8b7666d0270f7fac01e430e195bef7b58ac0116c` | [既存調査][PRE]。近接文献、law/control/費用の境界 |
| P0.5 | `ed3ddeacc88e998e7bc1d8c0db8163e106312791` | [定理・source・資源照合][P05]。対象norm/mean/channel差、D/R未確立 |

上の6ファイル（事前契約とそのindexを別計数）はGitHub connectorからbase64で取得し、復号bytesを固定commitのGit blobと照合した。既存の文献検索・科学監査・データ評価は再実行していない。

最新GPT原本はユーザーが本文で指定したrepository直下のファイルから取得した。

- 保存先：`docs/research/track_b_m1s_g0t1_claude_gpt_independent_scientific_review_20261016.md`。
- **52,629 bytes**、SHA256 **`cb24103f22ddfb12afb82c923fee4210fabae4b7dfec2355896a0686fb18f633`**。
- Git blob：`5b470fe248e1cc7c2c6ce3dc7e0832e5dda26a8d`。原本保存commitは表のGPT-T1。
- 保存前にpath履歴と同一blobの全ref履歴を確認し、既存保存がないことを確認。byte copy後にGitHubへ保存し、再取得bytesも一致。本文・改行・当時の権限記録を変更していない。

Claudeの今回の契約レビュー `レビューinner(1).md`（GPT記録では740行）は**今回Codexには直接提供されていない**。GPT-T1 §2.1が記録するClaude原本hashは `17cee6834f7ee0f731142279b15ac1aa96026f5e976377b227b89abe68221b69` だが、Codexによる原本照合済みとはしない。Claudeの主張はGPT-T1が引用・評価した範囲に限って利用し、過去の似たファイル名と同一視・再生成しない。

## 3. G0-T1の研究目的と成果の範囲

対象は、同位置のlabelを二枝で共有するdirectional qDRIFTの**coherent平均信号**について、減衰と位相を分けた条件付き保証が部分乱択へ移るかを調べるための理論前段階である。方向制御、回文、qDRIFT、量子枝の共有乱数そのものを新しい発明とする目的ではない。

T1aは**Q1-core＋Q2-min**として、完全乱択の信号近似に必要な最小限の平均化・スペクトル条件を扱う。T1bは**Q3-core**として、exact deterministic halfとPauli-tailの対称stepで、その先導機構が成立・要修正・不適用のどれになるかを判定する対象とする。これは証明対象の整理であり、成立を予告するものではない。

T1aの完全な一般化を目的化せず、T1bで使える出力をCheckpoint 1へ渡す。T1bまで条件付きの保証が得られても、分子での有効性・RPE成功・総資源改善・独立新規性はまだ別の判断である。T1aのみの成果、十分条件、適用限界、反例・不能性も区別して記録する。

## 4. T1a：完全乱択の対象・比較モデル・証明義務

### 4.1 参照lawと既知の出発点

GPT-T1 §§3, 4.1の参照を使う。有限次元で、`P_j=P_j^dagger`、`P_j^2=I`、`p_j>=0`、`sum p_j=1`、`h=sum p_j P_j`、`A_j=exp(+i theta P_j)`。右端先作用の表記、時系列label `j_1,...,j_N` は位置間IID、同位置の二枝では共有する。

\[
B_1=A_{j_N}\cdots A_{j_1},\qquad
B_0=A_{j_N}^{\dagger}\cdots A_{j_1}^{\dagger},\qquad
\mathbb E[B_0^\dagger B_1]=\Phi_\theta^N(I),\quad
\Phi_\theta(X)=\mathbb E_j[A_jXA_j].
\]

この恒等式は既存レビューで確認済み。負枝は**同じ時系列で各gateの符号を反転**したもので、全wordのliteral adjointとは異なる。Φ自身をCPTPと呼ばず、通常の `E[A_j X A_j^dagger]` と区別する。対象信号は正規化初期状態ψに対する `z_N(psi)=⟨ψ¦Phi_theta^N(I)¦ψ⟩` である。

GPT-T1 §4.1が確認した二次展開の記号を参照する。

\[
\mathcal L_1(X)=hX+Xh,\quad
\mathcal C(X)=\mathbb E_j[(P_j-h)X(P_j-h)],\quad
\mathcal D(X)=-\tfrac12\{I-h^2,X\}-\mathcal C(X).
\]

**一stepの二次展開があることと、長時間の一様保証が完成していることは別**である。ここでの演算子展開の `O(theta^3)` を、今回明示定数付きの一般剰余へ格上げしていない。

### 4.2 比較対象の候補を分ける

**C-sec候補**はGPT-T1 §4.2の二次secularモデルである。

\[
z_N^{\rm sec}(\psi)=
\langle\psi|e^{iN\theta\mathcal L_1}
e^{N\theta^2\mathcal D_{\rm sec}}(I)|\psi\rangle.
\]

この候補でいうsecular部分は、まず `L1` の**同一周波数block内の作用を残す射影**として定義する。どの固有operator・周波数・縮退blockを使うかをstatementに添付する。厳密共鳴を残すblockと、近共鳴を追加でまとめるclusterは別であり、後者を採るなら比較モデルと誤差定義を改訂する。低周波cutoffやweight normはまだ採択しない。

**C-log候補**はGPT-T1 §§0.2, 4.4が挙げた、正確な一step Φの対数に基づく有効生成子／位相補正を参照する案。対数の存在域・branch、どの次数・modeを残した近似か、補正をどの誤差項へ入れるかを指定する必要がある。正確な写像を別記号で書くだけを新しい近似保証や研究差分としない。

**C-sec/C-logの最終選択は未決定**。本書は異なる比較対象の誤差項を都合よく混ぜた一つの定理を作らない。採択する比較対象を確定してから証明を開始する。

### 4.3 将来のT1a提出物

T1aのstatementには、対象state class、近似信号、angle域、`N theta^2<=D_cap` の全対象N、norm/weight、明示した誤差会計、定数の依存性を含める。signal floorが必要な位相保証は複素信号誤差と分ける。非共鳴・低周波・pair-sumだけでなく**局所高次誤差を含む**ことを§7で要求する。

Checkpoint 1へは、証明済み範囲・未完部分・既知保証との差・T1bへ渡せる構造を一緒に提出する。少なくとも非可換の領域で減衰と位相を分ける有用性を検討する対象とし、可換例の整合だけで独立claimを認めない。今回はその検討・証明を実行していない。

## 5. T1b：部分乱択の参照stepと証明義務

### 5.1 Exact deterministic halfの候補定義

GPT-T1 §§5.1–5.3とユーザー指定に従い、参照モデルを `H=H_D+H_R`、`H_R=lambda_R sum_j p_j P_j`、`lambda_R>0` のPauli-tailに限定する。one-body correction・定数をどちらへ置くかはHamiltonian分解とphase台帳に明記する。特定の分子・DF rank・L_D・数値stepは選んでいない。

正の相対step幅 `delta_rel` に対する線形角の候補辞書は§8。**本書のexact-half参照では**

\[
D=e^{+i\alpha H_D},\qquad D'=D,\qquad
\alpha=\delta_{\rm rel}/4.
\]

`D`と`D'`は正枝の前後halfの位置を表す記号で、primeをadjointと読まない。既存資料のforward/reverse sweepの一般表記を参照する場合も、初期証明scopeのexact参照では両者は等しい。内部をPF gate列へ置き換えた版はこの定義と分離し、今回は作成・採択・同値検証しない。

### 5.2 Branch・順序・平均写像

一step内のlabel列を `omega=(j_1,...,j_r)`、`R_l=A_{j_l}` とし、

\[
R=R_r\cdots R_1,\quad R^{\rm rev}=R_1\cdots R_r,\quad
R_-=R_r^\dagger\cdots R_1^\dagger.
\]

exact参照の正枝は `G_+=D'RD`、負枝は `G_-=D^dagger R_- D^dagger`。時系列では前half→j_1からj_r→後halfで、負枝は同位置labelを共有して符号反転する。`R_-`を `R^dagger` と置かない。GPT-T1 §5.1の順序規約により `G_-^dagger=D' R^rev D` と書ける参照である。

同じ確率lawで位置間IIDの場合、GPT-T1が確認した厳密な恒等式を引用する。

\[
\boxed{\Psi(X)=\mathbb E[G_-^\dagger XG_+]
=D'\Phi_\theta^r(DXD')D.}
\]

step間も同じ `delta_rel,r,theta,D` と同一lawで独立なら、s stepの枝干渉平均は `Psi^s(I)`。位置やstepごとにlawが変われば固定Ψのs乗とは扱わない。同位置共有、位置間IID、step間IIDを独立した条件として記録する。

既知の整合極限は `H_D=0` なら `D=D'=I`、`Psi=Phi^r`。GPT-T1 §5.1の形式的な `r=0` の恒等式も参照できるが、§8の `theta=lambda_R delta_rel/(2r)` はその境界では使えない。空tail／`lambda_R=0` は別の決定論境界として扱い、rhoを補完しない。

### 5.3 Phase・対象mode・移送の境界

controlled operatorのbranch polarityと観測は `diag(B_0,B_1)`、`⟨ψ¦B_0^dagger B_1¦ψ⟩` を参照し、X/Yの符号・読出し規約は記号台帳に対応付ける。identity項やsystem global phaseはbranch relative phaseとして保持する。system上のprojective同値だけで受理しない。今回測定・回路は生成しない。

T1bの対象modeは原則として**全Hの対象固有stateと、それに対応させるΨの有効mode**であり、h_Rの固有stateを全Hの固有stateとみなさない。有限stepで有効Hamiltonianを使うなら、全Hからのbias・mode差と連続分枝を証明対象に含める。Ψの因数分解だけから減衰率、rho、RPE、shot数は得られない。

将来のT1b提出物は、ordered lawと同じjoint scalingの下で、先導減衰係数・位相・mode/weight・dressing誤差を定義し、rho候補を**成立／要修正／不適用／未解決**に分けるものとする。outer symmetric-step biasとrandom誤差も区別する。Exact H_Dを内部PFへ戻す追加誤差・価格は初期scope外である。

## 6. 数学的仮定・定義域・独立性

記号 `D` は決定論unitary、`mathcal D` はT1aの二次生成子、`D_cap` は減衰予算の上限とし、混同しない。`D_damp` は§8の実際の無次元減衰スケールである。

- **有限次元・有界Hermitian・involution**を初期参照とする。次元・sampling support・coefficientが変わるfamilyへ移す場合は定数依存を開示する。次元非依存を最初から必須にせず、次元依存を隠してよいとも扱わない。
- `p_j>=0`、総和1、同一の線形角を使う。符号付き係数をP_jへ吸収する規約、identity、重複項の扱いを明記する。分布の再重み付けや一般DF eventへの変更は別契約。
- T1aの参照固有stateは `h|k>=e_k|k>`、T1bは `H|k>=E_k|k>`。一般初期状態まで主張する場合はoverlapとmode weightsの条件を別途固定する。今回、具体的なstate・overlap閾値を採択していない。
- normはoperator/superoperator normまたは特定state上のweightを混ぜず定義する。Φのoperator-norm収縮を、他のnorm・各非正規modeの条件数の良好性へ無条件に移さない。
- 共鳴blockを保持する場合と、隔離したmodeの摂動を行う場合を区別する。必要なdiagonalizability／射影・resolvent条件を記録し、一般Jordan理論の完遂を初期完了条件に追加しない。
- 位相を主張する領域には、対象信号または参照信号の**どちらに対するsignal floorか**、複素誤差との関係、phase branchを明示する。near-zeroでは複素信号保証と位相未定義／未判定を分ける。
- shotごとのfresh drawは無条件平均恒等式の必要条件ではない。列再利用・X/Y/round間のcouplingは統計保証の別契約。今回は平均信号の理論候補であり、shotモデルやRPE roundを採択しない。

λ_Rの零、V_kやrho分母の零・極小、mode crossing・共鳴・alias、sector外結合、signal近零を適用域の外側／保留域として明示する。いずれかの存在だけで全M1-Sの一般no-goを認定しない。

## 7. 局所高次誤差を含む誤差会計

### 7.1 最新GPTレビューによる必須修正

GPT-T1 §4.3は、可換Pauli `P_1=Z_1, P_2=Z_2` を確率p,1−pで選び、`|k>=|01>`、`e=2p-1` とした**既存の紙上検算**を示した。pは `0<p<1`、`e≠0,±1` の条件である。その厳密信号と二次secular候補は

\[
z_N(k)=(\cos2\theta+i e\sin2\theta)^N,\qquad
z_N^{\rm sec}(k)=e^{2iN\theta e-2N\theta^2(1-e^2)}.
\]

この模型では別modeへのoff-resonant/near-resonant leakageがなくても、**`N theta^2=D_damp` を固定した比較に `O(D_damp |theta|)` の位相誤差が残る**とGPTレビューは確認している。本書はその結果を引用し、新しいTaylor展開・数値検算・反例探索を行っていない。

したがって、**`theta E_off + E_LF` だけを一般の厳密誤差式として採択しない**。局所誤差を `E_off` にまとめる案を採る場合にも、その定義・非零寄与・局所高次の累積への対応を明記する必要がある。項の名前を変えるだけで修正完了としない。

### 7.2 将来statementが説明すべき会計欄

以下は**証明義務の欄**であり、総和を今回証明済みのboundと宣言するものではない。

| 誤差・条件 | 比較対象と受理のために必要な説明 |
|---|---|
| 局所高次誤差の累積 | 一stepの打切り／log補正の剰余、norm、N反復後の定数と `N theta^2<=D_cap` での一様性。上記の `O(D_damp abs(theta))` 寄与を取りこぼさない |
| 有限角度の位相歪み | 比較モデルへ吸収するか残差として評価するかを固定。局所高次と同じ起源の項を二重計上しない。補正後の対象時間・targetを明示 |
| 非共鳴mode | 周波数差、実際のcoupling、重み付き和／normと分母、射影条件数。bare Hamiltonian gapだけで代用しない |
| 低周波・pair-sum | `L1` の周波数block、結合するnear resonance、cutoffとweight、secular部分へ残すもの／誤差へ出すものを明示 |
| 初期状態・mode weights | 対象modeへの入力weight、他modeの寄与、複素左右weightと近縮退。状態を一つの位相に無条件置換しない |
| 信号から位相への接続 | 適用域内のsignal floorとphase branch、複素信号誤差と位相誤差を区別。位相が定義できない境界を別recordにする |
| T1bへの追加項 | exact-Dによる二側作用、有効mode、有限alpha、outer-step bias、joint scalingで縮む隔離距離。T1aと同じ係数と仮定しない |

すべての対象Nに一様であることを要求し、固定N極限で代用しない。定数の `D_cap,theta,dimension,gap,weight,condition number` 依存を開示する。gap等が同時に小さくなるfamilyでの有効性は、固定モデルの結果とは別に判定する。

## 8. Joint scalingと時間・角度の辞書候補

T1bの線形角・exact-half参照では、同じHamiltonianとλ_Rを固定した上で、

\[
T_{\rm rel}=s\delta_{\rm rel},\qquad
\theta=\frac{\lambda_R\delta_{\rm rel}}{2r},\qquad
\alpha=\frac{\delta_{\rm rel}}4,\qquad
D_{\rm damp}=sr\theta^2\le D_{\rm cap}.
\]

`r,s`は正整数の内部参照、`delta_rel>0`。N=0や空tailの境界は別記する。`alpha`は `exp(i alpha H_D)` の指数係数で、H_Dの各係数を含めたgate角とは区別する。T_relは干渉の相対時間、理想正負枝の時間絶対値は各T_rel/2。全相対stepの決定論部分は前後に各delta_rel/4を置く辞書である。

T1aの線形角の対応も、`N theta=lambda T_rel/2` として同じ時間規約に結ぶ。これは**辞書候補**であり、平均信号が理想発展に厳密一致するという保証ではない。ordinaryの片枝の時間と±の各枝の時間を区別し、同じN・thetaだけを公平比較の条件にしない。

| 同時極限の候補 | 必ず残す検討事項 |
|---|---|
| delta_relを固定しrを増やす | thetaは小さくできるがalphaは固定。**theta→0だけでdressingが消えるとしない**。有限alphaのΨと係数を扱う必要がある |
| delta_rel→0を含む小step | alphaも小さくなるが、s・r・T_rel・D_dampの同時変化を明記。outer-step bias、長時間累積を含める |
| 有限減衰予算の長時間family | `sr theta^2<=D_cap` の全対象へ定数を管理。固定T_relと、T_relが伸びるfamilyを同じ極限と呼ばない |

GPT-T1 §5.3が指摘するように、小stepでは有効PFの固有位相分離が `O(delta_rel Delta E)` まで縮むことがある。摂動の大きさだけでなく、**摂動／分離の比**を同時に制御する条件を証明対象に含める。有限stepのphase aliasとpair-sumも別欄とする。

どのjoint path・許容域で先導移送を主張するかは最終確認事項。今回、r/s/delta/angle/time窓、分子入力、数値予算を選定していない。arctan角やfinite-RTE角への置換を黙って含めない。

## 9. Dressing：状態・有効mode・絶対誤差・相対誤差

GPT-T1 §6.1が確認した状態ノルム評価だけを、確認済み結果として引用する。全H固有state `H|k>=E_k|k>`、`a=⟨k|H_D|k⟩`、`D=exp(i alpha H_D)` について、

\[
\|D|k\rangle-e^{i\alpha a}|k\rangle\|
\le |\alpha|\sigma_k(H_D)
=|\alpha|\sigma_k(H_R).
\]

これは状態の近さの上からの評価で、dressingの必要十分条件ではない。本書はDuhamel恒等式の新証明や実際の分子値取得を行っていない。

T1bは次の5層を区別して提出する必要がある。

1. **状態ベクトルの変化**：上のnorm評価と適用するstate。
2. **有効固有modeの変化**：全HのstateからΨの右/左mode・射影へ移る差。状態ベクトルの近さだけで代用しない。
3. **減衰係数の絶対誤差**：比較する係数、単位、normalizationを先に定義し、その差のboundを記録する。
4. **減衰係数の相対誤差**：絶対誤差と分母下界を両方要する。小さい元係数では絶対誤差が小さくても相対誤差が大きくなり得る。
5. **改善因子rhoの誤差**：比較双方の係数・mode・分母・価格/時間規約に依存する。単一の状態ノルム評価から同程度のrelative boundを出さない。

`DXD'` は同符号の二側作用。exact-DでD'=Dでも、`D|k><k|D` は密度演算子 `D|k><k|D^dagger` ではない。dressingを単に変換後の物理状態の分散と解釈せず、coherent superoperatorの左右作用として扱う。

`alpha sigma_k(H_R)` や `alpha^2 norm([H,H_R])` はGPT-T1 §§6.2–6.3に記録された**候補尺度**であり、Ψのspectral保証・相対係数誤差の十分条件としない。これらを数値取得することも今回の承認外である。

## 10. 改善因子rhoの定義域と未証明部分

完全乱択についてGPT-T1 §§3.2/8.1の分散分解と先導係数を参照する。h固有state、`V_k=1-e_k^2>0`、`v_kk=Var_j⟨k¦P_j¦k⟩` の範囲で、

\[
\rho_\infty=\frac{V_k+v_{kk}}{2V_k}
=\frac12+\frac{v_{kk}}{2V_k}\in[1/2,1].
\]

この**先導係数の代数**と、有限N/thetaの必要イベント数比を区別する。イベント比として使うには同relative time・同減衰予算・比較law・有限誤差・位相条件を閉じる必要がある。V_k=0で比を定義しない。

部分乱択の `rho_infinity^(H)` は、[前回レビュー][PREV] §5.1／[前回事前契約][OLD] §10の

\[
\rho_\infty^{(H)}\stackrel{\rm candidate}{=}
\frac12+\frac{v_{kk}^{(H)}}{2(1-\langle k|h_R^2|k\rangle)},\qquad h_R=H_R/\lambda_R
\]

という**未証明の移送仮説**である。`v_kk^(H)`のsampling/中心化規約、全H stateと有効modeの差、比較する両減衰係数を明示した上で、T1bが成立・修正・不適用を判定する対象として残す。今回この式を認証済みのcontractへ採択していない。

分母が零・極小、共鳴・alias、Pauliによるsector外結合、finite-alphaの補正、非正規な左右weightを明示する。ratio undefined／未判定の領域を数値で補完しない。有限時間rho(T)のsinc一次kernelやその区間はGPT-T1 §8.2の**一次近似の示唆**に留め、全次数の有限時間保証にしない。

rho単独の科学的GO/STOP、0.7/0.85等の固定閾値、rhoの値に基づくGate 0自動実行はいずれも設定しない。

## 11. 比較対象・情報アクセス・資源会計の意味論

比較taskは、同じHamiltonian/分解、同じ物理targetとT_rel、同じ初期状態条件、同じ減衰予算、同じ対象信号、同じ精度・保証対象、同じ情報アクセスを一組とする。複素信号、mode位相、RMS energy errorは違う出力であり、T1a/T1bで減衰を扱ったことをQ4の完了としない。

| baseline | 区別する意味論 |
|---|---|
| ordinary qDRIFT | `diag(I,U_omega)`。片枝のfull relative time、実際のmean信号・有限角度bias |
| 独立± qDRIFT | 正負branchを独立draw。負時間law・順序、半時間と相対時間、branchごとの正確なmean |
| gate共有± M1-S | 同位置label共有・同順sign reversal。Φまたは今回のexact-half Ψのcoherent平均 |
| 既存PR-RTE | 独立±、normalizer、ordered PF、有限cutoff等を元の保証・scopeに対応させる。qDRIFTへ同じ保証を転記しない |

full-wordを共有してliteral adjointを負枝に使う方式も別lawであり、gate共有と混同しない。native比較ではcontrol polarity、identity/relative phase、ancilla込みsupport、precision、fusion/HWP等の最適化機会を公平にする。今回回路価格を取得・再最適化していない。

次の3層は別のrecordとする：**既知qDRIFT/RTEの費用係数差、M1-Sの候補イベント比、RPE全体の期待総資源改善**。STARのC_gate、FT/HWP、compiled RZ/CX、Clifford+Tは異なる価格座標。既知係数差とrhoとnative仮定を無根拠に乗算しない。

GPT-T1 §8.3の `rho*=1−S*/f_R` は固定加法モデルの代数条件であり、採択済み閾値ではない。f_Rは**選んだ評価対象全体**に対する同じ座標の費用割合として定義する必要があり、現在のM1-SではNOT_ESTABLISHED。後半round割合で全RPEを代用せず、shot/bias/再最適化・state-prep/readout・online古典費用を無視した式を実用GO条件にしない。今回はS*もf_Rも数値化しない。

exact-ground/gap/weightを後日研究診断に使う場合はoracle情報と明記し、本番アルゴリズムの無料predictorへ流用しない。H4/H6の保存入力・未取得量は[旧契約§9][OLD]へ参照し、今回は再開封・再集計・Pauli-tail評価を行っていない。

## 12. Checkpoint 1・2の提出物と進行権限

**Checkpoint 1（T1a後）**では、局所高次誤差を含む一様信号保証、nonresonant/low-frequencyの定義、条件・定数依存、既知平均化/摂動論の単純な直接適用以上の価値、T1bへ渡せる構造を確認する。結果がない現在は判定していない。

判断候補は「T1bへ進むことを推奨」「T1a縮小」「technical noteとして整理」「STOP」。技術的完了・定理の形式的完成だけでT1bを開始しない。重要な仮定・statement・価値の判断が必要ならGPTへ戻すが、文書修正やsource版の変更だけを理由に同じ科学レビューを反復しない。ユーザーのT1b実施承認は別に必要。

**Checkpoint 2（T1b後）**では、Ψの対象law、全H/effective mode、joint scalingとdressing、rhoの意味・分母・適用不能域、実分子で評価する意義をまとめる。**GPTによる重要科学レビューを必須**とし、ユーザーのレビュー開始承認を得て実施する。必須であることを自動的なレビュー開始許可としない。

Gate 0の候補量・系・分割・state/情報・誤差・予算・STOPは、T1bの結果とCheckpoint 2に基づいて別途判断する。Gate 0が提案されても実行は独立した科学計算承認後。Q4/Q5への拡張、一般DF event研究、M2への転換も別判断である。

## 13. 成功・失敗・縮小・STOP条件の候補

| 将来の結果 | 区別する記録・判断 |
|---|---|
| T1aで明示域内の一様保証が得られる | 理論上の条件付き成功。T1bへの可搬性と既知法との差をCheckpoint 1で判断。自動実行許可なし |
| T1aが固定有限モデルで閉じ、次元非依存は未達 | 依存性を明示した部分結果。直ちに理論失敗とせず、化学でboundが空になる危険を別評価 |
| 局所高次誤差を含まない候補式しかない | 受理statementの不足として差し戻す。平均化研究そのものの否定とはしない |
| 有効な十分条件・解析反例・不能性が得られる | 証拠と対象を明記し、単なる未完了/INCONCLUSIVEと区別する。今回は新しい反例を作っていない |
| T1bで自然な明示条件の移送が閉じる | 限定coherent-signal claim候補。Checkpoint 2の科学レビューへ。分子・RPE・資源の成功とはしない |
| T1bが要修正／一部適用外 | 旧rhoを保持するために仮定を黙って変更しない。修正量・適用域・情報負担を提示して判断 |
| 共鳴・小分母・finite-alphaで相対誤差が閉じない | 当該ratio claimを未認証／不適用とする。別のabsolute signal保証の価値は別に判断 |
| T1bが未完了、または条件が過大 | technical note・scope縮小・STOPを検討。未承認のGate 0で不足を埋めない |
| 既知保証の直接言換えに留まる | 独立新規性claimを限定。M1-Sの結果だけからTrack B全体のno-goを出さない |

各結果で「証明済み／条件付き十分条件／反証／未解決／適用外／情報不足」を分ける。非可換の有用な位相・減衰分離が示されたかは研究価値の判断材料であり、今回成功基準を満たしたと宣言しない。

## 14. 初期scopeから除外する研究課題

一般Hermitian event、DF fragment/Gaussian event、非involutionのnormalizer・一event価格・実装loweringは**T1b後の追加候補**。GPT-T1 §7.1の二次展開が既存の紙上等式として正しいことを、初期必須証明の拡張に使わない。P_j^2=Iの式を一般eventへ代入しない。

粒子数を保存するだけで、目的stateが到達可能な不変部分空間の端点固有stateであるとは認定しない。GPT-T1 §7.3の共鳴抑制は、全event/Dが共通に保存する空間K、そのK内の非縮退端点性、基準Hamiltonian、finite-step aliasを明示した場合の条件付きの話である。spin等の狭いsectorのgroundをK全体の端点へ読み替えない。

任意の縮退cluster/Jordan・全次元一様な一般理論の完成、内部H_D PF実装・native同値、Q4/RPE RMS保証、Q5/end-to-end費用、Gate 0/E0の分子値取得、数値反例探索、sampling/trajectory/operator guard、回路/compile/synthesis、H12、既存停止系列の再開も初期scope外である。

## 15. 未証明事項と最終確認待ち事項

**未証明**はT1aの修正した一様保証、nonresonant/low-frequency/局所誤差の明示bound、T1bのfull-H有効modeと先導係数、finite-alpha/小stepのjoint移送、dressingの絶対・相対誤差、rho^(H)の正当化・修正・不適用の判定である。これらは将来の研究で扱う対象であり、S0文書作成で解消したとしない。

**最終確認待ち**は、C-sec/C-log、state class・signal floor、採用norm/weightと共鳴block、joint pathと保証域、必要な明示定数・十分条件、baselineの同時間・同保証の対応、Checkpoint 1で判断する最小提出物である。各選択に新しい科学判断が必要なら、候補欄を残してユーザー/GPTへ戻す。

**後段まで未採択**は、数値Gate 0の系/rank/L_D/窓・評価指標・閾値、rho(T)の有限時間保証、D/R費用割合、RPE保証、実用native比較、一般DF event研究、独立論文claimである。前回事前契約の広い表を今回の実施項目へ再採択しない。

## 16. 次に必要な承認

1. **S0候補の正式採択判断**：暫定方針G0-T1/A′と、本候補の対象law・statement・誤差会計・未確定selectorを照合し、必要な科学的選択を確定する。
2. **T1aの新規理論実施承認**：採択済みstatement・scope・提出物・STOPを指定する。候補文書の公開はこの承認を代行しない。
3. **Checkpoint 1後のT1b進行判断と実施承認**：T1aの結果を根拠に行う。A′の暫定承認だけでT1bを自動開始しない。
4. **Checkpoint 2の重要GPTレビュー開始承認**：必要資料を先に整え、必須レビューを実施する。T1b完了だけでGate 0へ進まない。
5. **後段の個別採択・実行承認**：Gate 0/E0、Q4/Q5、一般Hermitian/DF拡張、compile/forecast等。理論・レビュー・計算の承認を区別する。

今回の完了後に直ちに必要なのは**S0候補契約の正式採択判断**である。Codexは今回、新しい重要科学レビューを始めていない。

## 17. 既存研究・前回事前契約との対応

今回の変更は、[旧契約][OLD]のQ1/Q2をT1aの必要部分へ、Q3をT1bの先導移送へ絞り、Q4/Q5・数値候補を後段へ残す技術整理である。旧文書を改稿・削除せず、scope変更の理由と今回の暫定承認を研究概要・日付付きノート・manifestへ追加する。

- **GPT-T1 §§4.2–4.4 → 本書§§4,7**：比較モデルの区分、可換Pauli例が示す局所高次誤差、一様性・次元依存の境界。
- **GPT-T1 §§5–6 → 本書§§5,8,9**：厳密Ψ、exact-half、joint scaling、state normと相対係数誤差の区別。
- **GPT-T1 §§7–8 → 本書§§10,11,14**：一般eventを後段へ、到達空間の端点性、rhoと費用閾値の非認証。
- **GPT-T1 §§9–10/12 → 本書§§1,3,12–16**：A′、二つのCheckpoint、claimと実行権限の分離。

近接先行研究は[G0-pre][PRE] §3と[P0.5][P05] §2の固定locatorを再利用する。PR Appendix A.4は対称PFの厳密halvingと独立RTE ±の条件、Campbellはcontrolled qDRIFTの既知構成、Gu/Li/CHENはnorm・spectral・集中/測定方式の対象差、COS/MLMC/SCU等は回文・共有乱数・cross termの先行例として位置付ける。Kato/DaviesはGPT-T1 §13.2が挙げる一般的背景で、今回M1-Sへの適用定理を検証したとはしない。

PR原論文を誤りとして覆した、新規性が世界初と確定した、既存channel保証より有用な一般定理が既に得られた、という結論は出していない。

## 18. 公開確認・未実施作業・mandatory STOP

専用branch `track-b-m1s-g0t1-s0-contract-repair-20261016` を旧契約の固定commitから作成した。原本保存を先にcommit/pushし、再取得を照合済み。候補文書・索引の公開もremote SHA、再取得bytes・SHA256を確認し、payload commitと後付けの[公開確認receipt][ARTDIR]を区別する。

文書検査は18節の要求対応、数式参照の出典、Markdown表・参照・固定Git path、原本bytes/hash、旧索引本文・manifest既存entry、親HEAD/index/既存dirty file保護に限定する。**科学runner・科学実験テストは実行しない**。文書検査PASSを定理成立や科学的GOとしない。

新規定理証明、T1a、T1b、数値反例探索、Gate 0/E0、行列評価・simulation、sampling/trajectory/operator guard、回路生成・compile・synthesis、新資源forecast/最適化、Q4/Q5の研究実行はすべて0。既存source/result/classification/freeze/authorization/markerは変更しない。文書・manifestには今回の暫定方針と候補準備状態だけを追加する。

将来科学計算が承認された場合は、既存の性能方針に基づく支配処理・CPU/GPU/メモリ・batch/cache・thread整合のpreflightをその実行前に行う。今回backendやworker数を選んでいない。将来新規契約に科学phase/total wall-time capを既定で追加せず、tool polling timeoutと区別する。既存凍結runの条件は変えない。

原本欠落・版不明、保全不能、今回の整理だけでは閉じない科学判断が不可欠となった場合、推測で埋めず未完了欄として止める。**今回の作業完了後はmandatory STOP。S0候補の正式採択判断をユーザー・GPTへ戻し、T1a/T1b・Gate 0・Q4/Q5を開始しない。**

[GPT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d1039afae54520f15545177a7c852407830a0927/docs/research/track_b_m1s_g0t1_claude_gpt_independent_scientific_review_20261016.md
[OLD]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0f0952614dcf5c06d3d63c21b5d49091b8ecce37/docs/tracks/algorithm_codesign/m1s_g0_proof_and_gate0_precontract_20261015.md
[OLDINDEX]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0f0952614dcf5c06d3d63c21b5d49091b8ecce37/artifacts/track_b_m1s_g0_contract_preparation/2026-10-15/v1/evidence_index.json
[PREV]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4f5114e137563bb871c9f881f9eadab0c180bdd4/docs/research/track_b_m1s_claude_final_gpt_independent_scientific_review_20261015.md
[PRE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8b7666d0270f7fac01e430e195bef7b58ac0116c/docs/tracks/algorithm_codesign/m1s_g0_pre_prior_art_protocol_and_cost_decision_packet_20261015.md
[P05]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ed3ddeacc88e998e7bc1d8c0db8163e106312791/docs/tracks/algorithm_codesign/m1_p05_theorem_protocol_and_resource_preflight_20261015.md
[COMMON]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0f0952614dcf5c06d3d63c21b5d49091b8ecce37/docs/research/research_project_instructions_gpt_codex_review_approval_2026-10-12.md
[INDEX]: ../../../artifacts/track_b_m1s_g0t1_s0_contract_repair/2026-10-16/v1/evidence_index.json
[ARTDIR]: ../../../artifacts/track_b_m1s_g0t1_s0_contract_repair/2026-10-16/v1/
