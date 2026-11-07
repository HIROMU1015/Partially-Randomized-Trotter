# Track C（O3）とTrack B：接続可能性の静的調査

2026-11-08 JST。主対象は **`94b6499e4ce3675dd6eabb0e7c0e3657e1911dc9`**。
専用branch：`codex/track-c-b-connection-survey-20261108`。

**同じ有限Taylor平均へ接続することは数学的に可能で、最小候補は既存closed P3をCのtailへ適用する方式である。ただしadapter・有限bit/native誤差・重み付き統計・18round価格の更新が必要で、統合済み・資源改善済みとは認定できない。** General Full Returnも条件付きで接続可能だが、通常の無重みHadamard測定へのnormalizerだけの置換は不適格である。

本報告は既存SOURCE・報告・保存結果の読解と、Git/blob・opaque raw bytesの照合に限る。科学moduleのimport、runner/test起動、新しい数値代入・再集計・再ランキング、sampling、回路build/compile、アルゴリズム実装は行っていない。以下の数値は既存報告の引用である。提案は未採択・未実装・未実行であり、公開照合後 **mandatory STOP**。

## 1. 対象・版・取得範囲

| 系列 | 確認した固定版・役割 | 今回の扱い |
|---|---|---|
| C O3 | [指定commit][C0]。D内部調整→bottleneck→共同isometry→対角融合 | 主対象。既存の保存H4モデル・保証・価格の条件を保持 |
| B G6/G7 | [方法ノートv1.1][B11]、`243eea8e840f3c472753f7935e510391626fd325` | 正有理p・involution・odd m・short stepの既存構成とgenerator |
| B後続草稿 | [v1.2 claim表][B12]、`5f750a23f023d75c8c4fbcabc6937d6953ce61f2` | 後続の解釈・限界。新しい科学SOURCEや採択済みの統合方式ではない |
| B分子C1 | [成果総括][BOUT]、`185ec4e9d6591f4ee8d7b047c2697bed7be7bc2e` | R1入力、adapter、選択correctness、tail-only探索と一時凍結 |
| B synthetic G9/G10/D1–D3 | [利用範囲マップ][BMAP] | P3/P5、保証/単価の交換、負結果の参照。Cの分子価格には流用しない |
| B M1-S | [統合報告][M1]、`5f079dbe67bb35eff35613857b169fdf812af4b8` | 共有±branchの別平均。CのP3 samplerとしては移植対象外 |

指定C commitのGitHub commit/treeは実取得した。BのG7 SOURCE、v1.2 claim表、C1総括、旧Ordinary/Full summary・旧P5 terminal・P5R terminalも固定SHAでGitHubから実取得し、取得したblob SHAをローカルGit objectと照合した。旧branch名が消えている場合があり、C1総括のbranch ref取得は404だったが、固定commitのファイルは取得できた。branch名の消失と成果の欠落を混同しない。

SERVER 1の今回のB–C接続調査は、2026-11-08 06:45 JSTまでの公開branch検索（`connection`、`compat`、`20261108`）と、関連語・日付付きcommit検索で**特定できていない**。これは未公開や不存在の証明ではない。旧SERVER 1収録索引の存在も、今回の調査の取得を意味しない。SERVER 1の新しい結果を取得済みとして引用していない。

## 2. Track Cの現構成

### 2.1 科学条件とD内部BCH調整

[指定版の最新報告][CDIAG]と[D内部報告][CD]の条件は、Linear H4、1.00 Å、STO-3G、保存DF rank12・generation order、Partial L_D=3 / Full L_D=12、8 system qubits、4電子sector70、保存元Hのexact groundを利用可能とするもの。状態準備費は除外。18round（m=0,…,17）、δ0=.02、ε_E=1.5936e-4 Ha、α総和≤.05、位相予算.4 radである。

Dはone-body＋先頭3 DF block。既存D対称S2をS_D(t)とすると、調整構成は

\[
D_v(t)=S_D(vt)S_D((1-2v)t)S_D(vt),\qquad
U_h=e^{-ihc}\,D_v(h/2)R(h)D_v(h/2).
\]

cは保存constantと抽出identityの和。SOURCEの時間順は`track_c_d_internal.schedule`、行列では各applicationを左へ掛ける。D/Rの境界を跨いでR同士をまとめたり、異なるbasisを可換扱いしたりしない。

既存の関係は`H2_PR = H2_outer + η(v) K_D/4`、η(v)=2v³+(1−2v)³。ground期待値でD内部とmixedの三次biasを相殺する。調整0123はv=1.4768778287800015、0321はv=1.490624664134615。負時間はD内部にあり、Rは正時間h一回。H2演算子自体はゼロでなく、一般の作用素4次公式ではない。[CD]

最新policyはroundごとの既存構成/h/resolutionを選ぶため、全roundを一律D調整と呼ばない。[bottleneck報告][CB]では支配的なm16/17がD調整0321、h=.32、resolution32/64。m17は8192 outer step・8388608 RTE event、log B≈.27869543、rho下限≈.756675。最新対角融合はこれらのD係数・選択点・canonical信号を変更していない。[CROUNDS]

### 2.2 Canonical K2はP3平均

`rte.finite_taylor_operator`は偶数cutoff Kに対して次数K+1まで残す。したがって **CのK=2 ↔ Bのm=3**。Bのm=5、closed P5や「P3+tailをm=5で使った試行」はCのK2と同じfinite targetではない。

Cのsigned Hermitian involutionをQ_i、正規化label確率をp_i、A=Σp_iQ_i、x=λ_R |t_R|/n、σ=sign(t_R)と書けば、ideal microstep targetは

\[
M_3=P_3(-i\sigma xA)=I-i\sigma xA-\tfrac{x^2}{2}A^2+
\tfrac{i\sigma x^3}{6}A^3.
\]

正時間ではσ=+1。Cは偶数order k∈{0,2}を

\[
w_k=\frac{x^k}{k!}\sqrt{1+\bigl(x/(k+1)\bigr)^2},\quad
B_{\rm ord}=w_0+w_2
\]

に比例して選び、rotation label＋k個のproduct labelを独立に選ぶ。eventは

\[
V=(-1)^{k/2}e^{-i\sigma\arctan(x/(k+1))Q_j}
Q_{i_k}\cdots Q_{i_1},\quad \mathbb E[V]=M_3/B_{\rm ord}.
\]

SOURCEの`_make_event`では入力indicesの先頭がrotation、残りが**回路時間順**のproduct。`event_unitary`はproductを左へ掛ける。CのBはnormalization、microstepのn回とouter stepを合わせた積が全roundのattenuationの逆数になる。[CRTE][CH4]

行列referenceの`tail_operator`はP3を直接使う。一方、実SOURCEのfloat確率・atan/sin/cos・finite-bit CDFはideal P3とbitwiseに一致するという契約ではなく、既存のSOURCE→mean・角度・CDF・数値diskで差を会計する。統合時にも「ideal finite operator一致」と「digital/native実装差」を分ける。[CSG][CDQ]

### 2.3 回路interfaceと相対位相

Cのeventにはcomponent ID、係数符号、identity、basis ID/hash、Z/ZZ support、product/rotationのrole、phase、rotation angleがある。signed Q_i=B_i Z_S B_i†として、basisをcontrolせず、中央Z/ZZ product・rotationをcontrolする。負product符号、Taylorの−1、constant/identityはancillaの相対位相に残す。

[共同isometry SOURCE][CJI_SRC]はevent/application列をflattenし、最大same-source-basis runのsupport和集合を作る。各列のprojectorを保存するpartial QRとsupport mappingを用い、元basisに戻す。span一致だけでは足りない。639登録集合中625採用というcatalogueは有限の既存証拠で、未登録・高rankはfull-basisへfallbackする。[CJI]

[対角合成SOURCE][CF_SRC]は、run内の可換なZ/ZZ rotationとproductを受け取る。符号位相はcallerが保持し、角度はexact rationalで加算後1回丸める。controlled rotationをancillaを含むparityへ写し、既存GraySynthでidentity-ending networkを作る。異なるbasisやD/Rを跨ぐreorderは行わない。これは**固定event列を安く実装する層**で、returnをTaylor係数へ集約するBとは別の処理である。

### 2.4 18roundのshot・価格と保証の限界

Cは通常の各軸±1 Hadamard測定を使う。rhoはraw信号の半径下限で、β_statはPF、有限RTE、SOURCE/native、数値等を差し引いた残余。一般の軸別αでは

\[
N_{m,b}\ge\left\lceil\frac{4\log(2/\alpha_{m,b})}
{\rho_m^2\sin^2\beta_{{\rm stat},m}}\right\rceil,
\quad \sum_{m,b}\alpha_{m,b}\le .05.
\]

一様α=.05/36ならlog(1440)。新しいweighted variableに同じrange=1の式を無条件で使わない。E0を渡さないdecoder、alias条件、3×.4<πと最終energy幅も保証の一部である。[CD][CSG]

最新[対角融合報告][CDIAG]のRZ優先policyは、PR 17164 shots・総RZ **1.42343e12**・CX **1.97230e11**、同じ既知control＋融合を与えた強Fullは14232 shots・RZ **5.66091e11**・CX **1.01463e11**。PR価格のMC SEはRZ1.37133e10、CX1.57153e9。これらは保存報告の引用で、新しい比較計算ではない。m17巨大whole回路の直接compileやglobal最適価格ではなく、具体的modular回路の加算価格。depthはserial上界である。

Compiler AはQiskit1.3.0、rz/sx/x/cx、definition lowering・固定inverse cancellation、微小角cutoffなし。**既にcontrolledなjoint回路全体に1個だけ共通global phaseを許す**。systemだけをlowerして後からcontrolする経路やbranchごとのphase fittingは未資格。元opt2のSTOPは維持する。[CA]

旧全18round資格をそのまま新RAへ付け替えられない。Cの既存資格はcanonicalのatan(x)、atan(x/3)・全support・basis/native区間、SOURCE generator、fresh独立finite bits、既存角度和diskに結合している。RAでは新角度・law・meanの評価が要る。保証は保存Hermitian-H・exact ground・ideal native・fresh bitsモデルの条件付き十分条件であり、元の連続積分Hの無条件包含、実機ノイズ、状態準備、T合成、一般分子の保証ではない。

## 3. SERVER 2に残るTrack B成果と原本

### 3.1 既存資産の所在と公開/ローカル版の区別

以下のローカル相対pathは`partially-randomized-trotter-worktrees/`を基準とする。原本を本branchへコピーしていない。

| 資産 | 版・科学条件・保存状態 | Cへの適用可能性 |
|---|---|---|
| G6/G7 SOURCEのGit object | v1.1固定SHA `243eea8…`。`return_aggregation.py`と`g7_generator.py`はC指定版に収録された同pathとbyte一致 | 正有理p、0<x≤1、odd m、involutionの構成を再利用可能。Cの新しいp/xへの保証適用は別 |
| v1.2草稿/claim表 | `5f750a23…`。C指定tree内には当該claim表がないが、別固定objectとGitHubから取得できた | C treeにないことは未公開の意味ではない。ideal moment/price相殺・small-x・H4の限界を保持 |
| C1 R1入力 | [R1][R1] `2937e6b…`。snapshot `3bc92e92…`、新partition、13 basis/324 labels、exact dyadic係数と正有理p | Cの218成分の直接入力ではない。旧PR2の元basis列がなく、bindingは`MODEL_BINDING_UNCLOSED` |
| C1 sidecar adapter | [adapter][BADAPT] `f3a6aa11…`、R2 [限定correctness][BR2] `068f0ae8…` | signed DF-Q、actual adjoint、controlled phase、X/Yの再利用材料。元bindはm=5固定、R1固定。一般C/m3へそのまま呼べない |
| C1 whole-path | [SOURCE接続監査][BC145] / [whole_path.py][BWHOLE] `af75af03…` | ordered event・weight積、全trial分母、prequantum zeroを人工接続。`SYNTHETIC_ONLY`・L≤3・m5等の固定条件を削除して済ませない |
| C1-5A旧run | [失敗報告][BFAIL] `b022f2ea…`。T=.8、q_outer=1、r=4、m=5、σ=+、microstep=.2、各1024 path | tail＋scalarのlogical探索。Ordinary/Full完走、P5 worker異常終了。C全18round価格/shot保証ではない |
| 旧runのraw原本 | `track-b-c1-5a-h4-limited-resource-pilot-20261021.runtime/raw_preserved/result/`、同runtimeのordinary/full_return/closed_P5_full.tar.gz | 3 archiveのhashが[公開raw索引][BRAW]と一致。登録229/246/38 files。trajectoryを展開・科学再解析していない |
| 旧runのsetup/summary/terminal | 上記raw_preservedのOrdinary3件、Full3件、旧P5 setup/terminal2件 | 8件すべて公開固定Git blobのbytesと一致。別の未公開成功結果にはなっていない |
| P5-only別実行 P5R | [報告・比較][BP5R] `bb9059ba…`、SOURCE/新実行来歴を持つ別run。runtime `track-b-c1-5a-p5-worker-recovery-20261021.runtime/` | 1024 path完走。旧P5 STOPのresume/成功への書換えではない。m5・tail-only・exploratory |
| P5R保存raw | 同runtimeの`raw_preserved/result/` | 公開raw索引の406 memberをopaque bytes/hashで照合し、406一致・欠落0・不一致0。内部trajectoryの数値は読出し再集計していない |
| C1凍結記録 | [一時凍結と再開条件][BPAUSE] `185ec4e9…`。同runtimeに最終STOP確認が残る | C1追加native/大規模比較を再開しない。今回の調査は旧one-shotの再利用ではない |
| 非公開原本2件 | C1 freeze runtimeの既存非公開保全metadataから、2件の存在・bytes/hash一致を確認 | 原文、私有path、hashは非公開を維持。本報告の公開根拠に代用せず、転載しない |
| 旧worktreeの保存 | `_workspace/WORKTREE_CLEANUP_20261104.tsv`にv1.1等のcheckout整理・固定SHA保持の記録 | 消えたcheckoutを欠落と扱わずGit objectを直接読む。既存archive/bundleを変更・復元実行していない |

今回調べたSOURCE/summary/raw範囲では、公開版と異なる未公開の新しい科学的成功結果は確認しなかった。非公開原本、runtimeの補助記録、rawの展開済みコピーはローカルに残るが、独立した新成果や別試行として数えない。全ローカル資産・全archiveの不存在/同一性を網羅的に証明したという主張ではない。

### 3.2 CとBの分子入力は同一視できない

| 条件 | C指定版 | B C1 R1 |
|---|---|---|
| 分子ラベル | Linear H4/1Å/STO-3G/DF12/LD3 | 同じ名目条件 |
| 保存入力 | DF NPZ hash `13e4b10d2347ed900fe4aa4b2238128ebe735a48988e321b9e4fabac4d092a3b` | snapshot hash `3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a` |
| label/basis | 218 signed tail components | 324 labels/13 basis、R1新partition |
| state/context | 元H ground参照hash `6a350e7d…`、4電子sector70 | 旧PR2保存state、snapshot内sector36、state hash規約も別 |
| 時間・次数・scope | round別h/resolution、K2/P3、D＋R＋X/Y・18round | T=.8、r4、m5/P5、TAIL_ONLY_PLUS_SCALAR |
| 元Hへのbinding | 保存行列モデルの条件付き保証 | R1 exact model定義と数値binding。rigorous binding未閉鎖 |

Cのλ_R=.5832704246339503とB旧報告の近い値を理由に、係数・basis・state・SOURCEの同一性を主張しない。NPZ hashの違いだけから物理Hamiltonianの差の大きさを推定することもしていない。[CINPUT][R1][BOUT]

### 3.3 負結果・途中停止の意味

C1-1のpartition hash不一致→D0→R1新来歴、C1-3のstate hash定義不備、R1 recoveryのreference bytes不一致、R2のsigned-zero復元修正は別記録である。R2で保存されたcorrectnessは選択6 event、label0/2/9/36、動的reference basis2件の範囲。全324 label・全native/全pathへの保証ではない。[BOUT][BR2]

旧C1-5A P5は`BrokenProcessPool`でSTOP。workerのexit/signalと処理済trial数はUNKNOWN、OOM等を断定しない。欠測をzero/rejectや費用0へ置換しない。P5Rはsingle producerとfresh exec provider等による別実行であり、科学的失敗を覆した試行とは呼ばない。[BFAIL][BP5R]

C1の保存3方式では、accepted1024/1024、zero path0、degree0 parentが大半。各pathに4 child rotationが残るので、degree0は無料identityではない。保存weight二乗平均はOrdinary/Fullが1.1141049662602474、P5Rが1.1105279232736538。これを認証M2/母分散や十分Nへ使わない。logical差は小さく、method別PRNG系列のためpaired改善を認定していない。[BCOST]

G7の旧比較利益、G9/G10の強closed対照による反転、D3の保証/単価によるP3+tail/P5の順位交換、P_U上界の反例を保持する。これらは固定synthetic provider等の限定結果で、分子Cのno-goでも資源改善証拠でもない。B旧D1のP5用x⁶ remainder、固定α/shot cap、全event価格列挙をCのP3へコピーしない。[BMAP][B12][BDACC]

## 4. 接続互換性表

「接続可能」は既存の数式/APIの意味が接続できる範囲、「条件付き可能」は新しいbinding・資格・会計が必要な範囲。「不適格」は提案された同一視/無変更移植に対する判定で、RA一般の数学的不可能性ではない。

| 調査項目 | 照合結果 | 分類・必要条件 |
|---|---|---|
| 1. 有限Taylor平均 | C K2とB m3は同じP3を構成可能。G6はfirst operator momentの恒等式 | **接続可能（ideal定義）**。同じA/x/σ/ordered operator、Bの0<x≤1を各microstepで満たす。B m5の結果は直接不適格 |
| 2. label・符号・作用順・identity | Q_iをsigned basis-conjugated Z/ZZとすればinvolution条件に合う。G6のwordは左→右operator順、時間順はreverse | **条件付き可能**。Cの218 componentへ新binding。signed-pを渡さず符号をQへ入れ、identity scalarを一度だけancillaへ。B R1のlabelを直結しない |
| 3. proposal/weight/rejection | closed P3はgroup/conditional child law、Fullはordinary envelope＋raw非reduced/coin拒否。実dyadic πW=α̃ | **条件付き可能**。rejectionを後述の全trial zeroへ。元raw語をreduceして別parentへ再割当てしない。技術failureをzeroにしない |
| 4. normalization/測定 | Cのraw平均はM/B、B digitalはweighted meanがM̃。FullのBはordinary envelopeで、B_fullではない | **無変更置換は不適格**。閉P3の理想constant-weight場合と、digital weighted/zero-fillの場合を分けたsignal/shot adapterが必要 |
| 5. finite-bit/native | G6 F1/F2とFull F3は利用可能。canonicalの一様preflightは未完成。RA角度はparent依存 | **条件付き可能**。CのCDF誤差・旧native angle表だけでは覆わない。係数root・angle・mean/native積誤差・全support/既存fallbackを対応付ける |
| 6. controlled/X/Y | B eventのparent phaseとsigned-Qを保ち、Cのdiag(I,V)、X:H、Y:Sdg→Hで読める | **接続可能（構造）／資格は条件付き**。joint共通phaseのみ許容、actual adjoint保持。B system-only phase最適化・後付controlは不適格 |
| 7. 共同isometry/対角合成 | RA eventもsigned product列＋単一Q rotationなので中央applicationへ展開できる | **条件付き可能**。event→application adapter、任意parent angle、phase ownershipが必要。未登録supportは既存full fallback。既存catalogueの利益率は移送しない |
| 8. 全18round shot/価格 | ordered mean積とweight積、36軸failure合成、C modular費用への接点はある | **条件付き可能・未検証**。認証range/M2、信号半径/残余位相、全trial数とaccepted quantum calls、同一taskの期待native費用を新しく結ぶ |

追加分類：

| 候補 | 判定 | 理由 |
|---|---|---|
| **既存closed_P3_tailをm=3でCへ適用** | **推奨・条件付き可能** | Cの有限target/最大parent長に合わせられる。Green/Full拒否を必須にせず既存generatorを使える |
| General Full Returnをm=3で適用 | 条件付き可能、比較/拡張候補 | idealには同じRA pairing。未知normalizerなしの生成は可能だが、zero-fill・range/M2・古典query費が残る |
| Partial Returnをm=3で適用 | 条件付き可能、軽量対照 | 既存group生成を使える。closed P3と同じlabel-dependent child lawではない |
| Closed P5 / m5のP3+tailをそのまま採用 | 現C固定条件へは不適格 | P5はP3と別finite mean。m5への変更は新科学条件 |
| M1-S shared ±をnormalizer対策として採用 | 現Cへの無変更移植は不適格 | 二枝のcoherence superoperatorはP3 first operator momentと別。ordinary diag(I,U)と同じ保証ではない |
| B toy native/CTS priceや旧C1 logical単価の流用 | 不適格 | provider、入力、精度、control、回路scopeが異なる |

## 5. 数式・推定量をどう接続するか

### 5.1 推奨する最小接続：C入力を保持したclosed P3

既存`CanonicalGenerator(..., m=3, arm='closed_P3_tail')`を使う案とする。m3ではdegree4以降のordinary tailはなく、closed P3そのものになる。G6のadjacent return reductionはQ_i²=Iだけを使うので、異なるDF basisが非可換でも構成自体は適用できる。Pauli交換・反交換や異なるlabelの物理的同一性まで自動収集する方式ではない。[BGEN][B11]

SOURCEとの照合で、χ=Σp_i²、μ3=Σp_i³に対し空parentの係数は

\[
a_\varnothing=1-\chi x^2/2,\quad
s_\varnothing=x-(2\chi-\mu_3)x^3/6,
\]

child iの質量は`p_i[x−(2χ−p_i²)x³/6]`。非空parent u=(j,k)、j≠kでは角度のtangentは`x(1−p_j)/3`、childはi≠jとなる。これは既存G7 SOURCEのm3への記号的対応であり、Cのχ・return率・normalizer改善率を今回計算したものではない。

**角度がorderだけで決まらなくなる**ため、`RTEEvent`のorderからatan(x/(k+1))を再構成する既存入口へ偽装して渡さない。B eventから明示的なrotation angleとsigned product applicationへ写す小さいadapterを用意する。parent phase、wordのoperator/時間順、元label、proposal、weightは保持する。

入力はB R1へ差替えず、Cの既存component probabilityをexact binary rationalとして正規化したp、既存λ_R・時間規約からのxを使うのが最小である。CのSOURCE保証もfloat probabilityをexact rationalとして正規化したAを扱う。`Fraction(p_float)`を各々渡すだけでΣp=1と仮定しない。元係数・λ・保存tail行列との差はCのgenerator/binding誤差として保持し、新しい定義との差を必要な範囲だけ照合する。今回この入力変換を実行していない。[CSG]

C側ではD係数、state、Hamiltonian、18roundの時刻、h/resolutionを最初の比較で保持する。sampler変更とstep数変更の寄与を同時に混ぜない。closed P3も新しい実行範囲として資格を付け、既存C/Bの封印SOURCEを編集して再開しない。

### 5.2 重みと全trial分母

Bのaccepted eventの実proposalをπ_e、有限係数をα̃_e、weightをw_e=α̃_e/π_eとする。全pathを量子実行前に選び、j番目のmicrostepで正当なrejectがあればA_path=0、全てacceptedなら1。X/Yの観測をZ_b∈{−1,+1}として

\[
Y_b=A_{\rm path}\Bigl(\prod_jw_{e_j}\Bigr)Z_b,\qquad
\widehat z_b=\frac1{N_{\rm trials}}\sum_{a=1}^{N_{\rm trials}}Y_{a,b}.
\]

分母は完走した固定全trial数である。accepted数で割らず、完走前prefixに全Nのconfidenceを付けない。技術failureは母集団の正当なzeroではない。各stepを独立に選べば、weighted operator meanはD/scalarを挟んだordered M̃_jの積に対応する。M̃_jをunitaryと仮定しない。[BWHOLE][BDACC]

理想closed P3をそのnormalizerで直接sampleする場合は各stepのweightが定数B_RAとなり、raw無重み信号を`z_P3/ΠB_RA`と書ける。この**理想定数weightの場合に限って**C型のattenuation会計に対応づけられる。現実のdyadic補正weightはevent依存で、係数・角度・native近似も残る。

General Fullはordinary envelopeからraw語を引き、非reduced/coin rejectを量子前zeroとする。実装の`generator.B`をB_RAと読み替えない。Fullの理想accepted ensembleとclosed法が一致しても、実proposal・zero率・range・必要Nは一致しない。

weighted変数には認証一様rangeとsecond momentを使う。独立stepについて、pathのrangeは各stepのweight上界の積、second momentはzero込みE[A_jw_j²]の積で評価できる。既存Bernstein型会計を**任意の36軸αとCの残余幅に適用できる形へ**結ぶ。`g7_generator.budget`は旧16軸固定confidence、D1 `whole_budget`は旧P5契約なので、関数の出力値をCのNとみなさない。

### 5.3 正規化だけで総資源改善を言えない理由

[既存v1.2のC08][B12]は、ideal ±1 outcome・同じprovider・accepted平均価格・second-moment支配の尺度に限ると、General Fullのrejectによる量子呼出し減とweight二乗増が相殺する関係を整理している。range項、digital bias、有限N、古典費用、task全体の費用定理ではない。

さらにRAはchild law・親word・角度を変えるため、同一basis runの分布とpriceも変える。既存融合は高次数returnでなく通常rotationのbasis費が支配する条件を報告している。正規化が下がっても、親/childのbasis、価格/層確率、angle key数、weight range、準備されるpath数がどう変わるかを評価しなければ総Gは決まらない。

全pathを事前選択する構成なら、量子価格は各軸について

\[
G_j=\sum_{m,b}N_{m,b}^{\rm trials}\,
\mathbb E\big[A_m C^{\rm accepted}_{j,m,b}\big]
\]

で、accepted回路にはD、RTE、scalar、X/Y wrapperを含める。zero pathでも古典sampling/query/angle処理等は支払う。逐次生成で量子prefixを既に実行した構成ならprefix費を別に加える。旧Cの`N_quantum shots × unconditional occurrence price`との単位を混ぜない。認証十分Nを最小Nと呼ばず、価格MC SEと科学誤差diskも分ける。

## 6. 再利用するコードと必要なadapter

| 再利用対象 | 既存機能 | 接続時に必要な最小変更・留意点 |
|---|---|---|
| [G7 `CanonicalGenerator`][BGEN] | ordinary / partial_return_tail / closed_P3_tail、group/conditional law、実π/weight | Cのp/x/m3を受け渡す。既存m3入口があるのでsamplerを再実装しない |
| [G6 `ReturnKernel`, `FullReturnGenerator`][BKERNEL] / [BGEN] | 局所return/parent/child、Full拒否、有限weight | Full比較時だけ使用。F3の既存Full preflightを適用し、canonical保証と区別 |
| [F3][BF3] / `g6_f3_precision.enclose_atan` | FullのP0–P5、有限bit、angle enclosure | Cの入力・precisionへ結合。P0–P5の検査名とTaylorのP3/P5を混同しない。canonicalの未完成証拠を補う範囲を限定 |
| [C1 `DFAdapter.append_q`][BADAPT]、[family][BFAMILY] | signed Z/ZZ、basis、actual adjoint、phase、X/Y | SOURCE例として再利用。R1/m5/synthetic固定条件を無断解除しない。C label registryへ薄いadapterを作る |
| [C1 `WholePath` / `AllTrials`][BWHOLE] | 独立stepのweight積、zero、全trial分母、未完走拒否 | CのD/R schedule・36軸へ結合。既存は人工scopeであり、production RPE完成と呼ばない |
| C `track_c_d_internal.schedule`, `track_c_d_resources.split/assemble` | D内部構成、D/D junction、scalar・D/R順 | sampler変更の最初の対照では保持。R occurrenceを新providerへ差す |
| C `track_c_joint_isometry.append_run` / `build_joint` | 同一basis共有、列projectorとmapping、fallback | 現`runs()`はcanonical choicesからeventを作るため、RA applicationを受ける入口が必要。部分basis構成そのものを再実装しない |
| C `track_c_diagonal_fusion.diagonal_word/emit` | signed product、exact角度和、parity合成 | 明示RA角度を渡す。phaseをcallerと二重に足さない。未知supportはfallback。angle/support由来の新誤差だけ追加 |
| C `compile_reference`、既存local interval関数 | already-controlled loweringと定義別区間評価 | 同じAを使い、新RA角度・符号・control文脈だけ必要な資格を追加。旧角度表を新角度の証拠にしない |
| C `track_c_source_guarantee`、`track_c_d_qualify`、joint/diagonal資格 | generator、保存H/ground/alias、source/native/数値disk | p/x/有限平均の差、weighted native平均、nonunitary telescopingを対応付ける。過去diskを二重加算しない |
| C price/α/decoder pipeline | D/R分解、paired共分散、holdout、36軸α、branch復元 | weight/range/M2に対応するNとaccepted価格へadapter。新lawの価格・rare層確率を用いる。E0非依存decoderは保持 |

旧D1価格pipelineは参照eventを全列挙する経路があり、Cの218-label高次数価格取得へ無条件で移さない。新しい一般framework、approval証明・認証server・ledgerの追加は不要。必要なのは科学的input/mean/measurement/費用の薄いbindingである。既存固定実行契約の条件・STOPは別途保持する。

## 7. 最小科学検証案と評価指標（未実行）

これは次回の実行承認を検討するための必要項目で、実行計画の採択や計算許可ではない。最初はCの保存入力・時刻・D/h/resolution・Compiler Aを保持し、**m3 closed P3対ordinary**を主比較、partial/Fullを必要な範囲の対照とする。P5/new molecule/new scheduleを混ぜない。

| 最小項目 | 何を確かめるか | 指標・判定の意味 |
|---|---|---|
| 入力/finite mean | Cの218 labelsのp・符号・basis・identity、全使用xのshort-step条件。ordinary/RAが同じP3を対象とする | operator差と説明済み丸め/binding上界。小人工非可換例で語順・negative sign・identity・x=0を確認し、C入力は既存保存参照との必要な照合に限定 |
| 実sampler/weight | group/conditional proposal、raw/coin zero、πW=α̃、positive support。fresh bits、失敗とzeroの区別 | 認証range/M2/root/angle bias。経験的sample meanや観測最大weightだけで証明しない |
| controlled/native | parent位相−1、負Q、actual adjoint、全512次元jointとX/Y、新RA角度、既存run合成 | joint共通phase下の作用素差・X/Y差、追加native区間上界。独立branch phase fitting不可 |
| whole path・18round | 全trial推定量とordered mean積、PF/RTE/native/finite-bit/数値誤差、alias/decoder、36軸α | rho下限>0、βstat>0、α和≤.05、整数N、最終energy誤差/branch条件。旧保存Hモデルの条件を超えない |
| 期待量子価格 | 新lawのwhole occurrence価格、D・junction・wrapper、zeroで回避する回路、paired covariance | 総RZ/CX/size、serial depth上界、trial数/accepted quantum calls、MC SE。価格holdoutを使って再選択しない |
| 古典費用と機構 | setup、parent/child query、selector/cache、angle種類、bits/weight演算、I/O。same-basis run/parentの実影響 | CPU/wall・memory・angle misses・受理率。normalizer、M2、basis費の寄与を区別。raw rare頻度を物理確率としない |
| 公平な総比較 | ordinary＋同じ共同isometry/対角合成、実行可能な同条件強Full PFにも同じ最適化を認める | 同ε_E/α・state/cost scopeでΣN E[C]を比較。RZとCXを別々の回路の最小値で合成しない |

最初の科学的問いは「Cの同じP3 targetで、RAのnormalization/weight改善がbasis/native価格と古典費用を含めても利益として残るか」。利益が残らなければ、固定小x条件の負結果として保全し、別次数・条件・研究の再開へ自動進行しない。結果が僅差ならその差に相応のMC精度と誤差余裕を要求する。統合前に全未知を網羅する大量test/証明を追加する案ではない。

## 8. SERVER 1から必要な情報

今回の新しい調査について、以下は未取得である。

1. **取得可能なrepository/ref/完全commit SHAと報告path**。公開済みなら固定リンク、未公開なら共有可能な調査結果とSOURCEの指定。未公開原本の一括公開は不要。
2. Bの対象がG6/closed P3/General Full/closed P5/M1-Sのどれか、SOURCEとinput identity、m/K、時間・p・label/符号/basis・identity・作用順。C指定版を前提にしたか。
3. 提案する実proposal・weight・reject、finite-bit/angle誤差とcertificateの適用範囲。無重みmeasurementとの対応があるなら、その成立条件。
4. controlled実装、joint共通phase、actual adjoint、X/Y、Cのisometry/対角合成を含むか。価格・shot数がtail-onlyか全D/Rか、全trialかacceptedか。
5. 保存結果・raw/途中停止・負結果、同task強対照、全18roundへの資格。新計算を行った場合は承認済みscopeと結果の範囲を示し、本調査の未実行事項と区別する。

情報取得後の静的照合は、両serverのSOURCE/input/推定量/誤差/価格の差を確認する作業になる。SERVER 1未取得を埋めるためにSERVER 2で新計算を代行していない。

## 9. 完了条件とSTOP

公開物は本調査報告と入口リンクのみ。非公開原本・cache・raw・実行SOURCE・旧branchの内容は変更していない。今回追加した科学計算、sampling、compile、アルゴリズム実装、他Track研究の再開は **0件**。

公開前にMarkdown/固定参照path、引用と原報告、diffの範囲を静的確認する。commit/push後はremote branch SHA・treeと報告blob/内容を照合する。これらの文書/Git確認を科学的PASSや独立科学レビューと呼ばない。

**その照合後mandatory STOP。統合アルゴリズムの実装・科学検証には進まない。**

次の担当：**ユーザー**がSERVER 1の固定調査情報を指定し、closed P3の限定統合検証を行うか、その科学条件を判断する。**Codex**は新しい明示指示・必要な科学実行承認がある場合に限り、その範囲を担当する。現在実行中の後続作業はない。

## 固定出典

[C0]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/94b6499e4ce3675dd6eabb0e7c0e3657e1911dc9
[CDIAG]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/94b6499e4ce3675dd6eabb0e7c0e3657e1911dc9/docs/research/track_c_diagonal_fusion_20261108/REPORT.md
[CD]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/94b6499e4ce3675dd6eabb0e7c0e3657e1911dc9/docs/research/track_c_d_internal_20261108/REPORT.md
[CB]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/94b6499e4ce3675dd6eabb0e7c0e3657e1911dc9/docs/research/track_c_bottleneck_20261108/REPORT.md
[CROUNDS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/94b6499e4ce3675dd6eabb0e7c0e3657e1911dc9/artifacts/track_c_diagonal_fusion/2026-11-08/ROUND_RESOURCES.csv
[CJI]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/94b6499e4ce3675dd6eabb0e7c0e3657e1911dc9/docs/research/track_c_joint_isometry_20261108/REPORT.md
[CJI_SRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/94b6499e4ce3675dd6eabb0e7c0e3657e1911dc9/src/trottertracks/track_c_joint_isometry.py
[CF_SRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/94b6499e4ce3675dd6eabb0e7c0e3657e1911dc9/src/trottertracks/track_c_diagonal_fusion.py
[CH4]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/94b6499e4ce3675dd6eabb0e7c0e3657e1911dc9/src/trottertracks/track_c_h4.py
[CRTE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/94b6499e4ce3675dd6eabb0e7c0e3657e1911dc9/src/trotterlib/rte.py
[CSG]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/94b6499e4ce3675dd6eabb0e7c0e3657e1911dc9/src/trottertracks/track_c_source_guarantee.py
[CDQ]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/94b6499e4ce3675dd6eabb0e7c0e3657e1911dc9/src/trottertracks/track_c_d_qualify.py
[CA]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/94b6499e4ce3675dd6eabb0e7c0e3657e1911dc9/src/trottertracks/track_c_compiler_a.py
[CINPUT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/94b6499e4ce3675dd6eabb0e7c0e3657e1911dc9/artifacts/track_c_h4_pilots/2026-11-06/run01/INPUTS.json
[B11]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/docs/manuscripts/track_b_return_aggregation_native_resource_note_v1_1.md
[B12]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f750a23f023d75c8c4fbcabc6937d6953ce61f2/docs/research/track_b_g6_v1_2_draft_review_integration_20261022/CLAIM_EVIDENCE_MATRIX.md
[BKERNEL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trottertracks/algorithm_codesign/return_aggregation.py
[BGEN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trottertracks/algorithm_codesign/g7_generator.py
[BF3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trottertracks/algorithm_codesign/g6_f3_generator.py
[BMAP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/94b6499e4ce3675dd6eabb0e7c0e3657e1911dc9/docs/research/track_b_paper_claim_survey_20261018/results_usability_map.md
[BOUT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/185ec4e9d6591f4ee8d7b047c2697bed7be7bc2e/docs/research/track_b_c1_freeze_synthesis_20261021/C1_RESEARCH_OUTCOMES.md
[BCOST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/185ec4e9d6591f4ee8d7b047c2697bed7be7bc2e/docs/research/track_b_c1_freeze_synthesis_20261021/C1_RESOURCE_EVIDENCE_AND_LIMITATIONS.md
[BPAUSE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/185ec4e9d6591f4ee8d7b047c2697bed7be7bc2e/docs/research/track_b_c1_freeze_synthesis_20261021/C1_PAUSE_AND_RESTART_CONDITIONS.md
[R1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2937e6b655ba3cd77f982d6901d766355a8b92c1/docs/research/track_b_c1_1_r1_new_partition_exact_model_20261021/result/model_binding.json
[BADAPT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f3a6aa1108bbc8359e585c470e23033458c57466/docs/research/track_b_c1_2_c1_4_integrated_adapter_validation_20261021/adapter.py
[BR2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/068f0ae8771fccf28c529bda12a84abe8b7e7648/docs/research/track_b_c1_3_r2_reference_recovery_c1_4_20261021/CORRECTNESS_REPORT.md
[BC145]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/af75af03c0b334c29b5a4311bd2b4578a999460c/docs/research/track_b_c1_4_5_c1_5_technical_preflight_20261021/SOURCE_CONNECTION_AUDIT.md
[BWHOLE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/af75af03c0b334c29b5a4311bd2b4578a999460c/docs/research/track_b_c1_4_5_c1_5_technical_preflight_20261021/whole_path.py
[BFAMILY]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/af75af03c0b334c29b5a4311bd2b4578a999460c/docs/research/track_b_c1_4_5_c1_5_technical_preflight_20261021/family.py
[BFAIL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b022f2ea421649015a4c21ea3c904fe619d0c198/docs/research/track_b_c1_5a_h4_limited_resource_pilot_20261021/FAILURE_REPORT.md
[BRAW]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b022f2ea421649015a4c21ea3c904fe619d0c198/docs/research/track_b_c1_5a_h4_limited_resource_pilot_20261021/raw_archives/index.json
[BP5R]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bb9059bacb697e1105e63c1b3826b0c34886ad15/docs/research/track_b_c1_5a_p5_worker_recovery_20261021/comparison.md
[BDACC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trottertracks/algorithm_codesign/rte_development/accounting.py
[M1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f079dbe67bb35eff35613857b169fdf812af4b8/docs/research/m1s_integration_20261017/integrated_research_summary.md
