# Track B：他PR研究の横断調査を踏まえた次の研究課題の選定

- 作成日：2026-10-13 JST
- レビュー開始承認：利用者の「レビューを開始して」
- 主対象：`HIROMU1015/Partially-Randomized-Trotter` のTrack B
- 参考資料：Codexによる親リポジトリ横断調査、Track A H6の公開済み科学レビュー、過去の方法差・費用比較・closure
- 運用指示：`research_project_instructions_gpt_codex_review_approval_2026-10-12.md`
- 状態：**研究方針レビュー完了。次の限定的な候補監査のRQを選定。新アルゴリズムの新規性・優位は未認定、新しい計算実行は未承認。**

## 0. 結論

**Track Bを他系列へ置き換えず、現generic whole補正系列のclosureも維持する。次の検討は「既存DF-RTEの各eventを保った、積部分の構造的な生成・圧縮」に一度だけ絞る。**

具体的な問いは、次のとおりである。

> 既存DF-RTEでsampleされるparity primitiveの順序付き積を、少数軌道に支えられた一粒子反射積として扱い、各eventの演算子・制御位相・sampling lawを変えずに生成できるか。その生成手順に、既存のthin basis実装と一般Gaussian圧縮を同じ情報で使う対照を超える、具体的な方法差または資源上の利点が残るか。

前半のGaussian閉包自体は既知の数学から従う。**新規性の候補は閉包の事実ではなく、DFの低rank入力から実際の制御回路を生成する手順・計算量・精度会計・期待資源の差である。** 現時点では、その差があると確認できていない。

したがって、今回選んだのは大規模pilotや新手法の採択ではなく、**既知法への還元・実装上の追加手順・改善可能な費用の大きさを、先に判別する候補監査**である。独立した差がない、または改善対象の期待費用が小さく研究目標に届かないと分かれば、新しい本番検証を開始しない。

Track Aは既存レビューの下で、具体的なPR資源量と適用範囲の研究を継続する。H6の希少次数費用を先に確認する判断は変更しない。N3の縮約保証研究も、別taskとして扱う。他研究の改善率をTrack Bの利益に足したり、Track Bを無言で資源評価・縮約研究へ転換したりしない。[R1,R2,R7]

## 1. レビューの範囲と、資料の扱い

今回の入口は、ユーザーが受領したCodexの横断調査報告である。Codexは133 branchの名前・HEADと、主要系列の資料を確認し、16資料をGitHubから再取得したと報告している。これはCodex側の調査範囲であり、GPTが133 branchを全て再監査したという意味ではない。[R1]

GPT側では、次の科学的判断に関係する一次資料を選んで取得した。

| 資料 | 固定参照と今回の用途 |
|---|---|
| Track A H6科学レビュー | `0dd5c3d204dab9b731268a1aa2add621080f3d20`。既存の継続判断と未解決費用の位置付けを継承。今回取得は冒頭・対象・履歴・確認範囲を中心とし、全レビューの再審査ではない |
| DF補正closure | `4192077723770f4846cc391ea0da96c1e115f2fa`。現generic wholeの自動延長をしない境界 |
| G5終了証拠 | `23adf61f9342abff20f059be69089e89f66f7797`。指定classに強いCTS対照を戻した後の限界 |
| G10技術ノート | `22a0d032717b4139cdfa806850643e93ba25624f`。既存の非列挙生成・finite-bit会計との重複回避 |
| P-A mechanism validation | 同`22a0d032...`。同source-basis run内分割の不成立とrun-level改善の区別 |
| DF-RTE builder | `321cd8c3066194fe98741cbebbcdba1515d1deaa`、`src/trotterlib/df_rte_qiskit.py` の1–125、260–470行。product列・終端rotation・既存basis reuseの確認 |
| 以前のTrack B方針レビュー | 会話添付の`track_b_overall_research_strategy_review_20261011.md`。独立した方法研究という目的と、過去の選定理由を確認 |

BF、BM、SP、N3等の全証明や全rawを今回新たに検算したわけではない。それらの概要・最終判断はCodexの固定資料付き調査と以前の科学判断を継承する。[R1]

重要な資料の参照先が取得できることと、全rawの完全再認証は異なる。今回の結論は、全実装の最適性や全先行研究の不在証明を必要とする結論ではない。

## 2. 他の研究から何を参考にするか

### 2.1 Track A：結果そのものではなく、評価を成立させる方法を参考にする

Track AのH6では、固定8構成の独立費用標本でも、選択集合内のPR点推定がfull PFより低い結果が保持された。公開レビューは継続を推奨しつつ、希少Taylor次数の期待費用、経験的数値allowance、準備費用等の限定を残している。独立確認は同じHamiltonianと固定条件での費用標本の再現性であり、別分子一般化・厳密数値certificate・最終energy taskではない。[R2]

今回参考にするのは、以下の評価構造である。

- 同じ物理的signal task・精度・confidenceで費用を戻す。
- 探索に使った標本と確認標本を分ける。
- wrapper費用、sampling負担、準備費用、未観測eventを区別する。
- 未観測の高次数を費用0とみなさない。

Track AのRZ-work比を、新候補の見込み改善率として流用しない。新候補が役立つとすれば、その候補を評価した同入力の結果が必要である。

### 2.2 BF/BM/G系列：候補だけに自由度を与えない

BFではfinite objectiveの追加価値が強いleading対照に対して残らず、BMでは固定列のBCH scoreが既知compact再帰と一致したと記録されている。[R1]

G5は、指定toy・辞書・非負degree matching・precision自由度・full-support IS・共通confidence policyのclassを対象に、保存CTS lawとの分離を得た。この結果は、全辞書・全sampling policy・全Gaussian実装・一般PRの不可能性を示すものではない。[R5]

ここから継承するのは、**大規模計算の前に、同じ入力・同じ実装機会を与えた既知法との差を確認する**という方法である。既知枠組みに包含されることだけで新規性を否定するのではなく、具体的な生成法・保証・計算量まで同じかを確認する。

### 2.3 G6–G10：非列挙生成という言葉だけでは新しくない

G系列には、形式returnの係数query、global normalizerの全列挙を要求しない生成、有限bitの実確率と補正weight、受理・拒否を含む会計が既に存在する。[R6]

従って、新候補に「on-demand」「非列挙」「有限bit」と名付けるだけでは差にならない。今回の候補は係数やreturn lawを作り替えるのではなく、既存lawから得た一つの順序付き積の実装を変えるものとして定義する。

### 2.4 P-A：同基底内の分割が無効だったことを忘れない

P-Aのforced order-2検証は、各taskが一つのsource-basis runで、`interval_union_dp`と`one_segment_per_source_run`が全30 taskで同じplanとなり、run内分割の独立寄与が残らなかった。過去のrun-level full/support-unionの改善は別に保持されている。[R4]

今回の候補は、原則として**異なる軌道基底に属するparity因子も順序を保持して掛け合わせる**点を検討する。これは同基底内の区間分割とは別の問いだが、それだけで新規とは言えない。既存run-level選択は対照側にも残す。

### 2.5 DF補正closure：小さい支持と安い回路を同一視しない

現generic whole系列では、指定primitiveから最大6軌道に縮約できることと、その補正回転が資源的に有利であることが別だった。固定計画の局所T分離、A/Bの平均とbiasの違い、全PR費用の未確定を保ち、系列を区切る判断を継承する。[R7]

今回の提案は、その未完了T表を埋めたり、同じ非Gaussian交換子指数を別compilerで救済したりするものではない。

### 2.6 N3：別taskの成果として尊重する

N3の幅改善・条件族分離は、基底エネルギーの射影誤差を扱う証拠である。PRの時間発展の誤差や総回路費用ではない。既存のnegativeと後続の幅改善の両立、同情報の対照、保証幅と採否の区別は参考になるが、Track Bの新アルゴリズムとして移植しない。[R1]

## 3. 今回比較した研究方向

| 方向 | 判断 | 主な理由 |
|---|---|---|
| 現generic whole補正の追加T合成・phase対応・compiler探索 | 自動再開しない | closure済み。未確定部分を全て埋めることを研究目的にしない |
| 同じ有限Taylor辞書の係数・proposal再最適化 | 新候補として優先しない | G5等の既存範囲との重複を避ける。新しい辞書・生成法がなければ同じ問いを反復する |
| 単純なcost-aware importance sampling | 単独の新規性の中心にしない | 一般の回路単価に対する資源最適ISが既知。再weight自体はbiasを変えない [P1] |
| 相関付き乱択・Markov chain | 具体的な新構成なしに採択しない | 関連法が既にある。同じ周辺分布だけでは非可換な全体平均を保持しない [P8] |
| 既存closed P5を物理PRへ接続する | 応用・資源評価の候補として保持 | 正当な問いだが、新アルゴリズム研究へ無言で読み替えない |
| N3の縮約保証へTrack Bを転換 | 行わない | taskと目的が別。今回の他研究参照依頼を超える |
| **既存DF-RTEの積部分の生成・圧縮** | **限定候補監査の対象に選ぶ** | 各eventを保つ明確な不変量があり、既知法への還元と改善余地を本番前に検査できる |

これは、最後の候補に新規性や性能が確認されたという順位付けではない。**実装の有無と既知法との差を小さい労力で判別しやすい具体的なRQ**として選んだ。

## 4. 新しいRQと対象を明確に定義する

### 4.1 RQ

> DF由来の1軌道・2軌道parity primitiveのsampled product列を、低rank反射積から生成する制御付きGaussian回路へ置き換えるとき、各eventの作用・位相・sampling lawを保ちながら、強い既知実装を超える方法差と期待量子資源の削減が成立する条件はあるか。

RQの対象はHamiltonianの再分解や軌道最適化ではない。**与えられたDF入力・sampled wordを実際の回路へ変換する、PR内部の生成アルゴリズム**である。

対象候補を記述するため、以下では「積部分」と「終端回転」を区別する。現builderはproduct roleを並べ、その後に一つのrotation roleを置く。非identityのsupportはZまたはZZである。[R3]

### 4.2 変えないもの

初期の候補監査では、次を固定する設計を選ぶ。

- Hamiltonian、DF入力、決定論／乱択のsplit。
- sampled label列・元の順序・次数・各eventの確率。
- 係数sign、Taylor由来phase、元のRTE normalizationとweight。
- 最後のrotationのtarget、制御枝、初期状態と測定task。
- 比較時の最終signal精度・confidenceの意味。

変更するのは、同じ順序付き積を生成・合成する処理である。ここでの不変性は理想構成の数学的性質であり、有限精度の合成誤差が自動的にゼロになるという意味ではない。

## 5. 数学的な出発点：新規性の主張ではなく、候補の成立範囲

以下は既存のfermionic Gaussianの表現から本レビューで整理した導出である。未実行の回路検証結果ではなく、独立した新定理という主張でもない。

### 5.1 真空位相を固定した第二量子化

一粒子空間の次元をNとし、unitary Mに対して

\[
\Gamma(M)=\bigoplus_{\eta=0}^{N}\wedge^\eta M
\]

と定義する。真空sectorでは1とする。この規約で

\[
\Gamma(M_2)\Gamma(M_1)=\Gamma(M_2M_1)
\]

が成立する。global phaseを無言でmod outした表現ではない。

正規直交した軌道列V_j=(v_{j1},...,v_{jr_j}), r_j∈{1,2}とsign σ_j∈{−1,+1}を考える。

\[
R_j=I-2V_jV_j^\dagger,
\qquad
Q_j=\sigma_j\prod_{\ell=1}^{r_j}(1-2n(v_{j\ell})).
\]

占有basisで確認すると

\[
Q_j=\sigma_j\Gamma(R_j).
\tag{1}
\]

このQ_jは指定されたparity primitiveである。任意のDF block全体を、rank2のGaussian操作と呼んでいるのではない。

### 5.2 順序付き積

回路上でQ_{i_1},...,Q_{i_k}の順に作用させる場合、全演算子は

\[
P_\omega=Q_{i_k}\cdots Q_{i_1}
=\chi_\omega\Gamma(M_\omega),
\quad
M_\omega=R_{i_k}\cdots R_{i_1},
\quad
\chi_\omega=\prod_{s=1}^{k}\sigma_{i_s}.
\tag{2}
\]

非可換な因子の順序を並べ替えてはいない。奇数の積がinvolutionになるとも仮定しない。負のscalar signとTaylor event phaseは別に保存する。

終端回転の後に全作用を書くと、例えば

\[
U_\omega
=e^{-i\theta_\omega Q_j}P_\omega
=\chi_\omega e^{-i\theta_\omega Q_j}\Gamma(M_\omega).
\tag{3}
\]

左／右の並びは、利用するsourceの時間順規約に合わせて実装時に固定する。

### 5.3 支持はsampleされた軌道のspanに閉じる

\[
\mathcal S_\omega=\operatorname{span}\{\operatorname{col}(V_{i_s}):1\le s\le k\},
\quad d_\omega=\dim\mathcal S_\omega.
\]

各R_jはSの直交補空間で恒等なので、M_ωもそうである。WをSの正規直交basisとすると

\[
M_\omega=I+W(M_{\omega,S}-I_d)W^\dagger,
\qquad
d_\omega\le\min(N,\sum_s r_{i_s})\le\min(N,2k).
\tag{4}
\]

従って一粒子行列の低rank更新またはactive-span上の積で扱える。全Fock空間の2^N×2^N行列を作ることは理論上必要ない。

**dは語長kに応じて増える。以前の補正系列の「最大6軌道」をそのまま使ってはいけない。** また、軌道spanが小さいことと、JW qubit上の短い連続区間に支持されることは別である。

有限精度でほぼ従属した列を小さいthresholdで捨てる処理は、式(4)のexact rankと別である。初期候補は無断のrank切捨てを含めない。rank認証ができない場合のfull-rank扱い・停止等は事前に定める。

### 5.4 終端rotationは一般にはGaussianではない

rank2ではQ_jは回転基底の二軌道parity積であり、一般角度のexp(−iθQ_j)はquartic相互作用を含む。**Q_j自体がGaussianであることから、その任意角度指数関数もGaussianであるとは言えない。**

初期案はproduct roleだけをまとめ、終端rotationと他の非Gaussian blockを境界として保持する。相互作用Hamiltonian全体をfree fermionとして圧縮する提案ではない。

rank1の特例を使えば終端rotationもphase付きGaussianにできる場合があるが、それを初期案へ追加しない。特殊例の成功を一般rank2へ延長しない。

### 5.5 制御位相

式(2)のsignを落とすと、制御回路では相対位相が変わる。

\[
\operatorname{diag}(I,\chi_\omega\Gamma(M_\omega))
\]

をtargetとし、control-zero枝、active枝、wire順序を保持する。単に一粒子行列が合うこと、あるいはstate preparationとして合うことだけで、このtargetを実装したとは認定しない。

Gammaをvacuum=1で定義することは位相を無料にする操作ではない。コンパイルしたcircuitのglobal phase、orbital eigenphase、符号、controlled phase、合成の実費用は記録する。

### 5.6 候補の具体的な生成手順

単に「Gaussian性を使う」とCodexへ委ねるのではなく、比較する候補を次の有限手順として置く。

1. sampled wordに現れた軌道列を、元のlabel順を保持して受け取り、そのspanのbasis Wを一回だけ作る。入力を最適化・再分解しない。
2. 小空間の反射 $R_j^{S}=I_d-2(W^\dagger V_j)(W^\dagger V_j)^\dagger$ を元の時間順で掛けて、$M_S$を得る。全Fock行列は作らない。
3. $M_S=Z\operatorname{diag}(e^{i\phi_1},\ldots,e^{i\phi_d})Z^\dagger$ とunitaryとして分解する。数値実装のunitarity・残差・角度誤差は別に認証する。
4. 最初のd列がWとなるunitary completion $\widetilde W$を用い、$B=\widetilde W(Z\oplus I)$とおく。このとき

\[
\Gamma(M)=\Gamma(B)\left[\prod_{\ell=1}^{d}e^{i\phi_\ell n_\ell}\right]\Gamma(B)^\dagger.
\tag{4a}
\]

5. controlled版では両側のGaussian basis変換を無制御で実施し、中央のoccupation phaseだけを制御する。式(2)のscalar signはcontrol上の相対phaseへ戻す。終端RTE rotationは元のまま接続する。

式(4a)のbasis変換とその逆を実施する時間順は、演算子の右から左である。$B$の未使用列に自由度があるのは、中央作用がその直交補空間で恒等だからであり、任意入力に対するunitaryの正当性を省略してよいからではない。

この手順は、各primitiveでbasisを開閉する方法と異なり、word全体で共有するbasisを一度作る。**ただし、この手順も既知Gaussian diagonalization／partial basis rotationの組合せであり、これを示しただけで新規とは認定しない。** compact WY、通常のGaussian積圧縮、thin primitive連結との直接比較によって、同じ処理か追加の価値があるかを調べる。

古典処理は全軌道列数s、span次元dに依存する。例えば逐次直交化と小行列更新では$O(Nsd+sd^2+d^3)$程度の演算を要する構成が考えられ、無料ではない。この式は実装案の算術モデルであって、認証・有限precision・bit complexityまで含む完成した複雑度定理ではない。

## 6. 既知法との境界：ここが最初の関門

Gaussian回路の圧縮自体は既知である。Campsらの代数的圧縮があり、Kökcüらはcontrolled free-fermionic evolution、Hadamard-test型の応用まで扱っている。[P2,P3]

また、Householder反射積のcompact WY表現は古典数値線形代数の既知手法である。複素の場合もLAPACKのZLARFTに、順序付きreflector積をI−VTV†として表す処理がある。[P4,P5]

DFの基底回転・低rank表現も既知である。[P6]

したがって、次のどれも単独の独立新規性にはならない。

- parity積がGaussianになる。
- 複数のGaussian操作を一つにまとめる。
- 反射積を低rank行列として保存する。
- 制御付きGaussian回路を作る。
- N×N行列を使い、2^N行列を避ける。

今回確認した現builderの該当経路は、basis ID/hashが同じときの隣接reuseを実装している。[R3] ただし、これをもって**リポジトリのどこにも一般Gaussian積圧縮がない**とは言わない。別helper・既知実装の利用経路をCodexが追う必要がある。

残る可能性は、**DFの少数軌道反射列を入力としたとき、汎用圧縮に比べて生成・制御・合成のどこをどう省けるかを、実際の手順として提示できること**である。現時点で、その追加手順は確立していない。

独立した差が見つからず、「compact WYで積を作り、既知Gaussian compilerへ渡す」だけなら、既知部品の適用と分類する。アプリケーションとして価値が残る可能性はあるが、新規アルゴリズムの主要claimへ昇格させない。

逆に、新しい手順の効果まで対照へ無料で移植し、架空の最適compilerを要求して候補を消すこともしない。対照は、実際に記述・利用できる既知手順で固定する。

## 7. 資源上の見込みと、楽観してはいけない点

### 7.1 弱い対照に対する改善を新規性と混同しない

各primitiveにfull N-mode basisを毎回適用する弱い見積もりは、積長kに対しO(kN²)の基底回転費用を持ち得る。一方、一般Gaussian積をまとめる既知方法でO(N²)規模にできる場合がある。

しかし、同じ入力列を使うthin basis・support restrictionを対照へ与えると、元からO(N∑r_j)程度で実装する余地がある。今回のactive-span構成も、標準的なbasis completionとd-mode Gaussian合成なら概ねO(Nd+d²)の構造を持つ。この次数は、既知のpartial rotation構成を利用するモデル上の評価であり、新しい実回路費用ではない。[P6]

一般の短い語でd≈∑r_j≈2kなら、両方ともO(Nk)であり、**次数の改善は自動的には残らない**。active-space内部の任意角度や古典分解費用が増える可能性もある。

使える可能性があるのは、非隣接の反射列で軌道spanが繰り返し共有される、語長が増えてもdが増えにくい、境界の既知basisと一体化できる、といった具体的な条件である。これらがPR分布で十分起きることは未検証である。

### 7.2 古典費用も別の資源として記録する

低rank化によって一粒子行列のdense積を避けても、span構築、QR、small unitary合成、angle生成、有限精度認証には費用がある。既知compact表現を使うこと自体は、その費用のゼロ化ではない。

古典CPU／RSSと量子RZ／T／CX／depthは別の座標として記録する。換算係数を定めずに秒とT数を足して「総費用」を作らない。冷開始とcache利用を分け、対照にも同じ既存cache機会を与える。

### 7.3 低次数の頻度を無視しない

product roleが0個のeventには、今回の積圧縮の利益はない。k>0のeventで局所回路が大幅に短くなっても、元分布における頻度と条件付き費用を戻さなければ、全PRの改善は分からない。

Track A H6では、非ゼロ次数・order4の未観測が残ることが重要な課題になっている。[R1,R2] これは本候補の数値結果ではなく、**低頻度を無視せず、逆に低頻度だけで費用寄与も小さいと即断しない**ための参考である。

## 8. 改善可能な期待費用の大きさを、先に判別する

以下は本レビューの会計上の導出であり、新しい実測値ではない。

完全なtrajectoryをω、測定axisをa、既存lawをp_a(ω)、十分shot予算をn_a、wrapper費用をC^0_{a,x}(ω)、変更後をC^1_{a,x}(ω)とする。xはRZ、CX等の同じ費用指標である。

同じ理想event・同じ実装誤差予算で、同じshot規則を使える場合、

\[
\Delta G_x=\sum_a n_a\sum_\omega p_a(\omega)
\big(C^0_{a,x}(\omega)-C^1_{a,x}(\omega)\big).
\tag{5}
\]

変更可能なtrajectory集合Ω_chgの外でC^1=C^0、全費用が非負なら、

\[
\Delta G_x\le
\sum_a n_a\sum_{\omega\in\Omega_{\rm chg}}
p_a(\omega)C^0_{a,x}(\omega).
\tag{6}
\]

右辺は**変更対象trajectoryの全wrapperを仮に無料にした、極端に楽観的な改善上限**である。共通PFや測定を実際に消せるという主張ではない。これほど楽観的に見ても研究目標に足りない場合、本番compilerの開発前に候補を区切れる。

逆に、右辺が大きくても改善を証明したことにはならない。さらに現実的な下限費用や、実際の対照との差が必要である。未観測eventのCが不明なら、既知確率だけで上限を計算できたとは言わない。

各slotのorder0確率が既知で独立なら、変更対象の一つの上包として、少なくとも一つの非ゼロ次数を含む確率は1−∏p_{0,s}で書ける。ただし、この確率と式(6)の費用寄与は同じではない。

全体compilerのfusionによって費用が非加法になる場合も、式(5)は完全wrapperを対象とする。局所gate費用を恣意的に割り当てた和で置き換えない。

## 9. 精度と推定量の保証

### 9.1 理想eventの一致は、平均より強い不変量である

各ωでU_new(ω)=U_old(ω)、確率pとweight wが同じなら、

\[
\sum_\omega p(\omega)w(\omega)U_{\rm new}(\omega)
=
\sum_\omega p(\omega)w(\omega)U_{\rm old}(\omega).
\tag{7}
\]

有限Taylor target、元のtruncation bias、理想weight moment・rangeを変更しない。元RTE normalizationが消えるわけではない。

新しい合成で実装誤差が変わる場合は別である。first-operator-momentの差には

\[
\left\|\sum_\omega p_\omega w_\omega
(\widehat U_\omega-U_\omega)\right\|
\le\sum_\omega p_\omega|w_\omega|\,
\|\widehat U_\omega-U_\omega\|
\tag{8}
\]

を戻す。実際の信号・測定規約の係数と予算は既存契約に従う。同じshot数を主張できるのは、最終的に同じ許容biasと統計幅が確保されたときである。

### 9.2 一粒子誤差を、そのまま多粒子誤差にしない

正確にunitaryなMとM_tildeについて、η粒子sectorではtensor積のtelescopingと反対称部分空間への制限から

\[
\|\wedge^\eta M-\wedge^\eta\widetilde M\|
\le\eta\|M-\widetilde M\|.
\tag{9}
\]

全Fock空間を扱うなら保守的にN倍まで必要である。固定粒子sectorに限定した保証を使う場合は、そのsectorがtaskと契約で正当化されている必要がある。

M_tildeが単なる非unitary浮動小数配列なら、この式をそのまま適用しない。理想unitaryと計算近似の距離を別に認証する。基底のphase、control、JW規約も含むfull unitaryの検証が必要であり、Slater determinantを一つ準備できたことでは代用しない。[P7]

## 10. 最小検証の順序

ここでは、次に準備すべき科学的目的を定める。新しいregistered run、native compile、T合成の実行許可を与えるものではない。

### Gate 1：既存法との差があるか

Codexが既存sourceと一次文献の対応を追い、式(1)–(4)を支えるprimitive access、符号、時間順を確認する。現builder、P-A run-level、thin basis、一般Gaussian圧縮、controlled圧縮、compact WYを対照候補として整理する。

成果物は「新しい処理は具体的にどれか」「どの既知処理と同じか」「既存実装に既にあるか」「古典・量子計算量のどこが変わるか」の対応表である。新規性の正式採択はGPTに残すが、具体的な同値性・関数の有無・式の照合はCodexで完結させてよい。

**ここで全処理が既知の組合せと実質的に同じであれば、新規アルゴリズムの本番pilotへ進まない。**

### Gate 2：利益の上限と反例になり得る条件を確認する

先に式(6)の会計と、d≈2kで漸近差がなくなる条件を調べる。保存された確率・語・費用・基底情報だけで確認できる部分を優先する。

保存データが足りなければ、不足する条件付き費用を明示する。未観測を0にせず、その取得だけが必要な最小の計画を提案する。H6の新しい実験、H8、分子gridをこの候補のために自動起動しない。

### Gate 3：必要な場合だけ、小さい機構識別用の実装比較を準備する

Gate 1–2に具体的な余地が残ったときに限り、登録外の少数のcontrolで、次を区別できる検証を設計する。

- 同じ基底の単純な反復：既存reuseが効く対照。
- 異なる基底で非可換、かつ軌道spanが独立に増える場合：候補が不利にもなり得る対照。
- 異なる基底でspanが共有される場合：提案機構の仮説を検査する例。
- productが空の場合、sign・controlled phase、rank2の非Gaussian終端回転。

これらは条件の意味を示す設計クラスであり、今回新しいseed・matrix・row数・閾値を確定したわけではない。技術的なfixture選定と資源上限の具体案はCodexでまとめ、実行前に固定する。

候補と対照を同じgate set、control、精度、coupling、最適化自由度で評価する。小さいFock行列の検算は診断であり、大きい実装の生成法にdense Fock exponentialを要求しない。

### Gate 4：全PRへの接続

各eventの機構差と生成費用が確認されて初めて、同じ物理taskで期待wrapper費用・誤差・十分shot予算へ戻す。forcedな高次数だけで優位を決めない。

この段階の新しい科学的scopeと計算実行は、別途の判断・明示承認を要する。Gate 1–3の準備が終わったことを、分子一般化や本番GOと読み替えない。

## 11. 対照に必要な公平性

初期比較では、次の三層を分ける。

1. **既存PR builder**：同基底reuse、run-level full/support選択、既存fusionを保持した実在の対照。
2. **既知の構造的対照**：thin parity実装、Gaussian行列合成・controlled compression。実際に適用可能なaccessと費用を示す。
3. **候補の追加手順**：DF低rank列を使うことで省く処理、固有のデータ構造、control/precisionの取得方法を明示。

対照を弱いfull-basis連結だけに固定しない。一方、未発明の最適compilerを無料で対照へ与えない。既知法が固定gate setへ適用できない場合、その理由を確認し、対応する代替法・共通費用範囲を固定する。

runtimeの高速化は古典資源の改善、native gate削減は量子資源の改善、T合成削減は指定精度のfault-tolerantモデルでの改善であり、相互に自動換算しない。

## 12. 先行研究に関する今回の評価

### 12.1 Cost-aware sampling

Cuginiらは任意の回路単価cに対し、平均単価とweight二次momentの積を最適化するISを扱っている。従って、回路全体の費用を使うこと自体も新規ではない。その目的関数と、有限confidenceでrange・残余biasを含む本研究の実行予算は区別する。[P1]

非列挙で実際のqを生成する新手順や、新しい実装との結合に差が残る可能性はあるが、今回その具体的な差を確立できていない。Gaussian候補と同時に別のsampling探索を立ち上げない。

### 12.2 相関・Markov型の乱択

Markov-chain random compiler等が既に存在する。[P8] 量子channelの保証をcoherent first momentへ移すには別の確認が必要である。また、同じmarginalを保って相関だけ変える操作は、一般には非可換な積の平均を保存しない。

今回は「相関を導入すればよい」という未定義の案は採択しない。論文の全性能比較や全同値性を結論したものでもない。

### 12.3 Gaussian圧縮

Campsらの一般圧縮、Kökcüらのcontrolled free-fermion圧縮までを近接対照とする。controlled対応の欠如を仮定して新規性を作らない。[P2,P3]

JiangらのGaussian state preparationは重要な関連結果だが、特定状態の準備最適化で省ける操作を、任意入力に作用するGamma(M)へ無条件に使わない。[P7]

### 12.4 低rank反射積

compact WY／ZLARFTは直接的な既知計算である。低rank反射積の保存・順序・複素共役を正しく処理するための対照として使い、これを新発見と呼ばない。[P4,P5]

### 12.5 文献アクセスの限界

今回、新しい候補が未公表であるという網羅的な不在証明は行っていない。Camps/Jiangは一次abstract・関連公式説明、Kökcü2303.09538v2はHTML本文のcontrolled圧縮節、Cugini2603.13495v1は最適ISの本文、Motta2021は出版社本文、WYは出版社abstractとLAPACK公式説明を確認した。

PR原論文arXiv2503.05647のversionless PDFでは、取得本文とページ画像に版表示の不整合があった。版に依存する図の数値・最新式・比較結果を今回の根拠にしない。別の2026年の相関乱択文献にはabstractのみ取得できたものもあり、その全文を独立確認したとは扱わない。今回の候補選定はそれらの未確認な詳細に依存させていない。

## 13. 研究としての着地点

### 成立し得る主張

DF-RTEの既存有限meanを変えないevent生成手順について、特定の入力クラスで、既存の構造的compilerより生成計算量や制御回路資源が少なく、その効果が元のsampling分布を戻した同taskでも残ることを示せれば、実装・アルゴリズム上の貢献となる可能性がある。

これは漸近次数の改善を必須条件とするものではない。一定の適用範囲で再現する定数因子の利益や、従来扱えなかったaccess条件への拡張も候補となり得る。ただし、一般Gaussian圧縮の直接適用との差が説明できる必要がある。

### 独立新規性が残らない場合

既知手法の組合せによる実装改善だけなら、その事実を記録する。ユーザーが別途応用研究として進めることを選ばない限り、Track Bの新規アルゴリズム成果として採択しない。

### 資源利益が残らない場合

積をまとめられても、thin対照、任意角度の合成、古典処理、元分布での頻度を戻すと利益が消える場合は、当該候補を区切る。補正・sampling・Hamiltonian表現を次々に変更して、同じ候補の成功を作りに行かない。

### Track B全体の扱い

今回の監査で独立した差も検証価値も残らなければ、Track Bの新規計算をいったん休止する判断は妥当である。これはPR改善一般が不可能という意味ではなく、現時点で具体化した候補への研究資源配分である。

Track AやN3にはそれぞれ独立した継続・判断課題がある。他研究の進展を理由にTrack Bを強制的に統合・終了する判断はしない。

## 14. 次の担当・作業・承認境界

**次の担当はCodex。最初の依頼は、方法差監査・既存sourceの対応・改善余地の会計・必要最小限の検証計画を一つにまとめる。**

- まず既存文献・source・保存資料の読取りと静的比較を行う。
- 今回示した式と位相規約が実際のprimitive定義に合うかを独立確認する。
- 既知Gaussian/controlled圧縮とcompact WYへの還元を記録する。
- 対照へ同じthin access・既存最適化を与えた場合の差を明示する。
- 保存確率・費用が使える範囲で、改善可能な期待費用の上限を整理する。不足は不足と記録する。
- 新しい行列生成、native compile、T合成、登録sampling、分子計算が必要なら、その最小scope・資源上限・実行承認を分離する。

科学的に決めた条件を変えない通常の技術検証や資料整備を、commitごとのGPT詳細レビュー待ちにしない。技術的な同値性・不具合・アクセス不能はCodexで可能な限り完結する。

GPTへ戻すのは、具体的な独立差が得られた場合の新手法採択、誤差・control・taskの本質的変更、候補の継続／終了、論文claimの決定である。旧mandatory STOP、消費済みmarker、source・result保全は維持する。

**今回のレビュー開始承認は、新しい検証runnerの実行承認ではない。** このレビューではリポジトリの変更、commit/push、authorization作成、registered計算、回路生成・compile・T合成、repository testsを行っていない。数学式の検討と資料の読取り、レビュー文書の作成のみである。

## 15. 最終判断一覧

| 項目 | 判断 |
|---|---|
| 現generic whole補正系列のclosure | 維持 |
| Track A H6レビューの再審査 | 行わない。既存判断を参考として継承 |
| Track A/N3へのTrack Bの自動転換 | 行わない |
| 新しいPRアルゴリズムの確立 | 未確立 |
| 次の限定RQ | DF-RTE積部分の低rank反射／Gaussian生成の方法差と期待資源 |
| Gaussian閉包・compact WY・controlled圧縮自体の新規性 | 主張しない |
| 次に必要なこと | Codexによる方法差・既存実装・改善余地の技術監査 |
| 大規模compiler開発・分子pilot | 現時点で採択しない |
| 新しい計算実行 | 別途明示承認待ち |
| 詳細GPTレビューの再開条件 | 新しい科学的な方法差・結果・契約判断が揃ったとき |

**総括：他研究の調査から学ぶべきなのは、別系列の成果へ乗り換えることではなく、強い既知法と同じ情報を使い、改善できる費用の大きさを先に見極めることである。Track Bでは、既存eventを保つ積生成の具体的な候補監査を一つだけ行う方針を選ぶ。既知法との差が残らなければ、新規性があるかのように名前を付け替えて計算を続けない。**

## 参考資料

### リポジトリ・提供資料

[R1] Codex横断調査報告。会話添付 `貼り付けたテキスト（1）.txt`。主要系列表、方法差監査、参考価値、調査限界を参照。個別branchの全raw再監査ではない。

[R2] Track A H6 scientific review。固定commit `0dd5c3d204dab9b731268a1aa2add621080f3d20`。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0dd5c3d204dab9b731268a1aa2add621080f3d20/docs/research/track_a_h6_independent_confirmation_scientific_review_20261013.md

[R3] DF-RTE Qiskit builder。固定commit `321cd8c3066194fe98741cbebbcdba1515d1deaa`。今回の重点は1–125、260–470行。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/321cd8c3066194fe98741cbebbcdba1515d1deaa/src/trotterlib/df_rte_qiskit.py

[R4] P-A非退化mechanism validation。固定commit `22a0d032717b4139cdfa806850643e93ba25624f`。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/22a0d032717b4139cdfa806850643e93ba25624f/docs/research_direction_joint_synthesis_mechanism_validation.md

[R5] G5固定辞書終了認証。固定commit `23adf61f9342abff20f059be69089e89f66f7797`。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/23adf61f9342abff20f059be69089e89f66f7797/docs/tracks/algorithm_codesign/g5_results_and_gpt_handoff_20261010.md

[R6] 有限Taylor return集約技術ノートv1.0。固定commit `22a0d032717b4139cdfa806850643e93ba25624f`。今回の重点は局所query、非列挙生成、有限bit会計。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/22a0d032717b4139cdfa806850643e93ba25624f/docs/manuscripts/track_b_return_aggregation_native_resource_note_v1_0.md

[R7] DF三次補正generic whole系列closure。固定commit `4192077723770f4846cc391ea0da96c1e115f2fa`。
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4192077723770f4846cc391ea0da96c1e115f2fa/docs/tracks/algorithm_codesign/df_correction_et_series_closure_20261012.md

[R8] 過去のTrack B全体方針レビュー、会話提供MD `track_b_overall_research_strategy_review_20261011.md`。独立した方法研究という目的、既知枠組みへの包含と具体的同値性の区別を継承。

### 外部一次資料・公式技術資料

[P1] Cugini, Atif, Subaşı, *Resource-Optimal Importance Sampling for Randomized Quantum Algorithms*, arXiv:2603.13495v1。Sec.II、Theorem1、Sec.III。回路費用とweight二次momentを用いるnet-cost。有限confidenceの全てを自動的に最適化する定理とは扱わない。
https://arxiv.org/html/2603.13495v1

[P2] Camps et al., *An Algebraic Quantum Circuit Compression Algorithm for Hamiltonian Simulation*, arXiv:2108.03283v2; SIAM J. Matrix Anal. Appl.43(3),1084–1108 (2022)。一次abstractと著者研究ページを参照。
https://arxiv.org/abs/2108.03283
https://kemperlab.github.io/papers/Camps-compression/

[P3] Kökcü et al., *Algebraic Compression of Free Fermionic Quantum Circuits: Particle Creation, Arbitrary Lattices and Controlled Evolution*, arXiv:2303.09538v2 (2025-05-15改訂)。Sec.IV.2、Appendix C、controlled free fermionへの適用を確認。
https://arxiv.org/html/2303.09538v2

[P4] Schreiber and Van Loan, *A Storage-Efficient WY Representation for Products of Householder Transformations*, SIAM J. Sci. Stat. Comput.10(1),53–57 (1989)。出版社abstractでcompact表現を確認。
https://epubs.siam.org/doi/10.1137/0910005

[P5] LAPACK公式、ZLARFT。複素block reflector、前進／後退積の順序とI−VTV†表現。
https://www.netlib.org/lapack/explore-html/d7/d0d/group__larft_ga99d53d027cc196a5e5f7e2bbacdf4979.html

[P6] Motta et al., *Low rank representations for quantum simulation of electronic structure*, npj Quantum Information7,83 (2021)。DF構造、基底回転、切捨てと費用の前提。
https://www.nature.com/articles/s41534-021-00416-z

[P7] Jiang et al., *Quantum algorithms to simulate many-body physics of correlated fermions*, arXiv:1711.05395v3; Phys. Rev. Applied9,044036 (2018)。今回参照した一次abstractのstate-preparation成果をfull-unitary最適性に転用しない。
https://arxiv.org/abs/1711.05395

[P8] *New random compiler for Hamiltonians via Markov Chains*, arXiv:2411.06485。相関型random compilerが存在することと、coherent first momentへの適用条件を分ける。
https://arxiv.org/abs/2411.06485
