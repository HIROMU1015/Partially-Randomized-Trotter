# Hamiltonian表現探索：初期検証結果とGPTレビュー資料

2026-10-10 JST。**A/B/Cを比較する初期機構検証を完了し、GPTによる研究方針レビュー待ちでSTOPする。**
中心仮説・採択アルゴリズム・新規性・論文化可能性は未決定。
今回得たのは構成自由度の具体例と障害であり、量子化学での精度一致総資源削減の実証ではない。

## 1. 実施概要と証拠への入口

対象は `HIROMU1015/Partially-Randomized-Trotter`、独立branchは `representation-exploration-20261010`。
基点は `b2e1bf65e21893b6c617223b42313623d3186f12`、run2実行sourceは
`25d7135f7a0285b6cf415349191b00c00acfb75f`。初回sourceは `451bfa523f548ad5e1563428474f62d44e7a3366`。
publication commitは、この報告書を含むGit commitで解決する（自身のSHAは埋め込まない）。
公開後の取得記録は独立した [remote retrieval receipt](../../artifacts/representation_exploration/2026-10-10/remote_retrieval_receipt.json) に保存する。

| 証拠 | 用途 |
|---|---|
| [利用者指示](representation_exploration_inputs/user_request_20261010.txt)・[GPT初期レビュー](representation_exploration_inputs/PR_representation_codesign_research_review_2026-10-10.md) | 提案を証明済みと扱わない入力原本 |
| [固定scope](representation_exploration_scope.md) | 入力、対照、指標、資源上限、初回STOPと修正範囲 |
| [mechanisms.py](../../src/trottertracks/representation_exploration/mechanisms.py) | 独立実装。既存DF/RTE helperを再利用 |
| [runner](../../scripts/run_representation_exploration.py)・[tests](../../tests/test_representation_exploration.py) | clean tracked source gate、実行上限、数学・回路の意味論検査 |
| [run2 result](../../artifacts/representation_exploration/2026-10-10/run2/result.json) | 全20 A / 9 B / 3 C行、入力行列、全比較値 |
| [B一次イベント](../../artifacts/representation_exploration/2026-10-10/run2/b_events.json) | 全24,678件の確率、選択項、回転、位相、leakage |
| [run2 audit](../../artifacts/representation_exploration/2026-10-10/run2/run_audit.json) | 実source SHA、commit、依存版、資源、出力SHA |
| [run1 failure](../../artifacts/representation_exploration/2026-10-10/run1/failure.json)・[audit](../../artifacts/representation_exploration/2026-10-10/run1/run_audit.json) | 成功に読み替えないtechnical STOP |
| [保存値verifier](../../scripts/verify_representation_exploration.py)・[audit](../../artifacts/representation_exploration/2026-10-10/saved_evidence_audit.json) | stdlibのみでcommit blobs、出力hash、イベント集計、保存Taylor行列を独立照合 |
| [validation directory](../../artifacts/representation_exploration/2026-10-10/validation/)・[provenance](../../artifacts/representation_exploration/2026-10-10/provenance.json) | テスト・実行log、依存pin、改変拒否、保護元worktreeの照合 |

上表のrepository相対リンクは公開commit上でsource・artifactへ直接到達する。sourceの固定GitHub URLは
[25d7135 / mechanisms.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/25d7135f7a0285b6cf415349191b00c00acfb75f/src/trottertracks/representation_exploration/mechanisms.py)。
既存remote資料は複製せず、今回初めて供給された指示・レビューだけを入力として追加した。

## 2. Source監査と比較条件

`PROJECT_MAP.md`、研究概要、現行設計、validation status/manifest、DF block・tail・RTE・compiled cost sourceを確認した。
レビューが参照したmainは `0babed07006c4cfc34b2b4191f4f0c8a9e9bceaf` であり、最新研究全体を含むとは仮定しない。
基点は後続研究のsource/evidenceを含むb2e1bf6を選んだ。
さらにTrack A `8a3189e`、Track B `f9d2665` の文書を別commitで読み取り専用確認した。
後者G10 v2はserializationのtechnical inconclusiveで科学比較が保存されていない。救済・再実行はしていない。
このbranchの旧索引は基点時点の履歴であり、後続Track A/Bの最新状態を上書きする資料ではない。

既存DF sourceはorbital matricesからfast-forwardable square blocksと具体的I/Z/ZZ tailを構成する。
rank/Frobenius proxyと実tail係数1ノルムは異なる。一般のDF square全体をRTE involutionとは扱えない。
既存finite RTEのpaired-Taylor意味論をそのまま使い、sampling law・return・aggregationを変更していない。
反射辞書、独立JW参照、conditional mixed oracle、global-phaseを含むtoy compile監査を新namespaceへ追加した。

全実験はsynthetic exact-data開発診断。geometry・化学basis・分子DF rank policyは該当しない。
ground-state情報、分子snapshot、GPU、量子shots、未知データholdoutは使用していない。
行列normはspectral norm（保存値verifierの別照合はFrobenius norm）。Hamiltonian近似は行わず、残差は正確に保持した。
回路はQiskit1.3.0、basis rz/sx/x/cx、optimization1、seed20261010、topologyなし、通常control。
state preparation・測定・fault-tolerant合成を含まない。RZ/CXを単一の任意重みで合算しない。

## 3. A：分解とD/Rの同時構成

### 3.1 数学的成立条件と改善できない条件

正の係数を吸収したHermitian factorsを $F_a$、$G_\mu=\sum_a O_{\mu a}F_a$、実直交 $O$ とすると、

\[
\sum_\mu G_\mu^2=\sum_{ab}(O^TO)_{ab}F_aF_b=\sum_aF_a^2.
\]

factor間が非可換でも成立する。$F_a=d\Gamma(g_a)$ なら同じ恒等式がfull Fock spaceで成立する。
ただし未吸収の不等係数をlabelへ残す変換や、符号付きsumを通常直交回転する変換は一般に不正。
正係数は平方根を吸収し、負係数を含む場合は符号別回転、または符号計量Jを保存する別条件が必要になる。
toyの誤変換残差は3と6、正しく吸収した変換は1.44e-15だった。

Gram $M_{ab}=\mathrm{Tr}(g_a g_b)$ に対し、k本のretained Frobenius weightの最大値はMの上位k固有値和。
Gramが対角で既存weight降順なら、元factor選択はこのproxyにすでに最適である。
この目的だけの回転は既知のspectral選択に還元される。実tail lambda、交換子、basis費用への最適性は導けない。

可逆factor混合はspanを保存する。変換後の全factor matricesが同時対角化できれば、元factorも線形結合として可換である。
したがって非可換factor span全体を直交混合だけで可換化することはできない。
一部block、あるいはfactorの**平方**が可換になることは別条件で、下のreview例は後者。
また、同じ全blockを同一unitaryで共役しただけなら、PF誤差のunitary-invariant normは変わらない。

### 3.2 数値と反例

全A：$H=\sum_{a=0}^1 d\Gamma(g_a)^2$、one-body/constant=0、rank2、$L_D=1$、JW full Fock。
2または3 modes、固定5角度 $0,\arctan(.1),\pm\pi/8,\pi/4$、delta=.05/.2/.4。
最大Hamiltonian保存残差2.67e-15、既存DF builderとの最大照合残差1.43e-15。
以下はdelta=.2のoperator誤差とcontrolled exact-tail S2 wrapperのcost。
tail lambdaはidentityを厳密位相へ分離したI/Z/ZZ辞書で、identityを含むfaithful値も全rowに保存する。
実RTE wrapperや精度一致costの比較ではない。

| 例・角度 | retained weight | tail lambda | S2誤差 | RZ / CX |
|---|---:|---:|---:|---:|
| review $g=[I+X,I+Z]$、0（2 modes） | 4 | 2 | 2.3210e-2 | 206 / 150 |
| 同、pi/4 | 6 | .5 | 5.66e-16 | 381 / 282 |
| proxy mismatch、0（3 modes） | 3 | 1.480189 | 6.9459e-5 | 391 / 280 |
| 同、atan(.1) | 2.981616 | 1.388289 | 1.0028e-4 | 762 / 544 |
| 同、-pi/8 | 2.728078 | 1.361864 | 3.8754e-5 | 未compile |
| 同、pi/4 | 2.0716 | .908071 | 2.1992e-5 | 未compile |

proxy mismatchは $g_0=\mathrm{diag}(1,1,-1)$、

\[
g_1=\begin{pmatrix}.1&0&.04\\0&-.8&0\\.04&0&-.7\end{pmatrix}.
\]

Gramは丸め誤差を除きdiag(3,1.1432)で、0はFrobenius proxy最適。
atan(.1)でlambdaは6.21%改善するがS2誤差は44.4%悪化し、compiled costも増える。
faithful lambdaは2.255989→2.235616、改善は約.903%に縮む。
元factorの全 $L_D=1$ 選択の最良lambdaより回転後が小さいため、単なる元factor rankingでは表せない自由度はある。
一方、この自由度を資源有利な構成へ変える方法は未確立。
pi/4のweightは数値的なtieで、D indexの違いを意味のある順位差とは扱わない。

review例の平方可換化・lambda減少は共有レビューの代数の再現であり、新規発見とは扱わない。
genericなcontrolled DF synthesisではcostが増えたが、共通可換構造を専用合成する最適化は実施していない。
よってこのcountは構成法の実装依存の診断で、最適回路の下界ではない。

可換対照（g1の.04を0）は全角度でPF誤差が丸め水準でもcostは増える。
isotropic対照 $g=[X/\sqrt2,Z/\sqrt2]$ は全角度で各Fock square自体が不変、lambda=.25のまま。
0→atan(.1)でRZ/CXが260/186→373/274になるのはframeを通す実装の違いで、物理演算子の改善ではない。
これらは「回転すれば改善」という仮説への反例。

共有computational frameの対角coreと**正確な**残差も保存した。
proxy mismatchでは残差normとPauli l1はともに.0896、再構成残差0。
Pauli辞書は丸め水準の項も捨てていない。しかしこの小例の一致は一般保証ではない。
一般に $\epsilon Z=(1+\epsilon)Z-Z$ の未統合l1は2+epsilon、normはepsilonなので、
小さいoperator残差から、与えられた辞書の小さいlambdaは導けない。

**予備評価：** proxyとPR関連量の不一致は確認できた。新しい構成自由度はあるが、既存factor最適化と重なる。
新規性には、安価なframe移動・正確残差・誤差の関係を保証する構成や、扱える入力classの拡張が必要。

## 4. B：生成子反射と補助空間leakage

### 4.1 成立する保証と成立しない保証

$Q=I-P$、$S=2P-I$、$\overline H=(\widetilde H+S\widetilde HS)/2$ とすると、

\[
Q\overline HP=0,\qquad P\overline HP=P\widetilde HP.
\]

物理Hamiltonianが $PHP=P\widetilde HP$ と正しくencodedされていることが必要。
involution辞書の各項を半分ずつ元項とS共役へ置き換えると、未統合係数l1は増えない。
finite RTEの補正平均は $B_K\mathbb E[U]=T_{K+1}(-it\overline H)$ であり、物理部分空間を保存する。
各occurrenceで必要な独立選択を、word全体で共通の反射bitに置き換えると、

\[
\tfrac12\{T_{K+1}(-it\widetilde H)+S T_{K+1}(-it\widetilde H)S\}
\]

になり、一般に正しい多項式とは異なる。反例 $\widetilde H=X,P=|0\rangle\langle0|,S=Z$ では、
正しいexpはI、完成したexpのtwirlはcos(t)I、t=.2のphysical biasは.0199334。
generator平均と完成したunitary平均を同一視できない。

さらに、平均演算子のblock保存は各trajectoryや平均channelのblock保存を意味しない。
PRに決定論PF blockを残す場合は、それらも個別にPを保存する条件が必要。
漏洩がsumで相殺する2blockのtoyでは、total generator leakage0でもS2 at .2のleakage normは3.0289e-4。
一方、Pを保存する全unitaryへ変更すると、元の圧縮primitiveの利点を保てるかが別問題になる。

### 4.2 数値、有限打切り、回路費用

1 auxiliary+1 system qubit、auxがtensorの上位、P=aux vacuum、
$\widetilde H=.7 I_{aux}Z_{sys}+.2 Z_{aux}X_{sys}+.4 X_{aux}X_{sys}$。
DF rank該当なし、$L_D=0,r=1$、t=.05/.2/-.2、K=0/2/4。
3項原辞書→6項未統合反射辞書、lambda=1.3→1.3。全eventsを列挙しMC/量子shotsは0。

| t=.2の診断 | K=0 | K=2 | K=4 |
|---|---:|---:|---:|
| 補正係数B | 1.033247 | 1.067174 | 1.067365 |
| 補正平均leakage norm | 丸め水準 | 5.80e-20 | 4.36e-20 |
| 平均channelの漏洩確率（初期aux=0,sys=0） | .019483 | .032419 | .032500 |
| 個々のevent最大漏洩確率 | .063320 | 1 | 1 |
| finite Taylor physical bias | .010594 | 1.8721e-5 | 1.3231e-8 |
| 誤ったwhole-word twirlのphysical bias | 0 | .003204 | .003191 |

補正平均と正しい有限多項式の最大残差は約4.91e-15。
K=2,t=.2のB²=1.138860、信号は補正前に1/B倍になる。diagnostic Hoeffding axis shotsは3993
（epsilon=.05、total failure=.05、両軸union bound、各軸 $\lceil2B^2\epsilon^{-2}\log(4/.05)\rceil$）。
Taylor biasをshot budgetへ戻した最終計画ではなく、QPE/RPE error・round全体の総costではない。
N=1024の補正leakage Frobenius RMS=.00849174は正確な列挙二次momentからの予測で、MC実測ではない。
期待unfused reflection回数1.063582/trajectoryも、gate cancellation前の診断。

既知echoもexact Htilde oracleを2r回呼ぶ密行列参照としてr=1/2/4/8で保存したが、compiled totalcostとは比較しない。
このtoyでは反射後のleaky Pauliが打ち消せ、統合辞書lambda=.9の簡単なHamiltonianを直接構成できる。
したがって反射RTE固有の利益を示す例にはなっていない。統合した辞書のfull wrapper比較は未実施。

| aux数 | vacuum S単独 RZ/CX | controlled toy回転 RZ/CX | 無制御Sで挟んだ同回転 RZ/CX |
|---|---:|---:|---:|
| 1 | 1 / 0 | 9 / 2 | 9 / 2 |
| 2 | 4 / 1 | 9 / 2 | 13 / 4 |
| 3 | 7 / 6 | 9 / 2 | 23 / 14 |

これはaux上の回転を使ったprimitive probeであり、実圧縮Hamiltonian、Bの全event circuit、GRADE wrapperではない。
1 auxの相殺を一般化せず、多auxの反射・追加qubit・normalization費用を含む比較が必要。

**予備評価：** 平均振幅推定に有効な代数は支持された。単一軌道leakage消去は否定された。
群平均・symmetry protectionは既知で、圧縮によるstep費用利益を維持する構成は未確認。
真のGRADE/isometric-THC factor、直接block-preserving構成、reset/echo等との精度一致比較が次の課題。

## 5. C：安価な混合propagatorの構成

2 system qubits、$A=Z_0+.7Z_1+.3Z_0Z_1$、$B_0=X_0,B_1=X_1$、
$H=A+\alpha(B_0+B_1)$、t=.2、alpha=.05/.1/.2、geometry/化学basis/DF rank/
$L_D$ は該当しない。THRIFT一次式は

\[
e^{-it(A+\alpha B_0)}e^{itA}e^{-it(A+\alpha B_1)}.
\]

各mixed oracleはspectatorの2枝ごとにconditional SU(2)で厳密に構成した。
密行列diagonalizationを効率的primitiveとして数えていない。mixed circuitの最大operator残差は丸め水準。

| alpha | THRIFT誤差 | 通常一次PF誤差 | mixedを通常splitで代用した誤差 |
|---|---:|---:|---:|
| .05 | 3.9848e-6 | .00359163 | .00359163 |
| .1 | 1.5939e-5 | .00718295 | .00718295 |
| .2 | 6.3748e-5 | .01436336 | .01436336 |

固定tのこのwindowでTHRIFTはalpha²、通常PFはalphaに応じた変化を示した。
これは既知THRIFTの機構再現。mixed oracleを通常splitで代用した積は通常PFと代数的に一致し、利益が消える。
Aだけが可解という条件は十分ではなく、mixed oracleの誤差と費用が必要である。

alpha=.1のcontrolled wrapperはTHRIFT RZ/CX=294/210、通常PF=37/22。
同一H・t・compilerだが**精度が異なる**ので、費用比からwinnerは決めない。
精度一致のstep反復、合成最適化、state preparation、測定は未評価。
Lie closureの実測次元は可換対照2、single SU(2)3、片側mixed7、全access generators15。
全体が小Lie algebraへ閉じなくても、各mixed oracleが安価な条件分岐を持つ例である。
この小例から一般化学Hamiltonianへの可否や下界は導けない。

**予備評価：** 狙うべき条件はnear-integrabilityだけでなくoracleアクセスの構成可能性。
2枝conditional SU(2)自体は既知で、新しいHamiltonian分解算法はまだ構成していない。

## 6. 近接研究と新規性の限界

今回の調査は一次資料のscoped監査で、網羅的な新規性認定ではない。

| 一次資料 | 確認した近接点と今回との差分 |
|---|---|
| [CDF, PRX Quantum 2, 040352](https://doi.org/10.1103/PRXQuantum.2.040352) | tensor圧縮と量子実装。factor化変更を提案するだけでは差分にならない |
| [RC-DF, Quantum 8, 1371](https://quantum-journal.org/papers/q-2024-06-13-1371/) | regularization、factor fitting、資源関連指標。Aは厳密保存する自由度のtoy診断に限る |
| [SCDF](https://pubs.acs.org/doi/10.1021/acs.jctc.4c00352) | symmetryとfactorizationの共同最適化。一般的な同時最適化という主張は重複する |
| [Symmetry protection, PRX Quantum 2, 010323](https://doi.org/10.1103/PRXQuantum.2.010323) | symmetry操作によるPF誤差抑制。Bの群平均・echoを新規とは呼べない |
| [Isometric THC v2](https://arxiv.org/html/2407.04432v2)・[PRX Quantum 6, 010355](https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.6.010355) | auxiliary空間とprojected diagonal構成、stepごとのreset。Bは実factorの圧縮利益を検証していない |
| [THRIFT v3](https://arxiv.org/html/2403.08729v3)・[Nature Communications](https://www.nature.com/articles/s41467-025-57580-5) | Eq.(7)/(8)のmixed propagatorとalpha²の誤差。Cは既知式のアクセス条件検査 |
| [Casares et al., SPRINT/GRADE, 2606.30741v1](https://arxiv.org/html/2606.30741v1) | Sec.II.1/II.3とAppendix E.5/F.4にfactorization、補助空間、保護・reset、residual処理が近接。広い「圧縮＋PR」だけでは新規性を主張できない |

本実装のfinite-RTE意味論は基点の `docs/rte_conventions.md` と実sourceに照合した。
PR原論文の全版、全先行研究との厳密同値性を網羅再監査したとは主張しない。

## 7. 候補比較とGPTに戻す論点

| 観点 | A | B | C |
|---|---|---|---|
| 科学的意義 | proxy・tail・PF・frame費用の不一致を構成問題にする | 圧縮とphysical保存を同時に成立させる | 理論が仮定する安価なmixedアクセスを生成する |
| 確認した機構 | 元factor選択を超えるlambda変更と平方可換化 | generator/有限補正平均のblock保存 | 2枝mixed oracleとTHRIFTのalpha²挙動 |
| 否定した一般化 | proxy改善＝実資源改善、回転で全factor可換化 | 平均保存＝各trajectory保存、whole-word twirlの同値性 | A可解だけで十分、通常split代用でも同じ利益 |
| 実資源の現状 | 同時に悪化する例あり、matched accuracy未評価 | 多aux反射増、full圧縮wrapper未評価 | single-step費用増、精度一致未評価 |
| 既知との重複 | CDF/RC-DF/SCDF、spectral proxy | 群平均、LCU、echo、GRADE/isometricTHC | THRIFT、conditional SU(2)、solvable grouping |
| 古典前処理 | factor混合O(R²n²)、各factor対角化O(Rn³)、探索costと目的の非平滑性 | 辞書倍増は軽いが統合/encoding/圧縮factor構成は別費用 | bounded-degree枝は2^degree、一般dense支援は指数的になり得る |
| 主な失敗要因 | basis費用、identity会計、誤差との競合、過学習 | 残差重量、反射、B²、物理状態と振幅推定の混同 | mixed oracle費用と精度、dense相互作用、support拡大 |
| 次検証の労力 | 小～中：費用を考慮した構成と固定対照 | 中～大：実圧縮factorと保護baseline | 小～中で構造条件、分子接続は大 |

数値改善率による順位はつけない。Aは構成自由度が確認されたが、資源有利性は未成立。
Bは有用な平均保存性と厳しいtrajectory制限が明確になった。Cは必要条件を具体化できたが既知構成の再現。
今回、**新規性と量子資源削減を共に立証した候補はない**。
初期検証の否定結果は、toyからの一般的不可能性ではなく、次の仮説を絞る証拠である。

追加候補として、AとCに接続する「bounded supportの可解core＋正確残差」を検討できる。
例えばdiagonal Ising coreについて、$A+\alpha X_i$ のdetuningがd個の隣接Zだけに依存すれば、
2^d枝のconditional SU(2)へ構成できる。全体の小Lie closureは必要条件ではない。
ただしdense chemistryでd=O(n)ならnaive枝分けは指数的。arithmetic方式にもoracle合成費用がある。
bounded-degree coreの生成、残差lambdaと誤差を同時に管理できるかが仮説で、
既知solvable groupingとの差分も未確定。**この候補の追加実装・数値実験・採択は行っていない。**

GPTに判断を戻す項目：

1. Aを単なるfactor目的関数変更から、frame/残差/誤差に保証のある構成へ発展させる課題が明確か。
2. Bの目標を平均振幅推定に限定する科学的意義があるか。実圧縮factorで直接保存構成やreset/echoより利益を残せるか。
3. Cのmixedアクセス構成、またはbounded-support coreを中心候補として比較する価値があるか。
4. 次に必要なbaseline、精度予算、physical encoding、資源指標をどう固定するか。

## 8. Technical failure、検証の限界、STOP

run1はBのcontrolled回転をcompileした際、絶対operator同値検査で停止した。
scalar phaseだけの差をsmall matrixで認証できたため、入力・候補・角度・予算を変えず技術修正してrun2へ進んだ。
補償はcompiled.global_phase metadataのみで、native countは変わらない。
relative branch errorは拒否し、補償後および追加control後も絶対同値を検査した。
このdense certificateはtoy用で、大規模truth-free compiler修正ではない。原因をQiskit一般の不具合として断定しない。
run1を上書きせず、旧研究のsource/costへ影響を外挿しない。

run2は19 compiles、wall3.84725秒、CPU3.860964秒、peak RSS447,684,608 bytes（約427 MiB）。
CPU1/BLAS各1、wall300秒/CPU240秒/address space4 GiB/per-file16 MiB/compile64件/row events10000のguard内。
これらのwall/CPUはrunner内計測で、環境作成・テスト・文書監査を含む作業総時間ではない。
初回technical STOPの消費は別auditに保存している。

専用23 tests＋既存RTE16 tests＝39 passed、fail/skip0。
qiskit-natureの実行経路でreal castに関するComplexWarning2件を記録した。今回の実入力はrealで、
独立JW参照とDF builderの同値検査は通った。complex orbital入力の正しさはこのscopeに含めない。
保存値監査74,389 checksと3改変拒否もPASS。checksの多くはevent単位で、独立科学実験数とは数えない。
全テストはlocalで、immutable CI、未知入力検証、外部科学再現ではない。

run2後は保存値監査・文書化・公開確認だけで、新科学run0。
元worktreeの保護1312 pathsは、初めから欠落していたpathsも含め同じ状態を維持した。
作業中に別の未追跡reviewファイルが追加されたため、元worktree全体のstatus同一とは主張しない。
既存dirty差分・Track A/Bのsource、実験契約、STOPは変更していない。

**STOP理由：主要候補を比較する材料と重要な反例が揃い、次は研究方針・実問題への接続の判断が必要。**
`next_stage_authorized=false`、`central_hypothesis_adopted=null`。
大規模分子benchmark、精度一致総cost最適化、Track B修正、次stageは実施しない。
この報告書でGPTの科学レビューが完了したとは扱わない。
