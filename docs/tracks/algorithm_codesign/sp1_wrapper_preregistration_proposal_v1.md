# SP-1：wrapper累積・placementの結果前契約案 v1

2026-10-06 JST。**契約案・小さい共通会計の技術検証。科学実行0、RUN_READY=false。**
受領reviewは `PROCEED_TO_WRAPPER_ACCUMULATION_AND_PLACEMENT_PILOT`。
研究B全体を再設計せず、primitiveからwrapper累積へ進む方針を具体化する。
reviewにない数値・配列は下記の**提案**であり、採択済みの実行契約／authorizationではない。

## 固定資料と証拠境界

| 資料 | identity／公開入口 |
|---|---|
| SP-0.5固定結果 | [e57c1fdd… 結果・監査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e57c1fdd28589422e9c973e34e53f6c725b921e9/docs/tracks/algorithm_codesign/sp05_one_shot_result_validation_20261006.md) |
| SP-0.5 source／一回authorization | `65f6fcdb3dc1ad8bfccfaee6e1413336aef91184`／`9477cd2fcfca69f3f24b801770a1f02805907eac` |
| 旧placement数学案 | [3861e6b… full-wrapper accounting](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3861e6b941745e43863f6a62cd25fe36f3b3e108/docs/tracks/algorithm_codesign/synthesis_placement_design_review_20261006.md) |
| 今回のGPT review原文 | [sp1_wrapper_gpt_review_20261006.txt](inputs/sp1_wrapper_gpt_review_20261006.txt)、SHA256 `7461d811e2d2b040ffb02b49b5a7fb66f46edfa50a3eb50bca42cd85f9662dd9`、9,877 bytes |
| 機械可読提案・準備監査 | [SP-1 preparation](../../../artifacts/track_b_sp1_wrapper_preparation/2026-10-06/) |

独立branchは `track-b-sp1-wrapper-preparation-20261006`、基点は上記SP-0.5結果commit。
worktreeは `/home/abe/Project/prt-worktrees/track-b-sp1-wrapper-preparation-20261006`。
必要text/JSONだけをsparse checkoutし、今回のreview原文だけを明示copyする。
SP-0.5のsource・contract・authorization・result・markerと旧STOPを変更しない。
SP-0.5の23件はliteral key rowsで、独立な23角度とは呼ばない。

B-Fの限定negative closure、BM現adapterのnew-method closure、旧BM-1未実行を保持する。
A/root worktree・共通APIは編集しない。M1/M2はAのsource-bound local evidenceであり、Bの新held-outではない。
今回は分子／NPZ／DF／Hamiltonian／trajectory／GPU／旧16-cell pilotへ進まない。

## 問いと一次claim監査

狭い問いは、固定したsynthetic coherent-signal wrapperで、PAI対象数・角度分布・D/R maskを変えたときに、
同じfinite-confidence会計の**十分shot数に基づく期待T費用**の順位・適格性が変わるか。
最小必要shot数、実測runtime、最良compiler、DF-native improvement、新規性成立は主張しない。
SP-0.5のprimitive strict witnessやreview中の事後crossover推定をwrapper結果へ格上げしない。

2026-10-06に [Sparse Probabilistic Synthesis v2](https://arxiv.org/html/2402.15550v2) の本文該当箇所を確認した。
固定版は arXiv:2402.15550v2、PRX Quantum 5, 040352。新手法の採否はGPT側で扱う。

| 一次資料・本文箇所 | 既知のclaim | SP-1で追加し得る限定的確認／非claim |
|---|---|---|
| Sparse PS §II.1 Statement 2、App.B.2 Eq.(18)–(19) | gate別L1 normの積による測定負担 | 同じ公式のwrapper会計を検証。積則を新定理にしない |
| Sparse PS §II.3 Eq.(5)–(6)、§III.1 | 低T libraryと係数最適化 | catalogue／係数最適化を発明したとはしない。今回は最適化を追加しない |
| Sparse PS §III.1 Fig.2、Eq.(7)–(8) | 回路全体のT費用と累積の損益反転 | crossover自体は既知。SP-1は固定合成列・bias・range・shot切上げ・配置を戻す確認 |
| Sparse PS §III.4 Fig.4 | 3-notch PAIと最適化解の対応 | 同じπ/4 catalogueの再利用。PAI原理の新規性なし |
| [PAI v2](https://arxiv.org/html/2305.19881v2) §II.2–II.4／App.A,C | channel QPD、符号・normalization・合成 | ancilla込みloweringの適用。controlそのものを独立新規性にしない |
| [TE-PAI v2](https://arxiv.org/html/2410.16850v2) §II/IV／App.A,B、[Resource-Optimal IS v1](https://arxiv.org/html/2603.13495v1) §II Thm.1 | evolutionのgate/shot交換、平均costと二次モーメントの設計 | 新sampler／一般cost×variance最適化の発明は主張しない |

上表から、**「wrapperで積むと反転する」だけでは新規性候補を支持しない**。
将来の候補はactual finite-RTE外側乱数・normalizationとD/R配置の具体的相互作用である。
下記SP-1案はその手前のmechanism pilot。synthetic roleを実際のDF/PF/RTEと呼ばない。
Sparse PS／TE-PAIに勝つ、独立論文が成立するという判定は行わない。

## 有限domainの具体案（未採択）

共通でancilla＋systemの2 qubit、system |0⟩、ancilla |+⟩、X/Y測定によるRe/Im信号とする。
全maskに同じcontrolled lowering・basis/inverse・合成列会計を使う。
maskは **NONE / D / R / DR**、catalogueはSP-0.5のexact **kπ/4, k=0,…,7一つだけ**。
角度は既存SP-0.5 targetから選び、精度はnative operator error **10^-6**を変えない。
新target合成は提案上0。既存Bのsequence bytes・T/T† count・guardをsource-bound入力として使う。
これをwrapper科学結果として再利用するのでなく、将来の新wrapper計算の固定primitive入力とする。
旧SP-0.5 runner／PAI／J／guardを再実行して結果を変更しない。

| template | n | ordered logical列／role／外側分布 |
|---|---|---|
| A：同角度累積 | 8,16,32,64 | 全D、角度+3π/16、Pauli Z,Xを交互。外側path1、weight1 |
| B：mixed angles | 同じ4値 | 全D、(+π/16,+π/8,+3π/16,+1/5 rad)を1:1:1:1で反復、Pauli Z,Xを交互。外側path1、weight1 |
| C：D/R placement | 同じ4値 | (D:+π/16,Z ; R:σπ/8,X ; R:σπ/8,Z ; R:σπ/8,X)をn/4回。σ=±1をwrapper単位で各1/2、外側weight1 |

時系列は表の左から右。制御回転のjoint native列は
`R_(I⊗P)(θ/2)`、`R_(Z⊗P)(−θ/2)`。各native位置に同じ元roleを継承する。
Rの符号coinはPAI choicesから独立、**有限列挙するtoy外側分布**であり、Taylor RTEではない。
同じCでは全maskが同じ二つのpathと同じ平均target νを推定する。A/B/Cの異なるν同士でGを比較しない。
実finite-RTEの打切り／attenuation／実際のDF角度分布はSP-1案の入力にない。

計12 wrapper ×4 mask＝48比較、二軸96 records。
外側pathの合計は4+4+8＝16、mask/axisを掛けたpath-axis上限128、native rotation最大128/shot。
A/BのRはNONE、DRはDと同じになるduplicate control。独立positiveに数えない。
全templateで固定Clifford state/basis/control/measurementのT費用0、batch初期化0。
mask対象外のnative rotationは保存された通常合成の実T/T† countを全て加える。
共通Cliffordを無料化した**T指標**であって、全gate費用が0とは言わない。

### 融合と指数branch数の扱い

同じPauliの単純反復は `R_P(θ)^n=R_P(nθ)` へexact融合できる。
通常baselineだけn個の高T列として扱うと利益を過大にするため、Aは**同角度・交互Pauli**を提案する。
これはreviewの「同じprimitive反復」を数式上そのまま採択したものではなく、review対象の明示修正である。
同一Pauli融合はsemantic controlへ置く。fusion policyは全mask共通、D/R横断fusionは初期案で禁止する。
列順・Clifford basis・exact融合可能な隣接joint factorsを実装前に列挙し、NONEにだけ不利な融合禁止を与えない。
この未完了監査により本提案はまだRUN_READYではない。

64 controlled gateのPAI branchesを3^128個列挙しない。
primitive channelのsigned平均を時系列合成し、conditional independenceからmoment/costを解析集計する案。
小さいbranch列挙はsemantic fixtureだけに使う。全mask同一のnative列から始め、合成後の都合のよい列変更をしない。
通常／notch sequenceは同一tool identity・precision。depth、Clifford、workspaceはsecondary ledgerへ残す。

## 共通有限confidence会計

外側path ωの確率p_ω、外側signed weight b_ω（normalizationを一度だけ含む）、
PAI coefficient g、γ=Σ|g|、canonical probability |g|/γ、Γ_ω=Πγ_iとする。
norm-oneの±1 outcome Y_aに対して

\[
Z_a=b_\omega\Gamma_\omega sY_a,\quad
V_{2,a}=\sum_\omega p_\omega b_\omega^2\Gamma_\omega^2,\quad
M_a=\max_\omega |b_\omega|\Gamma_\omega.
\]

合成後もY²=1なのでpopulation二次モーメントはこの式。
oracle mean／trajectory varianceをshot条件へ代入しない。
合成biasはnative channel diamond error δ、operator guard ηならδ≤2ηとして

\[
b_{\rm synth,a}\leq\sum_\omega p_\omega|b_\omega|\Gamma_\omega
 \left(\delta_{\rm fixed,\omega}+\sum_i\sum_j p_{ij}\delta_{ij}\right).
\]

PAI notchがexactでも、対象外の通常合成biasはweightで増幅されるため落とさない。
coefficient/probability/normalizationの有限数値誤差u_aは別項。
proposalではtargetをideal synthetic平均νと定義するのでb_PF/RTE=0。
これはactual Hamiltonianのapproximation errorが0という主張ではない。

提案値：complex ε=1/20、全48 taskに対する同時confidence budget α_tot=1/20。
ε_a=ε/√2、α_a=α_tot/(48×2)=1/1920とし、96軸をunion boundする。
accounting kernelへε_aの下側有理数を渡す場合は `2ε_a²≤ε²` を機械的に確認する。
u_a上限10^-8、実際のenclosureを取得してこの上限以内か確認する。根拠なしにu_a=0へしない。
mask/axis共通で

\[
s_a=\epsilon_a-b_{\rm synth,a}-u_a>0,\qquad
n_a=\max\left(1,\left\lceil
 \frac{2\overline V_{2,a}+\frac43 M_as_a}{s_a^2}
 \log\frac2{\alpha_a}\right\rceil\right).
\]

population上界・range・biasはoutward enclosure、Bernstein規則は全maskで共通。
**十分shot数によるbound-based resource comparison**であり、必要最小costの比較ではない。
実際にshotやtrajectoryを生成しない。モデル上shot capは各軸10^9の案。
超過は切り詰めてwinnerにせず `SHOT_CAP_EXCEEDED`。
bias marginなしは `BIAS_BUDGET_EXHAUSTED`、intervalで未確定なら `NUMERIC_INCONCLUSIVE`。

\[
G_T=C_{\rm init,T}+\sum_{a\in\{\Re,\Im\}} n_a\,E_{\omega,\xi}C_{T,a}.
\]

E[W²C]は診断として別保存する。一般にE[W²]E[C]と一致せず、primary GのE[C]の代用にしない。
batch初期化とshot毎の固定costを二重計上しない。ゼロcost NONEに改善ratioを作らない。
同一template/n内のNONE比Gを保存し、5%以上の改善／悪化をmaterialとする**提案**。
ratio enclosure upper≤.95がgain、lower≥1.05がloss、完全に(.95,1.05)内ならno-material-separation、
境界を跨ぐならnumeric-inconclusive。係数値・角度値の違いだけでは利益としない。
Gの厳密shot切上げとenclosure区間の構成はfuture adapterでreviewする。

## semantic control、保存項目、資源案

single controlled rotationはintegration smokeでありscience positiveへ数えない。
signed time、U/−Uの相対control phase、basis inverse、wrong shared randomness、
同一Pauli融合、exact-notch zero-costを小fixtureで検出する。
**現在のanalytic controlsは将来の実wrapper adapterのsemantic validationではない。**

scienceが承認された場合に全48比較・96軸について保存するもの：
ordered native列とrole／signed angle／basis／fusion lineage、各外側pathとprobability／weight、
notch g/p/sign/γ、Γ、V₂/range、source sequence identity／T/T†／guard、
bias enclosures／u／margin、shot sufficient count／cap、G interval／NONE ratio／classification、
ideal/finite wrapper signal residual（診断のみ）、tool/source/contract/auth identities、
wall/CPU/RSS/error、marker、retry0、次stage unauthorized。

classical資源の具体案：wall20分、CPU15分、RSS1GiB、出力8MiB、1 process、retry0。
実行前にguardを実装する。原SP-0.5資源上限を変えたとは扱わない。新stageの提案値である。
追加catalogue／angle／精度／mask／分子条件は結果後も自動追加しない。
全rowの処理終了は `SP1_RESOURCE_MAP_COMPLETE_AWAITING_REVIEW`、
technical partial failureは `INCONCLUSIVE_MANDATORY_STOP_NO_RETRY` とする案。
rowのcap-hitも保存し、全outcomeでmandatory STOP。研究GOはrunnerに持たせない。

## 今回完了した技術作業と残るgate

[wrapper_accounting.py](../../../src/trottertracks/algorithm_codesign/synthesis_placement/wrapper_accounting.py) は
exact Fractionのpopulation会計とBernstein十分shot数の小さいkernel。
logはrange reduction＋atanh有理series remainder上界を使う。
[17 focused tests](../../../tests/tracks/algorithm_codesign/test_sp1_wrapper_accounting.py) はtiny人工fixtureを独立列挙し、
controlled積moment、outer相関、同じ平均長の相違、bias増幅、capを切り詰めない処理、初期化を検証した。
五つのanalytic phase/fusion controlsも含む。system Python3.10のlocal pass、CI／外部再現ではない。
SP-0.5を開いて係数・Jを再採点するtest、SP-1 domain sweep、合成器呼出し、full suiteは0。

実行前に必要なgateは次の通り。

1. GPTが具体template／交互Pauli修正／toy外側分布／精度・confidence・materiality・capsを採否判断。
2. 共通fusion監査、保存interval adapter、ordered joint channel adapter、bias/u enclosure、guarded runnerとresult schemaを実装。
3. 実adapterのsource-bound focused semantic review/testを通し、science source Sとcontractを結果前固定。
4. **その新Sの直接子**に新SP-1 authorization-only commitを作り、別の明示実行指示を記録。
5. 一回だけSP-1、全outcomeでSTOP。研究Bの方針・RQ・新規性・論文着地点はGPTへ戻す。

現在は1のreview用に具体案を公開する段階。kernel commitをscience source freezeと呼ばない。
SP-0.5の承認・markerをSP-1へ転用しない。BF/BM/旧16-cellを再開しない。
ケースA/B/C/Dは受領reviewの**結果後GPT判断の入口**として保持し、Codexが自動進行条件へ変換しない。
必要資料をcommit/pushしてから、**STOPしてGPTの契約案reviewへ戻す**。
