# finite-RTE短block：小型pilotの未承認案 v1

2026-10-06 JST。**PROPOSAL_ONLY / RUN_READY=false / science_execution_authorized=false。**
[入口・意味論](block_synthesis_design_review_20261006.md)、[公平な対照](block_synthesis_claim_and_baseline_matrix_v1.md)。
受領GPT計画の最大12程度のperformance targetを、採否判断できる一つの案へ具体化する。
以下の数値・gate templatesはCodexのレビュー用提案。採用済み入力、preregistration、実行sourceではない。

## 一つのperformance domain案

全performanceは2 system qubits＋one control ancilla。RTE normalized weight λ=1、r=1、even K=2。
係数比u={1,3}、dimensionless τ={1/8,1/4}。以下の三familyの各4組、合計12 target案。
signed-time correctnessはcontrol枠で扱い、性能条件へ後から追加しない。

| family | Pauli規約とfinite mean M | ideal coherent target／目的 |
|---|---|---|
| commuting | Rc=(ZI+u IZ)/(1+u)、M=P3(−iτ Rc) | exp(−iτ Rc)。有限平均/phase/補正と可換control |
| noncommuting | R1=(ZI+u XX)/(1+u)、M=P3(−iτ R1) | exp(−iτ R1)。小閉代数で同辞書比較 |
| boundary | R2=(IZ+u XX)/(1+u)、W=R_XZ(π/2)、M=P3(−iτ R2/2) W P3(−iτ R1/2) | exp(−iτ R2/2) W exp(−iτ R1/2)。noncommuting境界と局所block接続 |

Pauli stringはsystem qubit順q0,q1。R_P(θ)=exp(−iθP/2)。timeとrotation angleを別fieldにする。
性能familyはactual finite paired-RTEの補正後平均を使う案で、SP-1符号coinを代用しない。
δsim=||M−Uideal||は全armへ共通に戻し、finite targetをexact evolutionと呼ばない。
boundaryは二finite blockの積を保持し、joint width拡張はこの一つの境界controlに限定する。
各targetのexact gate列・normalization・source identityは将来manifestへ固定する必要がある。

## control枠案（performanceを増やすために使わない）

| control | 検査内容 |
|---|---|
| 単一known rotation R_Z(π/8) | Granet–Dreyer系first-moment補間。既知primitiveの再現、phase含む強い直接control |
| Clifford/exact target R_Z(π/2)とidentity | 不要なweight/costの増加、zero-T referenceをpositiveにしない |
| SP-1型符号coinの短列一つ | outer weight1 toyとfinite-RTE補正を区別。旧result/runnerの再実行ではない新fixture案 |
| signed timeとphase、二block接続 | τ→−τ、±i atom、非unitary norm伝播。performanceとは独立のsemantic fixtures |

control数・exact列・diagnostic state一覧は未固定。正解state一つへ係数fitしない。
interval operator residualを独立検証し、少数stateのsignalは補助diagnosticだけ。

## 一つの有限辞書案

同じoperator比較には、各n_sysのPauli strings Pとphase ζ∈{1,−1,i,−i}を使う。
D0={ζP}を必須Pauli-LCU辞書とし、D1={ζ R_P(σπ/2): P≠I, σ=±1}を追加する案。
2 qubitのliteral上限は64+120=184 atoms。これはgate-templateの仕様上の数で、library生成結果ではない。
phase-preservingな重複だけを全arm共通に結合し、符号/±iをprojective同値で消さない。
dictionaryのT cap案は**controlled atomあたり2**、追加workspace0。実gate列とsemantic testで後に閉じる。

exact template案はPauli basis ladder→native joint Pauli rotations→inverse ladder。
Ctrl(R_P(θ))=R_(I_a P)(θ/2) R_(Z_a P)(−θ/2)を用い、θ=±π/2なら二つのR_(joint)(±π/4)。
T/T†実装時のscalar phaseを対として保持する。ζのcontrol phaseも明示し、Clifford/CX費用を別に数える。
controlled Pauliと±i phaseの列、bit/endian/order、basis ladderの標準順序はsource作成時にfreezeする。
この案を合成器/controlled wrapper検証済みとは呼ばない。

ordinary baselineの有限precision合成列はD0/D1と別の実装入力。
source freeze前に共通のprimitive precision候補と生成上限/identityを確定する。SP-0.5の未保存角度を推測しない。
positive/channel対照はjoint-spaceで同じ実装制約を守る別target tag。systemのみのdiamond保証をcontrol後に流用しない。
nonunitary Mを単一unitary/無補正convex mixtureで表せないことをcandidate勝利と数えない。

## 比較profileと予算の案

performance profileはcandidate、普通RTE、gatewise PAI/Sparseの採用一種類、positive mixture、
block full-channel Sparse、同辞書standard operator LCU、Pauli-LCUまでの7案。
controlled unitary直接再合成はordinary/event controlに含め、PR roundingは将来DF段階の対照。
PAI対Sparseを両方追加したfull gridにはしない。必要性はGPTが採否判断する。

ordinary/channel対照のprecision候補案は{10^-3,10^-4,10^-6}の3点。
全体biasに戻して適格な最安候補を選ぶ。candidateだけ追加辞書/情報/budgetを与えない。
最初は全arm canonical q。profileごとの出力は一つ、最多12×7=84 performance rows案。
探索内の評価数、primitive合成unique keys、solver call数とretry policyは最終manifestで別途数え、84と同一視しない。
I2 dense LP/SDPはoracle comparatorとして記録し、structure法にも同じPauli集約を許す。

共通taskの暫定案はcomplex ε=.05、familywise α=.05、common Bernstein sufficient shots。
これはSP-1の値の自動継承ではなく、採否と配分を要review。δsim/δblock/δimpl/u/axis予算は未固定。
第一報告は資源vectorと準備/context感度。T-only scalar primaryはzero-T Pauli対照の退化を解消してから決める。
materiality、shot cap、classical取得cost差のthresholdはnull/未固定。旧5%を自動適用しない。
task-tuned precision、positive finite bias、exact/signedの違いを同じ最終誤差へ戻す。

資源上限の候補はwall40分、CPU30分、RSS1GiB、output16MiB、一process、runs1/retry0。
seedはstochastic solverを選ぶ場合だけ結果前固定し、deterministicならnullの理由を記録する。
これらも未承認案で、今回の実行許可ではない。infeasible/span不足/technical failureも保存し、辞書やbudgetを追加しない。

## 結果前に閉じる事項と保存項目

| 未固定事項 | 次reviewで確定するもの |
|---|---|
| domain | 上記12の係数・時間・K/r・境界、control exact列、geometry/DFは適用外というscope |
| dictionary | literal/gate列、phase/endian/reduction、atom source hashes、T/workspace caps、primitive key上限 |
| algorithms | structured取得手順、standard LCU/positive/channel solverの具体版・objective・tolerance・budget・seed |
| accuracy | ideal/finite区別、norm/residual interval規則、全biasとstatistical配分、αfamilywiseと軸数 |
| primary | 資源vector/準備感度の扱い、比較基準、materialityとnumeric guard、C=0/reference0規則 |
| resource | science/technical cap、solver/合成call総数、stage/retry/partial failureの記録 |
| execution | 新source S・focused semantic verification・GPT review・別authorization・明示一回指示 |

保存はtarget/finite mean/normalization、辞書全sequence/phase/error/count/hash、係数/q/support、
operatorと別tagのchannel residual、composability/norm bounds、V2/range、ε/α/bias/整数sufficient shots、
全資源vector/prep/context感度、I0/I1/I2 access、生成/solver/residual取得runtime/RSS/calls、
不適格/未達/失敗、source/contract/tool/authorization/marker/retry/STOPとする。
partial/numeric failureでも自動retryやdictionary拡張をしない。

## 判断を結果後にGPTへ戻す

gamma低下だけ、同じscoreだけ、単一rotationの改善、T=0 blockだけではGO/STOPを決めない。
制約緩和、unitary全体再合成、task-tuned精度、positive mixture、同一辞書最適化の寄与を分ける。
standard LCUと同じ解なら、古典取得費用・再利用・適用可能性に独立差があるかを確認する。
そこも同じならmethod noveltyを主張せず、追加設計知見の価値はGPTが判断する。
どのoutcomeでも一回後mandatory STOP。DF接続/別geometry/独立validationは自動進行しない。

今回は**実装0・matrix/solver/library/synthesis/tests0**。仕様資料の公開だけでSTOPし、上記未固定事項の採否をGPTへ戻す。
