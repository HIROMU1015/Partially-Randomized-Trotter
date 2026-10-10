# 独立レビュー後の限定構成・比較scope

2026-10-10 JST。利用者の「こんな感じで進める」により、
[独立レビュー](representation_construction_inputs/representation_exploration_independent_review_2026-10-10.md)
第9節のA/C構成比較とB接続確認を実施する。中心仮説や新規性を採択しない。
基点39345830ddfe7c3e2a488c284a0623f489764087、新branch `representation-construction-comparison-20261010`。
前回結果branch・raw artifacts、Track A/Bのsource、契約、STOPは保護する。

## A：入力からframe/core/正確残差を生成する

3 fermionic modes、JW full Fock、正square rank2、独立one-body/constant correction0。geometry・化学basis・分子rank policyなし。
g0=[[.8,.09,0],[.09,-.35,0],[0,0,.15]]、g1=[[-.2,0,0],[0,.6,.07],[0,.07,-.45]]。
0–1と1–2 hoppingを持つconnected入力で、one-particle sectorは3次元。
前回のoccupation条件付き二準位toyだけで比較を終えないために選ぶ。

constructorsはcomputational frame、g0/g1の固有frameを入力から生成し、
全factorの対角square coreと正確なJW Pauli残差を返す。係数は多項式のPauli積で取得し、
dense Hの固有vector・ground state・angle gridを使わない。固定frameの全factor射影coreの
label回転不変性により、不要なO探索は除く。frame候補の全結果を保存し、最良角だけを報告しない。

baselineは元factorのnative DF weight-prefix L_D=0/1/2。basis変換を無制御にし、
diagonal中心だけを制御する。新旧ともidentityを厳密なancilla-relative phaseとして扱う。
共通frameは全反復の外側に置き、隣接deterministic half-blockは結合する。
固定T=.4、q=1/2/4、delta=.4/.2/.1、RTE r=1/step、K=2、各cost row8 classical trajectories。
deterministic-onlyは同じ回路を8回再compileせず1件/axis。
shot taskはfixed-time complex Hadamard signal、stateはmode0占有（X0の準備countを含む）。

## C：graphからmixed oracleを作り、時間次数を揃えた対照を加える

旧2qubit：fields=(1,.7)、edge01=.3、T=.2。
新3qubit chain：fields=(.9,.6,-.4)、edge01=.2、edge12=-.15、T=.4。
各入力H=A+alpha sum_i Xi、alpha=.1/.2、q=1/2/4/8。
geometry・化学basis・DF rank・L_Dは該当なし。
通常一次PF、対称S2、既知THRIFT一次積を同条件で実装する。
mixed constructorはgraph neighborsを取得し、最大degree2、最大4枝のconditional SU(2)を返す。
Hadamard ancillaに対してbasis回転を無制御化し、中央rotationだけを制御する。
対称S2の反復boundaryは明示的に結合する。
stateは全plusで、system H gatesの準備を含む。全qの誤差/countを保存する。
chain/starのn=2..8ではdegree/branch metadataのみを生成し、回路・行列を大系へ拡張しない。
degree超過はこのconstructorの適用除外で、他の算術実装の不可能性とは扱わない。

## B：isometric THC形式への小さい接続

physical3 modes＋aux1 mode、Gaussian completionを固定3つのorbital Givensから構成する。
diagonal enlarged density interactionsは01=.7、12=.4、23=.25、03=-.2。
uはcompletionの上3rows、uu†=I。normal-ordered quarticから得たphysical Hと
aux vacuum射影を独立に照合する。これは実isometryの構造fixtureで、分子fittingや圧縮利益の実証ではない。
identity分離後のdiagonal involutionsをGaussian basisとreflectionで実装し、generator-reflected RTEと
projected physical Hの直接Pauli RTEを第一moment signal taskで比較する。
後者のdense Pauli取得は小系baseline用で、大系の効率的oracleとは主張しない。
T=.2、K=2、L_D=0、q=1/2/4のmean/biasを保存し、最大epsilonに初めて適格なqだけで8 cost draws/axis。
stateはphysical modes0/1占有、aux vacuum、X0/X1準備を含む。
reset/channel保護の費用を第一momentへ流用しない。分子圧縮・echo/resetの完全比較は未実施事項として残す。

## 精度・会計・guard

epsilon_complex=.05/.02、各軸epsilon/sqrt(2)、総X/Y failure=.05。
corrected meanの小系exact operator誤差bを両軸bias上限として診断使用し、
epsilon_axis>bのとき N_axis=ceil(2 Gamma²/(epsilon_axis-b)² log(4/.05))。
これはbinary64 exact-data診断に基づく十分条件の会計で、解析的・validated numericsの厳密保証ではない。
corrected mean、normalization、Taylor/PF bias、actual sampled RTE wrapper costを接続する。
shotsは必要数の会計のみで量子測定は行わない。X/Yは同trajectoryを共有し、cost sumのSEはpairの和から求める。
8 drawsのpoint/SEを正式CIやwinner認定にしない。RZ/CX/sx/x/size/depthを別軸で報告する。
計数には準備、control、basis、relative phase、読み出しbasisを含める。Z測定1/axisはmetadataへ記録しunitary IR外。
任意重み付きgate合計、fault-tolerant T評価、QPE/RPE/energy総予算への外挿は行わない。

CPU1、BLAS/OpenMP/Numba/MKL各1、wall900秒、CPU600秒、address space4 GiB、
各output32 MiB、compile512件、native10000 gates/circuit、最大5total qubits。
sourceとscopeをcommit固定しclean tracked treeから起動、実source/input SHA・依存版・失敗・資源を保存。
native IRとabsolute phase metadataを保存し、保存IRを復元してoperator同値検査する。
scalar phaseだけのcompiler差は密行列certificateで補償できる場合に限定し、relative branch差は拒否する。
source/input固定後の重要な機構・構成・資源結果、またはtechnical/resource STOPでGPTへ戻る。
全output namespaceは新規作成のみ。大規模分子、Track B内部変更、中心仮説採択、次研究stageは含まない。

## 再現入口

- [module](../../src/trottertracks/representation_exploration/construction_comparison.py)
- [runner](../../scripts/run_representation_construction_comparison.py)
- [tests](../../tests/test_representation_construction_comparison.py)
- output：`artifacts/representation_construction_comparison/2026-10-10/run1/`
- ZIP参照原本の展開は[algebra inputs](representation_construction_inputs/algebra_checks/)。原本JSONを上書きしない。
