# N3 保証生成・限定移送契約 v1

状態：`DESIGN_PREPARED_IMPLEMENTATION_AND_SCIENCE_NOT_AUTHORIZED`。
この資料は数学契約と結果前scopeであり、新しい科学結果ではない。
今回の利用者指示では科学batchを認可しない。実装・合成科学検証・移送実行は後の明示指示に従う。
[受領済みGPTレビュー](n3_generator_n1_recovery_inputs/n13_direction_certificate_scientific_review_2026-10-11.md)を再レビュー・再実行しない。
[機械契約](n3_generator_transfer_contract_v1_20261011.json)と[次作業の入口](n3_generator_n1_recovery_preparation_20261011.md)を一組として読む。

## 1. 課題と適用class

入力から小さい行列の証拠を自動生成し、指定された一外部modeを除く縮約を保証できるかを判別する。
全軌道集合での最適active space探索は今回の課題にしない。
旧4入力はdevelopment履歴として保持し、その再実行で移送成功を数えない。
新入力に対して、旧scalar、共通U/cのnorm-only、五点方向保証、新生成法を比較する。

Hamiltonianは実対称一体係数hと非負density相互作用

\[
H=d\Gamma(h)+\sum_{i<j}J_{ij}n_i n_j,\qquad J_{ij}\ge0
\]

に限定する。固定粒子数は1≤ν≤m、active mode数m、全mode数m+1。
外部mode dを空にしたPと、同じ粒子数sectorのn_d=1のQに分ける。
geometry、化学basis、DF rank、L_D、時間刻みは該当しない。精度はground-energy縮約幅δ=1/250である。
一般二体項、pair hopping、exchange、任意の軌道回転、多外部、逐次削減は範囲外。

係数からh_A、v=(h_{i,d})、検証済み一体下界ℓ、試行状態上界Uを取得する。
非負densityを捨てた下界c=h_dd+(ν−1)ℓを使う。
Uの状態証拠とh_A−ℓIの正半定値証拠を必ず保存する。
真のactive/full基底状態・真のgap・真の縮約誤差は保証生成器へ渡さない。

## 2. 証拠条件とsoundness

本契約ではc>Uを前提とし、D=c−U+δ>0を置く。条件は

\[
0\le\kappa\le D,\quad
\kappa h_A-\mu I-vv^T\succeq0,\quad
\kappa U-\nu\mu\le\delta D.
\]

証明にのみa=λ_min(A)、A=PHP、C=QHQを使う。aは実装の入力ではない。
CARからBB†=dΓ_ν(vv^T)。一体LMIを固定ν sectorへliftし、κ≥0と非負active densityを使うと
BB†≤κA−νμIを得る。z=a−δに対してC−zI≥DI>0であり、

\[
A-zI-B(C-zI)^{-1}B^\dagger
\succeq(1-\kappa/D)(A-aI)
+[\delta-(\kappa a-\nu\mu)/D]I\succeq0.
\]

最後にa≤Uを使った。Schur条件からH≥zI、変分原理からE0(H)≤a。
従って0≤a−E0(H)≤δ。c−Uは証明に使う下界差であり、真のgapではない。
v=0でもc>Uという共通gateを保つ。c≤Uは`BOUND_PRECONDITION_FAILED`とし、物理的な縮約不能とは報告しない。
この証明は時間発展誤差Tδや量子資源改善を与えない。

## 3. 二つのscalarを一変数へ整理する

ν>0なので、第三条件はμ≥μ0(κ)=(κU−δD)/νに等しい。
任意の実行可能なμをμ0へ下げると一体LMIは保たれる。
逆にμ=μ0でLMIが成立すれば三条件すべてが成立する。従って与えられたU,c,δでは、

\[
M(\kappa)=\kappa(h_A-U I/\nu)+\delta D I/\nu-vv^T\succeq0,
\qquad 0\le\kappa\le D
\]

という一変数の実行可能性で十分かつ等価である。これは上記十分条件の内部の等価性であり、
物理的な縮約可能性との等価性ではない。
x=κ/D∈[0,1]とし、f(x)=λ_min(M(Dx))を数値候補生成にのみ使う。
fは単位vectorのRayleigh商というxのaffine関数の下限なので凹。
新しいSDP理論・世界初の定理とは主張しない。生成の有用性と検証費用は未評価。

## 4. 候補生成と厳密確認を分離する

将来実装の候補生成は小さいm×m実行列だけを使う。
M(Dx)のaffine部分を一度作り、正の共通scaleで数値conditioningを揃える。
端点0,1を評価し、[0,1]内のgolden-section探索でfの大きい点を探す。
最大128回の小行列固有値評価、bracket幅2^−40、同じxの評価cacheを固定する。
停止は候補生成の終了にすぎず、数値最大値・solver SUCCESSから保証や実行不能を結論しない。

best数値点を2^16、2^24、2^32、2^40の各分母へ下・上丸めする。
端点二つを加え、重複除去後の有理数x候補は最大10。
各候補についてκ=Dx、μ=μ0(κ)を**元の有理数係数から**再構成する。
候補を決定的順序で確認し、最初のvalid witnessを返す。旧五点λを候補生成へ追加しない。

別checkerは0≤κ≤D、第三条件、一体LMIを有理数で確認する。
PSD確認は対称Schur消去：正pivotならSchur補行列へ進み、負pivotなら拒否。
zero pivotでは残りの該当行もすべてzeroの場合だけその行を除く。
zero pivotに非zero off-diagonalがあれば2×2 principal minorが負なので拒否できる。
非strict PSDとκ=0も扱い、旧strict-SPD専用LDLをそのまま流用しない。
L、対角D、zero-row evidenceと元行列との一致を保存し、独立checkerで再構成する。
各中間分子・分母は最大4096 bits。cap超過は`UNRESOLVED_ARITHMETIC_CAP`。

μ0の第三条件の余裕は常にzeroなので、その値を正の安全余裕と呼ばない。
κの区間余裕、各PSD pivot/zero row、数値提案のf、最大bit長を別fieldへ記録する。
数値丸めやtoleranceでnegative pivotをzeroへ置き換えない。
有理数witnessが見つからない場合は`UNRESOLVED_NOT_CERTIFIED`。
探索が完全である、境界の全証拠が取得可能である、縮約が不可能であるとは主張しない。

## 5. 全方法に渡す共通係数量

Uは全determinantと、一体hoppingで直接つながる二determinantの等重み状態から得る。
後者のRayleigh値は(H_xx+H_yy)/2−|H_xy|で、relative signの状態証拠も保存する。
ν=1,2,3、m≤7の宣言入力に限定し、determinant最大35、trial最大128。
一般fillingでtrial列挙が多項式時間であるとは主張しない。

一体下界の初期値rはGershgorin row lower、sはone-body diagonal span。
旧ρ={0,1/20,1/10,3/20,1/5}のλ=r+ρsを先に固定し、strict-SPD確認を共通でcacheする。
追加で係数一体行列の最小固有値を一回だけ数値提案に使う。
E=max(1,max_ij|h_A,ij|)という有理scale、step=E/2^24とし、
提案をstep単位へ下丸めした値からstep×{1,2,4,8,16,32}を引いた最大6候補を作る。
その候補ごとにh_A−ℓIの有理PSDを確認する。提案値そのものは下界と呼ばない。
共通ℓはr、valid旧λ、valid追加候補の最大。追加提案が取得不能なら旧valid集合へ戻す。
この一体係数処理はmany-bodyのa/E0/gapの使用ではない。

U,c、trial集合、候補active space、δは全primary対照で共通にする。
改善したℓの生成・検証費用は共有処理として計測し、すべての方法へ同じ会計規則を適用する。
旧scalar原法のc/Uは元定義として別fieldに残すため、同情報のnorm-onlyとの差の代用にしない。

## 6. 比較と結果前の入力固定

primaryは次の五つ。method別判定と、valid certificateのunionによる運用判定を分ける。

| 方法 | 確認条件 |
|---|---|
| legacy scalar | 旧係数triangle下界・元の先頭determinant U。旧定義の一般ν版を証明確認して使う |
| common norm-only | c>U、δD≥Σ_i v_i² |
| strengthened scalar | 共通U/c、δD≥(Σ_i|v_i|)² |
| five-point common | 元の五点λ、κ=v^T(h_A−λI)^−1v、μ=κλ、共通U/cの三条件 |
| direct one-variable | 第3・4節。独自生成成功を記録し、五点fallbackを混ぜない |

secondaryとしてfive-pointの旧ℓ/c規則も保存し、共通cの改善と探索方法の改善を分ける。
各方法が不受理ならfull spaceへ戻る。このfallbackをδ達成の縮約成功として数えない。

[holdout coefficient pack](../../artifacts/n3_generator_n1_recovery/2026-10-11/preparation/n3_transfer_inputs.json)
には、三つのbase×二つの結合振幅×二つの方向、計12入力を結果前に固定した。
active m=4/5/7、ν=1/2/3、全mode5/6/8、one-body scale=1/2,1,2。
外部位置d=0/2/7。active順序に対して方向は(4/5,3/5,0,…)と(…,0,3/5,4/5)、
振幅は1/40と1/10。同じbase・振幅の二方向は同じ結合normを持つ。
数値は有理係数として保存し、旧binary64開発入力の解釈と混ぜない。
density packの(i,j)は順序を持たない異なるmode対で、各対を一回だけ数える。
添字の表示順によらずn_i n_j=n_j n_iであり、双方向の二重加算を行わない。

Hamiltonian係数は今回事前recipeから組み立てたが、保証・スペクトル・受理結果を計算していない。
生成器の仕様に入力固有の証拠点を埋め込まない。統計的blindや実分子分布の無作為標本とは呼ばない。
寸法・scale等はbase間で同時に変わるので、独立のfactor効果を推定したと主張しない。
このscopeは呼び出し側が指定する一外部候補の保証生成であり、dやactive数を真値から選ばない。
結果後のseed、係数、外部位置、δ、solver予算変更・入力追加は行わない。

## 7. 将来の実行順序・性能準備・保存

1. 別の利用者指示後にversion付きmodule/runner/checkerと限定synthetic testsを実装する。
   valid/invalid/zero-pivot PSD、κ端点、非対称/負density/ν不正の拒否、虚偽U/c・改変witness拒否、mode permutationを含む。
   synthetic suiteは最大24小fixture。旧四例の再実行・GPT付録replayは行わない。
2. CPU/memoryを再確認し、独立case最大2 workers、各BLAS/OMP/Numba/Rayon thread1、GPU0。
   rational係数・共通trial/LDLをcase内cacheし、数値候補生成を小行列に限定する。
   referenceとoptimized経路の同じsynthetic結果を確認してからsource/planを固定する。
3. N3だけの別authorizationと新しい空run directoryを作り、明示launch後一回だけ実行する。
   source/input SHA、比較条件、依存versionをlaunch前に照合。既存N13 runnerを呼ばない。
4. 12入力の全証拠判定ledgerを保存・hashして閉じる。その後だけ別診断phaseへ移る。
   full/active sectorの真値診断は各一回、計24 solve以内、最大dimension56。
   numerical診断toleranceは1e−10であり、証明条件の緩和や新しい誤差予算ではない。
5. 有理checker、artifact hash、旧source/結果の不変性を別監査で確認し、報告後STOP。

上限は72 method records、128 numerical search calls/case、10 direct rational候補/case、
旧五点5＋下界追加6の確認/case、trial128/case、rational4096 bits。
memoryはworker4 GiB・aggregate8 GiB、JSON shard32 MiB、全output512 MiB、log16 MiB。
scientific wall-time/CPU-time上限は置かない。tool polling timeoutとは別である。
実際のRSS・CPU/wall、行列寸法、評価/確認回数、最大bit、未判定理由を保存する。
速度倍率は同じ条件の測定比較がなければ主張しない。

## 8. 判別とSTOP

soundness violationまたは証拠改変の受理は`SOUNDNESS_STOP`。技術不備は`TECHNICAL_STOP`。
通常終了でも追加科学計算へ自動進行しない。
未使用base二群以上で新生成法のvalid受理がnorm-only不受理を変え、共通五点が不受理の入力にも
一件以上valid証拠を生成した場合は`GENERATED_TRANSFER_GAIN`として科学判断へ返す。
二群以上でnorm-only差が残っても共通五点との差がなければ`TRANSFER_CERTIFIED_NO_NEW_GRID_GAIN`。
それ以外は`NO_TRANSFER_ADVANTAGE_IN_FROZEN_SCOPE`。境界/cap等の未判定率も併記する。
これらは限定scopeの判別ラベルであり、中心テーマの自動採択や一般的有用性の証明ではない。

負の結果でも入力追加・precision/δ変更・solver再調整を行わず保存する。
量子資源評価、一般二体、多外部、逐次削減、別taskへの移行は後の科学判断事項。
今の状態は実装未着手・新科学実行0・新科学tests0、mandatory STOP。
