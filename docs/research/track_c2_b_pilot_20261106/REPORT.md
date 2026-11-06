# C2-B：共役block構造と決定論的合成精度の限定pilot

**登録した非一様配分同士では全18round総Tが0.5470%減った。ただし、通常非一様配分の最適値を包含する下界に勝てず、強baselineに対する改善の確立は未資格。固定保存行列・指定のcompositional回路・有限精度候補に限る数値的改善として保存し、mandatory STOPした。**

基準SOURCE `6f6c084bdfaf04ba951fd5d8ea0c521313ceb0b3`、科学資産 `720416c5b88b227cf76ff7aa729d91a0a98ed58b`。
SERVER 2 `GPUA100-gleap02`、2026-11-06 JST。条件と科学call範囲は[SCOPE.md](SCOPE.md)、判断の出発点は[科学レビュー](SCIENTIFIC_REVIEW.md)。
専用branch `codex/track-c2-b-block-synthesis-pilot-20261106`。

Linear H4、1.00 Å、STO-3G、DF rank12、元Hの供給される厳密ground、8 qubit/4 electron。
状態準備費を除く。delta0=.02、18round、t_m=.02*2^m、energy精度1.5936e-4 Ha、alpha総和.05、位相総幅.4 rad。
保存行列と元の連続積分入力の無条件な包含は引き続き未資格。
保存ground/eigenspectra/common preparationを再利用し、SCF・DF fit・ground eigensolveは0件。
E0は小系の信号診断にのみ使い、未知biasの補正やdecoderのalias選択には使っていない。
A/Bは独立SOURCE・独立worktree・独立出力のprocessとして分離した。B計算中にA実装を進めたが、B約54秒の終了後にA科学runが始まり、科学run同士の実時間重複はなかった。内部taskはprocess並列。AIサブエージェントは0。並列指定とのこの実行上の差を保存し、STOP後の再計算は行わない。
共有環境、他ユーザーのprocess/file、旧C-N/C-S/C-P/P-D/BF/BM/PR-3のSOURCE・raw・grant・ledger・marker・STOPを変更していない。

## 固定PFと再利用

PF scheduleは既存Full Y4、LD12、h=.02を全18roundで固定。tail-free endpointなのでnormalizationは全roundで1。
C2-Aの結果・選択はBへ移していない。partial LD3や新しいRTE residual lawへ拡張していない。

既存`track_c_h4.common/weights/diagonal_primitives`、reference A、
`track_c_h4_synthesis.acquire/strict_guard/exact_word_matrix`、
`track_c_compiler_a_18round.interval_check`、720416c5の保存18round signal/alias/ground条件を再利用した。
新adapter：[track_c2_b_pilot.py](../../../src/trottertracks/track_c2_b_pilot.py)。
保存値だけの完了検査：[track_c2_b_verify.py](../../../src/trottertracks/track_c2_b_verify.py)。

## 回路と全空間の誤差会計

全armに、隣接する同じHermitian fragmentの時間融合、同じ基底回路、同じ中心回路、同じactual adjointを与えた。
各basis/centerの定義をreference Aで局所loweringし、中心のcontrolled scalarを保持した。
これをnative RZ/SX/X/CXの明示compositional回路として定義した。
whole-wrapperの追加global再最適化や、無制限のClifford+T word再最適化の最良価格ではない。

合成済みbasisの帰りは、その回路の実adjointを使う。負角RZも正角wordの実adjointを使う。
中心Dのphaseをprojectiveに捨てていない。
fragmentのbinary64係数を正確な有理数として、full 256占有状態を含むスペクトル幅Wを得た。
基底を数保存sectorへ射影せず、局所gateの全空間operator boundを使う。

通常加算は2 epsilon_B + epsilon_D、構造上界は

`min(2, abs(tau)*W) epsilon_B + epsilon_D`。

基底RZ一回の合成誤差をbasis内で加算し、物理T費用にはBとB†の両方を課金した。
高次PFの負時間もabs(tau)で会計し、符号を無料相殺に使っていない。
新しい融合角度について、256占有状態のdiagonal phaseと各局所controlled gate loweringを192-bit区間で照合した。
既存のSOURCE生成・compiler diskを保持し、今回のlowering/融合に対する追加誤差も保守的に加えた。
正のfloat和の丸めにはgamma_4096の補償を加え、最終のdisk→phase/radius/shot ceilingは192-bit区間で評価した。

周期SOURCEは同一fragmentの境界融合を含むcounterで表し、
`counter(q)=counter(1)+(q-1)*(counter(2)-counter(1))`。
これは回路の定義から得る多重度であり、q1/q2の価格fitではない。q4の直接SOURCE列と多重度の一致も検査した。
H/Sdg/readoutのTは0、controlled scalarの合成Tを含む。状態準備は除く。
全RPE Tは全roundの必要shot×このfull interrogationのTを積算した。

## 強対照と有限配分

2×2対照：通常/構造上界 × 一様/既知の非一様精度。
非一様対照はrole/actual-angle keyごとに、共通基準precisionと1 decade緩和の2択を持つ。
離散実Tの削減量/追加誤差の比で並べ、正確な有理誤差予算内で緩和する既知の貪欲配分を両armに同じように与えた。
全てのprecisionや配分の大域最適を主張しない。通常非一様対照には後述のLP下界も付けた。

phase cap .01/.02 radを両armに同じ有限候補として与え、各roundで選択した。
合成disk上限と保守的なrho/beta/Nを揃えるablationを行い、その後も両armへ同じcap選択・cost-sensitive alpha配分を与えた。
alphaの正確な有理数としての和が.05以下であることを確認した。
合成meanの実位相を無料oracleで補正していない。

合成operator上界Eを既存の変換定数2sqrt(2)でraw signal diskへ渡し、
追加compiler diskと合成diskを**合算して一度**asin変換した。
`rho_new = rho_saved - disk_synthesis - disk_additional`、
`beta_new = beta_saved - asin((disk_synthesis+disk_additional)/rho_saved)`。
全roundで正の半径・統計幅が残り、既存alias条件と3*.4<piを維持した。
初回集計の別々のasin加算は微小補正が必要だったため不採用とし、初回SOURCE/結果/STOPを残して`verified_final`へ補正後を保存した。
同じ保存wordを使う会計補正で、新規合成・科学compile・新しい物理条件は0。
整数shotおよび下表の総Tはこの補正で変わらなかった。

## 数値結果と限界

| arm | 総T・一様alpha | 総T・cost-sensitive alpha | 総shot・cost-sensitive alpha |
|---|---:|---:|---:|
| 通常加算・一様 | 651,188,215,774,372 | 514,819,419,765,044 | 11,594 |
| 通常加算・非一様 | 625,828,987,268,492 | 493,576,769,775,156 | 11,608 |
| 構造上界・一様 | 632,472,887,496,464 | 497,982,472,346,044 | 11,654 |
| 構造上界・非一様 | 623,006,371,766,644 | 490,876,793,924,848 | 11,610 |

主比較は通常非一様493,576,769,775,156 Tと構造非一様490,876,793,924,848 T。
登録policy間の数値差は0.547022%削減。shotは11,608→11,610で、shot減少による改善ではない。
一様精度だけを対照にして利益を大きく見せていない。
[全round TSV](../../../artifacts/track_c2_b_pilot/2026-11-06/verified_final/ROUND_TABLE.tsv)にT/CX/半径/位相幅/shotを保存した。
RZ、CX、T、depthは合算していない。depthの全RPE最良値やhardware wall timeは未評価。

さらに、通常非一様の登録2精度候補に対するfractional-knapsack LP緩和を正確な有理数で求めた。
一様alpha下では、強baselineの全round T下界は618,080,180,649,592、構造候補の実現値は623,006,371,766,644。
候補はこの下界より約0.797%大きく、登録baselineの別の離散配分が候補差を消す可能性を除けなかった。
この下界自体は、実現するbaselineのTや不改善の証明ではない。
[BASELINE_BOUND.json](../../../artifacts/track_c2_b_pilot/2026-11-06/verified_final/BASELINE_BOUND.json)。

従って、数値policy差と、強い既知法に対する資格済み優越を区別する。
無制限のprecision/fusion最適化、元の積分Hamiltonianへの無条件保証、一般のpartial PRへの利益、世界的新規性は未資格。
構造式は既知の共役誤差評価の適用であり、project内の既存generator誤差評価との連続性もある。

## 実行・検証・保存

必要なactual-angle/precisionは7,603 key。うち既存strict record再利用4、新規取得7,599。
新規pygridsynth backendは5,374 call、残りの新規取得2,225はstrict Clifford fast path。
seed17、pygridsynth2.0.0、up_to_phase=False。各recordに実word、T数、strict区間guard、角度/precision keyを保存した。
独立key taskを利用可能64 CPU affinityで並列実行し、science本体約53.95秒。
precision keyは固定回路と18round誤差予算から導出したもので、新PF係数のgrid探索ではない。
全keyを独立に再合成した第二runは行っていない。

全7,603 recordのword hash/T count/strict guard状態、8 wordのcyclotomic-ring exact adjoint、
実emitterのT countとactual inverse、D=Iを含む5つの合成人工full-space bound、正確なalpha和を検査しPASS。
実inverseの数値identity差は8.64e-14。これは著者自身の実装検査で、別reviewerの独立レビューではない。
[TECHNICAL_CHECKS.json](../../../artifacts/track_c2_b_pilot/2026-11-06/verified_final/TECHNICAL_CHECKS.json)。
strict errorの理論的根拠は各取得時のoutward guardであり、人工test PASSを科学性能の保証とはしていない。

採用結果は[verified_final/SUMMARY.json](../../../artifacts/track_c2_b_pilot/2026-11-06/verified_final/SUMMARY.json)。
raw wordは`raw_synthesis.tar.xz`、全allocationは`RESULTS.json.xz`、precision requestは`PRECISION_REQUESTS.json.xz`に保存する。
各bundleのchecksumとmember数は`BUNDLES.json`。初回結果・補正履歴・各STOPも保持する。

**mandatory STOP。** 微小改善を救うための精度候補拡大、追加PF/partial split、追加合成・旧残差再開は行わない。
次の判断担当はユーザー：0.547%の有限policy差に対し、より強いbaselineの離散最適化やfusion比較を別途行う価値があるかを判断する。
別途許可された場合だけCodexがその限定比較を行う。現在、次の科学実行予定はない。
