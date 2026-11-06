# 新A：強baseline B′の限定検証と18round再会計

2026-11-07 JST、SERVER 2。基準commit `da9fceb944dae27f397e48734be30c6864e15f44`、専用branch `codex/pr-reference-strong-baseline-bprime-20261107`。

**固定モデルでB′に対する期待総CX削減が残った。保証できる相対削減率の下限は4.681782236153505%。総RZは少なくとも0.15059756533579838%増加し、旧RZ削減の優位はB′比較では消失した。** 旧14-CX実装に対する約19.03%/4.24%を、B′に対する改善率として使わない。

判定は `CONDITIONAL_CX_REDUCTION_VS_BPRIME`。V1・必要最小限V2を完了し、[mandatory STOP](../../../artifacts/pr_reference_bprime/2026-11-07/STOP.json)を記録した。新規性・研究価値・継続の正式な科学判断は行っていない。今回の検査は著者Codexによるもので、独立再実行・CI・第三者形式保証とは扱わない。

## 比較条件と変更点

Linear H4、1.00 Å、STO-3G、DF12、Partial S2、LD3、h=.02、finite RTE K=2。元の物理Hamiltonianの厳密groundを各shotで供給する仮定で、状態準備費を除く。18round q=2^m (m=0,…,17)と保存r=1/2/4/8/16/32、energy精度1.5936e-4 Ha、全失敗確率.05、位相cap.4 rad。ordinary diag(I,V)、canonical finite-bit lawと測定規約を固定した。

Qiskit1.3.0/compiler A、native RZ/SX/X/CX、全結合、近似commutation・angle cutoffなし。D融合、basis共有、eventのproduct・符号・scalar・実角、X/Y wrapperを変更していない。reference側のF/G、vacuum位相生成、2-register構成、8-CX CSWAPも固定した。

新しい `rewrite_rte()` は既存RTE builderの出力を受け、中央の `RZZGate(theta).control(1)` nodeだけを `CX(a,b); CRZ(theta)(c,b); CX(a,b)` に置き換える。thetaを実nodeから読み、追加の2倍化や符号変更を行わない。Dを含むwhole回路を無差別にrewriteしない。旧scientific SOURCEは変更していない。

[SCOPE](SCOPE.md)に実施範囲、[構造補遺](STRUCTURAL_APPENDIX.md)にV2の論証を記載した。[受領Claudeレビュー](CLAUDE_REVIEW_RECEIVED.md)は原文を保全し、その予測や相殺一致仮説を結果として自動採択していない。レビューの閾値・7-CX CSWAP提案より、今回のユーザー指示を優先した。

## 18roundの3方式比較

同じ成功条件を満たす十分shot scheduleで、各round・軸の `N × finite-bit lawでの1-shot期待native価格` を合計した。期待総費用であり、特定のsampling結果の総gate数やほぼ確実な費用上界ではない。[TOTALS_18ROUND.json](../../../artifacts/pr_reference_bprime/2026-11-07/TOTALS_18ROUND.json)に厳密な有理価格・上下界と外向き数値端点を保存した。

| 指標 | 登録14-CX baseline | B′（4-CX） | reference |
|---|---:|---:|---:|
| 18round期待総CX | [593,162,239,027.1154, 593,162,239,027.1156] | [503,880,186,497.76355, 518,760,528,585.989] | ≤480,289,613,434.8142 |
| 18round期待総RZ | [3,735,644,723,605.280, 3,735,644,723,605.281] | [3,571,960,960,634.802, 3,571,960,960,634.8027] | [3,577,340,246,876.2627, 3,577,340,249,201.294] |
| 十分shot数、X/Y計 | 8,598 | 8,598 | 8,606 |
| qubit | 9 | 9 | 17 |
| 全18round depth・size・T | 未資格・未資格・未取得 | 未資格・未資格・未取得 | 未資格・未資格・未取得 |

referenceの追加resetは68,848、source CSWAPは137,696。CSWAPのnative CX/RZ費も表に含む。異なる資源指標を合算しない。

| 比較 | 保証できる絶対差 | 保証できる相対差 |
|---|---:|---:|
| 旧登録baseline比のreference CX削減 | ≥112,872,625,592.30113 | ≥19.02896343796783% |
| 旧登録baseline比のreference RZ削減 | ≥158,304,474,403.9858 | ≥4.237674782178048% |
| **B′比のreference CX削減** | **≥23,590,573,062.94945** | **≥4.681782236153505%** |
| **B′比のreference RZ増加** | **≥5,379,286,241.461207** | **≥0.15059756533579838%** |

CXの4.681782%は保守的な**保証下限**であり、実際の改善率やMC点推定ではない。B′の下界は全parity CXを除外するため、上下界に幅が残る。下界を実行可能なbaseline価格とは呼ばない。RZ増加は単なる上下界の重なりではなく、reference下界がB′上界を上回ることで示した。

登録baselineの回転ZZ当たり24 RZがB′では2 RZになる一方、referenceには位相丸めを含む十分shot数を使用する。1-shot価格の比較だけを全RPEの利益とせず、最終roundの各軸626/630を含む総会計では上記のRZ増加となった。「何%以上なら価値がある」という閾値は設けていない。

## V1-A：correctnessとnative kernel

保存r6種、order0/2、正負の24実角について、全3qubit operatorを旧controlled-RZZと直接比較し、control0の恒等作用・control1のRZZ作用・actual inverseを確認した。branch別のglobal-phase fitは使っていない。compiler Aで全24角が4 CX・2 RZとなり、実native列も解析式と一致した。

さらに、既存q256/r2/common/j0の最初のoccurrenceについてB′を構築し、uncontrolled Uから作った全512次元diag(I,U)と比較した。joint operator差は5.655219420536211e-15。これは数値correctnessの結果であり、長round誤差の厳密上界へ移送していない。[CORRECTNESS.json](../../../artifacts/pr_reference_bprime/2026-11-07/CORRECTNESS.json)に全24角、実lowering、inverse検査を保存した。旧correctness suiteは再実行していない。

任意の実thetaについて、c=0ではparity計算・解除が恒等、c=1では `CX Z_b CX=Z_a Z_b` により目標controlled-RZZとなる。nativeのCRZは±theta/2を用い、このbinary scalingは実角に対して正確である。従って理想native gateモデルで新kernelの解析的operator誤差は0。数値差を0と丸めた結果ではない。

## V1-B：局所context、whole価格、相殺

B′の新contextはCZ product有無×Y有無の4通り。旧CCX族を含む8通りを無条件に流用していない。D、frame/inverse、CRZ/CZ/Pの既存loweringは対応する旧contextと一致した。frameとDの最終消去前のprefix・suffixに露出CXがないことを確認した。[catalogue](../../../artifacts/pr_reference_bprime/2026-11-07/Bprime_local_catalogue.tar.xz)。

同一support、異なるsupport（重複・disjoint）、CRZ/CZ/Pを挟む9局所境界を確認した。B′のparity CXは実際に露出し、旧14-CXの単純加法性を適用できない。[BOUNDARY_CHECKS.json](../../../artifacts/pr_reference_bprime/2026-11-07/BOUNDARY_CHECKS.json)。

保存済み64 realization（旧48＋16）のevent列を復元し、B′だけX/Y計128 whole-wrapper compileを行った。新samplingは0、旧baseline/referenceのcompile再取得は0。q≤256/r≤2の既存cohortを超えるlong-r whole compileはない。

全128軸でB′の実CXが安全な上下界に入り、RZは旧実価格から22×ZZ数を引いた値と一致した。保存cohortではB′とreferenceのCX相殺数も一致した。ただし全r・全supportにおける相殺集合一致の定理とはしていない。q256/r2の4 pathでは初段source passが18/14/22/8個のparity CXを削除し、これはB′の加法上界からの差と一致した。両軸・全pathの位置はDAG node IDとwireで保存した。

[paired whole価格](../../../artifacts/pr_reference_bprime/2026-11-07/paired_Bprime_prices.tar.xz)、[転記TSV](../../../artifacts/pr_reference_bprime/2026-11-07/PAIRED_PRICES.tsv)、[上下界照合](../../../artifacts/pr_reference_bprime/2026-11-07/PRICE_BOUND_CHECKS.json)を参照。stratum確率を保存し、rare/nonrareの無加重平均をcanonical期待価格には使用していない。depth・sizeはこの有限cohortの実DAG価格として保存しただけで、全roundへ外挿していない。

### 全support価格の安全な境界

旧14-CX加法価格をC14、ZZ回転数をZとする。D/frameの両側fenceと中央CRZのRZによる保護を使い、境界で削除され得るparity CXをすべて除外する下界を構成した。

`C14 −12Z ≤ CX_B′ ≤ C14 −10Z`。

相殺がないことは仮定しない。下界ではZZ回転の2 parity CXを全て捨てても、中央CRZの2 CXとD/frame/productが残る。上界ではparity CXを全て数える。RZは削除passの対象ではなく、B′は `RZ14−22Z`。reference RZ下界には全product-ZのRZを除外した。こちらも全128保存軸の実価格を包含した。[REFERENCE_RZ_LOWER_CHECKS.json](../../../artifacts/pr_reference_bprime/2026-11-07/REFERENCE_RZ_LOWER_CHECKS.json)。

これらの全support論証と有限holdoutの役割は[補遺](STRUCTURAL_APPENDIX.md)で区別した。価格式を都合よくfitしたり、観測相殺率を長rへ外挿したりしていない。

## V1-C/D：有理期待値、誤差、shot

既存floor-CDFによる成分・order確率、basis-run期待値、D融合価格を再利用した。各roundのP_ZZ、p₂、期待support、D/basis/rotation/product/CSWAPのCX寄与と価格上下界は `TOTALS_18ROUND.json`、[ROUNDS.tsv](../../../artifacts/pr_reference_bprime/2026-11-07/ROUNDS.tsv)に保存した。P_ZZ≈1などの推測値を代入していない。

新B′の全局所loweringを対応する旧contextと照合し、旧192-bit区間結果を必要な族だけ再利用した。旧14-CX CRZZの誤差項qr·e_CRZZを、新kernelの解析的0に置換した。D、basisとactual inverse、CZ、wrapperは既存の上界を保つ。全supportのtelescopingで、粒子数sectorへの射影やcontrol分枝別のphase fitは使っていない。

既存SOURCE/finite-RTE/event phase/proposal diskを同じraw信号単位で一度だけ継承し、compiler diskを新B′ diskで置換した。旧compilerと新compilerを二重に加算しない。referenceの既存位相丸めdiskと十分shot scheduleは、構成・算術方式が不変なのでそのまま再利用する。

B′の最終roundのnative joint operator誤差上界は5.828249834557184e-07、total raw disk上界は3.6648120444602677e-06、raw半径下界0.5726873297263416、残存統計位相下界0.38598699711850315 rad。全18roundで正半径・正統計予算が残った。[ERROR_RPE.json](../../../artifacts/pr_reference_bprime/2026-11-07/ERROR_RPE.json)。

alphaは各36軸1/720、合計.05。旧と同じ `epsilon=rho sin(beta)/sqrt(2)`、`N=ceil(2 log(1440)/epsilon²)` を192-bit外向き区間で再評価した。B′の十分shot計8,598は旧値の流用ではない。新kernelで誤差上界は小さくなったが、整数ceilの境界を跨がず、全roundで同じ整数scheduleとなった。fixed alias endpoint条件も維持する。alpha最適化・追加samplingは行っていない。

## V2：保証補強

- 常在/任意gate族、BasisTranslatorの規則適用、初段・最終InverseCancellation、旧fence補題、B′での適用限界を[構造補遺](STRUCTURAL_APPENDIX.md)に記述した。Qiskit1.3.0のUnroll、basis_search、compose_transforms、apply_translation、inverse_cancellationを一次SOURCEで確認した。
- 固定builderからDのRZ係数336と端点24、tailのevent係数49を導いた。結果laneのglobal-phase更新8n+2≤20nも確認し、旧定数を増やしていない。既存q4 pathでは各lane730 subtraction≤794、18 phase更新≤40。q1の小端点も観測した。[V2_INPUTS_CONSTANTS.json](../../../artifacts/pr_reference_bprime/2026-11-07/V2_INPUTS_CONSTANTS.json)、[builder witness](../../../artifacts/pr_reference_bprime/2026-11-07/V2_BUILDER_WITNESS.json)、[q1 witness](../../../artifacts/pr_reference_bprime/2026-11-07/V2_Q1_BUILDER_WITNESS.json)。instrumentationは当該process内だけで行い、finallyで復元した。
- lambda_prepとlambda_tailはともにhex `0x1.2aa26bcd0dbf3p-1`。r6種×order0/2の実角と旧区間代入角がbinary64で完全一致した。従って旧14-CX区間のこの入力対応について保留へ戻す不一致はなかった。
- Linux x86_64、glibc2.39、Python3.12.3/GCC13.3.0、Qiskit1.3.0のcp39-abi3/manylinux x86_64 wheel、CPU/Python build情報を記録した。fegetround観測は0で丸めmodeを変更していない。[環境](../../../artifacts/pr_reference_bprime/2026-11-07/ENVIRONMENT.json)、[算術platform補足](../../../artifacts/pr_reference_bprime/2026-11-07/V2_PLATFORM_DETAILS.json)。wheel/libmの形式検証や環境再構築はしていない。

## SOURCE、再利用、実行資源、provenance

新SOURCEは `src/trottertracks/pr_reference_bprime.py` と `pr_reference_bprime_accounting.py`。新規実装は中央nodeの限定rewrite、B′ context/trace、価格境界、既存会計の差分adapterに限る。新しい共通compilerやoptimizerは作っていない。

再利用した資産：`pr_reference_transfer_{qualification,catalogue,cost_bounds,error,accounting}.py`、旧pilotのbuilder/wrapper/fusion、`track_c_compiler_a.py`、`track_c_compiler_a_18round.py`の区間helper、DF/RTE/repeated-S2 builder、保存common入力、verified SOURCE/alias/finite-bit/RPE結果。旧moduleの出力書込関数・runnerは起動していない。旧catalogueは新専用領域のsnapshotへ展開し、読取helperの出力先参照だけを当該process内で向け直した。

新SCF/DF fit/ground solveは0。入力pickle hashは従来の `68269aa0415f048a0220beb4054daa1bba715c6eb29fbbe9d5ba779baf0346f2`。受領レビューはSHA256 `15f2fc28c1796bbd9983fde5a704a5cc4de0872493d923360adaa95ea4cf210d`。

実affinity64、開始時MemAvailable約401GBを観測。64独立pathを64 process、4局所contextを4 processで処理した。CPU/RAMに新しい固定上限を設けず、既存venvを読み取り使用した。共有環境・他ユーザーのprocess/fileを変更していない。ユーザー中断直前には価格jobが開始されていなかったことを確認してから起動し、価格の重複取得はなかった。

[SOURCE.json](../../../artifacts/pr_reference_bprime/2026-11-07/SOURCE.json)に実行順・SOURCE hash・入力/結果/実行中snapshot・archiveを記録した。[成果物README](../../../artifacts/pr_reference_bprime/2026-11-07/README.md)が全出力への入口。correctness24、局所context4、境界例9の計37 local compileと128 whole compileを実施。新しい局所区間jobは不要で、保存区間を照合・再利用した。旧full suiteを重複実行していない。

## 成立範囲、未資格、次の判断

成立するのは、固定保存Hermitian-H4、厳密ground供給、理想native gate/reset/測定、独立finite bits、従来のbinary64 remainder/丸めモデル、compiler Aに限定した**期待**総CXの削減と総RZの増加である。強baseline B′の最適性や、他の制御実装全般への優越は主張しない。

元の連続積分Hから保存行列への無条件包含、hardware/topology/routing/noise/wall time、全18round depth/size/T、space-timeは未資格。RZをT数やFT資源へ換算していない。任意角/Clifford/T型分類V3、長r追加compile V4、depth V5、真空位相方式変更V6、7-CX CSWAP、別PF/law/gadgetは実施していない。

**次担当はGPT／ユーザー**：4.681782%という保守的CX保証下限、RZ増加、追加8 qubit、未資格指標を踏まえ、今回の構造境界・算術条件と研究上の採用可否を判断する。Codexは追加の科学作業へ進まない。旧新A・Track C/C2・新B/C・P-D・BF/BM・PR-3等のSOURCE・結果・STOPを保全したまま、mandatory STOPを維持する。
