# A1新finite target：N1限定科学計算の結果

2026-11-06 JST、SERVER1（GPUA100-gleap-01）。**N1成功、MANDATORY STOP。N2/N3は開始していない。**
固定契約 `2b2e6c1bc3b2d201d1a6d1238172187d68aea6ab` の新finite targetを、ユーザーが保存DFモデル限定のA1診断対象として採択し、N1だけを明示許可した。
これはclassical law・control・basis対応のlocal科学取得であり、CVの改善や測定準備完了、独立再現/CIの認定ではない。

## 条件・SOURCE・一回の取得

対象は保存linear H4、1.00 Å、STO-3G、legacy DF rank12、8 system qubits、L_D=3、m0/q1/δ=0.02/r1/K2、coefficient_atol=0.0。
全4電子number sectorは後段stateの指定であり、N1ではstateを生成していない。元未近似Hamiltonianへの保証は扱わない。

- 実科学SOURCE：[b6dbec9f80da35a88fba28795ad02ca74c9adfa0][SOURCE]。公開・取得照合後のcleanな専用branchから実行した。
- [実行仕様・command](EXECUTION.md)、[SOURCE記録](../../../artifacts/o2_a1_new_target/2026-11-06/n1/SOURCE.json)、[環境lock](PROFILE_LOCK.json)。Python 3.12.3 / NumPy 1.26.4 / Qiskit 1.3.0。533 SOURCE、132 distribution metadata、332 native libraryを入力前に固定し、実行時の一致を確認した。
- 保存[入力NPZ](../../../artifacts/rte_connected_cluster_cost_validation/h4_sto3g_d100_rank12_ld3_dt0p1_ref4_k2_connected_pilot30_max1500_hold1500_rare375_v1.hamiltonian.npz)のSHA256は `13e4b10d2347ed900fe4aa4b2238128ebe735a48988e321b9e4fabac4d092a3b`。
- [CALL_COUNTS](../../../artifacts/o2_a1_new_target/2026-11-06/n1/CALL_COUNTS.json)：NPZ load・partition・preparation・CDF接続・policy登録は各1回。retryは0。既存core内部のrank検査・policy用軌道基底再構成もN1に含む。
- state solve・sampling/RNG・Hadamard測定・full interrogation build/compile・variance/N/費用評価は0件。[RESULT](../../../artifacts/o2_a1_new_target/2026-11-06/n1/RESULT.json)と[STOP](../../../artifacts/o2_a1_new_target/2026-11-06/n1/STOP.json)に保存した。

既存split/preparation/tail/CDF/CV/policy関数を再利用した。新runnerとserializerは旧hash guardを変更せず、新identityを保存・自己照合する。
coreの追加は計算済みtail canonical hash入力の返却のみで、数値計算・既存hash生成のAST不変を人工テストで確認した。

## 新identityと完全payload

| identity | SHA256 |
|---|---|
| Hamiltonian | `56e4df83655aa2f2f8132126f2635996516dc2cfb1bf4cb8d95490184ad631e5` |
| partition | `900342ad6262021ff9b919c36ee182a8ed83914a697524e3fae07248d0d082f8` |
| preparation | `f02980a5cbec9478aa1f4e6dd010f117d152f89770c40389544cf876dd197299` |
| tail | `890442900e605a262201338f19afdad638302d670c75a9d95371c10b83ea38d8` |
| catalog / PREPARATION.json bytes | `ca897db7be52ad30dcfab0512d7530188fc13a9d8ca0c78430809492d01f738b` |
| LAW.json bytes | `b77980d198cc9edc620daec9bf87df02b66e96575fd85327aed0793f3e86ecef` |
| selected policy fingerprint | `f6eba8633890dbf09b5388928a369650251bf53567d51ab4179282c7c239b2cc` |

[PREPARATION](../../../artifacts/o2_a1_new_target/2026-11-06/n1/PREPARATION.json)は全順位・weight・indices、D block4件、全catalog、符号/support/係数hex、basis操作13種、scalar/identity/λを保持する。
[PARTITION_HASH_INPUT](../../../artifacts/o2_a1_new_target/2026-11-06/n1/PARTITION_HASH_INPUT.json)、[PREPARATION_HASH_INPUT](../../../artifacts/o2_a1_new_target/2026-11-06/n1/PREPARATION_HASH_INPUT.json)、[TAIL_HASH_INPUT](../../../artifacts/o2_a1_new_target/2026-11-06/n1/TAIL_HASH_INPUT.json)はcanonical入力をbyte単位で保存し、上記3hashを再現する。tail pre-thresholdは333件（identity9件、exact-zero0件）、非identity retainedは324件。
旧opt2とのhash一致は要件にしていない。旧価格・bias・shot数・control平均・training fingerprintを移植していない。

## Actual CDF・finite-grid Q・control

[CONFIG](../../../artifacts/o2_a1_new_target/2026-11-06/n1/CONFIG.json)と[LAW](../../../artifacts/o2_a1_new_target/2026-11-06/n1/LAW.json)は実choice入力/CDFのbinary64 hexと、M=2^53上のexact Fraction質量を保持する。
格納CDFから `ceil(M*C_j)-ceil(M*C_{j-1})` によるQを照合した。独立uniform 53-bit wordというモデルのlawであり、PRNG独立性の証明ではない。
CONFIGのseed/PRNG欄は既存configの記述で、generatorは生成・使用していない。

324 symbolic labelsのうち正のgrid質量は232、zero-gridは92。zero-gridを除いた再正規化はしていない。
event supportは `232+232^3=12,487,400`（n=0と2）で、全eventを列挙せず積lawで取得した。
Q(n=0)=`9006586475752217/9007199254740992`、Q(n=2)=`612778988775/9007199254740992`。

固定a=`df-fragment-3:Z0Z1`、b=`df-fragment-6:Z0`は[CONTROLS](../../../artifacts/o2_a1_new_target/2026-11-06/n1/CONTROLS.json)に新係数・basisを保存した。fragment/support/符号とpre-threshold係数の対応、positive Qを確認し、別labelへの変更はない。
以下のL0/L1は**適用順**products→rotationの位置を表す。n=0ではL0がrotation、n=2ではL0/L1がproducts、L2がrotation。

| control | νの小数表示（exact FractionはLAWに保存） |
|---|---:|
| g1=1{n=0,L0=a} | 0.04983005213711758 |
| g2=1{n=2,L0=a} | 3.3902754435762208e-6 |
| g3=1{n=2,L0=a,L1=b} | 1.8920818202181145e-9 |

E[ggᵀ]は `[[ν1,0,0],[0,ν2,ν3],[0,ν3,ν3]]` とexactに一致する。これはclassical control joint momentで、未知quantum outcomeのvariance/covarianceではない。

| 5層（order indicator,g1,g2,g3） | 質量の小数表示 |
|---|---:|
| (1,1,0,0) | 0.04983005213711758 |
| (1,0,0,0) | 0.950101915728648 |
| (0,0,0,0) | 6.464185879087631e-5 |
| (0,0,1,0) | 3.388383361756003e-6 |
| (0,0,1,1) | 1.8920818202181145e-9 |

2層はn=0/2のQそのもの。2層・5層とも全層正、exact質量和1、互いに排他的で全supportを被覆する。
g3は非常にrareである。positive supportは候補の定義上の適格性を示すだけで、有限trainで観測できること、学習rank、CVの有効性や必要shot数を認定しない。

## Selected basis policy・correctness・保全

[POLICY](../../../artifacts/o2_a1_new_target/2026-11-06/n1/POLICY.json)は `support_run_le_1` / `boundary_optimized` / `ordinary_diag_I_U` を新preparationとSOURCEへbindした。
全324 componentの必要keyにselected basis操作定義と対応記録があり、両controlのkeyも含む。source basisの操作一致、selected hashとの対応を確認した。
記録された保持column残差の最大値は0.0、既存許容値1e-10内だった。これは浮動小数点でのcolumn対応確認であり、全pathのfull回路同値、厳密なoperator/bias上界、compile精度の証明ではない。

人工・mock [40 tests PASS](TESTS.json)（[log](TESTS.txt)）後にSOURCEを固定した。科学取得後は再preparationを行わず、保存JSONだけを独立なFraction式・byte hashで照合した。[CORRECTNESS](CORRECTNESS.json)の16項目はすべてPASS。
[PRESERVATION](PRESERVATION.json)：元mainのHEAD/statusと既存dirty/untracked 2,136ファイルのbytes、旧A1 artifact26ファイル、元WP11 HEADを保全した。C1凍結を維持する。
共有環境・他jobは変更していない。人工CPU/RAM/worker上限は新設せず、自分専用cacheだけを使用した。[resource observation](../../../artifacts/o2_a1_new_target/2026-11-06/n1/RESOURCE_OBSERVATION.json)は実行時の観測で、性能・総費用評価ではない。

## 未取得事項とN2の条件

今回の適格性は、**新identity内でのlaw/候補control/基底対応というN1の範囲**に限る。state、複素mean実測、engine error、PF/finite-RTE/selectedの参照bias、測定variance、N、価格・費用、CV改善は未取得。
一次targetはactual-Qと実selected logical full列に対するattenuated複素meanである。そのfull列・state/測定engineの接続はまだ検証していない。

N2に進むには別の明示許可が必要である。上記N1 payloadを固定inputとし、同じ全保存DF H・全4電子number sector・qubit orderingを持つ保存stateの検証、または一回の限定solveをscopeに定める。
groundの一意性・精度、sector/norm/layout、符号・scalar/identity phase・readoutの接続と、state/数値/bit-grid等を含むengine allowanceを確認する。residualだけでgapなしのstate誤差を発行せず、不足時はSTOPする。
ε/α/測定予算・取得法・価格指標の草案は今回採択しておらず、N3の準備完了とはしない。旧opt2値で補完しない。

成果物を専用branchで公開し、remote commit・本文/blob/treeを照合した後も**mandatory STOP**を維持する。新科学call・retry・C1解凍は行わない。

[SOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/b6dbec9f80da35a88fba28795ad02ca74c9adfa0
