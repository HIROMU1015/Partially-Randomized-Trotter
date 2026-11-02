# H4 B0+I：代表6回路の実コンパイル結果

2026-10-14 JST（利用者側日付）。**6/6 COMPLETE、PILOT6_COMPLETE_STOP。追加計算へは進まない。**
[実測・認可・終了監査・公開再会計のbundle](../../artifacts/resource_applicability/track_a_h4_b0i_pilot6/2026-10-14/RESULTS.md)を一つの入口とする。

GPT科学レビューでB0+Iを元B0に追加する対照として採択し、利用者が固定SOURCE・6回路・CPU/資源契約を承認した。
SOURCE `8f6c1bc8ceeeffddbecf170ef6ddb0fe6d18b63b`、準備REVIEW `34ad305d55a797b57d00aebed99fc03a49a3ed2f`、
実行grant commit `449614525d5de74daccf9f193e3199eb931fb388`。
旧source・B0・raw/cache/checkpoint・原稿v0.3を保持し、今回の実測を別資料へ追加する。

H4 linear/STO-3G/DF rank12、8 system＋1 ancilla、T=0.8、ordinary control、structured Gaussian lowering、
二次DF-prefix PF、q1を使用した。対象は0.70 Å/prefix5、0.80 Å/prefix5、1.40 Å/prefix3のcosine/sine。
固定Qiskit1.3.0、basis rz/sx/x/cx、optimization_level1、seed17、num_processes1等のcompiler optionsを維持した。

| 距離 Å／prefix | RZ cos／sin | CX 各軸 | depth 各軸 | 旧B0からのRZ差 cos／sin |
|---|---:|---:|---:|---:|
| 0.70／5 | 8454／8453 | 5160 | 9217 | 0／0 |
| 0.80／5 | 8458／8457 | 5136 | 9155 | 0／0 |
| 1.40／3 | 5218／5217 | 3132 | 5615 | 0／0 |

これらは各新full wrapperを実際にcompileした値であり、旧B0 metrics/cacheを代用していない。
数値circuit FPとsource/input/phase/axisを含むvalidation FPは両軸別に保存した。
RZ数の一致を、unitary/circuit全bytesの同一性、旧host compilerとのoutput完全同一性、未測定構成の費用一致とは扱わない。

## 保存shot会計への反映

ε=0.05、P=0、axis alpha=0.025、元の補正Hoeffding十分条件を使う。B0+Iの平均signal・biasは既存保存値から位相回転した値、
B2は元32 paired trajectoryの費用点推定である。実hardware測定shotsは実行していない。

| 距離 Å／prefix | shot cos／sin | B0+I百万RZ-work | 保存B2最小点の百万RZ-work | B2点推定比の費用低下 |
|---|---:|---:|---:|---:|
| 0.70／5 | 10092／7992 | 152.874144 | 206.510352 | 25.97% |
| 0.80／5 | 10318／7762 | 152.912878 | 171.335623 | 10.75% |
| 1.40／3 | 9077／9880 | 98.907746 | 102.934381 | 3.91% |

各点でB0+I費用は保存B2の点推定−2SEも下回るが、SEは記述的である。同じ32 trajectoryによる選抜と推定、
希少support・未知tailの制限を保持し、母平均の優劣保証や独立確認とは呼ばない。

測定した**固定3構成**を各302精度へ再会計した906行を公開する。B2最小点推定を下回るgrid点はそれぞれ178、103、87点。
これは368/906点であり、B0+Iの全family最小や未測定geometryを含む全1812点の新winner mapではない。
旧397/1812点は未測定構成を含むΔ=0感度であり、その全数を今回の実測結果へ置換しない。

## 終了・資源・予算

実行stageは11.734542550519109秒。driver CPU1、worker CPU0、observer CPU2のown-run限定、内部thread1。
Driver native exit0、worker exit0、observerは全6件受理後の意図的SIGTERMでexit−15。wait/reapとcleanup完了、3 identity残存0。
観測12回、最大interval1.007470秒、観測所要時間0.006941秒、staleness0.006940秒。Pressureは全観測0、cgroup OOM counter増加なし。
Worker sampled RSS最大110.65625 MiB、driver25.1328125 MiB、observer21 MiB。
実装が保存していないtranspile単体の正確なwall timeと瞬間peak RSSは未取得であり、phase間隔・sampled最大で補足した。

旧carryを保持した累積actual消費/予約は73029/74805、残1776。runtime保存のallocated blocksは151552 bytes（約148KiB）、27 inodes。
元chargeにこの保存量を加えたreceipt時点の会計は13615239890 bytes、全256MiB stage枠をchargeしても13883523794 bytesで17GiB上限内。
記録wallは旧268318.34715263546＋今回stage11.734542550519109＝268330.081695186秒。失敗分の返却・budget resetなし。

起動前にsource48 Git blobとbytes、既存environment/compiler profile、3入力とfreezeのstreaming bytes/SHA、
CPU/physical core、memory、pressure/OOM、容量/inode/quota、未使用output/controlを確認した。
既存44人工testsと科学レビューを繰り返していない。事後公開検査は実receipt bindingとFraction費用会計のlocal照合でありCIではない。

## 次の科学判断とSTOP

今回の実費用を含めて、同一domainの全面H4 B0+I費用地図へ進むかをGPTへ戻す。
候補は6距離×prefix[3,4,5,6,9]×q[1,2,4,8]×2軸＝240 compile。
既存6件のcomplete/source/input/compiler/phase/axis/namespace対応を新実装へ明示して証明できれば追加234、できなければ新規240となる。
[次段の未承認scope・資源・再利用条件](../../artifacts/resource_applicability/track_a_h4_b0i_pilot6/2026-10-14/NEXT_SCOPE_PROPOSAL.md)を参照する。
今回のgrantは消費済み、retry/resumeなし、next_stage_authorized=false、mandatory_stop=true。
追加6/10/234/240の起動、H8の変更、入力再生成は行わない。
