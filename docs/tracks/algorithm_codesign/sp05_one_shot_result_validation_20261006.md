# SP-0.5 one-shot結果・保存値監査・GPT handoff

2026-10-06 JST。固定source／別authorization／利用者の明示指示に従う**一回の実行を完了**。
preregistered primaryは **`PRIMITIVE_TRADEOFF_EXISTS`**。
23/23 synthesis keysと16/16 primitive rowsを保存し、14 strict witness、zero-cost control 1、J=1 control 1。
全error guardsが通過し、cap／runtime／numeric failureなし。**mandatory STOPに到達、retry0、次stage未認可。**

本結果は、固定pygridsynthとcheap exact catalogueのprimitive cost–second-moment trade-offを確認した
source-bound local execution evidenceである。科学実行の継続、研究方針・追加検証scopeの判断はGPT側へ戻す。

## 1. 固定sourceと一回実行

| 種類 | identity／資料 |
|---|---|
| source S | `65f6fcdb3dc1ad8bfccfaee6e1413336aef91184`、[結果前契約](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/65f6fcdb3dc1ad8bfccfaee6e1413336aef91184/docs/tracks/algorithm_codesign/sp05_synthesis_economics_preregistration_v1.md) |
| review／authorization準備P | `b0fa5afefd437aa0953593af2e5981c0826363a7`。参照記録として保持。実行HEAD・親には使用していない |
| 実行authorization A | `9477cd2fcfca69f3f24b801770a1f02805907eac`、[authorization JSON](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9477cd2fcfca69f3f24b801770a1f02805907eac/artifacts/track_b_sp05_economics_preparation/2026-10-06/authorization.json) |
| 明示指示receipt | [Aで固定した利用者の全文](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9477cd2fcfca69f3f24b801770a1f02805907eac/docs/tracks/algorithm_codesign/sp05_execution_authorization_receipt.md)。同じ全文とUTF-8 SHAをJSONへ記録 |
| contract SHA-256 | `ee126a0fcde4aa5eff96dfb5835647deec42150c27703e86a3023164b888d09f` |
| authorization SHA-256 | `8c543ee89d58b12d17e26fef200142e9813d647d9e5c8f5fc4c776446bd783f8` |
| tool identity | [固定runtime・wheel／source bytes](../../../artifacts/track_b_sp05_economics_preparation/2026-10-06/tool_identity_v1.json)。起動前に全版とpygridsynth／mpmath source-tree hashを照合 |

独立branch／worktreeは`track-b-sp05-one-shot-execution-20261006`、
`/home/abe/Project/prt-worktrees/track-b-sp05-one-shot-execution-20261006`。
Aのparents=[S]、変更はauthorization JSONとoptional receiptの二pathのみ。
clean HEAD=Aでsource critical 11 hashesとruntime identityを照合してから実行した。
result commitはAの後に保存資料を追加するcommitであり、実行HEADには用いていない。
source／contract／algorithm／target／catalogue／precision／判定閾値／capsは変更していない。

モデルはordinary `Rz(θ)`とcontrolled Pauliのsigned joint native pairのみ。
分子、geometry、basis、DF rank、splitは適用なし。NPZ・DF Hamiltonian・trajectory・wrapper circuit・GPU入力は使用しない。
8 target、one exact `kπ/4` catalogue、native operator ε=`10^-6`、seed0、canonical independent PAI。
23はsourceのliteral-key policyで登録したkeys数であり、23独立physical anglesとは呼ばない。
`pi:1/2`と`pi:2/4`は同じ角度の別keyだが、契約どおり両recordを保持した。
実合成APIの一般角routeは13 keys、共通exact Clifford/T fast pathは10 keys。

## 2. 保存された全16 primitive rows

以下は**保存値の表示**。Jはdecimal endpointsを外向きに10桁へ表示し、mean／momentは約値とする。
完全なg/p/γ・J interval・native signed angles・costs・分類は[元result JSON](../../../artifacts/track_b_sp05_economics_result/2026-10-06/v1/result.json)に保存する。
合成・PAI・J・error guardの関数を再実行せず、元のprimary分類を保持した。

| target θ | primitive | deterministic T | expected T ≈ | weight second moment ≈ | J interval（外向き表示） | 保存分類 |
|---|---|---:|---:|---:|---|---|
| +π/16 | ordinary_Rz | 68 | 0.25989153 | 1.12698254 | [0.0043072532, 0.0043072533] | STRICT |
| +π/16 | controlled_pair | 132 | 0.26765632 | 1.15100725 | [0.0023338967, 0.0023338968] | STRICT |
| −π/16 | ordinary_Rz | 68 | 0.25989153 | 1.12698254 | [0.0043072532, 0.0043072533] | STRICT |
| −π/16 | controlled_pair | 132 | 0.26765632 | 1.15100725 | [0.0023338967, 0.0023338968] | STRICT |
| +π/8 | ordinary_Rz | 69 | 0.50000000 | 1.17157288 | [0.0084896585, 0.0084896586] | STRICT |
| +π/8 | controlled_pair | 136 | 0.51978306 | 1.27008964 | [0.0048541991, 0.0048541992] | STRICT |
| −π/8 | ordinary_Rz | 69 | 0.50000000 | 1.17157288 | [0.0084896585, 0.0084896586] | STRICT |
| −π/8 | controlled_pair | 136 | 0.51978306 | 1.27008964 | [0.0048541991, 0.0048541992] | STRICT |
| 3π/16 | ordinary_Rz | 69 | 0.74010847 | 1.12698254 | [0.0120882509, 0.0120882510] | STRICT |
| 3π/16 | controlled_pair | 136 | 0.76222072 | 1.34633590 | [0.0075456258, 0.0075456259] | STRICT |
| +1/5 rad | ordinary_Rz | 69 | 0.26446907 | 1.12860475 | [0.0043258123, 0.0043258124] | STRICT |
| +1/5 rad | controlled_pair | 126 | 0.27246564 | 1.15355083 | [0.0024944679, 0.0024944680] | STRICT |
| −1/5 rad | ordinary_Rz | 69 | 0.26446907 | 1.12860475 | [0.0043258123, 0.0043258124] | STRICT |
| −1/5 rad | controlled_pair | 126 | 0.27246564 | 1.15355083 | [0.0024944679, 0.0024944680] | STRICT |
| π/2 | ordinary_Rz | 0 | 0* | 1* | — | ZERO_COST |
| π/2 | controlled_pair | 2 | 2.00000000 | 1.00000000 | [1, 1] | NO_STRICT |

STRICT=`STRICT_TRADEOFF`、NO_STRICT=`NO_STRICT_TRADEOFF`、ZERO_COST=`ZERO_COST_BASELINE_NO_STRICT_GAIN`。
`*`：zero-cost rowはsourceのearly returnによりmean／moment fieldを出力していない。
保存済みγ=1、p=(1,0,0)、active notch T=0から、監査の記述的補足としてmoment=1、mean T=0を示した。
元resultは編集していない。Jはnullのまま、positive witnessには含めない。

π-rationalの非exact target 5件と±1/5 rad 2件で、それぞれordinary／controlledの両rowがstrictだった。
ordinaryのgeneric deterministic Tは68–69、controlled pairの合計は126–136。
正負1/5 radの保存Jはordinary約0.00432581、controlled約0.00249447。
正負π/16、正負π/8、3π/16を含むnonexact controlled pairの保存momentは約1.15101–1.34634。
独立二nativeの`γ_1²γ_2²` penaltyを含めたJでもstrictだった。
π/2 controlはordinary T=0で除外、controlled native ±π/4は各1 T、合計2、moment1、J=[1,1]でstrictではない。

この表示はwhole-wrapper shot数やcompiled総costの改善率ではない。

## 3. sequence／guard／resource／failure

各keyにはgate string、そのSHA-256、T/T† count、Clifford count、scalar W count、
phase witness、operator／diamond upper、error_pass、wall／CPU／peak RSSを保存した。
保存sequenceの文字とhashからT/T†を照合し、全23 guardsの保存upperと契約εを比較した。
native operator upperの保存最大は約`1.46314903e-7`で、全23件が`10^-6`以内。
guard自体やchannel matrixを再計算してはいない。

| 項目 | 保存・観測値 | 契約cap |
|---|---:|---:|
| synthesis keys／primitive rows | 23／16、欠落0 | keys<=32 |
| runner wall | 4.271187 s | 1200 s |
| 外側GNU time wall | 4.34 s | 1200 s |
| worker CPU＋parent CPU（runner） | 4.206252＋0.106153 s | 900 s |
| 外側GNU time user＋system | 3.89＋0.47 s | 900 s |
| GNU time max individual RSS | 162624 KiB（158.8125 MiB） | combined RSS<=1024 MiB |
| conservative combined RSS upper | 325248 KiB（317.625 MiB） | 1024 MiB |
| 元result bytes | 75972 | 2097152 |
| cap hit／runtime exception／error failure／numeric inconclusive | 全て0 | hitならSTOP、retry0 |

同時parent＋workerは二processまで。GNU timeのmax individual RSSの2倍をcombined RSSの保守的上限として照合した。
combined RSSの精密なtime-series peakはsource resultに保存されていない。runnerの監視にcap hitはなく、
上記の保存ログに基づく上限もcap以内だった。
全保存keyのper-key CPU／wall／sequence長もcap以内。child address limitは固定sourceで適用され、例外なし。
stderrは空、process exit0、partial failureなし、retry0。

## 4. 保存値・provenance監査

- [result.json](../../../artifacts/track_b_sp05_economics_result/2026-10-06/v1/result.json)：元23 keys／16 rows。
- [one-shot marker](../../../artifacts/track_b_sp05_economics_result/2026-10-06/v1/one_shot_consumed.json)：測定前にexclusive create。consumedのまま保持。
- [prelaunch record](../../../artifacts/track_b_sp05_economics_result/2026-10-06/v1/prelaunch.json)と
  [execution receipt](../../../artifacts/track_b_sp05_economics_result/2026-10-06/v1/execution_receipt.json)：HEAD/source/runtime、run1／retry0、STOP。
- [GNU time resource log](../../../artifacts/track_b_sp05_economics_result/2026-10-06/v1/process_resources.txt)、
  [stdout](../../../artifacts/track_b_sp05_economics_result/2026-10-06/v1/stdout.txt)、
  [stderr](../../../artifacts/track_b_sp05_economics_result/2026-10-06/v1/stderr.txt)。
- [saved-value audit](../../../artifacts/track_b_sp05_economics_result/2026-10-06/v1/saved_value_audit_v1.json)：
  coverage、sequence/count/hash、保存guard、row分類predicate、zero-cost扱い、caps／identityを照合。
- [保存interval間の代数照合](../../../artifacts/track_b_sp05_economics_result/2026-10-06/v1/saved_interval_relations_v1.json)：
  全16 rowsのg/p/γ、moment、mean、`保存J×C_det`と`保存moment×保存mean`のenclosure整合をFractionで照合。
  新しいJの除算・評価や三角関数計算を行わず、保存関係の矛盾を検査する。
- [監査script](../../../scripts/tracks/algorithm_codesign/audit_sp05_saved_result.py)：stdlibだけで保存fieldを検査。
  synthesis／PAI／J／error-guardの再呼出し、primary分類変更、science rerunは0。
- [result manifest](../../../artifacts/track_b_sp05_economics_result/2026-10-06/v1/result_manifest_v1.json)：選択保存pathのSHA-256と証拠・STOP状態。

本科学実行の後にfull test suite／synthetic testの再実行は行わず、固定sourceで保存された34 focused testsの記録を保持した。
resultはsource-bound local evidenceであり、immutable CI、外部再現、independent validationではない。

## 5. mandatory STOP・GPTへの引き継ぎ

`PRIMITIVE_TRADEOFF_EXISTS`はcheap exact notchとsampling momentの交換がこの実装で存在したことを示す。
既知PAI機構のprimitive確認であり、新規性成立、DF-native improvement、D/R placement advantage、
wrapper-level pilot／16-cell pilotのGOを認可しない。
初期状態、wrapper全gate列、累積weight、finite synthesis／finite-RTE bias、Bernstein range、shot ceiling、
compiled／workspace／state costを含む研究判断は未実施。

GPT側で本16 rowsを確認し、whole-wrapper pilotの情報価値・必要性・scope、研究BのRQ／新規性／着地点を判断する。
Codex側は追加target・第二catalogue・precision／threshold変更・科学実行を行わない。
B-F限定closure、BM現adapter new-method closure、旧BM-1未実行、過去STOP、Aの証拠境界は維持する。
必要最小限のresult／監査／本文・索引をcommit/pushし、固定commitでGPTへ渡して停止する。
