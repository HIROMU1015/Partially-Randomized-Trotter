# RPE m17 B-1a：構成の高精度化でgapは変化したが診断超過が残る

2026-11-03 JST。**B-1aは1回で正常完了。96/160 bitでmeanとtailの収束診断を満たした。**
保存A以降の構成算術を高精度化するとLaw/referenceのcomplex mean gapは約3.2584e-10となった。
Task Aの保存factor固定gap約6.2431e-10から変化したが、元1e-10の数値診断閾値を超える差は残る。
scalarとmatrixの個別の寄与、保存前の入力表現誤差、実装・数式定義の違いは一意に分離できていない。
旧numeric_diagnostic_status=INCONSISTENT、u_num_bound=null、RPE_tolerance_certified=false、
scientific_status=UNDETERMINEDを保持し、mandatory STOPした。

## SOURCE・条件・入力境界

科学実行SOURCE：`9d27f931d6a572210eb206569806f96a46c2d094`。
[固定計画](../rpe_m17_b1_construction_precision_preparation_20261103/B1_SCIENTIFIC_EXECUTION_PLAN.md)、
[入力境界](../rpe_m17_b1_construction_precision_preparation_20261103/B1_INPUT_BOUNDARIES.md)、
[adapter](../rpe_m17_b1_construction_precision_preparation_20261103/b1_adapter.py)、
[driver](../rpe_m17_b1_construction_precision_preparation_20261103/run_b1_construction.py)を変更せず実行した。
実行branch：`codex/rpe-m17-b1-construction-precision-execution-20261103`。
SOURCE commitと結果公開commitは異なる役割である。

linear H4、隣接1.00 Å、STO-3G、保存legacy DF12、Bの保存physical ground state、
8 qubit・4電子sector dimension70、Partial L_D3、単一δ=.02、m17/q131072/r32/K2。
identity policy `extract_identity_phase`、`coefficient_atol=1e-12`、
保存N411/axis、α=.016957089147912167、ε=.15244325851620388、保存PF/RTE/stat予算を保持し、再評価していない。

元m17のRESULT/7行列（結果commit `cd749066904d3f29b28b03c19ab69370372b157f`、科学SOURCEは
`128e9a5bf2630cd0333ec5f69b81b25def78eb79`）、
Bのstate（`e22d7718c47358ff650dbc04e8e72f6ab1c8bf41`）、Task AのRESULT/S64
（`a5304864ec5e1412a36c31574bcb95fb13f80202`、科学SOURCEは`4479a1c9cf150250254e2c2650f70fb43841fed1`）
のbyte/hashを照合した。library carrier `4591b8d75a2a6a55c74a51225f0c61648023c8ee`は実装の出所。
5入力のfull path/hashは固定計画と新RESULTに記録されている。

A_probability/A_coefficient、F/R、ψ、保存event確率π、τ64/B64は固定した。
θ64/cos64/sin64/logB64/float normalizerは元保存値ではなく、今回初めてSOURCE replayした値。
新RESULTにbinary64 hexと由来を記録した。
HP経路は保存値と新θ64を正確なdyadicとして持ち上げ、HP cos/sin、独立なevent和とTaylor recurrence、
内側r乗、referenceのHP exp(r logB64)による一回のnormalizationを評価した。
π/Bの定義式からの再生成やπの再正規化はしていない。
新tailをbinary64へ戻さず、Task Aと同じHP `(R*tail)*F`、q乗、ψ期待値を使った。

## 実行結果

原記録：[RESULT.json](../../../artifacts/rpe_m17_b1_construction_precision/2026-11-03/run01/RESULT.json)、
[float_source_replay.npz](../../../artifacts/rpe_m17_b1_construction_precision/2026-11-03/run01/float_source_replay.npz)、
[high_precision_operators.zip](../../../artifacts/rpe_m17_b1_construction_precision/2026-11-03/run01/high_precision_operators.zip)。
以下は原RESULTと既存Task A結果の転記・表示桁の省略。報告作成時にoperatorやmeanを再評価していない。

float SOURCE replayのLaw event/Law tail/Reference tailは全て保存配列とarray_equal=true、Frobenius差0.0。
したがって比較基準はTask Aの保存factor固定・160 bit meanをそのまま再利用した。
conditional replay-tail基準の新meanは0件、旧tailのmean再計算も0件。

| 段階 | Law/referenceのcomplex mean gap | 元1e-10との関係 |
|---|---:|---|
| 元m17 float64（既存結果） | 6.2594202981e-10 | 超過 |
| Task A・保存factor固定160 bit（既存結果） | 6.2430653168e-10 | ABOVE |
| B-1a・保存A固定、構成から160 bit（今回） | 3.2583605418e-10 | ABOVE |

新しい2path×96/160 bitの計4 meanを評価し、全て計画内で正常完了。
224 bit、中断、retry、追加samplingは不要だった。

| Path | B-1a 160 bit Re（省略） | Im（省略） | Task A保存factor meanからのcomplex変化 |
|---|---:|---:|---:|
| Law | .32902182391106079299 | −.46875529437102468785 | 3.7642106964e-10 |
| Reference | .32902182372385757587 | −.46875529410433368743 | 7.7950592981e-11 |

Law−Referenceの符号付き差はRe約1.8720321712e-10、Im約−2.6669100042e-10。
新meanの表示桁はbackend値であり、真の平均に対する保証された桁数ではない。
path別の変化量はgapへの寄与率ではない。

### 収束と中間tail

| Path | 96/160 bitのcomplex mean変化 | 同tailのFrobenius変化 | 160 bit tail対元保存tailのFrobenius差 |
|---|---:|---:|---:|
| Law | 2.4820384236e-24 | 1.2495370698e-27 | 2.0680536763e-14 |
| Reference | 1.3411450316e-23 | 1.1514839379e-27 | 1.3880051835e-14 |

mean基準1e-13、tail基準1e-13/q（約7.6294e-19）の両方を満たした。
160 bitの新Law/Reference tail間のFrobenius差は約3.6316848193e-14。
tail差をmean gapの誤差上界や原因の寄与率に読み替えない。
収束差と288 bitのscalar比較は、外向き丸めや厳密包含によるu_numの保証ではない。

## 判別できた内容と限界

同じ保存A・同じHP後段のまま構成算術を変えると、両pathのmeanとgapが変化した。
この固定入力条件でtail構成の算術が数値結果に影響することは観測できた。
高精度化した箇所にはcos/sin、Taylor除算、行列積・和、r乗、normalizationが含まれるため、
その内訳や単一の主要原因まではこの比較で識別しない。

事前規則による候補labelは`FROZEN_INPUT_OR_DEFINITION_DIFFERENCE_POSSIBLE`。
差は診断閾値を超えたままであり、保存A_prob/A_coeff、π64、θ64、B64の表現、定義・規約の違い等が残り得る。
両式の代数的同値性には入力の整合が必要で、保存された丸め済み入力では自動的に満たされない。
残差を実装bug・定義差・component平均誤差のいずれかと確定した結果ではない。
LawをTaylorに置換したり、診断閾値を緩めたりしていない。

保存統計残余約8.60e-5は元m17結果の引用。今回のgapが1e-10を超えたことだけでRPE必要精度への不適合とはしない。
逆に、gapが残余より小さいことや精度間で安定したことだけでもRPE適合を認定しない。
真の周辺平均への誤差保証が未確立なので、旧INCONSISTENT、u_num=null、科学UNDETERMINEDを保持する。

## 資源・保全・出力照合

起動前にCPU affinity32、load average約1.3、利用可能RAM約52 GiBを確認。
GPU driverと接続できず、mpmath1.3.0/Python backend、NumPy1.26.4を使用。
独立4jobをspawn worker4で並列化し、float replayのBLAS/OpenMP/MKL設定は各4 thread。
mpmathの主要な多倍長matrix演算はPython backend。共有環境や他jobの設定は変更していない。
実行wall timeは約12.654秒、exit0。速度向上率、実測CPU占有率やprocess RSSは報告していない。
準備SOURCEの人工23＋回帰31＝54 PASSを利用し、科学実行のために旧SOURCEを変更していない。

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python docs/research/rpe_m17_b1_construction_precision_preparation_20261103/run_b1_construction.py --run --output-dir artifacts/rpe_m17_b1_construction_precision/2026-11-03/run01
```

| 原本 | Bytes | SHA256 |
|---|---:|---|
| RESULT.json | 17011 | `d392c2430882ca239e1e4ed5f8095dc912b5ef3597313b98cf0bf51e5ea7415e` |
| float_source_replay.npz | 289262 | `f3831949a05207a8c677a257577d29d24fc34264297f5252fcdf8c96b3d93d8c` |
| high_precision_operators.zip | 2005532 | `70c368097986b295914b776133789c2d254b8ab1ae72d7fedb75a40a650cb1a5` |

ZIPは4member、各path/bitのone-eventまたはpolynomialとtailをHPのdecimal round-tripで保存。
原本のbyte hash・ZIP整合性・記録schemaを確認した。科学計算の再実行はしていない。
元m17/Task A/V1/V2/V3/R1/A/BのSOURCE・RESULT・保存行列・STOP、N/α/β/εを保全した。
新SCF/DF/H/state/basis/component/F/R生成、sampling/shot/compile、価格・成功率再評価、独立詳細科学レビューは0。

次段案：**Codex**が別指示後に、残差を分けるための固定成分入力の可用性とB-1b/B-2の要否を静的に整理し、最小計画を提案する。
**ユーザー**がその対象・入力境界を判断し、新しい科学実行を別途明示許可する。
今回は追加scalar再解析、B-1b/B-2、他round・samplingへ進まずmandatory STOPする。
