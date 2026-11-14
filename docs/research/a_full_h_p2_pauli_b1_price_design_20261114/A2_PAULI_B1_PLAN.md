# A2：PR-Pauli LD1/LD2のB1と期待価格の検証計画（提案・未採択・未実行）

[REPORT](REPORT.md)／[候補設計（人工SE）](DESIGNS.json)／[HANDOFF](HANDOFF.md)。

**扱いの区分**
- B1のcompile、新しい価格の評価、fitはいずれも0。
- 数値は、保存されている記録の引用と、設計上のmetadata（R・q・s）だけである。

## 1. B0の確認（D `00f74f86`、raw `8502dd33`、実行SOURCE `9abbcc56`）

| 項目 | 内容 |
|---|---|
| 系 | H4 DF10（保存tol-only）、8 system qubit＋ancilla |
| PR構成 | LD1、S2、ΔPF=0.1、m0（T=0.1、qPF=1、R_outer=3）、K2、P=0 |
| 辞書 | A P0 `4220f300`、LD1は184項、SHA-256 `dbdde5a1…` |
| λ_P・s_R | λ_P＝3.3066052942233157、s_R＝0.4177175120626939。c＝2.29310124732との和（total T）を一回だけ適用 |
| event | normal 1 path（cohort `D_PAULI_LD1_M0_WIRING_V1`、index 0、seed `14788159701244282917`）。3 eventで、order0が2、order2が1 |
| 回路 | DF headは既存native、Pauli tailは`append_pauli_event`（`9c4006bf`）。D2は中心回転だけを制御する |
| route・lowering | `PR_PAULI_TAIL`／`PAULI_D2_CENTRAL_ROTATION_CONTROL_9C4006BF` |
| compiler | Qiskit 1.3.0、rz/sx/x/cx、opt1、seed17、hash `1239304c…` |
| 価格 | X/YともRZ 1380、CX 578、depth 834、size 2695 |
| record | `DIRECT_COMPILED_FINITE_SAMPLE`／`FINITE_COHORT_ONLY`、n=1、C guardはPASS |

- **位置付け**：回路の配線・作用の検証であり、Pauli lawの期待価格ではない。
- `classify_saved_price`の判定は`SINGLE_PATH_DIRECT_FINITE_SAMPLE`。

## 2. 既存のPauli価格証拠の照合（§10.5）

| 保存証拠 | 条件 | 今回（LD1/LD2のDF head＋Pauli tail、T=0.1×2^m、qPF=2^m、K2、D2）への再利用 |
|---|---|---|
| B0（LD1 m0、1 path、X/Y） | 同じ辞書・同じroute・同じcompiler | **部分的に再利用できる**。LD1 m0 cohortのpath0として、**取得前に宣言した場合に限る**。long roundのmodel較正には使わない（m0はτ≈0.11でorder-2が多く、s=3） |
| LD2の価格 | なし | 新規（配線もまだ確認していない） |
| H4 P3（`h4_p3_*`、2026-11-02） | 案Aのresidual（Gaussian保持、M_B/M_C）、T=0.8、S2、q8・r4・**K4**、Pauliの価格標本（pilotとrun04の32 path） | **再利用不可**。分割・law・K・T・headが異なる。cohort設計（mean・SE・独立cohort）の方法の参考にとどめる |
| Track AのPauli-JW（H2/H4/H6、2026-10〜11） | 全JW Pauli H（H4は184項）。PauliのPF prefixまたは純RTE、T=0.8、q8 | **再利用不可**。Hの表現（DF headがない）、係数・確率、headが異なる。項数が184で同じでも、価格は移せない |
| PR DF-tail（LD1のE[S]・C1 cohort・D P2） | DF tail（Givens・DF event） | **再利用不可**。tailの表現が異なる。DFの構造価格・残差係数も流用しない |
| 旧Full-H・旧DFの価格model | 別route | 対象外 |

**調査範囲**
- D B0の公開物（`00f74f86`）
- A P0（`4220f300`）、Aのall-round（`714247619`）
- Cの経路計画（`c628e9e8`）
- P3・Track A Pauliの研究文書（`00f74f86`の`docs/research`）
- Cの価格coverage（`rpe_price_evidence_validation_20261113`）

## 3. compiled価格に影響する特徴量（D2 lowering、`9c4006bf`）

**1 eventの構造**

- 回転Pauli P（重みw、X因子数n_X、Y因子数n_Y）について、次の順に命令が並ぶ。
  1. 基底変換：XはH、YはSdg·H
  2. CXのladder：w−1個（targetは最後の非I qubit）
  3. **中心の`crz(θ)`（ancilla制御）**
  4. ladderの逆
  5. 基底の逆変換
- product（order-2 eventの2因子）は、非I因子ごとにancilla制御の単一qubit Pauli（cx・cy・cz）になる。
- event位相とproductの符号は、stepごとに一つのancilla `p(φ)`へまとめられる（`central_control_tokens`）。

| 特徴量 | 価格への経路 |
|---|---|
| 回転の重みw | CXのladder 2(w−1)。RZはcrzのcompile後の分と、基底変換の分 |
| n_X、n_Y | 基底変換のH、Sdg·H（RZ・SXを含む） |
| target qubit・軸の配置 | ladderの形。隣接eventと相殺できるか |
| eventのorder（0か2） | productの制御Pauliの数（cyはRZを伴う） |
| productの符号・event位相 | stepごとのancilla `p(φ)`（φ=0でも命令として置かれる） |
| **隣接するeventの組（順序）** | opt1は、隣接する単一qubit gateを統合し、CXを相殺する。同じqubit上で連続する基底変換や、共通のladderは消え得る。そのため価格はeventのi.i.d.和ではなく、組（Markov）の特徴量に依存する |
| DF head（step毎） | 決定論のnative S2 head（LD1はone-body＋fragment0、LD2は＋fragment1）。symmetric_directional制御。step数qに比例し、head・tail境界での統合も含む |
| scalar | total Tで一回（定数） |
| X/Y | Y側wrapperの`sdg`（P-1・B0ではX/Yの価格は同じだった） |

**期待価格**：lawのもとでの期待値は\(E[C]=a+h\,q+b\,s\)の形になる。
- b：定常的な1 eventあたりの費用（隣接を含む）
- h：headと、stepごとの境界の費用

## 4. 既存実装を再利用した取得経路

Dの`scripts/run_pauli_ld1_m0_pilot.py`（`62ae1bd5`、run02復旧版は`9abbcc56`）は、次の処理を一つの子processで行う。

- `pilot_core`、`records`
- Cのplanner（`I.plan`、`price_context`、`pauli_requests`）
- `pauli_tail_price_connection.compose`／`controlled_tail`
- 独立したfull-2^8 oracle
- 既存のcompile cache

期待価格の検証に必要な最小の拡張は、Dの担当である（[HANDOFF](HANDOFF.md)）。

1. prefix（LD1/LD2）とcell（round、または非configuration cell）を引数にする。
   - 非configuration cellは、q=1でT=0.1・任意のR、またはq>1。
   - recordはCの`price_model_validation.record`で受ける（Cの規定どおり）。
2. paired compositionのための**明示的なstep seed表**を受け付ける。
   - seedの既定は、canonical candidate ID・cohort・index・outerで決まる。
3. pathごとにevent metadataを保存する。order・回転と積のlabel・step位相などで、後の構造model用である。追加のcompileは要らない。
4. configuration roundのholdout（m5・m6）は、既存のsidecar・guard経路（`PR_PAULI_TAIL`）を使う。

## 5. 既存B1案（LD1/LD2各2 path、X/Y計8 compile）で検証できること

C（PRICE_VALIDATION_PLAN §3）の「辞書ごとにfresh normal 2 path×X/Y」は、roundを指定していない。B0と同じm0を想定して評価する。

| 検証できる | 検証できない |
|---|---|
| LD2の実回路配線（初回）：s_R・head（LD2）・scalar・X/Yの符号 | law平均・CI（2 pathでは自由度1） |
| m0での2 path間の差（ばらつきの目安） | 1 eventあたりの傾きb、headの項h、境界・隣接の効果 |
| order-2の扱い（sampleされた場合のみ） | m1〜m12への外挿（m12はLD1で4.46e7 event） |
| LD1でのB0の再現性（別path） | sampler lawの資格、低頻度eventの寄与 |
| | 価格modelの較正。設計行列はrank 1で、`check_plan`は拒否する（test 8） |

- **2 pathで期待価格の精度が保証されたことにはならない。**
- m0のpathは、long roundのeventの性質（τが小さくorder-2がまれ）を代表しない。

## 6. 有限cohort・ばらつき・理想law・実sampler lawの区別

| 量 | 内容 | 必要な証拠 |
|---|---|---|
| 1 pathの価格 | B0など | 配線の証拠に限る |
| finite cohortの平均 | cellごとのn path | n、SE、coverage（interval contract） |
| path間のばらつき | 隣接・order・labelに依存する | n≥2。過分散を確認するなら、さらに多いn |
| 実sampler lawの期待値 | PCG64・CDF写像のlaw | sampler距離の資格 |
| 理想lawの期待値E[C] | 辞書確率p_j・finite K2 law | 上記に加え、model（または構造kernel）の検証 |

## 7. B1以降に必要な最小限の有限検証（提案。件数はX/Y合計）

| 段階 | 内容 | 新規compile | 目的 |
|---|---|---:|---|
| **B1-W** | LD2 m0、normal 1 path | 2 | LD2の配線（LD1のB0に相当）。LD2の価格を使う前に必須 |
| B1-V（任意） | m0の2 path目。LD1はB0をpath0として**事前に宣言**し、1 pathを追加。LD2はB1-Wを含めて2 path | LD1：2、LD2：2 | m0のばらつきの目安（Cの元のB1を、B0の再利用で8から6 compileに減らす） |
| **B1-CAL**（期待価格modelの推奨案） | 辞書ごとに次を取得する。<br>・1 step cell（q=1、T=0.1）を、R∈{R_m6, R_m9, R_m12}で各n2。LD1は171・1361・10885、LD2は66・526・4205<br>・8 stepの独立cell（R_m6）をn2<br>・R_m6の2 pathを連結したpaired 2 step cell<br>・holdoutとしてm5（n2）・m6（n1） | 辞書あたり24（計48） | \(a+h\,q+b\,s\)の識別（rank 3）、境界・head加法性の直接検査、long roundと同じτでのb |

**B1-CALの人工SE**（[DESIGNS.json](DESIGNS.json)。σ_e単位、headは雑音なしと仮定）

| 辞書 | SE(a) | SE(h) | SE(b) | m5予測の相対SE/ρ | m6 | m12 |
|---|---:|---:|---:|---:|---:|---:|
| LD1 | 16.6 | 4.3 | 0.007 | 4.6% | 2.4% | 0.72% |
| LD2 | 10.3 | 2.7 | 0.012 | 7.4% | 3.9% | 1.15% |

- 8 stepの独立cellを除いた案（20 compile）では、SE(h)が18.7（LD1）と大きく、m5・m6のholdoutを判定できない。
- 相対値の分母はb·sだけである。head h·qを含めると、相対SEはさらに小さくなる。

**順序の提案**：B1-W → B1-CAL（LD1）→ 資源を観測 → B1-CAL（LD2）。B1-Vは任意。

## 8. 期待価格modelの候補

| 候補 | 内容 | 前提・未資格の点 |
|---|---|---|
| M-lin | \(E[C]=a+h\,q+b\,s\)（B1-CALで較正し、m5・m6で独立検証） | 1 eventの費用の定常性、step境界・headの加法性、m7〜m12の外挿（`UNQUALIFIED_MODEL`） |
| M-struct | Pauli構造kernel E[S]＋残差model。E[S]は、辞書の確率p_jと隣接（組）の相殺規則から解析的に求める | Pauli用kernelは未実装。相殺規則とopt1の対応、残差の定常性、Aの記述子 |
| M-feat | labelのclass別特徴量（E[w]、E[n_X]、E[n_Y]、組の相殺）による線形model | 特徴量の定義をAの記述子に合わせて固定し、事前に宣言する必要がある |

**未資格の仮定**（いずれも追加検証が必要）
- eventの組（隣接）の効果がstepの境界をまたがないこと
- headの価格がstep間で独立であること
- sampler lawと理想lawが一致すること
- 低頻度のorderの寄与
- long roundへの外挿
