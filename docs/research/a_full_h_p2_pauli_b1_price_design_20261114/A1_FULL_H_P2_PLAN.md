# A1：Full-H κ2 P-2の較正設計の修正（提案・未採択・未実行）

[REPORT](REPORT.md)／[候補設計の数値（人工）](DESIGNS.json)／[生成script](BUILD_DESIGNS.py)／[test](TESTS.txt)／[HANDOFF](HANDOFF.md)。

**扱いの区分**
- この文書は設計の提案である。較正点・件数・精度条件は、ユーザーが採択するまで正式な計画ではない。
- **P-2は実行しない。**
- 新しいsampling、compile、価格の取得・再解析、fitは0。

## 1. 現行案の欠陥：係数cが識別できない

C（`c628e9e8` PRICE_VALIDATION_PLAN §2）とD（`26d4f23e` FINITE_SCIENCE_PLAN P-2）の現行案は次のとおり。

- modelは\(E[C]=a+b\,s+c\,(q-1)\)。featureは`one`、`event_count`（\(s=qR\)）、`outer_boundary_count`（\(q-1\)）。
- 較正点は単一group cell（T=0.1、**q=1**、R∈{566, 4528, 36223}、各n=2）。
- holdoutはκ2 m5（n=2）とm6（n=1）。合計18 compile。

較正の設計行列は次のようになる。

\[
X=\begin{pmatrix}1&566&0\\1&4528&0\\1&36223&0\end{pmatrix}
\]

- 第3列が0なので、rankは2である。
- null方向は\((0,0,1)\)で、cはどの較正値にも影響しない。
- 既存の`price_model_validation.fit`も、正規方程式\(X^\top X\)が特異なので`Unidentified model design (singular)`で拒否する。
  - これは人工recordで確認した（test 1）。
  - ただしfitが拒否するのは**価格を取得した後**である。現行の18 compile案を実行すると、12 compileの較正が係数を決められないまま終わる。
- **したがって現行案は、正式な実行計画として扱えない。**

## 2. 識別可能性の条件

**full column rankの条件**

- 3係数が識別できる必要十分条件は、較正点の\((x,s)=(q-1,\;qR)\)が平面上で**一直線上に並ばない**ことである。
- 理由：列が従属することは、ある\((\alpha,\beta,\gamma)\neq0\)で全cellが\(\alpha+\beta s+\gamma x=0\)を満たすこと、すなわち全点が一つの直線上にあることと同じである。
- 一意解には、異なる点が3つ以上必要である。

**よくある従属の形**

| 設計 | 従属の理由 | 判定reason |
|---|---|---|
| q=1だけ | 全点がx=0の直線上にある | `ALL_Q_EQUAL_ONE_…` |
| qが1種類だけ | xが定数で、`one`と共線 | `SINGLE_Q_VALUE_…` |
| Rが固定でqだけ変える | \(s=R(x+1)\)がxのaffine関数になる（例：q∈{1,2,4}、R=566） | `FIXED_R_EVENT_COUNT_IS_AFFINE_IN_Q` |
| qとRの両方を変えても直線上 | 例：(q,R)=(1,12)、(2,9)、(3,8)、(6,7)。いずれも\(s=12+6x\) | `CELLS_COLLINEAR_IN_FEATURE_PLANE` |

**実装**：[`finite_price_design`](../../../src/trottertracks/finite_price_design.py)

- `identifiability`：exactな分数でrankを求め、従属する場合はnull方向と理由を返す。
- `check_plan`：計画を**取得前に**検査し、rank不足なら拒否する。同時に次も検査する。
  - modelの領域が一つであること（route、law、lowering・native・scalar、compiler profile、quantity、kind）
  - 較正とholdoutが分離されていること（同じcellを両方に入れない。holdoutを較正のgroup pathから合成しない）
  - wiring cohort（P-1・B0）を除外すること
  - n_pathsが整数で、law期待値を宣言していないこと
  - paired compositionの規則
- `fit_identified`：記録が計画の較正cell・軸と完全に一致する場合に限り、Cの`price_model_validation.fit`（`c628e9e8`のblobを無変更で実行）へ渡す。

## 3. q>1を含む候補設計（holdoutはm5 n=2、m6 n=1で共通）

**読み方**

- SEは人工の雑音modelによる比較値である。1 pathの価格の分散を\(\sigma_e^2 s\)とし、\(\sigma_e\)を単位にしている。
- 相対SEは\(\mathrm{SE}/(b\,s)\)を\(\rho=\sigma_e/b\)（未知）あたりで表したものである。
- **実価格の分散の推定ではない。**
- group内のbasis再利用（Markov依存）は定数を変えるが、設計間の順位を比べる目的では無視した。

| 設計 | 較正の内容 | 新規compile | rank | SE(a) | SE(b) | SE(c) | m5予測の相対SE/ρ | m6 | m12 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D0 現行 | q=1・R{566,4528,36223}、各n2 | 18 | **2** | 拒否 | | | | | |
| D5 R固定 | q∈{1,2,4}、R=566 | — | **2** | 拒否 | | | | | |
| D1 | D0＋q2・q4（R566、独立、各n2） | 26 | 3 | 23.7 | 0.004 | 13.8 | 4.6% | 2.4% | 0.40% |
| D2 | D0＋paired q2（後述） | 20 | 3 | 27.4 | 0.004 | 34.0 | 11.6% | 6.0% | 0.43% |
| D3 | G566をn4に＋paired q4 | 24 | 3 | 26.5 | 0.004 | 15.7 | 5.3% | 2.7% | 0.40% |
| **D6（最小案）** | D0＋**q8・R566を独立にn2** | **22** | 3 | 27.4 | 0.004 | **7.6** | 2.5% | 1.3% | 0.40% |
| **D7（推奨案）** | D6＋paired q2（R566の2 pathを連結） | **24** | 3 | 26.1 | 0.004 | **7.8** | 2.5% | 1.3% | 0.39% |
| D4 | κ2 m2・m3・m4のroundを較正に使う | 18 | 3 | 30.1 | 0.074 | 12.0 | 3.7% | 5.4% | 7.3% |

**読み取れること**

1. **m12の予測精度は、ほぼSE(b)で決まる。**
   - m12（s=1.48e8）の相対SEは、D1〜D7でほぼ同じ（ρあたり約0.4%）である。
   - 理由は、R=36223の単一group cell（m12の1 groupと同じτ）がbを強く決めるからである。
2. **holdoutのm5・m6を判定できるかは、SE(c)で決まる。**
   - m6には境界が63個あり、予測の不確かさの主成分は\(63\cdot\mathrm{SE}(c)\)である。
   - 境界を7個持つq8の独立cell（D6・D7）で、SE(c)が最小になった。
3. **paired compositionは、OLSの精度ではD6に及ばない（D2・D3）。** paired compositionとは、R566のgroup pathを連結し、共通乱数で作るcellのことである。
   - 主な価値は、\(\Delta=C(g_1\circ g_2)-C(g_1)-C(g_2)\)を観測できることにある。この量はevent由来の雑音を含まない。
   - これにより、**境界の加法性・局所性**（compilerの最適化がgroupをまたいで効かないこと）を直接検査できる。これは現在、未検証の仮説である。
   - D7は、識別（q8）と仮説の直接検査（paired q2）を、+6 compileで両立する案である。
4. **D4（小さいroundを較正に使う案）は安いが、外挿で劣化する。**
   - m2〜m4はτが大きく（0.06〜0.015）、order-2 eventの頻度が長時間roundと異なる。
   - s範囲（≤2272）からの外挿になるため、m12で相対SEがρあたり7%に悪化する。

**推奨（提案）**

- **D7**を推奨し、最小案として**D6**を併記する。
- holdoutの判定に許容差H-d（標本誤差を含める案）を使う場合は、m6をn=2にする（+2 compile、合計26）。n=1ではholdout自身のSEが定まらない。

**cellの定義**

- 全cellのgroup時間は0.1、新ZZ、K2で共通。
- q8・R566は、m6の連続8 group（T=0.8、s=4528）と同じ構造で、seedが独立である。
- paired cell（P2R566）は、G566の2 path（path0・path1）のgroup event列を、seedを指定してそのまま連結し、T=0.2・q=2として1 pathだけcompileする。
- seedの指定は既存callerにない（D担当、[HANDOFF](HANDOFF.md)）。

## 4. 既存P-2案との差分

| 項目 | 既存（C/D） | 改訂案（D7） |
|---|---|---|
| 較正 | q=1のgroup 3種、各n2（12 compile） | 同じ3種＋**q8・R566を独立n2（4）**＋**paired q2（2）** |
| 識別 | rank 2（cは不定） | rank 3（exactに検査済み） |
| holdout | m5 n2・m6 n1（6） | 同じ（H-dを使う場合はm6をn2に） |
| compile数 | 18 | 24（m6をn2にすると26） |
| 加法性の検査 | なし | paired Δで直接検査 |
| 取得前の検査 | なし（fitが後で拒否する） | `check_plan`でrank・領域・分離を検査 |

## 5. P-1の単一path価格とP-2の価格の区別

- P-1（`a04470c0a`）について
  - κ2 **m3全round**（q8・R71、s=568）の、normalな1 pathの直接価格である（RZ 92,448）。
  - `DIRECT_COMPILED_FINITE_SAMPLE`／`FINITE_COHORT_ONLY`、n_paths=1、cohortは`…_M3_WIRING_V1`。
  - `classify_saved_price`は`SINGLE_PATH_DIRECT_FINITE_SAMPLE`と判定する。
  - **用途は配線の証拠に限る**。E[C]、較正点、holdout、Gには使わない。
- P-2の価格について
  - 事前に宣言したcell（group cell・q>1 cell・holdout round）ごとのfinite cohortで、n_pathsとlawを付ける。
  - 記録は`price_model_validation.record`形式（`finite_sample_mean_is_law_expectation=false`）。
- P-1を較正に加える場合（m3は(x,s)=(7,568)で、識別には寄与する）
  - 取得の前に「m3 cellのcohortのpath0」として明示的に宣言し、追加pathを取得する必要がある。
  - τ（0.03）がlong roundと異なる点に注意が要る。
  - **既定では加えない**（`check_plan`はwiring cohortを拒否する）。

## 6. 有限cohortの観測値と理想law期待値の違い

| 量 | 意味 | 現在の資格 |
|---|---|---|
| 1 pathの価格 | 特定seedの回路のcompiled RZ数 | 観測値。lawの代表ではない |
| finite cohortの平均（n path） | 実sampler（PCG64・float64 CDF）による標本平均 | `FINITE_COHORT_ONLY`。SE・区間はinterval contract次第 |
| 実sampler lawの期待値 | 無限path極限 | sampler lawと理想lawの距離は未資格（`ideal_sampler_distance_qualified=false`） |
| 理想event lawの期待値E[C] | 理想finite-K lawでの期待値 | 未取得。modelの予測もholdoutを通過するまで`CALIBRATED_UNVALIDATED` |

**期待価格の資格に必要な追加証拠**

1. cellごとのinterval contract（n、SE、coverage。n≥2は必須で、n=2では自由度1）
2. 事前宣言した許容差でのholdout通過（A8）
3. 境界加法性の検査（paired Δの一貫性）
4. sampler lawの距離の資格
5. order-2など低頻度eventの寄与
   - long roundではτ²に比例して小さいが、観測されないorderの条件付き価格はUNKNOWNのまま残る。
6. m7〜m12の外挿範囲の扱い
   - holdoutのsは最大36224まで。
   - m12の1 group（R=36223）は直接測るが、q=4096への外挿は残る。

## 7. 未検証のまま維持する仮定

- **group境界でのcompiler最適化**：opt1の局所passが境界をまたいで効く量。paired Δで検査可能だが、未検証。
- **groupのi.i.d.性**：seedが独立で、同じlawに従うことは構造上成り立つ。ただし価格の加法性は別問題である。
- **長時間外挿**：m7〜m12はholdoutを通過しても`UNQUALIFIED_MODEL`の外挿scenarioのままとする（Cの規則）。

## 8. 既存成果の照合（§10.5）

**調査範囲**

- Dのκ schedule（`26d4f23e`）：KAPPA_CANDIDATES、`saved_asset_status`
- 旧Full-H（light、T2.0、R448）の価格記述：KAPPA_CONNECTION §6
- Cの価格coverage（`rpe_price_evidence_validation_20261113`）とPRICE_VALIDATION_PLAN
- P-1再検証（`a04470c0a`）

| 予定cell（X/Yそれぞれ） | 保存記録 | 判定 |
|---|---|---|
| G566・G4528・G36223（q=1、T=0.1） | なし。κ候補のphysical IDはすべて`NO_SAVED_EXACT_CANDIDATE`。旧Full-Hのdirect価格は旧lowering（generic crzz）で、T・q・R・Kも異なる | 新規取得候補 |
| Q8R566（q=8、T=0.8） | なし。P-1はm3（R71）で、physical IDとRが異なる | 新規取得候補 |
| P2R566（paired） | なし（G566のpathに依存） | 新規取得候補（G566の後に取得） |
| m5・m6 round | なし | 新規取得候補（holdout） |
| P-1 m3 | あり（1 path、新ZZ、X/Y） | **計画外**。配線の証拠としてのみ引用する |
| 旧Full-H価格（light・T2.0） | あり（旧lowering） | **再利用不可**（guardが拒否する。旧価格からZZ差を引く換算もしない、D KAPPA_CONNECTION） |

**資源の見通し（未計測）**

- P-1（s=568）は2 compileとstatevectorで87.9 s、peak RSS 0.6 GB だった。
- R=36223とm6（s≈36k）は、命令数でその約64倍になる。時間・RSSは線形とは限らないので、未知として扱う。
- 段階的に取得することを推奨する。
  1. まずG566・P2R566・Q8R566・m5を取得する（s≤9056）。
  2. 資源を観測してから、G4528・G36223・m6へ進む。
- 独立なcell・軸はprocess並列にできる。
