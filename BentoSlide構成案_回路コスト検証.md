# BentoSlide構成案：ランダムRTE回路のcompiled-cost近似検証

## 1. 制作依頼の要約

### 資料タイトル案

**ランダムRTE回路のcompiled-cost近似モデルと検証**

副題：

> DF＋部分ランダムコスト評価に向けた続編

### 前資料との関係

この資料は、`docs/research/DF+PRコスト評価.pdf`の続編だが、HTMLとしては別資料にする。
前資料では、PF誤差係数、有限RTEの解析誤差上界、RPE総コストへ至る変数関係を説明し、
「ランダム回路のcompiled期待コストをどう近似するか」を次の主要課題として残した。

本資料はその問いへの回答として、次の流れを説明する。

1. eventコストの単純加算を検証し、大きな過大評価のため棄却した。
2. event境界の相殺を短い局所回路から推定するconnected-cluster modelを導入した。
3. Taylor orderのrare event、Monte Carlo分散、条件移送を切り分けて検証した。
4. H4の複数条件とH5の独立holdoutを通じて、条件ごとの3-event modelと必要時4-event補正という運用規則を得た。

前資料の研究背景、PF係数$C$、有限RTEのoperator/signal誤差検証は繰り返さない。冒頭1枚で
前資料との接続だけを示し、今回の回路コスト検証へ直ちに入る。

### 資料の目的

共同研究者に、次の4点を一続きで共有する。

- なぜランダム回路の期待compiled costに近似が必要か。
- どの単純モデルが失敗し、なぜ局所境界補正へ進んだか。
- どの条件でどこまで検証し、例外条件をどう扱ったか。
- 大きな系のコスト推定に、現時点でどのモデルをどう運用するか。

意思決定を求める資料ではなく、「ここまで検証し、今後はこの規則で総コスト評価へ接続する」
という研究進捗の共有資料にする。

### 中心メッセージ

> RTE eventを個別にcompileして足す単純加算は使えない。一方、短い1--3 event局所回路から
> 境界相殺を較正するconnected-cluster modelは、H4・H5の指定条件で未使用の長い回路を
> おおむね5%以内に予測した。標準は3-eventまでとし、不通過条件だけ4-event補正へ進む。

### 主張の範囲

- 数値はlocal validationであり、immutable CIまたは外部再現結果ではない。
- H4・H5、H-chain 1.0 Å、STO-3G、明示したDF rank/snapshot、Qiskit 1.3.0、
  `rz,sx,x,cx`、optimization level 1、coupling mapなしの範囲である。
- 量子shot、noise、実backend、状態準備、full RPEおよび最終総コストは未評価。
- 約5%は運用上のmodel uncertaintyであって、数学的な厳密上界ではない。
- H12や別compilerへH4/H5係数をそのまま移送できるとは主張しない。

### 用語上の重要な指示

- `compiled cost`は、Qiskit transpile後のRZ/CX count・depth、total depth、circuit sizeと説明する。
- `holdout`は、係数調整に使っていない未使用full回路による検証と初出時に説明する。
- `paired`は、同一trajectoryのfull回路と局所窓を対応付けて差を取る構造検証と説明する。
- Taylor cutoffは$K_{\mathrm{Tay}}$と表記する。
- connected-cluster項は$\kappa_1,\kappa_2,\kappa_3,\kappa_4$と表記し、Taylor cutoffと混同しない。
- `K1--K3 model`とだけ表示せず、初出では「1--3 event connected-cluster model」と併記する。
- `z-scoreが大きい＝モデル誤差が大きい`とは限らない。残差の統計的非ゼロ性と相対的な大きさを分ける。
- 点wise正規近似95%上側値をrigorous boundまたは全metric同時信頼上界と呼ばない。

### 推奨表示仕様

- 16:9、1280×720。
- 本編16枚。補足スライドは原則作らない。
- 数式、表、フロー、chartは編集可能なHTML/CSS/SVGで作る。
- 検証の経緯が重要なので、成功結果だけでなく「単純加算の失敗」「旧IID検証の不足」も本文に含める。
- 色は、full compile基準を濃紺、単純加算を赤、connected補正を青緑、未検証を灰色にする。
- chartはsource JSONから値を読み、必要なら生成コードまたはartifactへの参照を発表者ノートに残す。

## 2. AIエージェントが最初に読む資料

資料生成前に、次を順に確認する。

1. `docs/research/研究概要・現状.md`
2. `docs/rte_compiled_cost_validation_summary.md`
3. `docs/random_circuit_cost_validation.md`
4. `docs/rte_boundary_cost_validation.md`
5. `docs/rte_boundary_pair_validation.md`
6. `docs/hierarchical_cost_validation.md`
7. `docs/rte_connected_cluster_cost_validation.md`
8. `docs/research/研究ノート/2026-08-25.md`
9. `docs/research/研究ノート/2026-08-26.md`
10. `VALIDATION_STATUS.md`と`artifacts/validation_manifest.json`

数値は`docs/rte_compiled_cost_validation_summary.md`から引用できるが、chartを作る場合は対応する
machine-readable artifactを直接読む。runtime cache、checkpoint、logはgitに含めない。

## 3. 資料全体のストーリー

```text
前資料で残った期待compiled costの問題
        ↓
event別コストの単純加算を検証
        ↓
48--58%過大評価：単純加算を棄却
        ↓
局所connected-cluster境界補正を導入
        ↓
K_Tay=0でpair/triple、same/different境界を検証
        ↓
K_Tay=2のrare eventを条件付きsamplingで再検証
        ↓
paired構造検証と独立holdoutを分離
        ↓
H4の条件移送、例外L_D=6の4-event補正、controlled qを確認
        ↓
H5の独立calibration/holdoutで系サイズ方向を確認
        ↓
条件ごとの1--3 event model＋不通過時4-eventという運用規則
```

## 4. セクション構成

| Section | Slide | 役割 |
|---|---:|---|
| A. 前資料からの接続 | 1--2 | 今回の問いと範囲を定める |
| B. 単純加算の検証 | 3--5 | 基準、検証設計、棄却結果を示す |
| C. connected-cluster model | 6--10 | 局所補正、K=0、rare order、paired検証を説明する |
| D. 運用holdout | 11--15 | H4移送、K4、controlled q、H5を示す |
| E. 結論 | 16 | 採用規則と次工程を示す |

## 5. スライド別の詳細構成

### Slide 1. タイトル

**タイトル**

> ランダムRTE回路のcompiled-cost近似モデルと検証

**副題**

> DF＋部分ランダムコスト評価に向けた続編

**入れる内容**

- 発表者名、日付。
- 小さく「最終総コストではなく、その入力となる回路コスト近似の検証」と入れる。

**見せ方**

- 左に長いランダムevent列、右に短い1--3 event窓から期待costを再構成する模式図。
- タイトルでは数値表を出さない。

**このスライドで言うこと**

- 前資料で次の課題としたランダム回路コスト推定について、検証結果と採用モデルを報告する。

---

### Slide 2. 前資料からの接続

**タイトル**

> 残っていた課題は、1回のランダム回路の期待compiled cost

**必須内容**

前資料の最終目的関数を再掲するが、説明は期待cost部分に絞る。

$$
G_{\mathrm{total}}
=\sum_m\sum_{b\in\{c,s\}}N_{m,b}\,\mathbb E[C_{m,b}^{\mathrm{no\text{-}prep}}]
$$

- PF係数$C$：指定条件で検証済み。
- finite RTE誤差上界：H4指定gridで検証済み。
- 今回：$\mathbb E[C_{m,b}]$を全trajectory compileなしで求める方法を検証。

**範囲外を下段に明記**

`shot数との接続・β/α配分・最終総コストは今回の結果ではない`

**参照元**

- `docs/research/DF+PRコスト評価.pdf` 13--15、20--27枚目。
- `docs/research/研究概要・現状.md` 3節。

---

### Slide 3. なぜ近似が必要か

**タイトル**

> 全trajectoryの一体compileは、event列が長くなると使えない

**必須式**

$$
\mathbb E[C_L]
=\sum_\omega p(\omega)C_{\mathrm{comp}}(\omega),
\qquad \omega=(e_1,\ldots,e_L)
$$

**説明**

- event種類数を$J$とすると列挙空間は概ね$J^L$。
- $C_{\mathrm{comp}}(e_1e_2)$は一般に$C_{\mathrm{comp}}(e_1)+C_{\mathrm{comp}}(e_2)$ではない。
- 基底変換、隣接回転、transpiler最適化がevent境界をまたいで作用する。

**図**

3方式を横並びにする。

1. 全列挙／full Monte Carlo：基準だが重い。
2. event単純加算：安いが境界効果を無視。
3. 局所窓model：短い回路だけcompileし、長い列を再構成。

**このスライドで言うこと**

- 目的はMonte Carloそのものをなくすことではなく、compile対象を短い局所回路に限定することである。

---

### Slide 4. 検証設計と判定基準

**タイトル**

> 局所係数のcalibrationと、未使用full回路のholdoutを分離する

**基本フロー**

`短い局所回路で較正 → 係数固定 → 別seedの長いfull回路で検証`

holdoutは「係数調整に使っていない未使用full回路」と注記する。

**H4共通条件**

- H4 chain、1.0 Å、STO-3G、DF rank 12。
- exact DF Hamiltonian snapshotを固定。
- Qiskit 1.3.0、`rz,sx,x,cx`、optimization level 1、seed 17、coupling mapなし。
- 主指標RZ count、補助5 metric。

**暫定判定**

- 全metric最大点誤差5%以下。
- RZ予測側95%相対半幅2%以下。
- 点wise 95%上側値は診断のみ。

**見せ方**

- 左に条件、中央にcalibration/holdout分離、右に判定指標。

---

### Slide 5. event単純加算の検証結果

**タイトル**

> eventを別々にcompileして足すと、48--58%過大評価した

**条件**

H4、rank 12、$L_D=3$、$\delta=0.1$、$K_{\mathrm{Tay}}=0$、$r=2$、同一DF表現、100 trajectory。

**結果**

- RTE event別compile和：全metricで48.30--57.58%過大評価。
- 別DF表現の300 trajectory replicate：52.99--61.15%。
- 長さ6の境界pilotでは単純和の最大誤差157.51%。
- forward deterministic half、RTE occurrence全体、reverse halfの3部分和：最大0.959%。

**推奨chart**

- 主chartはRZ count、CX count、total depthについて`full=1`、`event sum`を比率で表示。
- 小さなinsetでpartial-$S_2$ 3部分和が1%以内であることを示す。

**結論ボックス**

> event単純和は不採用。大きな差は主にRTE event間の境界を切ったことで生じる。

**参照artifact**

- `artifacts/random_circuit_cost_validation/h4_sto3g_d100_rank12_ld3_dt0p1_r2_k0_v1.json`
- `docs/random_circuit_cost_validation.md`

---

### Slide 6. connected-cluster境界補正

**タイトル**

> 短い局所回路から、境界相殺をconnected係数として取り込む

**必須式**

$$
\widehat C_L
=\sum_i\kappa_1(e_i)
+\sum_i\kappa_2(e_i,e_{i+1})
+\sum_i\kappa_3(e_i,e_{i+1},e_{i+2})
+\cdots
$$

$$
\kappa_2(e_1,e_2)=C(e_1,e_2)-C(e_1)-C(e_2)
$$

$$
\kappa_3(e_1,e_2,e_3)
=C(e_1,e_2,e_3)-C(e_1,e_2)-C(e_2,e_3)+C(e_2)
$$

**直感**

- $\kappa_1$：単event。
- $\kappa_2$：一境界で回収される相殺。
- $\kappa_3$：二境界を独立に足せない残差。
- $\kappa_4$：3-eventまでで不足した条件だけ追加。

**重要な表記**

スライドの隅に`Taylor cutoff = K_Tay / cluster係数 = κ1,κ2,…`を固定表示する。

---

### Slide 7. $K_{\mathrm{Tay}}=0$で必要な局所次数

**タイトル**

> count/sizeには3-event項、depthには2-event項で十分な傾向

**条件**

H4、rank 12、$L_D=3$、short-step 0.025、$K_{\mathrm{Tay}}=0$、同一compiler。

**結果**

- 独立2 seed・$L=4,6$：pair最大8.07--8.18%、triple最大2.33--3.73%。
- 未使用$L=8$・2000 trajectory：pairのcount/size最大8.851%、triple全metric最大1.744%。
- depthではpair-onlyの統計的不適合は確認されなかった。

**推奨chart**

- metricをcount/size群とdepth群に分け、pair/tripleの最大誤差を表示。
- 5%の水平線を引く。

**結論**

- count/sizeは$\kappa_3$まで。
- depthは$\kappa_2$まででも候補。
- この条件では$\kappa_4$を直ちに必要とする証拠はない。

---

### Slide 8. same/different-fragment層別

**タイトル**

> pair補正はsame境界が支配するが、different境界もゼロではない

**必須内容**

$$
p_{\mathrm{same}}=\sum_f p_f^2=0.7310604
$$

$$
\mu_2
=p_{\mathrm{same}}\mathbb E[B\mid\mathrm{same}]
+p_{\mathrm{different}}\mathbb E[B\mid\mathrm{different}]
$$

**結果**

- sameだけ補正：最大3.65%、最大z 2.59。
- same/different双方を解析確率で重み付け：最大0.849%、最大z 0.587。

**言い方の注意**

> same/different分類はpair補正と別の競合modelではなく、平均pair係数を解析確率から構成する内部分類。

**図**

same基底境界とdifferent基底境界を2色で描き、条件付き補正を確率重み付きで合成する。

---

### Slide 9. $K_{\mathrm{Tay}}=2$で旧IID検証が不足した理由

**タイトル**

> rareなorder-2 eventは、通常のIID samplingでは検証できていなかった

**必須数値**

$$
p_{2}=1.062925\times10^{-4}
$$

- 旧runの8000 event位置中、order 2は1回だけ。
- 旧$K_{\mathrm{Tay}}=2$結果は実質的にorder-0列を評価していたため、内部order-2の根拠から除外。

**再検証方法**

1. all-order-0、exactly-one-order-2、two-or-more-order-2を条件付き生成。
2. 各patternを個別に評価。
3. 解析的Taylor次数確率で再結合。

**見せ方**

- 左にIIDではrare eventがほぼ出ない図。
- 右にstratified samplingと解析重み付け。

**このスライドで言うこと**

- 標本数を単純に増やすより、rare stratumを直接sampleする必要があった。

---

### Slide 10. paired構造検証と独立運用推定を分ける

**タイトル**

> 同一trajectoryのpaired差分で、cluster打切り残差を直接測る

**比較**

| 方法 | 比較する量 | 最大点誤差 | 解釈 |
|---|---|---:|---|
| 独立係数差引き | 別標本の大きな平均$C_1,C_2,C_3,C_L$ | 9.119% | MC分散と構造残差を分離できない |
| paired局所窓 | 同一trajectoryの局所予測$-\,$full | 1.373% | 3-event打切りの構造残差を測る |

**補足数値**

- pairedのRZ 95%診断1.796%。
- 最大z 7.535なので、小さい高次残差は統計的に非ゼロ。

**重要な結論**

- pairedはmodel構造を検証する。
- 運用時はholdoutと独立した局所係数をpilot/Neyman配分で較正する。
- paired holdoutを使って係数を調整しない。

---

### Slide 11. 実際の運用推定器

**タイトル**

> calibration・transpileなしprediction・独立holdoutを分離した

**フロー**

1. exact DF Hamiltonian snapshotを固定。
2. order-0単eventを厳密列挙。
3. Taylor patternを条件付きsample。
4. same/different pairを層別。
5. pilot分散からNeyman配分でproduction標本数を決定。
6. 保存係数と解析確率だけで長い列を予測。
7. 代表full回路holdoutで検査。

**実装上の主張**

- predictionはQiskit回路構築・transpileを呼ばない。
- fixed chunk、incremental merge、adaptive allocation、SQLite cacheで再開可能。
- cache/checkpointはruntime stateで、科学的結果はJSONとsnapshotへ保存。

**再現単位の注意**

別processで名目上同じ分子からDFを再構築したholdoutは最大11.68%、z 5.215ずれたため結合しなかった。
分子条件だけでなく、exact DF snapshotとcompiler条件を固定する。

---

### Slide 12. H4の独立holdoutと条件移送

**タイトル**

> 1--3 event modelはH4の多くの条件で5%内、$L_D=6$だけ不通過

**主run**

H4、rank 12、$L_D=3$、short-step 0.025、$K_{\mathrm{Tay}}=2$、未使用$L=4,6,8$。

- 全metric最大点誤差2.936%。
- RZ最大z 2.074。
- 予測側95%半幅1.537%。

**移送grid表**

| $L_D$ | short-step | 最大点誤差 | 判定 |
|---:|---:|---:|:---:|
| 0 | 0.025 | 2.043% | pass |
| 3 | 0.020 | 2.140% | pass |
| 3 | 0.025 | 3.777% | pass |
| 3 | 0.030 | 3.541% | pass |
| 6 | 0.025 | 6.119% | fail |

**結論**

> 1--3 event modelを全条件へ無条件に移送せず、候補条件ごとにholdout判定する。

---

### Slide 13. 不通過条件の4-event補正

**タイトル**

> $L_D=6$だけ4-event項を追加し、5%点候補を回復した

**条件**

H4、rank 12、$L_D=6$、short-step 0.025、$K_{\mathrm{Tay}}=2$、未使用$L=8$。

**結果**

- 1--3 event transfer：6.119%でfail。
- 固定$\kappa_1$--$\kappa_4$独立評価：最大点誤差3.750%。
- 同一trajectoryのpaired構造検証：最大点誤差4.008%、RZ 95%診断4.902%。
- 独立係数評価のRZ 95%上側診断は7.045%で、統計的不確かさは残る。

**結論**

- 4-event項は構造上有効。
- 最初から全条件に追加せず、3-event holdout不通過時だけ使う。
- 4-event補正を厳密5%保証とは呼ばない。

**図**

`κ1–κ3: 6.119% → κ1–κ4: 約4%`のbefore/afterを大きく表示。

---

### Slide 14. controlled反復回路の$q$依存

**タイトル**

> controlled反復の期待costは、検証した$q$範囲でaffineに予測できた

**モデル**

$$
\widehat D_q=a+bq
$$

**条件**

H4、rank 12、$L_D=3$、$\delta=0.1$、$r=4$、$K_{\mathrm{Tay}}=0$、同一compiler。

**結果**

- $q=1,2$各300 trajectoryから未使用$q=4$：全metric最大0.307%。
- $q=1,2$各150 trajectoryから未使用$q=8$：全metric最大0.0529%。

**scope**

ordinary controlled time-evolution subcircuitのみ。ancilla Hadamard、軸変更、測定、状態準備、
量子shotは含めない。

**見せ方**

- 横軸$q$、縦軸compiled costの模式的直線。
- $q=1,2$をfit点、$q=4,8$をholdout点として色分け。

---

### Slide 15. H5の系サイズ方向・独立holdout

**タイトル**

> H5でも3-event modelが独立holdoutの5%点基準を通過した

**条件**

H5 chain、1.0 Å、STO-3G、10 qubit、project設定DF rank 9、$L_D=4$、short-step 0.025、
$K_{\mathrm{Tay}}=2$、$L=4,6,8$、同一Qiskit/compiler条件。

**左パネル：paired構造検証**

- all-order-0各長さ100、single-order-2各位置20。
- 1--3 event：最大点誤差1.665%。
- 1--4 event：最大点誤差0.551%。
- 5%を満たす最小cluster長として3-eventを選択。

**右パネル：独立calibration/holdout**

- calibration実現最大RZ相対SE：0.745%。
- holdout：all-order-0各長さ500、single-order-2各位置125。
- 最大点誤差3.776%。
- 最大予測側95%半幅1.459%。
- 最大z 2.009。
- 95%上側診断7.461%は硬い判定ではない。

**結論**

> H5のこのsnapshot・compiler条件では、独立運用推定器として1--3 event modelを受理する。

**参照artifact**

- `artifacts/rte_cost_system_size_h5/2026-08-25/outputs/h5_sto3g_d100_rank9_ld4_s0025_k2_paired_l4_l6_l8_zero100_single20_v1.json`
- `artifacts/rte_cost_system_size_h5_independent/2026-08-26/outputs/h5_sto3g_d100_rank9_ld4_s0025_k2_independent_l4_l6_l8_zero500_single125_v2.json`

---

### Slide 16. 現在の採用規則と次工程

**タイトル**

> 条件ごとに3-event modelを較正し、不通過時だけ4-eventへ進む

**採用フロー**

```text
exact snapshot・compiler条件を固定
        ↓
κ1–κ3を局所回路で較正
        ↓
代表full回路holdout
        ↓
5%点誤差・2%予測精度を通過？
   ├─ yes → 採用し、predictionはtranspileなし
   └─ no  → κ4または追加境界classを検証
```

**現在の結論**

- event単純加算は不採用。
- H4・H5の明示条件で局所connected-cluster modelを支持。
- 検証済み範囲のmodel uncertaintyとして約5%を持たせる。
- 候補cost差が約5%以下なら追加samplingまたは直接compileで確認。
- 回路コスト検証の広いpilotは一旦区切る。

**次工程**

- cost providerをRPE roundごとの$N_{m,b}$へ接続。
- $\beta$・$\alpha$配分を比較。
- 実際に必要なevent列長が8を超える場合のみ長さholdoutを追加。
- compiler、coupling map、backend、snapshotを変えるときは小規模再較正。

**最後の留保**

> 部分ランダム化の最終総コストおよび決定論PFに対する優位性はまだ評価していない。

## 6. 数値chart作成時のartifact対応

| Slide | 数値 | 主artifact／status |
|---:|---|---|
| 5 | event単純加算48--58%、3部分和0.959% | `artifacts/random_circuit_cost_validation/` |
| 7 | K=0 pair/triple、L8 | `artifacts/rte_boundary_cost_validation/`、`artifacts/hierarchical_cost_validation/` |
| 8 | same/different 0.849% | `artifacts/rte_boundary_pair_validation/` |
| 9--10 | K=2 rare order、9.119%、1.373% | `artifacts/rte_order_stratified_cost_validation/` |
| 12 | H4 operational/transfer | `artifacts/rte_connected_cluster_cost_validation/` |
| 13 | H4 K4 follow-up | `artifacts/rte_cost_followup_batch/2026-08-25/`、`artifacts/rte_cost_paired_k4_l8/2026-08-25/` |
| 14 | controlled q4/q8 | `artifacts/hierarchical_cost_validation/`、`artifacts/rte_cost_data_batch/2026-08-25/` |
| 15 | H5 paired/independent | `artifacts/rte_cost_system_size_h5/2026-08-25/`、`artifacts/rte_cost_system_size_h5_independent/2026-08-26/` |

## 7. 作成時に避ける表現

- 「ランダム回路コストを厳密に求められた」
- 「5%以内が保証された」
- 「H12でも同じ係数が使える」
- 「Monte Carlo誤差を排除した」
- 「large systemでfull回路をcompileする」
- 「z-scoreが大きいのでモデルは失敗」
- 「95%上側診断が5%を超えたので、事前基準を不通過」
- 「回路コストの検証から最終総コストが得られた」

代わりに、`指定条件のlocal holdout`、`点誤差基準`、`prediction precision`、`診断量`、
`条件ごとの再較正`、`最終総コストは未評価`という表現を用いる。
