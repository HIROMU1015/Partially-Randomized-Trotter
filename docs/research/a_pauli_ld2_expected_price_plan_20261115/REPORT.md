# PR-Pauli LD2：全13 roundの固定shot総費用に向けた期待価格評価の計画

2026-11-15 JST。Codex Aの担当分（chat A1109が実施）。**計画のみ。** 新規Q3、モデル再fit、価格予測の新規算出、sampling、compile、E[C]・CI・Gの算出は、いずれも0件である。

**成果物**
- 本REPORT と [HANDOFF](HANDOFF.md)（統合担当・C2・B・D2向け）
- [AXIS_MAP.json](AXIS_MAP.json)・[AXIS_MAP.csv](AXIS_MAP.csv)：26軸の対応表（metadataのみ）
- 作成script [`scripts/build_pauli_ld2_axis_map.py`](../../../scripts/build_pauli_ld2_axis_map.py)：固定Git objectだけを読む。114 checkがすべてPASS。private QPYは読んでいない。

## 結論

1. **26軸の対応**
   - Q3に適合したLD2候補（元configuration `d7dd4f52`、decoder側のtotal view `285bf481`、配分 `PAULI_LD2_USER_SCALE_3O4_01`）の物理IDは、C対応表・価格receipt・m9予測のbindingと26/26で一致した。
   - 直接観測した有限pathがあるのはm0（配線用1 path）、m5（2 path）、m6（1 path）だけである。m9はC2の担当で、予測の固定に続き、計画の公開直前にraw・receiptの公開を確認した（Aは値を開いておらず、本計画では使っていない。比較は未公開）。残る9 round・18軸には直接観測がない。
   - 対象lawの期待価格として使える軸は、現時点で0である。
2. **総費用を決めるのはm10〜m12**
   - 固定配分で実行するPauli eventの96.9%、outer stepの77.6%が、m10〜m12にある。
   - 直接観測のあるm0・m5・m6は、event数の0.07%、step数の2.9%にすぎない。
3. **推奨する評価方法（1案）：step分解による線形評価**
   - 1 pathの価格は「stepごとの費用の和 ＋ 決定論的な境界と端」という構造を持つ（SOURCE）。
   - この構造の下では、期待総費用は凍結モデルの3係数の線形結合 `G = w·β` になる。`w = (756, 205,844, 471,030,556)`。
   - 凍結係数は再fitせずに点推定として使う。有限pathのばらつきはstep単位の分散で扱い、round間の移送は保存compiled QPYで構造を確認して扱う。
4. **26軸を個別に取得する必要はない。** 総費用は、3係数（または各Rのstep平均）から直接評価できる。m10〜m12を丸ごと取得する案は提案しない。
5. **最初の有限作業 S0：保存compiled QPYのstep分解監査。**
   - 新規path 0、compile 0、QPYの読取は最大52件（compiled 26・wrapper 26）。
   - m9の作業とは重複しない（m9の保存先・driver・公開値を扱わない）。
6. **path数と精度目標は、保存証拠だけでは決められない。**
   - 理由1：必要精度の基準になる「決定論PFの条件付き基準 593,555,352 RZとの差」は、LD2の総額を算出しないと分からない。今回は算出が禁止されている。
   - 理由2：step単位の分散は、保存receipt（path合計の値）からは推定できない。
   - 決めるための最小の先行作業は、S0と、その後のS1（凍結モデルによる条件付き総額と標準誤差。0-compile、別許可）である。

## 1. 固定入力

| 固定commit | 内容 | 本計画での使い方 |
|---|---|---|
| `0b4b2d941cd669aaa45231e1236bc2d3158a9d12` | C：5方式の候補対応表（ALIGNMENT） | LD2の26行（物理ID・q・R・信号packet） |
| `50be68b803a90fd893a1499ebce74e033225144d` | B：Pauli LD1・LD2のQ3（3/4配分） | 配分vector、元物理ID（`canonical_origin`）、対象law |
| `5dac74959ed9ea49465d393fc792ef5b4727edba` | D2：LD2較正 5 cell・18 compile | 較正cellのq・R・T、個別receipt |
| `b74e4e536392c2bbe94cd371436615fc56c0564f` | C2：軸別2モデルとm5/m6予測の凍結 | 係数・特徴量・適用範囲（変更しない） |
| `705f59ea863e1b9fe90e023f1cb1057d00366b16` | D2：m5/m6の独立holdout | 6 receiptと4件の記述差 |
| `2d513c2755cfd88cfcf30cba4352a6075d9a9c4b` | C2：m9有限検証の計画 | m9の位置付け。本branchの分岐点 |
| `ad6ab45bbcc7139bf499760d28efad794e4eae9b` | 統合ロードマップ | 影響報告の対象（本体は編集しない） |

補助として次も読んだ（いずれも公開済みの固定commit）。
- LD2配線W：raw `6c4709f85`、受入 `1e2639a1b`。
- m9予測の固定：C2 `3f6e6604a`（§3）。
- 回路構成のSOURCE：D2 `13ab1acfa`、native `c31ef4241`。
- 過去の設計：A `7a0c9468b`（B1-CAL・M-lin・`finite_price_design`）。
- PF閉包監査：D `606881b04`（規則とcodeの固定は `d3338c7ef`）。

## 2. 26軸の対応

### 2.1 roundごとの対応

X軸とY軸は同じ状態なので、1行で2軸を表す。全26行・全locatorは[AXIS_MAP.json](AXIS_MAP.json)にある。

| m | q | R | 1 pathのevent数 | N（X・Y各） | 直接観測した有限path | 同じstep lawの較正cell | 既存の凍結予測 | event数の割合 | step数の割合 |
|---:|---:|---:|---:|---:|---|---|---|---:|---:|
| 0 | 1 | 2 | 2 | 48 | 配線用1 path（RZ 2236） | なし | なし | 0.0000% | 0.05% |
| 1 | 2 | 3 | 6 | 45 | なし | なし | なし | 0.0001% | 0.09% |
| 2 | 4 | 5 | 20 | 42 | なし | なし | なし | 0.0004% | 0.16% |
| 3 | 8 | 9 | 72 | 39 | なし | なし | なし | 0.0012% | 0.30% |
| 4 | 16 | 17 | 272 | 36 | なし | なし | なし | 0.0042% | 0.56% |
| 5 | 32 | 33 | 1,056 | 33 | holdout 2 path（68,228・68,300） | なし | C2 `b74e` | 0.015% | 1.03% |
| 6 | 64 | 66 | 4,224 | 29 | holdout 1 path（149,435） | S66、Q8R66、P2R66 | C2 `b74e` | 0.052% | 1.80% |
| 7 | 128 | 132 | 16,896 | 26 | なし | なし | なし | 0.19% | 3.23% |
| 8 | 256 | 263 | 67,328 | 23 | なし | なし | なし | 0.66% | 5.72% |
| 9 | 512 | 526 | 269,312 | 19 | C2担当：予測固定済み。raw・receiptは公開済みだが本計画では未使用（§3） | S526 | C2 `3f6e` | 2.17% | 9.45% |
| 10 | 1024 | 1052 | 1,077,248 | 16 | なし | なし | なし | 7.32% | 15.92% |
| 11 | 2048 | 2103 | 4,306,944 | 13 | なし | なし | なし | 23.77% | 25.87% |
| 12 | 4096 | 4205 | 17,223,680 | 9 | なし | S4205 | なし | 65.82% | 35.82% |

- 有限pathのRZ値は、公開receiptからの転記である（X=Y）。新しい平均・差は計算していない。m5・m6の記述差（+0.044612%、+0.157666%）は[705f HANDOFF](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/705f59ea863e1b9fe90e023f1cb1057d00366b16/docs/research/d_pauli_ld2_b1_holdout_results_20261115/HANDOFF_C2.md)の値である。
- 割合の分母は、配分全体（X・Yの両軸）で、event数 471,030,556、step数 205,844、shot数 756。これはshot数・q・Rだけから決まるmetadataで、価格ではない。
- 較正cell（m=None）は、round軸の価格ではない。表の「同じstep law」は、1 stepの中身が同じ確率法則に従うという意味である（§2.2）。

### 2.2 確認できた事実

| 項目 | 結果 | 根拠 |
|---|---|---|
| 物理IDの一致 | C対応表、Q3入力の`canonical_origin`、holdout receipt（m5・m6）、W（m0）、m9 bindingで26/26一致 | AXIS_MAPのcheck |
| 配分 | Q3の`N_X = N_Y = [48,45,42,39,36,33,29,26,23,19,16,13,9]`が指示と一致 | 同上 |
| 対象law | 13 roundとも `ideal_finite_RTE_law`、K=2 | 同上 |
| 1 stepの時間幅 | 13 round、較正5 cell、凍結予測4件のすべてで、`float(T)/q` がbinary64の0.1と完全に一致 | 同上 |
| 同じstep law | eventのlawは `finite_rte_distribution(λ_P·step_time/R, K=2)` で決まる。step幅が同一なので、Rが同じなら同じlaw。S66≡m6、S526≡m9、S4205≡m12 | SOURCE `pauli_tail_price_connection.sample_steps`（13ab） |
| X/Y | 保存済みの13 path（W 1、較正 9、holdout 3）すべてでX=Y。wrapperの差は、最後のhの前のsdgだけ | 公開receipt、SOURCE `build_native_hadamard_wrapper` |
| 凍結係数 | X・Yの係数は同一（保存値）。特徴量は`[1, q, qR]` | `b74e` FROZEN_MODELS |

**現在の適用範囲と、各roundの位置**
- 凍結モデルの適用範囲は「較正とm5/m6の宣言済み予測だけ」である（`application_scope`）。m9はC2が別許可で予測を追加した。
- 26軸すべてへの適用は、新しい許可を要する範囲の拡張になる。
- 較正cellのRは66・526・4205、qは1・2・8。
  - m0〜m5（R＝2〜33）は、Rが較正範囲より小さい。
  - m7〜m12は、qが本計画で使った観測（最大64）を超える。

### 2.3 三つの不確かさを区別する

| 量 | 意味 | 現在の証拠 | 何で決まるか |
|---|---|---|---|
| 少数pathでの予測誤差 | 凍結予測Pと、少数pathの観測平均Oの差 `d = P − O` | m5：+0.044612%（2 path）、m6：+0.157666%（1 path） | モデルの偏り ＋ path間の変動（分散 `q·σ²(R)/n`） |
| 母平均の推定精度 | 対象lawの平均 E[C]（または総額G）の推定値の不確かさ | 未評価。凍結時の`coefficient_uncertainty`はnull。各軸5 cellのfitで残差の自由度は2しかなく、X/Yは同じpath、P2R66はS66に依存する | 較正cohortの標本誤差（stepの分散と設計） ＋ モデル形の誤り |
| 実験1回の実現総費用のばらつき | 756 shotそれぞれが別のpathを引くことによる、実際の総費用の揺れ | 未評価 | `Σ 2·N_m·q_m·σ²(R_m)`。推定誤差ではない |

- 小さい記述差は、偏りが小さいことも、分散が小さいことも保証しない。
- Pauli用の数値許容差は未採択である。決定論PFの2%は移さない。

### 2.4 対象lawの期待価格として使える範囲

- **現時点：0軸。** すべての価格recordは `law_expectation=false`、`qualified_for_candidate=false` である。
- 有限pathの観測は、それぞれのcohortの有限価格にとどまる。m0のWは配線専用で、較正にもholdoutにも使わない。
- §4の方法でS0・S1を通り、C2が適用範囲の拡張を採択した場合は、26軸に**条件付きモデル価格**を付けられる。決定論PFの条件付き閉包価格と同じ位置付けで、標準誤差と構造上の条件を添えたものになる。
- 統計的な信頼区間付きのE[C]とするには、別途、精度・信頼水準の契約が必要である。
- m0〜m5（Rが較正範囲より小さい）の値は、Rの外挿であることを明記する。ただし総額への重みは、event数で0.02%、step数で2.2%である。

## 3. m9の状態（固定成果を一度確認）

C2 branch `codex/pauli-ld2-price-freeze-20261115` を、作業開始時（21:10 JST頃）に一度確認した。そのときのheadは `3f6e6604a`（20:24 JST）で、raw・receipt・比較はなかった。計画はこの状態を前提に作成した。

計画の公開直前（§10.5の再確認）に、C2のheadが `c9986c82f`（21:29 JST）へ進んでいた。新しいcommitは**件名とfile名だけ**確認し、観測値は開いていない。

| 段階 | 状態 | 根拠 |
|---|---|---|
| 予測の固定 | **確認済み** | `3f6e6604a` の `PREDICTIONS.json`：`PREDICTIONS_FIXED_BEFORE_SAMPLING`。X・Yの2予測、cohort `D_PAULI_LD2_M9_LONG_NORMAL_N2_V1`、n=2、compileは最大4 |
| 取得の開始 | **確認済み**（公開直前の再確認で） | 作業開始時は固定記録になく、同じユーザーのm9 driverのprocessが動いているのを観測しただけだった（操作も閲覧もしていない）。その後、rawの公開で取得の実施を確認した |
| 観測結果の公開 | **raw・receiptは公開済み、比較は未公開** | `a7fdd200d`：2 pathのrawと4 compile receipt。`c9986c82f`：canonical価格の束縛。予測との記述比較・HANDOFFのfileはまだない |

- Aは、m9の予測固定・sampling・compileを引き取らない。
- 公開済みの観測値も開いておらず、本計画の数値・判断には使っていない。m9を使う後続作業（S0+、S1）は、C2が比較を公開してSTOPした後に、その固定commitを入力として指定する。

**m9の観測で解消できる不足**
- 凍結モデルとq=512（本計画で使った観測の最大q=64の8倍）との初めての独立比較。step lawが較正cell S526と同じ（R=526）なので、「q方向の移送」だけを切り分けて見られる。
- 公開後にS0と同じ方法で分解すれば、R=526のstepを1,024個（2 path×512）得られる。これは較正の2 stepを大きく上回る。ただし、これはC2の範囲外の追加作業で、別許可を要する。

**観測後も残る不足**
- m10〜m12（q=1024〜4096）への移送。event数の96.9%を占める。
- 対象lawの母平均の精度（stepの分散と係数の標準誤差）。
- R=1052・2103（m10・m11）には、同じRの較正cellがない。
- order2のproduct・非自明な位相（UNKNOWN）。
- 理想lawと実sampler（PCG64・CDF）の差。
- 許容差が未採択であること。
- LD1。

## 4. 推奨する評価方法：step分解による線形評価

### 4.1 求める量

```
G_LD2 = Σ_{m=0}^{12} N_m · ( E_L[C_X(m)] + E_L[C_Y(m)] )
```

- **L**：Q3と同じ `ideal_finite_RTE_law`。
  - LD2辞書（184項、SHA-256 `dd986e52…`）、K=2、step幅0.1。
  - 1 stepはR_m個のeventをi.i.d.に引き、q_m step続く。scalarは総Tで一回。
  - Q3の模型（`IID_SHOTS_INDEPENDENT_AXES_ROUNDS_AND_REPETITIONS`）では、各shotが対象lawからi.i.d.に引かれる。したがって1 shotの期待費用はE_L[C]で、固定shot総費用の期待値は上の式になる。
- **C_a**：測定付きの全制御Hadamard wrapper（軸a）のcompiled RZ数。
  - compiler profile `1239304c…`、Qiskit 1.3.0、opt1、seed 17。
  - 状態準備は含めない。
- 保存価格は実sampler（PCG64・float64 CDF・hash seed）で引いたpathなので、推定するのは実sampler lawの平均である。理想lawとの差は、未資格のまま別項目として残す（`ideal_sampler_distance_qualified=false`）。

### 4.2 回路・samplingの構造（SOURCE）と分解式

**1 pathの回路**（`pauli_tail_price_connection.compose` → `ax2b_native_df_v5.build_partial_native`）
- 各outer stepは、次の順に並ぶ：U(one-body)・U(f0)・U(f1)・[R個のPauli event。D2で中心回転だけを制御]・D(f1)・D(f0)・D(one-body)。
- 最後のD(one-body)は、全system wireに `cx(anc,k)·rz·cx(anc,k)` を置く（`append_directional_diagonal`）。決定論PFの監査では、同じ形のblockがcompile後も全wireに残ることを観測した（`606881b04` A5）。それが回路を切ることは、passの局所性からの論証である。
- 各stepのeventは、stepごとに独立なseedで、同じlawから引かれる（`sample_steps`）。
- 最後にancillaで `p(−scalar·T)` を一回適用し、wrapper（h…[sdg]…h・測定）で囲む。
- compile設定はPF監査と同じprofileである。Qiskit 1.3.0のopt1（coupling mapなし）は局所的な書換えしか行わず、translation以後にCXを新しく作らない（`dd435d07e`・`606881b04`の解析）。

**仮説 H_loc**：compile後の回路はstep境界で切れる。境界付近のtoken列は、stepにもpathにもよらず同一である。境界付近は決定論的なhead block（D(f0)・D(one-body)・U(one-body)・U(f0)）で、乱択されたtailから離れている。

**H_locの下での分解**

```
C_a = E_a(T) + Σ_{i=1}^{q} X_i + (q−1)·B
E[C_a(m)] = (E_a(T_m) − B) + q_m·(μ(R_m) + B)
Var[C_a(m)] = q_m·σ²(R_m)
```

- `X_i`：stepの区間のRZ数。i.i.d.で、そのlawはRだけで決まる。
- `B`：境界の定数。
- `E_a`：wrapper・端・最後のancilla runの定数。

**μ(R) の形**
- order0のeventでは、隣接eventとの相殺（wire上の1q runの融合など）は定常的に起きる。そのため `μ(R) = α + β·R` に、R が小さいときだけ効く端の補正が加わる形になると考えられる。これは論証であり、S0でR=33・66と526・4205を並べて確かめる。
- order2のeventの出現率は、lawの引数 `λ_P·0.1/R` を通じてRに依存する。その追加費用は未観測なので、UNKNOWNの項として残す。

**凍結モデルとの対応**
- 上の式は、凍結モデルの形 `β0 + β1·q + β2·qR` そのものである（`β0 = E_a − B`、`β1 = α + B`、`β2 = β`）。
- X・Yの係数が同一なので、総額は次の線形結合になる：

```
G = w·β,  w = 2·(ΣN_m, ΣN_m·q_m, ΣN_m·q_m·R_m) = (756, 205,844, 471,030,556)
```

**感度の目安（価格の算出ではない）**：保存係数の比 β1/β2 は約302 で、1 stepのevent以外の費用はevent約302個分にあたる。一方、配分全体の平均は1 stepあたりevent約2,288個である。したがって、総額の不確かさは主にβ2（1 eventあたりの平均費用）と、m10〜m12への移送で決まる。

### 4.3 既存成果から再利用するもの

| 再利用するもの | 使い方 | 変更 |
|---|---|---|
| 凍結係数・domain・特徴量 `PR_STEP_V1`（`b74e`） | 点推定 `w·β̂`、26軸の条件付きモデル価格 | 再fitしない。適用範囲の拡張は別許可 |
| 較正5 cell・18 receipt（`5dac`） | 係数の根拠。S0では同じstep lawの標本として使う | 用途を保つ。P2R66は親に依存する |
| holdoutの記述差（`705f`）、m9（C2、公開後） | 独立な整合確認。path変動の尺度と比べて読む | 許容差は設けない。旧比較を書き換えない |
| `finite_price_design.coefficient_covariance`・`prediction_se`（`7a0c9468b`） | `Cov(β̂)` から総額の標準誤差 `sqrt(wᵀ·Cov·w)` を出す。paired依存も扱える | σ_eはS0で推定する（設計時の人工値を使わない） |
| PF閉包監査のtoken化・並列runner（`scripts/pf_qpy_closure_audit.py`、`d3338c7ef`） | S0のwire token、hex符号化、段階log | 周期抽出は使わない（stepは乱択） |
| A設計 M-lin（`7a0c9468b` A2 §8） | 仮定の一覧（境界・headの加法性、event費用の定常性） | 今回、これをS0の検査項目へ具体化した |

再利用しないもの：LD1の辞書・価格、PFの2%、B0・Wを較正標本として使うこと、P2R66を独立標本として数えること。

### 4.4 有限pathのばらつきの扱い

- **推定精度（母平均）**：`Ĝ = w·β̂` の標本誤差を、`Cov(β̂)` から求める。
  - Covの計算には、既存の設計補助関数を実際の5 cellに当て、S0で推定した `σ_e² ≈ σ²(R)/R` を入れる。
  - 結果は「モデルに基づく標準誤差」であり、統計契約が採択されるまでCIとは呼ばない。
- **少数pathの比較**：m5・m6・m9の記述差dは、path変動の尺度 `sqrt(q·σ²(R)/n)` と並べて読む。
  - H_locの下で σ²(R) がRに比例するなら、1 pathの相対的なばらつきは `(qR)^(−1/2)` の速さで小さくなる。そのため、長いroundの差は主にモデルの偏りを表す。
- **実現総費用のばらつき**：`Σ 2·N_m·q_m·σ²(R_m)` を別欄に出す。推定誤差と混同しない。

### 4.5 round間の移送のずれの扱い

| 方向 | 確認の手段 | 残る仮定 |
|---|---|---|
| q方向 | S0：保存path（q=1, 2, 8, 32, 64）でH_locを検査（step境界の切れ目、境界の決定性、合成の恒等性）。m9（q=512）はC2の比較で確認する | q>512（m10〜m12）は論証による。installされたRust binaryがtagのsourceと同じ挙動をすること、規模に依存する分岐がないこと、未観測の回路そのものは見ていないこと（PF閉包と同じ） |
| R方向 | S0：R=33（64 step）・66（82 step）のstep平均と、R=526・4205（各2 step）の単一step cellを記述的に並べる | m10・m11（R=1052・2103）には同じRのcellがない。Rについて線形であることは、i.i.d.のeventと局所的な相殺から導く |
| law | — | order2・非自明な位相の寄与はUNKNOWN。実sampler lawの資格もない。いずれも今回は追加調査しない |

数値の許容差は置かない。採否の基準は、PFの閉包の場合と同様に、C2（とユーザー）が決める。

### 4.6 全軸を個別に取得する必要はあるか

- **X/Y**：同じpathを2通りに測るだけで、wrapperの差は最後のancilla runの直前のsdgだけである。保存13 pathすべてでRZが同じだった。S0のP5で、全13 roundのTについて最後のrunの形が保たれることを確かめれば、軸ごとに取得する必要はない。
- **round**：H_locの下では、各roundの平均は「R_mでのstep law」と「q_m」だけで決まる。roundを丸ごと取得しなくても平均は求まる。
  - 1 pathのcompile負荷はq·Rに比例して大きくなる。m12では1 pathが17.2M eventになる。
  - m6の1 path（4,224 event）はX/Y込みでwall 4:35（`705f` REPORT、import・gateを含む）だった。m9の資源実測は本計画では参照していない。
- **総費用を直接評価できるか**：できる。`G = w·β`（またはRごとのstep平均の重み付き和）を、その標準誤差とともに直接評価できる。26軸の条件付き価格は、同じ係数から副産物として得られ、Cの価格表の形式に合わせるために使う。
- **精度が足りない場合の追加**：event数の96.9%を担うR ∈ {1052, 2103, 4205} の**単一step cell**を追加する。1 pathの負荷はqRではなくRに比例する。
  - 実測の目安：S4205 cell（2 path・X/Y）のwallは2:34.53（`5dac` REPORT、setup込み）。

### 4.7 比較基準と精度目標

- **基準**：決定論PFの条件付き基準 593,555,352 RZ（B0I p7 S2、B2 `1002f1dfe`）。
  - 156軸のうち60軸は閉包からの導出値で、PFの配分は `[59,…,11]`。
  - この基準自体にも統計的な誤差幅はない。
- **判断に必要な精度**：「`|Ĝ − 593,555,352|` が、標準誤差と移送のずれの尺度より十分大きいか」で決まる。Ĝ が分からないと決まらないので、S1の後に決める。
- **追加pathの数の式**：S2を行う場合、各R_jにn_j pathの単一step cellを加えた設計で、`sqrt(wᵀ·Cov(β̂)·w) ≤ ε·|Ĝ − 基準|` を満たす最小のn_jを選ぶ。ε（または信頼水準）はユーザーが決める。

### 4.8 採らなかった案

| 案 | 採らない理由 |
|---|---|
| 26軸をn pathずつ直接取得する | m10〜m12の1 pathは1.1M〜17.2M event。X/Yは重複になる |
| 長いroundのholdout（m10など）を増やす | 1 pathの負荷が大きいわりに、得られるのは1点の記述差だけ。m9が先行している |
| 構造kernel（辞書の確率と相殺規則から解析的に求める。A設計のM-struct） | Pauli用kernelは未実装で、compilerの規則との対応検証が別に必要。S0の結果次第で将来の候補 |

## 5. 最初の有限作業 S0：保存compiled QPYのstep分解監査

| 項目 | 内容 |
|---|---|
| 対象 | LD2の保存13 path × X/Y。W m0（1 path）、較正 S66・S526・S4205・Q8R66（各2）、P2R66（親に依存する1 path）、holdout m5（2）・m6（1） |
| round・軸 | 直接対応するのはm0・m5・m6のX/Y。較正cellはR=66・526・4205のstep標本（m6・m9・m12と同じstep law） |
| path数 | 新規0（保存path 13） |
| compile・sampling | 0・0。再fit・予測・E[C]も0 |
| 読取 | compiled QPY 26件＋wrapper QPY 26件＝最大52件（D2のprivate cache、読取専用、計約50 MB） |
| m9との重複 | なし。C2のm9保存先（`pauli-ld2-m9-longpoint-20261115-private`）、driver、予測、公開済みのm9 raw・receiptには触れない |

**規則の固定**：区切り規則・token定義・判定条件は、合成回路の人工testで確認してから、実QPYを開く前にcommitで固定する。PF監査（`d3338c7ef`）と同じ手順である。

| 検査 | 内容 | 判定 |
|---|---|---|
| P1 束縛 | receiptのfingerprint・6指標・cache JSONとQPYの対応。tokenから数えたRZ・CX・命令数が保存値と一致すること | RZが一致しなければSTOP（toolの不具合） |
| P2 step区切り | 固定した規則で、各pathにq−1個のstep境界を全wireで一貫して検出する。候補：ancilla wire上に決定論的なhead block（D(f0)・D(one-body)）が残すCX列の並びを境界とし、各system wireは同じCXの位置で切る。tail内の制御回転も同じ形のCX対を作るので、単独のCX対では判定しない | 検出できなければFAIL（規則は緩めない） |
| P3 境界の決定性 | 境界付近のtoken列（parameterのhexまで）が、全境界・全pathで同一であること | 一致しなければFAIL |
| P4 合成の恒等性 | P2R66の2つのstep区間が、S66のpath0・path1の本体と、宣言した端を除いてtoken単位で一致すること | 雑音のない加法性の直接検査 |
| P5 X/Yと終端run | X/Yの差が最後のancilla runだけであること。runが `rz·sx·rz` の形であること。13 roundのTすべてで終端角の余裕が1e−6以上あること（PFのA4と同じ基準） | 余裕不足はUNKNOWNとする |
| P6 step区間の表 | 各pathのstep区間ごとのRZ数と、端のRZ数（整数）。Rごとの件数・平均・分散（記述統計） | 判定ではなく出力 |

**Rごとの標本数**（metadataから確定。独立step数）
- R=2：1
- R=33：64
- R=66：82（P2R66の2 stepはS66と重複するので除く）
- R=526：2（m9の公開後は+1,024）
- R=4205：2

**得られる結論**
- 凍結モデルの形 `a + q·(h+α) + β·qR` が、保存回路の上で構造的に成り立つか（H_loc）。PF閉包監査に相当する。
- stepの分散 σ²(R)（R=33・66では十分な標本数、526・4205では各2）。これにより、H_locの下で各roundのpath変動の尺度が分かり、S1の標準誤差の入力になる。
- S1の差と合わせて、S2（追加sampling）が必要かどうか。

**得られないもの**
- E[C]、CI、G、モデルのPASS/FAIL、許容差。
- q>64への延長の観測（論証にとどまる）。
- order2・位相。

**STOP条件**
- Qiskitが1.3.0でない、またはQPYを読めない → STOP。
- tokenのRZ数とreceiptが一致しない → STOP。
- private fileへの書込みが必要になる → 禁止。
- 閾値の変更、再compile、retryはしない。
- 公開・remote照合の後、mandatory STOP。

**資源・並列化**
- PF監査のrunnerの形をそのまま使う：pathごとのprocess並列、`PROGRESS.jsonl`、子processの終了code、定期的なframe dumpは使わない。
- compiled fileで最大なのはm6（size 328,666）。PF監査では5.75M命令のfileを65.6秒で処理したので、S0は数分以内の見込みである（wrapper fileの命令数は未計測）。まず最小のfileで計測してから並列数を決める。

**実行担当**：指示で指定する。読取専用なので、D2の取得系列を止めずに実行できる。出力は新しいpublic directoryだけで、D2・C2のbranchやprivate directoryには書き込まない。

### 5.1 後続（それぞれ別許可。S0の結果で内容を確定する）

| 段階 | 内容 | compile | 担当の候補 |
|---|---|---:|---|
| S0+（任意） | C2がm9の比較を公開し、STOPした後に、m9の2 path×X/YのQPYを同じ規則で分解する（読取は最大8件） | 0 | S0と同じ |
| S1 | 凍結係数のまま、26軸の条件付きモデル価格・固定配分総額・`sqrt(wᵀ·Cov·w)`・基準593,555,352との差を出す（`law_expectation=false`） | 0 | 価格はC2、総額はB |
| S2（必要な場合だけ） | R ∈ {1052, 2103, 4205} の単一step cell。n_jは§4.7の式で決める | 2·Σn_j | D2 |

## 6. §10.5で照合する対象（S0〜S2の実行前）

1. C2 `codex/pauli-ld2-price-freeze-20261115`（計画公開時のhead `c9986c82f`）
   - headの前進。step区間やセグメントの監査が既にないか。m9の比較・HANDOFF・STOPの状態。
   - m9のprivate namespaceは読まない。m9の公開値は、S0+・S1の入力として指定されたときだけ使う。
2. D2 `codex/d-pauli-b1-caller-prep-20261114`（計画時のhead `705f59ea8`）
   - 13 pathのreceiptのblobが変わっていないこと。
   - private cache（`d-pauli-ld2-b1-{cal,holdout,w}-20261115-private`）のfile一覧とsha256がreceiptと一致すること。
   - SCOPE_DEVIATIONが保全されていること。
3. D `codex/d-price-records-c-bridge-20261114`
   - PF監査のcode（`d3338c7ef`・`606881b04`）を流用元として確認する。Pauliの監査はない。
4. B・B2（`codex/rpe-total-dp-acceleration-20261114`、`codex/b2-pf6-allround-cost-20261115`）
   - Pauliの総額が未算出であること。基準 `1002f1dfe`。
5. C `codex/m5-m4-receipt-admission-20261114`
   - Pauliの価格表や採択がないこと。
6. 統合ロードマップの最新
   - §8.2「RTE価格の26軸評価計画」の割当（Full-Hは別契約）。
7. A `1adb7e65a`、`7a0c9468b`
   - 設計（M-lin・M-struct・B1-CAL）。
8. Full-H系列（`d-full-h-*`）
   - paired合成（P2R566）の加法性検査のcodeが実装されていれば流用する。計画時に関連文書を検索した範囲では、境界加法性は未確認の課題として記載されているだけだった（codeは未確認）。
9. 同じユーザーのprocess
   - C2のjobには触れない。同じS0のjobがないこと。出力namespaceが未使用であること。

照合の単位は「path × 軸 × QPY」とする。同等の監査結果があれば再利用し、出所の不明なものはその分をSTOPする（§10.5の表3）。

## 7. 今回行っていないこと・保全

- 新規Q3、再fit、価格予測の新規算出、sampling、compile、E[C]・CI・Gの算出は0件。private QPYも読んでいない。
- 旧SCOPE_DEVIATION、固定SOURCE、原結果、旧STOP、凍結JSON、C2・D2・AのPauli・PF branchは変更していない。
- 統合ロードマップ本体は編集していない。影響は[HANDOFF](HANDOFF.md)に記載した。
- 未観測のorder2・非自明な位相はUNKNOWNのまま。今回の追加調査の対象にしていない。
- 決定論PFの593,555,352 RZは、条件付きの比較基準としてだけ扱う。

**mandatory STOP。** S0の実行、適用範囲の拡張、S1・S2は、それぞれ別の判断と指示に委ねる。
