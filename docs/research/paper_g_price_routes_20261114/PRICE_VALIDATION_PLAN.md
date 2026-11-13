# 最初に実行すべき有限価格検証の計画（未承認・未実行）

[REPORT](REPORT.md)の§4〜6と§8。

**前提**：いずれもユーザーの明示的な承認があって初めて実行する。compileはDの既存native・compile・receipt経路で行い、Cは受入・guard・model検査を担当する。新しいcompiler・builderは作らない。

**共通の条件**
- 価格量はP0・測定付きfull wrapperの全RZ数。X/Yは別々に取得する。
- compilerはQiskit1.3.0、basis rz/sx/x/cx、opt1、seed17、hash `1239304c…`。controlは`symmetric_directional`。
- scalarは総Tで一回。sidecarは固定JSONのassertionに束縛する。
- 既存receiptと完全に一致する点は再compileしない。

## 1. Full PFとTruncated PF p7の同条件比較（優先）

**目的**
- T2の結果から、同じtemplateでのTruncated p7とFull PFの合計固有値シフトの差は8e−6 Haにすぎない。比較は1shotあたりの価格で決まる。
- 第1水準のnominal Nは両者ともB=1で同一なので、G_paperの比は価格の比そのものになる。

**template**：S2 16h/7、Y4 16h/3、M5 16h（Aのshortlistと同じ幅で、T2でε_trot内）。

| template | 較正（定常幅） | 独立holdout | Full PF（p10） | Truncated p7 |
|---|---|---|---|---|
| S2 16h/7 | m4/q7、m5/q14 | m6/q28 | 3 round×XY＝6 wrapper | 6 wrapper |
| Y4 16h/3 | m4/q3、m5/q6 | m6/q12 | 6 | 6 |
| M5 16h | m4/q1（Full PFは保存directの受入を優先）、m5/q2 | m6/q4 | 4新規＋保存2件の受入 | 6 |

- Full PFは、Cの既存案（`full_pf_latest_price_guard_20261114/NEXT_PRICE_VALIDATION.md`）と同じで、最大16 wrapper。Truncated p7は18 wrapper。
- B0とB0Iの価格同値性の検査（p7 S2 m4、X/Y）として、2 wrapperを追加する。旧Track Aでは差0だったが、現在のprofileでは確認していない。
- **任意**：p8の同じ3 template（18 wrapper）。p7・p8・p10の3点で、価格のblock数依存（傾き`b_p`がp＋1に比例するか）を検証する。p8はoracleを使わないL1でも予算内になる。

**model と判定**
- 各（方式・template・軸）で`C=a+b·q`を2点からexactに解き（`price_model_validation.fit`）、m6のholdoutを**事前に宣言した**相対許容差で判定する（`validate`）。
- 許容差の値は実行前にユーザーが決める。案としては、比較したい価格比の差（構造proxyでは0.727対1）の1/3以下。
- holdoutに失敗した場合はrefitもfeatureの追加もせずSTOPし、判断を仰ぐ。
- 2点の較正と1点のholdoutだけでは、m7〜m12（最大q＝1792/768/256）への外挿は保証されない。外挿した値は`UNQUALIFIED_MODEL`のtierに残す。

**前提条件**：Dのtruncated native builder・receipt・sidecar（並行するD作業）。それまでTruncatedの取得は保留する。

## 2. Full-H κ2の新ZZ・group価格model

**段階**
1. **D P-1**（Dの`FINITE_SCIENCE_PLAN` P-1）：κ2 m3（q8/R71）、normal 1 path、X/Yで2 compile。新ZZが有効であること、旧RZZが残っていないことを確認する配線確認。期待価格ではない。
2. **D P-2とCの検査**
   - 単一group cell（T=0.1、q=1、R∈{566, 4528, 36223}）を`price_model_validation.record`のCALIBRATIONとして受け付ける（構成のroundではないcellとしてSOURCEを束縛する）。
   - κ2 m5/m6の全roundをHOLDOUTとする。
   - modelは`E[C]=a+b·s+c·(q−1)`（features：one、event_count、outer_boundary_count）。
   - finite cohortの平均は`FINITE_SAMPLE`（n_pathsとlawを付ける）として、law期待値と区別する。
   - 規模はDの案（例：group n=2、m5 n=2、m6 n=1で18 compile）。
3. **前提とする仮説（未検証）**
   - groupがi.i.d.であること、境界の効果が局所的であること、compilerのpassがgroupをまたいで効かないこと。
   - m7〜m12の直接compileはしない。これらのroundの価格は、holdoutが通っても`UNQUALIFIED_MODEL`の外挿scenarioとする。
   - 第1水準のG_paperはm10〜m12がほぼ全体を占めるので、この仮説の扱いが結論を左右する。
   - 旧Full-H価格は旧loweringのものなので、比較から外す（guardが拒否する）。

## 3. PR Pauli tailの接続要件

**最初の段階（D B0）**
- 内容：LD1/m0、T=0.1、qPF=1、normal 1 path、X/Yで2 compile。
- 入力：Aの辞書とpacket（`714247619a`、P0 `4220f300`）、λ_P、R、scalarの扱い（保存s_Rとcを一回）。
- 回路：DF headは既存native、Pauli tailは`append_pauli_event`＋D2を`controlled_tails`へ渡す。
- 価格の扱い：単一pathの価格であり、E[C]ではない。

**C側の受入**（[PAULI_BRIDGE](PAULI_BRIDGE.md)§1〜2で更新）
- `PR_PAULI_TAIL` routeを使う。loweringは`PAULI_D2_CENTRAL_ROTATION_CONTROL_9C4006BF`（Dの`8b28dce0`と同じ）。sidecarにrouteの明示が必要。
- physical IDはBのcanonical candidate ID。`split.tail_dictionary`（辞書のhash）はphysical contextで照合する。
  - 旧形式のphysical CIDには辞書が含まれず、DFとPauliで衝突し得る。
  - そのため旧形式IDは拒否し、canonicalの辞書identityで区別する。

**平均単価に資格を付けるのに必要なもの（B1以降）**
- 辞書のlaw（K2のnormal・order確率）での期待値。
- Pauli特有の特徴量（support重み、X/Y局所基底の数、product順と隣接label）の期待値と、それを裏付けるSOURCE。
- DF用のE[S]・Accounting・残差係数は流用しない。
- 必要なclassと有限数は、Aの記述子が出てから固定する。
- B1の目安：辞書ごとにfresh normal 2 path×X/Y、計8 compile。

## 4. PR DF tail（既存）

既存のC1 finite cohort、Cの構造kernel E[S]、DのC−S残差を、Cの`quote_model`（`STRUCTURAL_PLUS_RESIDUAL`）で結合する。全roundの残差modelとlawの資格は未取得で、第1水準ではtierが`UNQUALIFIED_MODEL`または`FINITE_COHORT_SCENARIO`になる。追加の取得計画は、既存のC/D計画に従う（本件では新規に提案しない）。

## 5. 承認が必要な範囲（まとめ）

（A11〜A13は§7に追加した。）

| 項目 | 新しいcompile・取得 | 前提 | 担当 |
|---|---|---|---|
| A1 Full PF定常幅 | 最大16 wrapper＋保存M5 m4（2件）の受入 | Cの既存案 | D（取得）、C（受入・検査） |
| A2 Truncated p7 | 18 wrapper＋B0/B0I同値の2 wrapper | Dのtruncated builder | D、C |
| A3 Truncated p8（任意） | 18 wrapper | A2 | D、C |
| A4 Full-H P-1 | 2 compile | Dの準備 | D |
| A5 Full-H P-2 | 最大18 compile程度（Dの案） | A4、許容差の宣言 | D、C |
| A6 Pauli B0 | 2 compile | Dのtail dispatch、Aの記述子 | D |
| A7 Pauli B1（任意） | 8 compile | A6 | D、C |
| A8 holdout許容差 | — | 比較したい差からユーザーが決定 | ユーザー |
| A9 比較policyの採択 | — | 各routeのSOURCEとreceipt | ユーザー |
| A10 G_paperの実計算 | 保存・新規価格と保存signalのlog Bを使う集計（新しい科学的集計） | A1〜A9 | B（またはC） |

**STOP条件**
- SOURCE、profile、route、scalar、作用gateの不一致。
- holdoutの許容差超過：refitしない。
- 欠測：0価格にしない。
- 継続不能な資源不足。

各段階の後はmandatory STOPとする。

## 6. 取得後にCが行う受入とmodel検証（Truncated p7・Full-H κ2・Pauli pilot）

**共通の受入条件**
- 価格recordは`FULL_WRAPPER_MEAN_COMPILED_PRICE`（P0、測定付きfull wrapper、X/Y別）。
- sidecarに必要なもの：
  - `implementation`（method、native・wrapper・scalarのSOURCE、lowering、scalar_quantity、`ONCE_AT_TOTAL_T`、profile、measurement）
  - `physical_context`＝`I.price_context(req, m)`
  - `profile_extension`
  - 固定JSONの`implementation_evidence`
- guard（`price_implementation_guard.check`）は、第1水準（`paper_cost`）でも第2水準（`compare`/`R.cost`）でも、価格の算術より前に通す。
- classification・expectation・cohort・n_pathsを、recordに明記する。classificationはBの`PRICE_KINDS`の名前を使う。
  - Dの`8b28dce0`のcallerの記録名（`DIRECT_DETERMINISTIC`／`EXACT_DETERMINISTIC_CIRCUIT`、`DIRECT_SINGLE_PATH_WIRING`）は、G会計で拒否される。
  - 実行前にDの側でそろえる（[PAULI_BRIDGE](PAULI_BRIDGE.md)§2.2）。Cでは読み替えない。
  - 決定論PFは`DIRECT_COMPILED_DETERMINISTIC`かつ`DETERMINISTIC`。
  - RTEを含む方式の単一path・有限cohortは`DIRECT_COMPILED_FINITE_SAMPLE`かつ`FINITE_COHORT_ONLY`。
  - law期待値として扱えるのは、`QUALIFIED_EXPECTATION`と`ideal_sampler_distance_qualified=True`の両方が外部で示された場合だけ。
- 較正点とholdout点は、`price_model_validation.record`で記録する。
  - 構成のroundではないcellも、SOURCE付きで扱える。
  - `fit`の前にfeatureとdesignを、`validate`の前に許容差を宣言する。
  - holdoutに失敗したらrefitしない。

| 方式 | 受け入れる価格 | route／policy項目 | 較正（fit） | holdout | 外挿の扱い |
|---|---|---|---|---|---|
| Full PF（p10） | 3 template（S2 16h/7、Y4 16h/3、M5 16h）のm4・m5・m6、X/Y。M5 m4は保存directの受入 | `FULL_PF`、lowering `NOT_APPLICABLE_NO_RTE` | (方式・template・軸)ごとに`C=a+b·q`を、m4・m5の2点からexactに解く | m6（同じtemplate・軸） | m7〜m12は`UNQUALIFIED_MODEL` |
| Truncated p7 | Full PFと同じ3 template・同じround・X/Y。B0/B0Iの同値性確認（p7 S2 m4、X/Y） | `TRUNCATED_PF`、`NOT_APPLICABLE_NO_RTE`、retained Hの束縛 | Full PFと同じdesign | m6 | 同上。m6でのp7/p10の比は直接の観測値として記録 |
| Full-H κ2 | P-1：配線確認だけで、価格には使わない。P-2：単一group cell（T=0.1、q=1、R∈{566, 4528, 36223}）、κ2 m5/m6のround | `FULL_H_RTE`、新ZZ（`CONTROLLED_ZZ_CX_CRZ_CX_V1`） | group cellをCALIBRATIONとし、`E[C]=a+b·s+c·(q−1)`（featureはone、event_count、outer_boundary_count） | m5/m6の全round（finite cohort平均、n_pathsとlaw付き） | m7〜m12は`UNQUALIFIED_MODEL`の外挿scenario。group i.i.d.・境界の局所性・compilerの非干渉は未検証の仮説 |
| Pauli pilot（B0） | LD1/m0、normal 1 path、X/Y | `PR_PAULI_TAIL`、`PAULI_D2_CENTRAL_ROTATION_CONTROL_9C4006BF`、canonical physical ID（Dの`legacy_physical_candidate`は置換が必要）、`pauli_requests`との一致 | 行わない（単一pathなので） | 行わない | 第1水準のtierは`FINITE_COHORT_SCENARIO`。平均単価の資格はB1以降（Pauli特有の特徴量はAの記述子が出てから決める） |

**比較の前提**：Full PFとTruncated p7は、同じ3 template・同じround・同じ軸・同じcompiler profileで取得する。片方だけのtemplateやroundは比較に使わない。

## 7. holdout許容差の選択肢（**採択しない**。A8でユーザーが決める）

| 案 | 判定式 | 適する対象 | 注意 |
|---|---|---|---|
| H-a 比較差の1/3 | `|C_obs−C_pred| ≤ r·C_pred`。rは、比較したい価格比の差の1/3（構造proxyではp7/p10≈0.727なので、r≈0.09） | Full PF対Truncated p7 | 比較の結論を変えない精度を目標にする。値は取得前に固定する |
| H-b 固定相対値 | r＝2%（厳しめ）、または5% | 決定論PF全般 | 経験上の目安であり、科学的な根拠は弱い |
| H-c 絶対値の下限付き | `|Δ| ≤ max(r·C_pred, n_abs)`（例：n_abs＝数十RZ） | 小さいqのcell | 2点のexact fitでは較正残差が0なので、残差に比例する許容差は使えない |
| H-d 標本誤差を含める | `|Δ| ≤ r·C_pred + z·SE_cohort`、または同時区間に予測値が含まれること | Full-H、Pauli（有限cohort） | interval contract（SE・coverage）を事前に宣言する。n=1〜2ではSEが定まらないので、判定は保留（UNKNOWN）になる |

**共通の規則**
- 許容差は、取得の前にSOURCE付きで宣言する（`tolerance={kind, value, source}`）。
- 失敗した場合はrefitもfeatureの追加もせず、STOPして判断を仰ぐ。
- 合格しても、modelの`qualified_for_candidate`は自動では昇格しない（外部の採択が必要）。

**承認範囲の追加**（詳細は[PAULI_BRIDGE](PAULI_BRIDGE.md)§5）
- A11：Bのguard再pin。
- A12：Pauli bridgeのB取込み。
- A13：新decoderのbridge（D1の採用後）。
- A6は、C側の受入準備が完了したので、残る前提はDのsidecar（emitter commitの固定）とassertionである。
