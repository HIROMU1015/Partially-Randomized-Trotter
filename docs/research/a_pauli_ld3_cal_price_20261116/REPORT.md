# PR-Pauli LD3 段階3：較正cellの取得と、同じVPでのLD2との費用比較

2026-11-16 JST。Codex Aの担当分（chat A1109が実施）。計画 [`a50dce8f1`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a50dce8f1cbb38b2914bb2fb4ff79bab35e6e8f6/docs/research/a_pauli_ld_split_plan_20261116/REPORT.md) の段階3。

## 冒頭の答え

| 問い | 答え |
|---|---|
| 較正cell | Q8R660（q＝8、R＝660、T＝0.8）を独立2 path × X/Yで取得。**4 compile**、全gate PASS。compiled RZは path0 57,379、path1 57,355（X＝Y） |
| 価格規則 | **監査が支持した。** S0-r1の手順で区切り・入口・末尾・終端・X/Y一致・13 roundの終端runの余裕がすべて成り立ち、2 path × X/Yで分解が厳密に一致した |
| LD3の参照価格 | `C_ref(m,a) = 313 + 2685·q_m + (7793/1155)·q_m·R_m`。E3＝6.7472 RZ/event |
| 同じVPでの総額 | **LD3：1,051,862,454.81 RZ** 対 LD2（`a7eceecba`）：3,378,503,016.52 RZ |
| 差額・比 | **−2,326,640,561.71 RZ、LD3/LD2＝0.3113**（約69%減） |

**意味すること**
- fragment 2をheadへ移すと、1 stepの決定論部分は785 RZ増えた（H：1,900 → 2,685）。
- 一方、tail event数は0.157倍になった（段階1）。
- 同じVPでは、eventの減少がheadの増加を大きく上回る。
- BのQ3では、LD3はこのVPで採択済みの閾値に適合している（`2de33a4f4`、MSE上界6.2062×10⁻⁷）。したがって、LD3はLD2と同じshot数で、約3分の1の参照費用で適合する。

**注意**
- これはモデル参照値で、E[C]・CI・G・winnerではない。
- 決定論PFの2基準（`a7eceecba` の固定総額）と比べると、LD3の総額はB0I p7 S2（593,555,352）の1.77倍、Full PF S2（838,693,580）の1.25倍で、まだ大きい。
  - 配分はVPのままで、最適化していない。
  - 方式の優劣は判定していない。

## 1. 取得（[run01](run01/)）

| 項目 | 内容 |
|---|---|
| cell ID | `PAULI_PRICE_CELL:9cd5c434acf2eebf43f2f8795db4725a3db4a940a9c1c4f9112d6ab8f8946eb8` |
| m12のID | `RMSE:95014c1c…`（q＝4096、1 stepあたりのR＝660）。cellとは別IDで、cellのreceiptはround価格のreceiptではない |
| cohort・seed | `A_PAULI_LD3_CAL_Q8R660_N2_V1`、16個。既存のLD1・LD2の1,325個と重複なし（[BINDING.json](BINDING.json)） |
| 科学SOURCE | D2の `13ab1acfa`（LD2 CAL・C2 m9と同じ）を専用checkoutから無変更で使用 |
| driver | 凍結 `158bbcc7b`、binding `3074433d1`。C2 m9のworker（`cb034e547`）の観測・高速化をpinned blobから再利用 |
| 実行 | 2 path process並列、wall 1分31秒、peak RSS約0.89 GB、exit 0/0。retry・追加compileは0 |

**path別の結果**

| path | X RZ | Y RZ | CX | 数値gate（最大） | event cache |
|---|---:|---:|---:|---|---|
| 0 | 57,379 | 57,379 | 60,536 | 9.5×10⁻¹² | 159 miss / 5,121 hit |
| 1 | 57,355 | 57,355 | 60,624 | 9.4×10⁻¹² | 168 miss / 5,112 hit |

- 数値gateは、低制御・高制御漏れ・独立full-space vector・Hadamard X/Y・compiled vectorの7種（閾値1e−9、compiledは1e−7）。
- event orderは全てorder 0（order 2の確率は7.6×10⁻⁹）。

**実行前に確認したこと**
- cell identityの作り方で、保存LD2 Q8R66のrequestがbit単位で再現した。
- native LD3 headは one-body＋fragment 0〜2 で、nativeのtail identityは保存s_Rとbit単位で一致した（[INPUT_PREFLIGHT.json](INPUT_PREFLIGHT.json)）。

## 2. S0-r1による監査（[audit_run01](audit_run01/)、QPY読取8件）

S0-r1の `audit_path`（`5a18b792a`）を無変更で使った（wall 7.5秒）。規則は実行前に [AUDIT_PRICE_RULES.md](AUDIT_PRICE_RULES.md) に固定した。

| 検査 | 結果 |
|---|---|
| P1 束縛 | 4/4 PASS |
| P2 step区切り | 4/4 PASS。D対864組すべてhexまで一致。identity対は各step 5組。D blockは各step 4（D(f2)〜D(1b)）。全16 stepでevent＝660、product 0 |
| P3 境界の形 | PASS。D領域・障壁・compiledのancilla D領域が全stepで同一。末尾SはD領域の全4 blockを全wireで覆う。終端は2 pathで同一 |
| P5 X/Yと終端run | PASS。compiledのX/Yは最終run以外token一致。compile前の差はsdg 1個。最終runは `rz·sx·rz`、θ₁・θ₂は2 pathで同一。13 round × X/Yの余裕は最小0.224 rad（m2、Y） |
| P6 分解 | PASS。2 path × X/Yで `313 + 2685·8 + Σ本体` が保存rz_countと一致 |

**決定論部分**（全wireの合計、[EDGES.json](audit_run01/EDGES.json)）

| 部分 | LD3 | 参考：LD2（S0-r1） |
|---|---:|---:|
| 最初のstepの入口 F | 1,487 | 1,092 |
| 2番目以降の入口 P | 1,332 | 942 |
| 末尾 S（D(f2)〜D(1b)） | 1,353 | 958 |
| 終端 E | 156 | 156 |
| 最終run | 2 | 2 |
| **A（切片）**＝F＋S＋E＋2−(P＋S) | **313** | 308 |
| **H（stepあたり）**＝P＋S | **2,685** | 1,900 |

**event本体**（[SEGMENTS.csv](audit_run01/SEGMENTS.csv)）
- 第2〜第8 stepの14区間の本体RZ数の合計は62,344。
- **E3＝62,344/(14×660)＝7793/1155＝6.7472 RZ/event**。X軸だけを使い、同じpathのYは数えていない。
- 14区間の標準偏差は42.8（分散/R＝2.77）。
- 最初のstepの本体（4,378・4,426）は、後のstepの平均（4,453）より小さい。そのため、cellにモデルを当てはめた値（57,418.1）は、観測平均（57,367）より51（0.09%）大きい。

## 3. 26軸の参照価格と同じVPの総額（[PRICE_26AXIS.json](PRICE_26AXIS.json)・[CSV](PRICE_26AXIS.csv)）

VP（X/Y各軸）`[48,45,42,39,36,33,29,26,23,19,16,13,9]`。26軸はすべて「異なるq・Rへのモデル適用」で、較正cell自身の直接の数え値は別の行（`calibration_cell`）にある。X/Yは同じ値。

| m | q | R | C_ref LD3 | C_ref LD2（`a7eceecba`） | LD3/LD2 | X＋Yの加重項 LD3 | 総額に占める割合 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 1 | 3,004.7 | 2,226.0 | 1.350 | 288,456 | 0.03% |
| 1 | 2 | 1 | 5,696.5 | 4,167.2 | 1.367 | 512,684 | 0.05% |
| 2 | 4 | 1 | 11,080.0 | 8,087.6 | 1.370 | 930,719 | 0.09% |
| 3 | 8 | 2 | 21,901.0 | 16,080.5 | 1.362 | 1,708,274 | 0.16% |
| 4 | 16 | 3 | 43,596.9 | 32,674.3 | 1.334 | 3,138,974 | 0.30% |
| 5 | 32 | 6 | 87,528.5 | 68,294.5 | 1.282 | 5,776,878 | 0.55% |
| 6 | 64 | 11 | 176,903.0 | 149,670.6 | 1.182 | 10,260,375 | 0.98% |
| 7 | 128 | 21 | 362,129.4 | 352,560.6 | 1.027 | 18,830,731 | 1.79% |
| 8 | 256 | 42 | 760,218.7 | 917,269.4 | 0.829 | 34,970,062 | 3.32% |
| 9 | 512 | 83 | 1,661,761.4 | 2,687,267.7 | 0.618 | 63,146,934 | 6.00% |
| 10 | 1,024 | 165 | 3,889,757.6 | 8,786,343.7 | 0.443 | 124,472,242 | 11.83% |
| 11 | 2,048 | 330 | 10,059,211.3 | 31,207,839.3 | 0.322 | 261,539,493 | 24.86% |
| 12 | 4,096 | 660 | 29,238,146.1 | 116,957,179.0 | 0.250 | 526,286,631 | 50.03% |

**総額**

| 量 | 値（RZ） |
|---|---|
| T（LD3、exact） | 173557305044/165 |
| T（LD3、表示） | **1,051,862,454.81** |
| T_ref（LD2、`a7eceecba`） | 3,378,503,016.52 |
| 差額 LD3−LD2 | **−2,326,640,561.71** |
| 比 LD3/LD2 | **0.3113** |

**内訳**（VPの重み：shot 756、step 205,844、event：LD3 73,947,076・LD2 471,030,556）

| 部分 | LD3 | 割合 |
|---|---:|---:|
| 切片 313 × 756 | 236,628 | 0.02% |
| head・決定論 2,685 × 205,844 | 552,691,140 | 52.5% |
| event 6.7472 × 73,947,076 | 498,934,687 | 47.4% |

- 前半のroundではheadの増分で1.18〜1.37倍高い。
- m8以降はeventの減少が効いて安くなり、m12では0.25倍になる。

**LD2の基準との作り方の違い**
- LD2のT_refは、C2の凍結線形モデル（`b74e`）によるもので、今回の直接分解とは作り方が違う。
- 参考に、LD2のS0-r1の定数（308、1,900）と、R＝4205の本体平均/R（26646/4205）で同じ式を作ると、3,376,135,543になる。これはT_refと0.07%しか違わない。
- したがって、69%の差は作り方の違いでは説明できない（この照合値は比較の基準には使っていない）。

## 4. 前提と、確認していないこと

**qの外挿**
- 較正はq＝8だけである。q＞8（m4〜m12）での区切りと分解の成立は、LD3では観測していない。
- LD2ではq＝512まで成り立った（S0+、`2d3aa21f5`）。

**Rの外挿**
- E3はR＝660のstepだけから求めた。m12はR＝660で一致し、m11（330）・m10（165）などには同じ値を当てはめた。
- LD2では、eventあたりの費用がRとともにやや下がった（R＝33で6.78、R＝526で6.30）。このためLD3でも、小さいRのroundではE3が実際より小さい可能性がある（m10・m11は総額の36.7%）。

**標本の範囲**
- 2 pathの有限標本で、order 2のeventは観測していない。実sampler lawと理想lawの差は未資格のままである。

**評価していないこと**
- E[C]、CI、G、winner、shot配分の探索、Q3（新規0）。
- LD2のOWN-Bは別の配分なので比較に使っていない。

## 5. 照合・SOURCE・行っていないこと

**§10.5の照合**
- 照合範囲：2026-11-16に更新されたremote branchと、Cのlocal catalog branch。
- 同じ辞書・law・cell・compilerのLD3価格はなかった。旧資料のLD3は、PR-DFのLD3（旧rank・旧R）だけだった。
- 関連する稼働中のprocessはなく、出力先とprivate directoryも未使用だった。

**新規の作業と上限**
- compile：4（上限4）。QPY読取：8（上限8）。
- 新しい信号・Q3・shot配分探索：0。

**保全**
- 固定SOURCE（`13ab1acfa`、S0-r1 `5a18b792a`、LD3段階1・2の成果）、原結果、旧STOPは変更していない。
- ロードマップ本体は編集していない（影響は [HANDOFF_BC](HANDOFF_BC.md)）。

**commit**

| commit | 内容 |
|---|---|
| `158bbcc7b` | code・規則・test |
| `3074433d1` | binding |
| `04231b98c` | raw |
| `73f2dba54` | 監査・価格 |
| このcommit | 報告・HANDOFF |

**検査**
- 新テスト4件PASS：metadata、prefix 3のtoyでの旧coreの経路と高速化の一致、合成token列での分解・価格、親process観測。
- S0-r1のテスト14件もPASS（[TESTS.txt](TESTS.txt)）。
- これらは作成者自身による検査で、独立レビューではない。

**性能**
- C2のm9で確立した手法を流用した：2 path process並列、OPENBLAS 1 thread、event行列memo、同一byte fingerprint。
- 新しい手法は導入していない。

**mandatory STOP。**
