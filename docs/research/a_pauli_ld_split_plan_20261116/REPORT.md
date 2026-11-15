# PR-Pauli：決定論部分の大きさ L_D を変えたときの比較相手の提案

2026-11-16 JST。Codex Aの担当分（chat A1109が実施）。**照合と計画のみ。** 新規sampling・信号生成・QPY再監査・build・compile・fit・価格予測・配分生成・Q3は0件。

## 冒頭の答え

| 問い | 答え |
|---|---|
| **比較相手に選んだ L_D** | **L_D＝3**。headは one-body ＋ fragment 0・1・2、tailは fragment 3〜9 |
| **選んだ理由** | LD2の総費用の大部分を占めるevent数を、大きく減らし得る唯一の隣接分割だから（下記） |
| **何を一つ追加すれば比較が前進するか** | **LD3のP0 Pauli辞書（λ_P(LD3)）と、共通κによる13 roundのschedule（各roundのR_m）。** compileも信号も不要で、費用の主因であるevent数が確定する |

**選んだ理由**

1. **LD1は、保存資料から方向がほぼ読める**
   - 同じκでは、event数は全roundで約2.59倍になる（λ_P²の比。VP配分で重み付けしたevent位置は12.19億 対 4.71億）。
   - 一方、shotの必要量はほとんど変わらない。同じVP配分でMSE上界は6.2080×10⁻⁷ 対 6.2075×10⁻⁷、V0でも4.7246×10⁻⁷ 対 4.7242×10⁻⁷。
   - headが1 fragment分小さくなる節約は、m0の単一pathの比較で約856 RZ。これに対し、LD2のm12の1 stepは、決定論部分1,900 RZとevent本体約26,600 RZからなる（S0の実測）。
   - したがってLD1は、eventあたりの費用がLD2の約4割未満でない限り、費用が上がる方向にある。LD1と比べても、PR-Pauliの達成費用を下げる情報は得にくい。
2. **LD3は、その逆向きの一段階である**
   - tailのDF λ_Rは1.603 → 0.583（同じsnapshotの保存descriptor）で、λ²の比は0.132。fragment 2は、LD2のtail λの約64%を占める。
   - 同じκではevent数はλ_P²に比例する（LD1/LD2で成り立っている）。λ_P/λ_Rの比がLD1・LD2並み（1.15〜1.28）なら、event数は約0.11〜0.13倍になる。
   - 代わりにheadが1 block増える。fragment blockのD領域は、LD2ではD(f1)が33対、D(f0)が36対の規模である。
3. **それでも費用順位は、保存資料からは決まらない**
   - 未取得のもの：λ_P(LD3)、headのPF誤差の変化とそれによるshot必要量、LD3辞書のeventあたり費用、fragment 2のblockの費用。
4. **LD3は、現在のSOURCEから構成できる**
   - 同じsnapshotのDF descriptor（λ_R・s_R）が保存されている。
   - P0と信号のcodeはprefixを(1, 2)に固定しているが、一般化の差分は小さい。
   - LD4にはdescriptorがない。LD5以降はtailがほぼ消え（λ_R ≤ 0.054）、PR-Pauliの性格がPF側へ寄る。まず隣接する一段階で傾向を見るのが先である。
   - PR-DFでも、D2が次の候補にLD3を挙げている（`f42f6e9b9`）。

## 1. PR-Pauliの L_D が数えるもの（固定SOURCE）

**定義**：L_D は、決定論的なheadに置く**二体DF fragmentの本数**である。one-body blockは常にheadに入り、L_Dには数えない。fragmentは保存順（index 0〜9、DF10）のまま使い、重みで並べ替えない。

**head（決定論側）**
- one-body と fragment 0〜L_D−1。
- 各outer stepは S2 の形で並ぶ：U(one-body) U(f0) … U(f_{L_D−1}) ［tail］ D(f_{L_D−1}) … D(f0) D(one-body)。
- 出典：native `ax2b_native_df_v5.build_partial_native`（`c31ef4241`）。P0では `head = sum(full.matrices[:prefix+1])`（`fd111f989` `df_head_pauli_tail_p0.py`）。

**tail（ランダム側）**
- fragment L_D〜9 を、保存されたweighted-Hermitian completionごとJordan–Wigner変換し、同じPauli文字列をglobalに集約する。
- identity成分は s_R として抜き出し、総Tで一回だけ位相として適用する。
- eventは |c_j|/λ_P の確率で引く。K=2（paired Taylor）、1 stepあたりR個。
- 出典：P0 `square_factor_dictionary` は `ham.lambdas[prefix:]` を使う（`fd111f989`）。辞書は `4220f3006`（LD1・LD2とも非identity 184項）。eventの抽出は `pauli_tail_price_connection.sample_steps`（`13ab1acfa`）。

**PR-DFとの違い**
- PR-DFも同じ数え方のL_Dと同じheadを使う（D2 `f42f6e9b9` REPORT §1）。
- ただしtailのeventは、DF fragmentの成分（DF基底の回転と制御ZZ/Z）である。λはDFのλ_R（LD1で2.863、LD2で1.603）。
- PR-PauliのλはPauliのλ_P（LD1で3.307、LD2で2.055）。
- 同じκでもR_mが違い（m12：PR-DF LD1は8160、PR-Pauli LD1は10885）、価格も別物である。

**固定条件**（各L_Dで共通）
- H4 linear 1.00 Å、STO-3G、tol-only DF10、sector (2,2)/36、同じH・state・snapshot・target。
- S2、ΔPF＝0.1、K＝2、元のexactな共通κ、h＝0.1、M＝12、P＝0。

## 2. L_D別の保存状況

機械可読表：[CANDIDATES.json](CANDIDATES.json)（全locator・check 11件）／[CANDIDATES.csv](CANDIDATES.csv)。作成script：[`scripts/build_pauli_ld_split_candidates.py`](../../../scripts/build_pauli_ld_split_candidates.py)（固定Git objectと保存context 1件を読むだけ）。

| L_D | 構成・schedule | 保存信号・Q3・評価済み配分 | 直接価格・モデル参照価格 | 配分を選び直して総費用を出すための不足 |
|---:|---|---|---|---|
| 0 | 現行の信号adapterの範囲外（1 ≤ prefix < rank）。descriptorなし | なし | なし | 境界の定義、descriptor、以下LD3と同じ |
| **1** | `d905d219…`。S2・0.1・K2・共通κ。m12でR=10885。scheduleは `714247619` MEAN_HANDOFF packets[0] | 13 round信号あり。V0：診断のみ（MSE上界4.7246×10⁻⁷）。VP：**適合**（6.2080×10⁻⁷、`50be68b80`） | m0 X/Y 1 path（配線用、1380 RZ）。モデル参照価格なし | LD1固有の26軸参照価格。Bの配分探索とQ3 |
| **2** | `d7dd4f52…`。m12でR=4205。packets[1] | 13 round信号あり。V0：診断のみ（4.7242×10⁻⁷）。VP：適合（6.2075×10⁻⁷）。FPN-B：**不適合**。OWN-B：**適合**、T＝2,442,845,584.51（`738df2262`） | 配線m0、較正5 cell・18 compile、holdout m5・m6・m9、凍結モデル（`b74e`）、T_ref 26軸（`a7eceecba`）。構造：q ≤ 512で `308 + 1900·q + Σ本体`（S0） | Bの配分探索に必要なものはそろっている。m10〜m12は外挿のまま |
| **3** | 未作成。DF descriptor：λ_R＝0.5832704246339966、s_R＝0.09261516589219046 | なし | なし | P0辞書（λ_P）、configurationと13 roundのschedule、信号、total view、固有の価格規則、Q3 |
| 4 | descriptorなし | なし | なし | descriptorと出典を整えてから、LD3と同じ |
| 5 | 未作成。λ_R＝0.0544 | なし | なし | LD3と同じ |
| 6 | descriptorなし | なし | なし | LD4と同じ |
| 7 | 未作成。λ_R＝3.13×10⁻⁴ | なし | なし | LD3と同じ |
| 8 | descriptorなし | なし | なし | LD4と同じ |
| 9 | 未作成。λ_R＝1.56×10⁻⁷ | なし | なし | LD3と同じ |
| 10 | tailが空。PRのruntimeは正のλを要求する。Full PFが別の決定論的な端点 | — | — | PRとしては定義しない |

**表の補足**
- DF descriptorの出典は、保存context `H4_context.json`（sha256 `8396605f…`。索引は `7a8bc7f13` SOURCE_INDEX）。LD1・LD2の値は、P0のPREFIX_SUMMARYとbit単位で一致した。
- **再利用したもの**：LD1・LD2の同一配分のQ3（`50be68b80`）、LD2のB_PRICE評価（`738df2262`）、T_ref（`a7eceecba`）、m9の構造確認（`2d3aa21f5`）。いずれも再計算していない。
- **保存候補があるもの**：LD1・LD2。**現在のSOURCEから新たに構成する必要があるもの**：LD3・5・7・9（descriptorあり）、LD4・6・8（descriptorから必要）。

## 3. 分割を変えたときの増減を、何と何の比較で調べられるか

| 比較 | 分けて見られること | 保存資料で分かっていること |
|---|---|---|
| **LD1 対 LD2**（保存） | headから1 fragmentを外し、tailに入れたときの影響 | event数は約2.59倍（各roundのR_mの比。λ_P²の比2.589と一致）。shotの必要量はほぼ同じ（同じ配分でMSEがほぼ一致）。head側の節約は、m0の単一path（LD1：R=3で1380 RZ、LD2：R=2で2236 RZ）で約856 RZ。**費用は上がる方向**だが、LD1のeventあたり費用は未測定 |
| **LD2 対 LD3**（提案） | headに1 fragmentを加え、tailから外したときの影響 | DFのλ²の比は0.132。λ_P(LD3)、headのPF誤差、LD3辞書のeventあたり費用、fragment 2のblockの費用は**未取得**。**費用順位は保存資料から決まらない** |
| LD3 対 LD5（後続） | tailがほぼ消える側（λ_R ≤ 0.054）への延長 | LD3の結果が良い場合だけ意味がある |

**LD2のstep費用の内訳**（S0の実測。eventの比重を見るための値で、価格予測ではない）

| 部分 | 値 |
|---|---|
| 決定論部分（1 stepあたり） | 1,900 RZ |
| 2番目以降のstepの本体の平均 | R=33で224、R=66で430、R=526で3,334 |
| 最初のstepの本体の平均（R=4205） | 26,646 |

長いroundでは、eventの部分が1 stepの約93%を占める。分割の比較では、「event数がどれだけ減るか（λ_P²）」「その代わりに増えるheadの費用」「headのPF誤差によるshotの変化」の3つを、同じκ・同じQ3条件のもとで、LD3自身の信号と価格から測る必要がある。

## 4. LD3をBの共通shot配分探索へ渡すための最小案（未実行・要許可）

**Bに渡す入力**
- LD3自身の13 roundの信号（total view付き）。
- X/Y別の26軸の1 shot参照価格。

### 段階1：P0辞書とschedule（compile 0）

- 既存のP0 code（`fd111f989`）のprefixを(1, 2, 3)に広げ、LD3のPauli辞書・λ_P・s_Pの照合・operator/sector gateを得る。
- Bのcanonical `generate_configuration`（P1 adapter `05447d1d4`）で、共通κのconfigurationと13 roundのR_mを作る。
- これでevent数が厳密に決まる。

### 段階2：13 roundの信号

- 既存のprovider（`7084fbc9a`・`05447d1d4`。prefixの集合を拡張）で、sector36のideal finite meanを13件計算する。
- Bがtotal viewを作る（LD1・LD2の `9fa3c47d`・`285bf481` と同じ手順）。

### 段階3：価格規則（最大4 compile、監査は0-compile）

- **測るもの**：LD3の非configuration cellを1つだけ取る。q=8、R＝R_12(LD3)（m12の1 stepのR）、T＝0.8。fresh 2 path × X/Y（最大4 compile）。既存のD2 callerと受入の経路を使う。
- **監査**：その2 pathを、S0-r1の規則（`5a18b792a`）で0-compile監査する。これで次が得られる。
  - 切片 A₃ と、stepあたりの決定論定数 H₃（厳密値）。
  - 2番目以降のstepの本体 14区間の平均（eventあたり Ē₃）。
  - X/Yの一致と、LD3の13個のTに対する終端runの余裕。
- **価格規則**（26軸、X/Y共通）：

  ```
  C_ref^LD3(m, a) = A₃ + H₃·q_m + Ē₃·q_m·R_m^(3)
  ```

- **妥当な理由**
  - LD2では、q ≤ 512の保存回路でこの分解が厳密に成り立った。
  - LD3の主要round（m10〜m12）のRは、R_12(LD3)の近くにある。
  - eventあたりの平均は、LD2ではRに弱くしか依存しない（6.78、6.52、6.34）。小さいRのroundでの差は、重みが小さい。

### LD2の係数をLD3に流用する場合の境界

測定前の暫定値としてだけ使える。採否の判断には使わない。

| 区分 | 内容 |
|---|---|
| **共通** | native headのbuilder、各stepのD(one-body)障壁と区間の局所性（LD2で確認済み）、D2によるtailのlowering、wrapper・compiler、X/Yの構造 |
| **仮定** | eventあたり費用がLD2と同じ（約6.3 RZ）。fragment 2のblockによる増分ΔH（直接の値はない。LD1/LD2の単一path差856 RZが唯一の類例）。切片308がそのまま使える |

### 段階4：B

- VP配分でのQ3（1回）。
- LD3の価格で、Bの配分探索を行う（アルゴリズムはBに任せる）。
- LD2のOWN-B（2,442,845,584.51）およびPFの基準と比べる。

### §10.5に沿った照合と、上限

**重複を照合する範囲**
- A：`714247619` 系のdf_head_pauli_tail、`7a0c9468b`、本branch。
- B：`rpe-total-dp-acceleration`、`b-rpe-pauli-integration`、`b2-pf6-allround-cost`。
- C：`m5-m4-receipt-admission`。C2：`pauli-ld2-price-freeze`。D2：`d-pauli-b1-caller-prep`、`d2-pr-df-ld1-m3-acquisition`（PR-DFのLD3計画）。
- 旧資料：Track CのH4 LD3（`c5a3c102`、DF成分218・324。PR-DF側）、O1-M1のLD0 Pauli（`5e4f0048`）、P3（bare W−D）。いずれもPR-PauliのLD3としては使えない。
- 稼働中のprocessと、出力namespace。

**再利用するもの**
- H・state・snapshot、LD3のdescriptor、κ。
- P0・P1・provider・B canonical・Q3経路・D2 callerとreceipt・S0-r1の監査code。
- LD1・LD2の結果は参照用。

**新規の対象と上限**

| 段階 | 対象 | 上限 |
|---|---|---|
| 1 | P0辞書 | 1件。compile 0 |
| 2 | 信号 | 13 mean。compile 0 |
| 3 | 価格 | 1 cell・2 path・X/Y で最大4 compile。監査のQPY読取は最大8件 |
| 4 | Q3 | VPで1回（Bの探索は別） |

**得られる判断**
- 段階1：event数が、DFのλから見込むほど減るか。
- 段階2と段階4：LD3が同じ精度を満たすshotの必要量。
- 段階3と段階4：LD3の適合配分のモデル参照費用が、LD2（2.44×10⁹）より低いか。
- 低ければ、次はLD5（descriptorあり）を検討する。低くなければ、PR-Pauliの最良の分割はLD2付近と判断する。

## 5. 保全・範囲

- 原結果、旧STOP、固定SOURCE、他担当のbranchは変更していない。
- ロードマップ本体は編集していない（影響は [HANDOFF_B](HANDOFF_B.md) 末尾）。
- 段階1〜4は、いずれも別の明示許可を要する。

**mandatory STOP。**
