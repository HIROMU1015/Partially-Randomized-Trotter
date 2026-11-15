# PR-Pauli LD5：較正cellによる固有価格と、同じ配分でのLD3との比較

2026-11-16 JST。Codex Aの担当分（chat A1109が実施）。LD5段階1の正本：[HANDOFF_B](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bf0a930fd75fb2eb693622edcf3b7204e30f5eff/docs/research/df_head_pauli_tail_ld5_stage1_20261116/HANDOFF_B.md)（`bf0a930fd`）。LD5の13 round信号は、別branchの [df_head_pauli_tail_ld5_signals_20261116](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/codex/pr-r-dependent-n-model-20261111/docs/research/df_head_pauli_tail_ld5_signals_20261116)（packet `ac46d100e`）にある。

## 冒頭の答え

| 問い | 答え |
|---|---|
| 較正cell | Q8R4（q＝8、R＝4、T＝0.8）を独立2 path × X/Yで取得。**4 compile**、全gate PASS。compiled RZは path0 34,315・path1 34,314（X＝Y） |
| 価格分解 | **支持された。** S0-r1の監査P1〜P6がすべてPASS。2 path × X/Yで分解が厳密に一致 |
| LD5の係数 | **A5＝311、H5＝4,215、E5＝255/28＝9.107 RZ/event**（E5の分母は56 event） |
| H5−2685 | **＋1,530 RZ/step**（fragment 3・4の2 block分。1 blockあたり765。LD2→LD3は785） |
| VPでの総額 | LD5 **872,241,518.14** 対 LD3 1,051,862,454.81 → 差 **−179,620,936.67**、比0.8292 |
| FULL_PF__B_PRICEでの総額 | LD5 **541,743,905.57** 対 LD3 541,936,295.95 → 差 **−192,390.38**、比0.99964 |

**意味すること**
- headが2 block増える費用（1,530 RZ/step）と、tail eventがほぼ消える効果が、配分によって違う形で釣り合う。
- VP（m11・m12に多くのshot）では、LD5が約17%安い。
- 現在の最安適合列FULL_PF__B_PRICE（m11・m12のshotが8・2）では、差は−0.04%で、ほぼ同額である。
  - この差は、本体のstepごとのばらつきから見た目安（LD3側で約±0.51M RZ）より小さい。費用の優劣は判定できない。
- round別に見ると、LD5はm0〜m10でLD3より1.11〜1.56倍高く、m11で0.86倍、m12で0.60倍と安い。したがって、有利な分割は配分の形に依存する。

**未評価**
- LD5のQ3適否（VP、FULL_PF__B_PRICEのどちらも）。
- E[C]・CI・G。

**段階1の分岐値との関係**：段階1の目安（FULL_PF__B_PRICEで約1,535、VPで約2,408 RZ/step）は事前の目安として残した。今回の比較には、LD5から得た係数だけを使った。実測のH5−H3＝1,530は、FULL_PF__B_PRICEの目安とほぼ一致した。

## 1. 取得（[run01](run01/)）

| 項目 | 内容 |
|---|---|
| cell ID | `PAULI_PRICE_CELL:53f1503d5e53f9190730006845b8e89605dcbbc270349fd79d24701745256318` |
| roundのIDとの区別 | m3（q＝8、R＝1、`RMSE:c943efea…`）ともm12（q＝4096、R＝4、`RMSE:795c5535…`）とも別IDで、cellのreceiptはround価格のreceiptではない |
| cohort・seed | `A_PAULI_LD5_CAL_Q8R4_N2_V1`、16個。既存のLD1〜LD3の1,341個と重複なし（[BINDING.json](BINDING.json)） |
| 科学SOURCE | D2の `13ab1acfa` を専用checkoutから無変更で使用。driver：凍結 `1097c37f9`、binding `aedf55d9a` |
| 実行 | 2 path process並列、wall 41秒、peak RSS約0.70 GB、exit 0/0。retry・追加compileは0 |

**path別の結果**

| path | X RZ | Y RZ | CX | 数値gate（最大） |
|---|---:|---:|---:|---|
| 0 | 34,315 | 34,315 | 16,292 | 9.1×10⁻¹³ |
| 1 | 34,314 | 34,314 | 16,288 | 8.9×10⁻¹³ |

**実行前に確認したこと**
- LD3のspecで公開LD3 bindingを再現した（driverのpath以外の全欄）。
- native LD5 headは one-body＋fragment 0〜4 の6 blockで、tail identityは保存s_Rとbit単位で一致した（[INPUT_PREFLIGHT.json](INPUT_PREFLIGHT.json)）。

## 2. S0-r1による監査（[audit_run01](audit_run01/)、QPY読取8件、wall 3.1秒）

| 検査 | 結果 |
|---|---|
| P1 束縛 | 4/4 PASS |
| P2 step区切り | 4/4 PASS。D対1,376組すべてhexまで一致。identity対は各step 13組（LD3は5）。D blockは各step 6（D(f4)〜D(1b)）。全16 stepでevent＝4、product 0 |
| P3 境界の形 | PASS。D領域・障壁・compiledのancilla D領域が全stepで同一。末尾SがD領域の全6 blockを全wireで覆う。終端は2 pathで同一 |
| P5 X/Yと終端run | PASS。X/Yは最終run以外token一致、compile前の差はsdg 1個、`rz·sx·rz`、θ₁・θ₂は2 pathで同一。13 round × X/Yの余裕は最小0.082 rad（m8、X） |
| P6 分解 | PASS。2 path × X/Yで `311 + 4215·8 + Σ本体` が保存rz_countと一致 |

**決定論部分**（全wireの合計、[EDGES.json](audit_run01/EDGES.json)）

| 部分 | LD5 | LD3 |
|---|---:|---:|
| 最初のstepの入口 F | 2,247 | 1,487 |
| 2番目以降の入口 P | 2,094 | 1,332 |
| 末尾 S | 2,121 | 1,353 |
| 終端 E | 156 | 156 |
| 最終run | 2 | 2 |
| **A**＝F＋S＋E＋2−(P＋S) | **311** | 313 |
| **H**＝P＋S | **4,215** | 2,685 |

**event本体**（[SEGMENTS.csv](audit_run01/SEGMENTS.csv)）
- 第2〜第8 stepの14区間の本体RZ数の合計は510。**E5＝510/56＝9.107 RZ/event**。
- X軸だけを使った。Yは同じpathなので、二重計上していない。
- 1区間（4 event）の本体は21〜48、標準偏差6.9。
- E5はE3（6.747、R＝660）より大きい。LD2でも、Rが小さいstepほどeventあたりの費用が大きかった（R＝2で14.0）。
- cellにモデルを当てはめた値は34,322.4で、観測平均（34,314.5）より8（0.02%）大きい。

## 3. 26軸の参照価格（[PRICE_26AXIS.json](PRICE_26AXIS.json)・[CSV](PRICE_26AXIS.csv)）

- 直接観測したのは、q＝8・R＝4の2 pathだけである。
- 26軸はすべてモデル適用である（`value_source=MODEL_APPLICATION_FROM_Q8R4_DECOMPOSITION`、`direct_observation_at_this_q_R=false`）。R＝1・2のround（m0〜m11）やq＞8への適用もモデル値である。
- X/Yは同じ値。

| m | q | R | C_ref LD5 | C_ref LD3 | LD5/LD3 | 加重差（X＋Y、百万RZ）：VP | 加重差：FULL_PF__B_PRICE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 1 | 4,535.1 | 3,004.7 | 1.509 | +0.15 | +0.15 |
| 1 | 2 | 1 | 8,759.2 | 5,696.5 | 1.538 | +0.28 | +0.28 |
| 2 | 4 | 1 | 17,207.4 | 11,080.0 | 1.553 | +0.51 | +0.51 |
| 3 | 8 | 1 | 34,103.9 | 21,901.0 | 1.557 | +0.95 | +0.95 |
| 4 | 16 | 1 | 67,896.7 | 43,596.9 | 1.557 | +1.75 | +1.75 |
| 5 | 32 | 1 | 135,482.4 | 87,528.5 | 1.548 | +3.16 | +3.16 |
| 6 | 64 | 1 | 270,653.9 | 176,903.0 | 1.530 | +5.44 | +5.44 |
| 7 | 128 | 1 | 540,996.7 | 362,129.4 | 1.494 | +9.30 | +9.30 |
| 8 | 256 | 1 | 1,081,682.4 | 760,218.7 | 1.423 | +14.79 | +14.79 |
| 9 | 512 | 1 | 2,163,053.9 | 1,661,761.4 | 1.302 | +19.05 | +19.05 |
| 10 | 1,024 | 1 | 4,325,796.7 | 3,889,757.6 | 1.112 | +13.95 | +13.95 |
| 11 | 2,048 | 2 | 8,669,933.9 | 10,059,211.3 | 0.862 | −36.12 | −22.23 |
| 12 | 4,096 | 4 | 17,414,162.4 | 29,238,146.1 | 0.596 | −212.83 | −47.30 |

## 4. 同じ配分でのLD3との比較

- 重み：shot Σ2N、step Σ2N·q、event Σ2N·q·R。
- LD3側の分解は、LD3の係数（313、2,685、7793/1155）で保存総額をexactに再現することを確認してから使った。
- LD5のQ3適否は、どちらの列も未評価である。

| 列（X/Y各軸のN） | LD5 総額 | LD3 保存総額 | 差 LD5−LD3 | 比 |
|---|---:|---:|---:|---:|
| VP `[48,45,42,39,36,33,29,26,23,19,16,13,9]` | 872,241,518.14（6105690627/7） | 1,051,862,454.81（`f25a05f8a`） | **−179,620,936.67** | 0.8292 |
| FULL_PF__B_PRICE `[…,16,8,2]` | 541,743,905.57（3792207339/7） | 541,936,295.95（`fb4d78781`） | **−192,390.38** | 0.99964 |

**費用差の内訳**

| 列 | 切片（A5−A3）×shot | head（H5−H3）×step | event E5·W5−E3·W3 |
|---|---:|---:|---:|
| VP（shot 756、step 205,844、event LD5 480,276・LD3 73,947,076） | −1,512 | **+314,941,320** | **−494,560,744.67** |
| FULL_PF__B_PRICE（shot 732、step 128,020、event LD5 209,940・LD3 29,341,636） | −1,464 | **+195,870,600** | **−196,061,526.38** |

**差の大きさの目安**（記述的。CIではない）
- E3とE5の推定には、本体のstepごとのばらつきがある。
- FULL_PF__B_PRICEでは、そのばらつきから見たevent部分の目安が、LD3側で約±0.51M RZ、LD5側で約±0.10M RZである。差−0.19Mはこれより小さく、同額とみなすのが妥当である。
- VPでは、目安（約±1.3M）に比べて差−180Mは十分大きい。

## 5. 前提と、確認していないこと

**外挿**
- qの外挿：q＝8 → 4096。LD5の分解はq＝8でだけ確認した。LD2ではq＝512まで成立している。
- Rの外挿：R＝4 → 1・2。eventあたりの費用は、R＝1では、R＝4より大きい可能性がある（LD2の傾向）。
  - そのため、m0〜m10のLD5の価格は、やや低めに出ている可能性がある。
  - ただしeventの部分は小さい。例えばm10では、1 stepあたりhead 4,215に対しeventは約9 RZ。

**標本の範囲**
- 2 pathの有限標本で、order 2のeventは観測していない。実sampler lawは未資格のままである。

**評価していないこと**
- Q3、E[C]、CI、G、winner、shot配分の探索（新規0）。

## 6. 照合・SOURCE・行っていないこと

**§10.5の照合**
- 同じ辞書・law・cell・compilerのLD5の保存信号、較正cell、receiptはなかった。関連する稼働中のjobもなかった。
- LD5の信号は、同じ作業の別branchで今回取得した。

**新規の作業と上限**
- compile：4（上限4）。QPY読取：8（上限8）。

**保全**
- 固定SOURCE（`13ab1acfa`、S0-r1 `5a18b792a`、LD3段階3 `158bbcc7b`、LD5段階1）・原結果・旧STOPは変更していない。
- ロードマップ本体は編集していない（影響は [HANDOFF_BC](HANDOFF_BC.md)）。

**commit**

| commit | 内容 |
|---|---|
| `1097c37f9` | code・規則・test |
| `aedf55d9a` | binding |
| `f78ec5c2e` | raw |
| `fb0d60b0a` | 監査・価格 |
| このcommit | 報告 |

**検査**（作成者自身による技術検査で、独立レビューではない）
- 新テスト3件PASS
  - LD3 bindingの再現
  - prefix 5のtoyで旧coreとreceiptの経路を確認
  - LD3の保存tokenから、LD3の係数と保存総額（2列）を再現
- LD3段階3のテスト4件、S0-r1のテスト14件もPASS（[TESTS.txt](TESTS.txt)）。

**mandatory STOP。**
