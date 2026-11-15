# A → B・C（統合担当にも）：PR-Pauli LD3 の26軸参照価格と同じVPの総額

根拠は [REPORT](REPORT.md)。機械可読の正本は [HANDOFF_BC.json](HANDOFF_BC.json)（26行の結合表）と [PRICE_26AXIS.json](PRICE_26AXIS.json)（係数・cell・総額）。これらはモデル参照価格であり、E[C]・CI・Gではない（`qualified_for_candidate=false`）。

## 渡すもの

| 対象 | 値・locator |
|---|---|
| 価格規則 | `C_ref(m,a) = A3 + H3·q_m + E3·q_m·R_m`、**A3＝313、H3＝2,685、E3＝7793/1155 RZ/event**（X/Y共通） |
| 係数の根拠 | 較正cell Q8R660の2 pathの監査（S0-r1、`5a18b792a`）で直接数えた値（fitではない）。`audit_run01/EDGES.json`（wireごとのF・P・S・E）、`SEGMENTS.csv`（16区間の本体）、`AUDIT_RESULT.json`（P1〜P6） |
| 26軸の価格 | `HANDOFF_BC.json` の `rows`：m・axis・q・R・N・C_ref（exact）・N×C_ref（exact）・value_source（全行がモデル適用） |
| configurationとの対応 | 各行に `original_configuration_id`（`dd5b881d…`）と `original_candidate_id` |
| total viewとの対応 | 各行に `total_configuration_id`（`6757998a…`）、`total_candidate_id`、`binding_id`、`allocation_id`（`PAULI_LD3_USER_VP_MATCH_LD2_01`）。Bの `C_RESULT_JOIN.json`（`2de33a4f4`）の行と、物理ID・Nが26/26で一致することを確認した |
| VP総額 | **T＝173557305044/165＝1,051,862,454.81 RZ** |
| LD2との比較 | LD2の同一VPのT_ref（`a7eceecba`）3,378,503,016.52との差 −2,326,640,561.71、比0.3113 |
| 較正cellの直接値 | `PRICE_26AXIS.json` の `calibration_cell`：2 path × X/Y の compiled RZ（57,379・57,355）、receipt locator（raw `04231b98c` の `run01/COMPILE_RECEIPTS.json`） |

**IDの区別**
- 較正cellのID `PAULI_PRICE_CELL:9cd5c434…`（q＝8）は、m12のround ID `RMSE:95014c1c…`（q＝4096）とは別物である。
- cellのreceiptを、roundの価格receiptに付け替えない。

## 使い方の注意

**適用の範囲**
- 直接確かめたのは、q＝8・R＝660の分解だけである。
- 26軸はすべて外挿である：qは4096まで、Rは1〜330。
  - qの外挿：LD2ではq＝512まで分解が成り立った。
  - Rの外挿：LD2ではeventあたりの費用がRとともにやや下がった。そのため、小さいRのround（m10・m11など）では、LD3の価格がやや低めに出ている可能性がある。

**LD2との作り方の違い**
- LD2のT_refは、C2の凍結線形モデル（`b74e`）による。
- LD2のS0-r1の定数で同じ式を作ると、T_refとの差は0.07%である。比較の結論は、作り方の違いには左右されない。

**Q3との組合せ**
- このVPでのQ3適合は、Bの `2de33a4f4`（LD3のMSE上界6.2062×10⁻⁷）による。
- 配分探索で別の配分にする場合は、同じ26軸の価格をそのまま使ってよい（価格は配分に依存しない）。

## 次の段階（未実行・要許可）

1. **B：LD3の配分探索**
   - この26軸の価格と、LD3のtotal viewを使う。
   - LD2のOWN-B（2,442,845,584.51）、およびPFの2基準（B0I p7 S2 593,555,352、Full PF S2 838,693,580）と比べる。
   - 現在のVPでは、LD3の総額はPF基準の1.25〜1.77倍である。
2. **A（必要なら）**：q＞8でのLD3の分解の確認。保存済みのcellまたは新しいcellで、S0+と同様に行う。
3. **分割の次の候補（判断材料のみ）**
   - LD3では、headの決定論部分が総額の52.5%を占める。
   - 1 block増やすごとにstepあたり約785 RZ増える（VPの重みで約1.6億RZ）。
   - LD5（descriptorあり）を試すかどうかは、Bの配分探索の結果を見て決めるのがよい。

## ロードマップへの影響（本体は編集していない）

| 箇所 | 更新の提案 |
|---|---|
| PR-Pauliの行 | 「LD3の段階3：較正cell Q8R660（4 compile）の監査で価格規則が成立（A3＝313、H3＝2,685、E3＝6.747 RZ/event）。同じVPのモデル参照総額は1.052×10⁹ RZで、LD2（3.379×10⁹）の0.311倍。Q3は同じVPで適合（B）。PF基準の1.25〜1.77倍。E[C]・CI・Gは未認定」 |
| 次の判断 | BのLD3配分探索を許可するか |
| 依存関係 | B：配分探索 → 必要ならA：q＞8の分解確認 → 分割の次の候補（LD5）の判断 |

**mandatory STOP。**
