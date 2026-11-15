# A → B（統合担当にも）：PR-Pauliの分割比較で推奨する候補 LD3

根拠は[REPORT](REPORT.md)、表は[CANDIDATES.json](CANDIDATES.json)。計画のみで、新しい計算は0件である。

## 推奨候補

| 項目 | 内容 |
|---|---|
| 候補 | PR-Pauli **L_D＝3**（head：one-body＋fragment 0〜2、tail：fragment 3〜9 のJW Pauli辞書） |
| ID | **未作成**（configuration IDとtotal viewは、段階1・2で作る） |
| 固定条件 | LD2と同じ：H4/1.00 Å/STO-3G/tol-only DF10、H `c1a6c428…`、state `9bcc0b0e…`、snapshot `12219b7d…`、S2・ΔPF 0.1・K2・元のexactな共通κ・h 0.1・M 12・P 0 |
| DF descriptor | λ_R＝0.5832704246339966、s_R＝0.09261516589219046。出典は保存context `/home/AbeHiromu/projects/h46-long-time-pf-pr/20261025/H4_context.json`（sha256 `8396605fa0c5ee34…`、selector `prefixes/3`。索引 `7a8bc7f13` SOURCE_INDEX） |
| 比較する基準 | LD2 `d7dd4f52…`（total `285bf481…`）。OWN-B が適合で、T＝2,442,845,584.51（`738df2262`） |

## Bに渡す入力と、そのlocator（予定）

**1. 13 roundの信号**
- 段階2で作成する。形式はLD1・LD2と同じMEAN_HANDOFF（`714247619` `docs/research/df_head_pauli_tail_all_round_results_20261114/MEAN_HANDOFF.json` の packets と同形）。
- total viewは、LD1・LD2と同じ手順（`9fa3c47d`・`285bf481`）で作る。

**2. 26軸の1 shot参照価格**
- 段階3の価格規則による：

  ```
  C_ref^LD3(m, a) = A₃ + H₃·q_m + Ē₃·q_m·R_m^(3)
  ```

  - A₃・H₃：LD3の非configuration cell（q=8、R＝R_12(LD3)）の2 pathを、S0-r1で監査して得る厳密値。
  - Ē₃：2番目以降のstep 14区間の平均。
  - X/Yは同じ値（X/Yの構造の一致はS0で確認する）。

**3. Q3の経路**
- LD1・LD2と同じPauliの固定経路（`f54b41a2e`、`scripts/run_pauli_q3_three_quarter.py`）。
- 最初はVP `[48,45,42,39,36,33,29,26,23,19,16,13,9]` で1回。

## 使える価格規則と、その限界

| 規則 | 使ってよい場面 | 共通な部分と仮定 |
|---|---|---|
| **測定規則**（上記、段階3の後） | Bの配分探索と、LD2との比較 | step分解の形はLD2の保存回路で確認済み（q ≤ 512）。LD3でもS0で確かめる。Ē₃はR_12(LD3)の近くで測る。小さいRのroundでの差は重みが小さい |
| 暫定のLD2流用（`308 + (1900+ΔH)·q + 6.34·q·R^(3)`） | 段階1のevent数を見た後の、大まかな目安だけ | **仮定**：LD3辞書のeventあたり費用がLD2と同じ。fragment 2のblockの増分ΔHは未測定（LD1/LD2の単一path差856 RZが唯一の類例）。採否の判断には使わない |

## 不足しているもの（すべて未実行・要許可）

1. LD3のP0辞書（λ_P(LD3)）、configuration、共通κの13 roundのschedule。code `fd111f989` と `05447d1d4` は、prefixを(1, 2)に固定しているので、拡張が要る。
2. LD3の13 roundの信号と、total view。
3. 価格：1 cell・2 path・X/Y で最大4 compile。S0-r1による0-compileの監査。
4. Q3：VPで1回。その後、Bの配分探索。

## LD1について

- 信号と、VPでのQ3（適合）は保存されている。足りないのは、LD1固有の価格規則だけである。
- ただし、同じκではevent数が約2.59倍で、shotの必要量はLD2とほぼ同じである。そのため費用が上がる方向にあり、最初の比較相手にはしない。
- LD1を後で確かめる場合は、同じ価格規則の段階3（q=8、R＝10885）だけで足りる。

## ロードマップへの影響（本体は編集していない）

| 箇所 | 更新の提案 |
|---|---|
| PR-Pauliの行 | 「分割を変えた比較：L_Dは二体DF fragmentの本数（one-bodyは数えない）。LD1/LD2は保存済み（同じ配分で適合、event数はLD1が約2.59倍）。次の比較相手はLD3（DFのλ_Rが0.583で、event数を大きく減らし得る隣接分割）。最初の追加は、LD3のP0辞書と13 roundのschedule」 |
| 次の判断 | LD3の段階1（P0・schedule）を許可するか。続く段階2〜4（信号13・最大4 compile・Q3）は、段階1の結果を見て判断する |
| 依存関係 | A：段階1〜3 → B：Q3と配分探索 → LD2との比較 |

**mandatory STOP。**
