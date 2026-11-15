# A → B・C（統合担当にも）：PR-Pauli LD5 の固有参照価格と、同じ配分でのLD3比較

根拠は [REPORT](REPORT.md)。機械可読の正本は [PRICE_26AXIS.json](PRICE_26AXIS.json)。中身は、係数とその根拠、26行の価格（configurationのphysical ID付き）、2列の比較、較正cellの直接値である。モデル参照価格であり、E[C]・CI・Gではない。

LD5の13 round信号は、別branch（`codex/pr-r-dependent-n-model-20261111`）にある：
- MEAN_HANDOFF：commit `ac46d100e74f3f56b4a3ff4d95c15fdc7fe8ca98`、`docs/research/df_head_pauli_tail_ld5_signals_20261116/MEAN_HANDOFF.json`
- 説明：同じdirectoryの `HANDOFF_B.md`

## 渡すもの

| 対象 | 値・locator |
|---|---|
| 価格規則 | `C_ref(m,a) = A5 + H5·q_m + E5·q_m·R_m`、**A5＝311、H5＝4,215、E5＝255/28 RZ/event**（X/Y共通） |
| 係数の根拠 | 較正cell Q8R4の2 pathを、S0-r1（`5a18b792a`）とLD3の計数関数（`158bbcc7b`）で監査し、直接数えた値（fitではない）。`audit_run01/EDGES.json`・`SEGMENTS.csv`・`AUDIT_RESULT.json` |
| 26軸 | `PRICE_26AXIS.json` の `rows`：m・axis・physical_id（configuration `4d6e2c9c…`の各round）・q・R・N（VP）・C_ref（exact）・N×C_ref。全行がモデル適用（`direct_observation_at_this_q_R=false`） |
| 2列の比較 | `comparison`：VP・FULL_PF__B_PRICEごとに、LD5・LD3の総額（exact）、差、比、切片・head・eventの内訳と差、重み、LD3総額の出所、`Q3_for_LD5=NOT_EVALUATED` |
| 較正cellの直接値 | `calibration_cell`：2 path × X/Yのcompiled RZ（34,315・34,314）、receipt locator（raw `f78ec5c2e` の `run01/COMPILE_RECEIPTS.json`） |

**IDの区別**
- 較正cellのID `PAULI_PRICE_CELL:53f1503d…`（q＝8、R＝4）は、m3（`RMSE:c943efea…`、R＝1）とも、m12（`RMSE:795c5535…`、q＝4096）とも別物である。
- cellのreceiptを、roundの価格receiptに付け替えない。

## 結果の要点

| 列 | LD5 | LD3 | 差 | 比 | head差 | event差 | 切片差 |
|---|---:|---:|---:|---:|---:|---:|---:|
| VP | 872,241,518.14 | 1,051,862,454.81 | −179,620,936.67 | 0.829 | +314,941,320 | −494,560,744.67 | −1,512 |
| FULL_PF__B_PRICE | 541,743,905.57 | 541,936,295.95 | −192,390.38 | 0.9996 | +195,870,600 | −196,061,526.38 | −1,464 |

- H5−2685＝＋1,530 RZ/step。
- FULL_PF__B_PRICEでの差（−0.04%）は、本体のばらつきから見た目安（約±0.5M RZ）より小さく、同額とみなす。
- round別では、LD5はm0〜m10でLD3より1.11〜1.56倍高く、m11・m12で安い（0.86倍・0.60倍）。

## 使い方の注意

- **外挿の範囲**：直接確かめたのは q＝8・R＝4 の分解だけである。q＞8とR＝1・2への適用はモデル値である。
- **eventあたりの費用**：E5（9.107）はE3（6.747）より大きく、Rの小さいstepほど1 eventあたりが高い傾向と整合する。R＝1のroundでは、E5がやや低めの可能性がある。
- **Q3**：LD5のQ3適否は、VP・FULL_PF__B_PRICEのどちらも未評価である。上の総額は、Q3に適合したと仮定したときの条件付き値ではない。

## 次の段階（未実行・要許可）

1. **B**：LD5のtotal viewと、2列でのQ3。
2. **B**：LD5の価格を使った配分探索。
   - LD5はm11・m12が安いので、LD3と比べて、そこへshotを寄せた配分が有利になりうる。
   - 分割（LD3とLD5）と配分は、一緒に比較するのが妥当である。
3. **A（必要なら）**：q＞8またはR＝1でのLD5の分解の確認。

## ロードマップへの影響（本体は編集していない）

| 箇所 | 更新の提案 |
|---|---|
| PR-Pauliの行 | 「LD5の固有価格：A5＝311、H5＝4,215（LD3＋1,530）、E5＝9.107 RZ/event（Q8R4、4 compile、分解成立）。VPではLD3の0.829倍。FULL_PF__B_PRICEではほぼ同額（0.9996倍）。LD5の13 round信号を取得済み。Q3は未評価」 |
| 次の判断 | BのLD5 Q3と、分割と配分を合わせた探索 |
| 依存関係 | B：LD5 total view・Q3 → B：LD3・LD5の配分探索 |

**mandatory STOP。**
