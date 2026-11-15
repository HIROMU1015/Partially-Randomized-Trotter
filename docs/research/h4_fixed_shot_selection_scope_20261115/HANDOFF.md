# 統合担当へ：H4固定shot比較の候補・配分と主張範囲

2026-11-15 JST、B。[REPORT](REPORT.md) §1が選定表、§2が研究結果用の短い主張文、§3が追加確認の分岐条件。新しい科学計算・費用比・Gの算出は0件。価格担当の完成待ちをせず、保存記録だけで整理した。

## 初回比較の入力

各配分の全13整数列と固定SOURCEはREPORTを正本とする。m0〜m12、X/Y各軸の配分であり、軸数や価格標本数を再乗算しない。

| 方式 | 初回に結合する物理候補・配分 | 選択の意味 |
|---|---|---|
| PR-DF | LD1・V0 | 価格証拠の先行した代表。LD2・V0もQ3適合。LD1の方式内費用最小は未確認。当初の比較選定理由は記録上未確認 |
| PR-Pauli | LD2・VP | W確認後、tail列の小さい較正計画を先行する記録（043c7567 §6）。LD1・VPも適合で、LD2の費用最小を示したものではない |
| Full PF | S2・16h/7・VD | B2のFull PF 3候補内で条件付き最小、838,693,580 RZ（1002f1dfの引用） |
| B0I p7 | S2・16h/7・VD | B2のp7内かつ決定論PF全6候補内で条件付き最小、593,555,352 RZ（同じ引用） |
| Full-H pure RTE | κ2/K2・VH1.5 | 保存3配分中の判断。V0不適合、VH1.5/VH2適合。同一非負価格で各軸Nの少ないVH1.5を初回に使う |

Pauli V0は旧opt-in診断として保全し、新しい閾値判定を作っていない。PF6はcomponent適合と、原API return・CPU/peak RAM欠測の部分保存を別々に記す。原論文nominal Nや十分N packetには置き換えない。

## C/D2/Aへの結合境界

- Cの0b4b2d94 ALIGNMENTとBの各C_RESULT_JOIN／INPUTSを再利用する。元configuration・round physical IDとtotal decoder用IDを区別し、元receiptを改名・上書きしない。
- 保存H/state/snapshot、候補固有split/辞書/κ/R/K/normalization/scalar、round・axis、対象law、compiler/lowering、測定付きfull controlled Hadamard wrapper/P0、生成SOURCEを既存bridge/guardで照合する。
- D2/A/Cは、ランダム化3方式の同law全26軸E[C]と不確かさ・適用範囲を供給する担当。本件では価格表を新設・再計算しない。有限path平均・構造E[S]・未資格モデルを全law期待価格へ昇格しない。
- PFの価格入口は1002f1dfの既存表。96直接＋60条件付き閉包、60軸のCI/誤差上界null・2%保証なしを維持する。価格総額は再集計しない。
- 全方式の実際のQ3適合配分を使う。同じshot数に揃えない。N差は資源差の一部だが、異なる配分・方式での最小Nを意味しない。

## ロードマップへの差分

基準ad6ab45bの§4.4・§5.1–5.2を再利用し、統合担当が次の解釈を追補できる。**ロードマップ本体はBで変更していない。**

1. 比較ラベルを「PR対Full PF」「PR対B0I p7」「PR対今回の決定論PF最小」に分ける。後二者の初回比較対象は同じp7 S2である。
2. PR-DF LD1／Pauli LD2は価格整備先行の代表であり、方式内費用最小の認定ではない。Pauli LD2先行には取得規模の記録がある。PR-DFの初期比較選定根拠は確認した資料では未確認。
3. 期待価格が揃えば報告するのは**今回評価したQ3適合候補・固定配分での条件付き費用比**。Full-Hは1.5倍を使い、2倍を初回基準としてPRの削減を大きく見せない。PFの最良候補をFull PFだけの最良と混同しない。
4. 初回差額を見て、別LDの既存Q3適合結果＋固有価格で結論が変わり得る場合だけ、価格証拠の追加要否を判断する。PF高次公式の別shot配分やFull-H未評価vectorは、方式最適化後まで主張する必要が生じた場合の別計画である。
5. 差が価格不確かさの範囲内なら、価格側の不足を先に特定する。一律の精度基準・全候補探索・新N保証の発行を初回比較の前提にしない。

必要な科学的境界は保存ideal raw/IID・solver proxy・採択Z1＋T1/Q3閾値の条件付きモデル適合。厳密ground/state、実samplerへの移送、数値certificate、十分N・global RMSE・G_qualified・方式全体のwinnerは新認定しない。AのδEは二重加算せず、元Minkowskiを保全する。

## 今回の保存状態とSTOP

追加はREPORT/HANDOFFの2文書のみ。SOURCE/path・引用整数列と金額・文書間整合を静的照合する。科学module・Q3/DP/二項確率/MC・cost/selectは起動しない。固定原結果・旧STOP・他担当branch/worktreeは保全する。

公開後 **mandatory STOP**。期待価格の新評価、費用比・Gの新規算出、候補選定の追加科学検証には進まない。後続の数値結合が必要になった時点で、対象と有限範囲を別途明示する。
