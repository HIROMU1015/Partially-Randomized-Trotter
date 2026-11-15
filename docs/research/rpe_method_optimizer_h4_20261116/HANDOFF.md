# 統合担当・A/B/C/D/D2へ：方式別RPEコスト最適化の最初の実装

正本は[REPORT](REPORT.md)。branch `codex/b2-rpe-method-optimizer-20261116`（Cの集計branch `08c6a694b` から作成）。新規の科学計算・compile・Q3は0件。

## 何ができるようになったか

一つのコマンドで、H4の問題入力から次のものを出せる。
- 方式別（Full PF・B0I p7・PR-Pauli・PR-DF）の評価済み集合内の最安適合構成：構成、round別X/Y配分、MSE上下界、条件付き費用。
- 停止理由と探索範囲。
- 探索規則が選んだ次の新規Q3要求。
- 信号・価格・configurationが足りない候補と、その不足内容（機械可読）。

```bash
cd /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/b2-rpe-method-optimizer-20261116
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  /home/AbeHiromu/projects/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/venv/bin/python -B \
  scripts/run_rpe_method_optimizer.py \
  --problem configs/rpe_method_optimizer/h4_df10_problem.json \
  --output docs/research/rpe_method_optimizer_h4_20261116/run01
```

新しいQ3結果は `--ingest results.json` で取り込み、次の要求を計算し直せる。

## 現在の出力（保存結果のみ、新規Q3予算0）

| 方式 | 最安適合構成・費用 RZ | 次の要求（構成・X/Y各軸・費用 RZ） |
|---|---|---|
| Full PF | S2・16h/7、559,457,386 | M5・16h：[30,28,26,24,22,20,17,15,13,11,9,7,3]、541,075,202 |
| B0I p7 | S2・16h/7、395,962,678 | M5・16h：[30,28,26,24,22,19,17,15,13,11,9,7,3]、382,916,152 |
| PR-Pauli | L_D=5、541,743,905.57 | L_D=3：[48,45,42,39,36,33,29,26,23,19,16,8,1]、483,460,003.67 |
| PR-DF | L_D=7、1,235,340,731.91 | L_D=7：[48,45,42,39,35,32,29,25,22,19,16,12,7]、1,134,606,918.91 |

最安点は、Cの保存集合内の方式別最安点と一致する。次の要求は、探索規則（`FINITE_LOCAL_DESCENT_V1`）が選んだものである。要求の詳細は `run01/RESULT.json` の各方式の `pending_request` と `NEEDS.json` の `pending_Q3` にあり、次の項目を含む。
- 配分の全26軸
- どの適合点からどう変えたか
- 既存Q3経路の案内
- 次点の4候補

## 動作確認

- Cの最安点を4方式とも再現した。12 configurationの自分の単価が、保存総額をすべてexactに再現した。
- REPLAY：p7 S2・Full PF S2とも、m12各4 shotの適合点だけを初期知識にして、より安い適合点（m12=2）へ自分で進んだ。続いてm12=1の不適合を受け、次の要求を決めた。正解の列や経路はcodeに埋め込んでいない。
- INGESTで、Q3結果1件を受け取った後に次候補が変わることを確認した。FIXTURE（人工MSE）で、最初の改善の後も停止規則まで続くことを確認した。
- OWN_PRICEで、PRの6 configurationについて、同じ配分を各configuration自身の単価で比較した。保存点の費用はすべて一致した。

## 人手が残る箇所（REPORT §5）

1. 新規Q3の実行：このversionには実行器が無い。探索1回分の対象と予算の採択、既存callerでの実行、`--ingest` での取り込みが必要。
2. 不足のある候補の信号・価格の取得と、Cの集計INPUTSへの登録。
3. Cの集計に無い証拠は見えない（例：PR-PauliのL_D=1の保存信号）。
4. 問題入力（H4）の作成と、探索パラメータの決定。
5. PFの新しい幅のconfiguration生成。
6. 別のHの保存結果の集計。

## 次段階の案（採択は別判断）

- 探索1回分の採択：方式ごとの新規Q3上限（例：各4件）、対象方式、停止規則。
- その範囲で、要求を既存のQ3 callerへ渡す実行器を最適化器につなぐ。B・D2の既存callerのpatternを再利用する。
- 新規Q3の結果を、Cの集計とこの最適化器に自動で戻す。

## ロードマップへの差分（本体は統合担当が更新）

- 「方式別の自動最適化」に最初の実装ができた。保存結果の範囲では、最安点の再現と次の要求の自動決定まで動く。
- 新規Q3・信号・価格の実行と取り込みは、まだ人手である。

Cの集計の入力・code、原結果、旧STOPは変更していない。公開・remote照合後に **mandatory STOP**。
