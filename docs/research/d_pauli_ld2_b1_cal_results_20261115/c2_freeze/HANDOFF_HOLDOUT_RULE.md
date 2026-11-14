# C2 → ユーザー・D2：LD2 holdout規則案の引渡し

**[規則案](HOLDOUT_RULE_PROPOSAL.md)：記述的検証を推奨。提案・未採択、未実行。**
ユーザーに判断いただく対象は、m5-X、m5-Y、m6-X、m6-Yの4件について、誤差を報告し、
モデルのPASS/FAILを付けない運用の採否である。数値許容差は提案していない。

m5は同軸の固定fresh 2 pathの算術平均、m6は同軸の固定fresh 1 pathの観測値。
`d=P−O`、`A=|d|`、符号付き相対誤差`d/O`、相対誤差`A/O`を用いる（O>0）。
O=0は相対誤差を未定義とし、6 receipt・3 path・4比較単位を区別する。
取得・実装不成立と精度不適合、campaignの完備性とモデルの精度判定を分離した。
規則案§3は、次回D2指示へ転記できる比較式・状態・STOP条件である。

## 固定資料と保全

| 役割 | commit / path |
|---|---|
| 凍結正本 | `b74e4e536392c2bbe94cd371436615fc56c0564f` / 同directoryの`run01/FROZEN_MODELS.json` |
| 既存D2引渡し | `87c525a7b72d70314d87f59042efd663d5566735` / `HANDOFF_D2.md` |
| D2較正 | `5dac74959ed9ea49465d393fc792ef5b4727edba` / 親directoryの`HANDOFF_C.md` |
| cell/holdout metadata | `043c7567451e8f502a1cf0f425d32efe0a50f5dc` / `docs/research/d_pauli_b1_caller_preparation_20261114/CAL_AFTER_W_METADATA.json` |

既存C2の関連履歴・引渡しとCのPRICE_VALIDATION_PLAN§7を確認した。
既存資料は許容差の選択肢・未採択状態までで、今回の4単位の記述式・campaign状態・
D2転記条件をまとめた同等文書は確認範囲に見つからず、規則案だけを追記した。
閾値の用途側の根拠と区間契約が不足するため、既存の2%や5%を自動採用しない。
既存の2モデル・4予測・fit SOURCE・旧HANDOFF・raw・receipt・STOPは変更していない。

## ロードマップへの影響

- 追加成果は「4単位の独立holdoutに向けた記述的比較規則の提案」。
- 科学的状態は「LD2有限cohortで較正・係数と予測を凍結・独立holdout未評価」のまま。
  新しい精度証拠、モデル資格、実order2・非自明位相coverageは得ていない。
- 残る判断はユーザーによる本案の採否。採択しても、取得・開封・比較計算の実行許可は別。
  数値的なPASS/FAILが必要なら、開封前に用途・精度予算と閾値を別途採択する。
- Cへのmerge・再fitは不要。ロードマップ本体は統合担当が扱い、本作業では編集しない。

今回は文書と固定metadataを静的に確認しただけで、holdout観測値の取得・開封、新規sampling・
compile、数値再解析、再fit、実データ／人工データの比較計算は0。
文書のリンク・対象metadataと既存成果の不変性を確認し、計算testやrunnerは実行しない。
規則案をC2 branchへcommit/pushして**mandatory STOP**する。
