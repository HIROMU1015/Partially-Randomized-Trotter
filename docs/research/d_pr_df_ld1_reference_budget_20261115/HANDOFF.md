# 統合担当・B/C向け：LD1参照総額とPF 2基準の符号付き境界

**T_ref＝261,314,830,475.559717693 RZ**。26軸の固定shot加重をexact有理数で一回計算した。参照総額は両PF基準より大きいが、未知Δを0とせず、対象lawの順位は未判定。

| PF基準 | T_PF (RZ) | T_ref (RZ) | D_PF=T_PF−T_ref (RZ) |
|---|---:|---:|---:|
| B0I p7 S2（16h/7） | 593,555,352 | 261,314,830,475.559717693 | -260,721,275,123.559717693 |
| Full PF S2（16h/7） | 838,693,580 | 261,314,830,475.559717693 | -260,476,136,895.559717693 |

表示は小数9桁。exact値は次のとおり。

```text
T_ref = 329967563350580055747055/1262720385024
D_B0I = -329218068907969559898607/1262720385024
D_Full = -328908527870325298801135/1262720385024
Delta=T_LD1−T_ref は未評価
Delta<D_PF: LD1低費用、等号:同額、Delta>D_PF:PF低費用
```

- B0I基準：Δの上限を約 **-260,721,275,123.559717693 RZ** のexact境界より小さく示せればLD1が低費用、下限を同境界より大きく示せればPFが低費用。
- Full PF基準：Δの上限を約 **-260,476,136,895.559717693 RZ** のexact境界より小さく示せればLD1が低費用、下限を同境界より大きく示せればPFが低費用。

いずれも許容差や誤差の実証値ではない。26軸一律の許容差を導入せず、固定Nで加重した総差Δの片側証拠を次に求める。今回その証拠は取得していない。

## 固定locatorと読取

- **実行SOURCE `68a808872842e3325e8a20a9118f19117483b9ad`**：`docs/research/d_pr_df_ld1_reference_budget_20261115/evaluate.py` とINPUTS.json。数値処理前にcommit固定した。
- **元結果公開 `1fb47bf9291c9cc2328bbfe42a8b3d40f13befa5`**：同ディレクトリの `run01/RESULT.json#rows[0..25]`、`T_ref`、`boundaries[0/1]`。26行は各physical ID/m/axis/N、C_ref/weighted_termのexact分子・分母、元入力の{commit,path,selector}を持つ。CSVもexact比を保存。
- 計画/26軸join44f4e40f、凍結3cac53da、Q3 a82b0f0e、元C索引c99d511e（8eef0ad7 HANDOFF指定）、保存構造9d95d3ea、PF RESULT1002f1dfへ元参照で戻れる。SOURCEと公開consumer SHAを混ぜない。
- [REPORT](REPORT.md)／[RESULT](run01/RESULT.json)／[CSV](run01/AXIS_TERMS.csv)／[CHECKS](run01/CHECKS.json)／[資源](RESOURCES.txt)。最終公開SHAは最終チャットで示す。

対象lawは保存SOURCE ideal finite-RTE law I、比較目的は同じLD1固定候補の期待費用。C_ref＝保存E_I[S]のbinary64 exact値＋凍結残差モデルの参照値。m4は保存済み最終予測を使い、残差を二重加算しない。軸ごとの整数丸めなし。

保存モデルは元PCG64有限cohort較正に由来する。m0/m3/m6の6残差＋m4の2を再利用、新しい残差評価18件。m4を含む較正点以外の10 roundへの適用を区別し、とくにm7〜m12の移送は未検証。m4は独立holdoutのまま、観測値への差替え・fit再投入なし。X/Y一致を独立標本数2倍とは扱わない。

## 科学的境界・件数・ロードマップ

F0/range dcf905fcはI/J間の期待残差差の上界で、期待残差自体やΔ全体の上界ではない。有限cohort誤差/共有モデル誤差/長round移送/保存数値のずれは未評価のまま。実PCG64移送も今回0。Delta/Delta_bound/ranking/正式E[C]/CI/G/winnerはnullを維持した。

参照価格26＝再利用2＋新24、残差参照26＝再利用8＋新18、加重項26、総額1、PF総額再利用2、符号付き境界新2。有効は全件、欠測/保留0。科学実行1回・retry0、人工6検査PASS、26項/2境界/142 locatorの出力整合チェックPASS。新path/H4 build/compile、Q3、再fitは0。元full-wrapper作用/statevector未実施の範囲も変えていない。

GNU wall0.15秒、max RSS18,432 KiB、exit0。小さいexact算術なのでcacheを再利用して単一processで実施した。

統合ロードマップへの差分：参照総額とPF差額の未数値化を解消し、**大きい負の順位反転境界を超える片側Δ証拠が必要**と具体化。PF価格の再取得やA/B/C待ちは不要だった。参照上の大小を正式な期待費用順位へ昇格させない。正式E[C]/CI/G・winner、モデルの保証は未解消。ロードマップ本体は統合担当が更新する。

元SOURCE/raw・保存S/凍結係数・m4独立holdout・旧STOPを保全。指定branchへcommit/push・remote照合後、**mandatory STOP**。
