# B2・統合担当への引渡し

PR-Pauli／PR-DFの価格適用判定を追加した。保存モデル156軸・12総額をexactに再現、17検査PASS。新規sampling・build・compile・fit・Q3は0件。有限cohort価格とモデル参照価格の資格を保持する。

固定SOURCE：`c1f95e0806f5478a1ee5ce9ecb148ce9d10461aa`。

| 用途 | 同SOURCEのpath |
|---|---|
| Python API | `src/trottertracks/pr_price_rules.py` |
| 固定規則・価格・configuration・receiptのlocator登録 | `configs/pr_price_rules/h4_saved_rules.json` |
| JSON CLI | `scripts/quote_pr_price_rules.py` |
| 対象を絞った保存再現・人工検査 | `tests/test_pr_price_rules.py` |

公開成果の機械可読入口は、このHANDOFFと同じ最終公開commitの [C_HANDOFF.json](C_HANDOFF.json)。モデル・直接価格の元commitは登録表から個別にたどれる。実装SOURCEと今回の成果公開commitを区別する。

## そのまま使う入口

```python
from trottertracks.pr_price_rules import PriceRules, frac

rules = PriceRules.from_repo(repository_root)
result = rules.quote({
    "configuration": generated_configuration,
    "implementation": execution_contract,  # 登録表の候補固有契約と照合
    "policy": "MODEL",                    # 保存モデル総額の再現用
    "shots": {"X": N_X, "Y": N_Y},        # 任意、各round順
})
if result["complete"]:
    # B2の既存cost(vector, prices)に渡せるexact値。探索は呼ばない。
    prices = [frac(row["unit_price_RZ"]) for row in result["rows"]]
else:
    missing = [row for row in result["rows"] if row["unit_price_RZ"] is None]
```

`rows`の順序はm順・X/Y順。`total_RZ`は全軸が揃ったときのみexact有理数、欠測があればnull。金額の形式は `{numerator,denominator}`。B2の `frac`／既存exact費用計算に接続でき、Cのモデルを再fitする必要はない。

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/quote_pr_price_rules.py \
  --input docs/research/d_pr_price_rule_applicability_20261116/EXAMPLE_INPUT.json \
  --output /tmp/pr-price-quote.json
```

[EXAMPLE_INPUT.json](EXAMPLE_INPUT.json) はLD7の固定configuration locator・実装契約・V0を使用し、[EXAMPLE_OUTPUT.json](EXAMPLE_OUTPUT.json) はその出力。新候補にはB2が生成したconfiguration本体を渡すか、CLIの`configuration_source`を置き換える。実装契約を省略して暗黙にloweringを決めることはしない。`--template DF_LD7`／`--template PAULI_LD5`で、各既存契約とlocatorの入力雛形も取得できる。

## 出力の扱い

- `DIRECT_FINITE_COHORT`：一致した実回路条件の保存平均。`direct_alternatives`の役割・cohort・n_paths・sourceから元receiptへ戻れる。X/Yを独立pathとして足さない。
- `MODEL_REFERENCE`：候補固有規則の適用。`model_alternative`のsource、較正範囲、外挿変数、event stepの変化、caveatsを保持する。
- `NEEDS_PRICE_EVIDENCE`：該当軸の不足理由と係数／量、埋める較正を`needs`から返す。候補に他prefixの価格を付けない。

shot配分やκの宣言だけでは単価を変更しない。実際の整数化後schedule・law・物理条件を比較する。実stepや公式、分割、scalar、wrapper、compiler変更は別途の根拠が必要。DF LD7はR=1を外れるとhead／tail傾きの分離不足、DF LD1の新点はE_I[S]不足、DF LD9はm6の直接2軸のみ。PauliのA監査は較正cellをm=Noneのまま保持し、そのcellを長roundの直接価格とは扱わない。

`policy=MODEL`で保存単価・総額を再現できる。`AUTO`は直接平均が一意なら優先するため、holdout点ではモデル表と異なることがある。`DIRECT`で直接証拠のみを要求できる。複数cohortは自動poolせず`direct_cohort`で選ぶ。これらの選択から精度適合や期待値の保証は生じない。

## 次回差分照合・ロードマップ

次回利用前に、同じ候補の新しいモデル・直接receipt・A監査・compiler/loweringの変更と、B2側の候補生成入力を確認する。既存入力は登録表の固定locatorで再利用する。新しい根拠は登録表を明示的に更新するまで自動取得しない。変更SOURCEの同等性は本入口が証明しない。表の更新時は保存価格・総額のexact再現検査を通す。

今回B2の進行中optimizer／候補生成jobを読取観測したが、別worktree・出力を変更していない。B2本体へのmerge・検索・列挙・科学実行は今回行っていない。本入口を呼ぶ接続作業はB2側で行える。ロードマップ本体の更新は統合担当へ渡す。

許可された技術整備・検査・公開で終了。保存信号・配分・独立holdout・固定モデル・旧STOPを保全し、追加価格取得や科学callへは進まない。
