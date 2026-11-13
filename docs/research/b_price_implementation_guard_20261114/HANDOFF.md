# B本線への取込み手順：価格実装guardの直接経路接続

結果と根拠は [REPORT](REPORT.md) にある。**このpatchは、まだB本線に適用していない。** 適用するかどうかと、その時期はBとユーザーが決める。

## 適用条件

- Bの固定DP runner（`scripts/run_h4_dp_risk_fixed.py`）は、実行時に`rpe_shot_allocation.py`と`pr_rpe_rmse_controller.py`がENGINE `06db9aef`から変わっていないことを確認する。
  - このpatchは`rpe_shot_allocation.py`を変更する。
  - そのため、DP実行（run01）と結果のcommitが終わってから取り込む。実行中のworktreeには適用しない。
  - 取込み後のtreeでは、このrunnerを再実行できない。runnerはone-shotであり、再実行は予定にない。
- 固定科学契約が旧SOURCEを指定している実行は、新しいtreeで再開しない。

## 手順（B head `ce0517de6`で確認済み）

```bash
cd <Bのworktree>            # DP結果をcommitした後
git fetch origin            # pruneは付けない
git cherry-pick 434458be2f25b200c54636d941b6647d62df9859 \
                540ea9e04cd1be6290a8b089403a54bf6f2aea07 \
                3b7b5b4f81c6e8b34e5a5c1de93cb1fb48a2df15
# cherry-pickの代わりに:
# git show origin/codex/b-price-implementation-guard-20261114:docs/research/b_price_implementation_guard_20261114/B_guard_connection.patch | git apply
cd tests
/home/AbeHiromu/projects/h6-environments/20261011/venv/bin/python -B check_rpe_price_implementation_guard.py
/home/AbeHiromu/projects/h6-environments/20261011/venv/bin/python -B check_rpe_shot_allocation.py
/home/AbeHiromu/projects/h6-environments/20261011/venv/bin/python -B check_rpe_deterministic_risk.py
```

- 変更するのは3 fileだけ（`rpe_shot_allocation.py`、新規`rpe_price_implementation_guard.py`、新規test）。B headでは、3 commitのcherry-pickがconflictなしに当たり、生成されるblobは本branchと同一だった。
- I1/I2/E1修正（`3a27f99fe`）とは独立に適用できる。ただし、I2の既存5 file失敗を解消するには、その修正の取込みも必要である。
- 回帰一式は [run_regression.sh](run_regression.sh)（`tests/`で実行し、出力directoryを引数に渡す）。

## 正式なG計算での呼出し

```python
from trottertracks import rpe_shot_allocation as W
from trottertracks import rpe_price_implementation_guard as R

out = W.compare(config, allocations, evaluations, prices, cohorts=cohorts,
                statistical_contract=contract, price_extension=extension,
                price_bindings=bindings, interval_contract=interval,
                root=REPO_ROOT, comparison_policy=policy)          # guardは算術より前に動く
one = R.cost(REPO_ROOT, config, n_packet, prices, cohorts=cohorts,
             price_bindings=bindings, comparison_policy=policy)    # Cの G.cost と同じ順序
best = R.select(configs, [r['outer_evaluation'] for r in results])  # 共通policyを確認してから B.select
```

- `rpe_deterministic_risk.compare(..., root=..., comparison_policy=...)`は、そのまま`compare`へ転送される。
- `policy`の形と、`price_bindings`の`implementation`・`implementation_evidence`（固定Git JSONのassertionを選択するもの）は、Cの [HANDOFF](../full_pf_latest_price_guard_20261114/HANDOFF.md) の契約に従う。元の価格recordは変更せず、sidecarで渡す。
- policyなしでも`SYNTHETIC_ONLY`のfixtureは従来どおり動く。科学N packetは`MISSING_PRICE_COMPARISON_IMPLEMENTATION_POLICY`でpendingになる。
- controllerの`B.cost`と`B.select`を、科学packetで直接呼ばないこと。これらは算術primitiveとして残しており、guardを通らない。

## まだ決まっていないこと（ユーザー・C・Dの判断）

- 比較scenario（policy）の採択と、旧Full-H価格の扱い：新loweringで取り直すか、旧domainとして比較から外すか。
- 実保存価格に付けるsidecarと固定assertionの作成（producerのreceipt照合を含む）。
- 上の2つが揃い、別途科学実行の許可が出るまで、policy付きの実G計算には進まない。
