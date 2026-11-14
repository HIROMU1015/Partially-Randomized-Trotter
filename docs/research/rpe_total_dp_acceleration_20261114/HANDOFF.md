# 高速total DPの利用と科学SOURCE

実装：[rpe_total_dp_parallel.py](../../../src/trottertracks/rpe_total_dp_parallel.py)。性能・数学的同値性・制限は[REPORT](REPORT.md)。

## 直接呼出し

```python
from trottertracks import rpe_total_dp_parallel as P
engine = P.Accelerator(workers=8, outcome_batch=8)
result = engine(tables, h=h, target=target, scope=scope)
print(engine.observations)   # 各roundのactive/omitted states、pair数、worker数
```

`scope`は元APIの`source/maximum_frontier_states/maximum_transition_pairs`をそのまま渡す。科学的な有限範囲と計算worker数は別である。結果schemaは元total DPと同じ。extra performance情報を科学output dictへ加えないので、元kernelと全dict比較できる。

## 既存Q3/canonical callerへ接続

```python
from trottertracks import rpe_total_binding as T
from trottertracks import rpe_total_dp_parallel as P
options['total_dp_acceleration_source'] = {
    'commit': accelerated_execution_source,
    'path': 'src/trottertracks/rpe_total_dp_parallel.py',
    'workers': 8, 'outcome_batch': 8,
}
with P.installed(workers=8, outcome_batch=8) as engine:
    records = T.evaluate(root, canonical_bridge, signal_packet, allocations,
        evaluation_target=evaluation_target, statistical_contract=contract,
        source=execution_source, options=options)
```

この呼出しは実データなら科学実行であり、固定科学SOURCEを変更する。現行2件には未適用。新しい実行範囲とSOURCEを明示してから使う。

- contextを抜けると元関数に戻る。現在のprocess内だけを変更する。
- 同じprocessでnested/concurrent contextは不可。独立jobは別processを使用する。
- kernelの元geometry・guard・allowance・normalization・momentの意味は不変。
- transition budgetは元と同じ前状態prefixを処理する。outcome/task/worker数でmassを省略しない。
- 有理数は最後にFractionで正規化され、巨大整数N・probability massをfloatへ丸めない。
- zero/tieは元Z1＋T1のまま、signed-zero phase bitsを維持する。
- scope超過・mass不保存等のUNKNOWNは元kernelと同じ。
- 新しいN資格・G・精度filterを発行しない。

## 現jobの切替が必要な場合

元2 jobのDP state checkpointは未保存であり、途中の数値状態を安全に取り出して移植する実装はない。元processのkernelをhot patchしない。

切替をユーザーが明示許可した場合の具体案：

1. 自分の旧supervisor/workerとcollectorのみを対象にし、途中raw/partial/log/STOPを保全する。他担当には触らない。
2. 新SOURCEと同値性・新backendのprovenanceを固定し、別run namespaceを使う。
3. 同じPauli LD1/LD2・同じtarget/loss/IID、同じUSER_SUPPLIED vector、ρ/count/frontier/transition上限を各1件、計2件再実行する。
4. 元jobの開始分も履歴として残し、新しい2件を「最初の1回だった」ことにしない。無断retryなし。
5. 終了後はMSE/moments/残余だけを報告し、十分N・G・主推定量採択へ進まない。

現時点では技術実装と人工benchmarkだけ完了。**現行固定2 jobは旧SOURCEで継続中**。切替許可は未取得。

## 技術公開後の切替・実行結果

2026-11-14：ユーザーが旧run保全停止と高速版での代替2評価を明示許可した。固定条件は同一で、各1件・計2件が約235秒で完了。[結果](../pauli_total_q3_accelerated_results_20261114/REPORT.md)。上記の「未適用・旧job継続」は技術公開時点の履歴。旧2 jobは未完了のまま別branchに保全されている。新しいshot探索・十分N/G・主推定量採択へ進まずSTOP。
