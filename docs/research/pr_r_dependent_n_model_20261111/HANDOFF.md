# Codex B向け：PR R依存N provider

基点は`89c91c692e1d35de94d2e7afbe890aabe6d80b7e`。このbranchの追加module/CLI/testだけを取り込める。既存kernel・fixed科学runner・C interfaceは変更していない。実科学計算は別途ユーザーの固定範囲承認後に開始する。

## metadata（科学計算なし）

```python
from trottertracks import pr_r_dependent_n as rmodel
base = rmodel.saved_task(root, "H4")
proposal = rmodel.with_round_R(base, {11: 8, 12: 16}, adapter_source)
task = proposal["task"]
reuse, excluded_old_R = rmodel.partition_saved(task, saved_N_records)
```

CLIもmetadataのみ：

```sh
python -B scripts/plan_pr_r_dependent_n.py --system H4 \
  --r-by-round '{"11":8,"12":16}' --output <fresh-proposal.json>
```

上記Rは使用例であり、科学採択ではない。sourceには固定commit/module pathを記録する。H6はm≥4、R≥11。H4はm0..12、R≥1。beta/alpha/h/prior/decoder、H/state/binding/contract/profile、PF menuは変更不可。旧Rのsignal/bias/B/NやCは新candidateへ付け替えない。

## 将来の科学接続

`PYTHONPATH=src:scripts`を設定する。Bの既存context loaderをそのまま呼ぶfactoryを渡す。

```python
# 承認された科学範囲でのみ呼ぶ。
def context_factory(task):
    ctx, metadata = existing_bound_context_loader(task)
    ctx.scientific_binding = task["binding"]  # loaderのH/state/hash検査後
    return ctx, {"state_energy": metadata["state_energy"],
                 "state_residual": metadata["state_residual"]}

provider = rmodel.RBatch(task, base, numerical_source, context_factory,
                        reference_factory=bound_saved_reference)
result = pipeline.run_missing(task, reuse, [], n_provider=provider,
                              c_provider=None)
```

H4 factoryの再利用先は`h4_df10_rpe_batch_provider.prepare/load_saved_context`（保存H4専用loader、metadataはstate_energy/state_residual）。H6は`h6_rpe_batch_provider.load_saved_context(root, existing_reuse_plan)`（metadataのsolver_energy/solver_residualを上記共通keyへ写す）。H4 layoutをH6 loaderへ渡さず、completion/scalar/tail/fragment順・保存sectorを保持する。contextは既存action kernelの`ham/state/matrices/tails/full/half_steps`を持つ。scientific_bindingの代入自体は検証ではなく、既存loaderの検査完了後の宣言である。

`bound_saved_reference(ctx,T)`は次のdictを返す。input bindingと実時刻Tを照合する（binding中の歴史的T=.8とは別）。

```python
{"binding": task["binding"], "T": T, "source": pinned_saved_reference_source,
 "evidence": {"reference": {"real": ..., "imag": ...},
              "independent_reference": {"real": ..., "imag": ...},
              "signal_difference": ..., "certified": False}}
```

referenceはH/state/Tが同じならRに依存しない。保存された2つのcomplex値の差も検査し、allowanceには報告差と観測差の大きい方を用いる。保存経路の資格/SOURCEをBが検証し、旧reference不在時に科学再計算へfallbackしないcallbackを使う。`reference_factory=None`は既存の2経路新referenceを起動する意味なので、このcallも承認範囲に含める。m4の単一保存spectral referenceを新2経路証拠へ昇格させない。

providerに渡せるのは`task.source.R_by_round`へ明示した不足PR候補だけ。範囲外PRをmissingとして渡しても計算せず拒否する。PFやH6 m0..3不足は先に保存recordを用意する。exact identityの保存FAILは`run_missing`でSTOPし、別Rを自動探索しない。candidateのNUMERICAL_GATE_OR_RANGE_UNQUALIFIED/位相・半径FAILはN=nullで保存する。source/context/reference不一致は例外・STOP。Rの修正・精度変更・再実行は自動で行わない。

## 出力と比較

common N recordをそのまま返す。candidate ID、m/T/q_RPE/q_PF、R、signed weights/allocation/event_count、binding/configuration/profileは従来interfaceに含む。diagnosticsにはsignal/bias/logB/B/attenuation、reference SOURCE、`R_AWARE_EMPIRICAL_ACTION_COUNT_V1`のu/margin/stages、Integer checkを追加する。actual N/実装law誤差・厳密ground・rigorous/full_RPEは未資格。失敗理由・計算完了数も保持する。

巨大NはPython intまたは`diagnostics.N_X_decimal/N_Y_decimal`で全桁を保持する。既存binary64 Hoeffding値に不足がある場合だけ高精度ceilへ上方向補正する。source hash/recordを保存し、未知誤差をゼロとしてPASSさせない。

旧固定R Nは旧経験的allowanceの証拠として保存したまま使える。ただしRによる微小差を比較する科学検証では、保存baseline signalも**同じ新allowance policy**でscalar再解析する（別途承認する範囲）。これを旧Nの無条件流用や原本上書きで代替しない。

C record schemaの変更はない。新Rは新candidate/law/event構成なので、対応するCのcoverageと資格はC担当が改めて判断する。B/attenuationをCへ掛けず、複数R taskを同一全費用和へ重複投入しない。今回の作業で価格lookup/予測/fitや実Gは開始していない。
