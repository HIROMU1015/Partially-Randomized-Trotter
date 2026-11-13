# A → B・価格担当：全round physical signal API

2026-11-13 JST。[REPORT](REPORT.md)／[有限科学案](SCIENCE_PLAN.md)。実科学・保存数値再解析0。B正本は`3216d27a69ac8532d2a22e78f5f94491d202d238`。N/G selector・κ mapper・decoder・価格providerは追加していない。

## Bから受け取るもの

`PR_RPE_RMSE_V2` configurationと、全m=0…Mの元physical candidate対応を既存B APIで用意する。constructorは`pr_rpe_signal_plan.plan`を使い、Bのvalidatorをそのまま呼ぶ。`q_RPE/T/qPF/actual_delta/R/K/L/formula/weights/allocation`、whole configuration/identity/lawを保持する。新IDを作り直さない。

```python
from trottertracks.pr_rpe_signal_provider import (
    SignalBatch, load_h4_context, saved_h4_reference,
)
from trottertracks.pf_pr_rpe_pipeline import SavedArtifacts

store = SavedArtifacts(root)  # pinned JSON cache; constructor has no read/action
batch = SignalBatch(
    config, physical_by_round,
    b_validate=B._validate, b_exact_scalar=B.a.exact_scalar,
    source=fixed_numerical_source, root=root,
    context_factory=lambda cfg: load_h4_context(root, cfg),
    reference_provider=lambda job: saved_h4_reference(root, job, store=store),
    saved_inventory=metadata_only_inventory,
)
pending = batch.pending  # no array load, normalizer, signal or N evaluation

# ONLY after permission covering this fixed config/SOURCE/scientific scope:
# result = batch.run(fresh_output_directory)
```

`B`はcallerが固定SOURCEからimportした実controller。上のB APIはfixtureを本番へ持ち込む意味ではない。SOURCEは実行codeのcommit/path/recipeを示し、承認証明書ではない。API importだけでNumPy/SciPy/H/state配列をloadしない。

## 保存H4 factoryの前提

H4 DF10 / sector(2,2), dimension36、同H/state/snapshot/rank policyを元保存loaderで照合する。H・state・snapshotはS1＋S2と同じ保存条件、targetは全保存DF-H、stateは条件付き数値state、P0のまま。one-bodyはdeterministicに含める。

tail λ/scalarの正本はINDEX `7a8bc7f132e166390c943e6b32c92cc6abe1baf4:artifacts/resource_applicability/df_full_h_pure_rte_t20_opt/2026-11-09/SOURCE_INDEX.json`が示すH4 context。B `spec.split.lambda_source`は、次のdescriptorと完全一致する必要がある。

```python
{
  "index": {"commit": "7a8bc7f132e166390c943e6b32c92cc6abe1baf4",
            "path": "artifacts/resource_applicability/df_full_h_pure_rte_t20_opt/2026-11-09/SOURCE_INDEX.json"},
  "context_sha256": context_spec["sha256"],
  "selector": ["prefixes", str(L_D)]
}
```

λもそのfieldの保存値と一致させる。旧configurationのSOURCEだけを後付けで変えず、Bがこのdescriptorを用いてcanonical configurationを生成する。κの定義とmapperはBの元仕様を保持する。prefix1/2/3/5/7/9以外は追加norm生成へfallbackしない。

referenceはbba1508の`PHYSICAL_INPUTS.json:rounds[m].diagnostics.reference_pair`を同H/state/Tでlookupする。prefix/R/Kが異なるPRにも同full-H referenceを使えるが、別時刻や別stateを使わない。今回このlookupを実H4の新診断へ進めていない。欠測・pair gate失敗時は新reference actionを勝手に取得しない。

## Bへ返す値

| field | 契約 |
|---|---|
| B_configuration_id / B_candidate_id / B_identity / B_law_binding | B正本のコピー。physical candidate IDとは別に保持 |
| physical_context / candidate | 元H/state/profile/split/law/numerical recipe、qPF・R・K・全piece条件。全Mの文脈を落とさない |
| signal.raw | ideal finite-K lawの未補正complex mean。既存軸規約：X=Re(raw)、Y=Im(raw)、p(+)= (1+coordinate)/2。座標をclipしない |
| signal.corrected / exact_tail / reference | 有限normalization補正mean／同PFのexact-tail内部対照／全保存DF-H独立pairのreference。相互代入しない |
| signal.total_signed_bias / outer_PF_or_discard_bias / finite_RTE_bias | corrected−reference／exact_tail−reference／corrected−exact_tail。PRではtail discardなし。signal空間の量で、energy biasではない |
| log_B / B / attenuation | whole interrogationの有限式。N側で一回だけ考慮する。Cへ追加乗算しない。overflowのBはnull、logBと停止理由を保存 |
| allowance / allowance_evidence / numerical | 元R-aware経験式、logical event数、最大normの観測境界、normalizer ULP感度、exact-tail gate、closure、総gate結果。rigorous=false |
| reference_evidence / state_evidence / source群 | 元pair・state metadataと入力/数値/reference SOURCE。欠測はnull。ground資格を自動付与しない |
| N_X/Y / C / G / global_RMSE_bound / scientific_PASS_FAIL | 全てnull。nominal/十分N・global RMSE・winnerを返すAPIではない |

新status `LOCAL_FINITE_MEAN_NOT_RMSE_QUALIFIED`は局所数値取得だけを意味する。`NUMERICAL_FAILURE`やbatchの`STOP_*`を科学的RMSE FAILへ読み替えない。

保存mean受入・旧N packetのtranslationは既存A/B経路で行う。`SignalBatch`は保存inventoryがhitすると自動再解析をせずSTOPする。fresh mean部分だけを明示的なscopeで扱う場合は`evaluate(context, job, saved_reference, root=root)`が単一leafであり、whole-schedule検証済みjobを使う。異なるM/κ/sourceのpacketへ値だけを移植しない。

## 価格担当へ

B/physical ID、prefixとfragment順、formula/signed weights、T/qPF/δ、R・piece allocation・K・law、scalar一回、wrapper/axis/compiler/control/profile、P0を渡す。行列累乗は古典計算の高速化で、回路event数やcompiled Cを縮約しない。L/K/formula/幅/Rが変われば既存価格domainとの一致を担当側で判断する。finite-cohort価格とlaw期待価格も区別する。

## 未対応・次の担当

PR-Y4、H6初期例外、任意round別split、Kを跨ぐ統計/価格比較器は追加していない。必要になった時にB/各担当が最小接続を行う。今回新10＋旧7技術test PASS、科学実行0。

**ユーザー**：menu・exactκ/外側条件・有限signal実行範囲を選択/承認。**A**：承認後の全round signal。**B**：同一RMSEのN/decoder・資格・全N×C比較。**価格担当**：同identityのC coverage。公開照合後はA mandatory STOP。
