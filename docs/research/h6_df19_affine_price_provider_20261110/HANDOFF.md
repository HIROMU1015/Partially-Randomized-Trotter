# Codex BへのH6 Affine provider取込み

Cのコード/テスト固定SOURCE：**`0b5c493f2b2728b0c7c6c848d2cbf084db8f7f76`**。
互換を試験したB SOURCE：`cb2b1d6b4309ca7d705f397e25e27d14e8ba125a`。
CはBのbranch/worktreeへ書き込んでいない。これは技術部品の引継ぎで、全13roundの科学実行・外挿採用・N×C/Gの承認ではない。

## 取込み

Bの作業directoryで、既存差分を確認してから次の2ファイルを選択して取り込む。C branch全体のmerge/cherry-pickは不要。

```bash
git fetch origin codex/resource-eval-h6-df19-price-design
git restore --source=0b5c493f2b2728b0c7c6c848d2cbf084db8f7f76 -- src/trottertracks/h6_df19_price_provider.py tests/check_h6_df19_price_provider.py
PYTHONDONTWRITEBYTECODE=1 python -B tests/check_h6_df19_price_provider.py
```

このtestは固定B snapshotを一時領域で再現する。Bの最新コードが進んでいる場合は、その実環境での追加の技術結合確認をBが担当する。Cの互換PASSを将来のB HEADへ自動読み替えない。単独C checkoutでtestを再現する場合は、B固定objectがなければ`git fetch origin codex/pf-pr-rpe-cost-integration-20261110`も先に行う。係数/保存価格のGit objectはC branchから取得する。object不足時にcompileや新予測へfallbackしない。

## 共通interface

```python
from trottertracks import pf_pr_rpe_pipeline as p
from trottertracks.h6_df19_price_provider import h6_saved_price_provider

store = p.SavedArtifacts(repository_root)
provider = h6_saved_price_provider(store.read, pipeline=p, selections={
    'ef41b65942d56977d6cd': {
        'kind': 'model_prediction',
        'price_unit': 'cohort_mean',
        'cohort': 'H6_PRICE_V2_20261110_Q4_INDEPENDENT',
    },
})
# 将来の許可済みpipelineで使うcallback。
# provider(task, requested_rows) は各行X/Yの保存lookup recordを返す。
# p.run_missing(..., c_provider=provider) にそのまま渡せる。
```

上記は接続例。Cが実Nや実Gを与えて実行した例ではない。PF既定は保存direct。PRにunit/cohortの既定はなく、未指定はnull。modelとdirectを同一cohort scenarioへpoolしない。

共通pipelineの変更は必須ではない。便利な`p.h6_saved_price_provider`入口を用意したい場合だけ、Bが次の薄いwrapperを追加できる。

```python
def h6_saved_price_provider(read_json, *, selections=None):
    from .h6_df19_price_provider import H6AffinePriceProvider
    return H6AffinePriceProvider(read_json, pipeline=sys.modules[__name__],
                                selections=selections)
```

`identity/measurement/validate_task/candidate_id/SavedArtifacts.read`はB自身のものを渡す。N/G会計器や共通coreの差分はC側に含めていない。B snapshotの既存`UNAVAILABLE`/`MODEL_PREDICTION`/`UNVALIDATED_EXTRAPOLATION`と期待価格policyをそのまま使用できる。

## selectorとcoverage

| selector | 動作 |
|---|---|
| `kind='direct_compiled'` | 固定domainの保存directだけ |
| `kind='model_prediction'` | q2/q4の保存Affine scalarだけ。式を起動しない |
| PF `price_unit='deterministic'` | 既定。保存1回路の価格 |
| PR `price_unit='cohort_mean', cohort=...` | 保存cohort平均/その保存forecast。全law平均ではない |
| PR `price_unit='path'` | directのみ。`path_index,event_digest,digest_kind='private_events_file_sha256'`必須 |
| `price_unit='law_mean'` | 常にunavailable |
| 未取得q・対象外・保存値のないunit/kind | null。自動再予測・再sampling・compileなし |

PR raw cohort名：q1は`EXPLORATION` n9または`CONFIRMATION` n33、q2は`H6_PRICE_V2_20261110_Q2_CALIBRATION` n8、q4は`H6_PRICE_V2_20261110_Q4_INDEPENDENT` n8。cohort平均同士をpoolしない。path digestはV1 private bindingsまたはV2 `COMPLETION.json.private_path_bindings`にある**file SHA256**であり、event配列をJSON化したhashではない。requestに別payloadを添えて内容を証明したことにしない。

q1のmodel lookupはnull。frozen JSONのq1価格は校正anchorで、保存済み予測quoteとして再ラベルしていない。q2 modelは`CALIBRATION_Q2`・資格false。q4だけ`FINITE_INDEPENDENT_Q`・モデルdomainの有限検証資格true。qPF>4のlookupはnullで`UNVALIDATED_EXTRAPOLATION`を保持する。

返答の`cohort`はB用scenario key（`DIRECT:`/`MODEL:`/`UNAVAILABLE:`＋raw cohort、pathならindex付き）、`source_cohort`は元cohort。`run_missing`の返答`prices`を保持し、nullを取得済みへ数えず、同じ未資格返答を再呼出ししない。selector/policyを変える場合はB側で明示的な要求として扱う。

## 科学条件と採用policy

H6保存DF19/tol-only1e-8/weighted-Hermitian completion・sector(3,3)、保存H/state/snapshot、元順fragment/11 signed m5_best piece/scalar、P=0。δ=1.6、PF Full19、PR p7/R16/K2、元finite law・compiler・測定付きfull Hadamard wrapper。bindingのT=.8は元inputのreference metadataで、candidate.Tへ読み替えない。要求のT/qPF=1.6を確認する。

contractはB既存形式：`completion='saved_weighted_Hermitian_completion'`、`fragment_order='saved_snapshot_order'`、`state_preparation_cost=0`、明示`input_source`、`preparation_binding.snapshot`と元`provider_source='c31ef424175c02379eafebe8fc6c7548c1d9aabe'`。method別prep hashを渡す場合はPF/PRの一致が必要。共通taskでは省略して固定snapshot/prefixから解決するか、selectorでcandidate別に渡す。別compiler/backend/target/layout等へ適用しない。

m0〜3のp5/R32や別δは本providerのdomain外。Cのlookup missを理由にAffineで全13roundを補完しない。`qualified_for_candidate=True`はfull-RPE採用・状態の厳密ground資格・一様誤差上界を意味しない。

- 既定`qualified_expectation`：PR有限cohortをlaw期待値へ昇格させない。
- 明示`finite_cohort_scenario`：保存cohort scenarioに限る。q2校正modelの資格falseは維持する。
- PF direct/q4条件付きmodelが値を返しても、全N/Cと科学policyの充足をBが別途確認する。新signal/NやGは今回の承認外。

## 明示数値APIと未資格

`predict_affine(request)`は将来の科学実行入口。`kind='model_prediction'`、PFは`deterministic`、PRは`model_mean`だけを受ける。PRの`model_mean`は実cohort/path/law平均ではなく、B recordの`expectation='UNQUALIFIED_MODEL'`・`n_paths=None`で採用を止める。q>4の数値返答はさらに`UNVALIDATED_EXTRAPOLATION`。このAPIを`c_provider`のmiss補完へ接続しない。Cは実H6の未取得qでは実行していない。

将来、ユーザーが許可した未取得qの数値予測やPR model_meanを実Gへ採用する場合、現B policyは`UNVALIDATED_EXTRAPOLATION`/`UNQUALIFIED_MODEL`を拒否する。共通部分の採用policy変更が必要ならBが別scopeで担当する。今回のC接続はその資格を緩めず、finite cohort/lawタグの付替えで回避しない。明示数値quoteは`provider.pipeline_record(task, row, axis, quote)`でB recordへ変換できる。

保存nominal係数共分散・保存q4予測平均共分散・paired誤差統計は参照可能だが、完全な係数不確かさ/joint covariance/長q systematic/order2価格/全law期待値/total CIはnull。modelの`shared_calibration_group`と校正directの`calibration_data_group`が同じ場合、独立標本として扱わない。対応X/Yも共有する。Bの全体Sigmaへ独立仮定で0を補わず、必要な依存を別途資格する。nominal部分を全体CIへ変換していない。

Cの作業は[10技術テスト](TESTS.txt)・[保全確認](CHECKS.txt)まで完了。B取込み・最新環境での技術確認は未実行。q8の追加取得、未取得qの実予測、全law推定、全RPE科学計算には、ユーザーの別途の具体的な実行範囲承認が必要。公開照合後mandatory STOP。
