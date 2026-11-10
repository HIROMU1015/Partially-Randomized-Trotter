# A/C・将来B実行向け統合HANDOFF

[REPORT](REPORT.md)／[最小科学案](MINIMUM_SCIENTIFIC_PLAN.json)。基点7bca8f964にA 5ca2dce323、C 4cb61ef463の固定部品を選択取込みした。現作業は技術統合のみ。下記の数値APIは将来の別途許可範囲で使用する。

## metadataだけを利用する

```python
from pathlib import Path
from trottertracks import rpe_provider_integration as b

root = Path(repository_root)
packet = b.prepare_menu(root, 'H6', {12: [16, 32]})
menu = packet['menu']
inventory = b.r.coverage(menu, packet['shots'], packet['prices'],
                         cohort_selection={})
```

`prepare_menu`はCの公開N loader、Aの固定base task、BのR-only builderと価格catalogを再利用する。NPZ/state/private signalをloadせず、N/Cの新予測・scalar評価・積和をしない。H4 Nは89c91c692の`N_HANDOFF.json.shots`、H6 Nはf4c0bdabdの保存26 record。巨大整数はcopyだけで、別RのIDへ付け替えない。除外した旧Rは`excluded_old_R_N`へ残す。

`cohort_selection={}`はscenario未選択という意味で、価格不存在の宣言ではない。既存値は`price_contexts`に残り、Nの保存状態、C availability、選択・資格不足を分離する。固定全13round inventoryは`report_rpe_nc_coverage.py --output-dir <fresh-dir>`で再現できる。これは実G/R比較の入口ではない。

## AのN providerと一括入口

```python
# 以下は実科学実行の接続例。今の技術許可だけでは呼び出さない。
ctx_factory = b.bound_context_factory(root, 'H6')
n_provider = b.RMenuBatch(menu, packet['base'], numerical_source,
                         ctx_factory, reference_factory=bound_saved_reference)
c_provider = b.FixedSavedPrices(b.p.SavedArtifacts(root).read, 'H6',
                               selections=explicit_price_selectors)
batch = b.run_missing(menu, packet['shots'], packet['prices'],
                     n_provider=n_provider, c_provider=c_provider,
                     cohort_selection=explicit_scenario_selection)
# 別途許可された実費用/R比較には既存exact R選択を束ねた一括入口：
result = b.run_evaluation(menu, supplied_N, supplied_C,
    n_provider=n_provider, c_provider=c_provider,
    cohort_selection=explicit_scenario_selection,
    price_policy='qualified_expectation',
    simultaneous_bounds=supplied_bounds_or_none,
    schedule_success=upstream_success_or_none)
```

`bound_context_factory`は呼ばれるまで科学入力をloadしない。H4は既存`h4_df10_rpe_batch_provider.prepare/load_saved_context`、H6は既存`h6_rpe_batch_provider.load_saved_context`を使う。元loaderのhash/H/state/layout/sector等の検査後、Aへscientific_bindingとstate_energy/state_residualを渡す。H6のsolver_* metadataを共通keyへ対応付け、H4 layoutを転用しない。

`bound_saved_reference(ctx,T)`はAのHANDOFFと同じ`{binding,T,source,evidence:{reference,independent_reference,signal_difference,...}}`を返す。Aが2つのcomplex値と差を検査する。Bは同じTのcallbackをcall内でcacheし、R間で再利用。callbackを省略すると既存backendで新2経路referenceを計算する意味なので、別途承認範囲へ含める。保存referenceがない場合の暗黙fallbackには使わない。

同roundの複数Rは、重複roundのないA task群へ分割する。全candidateとconfiguration/identityを照合し、Aの既存`RBatch`を実際に呼ぶ。contextもcall内で共有するが、Aの既存half-step/worker機構は変更していない。AのFAIL/null/status/diagnostics/整数/actual未資格を保存する。N不成立後の後続groupは未実行statusを持ち、価格段・自動再試行へ進まない。context/reference例外時も`last_run`にpartial recordとAのSTOP情報を残す。

元固定Nは保存値として使えるが、R比較roundはmenuの`N_comparison_policy=R_AWARE_EMPIRICAL_ACTION_COUNT_V1`を要求する。対応policyの証拠がない旧baseline Nを再ラベルしない。必要な保存baseline signalのscalar再解析は別の科学計画で実行し、旧handoffを上書きしない。PFとR変更しないroundを不要に再計算しない。

## CがR依存価格を供給するとき

固定provider wrapperはH4 R4/H6 R16だけ。H6 qPF>4はlookup null/外挿のまま。価格値を持つ既存quoteも、元providerのdomain検査とqualified flagを保持する。旧V1等の未資格modelは別scenarioのまま残し、資格を追加しない。

R依存Cが完成した際は、`c_provider(menu, requested_rows)`を差し替える。固定providerを書き換える必要はない。各candidateにつきX/Yを1 recordずつ返し、次を満たす。

- `identity=b.p.identity(menu,row)`、`law_binding=b.r.request_context(menu,row)['law_binding']`。定義は`b.LAW`（Aの元finite RTE meanのSOURCE、R/K/qPF付き）。一致を上流証拠で確認してから返す。Bがlawを数値証明したものではない。
- method/formula/T/qPF/δ/prefix/R_outer/K、H/state/snapshot、completion/scalar/fragment、compiler/control/full wrapper/P0、measurement/符号/RZ単位を一致させる。
- `classification/expectation/cohort/source/mean.rz_count`を保持。direct/model/外挿/null、path/finite cohort/law期待値を区別し、X/Yの同scenarioを明示する。未知は0にしない。
- modelは`domain=full candidate object`、`binding/compiler`、q_variable=q_PF、q、coverage/source/validation_source、候補別のqualified_for_candidateを持つ。R4/R16の係数に新Rを代入しただけでは資格しない。
- `law_context`にtarget N lawと実sampler law・距離資格を残す。有限cohort policyでも全law保証にはしない。normalization_is_price_multiplier=False、価格標本数を量子Nへ乗算しない。
- `uncertainty`に保存残差/nominal量/SOURCE、shared calibration group、paired XY、cross-R/round依存を保持。未知systematic/joint covariance/CIはnull。新しい同時区間や全RPE成功は別の上流証拠として渡す。

新R価格が欠測でも、common responseはnullと理由/validationを返せる。Bはそれを取得済み価格へ数えず、同じ返答を自動再呼出し・再compileで埋めない。SOURCE/domain/lawが未完成の段階で仮価格を実データに補わない。

## 保全・再現と担当

元A/Cのproducer/coverage/CLIは固定commitとbyte一致。元testの保存assertはH6 factory追加とBの既存exact会計だけを認め、他の検証を保持した。新テストでは実A RBatchの下位leafをmockし、C保存値はmetadata確認に限定。実costに使ったN/Cは全てmock。83技術test PASS、独立科学レビューではない。

新cloneで固定Git objectが足りない場合は、既存B/A/Cの該当branchをfetchしてから使う。reader missを科学再計算へfallbackしない。ローカルの他担当worktreeや進行中SOURCEを取り込むために書き換えない。

次はユーザーが最小科学案と必要価格資格を判断する。A/Bが許可されたNと比較baseline policyを供給し、Cは既存指示のR依存価格設計を継続する。Bの実G/R評価も別途許可後だけ行う。本担当は公開・remote照合後mandatory STOP。
