# B2・Skill・D2・高速化担当への引渡し

固定実装SOURCEは `SOURCE.json`。通常入口は従来の `scripts/run_rpe_schedule_search.py`。元round05を編集せず、[両方式の入力例](../../../configs/rpe_method_optimizer/rounds/example_h4_pr_requested_widths_unapproved.json)の新round/runを使う。

## 利用者が開始時に宣言する幅

```json
{
  "methods": ["PR_PAULI", "PR_DF"],
  "science_approved": false,
  "search_overrides": {
    "requested_widths": {
      "PR_PAULI": [0.1, 0.2, "1/5"],
      "PR_DF": [0.1, 0.05, 0.2, "1/5"]
    }
  }
}
```

値は**絶対要求ΔPF**。hは元のbinary64 .1、T=h2^m、M=12を固定する。JSON数値はそのbinary64 exact有理数として、文字列1/5やnumerator/denominator辞書は厳密有理数として保持する。mapperが拒否した幅は理由付きで保留し、別幅は継続する。L_D/κ/shot数の候補別入力や、幅別callerの追記は不要。

宣言が無い既存round05は従来WideDriverへdispatchする。新しい宣言があるとWidthDriverになり、既存mapper由来のfamily定数を使う。保存基準がないLDは `preparation_request` を通してD2/Aのdescriptor/dictionary準備へ接続する。SOURCE不足時は別LDのλを借りず必要入力を返す。

## 起動・再開

技術的な計画確認は次の同一コマンドで開始・再開する。実H4科学実行の許可は含まれない。

```bash
cd /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/a-pr-step-width-search-20261117
PR_WIDTH_PYTHON=/home/AbeHiromu/projects/h6-environments/20261011/venv/bin/python
"$PR_WIDTH_PYTHON" -B scripts/run_rpe_schedule_search.py \
  --problem configs/rpe_method_optimizer/h4_df10_problem.json \
  --round docs/research/pr_requested_step_search_20261117/entry_plan/ENTRY_ROUND.json \
  --run-dir docs/research/pr_requested_step_search_20261117/entry_plan/run02 \
  --max-cycles 1 --plan-only
```

Skillからも同じround/problem/runを渡す。実行済み科学STOPは同runで開かない。後日、採択済みの実行SOURCE・有限scopeに科学実行許可がある場合だけ、所有する新roundのscience/予算/scopeとworkspaceを整えて `--plan-only` を外す。元の未承認round05を変更しない。上のENTRY_ROUNDはLD1/LD7・最初のκ範囲に絞った技術smoke入力であり、科学探索範囲の採択ではない。全分割の利用者向け入力例はconfigs側に保持する。計画専用の再開も次のmetadata候補を選ぶことがあり、完了済みstageは再起動しない。

例のPauli workspaceは当独立checkout。今SOURCEには旧A metadata/信号入口、採択済みP0、既存数値監査helperと新scoped adapterが揃っている。他担当のcheckoutを書き換えない。別所有workspaceを使う場合は、この固定SOURCEを展開し1回設定すればよく、幅・分割ごとの設定追加は不要。DFのsignal/Q3 viewは元D2固定SOURCEを使う。

## 自動接続の場所

1. `WidthDriver` が `(method,LD,formula,exact requested width,K)` ごとにκ scheduleを選び、`Completion.complete({method,design:{L_D,formula,delta_pf,kappa,K},selection})` を呼ぶ。`kappa` は絶対κ。declared widthはrawとexactの両方を保持する。
2. `Completion.plan/prepare_family` は科学leafを呼ばずcanonical配置・各roundのsignal reuse/request/unsupported・価格不足を返す。対象LD自身の辞書/descriptor/sourceを使う。
3. `Completion.price` は現在のquoteをactual-step registryとmergeし、不足だけをCのPauli要求またはDFの `design_groups` へ渡す。Pauliは `scoped_registration=true`、DFは `--covered-rounds`。同物理calibration cellと保存roundは共有する。
4. driverが返す `evidence_packages` を `Completion.ingest_price_results()` が自動監査/登録し、同candidateの26軸を再照会する。run専用 `completion/STEP_PRICE_RULES.json` に保存する。本番catalogへの人工登録はしない。
5. READYだけを `Completion.install` が同じcontext/routesへ登録し、V2がその候補自身の価格・信号でshot提案する。Q3のREQUEST/OUTCOME検査・prior再利用・loss/target/MSE閾値は従来と同じ。最安は `ROUND_RESULT.json` と `PER_METHOD.csv`、幅別状態は `KAPPA_STATE.json`。

戻る状態はconfiguration完成、signal13 round完成、価格26軸完成を分離する。`WAITING_INPUTS` は保存価格の追加で再判定できる。`HELD_*` の科学失敗、未解決な既起動、旧STOPは自動retryしない。規則の適用外公式/K/表現不能幅やDFのmodel rank/holdout不足は、対象round・不足理由を返す。

### 外部から固定較正結果が届く場合のAPI

通常の自動driver経路では転記不要。B2/D2の別workerから返す場合も同じ `StepPriceRegistry.ingest(package)` を用い、run設定の `candidate_completion.price_evidence_results` またはCompletionの同名APIへ渡す。

```json
{
  "kind": "PR_PAULI_STEP_AUDIT",
  "configuration_source": {"commit": "<fixed>", "path": "CONFIGURATION.json"},
  "implementation": {"...": "candidate-specific adopted contract"},
  "preparation_group": {"covered_rounds": [1,2,3,4,5,6,7,8,9,10,11,12], "...": "C request"},
  "price_source": {"commit": "<audit>", "path": "PRICE_26AXIS.json"},
  "audit_source": {"commit": "<audit>", "path": "audit_run01/AUDIT_RESULT.json"},
  "audit_inputs_source": {"commit": "<audit>", "path": "INPUTS_AUDIT.json"},
  "binding_source": {"commit": "<binding>", "path": "BINDING.json"},
  "raw_source": {"commit": "<raw>", "path": "run01/RESULT.json"}
}
```

DFは `kind=PR_DF_STEP_MODEL` とconfiguration_source・implementation・inputs_source・model_source・price_source・holdout_sourceを渡す。単独セルのDIRECT_FINITE_COHORTだけはholdout_sourceをnullとし、同物理点だけを使う。モデル価格は独立holdoutと元receiptを要求する。モデル入力とraw/audit/frozenのSOURCEを分ける。

各価格行の `model_alternative` にactual_step・covered_rounds・calibration_domain・extrapolation(q/R)・event_step_changed_from_calibration・SOURCEが残る。`qualified_E_C=false` とCI=nullを維持する。較正のモデル参考価格を期待価格や精度保証へ昇格しない。

## 並行高速化版との統合点

読取照合した高速化成果はe839b0d3f、実装a50aadc1f。並行worktreeは変更していない。

- candidate識別：`width_family_key` とcanonical mapperの物理IDを保持する。κ境界はfamilyごと。真のaliasは `physical_evaluation_contract` と既存登録機構で照合し、requested IDとevaluation IDを分離する。
- 準備：`Completion.prepare_family/complete/plan` とREADY recordを共用する。別のsignal/価格workerループは作らない。
- 価格登録：旧26軸registryと `STEP_PRICE_RULES.json` の両digestが更新トリガー。新audit metadata adapter・DF scopeを取り落とさない。
- 探索復帰：当branchの `WidthDriver.batch` のcontext/routesを、高速化版の共通executor・optimizer factoryへ渡す。旧CLI全体で当branchを置き換えず、両変更を保持する。高速化版の `continue_selected_preparation(...,optimizer=...)` も維持する。
- cache：元Q3 job key/contract/prior/sourceを保持し、部分価格や異辞書を既完了点へ混ぜない。共通queue/DP cacheによる実速度効果は当branchで測定していない。

## 方式別の確認範囲と次に必要な実取得

|方式|保存証拠|人工入力|実H4で未取得|
|---|---|---|---|
|PR-Pauli|単一幅9構成、LD1/2/3/5のround04別幅request、公開LD4 auditの26軸|二実幅のdriver監査/登録・部分価格・再判定・V2/Q3・安い別幅選択・再開0起動|選ばれた別幅の欠測mean、未較正actual-step cell、実Q3|
|PR-DF|単一幅9構成、LD7 .05のmapper受理/.2拒否、公開LD6model/holdoutの26軸|scoped model/direct価格、欠測descriptor、共通探索復帰・安い別幅選択・再開0起動|未保存分割のdescriptor、選ばれた別幅の欠測mean/calibration/独立holdout、実Q3|

再現検査：`scripts/check_pr_requested_width_saved.py`、`tests/check_pr_width_search.py`、`tests/test_pr_width_completion.py`、`tests/test_pr_step_price_registry.py`、`tests/test_pr_df_step_groups.py`。公開検収は `TESTS.json` と `SAVED_ACCEPTANCE.json`。確認したのは作成者と担当subagentであり、独立科学レビューではない。公開・remote一致確認後mandatory STOP。
