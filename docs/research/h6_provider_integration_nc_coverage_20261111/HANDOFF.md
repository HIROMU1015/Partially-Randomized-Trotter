# Codex B向け：H6 provider統合とN/C coverage

固定B基点 `cb2b1d6b4309ca7d705f397e25e27d14e8ba125a` に対する統合コードcommit：`f58428f3021f1ac9d51fad038faa8f78eb560aed`。独立branchは `codex/rpe-h6-provider-coverage-20261110`。Bの作業領域は変更していない。B自身の科学SOURCE/STOP保全期間が終了した後、Bが現在の差分と照合して取り込む。C branch全体の強制merge、履歴変更、Bの新しいN側修正や成果物の上書きは不要。

## 取込み単位

コードcommitは次の6ファイルだけを追加/変更する。

- `src/trottertracks/h6_df19_price_provider.py`、`tests/check_h6_df19_price_provider.py`：C固定0b5c493fからbyte一致。
- `src/trottertracks/pf_pr_rpe_pipeline.py`：6行のH6 factoryだけ。既存関数はB基点と同じ。
- `src/trottertracks/rpe_nc_coverage.py`、`scripts/report_rpe_nc_coverage.py`：固定公開JSONを読む静的inventory。
- `tests/check_h6_price_pipeline_coverage.py`：実checkoutのB APIとの技術結合確認。

Bの作業保全期間終了後に使う候補手順（CはBへ実行していない）：

```bash
git fetch origin codex/rpe-h6-provider-coverage-20261110
git fetch origin codex/resource-eval-h6-df19-price-design
git show --stat f58428f3021f1ac9d51fad038faa8f78eb560aed
git cherry-pick f58428f3021f1ac9d51fad038faa8f78eb560aed
PYTHONDONTWRITEBYTECODE=1 python -B tests/check_h6_price_pipeline_coverage.py
PYTHONDONTWRITEBYTECODE=1 python -B tests/check_h6_df19_price_provider.py
PYTHONDONTWRITEBYTECODE=1 python -B tests/check_h4_price_pipeline_integration.py
```

Cの価格branchのfetchは凍結model/価格/validationのGit objectも取得するため。共通moduleで競合する場合、Bの最新版を保ちH6 factoryだけを追加する。B N batchコードを過去版へ戻さない。結合テストの「H4/N/会計と共通関数がcb2と一致」のassertはCの独立基点を検査するためであり、最新Bへそのまま適用する際は意図されたB追加差分を照合して基点保存assertだけを調整する。他のdomain/資格/null/coverage検査を無効化しない。

静的coverage再現には公開H4 N commit `89c91c692e1d35de94d2e7afbe890aabe6d80b7e` とH6 N commit `f4c0bdabdf2afdc5bc4f20573bb10677917f4c1a` も必要。B branchに公開H4結果がある場合は追加コピー不要。objectが不足するcloneでは次の読取fetchで補う。

```bash
git fetch origin codex/pf-pr-rpe-cost-integration-20261110
```

object不足時に新N/価格予測/compileへfallbackしない。文書と静的成果は本branchの `docs/research/h6_provider_integration_nc_coverage_20261111/` を選択して参照/取込みする。共通索引・manifestはB最新版と手動で必要な追記だけを照合し、branch全体で上書きしない。

## 共通API

```python
from trottertracks import pf_pr_rpe_pipeline as p
store = p.SavedArtifacts(repository_root)
c_provider = p.h6_saved_price_provider(store.read, selections={
    'ef41b65942d56977d6cd': {
        'kind': 'model_prediction',
        'price_unit': 'cohort_mean',
        'cohort': 'H6_PRICE_V2_20261110_Q4_INDEPENDENT',
    },
})
# c_provider(task, requested_rows) -> 保存lookupのX/Y record
# 将来の許可済みpipelineの run_missing(..., c_provider=c_provider) へ渡す。
```

この例は配線の説明で、実N×C/Gを実行した例ではない。PF既定は保存direct、PRは明示unit/cohort必須。q1 PRはEXPLORATION n9/CONFIRMATION n33、q2はH6_PRICE_V2_20261110_Q2_CALIBRATION n8、q4はH6_PRICE_V2_20261110_Q4_INDEPENDENT n8。探索/確認をpoolしない。

`quote/__call__/run_missing`は保存lookupだけ。q2/q4のmodelは保存予測をコピーし、q1 anchorをforecastへ再ラベルしない。qPF>4はvalue=null、UNVALIDATED_EXTRAPOLATIONを保持。別δ・H6 m0..3のPR p5/R32や別compiler/full wrapperは対象外。明示数値`predict_affine`をmissing補完へ接続しない。未観測order2も対象外。

既定`qualified_expectation`はPR有限cohortを拒否する。明示`finite_cohort_scenario`でもq2校正modelは未資格。q4の有限検証資格はfull RPE採用/全law期待価格保証ではない。pathはdigest/indexを含む保存bindingが必要で、平均や全lawへ変換しない。未知joint covariance/長q systematic/total CIはnullのまま、共有校正groupを独立標本としない。normalization Bは価格へ再乗算しない。

## 公開Nの受入れと更新

今回の[静的成果](static/COVERAGE.json)は両system各26行の条件付きPASSを含む。H4は作業中に公開済みであることを確認した89c91c69の `N_HANDOFF.json.shots` を使用。Bが補正した巨大Nの整数を再計算せず、原resultの小さい整数へ戻していない。`N_X/N_Y`はPython整数、`N_X_decimal/N_Y_decimal`は全桁十進文字列。binary64だけのreaderは文字列を使う。actual Nは全てnull。

後日の公開H4 handoffを差し込む場合は、commit/pathを明示し、既存結果を上書きしない新outputへ保存する。

```bash
PYTHONDONTWRITEBYTECODE=1 python -B scripts/report_rpe_nc_coverage.py \
  --h4-n-commit 89c91c692e1d35de94d2e7afbe890aabe6d80b7e \
  --h4-n-path docs/research/h4_df10_rpe_batch_results_20261110/N_HANDOFF.json \
  --output-dir /tmp/rpe-nc-coverage-published
```

`accept_saved_N`は既存identity/measurementでtaskの候補と照合する。等しい重複のみ受入れ、別candidate/foreign条件/重複entry/既存Nの競合はエラー。FAIL/nullは保持し、欠測を0にしない。H4 m3のみの基点を再現する技術用途には `--baseline-h4-n` を使える。未公開raw/checkpointを自動探索せず、公開taskの科学条件が変わる場合は単なるhandoff差替えで処理しない。

H6は既存passive `prepare` が返す保存m0..7と、公開m8..12のnew_recordsを転記する。execute/context/state配列を起動/読込しない。coverage helperは価格selectorのmetadataと保存scalarだけを読む。

## 残る判断とSTOP

保存CはH4 m3/m4両方式とm5 PF、H6 m3..6両方式。旧V1の未資格保存modelも別scenarioとして残す。H4 m0..2・m5 PR・m6..12とH6 m0..2・m7..12のCが不足。H6 qPF>4の価格は未検証・未計算。PR全law期待価格と全体依存/CIの資格も不足する。

次の担当：Codex Bは自身の科学STOP/成果保全を終えてから本コードと公開Nを統合し、最新APIで技術確認する。ユーザーは価格不足の追加検証範囲、有限cohort scenarioの採用可否、実G集計の具体的科学scopeを判断する。Cは24技術test・静的coverageの公開/remote照合でmandatory STOPし、新価格/N/G科学計算へ進まない。
