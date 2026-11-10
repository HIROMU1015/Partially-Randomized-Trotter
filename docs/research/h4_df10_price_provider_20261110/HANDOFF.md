# Codex BへのH4 price provider handoff

[interface](INTERFACE.md)／[REPORT](REPORT.md)。Aは本branchのH4 adapterだけを実装した。Bのworktree・固定SOURCE・共通frameworkを変更せず、Bの共通interface最終設計を待つために作業を止めていない。

## 調査したB interfaceと取込み

読取先は`h6-df19-rpe-preparation-20261110` worktreeの`src/trottertracks/pf_pr_rpe_pipeline.py`。作業中の未commit版を読取snapshotとしてテストした。テストしたbytesのSHA256は`c91d5c59f4ea7ae65b37bc2c850e782888834842c0626fccd72b5faa34c1c9ba`（[TESTS](TESTS.txt)）。調査開始時HEADは`14c4f871fb9e8d4a23a6cc3244f5435ebc65a7ce`だが、このHEADに作業中moduleが固定済みだとは扱わない。他worktreeの変更を拒否する条件にはしていない。

Bの`SavedArtifacts.read`、`identity`、`measurement`、`validate_task`、`candidate_id`を使用し、`c_provider(task, requested_rows)`へ適合させた。既存のBの`saved_model_prices`にはS1の保存mean転記があるが、今回必要な独立q証拠・path binding・未取得値・明示数値entryの区別がないため、H4 adapterを追加した。

試験ではこのB module snapshotをA側Python環境へloadし、相対importの`h4_df10_rpe_connection`はA側版を使用した。B側の同moduleには保存point検査・部分round会計の拡張があるが、今回利用したscope/価格種別定数は同じで、拡張された科学会計は呼んでいない。B環境全体の結合試験は取込み後にBが行う。

取込み対象は新しい`src/trottertracks/h4_df10_price_provider.py`、`tests/check_h4_df10_price_provider.py`、本directory。依存する既存moduleは下記の固定版を再利用する。調査時Bには前3つが未配置、counterは同版で存在した。

- `src/trottertracks/h4_df10_price_v1.py`
- `src/trottertracks/h4_df10_price_v2.py`
- `src/trottertracks/h4_df10_price_v2_independent.py`
- `src/trottertracks/track_d_native_resources.py`

これらは`1cf87201ebc6c9e0296fe647354981b6f7f8b0c1`を参照する。数値entryではV1会計の版を混合しない。B側で既存counterを無条件に上書きせず同版/適合性を確認する。catalog用の固定JSON objectも取得できるよう、本branchをfetchする。取込みはBが自分のbranchで行い、Aは実行していない。

## 最小接続例（lookup・静的planのみ）

```python
from trottertracks import pf_pr_rpe_pipeline as p
from trottertracks.h4_df10_price_provider import H4PriceProvider

store = p.SavedArtifacts(repository_root)
selection = {
    "64e4f7e9ec7fba28815e": {
        "kind": "direct_compiled",
        "price_unit": "cohort_mean",
        "cohort": "H4_DF10_V2_INDEPENDENT_Q_20261110",
    },
    "14bd81a375400c9b0305": {
        "kind": "direct_compiled",
        "price_unit": "cohort_mean",
        "cohort": "CONFIRMATION",
    },
}
provider = H4PriceProvider(store.read, pipeline=p, selections=selection)
prices = provider(task, requested_rows)  # 1 candidateにつきX/Yの2 record
inventory = p.plan(task, supplied_shots, prices)  # N/Gは計算しない
```

`task/row`はBの共通identityを使う。binding/profile/contractの条件は[INTERFACE](INTERFACE.md)に記載。selectionsはkind/unit/cohort/path/hashだけを選び、candidateや科学identityを置換できない。PFは既定で保存direct、PRは明示selectorがなければunavailable。Mq保存値を使う場合は`kind='model_prediction'`を明示する。

## 数値entryの将来接続

新しいH4予測が別途許可されたときだけ、Bの明示的な数値実行経路から`predict_path(request, descriptors)`を呼び、`pipeline_record(task, row, axis, quote)`へ渡せる。既存descriptor抽出元は`run_h4_df10_price_v1.inputs_and_blocks`で、新たな抽出も科学実行に含まれる。PRには実際のordered event/application・source・digest・path IDを渡す。家族名からlaw期待値を生成しない。

`provider(task, rows)`自身には数値予測・compile・sampling・fitのfallbackがない。q1…4096のmissing priceを一括して新予測することもない。未検証qの数値quoteをB recordへ変換すると`UNVALIDATED_EXTRAPOLATION`を保持する。今回この数値entryは人工/mockでだけ確認した。

## B側で残る作業・科学判断

1. **B**：上記component/dependencyを自分のbranchへ取り込み、共通APIの最終版で技術再確認する。Aが試験したのは上記作業中snapshotで、B完成版へのmerge済みとはしない。
2. **B**：`run_missing`の`c_provider`へlookup adapterを登録し、`provider_quote.reason/price_unit/validation`をcoverageへ保持する。現在のB plannerでは値のないrecordもC_axesへ現れ得るため、単なる軸存在を価格取得済みとしない。
3. **B**：`FINITE_COHORT_ONLY`が残ることを「再compileすれば解消するmissing」と扱わず、同じC providerを再呼出しして重複cohortを追加しない。saved directとMq scenarioを区別し、X/Yは同じcohort/pathを選ぶ。
4. **B**：価格policyを明示する。現在のBの既定`qualified_expectation`はPR有限cohortを受け付けず、明示`finite_cohort_scenario`は条件付きscenarioとなる。path単体はlaw平均ではない。`qualified_for_candidate`は有限条件の記録で、全RPE価格の正式採用とは別。
5. **ユーザー/B**：新N・新価格予測・N×C/Gを実行する前に、対象候補とSOURCE/inputs、law/finite-cohort解釈、外挿資格、必要な誤差精度を決定し、その科学範囲を明示許可する。今回の接続実装だけでは科学実行を許可していない。
6. **B**：将来の誤差伝播で同じbを使うround/XYの相関を保持する。校正係数・long-q systematic・全law・joint covarianceは未取得。観測maxや標本SEから全体CIを作ったことにしない。これらの新推定や総費用会計はA側で行っていない。

科学的証拠はPF q32/PR q16の有限独立q、およびq8/q16の既知S1結果に限定する。長q・q4096・全13round期待価格/CIは未資格。Aの今回作業は技術完了後mandatory STOP。B/Cへツールでメッセージを送る処理は行わず、本公開資料を引継ぎとする。
