# 保存receiptの技術受入bundle

`ADMISSION.json`は保存M5/m4（Full PF p10、q1）の原receipt・carrier・D catalog・現在のD metadataを照合した結果。原trialのbytesは`ORIGINAL_TRIAL.json`に保全した。価格は両軸とも保存整数のcopyであり、平均・価格予測・fit・N・Gを新計算していない。

`PRICES.json`はD `62ae1bd5`の`price_bindings`をそのまま使って作ったC受入record。原cohort `EXPLORATION`、原price SOURCE、原trialのlocator/hashを保持する。新取得cohortへ移していない。`SAVED_CALIBRATION_INPUTS.json`は同Dの`fit_inputs`→既存C `price_model_validation.record`による2入力の整形だけ。m5が未取得なのでモデルは存在しない。

`IMPLEMENTATION_EVIDENCE.json`をこのbundleのGit commitに固定した後、既存C guardで技術的互換性を確認する。`SIDECARS_UNBOUND.json`のcommit placeholderは消費不可。固定commitへ結合したsidecarとguard結果を次の文書commitで公開する。比較policyの採択、科学的精度保証、G評価を意味しない。

`CELL_PLAN.json`はDの固定`plans`からcopyした18セル×X/Yのmetadata。Full PF16新wrapper＋保存2、Truncated p7の18新wrapperというscopeを維持する。m4/m5 CALIBRATION、m6 HOLDOUT。実行承認は含まない。

既存run・SOURCE・raw・STOPは変更していない。以後のREPORT/HANDOFFに、推奨する未採択policyと実行前条件を記す。
