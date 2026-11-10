# Phase 1準備：A/B向けHANDOFFと将来の実行手順

コードSOURCE `1a87e5331b52a49732b7db9d49d28f1fe6d9c9a7`、[PLAN](PLAN.json)、[REPORT](REPORT.md)。今回の科学実行は0。以下のexecute手順は、ユーザーがこの固定Phase 1範囲を明示承認した後にだけ使う。--execute自体は許可の証明ではなく、誤起動を避ける明示入口である。新たな承認書/grant発行は不要。

## 取込み

BのSOURCE保全期間/STOPを確認後、Bが自分の変更と照合してC固有5ファイルを選択して取り込む。CはA/B worktreeへ実行していない。

```bash
git fetch origin codex/r-variable-compiled-price-design-20261111
git restore --source=1a87e5331b52a49732b7db9d49d28f1fe6d9c9a7 -- src/trottertracks/r_price_phase1.py src/trottertracks/r_price_phase1_runner.py src/trottertracks/r_price_phase1_science.py scripts/run_r_price_phase1.py tests/check_r_price_phase1.py
PYTHONDONTWRITEBYTECODE=1 python -B tests/check_r_price_phase1.py
```

共通pipeline/N/会計や旧providerを過去版へrestoreしない。試験の保存byte一致assertはC基点8457c4435の保全確認なので、Bの意図した差分がある場合は該当assertを照合して調整し、機能/資格/STOP検査は保持する。fixed Gitから読める旧SOURCE/data objectが必要。新clone等では元A/Cと公開A/B branchをfetchして用意し、不足を再compile/予測で補わない。

```bash
git fetch origin codex/resource-eval-h4-df10-longtime-price-design
git fetch origin codex/resource-eval-h6-df19-price-design
git fetch origin codex/r-variable-rpe-cost-pipeline-20261111
git fetch origin codex/pr-r-dependent-n-model-20261111
```

固定Git objectの参照一覧は[SOURCE_INDEX](SOURCE_INDEX.json)。sealed providerはc31ef424の元runtime・NumPy1.26.4/Qiskit1.3.0を使用する。PLAN.runtimeのinput_config/provider_root/old_cacheは既知の保存先であり、実分子入力を今回openして存在やruntimeを保証したものではない。別保存先が必要な場合は内容bindingを保った明示変更として扱い、共有環境を変更しない。

## Metadataだけの再現

```bash
PYTHONDONTWRITEBYTECODE=1 python -B scripts/run_r_price_phase1.py prepare --plan /tmp/r-price-phase1-new-plan.json
```

fresh fileを指定する。実行SOURCEには、その時点でcommit済みのコードが記録される。公開済みPLANを将来の承認対象として使う場合は、そのPLANと固定SOURCEをそのまま使う。N/価格/fit/compile/科学runtimeはprepareで呼ばれない。

## 明示承認後のPhase 1（現在は未実行）

public outputはこのrepo内の新directory、private rawはrepo外の自分の新directoryとする。以下の変数は例示で、既存resultを指定しない。

```bash
phase1_plan=docs/research/r_price_phase1_preparation_20261111/PLAN.json
phase1_public=docs/research/r_price_phase1_results_20261111/run01
phase1_private=/home/AbeHiromu/projects/r-price-phase1-20261111-private/run01
PYTHONDONTWRITEBYTECODE=1 python -B scripts/run_r_price_phase1.py calibrate --plan "$phase1_plan" --public-output "$phase1_public" --private-output "$phase1_private" --execute
```

このstageで保存C/S 4セル＋新calibration 2セル（各n8）をjoin/取得し、system別fitとcalibration評価を保存する。独立validationの価格はまだ取得しない。続いて**係数をcommitする**。これは校正と検証を分離する手順で、同じ科学scope内で新しい承認を取り直すものではない。

```bash
git add "$phase1_public/FROZEN_MODELS.json" "$phase1_public/PLAN.json"
git commit -m 'Freeze R-price Phase 1 coefficients before independent validation [skip ci]'
phase1_frozen_commit=$(git rev-parse HEAD)
PYTHONDONTWRITEBYTECODE=1 python -B scripts/run_r_price_phase1.py validate --plan "$phase1_plan" --public-output "$phase1_public" --private-output "$phase1_private" --frozen-commit "$phase1_frozen_commit" --execute
PYTHONDONTWRITEBYTECODE=1 python -B scripts/run_r_price_phase1.py rare --plan "$phase1_plan" --public-output "$phase1_public" --private-output "$phase1_private" --frozen-commit "$phase1_frozen_commit" --execute
```

private rawをgit addしない。Git freezeは40桁commit、repo-relative frozen fileと同一bytesで確認する。current HEADや無関係なworktree更新を一括凍結しない。校正後はSOURCEの4 production fileが固定SOURCEと一致することを検査し、科学source変更時はそのまま続行しない。

SCI stageのエラーはSTOP_NO_IMPLICIT_RETRY。結果を残し、run01の再利用、R/seed変更、失敗pathの再sampling/再compileを自動開始しない。source/入力の不足を修正して科学処理をやり直す場合は、必要な範囲を明示してユーザー判断を受ける。正常な校正後のGit freeze待ちは再試行ではなく、予定された手順。

## 出力と共通interface

public：PLAN/EXECUTION_PATHS、started markers、CALIBRATION、FROZEN_MODELS、CALIBRATION_EVALUATION、VALIDATION、PATH_RESIDUALS.csv、RARE_DIAGNOSTICS、REPORT、STOP。private：events_PRIVATE、STRUCTURAL、axis開始/価格/fingerprint、path結果。旧cacheを読み取るだけで変更せず、原本は上書きしない。

CALIBRATION_EVALUATION.jsonとVALIDATION.jsonのpricesはB旧形式＋law_binding、full candidate domain/compiler/sourceを持つ。B公開7bca8f9のcoverageで人工整合を確認済み。direct/modelはscenario名も別。PRはFINITE_COHORT_ONLY、量子shot数Nとは別のn_paths、normalization_is_price_multiplier=false。modelはcalibration/finite independent point evaluatedを区別するが採用閾値未設定のためqualified=false。未観測q/Rへの評価は拒否し、全law/total CI/実Gはnull。

A公開5ca2dceのwith_round_Rから得るcandidate/identityは同じ。NのR-aware allowance・phase/radius・α・巨大整数の資格はAの責任で、CがNを変更・計算しない。同じlaw_binding宣言だけでfinite CDF samplerとideal lawの距離が保証されたことにしない。

Bの広いmenuにはPhase 1外のR/q/早期H6 domainが残る。Bが有限cohort policyを明示してdirect Cを使う場合も、対応N/科学資格と全menu coverageを別に確認する。全RPE実数値評価や新R最適化は今回も、このrunner自体の処理にも含めない。raw価格を見てモデルを再fit/選択し直す機能はない。

次：ユーザーが固定SOURCE/PLANの科学実行範囲を承認した場合のみCが実施し、公開・remote照合後STOP。Bは取込みと最新共通APIの技術確認、Aは対応N側の資格整理をそれぞれ担当する。現在は準備公開でmandatory STOP。
