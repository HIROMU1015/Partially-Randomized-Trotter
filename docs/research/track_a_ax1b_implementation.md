# Track A AX-1b Preparation：実装・検証・実行前固定

2026-10-09 JST。`AX1B_PREPARATION_COMPLETE_EXECUTION_NOT_AUTHORIZED`。
**保存科学値のfit・性能評価・regretは未実行。これは準備証拠であり科学結果ではない。**

## 正本と今回の追加条件

正本はAX-1a commit `6b6959b9e76ead049a9a247dadfc1f85161916e8` の[結果前契約](track_a_ax1a_preanalysis_contract.md)、[モデル・fit](track_a_ax1a_model_comparison_and_fit.md)、[評価仕様](track_a_ax1a_evaluation_protocol.md)と、`artifacts/resource_applicability/track_a_ax1a/2026-10-09/`のallowlist/実行draft。AX-0/AX-1aの既存契約、model_configuration、tolerance、四case、membershipは変更していない。

利用者の独立review `APPROVE_AX1A_WITH_MINOR_AMENDMENTS` に従い、q別4foldを結合した結果を **cross-fitted内部診断** と明記する。異なるtraining集合で学習したモデルを混ぜるため、単一の凍結モデルによる候補選択・将来H6/H8移送のregretとは異なる。既存base metric名は保ち、追加の`diagnostic_kind`、`single_frozen_model=false`、cross-fitted名で区別する。fold別cost/selection、pooled OOF cost、pooled cross-fitted conditional regretを別recordにする。eligible集合が空等の場合は数値を作らない。

## 新規実装の入口

| ファイル | 機能 |
|---|---|
| [ax1b_contract.py](../../src/trottertracks/resource_applicability/ax1b_contract.py) | AX-1a五fileの固定byte hash、情報区分、STOP、canonical JSON、path/selector、operational N/A |
| [ax1b_models.py](../../src/trottertracks/resource_applicability/ax1b_models.py) | 型付きI1 Features、action index、axis別SINGLE/FEW fit、source finite normalization、complexity gate、STRUCT N/A |
| [ax1b_data.py](../../src/trottertracks/resource_applicability/ax1b_data.py) | permit必須の45入力reader、schema/hash、full candidate join、membership、PM1/M2 adapter、metadata folds |
| [ax1b_evaluation.py](../../src/trottertracks/resource_applicability/ax1b_evaluation.py) | RZ error、coverage、index順位、I4 reference shot、conditional work/regret、共通support、paired covariance/SE |
| [ax1b_analysis.py](../../src/trottertracks/resource_applicability/ax1b_analysis.py) | 将来認可後の保存値orchestration、group/cross-fitted診断、四anchor、PM2保存ledger照合、登録12出力の内容 |
| [ax1b_execution.py](../../src/trottertracks/resource_applicability/ax1b_execution.py) | stdlib gate、準備validator、commit/blob/environment/budget/output照合、resource enforcement、hash再確認、STOP付きmanifest |
| [runner](../../scripts/resource_applicability/run_track_a_ax1b.py) | default拒否、metadata-only preparation検査、別authorizationと明示launchの将来入口 |
| [synthetic test入口](../../scripts/resource_applicability/run_track_a_ax1b_preparation_tests.py) / [専用tests](../../tests/tracks/resource_applicability/test_ax1b_preparation.py) | 科学artifact・旧moduleを遮断して合成値のみで検証 |

既存科学runner/moduleをimport・変更していない。旧sourceの必要な定義をテキスト確認し、新moduleに会計だけを実装した。旧indexは未コミット整理を保護するため変更せず、本書と新manifestを新経路の入口とする。

## AX-1aと実装の一致

| 契約 | 実装と検証 |
|---|---|
| A_exact/A_ceil、未校正action単位 | Features.actions、ceil(E−1e−15)。RZ error関数にaction unitを渡すとSTOP。順位は平均tie rankのSpearman |
| SINGLE：axis別、均等weight、interceptなし | trainだけのRMSを使う閉形式β≥0。同じleast-squares解。参照bias/shot/test Cを受け取るfeature keyなし |
| FEW：固定5column、NNLS非負 | train column/target RMS、zero column drop、SVD relative1e−12、保持順intercept/E_rand/n_det/q/n_fixed、min rows=2p |
| solver/tolerance | SciPy1.14.1 nnls(maxiter10000,atol1e−12)。KKT active1e−12/residual1e−8。fallback/negative clippingなし |
| dropped featureのtest外挿 | train-only linear relationを保存。関係を破るtest値はflag、係数を復活/refitしない |
| 欠測・zero・ineligible | 0補完なし。zero target/zero Cはfitに残す、比率N/A。accuracy不適格もone-shot cost評価。fit不能を別modelへ置換しない |
| PM1 n_fixed未保存 | full one-body/constant・empty-tail source invariantと同q/同identityのB0 anchors全一致だけを利用。不明ならPM1は両modelともN/A（従属columnとしてdrop済みでも救済しない） |
| STRUCT/shot model | STRUCTは未保存入力によるN/A。bias/N/operational eligibility/work/regretはnullと理由を保持 |
| canonical finite B | lgamma paired weights、fsum、source順q*(r*log b)、exp。有限normalization/全B overflowでSTOP。paper upperのoverflowはnull+flag、bound slackを別分類 |
| trainingと候補集合 | M1の210だけfit。PM1 8/M2 5は診断。q/prefix/method/random K folds。deterministicをK-testへ混入しない |
| complexity gate | q-balanced abs-relative errorの相対10%かつ絶対.01改善、各q悪化≤.02、4fold・同full coverage。default SINGLE、PM1/M2/regretによるoverrideなし |
| 四factor cases | REF×DIRECTとREF×PREDの条件付き評価のみ。predicted-shot二caseはN/A。M2 legacy conditional分類を保存 |
| 比較集合・selection | direct218/M2別5、OOF210。available calibrated modelsの同common supportを明示。full-set欠測をN/A、false acceptanceをfinite regretより先に処理 |
| 統計 | IID unweighted n32/1、pair index/seed/step/evolution照合、n−1 covariance、同NのSE。±2SEはengineering、cross-candidate covariance=null、rare-event unresolved |
| 出力・終了 | AX1a登録directoryと12 filenameを維持。JSON/JSONL/CSV/Markdownを専用serializerで出す。input/source前後hash、disk cap、no retry/resume、全terminal後STOP |

β/θの物理的因果解釈、QPE/FT対native RZの直接比較、DF norm保証、H6/H8検証は行わない。保存biasから算術でshotを再評価する将来処理はI4でありoperational predictorではない。ε=.001も保存biasを使う会計点で、新signal精度検証ではない。

## Synthetic検証

実行command（対象worktree、追加専用suiteだけ）：

```bash
'/home/abe/Project/Partially Randomized Trotter/.venv311/bin/python' \
  scripts/resource_applicability/run_track_a_ax1b_preparation_tests.py \
  --audit-output /tmp/ax1b_synthetic_test_audit.json
```

記録環境：Python **3.11.0rc1**（releaselevel candidate）、NumPy1.26.4、SciPy1.14.1、pytest9.0.3、process1/BLAS1。専用suiteのpass/fail/skipと正確なcommandは[synthetic audit](../../artifacts/resource_applicability/track_a_ax1b_preparation/2026-10-09/synthetic_test_audit_v1.json)に固定する。全repository testsや旧科学testsは実行していない。local synthetic evidenceでありimmutable CI・保存科学値の再現証拠ではない。

テストは合成既知係数、NNLS制約/失敗、rank/scaling/zero/missing、情報漏洩、PM1 anchor、M2 root geometry、join変異、path/hash/schema、complexity、cost/selection status、common support、paired covariance、認可/環境/予算/source/output gateを検査する。合成210+8+5のorchestrationも含むが、全値をrange/式から作成しており保存M1/M2値を使っていない。

pre-import boundaryはNPZ/NPY/pickle/runtime/cache/registry、旧科学artifacts、trotterlib importを拒否する。pytestの親conftest・plugin自動load・cacheを無効にし、単一file・隔離temp directoryを使用する。最初のcollectorがartifact directoryを探索したため遮断された件と、synthetic overflow例の訂正、Python gateの直接releaselevel確認の追加は準備中に解決した。最終auditに科学アクセス/import試行0を要求する。科学データを読んで修正した履歴はない。

## 実行前bundleとhash

専用directory：[track_a_ax1b_preparation/2026-10-09](../../artifacts/resource_applicability/track_a_ax1b_preparation/2026-10-09/)。

- [preparation_manifest_v1.json](../../artifacts/resource_applicability/track_a_ax1b_preparation/2026-10-09/preparation_manifest_v1.json)：新source/test/schema/doc/draft/blocker/auditのbyte hashes、AX1a hashes/config hash、状態。
- [schemas_v1.json](../../artifacts/resource_applicability/track_a_ax1b_preparation/2026-10-09/schemas_v1.json)：bundle/audit/authorization/prediction/terminalの必須fieldと制約。
- [execution_authorization_draft_v1.json](../../artifacts/resource_applicability/track_a_ax1b_preparation/2026-10-09/execution_authorization_draft_v1.json)：認可false、source commit/env/予算/review/launchはnull。
- [blockers_v1.json](../../artifacts/resource_applicability/track_a_ax1b_preparation/2026-10-09/blockers_v1.json)：確認方法と解除前に禁止される処理。

source bytes hashは固定できる。自身を含むcommit SHAは作成前に埋められないのでbundleのsource_commitはnull、実commitは完了報告で示し、将来の**別authorization**でbindする。manifest自身のhashは外部review/authorizationからbindし、self hashを捏造しない。`preparation_manifest_sha256`とlaunchのauthorization hashは**sorted-key compact JSON payloadのUTF-8 SHA-256**。input/source/doc/test file hashesは**raw bytes SHA-256**。二種類を混同しない。

元のAX1a execution draftは変更しない。新authorizationの環境・予算が揃っても、review/明示launchなしに実行できない。source commitのHEAD/blob、テストaudit、五契約bytes、output新規性も確認する。通常のdefault runnerはbundleすら開く前に拒否する。metadata-only検査は五契約と新bundle関連fileだけを読む。

## 未確定blockerと必須STOP

Python3.11.0rc1を本解析のstable Python≥3.11として承認しない。今回のsynthetic test環境と、将来の実行環境確認を分離する。stable環境の正確なPython/NumPy/SciPy identityと同sourceのsynthetic成功が必要。SciPy1.14.1不一致のadapterは未実装・未承認でありSTOP。

CPU割当/affinity、RAM bytes、wall seconds、output disk bytesは未確定null。host総量から推測しない。future runnerはprocess1/BLAS1、affinity、RLIMIT_AS、wall alarm、write前disk capを適用するが、共有host全体の容量保証ではない。出力directoryの衝突、source/input変更、solver/KKT/identity failureはSTOP。partial出力は完了証拠でなく、自動retry/resumeしない。

今後必要なのは今回の独立GPT review、source commit/hash binding、stable environment/test確認、割当予算、別execution authorization、利用者の明示launch。STRUCTとoperational shotのN/Aはcost-only研究範囲の制約であり、無断のscience補完を求めるblockerではない。

`ax1b_analysis_authorized=false`、`science_authorized=false`、`explicit_user_launch_required=true`、`mandatory_stop=true`、`next_stage_authorized=false` を維持する。Preparation完了はAX-1b解析・AX-2以降への自動認可にならない。
