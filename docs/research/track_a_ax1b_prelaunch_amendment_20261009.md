# Track A AX-1b Prelaunch Hardening：結果前amendmentと実行条件

2026-10-09 JST。独立review `APPROVE_AX1B_PREPARATION_WITH_TARGETED_FIXES`への対応。
基準sourceは `7c4d1a9c098ba4b8f48896fa4b544d4f106c89b3`。
**保存科学値の読取・fit・誤差評価・regret実測は0。準備証拠であり科学結果ではない。**

修正とstable環境のsynthetic検証は完了したが、実行割当予算は未確認。
状態は `AX1B_PRELAUNCH_HARDENING_PARTIAL_EXECUTION_NOT_AUTHORIZED` とする。
独立review・別authorization・明示launchも未完了で、AX-1b/AX-2を開始しない。

## 正本と履歴の保存

正本はAX-1aの[結果前契約](track_a_ax1a_preanalysis_contract.md)、[fit仕様](track_a_ax1a_model_comparison_and_fit.md)、
[評価仕様](track_a_ax1a_evaluation_protocol.md)、45入力allowlist、execution draftの五件。
[Preparation v1説明](track_a_ax1b_implementation.md)と既存bundleは当時のsourceの履歴として保存する。
AX-0/AX-1a、科学結果/status、原稿、Track B、既存dirty整理差分は変更しない。
既存索引は利用者の保護対象なので、本書と新manifestでsource/test/bundleを結ぶ。

## Amendment AX1B-PRELAUNCH-ELIGIBILITY-20261009-V1

旧 `selection()` は `eligible_ref is True` の集合だけから分母を作り、Noneを除外したまま
全登録集合の最適値とregretを表示できた。Noneの候補が実際には適格かつ低費用である可能性を
否定できないため、解析前に意味論を修正する。研究結果を見て閾値・候補・モデルを変えたものではない。

`eligible_ref` は厳密にTrue/False/Noneの三状態とし、それ以外はSCHEMA STOP。
全登録集合のidentity hash、確定適格集合hash、未確定fingerprintsを保持する。

| 条件 | 主statusと処理 |
|---|---|
| 全参照適格性が確定、適格あり、予測完全 | 従来のVALID_CONDITIONAL_ORACLE。regret式・tie規則は不変 |
| 確定適格あり、未確定あり | 新REF_ELIGIBILITY_UNDETERMINED。regret/full_set_min_G_ref/common_support_regretはnull |
| 確定適格なし、未確定あり | 新REF_ELIGIBILITY_UNDETERMINED。emptyと分類しない |
| 未確定なし、全件確定不適格（空集合を含む） | REF_ELIGIBLE_EMPTY |
| 明示選択候補が未確定 | 既存SELECTED_ELIGIBILITY_UNDETERMINED |
| 明示選択候補が確定不適格 | SELECTED_REFERENCE_INELIGIBLEを先に記録。通常のregretはnull |
| optional判定診断でpredicted eligibilityが欠測/None、acceptなし | 新PREDICTED_ELIGIBILITY_UNDETERMINED。MODEL_ALL_REJECTEDへ合算しない |

新二statusと `track_a_ax1b_selection_v2` はこのamendmentによる結果前追加である。
AX-1a五正本のbytesは書き換えない。prediction eligibilityの追加statusは既存optional audit引数の
欠測処理の訂正であり、operational bias/shot predictorを追加したものではない。

既存のoutside/false acceptance/selected-undetermined、missing prediction、reference cost未定義、
zero denominatorは専用statusを維持する。主statusがavailability問題でも
`reference_eligibility_status` と未確定一覧により、全集合の適格性未確定を同時に表示する。
未定義regretを0、未知eligibilityをFalse、未知costを0/∞として補完しない。

## Full setとsubsetの区別

- `regret` と `full_set_min_G_ref` は各recordの登録評価集合を対象にする。その集合にNoneがあればnull。
- `known_eligible_subset_regret` は確定適格集合だけの診断。確定適格全件の参照費用と予測が揃い、分母>0のときだけ数値。
- `known_eligible_prediction_subset_regret` は確定適格かつ予測が存在する集合内の診断。集合hashを併記する。
- これらの部分集合指標はfull-set regretへ転記しない。未知候補の仮の低費用は確定分母へ使わない。

`common_support_selection()` のsupportは従来通りavailable calibrated modelsのcomplete **one-shot cost**
predictionの共通集合。eligibilityによってsupportを絞らないので、None候補もそのまま残る。
そのsupportにNoneがあればcommon-setのregretもnull。None候補がcost欠測でsupport外となる場合は、
除外一覧と集合hashを出す。common-set内で定義できたregretはそのsubsetだけの値であり、
`full_set_diagnostic.regret` は依然null。full setとcommon setを別objectで保存する。

## 呼び出し側・schema・集計の監査

| 経路 | 反映・検証 |
|---|---|
| [ax1b_evaluation.py](../../src/trottertracks/resource_applicability/ax1b_evaluation.py) | selection/common support、三状態、集合hash、subset fields、status validator/summary |
| [ax1b_analysis.py](../../src/trottertracks/resource_applicability/ax1b_analysis.py) | 全selection経路は共通修正版を呼ぶ。status集計をshot_availability.jsonへ追加 |
| full210→development218、M2別5 | 候補集合不変。PM1/M2はfitに使わず、Noneを診断から削除しない |
| fold別・pooled Q cross-fitted | 同じNone伝播。pooledはsingle_frozen_model=false、単一凍結モデルと区別 |
| ε=.05/.01/.005/.001 | 全anchorで伝播を合成223候補により検査。新signal精度検証ではない |
| CSVとstatus集計 | nested full/common別に保持。registered_set/common_support/standaloneの別counts。未定義はnull/空欄 |
| [ax1b_execution.py](../../src/trottertracks/resource_applicability/ax1b_execution.py) | 出力前のstatus意味論/schema検査。不明statusや矛盾した成功をSCHEMA STOP |
| [runner](../../scripts/resource_applicability/run_track_a_ax1b.py) | default拒否は継続。manifest参照先だけ新Prelaunchへ変更、metadata検査は科学入力を読まない |

SINGLE/FEW/NNLS、特徴、scaling、rank/tolerance、membership、complexity gate、四case、
paired SE、operational N/Aは不変。モデル/readerのsourceも変更しない。
output directoryと登録12 filenameはAX-1aのまま。内容のselection schemaとstatus集計だけを拡張する。

## Stable environmentとsynthetic tests

既存Python/package inventoryをread-only確認した。stable pyenv3.11.1/3.12.3単体には必要packageがなく、
確認した他venvはRCまたはSciPy版不一致だった。新install/upgrade/venv上書きは行っていない。
stable **Python3.11.1 final** から既存cp311 site-packagesをprocess-local PYTHONPATHで参照して検証した。
NumPy1.26.4、SciPy1.14.1、pytest9.0.3。既存packageと他Trackの環境を変更していない。

```bash
PYTHONPATH='/home/abe/Project/Partially Randomized Trotter/.venv311/lib/python3.11/site-packages' \
PYTHONDONTWRITEBYTECODE=1 \
'/home/abe/.pyenv/versions/3.11.1/bin/python' \
  scripts/resource_applicability/run_track_a_ax1b_preparation_tests.py \
  --audit-output /tmp/ax1b_prelaunch_synthetic_audit.json
```

専用suite **107 passed / 0 failed / 0 skipped**。既存83件を含み追加24件。
科学入力読取、保護access試行、科学module import試行、実データfitは0。
全repository/旧科学testsは実行していない。資源enforcementのsynthetic testはOS操作をmockし、
テストprocessのaffinity/rlimit/timerは変更しない。実データのwall/RSS benchmarkも実行していない。

環境fingerprintは `314b059957876f54e3e82d97d0368ccd3ef9157aec335bb04d057123c2fc10f5`。
実行file hash、prefix/PYTHONPATH、package roots/METADATA hashes、NNLS source hashを含む。
authorizationでは環境全体のexact一致と、同環境・同sourceのsynthetic成功を要求する。
これは共有依存先を使う実行候補である。独立venvへ移す場合は別fingerprintになり、同sourceで再testが必要。

## 資源：観測、提案、未確認を分離

Linux affinity/cpusetは0–31。cgroup memory.max/highはmax、RLIMIT_AS/CPU/FSIZE/RSSはunlimited。
これは専有CPU/RAMや無制限実行の承認ではない。共有memory.currentやhost MemTotalから割当を推定しない。
statvfsの空き容量は新auditの観測値、quota確認toolと承認済みdisk割当は未確認。
外部wall-time割当情報も確認できない。authorization resourcesの四上限とaffinityはnullのまま。

提案はprocess1/BLAS1、1 core（例CPU0）、**AS 8 GiB / wall 300秒 / output 512 MiB**。
allowlist metadataの総入力45,995,823 bytes、210/8/5と固定fold数、小さいNNLS行列、sourceの
全入力保持・JSON展開・出力serializationを踏まえた保守的候補である。入力展開20倍でも約0.86 GiBに
加えlibrary/出力用の余裕を取る仮定であり、実測RSSや必要量保証ではない。wall/diskも計画上の上限案で、
完了時間・outputサイズの保証ではない。割当・quota・他processとの共存確認後に別authorizationで承認する。

runnerはaffinity、BLAS1、RLIMIT_AS、SIGALRM、write前disk capを適用する。
RAM値はRSS予約ではなくaddress-space上限、wall timerはauthorize後/import前から、disk capは新outputの合計。
共有host全体の容量を予約しない。入力hash/環境/予算/source不一致と出力衝突はSTOP、retry/resumeなし。

## 新bundleと次のreview

新namespace：[track_a_ax1b_prelaunch/2026-10-09](../../artifacts/resource_applicability/track_a_ax1b_prelaunch/2026-10-09/)。

- [prelaunch_manifest_v1.json](../../artifacts/resource_applicability/track_a_ax1b_prelaunch/2026-10-09/prelaunch_manifest_v1.json)：source/testと各証拠hash、旧契約参照、部分完了、禁止状態。
- [schemas_v2.json](../../artifacts/resource_applicability/track_a_ax1b_prelaunch/2026-10-09/schemas_v2.json)：selection statusと新bundleのschema。
- [synthetic_test_audit_v2.json](../../artifacts/resource_applicability/track_a_ax1b_prelaunch/2026-10-09/synthetic_test_audit_v2.json)：stable環境・command/env vars・107件・九source bytes hashes。
- [environment_resource_audit_v1.json](../../artifacts/resource_applicability/track_a_ax1b_prelaunch/2026-10-09/environment_resource_audit_v1.json)：inventory、環境fingerprint、cgroup/ulimit/空き容量、候補予算と未確認の区別。
- [execution_authorization_draft_v2.json](../../artifacts/resource_applicability/track_a_ax1b_prelaunch/2026-10-09/execution_authorization_draft_v2.json)：確認済み候補環境/testのみ記入、実行認可false、割当/review/launchはnull。
- [blockers_v2.json](../../artifacts/resource_applicability/track_a_ax1b_prelaunch/2026-10-09/blockers_v2.json)：未確定割当予算、review、source binding、別authorization/launch。

自身を含むcommit SHAはself-referenceにせず、今回の正確なSHAを完了報告し、将来の別authorizationでbindする。
source/input file hashesはraw bytes、manifest/authorization bindingはsorted-key compact JSON UTF-8 SHA-256。
次のreviewでは新status/subsetの意味論、共有依存を使うstable候補環境の採用、割当予算とquota、
commit/hash bindingを判断する。独立execution reviewと利用者launch記録は未取得null。

`ax1b_analysis_authorized=false`、`science_authorized=false`、`explicit_user_launch_required=true`、
`mandatory_stop=true`、`next_stage_authorized=false`。本作業はAX-1b解析やAX-2以降への認可にならない。
