# Track A AX-1b compiler identity互換性修正とmetadata-only preflight

2026-10-09 JST。利用者提示の独立review `APPROVE_TARGETED_IDENTITY_COMPATIBILITY_FIX` に基づく限定修正。
修正前sourceは `5412375e0e57cf3e114d5f68ed9ddbd433cfc4bc`。

**`AX1B_IDENTITY_COMPATIBILITY_PRECHECK_COMPLETE_EXECUTION_NOT_AUTHORIZED`**。
専用synthetic suiteは170 passed、fail/skip 0。45入力のmetadata-only preflightは `PRECHECK_PASS`。
本解析launch、実データmodel fit、性能・regret評価、新signal/sampling/compileはすべて0。
これはローカルの実装・準備検証であり、モデル性能やPR優位性の科学結果ではない。

## 保護対象と原本

AX-0/AX-1aの契約、旧科学結果/status、原稿、Track B、既存dirty整理差分を変更しない。
旧Preparation/Prelaunch bundleも当時の履歴として保持する。保護対象の既存索引は編集せず、
本書と[専用manifest](../../artifacts/resource_applicability/track_a_ax1b_identity_compatibility/2026-10-09/identity_compatibility_manifest_v1.json)で新source/test/runnerを結ぶ。

前回の `AX1B_STOP_INPUT_IDENTITY` はlaunch前の入力表現互換性問題である。
`/tmp/track_a_ax1b_execution_20261009_9dwp7iqq/` に残る停止監査とmanifestを読み、元manifestの5記録のhashも確認した。
原停止監査SHA-256は `2463c880b6d18d82c84b02a3c6c241941e542ec3656642489ee627e2aab4c040`。
2原本を専用bundleにbyte-identicalでコピーし、[来歴](../../artifacts/resource_applicability/track_a_ax1b_identity_compatibility/2026-10-09/prior_stop_provenance_v1.json)へ記録した。
原本のSTOP/status、0 launch、0 fitを成功へ書き換えていない。

## Amendment AX1B-COMPILER-IDENTITY-COMPATIBILITY-20261009-V1

### 静的な意味論確認

| source | 確認した処理 |
|---|---|
| [M1 candidate契約](../../src/trotterlib/pr2_matched_accuracy_m1_contract.py) | candidate ledgerでは基本5項目だけを保存する |
| [M1-B1 compiler契約](../../src/trotterlib/pr2_matched_accuracy_m1_b1_contract.py)・[実装](../../src/trotterlib/pr2_matched_accuracy_m1_b1_execution.py) | 実compileの `_compiler()` は8項目を `CompilerSettings` へ明示的に渡す。追加3項目はすべてNone |
| [PM-1契約](../../src/trottertracks/resource_applicability/pm1_discard_contract.py)・[実装](../../src/trottertracks/resource_applicability/pm1_discard_execution.py) | 同じ8項目・値でCompilerSettingsを作り、共通のmeasured Hadamard benchmarkを使用する |
| [共通transpile経路](../../src/trotterlib/rte_compiled_cost.py) | backend_name=Noneではbackend objectを受け入れない。layout/routing/couplingがNoneなら該当kwargsを渡さない |
| [M1 discard/action実装](../../src/trotterlib/pr2_matched_accuracy_m1_execution.py) | PM-1は `_prepare_discard()` を再利用。full one-body/scalarとempty-tailのaction policyを維持する |

両実装は `boundary_optimized`、同じwrapper semanticsを使い、backendを指定していない。
したがって今回の省略はcandidate ledgerの表現差であり、実compile policyの差ではないことをsourceから確認した。
この根拠は[意味論監査](../../artifacts/resource_applicability/track_a_ax1b_identity_compatibility/2026-10-09/compiler_semantics_evidence_v1.json)の9 source hashに結び付ける。
科学moduleのimport、build/transpile、保存costの置換は行っていない。
canonical比較の通過は、独立に再compileしたゲート列の一致確認を意味しない。

### 比較規則

[ax1b_data.py](../../src/trottertracks/resource_applicability/ax1b_data.py) の `canonical_compiler_identity()` は次の2形式だけを受け入れる。

1. 基本5項目のみ。
2. 基本5項目と `backend_name/layout_method/routing_method` の全3項目を持ち、追加項目はすべてNone。

基本5項目は固定のQiskit 1.3.0、basis `[rz,sx,x,cx]`（順序も一致）、optimization level 1、seed 17、coupling map None。
basisは文字列list、versionはstr、level/seedは厳密なint（bool/floatを拒否）。
追加項目の非None、部分的な追加、未知field、欠けた基本項目、基本値/型の差は `AX1B_STOP_INPUT_IDENTITY`。
未知fieldを一般的に削除するnormalizationは導入していない。

新しい比較viewだけを返し、raw dictionary、candidate identity、fingerprintを変更しない。
`validate_candidate_scope()` と `pm1_features()` は同じ関数を使用する。
保存record間のfull identity joinと凍結membershipはrawのexact比較を維持する。
wrapper semanticsの検査も従来どおり厳密である。

### PM-1 anchor共有

同じHamiltonian/state/state-vector/snapshot、identity policy、outer formula、coefficient tolerance、
wrapper semantics、T、qを要求し、M1 B0だけをanchorとする。
全該当anchorの `n_fixed` が非欠測かつ一致するときだけ共有する。anchor不足、不一致、欠測は従来のN/A。
参照bias・shot・compiled costをfeature導出へ使わない。

preflightではPM-1全8候補が各3件の同じqのM1 B0 anchorへ対応し、8件すべて `SOURCE_INVARIANT_SHARED`。
canonical比較view、元compiler identity、anchor fingerprint一覧、raw identity保護を候補ごとに監査した。
M1 210とM2 5のmembership/基本compiler policyに変更はない。M2をtraining/anchorへ混ぜない。

## Synthetic testsと環境

[専用test](../../tests/tracks/resource_applicability/test_ax1b_preparation.py) を
[専用test runner](../../scripts/resource_applicability/run_track_a_ax1b_preparation_tests.py) で実行。
既存107件と追加63件で **170 passed / 0 failed / 0 skipped**。
2形式、未知/部分field、非None、基本値/厳密型、wrapper、raw/fingerprint保護、全anchor条件、欠測・不一致、
foreign geometry/M2・非B0排除、参照値を使わないfeature導出、合成210/5のscope回帰、preflightの失敗・未認可拒否を検査した。
SINGLE/FEW_PARAM/NNLS、complexity gate、三状態eligibility/subset regret、operational N/Aの既存回帰も通過。

```bash
PYTHONPATH='/home/abe/Project/Partially Randomized Trotter/.venv311/lib/python3.11/site-packages' \
PYTHONDONTWRITEBYTECODE=1 \
'/home/abe/.pyenv/versions/3.11.1/bin/python' \
  scripts/resource_applicability/run_track_a_ax1b_preparation_tests.py \
  --audit-output /tmp/ax1b_identity_fix_20261009_ac6y2usu/synthetic_test_audit_v3_final.json
```

Python3.11.1 final、NumPy1.26.4、SciPy1.14.1、pytest9.0.3。
環境fingerprintは `314b059957876f54e3e82d97d0368ccd3ef9157aec335bb04d057123c2fc10f5`。
既存cp311 packageをprocess-local PYTHONPATHで参照し、新install/upgradeや他Trackの環境変更は0。
synthetic tests中の旧科学入力読取、保護access試行、科学module import試行、実データfitは0。
合成値によるfit/regret回帰はテスト内で実施した。実科学値による解析とは区別する。

## Metadata-only preflight

[新module](../../src/trottertracks/resource_applicability/ax1b_preflight.py) と
[専用runner](../../scripts/resource_applicability/run_track_a_ax1b_identity_preflight.py) を使用した。
本解析runnerは起動していない。MetadataReaderには解析用permitを発行しない。
事前の[request manifest](../../artifacts/resource_applicability/track_a_ax1b_identity_compatibility/2026-10-09/precheck_request_manifest_v1.json)へ
11 Python source/test hash、同環境synthetic proof、固定AX-1a契約/model hash、認可falseを固定した。
requestと結果を分け、監査が自身のhashへ依存する循環を避ける。

```bash
PYTHONPATH='/home/abe/Project/Partially Randomized Trotter/.venv311/lib/python3.11/site-packages' \
PYTHONDONTWRITEBYTECODE=1 \
'/home/abe/.pyenv/versions/3.11.1/bin/python' \
  scripts/resource_applicability/run_track_a_ax1b_identity_preflight.py \
  --metadata-only-preflight \
  --request-manifest artifacts/resource_applicability/track_a_ax1b_identity_compatibility/2026-10-09/precheck_request_manifest_v1.json \
  --audit-output /tmp/ax1b_identity_fix_20261009_ac6y2usu/metadata_preflight_audit_v1.json
```

45 allowlisted入力だけを開き、byte hash、登録JSON root/version/field cardinality、CSV header、identityを確認した。
source text/Markdown/TOMLは登録byte hashで照合し、実行や埋込path参照は行わない。
科学数値は必要fieldの存在/型だけを確認し、平均・正規化・精度・shot・予測誤差へ変換しない。
I1の整数action invariantと同じqのanchor `n_fixed` 一致判定だけは、利用者指定のfeature導出可能性確認として実施した。
raw fingerprintの再計算は行わず、凍結membershipとrecord間のexact一致を検査した。

| 検査 | 結果 |
|---|---|
| 45入力hash/schema・前後byte hash | PRECHECK_PASS |
| M1 210 / PM-1 8 / M2 5、分離・join・membership・scope | PRECHECK_PASS |
| PM-1全8候補のcompiler比較と各3件のM1 B0 anchor | PRECHECK_PASS |
| n_det/n_fixed/E_randの入力・導出可能性 | 全223候補で構造上利用可能。科学値の新算出はしない |
| normalization値の照合、PM-2再会計、fit、誤差、shot/eligibility、regret、paired cost統計、再compile一致 | PRECHECK_UNVERIFIED：今回の範囲外で未実行 |

新しいidentity不一致や必要action fieldの欠測は認めなかった。
STRUCT accountingとoperational bias/shot predictionは既存のN/Aを維持し、補完しない。
[preflight audit](../../artifacts/resource_applicability/track_a_ax1b_identity_compatibility/2026-10-09/metadata_preflight_audit_v1.json)には
数値library/科学module import試行と禁止計算呼出し試行が0と記録されている。
`project_saved()`、`analyze()`、`fit_cost()`等を呼ばず、入力変更や科学出力生成も0。
PASSは構造的な実行前検証の通過であり、本解析の成功保証ではない。

## 資源とsource binding

前回利用者が認めたprocess側capを維持：CPU1/affinity[0]、process1/BLAS1、AS8GiB、wall300秒、output512MiB、GPU禁止。
CPU0利用可能、output parent書込可能、空きdiskは512MiB以上。適用可能な指示にscheduler必須条件は見つからなかった。
quota toolはなく専有予約もない。これらを専有割当やquota保証と解釈しない。
isolated metadata probeでaffinity[0]、RLIMIT_AS8GiB、300秒timer、BLAS環境値1を確認した。
実データ解析のwall/RSS benchmarkは0。後のlaunch前に共有資源・環境・sourceを再確認する。

source/test/入力file hashはraw bytes、manifest/authorization bindingはsorted-key compact JSON UTF-8 SHA-256。
成功したsynthetic testとpreflightのsource bytesは同一。旧models/evaluation/analysis/AX-1a contract bytesも維持した。
自身を含むcommitのSHAはbundle内に自己参照させず、実commit後の `source_freeze_v1.json` と完了報告で固定する。
後続の別authorizationでは、最新review済みcommitと同じbytes・環境・証拠へ改めてbindする。
修正前5412375のauthorizationは流用しない。

## GPTレビューへ戻す事項

1. 限定した2形式のcanonical比較と、実compile sourceによる同等性根拠を確認する。
2. PM-1全8候補のanchor共有が、他identity条件を維持していることを確認する。
3. metadata PRECHECK_PASSと科学値依存のPRECHECK_UNVERIFIEDを区別する。
4. 新source commit/hash、170 synthetic proof、preflight audit、環境とprocess capを束ねた別実行認可を判断する。

[新authorization draft](../../artifacts/resource_applicability/track_a_ax1b_identity_compatibility/2026-10-09/execution_authorization_draft_v3.json) は認可false。
残る実行blockerは独立execution review、新sourceのauthorization binding、別authorization、明示的user launch。
未実行の科学値照合は本解析時の検査であり、今回の互換性修正で成功と認定しない。
登録12解析出力と `run_v1` directoryは未生成。AX-2以降も未認可。

`ax1b_analysis_authorized=false`、`science_authorized=false`、`explicit_user_launch_required=true`、
`mandatory_stop=true`、`next_stage_authorized=false`。この準備作業を完了して停止する。
