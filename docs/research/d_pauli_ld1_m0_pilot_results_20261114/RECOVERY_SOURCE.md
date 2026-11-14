# run02復旧SOURCEと有限scope

2026-11-14。ユーザーの「復旧再実行してよい」に基づき、元B0条件の一回の復旧実行を準備した。
旧run01の停止結果b5fc2f97d、SOURCE62ae1bd5、全rawとprivate jobは保全する。
今回のcode変更はcallerのrun02出力先、復元関数を呼ばないfresh provider-first import preflight、
SOURCE記録への元run01 locator追加だけ。`pilot_core/child/records/canonical_request/preflight`は
元62aeとAST同一、connection・C planner/guard・native/providerは不変。

このworktreeだけのsparse checkoutへtracked referencesを展開した。
参照2 JSONは元62aeのblobとbyte一致し、既存saved H4 inputも元configのSHAに一致する。
元run01 raw/private SHAは不変。shared環境・他worktreeを変更していない。

新しい人工確認は親9件（元8＋import鎖/科学関数不変1）、内部の子人工5件がPASS。
`--import-preflight`は元install・provider照合・実restore/head/native module importまでを通し、
restore/context/prepare/sample/build/compileは呼ばない。科学実行前にもfresh childで再確認する。

固定scopeは元configuration d905d219、physical RMSE:62a9a6b6…、Pauli LD1/m0、T.1/q1/R3/K2、
184項・cohort D_PAULI_LD1_M0_WIRING_V1/index0/seed14788159701244282917、
PR_PAULI_TAIL / PAULI_D2_CENTRAL_ROTATION_CONTROL_9C4006BF、元scalar c+s_R一回、
測定付きXY最大2 compile、元full-2^n oracleと数値gateである。
Bの新configurationへ置換せず、K/seed/辞書/compiler/phase/toleranceも変更しない。

新公開出力は `docs/research/d_pauli_ld1_m0_pilot_results_20261114/run02/`、
privateは `/home/AbeHiromu/projects/d-pauli-ld1-m0-20261114-private/run02/`。
本技術変更をcommitで固定後、同callerの `--execute-approved-pauli-b0` を一度だけ起動する。
これは明示承認された手動復旧で、旧run01の未使用枠を自動転用したものではない。
さらなる自動retry・追加path/seed/compileは0。結果が失敗ならpartialを残してSTOP。
成功しても有限1 pathだけで、law平均/CI/model/B1/N/Gへ進まない。公開照合後mandatory STOP。
