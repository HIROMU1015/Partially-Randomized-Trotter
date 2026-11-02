# H4 geometry server-native source実装・科学未実行

2026-10-06 JST。base `b662dbd72e49fa713a25c716f323843e547e973b`。
利用者の添付指示で契約v2 D1〜D4を今回の実装条件に採用した。過去のv2未承認記録は不変。
現在は `H4_GEOMETRY_SOURCE_FROZEN_AWAITING_REVIEW`。別入力生成authorizationの作成へ進めるかをレビューする段階。
科学入力アクセス・生成、実SCF/DF/state、実signal/sampling/build/compile、GPU、環境・他job変更、authorization発行、production runner launchは0。

入口は[専用bundle](../../artifacts/resource_applicability/track_a_h4_geometry_source/2026-10-06/README.md)。
SOURCE_COMMITと、sourceを一切変更しないREVIEW_BUNDLE_COMMITを分ける。実SHAとblob/hashは後者の
`source_freeze_v1.json`に保存し、source自身へ自己参照SHAを埋め込まない。

## 構造と起動境界

`src/trottertracks/resource_applicability/h4_geometry/`の新namespaceに追加した。

| module | 役割 |
|---|---|
| identity.py | tagged IEEE hex/complex、signed zero、source/input-bound seed、axis別key |
| gates.py | production構造検査と意味論・二段階認可、actual checkout/依存45/compiler/契約37blob gate |
| inputs.py | checkpointなしRHF、raw gradient、積分/MO、raw DF eigenpair/order、sector/state数値gate・freeze |
| signal.py / review.py | finite-RTE、paired sampler、corrected/raw signal、302保存値表示、共通P affine感度 |
| circuits.py | production Qiskit数値serializer/round-trip、full Gaussian basis境界共有、controlled wrapper |
| ledger.py | 独立expected identity/registry、外部digest、非cache COMPLETE owner、予約とatomic delta ledger |
| resources.py | AS/RSS別、CPU/memory admission、上位cgroup、owned監視、72h/10GiB、exclusive出力 |
| workers.py / execution.py | 匿名pipeのowned spawn workers、generation freeze STOPと別signal map STOP |

future runnerは `scripts/resource_applicability/run_h4_geometry_input_generation.py` と
`run_h4_geometry_signal_compile.py`。今回はどちらも起動しない。module importは分子import・入力解決・出力作成を行わない。
新production wireのstructure gateを先に検査し、その後にstage/run/source/plan/authorization/review/explicit launchの意味論を検査する。
旧pure JSON validatorをproduction controllerとしてimportしない。authorizationsを作る機能は実装しない。

source checkout rootはこの専用worktree。artifact anchorは契約案の
`/home/AbeHiromu/projects/partially-randomized-trotter`、output rootも契約案のfixed run01を使い、両者を混同しない。
source gateは実際にimportされたcheckoutのpathとblob/hashを照合する。本番output/registryは作成・statしていない。

## 科学scopeと移植

H4 linear/STO-3G、all-electron/4 spatial/8 system＋ancilla index8、計9 qubits、requested/returned DF fragments12。
距離0.70/0.80/0.90/1.10/1.40/1.60 Å、T=0.8、二次DF-prefix PF/canonical finite-RTE、delta=T/q。
契約218 templateをそのまま参照する。B0=20/B1=4/B2=145/B3=49、random194各32 paired trajectories、baseline24。
1点12,464、6点74,784 logical wrappers/actual science invocation cap、signal1,308。
r64はB2-rank3-q1-r64-K2とB3-rank0-q8-r64-K2だけ。ineligibleも保持してshots/workはnull。

RHF constructorの暗黙NamedTemporaryFileを省いた新constructorを使う。通常CDIISのCorth初期化を保ち、DIIS保存はメモリ内に固定。
minao/dm0=None、50 cycles、conv_tol1e-9、raw gradient sqrt(1e-9)とstrict main-cycle/final gate。
AO積分配列をao2mo.kernelへ渡し、分子objectによるoutcore一時HDF5経路を避ける。
NoSaveRHFの実分子検査はまだ行っていない。PySCF共有設定の変更・monkeypatchは行わない。

OpenFermion低rank関数の数式と生成weight/reversed argsortを新sourceへ移植し、一度のfull16 eighからraw pair/permutationを保存する。
sign canonicalizationを追加し、null modeを残す。有意縮退・significant prefix weight tieでSTOP。
固有値数値閾値によるrankの再定義やFrobenius再ソートはしない。sector36・元Hの数値gate・8-bit reversalを固定する。
signal stageは保存stateのgate検査だけを行い、stateを再solve・置換しない。

既存RTEのpaired order weight、degree K+1 polynomial、rotation/product積順序とnormalization/shot式を移植した。
新D3独立streamへ変更し、cosine/sineは同じevolutionを共有する。wrapper keyにはaxis、seed/indexを入れる。
同じgeometry/cell/axis/数値fingerprintの非cache COMPLETE ownerだけを再利用し、各論理標本1/32を保つ。
full registered Gaussian basisの境界共有を明示した新wrapper identityで、旧geometry/cost/seedのidentityを流用しない。
元path/base commit/hashと意味論差分は監査JSONの `migrations` から辿る。旧source/guardは変更しない。

## serializer・atomic・資源

ordered bit/register/operand、exact parameter、global phase、condition/control state、custom definition閉包、axis/measurementを保存する。
open-controlはclosed/effective definitionを区別してround-tripする。非有限・symbolic・未対応operationを拒否し、角度の丸めはしない。
元operatorとserializer round-tripは全matrixを比較した。transpileの検査もcontrol qubitを含む全wrapper matrixを比較し、
全wrapperに共通するoverall phaseだけを許し、branch relative phaseを保つ。precompile fingerprintはそのoverall phaseも保存する。

compile前にinvocationを永続予約する。exclusive temp作成、file fsync、exclusive link、directory fsync、ledger commitの順。
中断窓のorphan/RESERVEDは消費済みのままSTOPする。delta ledgerは前digestに連結し、監査は独立のhead/stateへ照合する。
old SQLite/runtime/cacheや別runを再利用しない。output quotaはtemp＋final bytesとbudget journalも保守的に課金し、削除でquotaを戻さない。

required_available(w)=8+8w+16 GiB、w<=min(12,requested,明示許可CPUとprocess CPUの共通数)。64 GiBで最大5、12には120 GiB。
開始前だけworker数を決める。driver/worker RLIMIT_AS各8 GiB、RSS別監視、pressure/OOM/allocation failureでown run STOP。
cgroup v1/v2の見える全上位階層を検査し、mountで上位limitが隠れる場合は推測せずSTOPする。
spawnは匿名pipe接続の専用Python childにし、multiprocessingの補助resource trackerを作らない。監視と通信はdriver内の限定thread。
許可CPUがprocess CPUより狭い場合も、affinityを変更せずSTOPしてlaunch context reviewへ戻す。
generation/map間で消費wall/bytesを引き継ぐ。固定runの入力freeze後STOP、別review済みsignal認可後だけhandoffする。

## 合成検査と未検証事項

専用 `run_h4_geometry_source_tests.py` だけを実行。最終94 tests pass、fail/error/skip0。
合成transpileは全9 attempt合計25件（初回1＋後続8回各3）、上限64以内。旧benchmark128は比較120＋axis/phase4＋full-operator4で保存・再実行0。
開発中の失敗3 attemptは削除せず保存した。NumPy helperのsubprocess拒否、compiler計数wrapper signature、open-control閉包の修正履歴。
合成行列/回路は一時メモリだけ、ledgerはprivate `/tmp/h4-synthetic-*`だけ。本番data/outputのboundaryをnegative testsでmock禁止した。

実SCF/minao収束、real returned-rank/縮退、6分子入力・物理operator、live spawn/AS/RSS/cgroup、production filesystemのpower loss、
全74,784件のwall/memory/output内完了は未検証。unit/synthetic合格を科学成立・資源上界・CI/外部再現・完成総costとはしない。
今回発行した科学認可は0。公開後STOPし、入力生成authorizationの作成へ進めるかを別レビューする。
