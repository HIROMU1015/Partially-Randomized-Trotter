# H4入力生成 resource observer修正・new source-bound草案v2 review

状態は `H4_INPUT_GENERATION_RESOURCE_FIX_FROZEN_AWAITING_REVIEW`。
保存reviewはapproved=false、allowed_cpus=[]、本番未認可。CPU許可・最終review・明示launchを代行しない。

## 固定identityと読む資料

- Repository: `HIROMU1015/Partially-Randomized-Trotter`
- Branch: `track-a-h4-geometry-resource-observer-fix-20261006`
- 起点: `5245a29ca26cad7421640410934907647459b822`（fetch後remoteと一致）
- NEW_SOURCE_COMMIT: `9ab38665920dfb5ac0a9d038233e1f3bf5d8fe5a`
- 旧science SOURCE: `6a121725ce751affd2d3d131a84944728e6b2343`
- 契約base: `b662dbd72e49fa713a25c716f323843e547e973b`
- 更新bundle commit: 本資料を収録する次commit。実SHAはGit履歴・公開時の報告で確認する。
- new science/source_root: `/home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/worktrees/track-a-h4-geometry-resource-observer-fix-20261006`
- artifact anchor: `/home/AbeHiromu/projects/partially-randomized-trotter`
- fixed run: `track-a-h4-geometry-v2-20261006-run01`
- future output: anchor下 `artifacts/resource_applicability/track_a_h4_geometry_execution/track-a-h4-geometry-v2-20261006-run01`

1. [source実装資料](../../../../docs/research/track_a_h4_geometry_resource_observer_fix.md)。
2. [new source freeze audit](../../track_a_h4_geometry_resource_observer_fix/2026-10-06/source_freeze_v1.json)。
   17 source＋parent2件のnew SOURCE blob、変更対象外15 source、source import closure、依存45/installed source11/compilerを照合。
   resourceのadmission/Monitor/AS/wall/output等はAST不変。gates変更はnew audit pathだけ。
3. [source-bound plan](input_generation_plan_v2.json)、[authorization草案](authorization_draft_v2.json)、[approved=false review](stage_review_v2.json)。
   科学範囲/資源/permission条件は旧草案不変、source commit/hash/audit/rootと関連fingerprint/digestだけを再bindingした。
4. [resource/CPU未解決事項](resource_cpu_review_v2.json)、[最終audit](final_audit_v2.json)、[hash/fingerprint一覧](identity_summary_v2.json)。
5. [binding7件の結果](guard-binding-tests-attempt-01.json)、[ログ](binding-tests-attempt-01.log)。
   [observer33件・全実環境ログ](../../track_a_h4_geometry_resource_observer_fix/2026-10-06/README.md)も参照する。
6. [全artifact manifest](artifact_manifest_v2.json)、[commit対象監査](publication_scope_audit_v2.json)、[将来command・未実行](FUTURE_INPUT_GENERATION_COMMAND_v2.md)。

| 資料 | bytes SHA-256 | domain fingerprint |
|---|---|---|
| new source audit | `40e3e811326715270a86e22a33dd4fe57f80e07a4f544a0a4449d20c5dde91da` | `8576706fb69029428583d9690967e8f0b8cc5fc638e625357eedcd153e361cda` |
| plan v2 | `8d4ee43c3d7d74ba30cbd495a4c49dd0798ca27d5935df9df97125f3c069a256` | `8514368280d33bd89aff09891bbb25204d5f6e41acc7adcd176b317c0e5d6a5d` |
| auth草案 v2 | `ca6a73e884c4e5091d1aee5f14c65e782467fd720b1dd14903d06ce2bb2c2612` | `33beb412c139042b82cab1c4859c74a7d7e96b29ca9ddaf26b71c48688393c2c` |
| review v2 | `56d7cb60d2d1ce6a8182f52a1b039c863afb5a9327539d2ce160a611250c3e27` | `d7fdb69261b28342f4779f367770edeada92b2357a6849d46a77b151d8ec8833` |
| final audit | `e41d08bafbc42c66da84e37ca47dd120dda6e32ced3a56b89a87cdde12105fdd` | `ffd70107b1024d7835c8aed3dc8ee11e2ba2200bde65fa9c5dd6ec407fa80803` |

domainはidentity_summary各entryに記載。manifestは自己参照を避け、bytes SHA-256と
`h4-resource-fix-artifact-manifest-v2` fingerprintを公開時の外部報告で示す。

## root判定の根拠と修正

Linux v2のmemory.max/current/eventsは非root interfaceであり、真のrootでのENOENTを要求していた旧observerが原因だった。
[Linux v6.8 memory interface](https://www.kernel.org/doc/html/v6.8/admin-guide/cgroup-v2.html#memory-interface-files)を参照する。
root判定は初期cgroup namespace marker4026531835（[kernel proc_ns.h](https://github.com/torvalds/linux/blob/v6.8/include/linux/proc_ns.h)）、
mountinfo full-root `/`、所属path、shadow mountなし、観測前後の所属/namespace/mount一致で行う。
初期namespace以外で上位制限を排除できなければSTOPする。mount root `/` やfile欠測だけをroot証拠にしない。
全非root祖先のmax/current/events/pressureは必須。非root欠測/permission/不正値は拒否し、host-only fallbackしない。
root/global pressureはhost PSI、加えて全非root pressure/OOMを監視する。resource閾値・CPU guardは緩和していない。

## 実観測・検査と次レビュー条件

実環境の読み取り専用observerは成功した。kernel6.8.0-49-generic、初期cgroup namespace、3非root祖先＋rootを確認。
全3祖先maxはunlimited、観測host/effective available約981.826GiB、OOM0/PSI full avg10=0。
これは準備時点のmetadataで、memory予約、CPU許可、live admission/launch成立ではない。
unlimitedを確認した全祖先を検査済みなのでhost値が最小になったのであり、欠測によるfallbackではない。

observer33＋binding7＝40 tests PASS、fail/error/skip0、各suite1attempt、検査失敗0。全attemptログを保持した。
private science/output/launch/worker境界をmock禁止し、actual launch/OwnedRun/OwnedPoolは呼ばない。
metadata-only合格authorize/checkoutはメモリ内review承認＋架空CPU[0]だけで、保存review/CPU許可へ転記しない。
new/old source、plan、auth、review、audit、rootの混在を拒否した。
全repo tests・111-suite・benchmark・追加transpileを実行せず、旧系列累積28/64を維持する。

実行済みcommandは全ログのcommand_argvとfinal_auditへ記録する。各commandは以下のprocess限定環境を使用した。

```bash
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
QISKIT_PARALLEL=false QISKIT_NUM_PROCS=1 \
/home/AbeHiromu/venvs/trotter-common/bin/python -B \
  scripts/resource_applicability/run_h4_resource_observer_fix_tests.py --mode observer-tests

PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
QISKIT_PARALLEL=false QISKIT_NUM_PROCS=1 \
/home/AbeHiromu/venvs/trotter-common/bin/python -B \
  scripts/resource_applicability/run_h4_resource_observer_fix_tests.py --mode binding-tests
```

source audit/live observation/draft更新も同環境のguarded runnerで行った。
上記は実行記録であり、source固定後に再実行して既存監査を変更する指示ではない。

requested workers6、allowed_cpus=[]、approved=falseを保持する。process CPU0–255や候補0–5を利用許可とみなさない。
次には明示CPU許可、`process_cpus <= allowed_cpus` と初期namespace/root visibilityへ適合するlaunch context、
必要なauth/review再binding、独立最終review、利用者の明示launchが必要。今回affinity/priority/cgroupを変更していない。
将来launchでfresh memory/pressure/OOM/CPU/admissionを再確認する。private namespaceの実行contextを受け入れる拡張は未実装。

## 不変性・未検証・STOP

H4 linear/neutral singlet/STO-3G、6距離0.70/0.80/0.90/1.10/1.40/1.60、4 spatial/8 system、DF fragments12。
SCF/DF/order/solver/gates/master seed、218 templates、入力生成6件freeze後STOP、将来74,784 wrapper capを維持する。
inputs/generation freeze digestはnull、signal/compileは未認可。
requested6/max12、AS各8GiB/RSS別/headroom16GiB、8+8w+16GiB、wall72h/output10GiB、one-shot/no retry/resume/fixed runも不変。
SOURCE固定後の草案commitで17 source＋親2件、独立helper/tests、source段25ファイルは一切変更しない。
旧source247/保存6 JSON、旧契約/source/review/認可v1 bundle、原稿・科学結果・Track Bを保持する。

実SCF/DF/state/分子入力、実worker/production性能、実障害・pressure下のprocess停止、全campaign資源内完了は未検証。
分子アクセス/生成、実signal/sampling/seed/build/compile、追加transpile、GPU、本番起動、共有環境/他job変更0。
有効execution authorization0、最終review/明示launch未実施、実行準備完了とはしない。
認証失敗時は設定を変更せず未公開でSTOP。公開成功時もreport後STOPし、入力生成や認可有効化へ進まない。
