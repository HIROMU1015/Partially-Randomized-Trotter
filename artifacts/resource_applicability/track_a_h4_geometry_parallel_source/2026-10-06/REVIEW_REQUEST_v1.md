# H4 geometry bounded compile source再固定レビュー

`H4_GEOMETRY_PARALLEL_SOURCE_FROZEN_AWAITING_REVIEW`。科学未実行、公開後STOP。
レビュー対象はbounded compile制御・処理中owner・failure/accounting・new source gateの修正と固定である。
入力生成専用plan/authorizationを作成せず、別指示なしに本計算・H6・Track Bへ進まない。

## 固定identity

- Repository: `HIROMU1015/Partially-Randomized-Trotter`
- Branch: `track-a-h4-geometry-parallel-source-20261006`
- Branch base / old REVIEW: `4d5eba454dda06bc2735730cf7c3f132e456db29`
- Contract base: `b662dbd72e49fa713a25c716f323843e547e973b`
- Old SOURCE: `d6c7afd02dc0603216982a3cd4ea71b3d047dd8d`
- New SOURCE_COMMIT: `6a121725ce751affd2d3d131a84944728e6b2343`
- New REVIEW_BUNDLE_COMMIT: このreview資料4ファイルだけを追加する次commit。実SHAはGit履歴と公開時の報告で確認する。
- Contract plan SHA-256: `18aa36a2776d38852657f154a88c381b3299fbc009007f20a2d95fcb865d9f7a`
- Contract manifest SHA-256: `14cc5d0cc4da2b82168a0640cf8ff70ddf382b79810842bab1d7fedfae029f70`
- Old source audit SHA-256: `8c7d68c2f50753e22152079206e6d9a8e6e174a87f4b940b4cfed218c92e5eef`
- New source audit SHA-256: `5c9a1997339fa0f1f5479c62b11b6e2ef2ee024ce5a584aa958cfa80c4addd5f`

New actual checkoutは
`/home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/worktrees/track-a-h4-geometry-parallel-source-20261006`。
contract artifact anchorは `/home/AbeHiromu/projects/partially-randomized-trotter`。
future outputはそのanchor下の
`artifacts/resource_applicability/track_a_h4_geometry_execution/track-a-h4-geometry-v2-20261006-run01`。
本番output/registryをresolve/stat/作成していない。source gateは実際にimportするnew checkoutへ結合する。

## 読む資料と検査結果

1. [修正の実装資料](../../../../docs/research/track_a_h4_geometry_parallel_source_implementation.md)。
2. [New source audit](source_freeze_v1.json)：actual SOURCE_COMMITの17 Python blob、parent2件、AST imports、
   dependency45のversion/RECORD hash、installed critical source11件、compiler defaults/plugins/optionsを照合。
   old/sourceとの差分7件と変更対象外9件のhash、old247 source・保存6 JSON・旧bundleの不変性も記録する。
   全binary内容や独立環境での再現性は検証していない。
3. [Source段manifest](source_stage_manifest_v1.json)、[全artifact manifest](artifact_manifest_v1.json)、
   [commit/blob scope監査](publication_scope_audit_v1.json)。Source段24ファイルを固定し、review段では4新規資料のみを追加する。
4. [起点/実行前budget確認](identity_preflight_v1.json)、[test結果](test-attempt-01.json)、
   [testログ](test-attempt-01.log)、[new transpile予約](synthetic_transpile_reservations.jsonl)。

最終**111 tests PASS（既存94＋回帰17）**、fail0/error0/skip0。
今回test attemptは1回で失敗履歴0。old sourceの失敗attempt01/02/04は旧bundle内へ保持する。
新synthetic transpileは**3件**、old25＋new3＝**28/64件**。旧予約台帳はbyte/hash不変。
旧benchmark128は比較120（30 tasks×workers1/6/12/16）＋axis/phase4＋full-operator4で、再実行0。
実worker/production runnerは起動しない。fake futures/mock workersと小型synthetic fixtureだけで検査した。

既に実行した最終commandは次のとおり。同じ監査先への追加検査は今回の固定後に行わない。

```bash
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
QISKIT_PARALLEL=false QISKIT_NUM_PROCS=1 \
/home/AbeHiromu/venvs/trotter-common/bin/python -B \
  scripts/resource_applicability/run_h4_geometry_source_tests.py \
  --audit-dir artifacts/resource_applicability/track_a_h4_geometry_parallel_source/2026-10-06 \
  > artifacts/resource_applicability/track_a_h4_geometry_parallel_source/2026-10-06/test-attempt-01.log 2>&1
```

同じprocess限定環境でsource固定後に実行したread-only audit：

```bash
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
QISKIT_PARALLEL=false QISKIT_NUM_PROCS=1 \
/home/AbeHiromu/venvs/trotter-common/bin/python -B \
  scripts/resource_applicability/audit_h4_geometry_source.py \
  --source-commit 6a121725ce751affd2d3d131a84944728e6b2343 \
  --output artifacts/resource_applicability/track_a_h4_geometry_parallel_source/2026-10-06/source_freeze_v1.json
```

## レビューで確認する点

- `parallel.py` がadmitted w件以下に投入を制限し、全回路を先に生成せず、完了順に枠を補充すること。
  workers1/2/5/12で先行job完了前の複数投入と最大未完了数wを確認した。
- callback/ledger更新はdriverで行い、結果をlogical positionへ戻すこと。
  out-of-order completionでもtrajectory index・cosine/sine・6metric・weightは一致した。
  cosine/sineは同じseed/evolution、wrapper keyはaxis別である。
- 処理中ownerは結果待ちの依存として追跡し、RESERVEDの結果をcacheと呼ばないこと。
  3同一回路はactual fake compile1、両axis各32logicalはactual2、標本weightは各axisで合計1だった。
  正常COMPLETE＋独立expected/registry/digest検査の後だけ既存reuse gateへ渡す。
- identity不一致、failed/RESERVED/missing owner、cache chain、cross scope、digest改変を拒否すること。
  実pool.submit前のdurable予約を維持し、logical数とactual invocation数を分離すること。
- compile/worker/monitor failureを観測したら新規投入を止め、own runを停止すること。
  回収batch外の失敗も検査し、pool failure latchでqueued dispatchも拒否する。
  部分record/消費済み予約を保持し、retry/resume/補充/払い戻しを行わない。
- 同一合成fixtureで旧直列手順と新schedulerのsignal、metric pairing、302表示、P感度、weight、件数が一致すること。
  比較のsignalは2×2架空matrixだけで、実H4 signalや研究結果ではない。
- `gates.SOURCE_AUDIT` がnew auditを指し、actual new SOURCE_COMMIT/source closure/audit SHAを検証すること。
  old auditを新sourceの監査へ流用しない。review commitでSOURCE段を変更しないこと。

## 不変条件・未検証事項・停止

H4 linear/STO-3G、DF requested/returned fragments12、8 system＋ancilla1(index8)、T=0.8、二次DF-prefix PF/canonical finite-RTE。
6距離0.70/0.80/0.90/1.10/1.40/1.60 Å、各218 templates、random194各32 trajectories、signals1,308、logical cap74,784。
candidate/seed規則/sampling/normalization/shot、compiler/Gaussian/wrapper意味論は変更しない。
最大12 workers、内部thread/process1、AS各8GiB/headroom16GiB、wall72h/output10GiB、fixed run/root、二段階認可/STOPも不変。
`identity.py / inputs.py / signal.py / circuits.py / resources.py / review.py`等、変更対象外Python9件はbyte-identical。
旧bundle28、準備25、v1 26、v2 30、旧247 source、保存6 JSON、旧結果/原稿/図/Track Bも不変。
旧worktreeやmainの未commit変更を保持する。

実SCF/DF/state・分子入力、live workerの同時compile/性能、実IPC/AS/RSS/cgroup/CPU/pressure/OOM、
production filesystemのpower loss、全campaignの72h/10GiB内完了は未検証。
このbundleはlocal synthetic evidenceであり、実speedup・資源上界・科学成立・immutable CI/外部再現を示さない。

分子アクセス、科学入力生成、実signal/sampling/build/compile、GPU、実worker/本番起動、actual authorization発行、
共有環境/shell/package・他job変更はすべて0。分子NPZ・snapshot・runtime/checkpoint/cache/registryはcommit対象外。
このreview作成時点ではremote push未試行。成功とremote SHA一致は公開後の別確認が必要である。
認証失敗時は設定を変更せず未公開として停止する。
