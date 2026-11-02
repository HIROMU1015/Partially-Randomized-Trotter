# H4入力生成run02：worker bootstrap修正・6入力freeze完了

2026-10-06 JST。利用者の明示指示で修正して一度再実行し、**INPUTS_FROZEN_STOP**。
CPU [3,5,6,7,8,9]、6 workers、own-run限定mask0x3e8。fresh resource/容量/quota/CPU検査PASS。
6入力NPZを保存し、generation-freeze.jsonを固定。6 fileのbytes SHAをfreeze recordと照合し、
own driver/worker残存0、pending fileなし、next_stage_authorized=false、mandatory_stop=trueを確認した。
signal/sampling/circuit build/compile/transpile/GPU/次段は開始しない。共有環境/他jobの変更0。
これは入力生成stageの完了であり、geometry資源mapや最終total-cost評価の完了ではない。

## 修正と不変性

失敗run01はworkerのscript directoryにある研究signal.pyがstdlib signalをshadowし、
subprocess import時にrelative-importエラーとなっていた。private entry引数なしのbootstrapで再現した。
worker Popenへ-Pを追加し、回帰test1件PASS。科学jobを回帰testでは実行しない。
run01のsource/plan/review/logと空byte-budget.journalは保存し、削除・移設・resumeしない。
run02は別出力先/別run identity。source19の変更はgates.pyのrun/audit参照とworkers.pyの-Pだけ。
科学inputs/execution/signal/circuits/parallel/resources/seed、compiler条件、thread数、予算は不変。

SOURCE_COMMIT: `049e69919af16ad29a67a217dc7a407d6b1754a6`
prelaunch binding commit: `b3caa9923a83f5ec3b3d4cafed1998695e422a38`
source root: `/home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/worktrees/track-a-h4-worker-bootstrap-run02-20261006`
実行時のscience checkoutはclean、source19はSOURCE blob一致、plan/auth/review bindingを固定してから起動した。
旧run/outputを再bindingしたplanもnegative gateで拒否する。bootstrap修正でguardを緩めない。

## 固定科学scopeと実行結果

H4 linear/neutral singlet/STO-3G、4 spatial/8 system qubits、DF fragments12。
追加6距離0.70/0.80/0.90/1.10/1.40/1.60 Åの入力生成だけを今回実行した。
PF split L_D/delta window、218 templates、paired trajectories、signal/cost/mapは今回実行しない。
凍結sourceのSCF/DF/state acceptanceを通った入力を保存したが、独立CI再生成や次段の科学結論ではない。
6 NPZ合計 **82,321,596 bytes**。各32 array identitiesをfreezeへ保存。
NPZ bytesをhash照合するpost-checkではarrayを読み込んで再計算していない。
generation sourceのconsumed_secondsは約2.680秒。短時間を事前容量条件の根拠にはしていない。

fresh preflightのeffective memory約983.124GiB、nonroot available約3.554GiB、
quota user/group/project3種非有効、6 workers admitted。
必要容量3GiB/260000 inode、memory72GiB、AS/RSS8GiB/headroom16GiB、
wall72h/output10GiBの条件を維持した。観測・監視記録はserver-local controlに保存する。
10GiBはcampaign charge cap。次段signal/compileの容量確認・別認可は今回行わない。

freeze fingerprint: `0fd1de52bce01c351c99efe0535913b2b3fd9a359ee72be36bc75cf847d193e7`
freeze path: `/home/AbeHiromu/projects/partially-randomized-trotter/artifacts/resource_applicability/track_a_h4_geometry_execution/track-a-h4-geometry-v2-20261006-run02/generation-freeze.json`
output root: `/home/AbeHiromu/projects/partially-randomized-trotter/artifacts/resource_applicability/track_a_h4_geometry_execution/track-a-h4-geometry-v2-20261006-run02`
control root: `/home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/executions/h4-input-generation-run02-20261006T091149Z`

## 追跡入口

- [source/plan/auth/review・prelaunch bundle](../../artifacts/resource_applicability/track_a_h4_worker_bootstrap_run02/2026-10-06/README.md)
- [stage完了監査](../../artifacts/resource_applicability/track_a_h4_worker_bootstrap_run02/2026-10-06/input_generation_completion_audit_v1.json)
- [完了報告](../../artifacts/resource_applicability/track_a_h4_worker_bootstrap_run02/2026-10-06/COMPLETION_REPORT_v1.md)
- [完了資料manifest](../../artifacts/resource_applicability/track_a_h4_worker_bootstrap_run02/2026-10-06/completion_manifest_v1.json)

NPZ/generation-freeze/runtime/journal/監視logはserver-local科学outputに保持し、commitしない。
source系列の旧synthetic transpile28件は不変、今回追加transpile0。
利用者の別指示がない限り、freeze後STOPから次段へ継続しない。
