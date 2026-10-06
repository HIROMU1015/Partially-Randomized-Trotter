# H4の6入力生成：CPU・own-run条件・最終承認案

**CPU [3,5,6,7,8,9]、requested workers6、利用者承認未取得。**
状態は `H4_INPUT_GENERATION_CPU_LAUNCH_PROPOSAL_FROZEN_AWAITING_APPROVAL`。
今回は準備だけ。保存stage reviewはapproved=false、有効CPU使用許可・本番launch・独立最終reviewは未成立。

## 一度で判断する具体的事項

> CPU [3,5,6,7,8,9]、6 worker、提示したown-run限定起動条件で、H4の6入力生成を一度行い、freeze後STOPすることを承認するか。

これは承認判断用の文面で、承認や明示launchを記録したものではない。
CPU/worker/own-run条件への承認、独立最終review、承認済みreview文書の更新、利用者の明示launchが必要。
**現在はfilesystem容量条件も未解決。これらが揃うまで実行しない。**

## CPU集合とresource/context

| logical CPU | physical package | physical core | SMT sibling（観測） | NUMA node | sample core busy |
|---|---|---|---|---|---|
| 3 | 0 | 3 | [3] | 0 | 0% |
| 5 | 0 | 5 | [5] | 0 | 0% |
| 6 | 0 | 6 | [6] | 0 | 0% |
| 7 | 0 | 7 | [7] | 0 | 0% |
| 8 | 0 | 8 | [8] | 0 | 0% |
| 9 | 0 | 9 | [9] | 0 | 0% |

OS topologyで異なる6 physical cores、全てNUMA0。
2026-10-06 16:05:51.914–16:05:54.915 JSTの約3.001秒の受動sampleを使った。
CPU0–5を固定採用せず、全online CPUのSMT sibling負荷を比較した。
online/process集合0–255、可視cpuset有効0–127の共通部分内。
空き/OS利用可能性はCPU使用許可や専有予約ではない。将来の負荷・性能保証でもない。
topology・raw CPU counters・NUMA候補・時刻は[観測JSON](cpu_resource_observation_v1.json)に保存した。

own namespaceは初期cgroup `cgroup:[4026531835]`。
既存observerが全非root祖先をread-only確認し、実効available約981.887GiB、PSI full avg10=0、OOM counters0。
現在の広いprocess CPU集合は提案6 CPU以下ではない。起動時に新規own-runだけにtaskset maskを指定する案。
shared cgroup作成/移動、既存process/他jobのaffinity/priority変更は行わない。
今回はtasksetやworkerを起動しておらず、live mask継承・fresh admissionは未検証。

固定outputを置くfilesystemは既存上位`artifacts/`から確認した `/home` ext4。
availableは3,991,597,056 bytes（約3.717GiB）で、総output cap10GiBより少ない。
全cap分の空き確保という保守的運用案は未達で、launch容量条件を別レビューへ送る。
10GiBは実生成量ではない。6入力の実出力量は未測定、quotaとfuture容量も未保証。
output/registryへのresolve/stat/作成、領域削除、移設、共有設定変更はしていない。
初回の未作成親directory・二回目の容量条件停止を含む[全attemptログ/手順](OBSERVATION_METHOD_v1.md)を保持した。

## own-run限定launch案・不変条件

[将来command](FUTURE_LAUNCH_COMMAND_v1.md)は `/usr/bin/taskset --cpu-list 3,5,6,7,8,9` を使う。
maskは `0x3e8`。既存PID用のtaskset --pidを使わず、新規science runnerとそのspawned workersへ同じCPU集合を継承させる案。
science root/runnerは既存のresource-fix checkoutを使用し、proposal checkoutへ移さない。
Pythonは固定absolute interpreter、内部thread/process各1、環境/packageを変更しない。
対象は入力生成runnerだけで、plan/auth/reviewは新proposal bundleの絶対path。
既存subset guard、初期namespace/ancestor visibility、fresh pressure/OOM/admissionを維持する。

H4 linear/neutral singlet/STO-3G、6距離0.70/0.80/0.90/1.10/1.40/1.60、4 spatial/8 system、DF fragments12。
SCF/DF/order/solver/gates/master seed、218 templates/geometryは変更しない。
inputs/freeze digestはnull、入力生成6件のみ、一度freezeしたらSTOP。signal/compileと将来74,784 wrappersは未認可。
requested6/上限12、driver/worker AS8GiB/RSS別/headroom16GiB、8+8w+16GiB、6workerに72GiB、wall72h/output10GiBを維持。
actual wは既存admissionで6以下に決まる。worker増加・scope拡張・retry/resume・guard変更は行わない。

## source/plan/bindingとgate検査

- 起点branch: `track-a-h4-geometry-resource-observer-fix-20261006`
- 起点bundle commit/remote SHA: `a1b0ba5e1ae14c2fd7c345a4d979b85b1eff2538`
- science SOURCE_COMMIT: `9ab38665920dfb5ac0a9d038233e1f3bf5d8fe5a`（source19件不変）
- science checkout: `/home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/worktrees/track-a-h4-geometry-resource-observer-fix-20261006`
- proposal branch: `track-a-h4-geometry-input-generation-cpu-launch-proposal-20261006`
- proposal commit: この資料・manifestを収録する次commit。実SHAはGit履歴と最終報告で確認する。
- [byte-identical plan v2](input_generation_plan_v2.json)、[候補CPU入りauthorization草案](authorization_proposal_v1.json)、[未承認review](stage_review_proposal_v1.json)。
- [source/resource/binding監査](proposal_audit_v1.json)、[hash/fingerprint一覧](identity_summary_v1.json)、[manifest](artifact_manifest_v1.json)。
- [launch-contextの機械記録](launch_context_proposal_v1.json)、[12 gate tests](gate-tests-attempt-01.json)、[検査ログ](gate-tests-attempt-01.log)。

| production資料 | SHA-256 | domain fingerprint/digest |
|---|---|---|
| plan v2（不変） | `8d4ee43c3d7d74ba30cbd495a4c49dd0798ca27d5935df9df97125f3c069a256` | `8514368280d33bd89aff09891bbb25204d5f6e41acc7adcd176b317c0e5d6a5d` |
| auth候補 | `801ee37356a54e28be315ebf0570d0c2421fde7e66e5daca4500e22a2078ebe4` | `86d689c5dbdf496e0393b1a6442bcb975ffb0bbde4d0ea959962dd87353fbc1c` |
| approved=false review | `536122cf1d1aaae383b8a5e5873c3c58d928209ac645b752680c7db308233d22` | `f583dec002a674193e1605ca1af6f37e06ec67985ea02fd19049c902a949b684` |

plan/source_root/source19は完全不変。authはallowed_cpusだけ、reviewはauthorization_digestだけ変更した。
production JSONに独自承認fieldを追加せず、承認待ちは別文書/auditへ記録する。
12 metadata tests PASS、fail/error/skip0。保存review=falseの拒否、digest整合、source/plan不変、
旧auth/review混在・不正CPU・別stage/no launch・proposal rootの混入拒否を確認した。
合格authorize/checkoutはメモリ内だけの模擬review承認。approved=trueのreviewを保存しない。
private science/output/registry/production launch/worker境界はmock禁止した。

追加transpile0、旧系列28/64不変。benchmark/GPU/taskset/worker/科学処理/共有環境・他job変更0。
science source・旧bundle・証拠・原稿・Track Bは変更しない。
このlocal metadata案はCPU許可・予約・CI・独立外部再現・実行準備完了を示さない。
公開成功または認証失敗の報告後、承認待ちでSTOPする。
