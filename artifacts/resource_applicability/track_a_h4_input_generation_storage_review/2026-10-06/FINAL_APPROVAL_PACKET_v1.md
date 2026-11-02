# H4入力生成6件：容量確認済み・最終承認待ち

2026-10-06 JST。容量準備の判定：**足りる**。CPU使用許可・最終review・明示launchは未取得。
status: `H4_INPUT_GENERATION_STAGE_STORAGE_CONFIRMED_AWAITING_FINAL_APPROVAL`。
本資料と未承認reviewでは計算を開始できない。公開後STOPする。

## 容量と根拠

必要量は **3 GiB（3,221,225,472 bytes）＋260,000 free inodes**。
source由来planning boundは2.883205 GiB、259,227 inodes。保存32配列、NPY/ZIP overhead、
6入力の各64MiB IPC hard ceiling、一時/final二重余裕、72h監視259,202 files、freeze JSON、
byte-budget journal、filesystem4KiB丸め、inode/directory/misc metadata余裕を含む。
metadata余裕は明記した仮定で、生成実測ではない。短時間で終わる期待は使用しない。

2026-10-06 **16:58:53.317 JST**、既存親
`/home/AbeHiromu/projects/partially-randomized-trotter/artifacts` でread-only確認：

| 項目 | 観測・判定 |
|---|---|
| filesystem/mount | ext4、/dev/nvme2n1p1、/home、rw、4KiB |
| nonroot available bytes | 3,881,209,856 ≈3.614658 GiB |
| available/free inode | 225,817,022 |
| own user/group/project quota | Q_GETFMTを3種へ照会、全ESRCH＝非有効 |
| quota残量・inode | 非有効のため別quota上限は適用されない（不明という意味のnullではない） |
| 親のアクセス | uid/gid30038、mode775、current euid write/search可、probe fileなし |
| 実効容量 | statvfs f_bavail/f_favail、root reserved blocks不使用 |
| 必要量との差 | +659,984,384 bytes ≈0.614658 GiB、+225,557,022 inodes |

初期project GETQUOTAのEPERM履歴も保存し、権限不要のGETFMTで状態を確定した。
quotaの新規設定・install、disk予約・dummy file、output/registry作成・探索、削除・移設なし。
観測は予約/将来保証ではない。launch直前にfresh再確認し、不足/不明なら起動しない。

根拠：[全保存配列・整数容量計算](array_and_storage_estimate_v1.json)、
[filesystem/quota観測](filesystem_quota_observation_v1.json)、[判定JSON](storage_verdict_v1.json)、
[計算・観測手順](READONLY_METHOD_v1.md)、
[詳細説明](../../../../docs/research/track_a_h4_input_generation_stage_storage_review.md)。

旧CPU proposalの「全10GiB空き確保案」は当時の履歴として保持する。
10GiBは全campaignの累積出力charge cap。今回stageは上のsource境界で足りるため、
入力生成開始前の10GiB全量空き要件にはしない。source/契約/capは変更しない。
signal/compile段の容量は別認可で確認する。

## 公開直前のread-only容量再観測

2026-10-06 **17:17:59.216 JST**、同じ既存親とquota getterだけを再観測した。
nonroot available **3,880,157,184 bytes ≈3.613678 GiB**、available/free inode **225,816,970**。
user/group/project quotaは今回も3種非有効、write/search access可。
必要量との差は **658,931,712 bytes ≈0.613678 GiB**、**225,556,970 inodes**。
判定「足りる」を維持する。[公開直前観測JSON](prepublication_capacity_observation_v1.json)に保存。
容量計算やscienceは再実行せず、この観測もCPU許可・review・launch直前検査の代用ではない。

## 固定identity・binding

| identity | 固定値 |
|---|---|
| Repository | HIROMU1015/Partially-Randomized-Trotter |
| base commit | 7c641ea8a2610e40667a311ebc48682702cc3a82 |
| science SOURCE_COMMIT | 9ab38665920dfb5ac0a9d038233e1f3bf5d8fe5a |
| source19 paths | base/SOURCE/science checkoutのblob/hash一致。変更0 |
| plan v2 SHA-256 | 8d4ee43c3d7d74ba30cbd495a4c49dd0798ca27d5935df9df97125f3c069a256 |
| plan fingerprint | 8514368280d33bd89aff09891bbb25204d5f6e41acc7adcd176b317c0e5d6a5d |
| auth SHA-256 | 801ee37356a54e28be315ebf0570d0c2421fde7e66e5daca4500e22a2078ebe4 |
| auth digest | 86d689c5dbdf496e0393b1a6442bcb975ffb0bbde4d0ea959962dd87353fbc1c |
| review SHA-256 | 536122cf1d1aaae383b8a5e5873c3c58d928209ac645b752680c7db308233d22 |
| review digest | f583dec002a674193e1605ca1af6f37e06ec67985ea02fd19049c902a949b684 |

science checkout/source_root：
`/home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/worktrees/track-a-h4-geometry-resource-observer-fix-20261006`。
plan/auth/reviewの本bundleコピーは旧CPU proposalとbyte-identical。
reviewはauth digestにbindingし、**approved=false**。auth/planは同じsource/plan bindingを維持。
[監査](final_storage_audit_v1.json)と[identity一覧](identity_summary_v1.json)にhash/bindingを保存する。
CPU案を文書へ記載したことはCPU使用許可の発行ではない。

## CPU案・fresh launch条件

CPU **[3,5,6,7,8,9]**、requested workers **6**、mask **0x3e8**。
own新規driver/workerだけにtasksetを適用する案。既存PID/他job/cgroup/priorityは操作しない。
既存single thread/process環境を維持する。共有設定は変更しない。
旧約3秒のCPU観測は履歴で、launch直前負荷の代用にはしない。

起動直前の条件：

- CPU集合/online/cpuset/topology/負荷が許可済み集合とown-run条件を満たす。
- 初期cgroup namespace・全祖先可視性、memory制限、AS/RSS guardを固定sourceのまま検査する。
- 6 workersにはfresh effective available **72GiB**、headroom16GiB、pressure/OOM等の既存条件を満たす。
  actual worker数はsource admissionで6以下に決まり、条件不合格をhost-only fallbackで通さない。
- 同じfixed-output filesystemでnonroot available **≥3GiB**、available inode **≥260000**。
- quota3種が非有効のまま、または有効quotaのown user/group/project残bytes/inodesを確認し必要量以上。
  quota確認不能・filesystem変更・容量不足ならSTOPする。
- source19/plan/auth binding、独立最終review、利用者承認、明示launchの全条件を満たす。
- fixed run/root、72h wall/10GiB output charge cap、one-shot/no retry/resumeを保持する。

未実行のabsolute commandは[FUTURE_LAUNCH_COMMAND_v2.md](FUTURE_LAUNCH_COMMAND_v2.md)。
保存reviewは未承認であり、commandの掲載は実行指示ではない。
fresh検査のためにsourceを修正せず、不合格時にlaunchしない。

## 利用者に提示する承認対象・停止地点

> CPU [3,5,6,7,8,9]、6 worker、提示したown-run限定条件で、
> H4の6入力を一度生成し、freeze後STOPする。
> launch直前のfresh resource・容量検査が不合格なら起動しない。

残る事項はこのCPU/own-run条件の明示使用許可、独立最終reviewと別の承認済みreview更新、
利用者の明示launch指示、fresh検査である。今回はいずれも自作・実施しない。

H4 linear/neutral singlet/STO-3G、6距離0.70/0.80/0.90/1.10/1.40/1.60 Å、
4 spatial/8 system qubits、DF fragments12の固定条件を保持する。
6入力を一度だけgenerateし、H/DF/state/sector/order/coordinate bytes等をfreezeしたらSTOP。
signal、seed/trajectory、circuit build/compile/transpile、GPU、H6/Track Bは開始しない。
今回は科学array/NPZ/分子生成0、worker/taskset/runner起動0、source修正0、
大きなtest campaign0、共有環境・他job変更0。source系列の旧transpile28件は不変、追加0。
