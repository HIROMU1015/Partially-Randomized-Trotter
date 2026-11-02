# H4の6入力生成stage：容量確認と最終承認資料

2026-10-06 JST。対象は **6入力生成→freeze→STOP** だけ。
保守的必要量は **3 GiB（3,221,225,472 bytes）＋260,000 free inodes**。
2026-10-06 16:58:53.317 JSTの読み取り専用観測ではavailable **3,881,209,856 bytes ≈3.615 GiB**、
available inode **225,817,022**。user/group/project quotaはkernelの状態照会で非有効と確認した。
したがって、下記のsource境界と明記したmetadata余裕の下で、入力生成stage容量は **足りる**。
byte余裕は659,984,384 bytes ≈0.615 GiB、inode余裕は225,557,022。
これは準備時点の判定で、予約・将来の空き保証・科学実行承認ではない。

公開直前の17:17:59.216 JST再観測ではavailable **3,880,157,184 bytes ≈3.613678 GiB**、
available inode **225,816,970**、quota3種非有効。必要量余裕658,931,712 bytes ≈0.613678 GiB、
225,556,970 inodesで「足りる」を維持した。
[再観測JSON](../../artifacts/resource_applicability/track_a_h4_input_generation_storage_review/2026-10-06/prepublication_capacity_observation_v1.json)を追加し、
最初の観測履歴と見積りを保持した。launch直前のfresh検査は引き続き必要。

入口は[storage review bundle](../../artifacts/resource_applicability/track_a_h4_input_generation_storage_review/2026-10-06/README.md)と
[一括最終承認資料](../../artifacts/resource_applicability/track_a_h4_input_generation_storage_review/2026-10-06/FINAL_APPROVAL_PACKET_v1.md)。

## 旧10 GiB案を保存し、今回stageだけを判断する理由

旧CPU proposal commit `7c641ea8a2610e40667a311ebc48682702cc3a82` の「全10 GiB分の空き確保案」は当時の監査履歴として不変。
10 GiBはgeneration＋将来mapの**累積出力charge上限**であり、入力生成開始前に全量空き確保を要求する契約field/source gateではない。
今回sourceの配列/保存/監視/IO境界からstage-specific容量を計算したため、その旧案を入力生成の開始必須条件へ継承しない。
campaign cap10 GiBや科学条件を変更しておらず、signal/compile段の容量は今回承認・評価しない。

sourceは `9ab38665920dfb5ac0a9d038233e1f3bf5d8fe5a`、science checkoutは既存resource-fix worktreeのまま。
17 Python source＋親2件、plan、CPU入りauth草案、approved=false reviewはbyte-identical。
新source実装、科学array/NPZ生成、分子snapshot・runtime/cache/checkpointのアクセスは行っていない。

## 保存配列・NPZの見積り

`inputs.generate_input` の14 explicit entries、`factorize` の11 entries、`ground_state` の7 entriesをASTで照合し、計32配列を一覧化した。
全shapeとdtype basisはbundleの `array_and_storage_estimate_v1.json` に保存する。以下の表は科学arrayではなくmetadataだけ。
Hとblocksはcomplex128・256×256を13個、計13 MiB/入力。これは全保存内容ではない。
native f64/c128/i64を固定環境/sourceから推定し、AO積分等とraw eigenvectorsには16 bytes/itemの保守的ceilingを使う。
実dtype/bytesは未生成であり、実測値ではない。shape由来NPZ ceilingは約13.115 MiB/入力、6入力約78.7 MiB。

NPZは固定sourceの `np.savez`、installed NumPy1.26.4のZIP_STORED（非圧縮）、NPY member32個、force_zip64 local header。
NPY header/padding・ZIP local/central/filename/ZIP64を合わせて1,024 bytes/member、archive tail256 bytesを余裕として加算した。
この小さな数値shape/非object dtype群のheaderより大きいallowanceだが、実NPZ/headerは作成していない。
最終容量判定はdtype推定だけに依存せず、generation responseの **MAX_FRAME64 MiB** という凍結sourceのhard boundaryで各入力NPZを覆う。
6入力×64 MiBに加え、全6 fileのtemp/finalを別copyとみなす過大側allowanceを入れた。

実際の `OutputBudget.write` はtemporary fileからfinalへhard linkしてunlinkするので、同時の2名は同じinode/dataを共有する。
一方、source byte-budgetのchargeは `2*len(data)+128`。physical占有とchargeを区別し、容量見積りでは安全側に二重data allowanceも保持した。

## 72時間監視・journal・metadataを含む容量内訳

monitorは `finished.wait(1)` の後に毎回1個のwall JSONを保存する。
短時間終了を仮定せず、既存72h全量＋境界2 recordとして **259,202 files** を見積もった。
JSON payloadは1 record128 bytesを許容するが、filesystem4KiB丸めで約1,012.508 MiBになる。
worker text logはgeneration6 jobs各8KiB上限。generation-freeze JSONは6×32配列のshape/dtype/hash等のmetadataで、128KiBを余裕として使用する。
成功freezeのstdoutを保存する場合にも、freeze JSONのtemp/final二重allowanceに小さな二重metadata余裕を含む。
各publicationのbyte-budget journalは128 bytes fixed record。最大259,215 publicationsで約31.645 MiB。

| 項目 | 保守的allocation/allowance MiB | 根拠 |
|---|---:|---|
| 6入力NPZ、64MiB IPC ceiling×temp/final二重 | 768.000 | source hard ceilingと過大側二重allowance |
| 72h wall JSONの4KiB丸め | 1012.508 | source wait1秒、259202 records |
| generation-freeze JSON | 0.250 | metadata128KiB×2、仮定 |
| 6 worker text logs | 0.094 | source各8KiB×2 |
| byte-budget journal | 31.645 | source128 bytes/publication、block丸め |
| 新inodeごと1 blockのmetadata余裕 | 1012.605 | 259227 inodes×4096、保守的仮定 |
| directory entry/index余裕 | 63.289 | entryごと256 bytes、保守的仮定 |
| 新祖先directoryの初期block | 0.012 | 最大3 directories×4KiB |
| extent/journal等metadata追加余裕 | 64.000 | 保守的仮定 |
| 合計planning bound | **2952.402 ≈2.883 GiB** | 整数計算値はJSON参照 |
| 丸めた必要量 | **3072 MiB＝3 GiB** | 上方丸め |

inode内訳はwall files＋6 NPZ＋6 worker logs＋freeze＋journal、同時temporary8、ancestor directories3。
source由来最大259227個を260000へ切り上げた。temp/final hardlinkを実際は別inodeと数える必要はないが、追加temporary余裕を残した。
ext4 inode tablesは通常既存で、実inode sizeを256 bytesと決めつけていない。
1 inodeあたり全4KiB、directory entry256 bytes、追加64MiBはmetadata負担の**仮定/余裕**で、実生成filesystem占有を測ったものではない。
sourceのlogical cumulative byte charge ceilingは約863.268 MiBで、campaign cap10 GiBの内側。
RAM内のBytesIO・arrays・IPC buffersはdisk容量へ混同しない。

## filesystem・quotaの読み取り専用確認

fixed outputをresolve/stat/作成せず、既存上位 `.../partially-randomized-trotter/artifacts` のdescriptor/statvfsで確認した。
mount `/home`、ext4、device `/dev/nvme2n1p1`、4KiB block/fragment、rw。
nonrootが使えるf_bavailを使用し、root reserved blocksを含むf_bfreeを空きとして流用しない。
既存親はuid/gid30038、mode775、current euidでwrite/search accessをread-only確認した。probe fileは作成しない。

quota binaryは未導入なのでinstallせず、既存libc/kernelの `quotactl_fd Q_GETFMT` を使用した。
user/group/projectの3 typeとも **ESRCH** を返し、当該filesystemのquotaが非有効であることを確認した。
これは「未確認だからquotaなし」という推定ではない。
[Linux v6.8 quota_getfmt](https://github.com/torvalds/linux/blob/v6.8/fs/quota/quota.c)は非active quotaでESRCHを返す。
[quotactl_fd](https://man7.org/linux/man-pages/man2/quotactl.2.html)は既存file/directory descriptorを対象に状態照会できる。
既存親のproject id0・inherit flagなしもread-only FSGETXATTRで確認した。
初期のown GETQUOTAではuser/group ESRCH、project EPERMだった履歴を保存し、project未確認をQ_GETFMTで解消した。
quota設定の変更、quotaon/off、quotacheck、sync、容量予約、dummy file、削除・移設は行っていない。

観測時点ではquota残量の別上限は適用されず、実効availableはstatvfs f_bavail、inodeはf_favail。
将来quotaが有効になったり照会不能になれば、この判定を流用せず、残容量・inode確認までSTOPする。

## 最終承認とlaunch直前条件

承認対象：

> CPU [3,5,6,7,8,9]、6 worker、提示したown-run限定条件で、H4の6入力を一度生成し、freeze後STOPする。
> launch直前のfresh resource・容量検査が不合格なら起動しない。

CPU使用許可、独立最終review、承認済みreview文書の更新、利用者の明示launchは未実施。
保存reviewはapproved=falseのまま。source・plan・authorizationを変更しない。
taskset mask0x3e8は新規own-runとspawned workersだけの未実行案。既存process/他jobのaffinity・cgroup・priorityは変更しない。
launch直前にはonline/cpuset/CPU集合、初期namespace/祖先可視性、memory≥72GiB（6 workers）、pressure/OOM、
同じfilesystemのavailable≥3GiB・free inodes≥260000、quota非有効または残量確認済みを再検査する。
sourceのactual worker admission、AS/RSS/headroom、wall72h/output10GiB、one-shot/no retry/resume/fixed runを維持する。
容量不合格・確認不能なら起動しない。必要量との差分だけを解決案として示し、output変更/削除は別承認の提案に留める。

H4 linear/neutral singlet/STO-3G、6距離0.70/0.80/0.90/1.10/1.40/1.60、4 spatial/8 system、DF fragments12。
SCF/DF/order/solver/gates/master seed、218 templates、入力生成6件freeze後STOPは不変。
signal/compileの容量や次段認可は今回対象外。科学/Numpy array/NPZ/worker/taskset/GPU/共有環境変更0。
旧bundle・科学証拠・原稿・Track Bと旧10GiB案の履歴を保持し、資料commit/push後にSTOPする。
