# 別サーバーのCodexへの依頼：H4作業移行・環境監査・監視修正・起動準備まで

## 今回の目的と認可範囲

前サーバーのH4 geometry作業を、このサーバーの自分専用checkoutへ引き継いでください。
環境は前サーバーに近い見込みですが、同一と推定せず実測して照合してください。
今回認可するのはrepository取得、移行状況・環境/資源の読取監査、専用branchでの監視/serialization修正、
対象を絞った人工tests、source/plan/認可草案の固定と報告までです。
**本計算・入力再生成・GPU使用は開始しません。新サーバーのCPU許可・最終review・明示launchを別に確認してSTOPします。**
旧サーバーのCPU番号とapproved=true記録は旧host/runに限る履歴で、新サーバーの許可として流用しないでください。
共有環境や他ユーザーへの影響がある変更は絶対に行わないでください。

## 取得すべき固定identity

- Repository/origin：HIROMU1015/Partially-Randomized-Trotter / https://github.com/HIROMU1015/Partially-Randomized-Trotter
- 引継ぎbranch：`track-a-h4-lazy-identity-run05-20261006`
- 引継ぎcommit：`8f77bebf99c5bd15fa6419c1f58556c3bd2837a9`
- science SOURCE_COMMIT：`6d365257770e99022b91d6a38dbee49ee0077503`
- 凍結入力のgeneration SOURCE_COMMIT：`049e69919af16ad29a67a217dc7a407d6b1754a6`
- 契約base：`b662dbd72e49fa713a25c716f323843e547e973b`
- 資料入口：`docs/research/track_a_h4_lazy_identity_run05.md`
- source/plan/audit入口：`artifacts/resource_applicability/track_a_h4_lazy_identity_run05/2026-10-06/README.md`

2026-10-07に旧サーバーでoriginの上記branchをls-remoteした結果、refはありませんでした。
旧サーバーではcommit済みですがpushはHTTPS認証エラーで未公開です。
**通常のmain cloneだけでは最新H4 sourceと修正資料を取得できません。**
最初にoriginとbranch/40文字SHAを確認し、公開されていればclone/fetchして上記commitをcheckoutしてください。
refがまだなければ、既存repositoryの読取や環境監査は進めつつ、利用者へ旧サーバーからの
non-force push、またはそのbranchを含むGit bundleの受領を依頼してください。
不明な別branch/mainを代替sourceとして扱わず、Git履歴・source commitの照合前に修正や計算を開始しないでください。
認証設定変更・force push・reset/clean/stash・既存worktree上書きをしないでください。

受領後は上記commitから独立worktree・専用branchを作ってください。
推奨名：`track-a-h4-new-server-monitor-preparation-20261007`。同名があれば連番を付けます。
新repo/checkout/input/evidence/output/python pathをabsolute pathで記録し、自分所有の作業領域内で扱ってください。

## 最初に読む資料

AGENTS.md、PROJECT_MAP.md、`docs/research/研究概要・現状.md`を読み、関連normative文書を確認してください。
VALIDATION_STATUS.mdとartifacts/validation_manifest.jsonも読み、旧研究・原稿・Track Bと今回の運用修正を混ぜません。
本指示と同梱のHANDOFF_STATE_v1.jsonは、引継ぎcommit後に判明したrun05停止を補足する最新記録です。
commit内の「production completion pending」「起動準備済み」や旧草案の未承認記載は当時の履歴です。
**現在の状態はrun05監視STOP、全owned processes終了、H4 map未完了です。**

## 現在の不具合と残る修正

run05は2026-10-06 23:28 JSTに`monitor interval/freshness`で停止しました。
12 workerを起動し候補間に8 invocationsを投入しましたが、run05完成compile0、signal0/1308です。
75 local testsは合格したものの、production成功は確認できていません。

既存修正：候補をまたぐbounded compile queue、lazy exact fingerprint、64KiB hash投入、
1回のserialize内の同じndarray parameter metadata共有、workerの前job参照解放、最初のmonitor例外ログ。
残る停止stackは`candidate_wrapper_jobs -> numerical_fingerprint -> serialize -> number -> exact(complex)`です。
hash以前の行列parameter変換も巨大なPython containerを作り、同じdriver process内のmonitor threadが
既存5秒deadlineを維持できない場合があります。GIL/GC/観測I/Oの内訳は追加切り分けが必要です。
`sleep(0)`やhash分割の既存人工payload testsだけで解決済みとしないでください。

以下を対象を絞って検討・修正してください。

1. 行列parameter/回路serialization全体を分割・streaming化し、巨大な中間copy・反復encodingを抑える。
   gate内容、phase、control state、ordering、dtype/shape、signed zeroを保持し、従来canonical bytes/digestと照合する。
2. 重い回路処理に監視が妨げられない実行経路へ分離する。
   独立observer process等を採用するなら、新own PIDのUID/starttime/親子関係・異常時停止・FD/pipe・終了手順、
   driver/worker/observerのRSS/AS/CPU/累積wall/output chargeを全て明示して監査する。
   observerの費用を無視して既存budget内とみなさず、追加roleや上限変更が必要なら実装案でSTOPし利用者へ報告する。
3. 分子データを使わない、実際と同じshapeの人工8-system+ancilla回路/parameterで、serializeからhashまでを検証する。
   12 worker相当の所有PID/RSS観測もmock/fixtureで含め、GIL占有/重いGC、観測遅延、worker死、欠測、pressure/OOMを検査する。
   source・CPU・memoryの許可が揃わない状態で12実workerのstress試験を起動しない。
   大規模test campaignは作らず、追加transpileが必要なら件数/cap/累積を先に明記する。
4. 最初の停止理由と実監視間隔・観測所要時間・driver処理phaseを保存し、interval遅延と観測自体のstalenessを区別する。
   監視失敗を無視せず、5秒制限・fail-closed・own-runだけの停止を維持する。

## Gitに含まれない入力・停止証拠

凍結6 NPZとgeneration-freeze、旧runのjournals/ledger/control/logはGit管理外です。
**cloneだけではこれらはありません。受領するまで入力を生成して穴埋めしないでください。**
利用者が移送する元directoryは以下です（旧サーバー上のpathであり、新hostに存在するとは限りません）。

```text
元の6入力・freeze（このdirectoryの内容をbyte-identicalにcopy）:
/home/AbeHiromu/projects/partially-randomized-trotter/artifacts/resource_applicability/track_a_h4_geometry_execution/track-a-h4-geometry-v2-20261006-run02

旧run01/02/03/04/05の保存証拠:
/home/AbeHiromu/projects/partially-randomized-trotter/artifacts/resource_applicability/track_a_h4_geometry_execution/

旧起動・監査・ログcontrol:
/home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/executions/
```

まず受領manifestとabsolute pathを確認し、元ファイルを移動/削除/renameせず、専用evidence領域へコピーとして保持します。
旧partial checkpoint/metrics/cacheは新runの計算結果として再利用せず、旧NPZ/freeze/source lineageだけを明示的に再利用します。
旧hostのPID番号を新hostでkillしないでください。旧pid/starttimeは保存証拠で、新hostの所有証明には使えません。
SSH秘密鍵・credential・旧venv全体を移送対象にしないでください。

generation-freeze file SHA-256：`75d7ddc8dc71ebeec03a6c173397a9b941b492b74e4dc80814d613d83ce56c69`。
generation-freeze fingerprint：`0fd1de52bce01c351c99efe0535913b2b3fd9a359ee72be36bc75cf847d193e7`。
NPZはstreaming byte SHAを照合し、人工testsへ実入力を使わないでください。
後の明示計算許可後にnative input gateが全32 arrays identitiesを検査する設計を維持します。

| 距離 Å | file | bytes SHA-256 |
|---|---|---|
| 0.70 | `input-0.70.npz` | `fc0cc9676ee576d7b4e150ab555acba798ffb8659b98f0a8165118e4e59e52b9` |
| 0.80 | `input-0.80.npz` | `25d8d222a2798b9986926cabeaeddb963bf476db92991cfd4f10f6e454ddcacb` |
| 0.90 | `input-0.90.npz` | `06672860ed8f9298053b4d84693fbab6d13e38c3166c27dbabcc161f33673c3d` |
| 1.10 | `input-1.10.npz` | `ed83af66303cdcb6847be1b89f77a86193c725e942ef4d0d717f0cd5b7e714dc` |
| 1.40 | `input-1.40.npz` | `50a9cece3c2d14b830233c571b543e916126efe581593c194fd5649ee638ec22` |
| 1.60 | `input-1.60.npz` | `504c7647e35fe17359b6545078e30b85fee6795316081b52f28c426c4490e1e8` |

## 「ほぼ同じ環境」を読取監査で確認

新hostのhostname、UTC/JST時刻、OS/kernel/CPU architecture、existing Python/venvのabsolute pathとversion、
45 distributionsのversion/installed RECORD SHA、compiler defaults/options/plugins、installed source hashesを照合してください。
旧環境は`/home/AbeHiromu/venvs/trotter-common/bin/python`、Python3.12.3です。
旧dependency/compiler referenceはcontract bundleとsource_freeze_v1.jsonにあります。
version一致だけでenvironment fingerprint一致と扱わないでください。
install/upgrade、system Python・共有venv・共有config/cgroup/quota/mount変更はしません。
差異は機械可読に保存し、既存のprivate環境を選ぶか、解決案を提示してSTOPします。

旧sourceのgates.pyにはARTIFACT_ANCHOR/PYTHON/OUTPUT/REUSE_ROOT/installed-source-prefix、
plan/source audit/reuse manifestにはactual checkoutや旧journal pathが固定されています。
別hostで古いpathへsymlinkを作って監査を迂回せず、新hostの実pathを必要な実行設定として明示します。
環境/資源bindingの更新は専用branchの新source/audit/planへ固定し、旧固定sourceと歴史的JSONを保存します。
source closureは旧19件＋必要な新moduleまで漏れなく含め、旧source/blobと新source/blobの差分を一覧化します。

processだけに次を適用します。共有shell/環境設定には保存しません。

```text
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1
QISKIT_PARALLEL=false QISKIT_NUM_PROCS=1
```

Python runner/owned workersの`-P -B`を維持します。stdlib signal shadowを-Pで解決した履歴があります。
GPU query/use、nvidia-smi、CuPy importはしません。

## 新hostの資源と計算予算

旧CPU `[3,5,6,7,8,9,10,11,12,13,14,15]`/mask0xffe8は履歴のみです。
新hostのonline/cpuset/physical core/NUMA/SMT/CPU quota・3秒受動負荷、memory/cgroup visibility・全祖先limit/PSI/OOMをread-only確認します。
12 compile workersが候補ですが、利用者が新hostで許可するCPU集合は未確定です。allowed_cpusは許可が得られるまで空の草案にします。
own-runだけのtaskset commandは未実行で提示し、他jobのaffinity/processを変更しません。

既存上限：driver/worker AS/RSS各8GiB、headroom16GiB、12 workers admission120GiB、monitor5秒、
全campaign10GiB累積charge/72h累積wall/actual compile74784。科学scopeは変更しません。
入力受領後のfixed output filesystemについて、user向けavailable bytes/inodesとuser/group/project quotaを確認します。
未知quotaを「なし」と推定しません。ext4での既存stage planningは3.5GiB/560000inodes、全72h監視を含む保守的値です。
新filesystem形式やobserver/log出力が違うなら根拠付きで見積り直し、10GiB全量の空き確保を新たな必須条件にしないでください。
容量予約/dummy file・削除/移設は行いません。

同じcampaignの引継ぎとして予算をresetせず、停止run05までの次をcarryします。

- consumed/reserved actual invocations：**20**（旧12＋run05の8、未完了reservationも消費済みとして扱う）
- cumulative charge：**165214360 bytes**。新journal carry rowも別途課金する。
- cumulative wallの保守的carry：**5466.188392877579 s**（最終failure logまで＋60秒余裕）
- run05 journal SHA：`6b68368565cc336d283bc094f844a37ceb1966838ca5482f1f12347d0c5d669e`。
- 上限を維持する場合、新attemptの残actual invocation枠は**74764**。

旧停止runをresume/retryするのではなく、修正後の新SOURCE_COMMIT・新run ID・新empty output・新bindingで一度だけ実行する設計にします。
今回そのrunnerは起動しません。budget resetやcap変更が必要なら利用者へ提案してSTOPします。

## 維持する科学scope

H4 linear neutral singlet/STO-3G、4 spatial/8 system＋ancilla1、DF12。
距離0.70/0.80/0.90/1.10/1.40/1.60 Å、T0.8、二次DF-prefix PF/canonical finite-RTE。
各218 templates、L_D0/3/4/5/6/9/12、q1/2/4/8、delta0.8/0.4/0.2/0.1、固定r/K。
random32 paired trajectories、master20261006、1308 signal records、74784 logical wrappers。
既存sourceのseed identityはSOURCE_COMMITを含みます。sourceを変えた場合は同じseed算法/masterでも数値seedが再結合されます。
これを隠さず記録し、旧sourceのpartial/random結果と混合しません。
compiler seed17/optimization_level1/num_processes1等は固定referenceから機械転記し、独断で変更しません。
H6/Track B/追加trajectory/anchor/長RPE/最終総costへ拡張しません。

## 固定・公開・報告してSTOP

環境監査、受領file/hash manifest、停止証拠の照合、source修正理由/差分/closure、人工test結果/log、
resource/容量観測と新source/plan/auth-review binding草案、未実行absolute launch commandを一つの入口へまとめてください。
旧approved=true記録を新hostの承認に改変しません。新reviewはapproved=falseで固定します。
研究概要/関連normative文書/当日研究ノート/machine manifestと必要索引を、最新の停止・未承認状態で追記します。
軽量資料・source・testsだけを明示stage/commitし、Git originへnon-force push、remote SHAを照合してください。
NPZ/分子snapshot・実runtime/checkpoint/cache・credentialをcommitしません。
認証失敗なら設定変更せず、未公開commit SHAと手動push commandを報告します。

報告項目：取得commit/branch、environment差分、入力受領/hash、監視修正/人工testsの結果と未検証部分、
新CPU候補/memory/容量/quota・observer予算、累積budget、source/planのbinding、資料入口、commit/remote SHA、
残る新host CPU承認・最終review・明示launch事項。
**報告後STOP。利用者が新hostのCPU使用と計算開始を明示するまでproductionを起動しないでください。**

---

## 旧サーバーから資料を提供する際の参考command（未実行）

上記branchを公開する場合（旧サーバーで実行）：

```bash
git -C /tmp/track-a-h4-lazy-identity-run05-20261006 \
  -c maintenance.auto=false -c gc.auto=0 \
  push origin HEAD:refs/heads/track-a-h4-lazy-identity-run05-20261006
```

認証解決を待たずGit履歴を手渡しする場合は、旧サーバーでbundleを作る方法もあります。
保存先の同名fileを上書きしないことを確認してから実行します。本指示書作成時にはbundle/archive作成・転送はしていません。

```bash
git -C /tmp/track-a-h4-lazy-identity-run05-20261006 \
  -c maintenance.auto=false -c gc.auto=0 -c pack.threads=1 \
  bundle create /tmp/h4-run05-handoff-20261007.bundle \
  refs/heads/track-a-h4-lazy-identity-run05-20261006
```

Git bundleはNPZや未追跡runtime/controlを含みません。それらは上記専用directoryを別にcopyし、
コピー前後のSHA・bytes・file一覧を照合してください。転送先hostname/ユーザー/pathが不明な場合は推測して転送しません。
新serverのclone/fetch/bundle受領は認証設定変更不要な既存方法を優先し、Gitbundleも`git bundle verify`と取得SHAで検査します。
