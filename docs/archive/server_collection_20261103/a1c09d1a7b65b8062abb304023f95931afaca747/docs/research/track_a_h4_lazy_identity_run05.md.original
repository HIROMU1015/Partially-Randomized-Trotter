# H4 run05：全identity変換をlazy化・12 worker再実行

利用者の増員再実行指示と「続きを行って」に基づく修正。旧run04はfresh検査PASS後12 owned workersを起動し、5件の新compileを投入してmonitor interval/freshness STOP。完成compile/signalは0、全own processの終了を確認して全証跡を保持した。失敗stackはJSON encodingの前に巨大exact container treeを作る箇所だった。5秒deadlineを緩和せず、normalize/encodeを一緒にlazy走査し、64KiBごとにSHAへ渡してprocess内sleep(0)を使う。同じndarrayのparameter metadataは1回のserialize内で共有し、変更可能なglobal cacheやcircuitsの置換は使わない。

75 local tests（67関連tests＋正しいcounter初期化による人工serialization8件）がPASS、failures/errors/skips0。追加transpile0。最初のserialization8 pytestには既存runner COUNTS初期化不足による6fail/2passがあり、その履歴を保持した。controlled relative phase、params、Unicode、signed zero、ordering、nonfinite STOP、serial/12worker集計とlazy hash byte同値性を検査した。科学データを使わない256×256 complex-hex metadata128繰返しでは旧/新hashは8bc4b0421dd538a1c9078229e66ea184da5de5cc57b6c45a6485f9ffcf9cf216で一致。全監視PASS、39.49秒で終了。旧全JSON29.42秒/encodingだけstreaming59.60秒は診断比較であり、production速度や完了を保証しない。

- branch：track-a-h4-lazy-identity-run05-20261006
- SOURCE_COMMIT：6d365257770e99022b91d6a38dbee49ee0077503
- actual checkout：/tmp/track-a-h4-lazy-identity-run05-20261006。/home容量条件を圧迫しない/tmpの独立worktree。
- plan SHA：e6bccbe77e1728aa4238f2ee7113a81f8261b297030999f125f7a0dab18f2ed5
- plan fingerprint：191b5a4f191db3c32e91fa245b4d3460af72f3f6ad7b44a4544456d6baa8cae1
- auth digest：8c699e59e9ef102f2d24f0767b64ee5b81cf288feadbf570d744ce9bc1d7b127
- review digest：d62799a63cdcc161864cc7ffe28cfa5a939039214275ae698d0a33557bf92cc7
- CPU [3,5,6,7,8,9,10,11,12,13,14,15]、12 owned workers、own新driverだけmask0xffe8、内部thread1。
- fresh memory>=120GiB、nonroot available disk>=3.5GiB、inode>=560000、quota確認、PSI0/OOM増加0。
- cumulative carry：165184714 bytes、12 consumed invocations、5200.05529279015s。
- unchanged caps：10GiB累積charge/72h累積wall/74784 actual invocations。新attempt残invocation74772。
- driver/worker AS/RSS各8GiB、headroom16GiB、monitor freshness5秒。上限は増やさない。

H4 linear neutral singlet/STO-3G、4spatial/8system+ancilla1、DF12。凍結6距離0.70/0.80/0.90/1.10/1.40/1.60Åを元generation source049e69919af16ad29a67a217dc7a407d6b1754a6と元freeze/array identities付きでread-only再利用。分子アクセス・SCF/DF/state/input再生成をしない。T0.8、二次DF-prefix PF/canonical finite-RTE、218templates/距離、L_D0/3/4/5/6/9/12、q1/2/4/8、delta0.8/0.4/0.2/0.1、固定r/K、32paired trajectories/master20261006。科学式/回路/parameter数値とcompiler optionsは不変。SOURCE_COMMITを含む従来seed identityへ新sourceを結合するため旧random結果を混ぜない。

fresh gateはsource19/45dependencies/compiler・実checkout/plan/auth/review、6NPZ stream SHA/元freeze、run02/03/04 journal不変、全旧own processes停止、CPU online/cpuset/NUMA/physical cores/SMT低負荷/全祖先CPU quota、memory/pressure/OOM、fixed outputのmount/quota/available bytes/inodes、新output不在を照合。不合格や欠測ならSTOP。利用者の実認可を別reviewへ転記し外部reviewerを捏造しない。旧false草案は保持。

candidate間bounded compileはmax12 outstanding、候補/trajectory/axis集計順とcache scopeを保持。一度だけ実行し完成時signal1308/logicalwrappers74784のMAP_COMPLETE_STOP。失敗時もrunnerは自動retry/resumeせず全証跡を保持。旧source/入力/runtime/cache/checkpointを削除/移設/上書きしない。GPU/共有設定/他job変更なし、H6/TrackB/追加trajectory/長RPE/最終総costへ進まない。

実起動前source/plan/auth/reviewは本bundleに固定し、runnerはこのcheckoutのscripts/resource_applicability/run_h4_geometry_signal_compile.py。fresh検査後だけabsolute pathで一度execする。control入口：/home/AbeHiromu/projects/partially-randomized-trotter/.server-preparation/executions/h4-signal-compile-run05-lazy-identity/README.md。sourceはlocal実装検査済みでproduction成功・全campaign完了は未確定。科学NPZ/runtime/cache/checkpointはcommitしない。

[bundle入口](../../artifacts/resource_applicability/track_a_h4_lazy_identity_run05/2026-10-06/README.md)。
