# 2026-10-12 H6 hash再利用・8workerの実装


利用者の「ハッシュ再利用・8 workerの実装を入れて。今動かしている計算はそのままでよい」に従い、現4worker runを停止・変更せず、独立worktreeで実装した。

SOURCE `87b6b9f46b7e8ba9abe132ee21478c114b6f46f7`、[一つの資料入口](../../artifacts/resource_applicability/track_a_h6_confirmation_fast_preparation_v1/2026-10-12/README.md)。旧確認SOURCE73018cbcの245 source/science224はblob/SHA不変。追加runtime6件でclosure251、test2件を固定した。

## ハッシュ再利用

`h6_hash_reuse_v1.py`は呼出しごとに現在のordered graphを検証するguardを作り、変更のないoperationのcanonical bytesとcircuitの最終digestを内容で再利用する。永続cacheにmutable object IDを使わず、guard memo/strong referencesは呼出し内だけ。旧SHAは同じ `qiskit_recursive_numeric_circuit_v2` bytesに対するSHA256で、guard digestを科学結果へ置き換えない。

phase、dtype/shape/raw array bytes、ordering、conditionのbit context、control state/base gate、explicit definitions、signed zeroをguardに含め、symbolic/layout/calibration/cycle/object/nonfiniteはSTOP。数値arrayのnonfiniteもguardで拒否する（旧leaf helperはarray raw SHAだけだった）。正当なfinite circuitの旧/new bytes/digestを保つ。呼出し中のimmutable条件は旧規約と同じ。

固定Qiskit1.3.0のplain RZ/RX/RY/Phase/U/U1/U2/U3/R標準gateで、lazy definitionが未設定・元 `_define` methodと一致する場合に限り、同じnumeric headerの定義guardをcall内で再利用する。explicit definition・subclass・instance/class method上書きは再走査し、次callにはこのguardを持ち越さない。Qiskit packed instructionのreadで新Python gateが生成される際の重複を削減する。

canonical byte cache32MiB・entry1MiB・8192entries、digest1024entries、guard/standard memo各8192nodes。これはcache payloadの論理上限で、allocator/guard metadata/strong references/legacy call-local256KiB cache/hash buffer64KiB/一時copyを含む総RSS上界ではない。各9GiB RSS STOPで合計に算入する。oversize・容量不足はstreaming fallback。worker/groupを越えてcacheを持ち越さない。

固定scientific portそのものは変更せず、新worker内の単一thread contextだけでnative build・single-owner wrapper・matched portの既知3 fingerprint aliasesを置換し、例外時も元へ復帰しcacheを解放する。他processやdisk source/venvに作用しない。hash reuseのhit/guard/cache統計をworkerの上限付きwriterへ保存する。

## 8workerと引継ぎ

新fast contract/execution/runner/preparer/auditorを追加。元supervisorの所有確認、parent death guard、1秒観測/5秒fail-closed、PF_EXITING限定1秒待ち、欠測RSS保持、最初のSTOP、全wait/reap/pipe/log閉鎖はdirectory prefix以外同一。8 slotはaggregate reservationから計算し、予算が40GiBなら4へ減る。

提案CPUはworker20/21/24/25/26/27/28/29、coordinator22/23。読取時affinity内で物理core重複なし。各AS32GiB/RSS9GiB、coordinator soft2/hard32/RSS2GiB、aggregate76＝8×9＋2＋2GiB、jobRAM96GiB。fresh host248.25＝旧H4margin152.25＋96GiB/effective96GiB、running host168.25/effective16GiB、host full PSI5%・cgroup full PSI0・新OOMでSTOP、内部thread1/num_processes1、nice10・own affinityのみ。wall cap=null、元output2GiB/worker32MiB/log1MiB/journal32MiB/FS10GiB・10万inodeを保持。RAM予算はkernel予約ではなく、まだ実行認可ではない。

親4worker runのSOURCE/固定REVIEW3b6d733a/grant/profile/input/selection/task/seed、native launch binding・terminal/resource summary・retire/reapを確認する。生存中parentは拒否。supervisorで正常完了・exit0として登録されたwhole trajectory groupだけを、元sourceで両軸/event/compiler/call上限を受理し、metadata bytes/SHA inventoryを作る。partial/未観測/不一致を完成品として取り込まず、残りの元task/seedだけを新runへ登録する。

結果統合は確認group全32/cellの一度だけ、探索8をpoolしない。carryの既知完了callsは加算し、partialはfull registered taskで別予約して未知actual=nullを保持。新callsだけを加算して二重課金を避け、返却/resetなし。4partialならcompile8/trajectory4追加予約が生じ、compile1712→1720/trajectory840→844案となり得るが、実際の停止証拠で再計算する。新grantはSOURCE/hash policy/reuse/task/資源/caps/outputに厳密bindingし、旧4worker grantやdraftを拒否する。

Hamiltonian/DF/saved state、候補・epsilon、PF/PR/K/R、元seed/trajectory、normalization/allowance/shots、同じmeasured full wrapper/compiler・RZ/CX/depth定義は変えない。旧compiler/hostとのoutput完全同一性は未検証、既存の承認済みroundoff bridgeだけを維持する。

## 限定検証と現在の状態

101 local人工tests（7.26秒）が合格。13qubit人工回路・256×256 complex parameterのcanonical bytes/digest、各種mutation/非finite/上書き、cache上限/fallback/解除、8 mocked slotの二回dispatch/pressure/AS/RSS/欠測/cleanup、旧grant拒否、native complete-only親receipt・seed・carry/partial予約を検査。driver CPU30/thread1/AS2GiB/wall180秒/output16MiB、8実workerは0。科学NPZ decode/trajectory/SCF/追加transpile/GPUは0。

hash-only人工microbenchmarkは6 hash calls×2小回路、CPU30/AS2GiB/wall90秒。旧v5/newは0.709/0.097秒（7.28倍）と7.143/0.733秒（9.75倍）、全digest一致。これは実H6のwall短縮や8worker speedupの測定ではない。旧benchmark128やscienceの再計算はしない。

metadata-only auditで新worktreeの固定profile/input bytes・保存探索592・selection/taskを照合し、実際の生存中parentに対し `FAST_PARENT_STILL_RUNNING` でseal拒否を確認。approved=false/runtime_authorization=false/allowed_cpus=[]・absolute command=null、technical seal=false/new grant0/new runner0。

元runが継続しているため、ここで止めたり切替を行わない。利用者の切替指示後、旧run終了・own cleanup、最終receipt/remaining/carry、必要な予算変更と8worker資源の認可、fresh CPU/RAM/pressure/OOM/FS/inode/quota、未使用output/control・新one-shotを結合する。完了/STOP後mandatory STOP、自動retry/追加sample/H8/次stageなし。

資料はlocal source/review commitに固定。GitHub認証待ちは既存状態のまま、認証/共有設定を変更せず、remote公開済みを主張しない。新commit対象はsource/testsと軽量準備資料だけ、実runtime/checkpoint/cache/NPZは追加しない。
