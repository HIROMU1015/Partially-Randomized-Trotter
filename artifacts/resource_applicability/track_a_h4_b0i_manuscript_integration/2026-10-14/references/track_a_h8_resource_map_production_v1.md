# H8本格resource-map：実装・高速化・認可済み一回起動

2026-10-13 JST。SOURCE `cb8a56c6b2a772827275f9200b02638030f33f72`。branch `track-a-h8-resource-map-production-20261013`。

利用者の最新指示「並列化高速化してから計算を始めてください。計算を実際に始めるところまで進めて大丈夫です」により、先行のPREPARATION ONLYはこのcoreの一回起動について解除された。細かな実装判断を自律的に行い、実行前にsource/plan/auth/input/profile/carryを固定する。新しい科学条件へ変更する許可ではない。

対象は保存linear H8/1Å/STO-3G/tol-only DF1e-8/cutoff0/signed generation order/weighted Hermitian projection、actual rank28/16modes/Na=Nb4/sector4900/T=.8、snapshot SHA d4c4346127e97c241439ae1c4ae010962b0456b19f461cfd3d9b06e7d4a50955。元のHam/state/DF bytesを読み、SCF/DF/state生成を行わない。

## 固定した科学的比較

B0 prefix S2 12、B1 full S2/global Yoshida S4 8、B2 partial S2 72、計92。p7/14/21・q1/2/4/8・r1/2/4・K2/4、epsilon_sig .05/.01/.005/.001。測定込みpaired cosine/sineのfull Hadamard wrapper RZを主指標、CX/RZ depth/CX depth/total depth/size/shots/common stateprep P感度を副指標にする。Qiskit1.3.0/basis rz,sx,x,cx/opt1/seed17/topologyなしのcompiler hash1239304cを維持する。finite-time complex signal精度で、chemical accuracy/QPE energy精度とは別。

各適格PR候補8 trajectoryを探索し、精度別G_RZ点上位2・lexical cell ID tieのunion最大8を固定、その後の独立32 trajectoryで確認。探索と確認をpoolしない。seedはnamespace/source/input/cell/phase/replicaを結合しworker数や完了順から独立。q16、absolute p5 transfer、条件付きrare pilot、新H4/H6計算、GPU、次stageは含めない。q8は登録domain上限であり、境界での候補を自動拡張しない。

## 実装と数値品質

signalはqごと4groupに分割し、group内でDF preparation、CSR topology、spin primitiveと時間phaseを再利用する。B0は_準備をdiscard専用に分け、truncated exact referenceを独立CSRで評価してdiscard/PFのsigned分解を保持する。全92についてoriginal full DF reference、近似complex signal、bias、B/logB、u、headroom、eligibility、整数Hoeffding shotを保存する。axis配分/alpha .025/演算順/経験的uを旧実装から維持した。

16-modeでの数学的条件はspin保存・等しいalpha/beta block・Hermitian input・物理sectorのbasis対応である。native/spin/CSRの登録3probe、production/CSR/reference split-time、Horner/forward/raw-B closureは数値実装検査で、全operator/全数値allowanceの厳密保証ではない。小さいstate residualだけからbiasやallowanceを保証しない。ground-state/u/tail/population winnerのcertified flagsはfalseを維持する。

## 高速化と検証

Canonical v2 bytes/digestを維持し、32MiB operation payload cacheを飽和時にもLRU再確保する。古い永続的capture停止を避け、毎回current graphのphase/control/params/dtype/shape/order/nonfinite/signed zeroを検査する。final digestをobject identityだけで使い回さない。

Pinned standard gateのancilla-Z保存構造が確認できる場合に限り、両branchのsuperpositionを一回の実E作用へ渡して双方を取り出す。low branch、deterministic expected signal、ordinary対照、Hadamard X/Yを検査する。4 logical control probesを維持しphysical E作用を2→1に削減する。未知/ancilla mixing構造は元の2branch検査へfallbackする。単純なcontrol検査無効化ではない。

一つのretained evolution copyをsineに使い、native buildをcontrol当たり2→1へ削減する。元のsingle-owner wrapper、別axis compile、measurement/global phase/compilerを維持する。retained copy分をRSSに含める。旧／新mock compileのcanonical bytes・metrics・kwargs・events・seed・callsが一致した。16mode人工分岐も比較済み。旧保存deterministic4anchorは実際のfull-core測定後にcanonical/metric一致を検査するが、その費用やcallを再利用/返却しない。

新41 local artificial/mock tests PASS（8.57秒）：registry/budget/seed/selection、LRU bytes/mutation/nonfinite、joint/fallback、mock compile equivalence、B0 small dense reference、weighted admissions、native cleanup/STOP/failed reservation、rotation SHA、complete checkpoint/resume tamper、JSON roundtrip binding。np.load/実Qiskit transpileをtestsで禁止。旧15/44/40/23、実8signal/1164probe/20compileは再実行していない。CI/外部再現ではなく、実全体の速度倍率や8worker完走は未測定。

## 呼出予算・資源

Coreの保守上限はsignal92/normalization候補72、探索596 group、確認256 group、compile1708/trajectory832/occurrence4208/control376/primitive2172。normalizationはB評価を受けるsignal候補会計で、廉価な分布内部の各scalar演算数ではない。reference exponentialsは登録probe/ideal/truncated/exact-tailを合計2850上限、1actionあたりmatvec100000を実行中にguardし、reference matvec総上限285005104を設計した。この数は期待実消費やwall予測ではなく、保守的なcall capである。SCF/DF/state solverの新callは0。

H6 known+予約1736/852、H4補完532/256、H8 feasibility21/4から対象lineage subtotal2289 compile/1112 trajectory。新coreを全追加した提案・認可bindingは3997/1944。既存H4 production、unknown historyはこのsubtotalに含めず、global historical actualを正確と称さない。過去失敗1compile/4control、未知solver actual/null・予約10000を保持、返却/reset0。異なるfull namespace/source/seedにつきpreflight20compileの節約予約0。

Worker CPU20/21/24/25/26/27/28/29、coordinator22/23、全内部thread1。Worker AS32GiB、normal RSS9GiB、heavy/unknown24GiB、co RSS2＋margin2、aggregate76/job admission96GiB。全PRは最初24GiB、観測済みp7の4構成に限り実event全order0を受領した後に9GiBへpromotionし、新source実peak未測定/25%工学headroomという限界を保存する。p14/p21・未測定r/K・希少次数はheavy/unknownのまま。重み付きslotで8normal/3heavy/1heavy+5normal等だけを許す。signal開始は最大3worker、S4/q8が適格なら最初のcostは単独で測る。CPUの排他占有権をaffinityから推論しない。

Independent coordinatorのresource observation .2秒・staleness/所要時間上限5秒、host full PSI>=5%/cgroup PSI>0/新OOM/headroom/AS/RSS欠測超過はfail-closed。Fresh host248.25/effective96GiB、runtime168.25/effective16GiBを維持し、共有設定・他jobを変更しない。今回fresh native318files/profile/input5一致、available466.135GiB、disk374.473GiB、inodes219758187、quotaKNOWNを確認した。起動時にも改めてfresh gateを行う。

Output32GiB、storage admission64GiB、fresh inode200000・own output inode cap100000、worker output32MiB、log総128MiB。観測journalは8MiB segment・全phase合計1GiB内でexclusive rotation/SHA seal。checkpoint/index/一時cacheもown outputのbyte/inode計測へ含める。quotaUNKNOWNはSTOP。source/venvを変更せず、cache/MPL/tempはhome配下のown groupだけ。時間上限は設定しないが、資源とSTOPは維持する。旧実測S4/q8約7126秒/RSS6.82GiBを根拠とした旧条件付き8worker estimate17–69hはreference/signal/未観測p14/p21・rareを含む完走保証ではなく、新高速化の実倍率はまだ不明。

## STOPと手動complete-only再開

Fresh output/one-shot lockを消費して一度だけ進める。失敗時に後続phaseへ進まず、first STOP、native exit/wait/reap、失敗reservationとpartial evidenceを保存する。automatic retry/resumeは0。全scope終了後mandatory STOP。

完了groupだけcheckpointのtask/source/input/profile/seed/paired metrics/events/file SHA/native reapを照合して再利用可能。再開は同じ固定SOURCE、親coordinatorのnative退出、全own workers回収、別output/新grant/新lock/fresh gateを要求する。prepare --resume-parent/--native-exit/--sourceで手動設計し、旧grantを再使用しない。過去ledgerのknown/reservedを継承し、残仕事が既存call capに入らなければSTOP、勝手にcapを上げない。新SOURCEへの再開はこの版では拒否し、意味論・seed/carryの別bindingを要求する。

初回metadata準備2254d138はJSON tuple/list比較でFIXED_CONTRACTとなったが、runner/worker/lock/output/科学callに未到達。canonical digest比較へ修正し41回帰を通し、SOURCE cb8a56c6b2a772827275f9200b02638030f33f72・新auth02/run02へ固定。元auth01を保持する。

## 入口と公開

[一つの入口](../../artifacts/resource_applicability/track_a_h8_resource_map_production_v1/2026-10-13/README.md)。private `/home/AbeHiromu/projects/h6-handoff-evidence/20261013/h8-resource-map-v1/` のauth02/run02/control02。code/testsをSOURCE、registry/manifest/auth/profile/budget/tests/review/launch情報を別REVIEWに固定する。NPZ・実checkpoint/cache/raw result/credentials/internal SSHをcommitしない。
