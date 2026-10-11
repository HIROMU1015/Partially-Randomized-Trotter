# H4 resume01→resume02：高速化版への再開処理を準備

SOURCE `05a37cebaf18fa28e3a1e8582b86f54e32ba2a66`、branch `track-a-h4-resume02-preparation-20261011`。
起点REVIEW `28866c82ed5bc6f73de3eeb8063785502de878ba`、既検査の高速化SOURCE `def7d48f4af6c07f2d4e361fae868f3b01db8915`。
[固定source・限定mock tests・軽量review・binding・将来の手順](../../artifacts/resource_applicability/track_a_h4_resume02_preparation/2026-10-11/README.md)が資料入口。
現在の`h4-newhost-signal-compile-20261011-resume01`はそのまま継続し、今回の停止/引継ぎ/起動は0。

## 実装と来歴

[continuation_resume01.py](../../src/trottertracks/resource_applicability/h4_geometry/continuation_resume01.py)は親resume01のSOURCE `941749b8d4aedb43fd5f47b824116082493df393`、REVIEW `d0b4e1734b063c9fef353fdb3a7b63b26c961e72`、plan blob、開始receipt、live clockを固定する。
停止後に6 owned identitiesの残存0を2回・2秒以上間隔で確認し、全fileのbytes/streaming SHAとledger chainを照合する。
run10からimportしたrecordは元のbyte SHA・COMPLETE digest・元receipt tagに一致することを要求する。
resume01の新recordはexecution_source_commitが親SOURCEと一致し、元dataset identityから再計算したseed/key・数値fingerprint・metricsを検証する。
COMPLETEだけを再利用し、RESERVED/orphan/.pendingは保存して未完了分を再計算する。
source commitを置き換える場合でも、dataset/seed SOURCE `5aa6c3685afb9c3afcfbdc36f80244da0a7fe525`は保持し、新しいexecution SOURCEを別記録する。

[共有replay](../../src/trottertracks/resource_applicability/h4_geometry/continuation.py)と[Ledger](../../src/trottertracks/resource_applicability/h4_geometry/ledger.py)は累積actualを保持し、次のinvocation IDをcarry+1から発行する。
byte journalの先頭は既にprior_charge+128を含むので、journal sumに過去分を再加算しない。失敗/未完了の過去予約は返却しない。
wallもobserverの累積originを固定し、run10+resume01+次runの累積72hを維持する。
cache followerがlogical ownerより先に現れる場合は、検証済みownerを補助importし、最後に全logical位置へ登録されたことを検査する。
番号付きworker-logの`.txt`は実writerの保存形式と一致する限定patternでreceipt対象に追加した。未知fileは従来通りSTOP。

[閉じたscope schema](../../schemas/h4_newhost_launch_v2.json)と[launch gate](../../src/trottertracks/resource_applicability/h4_geometry/continuation_binding.py)は、旧run10→resume01に加えて`resume01_inverse_instruction_resume`だけを許可する。
次のrun IDは`h4-newhost-signal-compile-20261011-resume02`。未知generationは許可しない。
run10とresume01双方のoutput/control、入力、source、各receipt/native/clockのpathから新rootsを分離する。
compiler/environment/options、input/科学条件、CPU4 [2,4,5,6]・driver16・observer18、worker32GiB/driver8GiB/observer256MiB・64MiB、pressure方針、17GiB/74805/72hは不変。
科学H4 linear/STO-3G/DF rank12・6 geometry・218 templates・32 paired trajectories、structured Gaussian cost seriesを維持する。
基底inverseと命令/定義memoは既検査SOURCEのexact blobを要求する。旧compilerとのoutput完全同一性は引き続き未確認。

## 限定検証と軽量review

[実行前計画](../../artifacts/resource_applicability/track_a_h4_resume02_preparation/2026-10-11/test_plan_v1.json)は単一test process、内部thread1、Python -P -B、AS256MiB/RSS128MiB、wall45秒、output4MiB。
NumPy/Qiskit/分子/GPU import・NPZ読込・child/affinityを拒否するrunnerでmockだけを実行した。
最終24件PASS、failure/error/skip0。wall 2.124秒、kernel VmHWM 31457280 bytes。
最初の22件PASSにcache-owner順序とCLIを2件追加したところ、CLI guardがPython環境変数2件不足を検出した。
PYTHONNOUSERSITE/PYTHONDONTWRITEBYTECODEを新test processと計画へ追加して24件PASS。初回planと全試行logはprivate証跡に保持した。
実worker/科学input/transpile/新本体/affinityは全試行0。旧科学/人工compile/benchmark campaignは再実行していない。

2世代recordのimport、partialだけの再計算、seed保持、actual offset、journal二重加算防止、旧未完了費用保持、cache owner順序、signalのpaired metricsを検査した。
byte/lineage/source/carryの改変、欠測・orphan、native/clock/marker不一致、準備だけのauthority、unknown scope、科学source変更、根不足の容量boundを拒否する。
receipt CLIの`--parent resume01`と固定clock copyもmockで確認した。
[軽量review](../../artifacts/resource_applicability/track_a_h4_resume02_preparation/2026-10-11/delta_review_v1.json)は実装authorが差分と不変条件を確認、blocking findingなし。
独立最終launch reviewや全map成功の証拠ではない。source closure92は旧89中6変更/83不変、新protocol/runner/testsの3追加。
高速化circuits/signal/memo・sampling・execution/parallel・worker・observer・resource capsは変更していない。

## 今回確定したものと将来の引継ぎ

実装・mock回帰・SOURCE固定を完了。親source85件のbyte SHA一致を確認し、現在のrunを変更していない。
live clock contextは稼働中の6 owned identityを読み取り確認してhomeにexclusive保存した（630 bytes、SHA `8f12b48a1bca3d6a257149fe438a93b6c85ec0268ebc43717d71129e18cbb73e`）。
現在の観測でhistorical exit code・正確な終了時刻を補完しない。raw native/clock/runtimeはcommitしない。
親の停止proof・最終carry・再開receiptは親停止後にのみ確定し、新準備bindingはapproved=false/runtime_authorization=false/allowed_cpus=[]/未seal/absolute_launch_command=null。
今回の「対応を入れて」は準備だけのauthorityとして固定し、停止/引継ぎ/launchへの承認に変換しない。

将来の明示引継ぎ指示後、own resume01だけの停止とcleanupを確認し、停止receiptを作成する。
そのreceipt・SOURCE/profile/input/carry/output・plan/auth/reviewを再結合し、一度のrun02だけをsealする。
fresh CPU/memory/pressure/OOM/容量/inode/quota、未使用roots・one-shot lockが合格して初めて起動する。
carryは維持し、上限に達した場合にreset/返却しない。環境・資源・compiler条件の追加変更は今回行わない。

容量は祖先の一度のfresh全map bound + 現在の累積chargeで再見積りする。親のcarry込みboundを重ねて加算しない。
準備時のlive journal prefix概算は8,728,636,540 + 8,736,971,632 = 17,465,608,172 bytes（約16.27GiB）で、17GiB cap内。
これは停止receiptや起動許可ではなく、処理継続で増える。引継ぎ時の正確なcarry・容量・上限判定を再実施する。

## 固定・公開

source/tests/schema9fileをSOURCEに固定し、軽量監査・test結果・binding・資料入口・indexだけを別REVIEW_BUNDLEに固定する。
NPZ/実runtime/checkpoint/cache/raw native/log/認証情報はcommitしない。
originへのnon-force pushに認証が使えない場合は設定を変更せず、以下を利用者の認証済みterminalで実行する。

```bash
git -C /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h4-resume02-preparation-20261011 \
  -c maintenance.auto=false -c gc.auto=0 \
  push origin HEAD:refs/heads/track-a-h4-resume02-preparation-20261011
```
