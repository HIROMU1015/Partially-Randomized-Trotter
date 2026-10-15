# H6 full-family：重複fingerprint走査の削減とcleanup修正

SOURCE `ac6d28e35cf01398f169ce59577182a38da9b959`。
[監査・人工検査・再実行案](../../artifacts/resource_applicability/track_a_h6_family_fingerprint_parallel_fix_v1/2026-10-15/README.md)。
旧SOURCE `99abc20a29979f3b6b11d9c7127c1e8cf84c643f` と停止run03を変更せず、独立worktreeで準備した。今回は新しい科学実行・grant発行0。

## 修正と高速化の範囲

親processは全selected native correctness合格後、12軸を直列に再fingerprintしてからforkし、workerの処理後にも親の回路を再fingerprintしていた。
worker自身も実入力をコンパイル前後に照合するため、親の2巡を省き、workerが返すbefore/after SHAをschedulerと最終会計の両方で必須にした。
12軸の場合、measured wrapperの走査は合計60回から36回へ減る。省く24回は親の直列走査で、残るworker内24回は並列実行する。Phase Bの正しさ検査と12回の基準SHA取得は維持する。

全native検査を完了してからforkする障壁、fork後のCOW入力、worker所有identity/native exit、前後の実入力SHA、compiler hash、指標検査を保持する。
同一processで親回路を変更する処理は追加せず、古い回路hashを無条件に信用する汎用cacheも導入していない。
canonical serializer、科学provider、回路構成、compiler optionsは不変。実回路の高速化率は未測定であり、走査回数40%削減を全実行時間40%短縮とは扱わない。

停止後はSIGALRM/TERM/INT/XCPUのPython callbackをown driver内だけで延期し、TERM待ち1秒・KILL待ち2秒で全owned childのwait/reapとpipe closeを行う。
独立metadata observerとlauncherの既存5秒TERM deadlineは継続し、SIGKILLは延期しない。失敗時はsignalをterminal記録・process退出まで延期してfirst STOPを保持する。
成功時だけtimer/maskを復元してPhase Dへ進む。未回収・signal error・FD errorは成功扱いしない。

## 前回の停止と再実行案

run03は6構成のnative correctness合格後、host full PSI avg10=1.44%によりSTOPした。観測時の有効空きメモリは464427130880 bytes、観測cgroup PSIは全0。共有hostのpressure原因を自runだけに帰属しない。
launcher wait137、compile予約6件、受理した新費用0件。実transpileへの到達・完了数は不明のまま保持し、予約は返却しない。
元cleanupは3workerのreap/exitを記録できなかったが、後続のread-only監査では元6workerは残存0。過去のexit codeや正確な終了時刻は補完しない。

再実行はまず2workerを提案する。6workerより同時forkの負荷を抑え、上記の直列削減を適用する。
AS32GiB/RSS9GiB、内部thread1、host PSI>=1% STOP・cgroup PSI0・OOM増分なし・5秒監視・既承認admission240.5GiBを維持する。worker減少でhost PSIが必ず低下する保証はない。
旧run03のメモリ上の回路は失われており、この変更に途中resumeは含めない。旧pilot6軸は元SOURCEで再利用し、selected新構成のbuild/native検査からやり直す。
前回同じscreenなら新6構成12compile。新attemptの上限は既存18compileのまま、過去の予約・unknownをcarryへ別途保持する。run03予約6＋次回12＝18だが、過去のpilot等を含む全履歴が18という意味ではない。

## 限定検査と残る準備

stdlib/mockの61件PASS（既存49＋追加12）。実科学import・NPZ読込・fork・transpile0。
前後のmutation、proof欠落/改変、最終会計の再照合、2workerの全軸取得、停止signal延期、first STOP保持、TERM/KILL/reap/FD、未回収失敗を検査した。
最初の人工試行はfixtureに命令数上限がなく16件失敗した。fixtureへ明示上限を追加後、61件全合格。全試行を監査に保持する。system Python3.12と本番private Python3.11.0rc1の両方で61件PASSを確認した。
主担当による限定差分確認であり、別担当の独立reviewではない。subagents0。

SOURCE/runtime closure14件のうち変更3・不変11。新SOURCE/profile/input/provider/carry、未使用output/control、fresh CPU・資源・quota gateは再実行前に結合する。
今回の案は未seal、`approved=false`、`runtime_authorization=false`、`allowed_cpus=[]`、`absolute_launch_command=null`。以前の実行・修正認可を撤回するものではなく、新grantはまだ発行していない。
この準備では本計算を再起動せず、既存環境・他job・H8・共有設定・旧証拠を変更していない。
