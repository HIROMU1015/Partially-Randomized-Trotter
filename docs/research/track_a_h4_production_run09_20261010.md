# H4 run09：承認済みhost PSI猶予で再実行

利用者が「猶予案を承認して再実行」と明示承認した。新runは `h4-newhost-signal-compile-20261010-run09`、branch `track-a-h4-production-run09-20261010`。
host PSI1%以上5%未満は、全visible nonroot PSI0・OOM基準一致・effective available152.25GiB以上・観測間隔/freshness各5秒以内の場合だけ、最大30秒連続の猶予を与える。
host<1%でwarning timerをreset。5%以上・30秒継続・nonroot PSI非zero・OOM増分・欠測/不正/scope/namespace変化・余裕不足は即fail-closed STOP。
猶予中も毎秒resource observationを続け、warning_active/elapsedをtraceへ保存する。CPU/owned identity/AS・RSS・5秒違反には猶予を適用しない。
起動前は従来どおりhost PSI<1%。親/observer共通PressureGuardはoriginal launch OOM/hierarchyをbaselineに保持し、古いbaselineをtimer tickにはしない。
5%と30秒の最適性、共有負荷の発生者、全map完走は未証明。承認された限定条件として実装する。

SOURCE `251993785ef1dab2a3891bdbb5d079f5d2184f4d`、closure72。旧69の変更3/不変66、新active grace module/native run08 proof/test3。
旧pressure v1 strict1% profileと旧inactive案は履歴として保持。新v2 profile・利用者authority・plan/auth/reviewをbyte SHAとdigestへ再結合する。
科学条件H4 linear/STO3G/DF rank12、6距離0.70/0.80/0.90/1.10/1.40/1.60、T0.8、8system+ancilla1、218templates/32paired trajectories・compiler options・canonical serialization・parallel orderingは不変。
newSOURCEはseed identityへ結合し、旧partial/random/scientific cache/checkpointを混合・再利用しない。入力再生成なし、凍結6NPZの科学arrayは認可された実mapの内側だけで読む。

限定16integration回帰PASS、fail/error/skip0、wall0.215548秒、peakRSS29,884,416B。1process/thread1、AS256MiB/RSS128MiB/wall30秒/output1MiB。
closed v1/v2 profile、1/5/30境界・nonzero<1reset・exact5秒間隔・古いbaselineとstartup・observer warning/frame/32GiB role/所有・auth/freshgate・最新run08 proofを検査した。
初回tests-01はexact5秒fixtureのfloat offsetがわずかに5秒を超えFAIL、fixtureだけを修正しtests-02 PASS。実装の5秒基準は緩めず、初回ログ保持。
旧14/53/11campaign・benchmark128・synthetic28/64・追加人工compileは再実行しない。実child/worker/observer/affinity/科学array/transpile/GPU0（準備段階）。
独立reviewは新6sourceと新profile/authority/binding/launcherの差分に限定し、既存environment/input/旧proofの全面reviewを繰り返さない。

worker4 CPU[2,4,5,6]、driver16、observer18、own-run限定affinity、thread1/Qiskit num_processes1/Python -P -B。
worker AS/RSS32GiB、driver8GiB、observer AS256MiB/RSS64MiB、headroom16GiB、admission152.25GiB、17GiB output/74805actual/72h不変。
per-attempt carry0。全74784logicalのactual worst74784、charge bound8,736,971,632B、保存容量5GiB/301000inodes（records/ledger/signals/workerlog/observer/journal/temp/blocks/directoriesを含む）。
run08のoriginal STOPはhost1.26/nonroot0/OOM増分なし、予約6/complete2/signal1/charge4,263,867,374B。全6identity×2ABSENT。
原status receipt12213B/SHA10cd…を保持し、17322B/SHAd478…のnative/原17file bytes SHA inventoryを新stop_receipt_v19へ結合。過去exit code/正確な終了時刻を補完しない。
旧費用と証拠を保持し、次attemptへ失敗分を加算しない利用者指示を維持する。

prepare readonly precheckはavailable466.598GiB、FS411.046GiB、220440017inodes、quotaKNOWN、hostPSI0、selectedCPU最大busy0でPASS。
固定artifact後にもSOURCE/profile/input/carry/inherited kernel hard>=32・fresh CPU/memory/pressure/OOM/FS/inode/quota・未使用run09 output/control/one-shotを確認する。
launcherのpressure安定sampleは1秒間隔、最後のsampleからCPU3秒passive監査を含め次tickまで約4秒として5秒上限を維持する。
起動外部consoleはprelaunch metadata64KiB以内を検査し、nativeFD1/FD2をexec直前DEVNULLへ切替。Python診断はSOURCE予算付きCappedDriverLog/observer/workerlogへ保存する。
既知のnative stderr/formatter二次MemoryError診断2gapsは未修正の限界として保持する。

独立差分reviewとimmutable artifact・直前fresh gates全合格後、追加承認待ちでSTOPせず一度mapを起動する。未使用run09をclaimし、旧run08認可/output/control/one-shotは使い回さない。
完了はMAP_COMPLETE_STOP、問題時はown-run限定fail-closed STOP。次stage未認可、同run resume/盲目的retryなし。必要調査修正とfresh再実行は継続認可の範囲で行う。
共有host/cgroup/sysctl/既存venv/install/他job/GPU変更0。work/log/tempはhomeだけ。開始確認後chat終了可、observer監視は継続。
[固定source・承認・plan/auth/review・起動argv](../../artifacts/resource_applicability/track_a_h4_production_run09/2026-10-10/README.md)。
NPZ/科学runtime/checkpoint/cache/credential/内部SSH/private utilityをcommitしない。

独立最終review：`PASS_READY_FOR_FRESH_ONE_SHOT_LAUNCH_DELTA_ONLY`、blocking実装指摘0。新6source・pressureprofile/authority・最新nativeSTOP/carry0/認可/launcher差分を別担当が照合。旧SOURCE/profile/inputのreviewは保持して引継ぎ、全面再審査はしない。
review JSON SHA256 `422055f2aa7c4c407774fa0d622f4d4af2ff56d5fd2b871812fc4f9feacd308b`。直前fresh gatesは別途必須。
