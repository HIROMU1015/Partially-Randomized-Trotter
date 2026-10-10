# H4 run08：worker32GiBで本計算を再実行

利用者は「本計算をもう一度実行して。　処理が開始したらチャットはいったん終了してよい」と明示指示した。
worker32GiBの承認を保持し、独立reviewとimmutable artifact・直前fresh gates合格後、そのまま一度H4 signal/compile mapを起動する。
新runは `h4-newhost-signal-compile-20261010-run08`、branch `track-a-h4-production-run08-20261010`。
科学条件はH4 linear/STO-3G/DF rank12、距離0.70/0.80/0.90/1.10/1.40/1.60、T0.8、8system+ancilla1、218templates/32paired trajectoriesの既存契約。
compiler options・科学式・canonical serialization・parallel ordering・凍結input identityは不変。

SOURCE `8c917a5af7943969d4abede8b0bb012880efeee0`、closure67。旧memory32 SOURCE65からruntimeのRUN_IDと最新停止proof検査だけを追加した。
旧65の変更1/不変64、新run07 proof module/test2。future random seedを新SOURCEへ結合し、旧partial/random/checkpoint/科学cacheを混合しない。
[旧32GiB実装・53回帰と独立review](track_a_h4_worker_memory32_fix_20261010.md)を保持し、同campaignを再実行しない。
今回限定11metadata/byte回帰PASS、fail/error/skip0、wall0.081741秒、peakRSS29,360,128B。
単一test process/thread1、AS256MiB/RSS128MiB/wall45秒/output4MiB、実child/worker/observer/affinity・科学array・回路build/transpile0。
人工compileは利用者指示で省略。全map完走・旧compiler output完全同一性・元native例外/signalの確定は未検証。
既知のnative stderr破棄とformatter二次MemoryErrorの診断2経路は記録に保持し、本turnでは追加開発しない。

worker4はCPU[2,4,5,6]、driver16、observer18。own-run限定affinity/thread1/Qiskit num_processes1/Python -P -B。
worker AS/RSS各32GiB、driver AS/RSS8GiB、observer AS256MiB/RSS64MiB、headroom16GiB、5秒監視を維持。
fresh/OwnedRun二次gateの必要availableは152.25GiB。親soft8を保ってhard ceiling>=32をchildへ継承し、認可/SOURCE照合後worker32、全ready後driverhard8。
observerはtrusted driver PID8と登録済worker PID32を区別。所有不一致/欠測/退出/5秒違反/nonESRCH cleanup errorはfail-closed。
pressureは承認済host-only PSI<1%、全visible nonroot PSI0、OOM基準一致、fresh<=5秒。例外時available120.25GiB、通常headroom16GiBの既存条件を維持。
17GiB output/74805 actual/72h、per-attempt carry0を保持。全74784 logicalのstatic actual worst74784、charge bound8,736,971,632B。
保存容量は全records/ledger deltas/signals/worker logs/observer trace/journal/temp/blocks/directories/inodesを含み、5GiB/301000 inodesを必要とする。

run07は予約6/complete2/signal1/charge4,263,870,068B、6identity×2 native ABSENT。最新receipt13094B/SHA3156…と元18filesをbyte-only照合した。
worker native exit codeと正確な終了時刻はunknownのまま。旧費用/証拠は履歴として保持し、次runへ加算しない利用者指示によるcarry0。
新stop_receipt_v18は旧host/run01〜06と最新run07の固定証拠を結合し、runtime verifierも確認する。
凍結6NPZ/generation-freezeはstreaming bytes/SHAだけを照合、科学arrayは認可された実本計算内だけで読む。入力再生成なし。
準備時precheckはavailable466.9275GiB、FS411.2756GiB、220445710 inodes、quotaKNOWN、選定CPU最大busy0でPASS。
これは直前launch合格の代用ではない。固定artifact後にSOURCE/profile/input/carry/kernel ceiling・pressure/OOM・CPU/memory/FS/inode/quota・未使用output/control/one-shotを再確認する。
旧run07 output/control/one-shot/launch認可を使い回さない。新run08 approved/runtime/sealedは今回の明示launch指示と独立reviewの一致により固定する。

既存venv/install/共有設定/cgroup/sysctl/他job/GPUを変更しない。新work/log/tempはhome内だけ。
runは既定observerの監視下で継続し、MAP_COMPLETE_STOPまたはfail-closed STOPで終了する。次stageは未認可。
STOP後の調査修正とfresh再実行は利用者の継続指示に従い、同run resume・盲目的retry・旧partial/cache混合はしない。
利用者は開始確認後のチャット終了を許可している。起動/PID/結果は外部runtime receiptへ記録し、固定認可artifactと区別する。
[資料入口・plan/auth/review/source/固定argv](../../artifacts/resource_applicability/track_a_h4_production_run08/2026-10-10/README.md)。
NPZ/科学runtime/checkpoint/cache/credential/内部SSH/private utilitiesをcommitしない。

起動用外部consoleはexec前の固定metadataだけに限定し、64KiB以内を検査する。実runner exec直前にnative FD1/FD2をDEVNULLへ切替える。
本体のPython診断はSOURCEの予算付きCappedDriverLog（8MiB）・observer/workerログに保存し、scientific native出力を外部consoleへ流さない。
これはdriverのkernel AS/RSS・科学条件・SOURCEの変更ではない。既知のnative診断2gapsは引き続き保持する。

独立最終review：`PASS_READY_FOR_FRESH_ONE_SHOT_LAUNCH`、blocking実装指摘0。SOURCE67/profile/input/最新nativeSTOP/carry0/cap32/認可/launcherを別担当が照合。
review JSON SHA256 `bf8bac696fc3275120837cfe2af2c5687692903f33807c31244f9dceb14214bf`。直前fresh gatesは別途必須。
