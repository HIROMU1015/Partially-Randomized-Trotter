# H4 run10：承認済み構造化Gaussian・新cost系列で再実行

利用者は「新方式へ切替・再実行」と明示承認した。run09のown driverだけをUID/PID/start/parent+pidfdで照合してSIGINT、既存driver/observercleanupで全6native identityを2秒離した2回ABSENT確認した。
旧8予約/4compile完了/2signal/charge4,263,948,142B/13ledgersと全26原file bytes/SHAを保存。元driver exit code/正確な終了時刻は未記録で補完しない。
manualintentはUSER_APPROVED_STRUCTURE_SWITCH。observerがcleanup中に記録したfirstcauseはKeyError VmRSSであり、intentと混同してresource faultと推定しない。旧STOP原値を保持する。
新run `h4-newhost-signal-compile-20261010-run10`、branch `track-a-h4-production-run10-20261010`。旧partial/random/checkpoint/scientificcacheを新系列に再利用しない。

SOURCE `5aa6c3685afb9c3afcfbdc36f80244da0a7fe525`、closure77。旧74の変更4/不変70、新active Gaussian/nativeproof/integrationtest3。
circuits.gaussian_basisをactive Givens builderへ接続し、number PhaseGateと隣接4x4 two-qubit UnitaryGateでfull Γ(U)を構成する。generic256x256/8q QSD fallbackへ戻らない。
1つの8mode Gaussianは最大28two-mode rotations+8numberphases、vacuum=1・fulloccupancy determinant/JW signs/逆積順序を保持し、exactzeroだけskipする。
数値atol1e-12のunitarity/QR residual/reconstruction gateは閉じたgaussian_synthesis_profile_v2へ固定、利用者authorityとplan/auth/reviewへhash結合する。
wrapper_semanticsは `h4-full-gaussian-givens-paired-wrapper-v2`。科学unitary意図は保持するがcompiled RZ/CX/depthは旧denseの別cost系列になり、旧値と混ぜない。SOURCE+semanticsでtrajectory seed/keyも再結合する。

H4 linear/STO3G/DF rank12/6距離0.70/0.80/0.90/1.10/1.40/1.60、T0.8、8system+ancilla1、218templates/32paired trajectories、PF/RTE分布/shot/grid条件、compiler explicit optionsは維持する。
既存private venv・compiler1.3/options/profile・inputfreezeを変更しない。数学的同値性の証拠は[12人工行列tests・限定数学review](track_a_h4_gaussian_structure_proposal_20261010.md)を引継ぐ。
人工1/2/3/4/5/8modeのindependent minorsと旧式JW log/expm再実装でfullFock256まで1e-12一致、controlled cosine/sine相対位相は3mode fixtureを確認済み。
旧数学campaignは再実行せず、新active数学4関数のAST一致を確認した。これは実分子/Qiskit compiler出力の完全同一性や72h完走保証ではない。

新12integration回帰PASS（fail/error/skip0）、wall0.260886秒、peakRSS30,932,992B。1process/thread1、AS256MiB/RSS128MiB/wall30秒/output1MiB。
closedprofile、verified mathAST、basis dispatch/defaultbinding、newsemanticsのseed/key分離、false/oldcost/adoption missingの拒否、schema、最新manualstop/native26fileproofを検査。
Qiskit import/build/transpile・実NPZ/科学array・実child/worker/observer/affinity/GPU0（準備段階）。追加人工compileは利用者指示で省略し、Qiskit builder・実速度は認可された本計算の初期実行で確認する。
独立reviewは新7sourceとnewprofile/authority/binding/latestSTOP/launcher差分だけに限定し、旧environment/input/proofや回帰campaignの全面反復を行わない。

worker4 CPU[2,4,5,6]、driver16、observer18、own-run限定affinity、thread1/Qiskit num_processes1/Python -P -B。
worker AS/RSS32GiB・driver8GiB・observer AS256MiB/RSS64MiB・headroom16GiB・admission152.25GiB・5秒観測を維持。
host PSI1〜5%は全nonroot0/OOM同一/effective152.25GiB/fresh5秒の下で最大30秒猶予、5%以上/30秒継続/nonroot/OOM/欠測等は即STOP、毎秒監視継続、起動前host<1%。
17GiB output/74805actual/72h・per-attempt carry0、全74784wrapper worst74784/charge bound8,736,971,632B、保存容量5GiB/301000inodes（全records/ledger/signals/log/trace/journal/temp/blocks/directories）不変。
新GaussianprofileをSOURCE/environment/compiler/input/stop/carry/plan/auth/reviewへ再結合し、旧run09認可/output/control/one-shotを新run10へ使い回さない。

prepare readonly precheckはavailable466.684GiB、FS410.815GiB、220434306inodes、quotaKNOWN、hostPSI0、selected-core maxbusy約0.339%でPASS。
immutable artifact後もSOURCE/profile/input/carry/currentkernel hard>=32・fresh CPU/memory/pressure/OOM/FS/inode/quota・未使用output/control/one-shotを直前に確認する。
launcher pressure sample1秒+CPU passive3秒、console64KiB/exec前nativeFD1/FD2 DEVNULL境界と、source CappedDriverLog/observer/workerlogの予算を保持。
既知native stderr/formatter二次MemoryErrorの診断2gapsは未修正。全mapの実性能・完走は未測定で、初期actual compile gate数とelapsedを旧dense参照から区別して記録する。

独立限定review・固定artifact・直前fresh全合格後、追加確認を待たず一度mapを起動。開始後chat終了可、既定observer監視継続、MAP_COMPLETE_STOP/own-run fail-closed STOPで終了。
次stage未認可。通常の問題調査/修正/fresh再実行は既存の継続認可に従い、同run resume/盲目的retry/科学入力再生成なし。
共有host/cgroup/sysctl/既存venv/install/他job/GPU変更0、work/log/tempはhomeだけ。NPZ/科学runtime/checkpoint/cache/credential/内部SSH/private utilitiesはcommitしない。
[固定SOURCE・新Gaussiancost認可・plan/auth/review・argv入口](../../artifacts/resource_applicability/track_a_h4_production_run10/2026-10-10/README.md)。

独立最終review：`PASS_READY_FOR_FRESH_ONE_SHOT_LAUNCH_DELTA_ONLY`、blocking実装指摘0。新7source・Gaussian profile/authority・最新nativeSTOP/carry0/認可/launcher差分を別担当が照合。旧SOURCE/profile/inputのreviewは保持して引継ぎ、全面再審査はしない。
review JSON SHA256 `d72c1976e6a82c29141d31271a6552dab3301d3a7bc787c082c4563bf5a683e1`。直前fresh gatesは別途必須。
