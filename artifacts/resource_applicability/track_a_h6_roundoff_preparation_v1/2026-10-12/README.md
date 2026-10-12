# H6 丸め差許容・4 worker一回探索

SOURCE `0365f0a0c8e03031902837dbc2503bc1eed3a17b`、branch `track-a-h6-roundoff-execution-20261012`。
ユーザーが「丸め程度の細かい不一致なら構わないので計算して」と明示承認し、続きを指示した。
[丸め診断](roundoff_diagnostic_receipt_v1.json)、[67限定tests](limited_tests_v1.json)、[軽量差分review](limited_review_v1.json)、
[技術seal](sealed_v1/manifest_candidate_v1.json)、[利用者承認](launch_authorization_v1/user_authorization_record_v1.json)、
[起動直前gate](launch_authorization_v1/prelaunch_gate_v1.json)、[一回command](launch_authorization_v1/launch_request_v1.json)。

元450 seed/eventsの選択・順序・符号が一致し、数値metadata30件の最大差2.7755575615628914e-17を確認。
保存済みtail basis13件へnative固有値を対応付け、追加演算差の保守的重み付き評価合計4.865372669541938e-12 <1e-10。
既存nativeと入力の差は3.098852311480774e-10 <既存1e-9 gate。絶対1e-12/相対1e-10でfloatだけを比較し、seed・ID・sign・order・supportは厳密一致。
Gate名/位置/角度は元保存定義を維持。local matrix hash差だけは同じgate生成に結び付ける。変更hashの対応を保存する。
旧449と新取得のcompiler出力・gate countのbit同一性を確認済みとは主張せず、native環境来歴と小さい数値差を明示して再利用する。
量子資源の改善と古典実行時間短縮を混同しない。

独立worktreeで科学224/server基盤233不変。cost portの差分は旧partial event byte digest gateに丸め比較receiptを加える部分だけ。
Hamiltonian/state/DF/seed/92候補/4精度/normalization/shot/full-wrapper/compiler/RZ/CX/depthの定義・capsを保つ。
Deterministic preparation4種類は現在のnative hashと完全一致し、完了447を再計算せず、原142だけを実行する。
採用しなかった厳密old-export待ち案は元worktree/branchに保持した。旧環境fileの受領待ちは今回不要。

CPU20/21/24/25のworker4、coordinator22/23、内部thread1、own affinity/nice10。
Worker AS32GiB/RSS9GiB sampled STOP、coordinator soft2/hard32GiB、aggregate RSS40GiB、job RAM64GiB。
元5秒監視・host/cgroup PSI/OOM・容量/inode/quota・output/log/instruction guardを維持。wall cap=null。
H4は停止・変更せず、prelaunch時RUNNING/804 signals/46007 records。新science grantだけが今回の142一回を認可する。

Known carryはcompile904/control292/trajectory436/occurrence1040/primitive1470/reference400。
今回のbridge診断は失敗した1trajectoryと成功した1trajectory（occurrence計16）を確定消費へ加えた。
前回4worker STOPの実総数はnullのまま、下限trajectory4/occurrence32を独立した保守的予約として追加する。
予約は確定総数ではなく、古い不明attemptも保持する。返却/resetなし。
この残142を含むknown+reservation admissionはcompile1188/trajectory582/occurrence2208/control360、元cap1712/832/6656/1024以内。
これを全歴史のexact totalとは呼ばない。過去停止のreceipt bytes/hashと不明会計をbindingする。

Fresh prelaunchはCPU・profile/source/input/carry・メモリ/PSI/OOM・disk/inode/quota・未使用output/control PASS。
新output `/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h6-roundoff-execution-20261012/artifacts/resource_applicability/track_a_h6_roundoff_exploration_v1/2026-10-12/launch_parallel_v1`。
Run ID `h6-roundoff-phase-b-142-launch_parallel_v1`。
ログ/終了証拠 `/home/AbeHiromu/projects/h6-handoff-evidence/20261012/roundoff-execution-v1/`。
現在の資料commit時点はREADY_NOT_STARTED。実際の起動・PIDは同場所coordinator.pidとnative outputを参照。
計画flags approved=false/runtime_authorization=false/allowed_cpus=[]を保ち、新exact grantのみapproved true。
完了/STOP後終了、自動retry・selection・独立32確認・H8へ進まない。

前修正branchのremote SHA1293109e2c278d6c36c6337f0a9d885ea004b2e9を確認。新SOURCEはlocal固定で、今回も利用者push方針に従いagent push0。
最新利用者の明示再実行指示に基づき、local source/profile/inputの実SHA確認後に起動する。
公開コマンド:
```bash
git -C /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h6-roundoff-execution-20261012 \
  -c maintenance.auto=false -c gc.auto=0 \
  push origin HEAD:refs/heads/track-a-h6-roundoff-execution-20261012
```
