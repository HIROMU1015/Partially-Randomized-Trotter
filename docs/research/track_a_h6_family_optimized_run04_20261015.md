# H6 full-family optimized run04：明示再実行

利用者の「本計算をもう一度行って」を根拠に、修正済みSOURCE
`ac6d28e35cf01398f169ce59577182a38da9b959` で新one-shotを認可する。
[固定binding](../../artifacts/resource_applicability/track_a_h6_family_optimized_run04_v1/2026-10-15/README.md)。
[修正・61人工検査](track_a_h6_family_fingerprint_parallel_fix_20261015.md)を反復せず、主担当が差分と今回のbindingを確認した。subagents0、別担当の独立reviewではない。

旧run03はhost full PSI1.44%で停止し、compile予約6/受理費用0・native wait137を保持する。実transpile到達/完了数は不明のままで返却しない。
過去3workerの未記録exitを補完せず、後続の元6worker残存0と今回の起動前absenceを分けて記録した。旧output/grant/証拠は変更しない。

H6 linear1Å/STO-3G/保存DF19・T=.8/S2/symmetric_directional、既存構造化Gaussian、compiler rz/sx/x/cx・optimization1・seed17を維持する。
旧pilot3構成6軸は元SOURCEの実測としてREUSE_WITH_DECLARED_LIMITS。新9構成をscreenし、適格な構成だけbuild/全native correctness後にcompileする。
前回と同じscreenなら新6構成12軸、上限は既存18軸のまま。run03の回路checkpointはなく、build/native検査から再実行する。

今回のdriverはCPU63、workerはCPU62/61の2process、metadata observerは既存launcherがdriver coreで観測する。
thread1、各数値role AS32GiB/RSS9GiB、metadata AS512MiB/RSS256MiB、当stage合計CPU86400秒・metadata CPU1800秒、phase/total wallnull。
admission240.5GiBを据え置き、host full PSI>=1%・cgroup pressure非zero・OOM増分・5秒欠測等ではfail-closed STOPする。
CPU/環境/他jobの共有設定は変更せず、affinityは新runの自processと子にだけ適用する。

SOURCE14/provider230/profile34273/input18のbyte binding PASS、新grant/output/controlとcarryを結合する。
本資料固定後の直前fresh CPU/physical sibling・memory/pressure/OOM・容量/inode/quota・source/input/profile・未使用output/control/one-shot gateに全合格した場合だけ起動する。
fail-closed時は旧記録を保持し、盲目的retryや次stageへの進行は行わない。

private監視入口：
`/home/AbeHiromu/projects/h6-handoff-evidence/20261015/h6-native-b0i-full-family-run04/`
の `run01/`、`control01/launch_stdout.log`、`control01/external_resource_observation_v1.json`。
開始状態は `running_confirmation_v1.json`、終了時は `terminal_v1.json`/native wait/cleanupで確認する。
この文書は開始確認・完了結果そのものではない。既存venv/共有環境/H8・他job/GPU変更0。
