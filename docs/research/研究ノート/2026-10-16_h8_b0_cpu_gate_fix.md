# 2026-10-16 H8 B0 CPU gateの修正

利用者の「修正できるか」を受け、run01の再起動をせず、技術的なCPU選択・診断・停止記録を修正した。現在の読取専用1秒観測では旧CPU0が17/99 busy ticks、診断自身はCPU43・差0 ticks。過去run01も同じ原因だったとは断定しない。

grant前の3固定window候補probe、全選択coreのcounter/負荷率/自身のCPU使用を保存するreceipt、初回/refreshの1秒raw5% gate、science lock前の一回限りadmission記録を追加した。5%閾値・SMT観測・6物理core/176.25 GiB条件、固定4構成8軸/4 worker/2 waveは維持した。STOP後の自動再選択やretryなし。

新11＋既存27 methods PASS。SOURCE `db559a518214fd27fa96e2f1451911df05c4f4a8`、source15/provider265/runtime34,273/input8 byte照合PASS。新configと無効grantを作ったが、CPU[]/未承認で起動できない。新H8計算/compile/fit/有効grant0。旧677保護/116worktreeとrun01原証拠を保持し、公開後mandatory STOP。

[現在の修正と制約](../track_a_h8_b0_cpu_gate_fix_20261016.md)。
