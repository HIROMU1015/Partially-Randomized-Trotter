# 2026-11-13：PF有効Hamiltonian評価の技術基盤

ユーザー指示により、Claude統合レビュー§4.2/§7.3 T2'/付録A5を参照し、決定論stepからH_effを構成する技術moduleと人工testを追加した。手法は正式未採択のまま。branchの数値再構成、物理branch/ground対応、state overlap、全RPE bias/RMSEへの接続を分離する。

既存direct expm step・S2/Y4/M5 weights・q power・人工PR exact-tail macroを再利用し、PF/Hamiltonian/回路を再実装しない。旧target中心Schur helperはreferenceによる中心化・重み再正規化・QPE解釈を持つため、その全体を今回のRPE保証へ流用しない。reference energyからbranchを自動選ばず、spectral minimum shiftをrpe_biasへ昇格しない。

[REPORT](../pf_effective_hamiltonian_20261113/REPORT.md)／[HANDOFF](../pf_effective_hamiltonian_20261113/HANDOFF.md)／[有限案](../pf_effective_hamiltonian_20261113/SCIENCE_PLAN.md)。新15＋旧8＝23人工回帰PASS。実H4/H6固有値計算・保存signal再解析・PF幅探索・N/RMSE/C/Gは0。

PR exact-tailの共通stepはmetadataだけ確認し、finite-K corrected/rawは非対象とした。early actualΔが変わるscheduleやstate多成分・alias/縮退・数値未解決を単一energy biasで置換しない。次はユーザーが一候補/一幅・branch/referenceと有限pilotを選択/承認した場合だけA/Cが物理証拠を取得し、Bがriskへ接続する。今回公開・remote照合後mandatory STOP。
