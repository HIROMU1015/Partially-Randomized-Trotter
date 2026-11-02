# 2026-10-12 N3 occupation redesign preparation

利用者の2026-10-12指示に基づく理論・保存証拠の検証と比較準備。新科学実行は未認可。
base 64cf5ed227b7e4a74b37453cbe733db5f5d51f56を専用worktreeへ取得し、rootと既存結果を保持。
レビュー由来の固有値和条件をCAR/Schurから独立検証、T3有理証拠・旧族separator・強化norm対照を再検査した。
結果は同情報で条件族の厳密差がある一方、強い外部下界でnorm-onlyも受理。
従って新条件だけの算法全体の優位性は未確定とし、取得下界×判定法の比較を設計した。
過去12入力はdevelopment、新8入力は係数保存だけ。U/c/κ/受理/truthの新評価なし。
N1現候補scan停止、N2保留、一般二体/多外部/実分子/総量子資源へ拡張しない。
[詳細](../n3_occupation_redesign_mathematical_audit_20261012.md)、[契約案](../n3_occupation_transfer_design_v1_20261012.md)。
新9 synthetic testsはすべてPASS。付録JSONにはclosing fence用改行が追加されていたため、
抽出時に記載byte長とSHAを使ってその1改行だけを除いた。元付録コードは実行していない。
pyenvのpython aliasは未設定、stdlib checkerは明示/usr/bin/python3を使用。
GPU driver接続不能、CPU32/affinity32、memoryを確認し、単一保存証拠は1 process。
独立証拠auditの観測時間/RSSは一次JSONへ保存。倍率比較なし。
global manifestのrelevant_commits[7].description既存欠落は本作業では修復しない。

保全監査の初回はgit ls-treeの日本語pathがquoted表示されたためpath lookupで停止した。
NUL区切りのtreeを直接解析し、blob identityとSHA-256を照合する方式へ修正した。
scientific retryではなく保存検査の修正であり、通過済みsynthetic logは再利用した。
基点の保護対象1811 files、rootの1342 files/HEAD/dirty status、旧manifest90 entriesを保持。
