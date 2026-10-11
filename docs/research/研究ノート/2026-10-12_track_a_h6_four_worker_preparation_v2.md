# 2026-10-12 H6：4 worker指定を準備へ反映

利用者は2 worker提案に対して「4ワーカーでよい」と指示。既存2 worker worktree/資料を保持し、別home worktreeへ変更した。
CPU20/21/24/25、coordinator22/23。Aggregate RSS40/RAM64GiBを増やさず、worker RSSを9GiB/AS32GiBとした。
前回peak7.929GiBは参考値で、残142最大RSS・実speedupは未測定。超過時はown-run fail-closed STOPを維持する。
SOURCE `df2976f05c97c8c777f9a6264e83a53b637fef65`。科学224/旧runtime233・compiler/profile/input/task/carryに変更なし。
38限定検査PASS、最大4 tiny process、実science/人工transpile/approved grant0。
Phase A完了と旧449を再利用する技術sealを新SOURCEへ再binding。142 group起動は別指示待ちで実行していない。
[資料入口](../../../artifacts/resource_applicability/track_a_h6_parallel_preparation_v2/2026-10-12/README.md)。User push・H4継続・共有環境不変。
