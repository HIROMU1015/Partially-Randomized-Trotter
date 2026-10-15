# H6監視stalenessの必要修正

利用者の失敗時修正指示に従い、run02消費を保持して独立観測を既存metadata launcherへ追加。5秒を延長せず、長いGIL/C処理中にも継続観測できるよう修正。

H6 pilot run02はnative正しさ検証中のcallback stalenessでSTOP、compile0。既存metadata launcherの独立資源観測へ修正し、SOURCEb73580dd…/closure13、限定84testsとGIL保持C約12秒の2process実動作検査PASS。追加本番role/process0、5秒・資源・科学/compiler条件維持、旧消費/unknown保持、subagent0。新run03へ結合して再実行。

[技術記録](../track_a_h6_external_monitor_fix_20261015.md)。
