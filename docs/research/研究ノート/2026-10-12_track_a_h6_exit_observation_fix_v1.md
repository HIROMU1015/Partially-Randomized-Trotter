# H6 退出時のRSS欠測監視修正

[一つの資料入口](../../artifacts/resource_applicability/track_a_h6_exit_observation_fix_v1/2026-10-12/README.md)。
SOURCE00b8d7e3f061682eb7fc5503bd379215a0cea942。科学224/基盤233/環境profile/input/compiler/丸めbridge不変。
原roundoff runは8分後PROCESS_MEMORY_MISSINGでSTOP。完成7group/14wrapperをmetadata受理、原142の残135。
退出の20ms graceではmm teardown後の待機を誤判定し得る。Own identityとkernel PF_EXITINGを確認した場合だけ、最大1秒のnonblocking exit-pendingへ修正。
待機中も他worker/hostを観測し、RSSは直前値または既存slot予算を保持、aggregate capへ含める。
Live欠測・所有不一致・許可error・別ValueError・期限超過はfail-closed。WEXITED|WNOWAIT後にだけcleanup/wait/reap/slot解放。
5秒観測・PSI/OOM・CPU/AS/RSS/output/carry契約は変更しない。次のerrorで対象PIDと退出遷移を記録する。
58 local限定検査PASS、kernel退出flag fixtureとreal tiny-process wait/reapを区別して記録。修正後の実H6/独立agent review/CIは未実施。
原faultのPID/flagは未記録であり、完全特定したとは主張しない。原exit時刻や不明call総数を補完しない。
旧完了・partial・STOP・消費済みgrantを保持。新科学runner/grant/launch0、新planは未seal。
次回7group再利用/残135案を保存。科学条件/seed/quantum costを変えず、partial消費と不明過去分を返却/resetしない。
H4/venv/共有環境/他job/GPUを変更せず、利用者push方針を維持した。

原roundoff起動REVIEWc613a617・起動receipt19da6a6dを保持した。
