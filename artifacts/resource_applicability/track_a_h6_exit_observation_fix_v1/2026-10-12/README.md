# H6 PROCESS_MEMORY_MISSING監視修正

Branch `track-a-h6-exit-observation-fix-20261012`。
SOURCE `00b8d7e3f061682eb7fc5503bd379215a0cea942`。旧SOURCE0365f0a0・STOP・完了7 group/14 wrapperを旧worktreeに保持。
[原STOPと保存結果の受理](saved_stop_receipt_v1.json)、[source/profile/input binding](source_binding_audit_v1.json)、
[58限定tests](limited_tests_v1.json)、[検査予算](limited_test_plan_v1.json)、[軽量差分review](limited_review_v1.json)、
[次回135件の再利用案](reuse_proposal_v1.json)。新H6科学runner/approved grant/再起動0。

## 原STOPの確認範囲

H6 roundoff runは480.15151466988027秒後にPROCESS_MEMORY_MISSINGで停止。
監視側completedは450〜455の6件。cleanup中に正常終了した457をmetadata admissionで別途受理し、完成は計7件。
受理indices450/451/452/453/454/455/457、14 wrapper、原142の残り135。旧449＋Phase A447＋今回7 =457/592。
Native worker/coordinator全reap/残存0、cleanup errorなし。合計RSS peak5.434GiB。
456/458は未完了のまま保存する。partialを成功へ昇格せず、実call総数はnull、不明を0や返却/reset扱いしない。
既存compile snapshotと完成groupのknown callsは原bytes/hashで保存し、次回のcarry bindingに使う。

今回の原ログには、欠測した正確なworker PID・その時のkernel exit flagがない。
正常終了457と周辺の時刻は、kernel mm teardownとwaitable exitの間の競合に整合するが、原faultを完全特定したとはしない。
過去の精密exit時刻・flag・追加counterを後の観測から補完しない。

## 監視修正

旧20msのblocking waitを撤去。kernel PF_EXITINGとPID/start_ticks/group/sessionの一致を確認したown workerだけ、最大1秒のnonblocking exit-pendingとする。
Hostと他workerのresource observationをその間も継続する。WEXITED|WNOWAITを得た後だけowned group cleanup/wait/reap/受理を行い、slotを解放する。
欠測RSSは直前の観測値を保持し、初回未観測なら既存slot RSS/AS予算を保守的に予約する。ゼロで補わず、aggregate guardへ含める。
生存中の欠測、所有不一致、許可error、別のValueError、退出待ち期限超過はSTOP。
同じ連続欠測で期限を延長しない。5秒観測、PSI/OOM、AS/RSS/CPU/output/log/科学条件・compiler・丸めbridgeは不変。
将来の観測errorは最初のSTOP理由・対象PID/roleと退出遷移を小さなfailure receiptへ記録する。

58 local限定回帰PASS。Kernel flag/identity fixtures、250ms delayed-exit fixture、他worker継続観測、live欠測・所有不一致・期限・error STOP、全wait/reapと4 tiny children。
PF_EXITING値はhostのlinux/sched.h1617で照合。Fixtureのkernel stateを本番観測と混同しない。
旧科学224/基盤233/profile/input/旧raw hashは読取照合、新runtime差分は監視module1件だけ。
Immutable CI/独立agent review/修正後の実H6完走は未実施。

## 現在と次回

SOURCE/testsはcommit固定、軽量資料を別REVIEW commitへ固定。agent pushは行わない。
既存の一回grantは消費済み。新SOURCEに対する135件のruntime binding・fresh output/control・grantはまだ作成していない。
再利用案は実行許可ではない。approved=false/runtime_authorization=false/allowed_cpus=[]/absolute_launch_command=null。
保存済み7件を再計算せず残135を接続する際、過去不明attemptと今回partial消費を保全してcarryを再結合する。
現在H6は停止。H4はread-only確認でRUNNING/806 signals、変更なし。共有環境・venv・他job・GPU・科学input生成変更0。

利用者push用:
```bash
git -C /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h6-exit-observation-fix-20261012 \
  -c maintenance.auto=false -c gc.auto=0 \
  push origin HEAD:refs/heads/track-a-h6-exit-observation-fix-20261012
```
