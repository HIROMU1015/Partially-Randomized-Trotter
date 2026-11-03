# 2026-10-09：Track A AX-2B H4 pilotの一回実行と停止

利用者の「次の作業に進んで」を、直前に具体化したH4限定pilotへの継続指示として受け取った。
logical CPU 3、science worker 1/thread 1、RLIMIT_AS 8 GiB、phase 900秒/total 2700秒、
exclusive `launch_v1/run_v1`を固定し、v3 sourceを変更せず一回だけ実行した。
旧未認可draftは履歴として保存し、新authorizationへ指示とscopeを記録した。

linear H4、1.00 Å、STO-3G、legacy DF rank12、generation-prefix、T=0.8、
q1/q4（δ0.8/0.2）、保存state/36-dimensional spin sectorが対象。
登録8 cellの一致検査は通過し、random 4 trajectory/16 occurrenceと24/28測定wrapperを保存した。
global四次B1・prefix12・q4の回路準備中、control probe/compile前にMemoryErrorで停止した。
parent wall591.867752秒、worker exit1。再実行・上限変更・欠落の補完をしない。

保存JSON/hashの監査をscienceとは別のstdlib scriptに追加した。
prelaunch174 testsとsaved-audit15 testsは全てpass、fail/skip0。
8 cellと24 wrapperのpartial evidenceが整合することを確認したが、pilot完了とは扱わない。
総数値allowanceは未認定、accuracy適格性UNDETERMINED、shots/total cost/winnerの結論はない。

exception tracebackがないため正確なallocation原因は不明。
完了回路を保持するcacheと再帰的fingerprint payloadが静的なmemory圧力候補である。
次は回路の解放、phase/memory/traceback記録、互換stream fingerprintをv4準備として検討する。
S4 q4を外すとstrong baseline比較が失われるため、科学grid変更より実装修正を先に置く。
routine実装修正の形式的GPT reviewは不要だが、H6移行と数値allowanceの科学判断は別段階とする。

[結果・停止理由・次の修正案](../track_a_ax2b_h4_pilot_result_v1.md)と
[新inventory](../../../artifacts/resource_applicability/track_a_ax2b_h4_pilot/2026-10-09/launch_v1/pilot_evidence_inventory_v1.json)を入口にする。
旧v1/v2/v3準備、元dirty worktree、M1〜PM-2/AX-1b、原稿、Track Bは保持する。
現在 `H4_CORRECTNESS_PASS_COST_PARTIAL_MEMORY_STOP`、mandatory STOP、research decision=null。
H6/H8/main campaign、retry/resumeへ自動的に進めない。今回の結果は未commitのlocal証拠である。
