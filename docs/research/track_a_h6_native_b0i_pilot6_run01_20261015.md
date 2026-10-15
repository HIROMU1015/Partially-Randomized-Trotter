# H6 native B0+I pilot6 run01：Phase Aでfail-closed STOP

2026-10-15。利用者の「計算を実行して」と中断後の「続きを行って」に基づき、承認済み最大6並列pilotを**一度起動**した。約8.60秒でPhase Aのprepared B0 blueprint照合に失敗し、native build/control/compile前に終了した。**完了結果・費用は得られていない。自動retryなし、mandatory STOP。**

- Run ID: `h6-native-b0i-pilot6-20261015-run01`
- Branch: `track-a-h6-native-b0i-pilot6-launch-20261015`
- SOURCE: `7300b39df4c53d854d5bc8934430af19fbdb0085`（起動中・終了後もruntime10ファイル一致）
- 起点REVIEW: `6baf246cb46cf69536db6c16872286e5ed12e85a`
- 独立最終binding commit: `d9ef317e026160558a24b7accca5844f5ea11e54`
- 認可・seal commit: `8aaf4c2c23788180426c4ead9963f7a4fcae1546`
- [実行契約・承認・STOP監査入口](../../artifacts/resource_applicability/track_a_h6_native_b0i_pilot6_launch_v1/2026-10-15/README.md)
- Private output: `/home/AbeHiromu/projects/h6-handoff-evidence/20261015/h6-native-b0i-pilot6-one-shot-v1/run01/`
- Private log: 上記親directoryの `control01/launch_stdout.log`、監視/first STOPはoutput内 `terminal_v1.json`。

## 一度の起動と実績

Linear H6 1Å/STO-3G、保存tol-only DF19/H/state、T=.8/S2/symmetric_directional、p5q1・p10q2・p15q8のcosine/sine合計6compile契約。科学provider87b6b9f…、compiler hash1239304c…・basis rz/sx/x/cx/opt1/seed17、旧数学/u政策を維持した。

直前CPU/memory/host・全祖先cgroup PSI/OOM・filesystem/block/inode/quotaとSOURCE/input/profile/未使用namespace検査をPASSした。driver CPU2、compile worker予定CPU4/5/8/9/10/11、thread1、per-role AS32GiB/RSS9GiB、stage wall/総CPU4h、admission240.5GiB。実行前effective空きmemory約468GiB、filesystem約308GiB、PSI0/quotaoff。共有環境・venv・他jobの変更なし。人工70tests・科学reviewは再実施しなかった。

Launcher PID1437070、driver PID1437088。起動one-shot markerを一度消費した後、以下でSTOPした。

```text
phase: A_INPUT_SCALAR_SIGNAL
reason: Stop:B0_BLUEPRINT_FINGERPRINT_MISMATCH
native exit: 2
```

| 操作/資源 | 実績 |
|---|---:|
| snapshot load予約 | 1 |
| first analytic overlay予約 | 1（成功receiptなし） |
| native evolution build / measured wrappers | 0 / 0 |
| unmeasured copies / control probes | 0 / 0 |
| compile / compile workers起動 | 0 / 0 |
| launcher記録wall | 8.597542秒 |
| 観測peak RSS | 538.914 MiB |
| 観測CPU | 約7.307秒 |
| monitor最大start間隔 | 約0.514秒 |
| outputファイルlogical容量 | 40,045 bytes（cache等込み、監査時点） |

snapshotは認可実行内で数値loadした。最初のcaseのprepared B0を構成して照合したが、その後のtail scalar/overlay receipt・headroom・native buildへ進んでいない。overlay予約を成功件数と扱わない。新SCF/DF/state/trajectory/指数作用/hardware shotsは0。

## 確認できた不一致と限界

固定B0 blueprintはprepared hash、constant、extracted tail、fragment order、block order、mode、prefix/q/T/formula、H/DF/state/input/compiler bindingを含む。p5q1の期待digestは `e955c9c6e25da2b73888727860c8e16ce935cc2f4189b0b679e513f4f95a8fcc`。

実行時に作ったprepared blueprintのdigestがこの期待値と一致しなかった。input5件のbytes/SHA、runtime10 sourceと科学provider230 source、profile34,273 filesは起動前一致、input/runtime sourceは終了後も不変である。byte gateが通ったことはprepared/compiler出力同一性を証明しない。旧metadataの[期待blueprint](../../artifacts/resource_applicability/track_a_final_pre_review_implementation_v1/2026-10-15/h6_prepared_metadata_receipt_v1.json)は固定のまま保持した。

旧MatchedPortと新adapterは同じ固定 `_prepare_discard` APIを使うことをsource静的読取で確認した。ただし現STOPメッセージは実際のblueprint/hash/field diffを保存していないため、**どのfieldが違ったかは未特定**。input改変、科学条件差、固有値分解/数値環境の差、事前metadataの生成差などの特定原因へ断定しない。期待hashを実値へ置換してguardを迂回したり、追加の実H6再構成で埋めたりしていない。

## native終了・回収・carry

親launcherのnative wait2、recorder exit0、driver identity absentを確認。launcher自身も退出。compile phaseへ達していないためcompile子は生成0であり、parallel cleanup receipt欠測はこの場合に子回収失敗を意味しない。exact exit timestampは未記録/nullのまま。first STOP/terminal/native exit/wait/予約journalを保持した。

旧known/held/failed/unknown totalsはnull/UNKNOWNを維持し、8件の過去native終了metadata SHAへbindingした。これを完全compile ledgerや0件と扱わない。今回のsnapshot1/overlay1予約、wall/CPU/出力は失敗分として保持し、refund/resetしない。消費済みoutput・marker・grantを再使用しない。

今回のone-shot契約はfail-closed STOP後の自動retryを認めない。次は不一致fieldを識別できる技術診断・必要な修正/新SOURCEまたは契約bindingの固定と、新しいone-shot実行承認・fresh gateが必要。matched B0/全面map/次stageへ進行しない。原稿・新科学claim・共有設定・他job変更なし。軽量監査資料だけを公開しmandatory STOPする。
