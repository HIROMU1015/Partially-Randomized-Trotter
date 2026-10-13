# H4 B0+I12並列run02 起動確認

SOURCE `924b779c2bc6f6e844d6283c355f91c73aed6445`、grant `15010795611b156656f943f761e47e99820765e6`、run `h4-b0i-map240-parallel-run02-20261014`。利用者が「できるだけ並列化高速化して計算を開始して、そのためならメモリなどの上限は外してよい」と指示し、通常の並列化/修正/起動判断を継続した。

12worker、234 new compile＋6 exact numeric FP照合再利用＝240 wrapper。科学条件・compiler options・既存private venv・6凍結入力を維持し、旧科学source48不変、closure59。新25限定mock検査PASS、旧科学レビュー・人工compile campaignの再実施なし。group単位の保持ブロック再利用、重いprefix/qの先行配分、非同期回収で並列化した。

CPU役割 `{"workers": [1, 3, 4, 5, 6, 7, 0, 2, 10, 11, 12, 14], "driver": 15, "observer": 13}`、worker AS/RSS32GiB、driver8GiB soft limit（worker継承用hardは保持）、observer AS256MiB/RSS64MiB、共有headroom64GiB／admission456.25GiB。5秒監視・pressure/OOM・own-run stable pidfd STOP/cleanupを維持する。

最初のrun01はdriver8GiB hard limitの継承でbootstrap STOP、actual予約/compile0、cleanup完了。旧出力/lockを保持し、65536 bytesと0.7697398792952299秒をcarryへ加算した。run02でhard-limit継承を修正して起動した。budget resetや失敗返却はない。

Driver PID 603385、起動snapshotで81/240 complete、12 worker ready、observer first_failureなし。これは起動確認であり最終合格/費用地図完成ではない。stage4h/task10min、出力256MiB/監視160MiB/inode1024、累積actual74805/charge17GiBは維持する。

[固定認可/最終binding/検査/起動snapshot](../../artifacts/resource_applicability/track_a_h4_b0i_map_preparation/2026-10-14-parallel-run02/README.md)。private driver log: `/home/AbeHiromu/projects/h4-handoff-evidence/20261014/h4-b0i-map-parallel-run02/driver.log`、observer: `/home/AbeHiromu/projects/h4-handoff-evidence/20261014/h4-b0i-map-parallel-run02/future-map240/h4-b0i-map240-parallel-run02-20261014-control/observer.jsonl`、結果: `/home/AbeHiromu/projects/h4-handoff-evidence/20261014/h4-b0i-map-parallel-run02/future-map240/h4-b0i-map240-parallel-run02-20261014`。完了またはfail-closed STOPで終了し自動retry/次stageなし。共有環境/H8/他job/GPU変更0。
