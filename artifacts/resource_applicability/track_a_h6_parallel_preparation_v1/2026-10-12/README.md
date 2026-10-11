# H6 次回142 group：2 worker並列化の準備 v1

SOURCE `67fed3285acd757226fada25278d9a569cac76b8`、branch `track-a-h6-parallel-preparation-20261012`。
REVIEWのactual SHAはこの資料を固定したcommitの `git rev-parse HEAD` を参照。
現在はlocal commitのみ。利用者がpushする方針を維持し、remote検証済みとは扱わない。

[設計・制約](../../../../docs/research/track_a_h6_parallel_preparation_v1.md)、
[固定manifest](sealed_v1/manifest_candidate_v1.json)、[未認可grant草案](sealed_v1/authorization_draft_v1.json)、
[binding監査](binding_audit_v1.json)、[source/blob closure](source_closure_audit_v1.json)、
[Phase A完了監査](phase_a_completion_audit_v1.json)、[34限定test結果](test_result_v1.json)。

## 完成したこと

次の複数groupを最大2 processで同時処理するcontrolを実装した。
各group内は固定 `h6_matched_cost_port_v3.cost_task` を呼び、cosine/sineを従来順序で処理する。
Science SOURCE `5a903aea2492c2d211c1e15e7737df0d50694c12` の224 pathsとcompilerは変更0。
旧control/runtime233 pathsも変更0、新control/runner/preparer4 pathsを足し、closure237を固定した。
Testは別のSOURCE対象1 file。科学条件・seed・trajectory・full-wrapper・費用定義の変更はない。

Phase A index447 `B1_p19_q8_4th` はnative status COMPLETE、worker/coordinator exit0、
first STOPなし、cleanup errorなし、process回収・PID残存0を確認した。
coordinator wall 2756.640秒、worker peak RSS 7.929GiB、
VmSize peak 8.485GiB。
当該q8の32GiB AS下での完走を確認した。全142 groupの完走や旧compilerとのbit同一性を意味しない。
旧449 groupに447を足して450/592 group・904 wrappers。残りは固定index450–591の142 group。
Phase Aのraw/cache/runtimeは元のworktreeで保持し、監査hashだけを記録した。

## 次回の実行案・未承認

| 対象 | CPU | AS | RSS停止閾値 |
|---|---|---|---|
| worker slot0 | 20 | soft/hard32GiB | 18GiB |
| worker slot1 | 21 | soft/hard32GiB | 18GiB |
| coordinator | 22,23 | soft2/hard32GiB | 2GiB |

内部thread1・Qiskit num_processes1、own-run affinity、起動はnice10を維持する案。
CPU20/21は別のphysical core。Aggregate RSS40GiB、job RAM予算64GiBを維持する。
18×2+coordinator2+margin2=40GiB。ASは仮想address上限、RSSは1秒観測/max5秒のSTOP閾値で、RAM予約ではない。
他groupが18GiBを必要とするか、2 worker実science時のpeak/transient使用量は未測定。
Host/cgroup PSI、OOM増加、欠測、I/O遅延、pressure/headroomの従来guardを維持する。

固定候補profile SHA `4e32bb9ba57c38280562390100ebf916e9858218534504ead1519625a609bd75`、既存H6専用venvをread-only使用。
元233 closure・33凍結input・5220旧raw・449受理group・profile34273 critical filesを再照合し、Phase A native receiptを結合。
新planはtechnical seal済みでも `approved=false` / `runtime_authorization=false` / `allowed_cpus=[]`。
別の新しいexact grant、source/profile/input/carry照合、fresh resource gate、未使用output/one-shot lockが必要。
Draft grantでは起動できない。142 group完了/最初のSTOP後mandatory STOP、自動retry/resumeなし。
探索592完了時はreceiptまで保存し、selection/map作成・独立32確認・H8へ進まない。

## 予算・容量

Known carry：compile904、control_probe292、trajectory434、occurrence1024、primitive1470、reference_matvec400。
今回のstatic demand：compile284、control_probe68、trajectory142、occurrence1136。
既知合計はcompile1188≤1712、control_probe360≤1024、trajectory576≤832、occurrence2160≤6656。
過去STOPの未記録actualはunknownのまま保存し、0補完・budget resetをしない。
Phase/total wall capは従来null。実際の142 group wallとspeedupは未測定。
最大2 workerのためgroup処理の理想的throughputは約2倍が上限目安だが、負荷・group時間差・I/Oで低下する。
同じ1 groupのtranspile自体を2倍にする実装ではない。

Output2GiB、worker単体output32MiB、全worker log1MiB、monitor journal32MiBを維持。
Phase A当該groupの実保存は52 files/26569 bytes、run全体67 files/3073309 bytes。
同程度のgroup metadataなら142組で約3.60MiBだが、他groupの一時/cache/log量は未測定。
32MiB×142=4.44GiBという単体上限の最大和はoutput2GiBを超えるため、全groupの容量完走保証はない。
同時worker用64MiBとterminal余地を予約し、全体outputとjournal上限ではfail-closed STOPする。
監視journalの有限上限はwall cap nullでも維持される。
Filesystem直前読取では約404.56GiB/220M inodes空き。既存quota無効監査は保持し、将来launchでfresh再確認する。
Fresh launch要求は既存10GiB/100000 inodes、host216.25GiB/effective64GiB等を維持。
この資料作成時にfuture launch preflightをPASS扱いしない。

## 限定検証と公開

34 local metadata/mock/minimal-process tests PASS（0.91秒）。
最大2 tiny childの同時稼働、wait/reap、各worker exit、RSS超過、欠測/stale、cleanup非ESRCH error、first STOP保持、
32GiB結果受理と旧admissionの全科学checks parity、旧/draft grant拒否、task/CPU/output/lock/seed/compiled metadata bindingを検査。
AS2GiB・thread1・wall120秒・file16MiB、child AS128MiB、home内一時領域。
人工/実transpile追加0、科学array読込0、新科学runner0、approved grant0。
限定差分確認であり独立review・immutable CI・本science並列throughput検証ではない。
Test起動時の2件の環境選択失敗もhome evidenceで保全し、既存venvへinstall/updateはしていない。
H4は継続し、worker2/4/5/6・driver16・observer18のCPUと提案CPUは重ならない。

Pushは利用者側。必要なcommand：

```bash
git -C /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h6-parallel-preparation-20261012 \
  -c maintenance.auto=false -c gc.auto=0 \
  push origin HEAD:refs/heads/track-a-h6-parallel-preparation-20261012
```

次に承認が必要なのは、固定142 groupの一回探索、2 worker/CPU20,21と新RSS18GiB閾値の案。
承認後にREVIEW remote照合とfresh gateを通し、別grantで一度実行する。
現段階では未実行、absolute launch commandは未確定。現在のH4は変更しない。
