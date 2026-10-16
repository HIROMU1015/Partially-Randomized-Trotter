# V1 p7/q1 S2：登録2軸実compiled結果

SOURCE `84b9ea22bf78c7193f2137ce42a04400292b8e68`。linear H6/1Å/STO-3G/tol-only DF19/T=.8/δ=.8/B0+I p7 q1 S2。

cosine/sineとも実測14,492 RZ、固定予測14,524.2、相対誤差各161/72460（0.222191554%）で2%未満PASS。総費用再会計は実施していない。

- [結果報告](../../../../docs/research/track_a_v1_p7_compile_results_20261016.md)
- [固定契約・準備履歴](../../../../docs/research/track_a_v1_p7_compile_readiness_20261016.md)
- [各軸validation](validation_complete.json) / [CSV](axis_validation.csv)
- [cosine実測](cosine_measurement.json) / [sine実測](sine_measurement.json)
- [終了terminal](execution_terminal.json) / [外部監視・回収terminal](launcher_terminal.json)
- [終了監査・raw hash/bytes](completion_audit.json) / [資源観測](resource_observations.jsonl)
- [新承認receipt](authority_receipt.json) / [fresh gate receipt](launch_gate_receipt.json)
- [全予約](reservations.jsonl)

実測・control・開始・終了・予約・資源観測はprivate実行証拠のbyte copy。authority/gate receiptはprivate絶対path・他job一覧を含まないmetadata抜粋。snapshot/state/archive/config・承認grant原本・lock・private起動script・raw logはprivateに保全し、公開completion監査のbytes/SHAへ結合した。

wall73.241100秒、native exit0、全回収、first_stop=null、cleanup_errors=[]。SOURCE21/provider230/runtime34273/input15/旧grant等660は終了後再照合一致。追加compile・retry・refit・新signal/reference/scalar/primitive/SCF/DF/state/sampling・費用再会計・V2は0。

経験的allowance・厳密bias保証未解決・全grid費用誤差未保証を保持。primary agent自身のlocal実測/監査であり、外部再現・新レビューではない。

**mandatory STOP。保存shotとの総費用比較は別途の明示承認待ち。**
