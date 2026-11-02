# H4/H6 S4：実primitiveの発生箇所確認

**COMPLETE_MANDATORY_STOP**。承認済み66 primitive評価を一回実施し、元8行の差を全て照合した。

[結果報告・解釈・次の判断](../../../../docs/research/track_a_h46_s4_primitive_sites_results_20261018.md)

- [変更箇所4件の短い表](changed_sites.json)
- [全33 block×2時間の観測](primitive_sites.json)
- [8 metric行との照合](eight_row_reconciliation.json)
- [SOURCE・元source・入力byte binding](source_input_binding.json)
- [環境profile](environment_profile.json)
- [承認・planの消費済みprojection](plan_authorization_projection.json)
- [66件の予約・完了と旧ledger保全](call_ledger_summary.json)
- [本体terminal](terminal.json)／[native exit・wait/reap](native_exit.json)
- [追加12人工tests](artificial_tests.txt)
- [保存結果checker](verify_saved_results.py)／[ファイルbytes/SHA](manifest.json)

実行SOURCE：`4d60519e11ecf70a2d19f536ce69ed95b314724c`。元科学SOURCE `536a3671c0ec2939ff3886e40467db26a27ec65a`のcounter/nativeを変更していない。
branch：`track-a-h46-s4-primitive-site-validation-20261018`。最終REVIEW/remote SHAはチャット報告を参照。

H4はDF fragment index1/block index2/qubit0,1,7（全て0始まり）で保持RZが正味+1。H6はfragment index5/block index6/qubit11で正味−1。q反復によりRZ/CX差+2/−16となり、両軸の保存差と一致した。全blockのRZZ maskとphase有無は一致。原因はS4負時間の1 ULP差→共有producerの厳密zero選別で、Qiskit処理前に生じる。

Qiskit import・compile・trajectory・full-wrapper・full理論再会計は0。今回の数値評価では実diagonalを読み、元producerを66回呼んだ。先行の人工調査だけの結果とは区別する。raw codecや実diagonal配列を公開していない。

```bash
python -P -B artifacts/resource_applicability/track_a_h46_s4_primitive_sites/2026-10-18/verify_saved_results.py
```

このcheckerは保存結果のmetadata/byte検査のみ。実科学runnerの再起動はしない。

次の担当：ユーザーがcounterの最小互換性修正と新SOURCEでの限定受入確認を進めるか判断する。元counter修正・過去表更新は今回未実施。mandatory STOP。
