# G6/G7 W2c-B3 実行driver準備

状態：**TECHNICAL_DRIVER_PREPARED_ADOPTION_PENDING**。2026-10-19 JST。
実行engine・G7/F3接続・guard・測定・streamingを実装し、人工/mockで検証した。
実G7 correctness・sampling・CPU profiling・数値再解析・compile/synthesisは**0件**。
候補は未採択、実行承認・grant・marker・claimを発行／消費していない。公開照合後mandatory STOP。

直接の基点は`c8cbeb18f76aef2cb1329d8c810036372fa07cc0`。
専用branchは`track-b-g6-w2c-b3-driver-preparation-20261019`。
固定資料の取得元commitと基点は[source inventory](source_inventory.json)で区別する。
指定27件の公開bytesと固定Git blobを照合済み。原SOURCE・raw結果・親worktree・並行作業を変更していない。

- [Decision Handoff／Q1–Q5](handoff.md)
- [driver・費用座標とfairness](driver_and_measurement.md)
- [angle／cache／setup／保存／seed](angle_cache_output_contract.md)
- [実行契約候補](execution_contract_candidate.md)／[JSON](execution_contract_candidate.json)
- [16 cellとcall予算](comparison_schedule.json)
- [新driver](../../../src/trottertracks/algorithm_codesign/g6_w2c_b3_driver.py)／[人工tests](../../../tests/tracks/algorithm_codesign/test_g6_w2c_b3_driver.py)
- [人工テスト記録](artificial_test_results.json)／[技術・保全確認](verification.json)／[公開照合](publication_receipt.json)

対象はG7 `P5_general_order`、p=(1/5,3/10,1/2)、x=5/7、m=5、
Ordinary／Partial Return+tail／Closed P3+tail／Full、X/Y、σ=+1。
G9 Closed P5 armを追加していない。分子・geometry・basis・DF rank・L_D・δ窓は本形式入力に非該当。
固定Nを使用する記述的実費比較であり、母期待費用・一般計算量・量子native総費用の優位を保証しない。

実入力成立性は**NOT_ESTABLISHED**。資料不足や未採択を数学的不適用のNOT_APPLICABLEと呼ばない。
正式採択と別途実行承認が必要。これまでのmandatory STOPや旧消費済みclaimは維持する。
