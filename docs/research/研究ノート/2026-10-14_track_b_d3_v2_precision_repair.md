# 2026-10-14 D3 v2 precision adapter / source準備

利用者の限定修正・実行準備指示に基づき、別branchでFractionをmpmathへ明示変換した。exact epsilon/key/strict thresholdを保持し、変換後にbackend予約tokenを作る。科学条件・tool・precision・36 keys・48行・native会計1e-6・資源上限は変更しない。

新18 API接続/保護testsと旧63 guard/accounting testsがPASS。実backendはmock、登録比較・価格取得・pair・source実行0。初回のmock namespace解決エラーは訂正してraw logを保持。固定runtime/v1 identity一致。

旧source三blobは旧Sから照合し、v2実装を新current hashで保護。v1結果・authorization・消費済みmarker・STOPはbyte不変。pending v2 authority、新namespace未存在。旧approvalを流用しない。

[source/契約/試験範囲/次の別承認](../../tracks/algorithm_codesign/d3_v2_precision_repair_and_source_preparation_20261014.md)。準備資料をcommit/pushしremote取得後mandatory STOP。科学採否は変更しない。
