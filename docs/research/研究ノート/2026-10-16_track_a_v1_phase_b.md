# 2026-10-16 V1 Phase B

Phase Aのcommit bf865de1…を受け、[Phase B固定モデルと検証契約](../track_a_v1_phase_b_readiness_20261016.md)を準備した。
利用者は準備指示の後に「計算を実際に開始するところまで進めて」と明示したため、保存S2 fit・固定S4最大4軸の検証起動までを実施対象とした。
旧B0 binary profile未在庫、旧compiler出力bit同値未確認を残し、SOURCEが同じであることとprofileが同じであることを混同しない。現profileへの費用転用は未検証仮説として固定し、各軸2%以上でSTOPする。
compile前固定のmodelを新実測で自動再fitせず、検証後の5精度費用比較・V2は自動実行しない。H5と他作業、旧raw/grant/ledgerを保全する。

14:10 JST追記：[起動snapshot](../track_a_v1_phase_b_run01_started_20261016.md)。準備SOURCEfcf37e67…のremote一致後、fresh gate PASS、実compile開始を確認。既存650ファイル不変。全軸検証結果は未確定で、起動時点の記録として公開する。

終了確認追記：[登録全4軸PASS・完了STOP](../track_a_v1_phase_b_run01_results_20261016.md)。B0誤差321/326305、B0+I誤差0。wall3037.384秒、native wait0、全回収。SOURCE/provider/runtime/9入力/650旧ファイル不変。モデル再fitせず、5精度再会計・V2を開始せずmandatory STOP。登録4軸の局所検証と全prefixへの保証を区別する。
