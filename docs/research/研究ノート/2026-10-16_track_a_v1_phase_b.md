# 2026-10-16 V1 Phase B

Phase Aのcommit bf865de1…を受け、[Phase B固定モデルと検証契約](../track_a_v1_phase_b_readiness_20261016.md)を準備した。
利用者は準備指示の後に「計算を実際に開始するところまで進めて」と明示したため、保存S2 fit・固定S4最大4軸の検証起動までを実施対象とした。
旧B0 binary profile未在庫、旧compiler出力bit同値未確認を残し、SOURCEが同じであることとprofileが同じであることを混同しない。現profileへの費用転用は未検証仮説として固定し、各軸2%以上でSTOPする。
compile前固定のmodelを新実測で自動再fitせず、検証後の5精度費用比較・V2は自動実行しない。H5と他作業、旧raw/grant/ledgerを保全する。

14:10 JST追記：[起動snapshot](../track_a_v1_phase_b_run01_started_20261016.md)。準備SOURCEfcf37e67…のremote一致後、fresh gate PASS、実compile開始を確認。既存650ファイル不変。全軸検証結果は未確定で、起動時点の記録として公開する。

終了確認追記：[登録全4軸PASS・完了STOP](../track_a_v1_phase_b_run01_results_20261016.md)。B0誤差321/326305、B0+I誤差0。wall3037.384秒、native wait0、全回収。SOURCE/provider/runtime/9入力/650旧ファイル不変。モデル再fitせず、5精度再会計・V2を開始せずmandatory STOP。登録4軸の局所検証と全prefixへの保証を区別する。

利用者の「その方針で進めて」を受け、[5精度の保存値再会計](../track_a_v1_reaccount_readiness_20261016.md)を専用one-shotで進める。実測4軸を優先し他の追加候補には固定modelを適用、旧比較の完全再現をgateとする。非相殺分解未保存候補を未解決のまま保持する。V2保留・新科学計算0・終了時STOP。

保存値再会計の終了追記：[結果と範囲](../track_a_v1_reaccount_results_20261016.md)。SOURCE2261a3ad…、旧表2170照合一致、4320会計行/60比較、wall6.024秒/native0/全回収。通常bias・固定model条件下でε=.05のPR優位が反転、中間εで削減率が10pp以上縮小。新最良対照は全てmodel推定、p7は外挿。非相殺診断は旧B0I12候補未保存で全domain最良をnullとした。旧653ファイル保全、新signal/compile/fit/V2実行0、mandatory STOP。
