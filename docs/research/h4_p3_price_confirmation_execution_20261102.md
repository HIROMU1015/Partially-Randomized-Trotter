# P3追加価格確認：保存native再利用・新32 trajectory/方式

2026-11-02 JST。ユーザーは[公開設計](h4_p3_poststage3_cost_and_stage4_design_20261102.md)
（`60515a33ce6231e9ff4ffdb40124ed845373351c`）の限定範囲を許可した。
本資料はその実装接続を定める。旧文書・結果・全STOPの状態は保持する。

固定条件はH4/1Å/STO-3G/中性singlet/system8/full256、保存DF tol-only1e-8/rank10、
案A/T=.8/S2/q8/r4/K4、M_B/M_C、共通Gaussian保持、P=0、cosine/+Im sine。
P320201–320232、DF320301–320332の各32本、64 trajectory/2048 event位置を一度だけsamplingする。
新Full64 bodyと128 wrapper compile、各軸価格の平均・不偏分散・標本SE・観測min/max、cos/sin共分散のみを取得する。
方式間の平均差/SEは独立標本で計算する。pilot8本/36価格をpoolせず、E/R/Emptyを新生成しない。

[adapter](../../src/trotterlib/matched_partition_price_mean.py)は保存basis QPYのgate/qubit順をregistryへ戻し、
旧basis hash・law成分確率を照合する。Full-onlyのcomposeは旧trajectory_partsのFと同じ順・scalar。
新Givens合成・T/DF再対角化は0。保存half/actual inverse/controlled halfを復元する。
独立oracle用に保存Tの半指数を1回だけ計算し、保存system halfの行列を1回評価して同じ256次元基準で照合する。
各新Fullを独立DictionaryOracleと同じcomplex128/積誤差政策で照合し、全64 PASS後のみcompileへ進む。
phase alignment・閾値緩和・dense fallbackなし。

[runner](../../scripts/run_h4_p3_price_confirmation.py)は旧Stage 3入力loader・event・oracle・price・QPY保存を再利用する。
nativeはspawn processで独立targetを並列検証し、共通参照・component oracleは各process内でcacheする。
compileはrun03のspawn/max_tasks_per_child=1を使い、各128 targetを異なるprocessで各1回扱う。
Qiskit1.3.0/opt1/seed17/rz,sx,x,cxを維持し、共有package/環境は変更しない。
小さな人工testsで保存QPY/basis hash、Full-onlyと旧Fのliteral一致、order0/2/4・scalar/+Im・測定compiler、
processの復元と平均/SE/共分散を検査し、SOURCE固定・公開後に1 batchを実行する。
SOURCE固定前の最終人工検査は新規5件＋既存process回復4件の計9 PASS。

保存先は`artifacts/df_pauli_matched_partition_validation/2026-11-02/stage3/run04`。
全event/native/input/compiled QPY・worker・metrics・集計とSOURCE/inputを保全する。
失敗時は正常workerを回収し、原因と途中結果を残して自動retryせずSTOPする。
成功時も標本SEを母平均tail保証とせず、population上界・新N/Gはnull。
追加標本・新signal/B・finite-bit実効平均・高精度化・Stage 4・主要claim採択は範囲外。
結果公開・remote一致確認後mandatory STOPし、次の判断をユーザーへ返す。
