# H4 Pauli-JW PR central-control A_GATE：保存order2 index1

2026-10-24 JST。ユーザー添付35fb6e4cの明示承認で、保存index1のbody-only replay1回、D2 central-control1回、cosine/sine各wrapper1回・費用compile各1回を実施する。現時点は実行前のSOURCE記録であり、完了結果・終了監査を追記する。

固定対象：`trajectory_index=1`、ID `596897f2d5ec2fb9b28a2b73c16423192d1494fd490fc58ca52cfc818baae0aa`、trajectory hash `0d16d201c87fdbb5f3e219dffb7dca7d48a69799290453e9ebb110a267db0e4b`、event hash `9cea6ce6d3659094b6b0d0a6da8b935cf32860c97a2e01a75a0761eded9f1779`。

Linear H4/隣接1.00 Å/STO-3G、system8＋ancilla8、Pauli184非identity項・PR prefix27、S2/T=.8/q8/r4/K4、元law・係数・状態を維持する。Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、backend/coupling/layout/routing指定なし。元Hadamard wrapper、状態準備費用除外。

基点 `966c2cb323e3fe34032dc3be9f6b962936415e77` と契約をGitHubで実取得する。元H4 SOURCE0c329b30、D1 SOURCE9843b803、D2 SOURCE74515d8eを保持し、新SOURCEを科学call前にcommitする。

[接続](../src/trotterlib/h4_a_gate.py)、[runner](../scripts/run_h4_a_gate.py)、[人工検査](../tests/test_h4_a_gate.py)、[証拠入口](../artifacts/non_df_h4_a_gate_order2/2026-10-24/README.md)。D1 append関数・event復元・forward/event/reverse scheduleを使い、generic wrapperまで生成するA.buildは実H4で呼ばない。D2のcentral変換・wrapper/compiler・seal検証・並列compile batch・native witnessを再利用する。再利用関数のartifact root/axis/script bindingだけを新namespaceに接続し、旧関数そのものは変更しない。

Stage Aは保存hash・label/sign/角度・order2・phase規則の確認、Stage Bはbody/lift各1回とSOURCE token一致、Stage Cは共通controlled bodyから両軸wrapperを作り各1回compileする。異なる軸価格はそれだけで不一致としない。失敗callは非返還、retry・cohort・追加axes・N/G・residual診断・D3への自動進行なし。今回の1本は事後構造coverage選択であり、標本平均や未観測tailの保証にはしない。

人工12件ではbody-onlyと元D1 bodyのtoken/global phase一致、非可換order2と個別phase、両control branch・両軸wrapper、generic生成・sampling・compile不呼出し、grant/予算/STOP、source gate失敗でcompile0、duplicate axis拒否を確認する。既存20件の意味対応証拠もSHAで保全して使う。主agent自身の検査で、compiled全H4の数値誤差certificateではない。

科学grantは新namespaceで一度だけ発行・消費し、native exit/reap・SOURCE/input/runtime/outputを保全する。旧Pauli-PR historical hash FAIL/原因UNKNOWN、元H4 native UNKNOWN、旧N/G null、C/L COMBINATION_BLOCKED、旧compiler MISSING_EVIDENCE・正式certificate未受理は変更しない。

完了後mandatory STOP。次の担当はユーザー：結果を踏まえて残254 cosineのA_COHORT_COS等が必要か判断する。Codexは新たに明示承認された範囲だけ準備・実行する。
