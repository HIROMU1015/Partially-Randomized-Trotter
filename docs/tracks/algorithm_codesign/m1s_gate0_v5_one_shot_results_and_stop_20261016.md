# M1-S Gate 0 v5：初回one-shot結果・mandatory STOP（2026-10-16）

**HEADROOM_DIAGNOSTICS_COMPLETE_AWAITING_SCIENTIFIC_JUDGMENT。**
別途ユーザー実施承認により固定SOURCE・採択契約から一回だけ実行した。
全G3/G6/G9で事前精度予算を通過してW/v/rhoを取得し、到達1,239 gateはすべてPASS。
条件付きengineeringの保存数値oracle診断であり、科学的GO/no-go・finite-step・最終総資源優位は判断していない。
one-shot消費済み、retry0。追加科学計算は行わずmandatory STOP。

## 固定scope・実施承認と証拠

対象はlinear H4 1.00 Å、STO-3G、8 spin orbitals、保存DF rank12。
HD=保存DF順prefix G3/G6/G9（0–2／0–5／0–8）、HR=残りの非冗長signed Pauli-tail。
signed集約後abs・lex IXYZ、cutoff0、exact floating zeroのみ削除、HR identityをHDへ移動。
保存数値sector-ground oracleを再正規化・上書きせず、期待値を保存norm²で割る。delta window／finite-step評価なし。

|固定対象|識別子／資料|
|---|---|
|proof|b5a185cc0c3aae0e477bfc52b93061e86feb7dec|
|SOURCE S|69de022dc8142f703423b1ce0bd42f05e123954e|
|採択契約 C|50b3669b9253eb889fa4daa87e793646df2d5138|
|実施準備・外側launcher|40248f6844d4e0bd0ad975e94c39ce2c1b211abf|
|別実施authorization A|8da63d672f779c947dc295eb2c5d524d927ac66b|
|[実施承認文脈](../../../artifacts/track_b_m1s_gate0_v5_execution_preparation/2026-10-16/v1/user_execution_approval_basis.json)|ユーザーの別turn「次の作業に進んで大丈夫」と直前の科学実施承認境界を保存。IDE fields・旧v4 consent/authorityは根拠にしない|
|[新v5実施authorization](../../../artifacts/track_b_m1s_gate0_v5_execution_authorization/2026-10-16/v1/execution_authorization.json)|SOURCEと契約、未存在output一つにbinding。clean HEAD=Aから一回launch|
|[raw result](../../../artifacts/track_b_m1s_gate0_v5_result/2026-10-16/v5_one_shot/result.json) SHA256|8e32925e99b64ce6bc2c939e45ba51485d18b1eb4b62952e22a0debe1d9d4ec5|
|[execution context](../../../artifacts/track_b_m1s_gate0_v5_result/2026-10-16/v5_one_shot/execution_context.json)／[runner log](../../../artifacts/track_b_m1s_gate0_v5_result/2026-10-16/v5_one_shot/runner.log)|raw config/auditとnative FD logを保存。logは0 bytes|

SOURCE8/proof2/raw入力/契約hash、Python3.11.0rc1、NumPy1.26.4配布物と6build hash、threadsを非数値preflightで照合した。
入力8652 bytes、SHA256 3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a。
17精度条件・1e-8 Ha+B_res・local alpha_R=gamma_(713+r)S_R・global会計・runtime/capsは採択版と同一。

## 保存された数値診断

以下はraw resultの保存値の転記であり、監査・報告で算定式やmetricsを再計算していない。
全行metric_status=HEADROOM_DIAGNOSTIC_ONLY、technical_stop=false、scientific_decision=REQUIRES_USER_GPT_JUDGMENT。

|HD prefix|lambda_R [Ha]|W|v|rho|
|---|---:|---:|---:|---:|
|G3|0.8137790393725155|0.9625288642498284|0.13226039984301002|0.5687046408452855|
|G6|0.026294457645549787|0.38153576790356514|0.1400352453142982|0.6835152259560796|
|G9|5.47382434715216e-07|0.9753534991954069|0.24871838546505062|0.6275016625614326|

|HD prefix|条件付きW区間|条件付きv区間|条件付きrho区間|
|---|---|---|---|
|G3|[0.9625288623629953, 0.9625288661366616]|[0.1322603991924234, 0.13226040049359664]|[0.5687046403726475, 0.5687046413179235]|
|G6|[0.3815357660168907, 0.38153576979023957]|[0.14003524466434603, 0.14003524596425038]|[0.6835152241968483, 0.6835152277153109]|
|G9|[0.9753534973085102, 0.9753535010823037]|[0.24871838481420963, 0.24871838611589162]|[0.6275016619811269, 0.6275016631417383]|

## 局所精度予算と17条件

全3行で採択済みW/v ceiling 2^-24（5.960464477539063e-8）とrho uncertainty ceiling 2^-12を通過した。
raw ledgerの値を以下へ転記する。観測counterや結果をalpha_Rの選択に用いていない。

|split|gamma index|alpha_R [Ha]上界|W_error_budget|v_error_budget|rho uncertainty|
|---|---:|---:|---:|---:|---:|
|G3|722|3.3245164659834197e-13|1.8868329994921536e-09|6.505865970527515e-10|4.726380398167863e-10|
|G6|719|9.699383939055367e-15|1.8866743893013905e-09|6.499521562902885e-10|1.7592313161785002e-09|
|G9|716|2.323234013722741e-19|1.88689659135531e-09|6.508409645051397e-10|5.803056923880946e-10|

local係数台帳はcoefficient生成前、law誤差台帳はW/v/rho計算前に解決・hash保存。
global ledgerは旧v4保存値と同一であり、各canonical Pauli table hashもv4と同一。
変えたのは採択済みの係数算術誤差モデルであり、係数生成target・cutoffや精度要求ではない。

入力・基底埋込み・16 ladderのJW/occupation reference対応・exact Hermiticity・full/HD/HR再構成・identity phase・state norm/residualに加え、
今回初めてprobability normalization・Pauli expectation虚部・mean consistency・W/v範囲と区間整合・rho区間のgateへ到達し、全1,239件PASS。
[17項目適用表](../../../artifacts/track_b_m1s_gate0_v5_execution_audit/2026-10-16/v1/precision_application_status_17.json)はSOURCEのgate対応と保存statusを示す。
有限入力でのSOURCE内correctness/整合検査の成功であり、全経路の独立数値実装照合やplatform certificateを得たものではない。

## Oracle品質・coverage・解釈の限界

full-H residualは1.022703352687838e-15 Haで、固定quality 1e-8 HaとB_res=5.333090193902594e-9 Haの合計1.5333090193902595e-8 Haを通過した。
旧solverと新referenceの算術経路差は未測定・未認証。厳密固有状態への距離、full-Fock ground、非縮退性、pair-sum/Jensen floorの適用条件は認証していない。
IEEE/BLAS/math.fsum/underflow/FTZのモデルはユーザー採用の条件付き仮定であり、CONDITIONAL_ENGINEERING_NOT_CERTIFIEDを維持する。
raw SOURCE/ledgerのAUTHOR_CONDITIONAL_DERIVATION_NOT_INDEPENDENTLY_REVIEWEDは導出時の起源ラベルであり、後続の独立レビューと採択は[正式採択記録](../../../artifacts/track_b_m1s_gate0_v5_formal_adoption/2026-10-16/v1/user_final_contract_adoption_record.json)で追跡する。

本runはv4停止後、同一H4 development条件で精度モデルを改訂した事後的技術評価。未使用系への独立transferではない。
rhoやleading診断をfinite-step visibility、finite RPE、shot/native費用、基底エネルギー推定の最終総資源優位へ読み替えない。
GPT詳細科学レビューを今回新規開始しておらず、科学的GO/no-go・方向判断をCodexは代行しない。

## Runtime・保存・停止

CPU1 worker、5 thread環境各1、GPUなし。SOURCEのcacheを保持し、別benchmarkやspeedup比較は行っていない。
runner elapsed=0.4421113417483866 s、maxrss_linux_KiB=62232、product contributions=27781。
RLIMIT_AS=4 GiB、result/log各256 MiB、context4 MiB、scientific wall capなし。これはこの一回の実測値である。

[消費済exclusive markerのbytes copy](../../../artifacts/track_b_m1s_gate0_v5_execution_audit/2026-10-16/v1/exclusive_launch_marker.json)、[launch完了](../../../artifacts/track_b_m1s_gate0_v5_execution_audit/2026-10-16/v1/launch_completion.json)、
[saved-only auditor](../../../scripts/tracks/algorithm_codesign/audit_m1s_gate0_v5_saved_result.py)、[保存監査](../../../artifacts/track_b_m1s_gate0_v5_execution_audit/2026-10-16/v1/saved_identity_status_audit.json)を保全した。
新科学one-shot1、runner invocation1、retry0、canonical tables3。保存監査はbytes/JSON/hash/statusと記録済みceiling比較のみで、NPZ配列decode・Pauli再生成・budget/metric再評価0。
既存v4 TECHNICAL_STOP/null/消費済marker、旧SOURCE・結果・採択記録・authorization、親worktreeを保持する。
新しい計算・研究拡張へ進まず**mandatory STOP**。以降の科学的判断・追加実施は別のユーザー指示を要する。
