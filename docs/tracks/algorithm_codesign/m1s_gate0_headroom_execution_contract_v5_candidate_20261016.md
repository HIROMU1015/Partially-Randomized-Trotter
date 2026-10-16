# M1-S Gate 0 v5 最終実験契約候補・SOURCE接続（2026-10-16）

**CANDIDATE_NOT_ADOPTED / IMPLEMENTED_NOT_SCIENTIFICALLY_EXECUTED。**
tail局所resolverを独立v5 SOURCEへ接続し、SOURCE・入力・17精度条件・proof/model・runtime・authorization guardを固定した。
この候補はSOURCE未接続だった前段のv5設計候補を具体化するもの。採択済み契約や実行許可ではない。
adopted=false、coverage_model_adopted=false、adoption_record=null、execution_ready=false。
科学実行authorization・grant・markerは作成せず、旧v4のTECHNICAL_STOP、W/v/rho=null、one-shot消費済み状態を保全する。

## 1. 固定対象

|役割|commit／path|
|---|---|
|今回の基点・tail proof公開|b5a185cc0c3aae0e477bfc52b93061e86feb7dec|
|係数演算経路の元SOURCE|606036db52da2f080cf846e48824e5e418430762|
|旧v4正式契約|5e57955baa79e42f37aa233ac011c60539e431fd|
|v5固定SOURCE S|69de022dc8142f703423b1ce0bd42f05e123954e|
|v5 module|[m1s_gate0_v5](../../../src/trottertracks/algorithm_codesign/m1s_gate0_v5)|
|v5 runner|[run_m1s_gate0_headroom_v5.py](../../../scripts/tracks/algorithm_codesign/run_m1s_gate0_headroom_v5.py)|
|v5最終machine契約候補|[execution_contract_v5_candidate.json](../../../artifacts/track_b_m1s_gate0_v5_source_preparation/2026-10-16/v1/execution_contract_v5_candidate.json)|
|SOURCEとproof/model hash|[source_identity.json](../../../artifacts/track_b_m1s_gate0_v5_source_preparation/2026-10-16/v1/source_identity.json)|

固定SOURCEはv5 module4ファイル、runner、既存宣言的tail spec、package init2ファイルの計8ファイル。
proof文書と元model JSONのbytesも別にbindingする。
Machine契約候補SHA256：4bb4ea3078921e30749e0e85e8f7b6801ec3ab9466d5d5d53ae72050cf9b44cd。

対象はlinear H4、1.00 Å、STO-3G、8 spin orbitals、DF rank12。
HD=保存DF順prefix G3/G6/G9（0–2／0–5／0–8）、HR=残りの非冗長signed Pauli tail。
signed集約後にabs、lexicographic IXYZ、cutoff=0、exact floating zeroのみ削除、HR identityをHDへ移す。
保存数値sector oracleを再正規化・上書きせず、期待値は保存norm²で割る。delta window／finite-stepなし。
入力NPZは8652 bytes、SHA256 3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a。
H/state hash、occupation mode0 MSB、alpha-even/beta-oddも元scopeと同一。

## 2. 実装した接続と誤差台帳

同じ条件付き算術モデルで導出した[tail局所証明](m1s_gate0_tail_local_bound_proof_20261016.md)を参照し、

    G_l = sum_pq (abs(Re g_lpq)+abs(Im g_lpq))
    S_R = sum_l_in_tail abs(lambda_l)*G_l²
    alpha_R = gamma_(713+r)*S_R

をFractionで解くresolverを追加。r=9/6/3、index722/719/716を固定splitと宣言的specから決める。
実測counterや観測された再構成差・残差・W/v/rhoはalpha_Rの入力にしない。
raw尺度と3行分の局所上界は、将来の認可runでcoefficient生成前に一度解決する。

|台帳|用途と保存先|
|---|---|
|global|v4のglobal式・全H尺度・coefficient包絡・再構成・Hermiticity・phase・残差演算誤差。report.global_formula_resolution_before_metrics|
|local|S_R、r、gamma index、alpha_R、proof起源。report.tail_coefficient_resolution_before_metrics.G3/G6/G9|
|各splitのlaw|local alpha_Rをlambda区間・ec・W/v/floorへ伝播。row.error_ledger.tail_local_coefficient_errorとformula_resolution|

global側のBudget constructor既存prefixとglobal_limits/q/l1/gamma/upはv4と同じAST。
新しいtail cache・resolverを追加し、global alphaを上書きしない。
W/v/floorの算術はalphaの参照を局所値へ変更する以外、既存代数を保持した。
HR identity除去はnonidentity値を丸めず、HD scalar加算誤差はglobal ledger側に残る。
失敗時にも初期化済みの両台帳と全split statusを保存する。

## 3. 精度17項目

[項別inventory](../../../artifacts/track_b_m1s_gate0_v5_source_preparation/2026-10-16/v1/precision_inventory.json)を参照。

|項目群|v5候補|
|---|---|
|eps_basis、eps_hermiticity_factors|0／0、exact equalityを維持|
|eps_state_norm、eps_imag_expectation、eps_prob|2^-40／2^-40／2^-46を維持|
|eps_W、eps_v、eps_rho|2^-24／2^-24／2^-12を維持|
|eps_H、eps_hermiticity_H、eps_phase|旧global再構成・Hermiticity・phase式を維持|
|eps_imag_coeff、eps_imag_energy|旧global式を維持|
|eps_state_residual|1e-8 Ha+B_res、absolute_original_solver_acceptance_v1を維持|
|W_error_budget、v_error_budget、W_floor|局所alpha_Rを用いる新しい3 formula ID|

すべてのprecision値／selectorをSOURCE内の固定候補と照合し、ceiling変更・scalar fallback・任意expressionを拒否する。
独立したsplitごとに元selectorから解決し、前splitの解決scalarを流用しない。
lambda_lo<=0、eW>2^-24、ev>2^-24ならmetrics前にPRECISION_UNRESOLVED。
小分母やrho interval精度、probability／expectation品質も従来のgateを維持する。

証明のS_R/lambda_hat<=8等の十分領域は**新しい実行gateにしない**。
新SOURCEにそのratioで候補を選別する分岐はない。実入力への新式の代入・W/v/rho取得も今回は行っていない。

## 4. Runtimeと性能方針

Python3.11.0rc1、NumPy1.26.4、6 build-file hash、CPU1 worker、thread5環境各1、GPUなしを維持。
4 GiB RLIMIT_AS、result/log各256 MiB、context4 MiB、Pauli65536、global contribution1000000、scientific wall capなし。
Runtime/build/resource値の変更も固定SOURCE照合で拒否する。
metadata preflightは指定venvのversionと6 build hashの一致を確認したが、NumPyはimportしていない。

cached JW bilinear、共有DF block/reference、mask cacheを維持。
既に求めたraw G/lambda尺度を3tailのcacheへ再利用する追加だけで、並列化条件・比較条件を変更しない。
性能・memory十分性・数値correctnessの実測検証は未実施。speedupを主張しない。

## 5. Authorizationとone-shot

実行にはproof固定commit < v5 SOURCE S < ユーザー採択契約C < 別実施authorization Aの履歴とbytes整合が必要。
固定proof/model・SOURCE8ファイル・scope record・raw input・runtime/build/threadをguardで照合した後でのみscienceをimportする。
default/status/schema/validate-configは科学処理を起動しない。
contract completenessのexecution_readyと、実際の科学実施許可は別条件である。

v5 authority schemaはm1s_gate0_separate_execution_authorization_v5、scopeはGate0_H4_DEV_G3_G6_G9_W_v_rho_v5_once。
authorityの新規pathはartifacts/track_b_m1s_gate0_v5_execution_authorization/配下を要求し、旧v4 authorityを流用しない。
outputはartifacts/track_b_m1s_gate0_v5_result/配下の未存在directoryを要求し、旧v4出力namespaceを拒否。
clean HEAD=A、source/contract ancestry、exclusive mkdir、fileのxb保存、no overwrite／no implicit retryを維持。
採択JSONのflagやauthorization記録は、ユーザーの実際の承認を代行しない。

global failureは残る全行を未評価として保存。split failureもその行の記録を保持し、事前固定した全splitのstatusを残す。
自動再実行・precisionの緩和・科学的GO/STOP分類・T1c/Q4/Q5等への拡張は行わない。
旧v4の消費済み外側markerとresult directoryは保全する。v5実行用のauthorization/grant/marker/outputは未作成。

## 6. 証明とSOURCEの静的整合性

[static checker](../../../scripts/tracks/algorithm_codesign/check_m1s_gate0_v5_static.py)と
[最終静的検査](../../../artifacts/track_b_m1s_gate0_v5_source_preparation/2026-10-16/v1/static_source_validation.json)は122項目PASS。
指定Python3.11.0rc1でも実施した。科学module・NumPy等のimport、NPZ open、数値Budget methodへの呼出しをtrapして確認した。

- _sum/_combine/_multiply/_jw_ladder/_one_body_pauli、reference/Pauli作用、load・quality gateはv4と同じAST。
- prepareはlocal ledgerの取得だけを追加。splitの係数生成・射影・identity・再構成prefixは同じAST。
- global式とW/v伝播代数、rho区間、failure分類、resource/保存guardを照合。
- 新schema/namespace、3世代commit guard、proof binding、17項目改変拒否、古いauthorityの拒否を確認。
- 新しいratio gate、観測counterによるgamma変更、scalar fallbackがないことを確認。

[proof/source対応表](../../../artifacts/track_b_m1s_gate0_v5_source_preparation/2026-10-16/v1/source_proof_alignment.json)と
[安全なCLI metadata照会](../../../artifacts/track_b_m1s_gate0_v5_source_preparation/2026-10-16/v1/safe_cli_metadata_queries.json)も保存した。
静的PASSは証明の独立受理・実入力誤差のcertificate・runtime correctnessではない。
scalar複素積／IEEE／BLAS／fsum／underflow等の条件は未認証のまま、CONDITIONAL_ENGINEERING_NOT_CERTIFIEDを維持する。

## 7. 残る採択事項とSTOP

[未決事項](../../../artifacts/track_b_m1s_gate0_v5_source_preparation/2026-10-16/v1/unresolved_items.json)：
新しい係数上界の受理判断、v5全契約のユーザー正式採択、条件付きcoverage採択、別の科学実施承認。
独立した数学的確認／GPT詳細レビューは今回開始していない。
source接続は完了したが、初回の実入力誤差予算・numerical correctness・runtime・W/v/rhoはすべて未検証。

[evidence index](../../../artifacts/track_b_m1s_gate0_v5_source_preparation/2026-10-16/v1/evidence_index.json)、
[publication receipt](../../../artifacts/track_b_m1s_gate0_v5_source_preparation/2026-10-16/v1/publication_receipt.json)を参照。
新科学計算・予算の実/合成入力評価・Pauli生成・数値correctness test・回路simulation/compile・Gate0実行0、
新authorization/grant/marker0、旧成果と親worktree保持。mandatory STOP。
