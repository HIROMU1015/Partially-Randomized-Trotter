# 2026-10-16 M1-S Gate 0 tail局所誤差上界の技術的導出

ユーザーがtail局所尺度・SOURCE演算経路からの上界証明と新精度契約候補の設計を指示。
新しいGate0実行・数値検証・GPT詳細レビューを開始せず、旧v4 TECHNICAL_STOP/null・one-shot消費を保持した。
H4 linear1.00 Å/STO-3G/8 spin orbitals/DF rank12、HD G3/G6/G9、deltaなし、保存数値oracle。

Pauli積の出力語を固定すると左語ごとの右語は一意で、squareのgroup長<=121。
bilinear116、one-body220、square673、scaled block681、tail713+rのgamma indexをSOURCEから導出した。
exact Hermitian入力のRe射影とnonidentity除去を含め、alpha_R=gamma_(713+r)S_Rという条件付き係数上界。
global resource capと実測counterは丸め経路indexへ流用しない。
S_R/lambda_hat<=8、L/lambda_hat<=2の解析領域では、旧W/v伝播式と同じ2^-24精度要求を満たす十分条件を導出。
これは新しい実行gateではなく、実入力で新しい算定式を評価した結果でもない。

宣言的specとv5 precision design candidateを作成。実行resolverは未接続でexecution_ready=false、adopted=false。
17精度項目のうちlaw側3selector候補だけ変更し、8scalar/6global式・残差1e-8 Ha+B_res・runtime/capsを保持。
53 static checks PASSはAST/schema/count/整数不等式の確認であり、数値correctnessやplatform certificateではない。
新NPZ decode・Pauli生成・実/合成入力の予算評価・W/v/rho・Gate0起動は0。
詳細：[条件付き証明](../../tracks/algorithm_codesign/m1s_gate0_tail_local_bound_proof_20261016.md)、[v5設計候補](../../tracks/algorithm_codesign/m1s_gate0_tail_local_precision_contract_v5_design_candidate_20261016.md)。
独立namespaceへのSOURCE接続・数値検証・完全な契約採択・別の実施承認を残し、mandatory STOP。
