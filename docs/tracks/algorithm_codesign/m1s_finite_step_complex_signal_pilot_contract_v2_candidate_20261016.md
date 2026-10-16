# M1-S有限step複素信号pilot：実験契約v2候補（2026-10-16）

**CANDIDATE_NOT_ADOPTED / IMPLEMENTED_UNEXECUTED / CONDITIONAL_ENGINEERING_NOT_CERTIFIED。mandatory STOP。**
今回のユーザー指示は契約・SOURCE・静的検証・公開の準備だけ。研究方針・本契約・条件付きcoverageは未採択、科学実行は未承認。
Gate0/v5本体と独立one-shotを再実行せず、新しい有限信号、Pauli係数、模擬入力correctness、simulation/compileは0件。


本書は旧v1候補を保全した新しいv2候補。[旧契約](m1s_finite_step_complex_signal_pilot_contract_candidate_20261016.md)のraw-H差を一括してアルゴリズム誤差へ含める記述を改訂する。
[受領GPTレビュー原本](../../research/m1s_finite_step_pilot_gpt_independent_review_20261016.md)は27909 bytes、SHA256 `fb2ff8695b9bf0355a84fb3a368543e9d36aa6e97dc172a034116efb190432b4`。原本とbytes一致、全文読解済み。新しいGPTレビューを開始したものではない。
SOURCEはv2として新規固定し、旧S9758aeaa・旧候補b48d4cdc・公開a398a3faと全旧証拠を変更しない。

## 受領レビューと対象

[GPTレビュー原本](../../research/track_b_m1s_gate0_v5_independent_result_gpt_scientific_review_20261016.md)を指定ローカルファイルからbytes不変で保存した。
24927 bytes、SHA256=`07864ab817cc5fa8f3b8f2a834619c3ea4af090dec3e553cb7be3438a401f6fb`。
Gate0のavailable scope再現・代数的headroomを限定受理し、T1b単一rate適用域、有限step利益、RPE/native総資源、科学GOは未確立とした既存レビューである。
[今回の文脈](../../../artifacts/track_b_m1s_finite_signal_preparation/2026-10-16/v2/decision_context.json)に、現在の準備指示を実施承認と混同しないよう記録する。Codexが新しい詳細科学レビューを始めたものではない。

linear H4 1.00 Å/STO-3G/8 spin orbitals/保存DF rank12、保存生成順prefix G3/G6/G9、保存数値sector oracleを維持する。
Pauli eventはsectorを保存するとは仮定せず、全256次元を用いる。新しい分子・geometry・DF law・T1c・大規模スペクトル計算へ広げない。
[SOURCE/input/evidence台帳](../../../artifacts/track_b_m1s_finite_signal_preparation/2026-10-16/v2/source_and_evidence_identity.json)：base=`a398a3fab6f68abc2a47799d7454b312082af292`、SOURCE S=`fbc2edea3816bf8f68508f62c6da595631df03b6`。
[machine候補](../../../artifacts/track_b_m1s_finite_signal_preparation/2026-10-16/v2/finite_signal_contract_candidate.json) SHA256=`03944ae9aa462df10e2d9a0aab68e1db531413a731c20707b4048136f6a873f3`。

## 同じ理想taskと3対照

数学的なHD halfはexact exponent。実装は誤差予算を伴うTaylor scaling/squaringであり、HDに別のPF誤差を導入しない。
H_R=lambda*h、h=sum p_j P_j、P_jは係数符号込みinvolution。A_j(theta)=exp(+i theta P_j)。同じ位置のshared二枝はlabelを共有し、負枝は**同順sign reversal**とする。full-word adjointへ変更しない。

|方法|一stepの数学的平均|概念的独立label数|branch側Pauli exp数|
|---|---|---:|---:|
|shared ±|D Phi_(lambda*delta/(2R))^R(D X D) D、D=exp(+i delta HD/4)|R|2R|
|ordinary|O=Do M_(lambda*delta/R)^R Do、Do=exp(+i delta HD/2)|R|R|
|independent ±|V=D M_(lambda*delta/R)^(R/2) D、Psi_ind(X)=V X V|R（各枝R/2）|R|

M_a=cos(a)I+i sin(a)h、Phi_a(X)=cos²(a)X+i cos(a)sin(a)(hX+Xh)−sin²(a)sum p_j P_j X P_j。
sharedはPsi^s(I)、ordinaryはO^s、independentは(V^s)(V^s)を保存stateで期待値評価する。V†XVや密度行列channelへ置き換えない。
位置・step間IID。同じT_rel=s*delta、同じH/state/law、同じsR label数で比較する。sharedのbranch exp数は2倍であり、独立draw数だけでnative回路費用の公平性・資源改善を認定しない。

理想参照は z_ref(T)=<psi|exp(+i T H_DF_full)|psi>/<psi|psi>。
未認証のexact eigenstateやRayleigh energyへの指数置換は使わず、sector内の固有値・pair-sum/gapも評価しない。

## 入力のexact targetと旧v5の限界

raw NPZは固定保存binary64入力。full-HとDF prefixはoccupation-bitの整数/dyadic演算で構成する。旧独立SOURCEの15 helperを明示的に再利用し、独立の新実装とは呼ばない。
sampling lawは旧v5の保存canonical tableを再利用し、binary64係数のexact abs sumをlambda、p=abs(c)/lambdaとする。保存probability floatの再正規化や新係数cutoffは行わない。
新HDはDF prefixに**保存v5のHR identity scalar**を加えたもの。旧v5の未保存HDを回収した証拠とは呼ばない。
H_D+H_R(saved law)−H_fullのexact row-sum上界を必ず報告する。この差をHDへ黙って吸収・修正しない。raw-H基準の総複素信号差と、split-targetに対する有限stepアルゴリズム差を下記の別台帳で区別する。
このtarget選択は本候補の採択事項である。旧v5の全scope UNDETERMINEDは変更しない。


## Hamiltonianモデル差の独立会計（v2の採択対象）

各 j=G3/G6/G9 に対し、H_full はraw保存binary64 DF入力をexact dyadicとみなして構成した理想参照、H_tilde_j=HD_j+HR_saved_j は3方式共通の実際のsplit-targetである。
HDはraw DF prefix＋保存HR identity scalar、HR_savedは保存nonidentity Pauli lawのまま。identity移動・符号・samplingを調整して両targetを一致させない。

DeltaH_j=H_tilde_j−H_full。exact算術でfull、HD、HR、DeltaHのHermiticityを検査し、不成立ならtechnical STOP。
B_j=max_i sum_k (|Re DeltaH_ik|+|Im DeltaH_ik|) [Ha] は、Hermitian性により ||DeltaH||_2 <= ||DeltaH||_infinity <= B_j を満たす。
有限次元Hermitian両HのunitarityとDuhamel積分から

    ||exp(i T H_tilde_j)−exp(i T H_full)||_2 <= |T| B_j
    |z_tilde_ideal_j(T)−z_full(T)| <= eta_j(T)=|T| B_j.

Tの単位はHa^-1、etaと複素信号は無次元。保存stateの元norm²で割った同じ期待値なのでstateが厳密固有状態である必要はない。真の化学Hamiltonianへの誤差保証ではない。
exact整数・有理数実装の正しさも今回数値検証していない。将来実行でB/etaを初めて評価し、全3分割のT=0,1,4でexact numerator/denominatorを保存する。今回は数値代入0。

結果schemaでは各完了行に次を保存する。

|台帳|意味・解釈|
|---|---|
|numerical_rounding|方式信号・raw理想参照・その差の条件付き数値半径。signal精度2^-24のgateにetaを足さない|
|Hamiltonian_model_difference|B_jとeta_j(T)、単位、両target、モデルのみの位相上界・未解決理由|
|finite_step_algorithm_error|H_tilde基準の純アルゴリズム差の**大きさの区間のみ**。point_value=null、algorithm_only_error_measured=false|
|total_complex_discrepancy_to_raw_H|方式−raw理想参照の総複素差。純PF／乱択／有限stepアルゴリズム誤差とは呼ばない|
|phase_interpretation|TOTAL_PRINCIPAL_PHASE_DIFFERENCE_TO_RAW_H。既存phase_bias_radはこの意味だけ|

数値差の中心 d_hat と数値半径 r_num に対し、|d_hat|の条件付き下上界をl,uとすれば、純アルゴリズム差の大きさは [max(0,l−r_num−eta), u+r_num+eta] に含まれる。
これは三角不等式によるenclosureであり、モデル差の実ベクトルやアルゴリズム差を厳密に分離して測定した値ではない。z_tilde_idealは数学的に定義するだけで**評価しない**。将来one-shotにも追加のsplit ideal指数評価は含めない。
eta>0ならMODEL_TARGET_CONFOUNDEDを表示する（原因帰属の制限であり数値FAILや科学NO-GOではない）。eta=0では当該時刻のモデル差は0。モデル差の大きさに応じたgrid・threshold調整は行わない。
shared−ordinary/independentのvisibility差は同一H_tilde内の直接対照である。モデル差をその数値半径へ加えず、公平比較の意味を維持する。

モデルのみの位相上界は、raw信号の条件付きvisibility下界 L を用いる。L>1/4、L−eta>1/4、eta<L、かつ数値raw信号を中心とする半径 r_ref+eta のdiskがzero・負実軸branch cutから分離することが十分条件。
条件成立時のみ asin(eta/L) の上側評価＋既存library予算を保存し、対象はArg(z_tilde_ideal)とArg(z_full)の差に限る。判定にはdiskを覆う長方形を使い保守的にbranch接触を排除する。
条件不成立・ratioの上側丸めが1以上ならmodel_phase_status=UNDETERMINED、bound=null。unwrapしない。モデル位相上界はモデルの大きさであり、測定位相の数値半径2^-18 radを緩和する項ではない。
旧v5のfull-H/HD未保存によるUNDETERMINEDは維持し、新しい再構成で直接一致を確認したと主張しない。

## 評価点・出力・判定候補

|固定候補|値|
|---|---|
|delta_rel [Ha^-1]|1/8、1/16|
|R（一stepの独立label総数）|2、4|
|T_rel [Ha^-1]|1、4、ゼロcontrol|
|s=T/delta|T1で8/16、T4で32/64|
|対象|G3/G6/G9 ×3方法、非零24 cells/72 signals、zero36 signals|

[全108行の予定表](../../../artifacts/track_b_m1s_finite_signal_preparation/2026-10-16/v2/planned_rows.json)はmetadataだけで生成し、信号や入力尺度を代入していない。
保存stateを上書き・再正規化せず、exact元norm²で割る。Re/Im z、|z|、1−|z|、|z_ref|−|z|、複素誤差、principal phase bias、−log|z|と誤差台帳を保存する。
visibility差はshared−ordinary、shared−independentを絶対値で報告し、誤差区間が0を跨ぐとUNRESOLVED_ABSOLUTE_CONTRASTとする。
beta=lambda²/(4R)、u=s delta²、kappa_sh=beta*u*(W+v)、kappa_ord=2 beta*u*Wも**仮説的先導尺度**として旧保存W/vから報告する。T1b適用域認証・有限step予測の成立を主張しない。
G9も同じ絶対時間gridで評価し、微小tailを理由に時間窓や閾値を後付け変更しない。相対改善率は本pilotではnull。絶対差が精度以下でもrhoだけで利益を認定しない。
数値的なvisibility差の分解と科学的GO/no-GOは別であり、自動的な研究GO条件は置かない。

## 精度・正しさ・resource候補

[意味論と条件付き誤差設計](m1s_finite_signal_precision_and_semantics_v2_design_20261016.md)を参照。
signal absolute radius上限2^-24、phase radius上限2^-18 rad、phase visibility floor1/4を候補とする。
phaseはArg(z conj(z_ref))の主値。floor不通過・branch cutへの誤差球接触・位相精度不足はUNDETERMINED_PHASEとし、unwrapしない。
signal予算超過、入力/意味論/correctness/resource不合格はtechnical STOP。完了値・予算・失敗理由と全未到達行を保存する。
BLAS/IEEE/norm/trig/log/underflow等の条件は未検証であり、coverage_model_adopted=false。厳密certificateではない。

将来の別承認one-shot内で、実入力より先にexact rational 2x2 physical-word列挙（R2/s2・非可換HD、Taylor32＋解析的剰余）をbinary64再帰と照合する。
各H4 splitでもliteral tensor monomial conjugationとWalsh twirlを固定probeで照合する。いずれもSOURCE準備済み・**未実行**。失敗したらその一回を保存して停止する。
CPU1 worker・5 thread環境各1・GPUなし、4 GiB、result/log各128 MiB、context4 MiB、時間上限なし。
256-square行列、100000 matrix products、10000 twirls、exact-input counter5億、integer bit65536、rational toy48 paths/16 gates、保存law512 terms上限（cutoffではなく超過時STOP）。
[resource/高速化計画](../../../artifacts/track_b_m1s_finite_signal_preparation/2026-10-16/v2/resource_and_acceleration_plan.json)：sのcheckpoint再利用、D/ideal exp cache、Walsh twirl、整数入力再構成を使う。巨大65536-square superoperator、Monte Carlo、eig/SVDは使用しない。性能・速度向上・実数値成功は未検証。

## 実行境界・未採択事項

[SOURCE](../../../src/trottertracks/algorithm_codesign/m1s_finite_signal_v2)／[runner](../../../scripts/tracks/algorithm_codesign/run_m1s_finite_signal_v2.py)／[静的checker](../../../scripts/tracks/algorithm_codesign/check_m1s_finite_signal_v2_static.py)。
新scope専用authorityのみ受け付け、base < S < 正式採択契約C < 別実施authorization A、clean HEAD=A、全hash・runtime・threadを要求する。
採択済み契約だけでは実行不可。今回authorization/grant/marker・出力directoryは作らず、adopted=false、coverage_model_adopted=falseを維持する。
将来output directoryの排他予約でone-shotを消費し、xb/fsync保存、自動retry・上書き・結果を見たgrid/threshold変更は禁止する。
[未採択事項](../../../artifacts/track_b_m1s_finite_signal_preparation/2026-10-16/v2/unresolved_choices.json)はgrid/公平比較、input targetのモデル差、条件付きcoverage/精度、runtime/STOP。ユーザーが判断し、その後さらに科学実施承認を別途受ける。
旧v4 TECHNICAL_STOP/null、v5/独立one-shot原結果、SOURCE/契約/authorization/消費済marker、親/他worktreeを保全する。
新科学計算・finite signal・Pauli係数生成・実/模擬input数値test・compile0、**mandatory STOP**。

[result schema v2](../../../artifacts/track_b_m1s_finite_signal_preparation/2026-10-16/v2/result_schema.json)は完了行の別台帳と未評価行を区別し、STOP後の未評価値を0で埋めない。正式採択・科学実施承認は未実施。**mandatory STOP**。
