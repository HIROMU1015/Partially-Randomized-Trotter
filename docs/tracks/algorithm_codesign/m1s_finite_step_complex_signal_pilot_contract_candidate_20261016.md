# M1-S有限step複素信号pilot：実験契約候補（2026-10-16）

**CANDIDATE_NOT_ADOPTED / IMPLEMENTED_UNEXECUTED / CONDITIONAL_ENGINEERING_NOT_CERTIFIED。mandatory STOP。**
今回のユーザー指示は契約・SOURCE・静的検証・公開の準備だけ。研究方針・本契約・条件付きcoverageは未採択、科学実行は未承認。
Gate0/v5本体と独立one-shotを再実行せず、新しい有限信号、Pauli係数、模擬入力correctness、simulation/compileは0件。

## 受領レビューと対象

[GPTレビュー原本](../../research/track_b_m1s_gate0_v5_independent_result_gpt_scientific_review_20261016.md)を指定ローカルファイルからbytes不変で保存した。
24927 bytes、SHA256=`07864ab817cc5fa8f3b8f2a834619c3ea4af090dec3e553cb7be3438a401f6fb`。
Gate0のavailable scope再現・代数的headroomを限定受理し、T1b単一rate適用域、有限step利益、RPE/native総資源、科学GOは未確立とした既存レビューである。
[今回の文脈](../../../artifacts/track_b_m1s_finite_signal_preparation/2026-10-16/v1/decision_context.json)と[準備scope](../../../artifacts/track_b_m1s_finite_signal_preparation/2026-10-16/v1/preparation_scope.json)に、現在の準備指示を実施承認と混同しないよう記録する。Codexが新しい詳細科学レビューを始めたものではない。

linear H4 1.00 Å/STO-3G/8 spin orbitals/保存DF rank12、保存生成順prefix G3/G6/G9、保存数値sector oracleを維持する。
Pauli eventはsectorを保存するとは仮定せず、全256次元を用いる。新しい分子・geometry・DF law・T1c・大規模スペクトル計算へ広げない。
[SOURCE/input/evidence台帳](../../../artifacts/track_b_m1s_finite_signal_preparation/2026-10-16/v1/source_and_evidence_identity.json)：base=`cc2f451324b7292ae50cef94ff794d46f225d892`、SOURCE S=`9758aeaa968d1d0af55723bc6386959f32d680cf`。
[machine候補](../../../artifacts/track_b_m1s_finite_signal_preparation/2026-10-16/v1/finite_signal_contract_candidate.json) SHA256=`3ee7e999eebf1a62b2da9f760b647c78d5e956211b5fc8ce26739eb54501096a`。

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
H_D+H_R(saved law)−H_fullのexact row-sum上界を必ず報告する。この差をHDへ黙って吸収・修正せず、同じraw-H理想参照に対する有限stepアルゴリズム誤差に含める。
このtarget選択は本候補の採択事項である。旧v5の全scope UNDETERMINEDは変更しない。

## 評価点・出力・判定候補

|固定候補|値|
|---|---|
|delta_rel [Ha^-1]|1/8、1/16|
|R（一stepの独立label総数）|2、4|
|T_rel [Ha^-1]|1、4、ゼロcontrol|
|s=T/delta|T1で8/16、T4で32/64|
|対象|G3/G6/G9 ×3方法、非零24 cells/72 signals、zero36 signals|

[全108行の予定表](../../../artifacts/track_b_m1s_finite_signal_preparation/2026-10-16/v1/planned_rows.json)はmetadataだけで生成し、信号や入力尺度を代入していない。
保存stateを上書き・再正規化せず、exact元norm²で割る。Re/Im z、|z|、1−|z|、|z_ref|−|z|、複素誤差、principal phase bias、−log|z|と誤差台帳を保存する。
visibility差はshared−ordinary、shared−independentを絶対値で報告し、誤差区間が0を跨ぐとUNRESOLVED_ABSOLUTE_CONTRASTとする。
beta=lambda²/(4R)、u=s delta²、kappa_sh=beta*u*(W+v)、kappa_ord=2 beta*u*Wも**仮説的先導尺度**として旧保存W/vから報告する。T1b適用域認証・有限step予測の成立を主張しない。
G9も同じ絶対時間gridで評価し、微小tailを理由に時間窓や閾値を後付け変更しない。相対改善率は本pilotではnull。絶対差が精度以下でもrhoだけで利益を認定しない。
数値的なvisibility差の分解と科学的GO/no-GOは別であり、自動的な研究GO条件は置かない。

## 精度・正しさ・resource候補

[意味論と条件付き誤差設計](m1s_finite_signal_precision_and_semantics_design_20261016.md)を参照。
signal absolute radius上限2^-24、phase radius上限2^-18 rad、phase visibility floor1/4を候補とする。
phaseはArg(z conj(z_ref))の主値。floor不通過・branch cutへの誤差球接触・位相精度不足はUNDETERMINED_PHASEとし、unwrapしない。
signal予算超過、入力/意味論/correctness/resource不合格はtechnical STOP。完了値・予算・失敗理由と全未到達行を保存する。
BLAS/IEEE/norm/trig/log/underflow等の条件は未検証であり、coverage_model_adopted=false。厳密certificateではない。

将来の別承認one-shot内で、実入力より先にexact rational 2x2 physical-word列挙（R2/s2・非可換HD、Taylor32＋解析的剰余）をbinary64再帰と照合する。
各H4 splitでもliteral tensor monomial conjugationとWalsh twirlを固定probeで照合する。いずれもSOURCE準備済み・**未実行**。失敗したらその一回を保存して停止する。
CPU1 worker・5 thread環境各1・GPUなし、4 GiB、result/log各128 MiB、context4 MiB、時間上限なし。
256-square行列、100000 matrix products、10000 twirls、exact-input counter5億、integer bit65536、rational toy48 paths/16 gates、保存law512 terms上限（cutoffではなく超過時STOP）。
[resource/高速化計画](../../../artifacts/track_b_m1s_finite_signal_preparation/2026-10-16/v1/resource_and_acceleration_plan.json)：sのcheckpoint再利用、D/ideal exp cache、Walsh twirl、整数入力再構成を使う。巨大65536-square superoperator、Monte Carlo、eig/SVDは使用しない。性能・速度向上・実数値成功は未検証。

## 実行境界・未採択事項

[SOURCE](../../../src/trottertracks/algorithm_codesign/m1s_finite_signal_v1)／[runner](../../../scripts/tracks/algorithm_codesign/run_m1s_finite_signal_v1.py)／[静的checker](../../../scripts/tracks/algorithm_codesign/check_m1s_finite_signal_static.py)。
新scope専用authorityのみ受け付け、base < S < 正式採択契約C < 別実施authorization A、clean HEAD=A、全hash・runtime・threadを要求する。
採択済み契約だけでは実行不可。今回authorization/grant/marker・出力directoryは作らず、adopted=false、coverage_model_adopted=falseを維持する。
将来output directoryの排他予約でone-shotを消費し、xb/fsync保存、自動retry・上書き・結果を見たgrid/threshold変更は禁止する。
[未採択事項](../../../artifacts/track_b_m1s_finite_signal_preparation/2026-10-16/v1/unresolved_choices.json)はgrid/公平比較、input targetのモデル差、条件付きcoverage/精度、runtime/STOP。ユーザーが判断し、その後さらに科学実施承認を別途受ける。
旧v4 TECHNICAL_STOP/null、v5/独立one-shot原結果、SOURCE/契約/authorization/消費済marker、親/他worktreeを保全する。
新科学計算・finite signal・Pauli係数生成・実/模擬input数値test・compile0、**mandatory STOP**。
