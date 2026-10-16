# M1-S有限step複素信号pilot v2：技術修正後の計算完了（2026-10-16）

**PILOT_COMPLETE_AWAITING_SCIENTIFIC_JUDGMENT / CONDITIONAL_ENGINEERING_NOT_CERTIFIED / mandatory STOP。**
ユーザーの最新指示「計算完了まで行ってほしい　修正があれば行い計算完了まで行って」に基づき、技術修正・SOURCE pin更新・必要な別試行まで進め、試行02で全108行を完了した。科学的閾値・grid・比較・runtime・条件付きplatform仮定は変更していない。追加の科学計算・GPT詳細レビューは開始せず停止する。

## 承認と試行の来歴

[最新指示と修正・計算完了の範囲](../../../artifacts/track_b_m1s_finite_signal_v2_completion/2026-10-16/workflow/repair_and_completion_authority.json)／[元発言](../../../artifacts/track_b_m1s_finite_signal_v2_completion/2026-10-16/workflow/user_completion_instruction.txt)。
以前のno-retry／途中STOPは今回の明示指示により技術修正に必要な範囲で更新した。各試行は独立したSOURCE・契約・authorization・排他marker・出力を持ち、同じ失敗SOURCEを無修正で再実行していない。過去結果・消費済markerは上書きしない。

|段階|SOURCE／契約／authorization|結果|
|---|---|---|
|旧起動前STOP|旧S fbc2edea／元C c6254be7|精度文字列parse不具合。科学起動0、未消費の記録を保全|
|試行01|S 3d973027／C eea84105／A a6a00c7d|初回correctnessとG3 twirl PASS。16行まで評価し、G3 R2 delta1/16 T4 sharedの半径1.1354363398539135e-7が2^-24を超えtechnical STOP|
|試行02|S 2eac35ff／C 5afd57ed／A 3dfd4470|初回correctness・全3 split twirl PASS、108行すべて完了|

[試行01の保存結果](../../../artifacts/track_b_m1s_finite_signal_v2_result/2026-10-16/completion_attempt_01/result.json)と[失敗保存・次の修正](../../../artifacts/track_b_m1s_finite_signal_v2_completion/2026-10-16/attempt_01/failure_preservation_and_next_repair.json)を保全。
[試行02契約](../../../artifacts/track_b_m1s_finite_signal_v2_completion/2026-10-16/attempt_02/execution_contract_adopted.json)／[試行02 authorization](../../../artifacts/track_b_m1s_finite_signal_v2_execution_authorization/2026-10-16/completion_attempt_02/execution_authorization.json)。
最終SOURCE=`2eac35ff33f100fa96e296b737752addde5af5de`、契約C=`5afd57ed3feb900b8f97317c38cc62f52959981b`、実施A=`3dfd4470f8755937be1818c24580e8cbb76d3171`。S<C<A、各Aでclean HEADを要求した。
元の科学契約の各fieldはSOURCE commit／SOURCE hashes／そのpin記録以外すべて一致する。最新ユーザー指示は技術修正・実行の根拠であり、新しい科学的GOの採択ではない。

## 技術修正

第一に、`2^-24`／`2^-18`をFractionへ直接渡す箇所を、採択された表記だけを受理してexactな1/2^24／1/2^18を返すhelperへ変更した。閾値の値は不変。
第二に、exact Phi(X)=E[A_j X A_j]の既にレビューされたFrobenius収縮性を誤差伝播へ適用した。各A_jはunitaryなので ||Phi(X)-Phi(Y)||F <= ||X-Y||F。
丸められた中心Xhatに対する算術・h・probability・trigの誤差を従来モデルで計算し、元の入力半径eを一度だけ加える。||Xhat||2<=b+eを使い、入力不確かさの重複増幅を除く。
[条件と導出](../../../artifacts/track_b_m1s_finite_signal_v2_completion/2026-10-16/workflow/phi_contraction_radius_repair.md)。数値結果への一致を根拠に閾値やlibrary allowanceを緩和したものではない。平均写像の中心演算は不変で、両試行で重複した16行の複素信号中心もbit一致した。
IEEE／complex BLAS／abs・hypot／trig・log／underflow等の条件付き仮定は未認証であり、correctness PASSをplatform certificateへ格上げしない。

## 計算対象・完了状況

linear H4 1.00 Å／STO-3G／8 modes／保存DF rank12、生成順G3/G6/G9、保存数値oracleを元norm²で割った期待値、全256次元。
delta=1/8,1/16 Ha^-1、R=2,4、T=1,4 Ha^-1、zero control込み108行（非零72＋zero36）。shared ±／ordinary／independent ±を同じ概念的独立label数sRで比較する。native回路費用の等価性は主張しない。

[raw result](../../../artifacts/track_b_m1s_finite_signal_v2_result/2026-10-16/completion_attempt_02/result.json)、454422 bytes、SHA256 `ff6c5192d8cca769cd61a1478cea01671d058a85a8f6a4cdcdc9baa672a56fe0`。
[108行の信号CSV](../../../artifacts/track_b_m1s_finite_signal_v2_completion/2026-10-16/final/signals.csv)／[48対照のCSV](../../../artifacts/track_b_m1s_finite_signal_v2_completion/2026-10-16/final/contrasts.csv)／[保存結果summary](../../../artifacts/track_b_m1s_finite_signal_v2_completion/2026-10-16/final/saved_result_summary.json)／[保存結果auditor](../../../scripts/tracks/algorithm_codesign/audit_m1s_finite_completion_saved.py)。

|確認事項|結果|
|---|---|
|全予定行|108/108評価、すべてNUMERIC_DIAGNOSTIC_ONLY|
|初回correctness|非可換2x2 exact rational physical-word列挙の3方式、Walsh/direct twirl PASS|
|全空間twirl|G3/G6/G9各256次元literal tensor probe PASS|
|signal半径|最大3.8150467991527054e-8、要求2^-24以下|
|principal phaseの数値半径|全108行resolved、最大4.6659565697084285e-8 rad、要求2^-18 rad以下|
|保存結果監査|1,022 PASS、FAIL0。新numerical kernel呼び出し0|
|実行時間・メモリ|worker約52.286秒、最大RSS92496 KiB、4 GiB address-space cap|
|call/resource|6577 matrix products、1736 twirls、1990898 exact events。すべて上限内|

phase半径が小さいことはphase biasそのものが小さいこととは異なる。主値biasはCSV／raw resultに保存しており、taskとして許容されるかは未判断。

|split|sharedの絶対visibility差（ordinary／independentに対し計16対照ずつ）|
|---|---|
|G3|16対照すべてRESOLVED_HIGHER_VISIBILITY|
|G6|16対照すべてRESOLVED_HIGHER_VISIBILITY|
|G9|16対照すべてUNRESOLVED_ABSOLUTE_CONTRAST|

これは固定H4・law・time grid・条件付き数値モデルでの診断結果である。G9の精度以下の絶対差を相対改善率で成果へ格上げしない。rhoのsmall-step仮説と実測finite-step信号、visibilityとphase bias、概念label数とnative費用を区別する。

## モデル差・限界・停止

raw H_fullと実際のsplit target H_tilde_j=HD_j+HR_saved_jを区別し、全3分割でB_jとT=0,1,4のeta_j(T)=abs(T)B_jを保存した。数値丸めとモデル差は別台帳であり、純アルゴリズム誤差の分離測定や点値を主張せず、三角不等式の区間だけを保存する。
モデル位相はvisibility下界・branch条件を満たす場合の条件付き上界のみ。補助split ideal信号は新規評価していない。旧v5未保存full-H／HDの直接一致は依然UNDETERMINED。
保存oracleは認証された厳密固有状態ではなく、T1b単一rate適用域・RPE shots／native回路費用／総資源削減・一般的なアルゴリズム優位性・科学GOは未認定。
旧v4 STOP/null、v5／独立one-shot結果・SOURCE・契約・marker、起動前STOP、試行01、親／他worktreeを保全する。科学試行は計2回（失敗1＋完了1）、個々の試行内のretry0、旧科学run再実行0。
**計算完了。追加科学計算・研究拡張・新GPTレビューを開始せずmandatory STOP。**
