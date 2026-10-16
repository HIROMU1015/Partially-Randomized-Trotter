# M1-S V1：γ依存の成立条件・境界を調べる改訂契約候補

2026-10-17 JST。`CANDIDATE_NOT_ADOPTED_NOT_EXECUTABLE / MANDATORY_STOP`。

ユーザーは単一γ比較から、γ依存の記述的診断へ変更する**研究方針**を正式採択した。Task A先行・最終Task B・無補正raw直接route・全保存cellという既採択方針を保持する。実験契約、未決の科学値・resource候補の正式採択や、科学実行承認とは区別する。

## 固定資料と保全

基点は`be2da2c3c4ed1d826a76e8a08a98ff6c92ce90c5`。旧SOURCE `de622bd77ebd61a6d2cf5b69e29c4e620c8563b3`、旧候補`56ca76401e6cb332a4dbece764cadf3116a17285`と旧V0一式は上書きしない。新SOURCEは`src/trottertracks/algorithm_codesign/m1s_saved_signal_gamma_v1/`、新artifactは`artifacts/track_b_m1s_v1_gamma_preparation/2026-10-17/v1/`。

[受領GPTレビュー原本](../../research/m1s_v1_scientific_conditions_independent_review_20261017.md)を全文確認し、byte一致で新規保存した。SHA256 `a0d4a884d557e7f2337de8a88f31182b1176b7ab0d7c0bb3f2451a4712b8a575`。原本内の未採択という過去記録は変更しない。今回の方針採択は別intakeに保存する。

正式共通運用指示は[2026-10-16版](../../research/research_project_instructions_unified_2026-10-16.md)、SHA256 `8323be3f65856adf0f4d095d8e68546db3326822023c20e101ba5f596605ebfa`。新しいGPT詳細科学レビューは開始していない。

対象はlinear H4 1.00 Å／STO-3G／8 modes／保存DF rank12、G3/G6/G9、保存stateの元norm²商、`+iTH`。既存δ={1/8,1/16} Ha^-1、R={2,4}、T={1,4} Ha^-1。shared_pm／ordinary／independent_pmを全24非零cellで対応比較し、72方式行＋36保存zero control＝108行、48対照を固定する。新γ grid・時刻・分割・Hamiltonianは追加しない。

入力は旧完了raw結果1個のみ。

- commit：`025c5a90888b02b73d2815fbb10d6bd1d2823602`
- path：`artifacts/track_b_m1s_finite_signal_v2_result/2026-10-16/completion_attempt_02/result.json`
- 454422 bytes、SHA256 `ff6c5192d8cca769cd61a1478cea01671d058a85a8f6a4cdcdc9baa672a56fe0`

V0での科学artifactの使用はbytes/hash照合だけ。数値JSONを開いて再解析しない。新SOURCEはHamiltonian・Pauli・state・signalを再計算しない。

## γ依存の数学的対象

保存誤差の条件付きcoverageを前提に、各方式mについて

\[
 A_m^-=\max(0,\widehat A_m-r_{A,m}),\quad
 b_m^+=\min(\pi,|\widehat b_m|+r_{b,m}),\quad
 M_m(\gamma)=A_m^-\sin(\gamma-b_m^+)
\]

を定義する。共通領域は`0<gamma<pi/2`の中の

\[
 \max(b_{sh}^+,b_{cmp}^+)<\gamma<\pi/2,\qquad A_{sh}^-,A_{cmp}^->0.
\]

各方式の正の領域も全行に報告する。`b>=pi/2`なら領域なし、π区間を考慮して判別できなければ未解像、必要証拠がなければ比較不能とする。共通領域にあるbiasはπによるclipが働かないので、SOURCEはその領域で`abs(b_hat)+r_b`をexactに使えることを検査する。

`phase_bias_rad`は保存された`TOTAL_PRINCIPAL_PHASE_DIFFERENCE_TO_RAW_H`。既に含まれるモデル差`eta=|T|B`を二重加算しない。旧2^-24 signal／2^-18 radは数値解像度の仕様であり、新γや科学的許容biasへ流用しない。

### レビューのtrade-off境界式

`a=A_sh^->c=A_cmp^->0`、`0<=v=b_cmp^+<u=b_sh^+<pi/2`が成立する場合だけ、

\[
 X=a\cos u-c\cos v,\quad Y=a\sin u-c\sin v>0,\quad
 M_{sh}-M_{cmp}=X\sin\gamma-Y\cos\gamma
\]

を使用する。`X<=0`なら全共通領域でsharedのmarginが小さい。`X>0`なら一意な内点境界

\[
 \gamma_* = \arctan(Y/X),\quad u<\gamma_*<\pi/2
\]

があり、境界より大きなγでsharedのmarginが大きい。`X`や`Y`の符号を算術区間で示せない場合は`ARITHMETIC_UNRESOLVED`。閾値を緩めたりprecisionを上げたりしない。

逆のtrade-offはsharedとcomparatorを入れ替えた**同じ定理**を適用し、向きも明示的に戻す。式の仮定を満たさないcellへ無条件にarctanを代入しない。

同じbiasなら振幅下界だけで関係を分類する。振幅下界が大きくbias上界が小さい側は全共通領域でmarginが大きい。これらは`NOT_APPLICABLE_EQUAL_BIAS`／`NOT_APPLICABLE_*_DOMINANCE`と記録し、trade-offの内点境界があるとは書かない。全順序caseを扱う技術的な実装であり、比較対象・科学的評価軸の追加ではない。

境界区間`[g_lo,g_hi]`の外側でのみ向きを分解する。区間内は未解像で、唯一の点境界を任意の中心値で代表させない。定理上の一意存在が成立しても、区間が共通領域の端に重なる場合は`ROOT_EXISTS_LOCATION_UNRESOLVED`を保持する。

## 十分shot設計との関係

第一候補は`PER_METHOD_ROW`、fresh-IID、Hoeffding、X/Y等配分。全方式に同一のαとpolicyを適用する候補である。各shotでfreshな乱択labelと±1測定outcomeを仮定する。

\[
 N_{m,axis}(\gamma)=\left\lceil\frac{K}{M_m(\gamma)^2}\right\rceil,
 \qquad K=4\log(2/\alpha_{axis}).
\]

row policyは`alpha_axis=alpha_total/2`。代替family policyを選ぶなら`alpha_total/144`で全72方式行にunion boundを置く。前者を全表同時confidenceとは呼ばない。保存数値半径のcoverageを統計的αへ加算しない。

共通のK>0は連続因子の順序から消えるので、γ境界そのものはαに依存しない。一方、**整数shotのstrict削減には天井関数の同数があり得る**。SOURCEは連続因子の順序と、正確な記号条件

`ceil(K/M_sh(gamma)^2) < ceil(K/M_cmp(gamma)^2)`

を分ける。このversionは任意の一点γを選ばず、γ grid・絶対shot数・整数削減率・ランキングを生成しない。採択後の承認済み実行ではKの区間と境界区間・成立条件を保存する。αを具体化しても、連続marginのstrict差だけから整数Nのstrict差を認定しない。

Task Aの保存raw-H理想信号の円周位相を扱う事後的oracle診断であり、運用時に無料で取得できるbias/振幅の保証ではない。物理的shot保証、最適shot数、Task Bの多round energy/RPE、native総資源を保証しない。数値予測の分類は`IDEAL_SIGNAL_HYPOTHETICAL_SUFFICIENT_DESIGN_GAMMA_CONDITIONS`、証拠区分は`CONDITIONAL_ENGINEERING_NOT_CERTIFIED`。

## 分類とSTOP

|分類|意味|
|---|---|
|`NO_COMMON_PHASE_DOMAIN`|選んだbias上界で正の共通領域がない。一般的性能no-goではない|
|`EVIDENCE_UNRESOLVED_NOT_COMPARABLE`|正のvisibility下界・phase情報・coverageなどがない|
|`ARITHMETIC_UNRESOLVED`|domainまたは係数符号を採択precisionで判別できない|
|`ALL_COMMON_GAMMA_*`|共通領域での連続marginの順序。整数strict利得を意味しない|
|`UNIQUE_TRADEOFF_BOUNDARY`|レビュー仮定下の一意境界。位置区間の解像状態も別fieldに保存|
|`TECHNICAL_STOP`|schema・入力identity・cap・例外・時間超過等。prefixと全未評価行を保存しretryしない|

全G9対照を保持し、旧`UNRESOLVED_ABSOLUTE_CONTRAST`は`ABSOLUTE_CONTRAST_UNRESOLVED_NO_RELATIVE_RESCUE`としてcarryする。新しい相対因子や境界値で物理的有意差へ昇格しない。`scientific_GO`、native費用、総資源はnull、GO/NO-GO自動採点をしない。

## Resource・guardと採択前の状態

[精度・resource技術根拠](m1s_v1_gamma_precision_resource_and_handoff_20261017.md)を参照。CPU1 numerical worker／GPUなし、metadata supervisor、aggregate virtual memory2 GiB、入力1 MiB、result/log各16 MiB、rational65536 bits、series4096項、bound step100万の候補を保持する。今回の明示指示に従って有限CPU/wall-time候補を新規に提示するが、**採択前のmachine契約の時間上限はnull、実行guardは閉じる**。

machine候補：`artifacts/track_b_m1s_v1_gamma_preparation/2026-10-17/v1/execution_contract_candidate.json`。全science fieldはnull、`adopted=false`、`coverage_model_adopted=false`、`contract_grants_execution=false`。`PER_METHOD_ROW`は別fieldで第一候補と記録し、暗黙のdefaultにしない。

新SOURCE Sを先に固定し、採択契約C、別途明示実施指示を記録するauthority-only子Aが将来必要。S<C<A、hash、clean HEAD=A、入力identityを照合する。数値engineはguard通過後にのみimportする。exclusive出力dirでone-shotを消費し、完了／失敗でmandatory STOP。今回新しいexecution authorization／grant／markerは作成しない。

**研究方向の変更だけが採択済み。改訂実験契約の正式採択とV1数値解析には、それぞれ別途明示承認が必要。V0完了後mandatory STOP。**
