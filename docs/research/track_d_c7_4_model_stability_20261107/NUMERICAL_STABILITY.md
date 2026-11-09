# C7-4：数値安定性の技術検証

実行SOURCE `dd31e1e2878479eabe2e7cc27c21b26c8c7dc794`。C7-3 sampling/raw35bfdとcompile/post94e94を区別し、旧判定/精度は変更していない。[保存残差](../../../artifacts/track_d_c7_4_models/2026-11-07/run02/numerical_residual_trends.json)／[保存原本の限定再照合](../../../artifacts/track_d_c7_4_models/2026-11-07/run02/saved_numerical_recheck.json)／[人工比較](../../../artifacts/track_d_c7_4_models/2026-11-07/run02/artificial_stability.json)。全H4再シミュレーション/新H4 compile/旧run resumeは0。

## 保存q2〜32の残差

|q|raw max entry|compiled max entry|observable max Frobenius|
|---|---|---|---|
|2|2.9309888e-14|2.6708796e-12|1.3482111e-11|
|4|4.4408921e-14|4.7672783e-12|2.1341502e-11|
|8|7.4606987e-14|1.0275804e-11|3.6567448e-11|
|16|1.1413093e-13|1.8696899e-11|6.1043751e-11|
|32|2.1138646e-13|3.493461e-11|1.1971565e-10|

component1e−10、observable2e−10、rtol0、phase fitなし。full-wrapper全行列のFrobeniusはcomponent基準と別で、その値をobservable閾値で再判定しない。q32の最大observable残差1.197156474600893e−10は閾値の約60%。q64で同じ閾値に収まるとは言えない。q32は小演算子評価が公式Aer double/unitaryへ切り替わっており、qだけの因果効果と断定しない。

raw CAR積/AB/actual inverse/vacuum rayに比べ、compiled観測の残差が大きいことは、長いnative gate列の評価・compiler変換/位相処理・observable集約が追加寄与する可能性と整合する。各処理を独立に再propagateしていないため、最大誤差を特定のpass/cache/global phaseへ定量配賦できていない。CAR ordered積の演算順序、Gaussian double行列、宣言χ/fsum/MPFR160、unitary/statevectorの反復丸め、256列のGram/observable積を区別する。phase精度160bitは全回路の演算精度を160bitにするものではない。

## 最大残差を持つ保存1 pathの再照合

最大値（同値は辞書順）で選んだq32 `q32_sample05`、reference/Y。原本hashを旧公開索引へ照合した上で、保存A/B/actual adjoint/Vと保存4 observableだけを読んだ。全256×256要素を保持し、extended精度の積・差・normで再照合した。行列や参照状態の再伝播、phase fit、renormalizationはない。

|check|max entry|Frobenius|status|
|---|---|---|---|
|saved_AB_extended_product|1.0466375e-13|1.3397877e-12|PASS|
|saved_actual_adjoint|2.0794391e-14|1.9893772e-13|PASS|
|saved_CAR_unitarity_extended_product|7.7964816e-14|6.8392329e-13|PASS|
|symmetric_directional_X_vs_CAR_extended_subtraction|9.8967501e-12|1.1216664e-10|PASS|
|symmetric_directional_X_vs_saved_raw|9.9449338e-12|1.1334366e-10|PASS|
|symmetric_directional_X_Hermiticity|1.1364756e-16|6.4962432e-16|PASS|
|symmetric_directional_Y_vs_CAR_extended_subtraction|9.7883923e-12|1.1502587e-10|PASS|
|symmetric_directional_Y_vs_saved_raw|9.8637765e-12|1.1624248e-10|PASS|
|symmetric_directional_Y_Hermiticity|6.9388939e-17|5.6104383e-16|PASS|
|reference_X_vs_CAR_extended_subtraction|1.0465628e-11|1.1748289e-10|PASS|
|reference_X_vs_saved_raw|1.045708e-11|1.1722238e-10|PASS|
|reference_X_Hermiticity|1.1114421e-16|6.8277221e-16|PASS|
|reference_Y_vs_CAR_extended_subtraction|1.0192625e-11|1.1971565e-10|PASS|
|reference_Y_vs_saved_raw|1.0163093e-11|1.194152e-10|PASS|
|reference_Y_Hermiticity|1.1208655e-16|5.0619504e-16|PASS|

結果はPASSであり、旧残差は単なる保存normの丸めで消えるものではなかった。extended subtractionは以前のbinary64 simulationが生んだ誤差を回復しない。厳密区間oracleでもなく、元H ground vectorも使っていない。全88標本を再シミュレーションしていない。

## 人工経路比較と技術改善の成否

既存exact cache/実class/control state/binary64角度、actual inverse/global phase、Reference state-action batchの回帰を再利用した。新C7-4定義/native比較7件PASSと、旧cache6＋handoff5の不変回帰11件PASS、distinct18件。旧guard/17qubit全Operator禁止を維持。

固定3/5qubit blockの1024/8192/32768反復を、generic局所行列変換とnative rz/sx/x/cxを保持する別revisionで比較した。参照は短いQiskit/binary64 blockをlongdoubleで累乗したもので、短block誤差は反復される。厳密な高精度解/誤差上界とは扱わない。

|width|repeats|operations|generic max|generic Frob|native Frob|generic-native max|
|---|---|---|---|---|---|---|
|3|1024|11264|2.5909787e-14|1.0291e-13|1.0291e-13|0|
|3|8192|90112|2.4419538e-13|7.7946452e-13|7.7946452e-13|0|
|3|32768|360448|8.6615332e-13|3.0966499e-12|3.0966499e-12|0|
|5|1024|17408|1.1533055e-14|1.5741211e-13|1.5741211e-13|0|
|5|8192|139264|1.1625503e-13|1.2260737e-12|1.2260737e-12|0|
|5|32768|557056|5.6617164e-13|4.8774452e-12|4.8774452e-12|0|

6件ともcomponentと記録したfull-matrix Frobenius診断は現行許容値内。nativeとgenericは今回すべて同じ値で、**精度改善は確認できなかった**。native版を「改善済み長q verifier」として採択しない。cacheの共用は同じ公式局所行列を再利用し、この人工比較と既存回帰で適合したが、全support誤差boundではない。3/5qubit短blockのP0 compileは計2本だけで、raw/compiled全行列は許容値内。人工gate数が多いこと自体をH4/17qubit/256入力列の長q資格へ転用しない。

## 条件付きの誤差会計

理想unitary Ujに対する各実装誤差のoperator normをεjとし、その中へ局所行列・gate適用・演算順序・compiler/宣言位相の差を重複なく含められる場合、積の誤差は ∥ŨL…Ũ1−UL…U1∥ ≤ ∏j(1+εj)−1 ≤ exp(Σj εj)−1 と評価できる。これは各εjの一様な上界を別途得るという条件付きであり、machine epsilonをそのままεjへ代入した実H4 certificateではない。誤差が独立と仮定して√L則へ置換しない。

入力isometry Kとその近似K̃、測定Z（norm1）について、observable K†ZKの差は ∥K̃†ZK̃−K†ZK∥F ≤ (∥K̃∥2+∥K∥2)∥K̃−K∥F。全256入力列の誤差会計が必要で、単一のmax-entry残差が小さいだけでは同じFrobenius余裕を保証しない。Reference leakageもK̃の独立な成分として保持する。保存配列のextended再集約はこの後段の差/積の丸めを減らす候補だが、前段gate適用の誤差を取り消さない。

現在は全supportで有効なεj、Gaussian/actual inverse/位相の一様包絡、compiled長gate列の累積boundが未取得である。したがってこれらの式は次のverifier設計の条件整理に留まり、q4096の実装誤差や同じNの保証値を新規算定していない。

## q≥64前のpreflightと失敗時の対処

1. 新SOURCEでP0/入力law/ordered規約/actual inverse/χと固定精度を維持して人工回帰を通す。演算順序/phase扱い/extended集約を変える場合は旧結果を保持し別revisionへ固定する。
2. 次の承認済みdomainの先頭固定pathをcohortの一部としてraw→4 wrapper→postcheckし、CAR/raw/paired X/Y/全256列/reference leakageを照合。失敗時に除外/再抽選しない。
3. threshold近傍・超過なら依存する分子実行をSTOP。normの再計算だけでPASS化せず、誤差をcompiler位相・短gate列/小演算子・state-action/Gramへ分ける限定診断を次のSOURCEに設計する。必要ならより高精度のgate/Gram/ordered積経路を検証するが、真の相違をpolar補正/renormalization/phase fitで消さない。
4. 任意入力の丸め誤差包絡と、17qubit Reference経路の精度改善は未取得。q32→64の線形誤差スケールも証明されていないため、**現行閾値を長qで維持できるとの事前GOは出せない**。

本C7-4の数値比較は完了。q≥64の実H4計算と、旧失敗を上書きする再計算は実行していない。次の科学実行は[条件案](NEXT_EXECUTION_DRAFT.md)をユーザー/GPTが採択・承認してから行う。
