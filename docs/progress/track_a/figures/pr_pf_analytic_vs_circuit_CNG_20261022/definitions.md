## 定義とN監査

|量|解析側|回路側|今回の扱い|
|---|---|---|---|
|C_a|NATIVE_MIXED / explicit_rz_sx_x_cx_v1の法則期待RZ数|PRの保存compiled平均／PFの決定論計数|cos/sin別に比較。理論lower/upperを保持|
|N_a|保存bias・normalization・経験的allowanceをHoeffding十分条件へ代入|同じ推定器・精度・失敗率のもとでcompiled信号から独立評価する量|未取得null。実機の実測必要shot数ではない|
|G00|Σ N_theory,a C_theory,a|—|native費用モデル＋数値信号入力によるhybrid会計|
|G01|—|Σ N_theory,a C_circuit,a|**N固定・Cのみ置換した総費用比較**|
|G10 / G11|Σ N_circuit,a C_theory,a|Σ N_circuit,a C_circuit,a|独立N未取得のため両方null|

N_theoryは純粋に解析式だけから得た値ではない。corrected平均信号とfull-H参照との差を古典数値評価し、経験的u、finite-RTE正規化、α_cos=α_sin=.025、各軸ε/√2を使う。Nは必要最小回数でなく十分条件に基づく配分。補正付き独立有界±1測定平均にHoeffdingを適用する。PRでは量子shotごとにfresh IID RTE抽選を仮定する。

すべての抽選回路で同じ理想測定法則を実装し、推定器・bias・u・B・αを同じにすれば、同じ式は同じ十分Nを返す。これは条件付きの数学的帰結で、保存compiled費用だけからNを独立取得した証拠ではない。代表状態ベクトルの1e-7検査だけを、全抽選列のbias・正規化・数値誤差を保証する証明へ格上げしない。独立Nを得るにはcompiled平均信号または全法則に対する同値性／誤差移送と、同じ保証強度の数値誤差予算が必要。分散適応や別の統計保証を採用する計算は行っていない。

[N比較可能性監査](N_comparability_audit.json)に6項目のpreflightと不足情報を記録した。新推定器の科学的採用は未承認。今回のN図は取得済みshotの情報図であり、独立比較図ではない。

