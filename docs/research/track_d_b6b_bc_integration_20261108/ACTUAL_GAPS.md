# B6b：actual保証へ残る条件（今回は算定しない）

[B5固定版](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0a8b953aea1be20b110784960763d7f36ebccac8/docs/research/track_d_b5_actual_rpe_static_survey_20261106/REPORT.md)の三つの不足、implemented law、各軸shot列の統計条件、全使用path測定実装誤差を維持する。B6bの理想同値性の導出と、actual認定は別である。

## 誤差接続と二重計上防止

方式aについて

\[
E_{Q_m}z_{a,impl}\to E_{Q_m}z_{ideal}\to E_{P_m}z_{ideal}\to A_m e^{-it_mE_0},
\quad d_{a,m}=A_m\epsilon_m^{B1}+e^{raw}_{impl,a,m}+e^{raw}_{law,m}.
\]

|項|単位/出所|未資格・重複防止|
|---|---|---|
|ε_B1|同sectorのnormalization-corrected operator誤差。元H packet/CDFのDuhamel輸送＋PF＋有限RTEを含む保存13行|数値norm＋guardの条件を継承。DF reserve/Weylをenergyへ再加算しない|
|A_m ε_B1|raw complex-signal単位。Aは保存handoff/normalization|Aを一回だけ掛ける。価格には掛けない|
|e_law raw|ideal Pとfinite-bit Qの同じideal path信号の平均bias|B1有限Taylor truncationと別。unknown=null。TV定義sup_Sなら≤2TVは候補で、数値TVを取得したものではない|
|e_impl raw|Q平均での実装信号差、または全path一様bound|P下の平均だけをQへ流用しない。max-entry residualをoperator normへ同一視しない|
|χ/actual adjoint/leakage|同じreference初期化でのisometry/observable差|full-wrapper observable boundが包含するなら追加で重ねない|
|shot joint-law|濃度不等式・全transcript lawの条件|平均biasのboundだけではconcentrationを保証しない|

追加raw誤差へAを再度掛けない。全26軸のα配分和は既に.01であり、未取得TV/transcript失敗分を正の余裕として勝手に加えない。各roundのphase slackだけでは、半径縮小に伴うcoordinate許容幅と旧整数Nの適否を保証できない。

## 何をどこから得るか

1. **finite-bit categorical law**：固定216-atom/K0、Fraction係数→float/再正規化→NumPy choiceの閾値・有限uniform格子、bit generatorのSOURCE。law JSON hashはidentityだけで、距離0の証明ではない。固定seed列のPCG64と独立uniform bitsの仮定を分ける。
2. **joint law**：event/occurrence/shotの階層と、各軸内のfresh状態/trajectory。seed非重複だけでIIDにしない。round/axis間独立はunion bound自体には不要。price-only cohortは量子shot transcriptではない。
3. **native/raw実装誤差**：Gaussian分解、signed diagonal、scalar/identity、actual inverse、χ、gate適用・roundingを同じoperator/isometry/observable単位へ結ぶ。exact nativeは理想laneの定義で、binary64実装の誤差0認定ではない。
4. **P0/長q**：合法融合の対象、declared phase保持、compiled observable、全256入力/全reference出力のleakage。局所辞書＋telescopingによる一様伝播は取得候補で、未検証kernelの単純反復や有限postcheck最大値で全support boundを置き換えない。
5. **共通N**：同じlawとsignal-budgetに両方式を包含できればcommon最悪評価または旧整数Nの適否検査が候補。方式別law/誤差が違う、片方式だけcoordinate条件を外れる等なら方式別Nを同じtarget/failure配分で評価する。今回は再評価しない。

isometry差η（適切なnormと正規化前提）からobservable差2η、X/Yのaxis bound κ_X/κ_Yからcomplex raw差√(κ_X²+κ_Y²)へ結ぶ標準的な候補はB5と整合する。χだけなら|e^{iδχ}−1|≤|δχ|。これらを数値H4へ適用した新boundは未取得。B6bの弱いV-ray例は理想信号を保存しても最終reference非真空となるため、既存DFのzero-leakage checkerを緩める根拠にはしない。

## 理想会計との独立性

上記のactual不足を残したまま、明示したideal law/exact native/B4-2前提と、採択した条件付き価格モデルで資源の点推定を行うことはできる。そこでactual失敗確率やunconditional母平均保証まで主張しない。元H厳密groundの仮定はNの対象stateを固定するためで、B6bの任意state回路恒等式がstate準備やenergy推定の実装資格を保証するものではない。

今回はraw/law/実装誤差・正半径・新N・actual成功率の分子数値算定0。noise/routing/T合成も対象外。必要なactual資格は、条件付き理想資源評価とは別の次段階としてユーザー/GPTが優先順位と実行範囲を判断する。
