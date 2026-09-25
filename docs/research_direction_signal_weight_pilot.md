# P-B：energy biasとtarget weight／signalの判別pilot

最終更新：2026-09-25 JST

## 目的

研究テーマ再設計のP-Bとして、PFのtarget固有位相biasだけで候補を選ぶと、target weightまたは
coherent signalの観点で不適格な候補を選ぶ例が現行の小系証拠に存在するかを確認した。
論文用の一般結論を作る検証ではなく、案Bを主研究へ進める根拠があるかを安価に判別するpilotである。

## 固定範囲

- H4 linear chain、1.0 Å、STO-3G、8 qubits、DF rank 12
- 二次partial-$S_2$、$L_D=0,\ldots,11$
- $\delta=0.0125,0.025,0.05,0.1,0.2,0.4$
- $q=1,2,4$
- 既存`pf_delta_validation_v5`の12 artifact、計72候補
- 新しいHamiltonian計算、PF対角化、回路compileは行わない

各候補について、target-centered PF固有位相bias、支配分枝weight、q別物理signal半径・位相汚染を
再集計した。将来の安価な診断候補として、q=1のstate actionだけから

$$
d_{\perp}=\sqrt{1-|\langle\psi|S(\delta)|\psi\rangle|^2}
$$

も記録した。ただし今回のgridには実用的なsignal failureがないため、この診断を選択器として
検証済みとは扱わない。

## 事前screening規則

- target weight $w_0\geq0.995$
- 全$q$でsignal半径$|Z_q|\geq0.99$
- 各deltaで、energy-only選択が上記を外れ、別の候補が通る場合を選択不一致とする
- biasが最良値の5%以内の候補について、leakageが2倍以上減り、かつ最小signal半径が
  $10^{-3}$以上改善する場合を意味のあるnear-tie差とする

これはP-Bのscreening規則であり、普遍的なQPE受理基準ではない。

## 結果

- 最小target weight：`0.9999803238781496`
- 最小q別signal半径：`0.999974490356076`
- 各deltaのenergy-only最良候補：すべて`L_D=0`
- signal screening後のenergy最良候補：すべて同じ`L_D=0`
- energy/signal選択不一致：`0/6`
- 数値上のpairwise ordering inversion：118組
- 意味のあるpairwise inversion：0組
- 意味のあるnear-tie signal advantage：0件

小さなordering inversion自体は存在するが、最大leakage比は約1.0035であり、候補選択や
signal使用可否を変える大きさではない。全72候補がscreeningを通過した。

## 判断

現行H4・二次partial-$S_2$・prefix分割gridでは、energy-only選択とsignal-aware選択の
実質的な食い違いは得られなかった。従ってこの範囲を根拠に案Bを主研究へ進めず、P-Bの
停止条件を満たしたものとして次のP-Cへ進む。

これは、他のPF family、小gap系、別geometry・別分子でweight問題が存在しないという
一般的棄却ではない。また、state-action診断の選択能力も検証済みではない。将来、別の研究から
自然にsignal failure候補が現れた場合は安全診断として再利用できる。

## 証拠

- artifact：`artifacts/research_direction_signal_weight_pilot/2026-09-25/pb_h4_s2_prefix_grid_v1.json`
- implementation：`src/trotterlib/research_direction_signal_weight_pilot.py`
- runner：`scripts/run_research_direction_signal_weight_pilot.py`
- test：`tests/test_research_direction_signal_weight_pilot.py`

artifactはlocal worktree解析であり、外部再現またはimmutable CI evidenceではない。最終RPE総cost、
部分ランダム化の科学的優位性、H12、backend/noiseは評価していない。
