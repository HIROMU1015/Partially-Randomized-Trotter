# 研究方向screening WP03：PF係数選択とcandidate regretの感度

最終更新：2026-09-22 JST

## 位置づけ

WP03は、WP04のtask、候補、compiled-cost provider、round別schedule選択、$\beta$・$\alpha$配分を
固定し、Product Formula（PF）係数だけを差し替えたときに、最適$L_D$、$\delta$、candidate
regretが変わるかを確認するGate S1最後の検証である。

これはH4小系の`model_conditional_screening`であり、PF係数の厳密認証、最終総cost評価、または
部分ランダム化の優位性判定ではない。

## 固定条件

| 項目 | 条件 |
|---|---|
| model | linear H4、原子間距離1.0 Å、STO-3G、8 qubit |
| DF | rank 12、固定snapshot hash `56e4df83...31e5` |
| 係数監査 | $L_D=0,3,12$ |
| cost比較 | WP01-Sでcost証拠を作った$L_D=3,12$ |
| task | CA/10、$\delta\in\{0.01,0.0125,0.02\}$ |
| 配分 | WP04完全設定。$L_D=3$は$(0.015,0.005,0.38)$、$L_D=12$は$(0.015,0,0.385)$ |
| schedule | round別compiled RZ最小、cost感度重み$\alpha$ |
| cost | 状態準備なしHadamard、WP01-Sの$q=1,2$直接較正から軸別affine外挿 |

$L_D=0$はWP01-Sの拡張成分作用数proxyでscreen out済みであり、同じscopeのcompiled-cost証拠を
作っていない。このため係数の反例監査には残すが、WP03のcost順位へ戻していない。

## 比較した係数

| $L_D$ | $C_D$ | 論文D6 | 支配固有位相 | $C_D$/D6差 | D6/固有位相差 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0.0115338 | 0.0114931 | $-100\%$ | $+0.354\%$ |
| 3 | 0.0117236 | 0.0133991 | 0.0133569 | $-12.50\%$ | $+0.316\%$ |
| 12 | 0.0134411 | 0.0134257 | 0.0133833 | $+0.115\%$ | $+0.317\%$ |

- $C_D$は$H_D$だけの安価なscreening surrogateであり、厳密上界ではない。
- 論文D6はfull-$H$基底状態を用いる大規模系向けの現行推定器である。
- 支配固有位相係数は小系の主基準値である。

$C_D$もD6・支配固有位相と同じconditioned窓$\delta=0.05,0.1,0.2,0.4$で再fitした。既存artifactの
元の$C_D$ fit窓0.01--0.08と混ぜて比較していない。元の係数は$L_D=3$で0.0116395、$L_D=12$で
0.0133696だったが、WP03の係数policyには上表のcommon-window値を使用した。

$L_D=0$では$C_D=0$でもfull partial-$S_2$係数は非零である。$L_D=3$でも$C_D$はD6を12.50%
下回るため、$C_D$だけでPF予算または最終候補を決めない既存方針を支持する。一方、cost比較した
$L_D=3,12$のD6と支配固有位相の係数差は最大0.317%だった。

### D6 conditioningと符号

両候補ともD6は$\delta=0.05,0.1,0.2,0.4$の4点で$|\sin(E_0\delta)|\geq0.1$を満たし、
$0.0125,0.025$の2点はconditioning不足としてfitから除外した。conditioned点におけるD6と
支配固有位相の最大点差は$L_D=3$で0.781%、$L_D=12$で0.781%だった。

保存artifactでは$C_D$と支配固有位相のsigned biasが正、論文D6のsigned biasが負である。
これは記録された定義の符号規約が直接揃っていないことを示すため、係数比較は絶対値で行った。
符号の違いを交換子項の物理的相殺とは解釈しない。また、現artifactは個々のmixed/tail交換子を
分解していないので、$C_D$からfull partial係数への差は省略された効果の合計としてのみ扱う。

## 選択感度

3係数×2候補×3つの$\delta$、計18条件は全てPF予算と全roundの実行可能性を満たした。

| 係数 | 最良候補 | RZ点推定 | 選択変化 |
|---|---|---:|:---:|
| $C_D$ | $L_D=12,\delta=0.02$ | $1.6963\times10^{12}$ | なし |
| 論文D6 | $L_D=12,\delta=0.02$ | $1.6963\times10^{12}$ | なし |
| 支配固有位相 | $L_D=12,\delta=0.02$ | $1.6963\times10^{12}$ | なし |

係数はPF予算の実行可能性を変える入力であり、予算内では固定したshot式とcost providerの値を
直接変更しない。今回の係数群では全候補が同じ実行可能領域に残ったため、cost値と選択も同じに
なった。

### 論文D6でのcandidate regret

| 候補 | $\delta$ | RZ点推定 | 最良候補比regret |
|---|---:|---:|---:|
| $L_D=3$ | 0.01 | $2.3358\times10^{12}$ | 37.70% |
| $L_D=3$ | 0.0125 | $2.9599\times10^{12}$ | 74.49% |
| $L_D=3$ | 0.02 | $1.7848\times10^{12}$ | 5.22% |
| $L_D=12$ | 0.01 | $3.3927\times10^{12}$ | 100.00% |
| $L_D=12$ | 0.0125 | $3.3927\times10^{12}$ | 100.00% |
| $L_D=12$ | 0.02 | $1.6963\times10^{12}$ | 0% |

$L_D=3,\delta=0.02$を選んだ場合の点regretは5.216%である。ただしWP04と同じscenarioを伝播すると、
$L_D=3$対12の相対差区間はlocal 5%＋較正で$[-15.91\%,28.57\%]$、25%移送＋較正で
$[-46.20\%,90.91\%]$となり、どちらも0を跨ぐ。したがって、この5.216%を確定した科学的regret
または決定論endpointの優位性とは呼ばない。

全係数policyの選択を別のpolicyで再評価するcross-policy regretは全て0だった。今回比較した
係数定義の選択差より、長$q$ cost移送scenarioの不確かさの方が候補判断を支配している。

### PF予算境界までの余裕

$\delta=0.02$で論文D6を使った最大PF位相proxyは$L_D=3$で0.0140500、$L_D=12$で
0.0140778であり、予算0.015を満たした。同じ$\delta$が不適格になるまでの係数増加余裕は
それぞれ6.76%、6.55%である。D6と支配固有位相の差0.317%はこの余裕より小さい。

従って既存の二推定器間差では選択は安定だが、$\delta=0.02$が係数に無制限に頑健という意味では
ない。係数が約6.6%以上上振れする別instance・別fitでは、round horizonと$\delta$選択を再評価する。

## 判断

1. 現在の同一snapshotでD6と支配固有位相を選び替えても、最適$L_D$、$\delta$、点regretは変わらない。
2. $C_D$は$L_D=0,3$でfull partial係数を大きく過小評価するため、広い候補screening専用とし、
   shortlist後はD6を候補ごとに計算する現方針を維持する。
3. PF係数選択はWP04の候補区間重なりを解消しない。方向判定は引き続き`undetermined`である。
4. Gate S1に必要なWP01-S、WP02、WP04、WP03が揃った。次は追加計算ではなく、まずGate S1の
   研究方向判断をまとめる。

## 成果物と再実行

- 主artifact：
  `artifacts/research_direction_pf_sensitivity/2026-09-22/`
  `wp03_pf_coefficient_selection_sensitivity_v1.json`
- fingerprint：`2faa8d63...e4ff3`

```bash
.venv311/bin/python scripts/run_research_direction_pf_sensitivity.py
.venv311/bin/python -m pytest -q tests/test_research_direction_pf_sensitivity.py
```

artifactはlocal dirty-worktree evidenceであり、immutable CIまたは外部再現結果ではない。
