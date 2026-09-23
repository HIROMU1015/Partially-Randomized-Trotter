# WP01-D / C07 full-scope再最適化と判断区間

## 目的

WP05-bRまで通過したfull controlled Hadamard cost providerを使い、H4の暫定`CA/10` taskで
$L_D=3$と決定論endpoint $L_D=12$を公平に再集計した。各候補について$\delta$、
round別$(r_m,K_m)$、位相誤差予算$\beta$、軸・round別失敗確率$\alpha$、shot数を再最適化し、
全roundの状態準備なし期待RZ costと判断区間を作った。

これはH4、1 compiler、経験的PF係数、長$q$ affine proxyに条件付いた比較である。状態準備、
backend、noise、fault-tolerant synthesis、外部再現を含む最終科学的総costではない。

## 入力とcost provider

| 項目 | 条件 |
|---|---|
| 物理系 | H4直鎖、1.0 Å、STO-3G、8 qubit、DF rank 12 |
| task | WP00固定`CA/10`、$\beta_{\rm RPE}=0.4$、$\alpha_{\rm total}=0.05$ |
| 候補 | $L_D=3,12$、$\delta=0.01,0.02$ |
| $L_D=3$ provider | `support_run_le_1` full wrapper、$r=1,2,4,8,16,32$、$K=2$ |
| $L_D=12$ provider | tail-free deterministic full wrapper |
| 較正・holdout | $q=1,2$ affine。WP05-a/bの$q=4,8$、WP05-bRの$r=32,q=8$で検証 |
| 長round | $\delta=0.02$は$M=17,q_{\max}=131072$、0.01は$M=18,q_{\max}=262144$ |

$\delta=0.02,r=32$の$q=1,2$はWP05-bRの32標本へ置換し、それ以外はWP05-a/bの8標本を使った。
較正standard errorはround独立として縮小せず、shot重み付きの保守和として伝播した。

## 最適化

$L_D=3$は$\beta_{\rm PF}\times\beta_{\rm RTE}$の$32\times32$粗gridと
$25\times25$局所grid、$L_D=12$は対応する1次元gridを評価した。その後、上位20解を各scheduleの
実際のPF/RTE使用量まで反復的に締め、粗grid下端への依存を除いた。各点でcompiled RZを目的に
round scheduleを選び、cost感度重み$\alpha$を収束まで更新した。

4候補の全gridに実行可能解があり、選択解では$\alpha$が収束し、$\beta$和、PF/RTE予算、
全round feasibilityを満たした。

## 結果

両$L_D$で$\delta=0.02$が選ばれた。

| 候補 | no-prep RZ点推定 | local 5%＋較正区間 | transfer 25%＋較正区間 | shot数 |
|---|---:|---:|---:|---:|
| $L_D=3$ | $1.4557921\times10^{12}$ | $[1.306540,1.605045]\times10^{12}$ | $[1.015381,1.896203]\times10^{12}$ | 13,538 |
| $L_D=12$ | $1.6911234\times10^{12}$ | $[1.606567,1.775680]\times10^{12}$ | $[1.268343,2.113904]\times10^{12}$ | 11,162 |

$L_D=3$の点推定は$L_D=12$より13.916%低い。local 5%区間は分離するが、そのgapは
$1.523\times10^9$ RZ、$L_D=12$点推定の0.090%だけである。対称model discrepancyが
5.0484%を超えると区間は再び重なるため、採用5%との差は0.0484 percentage pointしかない。
25% transfer-sensitivity区間は重なる。

最終3 roundのcost割合は$L_D=3$で90.94%、$L_D=12$で83.03%だった。従って残るproxy精度の
検証資源はlate roundへ集中させる。

## 判断

- **local model条件付き**：5% discrepancy、同じH4・compiler・no-prep scopeでは
  $L_D=3$の区間が$L_D=12$より低い。
- **頑健性**：25%移送感度では区間が重なり、local分離の余裕も小さい。
- **研究上の結論**：点選好は$L_D=3$へ変わったが、部分ランダム化の頑健な科学的優位性は
  まだ確立しない。判定は`undetermined_under_transfer_sensitivity`とする。

後続G08/M08では支配的なlate roundを特定し、$q=16,32$の直接holdoutを実行した。selected RZ
2.466%と観測RZ最大3.286%による再集計区間はいずれも分離したが、直接domainは$q\leq32$で、
25%移送区間は重なる。従って頑健判定は変えず、追加のlocal $q$精密化より主張範囲の見直しまたは
外部移送検証を優先する。詳細は
[G08/M08後半round proxy精度](research_direction_late_round_proxy.md)に記録する。

## 成果物

- 最適化compute：
  `artifacts/research_direction_decision_cost/2026-09-22/wp01d_c07_full_scope_optimization_compute_v2.json`
  （fingerprint `709142f7...937474`）
- claim-scope synthesis：
  `artifacts/research_direction_decision_cost/2026-09-22/wp01d_c07_conditional_interval_synthesis_v1.json`
  （fingerprint `7d85b472...44250f`）
- runners：
  `scripts/run_research_direction_decision_cost.py`、
  `scripts/run_research_direction_decision_synthesis.py`
- tests：
  `tests/test_research_direction_decision_cost.py`、
  `tests/test_research_direction_decision_synthesis.py`

同じdirectoryに残る`wp01d_c07_full_scope_optimization_compute_v1.json`は、grid下端依存を検出した
予備診断であり、判断入力には使わない。現行computeは制約境界refinementを含むschema-v2である。

これらはlocal dirty-worktree evidenceであり、immutable CIまたは外部再現結果ではない。
専用testは合計`2 passed`、変更後のlocal全suiteは`530 passed, 4 warnings`だった。
warningは既存grouped-UWC test由来である。
