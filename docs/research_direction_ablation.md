# 研究方向screening WP04：有限RTE・配分・scheduleの寄与分解

最終更新：2026-09-21 JST

## 位置づけ

WP04は、WP01-Sで比較した部分ランダム化候補と決定論endpointについて、見かけの
cost差がround別有限RTE schedule、位相誤差予算$\beta$、失敗確率$\alpha$、または
schedule選択に使うcost providerのどこから生じるかを分ける検証である。

これはH4小系の`model_conditional_screening`であり、最終総cost評価、部分ランダム化の
優位性判定、またはdecision-grade比較ではない。

## 固定条件

| 項目 | 条件 |
|---|---|
| model | linear H4、原子間距離1.0 Å、STO-3G、8 qubit |
| DF | rank 12、固定snapshot hash `56e4df83...31e5` |
| 候補 | 中間分割$L_D=3$、tailなし決定論endpoint $L_D=12$ |
| task | CA/10、$\delta=0.02$、$M=17$、$q_m=2^m$、$q_{\max}=131072$ |
| RPE | $\beta_{\rm RPE}=0.4$、$\alpha_{\rm total}=0.05$ |
| PF入力 | 同じsnapshotで計算した論文Eq. (D6)の経験的係数 |
| 回路scope | 状態準備を除くHadamard interrogation |
| cost | WP01-Sの$q=1,2$直接較正から得た軸別affine RZ点推定 |
| compile | Qiskit 1.3.0、`rz,sx,x,cx`、optimization level 1、seed 17、coupling mapなし |

$q>2$のschedule別回路には未使用holdoutがなく、affine外挿である。従って、以下のRZ値は
WP04の条件付き集計値であり、最終総costとは呼ばない。

## 比較設計

### schedule

1. 全roundで一つの$(r,K)$を使い、RZ点推定を最小化する固定schedule
2. 各roundでshot重み付き成分作用数を最小化するschedule
3. 各roundで$q=1,2$ affine RZ点推定を最小化するschedule

部分ランダム化の候補集合はWP01-Sで直接較正した7種類、決定論endpointはtailなしの
$(r,K)=(0,0)$である。成分作用数とcompiled RZは別の目的関数として扱った。

### 位相誤差予算

| label | $(\beta_{\rm PF},\beta_{\rm RTE},\beta_{\rm stat})$ | 用途 |
|---|---:|---|
| legacy | $(0.08,0.08,0.24)$ | 旧基準 |
| guarded | $(0.02,0.02,0.36)$ | 既存短round感度検証の暫定値 |
| rebalanced common | $(0.015,0.005,0.38)$ | 両候補に共通 |
| deterministic rebalanced | $(0.015,0,0.385)$ | tailなしendpointで不要なRTE予算を統計へ戻す |

これは事前に固定した診断gridであり、連続的な大域最適化ではない。決定論endpointにも、
適用可能な予算の再配分と$\alpha$最適化を与えた。

### 失敗確率配分

- 一様：36 round-axisへ$0.05/36$を配分
- cost感度重み：固定scheduleでの連続緩和
  $a_i\log(2/\alpha_i)$から$\alpha_i\propto a_i$を反復適用

離散的なschedule切替を含むため、反復の絶対収束許容値は$10^{-8}$とした。全42 factorial
cellがこの条件を満たした。

## 結果

### 固定順の逐次ablation

| 候補 | 段階 | RZ点推定 | 直前からの削減 |
|---|---|---:|---:|
| $L_D=3$ | 固定schedule・legacy $\beta$・一様$\alpha$ | $5.9516\times10^{12}$ | -- |
|  | 成分作用数でround別schedule | $6.7662\times10^{12}$ | $-13.69\%$ |
|  | common $\beta$再配分 | $2.7798\times10^{12}$ | $58.92\%$ |
|  | cost感度重み$\alpha$ | $2.1270\times10^{12}$ | $23.48\%$ |
|  | compiled RZでschedule選択 | $1.7848\times10^{12}$ | $16.09\%$ |
| $L_D=12$ | 固定schedule・legacy $\beta$・一様$\alpha$ | $5.3316\times10^{12}$ | -- |
|  | round別schedule | $5.3316\times10^{12}$ | $0\%$ |
|  | tailなし用$\beta$再配分 | $2.1430\times10^{12}$ | $59.81\%$ |
|  | cost感度重み$\alpha$ | $1.6963\times10^{12}$ | $20.84\%$ |
|  | compiled RZでschedule選択 | $1.6963\times10^{12}$ | $0\%$ |

表の負の削減はcost増加を表す。$L_D=3$では成分作用数を最小化するround scheduleが、
RZを固定schedule比16.87%増やした。一方、compiled RZを目的に選ぶround scheduleは
同じ完全設定の固定schedule比1.93%減らした。したがって、schedule改善とcost providerの
変更を同じ寄与として扱わない。

### leave-one-outと交互作用

完全設定から一因子だけ戻したときのRZ増加は次の通りだった。

| 戻した因子 | $L_D=3$ | $L_D=12$ |
|---|---:|---:|
| round別schedule | $1.97\%$ | $0\%$ |
| $\beta$再配分 | $143.19\%$ | $149.61\%$ |
| $\alpha$再配分 | $35.44\%$ | $26.33\%$ |
| compiled-costに整合した選択 | $19.17\%$ | $0\%$ |

$\beta$再配分後は$\alpha$再配分の絶対利得が小さくなる。加法interactionは
$L_D=3$で$-9.076\times10^{11}$ RZ、$L_D=12$で$-6.507\times10^{11}$ RZだった。
逐次差分だけで各因子を独立な削減率として加算してはいけない。

### 完全設定での候補比較

| 候補 | RZ点推定 | local 5%＋較正区間 | transfer 25%＋較正区間 |
|---|---:|---:|---:|
| $L_D=3$ | $1.7848\times10^{12}$ | $[1.4978,2.0718]\times10^{12}$ | $[1.1408,2.4288]\times10^{12}$ |
| $L_D=12$ | $1.6963\times10^{12}$ | $[1.6115,1.7811]\times10^{12}$ | $[1.2722,2.1204]\times10^{12}$ |

点推定では$L_D=12$が4.96%低い。しかし5%区間、25%移送区間とも重なるため、方向判定は
`undetermined_between_intermediate_and_deterministic_endpoint`とする。WP01-Sの21.0%差が
縮んだ主因は、両候補へ公平に$\beta$・$\alpha$再配分を適用したことである。

### 信号、有限RTE bound、統計bound

$L_D=3$の完全設定scheduleは

- $q=1,2$で$(r,K)=(1,2)$
- $q=4,\ldots,1024$で$(2,2)$
- $q=2048$で$(4,2)$、$q=4096$で$(8,2)$
- $q=8192,16384,32768$で$(16,2)$
- $q=65536,131072$で$(32,2)$

となった。全選択点を含むsector行列gridでoperator・signal・半径等の既存validatorを通過した。
物理基底状態の最小観測半径は0.5727013937、最小の保守半径下界は0.5727013920で、
全点で下界を満たした。$L_D=12$のtailなし信号は最小半径0.9999999998、unitary defectは
$9.12\times10^{-15}$だった。

実観測半径をHoeffding式へ代入しても、保守半径下界とのRZ差は丸め誤差以下だった。
整数shotのceil overheadは$L_D=3$で0.171%、$L_D=12$で0.164%である。一方、既知の小系信号を
使った軸別厳密二項counterfactualはHoeffding点推定よりそれぞれ53.0%、68.2%低かった。
これは真の軸平均へ依存する診断であり、全RPE分枝復元を置き換える採用costではない。

### round支配度

完全設定の最終roundが占めるRZ割合は$L_D=3$で60.5%、$L_D=12$で43.9%、最後の3 roundは
それぞれ91.9%、83.0%だった。長$q$ holdoutとfull-scope接続は後半roundを優先すべきである。

## 判断

1. WP04で確認した最大の共通因子は、有限RTE固有のschedule改善ではなく$\beta$、次いで
   $\alpha$の再配分である。
2. $L_D=3$のcompiled-costに整合したround別scheduleには小さい追加利得があるが、完全設定の
   固定schedule比1.93%である。
3. 成分作用数proxyだけでscheduleを選ぶとRZ方向を誤る場合がある。
4. 公平な再配分後も$L_D=3$対12は不確かさ区間内で未決定であり、部分ランダム化の優位性を
   主張しない。
5. 次は予定通りWP03でPF係数だけを差し替え、候補順位とregretの変化を調べる。

## 成果物と再実行

- 主artifact：
  `artifacts/research_direction_ablation/2026-09-21/wp04_finite_rte_statistical_ablation_v1.json`
  （fingerprint `b7493a86...a6ab9`）
- $L_D=3$行列grid：
  `artifacts/research_direction_ablation/2026-09-21/wp04_ld3_full_schedule_signal_grid_v1.json`
  （validation fingerprint `7a723303...10cdb`）

```bash
.venv311/bin/python scripts/run_research_direction_ablation.py
.venv311/bin/python -m pytest -q tests/test_research_direction_ablation.py
```

両artifactはlocal dirty-worktree evidenceであり、immutable CIまたは外部再現結果ではない。
