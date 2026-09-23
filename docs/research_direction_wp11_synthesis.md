# WP11 限定判断統合

## 1. 目的

WP00からN07/P03までの実施済み部分を研究方向へ戻し、T1--T7を継続・保留・範囲変更へ
分類する。未実施項目を失敗とは扱わず、次の重い検証を一件だけ選ぶ。

本統合では新しい回路compile、statevector計算、状態準備計測を行っていない。11個の
fingerprint済みlocal artifactを検証・再集計した。

## 2. 固定範囲

- H4 linear chain、1.0 Å、STO-3G、8 qubit、DF rank 12
- `CA/10`、候補`L_D=3,12`、主比較`delta_time=0.02`
- cost指標はcomplete controlled Hadamard wrapperのcompiled RZ
- topology-free、`rz/sx/x/cx`、seed 17
- 現在の結論はH4、状態準備なし、local dirty-worktreeの範囲に限る

## 3. 証拠の進展

1. Gate S1ではcost区間を同点でなく未決定とし、T4/T7を主軸にした。
2. WP06-a/bではsupport限定basisが常に有利ではないことを確認し、singleton runだけを置換する
   sequence-aware policyを固定した。holdout RZは10.67%減り、additive bridgeの点順位が反転した。
3. WP05-a/b/Rではfull controlled wrapperへ接続した。初回q=8 triggerを消さずに保持し、独立32
   trajectory再検証ではselected RZ誤差0.52%で通過した。
4. WP01-D/C07ではalpha、shot、scheduleを候補ごとに再最適化し、状態準備なしのopt1点推定で
   `L_D=3`が13.92%低かった。
5. G08/M08では後半3 roundが`L_D=3` costの90.94%を占め、q=16/32 holdoutが通過した。ただし
   q>32移送は未測定である。
6. M06/L08ではopt2 q<=32 proxyが通過したが、直接opt2 domainは`L_D=3,r=32`だけで、focused
   実測幅区間は重なった。
7. N07/P03では`L_D=3`が2,376 shot多く、正の共通状態準備costがその点利得を縮めることを確認した。

## 4. 研究方向の判断

| 方向 | WP11判断 | 現在の位置づけ | 判断が変わる条件 |
|---|---|---|---|
| T1 優位領域 | 範囲変更 | 一般的優位性でなく、H4条件付き成立限界と不安定化要因を扱う | coherent compiler contextで区間が分離する |
| T2 PF誤差・分割 | 限定継続 | D6対支配位相係数は現H4 shortlistを変えない | PF境界接近、新instanceで順位変化 |
| T3 高次PF | 保留 | 現bottleneckは二次PF係数でなくcompiled-cost context | PF誤差またはround depthが支配要因になる |
| T4 cost予測 | 主軸継続 | sequence、scope、compiler domainを明示したproxy検証が中心 | coherent opt2でも判断に寄与しない |
| T5 schedule・配分 | 限定継続 | beta、alpha、shot、scheduleを次段でも共同再最適化する | opt2で選択rまたはdelta境界が変わる |
| T6 表現・sampling | 限定継続 | 固定済みsequence-aware policyを入力として維持する | opt2 holdoutまたは外部instanceでpolicyが破れる |
| T7 信頼性・否定的結果 | 主軸継続 | 簡略化したscreeningが不安定な判断を生む条件を成果とする | 結果が一つのcode固有bugだけへ縮退する |

これは外部文献に対する新規性を確定した表ではない。既存研究との差は、現在のproject内で
支持できる貢献形態を限定したものであり、別途literature novelty reviewが必要である。

## 5. 次に選ぶ一件

`M06-F all-r coherent optimization-level-2 reoptimization`を選ぶ。

### 新規計算範囲

- `L_D=3`のopt2未測定`r=1,2,4,8,16`
- 各rのq=1,2較正
- 各rで未使用q=4またはq=8を事前固定holdoutとして使用
- 既存の`r=32,q=1,2,16,32` opt2証拠とdeterministic `L_D=12` opt2 providerは、条件とhashが
  一致する場合に再利用
- q>32直接compileは第一batchに含めない

### 解析範囲

1. 全選択rについてopt2 cost proxyを作る。
2. `L_D=3,12`、`delta=0.01,0.02`のbeta、alpha、shot、scheduleを同一compiler contextで再最適化する。
3. per-r実測proxy discrepancyを伝播し、異なるcompiler contextを平均しない。
4. coherent結果でN07/P03の区間と状態準備break-evenを再計算する。

この次段はQiskit transpile中心なのでGPUは不要であり、独立trajectory・rのCPU並列が適する。

### 終了・分岐条件

- 実測幅区間がなお重なる、または狭い未測定compiler domainを残さず`L_D=12`が低くなる場合は、
  local compiler精密化を止める。
- coherent opt2候補が残り、外部移送を試す科学的価値がある場合だけ外部instance pilotへ進む。
- q>32移送だけが順位を変え得る残存要因になった場合だけ、長q holdoutを検討する。
- この次段だけから科学的優位性または最終総costを主張しない。

## 6. 選ばなかった候補

- 外部instance pilot：棄却ではなく延期。現在実行するとopt1/opt2混在を持ち込み、順位変化の原因を
  物理instanceとcompiler contextへ分離できない。
- 状態準備計測：具体的な準備回路が未固定で、opt2 focused区間はP=0ですでに重なる。候補別回路が
  固定されるか、現実的costが47,067,344 RZ相当/shotへ近づく場合に再開する。
- q>32追加holdout：M08の事前q=64 triggerは発火していない。まず既知のall-r opt2欠落を埋める。

## 7. 現在の結論と限界

状態準備なしの点推定候補は`L_D=3`だが、compiler、長q、状態準備を含む頑健な区間優位性は
確立していない。判定は
`undetermined_under_compiler_transfer_and_preparation_sensitivity`のままである。

H12、外部instance、q>32 opt2直接検証、状態準備、coupling map、backend/noise、fault-tolerant
synthesis、最終総costは本統合の検証範囲外である。

## 8. 証拠

- artifact：
  `artifacts/research_direction_wp11_synthesis/2026-09-23/wp11_scoped_direction_synthesis_v1.json`
  （fingerprint `45def7f696eddba574878cc7530837dfdfc5c6e9c2767ee3e115cf4a5f1ac092`）
- 実装：`src/trotterlib/research_direction_wp11_synthesis.py`
- runner：`scripts/run_research_direction_wp11_synthesis.py`
- test：`tests/test_research_direction_wp11_synthesis.py`
- 専用test：`3 passed`
- 変更後のlocal全suite：`557 passed, 4 warnings`（warningは既存grouped-UWC test由来）

証拠statusはlocal dirty-worktreeであり、immutable CIまたは外部再現ではない。
