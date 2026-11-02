# Gate S1 研究方向判断

## 位置づけ

Gate S1は、WP00、WP02、WP01-S、WP04、WP03を統合し、次に減らすべき判断不確かさを一件へ絞る。
新しい物理計算や回路compileは行わず、fingerprint済み成果物の結論と適用範囲を機械可読な判断記録へ変換する。
したがって、これは研究経路の決定であり、最終総cost評価または部分ランダム化の優位性判定ではない。

対象はH4鎖、距離1.0 Å、STO-3G、8 qubit、DF rank 12、固定Hamiltonian hash
`56e4df83...31e5`、CA/10である。比較costは状態準備を含まない単一Hadamard interrogationで、
長い$q$は$q=1,2$の直接較正からのaffine外挿である。

## Gateの四つの問い

### 1. 候補cost区間は分離したか

分離していない。WP04完全設定では、$L_D=3$のRZ点推定は
$1.784816502056\times10^{12}$、$L_D=12$は$1.696328651024\times10^{12}$で、
決定論endpointが4.958%低い。しかしlocal 5%とtransfer 25%の両scenarioで区間が重なる。
WP03の論文D6係数でも相対差区間はそれぞれ$[-15.91\%,28.57\%]$と
$[-46.20\%,90.91\%]$で0を跨ぐ。よって判定は「同点」ではなく`undetermined`である。

### 2. 改善の主因はpartial randomization固有か

現時点ではそう示せない。WP04で両候補へ同等に適用できる変更を分離すると、最大の共通因子は
$\beta$再配分、次いで$\alpha$再配分だった。$L_D=3$のcompiled-RZ整合round scheduleは固定schedule比
1.93%減に留まり、成分作用数で選んだscheduleは逆にRZを16.87%増やした。したがって、観測された
大きな改善をpartial randomization固有の利得として扱わない。

### 3. PF係数選択はshortlistを変えたか

変えなかった。共通$\delta$窓で比較した$C_D$、論文D6、支配固有位相係数の全てが
$L_D=12,\delta=0.02$を点推定最良とした。costed候補のD6対支配固有位相係数差は最大0.317%であり、
係数選択だけではWP04の区間重なりを解消しない。今後のshortlistでは論文D6を主係数、支配固有位相を
小系reference、$C_D$をscreening専用とする。

### 4. 最大の残存不確かさは何か

`full controlled interrogationのcost scopeと回路構造`である。現在の比較にはcontrolled partial-$S_2$
反復の外側境界、完全なHadamard wrapper scope、未使用の長$q$ holdout、shared calibration covarianceが
揃っていない。一方、PF係数は現shortlistを変えず、finite-RTEと統計配分の最大効果は両候補に共通である。

## 研究方向の判断

| 方向 | 判断 | 現時点の意味 |
|---|---|---|
| T1 優位領域と限界 | decision bridge後に条件付き継続 | 優位性も同点も主張せず、full scopeで成立境界を判定する |
| T2 状態依存PF誤差・分割 | 必要係数精度へ範囲縮小 | D6を主に使い、順位またはPF境界へ効く場合だけ深掘りする |
| T3 高次PF | 保留 | 現在のbottleneckは二次PF係数でなく回路scope・構造である |
| T4 長いランダム回路cost予測 | 高優先で継続 | WP06-aとWP05の中心課題とする |
| T5 finite-RTE・RPE schedule | 狭い範囲で継続 | 公平な$\beta/\alpha$規則と終盤round優先を維持する |
| T6 表現・sampling共同設計 | WP06-aだけ実行し条件付き保留 | 大幅な構造差が出た場合だけ広い探索を再開する |
| T7 信頼できる資源評価・否定的結果 | 主軸候補として継続 | 公平なbaseline最適化とscope欠落が判断を変える事例を積み上げる |

## Shortlist

WP05へ残す主条件は$L_D=3,12$の$\delta=0.02$である。$\delta=0.01$は比較対照として同じ二候補を残し、
$\delta=0.0125$は初回batchへ入れず感度候補として保存する。$L_D=0$はWP01-Sの解析的成分作用数proxyに
限ってscreen outしたもので、一般理論またはcompiled-RZ costによる棄却ではない。

## 次の一件：WP06-a

まず一つまたは二つの代表DF Z/ZZ eventと短い列だけを使い、次を一変更ずつ比較する。

- full Gaussian basis変換とsupport限定構成
- basis変換の融合
- 既知scalarとphaseの処理
- 適用可能なcontrol最適化

小行列で厳密同値性を確認し、controlled回路ではglobal phaseを捨てず、必要な既知phase補償と枝間relative
phaseまで照合する。このpilot専用の`eta_decision`はRZ相対変化5%とする。これは現在の4.958%点推定差と
同程度で、候補判断を変え得るためである。次のいずれかなら、WP05の前に回路構造とproxyを更新する。

- 代表RZ costが5%以上変化する
- 候補順位が反転する
- $q$方向の傾きまたはproxy適用domainが変わる
- 現構造に必要なcontrolled relative-phase補償が欠けている

該当しなければ現構造を維持し、測定した残差を不確かさとして持ってWP05へ進む。この5%はWP06-aの
研究経路判定専用で、普遍的な回路精度保証ではない。

## 成果物と再生成

- 成果物：`artifacts/research_direction_gate_s1/2026-09-22/gate_s1_research_direction_decision_v1.json`
- runner：`scripts/run_research_direction_gate_s1.py`
- test：`tests/test_research_direction_gate_s1.py`

```bash
.venv311/bin/python scripts/run_research_direction_gate_s1.py
```

成果物は各upstream content fingerprintとfile SHA-256を記録する。local dirty-worktree上の判断記録であり、
immutable CIまたは外部再現証拠ではない。

専用testは`4 passed`、変更後のlocal全suiteは`511 passed, 4 warnings`だった。warningは既存の
grouped-UWC testにおけるcomplex-to-real cast由来で、Gate S1の失敗ではない。

## 後続結果

WP06-aを実行した結果、support限定Gaussian completionは単一controlled Z/ZZのRZを61.76%・
39.23%減らし、事前の5% triggerが発火した。一方、異なるZZ supportの長さ3列ではfull basis共有
より15.01%高くなった。このため、一律置換ではなくsequence-awareなfull/support選択とproxy再較正を
focused WP06-bとして行い、その後WP05へ進む。詳細は
[WP06-a回路構造pilot](research_direction_structure_pilot.md)を参照する。

WP06-bではsingleton runだけsupport限定にするpolicyが独立holdoutを通過した。既存proxyへの
additive bridgeはRZ $q$ slopeを最大8.00%変え、$L_D=3/12$の点順位を反転させたが区間は重なる。
従って次は選択policyを明示入力とするWP05へ進む。詳細は
[WP06-b sequence policy](research_direction_sequence_policy.md)を参照する。

WP05-aではcomplete controlled partial-$S_2$／Hadamard wrapperへの接続を実施し、$q=1,2$較正から
未使用$q=4$をRZ最大2.29%、全metric最大2.43%で予測した。中央additive bridgeのfull-wrapper RZ
残差も最大2.63%で5%基準を通過した。固定WP04条件の点順位は$L_D=3$となったが区間は重なった。
後続WP05-b/Rで$q=8$と$\delta=0.01$を検証し、WP01-D/C07で候補別再最適化まで完了した。
$L_D=3$の点推定は13.92%低く5% local model区間は僅かに分離したが、25%移送区間は重なるため、
頑健な方向判断は未確定である。詳細は[WP05-a full-scope接続](research_direction_full_scope.md)、
[WP05-b/R](research_direction_full_scope_extension.md)、
[WP01-D/C07](research_direction_decision_cost.md)を参照する。
