# PR-2 matched-accuracy M1-B1 result validation

## 結論

M1-B1のcompile mapは、固定identity、resource上限、全checkpoint、candidate別cache、集約値を
再検査し、`M1_B1_RESULT_VALIDATED_RESEARCH_REVIEW_COMPLETE`としてlocal validationを通過した。
研究判断は

`CONTINUE_RESOURCE_STUDY`

とする。ただしこれはheld-out実行認可ではない。次の停止位置は、結果を先に固定した別の
held-out transfer reviewを作成するところであり、H4 1.30 Åはまだ開かない。追加96 trajectory、
transfer、S3、最終winner精密化も未承認のままである。

## 対象と証拠identity

- system: H4 linear 1.00 Å
- basis: STO-3G
- sector: 8 qubits
- DF rank: 12
- split rank: `0, 3, 6, 9, 12`
- total time: `T=0.8`
- matched-accuracy grid: `q={1,2,4,8}`、`delta={0.8,0.4,0.2,0.1}`
- compiler: Qiskit 1.3.0、basis `rz,sx,x,cx`、optimization level 1、seed 17、
  coupling mapなし
- primary resource: 状態準備を除くfull measured Hadamard wrapperのcompiled RZ期待値と
  M1-A解析shot数の積
- actual execution source commit:
  `33f436bb3a7d5b9cefa23604bb22c8d1fb17cd62`
- execution plan SHA-256:
  `5afc94fac0571b38c74b0b00cfcf68e34e491a5fc65ef579d3ec884d079e5aa5`
- execution plan fingerprint:
  `17c91d41e77d7c085629b60ea470abc87e9590f448ba9bcdfa4342410cd89607`
- M1-B1 result SHA-256:
  `71278113c32b26af0dbf6144a626237a0087478212f8a93fc908de3d4d52aee4`
- M1-B1 result fingerprint:
  `504d9c9089726800a291a8259e87b2d37c1fdea046263db6c9582bb659c77975`
- completion marker SHA-256:
  `c076d80bd584e646bce7877df3b13dac9aec5e93a52f7ed9b1464fa6a1e0e52c`
- validation artifact SHA-256:
  `c9a05babed99cd1e80eaec5b58e47f25d74513c7ba2e5a00775cfd5959c37a0f`
- validation fingerprint:
  `c3cf1c084ebfe343d576236de2803c9c69855e0247ca6a2d628c496ee0546214`

一次結果は
`artifacts/pr2_matched_accuracy_m1_b1_execution/2026-09-30/`、再検証結果は
`artifacts/pr2_matched_accuracy_m1_b1_result_validation/2026-10-03/`に置く。

## 完全性と再集計

次をすべて再計算またはbyte-levelで照合した。

- compile map: 210 cell
- accuracy eligible: 206 cell
  - B0: 8 eligible / 4 ineligible
  - B1: 4 eligible
  - B2: 145 eligible
  - B3: 49 eligible
- random cell: 194
- baseline cell: 16
- random trajectory: 6,208、seed衝突0
- wrapper record: 12,448
  - random: 12,416
  - baseline: 32
- checkpoint JSON: 210
- candidate-scoped SQLite cache: 210
- unique actual circuit transpile: 12,128
- 同一candidate内の意味的同一回路cache reuse: 320
- process worker: 6、BLAS thread/worker: 1
- extension trajectory、signal再評価、held-out access、transfer、S3、GPU query/use: 0

320件のreuseは、candidate fingerprint、axis、trajectory identityが異なる結果を横流ししたものではない。
同一candidate内で生成回路のsemantic fingerprintが一致した場合だけであり、cross-cell reuseはない。
全210 checkpointのtask fingerprintと全cache key、各axisの平均・最小・最大・分散・標準誤差を
一次recordから再集計し、保存結果と一致した。

## actual compiled resource map

accuracy適格206 cellを、`circuit_size`、`cx_count`、`cx_depth`、`rz_count`、`rz_depth`、
`total_depth`の6 work指標で同時比較したactual Pareto集合は2件だけだった。

| candidate | shots | circuit size work | CX work | RZ work | RZ-depth work | total-depth work |
|---|---:|---:|---:|---:|---:|---:|
| `B2-rank3-q1-r4-K2` | 20,563 | 249,472,886.375 | 69,446,391.750 | 130,774,896.656 | 53,615,452.125 | 116,808,764.094 |
| `B2-rank3-q1-r8-K2` | 19,489 | 253,752,870.312 | 68,976,443.250 | 132,704,255.187 | 52,970,492.969 | 115,424,211.531 |

primary RZのpoint minimumは`B2-rank3-q1-r4-K2`である。各methodの最小RZ workと比べると、
この値はB0最小より48.388%、B1より64.895%、B3より88.656%低い。RZ最小の10%以内に入る
6候補はすべてB2 rank 3、q=1、`r={2,4,8}`、`K={2,4}`だった。

ただしK2/K4まで含む厳密なpoint winnerは32 trajectoryでは確定しない。RZ 1位
`B2-rank3-q1-r4-K2`と2位`B2-rank3-q1-r4-K4`の差は0.741%で、paired Monte Carloの
gap診断は0.721 standard error相当である。この留保はwinner refinementを認可する理由にはせず、
method/split領域がB2 rank 3、q=1へ集中したという結論だけを採用する。

## fixed-q=8との比較

旧S2はq=8固定で、B2 rank 6とB3をmaterial frontierとしていた。M1-B1のq=8内ではB2 rank 3の
最小は`B2-rank3-q8-r2-K2`、RZ workは684,061,479.375である。matched-accuracy全gridの
q=1 point minimumはこれより約80.88%低い。

したがって可変qを戻すと、設計判断は「q=8のB2 rank 6/B3境界」から「q=1のintermediate
B2 rank 3領域」へ変わる。M1-B1はfixed-q比較の単なる再現ではない。

## 旧selectorとproxyの監査

- M1-A proxy frontier: 64 cell
- 旧selector: 16 cell
- actual six-metric Pareto: 2 cell
- actual Paretoのうちproxy frontier内: 2/2
- actual Paretoのうち旧selector選抜内: 1/2

旧selectorはprimary RZとcircuit-sizeのpoint minimumを含んだが、もう一つのactual Pareto候補を
落とした。選抜内だけで作る最小値のregretは、CX 0.681%、CX depth 2.255%、RZ depth 1.218%、
total depth 1.200%だった。random 194 cell内のRZ workに対する`W_action`のSpearman相関は0.958、
log-Pearson相関は0.996であり、proxyは全体傾向には強い。しかしfrontier completenessを保証しない。
M1-Aの`SELECTION_LIMITED`判定とbounded全grid compileへの切替は妥当だった。

## 状態準備感度

共通の状態準備RZ相当cost/shotを`P>=0`として各cellへ加えたlower envelopeは、走査した全breakpointで
B2だけから構成された。P=0では`B2-rank3-q1-r4-K2`から始まり、P増加に伴ってB2 rank 3の
q=1,2,4,8へ移り、非常に大きいPでB2 rank 6 q=8へ移る。B0、B1、B3はlower envelopeへ現れない。

これは状態準備実装を測定済みという意味ではない。Pをパラメータ化した感度診断であり、noise、
実backend、量子shot実行も未評価である。

## 研究判断

`CONTINUE_RESOURCE_STUDY`とする根拠は次である。

1. intermediate B2 rank 3がprimary RZ/circuit-sizeのpoint minimumかつactual Paretoに残った。
2. 旧16-cell selectorはactual Paretoを完全には保持せず、direct compileの追加情報があった。
3. matched-accuracy q最適化はfixed-q=8 S2から方式選択を具体的に変えた。
4. `P>=0`の状態準備感度でもlower envelopeはintermediate B2から成り、endpointだけには縮退しない。
5. r/Kの厳密winnerは未確定でも、RZ最小近傍のmethod/split/q結論は安定している。

この判断はH4 linear 1.00 Å、STO-3G、DF rank 12、Qiskit 1.3.0 optimization level 1の
development-only local resultに限る。resource-optimality一般、新algorithm、H12、最終RPE総cost、
状態準備実装込みの優位性を主張しない。

## 実行とtest

validatorは分子snapshotを再loadせず、保存済みM1-A/M1-B1 artifact、checkpoint、cacheだけを読む。
held-out pathへはアクセスしない。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  "/home/abe/Project/Partially Randomized Trotter/.venv311/bin/python" \
  scripts/run_pr2_matched_accuracy_m1_b1_result_validation.py \
  --project-root . \
  --output artifacts/pr2_matched_accuracy_m1_b1_result_validation/2026-10-03/pr2_matched_accuracy_m1_b1_result_validation_v1.json
```

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  "/home/abe/Project/Partially Randomized Trotter/.venv311/bin/python" -m pytest -q \
  tests/test_pr2_matched_accuracy_m1_b1_result_validation.py \
  -p no:cacheprovider \
  --basetemp=/tmp/pytest-pr2-m1-b1-result-validation
```

local focused testは4 passed。これはimmutable CIまたは独立外部再現ではない。

## 次の停止位置

次は、M1-B1結果を変更せず、developmentで固定するtransfer対象、成功条件、重大なunderestimate、
再最適化禁止を定めたresult-prior held-out transfer reviewを別commitで作る。レビューと別authorizationが
完了するまでH4 1.30 Åをload、hash、stat、評価しない。
