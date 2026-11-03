# m17 Task A：高精度でも保存factor間の差が残る

2026-11-03 JST。**Task Aは正常完了。96/160 bitで収束診断を満たしたが、
law/reference差はS64固定で約6.2484e-10、保存F/R/tail固定で約6.2431e-10と残った。**
今回の数値診断では、保存後のq乗・期待値やS組立てだけを高精度化しても、元1e-10の閾値超過は解消しない。
保存tail以前の丸め、入力表現、実装・数式定義の違いは未分離である。
元INCONSISTENTを変更せず、u_num_bound=null、RPE_tolerance_certified=false、scientific_status=UNDETERMINEDを維持する。

## SOURCE・物理条件・入力境界

実行SOURCE：`4479a1c9cf150250254e2c2650f70fb43841fed1`。
[固定計画](../rpe_m17_numerical_precision_preparation_20261103/PRECISION_ISOLATION_PLAN.md)、
[adapter](../rpe_m17_numerical_precision_preparation_20261103/precision_adapter.py)、
[driver](../rpe_m17_numerical_precision_preparation_20261103/run_precision_isolation.py)を変更せず実行した。
実行branch：`codex/rpe-m17-numerical-precision-isolation-execution-20261103`。

linear H4、隣接1.00 Å、STO-3G、保存DF12、Bのphysical ground state、8 qubit・4電子sector70。
Partial L_D3、単一δ=.02、m17/q131072/r32/K2。r32/K2/attenuationは保存tailに含まれる。
N411/axis、alpha=.016957089147912167、epsilon=.15244325851620388、保存PF/RTE/stat予算を保持し、再評価していない。

入力は、元m17結果commit `cd749066904d3f29b28b03c19ab69370372b157f`のRESULT/7行列と、
B commit `e22d7718c47358ff650dbc04e8e72f6ab1c8bf41`の保存psi。
元m17の科学実行SOURCEは別の`128e9a5bf2630cd0333ec5f69b81b25def78eb79`、library carrierは
`4591b8d75a2a6a55c74a51225f0c61648023c8ee`。carrierと実行SOURCEを混同しない。
3入力の固定hashとschemaは[実行計画の入力表](../rpe_m17_numerical_precision_preparation_20261103/PRECISION_ISOLATION_PLAN.md)と今回の原RESULTに記録される。

元NPZにS自体はないため、保存F/R/tailからfloat64で`S64=R64 @ tail64 @ F64`を各経路で一度組み、
今回の別archiveへ保存して固定した。これは元に存在したSのbyteを取得したものではない。
A-stepはこのS64とpsi64、A-factorは元のF/R/tail/psi64を正確なdyadic値として高精度へ持ち上げる。
psiを再正規化せず、元行列の生成前に失われた桁を復元したとも扱わない。

## 実行した段階と結果

原記録：[RESULT.json](../../../artifacts/rpe_m17_precision_isolation/2026-11-03/run01/RESULT.json)、
[frozen_steps.npz](../../../artifacts/rpe_m17_precision_isolation/2026-11-03/run01/frozen_steps.npz)。
以下は原RESULTの転記・表示桁の省略。記録後の新たなmatrix power・mean再評価は行っていない。

| 段階 | law/referenceのcomplex差 | 元の1e-10との関係 |
|---|---:|---|
| A0：float64基準 | 6.259420298065069e-10 | 超過。両pathとも元平均からのdriftは記録上0.0 |
| A-step：S64固定、160 bit | 6.248411459764507e-10 | ABOVE |
| A-factor：保存F/R/tail固定、160 bit | 6.243065316757413e-10 | ABOVE |

元平均を再現したためA-stepを実施し、A-stepの収束したgapが閾値を超えたため、計画どおりA-factorも実施した。
224 bitは不要だった。float基準2平均と、高精度8平均（2 mode×2 path×96/160 bit）を計算した。
高精度jobと基準はそれぞれ計画の範囲内で、中断・retryは0。実行はexit0、COMPLETED_MANDATORY_STOP。

### 160 bitの平均値

| 入力境界・path | Re（省略） | Im（省略） |
|---|---:|---:|
| S64固定・law | .32902182413189454432 | −.46875529468601182948 |
| S64固定・reference | .32902182377293157645 | −.46875529417457068389 |
| F/R/tail固定・law | .32902182412734365440 | −.46875529467910673136 |
| F/R/tail固定・reference | .32902182376865465701 | −.46875529416812644931 |

表示桁はbackendの計算値であり、物理的な平均の精度を保証する桁数ではない。
旧V2保存meanに対する新しい成功率・energy誤差を評価した値でもない。

### 収束診断

| Mode | Path | 96/160 bit間のcomplex平均差 |
|---|---|---:|
| S64固定 | law | 2.427404309003714e-25 |
| S64固定 | reference | 1.467816616082985e-25 |
| F/R/tail固定 | law | 8.911279830252193e-26 |
| F/R/tail固定 | reference | 6.521785390361012e-25 |

全4件が事前の1e-13以内で、両modeの収束診断を満たした。
収束差は外向き丸め・真の平均への誤差上界ではない。scalar比較に288 bitのcontextを使うことも保証への昇格を意味しない。

### 入力境界を変えたときのpath別の変化

| 比較 | lawのcomplex変化量 | referenceのcomplex変化量 |
|---|---:|---:|
| float64基準対S64固定160 bit | 1.067941581676008e-12 | 2.072589649381357e-12 |
| S64固定対F/R/tail固定160 bit | 8.269883867341983e-12 | 7.734351897716646e-12 |

これらは各pathの平均の差であり、law/reference gapへの寄与率ではない。
後段演算の丸めによる数値変化は観測されるが、高精度化した後にも上記のgapが残る。
「外側q乗の丸めが存在しない」とは結論しない。

## 原因候補の整理

事前規則から出力された候補は次の二つである。

- CONSTRUCTION_ROUNDOFF_PLAUSIBLE
- IMPLEMENTATION_OR_DEFINITION_DIFFERENCE_POSSIBLE

意味は「保存factorのまま高精度で評価しても差が残り、保存tail以前の構成・float入力・定義の違いが次の対象になる」という範囲。
入力生成時の丸めと実装・定義差を区別できたわけではなく、単一原因を確定した判定ではない。
POWER_ROUNDOFF_PLAUSIBLEは今回の事前規則では選ばれなかったが、これは後段丸めの寄与が厳密に0という意味ではない。

新たなorder/rotation/normalizationの再構成、成分ごとの確率・係数の再生成、DF/basisの再準備は行っていない。
旧1e-10を緩めたり、lawとreferenceを同じTaylor実装に統一して差を消したりもしていない。

## 精度・資源・保存物

NumPy1.26.4、mpmath1.3.0/Python backend。float基準はcomplex128、精度jobは独立clone contextの96/160 bit。
binary poweringでq131072を評価し、保存binary64はas_integer_ratioによって正確なdyadicへ変換した。
新S64 archiveにはlaw/referenceの2個の70x70 complex128行列がある。

起動時はCPU32、空きRAM約52 GiB。初期独立job数に合わせてspawn worker4を使用した。
NumPy基準はjob内BLAS/OpenMP等4 thread。mpmathはPython多倍長で、BLASによるnested並列ではない。
GPU driverと接続できずCPU経路を使った。実行wall timeは約15.942秒。
各jobのwall timeは約6.77–8.43秒で、段階内は並列。process RSSやserial速度比較は測っていない。

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python docs/research/rpe_m17_numerical_precision_preparation_20261103/run_precision_isolation.py --run --output-dir artifacts/rpe_m17_precision_isolation/2026-11-03/run01
```

出力SHA256：

- RESULT.json（15,315 bytes）：`e5c3cb1cd455d102c8a1e74d7556ec95b26ab79e72f0727e172f4483162f362e`
- frozen_steps.npz（145,460 bytes）：`2250e10bcae320c2fe1ec30662eb9ce6e14df50798f3db2d78f60c28dcfd7c30`

元SOURCE/RESULT/operators.npz、B/A/V1/V2/V3/R1、N/α/β/epsilonと既存STOPを保全した。
新H/state/DF生成、sampling・shot、compile、統計予算・energy成功率・費用の新評価、重要科学レビューは0。
B-1/B-2、他round、R2/R3は開始していない。

## RPEへの解釈と次段

元の数値診断INCONSISTENT、u_num_bound=null、RPE_tolerance_certified=false、scientific_status=UNDETERMINEDを保持する。
既存の統計予算残余約8.60e-5を再計算しておらず、これと今回の収束差やgapを同一の保証量にしない。
この検証は固定されたfloat入力への後段評価を調べたもので、保存前の丸め、真の周辺平均、
全18round成功、厳密半径、物理状態準備、backend/noise、費用順位を認定しない。

**次段案**：CodexがB-1の最小計画・SOURCEを準備する。
まず保存A_prob/A_coeffとRESULT中のorder確率/tau/Bを固定し、どこから高精度へ変えるかを明示する。
theta64/cos64/sin64は元archiveにないので、必要なSOURCE replayの出力を新しい入力として記録し、保存済みと偽らない。
成分単位やDFの再生成、数式からのB-2を直ちに追加する必要は、まだ判断していない。
新しい構成評価はユーザーによる別の明示許可後だけとし、今回のTask Aから自動移行しない。
結果の公開照合後mandatory STOP。
