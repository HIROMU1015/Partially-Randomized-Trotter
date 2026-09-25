# P-C geometry signed-error pilot

## 結論

固定H4条件の小規模pilotでは、geometryに依存する符号付きProduct Formula（PF）誤差係数を
未使用geometryへ補間し、さらに未使用deltaでgeometry間エネルギー差のPF biasを予測できた。
事前に固定した5条件を全て通過したため、案Cを次段へ残した。その後のgeometry tracking・breakdown
validationでは固定停止条件が成立し、現在はこのH4 familyを主研究候補として進めない。

ただし、これは0.80--1.20 ÅのH4 linear chain、STO-3G、8 qubit、DF rank 12、`L_D=3`、
二次partial-$S_2$だけのlocal dirty-worktree結果である。potential-energy surfaceの完成、別系への
移送、RPE/RTE cost、回路compile、最終総costまたは科学的優位性は評価していない。

## 問いと固定条件

P-Cで調べる問いは、各geometryの絶対エネルギーを個別に高精度化することではなく、

$$
\Delta E_{\mathrm{PF}}(R_b,R_a)-\Delta E_{\mathrm{exact}}(R_b,R_a)
=b(R_b,\delta)-b(R_a,\delta)
$$

という符号付き差分biasを、geometry依存のleading coefficientから予測できるかである。
ここで`exact`はuntruncated FCIではなく、各geometryの同じDF rank-12 Hamiltonianを物理粒子数sectorで
厳密対角化した参照値である。

- geometry grid: 0.80、0.85、1.00、1.15、1.20 Å
- coefficient training geometry: 0.80、1.00、1.20 Å
- geometry holdout: 0.85、1.15 Å
- coefficient fit delta: 0.0125、0.025、0.05、0.1、0.2
- delta holdout: 0.4
- PF: 二次partial-$S_2$、`L_D=3`、exact-tail参照
- QPE/RPE統計誤差: PF係数と混合しない

0.90 Åでは本計算前に実行時間と既存runnerの動作だけを確認した。この点はthreshold決定にも
統合artifactにも使用していない。

## 事前固定したpilot gate

小delta側の5点で、原点を通る

$$
b(R,\delta)\simeq C(R)\delta^2
$$

をfitした。次を全て満たすときだけ案Cを次段へ残す。

1. training geometry間の線形補間によるholdout `C(R)` 相対誤差が最大10%以下。
2. 各geometryでfitに使わない $\delta=0.4$ のsigned bias相対誤差が最大10%以下。
3. 0.85--1.15 Åかつ $\delta=0.4$ の二重holdout差分について、予測誤差が両endpointの
   大きい方のabsolute PF biasの10%以下。
4. `C(R)`のspanが最大絶対値の10%以上で、geometry依存が数値的に自明でない。
5. $\delta=0.1$の隣接geometry pairに、差分PF biasがendpoint biasの50%以下となる相殺がある。

これらはP-Cのテーマ選定gateであり、一般的な化学精度またはrigorous boundではない。

## 結果

| $R$ (Å) | DF rank-12 $E_0$ (Ha) | signed $C(R)$ (Ha) | $\delta=0.4$ holdout相対誤差 |
|---:|---:|---:|---:|
| 0.80 | -2.1675605441 | 0.0234382633 | 0.840% |
| 0.85 | -2.1783136329 | 0.0200349441 | 0.702% |
| 1.00 | -2.1663874486 | 0.0133580009 | 0.428% |
| 1.15 | -2.1208025758 | 0.0095350152 | 0.273% |
| 1.20 | -2.1026084810 | 0.0086011846 | 0.237% |

- geometry holdout coefficient誤差: 0.85 Åで4.409%、1.15 Åで2.678%
- coefficient span fraction: 63.303%
- 全geometryのdelta holdout最大誤差: 0.840%
- 二重holdout（0.85→1.15 Å、$\delta=0.4$）:
  - exact DF rank-12 energy difference: 0.0575110571 Ha
  - actual signed PF difference error: -0.0016984874 Ha
  - predicted signed PF difference error: -0.0017804494 Ha
  - endpoint-bias正規化予測誤差: 2.539%
  - difference error自身に対する相対誤差: 4.826%
- $\delta=0.1$の隣接pairで最小cancellation ratio: 9.786%

したがって、単に「近いgeometryで誤差が近い」という観測だけでなく、未使用geometryと未使用deltaを
同時に使った差分biasが固定gate内で予測された。案Cは主研究候補として残す。

## 訂正と後続判断

初版artifactでは表のexact-energy欄にraw inputのsurrogate ground-state energyを転記していた。
PF bias、`C(R)`、holdout予測、gateはraw validationがfull DF-rank-12 Hamiltonian ground energyから
計算していたため変わらない。訂正版artifactではexact-energy欄とexact energy differenceだけを
full Hamiltonian値へ直した。

後続の[geometry tracking・breakdown validation](research_direction_geometry_tracking_breakdown.md)では、
stretch側のblind予測と診断が固定gateを通らず、`stop_pc_current_h4_family_as_primary`となった。

## 証拠と再実行

- 統合artifact:
  `artifacts/research_direction_geometry_energy_difference_pilot/2026-09-25/pc_h4_geometry_signed_error_v1_corrected_energy.json`
- raw input 5件:
  `artifacts/research_direction_geometry_energy_difference_pilot/2026-09-25/inputs/`
- content fingerprint:
  `3c199dbc20d0892cf1bcfad4b27646ea8c90d1c1d32fc5e2c75be610bb1fe611`
- 解析runner:
  `scripts/run_research_direction_geometry_energy_difference_pilot.py`
- library:
  `src/trotterlib/research_direction_geometry_energy_difference_pilot.py`
- test:
  `tests/test_research_direction_geometry_energy_difference_pilot.py`
- test結果:
  専用`3 passed`、PF/P-B/P-C関連`10 passed`、全suite
  `576 passed, 2 skipped, 4 warnings`、失敗0

raw inputは各geometryについて`run_pf_delta_validation.py`を同じ引数で実行して生成した。
統合artifactは既存ファイルを上書きしない。provenanceには各raw fileのSHA-256、validation
fingerprint、解析source hash、commit、dirty-worktree状態、実行commandを保存する。
