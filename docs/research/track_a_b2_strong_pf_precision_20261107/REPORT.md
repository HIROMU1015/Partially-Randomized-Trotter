# B2 P1強PF：基本step構成精度の改善・人工検証

2026-11-07、SERVER 2 GPUA100-gleap02。

**SYNTHETIC_PRECISION_IMPROVED。** 固定人工入力の8 armで、native/CAR両経路の構成精度が独立高精度参照に対して改善した。
旧科学実装は変更せず、明示選択する精度adapterを追加した。数理的PFのschedule・係数・符号・scalarを維持し、projection・正規化・phase fitは用いない。
**実H4は一切再計算せず、正式判定は3 PASS／5 STOPのまま。公開確認後mandatory STOP。**

## SOURCEと再利用

- 新規[precision adapter](../../../src/trotterlib/b2_strong_pf_precision.py)、[人工検証runner](../../../scripts/validate_b2_strong_pf_precision.py)、[人工テスト](../../../tests/test_b2_strong_pf_precision.py)。[比較結果JSON](../../../artifacts/track_a_b2_strong_pf_precision/2026-11-07/run01/result.json)が実行コードのSHA256、人工入力hash、全比較値を保持する。local人工検証後、そのbytesをcommitへ固定した成果で、CI/外部再現の主張ではない。
- 起点は実H4診断 `7dc2113e1325ef87b594b04f2d638fef641de9b7`／SOURCE `b81734e177b3161f287aea908720eac5c9394a00`。元強PF `a7940b58c6cb939d77e9f8e44d214b3787b7e101` と8 arm結果 `2583ce4957ed44e914ce6571e3e676830439dfa7` は保全。
- P1 `7bc952772e451f27c1654dab6803192f2b204c9f`、強PF接続 `66a41ad9219f12377e7d325e272c1a90d66ab7e5` の科学条件・独立1e-9・allowanceは変更しない。
- `strong_baseline_schedule`と`native.runs`を直接再利用し、Full S2対称列、Yoshida正・負・正stage、既存binary64係数とFraction時間を保持。cube rootの再評価や時間の再正規化はしない。
- CARの合成には既存`_ordered_exponentials(...,arithmetic=...)`を使用。既存CAR helperのannihilatorから得た整数±1 bilinearだけを再利用し、分子係数の和・signed square・expm・積は高精度で処理。
- 直前診断のexact binary64変換・HP差/信号/累乗helperを再利用。`SectorCAR`も調査したが、用途固有の半充填sectorとsparse-toarray経路があるため、既存Qiskit-order整数CAR helperを用いる小さいadapterを選んだ。

## 何を改善したか

`prepare_precision(one_body, lambdas, factors, constant, digits=..., basis_indices=...)`を明示的に呼び、`precise_step(..., formula=..., substeps=..., delta_time=.02)`で二つのoperatorを取得する。
旧`prepare_xdf_basis`、`native_mean_step`、`StrongBuilder`、旧runnerは上書きしない。

| 部分 | 新しい演算と数理的意味 |
|---|---|
| 固定入力 | binary64実数・複素数をその値のままmpmathへ変換。元factorをfit/再定義せず、Fractionも分子/分母から変換する。 |
| orbital eigensystem | 元Hermitian factorのspatial blockを高精度`eighe`で対角化し、同じexact interleaved spin liftへ戻す。旧binary64固有ベクトルのcastではない。 |
| Fock/relative | 同じexterior-minor定義Γ(V)、Γ(V_next†V_previous)を高精度で評価。既存のchronological relative fusionを保持。 |
| diagonal | one-bodyのΣeps*n、signed factorのλ(Σeta*n)^2を高精度occupation位相として直接評価する。旧RZ/RZZ/global-phase primitiveと同じoperatorで、factor identityも含む。旧primitiveのfloat強制変換を経由しない。 |
| native合成 | Γ(V_0)†、diagonal、relative、終端Γ(V_last)、核scalar一回を高精度で合成。 |
| CAR構成 | 元one-body/Gを整数CAR bilinearへ線形結合し、signed λ[dΓ(G)]²を高精度で形成。既存合成hookでmatrix expm/chronological積を実行。CARを絶対的厳密参照とは仮定しない。 |

`fock_lift`の旧helperはcomplex128/NumPy detに固定されていたため、minorの同じ定義に高精度の小さいadapterが必要だった。
人工テストでmpmath1.3.0 `det`が構造的に特異なminorで例外となるケースを検出した。task-localなpartial-pivot LU determinantを追加し、**exact-zero pivotだけを0とする**。依存packageは変更していない。
構造的ゼロ、線形従属、2^-200の非ゼロpivotを検査した。小さい値のcutoff・unitary化ではない。

高精度matrix/eigensystem機能は既存mpmathを使用し、contextをcloneして共有global precisionを変更しない。[mpmathの行列・固有値API](https://mpmath.org/doc/1.3.0/matrices.html)

## 入力条件と意味の保全

- 入力matrixはfinite binary64、**exact Hermitian**かつexact interleaved spin liftを要求する。許容誤差内だからと事後Hermitian化したりoff-spin成分を落としたりしない。
- 指定basisは完全なNα/Nβ sector（またはその和集合）であることを確認し、渡されたQiskit index順を保持する。不完全なsubsetへのprojectionは拒否。
- 核scalarはδに一回、factor identityは各diagonalに保持。Yoshidaの負時間もそのまま評価。
- 厳密unitaryなorbitalについてのΓの積則とspectral theoremにより、同じhと全signed λG²のPFを表す。高精度で残るunitarity defectは測定し、補正しない。

比較対象は、①固定factor/時間から定義した数理的PF、②旧binary64 step、③新HP step、④native emitterのgate意味を区別する。
今回精度を高めたのは**①に対する③の数値近似**。④との実際のgate列一致は`UNASSESSED`で、裸のΓ(有限精度V)をemitterの厳密作用と同一視しない。回路build/compileは行っていない。

## 人工検証と独立参照

人工4 spin-orbital（2 spatial orbital）、complex hopping、正負2 signed factor、scalar=.371、Nα=Nβ=1のsector4を使用。
stateは固定の(.5,.5i,-.5,.5i)で、solve・入力調整をしない。分子H4/DF12の数値やprivate stateではない。
8 armはFull S2／Yoshida4×substeps1/2/4/8で固定し、候補探索や順位選択をしない。

独立参照は元の人工h/Gから**明示的Pauli tensorのJordan–Wigner行列をfull空間で構成**し、signed square後にsectorへ制限する。
その多体系generatorの高精度spectral exponentialを使う。nativeのorbital/minorやCAR adapterのbilinear・出力matrixから参照を作っていない。
合成scheduleは共通の既存helperであり、新しいPF係数・順序を導入しない。

**46テストPASS（5.43秒）**：新規16件＋既存人工18件＋直前人工12件。
Hermitian/spin/dtype不適格の無補正拒否、exact binary/Fraction変換、同一schedule、負stage、signed factor、scalar/identity、full/sector作用、OF→Qiskitのindex順、relative fusion、LUのexact-zero動作を確認。
private loader、分子保存JSONのloader、回路constructorを禁止した人工テストである。

基本stepの独立120桁参照に対する最大entry誤差：

| arm | 旧native64 | 旧CAR64 | 新native80 | 新CAR80 |
|---|---:|---:|---:|---:|
| Full_S2_s1 | 3.16176e-15 | 2.34487e-16 | 1.93103e-80 | 2.44203e-81 |
| Full_S2_s2 | 5.36956e-15 | 3.82398e-16 | 2.92014e-80 | 1.78972e-81 |
| Full_S2_s4 | 1.10854e-14 | 8.14144e-16 | 5.60014e-80 | 3.49505e-81 |
| Full_S2_s8 | 1.97814e-14 | 2.06358e-15 | 1.08791e-79 | 4.49248e-81 |
| Yoshida4_s1 | 8.62569e-15 | 2.9884e-16 | 3.89532e-80 | 4.20785e-81 |
| Yoshida4_s2 | 1.5584e-14 | 4.49884e-16 | 7.76449e-80 | 5.95233e-81 |
| Yoshida4_s4 | 3.04608e-14 | 8.17825e-16 | 1.65832e-79 | 8.96458e-81 |
| Yoshida4_s8 | 5.80767e-14 | 2.31091e-15 | 3.10741e-79 | 1.39845e-80 |

全8 armで両経路が独立参照に対して改善した。互いに近づいたことだけでの判定ではない。
80桁/120桁の独立参照間の最大entry差は約4.51e-79。
新120桁構成の参照誤差最大はnative約5.15e-119、CAR約6.14e-119。
これらは固定人工入力の数値観測で、全入力への厳密誤差保証ではない。80/120桁は試験精度であり、新しいP1科学契約ではない。

80桁構成のcomponent残差最大：orbital unitary約2.11e-81、元factor再構成約1.05e-81、Fock unitary約4.22e-81、CAR generatorとの整合約3.16e-81。
基本step全体での改善を確認したが、全substageのうちどれだけの精度向上が必要最小限かまでは切り分けていない。

## 長時間の構成誤差と累乗精度

人工Q=131072で同じ独立PF参照を用いて比較。以下は全8 armの最大entry誤差で、同じoperator指標を比較している。
**HP累乗は全て120桁**。JSONのnative80等のmethod名は基本stepの構成精度を指す。
baseの信号内積も120桁で共通化して構成差を比較し、binary64累乗controlだけは旧経路と同じbinary64内積を使う。

| 構成＋累乗 | native | CAR |
|---|---:|---:|
| 旧64構成＋HP120累乗 | 7.30e-9 | 2.72e-10 |
| 新80構成＋HP120累乗 | 3.56e-74 | 1.30e-75 |
| 新120構成＋HP120累乗 | 5.46e-114 | 5.29e-114 |

旧64構成＋64累乗も実行し、同じstateの信号・複素差・半径差・円周位相差をJSONに保存した。基本stepのmatrix差・unitarity・traceによる相対位相診断も保存し、phaseでoperatorを補正していない。
旧stepをHP累乗しただけで残る構成差に対して、新しい構成からのHP計算が改善することを人工系で確認した。

## 実H4への準備と残る作業

- 精度adapterと人工検証は実装済み。実H4へ進む**条件付きの技術的準備**が整った。新たなPF理論・再最適化・追加管理基盤は不要。
- ただし、保存H4入力が新adapterのexact Hermitian/spin条件を満たすかは今回確認していない。別途承認された実行の入力gateで確認し、不一致ならSTOPする。黙ってoperatorをHermitian化・変更しない。
- 次の実行段階で、既存`load_private_inputs`のh/λ/G/scalarとsector indexを`prepare_precision`へ接続し、保存元H stateを値保持してHP期待値へ渡す薄いrunner差分が必要。既存z_phys/z_DF、P1 cap・allowance、半径判定を再利用し、独立差をHPのまま評価する。元の15点診断runnerはold SOURCEの再現用として保持する。
- 実H4の正式再適格化は、固定新SOURCE・入力・P1で停止5 armの全18round（最大90点、arm内最初の不適格でSTOP）を別途明示承認した後に実施する。HP80/120等の参照安定性も確認し、旧3 PASSは旧証拠として保持する。今回その再計算・再ランキングを行っていない。
- 元の厳密ground state仮定・no-prep費用範囲を維持する。数値/実装誤差の厳密enclosure、全support/長compiled回路、branch/shot込み95%成功、価格・総RPE費用改善は未評価/未認証。

**担当**：ユーザーが新精度経路を用いる実H4の5 arm再適格化についてSOURCE・範囲を承認し、Codexがその範囲で薄いrunner接続と実行を行う。今回は開始しない。
GPT詳細科学レビューは実施していない。元PFの意味を変えない技術的精度改善として人工検証を終えた。

SERVER 2の利用可能RAM約410GiB、CPU affinity64を観測。小さい人工行列の比較は約3.52秒で完了し、恣意的CPU/RAM/worker制限や共有設定変更はしていない。private入力は読んでいない。

**mandatory STOP。実H4の正式判定は3 PASS／5 STOPのまま。**
