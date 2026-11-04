# RPE m17：局所周辺平均の計算完了・数値診断はINCONSISTENT

2026-11-03 JST。**実行は正常完了、固定1e-10の数値診断はINCONSISTENT、科学判定はUNDETERMINED。**
samplerの宣言lawとTaylor参照の最終complex平均に約6.26e-10の差があった。
固定N/α/epsilonから求めた統計予算残余は約8.60e-5。
診断差はこの残余とは別の量であり、今回のINCONSISTENTはRPEに必要な平均精度の違反を認定したものではない。
数値誤差上界u_numは未確立のままで、RPEの十分条件や全18round成功を認定しない。

## SOURCE・範囲・入力

科学実行SOURCE：`128e9a5bf2630cd0333ec5f69b81b25def78eb79`。
[固定実行計画](../rpe_m17_mean_validation_preparation_20261103/SCIENTIFIC_EXECUTION_PLAN.md)、
[runner](../rpe_m17_mean_validation_preparation_20261103/run_m17_mean_validation.py)、
[準備と人工13 tests](../rpe_m17_mean_validation_preparation_20261103/IMPLEMENTATION_REPORT.md)を変更せず使用した。
実行専用branch：`codex/rpe-m17-mean-validation-execution-20261103`。

| 項目 | 固定条件 |
|---|---|
| Model/geometry/basis/rank | linear H4、隣接1.00 Å、STO-3G、保存legacy DF12 |
| State/sector | Bの保存physical ground state、8 system qubit、4電子sector70 |
| PF/RTE | Partial L_D3、単一δ=.02、m17/q131072、r32/K2 |
| 係数・位相 | coefficient_atol=1e-12、extract_identity_phase、元の符号・scalar規約 |
| 保存N/α/epsilon | 各軸N411、α=.016957089147912167、epsilon=.15244325851620388 |
| β | PF=.014050012886294481、RTE=9.899584668737594e-8、stat=.38594988811785885 |
| 保存mean | V2 commit78c49d0b5f1d71e63b5959e78813237923bbc648、V2_ROUND_RESULTS.csvのld3/round_index17 |
| State/source | B commit e22d7718c47358ff650dbc04e8e72f6ab1c8bf41、legacy carrier4591b8d75a2a6a55c74a51225f0c61648023c8ee |

NはRPEモデルの保存入力であり、今回shotを生成していない。
Full・m8・残り17round、sampling・RNG replay、state solve、controlled/Hadamard build、compile、
成功率・費用の再計算は実施していない。

## 正常完了した結果

原記録：[RESULT.json](../../../artifacts/rpe_m17_mean_validation/2026-11-03/run01/RESULT.json)、
[operators.npz](../../../artifacts/rpe_m17_mean_validation/2026-11-03/run01/operators.npz)。
以下はRESULTの転記・表示桁の省略であり、保存演算子から追加の平均計算や再解析はしていない。

| 平均 | Re | Im |
|---|---:|---:|
| 保存μ | .32902182375994465 | −.46875529425442597 |
| Literal-law経路 | .32902182413176795 | −.4687552946849514 |
| Coefficient/Taylor参照 | .3290218237714639 | −.4687552941731073 |

| 比較 | Complex絶対残差 | Cosine座標残差 | Sine座標残差 | 固定complex閾値1e-10 |
|---|---:|---:|---:|---|
| law対reference | 6.2594202981e-10 | 3.6030406436e-10 | 5.1184412264e-10 | 超過 |
| law対保存μ | 5.6886265843e-10 | 3.7182329438e-10 | 4.3052544862e-10 | 超過 |
| reference対保存μ | 8.2130502275e-11 | 1.1519230014e-11 | 8.1318674017e-11 | 閾値内 |

閾値を変更せず、`numeric_diagnostic_status=INCONSISTENT`を保存した。
これは例外や中断ではなく、事前規則に従った正常完了時の診断である。
実行はexit0、stage=completed、COMPLETED_MANDATORY_STOP。追加retryは0。

## 統計予算残余と数値不確かさ

固定値に対する新しい科学的scalar再解析として、両軸それぞれ

    s = sqrt(2 log(2/alpha)/N),    R = epsilon - s

をDecimal80桁で求めた。両軸の入力が同じなので、保存された出力値も同じである。

| 項目 | 結果（省略表示） |
|---|---:|
| 保存epsilon | .15244325851620388 |
| Hoeffding半幅s | .15235725199419449503 |
| 残余R | .00008600652200938497108 |
| Cosine：R−観測law対保存座標差 | .00008600615018609059495 |
| Sine：R−観測law対保存座標差 | .00008600609148393634632 |

この残余は、既存N/α/epsilonを変えずに追加の座標mean差と数値不確かさを置くための、
前段で導いた保守的十分条件の量である。唯一の必要精度や最大許容差ではない。
Decimal80桁も外向き丸めの区間証明ではなく、表示した全桁の物理的正確さを保証しない。

観測したfloat64差を引いた余地は正であるが、真の実装平均差の上界ではない。
u_num_bound=null、RPE_tolerance_certified=false、科学status=UNDETERMINEDを保持する。
1e-10の数値診断の超過からRPE予算超過とは結論しない一方、正の余地だけから十分条件成立とも結論しない。
元V2の半径・入力/IID/alias/branchに関する条件や、全roundの確認も別に必要である。

## 到達点・identity・資源

保存H/state/B結果のbyte照合、再利用12 sourceファイル、準備時のrunner/test hashは実行前に確認した。
保存入力・旧SOURCE・固定N/α/β/epsilonを変更していない。

- preparation hash：`89145d8324bf139eaec67265f3134a5c1a3cebb7ed97792d54ed3221d31f8919`、218 component、4 deterministic block。
- state fingerprint：`a07d6c7fdbbb23a035f9fc54b9956ee79e557a2185e29b1870748b771958be90`。
- config：r32/K2、step_time=.000625、dimensionless_step_time=.0003645440153962189。
- probability sum：.9999999999999999。samplerと同じ再正規化をlaw側へ適用した。
- basis cache：11件。最大basis sector leakageは数値0.0。厳密な0の証明ではない。
- 非gating診断：A_prob対A_coeffのFrobenius差4.795333763004963e-16、tail平均の差5.983045574440441e-14。

保存scalar行1、DF preparation1、最終complex平均2経路を完了した。
operators.npzにはA_prob/A_coeff/F/R/reference tail/law one-event/law tailの7個の70x70 complex行列を保存した。
新Hamiltonian/state生成・ground-state solve・sampling・仮想shot・controlled build・compile・価格評価は各0。
0件の項目はSOURCEの構成に基づくもので、外部の独立したcall監査を実施した意味ではない。

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python docs/research/rpe_m17_mean_validation_preparation_20261103/run_m17_mean_validation.py --run --output-dir artifacts/rpe_m17_mean_validation/2026-11-03/run01
```

Python3.11.0rc1、NumPy1.26.4、SciPy1.14.1、Qiskit1.3.0、float64/complex128。
使用可能CPU32、開始前空きRAM約52 GiB、job内BLAS/OpenMP等4 thread。
GPU utilityはあるがdriverへ接続できず、準備済みCPU経路を使った。
basis集約とmatrix powerを使用し、記録wall timeは約1.515秒。
process RSSや実H4のserial比較は測っておらず、速度倍率を主張しない。

出力SHA256：

- RESULT.json（7,521 bytes）：`79fdf7e84f90c48230857b13f39f9becb30f19892e2e9c09d80805d346432a4a`
- operators.npz（484,458 bytes）：`e1a65238e9d9b61a6d0aa542ca7f509827b0a761fc751615f1f2018a0e7020df`

## 静的に確認できる差の候補と、今回の限界

SOURCEの読取りと既に出力された診断値から、次の違いを確認した。

1. law側はcomponent確率を再正規化する。係数側は保持係数/lambda_Rを使い、
   Aの出力差は上記の小さいFrobenius差として残る。
2. law側はorder確率とatan/cos/sinで一event平均を作ってからr乗する。
   参照側はpaired polynomialのr乗と`exp(r*log(B))`によるnormalizationを使う。
   数式上の同値性は、float64演算順や超越関数のbit単位の同一性を意味しない。
3. 両経路とも外側でq131072のmatrix powerを使う。長qで丸め差が増幅される可能性はある。

**これらを今回の差の原因と特定したわけではない。**
保存行列の高精度再評価、経路の入替え、個別誤差項の新しい上界評価は実施していない。
差を消すためにTaylor関数を両側で共有したり、1e-10を緩めたりもしない。
実装lawの科学的破綻、RPE必要精度への不適合、または単なる丸め誤差と断定する材料には不足する。

## 保全・STOP・次段案

元V1/V2/V3/R1、B、A run01/run02、前段設計・準備SOURCE/raw/status、M06-F/R2/R3のSTOPを保全した。
V2はCONDITIONAL_SUCCESS、R1費用順位とrun02はUNDETERMINEDのまま。
今回も独立重要科学レビュー・全18round検証・費用比較を開始していない。

正常完了のINCONSISTENTを「意図しない技術的中断」として修正・再実行する扱いにはしない。
結果を公開・照合し、計画どおりmandatory STOPする。

**次段案**：Codexが、同じ保存行列を固定した精度切分け計画と最小SOURCEを準備する。
外側powerの丸めと、order/rotation/normalizationの構成時の丸めを別に調べる方式を検討し、
固定float入力を高精度で評価する部分と、数式上の係数を再評価する部分を混同しない。
ユーザーが必要と判断した場合、その限定した新しい数値評価を別途許可する。
今回は追加計算せず、正の残余と診断超過の両方を原記録のまま残す。
