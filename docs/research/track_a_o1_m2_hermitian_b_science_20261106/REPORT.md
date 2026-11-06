# O1 M2：限定Hermitian-B実H4 operator-only検証

2026-11-06、SERVER 2（GPUA100-gleap02）。**固定事前条件によるoperator検査PASS、物理モデルの適格性は`UNASSESSED`。1回の実行完了後、科学計算はmandatory STOPした。** Bは今回の数値検証規約として限定採用し、元の物理Hamiltonianを正式再定義していない。旧FAIL・旧SOURCE・旧許容値・結果・grant・marker・STOPは変更していない。

## 使用SOURCE・入力・方法

- 固定実装：`7f9aea5f22e0e19dc3be002f6f35d4b9648fd872`。[事前条件](../track_a_o1_m2_hermitian_b_validation_preparation_20261106/REPORT.md)と既存`evaluate(raw, plan=H4_PLAN, projection_limits=None)`、微小係数保持mapper、独立CAR/density参照を再利用した。固定5 SOURCEのbytesをcommitと照合し、変更していない。
- 追加は[薄い実行driver](../../../scripts/run_o1_m2_b_limited_operator.py)のみ。実行時SHA256 `1ee1b8e4420253e3db43f06796584f0e6533a9010096ecf138e7d8de90c5803c`。同一invocation内で要求されたclosure・model差・一体補正診断を先に記録し、固定evaluateを1回実行した。判定式・閾値・演算数・精度は変更していない。
- 保存入力commit：`aced1647b81e8d28091058cdc7b6e3e1ee54896a`。coreは`artifacts/research_c_pr_integrated/2026-11-02/run01/update005_candidate.npz`（SHA256 `938340b8e09c85f5f4021582501ba6d81fc79d5382c37c1824863939bdf6e56e`）、元Hは`artifacts/research_c_pr_aware_df/2026-10-27/run01/prepare/input.npz`（SHA256 `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db`）。新run内へbyte同一コピーを取得しhash照合後decodeした。コピーは専用`.gitignore`で公開対象外とした。
- core生成SOURCEは`980b514c81eb415fe053f558d7d3c0b8cfb5b9b4`。4 spatial／8 spin orbitals、保存update005の全8 frame・元g由来κを維持し、旧LD2 tailは使用しない。
- 正規順序`H_H=P(H0), C_H=P(C0), R_H=H_H-C_H`をbinary64で形成した。CAR参照は整数疎monomialと`math.fsum`、P/subtraction形成診断は元binary64 bit値の正確なFractionを使用した。J/U縮約の無限精度再計算ではない。
- 新[raw結果](../../../artifacts/track_a_o1_m2_hermitian_b_science_20261106/run01/result.json)に全SOURCE hash・入力binding・時刻・runtimeを記録した。Python 3.12.3／NumPy 2.5.4／SciPy 1.18.1／OpenFermion 1.8.1／Qiskit 2.6.2、実行23.48秒。共有設定・ライブラリ・他ユーザーのprocessは変更していない。

## 独立参照との一致（max-entry絶対差）

| 比較 | full256 | sector36 |
|---|---:|---:|
| `C_H+R_H-H_H`（独立CAR） | `4.440892098500626e-16` | `2.220446049250313e-16` |
| H_H Pauli − CAR | `8.881784197001252e-16` | `4.440892098500626e-16` |
| C_H Pauli − CAR | `8.881784197001252e-16` | `8.881784197001252e-16` |
| **R_H Pauli − CAR** | **`8.271806125530277e-25`** | **`4.1359030627651384e-25`** |
| 独立density coreのP − C_H | `4.884981308350689e-15` | `1.5543122344752192e-15` |
| 独立density coreのP ＋ R_H − H_H | `5.329070518200751e-15` | `1.5543122344752192e-15` |
| **R_H − P(raw R)** | **`2.2551405187698492e-17`** | **`2.2551405187698492e-17`** |

sectorは`Nalpha=Nbeta=2`。full検査をsectorへ置き換えていない。固定countはtensor264192、κ10、core一体10、mapper33824、CAR4228、guard8。上流／形成allowanceは`5.5083244455663064e-08`、残差専用`tau_R=2.342260361136004e-18`（rtol=0、単位floorなし、総H allowanceの加算なし）。R_H参照max-entryは`2.2651961936020587e-09`で、固定1%のmapper・形成・独立core参照分解能条件を通過した。R_H参照／mappedのHermiticity defectはいずれも0。

## 形成・anti-Hermitian成分・model差

正確な`P(H_bits)-P(C_bits)`に対するcanonical係数形成誤差のCAR-L1 boundは`4.501607430616656e-16`、`P(raw R)`経路は`1.0901336641983205e-24`。canonicalと後者の差をmapper誤差へ混ぜていない。今回canonical Rは非零で、形成分解能は適格だった。これは保存binary64 H/core packet間のP/subtractionを対象にし、core生成／積分生成誤差の証明ではない。

| model変更の観測max-entry | full256 | sector36 |
|---|---:|---:|
| H_H − H0 | `5.551115123125783e-17` | `5.551115123125783e-17` |
| C_H − C0 | `1.1102230246251565e-16` | `5.551115123125783e-17` |
| R_H − raw R（projectionと差形成を含む） | `1.0104764247564901e-16` | `1.0104764247564901e-16` |

raw残差のanti-Hermitian半成分はfull`8.391724815037804e-17`／sector`8.272462576064399e-17`。projectionの係数L1変更量はH `5.919376153131022e-16`、core `1.6254909057165304e-15`、`P(raw R)` `1.936859054150246e-15`。これらのmodel許容上限は採択されていないため、**大小から物理モデルのPASSを推定せず、`UNASSESSED`を維持した。** 固定moduleの旧candidateラベル／`formal_adoption=False`は保持し、wrapperの`numeric_convention_B_adopted_for_this_validation=True`で今回の限定採用を区別した。

## scalar・一体補正・微小係数

- 核scalarはH/core各`2.29310124732`、残差0。Pauli電子identityは核scalarと別に保持し、R_Hの値は`4.6249802598818945e-11`、CAR traceとの誤差0。R_H nonidentity360項を保持した。
- rawの固定field名`identity.*.electronic`は既存`packet.constant`を格納している。H/coreでは核scalarを含むPauli Iの**総係数**であり、電子部分だけの値ではない。Rは核scalarが0なので同fieldが上記の電子I_Rそのものに一致する。field名・raw・固定SOURCEは変更せず、この意味を注記する。
- 元g由来κと独立`fsum`算出の差は`1.232595164407831e-32`。raw R一体項と`contract(raw R.g)/2`との差は`1.0408340855860843e-16`、raw Rのg符号`g_H-g_core`との差は0。raw R.h最大`8.000222706527893e-11`、g最大`7.998633243176693e-10`。一体補正・scalar・符号を省略していない。
- 残差専用mapper検査は微小係数に物理scaleを掛けたOpenFermionのthresholded和を使わず、既存unit-JW→fsum経路を再利用した。残差を零にした場合の参照差は`2.2651961936020587e-09`、逆符号なら`4.530392387204117e-09`という診断もrawへ記録され、今回の残差一致とは区別される。

## 不確かさ・GO/STOP・次の担当

固定判定の10個のfull/sector比較、残差自身のmapper/Hermiticity、scalar/identity、上流・形成分解能は通過した。SOURCEのengineering allowanceは厳密interval enclosureではなく、binary64／CAR参照精度と演算経路に基づく条件付き数値検証である。特に大きい全H側allowanceを残差mapperへ流用していない。今回の結果から物理Hamiltonianの正式再定義・性能改善・新fitの必要性を判定しない。

driverのAST／helpを実行前確認した。固定実装の既存人工46件PASSは準備commitの記録であり、今回の実H4結果とは別の証拠である。新しい科学再実行・事後許容値変更・旧FAILのPASS化は行っていない。状態は読み込まず、元H厳密基底状態仮定・prep費用除外は今回のoperator-only scope外へ変更しない。

**科学検証はmandatory STOP済み。** driver・raw JSON・本報告・専用input除外設定を専用branchへ公開しremote照合する。以降はユーザーがmodel差上限と元Hとの扱いを判断する。Codexが追加operator検証や次段階へ進むには、対象範囲を別途指定・承認する必要がある。PF/RTE、native回路build/compile、sampling、shot／費用、新fit／rank探索は実行していない。
