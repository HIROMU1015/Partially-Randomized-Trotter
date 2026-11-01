# P3 Stage 2：H4 matched有限RTE名目平均・normalizationの限定結果

2026-11-02 JST。**数値整合はPASS、finite biasのRe/Imは未解像。`STOP_STAGE2_UNRESOLVED_COMPONENTS`。**
科学batch1回、6複素信号＋2normalization bundleを完了。追加精度・retry・次Stageは実施していない。

## 固定条件・SOURCE・入力

Linear H4／隣接1.00 Å／STO-3G／中性singlet、system8／full256、保存DF tol-only1e-8／rank10。
案Aの共通T_eff保持・V_square残差、T=.8／S2／q8／r4／K4（5次）、M_B/M_Cのみ。
時間はbinary64 .8のexact rationalを8、4で割った共通operand。RTE位置32、保持halfは各outerの両側。

- 科学SOURCE：`4bfbbaad1f8178077489bf06b085377256d6fb80`（SOURCE公開・固定後に実行）。
- 契約正本：`495ed110ad8379663145791b788a63b64f05eedd`の[科学契約][CONTRACT]と[準備表][READY]。
- Stage 1：SOURCE `da13170e593e6d845b014e3f110691be10e29d43`、結果`5708e37080738c4ebf74590d799a3741246483b2`。旧34項目を再実行していない。
- 保存入力anchor：`4591b8d75a2a6a55c74a51225f0c61648023c8ee`、元生成SOURCE `0c329b30a81accb35c212be8defa66fee0dd0f12`。
- 共通state hash：`8339b1a8b0bec35ae07688e667fde7ba602d56c1182523d58e0889ea114e8b80`。signed zeroを含むbytesと保存exact νの対応を確認、再正規化なし。
- 保存DF C reference SOURCE：`e80d7510880f8db993b777c627bb8e95b05a070d`。raw区間両端を保存exact νで一度だけ除算した。新reference生成0。
- 固定保存ファイル15件のSHA、元MO/spin/JW/DF/state/νの対応は[source_and_inputs.json](source_and_inputs.json)。過去の科学受理statusは変更しない。

## 共通scalar・独立辞書・operator preflight

T_*は保存h_effのnative occupation作用、V_*は保存factorsのnative ΣλA²。残差辞書和からcanonical operatorを定義していない。
Pauliは元bare Wと−Dを既存OpenFermion/JW経路で変換。DFは保存Gを対角化し、basis別rotated Z/ZZを構築した。
Gaussian回路・再DF分解は0。cutoff0、exact zeroのみ除去、異なるbasisのsupport統合なし。

共通v̄=4.679666433618258、c_N=2.29310124732。T内部identityはTに残し、c_N＋v̄のexact operand和に由来する共通phaseを全体で一度適用した。
Pauli full-H constantを追加せず、identity差はbindingの内訳として保存・伝播し、二重加算していない。

| 辞書 | 非identity成分数 | 抽出identity | 共通v̄との差 | residual sum差（spectral） |
|---|---:|---:|---:|---:|
| P | 184 | 4.6796664336182632 | 5.3290705182007514e-15 | 8.3919995738065623e-15 |
| DF | 360 | 4.679666433618257 | -8.8817841970012523e-16 | 9.0707612013544502e-15 |

新しいoperator preflightは**57項目すべてPASS**。閾値は契約のscale依存τ/4・4τ。
詳細は[operator_preflight.json](operator_preflight.json)、成分係数・順序・10個のDF basis/固有値は[dictionaries.json](dictionaries.json)。
共通T/V0だけを[common_operators.npz](common_operators.npz)へ保存し、入力stateや各成分の巨大行列を重複保存していない。

## Normalization：理想nominal law、実効CDF評価なし

λ_Rは非identity辞書の係数L1 norm。βはk=0,2,4のpaired weight和、B=β^32、log B=32 log β。
既存binary64評価とdecimal50 scalar検算を併記し、Bを厳密上界certificateとは呼ばない。

| 辞書 | λ_R | β | B_total | log B_total | Bのdecimal50との差 |
|---|---:|---:|---:|---:|---:|
| P | 13.979010086592508 | 1.1214102312749288 | 39.125909965750068 | 3.6667849065004012 | 4.3716972830890743e-14 |
| DF | 13.72369864844825 | 1.1170377665047011 | 34.527999187486905 | 3.5417705655437182 | 1.627525645321377e-13 |

この固定条件ではDF辞書のλとBが小さい。一般的な辞書優位・native価格・総資源優位を意味しない。
成分確率、次数weight/確率、exact係数和、decimal50値と丸め診断は[results.json](results.json)の`normalization`/`ideal_moments`に保存。

## 6複素信号と独立照合

全信号はraw φ内積/保存exact ν、Re=cosine、Im=+Im sine。主精度complex128。
Pauliは既存vector paired recurrence、DFは独立辞書和から既存dense finite polynomial、nom_*はcanonical V0からの第三対照。
rawは成分momentから別に評価し、nom/Bの除算で定義していない。

| 信号 | Re | +Im | u_num（各軸に適用するengineering目安） | u_bind |
|---|---:|---:|---:|---:|
| split | -0.16151114650191167 | 0.98687085022588705 | 5.152401618712194e-10 | 0 |
| nom_P | -0.16151114650257203 | 0.9868708502304625 | 7.5028810198475445e-10 | 4.9675308027182293e-14 |
| nom_DF | -0.1615111465025699 | 0.9868708502304594 | 8.7047408980394704e-10 | 5.0381430682361941e-14 |
| nom_star | -0.16151114650257067 | 0.98687085023045917 | 8.7047293258117721e-10 | 0 |
| raw_P | -0.0041279844134987408 | 0.025222949475023963 | 1.6742999870221807e-11 | 0 |
| raw_DF | -0.0046776862344546965 | 0.028581756066193158 | 1.9971132657892207e-11 | 0 |

raw attenuationと補正後finite biasは別の量。補正済み多項式へBを二度掛けていない。

| 独立照合 | 複素差の絶対値（点診断） | 合成不確かさU | 判定 |
|---|---:|---:|---|
| nom_P_vs_star | 3.5976408378630544e-15 | 1.6208107098739587e-09 | PASS |
| nom_DF_vs_star | 8.0825456208805303e-16 | 1.7409974038158066e-09 | PASS |
| nom_P_vs_DF | 3.7724105748269452e-15 | 1.6208622485274109e-09 | PASS |
| B_raw_vs_nom_P | 1.2010469701998595e-14 | 1.4055376869458198e-09 | PASS |
| B_raw_vs_nom_DF | 6.9057300056503065e-15 | 1.5602061238930213e-09 | PASS |

補正済み名目平均は独立構築で数値的に整合。点差が約1e-15でも、それを1e-15の厳密精度保証へ格上げしない。
B rawとnomも両方式でPASSし、異なるBが同じ補正平均と異なるraw平均に対応することを確認した。

## Signed outer/finite bias・分解能

outer=split−共通reference、finite_X=nom_X−split。下表は符号付き点差と、その差に適用する不確かさ。
referenceはexact rational区間を保持し、midpointは表示だけに用いる。reference幅をfinite差へ転用しない。

| 差 | 軸 | 符号付き点差 | u_num | u_bind | u_ref/transfer | 合成不確かさ | 判定 |
|---|---|---:|---:|---:|---:|---:|---|
| outer | real | 9.0712413647664469e-05 | 5.152401618712194e-10 | 0 | 3.6876877399833535e-14 | 5.152770387486192e-10 | RESOLVED_ENGINEERING |
| outer | imag | 1.4812500507499493e-05 | 5.152401618712194e-10 | 0 | 3.6923406684212493e-14 | 5.1527708527790357e-10 | RESOLVED_ENGINEERING |
| finite_P | real | -6.6036065504704311e-13 | 1.265528263855974e-09 | 4.9675308027182293e-14 | 0 | 1.265577939164001e-09 | UNRESOLVED |
| finite_P | imag | 4.5754511290851951e-12 | 1.265528263855974e-09 | 4.9675308027182293e-14 | 0 | 1.265577939164001e-09 | UNRESOLVED |
| finite_DF | real | -6.5822347572463968e-13 | 1.3857142516751663e-09 | 5.0381430682361941e-14 | 0 | 1.3857646331058487e-09 | UNRESOLVED |
| finite_DF | imag | 4.5723425046162447e-12 | 1.3857142516751663e-09 | 5.0381430682361941e-14 | 0 | 1.3857646331058487e-09 | UNRESOLVED |

outerの両軸はengineering分解能で解像した。finiteの両方式・両軸は不確かさ以下であり、符号・大きさを確定しない。
finite biasが0、または上表の点差が真のbiasであるとは主張しない。絶対値を足して一意な因果寄与率も作らない。
共通exponentialのexpm/spectral差、固有再構成・直交残差、γによる積/内積/phase/νの誤差、非unitary polynomial norm増幅を含む政策が分解能を定める。
solverや精度を変更して解像し直す計算は実施していない。

共通Hと保存DF matrixの点norm差は`5.6820458411432899e-15`。保存C exact-operand↔snapshot上端と丸めを加えたtransfer目安は`3.6869987759881697e-14`。
新信号のengineering目安と保存reference区間の包含保証は別のstatusである。区間端点・binding感度・exponential診断はresults.jsonに保全。

## 実行・人工検査・保全

人工検査は新規9＋既存P1/bit-copy回帰10＝**19 PASS**（科学SOURCE固定前）。
同じV/異なる辞書でのnominal一致とB差、signed orderの明示小event和、非可換S2、degree/r/q、scalar/ν一回、+Im、早期STOPを検査。
実H4へ人工検査を拡張せず、今回の独立H4 batchは1回だけ。

```json
{
  "scientific_batches_started": 1,
  "dictionaries": 2,
  "common_operator_bundles": 1,
  "signals": 6,
  "normalization_bundles": 2,
  "automatic_retries": 0,
  "sampling": 0,
  "RNG_replay": 0,
  "finite_bit_law": 0,
  "circuit_build_compile": 0,
  "SCF_FCI_DF_factorization": 0,
  "new_state_reference": 0,
  "M_A": 0,
  "N_C_G": 0
}
```
内部の明示処理カウンタ（全library内部callの監査ではない）：

```json
{
  "native_one_body_assemblies": 91,
  "spectral_svd": 171,
  "frobenius_norm": 1353,
  "DF_G_eigendecompositions": 10,
  "explicit_matrix_products": 684,
  "common_expm": 2,
  "common_spectral_eigendecompositions": 2,
  "finite_polynomial_matrix_products": 10,
  "signal_dense_matvecs": 184,
  "physical_complex_signals": 6,
  "normalization_bundles": 2,
  "paired_polynomial_matvecs": 160
}
```

wall `7.319491`秒、process exit0、全thread worker join済み。DF basis worker10、BLAS1 threadは小matrix/独立fragmentの処理に合わせた設定。共有設定の変更なし。
旧SOURCE/input/raw/result/grant/ledger/marker、Stage 1/P1/P2のSTOP、main/元研究branchを変更していない。
科学コードと入力のbyte検査・scope/件数・保存manifestの保全は[verification.json](verification.json)に記録する。

## 限界・次の担当・mandatory STOP

支持できるのは、この入力/分割/tupleでの辞書operator整合、補正nominal平均の整合、辞書依存のBとraw attenuation、outer差の解像まで。
finite biasの符号・大きさ、53-bit CDF/binary64角度の実効law差、native回路・価格・実compiledρ、shot/N/C/G、RPE全費用は未確定。
Stage 3では同じDF-native Gaussian一体保持blockを両方式へ接続する方針だが、その作用・phaseと共通E_Tの照合、event/basis/compilerの条件固定、実行許可が必要。
Bが小さいだけでは費用順位を判断できず、保持/event価格、basis切替、compile非加法性、精度headroom等の交絡が残る。

次の担当はユーザー：この限定結果を踏まえて、未解像finite成分への別scopeが必要か、Stage 3設計等を希望するか判断する。Codexは新しい明示指示の範囲だけを扱う。
**公開・remote照合後mandatory STOP。追加計算・詳細科学レビュー・主要claim採択・論文改訂へ移行しない。**

[CONTRACT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/495ed110ad8379663145791b788a63b64f05eedd/docs/research/h4_p3_stage2_nominal_signal_contract_20261102.md
[READY]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/495ed110ad8379663145791b788a63b64f05eedd/docs/research/h4_p3_stage2_execution_readiness_20261102.md
