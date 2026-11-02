# P-D 既知モデルと baseline の位置づけ

日付: 2026-09-26  
状態: S0 固定済み

## 1. 既知として扱う内容

部分ランダム化 PF では、signed coefficient を持つ tail の実行負担が係数の符号付き和
ではなく絶対時間

`Gamma_R = sum_j |b_j|`

に依存すること、連続近似では short-step allocation を絶対係数に比例させること、leading
normalization/variance burden が `Gamma_R^2 / R` 型になることは既知モデルとして扱う。

したがって、次だけでは本研究の新規性としない。

- energy-minimal PF と `Gamma_R`-aware PF の選択が違うこと
- 負時間係数を absolute tail time へ入れること
- continuous proportional allocation を用いること

## 2. 本 S1 が追加で問う差分

S1 は、同じ物理時間と共通位相誤差予算の下で各 PF を公平に再最適化し、次を分離する。

1. stage/one-shot cost を入れるだけで判断できるか
2. leading `Gamma_R` model まで必要か
3. finite cutoff、integer allocation、exact normalization、finite truncation error が選択を変えるか
4. 差分の主因が tail model ではなく internal `H_D` 精度や nested/native 構成か

新規性候補は 3 または 4 が decision-relevant である場合に限る。

## 3. baseline の比較責任

| baseline | 入れるもの | 入れないもの |
|---|---|---|
| B0 | 総時間での deterministic phase bias | cost、tail burden |
| B1a | outer exponential-stage proxy | fragment 内部 work、shot inflation |
| B1b | realized component-action proxy | shot inflation |
| B2 | B1b、absolute tail time、continuous allocation、leading attenuation | integer rounding、finite cutoff error |
| B4 | B1b、integer allocation、exact paired normalization、finite truncation bound | compiled gate cost、実測 shot |

B3 という番号は選択 model に用いない。nested/native は構成軸として全 baseline に交差させる。

## 4. 解釈上の制限

`C_1shot` は解析上の component-action proxy であり、compiled RZ/CX count ではない。
`B_total^2` は model-predicted shot inflation であり、実際の RPE estimator で検証した shot
数ではない。S1 は最終 resource estimate ではなく、model-selection diagnostic である。

