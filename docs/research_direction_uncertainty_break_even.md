# N07/P03 不確かさ台帳・状態準備break-even

## 1. 目的

M06/L08後の候補比較について、異なる不確かさを一つのerror barへ潰さず、状態準備を除外した
比較がどこまで有効かを明示する。N07では不確かさの種類・適用範囲・合成規則を分離し、P03では
1 shot当たりの状態準備コストをパラメータ化する。

新しい回路compileやstatevector計算は行わず、WP01-D/C07、M08再集計、M06/L08
compiler-transferのfingerprint済みartifactを再集計した。

## 2. 固定条件

- H4 linear chain、1.0 Å、STO-3G、8 qubit、DF rank 12
- `CA/10`、`delta_time=0.02`、候補`L_D=3,12`
- 状態準備なしのcompiled RZを基準costとする
- WP01-D/C07で固定したshot数：`L_D=3`は13,538、`L_D=12`は11,162
- shot差：`L_D=3`が2,376多い
- opt1、opt2 focused、opt2一様比率反実仮想を別scenarioとして保持する

## 3. N07不確かさ台帳

| 誤差源 | 分類 | 現状 | 合成・解釈 |
|---|---|---|---|
| calibration sampling | sampling uncertainty | localで数値化済み | 共有fitを独立roundとして平均せず、保守的な和を使う |
| affine proxy discrepancy | model bias感度 | `L_D=3,r=32,q<=32`まで数値化 | sampling SEと二乗和にせず、対称discrepancy scenarioとして加える |
| compiler context | systematic context感度 | opt1/opt2を部分的に比較済み | compilerごとに別結果として示す |
| q>32移送 | 未測定外挿 | 未解決 | 5%・25% scenarioを残す |
| opt2の`r<32` | 未測定compiler domain | 未解決 | focused結果と一様比率反実仮想を分離する |
| state preparation | 除外cost | P03でパラメータ化、未測定 | `P*N_shots`を加える。候補で状態が違えば`P3,P12`を分ける |
| 別snapshot/backend/noise | 外部移送・実行 | 未測定 | 数値error barを割り当てない |

直接RZ relative SEはopt2でも最大1.387%だったが、compiler変更では候補のcost減少率が異なり、
focused区間が重なった。従って、現在の判断を支配するのは単なるMC標本誤差ではなく、
`r<32`を含むcompiler contextとq>32移送である。

## 4. P03共通状態準備コスト

両候補が同じ1 shot当たり準備コスト`P`を持つ場合、

```text
G3(P)  = G3_no_prep  + 13,538 P
G12(P) = G12_no_prep + 11,162 P
```

となる。`L_D=3`は2,376 shot多いため、非負の`P`は常に`L_D=3`の点推定上の利点を縮める。

### 4.1 点推定break-even

| cost context | no-prepでの`L_D=3`利点 | 共通`P`の点推定break-even |
|---|---:|---:|
| opt1 WP01-D/C07 | 2.353312e11 RZ | 99,045,126 RZ相当/shot |
| opt2 focused | 1.118320e11 RZ | 47,067,344 RZ相当/shot |
| opt2一様比率反実仮想 | 1.375839e11 RZ | 57,905,665 RZ相当/shot |

break-even未満では点推定は`L_D=3`、超えると`L_D=12`が低い。これは点推定だけの交点であり、
interval superiorityを意味しない。

### 4.2 区間を含むbreak-even

| scenario | P=0で区間分離 | `L_D=3`区間が確実に低い共通P | `L_D=12`区間が確実に低くなるP |
|---|---|---:|---:|
| opt1 M08 selected共通幅 | 分離 | 0--34,208,330 | 163,881,922以上 |
| opt1 local 5% | 僅かに分離 | 0--640,843 | 197,449,409以上 |
| opt1 transfer 25% | 重なる | なし | 462,341,285以上 |
| opt2 focused selected実測幅 | 重なる | なし | 99,043,760以上 |
| opt2 focused観測RZ最大幅 | 重なる | なし | 110,614,617以上 |
| opt2一様比率・selected幅 | 分離（反実仮想） | 0--6,799,763 | 109,011,567以上 |

opt1 M08幅はq<=32の`L_D=3`測定幅を両候補へ共通適用した反実仮想である。またopt2一様比率は
未測定`r<32`へ平均比を移した反実仮想であり、直接証拠ではない。直接証拠を尊重したopt2 focused
ではP=0ですでに区間が重なるため、compilerをまたいで`L_D=3`区間が確実に低い非負のP範囲はない。

## 5. 候補別状態準備コスト

候補ごとに準備状態または回路が異なる場合は、共通Pを使わず、

```text
G3  = G3_no_prep  + 13,538 P3
G12 = G12_no_prep + 11,162 P12
```

とする。opt2 focused点推定の境界は

```text
13,538 P3 - 11,162 P12 = 111,832,010,143.5 compiled RZ
```

である。この二次元境界より片側だけを、測定していない`P3,P12`の大小関係で自動採用しない。

## 6. 判断

- N07：sampling、model discrepancy、compiler、長q移送、状態準備、外部移送を分離した。
- P03：状態準備を測定済みとはせず、共通Pと候補別`P3,P12`のbreak-evenを得た。
- no-prep点推定では全scenarioで`L_D=3`が低い。
- ただしcompiler-robustなinterval superiorityは確立しない。
- 頑健判定は`undetermined_under_compiler_transfer_and_preparation_sensitivity`とする。
- 次はWP11相当の限定的な研究方向統合を行い、そこでfull opt2再最適化または外部instance pilotの
  どちらか一件に進む価値があるかを判断する。

本解析は状態準備回路、backend、noise、q>32、`L_D=3,r<32`のopt2直接compile、外部snapshot、
最終総costまたは科学的優位性を検証したものではない。

## 7. 証拠

- artifact：
  `artifacts/research_direction_uncertainty_break_even/2026-09-23/n07_p03_uncertainty_break_even_v1.json`
  （fingerprint `ff70308a64798c6ba8c9d20533c9e9e8c614e58c0d433dd861b7de45ac70c32d`）
- 実装：`src/trotterlib/research_direction_uncertainty_break_even.py`
- runner：`scripts/run_research_direction_uncertainty_break_even.py`
- test：`tests/test_research_direction_uncertainty_break_even.py`
- test結果：専用`2 passed`、変更後のlocal全suite `540 passed, 4 warnings`
  （warningは既存grouped-UWC test由来）

証拠statusはlocal dirty-worktreeであり、immutable CIまたは外部再現ではない。
