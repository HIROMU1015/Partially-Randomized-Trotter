# Codex C向け：E[S]とE[C−S]を結合するための条件

参照：[REPORT](REPORT.md)、[SCIENCE_PLAN](SCIENCE_PLAN.json)。

DはCの実行結果を待たずに残差側を準備した。CのK0/K1 runner・branch・cache・結果には介入していない。

## 1. Dの方法が成立する条件（Cとは独立）

- SとCが**同一path**（event digest・seed・cohort・index）・**同一軸**・**全wrapper**で取得されていること。
  - Sは、C1 `Accounting`による`r_price_phase1_science.structural`のword（head→tail→reverse、scalar一回、Y sdg、H/measure、finish一回）で数える。
  - Cは、同じrequestsの`compose`→`build_native_hadamard_wrapper(include_measurement=True)`→compiler `1239304c…`で得る。
- C1 `Backend.price_requests`がこの対を同じ呼出しで作る。新しい取得経路は不要である。
- forced rare pathは別stratumとして扱い、normal cohortへ混ぜない。

## 2. CのE[S]と結合する条件

| 条件 | 内容 |
|---|---|
| identity | 同じB canonical configuration（`587d20f7…`）、round、軸、physical candidate、law_binding、profile |
| S quantity | full wrapperのE[S]で、Bの構造envelope（`MEAN_STRUCTURAL_RZ_COUNT_NOT_COMPILED_C`）として受け入れられていること。K1はK0 PASSが前提 |
| S word | K1 kernelの遷移・報酬が、C1 `Accounting`（Dの対で使うS）とparityであること。wordが違えばC−Sの基準がずれる |
| law | E[S]のlaw（ideal descriptor）と、Dの標本path（PCG64 CDF sampler）のlawが一致すること。距離が未資格なら、Dは`S_AND_D_LAW_DIFFER_OR_UNDECLARED`を付けて未資格のまま扱う |
| 残差model | 1つだけ（paired direct D̄、C1 affine、local C−Sのいずれか）。C1 affineとlocal/paired Dを重ねない。raw κ(C)を足さない |
| 不確かさ | E[S]の数値区間、D̄の有限cohort分散、order2 stratum、law距離、長q/R systematicを別欄で持つ。S–D間の共分散は、E[S]がlaw期待値ならS側の標本誤差なし。それ以外はnull |
| 欠測 | E[S]またはE[D]がnullなら、E[C]はnull。0で補わない |

## 3. CのK1四値について

- 公開された場合も、`FULL_WRAPPER_MEAN_STRUCTURAL_RZ_COUNT`（E[S]）であり、compiled Cではない。
- Dの`combine`はE[S]を入力として受けるだけで、E[C]を資格しない。

## 4. Cへの提案（任意、ユーザー判断）

- SOURCE上、compiled側の制御ZZ回転は`RZZGate.control(1)`経由で**CCX·CRZ·CCX**になる。一方、`Accounting`の`crzz`は**CX·CRZ·CX**として数える（REPORT §3(a)）。
- この差が単位compile（P0）で確認された場合、二つの選択肢がある。いずれもC1のSや保存結果は変更しない。
  - 実loweringを数える別の構造量S′を用意する。そうすれば、C−S′はcompiler文脈の残差に近づく。
  - 機序項を残差modelで扱う。
