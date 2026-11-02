# P-D energy係数・random-tail負担Pareto監査

実施日：2026-09-25 JST

## 結論

固定した5つの対称Product Formula（PF）について、energy biasだけで選ぶ公式と、random
tailへ掛かる絶対時間

\[
\Gamma_R=\sum_j |b_j|
\]

およびstage数を併記して選ぶ公式が食い違う例を、developmentの`L_D=3`と事前固定した
blind holdoutの`L_D=4`の両方で確認した。判断点`delta=0.4`、energy tolerance
`1e-6 Ha`では、energy-only選択は両splitとも`8th(Morales)`、tail-aware選択は
`4th(new_2)`である。後者は前者より`Gamma_R`が56.702%小さく、full exponential stage数も
35から11へ減る。

従ってP-Dは
`advance_pd_as_conditional_candidate_pending_signed_time_and_inner_hd_validation`、すなわち
**条件付き研究候補**へ進める。ただし、これはexact two-block外側公式で得たpilot結果である。
負係数に対応するfinite-RTE sampled operatorとcontrol位相、実際の多数fragmentからなる
`H_D`内部近似誤差をまだ含めていないため、高次partial-RTEの実用的優位性や最終総costは
結論していない。

後続のP-D現実化Go/No-Goは2026-09-25に完了し、D1--D3を全て通過した。現在の判断と
限界は[P-D現実化結果](research_direction_pd_realization.md)を正本とする。本書の条件付き判断は、
後続gate前のpilot履歴として保持する。

## 条件と事前固定

- 系：H4 linear chain、1.0 Å、STO-3G、8 qubit、4 electron sector、DF rank 12
- 入力：固定Hamiltonian snapshot、sector dimension 70
- development：`L_D=3`。delta窓を決めるため、expected-task固定前に候補gridを確認した
- blind holdout：`L_D=4`。expected-task固定前にはP-D候補gridを確認していない
- 候補：`2nd`、`4th`、`4th(new_2)`、`8th(Yoshida)`、`8th(Morales)`
- direct delta：0.025、0.05、0.1、0.2、0.25、0.32、0.4
- 判断点：`delta=0.4`、energy tolerance `1e-6 Ha`、target branch weight 0.9995以上
- finite-RTE診断：1 outer step当たり総short-step数64、Taylor cutoff 2

`L_D=3`をblind transferの証拠には使わない。計算前に固定した条件と探索開示は
[事前登録](research/pd_energy_tail_pareto_preregistration.md)に記録する。

## K01：候補registryと次数

非可換3次元toyについて、局所operator errorは期待値`p+1`、固定総時間を反復したglobal
operator errorは期待値`p`を許容幅0.35で満たした。固有位相biasのslopeはstate-specificな
高次相殺を含み得るため、registry gateには使っていない。

| 公式 | 次数 | 局所operator slope | global operator slope | `Gamma_R` | full stages |
|---|---:|---:|---:|---:|---:|
| 2nd | 2 | 2.9956 | 1.9990 | 1.0000 | 3 |
| 4th | 4 | 4.9795 | 3.9821 | 4.4048 | 7 |
| 4th(new_2) | 4 | 4.9931 | 3.9945 | 2.3163 | 11 |
| 8th(Yoshida) | 8 | 8.8352 | 7.8366 | 20.8416 | 31 |
| 8th(Morales) | 8 | 8.9235 | 7.9106 | 5.3497 | 35 |

`4th(new_2)`は同梱2026 PDFの8桁係数を使う実用次数候補であり、ここでは厳密な記号的
order proofを主張しない。Morales係数は同梱arXiv:2210.15817v1 Table IIへ固定した。

## K02：内部高次化と全体高次化の区別

`H_D=A_1+A_2`、`H_R=B`の非可換toyで、`H_D`内部だけを四次化しても外側がStrangなら
局所operator slopeは2.9949のままだった。これに対し、`H_D/H_R`二block全体の四次化は
4.9795、`A_1/A_2/H_R`三term全体の四次化は4.9664だった。

従って、既存の`H_D`内部高次PFと、P-Dで問う外側`H_D/H_R`係数設計は同一ではない。
後続検証では両誤差を同時に戻す必要がある。

## H4でのselection reversal

判断点`delta=0.4`のabsolute energy biasは次のとおりである。

| split | 2nd | 4th | 4th(new_2) | 8th(Yoshida) | 8th(Morales) |
|---|---:|---:|---:|---:|---:|
| `L_D=3` development | 4.9040e-4 | 6.3723e-5 | 8.9360e-7 | 5.8957e-6 | 1.6524e-11 |
| `L_D=4` blind | 2.7926e-6 | 1.7492e-7 | 2.4443e-9 | 1.6540e-8 | 4.4853e-14 |

| split | tolerance内候補 | energy-only | tail-aware | `Gamma_R`減少 | stage変化 |
|---|---|---|---|---:|---:|
| `L_D=3` | new 4th、Morales 8th | Morales 8th | new 4th | 56.702% | 35 → 11 |
| `L_D=4` | standard/new 4th、Yoshida/Morales 8th | Morales 8th | new 4th | 56.702% | 35 → 11 |

両splitのPareto集合は`2nd`、`4th`、`4th(new_2)`、`8th(Morales)`で、
`8th(Yoshida)`は非劣位ではなかった。全direct unitaryの数値unitarityとtarget branch
weight gateも通過した。

## finite-RTE解析診断

各tail occurrenceを独立RTEとして、short-stepをequal配分した場合と`|b_j|`比例で整数配分した
場合を比較した。全行で比例配分のlog-normalizationがequal配分以下だった。
代表的な`delta=0.4`のattenuationは次のとおりである。

| split | 公式 | equal | `|b_j|`比例 |
|---|---|---:|---:|
| `L_D=3` | 4th(new_2) | 0.995261 | 0.995444 |
| `L_D=3` | 8th(Yoshida) | 0.590472 | 0.680479 |
| `L_D=3` | 8th(Morales) | 0.969992 | 0.975444 |
| `L_D=4` | 4th(new_2) | 0.999894 | 0.999898 |
| `L_D=4` | 8th(Yoshida) | 0.988204 | 0.991407 |
| `L_D=4` | 8th(Morales) | 0.999319 | 0.999445 |

これは有限分布のnormalizationと打切りboundの解析監査であり、sampled operatorの実測、
compiled circuit cost、あるいは総costの下界ではない。

## 固定gateと判断

次の7 gateは全て通過した。

1. 係数registry・toy次数
2. 内部高次化と全体高次化の区別
3. finite-RTE配分の数値整合
4. development selection reversal
5. blind selection reversal
6. direct unitaryの数値unitarity
7. target branch weight

この結果から、P-Dには「energy精度を最小にする公式がrandom-tail負担も最小とは限らない」という
独立した差分がある。ただし、現時点の選択規則はpilot診断であり、実装済みpartial-RTE全体の
資源最適化規則ではない。

## 当時固定した次の停止点

このpilot完了時点では、次の一件を`PD-1`として同じ検証内で行うことにした。

1. 負のtail係数を含む有限RTE sampled operator、identity phase、通常/control branchを
   小行列oracleと照合する。
2. exact `H_D`を使う二block参照と、実際のfragment列で近似した`H_D`を比較し、内部誤差が
   selection reversalを保つか確認する。

このgateを通るまでは、compiled回路、長RPE、H12、全PF family探索へ広げないと固定した。
後続gateは固定条件を全て通過したが、そこで計算を停止し研究再設計へ進む。

## 証拠

- expected-task artifact：
  `artifacts/research_direction_energy_tail_pareto/2026-09-25/pd_energy_tail_expected_tasks_v1.json`
- result artifact：
  `artifacts/research_direction_energy_tail_pareto/2026-09-25/pd_energy_tail_pareto_v1.json`
- expected fingerprint：`f302dafce37fb90f3acfe32aa83edf563dd1609015a1b3d50972880e13407c7d`
- result fingerprint：`846362ab808e9b26e5648f7f9d12d541dd01a6f9c2954e45d9194b4dfef8d835`
- result file SHA-256：`018cd6ad1a908748700a35a0213cc9775ead98360d4b2fa6fcaf2164ec672286`
- 実行時間：1.059 s
- evidence status：local dirty-worktree、外部再現・immutable CIなし

専用testはartifact生成時点で`4 passed`である。全suiteの結果は
[VALIDATION_STATUS.md](../VALIDATION_STATUS.md)に記録する。
