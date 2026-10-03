# PR-2 matched-accuracy M2 held-out transfer契約 v1

作成日：2026-10-04
M1-B1 evidence commit：`8e0814e70c14ecf526444fac8a2142799610dc96`
M1-B1 result SHA-256：`71278113c32b26af0dbf6144a626237a0087478212f8a93fc908de3d4d52aee4`
M1-B1 validation SHA-256：`c9a05babed99cd1e80eaec5b58e47f25d74513c7ba2e5a00775cfd5959c37a0f`
status：`M2_TRANSFER_CONTRACT_FROZEN_EXECUTION_NOT_AUTHORIZED`

## 1. 目的と暫定claim

M1-B1はH4 linear 1.00 Å、STO-3G、DF rank 12、8 qubits、`T=0.8`のdevelopment条件で、
matched-accuracy化とactual full-wrapper compileによりfixed-`q`/proxy比較から方式判断が変わり、
intermediate DF-prefix partial randomizationがresource frontierに残ることを確認した。

これはdevelopment一条件のlocal resultである。rank 3、`q=1`を一般的最適構成とは呼ばず、
`r=4,K=2`と`r=8,K=2`の厳密winnerも決めない。本契約は、developmentで固定した方式判断を
H4 1.30 Åへ一度だけ移し、再最適化せずtransferabilityを判定するM2を定める。

本書、schema、source、test、zero-compute planはheld-out開封またはM2科学実行を認可しない。
science module/runner/testを別commitで固定し、独立review後のresult-prior authorizationが必要である。

## 2. frozen candidate集合

M1-B1 result/validationから次の5構成だけを、表の順で固定する。

| role | candidate | method | rank | q | r | K |
|---|---|---:|---:|---:|---:|---:|
| development actual Pareto partial | `B2-rank3-q1-r4-K2` | B2 | 3 | 1 | 4 | 2 |
| development actual Pareto partial | `B2-rank3-q1-r8-K2` | B2 | 3 | 1 | 8 | 2 |
| best discard reference | `B0-rank6-q1-r0-K0` | B0 | 6 | 1 | 0 | 0 |
| full deterministic reference | `B1-rank12-q1-r0-K0` | B1 | 12 | 1 | 0 | 0 |
| random-dominant reference | `B3-rank0-q8-r32-K4` | B3 | 0 | 8 | 32 | 4 |

held-outではrank、`q,r,K`、method、splitを探索、追加、除外、置換しない。developmentのcandidate
fingerprintは出所identityとして保持し、held-out snapshot/stateを含む実行candidate fingerprintは将来の
science source/authorizationで別に固定する。

## 3. accuracyとshot

M1と同じ次を使う。

\[
\epsilon_{\rm complex}=0.05,\qquad
\epsilon_a=0.05/\sqrt{2},\qquad \alpha_a=0.025,
\]

\[
\nu_a=B_{\rm total}\mu_a,\qquad
b_a=|\nu_a-z_{H,a}|,\qquad s_a=\epsilon_a-b_a,
\]

\[
N_a=\left\lceil\frac{2B_{\rm total}^2}{s_a^2}\log\frac{2}{0.025}\right\rceil.
\]

held-outのcorrected signal、bias、normalization、analytic axis shotsは再計算する。一方、method/split/
`q,r,K,T`、compiler、32 trajectory上限は固定する。`s_a<=0`、非有限値、branch/correctness gate不通過は
accuracy-ineligibleとする。developmentでeligibleだった構成がheld-outでineligibleならcandidate単位の
accuracy feasibility misclassificationとして保存する。

## 4. compiled metricとmateriality

point Paretoはdevelopmentと同じ6指標を使う。

1. `rz_count`
2. `rz_depth`
3. `cx_count`
4. `cx_depth`
5. `total_depth`
6. `circuit_size`

primaryは状態準備を除く

\[
G_{\rm RZ}=N_{\rm Re}E[C_{\rm cosine,RZ}]+N_{\rm Im}E[C_{\rm sine,RZ}]
\]

である。B2の10% materialityは

\[
R=\frac{\min_{j\in B2}G_{j,{\rm RZ}}}
        {\min_{j\in\{B0,B1,B3\}}G_{j,{\rm RZ}}}\le1.10
\]

で判定する。random candidateのpaired trajectory標準誤差をdelta methodで比へ伝播し、`point ± 2SE`を
engineering intervalとして保存する。formal confidence intervalとは呼ばない。

共通状態準備costは`G_RZ(P)=G_RZ(0)+(N_Re+N_Im)P, P>=0`の全pairwise crossingとlower envelopeを
secondary結果として保存し、terminal statusを結果後に動かすためには使わない。

## 5. 重大なcost underestimate

candidate/axis/metricごとの事前予測は、held-outで再計算したanalytic shot数に、development M1-B1で
固定した対応axisの1-shot expected compiled costを掛ける。

\[
G^{\rm pred}_m=N^{\rm held}_{\rm Re}E[C^{\rm dev}_{\rm cosine,m}]
                 +N^{\rm held}_{\rm Im}E[C^{\rm dev}_{\rm sine,m}].
\]

primary RZについて

\[
u=\max\left(0,\frac{G^{\rm actual}_{\rm RZ}}{G^{\rm pred}_{\rm RZ}}-1\right)>0.10
\]

を重大なcost underestimateとする。10%ちょうどは重大扱いしない。secondary 5指標にも同じ値を保存するが、
terminal判定へ直接使わない。重大underestimateを持つB2はtransfer supportに使用できない。

## 6. terminal status

runnerが出せるstatusは次の4つだけで、どれでも必ず停止する。

- `TRANSFER_SUPPORTED`：accuracy-eligibleかつ重大underestimateのないB2の少なくとも一件が、6指標point
  Paretoに残る、またはprimary ratioのupper `2SE`が1.10以下。
- `TRANSFER_NOT_SUPPORTED`：B2が全件accuracy-ineligible、eligible B2が全件重大underestimate、または
  usable B2がpoint Paretoに残らずprimary ratioのlower `2SE`が1.10を超える。
- `TRANSFER_INCONCLUSIVE`：必要なendpoint比較が成立しない、またはpoint-Pareto B2がなくratio intervalが
  1.10を跨ぐ。
- `IMPLEMENTATION_GATE_FAILED`：source/data identity、correctness、numerical、resource gateの一件でも不通過。

`TRANSFER_SUPPORTED`でも次段階を認可しない。H5/H6、別分子、96 trajectory追加、winner精密化、S3、長RPEへ
自動進行せず、研究方針の全面reviewへ戻す。

## 7. resource capとseed

B2二件とB3一件は各32 trajectoryを用い、各trajectoryをcosine/sineで共有する。B0/B1は各axis一件である。

\[
3\times32\times2+2\times2=\boxed{196\ \text{full wrappers}}.
\]

future workerは最大5、各BLAS threadは1。random seedは固定master seed、transfer configuration fingerprint、
trajectory indexだけからSHA-256で導出し、axisを含めない。追加96、adaptive extension、別seed救済は行わない。

## 8. zero-compute planと禁止事項

zero-compute runnerが読む科学artifactはcommit済みM1-B1 result/validation JSONだけである。held-out pathは
repository-relative literalとして書き込むだけで、resolve、stat、hash、loadしない。development NPZ、signal、
trajectory、circuit、compiler、quantum shot、molecular calculation、candidate search、GPUは全て0である。

次を禁止する。

- held-outでのcandidate探索、rank/`q,r,K`再最適化、閾値変更。
- development `r=4`対`r=8` winnerの精密化。
- 追加trajectory、別geometry、H5/H6/H12、LiF、新PF、長RPE、最終総cost。
- runnerによる研究方針の自動決定または次段認可。
- science execution sourceより先に、その未実装コードを認可すること。

次の一件は、このbundleの独立reviewである。承認後も、science module/runner/test実装、source commit固定、
execution authorization、再reviewを経るまでheld-outを開かない。
