# PR-2／PR-3最小pilot検証

実行日：2026-09-27  
status：`PR23_MINIMAL_PILOTS_COMPLETE_MANDATORY_STOP_SELECT_PR2`  
判定：`SELECT_PR2_PRIMARY_CANDIDATE`  
強制停止：`STOP_AFTER_PR2_PR3_MINIMAL_PILOTS`

## 1. 結論

結果前に固定した
[事前登録](research/pr2_pr3_minimal_pilot_preregistration.md)どおり、PR-2とPR-3を各一条件だけ
評価した。PR-2はH4 rank-12参照に対するrank 3/6/9圧縮と明示残差、PR-3は固定2-qubit
非可換系のtail qDRIFT level 4/8/16である。

- PR-2：`GO_PR2`。rank 3と6でdiscard biasとdeterministic savingのgateを満たし、
  qDRIFT screening workを戻してもrank-12 deterministic RZ work未満だった。
- PR-3：`STOP_PR3_VARIANCE_BACKBONE_DOMINATES`。外挿biasは大きく減ったが、Re/Im shot分散と
  各levelのbackbone workを戻すと通常PR最良点の7.557倍だった。
- 5項目比較はPR-2が8点、PR-3が6点で、PR-2を次の主題候補に選んだ。
- これはpilot内の方向選択であり、PR-2の一般的優位性、新アルゴリズムとしての新規性、
  full RPE総cost、H12または別条件への移送を確立しない。

追加数値は許可しない。次はSPRINT/GRADE等との差を限定した中心主張、development anchor、
独立条件を含む完成基準を文書で監査する。

## 2. 凍結条件とprovenance

| 項目 | 値 |
|---|---|
| preregistration SHA-256 | `d8562545981be6aff382ff7b01ff2f7f857dab1bd5622ffcdab61ee0f557e6e8` |
| result fingerprint | `0ee2c3b9bcb7ef433d429336d84c116eace3df150c2e6bd5747eb6850998b725` |
| artifact file SHA-256 | `65d5c12a129cd50f521b069b63be140f7bbee1eddf7ea8b859b18737dbb8d302` |
| artifact | `artifacts/pr2_pr3_minimal_pilot/2026-09-27/pr2_pr3_minimal_pilot_v1.json` |
| evidence status | local dirty-worktree validation、immutable external CIではない |

artifactは実行commit、実行前worktree status、command、platform、環境version、module・runner・
事前登録のSHA-256を保存する。計算後に事前登録は変更しておらず、保存値と現ファイルのSHA-256は一致した。

## 3. PR-2：圧縮＋random residual

### 3.1 条件

- H4 linear chain、1.0 Å、STO-3G、8 qubit。
- OpenFermion low-rank decompositionのrank-12 Hamiltonianを参照とする。
- 圧縮rankは3、6、9。constantとone-bodyを保持し、残りのDF blockをexact residualとする。
- deterministic workは二次PF one step、$T=0.1$、Qiskit basis `rz,cx,sx,x`、optimization level 0。
- residual identityはdeterministic phaseへ移し、非identity componentのsampling $\ell_1$ normを
  `residual_lambda_r`とする。
- $\epsilon_{\rm tail}=10^{-2}$に対するqDRIFT boundから全点でscreening step数は1となった。

rank-12参照ground energyは$-2.1663874486$ Ha、deterministic workは33,594 RZである。

### 3.2 結果

| rank | discard energy bias (Ha) | deterministic RZ | saving | $\lambda_R$ | components | expected RZ/sample | hybrid/reference |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 0.1690855260 | 9,364 | 72.126% | 0.583270425 | 324 | 974.653 | 0.307753 |
| 6 | 0.02406110385 | 17,428 | 48.122% | 0.0220789204 | 216 | 984.639 | 0.548093 |
| 9 | $8.1391\times10^{-8}$ | 25,284 | 24.737% | $1.5606\times10^{-7}$ | 108 | 701.000 | 0.773501 |

全点でlower-rank decompositionはrank-12 prefixと最大差0、residual dense再構成相対誤差は最大
$3.27\times10^{-16}$、sampling確率和誤差は最大$1.11\times10^{-16}$だった。rank 3と6は
discard bias $>10^{-3}$ Ha、RZ saving $\geq20\%$、hybrid/reference $<1$を満たすため、固定規則で
`GO_PR2`となった。rank 9はbias gateを満たさず、GOの根拠には使わない。

`GO_PR2`は、粗い圧縮をrandom residualで補うtrade-offがこのone-step screening単位で消えていない
ことを示す。controlled wrapper、shot、状態準備、outer repetition、RPEを含まないため、
最終的なresource advantageではない。

## 4. PR-3：tail-only extrapolation

### 4.1 条件

$$
H_D=0.9Z_0+0.7X_0X_1,
\qquad
H_R=0.31X_0-0.27Z_0Z_1+0.19Y_1.
$$

$T=0.8$、固定state、tail qDRIFT level 4/8/16、target complex RMSE 0.05とした。主参照は
同じpartial-$S_2$ backboneの中央だけをexact $e^{-iH_RT}$にした信号であり、exact-$H$との差
0.0360656はouter-PF biasとして勝敗から分離した。

### 4.2 結果

| tail steps | tail-exact bias | one-shot primitive work | RMSE条件を満たすtotal work |
|---:|---:|---:|---:|
| 4 | 0.02973924 | 8 | 11,488 |
| 8 | 0.01488941 | 12 | 11,940 |
| 16 | 0.00745115 | 20 | 18,300 |

$z_{\rm ext}=2z_8-z_4$のbiasは0.000452072で、$z_8$比96.96%減となりbias gateを通過した。
しかしweights $(-1,2)$のcost-minimizing Re/Im shot allocationではtotal workが86,820となり、
通常PR最良のlevel 4に対して7.557倍だった。固定規則により
`STOP_PR3_VARIANCE_BACKBONE_DOMINATES`とする。

これは外挿が数値的に失敗したという意味ではない。bias cancellationは明瞭だが、この条件の
target RMSEではvariance amplificationとbackbone再実行費用がそれを上回るnegative resultである。
qFLOのfull-random漸近保証をpartial-tail回路へ移送しない。

## 5. 強制停止後の比較

| 評価項目（0--2） | PR-2 | PR-3 |
|---|---:|---:|
| 限定した先行研究差 | 1 | 1 |
| partial randomizationの本質性 | 2 | 1 |
| 一文の中心結果 | 2 | 2 |
| negative resultの再利用性 | 2 | 2 |
| 完成可能性 | 1 | 0 |
| **合計** | **8** | **6** |

一方だけがGO/CONDITIONALなので、規則1から`SELECT_PR2_PRIMARY_CANDIDATE`となる。点数は
判断を補強する監査記録であり、論文価値の定量評価ではない。

## 6. 許可される次作業と禁止事項

許可するのは、PR-2の中心主張、development anchor、独立条件を含む完成基準を文書で固定する
作業だけである。現在は次の計算を許可しない。

- PR-2の追加rank、geometry、basis、precision、split sweep
- PR-3の別model、別時間、別RMSE、別level
- PR-4/5/6の数値pilot
- H12、長RPE、最終compiled total cost

## 7. 実装検証

専用testは5件で、PR-3固定判定、synthetic DF residualのdense再構成とsample compile、主題選択の
強制STOP、payload改変検出、登録artifactの固定hashを確認した。実行時は`5 passed`。
これはlocal回帰検査であり、
別条件への科学的再現性または外部CIを意味しない。
