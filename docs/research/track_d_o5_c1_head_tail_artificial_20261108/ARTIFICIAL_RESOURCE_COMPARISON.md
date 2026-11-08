# 人工P0全7資源：Strong/Referenceの公平な融合前後比較

固定2-mode/3-block q2_mixed/cut1、h=.1/τ=.2/K0、frame角.23/−.37/.41、scalar .17/identity .04。分子geometry/AO basis/DF rank/L_Dは非該当。P0はQiskit1.3.0/opt1/seed17/rz,sx,x,cx/approx1/initially_zero=False/topology-free/MPFR160。

## Full-wrapper compiled結果

| 構成 | axis | RZ | CX | RZ depth | CX depth | Total depth | Size | Qubits |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| S0 | X | 182 | 94 | 114 | 94 | 265 | 377 | 3 |
| S0 | Y | 182 | 94 | 114 | 94 | 265 | 377 | 3 |
| S1 | X | 177 | 92 | 111 | 92 | 258 | 366 | 3 |
| S1 | Y | 177 | 92 | 111 | 92 | 258 | 366 | 3 |
| R0 | X | 217 | 90 | 77 | 56 | 170 | 428 | 5 |
| R0 | Y | 217 | 90 | 77 | 56 | 170 | 428 | 5 |
| R1 | X | 212 | 88 | 74 | 54 | 163 | 417 | 5 |
| R1 | Y | 212 | 88 | 74 | 54 | 163 | 417 | 5 |

全8 wrapperはrawに対するfull-unitary位相付き一致とX/Y/coherence/leakageのpostcheckにPASSした。S0/R0の4 baselineはC7-2原本のhash・SOURCE・canonical raw IR・full phase・wire mapping・旧postcheck PASSを照合して再利用し、S1/R1の4本だけを新規compileした。X/Yは同じwhole pathの二軸で、独立標本とはしない。

| 資源 | S1−S0 | R1−R0 | R0−S0 | R1−S1 | 差の差 |
|---|---:|---:|---:|---:|---:|
| rz_count | -5 | -5 | 35 | 35 | 0 |
| cx_count | -2 | -2 | -4 | -4 | 0 |
| rz_depth | -3 | -3 | -37 | -37 | 0 |
| cx_depth | -2 | -2 | -38 | -38 | 0 |
| total_depth | -7 | -7 | -95 | -95 | 0 |
| circuit_size | -11 | -11 | 51 | 51 | 0 |
| qubit_count | 0 | 0 | 2 | 2 | 0 |

上の差はX/Yとも同じ。**両方式でRZ−5、CX−2、RZ depth−3、CX depth−2、total depth−7、size−11、幅差0。Referenceの相対差は全7量で不変**。既知Reference対StrongのCX/depth差を、今回の接口融合の追加利益へ二重計上しない。

## Rawとの違い・P0重複の限界

| 構成 | axis | RZ | CX（明示のみ） | RZ depth | CX depth | Total depth | Size | Qubits |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| S0 | X | 77 | 46 | 47 | 46 | 121 | 154 | 3 |
| S0 | Y | 77 | 46 | 47 | 46 | 122 | 155 | 3 |
| S1 | X | 71 | 46 | 44 | 46 | 115 | 145 | 3 |
| S1 | Y | 71 | 46 | 44 | 46 | 116 | 146 | 3 |
| R0 | X | 76 | 2 | 22 | 2 | 50 | 124 | 5 |
| R0 | Y | 76 | 2 | 22 | 2 | 51 | 125 | 5 |
| R1 | X | 70 | 2 | 20 | 2 | 46 | 115 | 5 |
| R1 | Y | 70 | 2 | 20 | 2 | 47 | 116 | 5 |

Rawは未展開XXPlusYY/CSWAP等を含み、明示CX countをnative最終価格とは扱わない。両方式ともnative命令sizeは9減り、P0後のsizeは11減った。raw RZ数は6減るがP0後は5減る。raw削減を単体価格へ換算して足した結果ではなく、実full-wrapper比較である。

各方式で、異frameの6命令の閉包を既存relative3命令へ、同frameの6命令のinverse pairをidentityへ置換した。中心・tail内部run・step境界・global cutは維持した。P0は元から局所相殺を行うが、この固定full wrapperでは増分を全回収していなかったといえる。

ただし同frameだけをon/offした別ablationや局所P0 compileは今回追加していない。従ってcompiled CX−2/depth−7を異frameと同frameへ個別に割り振れず、どのP0 passが何個消したかも未同定。両接口を含むこの一種類の規則全体での有限観測に留める。さらに新しいrelativeが別の境界融合を常に改善するという主張はしない。

S1/R1の幅は3/5を維持し、既存reference register・二CSWAP cascade・χ/readoutを含む。RZをTへ、topology-free depthを実機時間へ換算しない。主資源・新しいエネルギー精度/成功率は採択していない。

正本：[RESOURCES](../../../artifacts/track_d_head_tail_artificial/2026-11-08/run02_metadata_completion/RESOURCES.json)／[COMPARISONS](../../../artifacts/track_d_head_tail_artificial/2026-11-08/run02_metadata_completion/COMPARISONS.json)／[REUSE](../../../artifacts/track_d_head_tail_artificial/2026-11-08/run02_metadata_completion/REUSE.json)。
