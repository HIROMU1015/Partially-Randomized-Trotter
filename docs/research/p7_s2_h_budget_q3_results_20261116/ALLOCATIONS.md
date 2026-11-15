# 幅h・粗幅達成費用予算：Q3実行前固定配分

価格SOURCE ecf45508a63ddf75a4806b47179f5f5102fa90e0、予算435,481,090 RZ。全数値はQ3前の保存価格算術。

探索上限N0=[59,55,51,47,43,39,35,31,27,23,19,15,11]、各軸1以上。直近の末尾13,4は上限に使わない。

| 規則 | N_X=N_Y（m0〜12） | モデル費用 RZ | 予算残額 RZ | 直接部分 RZ | 外挿部分 RZ | 外挿割合 |
|---|---|---:|---:|---:|---:|---:|
| A_SHAPE | [21, 19, 18, 17, 15, 14, 12, 11, 9, 8, 6, 5, 3] | 421323290 | 14157800 | 5645442 | 415677848 | 98.660068851% |
| B_PRICE | [59, 55, 51, 47, 43, 39, 35, 31, 27, 16, 7, 2, 1] | 434203864 | 1277226 | 16000932 | 418202932 | 96.314880330% |

X/Y各軸は独立の価格行へ結合。整数breakpointは既存F.generateの最大予算内λ、N=max(1,min(N0,floor(λw)))、Aはw=N0、Bはw=N0/C_h。次breakpointの価格が予算超過することもINPUTSへ保存。

| m | X/Y単価 RZ | 区分 | AのX/Y各N | BのX/Y各N |
|---:|---:|---|---:|---:|
| 0 | 5820 | DIRECT_COMPILED | 21 | 59 |
| 1 | 11333 | DIRECT_COMPILED | 19 | 55 |
| 2 | 22359 | DIRECT_COMPILED | 18 | 51 |
| 3 | 44411 | DIRECT_COMPILED | 17 | 47 |
| 4 | 88515 | DIRECT_COMPILED | 15 | 43 |
| 5 | 176723 | ADOPTED_RULE_EXTRAPOLATION | 14 | 39 |
| 6 | 353139 | ADOPTED_RULE_EXTRAPOLATION | 12 | 35 |
| 7 | 705971 | ADOPTED_RULE_EXTRAPOLATION | 11 | 31 |
| 8 | 1411635 | ADOPTED_RULE_EXTRAPOLATION | 9 | 27 |
| 9 | 2822963 | ADOPTED_RULE_EXTRAPOLATION | 8 | 16 |
| 10 | 5645619 | ADOPTED_RULE_EXTRAPOLATION | 6 | 7 |
| 11 | 11290931 | ADOPTED_RULE_EXTRAPOLATION | 5 | 2 |
| 12 | 22581555 | ADOPTED_RULE_EXTRAPOLATION | 3 | 1 |

全26軸それぞれのN・C・積・physical ID・元canonical ID・元receipt・固定価格表locatorはINPUTSのproposals[*].price_terms。直接10軸/外挿16軸を維持し、長q外挿誤差上界・CIはnull。ここでのモデル費用は正式G_qualifiedではない。
