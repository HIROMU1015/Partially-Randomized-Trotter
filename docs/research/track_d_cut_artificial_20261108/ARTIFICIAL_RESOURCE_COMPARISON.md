# 固定2人工caseのP0 full-wrapper比較

2-mode・3 native blockの人工path、q6・16 event/step、scalar .17/identity .04、h=.1/τ=.2。分子geometry/AO basis/DF rank/L_Dは非該当。準備費除外、Referenceの全CSWAP・位相補正と追加registerは含む。

Qiskit1.3.0、P0 opt1/seed17、basis=(rz,sx,x,cx)、approximation1、qubits_initially_zero=False、topology-free、既存MPFR160宣言global-phase compiler。O2 opt2/routing結果は混ぜていない。

## 全7資源

各行のX/Yはそれぞれ別wrapperで全7資源が同じ値だった。12結果行、10 unique IR compile、均衡caseの中央/選択X/Y計2件を同一IR cacheで共用。X/Yを独立path数として数えない。

| Case | 構成 | Cut | RZ count | CX count | RZ depth | CX depth | Total depth | Size | Qubits |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 不均衡 | Strong | 共通 | 1058 | 604 | 710 | 604 | 1636 | 2265 | 3 |
| 不均衡 | Reference中央 | 3 | 896 | 352 | 409 | 270 | 934 | 1871 | 5 |
| 不均衡 | Reference選択 | 2 | 896 | 352 | 283 | 190 | 646 | 1871 | 5 |
| 均衡 | Strong | 共通 | 1457 | 772 | 938 | 772 | 2197 | 3168 | 3 |
| 均衡 | Reference中央 | 3 | 1316 | 520 | 409 | 270 | 934 | 2795 | 5 |
| 均衡 | Reference選択 | 3 | 1316 | 520 | 409 | 270 | 934 | 2795 | 5 |

### 選択−中央（X/Y各）

| Case | RZ count | CX count | RZ depth | CX depth | Total depth | Size | Qubits |
|---|---:|---:|---:|---:|---:|---:|
| 不均衡 | 0 | 0 | −126 | −80 | −288 | 0 | 0 |
| 均衡 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

不均衡のtotal depthは934→646、約30.8%減。**限定された事前固定人工条件で、cut変更による追加depth削減が観測された。** Count/size/widthの追加改善は観測されなかった。均衡caseは中央を返したため追加効果0であり、隣接cutのcompiled最適性を証明してはいない。

### 選択−Strong（X/Y各）

| Case | RZ count | CX count | RZ depth | CX depth | Total depth | Size | Qubits |
|---|---:|---:|---:|---:|---:|---:|
| 不均衡 | −162 | −252 | −427 | −414 | −990 | −394 | +2 |
| 均衡 | −141 | −252 | −529 | −502 | −1263 | −373 | +2 |

Reference対Strongの効果には既知gadgetの利点も含まれる。候補Bに由来する追加効果は前表の「選択−中央」に限る。

## Proxy、lane、融合と原因の限界

偏りcaseのstep負荷は(33,33,33,18,18,18)。候補cut2/3/4のscoreは88/100/118で、compile前に2を選んだ。均衡caseは全33、score133/100/133で3を選んだ。このscoreの単位はcompiled gate depthではない。

| Case/cut | Proxy prefix/suffix | Raw A† depth | Raw B depth | Full P0 depth |
|---|---|---:|---:|---:|
| 不均衡/3 | 99/54 | 272 | 452 | 934 |
| 不均衡/2 | 66/87 | 421 | 303 | 646 |
| 均衡/3 | 99/99 | 452 | 452 | 934 |

A†はsuffixのactual inverse、Bはprefix。raw lane値は未compileの記述診断であり、standalone P0 lane/shell価格を取得した値ではない。新しいlaneだけのcompileは0。

Strongは全caseでq−1=5 step境界融合。Referenceは中央/選択とも内部q−2=4融合とglobal cutの1境界損失。cut移動によって不均衡caseで失われる場所がstep3後→step2後へ移るが、失う数は増えていない。closure/headは既存builderのまま、二層CSWAPも不変。

負荷をprefix/suffix間で移したこと、raw max lane depthが452→421となったこと、compiled longest-path depthが934→646となったことは観測事実。Full count/sizeが同じままdepthが減るため、操作のlane配置・依存経路が変わった効果を含む。Gaussian closure/最適化順序/CSWAP周辺の寄与を個別に分離する追加compileは実施しておらず、288の削減量の因果的な内訳はUNKNOWN。単純な「shell+max(lane)」の等式や全path保証へは拡張しない。C7-2の既知非加法性も維持する。

## 保存と負荷

[COMPARISONS.json](../../../artifacts/track_d_cut_artificial/2026-11-08/artificial_run02/COMPARISONS.json) がaxis別全結果。各wrapperのraw/compiled構造、weighted wire finish、DAG longest path、QPYはpayload archiveに保存した。判定と数値意味論は [correctness資料](IMPLEMENTATION_AND_CORRECTNESS.md) を参照。

run02全体14.14秒、10 compile taskを空き資源確認後に10 workerで実行（affinity128、投入時MemAvailable約1TB）。親peak RSS約1.10GiB、最大childのpeak RSS約326MiB。合算同時peakは未測定であり、単純合算を実測値としない。共有設定/venv/cache/他processを変更していない。q64の未完了出力は読まず、操作も0。
