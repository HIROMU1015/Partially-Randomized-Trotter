# C/A/D・統合担当へ：PF6の費用予算3/4・固定12配分Q3結果

2026-11-15 JST。正本は[REPORT](REPORT.md)、[SUMMARY](SUMMARY.json)、[18点のCSV](RESULTS.csv)、[C向けjoin](C_RESULT_JOIN.json)。具体列・価格参照・軸別積・予算/slackは実行前固定[INPUTS](INPUTS.json)、[配分表](ALLOCATIONS.md)。

## 確認できた最小適合点

| 方式 | 公式・要求幅 | 配分規則 | 条件付き費用T（RZ） | 直接モデルMSE区間 Ha²（表示） |
|---|---|---|---:|---|
| Full PF | S2・16h/7 | B_PRICE：w=N0/C | **615,304,502** | [1.495783385949356167254620E−6, 1.495783385949356167254624E−6] |
| B0I p7 | S2・16h/7 | B_PRICE：w=N0/C | **435,481,090** | [1.506660616251764083585358E−6, 1.506660616251764083585362E−6] |

両点ともm0〜m12の各軸：

~~~text
N_X = N_Y = [59,55,51,47,43,39,35,31,27,23,19,13,4]
~~~

allocation IDはそれぞれFULL_PF_S2_16h_over_7_B_PRICE_COST_BUDGET_3O4_01、B0I_P7_S2_16h_over_7_B_PRICE_COST_BUDGET_3O4_01。物理候補・元/total configuration・13round scheduleは元S2幅16h/7のまま。両列の一致は独立した費用生成の結果であり、方式間の共通Nという追加条件ではない。

これは元N0の6点＋新12点、**各方式9点の有限集合内で確認できた最小適合費用**。他の幅・配分族・N0超の振替を含む最適性や、十分N・物理的RMSE保証ではない。元基準点はFull PF S2 838,693,580 RZ／p7 S2 593,555,352 RZを保存引用し、REPORTで全18点を併記した。

## 実行・保存の状態

- 科学実行SOURCE：**06f4210e1bdfd5b07f42b941fa22d17e228533fc**、caller scripts/run_pf6_cost_budget_q3_fixed.py。結果公開SHAとは別。INPUTS・caller・実費用/slackをcommit/pushした後に起動した。
- β=3/4、形保持Aと価格振替B、各候補2件・計12件を各1回実行。重複・N0一致・再利用可能な新配分の保存結果は0件。旧6 Q3と全13round信号・価格は再実行していない。
- 新12件は**適合10／モデル不適合2／INCONCLUSIVE0／UNKNOWN0**。不適合は両方式のM5・B_PRICEで、下界>τ。数値gate失敗とは区別し、別列やretryを追加していない。
- 全件で新count格子の閉包、質量・残余会計、上下界/moments・exact閾値の検査と原API/RESULT整合を確認。worker自身が原API・3種partials・RESULT・RESOURCE_FINAL・PERSISTEDを保存。caller/記録済み子process全て終了code0。
- caller wall259.126秒、親7並列・各8 DP worker/batch8。CPUはjob区間差分、ru_maxrssはprocess-lifetime high-waterで、pool再利用時は過去jobの最大も含み得る。sampled subtree RSSを厳密な物理RAM peakと扱わない。
- 旧PF6のAPI return・CPU/peak RAM欠測はそのまま。元component適合と部分保存・運用UNKNOWNを、新12件の完全保存と混同しない。

## Cへの結合

C_RESULT_JOINは新allocationごとの元configuration、total view、binding、元signal locator、m/axis、physical ID、元/total candidate ID、N、固定B2価格行と区分、Q3 statusを対応付ける。元price生成ID・receipt・SOURCEは付け替えない。原API/RESULTのexact有理数がMSE正本で、SUMMARYの小数端点は表示用である。

価格は1002f1dfeaのAXIS_TERMS、**96直接＋60条件付き閉包**を引き継ぐ。閉包の固定compiler/障壁・反復・終端構造等の前提、価格誤差上界/CI=null・2%保証なしは変わらない。量子Nと価格標本数・normalizationは別で、二重乗算しない。Tは条件付き固定配分会計で、G_qualifiedへの入口を緩めるものではない。

元full DF10 solver energy proxy、Z1＋T1 total decoder、保存deterministic rawとideal IID二項、全事象平方loss、exact τを維持した。ground/state/target包含、実sampler移送、数値certificateは未認証。AδE/δE²を再加算せず、元Minkowskiは別参照。sufficient_N・global_RMSE_bound・G_qualified・方式全体winnerはnullのまま。

## ロードマップへの差分

ロードマップ本体は統合担当が更新する。

1. 現6幅で配分を選べる初回有限batchが完了。各方式の初回比較候補は今回もS2幅16h/7だが、評価済み最良配分と条件付き費用は上表へ更新できる。
2. 旧N0の費用と結果は達成済み基準点として残す。PR側との将来の比較では、どの配分・結果SOURCEを採用したかを明記し、旧N0費用だけを決定論PFの最良値として使わない。
3. 高次PFも同じ費用challengeで適合を直接確認した点がある。M5の価格振替不適合は、その配分の結果であり公式・幅全体の棄却ではない。
4. 別幅signal/価格の不足は7f27e18fの計画のまま。近い粗幅S2のm4〜m11、Y4のm6〜m11等を今回取得したことにはしない。幅・Nを含む大域最適性は未確認。

追加のshot族・β・別幅、PR/Full-Hとの新費用比、価格取得・G比較を開始する指示ではない。全12評価の終了・公開確認後 **mandatory STOP**。元SOURCE/raw/結果/grant/ledger/marker/KEEP/STOP、他担当の環境を保全する。
