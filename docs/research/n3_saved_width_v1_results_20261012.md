# N3保存証拠の事後保証幅評価 v1：一回結果

2026-10-12 JST。利用者の「次の作業に進んで」に基づき、凍結済みplanの一回評価を実施した。
**有限保存集合でOCCの幅が狭い入力2/8（S1の1base）、BASEが狭い入力6/8。独立有理数checker全8件PASS、Mandatory STOP。**
元科学実験のNO_ADDITIONAL_ACCEPTANCE_OBSERVED、追加保証0、2-base未達を保持する。
この報告はCodexの固定比較結果であり、中心テーマ採択・新規性・研究継続の科学判断は行わない。

## 固定条件と由来

source `3f295df0263059ce16d866f84c1132f8bace0670`、fingerprint `b190ea508afb9a10fad3d3dbfeb92d50f46ad5813baadd70965aa6fc5ad044d1`。
plan SHA `bd3f5f4a21deafb227a6794d3277ca3eab1d9a30c610236d42294103e91e53dd`、認可commit `8370128c94641f70f21113ea9a82c85cc33a08af`。
原payload `e36e37bf5413aeba1b94e31e1d7ea6dcbaaf297c`、primary seal `8b1e75e89da78c13b82f24896db387adec6e76597ec12669a7549279592d0d69`、input SHA `c8df923eebfd999bf389ece2eb1728e7897c90e0950b51137785c708404489f4`。
[authorization](n3_saved_width_v1_evaluation_authorization_20261012.md)、[結果前比較仕様](n3_saved_width_v1_comparison_specification_20261012.md)、[数学契約](n3_saved_width_v1_mathematical_contract_20261012.md)。

実対称one-body＋非負unordered density、一外部modeをemptyへ射影、固定粒子数。
C1=chain(full7/active6,ν2,external1,scale1,元δgoal.004)、R1=ring(8/7,ν3,external4,scale1.5,.006)、
L1=ladder(9/8,ν3,external0,scale.75,.003)、S1=star(10/9,ν4,external8,scale1.25,.005)。
各baseは同じ結合normを持つedge_pair/distributed_three。化学geometry/basis、DF rank/policy、split L_D、PF/time windowは適用なし。
全8入力をposthoc developmentとして扱う。保証対象は基底エネルギー射影差0≤a−E0≤δ、時間発展normやQPE/RPE shot誤差ではない。

各入力で同じ保存U,c_star（g=c_star−U>0）を全方法へ使用した。固定κを保持し、新しいμ/τ/Zを生成していない。
BASEは全3原下界行のNORM/FIVE/OLD_DIRECTのPSD-valid証拠のOR。元goalで不採択でもPSD-validなら含める。
OCC固有poolはrawの全保存selected spectralと途中spectral_attempts。fallback/継承コピーは0。
固定q=κU−sに対する下端max(0,κ−g,(sqrt(g²+4max(q,0))−g)/2)を128-bit外向き区間で囲み、独立多項式endpointと有理PSDで確認した。
表示小数で勝敗を決めず、OCC_hi<BASE_loまたは逆方向の厳密不等式だけを使用した。

## 幅比較

表は保存係数のenergy単位で丸めた安全側上端の表示。厳密区間、証拠ID、正の有理δ_proofはcase JSONに保存する。
縮小率は1−δ_OCC/δ_BASE、負値はOCCの方が広い。

| input | BASE saved | OCC all saved | 狭いpool | OCC縮小率（表示） |
|---|---:|---:|---|---:|
| C1_edge_pair | 0.00117914072202 | 0.00118821565104 | BASE | -0.769622% |
| C1_distributed_three | 0.00114127558648 | 0.00118821565104 | BASE | -4.112947% |
| R1_edge_pair | 0.00272056818527 | 0.00273324112293 | BASE | -0.465820% |
| R1_distributed_three | 0.00272056818527 | 0.00273317645277 | BASE | -0.463442% |
| L1_edge_pair | 0.00115517774857 | 0.00118334516364 | BASE | -2.438362% |
| L1_distributed_three | 0.00115517774857 | 0.00118323524342 | BASE | -2.428847% |
| S1_edge_pair | 0.0030888226617 | 0.0024579588333 | OCC | 20.424087% |
| S1_distributed_three | 0.0030888226617 | 0.00307821749914 | OCC | 0.343340% |

| input | NORM | FIVE | OLD-DIRECT | BASE unique数 | OCC all / selected数 |
|---|---:|---:|---:|---:|---:|
| C1_edge_pair | 0.00117914072202 | 0.00176928537708 | 0.00397720074014 | 21 | 1 / 1 |
| C1_distributed_three | 0.00117914072202 | 0.00114127558648 | 0.00397720074014 | 21 | 1 / 1 |
| R1_edge_pair | 0.00272056818527 | 0.00662668043893 | 0.006 | 11 | 1 / 1 |
| R1_distributed_three | 0.00272056818527 | 0.00557342997141 | 0.006 | 11 | 1 / 1 |
| L1_edge_pair | 0.00115517774857 | 0.00347818807081 | 0.002532671365 | 19 | 1 / 1 |
| L1_distributed_three | 0.00115517774857 | 0.00218869585359 | 0.002532671365 | 19 | 1 / 1 |
| S1_edge_pair | 0.0030888226617 | 0.00836288431149 | 保存PSD-validなし | 6 | 60 / 11 |
| S1_distributed_three | 0.0030888226617 | 0.00796101957939 | 保存PSD-validなし | 6 | 60 / 11 |

OLD-DIRECTのS1は有限保存集合にPSD-valid証拠がない。旧LMI族の数学的不可能性ではない。
全保存OCCとselected-only OCCの最小区間は全8入力で同じだった。途中証拠を加えたことによる最小幅の改善は観測していない。
unique候補総数240、原raw attempts371、保存spectral entry visits184（selected重複を含む）、invalid旧PSD除外105。
主比較のcoverage不足0、重なり未判定0、exact tie0。
[小さいcase summaries](../../artifacts/n3_saved_width_v1/2026-10-12/evaluation_audit/case_summaries/index.json)、[aggregate summary](../../artifacts/n3_saved_width_v1/2026-10-12/evaluation_audit/aggregate_summary.json)。

## 保存証拠から確認できる改善原因と限界

S1_edge_pairはBASE=NORMの約0.00308882266170に対してOCC約0.00245795883330、縮小約20.424087%。
S1_distributed_threeは同じBASEに対して約0.00307821749914、縮小約0.343340%。厳密縮小率区間はcase summaryへ保存した。
両方とも保存c0行のraw OCC証拠が最小を与え、κはそれぞれ約0.00145041750270、0.000318019550193。
元c0行ではoriginal_accepted=falseでも、保存PSD証拠自体は有効で、同じ強いc_starで幅を判定するという事前仕様により再利用できた。
強い下界への継承はBASEにも同じように適用し、fallbackコピーをOCC固有利益へ含めていない。
これらのqはそれぞれ約0.00137997562965、0.00173011770288で、NORMのβ²=0.00173611111111より小さい。
同じgでrootが単調増加するため、この保存証拠のq減少が今回の幅差を説明する。E_forced、真のactive基底状態、真のgapは数値評価していない。

C1/R1/L1の保存OCC最小はκ=0のspectral証拠で、有限精度の有理下界がNORM identityより弱く、BASEを超えなかった。
C1_distributed_threeではBASEの最小がFIVEで、ほか7件はNORMだった。
この結果はOCC条件族と最適化された旧条件族の厳密分離、旧族global optimum、新入力への移送、真の射影誤差の改善を示さない。
幅2件の利益は同じS1 baseに限られる。元2-base acceptance gateへ置き換えない。

既存原理との関係は凍結[数学契約第5節](n3_saved_width_v1_mathematical_contract_20261012.md)のKy Fan trace/Schurの整理を保持する。
今回新たな網羅的先行性調査は実施していない。保存集合の幅差を有用な研究課題へ結び付けられるか、強い既存族との差分が独立かはGPT科学判断事項。

## 費用・検証・資源

CPU32/affinity32、開始前available memory 55,567,331,328 bytes、Python3.12.3 stdlib、2 spawn workers、numeric threads各1。
支配費用は保存有理PSD/行列再構成で、凍結case-local cacheを使用した。新eig/optimizer/certificate/input/E_forced/旧batch rerunはすべて0。
実行wall 6.145470s、parent CPU 6.234912s、worker case CPU合計5.670028s。
max recorded worker peak RSS 65,732,608 bytes、sampled aggregate RSS peak 152,150,016 bytes、8GiB budget違反なし。
watchdog目標50ms、sample22、最大gap0.923657s。sampling間の瞬間peakは未測定。
case-local distinct PSD matrices合計719、width算術11650 operations、checker rational max1167 bits、width rational max289 bits、scaled整数max535 bits。
Fraction内部temporary integer全体の監視とは区別する。operand cap4096、scaled cap8192、候補512/case、精度128bitsを保持した。
evaluation output 538,339 bytes、最大shard 189,732 bytes、run.log 55 bytes、各出力・総出力cap内。wall/CPU time capなし。

親processのcase別独立監査8件PASS。さらに保存結果だけを読むCLI checkerも全8件PASS、元checker conditions合計5,284,307。
独立endpoint検査はroot生成を繰り返さず多項式を使用し、有限保存集合の完全性/PSD/κ/τ/Z/原pointer/比較labelを確認した。
sourceを変更していないため、準備時24 synthetic PASSをsource identity付きで再利用し、同じテストの再実行は行わない。
global manifest checkerは既存description欠落のFAILを保持し、今回のentry/prefix保全を別に確認する。immutable CIや外部科学再現とは呼ばない。

元producerのevent再構成見積り（[元費用表](n3_occupation_transfer_v1_results_20261012.md)）はS1_edge_pairでBASE-OR0.096838s/OCC operational0.390802s、S1_distributed_threeで0.099440s/0.392357s。
これは元δgoalでの運用経路再構成であり、今回の小さい幅をcold生成する費用ではない。事後評価wallとも分離し、速度改善・費用対効果・量子総資源の結論へ使わない。

## 保存・停止・次の担当

[一次result](../../artifacts/n3_saved_width_v1/2026-10-12/evaluation1/result.json)、[run audit](../../artifacts/n3_saved_width_v1/2026-10-12/evaluation1/run_audit.json)、[STOP](../../artifacts/n3_saved_width_v1/2026-10-12/evaluation1/MANDATORY_STOP.json)、[独立CLI監査](../../artifacts/n3_saved_width_v1/2026-10-12/evaluation_audit/independent_saved_verifier.json)。
実行attempt1/retry0。新source/旧pack/ledger/plan/旧result/STOPを保持し、専用HIROMU1015 branchへcommit/pushしてGitHubから独立再取得する。
新しい科学的知見は「保存集合のS1の2方向でstrictな幅差」に限る。次の担当はGPTの新結果に対する研究判断で、同じ旧結果の再レビューではない。
レビュー開始は別のユーザー認可に従う。中心テーマnull、次段実行false、Mandatory STOP。N1追加探索停止、N2保留、実分子/多外部/一般二体拡張は行わない。
