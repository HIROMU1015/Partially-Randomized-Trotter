# H6 DF19 RPE D2：初期round・T3.2の限定評価結果

2026-11-10 JST。**明示実行許可後、固定4round・8候補を一回評価し、全8候補がideal条件付きround PASSとなった。** 新規4roundと保存D1の2roundを合わせ、m0..5まで6roundの条件付きNが揃った。全13round成功・全G・actual実装保証・厳密ground同定は未資格。

科学SOURCEは **`e0503b769358f239ef63245dda78ef1ee6ffce9b`**、承認対象は[公開D2計画][PLAN]の8候補・4round。結果の正本は[run01/result.json](run01/result.json)、[実行出力](execution.txt)、[人工7件](synthetic_tests.txt)、[本担当の照合](CHECKS.txt)、[6roundのNとmetadata転記](COVERAGE.json)。公開commitは科学SOURCEと区別する。計画・D1結果・旧SOURCEは保持した。

## 条件・実装・実行範囲

Linear H6、隣接1.00Å、STO-3G、保存tol-only 1e−8／DF19、保存weighted-Hermitian completion、signed weights・fragment順・scalar、12 system mode、Nα=Nβ=3のsector400。targetは保存DF-Hの同sector ground energy、保存数値stateを用いる条件付きpilot、P=0。CA=.00159360010199040 Ha、各方式αtotal=.05、βtotal/stat=.4/.2、h=.1、M12・13round、αaxis=.05/26を**D1からそのまま再利用**した。

prior [−9.695399206821989,−3.1355322139814668] Haと無shift alias判定も再利用。E_stateは保存solver energy −3.2360662799087647 Ha、residual 6.1060388500553216e−15 Ha。E_stateでpriorやdecoder branchを選んでいない。元分子HへのDF誤差を新しいenergy reserveへ足していない。

短時間T=.1/.2/.4はFull19 PFとPR5/R32/K2、全てm5_best/q_PF1。T3.2はFull19 PFとPR7/R16/K2、m5_best/**q_PF2・δ1.6**。これはRPE倍率q_RPE32とは別。signed 11 piece・ABS_TIME_PER_OUTER整数配分・各piece最低1を保持した。候補ID・各piece配分・全signal・allowanceは正本の各rowに記録した。候補追加、閾値変更、q/R/K探索は0。

保存coefficientsを既存`occupation_one_body`で400次元の作用表現へ変換し、既存m5 schedule／finite Taylor numerator／finite normalizerを再利用した。deterministic blockと内部exact tailには行列指数を使い、**新しい固有値問題は解いていない**。保存tail scalar/lambda_Rを使うため、tail抽出の再対角化もない。新Hamiltonian/DF/stateの生成、full Fock行列、trajectory、回路build/compile、価格fitは0。

## Referenceと新signal

T=.4 full-H referenceは同じH/state bindingを持つ旧固定δ研究から再利用した。T=.1/.2/3.2だけ新規にfull-H actionを行い、`expm_multiply`対`expm`を照合した。計3時点・6 state actionであり、T=.4/.8/1.6のtarget referenceは再計算していない。

| T | full-H reference | 数値証拠 |
|---|---|---|
| 0.1 | 0.94809472227+0.31798804632i | 2経路差 2.001483e-16 |
| 0.2 | 0.79776720479+0.60296557692i | 2経路差 2.220446e-16 |
| 0.4 | 0.27286502609+0.96205232578i | 保存値・元allowance保持 |
| 3.2 | -0.59732552691-0.8019988871i | 2経路差 5.5511151e-16 |

以下はcomplex signalとcorrected-referenceのsigned bias。表示を丸め、正本にはbinary64値を保持する。rawは独立にattenuated列を伝播し、corrected/Bを事後代入していない。PRのouter PF成分とfinite RTE成分を分ける内部exact-tail対照4件も保存した。

| m / arm | corrected | raw | total signed bias |
|---|---|---|---|
| 0 / PF | 0.94809472226+0.31798804635i | 0.94809472226+0.31798804635i | -9.0999430213e-12+2.7129132274e-11i |
| 0 / PR | 0.94809472226+0.31798804635i | 0.947481294+0.3177823044i | -1.3700485191e-11+2.5487667532e-11i |
| 1 / PF | 0.79776720424+0.60296557765i | 0.79776720424+0.60296557765i | -5.5123750009e-10+7.2932615502e-10i |
| 1 / PR | 0.79776720418+0.6029655776i | 0.79570462847+0.60140664895i | -6.1177174437e-10+6.7983951801e-10i |
| 2 / PF | 0.27286499813+0.96205233371i | 0.27286499813+0.96205233371i | -2.7962374582e-08+7.9296711419e-09i |
| 2 / PR | 0.27286499788+0.96205233248i | 0.2700544565+0.95214308098i | -2.8212214009e-08+6.7006835636e-09i |
| 5 / PF | -0.59728482147-0.80202822245i | -0.59728482147-0.80202822245i | 4.0705433569e-05-2.9335352996e-05i |
| 5 / PR | -0.59728482075-0.80202822127i | -0.58383897748-0.7839732743i | 4.0706159417e-05-2.9334169502e-05i |

## Disk・位相・半径・十分N

共通PF政策のu floor1e−9・倍率1、stage数依存roundoff、closure、log_B marginを使い、新referenceの2経路差を追加guardへ含めた。T=.4の元empirical allowance／marginは縮小しない。D1と同じcomplex box diskにreference対保存数値state位相差とT×residualを加え、normalization marginを含めてraw誤差へ接続した。新しいεsignal gridや有限時間αaxis=.025のshot関数は実行していない。

| m / arm | log B | B | raw error disk | raw半径下界 | βdet rad | corrected位相差 rad | 十分N_X / N_Y |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 / PF | 0 | 1 | 1.4461564e-09 | 0.999999998554 | 1.4461564e-09 | 2.8613944e-11 | **705 / 705** |
| 0 / PR | 0.0006472209698 | 1.00064743 | 1.4429085e-09 | 0.99935298699 | 1.4438426e-09 | 2.8520631e-11 | **705 / 705** |
| 1 / PF | 0 | 1 | 2.3293943e-09 | 0.999999997671 | 2.3293943e-09 | 9.1420833e-10 | **705 / 705** |
| 1 / PR | 0.002588783591 | 1.002592137 | 2.3238936e-09 | 0.997414562096 | 2.3299175e-09 | 9.1122954e-10 | **708 / 708** |
| 2 / PF | 0 | 1 | 3.0330042e-08 | 0.99999996967 | 3.0330042e-08 | 2.9064994e-08 | **705 / 705** |
| 2 / PR | 0.01035353021 | 1.010407313 | 2.9921085e-08 | 0.989699853163 | 3.0232483e-08 | 2.8970005e-08 | **719 / 719** |
| 5 / PF | 0 | 1 | 5.0176056e-05 | 0.999949823944 | 5.0176056e-05 | 5.0168507e-05 | **705 / 705** |
| 5 / PR | 0.02276886499 | 1.023030054 | 4.9046404e-05 | 0.97743934305 | 5.0175946e-05 | 5.0168382e-05 | **737 / 737** |

位相差の基準はexp(−i E_state T)であり、真groundの厳密同定ではない。全8候補で正半径、βdet+βstat≤.4が成立した。最も大きい合計位相予算はPF/T3.2の約.2000501761 rad。十分Nは固定配分と数値diskを仮定した計画値であり、最小Nや実測shot回数ではない。

最大raw/corrected vector closureは約3.34e−15、exact PF norm差は約1.78e−15、reference対数値state位相のcomplex差は最大約2.46e−14。これらの数値診断や経験的uは厳密な対真ground誤差上界ではない。**actual laneは全8件UNRESOLVED_REFERENCE_OR_ERROR、N_X/N_Y=null**。未知のgate/law距離やground同定誤差を0で補完していない。ideal laneのexact-law仮定とactualの未知量を区別する。

## D1との接続・価格coverage

以下のNは各軸の値。D1 m3/4は保存値の読取だけで、phase/radius/N・部分Gを再評価していない。task IDはD1の`H6_DF19_GROUND_CA_CONDITIONAL_D1`を保持し、stage欄でD2を区別する。D1とD2の科学SOURCE・結果出典はCOVERAGE.jsonで別々に辿れる。

| m / T | q_PF / δ | PR prefix/R/K | PF N / PR N（各軸） | 出典 |
|---|---|---|---|---|
| 0 / 0.1 | 1 / 0.1 | 5/32/2 | 705 / 705 | D2 |
| 1 / 0.2 | 1 / 0.2 | 5/32/2 | 705 / 708 | D2 |
| 2 / 0.4 | 1 / 0.4 | 5/32/2 | 705 / 719 | D2 |
| 3 / 0.8 | 1 / 0.8 | 5/32/2 | 705 / 765 | D1 |
| 4 / 1.6 | 1 / 1.6 | 7/16/2 | 705 / 721 | D1 |
| 5 / 3.2 | 2 / 1.6 | 7/16/2 | 705 / 737 | D2 |

今回の新8候補には、調査済み保存価格表で同一candidateの直接Cがない。**新C/Gは全てnull、N×C集計は0回。** D1のT=.8/1.6には直接価格があり、PR探索n9と独立確認n33の分離、有限cohortと母平均期待価格の違いも保持する。D1の価格・部分Gを今回の別時点へ代入しない。

価格担当へ渡す結合keyはCOVERAGE.jsonと各rowの`candidate_id, m, q_RPE, q_PF, T, δ, formula, prefix, R_outer, K, allocation, H/state/snapshot binding, compiler, control, axis, P=0`。期待compilerはD1と同じQiskit1.3.0／rz,sx,x,cx／opt1／seed17、symmetric_directional、測定付きfull Hadamard wrapper、hash `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。これは新8候補のcompile証拠ではない。

H4側の進行中価格調査には介入・重複調査をしていない。同じH6条件の保存cacheがなければ、H4側の**方法**を参照したH6 domainの適格なモデル又は必要点の価格が要る。CDF8/DF12/DF10の数値係数や保存価格を移植せず、全長回路compileを既定にしない。

## 検証・保全・残りの最小範囲

作用backendの人工7件は、既存spectral finite meanとの非可換2-level比較、独立したordered積、負時間/scalar一回、normalization、固有値solver非呼出し、q区別とbinding、actual未知量保持を確認した。**人工PASSと実H6科学結果を別記録**にした。実H6については候補/政策、保存値転記、bias/normalization、正半径とβ/α、既存Hoeffdingと独立式のshot整数一致、12 SOURCE fileと原入力hash、D1・計画の不変性を照合した。本担当自身の技術確認で、独立科学レビューではない。

campaign wallは**38.802秒**。480個の共通半step指数をcacheし、実資源観測（affinity64、load約4.19、available RAM約526.7 GB）に基づく59 threadで並列化した。BLASはこのprocess内1 threadとして入れ子を避け、候補列をbatch伝播した。cache計数は1,228,800,000 byteであり、process全体peakの測定値ではない。共有設定・他process・他worktreeを変更していない。既存project環境はPython3.11.0rc1、NumPy1.26.4、SciPy1.14.1で、依存installは0。

新規科学範囲はreference3時点/6 action、finite mean8件、内部exact-tail4件、scalar8件。新prior/horizon、D1 point再計算、H/DF/state生成、新eigensolve、sampling、回路build/compile、価格fit、G集計、RPE成功率simulationは全て0。SOURCE変更・科学retryは0、原本・SOURCE/input/raw/旧結果/grant/ledger/marker/KEEP/STOPは保持した。run01/progress.jsonは途中checkpointであり、最終状態の正本はresult.jsonである。

**残るroundはm6..12、T=6.4/12.8/25.6/51.2/102.4/204.8/409.6**。6roundで全13roundの成功・化学的精度達成・全Gを達成したとは扱わない。α=.05/26の予約を維持し、6roundへ再配分しない。

次の最小科学案は、成立したm5のrecipeを継承して**m6/T6.4、q_PF4/δ1.6、Full19 PF対PR7/R16/K2の2候補**を別途評価すること。新reference1時点/2経路、2 candidate finite mean、PR内部対照1、同じgeometry/α/βによるNを対象とし、新prior/state solve/価格fit/compileを増やさない。これは未実行・未承認の候補案であり、成立性や費用を予測した結果ではない。残り7roundをまとめる場合も、ユーザーが有限menuと実行範囲を指定した契約として扱い、広いq/R/K最適化へ広げない。

公開remote SHA・主要成果物の全文一致を確認後**mandatory STOP**。次の担当は**ユーザー**：残りsignalとH6価格Cの次範囲を判断する。**Codex**：別途許可までm6以降、価格fit・compile、全G、詳細科学レビュー・Main Claim変更へ進まない。

[PLAN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/47b3157ecd8c651c326b1bcfa3e8037534d289e1/docs/research/h6_df19_rpe_d2_preparation_20261110/REPORT.md
