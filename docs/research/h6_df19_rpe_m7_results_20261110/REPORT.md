# H6 DF19 RPE m7：T12.8・固定2候補の結果

2026-11-10 JST。**明示実行承認を受け、m=7のPF/PR各1候補を一回評価した。両方ideal条件付きround PASS、十分NはX/Y各軸でPF705、PR846。** 保存m0..6と合わせ、8個の異なるround m0..7の条件付きNを取得した。actual保証・厳密ground同定・全13round成功・総RPE費用は未資格。

科学SOURCEは **`dcd6a7c87b6596fecf781db9a84306d9c18e615d`**。[固定計画][PLAN]・runner・数値kernelを変更せず実行し、科学retry・候補差替え・閾値緩和は0。数値正本は[run01/result.json](run01/result.json)、[実行出力](execution.txt)、[算術・保全照合](CHECKS.txt)、[coverageと価格担当handoff](COVERAGE.json)。結果公開commitと科学SOURCEは区別する。

## 固定条件・候補・入力binding

Linear H6、隣接1.00 Å、STO-3G、保存tol-only 1e−8 DF19・weighted-Hermitian completion、12 system mode、Nα=Nβ=3・sector400。保存DF-Hの同sector ground energyをtargetとし、保存数値stateの条件付き評価、P=0を維持した。元分子Hの厳密energyとの差を保証する評価ではない。保存H/state/snapshot、signed weights、fragment順、scalar、tail identity、normalization、lawの定義は不変。

CA=.00159360010199040 Ha、h=.1、m0..12の13round、各方式αtotal=.05、αaxis=.05/26、βtotal/stat=.4/.2 rad、prior [−9.695399206821989,−3.1355322139814668] Ha、energy shiftなし。D1/D2/m6の保存設定を転記し、prior/horizon/α/βを再計算・最適化しなかった。E_state=−3.2360662799087647 Ha、residual=6.1060388500553216e−15 Haはreference診断だけに使用し、priorやdecoder branchを決めるoracleには使っていない。

| arm | candidate ID | 固定構成 |
|---|---|---|
| PF | `5cc99f967c6a176c43fb` | Full19、4th m5_best、q_PF8、δ1.6 |
| PR | `47402cdbdb4fecab7a46` | prefix7、4th m5_best、R_outer16、K2、q_PF8、δ1.6 |

共通m7/T12.8、q_RPE128。既存11 signed pieceの順序・binary64値・負時間を固定し、PRのABS_TIME_PER_OUTER配分は`[1,2,1,2,1,2,1,2,1,2,1]`、1 outer全体で16・全128 event。q_RPE128、q_PF8、event128を別変数として保持する。各pieceへ16を重複適用しない。K2の既存paired numeratorは3次まで。

既存D2 action backend、m5 schedule、finite Taylor/normalizer、D1 geometry・disk・Hoeffdingを再利用した。prefix7の保存tail scalar=.0194666612606853、lambda_R=.1649333839054104を使用し、新しいtail解析・eigensolve・Hamiltonian/DF/state生成は0。保存係数を400次元sector作用表現へ復元し、full Fock行列は構築していない。H/state/snapshotの完全hashはresult.json/PLANとCOVERAGEのhandoffに保持する。

## Reference・finite mean・normalization

| T12.8 full-H reference経路 | signal |
|---|---|
| expm_multiply | -0.8359450051690894-0.5488132180742561i |
| expm | -0.8359450051690079-0.5488132180742231i |
| 独立経路差 | 8.790870419093639e-14 |

経路差は既存gate1e−10以内。表は表示桁を丸め、binary64正本はresult.jsonに保存する。以下のsigned biasはcorrected−referenceである。PRのcorrected/rawは独立した列を伝播し、raw=corrected/Bから事後生成していない。

| arm | corrected signal | raw signal | total signed bias |
|---|---|---|---|
| PF | -0.8358340949988196-0.548980142748266i | -0.8358340949988196-0.548980142748266i | 0.0001109101702698734-0.0001669246740099162i |
| PR | -0.8358340906758633-0.5489801392899269i | -0.7630737444766763-0.5011907688433163i | 0.0001109144932260886-0.0001669212156708566i |

PR exact-tail内部対照は **-0.835834095264912-0.5489801423129783i**、finite RTE成分は4.589048696601594e-09+3.023051364792195e-09i。outer PF/discard成分とfinite RTE成分は正本に分離し、diskではtotal signed biasを一回だけ会計した。

| arm | log B | B | attenuation A | log_B margin | vector closure |
|---|---:|---:|---:|---:|---:|
| PF | 0 | 1 | 1 | 5.0036419452226e-11 | 0 |
| PR | 0.091075459955989 | 1.0953516573278 | 0.91294881722237 | 2.1846706816262e-11 | 9.7460139616735e-15 |

scalar/identityは全時間で一回会計した。exact-tail norm差はPF 2.220446e-15、PR 6.6613381e-16で、既存gate1e−9以内。closureは数値診断でありactual law距離やrigorousな誤差上界ではない。

## Error disk・位相・半径・十分N

既存u floor・stage依存roundoff・closureにreference経路差を含め、両方式u_X=u_Y=1.000087908704191e-09を用いた。保存数値state位相へのreference complex offsetは1.434303738367548e-13、T×保存residualは7.815729728070812e-14。これらをcorrected diskに含め、normalization margin/closure込みのraw diskへ変換した。数値allowanceはempiricalであり、厳密ground同定誤差や未知actual implementation/law誤差を0に補完していない。

| arm | corrected disk | raw error disk | raw半径下界 | βdet rad | βdet+βstat rad | 十分N_X / N_Y |
|---|---:|---:|---:|---:|---:|---:|
| PF | 0.00020041324425288 | 0.00020041329429933 | 0.9997995867057 | 0.00020041329564094 | 0.20020041329564 | **705 / 705** |
| PR | 0.00020041275623004 | 0.00018296660871477 | 0.91276585061366 | 0.00020041277943298 | 0.20020041277943 | **846 / 846** |

| arm | 観測corrected半径 | 観測raw半径 | corrected位相差 rad | raw位相差 rad |
|---|---:|---:|---:|---:|
| PF | 0.99999891574661 | 0.99999891574661 | 0.00020040903346117 | 0.00020040903346117 |
| PR | 0.99999891023477 | 0.91294782232251 | 0.00020040851607946 | 0.00020040851607751 |

位相差はexp(−i E_state T)との比較。正raw半径とβdet+βstat≤.4が成立し、両方式`CONDITIONAL_ROUND_PASS`。既存Hoeffdingと独立式 `ceil(2/epsilon_axis² × log(2/alpha_axis))`、`epsilon_axis=rho_lower*sin(beta_stat)/sqrt(2)`で整数Nを照合した。これは条件付き十分shotの計画値であり、actual minimum Nや実測shot数ではない。有限時間評価のαaxis=.025は使用せず、未評価roundの割当も維持した。

**actual laneは両方式ともUNRESOLVED_REFERENCE_OR_ERROR、N_X/N_Y=null。** 厳密ground同定・rigorousなdisk資格・実装/law距離が未取得である。ideal laneのexact-law仮定とactualの未知量を区別する。

## m0..7のcoverageと価格担当へのhandoff

m0..6の14行は旧COVERAGEの保存値をそのまま転記し、新しいreference/signal/N/費用を再計算していない。m6 run01/run02は同じ一roundで、Nや統計標本数を合算しない。以下のNはX/Y各軸。

| m / T | q_PF / δ | PR prefix/R/K | PF N / PR N |
|---|---|---|---|
| 0 / 0.1 | 1 / 0.1 | 5/32/2 | 705 / 705 |
| 1 / 0.2 | 1 / 0.2 | 5/32/2 | 705 / 708 |
| 2 / 0.4 | 1 / 0.4 | 5/32/2 | 705 / 719 |
| 3 / 0.8 | 1 / 0.8 | 5/32/2 | 705 / 765 |
| 4 / 1.6 | 1 / 1.6 | 7/16/2 | 705 / 721 |
| 5 / 3.2 | 2 / 1.6 | 7/16/2 | 705 / 737 |
| 6 / 6.4 | 4 / 1.6 | 7/16/2 | 705 / 772 |
| 7 / 12.8 | 8 / 1.6 | 7/16/2 | 705 / 846 |

今回はCを取得・検証・model fit・予測せず、N×Cも集計していない。新m7のC/Gはnull。価格Cの存在やmodel資格は本成果で判定しない。`COVERAGE.json.m7_price_handoff`にcandidate ID・T/m/q_RPE/q_PF/δ/formula/prefix/R_outer/K・signed weights・allocation/event数・X/YのN/α・law model・H/state/snapshot binding・保存tail定数・normalization・P0を保存した。Bや古典価格標本数をNまたはCへ二重乗算しない。

価格結合の期待条件は測定付きfull Hadamard wrapper、symmetric_directional、Qiskit1.3.0／rz,sx,x,cx／opt1／seed17、compiler hash `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。これはm7回路のbuild/compile済み証拠ではない。保存contextの`binding.T=.8`は歴史的input reference時刻で、今回の発展時間はcandidate.T=12.8。別Codexの価格branch・worktree・SOURCEへ変更を加えていない。

## 実行範囲・保全・次の最小案

campaign wall **12.518秒**。共通半step指数120個をcacheし、観測affinity64・load0.3511・available RAM 527057498112 byteから63 threadを選んで並列化した。BLASはprocess内1 threadで入れ子を避けた。cache計数307,200,000 byteはprocess peak測定値ではない。独自の固定worker capや共有設定変更はない。

実callは新reference1時点/2 action、finite mean2、PR内部対照1、scalar2だけ。m8以降・候補探索・新eigensolve/SCF/DF/state生成・trajectory sampling・回路build/compile・価格・N×C・全RPE成功率simulationは0。科学SOURCE/計画16 file、原入力3 file/tail context、旧結果の保全、per-candidate JSONと最終正本の一致、disk/normalization/phase/α/β/N整数を照合した。[準備時の人工3件][TESTS]はSOURCEが不変のため保存記録を再利用し、同じ試験を反復していない。本担当自身の技術照合であり独立科学レビューではない。

run01/progress.jsonは途中checkpointであり、最終状態はresult.jsonを参照する。SOURCE/input/raw/結果/grant/ledger/marker/KEEPと過去STOPを保持した。固定科学SOURCEへの事後修正・科学retryは行っていない。

**残りはm8..12、T25.6/51.2/102.4/204.8/409.6の5round。** 各signal/N、全体phase/branch/failure資格、対応Cと全Gが不足する。8roundの条件付きNが揃っても、全13roundの化学的精度・成功確率・PF/PR総費用順位はまだ確定しない。

次の最小科学案は **m8/T25.6、q_PF16/δ1.6、Full19 m5_best PF対PR prefix7/R_outer16/K2の2候補**。同じ保存入力とbackendを使い、新reference1時点/2経路、finite mean2、PR内部対照1、phase/radius/N2だけを評価する案である。未実行・未承認で、成立性やNを予測したものではない。追加候補や残5roundの一括探索へ自動展開しない。

公開remote SHAと主要成果物の全文一致を確認後 **mandatory STOP**。**ユーザー**：m8以降の有限科学範囲を判断する。**価格担当Codex**：このhandoffを別途許可されたC側作業で利用する。**本担当Codex**：追加許可までm8以降・価格評価・詳細科学レビューへ進まない。

[PLAN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dcd6a7c87b6596fecf781db9a84306d9c18e615d/docs/research/h6_df19_rpe_m7_preparation_20261110/REPORT.md
[TESTS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dcd6a7c87b6596fecf781db9a84306d9c18e615d/docs/research/h6_df19_rpe_m7_preparation_20261110/synthetic_tests.txt
