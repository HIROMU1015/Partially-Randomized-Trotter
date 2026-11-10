# H6 DF19 RPE m6：T6.4・限定2候補の結果

2026-11-10 JST。**明示実行許可を受け、固定PF/PR各1候補を一回評価した。両候補ともideal条件付きround PASS、十分Nは各軸PF705、PR772。** 保存m0..5と合わせm0..6の7roundまで条件付きNが揃った。actual保証・厳密ground同定・全13round成功・総RPE費用は未資格。

科学SOURCEは **`c9aa89548a3ab337e1a8dbc5cf219a31860820d6`**。[固定m6計画][PLAN]の条件と2 IDを変更せず実行し、科学retry・SOURCE変更は0。数値正本は[run01/result.json](run01/result.json)、[実行出力](execution.txt)、[技術照合](CHECKS.txt)、[7roundの対応と価格担当handoff](COVERAGE.json)。結果公開commitは科学SOURCEと区別する。

## 条件・候補・再利用

Linear H6、隣接1.00Å、STO-3G、保存tol-only 1e−8／DF19、保存weighted-Hermitian completion、12 system mode・Nα=Nβ=3のsector400。保存DF-Hの同sector ground energyをtargetとし、保存数値stateによる条件付き評価、P=0を維持した。signed weights・fragment順・scalar・H/state/snapshot bindingを保持し、元分子HとのDF差を新たなenergy reserveに加算していない。

CA=.00159360010199040 Ha、h=.1、βtotal/stat=.4/.2 rad、各方式αtotal=.05、13round m0..12、αaxis=.05/26、prior [−9.695399206821989,−3.1355322139814668] Haと無shift alias政策はD1/D2の保存設定を転記した。新prior/horizon・α/β最適化は0。保存solver energy −3.2360662799087647 Ha・residual 6.1060388500553216e−15 Haはreference診断にだけ用い、priorやdecoder branchを選ぶoracleにしていない。

| arm | candidate ID | 構成 |
|---|---|---|
| PF | `201a8e365ab6f4af6ebc` | Full19、m5_best、q_PF4、δ1.6 |
| PR | `ef41b65942d56977d6cd` | prefix7、m5_best、R_outer16/K2、q_PF4、δ1.6 |

共通で**m6、T6.4、q_RPE64**。q_PF4とは別変数である。既存11 signed piece、負時間piece、ABS_TIME_PER_OUTER配分を維持し、PR配分は`[1,2,1,2,1,2,1,2,1,2,1]`、各piece最低1、合計16/outer。全interrogationのRTE event数はq_PF R_outer=64であり、q_RPE64とは意味が異なる。K2のpaired numeratorは3次まで。候補追加・差替えは0。

既存D2 action backend／m5 schedule／finite Taylor・normalizer／D1 geometry・Hoeffdingを再利用した。保存tail scalar=.0194666612606853、lambda_R=.1649333839054104を使い、tail抽出・新対角化は0。保存係数を400次元の作用表現へ復元しただけで、新H/DF/state生成やfull Fock行列の構築はしていない。

## Reference・finite mean・normalization

T6.4 full-H referenceを1時点・2経路で評価した。

| 経路 | reference |
|---|---|
| expm_multiply | -0.28640442981123+0.95810881562822i |
| expm | -0.28640442981123+0.95810881562822i |
| 経路差 | 2.278663452567752e-15 |

経路差は既存gate1e−10以内。以下は表示を丸めたcomplex signalで、正本にはbinary64値を保存する。signed biasはcorrected−reference。PRはcorrected/rawを別列として独立に伝播し、raw=corrected/Bを事後代入していない。

| arm | corrected | raw | total signed bias |
|---|---|---|---|
| PF | -0.28650023419895+0.95807917202979i | -0.28650023419895+0.95807917202979i | -9.5804387722331e-05-2.9643598425722e-05i |
| PR | -0.28650023316152+0.9580791694574i | -0.27374626029205+0.91542888746904i | -9.5803350293022e-05-2.9646170818465e-05i |

PR exact-tail内部対照は−.28650023395225366+.9580791720888409i、finite RTE成分は7.9072903780e−10−2.6314429524e−9i。outer PF成分とfinite RTE成分は正本に分離して保持し、diskではtotal signed biasを一回だけ数えた。

| arm | log B | B | attenuation A | log_B margin |
|---|---:|---:|---:|---:|
| PF | 0 | 1 | 1 | 2.50253152e-11 |
| PR | 0.04553772997799 | 1.046590491705 | 0.9554835515185 | 1.04748793e-11 |

scalar/identityは従来どおり全時間で一回会計した。PR raw/corrected vector closureは約5.08e−15（既存gate1e−9）、exact-tail norm差はPF約2.22e−15・PR約4.44e−16。いずれも数値診断であり、rigorousな対真ground誤差上界ではない。

## Disk・位相・半径・十分N

既存u floor1e−9・stage依存roundoff・closureにreference経路差を加え、両方式u_X=u_Y=1.0000022786634526e−9を採用した。保存数値state位相へのreference offsetは約4.7066124e−14、T×保存residualは約3.9078649e−14。これらを含むcorrected diskをnormalization margin/closure込みのraw diskへ接続した。未知actual implementation/law距離と厳密ground同定誤差を0で補完していない。

| arm | corrected disk | raw error disk | raw半径下界 | βdet rad | βdet+βstat rad | corrected位相差 rad | 十分N_X / N_Y |
|---|---:|---:|---:|---:|---:|---:|---:|
| PF | 0.00010028696102 | 0.00010028698605 | 0.999899713014 | 0.00010028698621 | 0.200100286986 | 0.00010028118254 | **705 / 705** |
| PR | 0.00010028673038 | 9.5822331328e-05 | 0.955387729187 | 0.00010028674103 | 0.200100286741 | 0.0001002809256 | **772 / 772** |

位相差の基準はexp(−i E_state T)。両方式で正raw半径、βdet+βstat≤.4が成立し、`CONDITIONAL_ROUND_PASS`となった。既存Hoeffdingと独立式 `ceil(2/epsilon_axis² × log(2/alpha_axis))`の整数が一致した。有限時間のαaxis=.025や別epsilon gridは使用していない。Nは条件付き十分shotの計画値であり、最小N・実測shot数ではない。

**actual laneは両方式ともUNRESOLVED_REFERENCE_OR_ERROR、N_X/N_Y=null。** 保存数値stateの厳密ground同定、経験的allowanceのrigorous資格、実装・law距離の不足を保持する。idealのexact-law仮定とactualの未知量を区別し、m6のPASSを全RPE成功保証へ拡張しない。

## 保存D1/D2との対応・価格担当へのhandoff

以下は各軸N。m0..5の12行は保存COVERAGEからそのまま転記し、旧reference/signal/phase/N/部分費用を再計算していない。

| m / T | q_PF / δ | PR prefix/R/K | PF N / PR N | 出典 |
|---|---|---|---|---|
| 0 / 0.1 | 1 / 0.1 | 5/32/2 | 705 / 705 | D2 |
| 1 / 0.2 | 1 / 0.2 | 5/32/2 | 705 / 708 | D2 |
| 2 / 0.4 | 1 / 0.4 | 5/32/2 | 705 / 719 | D2 |
| 3 / 0.8 | 1 / 0.8 | 5/32/2 | 705 / 765 | D1 |
| 4 / 1.6 | 1 / 1.6 | 7/16/2 | 705 / 721 | D1 |
| 5 / 3.2 | 2 / 1.6 | 7/16/2 | 705 / 737 | D2 |
| 6 / 6.4 | 4 / 1.6 | 7/16/2 | 705 / 772 | D2_M6 |

今回のm6について**Cは未評価、C/Gはnull**。価格照会・取得・fit・予測・N×C集計は全て0であり、別Codexの価格担当の作業に介入していない。本成果はCの存在・モデル資格を判定するものではない。

COVERAGE.jsonの`m6_price_handoff`へ2候補のcandidate ID、T/m/q_RPE/q_PF/δ、formula、prefix、R_outer/K、signed weights、整数配分、event数、X/Y別N/α、law model、H/state/snapshot binding、compiler期待条件、P0を保存した。normalization Bや古典価格標本数をNまたは価格へ二重乗算しない。

価格の結合条件は測定付きfull Hadamard wrapper、symmetric_directional、Qiskit1.3.0／rz,sx,x,cx／opt1／seed17、compiler hash `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。これはm6回路をbuild/compileした証拠ではない。歴史的`binding.T=.8`は保存入力reference時刻であり、今回の発展時間はcandidate.T=6.4である。

## 保全・不足条件・次の最小範囲

campaign wallは**11.961秒**。120共通半step指数をcacheし、観測affinity64・load約4.42・available RAM約526.9 GBから59 threadを選んで並列化した。BLASはprocess内1 threadで入れ子を避け、PF/PR2 jobで共有cacheを使った。cache計数307,200,000 byteはprocess peakの測定値ではない。共有設定や他process/worktreeを変更せず、独自の固定worker capは追加していない。

実callは新reference1時点/2 action、finite mean2候補、PR exact-tail内部対照1、scalar2。新eigensolve・SCF/DF/state生成、trajectory sampling、回路build/compile、価格評価、全RPE成功率simulationは0。科学SOURCE/計画14 fileと原入力/contextのhash、旧D1/D2結果、per-candidate outputと正本の一致、N・normalization・phase/α/βを照合した。[準備時の人工3件][TESTS]を再利用し、SOURCEが変わっていないため同じ試験を再実行していない。本担当自身の技術照合であり、独立科学レビューではない。

結果原本・SOURCE/input/raw・旧結果/grant/ledger/marker/KEEP/過去STOPを保持した。run01/progress.jsonは途中checkpointであり、最終状態はresult.jsonを参照する。

**残りはm7..12、T12.8/25.6/51.2/102.4/204.8/409.6の6round。** 本N側成果ではそれらのsignal/Nと全体phase/branch/failure資格が未取得であり、各候補に対応するCも別担当の成果と照合する必要がある。7roundのNが揃っても化学的精度・全13round成功・総費用はまだ主張できない。αaxis=.05/26を維持し、評価済み7roundへ再配分しない。

次の最小科学案は**m7/T12.8、q_PF8/δ1.6、Full19 PF対PR7/R16/K2の2候補**。保存state/係数/tail定数と同じbackendを再利用し、新reference1時点/2経路、finite mean2、PR内部対照1、phase/radius/N2を対象にする案である。これは未実行・未承認で、成立性・N・費用を予測した結果ではない。残り6roundをまとめる場合も、ユーザーが有限menuと科学範囲を指定した後に扱い、広い候補最適化へ自動拡張しない。

公開remote SHAと主要成果物の全文一致を確認後**mandatory STOP**。次の担当は**ユーザー**：m7以降の科学範囲を判断する。**価格担当Codex**：このhandoffを使い別途許可されたC側作業を進める。**本担当Codex**：追加許可までm7以降・価格評価・詳細科学レビューへ進まない。

[PLAN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c9aa89548a3ab337e1a8dbc5cf219a31860820d6/docs/research/h6_df19_rpe_m6_preparation_20261110/REPORT.md
[TESTS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c9aa89548a3ab337e1a8dbc5cf219a31860820d6/docs/research/h6_df19_rpe_m6_preparation_20261110/synthetic_tests.txt
