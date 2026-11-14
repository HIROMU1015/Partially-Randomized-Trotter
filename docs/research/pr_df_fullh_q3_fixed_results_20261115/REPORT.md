# PR-DF LD1/LD2・Full-H κ2：固定配分のQ3結果

2026-11-15 JST。保存Linear H4/1.00 Å/STO-3G/tol-only DF10/sector(2,2)・保存state、h0.1/M12の全13roundを使用。**3候補を各1回実行し、PR-DF LD1/LD2はモデル閾値適合、Full-H κ2はモデルMSE下界が閾値超過。全件の原API return・partials・CPU/peak RSSをworker自身で保存し、監視/保存エラーはなかった。**

## 実行・判定の件数

| 項目 | 件数・状態 |
|---|---|
| 対象／新規／受入済み同等Q3再利用／保留 | 3／3／0／0 |
| 科学call | 各1回、合計3。retry・別vectorなし |
| モデル適合／モデル不適合／INCONCLUSIVE／UNKNOWN | 2／1／0／0 |
| 数値・運用の終了 | 数値gate全件通過、全worker ARTIFACTS_SAVED、monitor/errorsとも空 |
| 再利用した入力 | 39保存round・既存source-bound bridge・既存control形状。信号再取得0 |
| 今回実行していないもの | 旧partial DP、Pauli/PFの再計算、MC、shot探索、価格/E[C]/G/winner |

固定計画0629a8cf3、科学実行SOURCE **27e70efe533862f38f4080aabb9cdd3ad1749fe9**。[INPUTS](INPUTS.json)／[事前照合・保存検証](SCOPE_AND_PREFLIGHT.md)／[COMPLETION](run01/COMPLETION.json)。同等結果は計画公開後の確認範囲で未同定で、今回3件の新規評価とした。D1価格較正を待たず、Dの価格processへ介入していない。

## 固定モデル・MSE上下界

Z1＋T1 total decoder `consecutive_dyadic_nearest_branch_principal_energy_total_v1`、zero HOLD/tie argmin、保存ideal finite-K raw・shots/axes/round間IID、`EXACT_AFFINE_OF_BINARY64_RAW`。別の明示evaluation_targetはsolver proxy −2.1663874486347634 Ha、全事象lossは`(Ehat−Eproxy)^2`。元packetのground/proxy/UNQUALIFIED欄は変更していない。

両軸共通vectorは`[64,60,56,52,47,43,38,34,30,25,21,17,12]`。allocation IDはINPUTSの計画値を保持。USER_SUPPLIED controlで、十分N・最適N・Full-H固有nominalではない。τは十進εE=`0.00159360010199040 Ha`のexact二乗（表示約2.539561285063813282041692e−6 Ha²）。閾値判定に表示floatを使っていない。

| 候補 | MSE下界 Ha²（表示） | MSE上界 Ha²（表示） | 数値判定 |
|---|---:|---:|---|
| PR-DF LD1 | 4.724625889346538019516433e−7 | 4.724625889346539607295338e−7 | MODEL_BOUND_MEETS_ADOPTED_THRESHOLD |
| PR-DF LD2 | 4.724097319480510549083992e−7 | 4.724097319480511948188040e−7 | MODEL_BOUND_MEETS_ADOPTED_THRESHOLD |
| Full-H pure RTE κ2 | 1.624431036689670410203549e−4 | 1.633618706479668178967357e−4 | MODEL_BOUND_EXCEEDS_ADOPTED_THRESHOLD |

[SUMMARY](SUMMARY.json)は25桁の表示用。exact端点・元入力・全出力の正本は各候補の`API_RETURN.json`／`RESULT.json`。Full-Hは**下界>τ**なので、保守的上界だけが大きいINCONCLUSIVEではない。この限定モデル・固定controlについての不適合で、Full-H方式全体の否定や物理的ground RMSEの失敗保証ではない。

## bias²・Var・有効上界

| 候補 | bias²区間 Ha²（表示） | Var区間 Ha²（表示） | first-unsafe lemma上界 Ha²（表示） |
|---|---|---|---:|
| LD1 | [1.099811847510291450829e−8, 1.099811847510291450833e−8] | [4.614644704595508874433e−7, 4.614644704595510462212e−7] | 8.555310350386932e−7 |
| LD2 | [1.093552127008909949581e−8, 1.093552127008909949584e−8] | [4.614742106779619554126e−7, 4.614742106779620953230e−7] | 8.538977310902529e−7 |
| Full-H κ2 | [2.868161477561078564583e−12, 2.872144182161031235037e−12] | [1.624431007968228588593e−4, 1.633618677798053403357e−4] | 6.668904604763954e−2 |

全3件で最良有効上界はtotal状態集約DP。coarse/lemma/DPの有効上界を既存kernelの規則でminとし、異なる上界を加算していない。bias²＋Varの区間とMSE区間の整合をexactで確認した。Full-Hのこの評価ではvariance診断がMSEの主成分で、bias²の小ささだけでは閾値を満たさない。個別zero/tie/branchのloss寄与を分解した因果評価は追加していない。

raw直接MSEはPF/finite-K等の位相ずれを含み得るため、δE/δE²を二重加算していない。旧Minkowskiは別参照として保全した。モデル数値の僅かなLD差を資源winnerと解釈しない。

## 閉包・質量・残余・zero/tie

ρは全round1.0のまま。3件とも`VERIFIED_FINITE_BINARY64_SUPPORT`、閉包検査2,318,425組。DP遷移は各94,048,712組、上限141,559,036以内。非終端frontier8192以下、終端状態数はLD1 18,560、LD2 18,612、Full-H 24,894。終端状態と非終端の制限を混同しない。

事前の粗い未集約count上限143,995,786は固定上限を超えたため、科学開始前に同一N・decoderの保存full m0 supportを使用して137,923,114へ絞った。固定範囲を拡張せず、確率/phase/DPを再計算しない計数証明である。実際のm0状態数は3件とも2593・初回残余0で、この保存support境界と整合した。詳細とSOURCEは事前照合に保存。

| 候補 | 残余mass（表示） | 残余loss上界 Ha²（表示） | any-zero確率（表示） |
|---|---:|---:|---:|
| LD1 | 1.0033781222840941e−23 | 1.5877789055289133e−22 | 9.9983650978587252e−8 |
| LD2 | 8.3187092726193720e−24 | 1.3991040481205447e−22 | 9.9693476447885825e−8 |
| Full-H κ2 | 2.5882786664803590e−8 | 9.1876697899977688e−7 | 3.3911621170297923e−3 |

各roundのbinomial/DP質量保存は既存kernel/backendのexact gateを通過。保存first-unsafe13分割＋all-good=1、各round good＋unsafe=1、保持loss＋残余loss上界=DP上界もexactで照合。Full-Hでも残余を捨てていない。

zero/tieは全関数decoderの規則で出力を定義し、全lossへ保持。全13roundのzero HOLD/tie argmin追跡massはSUMMARY/rawに保存。m12のtie追跡massはLD1約7.01136e−7、LD2約7.00797e−7、Full-H約1.07548e−3。追跡prefixの値を残余込みの厳密な全履歴tie率と同一視しない。

旧partial規則の未解決率診断はLD1約8.01120e−7、LD2約8.00492e−7、Full-Hの区間約[0.0044788527,0.0044788703]。これはtotal decoderの未定義出力率ではない。unsafeも誤branchと同一ではない。Full-Hのprincipal-guard失敗として残余に会計したmass約3.17749e−10を別途lossへ重複加算しない。

数値allowanceは元のper-transition1e−12 radというengineering allowanceのまま。exact質量計算と、binary64 raw/target/transitionの認証誤差区間は別であり、数値certificateを発行していない。

## SOURCE・保存・資源

PR-DFはA714247619の各B_MEANS、configuration a9abf8de、数値fefd548e、raw carrier16210932を消費した。元のκ・fixed-grid・R/signed allocation・normalization/scalarを保全。Full-Hはce6aacfaのκ2 packet、producer5ccc1dfc→5cc7c07e、one-body込み元grouping/law/scalarを消費。R448は含めない。固定bridge・kernel710aa69b・accelerator9e294778・C c628e9e8を変更していない。

新callerは前回PF workerの数値gateを再利用し、候補別metadata routeとworker側保存を追加した。返却直後に保存したAPI_RETURNはRESULT.recordと完全一致。各workerがAPI/partials/RESULT/RESOURCE_FINAL/PERSISTEDを保存してから、親へ小さいreceiptだけを返した。**全3件がWORKER_ARTIFACTS_SAVED、errors/monitor_errorsとも0**。前回PFの復元viewとは別で、今回の記録は欠測補填ではない。

| 候補 | worker wall 秒 | worker CPU 秒 | child CPU 秒 | 合計CPU 秒 | worker peak RSS KiB | 観測process-tree peak RSS KiB |
|---|---:|---:|---:|---:|---:|---:|
| LD1 | 235.597 | 219.143 | 198.257 | 417.400 | 788,400 | 5,050,512 |
| LD2 | 230.856 | 217.653 | 200.206 | 417.859 | 824,876 | 5,367,320 |
| Full-H κ2 | 349.261 | 337.297 | 222.724 | 560.020 | 1,265,364 | 8,006,676 |

3構成並列、各8worker/batch8を既存benchmarkのplateauと空き資源に基づいて使用。全体wall349.359秒。worker資源のscopeは原API/RESULT保存まで、child peakは個別child最大で合計ではない。process-tree RSSは親monitorの定期観測値で厳密peakとは別。各RESOURCE_FINALとroot COMPLETIONに出所・scope・未丸め値を保存。全scientific process終了確認済み。他担当のprice jobは停止・変更していない。

## 検証・引渡し・今後の境界

科学実行前の人工/metadata test5件PASS。[TESTS](TESTS.txt)。実際の子processが人工評価へ入った後で親monitorに例外を注入し、子が原API/partials/CPU/peak RSSを保存できること、未開始callの拒否、R448 ID混在拒否を検証した。初期count準備の保留を解消してから全5件を通し、科学SOURCEを固定した。科学計算を人工testとして再実行していない。

[C_RESULT_JOIN](C_RESULT_JOIN.json)と[HANDOFF](HANDOFF.md)は、元configuration・physical ID・total configuration/candidate・m/axis・allocation・SOURCEを分ける。新しい十分N packetは発行していない。価格receiptやlaw/cohortを再ラベルせず、既存bridge/guardで後続に接続する。

PR-DFは指定controlのモデル閾値適合が確認できた。Full-H κ2について同じ条件の適合vectorを求めるなら、別の有限shot計画と明示許可が必要。ただし新配分を提案/探索したり、shot数増加で必ず単調改善すると仮定したりしていない。価格がなければGによる優劣は決まらない。

ground/target包含/state誤差/実sampler移送/厳密数値certificate/global RMSEは未資格。`sufficient_N/global_RMSE_bound/G/G_qualified/winner=null`。今回のFull-Hモデル不適合を、物理方式の不可能性又は全設計空間での劣位へ一般化しない。

## ロードマップ影響とSTOP

統合担当への更新点：3候補の「total metadataのみ／Q3未評価」を、**指定controlでPR-DF2件モデル適合、Full-H κ2モデル不適合、完全API/資源保存済み**へ更新できる。候補固有の閉包・質量・有限scopeのgateも当該3評価について確認済み。十分N最小化・物理保証・価格/Gは未完了。D1較正は今回riskの待ち依存ではない。Full-Hの次の依存は、必要なら別の有限shot評価計画と許可である。

ロードマップ本体・旧PF6欠測・旧DF DP・Pauli結果は変更していない。公開・remote SHA/tree/主要file照合後**mandatory STOP**。追加配分・scope拡張・retry・価格/Gへ自動進行しない。
