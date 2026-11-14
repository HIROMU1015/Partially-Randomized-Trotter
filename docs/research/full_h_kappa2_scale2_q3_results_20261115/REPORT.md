# Full-H pure RTE κ2：2倍whole-shot配分の固定Q3結果

2026-11-15 JST。既存B branch `codex/rpe-total-dp-acceleration-20261114`。ユーザーが採択した4525e1b6の1候補・1配分・新scopeで、**新規Q3科学callを1回だけ実行し、完了した**。無断retry・追加配分・MC・sampling・価格・Gは0件。

## 結果と主張範囲

同じH4/1.00 Å/STO-3G/tol-only DF10、sector(2,2)、保存H/state/snapshot、Full-H κ2/K2（one-body込み）、h=0.1/M12・13round、Z1＋T1 total decoder、solver proxy、ideal finite-K raw/IIDモデルを維持した。次の2配分を同一物理候補に対する**条件付き直接モデルMSE**として並べる。旧controlはa82b0f0の保存値を再利用し、再計算していない。

| 配分 | m0〜m12のN_X = N_Y | MSE下界（Ha²） | MSE上界（Ha²） | exact τ判定 |
|---|---|---:|---:|---|
| 保存control | `[64,60,56,52,47,43,38,34,30,25,21,17,12]` | 0.0001624431036689670410203549 | 0.0001633618706479668178967357 | MODEL_BOUND_EXCEEDS_ADOPTED_THRESHOLD |
| 今回の2倍配分 | `[128,120,112,104,94,86,76,68,60,50,42,34,24]` | 7.602391255146027183321118E-7 | 7.603835082688183971535190E-7 | **MODEL_BOUND_MEETS_ADOPTED_THRESHOLD** |

τは十進εE=`0.00159360010199040 Ha`のexact二乗、`15136964828156312001 / 5960464477539062500000000 Ha²`（表示約2.539561285063813282041692e−6）。今回の**有効上界≤τ**をexact有理数で判定した。旧controlは下界>τだった。保存値との比率や1/Nモデルによる倍率推定は追加していない。今回の適合はこの固定モデル・配分だけの結果で、最小N・方式全体の優劣・物理的ground RMSE保証ではない。

[SUMMARY](SUMMARY.json)は25桁表示、[COMPARISON](COMPARISON.json)は元control行と新行の対応、exact端点の正本は`run01/FULL_H_KAPPA2/API_RETURN.json`／RESULT.json。価格がないのでG・winnerは計算していない。

## SOURCE・科学条件・重複照合

| 区分 | 固定SOURCE・資料 |
|---|---|
| 採択した計画 | **4525e1b68883bb6a4960346f6406da4d01af2ccd**、`docs/research/full_h_kappa2_next_shot_plan_20261115/FINITE_SCIENCE_PLAN.md`と同directoryの`PROPOSAL.json` |
| 今回の科学実行SOURCE | **8a8871021b83c139479d0f47d24d8304145e8cf4**、[INPUTS.json](INPUTS.json)、`scripts/run_full_h_kappa2_scale2_q3_fixed.py` |
| 保存control | **a82b0f0ec7f0f11d9981ddb2343fcd5378a87503**の`pr_df_fullh_q3_fixed_results_20261115/run01/FULL_H_KAPPA2/`。旧科学SOURCEは27e70efe5 |
| 元13round raw | **ce6aacfaca8249c510baa2d958849cffc7649cdc**、`d_full_h_kappa2_s1_results_20261114/run01/FULL_H_K2_QDYADIC_KAPPA2/` |
| decoder/risk・高速backend | kernel **710aa69b663ef3313439762d25c80a652a03948f**、accelerator **9e294778266c9c867a8b1b98e040c2e94b1cb635**。未変更 |
| 元/total/binding | `54ac3eb38de5f90a77e9921100576c89f6c870f68384f4c7d9efe61b2c80c07e` / `265b90dc4a9b81011736664205e23833c420d569e57c923ef45c39eb59489612` / `579c5de23a2336d402a1d5b195061b34b3c7a1b49e99ea2e1a674c1a086189a5` |

allocation IDは`FULL_H_KAPPA2_TOTAL_Q3_CONTROL_SCALE2_PROPOSAL_01`。名前にPROPOSALは残るが今回はユーザー採択済みのUSER_SUPPLIED vector。物理IDを変更せず、13physical行・law・grouping/q/R/event schedule・scalar・normalizationを固定索引へ照合した。旧3候補runnerは起動せず、そのFull-H metadata/worker/superviseを新1件callerから再利用した。

計画後の公開ref・関連履歴・自ユーザーの関連jobを確認した範囲では同等2倍Q3結果は未同定で、新規1件・再利用科学結果0件・保留0件とした。元13信号は全て再利用し、再取得なし。稼働中のD P2価格jobはQ3ではなく介入していない。詳細は[SCOPE_AND_PREFLIGHT](SCOPE_AND_PREFLIGHT.md)。確認範囲外の全記録の不存在を証明したものではない。

推定量は`consecutive_dyadic_nearest_branch_principal_energy_total_v1`。targetは保存full DF10 solver energy −2.1663874486347634 Haの条件付きproxy、全事象loss `(Ehat−Eproxy)^2`、確率則EXACT_AFFINE_OF_BINARY64_RAW、shots/X/Y/round間IID。原targetの認証欄を変更せず、normalization/scalarの再適用なし。finite-K等の近似も直接MSEへ反映し得るので、AδEは加算していない。元Minkowskiは別参照で保全した。

## bias²・Var・残余会計

| 診断（Ha²、確率以外） | 保存control | 今回の2倍配分 |
|---|---|---|
| bias²区間 | ['2.868161477561078564583462E-12', '2.872144182161031235037416E-12'] | ['1.057162675471084800192096E-10', '1.057162948357228814950166E-10'] |
| variance区間 | ['0.0001624431007968228588593236', '0.0001633618677798053403356571'] | ['7.601334092197669954506168E-7', '7.602777920012712886734998E-7'] |
| 残余mass | 2.588278666480358978222801E-8 | 1.858983689028632269958308E-11 |
| 残余loss上界 | 9.187669789997768763807948E-7 | 1.443827542156788214071640E-10 |
| any-zero確率 | 0.003391162117029792340028986 | 0.0001317055221362517231249688 |
| principal-guard失敗として残余に入ったmass | 3.177487945890296356572502E-10 | 3.206492272825792890383794E-15 |

今回のmean energy、second moment、bias、bias²、Varの区間はSUMMARYに保存。bias²＋VarとMSE区間の整合、各区間の順序をexactで確認した。bias²の変化だけから全riskを説明せず、手法固有の物理biasとこのdecoderのmodel biasを同一視しない。

今回の保持終端lossは7.602391255146027183321118E-7 Ha²、残余loss上界は1.443827542156788214071640E-10 Ha²で、両者の和がDP上界にexact一致する。残余を0として捨てず、guard失敗massも既にその会計へ含む。coarse上界8.897429133235258247521819、lemma上界0.00003299965881877342928779356、DP上界7.603835082688183971535190E-7の**有効上界のmin**を使い、別々の上界を加算していない。

## 新support・全13roundのgate

- 全roundのρ=1.0、round outcomes上限16,641、隣接verification上限847,519,356、非終端frontier8192、transition上限620,478,721を固定した。
- 新vector固有の閉包は`VERIFIED_FINITE_BINARY64_SUPPORT`、31,407,535組。DP遷移379,479,905組で宣言scope内。
- **m0は10,081 grouped outcomes→保持8192状態**。m0残余`6.597372487332931354661754E-63`を会計した。旧N0=64の2593状態・残余0を新supportの証明として使っていない。
- 非終端は全round8192以下、終端m12は34,719状態。終端へ非終端capを誤適用していない。kernel内で保持mass＋残余mass=1を全13roundでexact検査し、各roundの残余massを保存した。追跡外へ回した状態数は735,847。
- 保存first-unsafe13分割＋all-good=1、各round good＋unsafe=1、保持loss＋残余loss=DP上界、最小有効上界、上下界・moments・exact τ判定の整合を確認した。
- 最大観測更新幅超過は`1.0495077029659683e-15 rad`で、宣言engineering allowance `1e-12 rad/transition`以内。ただし区間演算等の厳密数値certificateはない。

unsafeは安全十分条件を外れた事象で、実際の誤branchや当該roundのMSE寄与と同一ではない。zero/tieは全関数decoderのHOLD/argminで定義し、未定義出力として捨てない。旧partial規則の未解決率診断は今回['0.0001524745256879538112771491', '0.0001524745352057249650534016']で、total decoderの未定義率ではない。

### zero/tieの保存診断比較

下表は保存された追跡prefix massを表示丸めしたもの（原値はCOMPARISON/SUMMARY、exactは原API）。m4/5/9/10/11は旧vectorが奇数、新vectorが偶数でcount格子も変わった。変化をshot回数だけの効果やMSE寄与と断定しない。残余込みの厳密全履歴tie率・zero率とは限らない。

| m | 旧N→新N（各軸） | zero旧 | zero新 | tie旧 | tie新 |
|---:|---|---:|---:|---:|---:|
| 0 | 64→128 | 2.310843E-9 | 2.715813E-16 | 0 | 0 |
| 1 | 60→120 | 1.886981E-8 | 1.699043E-14 | 1.103452E-9 | 1.782253E-16 |
| 2 | 56→112 | 9.633376E-8 | 4.136671E-13 | 5.196953E-9 | 3.109413E-15 |
| 3 | 52→104 | 8.701962E-8 | 3.137552E-13 | 5.916981E-8 | 2.193440E-13 |
| 4 | 47→94 | 0 | 6.628633E-12 | 0 | 2.112258E-13 |
| 5 | 43→86 | 0 | 1.605778E-10 | 0 | 2.132222E-12 |
| 6 | 38→76 | 3.557139E-6 | 3.851639E-10 | 1.330917E-6 | 7.641209E-11 |
| 7 | 34→68 | 1.525901E-5 | 6.356234E-9 | 2.129753E-6 | 1.682767E-10 |
| 8 | 30→60 | 3.650487E-5 | 3.219346E-8 | 8.794780E-6 | 2.212544E-9 |
| 9 | 25→50 | 0 | 4.162863E-7 | 0 | 1.664018E-8 |
| 10 | 21→42 | 0 | 1.577438E-6 | 0 | 2.551825E-7 |
| 11 | 17→34 | 0 | 1.804483E-5 | 0 | 1.237931E-6 |
| 12 | 12→24 | 3.335822E-3 | 1.116301E-4 | 1.075484E-3 | 1.925777E-5 |

## 実行負荷・原結果保存

| 項目 | 今回の記録 |
|---|---:|
| worker wall（API/RESULT保存まで） | 1106.2360859070905 s |
| supervisor wall | 1106.3959254338406 s |
| worker CPU | 989.018776213 s |
| 回収済みchild CPU合計 | 1245.3291900000002 s |
| process-tree CPU | 2234.347967553 s |
| worker peak RSS | 3,159,584 KiB |
| 最大単一child peak RSS | 2,255,016 KiB |
| 観測subtree RSS合計ピーク | 20,589,740 KiB |

8worker/batch8の既存高速backendを1構成で使用した。CPU/RAMへの新たな恒常制限や共有設定変更はない。child peakは各子の最大値で合計ではない。subtreeは約2秒間隔のRSS和の観測ピークで、PSSや厳密な同時RAM最大値ではなく、共有pageの重複を含み得る。時間・負荷を旧vectorとの純粋な高速化比較とは扱わない。

科学callは1、原API return/RESULT/3種partials/CPU・RSS/PERSISTED/親receiptを保存し、原API returnとrecordの一致を確認した。科学worker・parent monitorのerrorsは共に0、保存状態は**WORKER_ARTIFACTS_SAVED**、親完了は**FIXED_SCIENTIFIC_WORKERS_SAVED**。数値モデル適合とは別の運用状態である。科学processは終了を確認した。

新caller2件＋既存worker/親monitor故障fixture5件の計7人工・metadata test PASS。別の結果投影scriptで、momentsの補助`status/meaning`文字列を数値区間として読もうとしたTypeErrorがあった。readerを数値5区間の明示検査へ修正し、元の原結果を使って投影・exact整合確認を完了した。kernel・実行SOURCE・原結果は変更せず、科学retryは0。これは科学的gate失敗や原API return欠測ではない。

## C引渡し・ロードマップへの影響・mandatory STOP

[HANDOFF](HANDOFF.md)、[C_RESULT_JOIN](C_RESULT_JOIN.json)に元/total/binding・13physical×X/Y・allocation・law selector・結果SOURCEを対応付けた。価格のSOURCE/lowering/full-wrapper/cohort資格はCの既存guardで別途確認する。今回はcompiled価格・Gを評価していない。

統合担当への更新点：Full-H κ2は「control不適合、次配分未評価」から、**同じ物理候補・同じdecoder/targetで、採択2倍vectorの条件付きモデルMSE上界≤τを確認、完全API/資源保存済み**へ更新できる。必要Nの最小化、物理的RMSE保証、適格E[C]によるG比較は未完了。旧control以下の全vector不適合や、新vectorの最適性は推論しない。D1価格較正の完了はこのrisk結果の依存条件ではない。ロードマップ本体は統合担当が更新する。

`sufficient_N/global_RMSE_bound/G/G_qualified/winner=null`。ground/state/target包含、実sampler law移送、厳密数値certificateは未資格。旧結果/raw/SOURCE/grant/ledger/marker/KEEP/STOP、他担当worktree・jobを保全した。

**最大1call完了でmandatory STOP。追加shot配分・4倍案・round別探索・scope/backend変更・価格/G・十分N認定へ自動進行しない。**
