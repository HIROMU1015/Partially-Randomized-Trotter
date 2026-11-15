# Full-H pure RTE κ2：固定1.5倍shot配分のQ3結果

2026-11-15 JST。H4/linear 1.00 Å/STO-3G/tol-only DF10、sector(2,2)、保存state、Full-H κ2/K2（one-body込み全H）、h=0.1/M12・全13round。ユーザー承認の**1候補・固定1配分を新規Q3 call 1回だけ**評価し、条件付き直接モデルMSE上界が採択閾値以下となった。原API return・partials・資源記録は保存済み。元control・2倍は再評価していない。

## 3配分の結果

全配分でX/Y各軸に同じ整数列を用いた。今回の正本は`[96,90,84,78,71,65,57,51,45,38,32,26,18]`で、roundごとに`ceil(3*N_control/2)`へ一致する。評価結果を見て偶奇や配分を変更していない。

| 配分 | m0〜m12、N_X=N_Y | MSE下界 Ha²（保存表示） | MSE上界 Ha²（保存表示） | exact閾値判定 |
|---|---|---:|---:|---|
| 元control | `[64,60,56,52,47,43,38,34,30,25,21,17,12]` | 0.0001624431036689670410203549 | 0.0001633618706479668178967357 | 不適合 |
| 今回1.5倍（切上げ） | `[96,90,84,78,71,65,57,51,45,38,32,26,18]` | 0.000002098434139000992158910785 | 0.000002105802648067548936567344 | 適合 |
| 保存2倍 | `[128,120,112,104,94,86,76,68,60,50,42,34,24]` | 7.602391255146027183321118E-7 | 7.603835082688183971535190E-7 | 適合 |

τは十進εE=`0.00159360010199040 Ha`のexact二乗、`15136964828156312001 / 5960464477539062500000000 Ha²`（表示約2.539561285063813282041692e−6）。**今回upper≤τ**をexact有理数で確認した。今回の科学判定は`MODEL_BOUND_MEETS_ADOPTED_THRESHOLD`。表示丸め値で採否を決めていない。

保存参照は元control **a82b0f0ec7f0f11d9981ddb2343fcd5378a87503** `docs/research/pr_df_fullh_q3_fixed_results_20261115/`、2倍 **422db93fc7bb599a7ea46c5b58e5dec0fda9ef08** `docs/research/full_h_kappa2_scale2_q3_results_20261115/`。各SUMMARYのFull-H κ2行をSOURCE付きで[COMPARISON.json](COMPARISON.json)へそのまま収録した。これらの元API/DPは再評価していない。

この表は3本の固定whole-vectorの条件付きモデル評価であり、MSEの単調性、最小N、未評価配分の適否、G最適性の証明ではない。今回vectorはUSER_SUPPLIEDで、十分N・原論文型nominal Nへ自動昇格しない。

## SOURCE・入力・科学契約

- 今回の科学SOURCE：**47ccd1f3759b1dfe45f464421f89a669d1e3eb27**、[INPUTS.json](INPUTS.json)、`scripts/run_full_h_kappa2_scale3o2_q3_fixed.py`を実行前commit/push。
- 再利用callerのSOURCE：**8a8871021b83c139479d0f47d24d8304145e8cf4**、元worker保存経路はa82b0f0。2倍executeや旧3候補executeを起動せず、Full-H metadata・worker・superviseだけを再利用。
- kernel：**710aa69b663ef3313439762d25c80a652a03948f**、高速backend：**9e294778266c9c867a8b1b98e040c2e94b1cb635**。数値本体を変更せず、既存8worker/batch8で実行。
- 元13round packet：**ce6aacfaca8249c510baa2d958849cffc7649cdc** `docs/research/d_full_h_kappa2_s1_results_20261114/run01/FULL_H_K2_QDYADIC_KAPPA2/`。元q/R/grouping/event数、law、scalar、normalization、raw、H/state/snapshot、physical IDを保持し、再取得・再適用なし。
- 元configuration `54ac3eb38de5f90a77e9921100576c89f6c870f68384f4c7d9efe61b2c80c07e`、total `265b90dc4a9b81011736664205e23833c420d569e57c923ef45c39eb59489612`、binding `579c5de23a2336d402a1d5b195061b34b3c7a1b49e99ea2e1a674c1a086189a5`。
- 新allocation：`FULL_H_KAPPA2_TOTAL_Q3_USER_SCALE_3O2_01`。物理候補・canonical decoder契約は元/2倍と同じで、新しく識別するのはallocationと今回run。

decoderはZ1＋T1 `consecutive_dyadic_nearest_branch_principal_energy_total_v1`、zero HOLD/tie argmin・nearest branch・binary64順序・principal guardを保持。保存ideal finite-K raw、shots/X/Y/round間IID二項モデル、EXACT_AFFINE_OF_BINARY64_RAW。targetは保存full DF10 solver energy proxy **−2.1663874486347634 Ha**、全事象loss `(Ehat−Eproxy)^2`。原packetのground/proxy資格を変更せず、AδEを二重加算しない。元Minkowskiは別参照として保全した。

§10.5照合は[SCOPE_AND_PREFLIGHT](SCOPE_AND_PREFLIGHT.md)。関連B/Full-H/C/roadmap公開ref・履歴・関連jobを読取確認した範囲で、同一vectorの受入可能Q3結果/進行中Q3は同定されなかった。元13rawと既存bridgeを再利用し、新評価1・再利用比較2・保留0。稼働中D1価格holdout/publicationへ介入しない。全サーバー・非公開資料の不存在を証明したものではない。

## bias²・Var、残余、zero/tie

下表は保存表示値の転記。exact端点は各元API・今回APIに保全する。risk/DPの再計算や比率・補間の追加解析はしていない。

| 配分 | bias²区間 Ha² | Var区間 Ha² | 残余mass | 残余loss上界 Ha² | any-zero確率 |
|---|---|---|---:|---:|---:|
| 元control | [2.868161477561078564583462E-12, 2.872144182161031235037416E-12] | [0.0001624431007968228588593236, 0.0001633618677798053403356571] | 2.588278666480358978222801E-8 | 9.187669789997768763807948E-7 | 0.003391162117029792340028986 |
| 今回1.5倍（切上げ） | [1.202813987909073678941036E-10, 1.202818311291663244980878E-10] | [0.000002098313857169862992586287, 0.000002105682366668758029199450] | 2.211565844564397905783298E-10 | 7.368509066556777656558794E-9 | 0.0007219549874823928390041101 |
| 保存2倍 | [1.057162675471084800192096E-10, 1.057162948357228814950166E-10] | [7.601334092197669954506168E-7, 7.602777920012712886734998E-7] | 1.858983689028632269958308E-11 | 1.443827542156788214071640E-10 | 0.0001317055221362517231249688 |

今回の保持終端lossは0.000002098434139000992158910785 Ha²、残余loss上界は7.368509066556777656558794E-9 Ha²で、和がDP上界にexact一致する。pruning履歴を0lossとして捨てない。principal-guard失敗として残余に含むmassは3.790299198693241361457736E-12であり、別に二重加算しない。mean energy/second moment/biasを含む5数値区間の順序、bias²＋VarとMSEの区間整合を確認した。

coarse上界26.99934870829052645628689、lemma上界0.0009859684691524075000524019、DP上界0.000002105802648067548936567344の**有効上界のmin**が最終上界。各上界を加算していない。旧partial規則の未解決率区間は[0.0008799208746496365507951232, 0.0008799210057686877923989182]で、全関数total decoderの未定義率ではない。

全13roundのzero HOLD/tie argmin追跡mass、first-unsafe、残余は[SUMMARY](SUMMARY.json)・元partialsに保存。今回のm4〜m8は両軸とも奇数shotで、zero/tieの追跡massは0だった。一方、偶数roundのzero/tieは元規則で出力を定義して全事象lossへ保持した。配分間ではshot数とcount格子の偶奇が同時に変わるため、変化を回数だけの効果とは断定しない。追跡prefix massを残余込みの厳密な全履歴率と同一視しない。

unsafeは安全十分条件を外れた事象であり、実際の誤branch率・当該roundのMSE寄与と同一ではない。3配分の表から未評価配分のriskを補間・外挿しない。

## 数値gateと有限追跡

| 項目 | 実行前固定scope | 実行記録 |
|---|---:|---|
| phase radius | 全13round1.0 rad | 新vector固有の閉包PASS |
| round outcomes | 9,409 | 全count支持を包含 |
| verification pairs | 273,175,586 | 12,062,298組を検査、VERIFIED_FINITE_BINARY64_SUPPORT |
| 非終端frontier | 8,192 | 各非終端round8192以下 |
| transition pairs | 353,666,241 | 232,764,892組、上限以内 |

count上限は今回整数列だけを使う既存finite_bounds式で実行前固定し、旧2593状態・2倍10081状態・旧閉包PASSを移送していない。今回m0は**5741 grouped outcomes・5741保持状態・残余0**で、新countから評価した。終端m12の保持状態は34,988で、非終端capを課していない。残余へ回した状態数は521,249。

kernelの全round exact質量保存gate、各round good＋unsafe=1、first-unsafe13分割＋all-good=1、残余loss会計、有効上界min、MSE上下界・moments・exact閾値の整合を確認済み。最大観測更新幅超過は`1.0495077029659683e−15 rad`で、固定engineering allowance `1e−12 rad/transition`以内。厳密数値certificateは発行していない。

## 保存・性能・観測

| 項目 | 今回の記録 |
|---|---:|
| worker wall（API/RESULT保存まで） | 632.6786783570424 s |
| supervisor wall | 632.8193791368976 s |
| caller全体wall（外側observer） | 634.3889514051843 s |
| worker CPU | 580.4748119569999 s |
| 回収済みDP子CPU合計 | 617.826818 s |
| process-tree CPU | 1198.301631047 s |
| worker peak RSS | 1,817,164 KiB |
| 最大単一child peak RSS | 1,350,200 KiB |
| 観測subtree RSS合計ピーク | 11,613,732 KiB |

child peakは子の最大単一RSSで、合計ではない。subtreeは約2秒おきのRSS和の観測ピークで、共有pageの重複を含み得る。PSSや厳密な物理RAM最大ではない。CPU/RAM/workerへの恒常的上限を設けず、共有設定・他processは変更していない。今回と2倍の時間差を高速化倍率とは扱わない。

既存worker自身が原API_RETURN・RESULT・3種PARTIAL・RESOURCE_FINAL・PERSISTEDを保存してから親へreceiptを返した。原API returnとrecordの一致を確認し、保存状態は**WORKER_ARTIFACTS_SAVED**、親完了は**FIXED_SCIENTIFIC_WORKERS_SAVED**。worker/monitor errors0。caller終了code0/signalなし、DP子104件＋科学worker1件の計105件も全終了code0。数値判定と保存/運用状態を分けて記録した。

今回追加したのは協調stage observerとpool shutdown時の終了code記録で、数値kernelは不変。ACCEL_PROGRESSは各伝播、SEGMENTSは伝播間・終端処理を記録する。非同期frame dumpやtimer threadは使用しない。新caller4＋再利用worker5の**人工/metadata9test PASS**で、実supervisor→workerに観測contextが効き、人工原API出力が不変であることを確認した。実結果でもstageと終了code記録を取得した。手法は既存[acceleration_casebook](../../acceleration_casebook.md)に短く追記した。速度倍率は未測定。

## 成果物・ロードマップ差分・STOP

- [INPUTS](INPUTS.json)、[実行前確認](SCOPE_AND_PREFLIGHT.md)、TESTS_NEW_CALLER/TESTS_REUSED_WORKER。
- [原API return](run01/FULL_H_KAPPA2/API_RETURN.json)、RESULT、coarse/lemma/total-DP partials、資源・stage・子終了記録。
- [SUMMARY](SUMMARY.json)、[3配分比較](COMPARISON.json)、[C_RESULT_JOIN](C_RESULT_JOIN.json)、[HANDOFF](HANDOFF.md)。26軸を今回allocation→物理candidate→元/total ID・law/SOURCEへ結び付ける。

統合担当への更新点：Full-H κ2は**元control不適合・今回1.5倍適合・保存2倍適合**という3本の固定配分評価が揃った。今回の原API・資源・process終了まで完全保存。モデル閾値適合を確認した配分が増えたが、最小N・全配分の単調性・物理的ground RMSE保証・価格/G最適性は未認定。D1/C完了はこのrisk結果の依存ではない。ロードマップ本体の更新は統合担当に委ねる。

`sufficient_N/global_RMSE_bound/G/G_qualified/winner=null`。ground/state/target包含、実sampler law移送、数値certificateの未資格を維持。新signal/sampling/compile/価格/G/旧control再計算/2倍再計算/追加vector/無断retryは0件。旧SOURCE/input/raw/結果/grant/ledger/marker/KEEP/STOPと他担当の作業領域を保全した。

**最大1件終了でmandatory STOP。追加倍率・round別探索・追跡範囲拡張へ自動進行しない。**
