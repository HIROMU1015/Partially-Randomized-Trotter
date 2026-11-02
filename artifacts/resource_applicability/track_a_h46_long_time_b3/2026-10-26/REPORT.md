# H4/H6 長時間DF-native B3資源評価

対象はlinear H4/H6、1.0 Å、STO-3G、tol-only DF threshold=1e-8、rank10/19、T=.8/1.2/1.6/2.0、epsilon=.05/.01/.005/.001/.0005、P=0。
保存Hamiltonian・state・全Tのfull-H referenceを再利用した。H6保存weighted-Hermitian completionを維持し、SCF/DF/state/referenceの再生成は0。
**以下は取得済み適格候補の有限標本費用と経験的精度会計であり、全域最適性・母平均優位・厳密bias保証ではない。**

## 主結果

1. B3の探索点費用がB2より低い条件は **0/40**。固定B3 nomineeの独立確認で低い条件は **0/40**。B2側の独立標本がない比較は明示している。
2. B3が最良Full PFより低い条件は **0/40**、B0/B0+Iも含む決定論PF envelopeより低い条件は **0/40**。
3. epsilon=.0005の各公式が自身のq/R_outer/Kを選んだ事前固定nomineeの独立確認では、m5_bestがS2より低い条件は **0/8**。以下にformula比較を示す。
4. B3の取得域最良q/delta/R_outer/Kは次表。T依存性はsignalの最小適格q、探索点最小費用q、独立確認費用を区別する。
5. B3を加えた四方式の拡張取得集合でB3が探索点最良なのは **0/40**。旧研究の33/40は旧集合内のB2対PF結果として保持し、上書きしない。

## epsilon=.0005 のB3探索nominee

|系|T|公式|q|delta|R_outer|K|s|B|N cos / sin|C cos|G ± SE|B3/B2|B3/PF envelope|
|---|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---|---:|---:|
|H4|0.8|2nd|16|0.05|16|4|256|1.6006|205063739 / 183447965|38060|1.47868e+13 ± 3.25e+11|2.872|1.757|
|H4|1.2|2nd|32|0.0375|16|4|512|1.6979|214058784 / 222529025|75847|3.31139e+13 ± 2.99e+11|5.967|3.788|
|H4|1.6|2nd|64|0.025|16|2|1024|1.6011|183526991 / 191340401|1.5337e+05|5.74925e+13 ± 1.14e+12|9.314|6.638|
|H4|2|2nd|64|0.03125|32|6|2048|1.4446|165196280 / 153438100|2.7237e+05|8.67851e+13 ± 1.05e+12|11.63|9.056|
|H6|0.8|2nd|32|0.025|32|6|1024|1.6349|192218029 / 195303160|4.3876e+05|1.7003e+14 ± 2.08e+12|7.846|4.622|
|H6|1.2|2nd|64|0.01875|32|4|2048|1.7386|217856791 / 218411276|8.6463e+05|3.7721e+14 ± 2.72e+12|15.42|10.18|
|H6|1.6|2nd|64|0.025|64|4|4096|1.635|204628061 / 195785321|1.6832e+06|6.7397e+14 ± 3.76e+12|23.65|19.11|
|H6|2|2nd|64|0.03125|128|6|8192|1.4683|156634715 / 183406840|3.289e+06|1.11839e+15 ± 5.44e+12|32.29|27.97|

## 高次公式の独立確認

各公式のepsilon=.0005探索nomineeを事前固定し、別cohortで比較。確認後minへの選び直しや探索とのpoolingは行っていない。

|系|T|公式|高次/S2 G|差のSE|
|---|---:|---|---:|---:|
|H4|0.8|4th|14.8063|1.116e+12|
|H4|0.8|4th(m5_best)|1.96082|3.155e+11|
|H4|1.2|4th|15.4413|1.917e+12|
|H4|1.2|4th(m5_best)|2.03672|6.171e+11|
|H4|1.6|4th|15.3509|2.124e+12|
|H4|1.6|4th(m5_best)|2.03617|7.583e+11|
|H4|2.0|4th|15.7063|1.835e+12|
|H4|2.0|4th(m5_best)|2.07822|9.661e+11|
|H6|0.8|4th|17.5012|5.255e+12|
|H6|0.8|4th(m5_best)|2.27858|1.752e+12|
|H6|1.2|4th|18.1459|9.004e+12|
|H6|1.2|4th(m5_best)|2.38105|3.668e+12|
|H6|1.6|4th|29.478|2.908e+13|
|H6|1.6|4th(m5_best)|2.33813|4.025e+12|
|H6|2.0|4th|68.0943|9.739e+13|
|H6|2.0|4th(m5_best)|2.17369|4.913e+12|

## 実装・科学入力・数値正しさ

基準はcommit 155529e20a8fac39e00786c40b57830b71203c4f。h46_long_time_signal.evolve_batch/qualify、h46_new2_coreのspectral・paired polynomial・matrix_only_tail_coefficients、既存signed PF scheduleとABS_TIME_PER_OUTER整数配分をそのまま利用。
nativeはh46_long_time_nativeのsampling、compose、cost_trial、fingerprint/compiler cache、summarize_trialsとOccupationOracleを再利用。B3ではone-bodyだけが決定論側にあるため、oracleの不要な保持DF spectrum生成を省いた。新規の独立RTE実装は作成していない。
B3はL_D=0で全DF二体fragmentをrandom tailへ置く。one-body correctionは決定論Gaussian blockとして残り、Hamiltonian constantとtailから抽出したidentityをscalarとして一度適用する。全Hamiltonian純RTEとは呼ばない。
occupation演算によるtail再構成の最大作用vector差は7.185e-15。H4/H6・正負時間・K2/K4/K6の条件付きeventとactual adjointを確認した。条件付き確認は費用標本へ混ぜていない。
symmetric-directional control、scalar一回、actual adjoint、測定付きcosine/sine full Hadamard wrapperを使用。Qiskit1.3.0、basis rz/sx/x/cx、opt1、seed17、backend/coupling/layout/routing追加指定なし。代表native/compiled作用vectorを独立occupation oracleと照合。
B、bias、Nは同じ保存input・lawで評価。direct finite-time corrected complex-signal biasを主判定に用い、rawとのnormalization closureを確認。normalizationの二重適用なし。u floor1e-9、alpha_axis=.025、旧共通log_B margin・headroom/u政策を維持。厳密reference/compiled残差保証とは区別する。

## 探索範囲・候補選抜

signal 3432構成、精度endpoint 17160。基本1584構成に対し、R64境界・基本不適格のq/K sliceへR128/256/512を追加し、必要なsystem/T/formulaだけq32、続いてq64まで追加。詳細理由は[grid_extension.json](grid_extension.json)。T>2や別分子には拡張していない。
native価格は192候補、探索1728 trajectory、独立確認1553 trajectory。top2のshot×work proxy、K別、低normalization・低単回価格、delta=.2対照を残した。proxyはcompiled下界でなく、未計測候補の劣位は認定しない。
探索格子の未評価域と価格未取得域はcandidate_inventory、grid_extension、candidate_precision_accountingに明示。q64/R512の境界に達する候補もあり、その外の最適性は未確認。s=q R_outerが同じでもone-body/PF scheduleやcompiled費用が同じとは仮定していない。
旧B3 H4 rank12・1.30 Åの価格とPauli純RTEのs512は今回へ転用していない。旧コード・計算方法だけを条件照合して再利用した。

## 独立確認・rare event・不確かさ

探索9 trajectory/候補。探索最良の各T×epsilon nomineeと厳しい精度の各公式nomineeを固定し、探索分散×4、名目相対MCSE2%、最低33で独立確認nを取得前に固定。別seed namespace/cohort、確認後top-upなし。cos/sinは同じtrajectoryでpaired、異なる候補間はpairingしない。
独立B3対保存baselineは[independent_comparisons.csv](independent_comparisons.csv)。B2側は保存独立cohortがある場合に限って再利用し、ない場合は探索baselineと明記。正規近似95%区間・Bonferroni感度は有限標本診断であり、希少eventを含む母平均の厳密保証ではない。
Kは最大偶数event次数、polynomial最高次数はK+1。supportにないevent、数学的確率正だが未観測、実効float-CDFで到達不能、観測event、条件付き価格未取得を[rare_event_cost_status.csv](rare_event_cost_status.csv)で区別。未観測費用を0とみなさず、母平均tail上界はnull。条件付きgate正しさ確認は、条件付きcompiled価格測定を意味しない。
今回の取得済み順位は希少event全lawの母平均優位を確定しない。有限標本のゼロ分散も母分散ゼロとは解釈しない。

## 保存・図表

[全40比較](comparisons.csv) / [最良構成](best_configurations.csv) / [q依存](best_by_q.csv) / [R/K依存とB/N/C/G](R_K_sensitivity.csv) / [同一q/R/K対照](matched_B2_B3_controls.csv) / [delta=.2](fixed_delta.csv) / [全6 native指標](secondary_native_metrics.csv)。
[T対費用](cost_vs_time.png) / [最適q/delta](selected_q_delta.png) / [R増加のB/N/C/G](R_outer_B_N_C_G.png) / [四方式map](expanded_best_method_map.png)。各図はSVG/PDFも保存し、旧図を変更していない。
private event/seed、trial/checkpoint、fingerprint cache、代表QPYは PRIVATE_B3_ASSETS/ 。原研究・統合論文worktree・旧grant/ledger/marker/STOPは編集していない。
結果の主要claim正式採択・詳細科学レビューは実施していない。今回の科学実行・図表・公開終了後mandatory STOP。次の作業なし。

2026-10-27 09:36 JSTにユーザー用systemd管理プロセス全体がexit.targetへ移行し、B3と終了確認serviceを停止した。停止要求者は読取可能なjournalからは特定できない。完成1,691件を保持し、中断17件のdirectoryを元のまま保全して、未着手20件とともに同一固定計画の続きを実行した。検証済み高速worker・入力・cohort・n・seed・compilerは維持。ホストの既存KillUserProcesses=falseの下で端末から切り離した自分のprocessへ移し、ユーザーmanagerや共有設定は変更していない。科学的STOPの再開や新たな失敗の自動retryは行わない。[中断と継続](interruption_user_manager_20261027.json)。

実装高速化の別検証：保存済みH4/H6・B3 L_D=0、linear1Å/STO-3G/DF10・19、T=.8/q8/S2、H4 R4/K2・H6 R8/K6の各代表seedを元実装・高速版で再実行した。各実行で両軸を同じQiskit1.3.0/opt1/seed17で新規compileし、価格6指標、event byte、canonical fingerprint、全native/compiled作用検査が一致した。時間はH4 23.79→12.36秒（1.93倍）、H6 103.55→63.34秒（1.63倍）。統計標本には追加せず、全研究の加速率を示すものではない。同じcanonical byte列を得る検査実装を復旧・独立確認で使用した。[対応入力の検証](acceleration_validation.json)。

空きCPUで同一の固定探索seed・native builder・compilerを用い、未着手trialのcompiled cacheを先行作成した。先行cache完成41 trial相当、新規compile 82軸、既存cache hit 0軸。fingerprintごとの既存排他lockにより同じ回路の重複compileを防止。本体のnative/compiled作用検証とtrial受理は省略せず、先行処理を追加の統計標本として数えていない。本体の将来メモリ増加も予約し、比較可能な大きさのfresh full trialがPASSしてから25%の余裕を付けて補助側の予測を更新した。実行中の科学SOURCE・controllerは変更しなかった。[先行compile記録](parallel_prefetch.json)。

## 終了時の技術確認

人工・必要回帰13件と復旧cache/byte照合の人工2件PASS。atomic永続化writerの人工入出力も確認した。両cohortの全3281標本はPASS・worker回収済み。重複標本0、cohort内/間の新seed衝突0。最大native vector差5.6333e-12、compiled差3.8299e-08、mean/native log_B差1.7764e-15。元baselineと記録した実行SOURCEの使用fileは不変。eventは取得byteへの復元一致を確認したgzipで保存した。ただし、以下の欠損event再構成について元byteとの同一性は主張しない。これらは実装整合性の確認であり、母平均優位の証拠とは区別する。

元poolは106 trial完成後にworkerが異常終了し、pool全体が停止した。OOM記録0、CPU/AS/DATAのOS上限は無制限で、原因と元workerのexitcodeは未確定。元のSTOP・結果・progressをprivate interruptions/run01と[公開記録](interruption_run01.json)へ保持した。完成106件は再取得せず、同じ候補・n・cohort・seedで未完了trialだけを個別fork processへ復旧した。保存済み未完了eventは再現byte列を照合。検算のrotated-number行列を同じ式で事前cacheしfork共有した。復旧cacheの代表回路でもQPYを復元してcompiled作用チェックを実施した。旧研究のgrant・one-shot・STOPは再利用していない。

個別fork復旧では3件のSIGBUSを記録し、その後controllerとworkerが不在となった。完了記録134件のうち1件のresult JSONと2件のevent archiveが空であり、空の原本を保全した。有効な133 resultを再利用し、欠損result1件は同じ固定seedの未完了分として復旧した。resultが有効な1件のevent記録だけを固定seedから再構成し、保存済み両軸native回路fingerprint・compiled価格・log_B・event次数との一致を確認した。元resultのSHAは不変、新規価格観測・compile・signal再取得は0。この再構成で失われた元byteとの同一性は主張しない。以降はfresh Python spawn、自分専用transient user service、atomic fsync保存、fsync後の可逆archiveへ移行した。科学kernel・入力・候補・n・cohort・seedは維持し、旧STOPと欠損原本は保持した。SIGBUSとcontroller消失の根本原因は未確定。[中断](interruption_run02.json) / [欠損](data_damage_run02.json) / [event再構成](event_log_reconstruction.json)。

fresh Pythonのrun03でも31件のSIGBUSを記録した。正常workerは自然終了まで継続し、完成済みtrialと元の失敗記録を保全した。小さい固定seedの診断再実行から2件を同じcohort/indexの唯一の受理標本として転用し、他の未完了分のみ1回の復旧passで実行した。科学kernel・入力・compiler・候補・n・seedは不変。追加の独立標本や探索/確認のpoolingは行っていない。再発時のsiginfoをstraceで記録し、新たな異常の自動反復は行わない。復旧したこと自体は元SIGBUSの原因解明を意味しない。[run03と復旧の記録](interruption_run03.json)。

高速版への引継ぎでは、投入controllerだけをSIGSTOPで一時停止し、その時点の科学worker 63件が全て終了してからcontrollerだけにSIGINT/SIGCONTを送った。科学workerへのsignal送信は0、待機中の非正常終了は0件。完了した1147 trialを保持し、元の固定計画の残りを同じcanonical byte列を生成する高速実装へ引き継いだ。停止中の元controllerの進捗counterは凍結値として保全し、完了数は実際の保存結果から取得した。
