# Signed support fresh pilot：再利用と最小activation差分

2026-11-07 JST / SERVER1。**静的SOURCE調査と設計のみ。以下の新API/adapterは未実装で、人工検査も未実行。**
固定条件・未採択値は[実行草案](SIGNED_SUPPORT_FRESH_PILOT_EXECUTION_DRAFT.md)を参照する。

## 1. 確認した版・資料

| 固定版 | 静的に確認した対象・役割 |
|---|---|
| b4512675ae1d650858256547d359210361d77f68 | metadata結果REPORT、METADATA_RESULT、SOURCE、CALL_COUNTS、STOP、入力identity。Σ1>0とUNKNOWNの境界 |
| 71778dae6954ed9a7a6821062a2a6e6be17672c4 | metadata-only入口/設定/テストのSOURCE。fresh科学入口ではない |
| b23b3c059dd1fdcc86178659311983048dea2257 | scv_signed_support、SignedControl/SignedFrozenCV/fit_h1/paired_evaluate、人工テスト・実装報告 |
| 560c6092bfb5e71af8eedc4a4be5416922764f98 | scv_scalar、a1_control_variates、a1_n3_connection、scv224_pilot、a1_n3_campaignの基盤 |
| 996e434e63ce9b1d14c6ae191649c9abe213cf47 / ba483ab3038693048e58830e39a676493a892053 | 旧fragment indicator activation・EXECUTION・学習/価格/STOPの構造。旧bit/係数/実価格の新再解析なし |
| af47cf9161241c8710383ff8539a4a7f8f67492c | SCV_FINITE_CONFIDENCE_MAP、SCV_TOTAL_COST_COMPARISON、SCV_REUSE_AND_VALIDATION_MAP。探索的診断と後段保証の分離 |
| N1 744e1a09661ebba6841fe5faf456f88c41c47780 / N2 5e9bd08068738cdf1070bdc43bee8194b113ae33 / engine fdf00cee3c87d804845497afc89ac6a71f3b3f59 | 保存law/policy、保証ground、exact kernel/proofとloaderの接続 |

既報126人工PASSは保存報告の参照。今回はSOURCEを読むだけで、project moduleのimport・テスト実行・bundle load・PRNG作成・実measurementをしていない。af47の旧「実H4未取得」等の時点表示を現在へ転記せず、signedの取得状態はb4512675から読む。

## 2. 取得済みinput・controlの固定

N1の `artifacts/o2_a1_new_target/2026-11-06/n1/LAW.json` と `PREPARATION.json`全bytes、N2 state/witness、engine proof/policyは既存の固定bundle/loader照合を使う。LAW SHA256は `b77980d198cc9edc620daec9bf87df02b66e96575fd85327aed0793f3e86ecef`、catalogは `ca897db7be52ad30dcfab0512d7530188fc13a9d8ca0c78430809492d01f738b`。

b4512675の共通relative directoryは `artifacts/scv_signed_support_h4_metadata/2026-11-07/run01/`。

| 保存ファイル | SHA256 | 用途 |
|---|---|---|
| METADATA_RESULT.json | `c5895f7696711c92bf87e591a3c2264cd7672d9af70d13027db790950e305ba0` | exact p/ν/m₂/Σ、metadata/control identity、feature、構造的ELIGIBLE |
| INPUT_IDENTITY.json | `4b5b46e4acadc7fe5483bab799394e19045bfc2e12bc28f87ff73549f66e0e1e` | N1 input、H/preparation等の対応 |
| SOURCE.json | `814477d1916250af87c8922cde03a2680f51d519c1e2aea61f22c5d2c6b277be` | metadata科学SOURCEと環境・固定条件の来歴 |
| STOP.json | `45340ae0ff408782e2e7810c864a6123816006dba48535cd6ddd8c856d4befc5` | 旧metadata attemptの終了を保全。解除/再開しない |

期待metadata identity=`c5e0c5c26b926eb4a73d1808a1c61222dc45934248c30d8aea4f8fc32c44b7e2`、control identity=`5e252ff9fa5682850683ced97d007b59fecbf6d7617c561adb4a34d4cc7c7f89`。fixed a,b/qubits8/feature version/patternを元binding文書と照合する。hash不一致なら別版へ追従しない。

## 3. 重要な差分：保存momentを再計算せずに復元する

既存SignedControl(metadata)は__post_init__でlabel有限和を計算する。fresh実行で通常コンストラクタを呼ぶと、今回の「metadata-only計算を再実行しない」という条件に反する。単に同じ入力で一回作り直す案は採らない。

提案する薄い `restore_saved_signed_control` は次の限定操作だけを行う。

1. 固定result bytesのhash/commit/pathとsnapshot/各cellのexact文字列一致を確認する。
2. 既存signed.metadata_from_bytesでLAW/catalogをbindし、metadata.identityが保存値と一致することを確認する。新しいsigned momentは計算しない。
3. 保存されたmeans/second_moments/variancesをexact Fractionへdecodeし、feature version・field数・bounds/identity/provenanceを検査する。p_hも保存値を保持する。新しいlabel和やH4特徴の集計で再評価しない。
4. 既存scalar.fingerprintとb23のidentity payload形式で、**保存mean/m₂を使ったhash**が固定control identityと一致するか確認する。これは保存値のbindingで、momentの数値再計算ではない。
5. SignedControl互換のfrozenな保存オブジェクトを復元する。既存のisinstance(control,SignedControl)を維持する専用の小さいsubclass/restore factoryで、metadata/means/second_moments/variances/identityの既知fieldだけを復元する。通常__init__/__post_init__は呼ばず、既存value/snapshotメソッドを継承する。汎用オブジェクト注入/任意field復元にはしない。

この復元経路は**新しい最小adapter**であり、既存SOURCEに存在すると表示しない。人工検査で通常momentコンストラクタを禁止しても復元・fit/pairedが動くこと、誤ったhash/mean/identity・mutable field混入でfail-closedすることを確認してから採用する。適合しない場合は再計算へfallbackせず、実科学前に停止・判断へ戻す。

## 4. APIごとの再利用と不適合箇所

| SOURCE/API | そのまま再利用する部分 | 新しい接続/注意点 |
|---|---|---|
| scv_signed_support.Metadata / metadata_from_bytes | sign/support、qubit、LAW/catalog/feature binding | saved resultへのhash接続を外側で追加 |
| SignedControl.value/snapshot | 最後のrotation labelのsigned feature、保存field表示 | 通常moment構築は不使用。上のtyped復元が必要 |
| signed.fit_h1 / SignedFrozenCV / SignedModels | scalar OLS・rank2/finite、binary64→Fraction、conditional centering、signed range、inactive4層0、stream/scope検査 | 外側にD≤6のguardを追加。kernelのguard=NOT_ADOPTED表示は「kernel自身に数値guardなし」の意味で保ち、外側の採択guard/判定を別fieldに保存 |
| signed.paired_evaluate / scalar.paired_evaluate / production_mean | 明示quota、同じrecord、全5層mean/variance/covariance、signed値counts | fullと固定half用quotaをcallerが渡す。CIや十分Nを返す関数ではない |
| ConditionalSampler / Domain / Streams / pcg64_streams | actual Q、直接条件付き取得、products→rotation、独立domain・重複/word上限 | 新seed/24子mapping・campaign namespaceだけを外側で固定。旧3-control bankは層定義の道具で、旧推定器を学習しない |
| scv224_pilot.Envelope / cost_summary / cv.account_costs | physical一回課金、stage別/参照費、UNKNOWN・重複検査 | new purpose/controlを外側validatorで確認。未完了groupはworker reconcileを正本にする |
| scv224_pilot.Driver / LegacyCollector | SOURCE/受領先行保存・独立性/価格検査・STOP先行の設計 | **そのまま不可**：PURPOSE、32-train、旧quota/224-cap、ν1∈[1/8,7/8]のindicator balance、scalar.fit_cells/indicator rangeが固定。module globalを書換えてsigned化しない |
| a1_n3_campaign.prepare_trial / Campaign.group / worker_trial / reconcile | path/word先行journal、PinnedSampler、bit-before-price、同full価格、全tail分類 | new purpose/configの薄いcollectorでraw groupを呼ぶ。legacy.run/learn/executeは呼ばない。prepare/submit/call境界へSTOP/budgetチェックを追加 |
| SavedEngine / MeasurementFacade | 認証済みstate/gate/probability・53-bit測定。CVへbit/pathだけ | load_certified_servicesはparent一回、saved proof attach、forkで継承。state/真信号はfitへ渡さない |
| SavedPrices.price / CompilerSettings / cost cache | 同じselected full、Qiskit1.3 opt2 seed17、cache hit課金 | 新cacheのみ。測定/価格でfull buildが計2/trial。旧実現価格は使わない |
| Runtime / ManagedPool / observer / qualified cron/outer/manager | signals、CALL記録、durable STOP、owned child終了・回収、consume-once/Restart=no | new campaign/spec/output/semantic exitのroutingを最小接続。CALL_STARTとSTOP遷移の短い共有排他・有限counterはadapterで接続 |
| 既存classical timers | parent/group/worker、engine/price/compile、RSS | probability inclusive/build/arithmetic/cacheを区別する観測adapterを追加。worker wall和をCPUと呼ばない |

実装差分の第一候補は、**saved-control復元1箇所、新purpose/quota/guard/phaseを持つ薄いcollector・orchestrator、既存managerへのactivation/CLI、必要な観測/STOP境界**。新sampler、engine、回路builder、OLS、統計/会計framework、持続managerを作り直さない。

旧Driverの多くのvalidation原則は再利用できるが、固定constant/indicator仮定のため親constructor/runを無理に呼ばない。新adapterの必要箇所に限って元検査の小部分を適合させる。metadata旧balanceを外すのは安全条件の無断緩和ではなく、新しいsigned controlに不適合な旧indicator選択規則を採用しないという差分である。

## 5. Engine・価格・managerの接続とcall観測

parentのload_certified_services一回で、saved N1/N2/policy/proofをdecodeして既存facade/prices/scopesを作る。metadata samplerとの完全一致後に同じsamplerへ統一し、fork poolへ渡す。worker_initはcompiler/cache/observerだけで再decodeしない。既存runtime dependency/native profileとstrict engine certificateを維持し、O1/O2高速化SOURCEを探索・取り込まない。

managerは旧SCV224で用いたconsume-once cron→outer recorder→manager→private parent/sessionの方式を再利用する。既存own crontabがあれば上書きせずSTOP。triggerをapplication前に撤去し、想定cgroupと親/worker identityを確認。Linger=noの下でnohup/tmuxのみを完全持続性としない。現在の別job/serviceは操作しない。

新collectorのpurpose/source/stream/price検査は任意parameterで曖昧化せず、新campaignの凍結設定へbindする。global trial IDにはcampaignを含め、旧local ticketの再利用をglobal一意性と誤認しない。fresh output内のjournalとPRNG word来歴を正本とし、旧Envelope/rawを読まない。

既存ManagedPoolはsubmit/worker/callbackのSTOP原則を持つが、raw Campaign.groupはprepare_trialをsubmitより先に呼ぶ。従って、STOP後に新pathを生成しないためのprepare前guardも必要。call境界の短い排他でSTART/STOPだけを線形化し、engine本体は並列のままにする。共有STOP・counter・観測は自分のjobに限定し、旧kernel fileを変更しない。未開始cancel、受理済みin-flight回収、bit先行保存・価格欠測UNKNOWN・停止費を含めてfinalizeする。

compileやexact kernelをtiming目的で置き換えず、同じ関数の前後を観測する。追加timerでsource科学意味が変わらないことを人工に確認する。現在のlegacy「cache」timerは価格build等も含むため、そのまま純cache費と表示しない。

## 6. 次回必要な人工検査：今回は実行していない

既存test_scv_signed_support、test_scv_scalar、test_scv224_pilot、test_scv224_activation、test_a1_control_variates、test_a1_n3_connection、test_a1_n3_campaign、resource admissionとmanager routingの検査を優先再利用する。旧テストを新条件の検証に読み替えず、次の差分だけを追加する。

| 検査 | 必要な確認 |
|---|---|
| 保存controlの復元 | 手書きtoy保存moment/hash、通常SignedControl.__post_init__を禁止、同一metadata/control identity、immutable field、別LAW/catalog/値混入拒否 |
| signed fit/guard | h1の複数signed値、axis独立fit一回、rank不足/nonfinite、actual Fraction/range、D=6境界/超過STOP、indicator rangeやbalance guardが入らない |
| stage凍結 | X/Y trainを先に両取得、X→Y fit、両PASS前のeval word/bit取得拒否、guard失敗全費用保全、refit/追加shot禁止 |
| quota/paired | full(8,128,8,8,8)、各half(4,64,4,4,4)、trial ID順、同じfrozen model、全5層合成、6解析でphysical価格を増やさない |
| stream/identity | 新campaign/seed/12domain/24child、重複claim・別軸/別stage/旧namespace拒否、worker replayが新PRNGを呼ばない |
| call budget | mock448の一campaign、decode1・fit2・full build≤896・compile≤448・word≤29,120、上限後のbody拒否。静的上限と観測値を分離 |
| STOP競合/部分保全 | worker failure/親signal/STARTとSTOPの競合、STOP後の新START拒否、短いlockの外で並列維持、未開始cancel、bit-before-price、UNKNOWN tailと未返却費 |
| 費用/時間 | cache hit課金、train128込み、common eval320は一度、未完了envelopeだけで完備扱いしない、inclusive timerとCPU/RSSの区別 |
| manager/環境 | 最小cron環境・新routing、consume-once/Restart=no、existing STOP/owned crontab保護、Source/runtime/compiler mismatchで実science前STOP |
| 漏洩/禁止 | fitにstate/Born/E0/gap/path信号を渡さない、N1/N2/認証/metadata有限和の再実行0、未認証float/GPU不使用 |

実H4をsmoke testやbenchmarkにしない。人工テストのPASS後にSOURCE/env/inputを固定公開し、実scienceを一回開始するには別途の明示許可が必要。人工で失敗したら人工だけを修正できるが、実campaign開始後はscience retryしない。

## 7. 保存・原本保全と現時点の未解決

新activation/設定/人工testsを将来の専用branchに追加し、実行SOURCE SHAを確定する。保存input/control/cert/manager/profile、全stream/word/bit/価格、X/Y fit/guard・凍結、full/half診断、全tail費用、STOPを新namespaceへ保存する。旧SOURCE・bit・係数・cache・grant/ledger/marker/STOPは変更しない。

未解決なのは、保存controlのtyped復元、new purpose/guard/quota/診断のadapter、START/STOP/予算の最小接続、timerと新activationの人工検証・公開固定。これらは設計できたが、実装済み・新pilotが稼働可能とは認定しない。科学性能は次回のfreshデータまでUNKNOWN。

今回の変更は本directoryの4 Markdownだけ。元main/dirty/untracked、metadata-only完了・224/752・N1/N2/engine・C1、別O1/O2のworktree/process/cacheへ操作しない。新科学/実装/テストは0で、公開照合後mandatory STOP。
