# H8 s2条件付きpilot：停止signal診断の修正と並列化点検

2026-10-18 JST。利用者の「修正を行って、並列化と高速化も確認」に対応した技術修正。科学sampling・実H8 compile・retry・有効grant発行は0。run01のSTOPを保持し、修正は未稼働の独立clone内の専用branchに固定した。サブエージェントは使っていない。

## 修正内容と根拠

[run01停止報告](track_a_h8_conditional_s2_pilot_STOP_20261018.md)の主原因は、旧receiptにないpartition_hashを要求したmetadata参照ミス。先行SOURCE `7522d74be3f59a8be1b435b9d05980bf6a1485b3` のschema修正は維持し、今回、16-mode・prefix7の人工DF native preparationと旧5項目receiptの実接続も検査した。元preparation hash・lambda/scalar・thresholdの完全照合は緩和していない。

副次的なBasisFingerprintErrorは、旧workerのSIGTERM handlerがRuntimeErrorを投げるため、元DF basis fingerprintの `except Exception` に捕まり変換され得る。固定providerの同じ関数へ人工例外を注入し、この変換経路を再現した。元run01には例外chainがないため、当時の3件がすべてこの原因だったと断定しない。

新SOURCE `03479a7546a994bf9d02561d1ab74cec67201b4f` では以下を修正した。

- own workerの停止signalを専用BaseExceptionへ変更し、通常の回路変換例外で包み直されるのを防ぐ。workerは非zero退出し、成功扱いしない。
- STOP receiptへsignal番号、元exception chain、frame localsを含まない上限付きtracebackを保存する。
- native preparationと保存receipt照合を別phaseで観測し、開始・所要時間・COMPLETE/STOPを区別する。
- 最初のSTOP保持、所有PID/start/parent/uid・pidfd限定停止、wait4/reap、非ESRCH時のfail-closedを維持する。

provider・既存venvは変更していない。sampling law・event規約・native gate生成・compiler options・科学候補・ledger/carry・call上限に変更はない。fixed compiler/profileは元契約のまま。SOURCE closure8と既存補助testのhashを[検証bundle](../../artifacts/resource_applicability/track_a_h8_conditional_s2_worker_fix/2026-10-18/README.md)へ固定した。

## 人工回帰

pilot28件＋既存owned worker4件、計32件PASS。pilot28件は前回20件を含む。前回のnative人工22件を今回再実行・加算したわけではない。

| 対象 | 確認したこと |
| --- | --- |
| 旧エラー変換 | RuntimeErrorの停止要求がBasisFingerprintErrorへ変換される人工再現 |
| 修正signal | 同じ固定provider経路を専用停止例外が通過し、signal番号を保存 |
| 真の不良 | 人工matrix変換不良は引き続きBasisFingerprintError、元ValueErrorをchainへ保存 |
| 実owned child | 自分の最小人工childへSIGTERM、非zero退出・回収・PID不在 |
| 並列scheduler | 最小2 workerが両方待機へ到達してから解放され、同時稼働・全回収 |
| preparation receipt | 実データを使わない16-mode DF/prefix7のnative表現と旧schemaの完全照合 |
| 既存保護 | first STOP・ESRCH/FD・所有不一致・予約・無効grant・one-shot・改変拒否 |

追加人工fixtureの初回はexact-zero成分を含み、既存のthreshold_dropped==0検査でFAILした。契約を緩和せず、非zero人工fixtureへ変更して最終PASSした。元の失敗logは保全しhashを記録した。

最終pilot suiteは3.334秒、既存pool suiteは0.239秒。実H8入力読込・sampling・full-wrapper compileは0。人工DFのbasis preparationでは固定provider内部の回路処理を使用しており、その処理まで「compile呼出0」とは主張しない。新しい大きなstress campaignやnative full-wrapper人工compileは行わない。これは主担当の自己検査であり、別担当の独立reviewではない。

## 並列化・高速化の評価

現在の固定pilotに対しては、独立trajectoryの並列化と重複削減が実装されている。

- 各候補8 trajectoryを最大8 workerで並行実行する。別processなのでPython GILを共有しない。
- 各workerは内部thread1。Qiskit/Rayon/BLAS等の過剰な入れ子並列を避け、OSや他jobの制御を変更しない。
- cosine/sineは同じevent列で、native evolutionを1回生成して保持copyから両軸へ接続する。whole-wrapper compile自体は各軸で実施する。
- canonical streaming hashとcontent確認付きLRU cacheを使い、大きなexact treeや同一operationの重複byte生成を減らす。
- 条件付き次数はpositive-binomialから直接生成し、希少s2を大量棄却で探さない。
- control検査は元契約の各候補先頭trajectoryで実施し、joint-branchの構造証明ができなければ元fallbackを使う。

q4の全8本を保存・検証・回収してからq8へ進む順序は明示契約なので、2候補の16 trajectory同時実行はしない。worker上限8は独立task数に対応しており、恣意的にCPU/RAMを制限するためのものではない。1 trajectory内の両軸を分けた最大16 compile並列の余地はあるが、追加の回路handoff/copyとRAM、call/cleanup bindingの検証が必要になる。今回のSTOP修正には含めていない。

この構成は8本ずつの限定pilotとして合理的だが、速度が最大化済みとは断言しない。実H8 compileへ到達した標本が0なので、実効speedup・終了時刻は未測定。人工2 workerの同時稼働をH8の8倍速実測と呼ばない。

## 保全・次の担当

元run01の51 file、使用済みgrant・lock・元承認・config・停止receiptの既知bytes/SHAを照合して不変。旧SOURCE checkout、既存結果・raw・cache・他worktreeへ書込しない。失敗run01の予約16compile/8trajectory/32occurrence/4controlは保持し、返却・resetなし。旧scopeのcarryを含む将来案も変更していない。

未承認plan・無効grant・未seal configを新SOURCEへ結合した。approved=false、runtime_authorization=false、allowed_cpus=[]、absolute_launch_command=null。無効grant CLIは科学import/lock前にexit1で拒否。新SOURCEの科学実行は0。

次の担当：ユーザーが新しい一回実行を認可するか判断する。明示承認後に限り、Codexが新grant・未使用output/control/lock・fresh資源gateを結合して実行する。今回はsource・人工検査・軽量資料の固定と公開試行まででmandatory STOPする。
