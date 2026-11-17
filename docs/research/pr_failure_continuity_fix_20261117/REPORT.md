# PR探索：例外・回収・局所UNKNOWN・再開の修正

固定修正SOURCEは `314524a913ba6e66b9069ab1514b1df765f9a5e2`。branchは `codex/b2-rpe-pr-failure-continuity-20261117`、基点はR08入力commit `8de068f60e29f0936fdfb13ada9026e709ce3e41`（code `07b9c1ce826ba67e7e8237b1d1c781a2e1357e6d`）。

依頼された4点を修正し、既存82件＋追加10件＝92件、native Pythonで追加10件がPASSした。R08の稼働SOURCE・入力・旧STOPには適用していない。今回の新規H4信号・sampling・回路build/compile・fit・実Q3・科学retryは0件。

**旧R08の16:00 JSTのERRORについて、具体的な例外名・原因は依然未確定**。当時はtracebackが保存されず、main側のunwindが準備thread待ちに入るため、準備だけが続く状態だった。今回、同じ制御経路の人工例外を再現して修正したが、「旧R08の具体的な例外そのものを再現・解消した」とは主張しない。[読取観測](READ_ONLY_R08_OBSERVATION.json)。

## 1. 例外を回収待ちより前に保存する

既存 `rpe_method_optimizer` に小さい `record_runtime_error` を置き、例外名・message・同期的なtraceback・method/phase・単調時計・UTC・PID/threadを `RUNTIME_ERRORS.jsonl` へflush/fsyncする。localsや全科学payloadは取得しない。記録失敗時は元tracebackをstderrへ出し、元例外を保つ。

`round_phase` は従来のSTART/ENDを保ち、ERRORの詳細を直ちに記録する。method futureには完了callbackを置き、別methodの終了待ちで例外の保存が遅れないようにした。起動・receipt保存・終了記録の例外も、その例外が存在する箇所で記録する。

最初の例外objectをshared executorに保持し、他methodの回収中の二次例外が原因表示を置き換えない。通常driverは `RUNTIME_EXCEPTION` とerror locatorを保存する。回収中はerror記録に `controller_state=DRAINING_AFTER_EXCEPTION` があり、process生存を正常探索の根拠にしない。

## 2. 実行中のstageを回収し、次stageへ進まない

例外通知はログ保存より先に出す。新規Q3投入は既存のfailure処理で止め、実行中のworkerは強制終了しない。準備には既存 `stop_requested` のeventを渡し、次候補だけでなく `Completion.call` の未開始stage直前でも確認する。

- 保存RETURNが成功なら、その結果を再利用できる。
- 既にSTARTEDだけがある、またはRETURNが失敗なら、元のNO_DUPLICATE / NO_RETRYを優先する。
- global failure後、まだ始めていないstageは `PreparationYield` で返し、候補はSELECTED/COMPLETIONのまま、yieldしたstageを残す。失敗した科学callとして再ラベルしない。
- 走行中stageのPopenをkill/terminateしない。終了receiptを保存してから境界でyieldする。
- producerを回収してから共有KAPPA stateを保存する。cleanup自身の失敗も別に記録し、一次例外を隠さない。

人工検証で、ログを書いてから停止eventを立てる順序では、native Python3.11でproducerが先へ進める競合を検出した。[修正前の再現](checks/BEFORE_NOTIFICATION_RACE_NATIVE.txt)。停止通知→queue保護→例外保存へ順序を改め、phase直下の例外も保存前に通知する。

## 3. 証拠のある局所UNKNOWNだけを保留する

UNKNOWNは採用可能なMSE結果にはしない。`Executor.scope_unknown` は元REQUESTのconfiguration/shot/job key、既存 `same_route` のexpected bindingとrouteを確認する。adapterが任意に付けたfailure_scopeを信じる処理ではない。

| 証拠・原因 | 扱い |
|---|---|
| `COUNT_ONLY_SCOPE_EXCEEDED`、科学API call数が整数0、元REQUEST/route一致 | CONFIGURATION保留。試したallocationはscope不適合だが、方式/law全体の不適合とはしない |
| OUTCOMEなし、同じREQUEST/route、終了receiptの明示SIGINT/SIGTERM | CONFIGURATION保留。再実行しない |
| identity/contract/SOURCE不一致、数値gate失敗 | 元のglobal UNKNOWN保護を維持 |
| SIGKILL/SIGSEGV、MemoryError、原因未分類、共通保存障害 | global保護を維持。SIGKILLだけをOOMと断定しない |

局所保留は元の `execution_holds` と `SAVED_JOB_REVIEW_REQUIRED` を再利用する。既存適合点は保持し、独立候補・他方式を続ける。試行budgetは消費済みattemptを含め、API call数0とattempt数1を混同しない。数値UNKNOWNを局所保留へ一括変換してはいない。

## 4. 保存結果から二重実行なしで再開する

Completion内部のtail descriptor補完がcallerのrequestを変更していた。後続呼出しでkeyが変わり、前のstage directoryを見落とすリスクがあったため、内部処理はrequestのdeepcopyへ限定した。

過去に補完後requestが元keyのrecordへ保存された場合、元のJSON表現（signed zeroや整数/floatの区別を含む）の**完全一致が一意**な場合だけ、その元key/directory/STOPを保持する。複数候補が一致する場合は保留し、新attemptを作らない。別候補・物理scheduleの類似性では結び付けない。

今回のyieldからの続行は、既存RETURNの再利用と未開始stageだけが対象。過去の失敗RETURN、曖昧なSTARTED、科学STOP、実行SOURCE不整合を解除しない。通常CLIの技術STOPも自動で消さず、修正SOURCEを用いる引継ぎと区別する。

## 検証

全てこの担当による人工・回帰検査で、独立レビューではない。

- [CORE](checks/REGRESSION_CORE.txt)：58 PASS（既存48＋追加10）。候補準備、保存価格、route、scalar/context、局所保留、通常CLI失敗・再開。
- [QUEUE](checks/REGRESSION_QUEUE.txt)：34 PASS。並行準備、遅着READY、有限budget、prefix経路、通常/再開、元の候補選択・価格計算・更新規則のAST照合。
- [NATIVE](checks/NATIVE_FAILURE_REGRESSION.txt)：Python3.11.0rc1で追加10 PASS。元Python3.12.3環境と両方で確認。
- [通常入口の人工証拠](checks/NORMAL_ENTRY/)：元 `run_rpe_schedule_search.main` とconcurrent executorから故障を注入。producer回収前の例外保存、通知済み状態、後続準備0、RUNTIME_EXCEPTION保全、同run再開の追加LAUNCH 0を確認。
- 実際の人工childを動かし、停止eventが立っても終了コード0で回収すること、次stageのSTARTEDが作られないこと、成功RETURNの再利用・失敗RETURNのNO_RETRYを確認。
- 最初に失敗したmethodがmethod一覧の後方でも、先行methodの二次例外で隠れないことを確認。
- request不変、過去の補完済みrequestから元keyへの再接続、signed zeroの非混同、旧失敗STOP保持を確認。
- 局所UNKNOWNで科学API callが0でもattemptを1件消費し、残りの改善提案があっても固定budgetを超えて呼ばないことを、旧選択規則とV2の両方で確認。

科学的な結果・精度・価格・実H4性能の改善を示す検証ではない。

## 非変更・残る条件

H/state/snapshot、scalar、lowering、compiler、wrapper、law、seed、q/R、DP kernel/AST guard、MSE判定、価格モデル、shot候補の順位、小改善閾値、κ打切り、cycle上限は変更していない。追加の承認証明・grant・queue・管理frameworkは作っていない。

R08はread-onlyであり、進行中の価格取得・旧raw・旧SOURCE・cache・marker・STOPを変更していない。新SOURCEをcommitしただけでは、そのprocessへ適用されない。今後の引継ぎでは、最初にcontroller/所有workerと保存receiptを照合し、完了済みの信号・価格・Q3を再利用する。現R08の具体的な例外は、自然終了後のtraceback等を回収してなお確認する必要がある。

[HANDOFF](HANDOFF.md)にcode・通常入口・利用法・引継ぎ条件を集約した。ロードマップ本体は編集せず、「失敗の即時可視化と協調回収は人工検証済み、R08の具体例外と実runへの適用は別状態」という差分を渡す。
