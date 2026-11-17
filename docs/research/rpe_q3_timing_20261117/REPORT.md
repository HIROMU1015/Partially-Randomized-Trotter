# 分散Q3：配送・回収・探索採用の計時

保存実3ホストの18人工jobから、初回割当前のqueue待ちは約 **234.916秒**、後続queue待ちはhost別平均0.560〜2.598秒と確認した。遠隔の「検証・保存後→探索採用」は平均約5〜6 ms。**Q3本体時刻が旧公開ログにないため、B配送とC回収の片道時間は分離できない。中継並列化の効果はまだ認定しない。**

専用branch `codex/rpe-q3-timing-20261117`、base `3cd8b1769ffc4df71d85bdaefce0e337becf028e`。
transport/body計時SOURCE `19c43576c83ec9de24f6059f4cb39762458c6519`、最終集計SOURCE `5897d26abecf224b8968aa7cf6ea0b4d4732c733`。
通常入口・導入条件・集計コマンドは [HANDOFF](HANDOFF.md) に一元化した。

## 既存ログから測れた範囲

quantityは運用の経過秒。RZ価格・MSEの新規集計ではない。

| 区間 | 旧18件で測れるもの | 足りないもの |
|---|---|---|
| A 候補READY→割当 | allocationをENQUEUEしてから中央LAUNCH（割当）までの既存queue_wait_seconds | 物理候補の入力READYからshot案生成まで、全適合slot使用中か参加待ちかを示す状態列 |
| B 割当→Q3本体開始 | 中央の割当から検証済み結果保存までの包絡時間 | worker本体開始、各hostの起動・dry-check時間、片道転送 |
| C 本体終了→server2受領 | worker/relayで測ったupload+ACK往復17/18件、中央の検証・永続化18件 | Q3本体終了、HTTP受信開始（旧TRANSFER_COMPLETEは検証・保存後） |
| D 受領→探索に利用可能 | 中央TRANSFER_COMPLETE→ADOPTとSEARCH_ADOPT、18/18件 | raw受信境界、次候補判断を実際に止めたdependency-wait。旧SEARCH_ADOPTはjobを持たず、同method・configurationの唯一のpending ADOPTから推定した対応 |

AのREADYは「探索器が選んだallocationが配送queueに入る」時点。configurationの物理入力READYと同じではない。旧LAUNCHは中央割当であり、worker PopenでもQ3開始でもない。旧Dは受信開始からではなく検証・保存後からの値。

## 保存実3ホスト：18件の値

出所は `36a49d6df6705425c9b1af4ce8a62cfb6f56e8c5:docs/research/rpe_distributed_q3/checks/REAL_THREE_HOST_TIMELINE.jsonl` と同dirの `REAL_THREE_HOST_PUBLIC_SUMMARY.json`。元source `1ffdf610a`、実ホストserver2/server1/RomuTail(local)、人工jobのみ、H4 Q3 0件。

初回は「全runの最初の割当前にenqueue済み」の4jobと、以降enqueueの14jobに分けた。初回host参加の完了を証明する分類ではない。hostの初回割当自体も各行のhost_phaseに保持した。全slot busyの割合は旧ログではUNKNOWN。

| host | 件数 | 初回queue群 | 後続群 | 後続A平均秒 | 割当→保存平均秒 | 中央検証保存平均秒 | 保存→探索採用平均秒 |
|---|---:|---:|---:|---:|---:|---:|---:|
| local | 5 | 2 | 3 | 0.722706 | 3.742188 | 0.021110 | 0.004995 |
| server1 | 2 | 0 | 2 | 2.598341 | 4.219034 | 0.023727 | 0.005625 |
| server2 | 11 | 2 | 9 | 0.559938 | 1.355015 | 0.017010 | 1.569353 |

Q3本体は18/18欠測。上表の「割当→保存」を計算時間や転送時間に読み替えない。server1は2件だけで、各値は次の通り。

| job_id | attempt_id | A秒 | 割当→保存秒 | 保存→探索採用秒 |
|---|---|---:|---:|---:|
| 1e889d90de500eaac5a5 | f821f4fb5c5e4c68a1488f16fb0d498b | 3.353521 | 5.038512 | 0.004750 |
| b4aa6fd9fbe1c311c3b3 | bd03cb0da23e4659bda33a6c0f907671 | 1.843160 | 3.399557 | 0.006501 |

localの5件、server2の11件も全件を [CSV](checks/LEGACY_PER_JOB.csv) と [JSON](checks/LEGACY_THREE_HOST_TIMING.json) に載せた。job/attempt/hostを一意に対応させ、同一jobのattemptが曖昧なら集計へ推測採用しない。人工MSE数値は読み直していない。

- queue待ちのjob秒合計 **953.544295秒**、同時計domain内で重複を除いたqueue区間 **243.219403秒**。後者も「次候補判断が遅れた秒数」ではない。
- 保存→探索採用のjob秒合計 **17.299110秒**、区間union **11.370422秒**。主にserver2の先着結果が後から利用された記録が寄与する。探索順序待ちと整合するが、旧dependency-wait欠測のため因果の内訳はUNKNOWN。
- 旧公開要約のqueue合計953.511335秒と新集計953.544295秒には約0.033秒差がある。新集計はLAUNCHに保存されたmonotonic由来queue_wait_secondsをそのまま合算し、旧要約・DBのepoch由来計数を置換していない。
- 中央検証保存は全18件合計0.340113秒。ACK往復は17件合計1.165449秒、server1最終通知1件が欠測。SSH status往復は複数pending attemptの一括取得を含み、各jobの値を足してネットワーク占有時間とはしない。

## 次の通常runに加えた観測

既存PROGRESS、JobStore events、worker/relay event、JOB_STAGESを再利用した。数学kernel、PF/Pauli/DF信号・price・seed・shot決定・exact閾値、割当優先列、再開・SOURCE guardを変更していない。

1. Linuxのhostname+boot IDで単調時計domainを明示。別OSはprocess単位domainへ保守的に限定。epoch/UTCは表示用だけ。
2. queueの適合host/予約slot/required-host参加状態を既存claim lock下で観測。`QUEUE_WAIT_STATE`は状態変更時だけ。全枠使用中・required host参加待ち・readiness/poll待ち・不明を区別し、未観測時間は帰属させない。処理を重くしないためDBの小さい運用列だけを読む。
3. worker durable受領、adapter process開始/終了、bundle準備をTRANSPORT_TIMINGへ保存。元Accelerator `__call__`をそのまま呼ぶ薄いwrapperでQ3 API開始/終了をQ3_BODY_TIMINGへ保存。複数callの場合は同一clock内のspanと各call時間を区別。対応Acceleratorがないrouteは欠測であり、route envelopeを本体扱いしない。
4. 中央`RESULT_RECEIVED`をcomplete RPCのpayload parse後・検証保存前に記録。HTTP body全体のネットワーク受信前ではない。`TRANSFER_COMPLETE`は既存の検証保存後を維持。
5. SEARCH_ADOPTにjob IDを渡し、保存LAUNCHからattempt/hostを結合。転送ACK、executorのADOPT、探索知識更新後のSEARCH_ADOPTを別々に保持。
6. relayのsubmit往復、poll応答後から各submitまでの待ち、status往復、ACK往復を既存metric RPCへ送る。relay本体のEVENTSにも単調時計とphase時刻を追加。optional metric失敗でdurable受領を取り消さない。

B/Cの両端が別clockなら、中央で測った割当→受領時間からworkerで測った本体spanを除いた非本体包絡を返す。BとCの各上限はその包絡を共有し、**両上限を加算しない**。片道の点推定は行わない。worker内の受領→本体、終了→bundle準備、relay往復などの測れる部分は別欄に残す。時計rateの校正を伴う厳密ネットワーク保証ではない。

## 人工・回帰検証

- Q3 store/worker/relay/計時：53 tests PASS。distributed通常入口/回収/再開：7 tests PASS。
- 元Python3.11.0rc1の人工DP回帰：10 tests PASS。別途2 callで、計時あり/なしの元DP結果exact一致、cache HIT、元AST guard保持。[native記録](checks/NATIVE_TIMED_DP.json)。初回3.12実行は既存AST guardに拒否され、guardを変更せず元環境へ戻した。
- 新計時の9 testsには、100万秒のhost時計offset、boot変更、attempt取り違え、重複受領、waitの重複、戻り値/例外保全、旧18件を含む。最終計時変更後も9 PASS。
- [通常入口受入](checks/ARTIFICIAL_ACCEPTANCE.json)：同一OS上の独立spool、loopback HTTPの模擬1/2/3hostで各18結果、参照のMSE上下界/判定/最安と全件exact一致。全ケース再開追加0件。切断・中断回収の既存人工受入もPASS。実3台の再実測ではない。
- [公開人工runの同コマンド集計](checks/ARTIFICIAL_THREE_HOST_TIMING.json)：各host6件、本体18/18と明示job/attempt付き探索採用18/18を測定。本体平均約0.121秒。全枠busy観測union 0.957077秒。Dのjob秒和3.128334秒、union 1.856936秒、要求したjobのdependency-waitとの交差union 0.261330秒。後者はログに基づく待機区間で、relay並列化した場合の反事実的な短縮量ではない。
- 人工復旧fixtureに元WORKER_PROFILE保存が欠けていたため、実workerが必ず保存するprofileをfixtureへ追加した。guardや本番復旧条件は緩和していない。

最終照合で後続e64e377a8（optional WSL）が公開済みだった。相手のadmission/dispatch/WSL transportは取り込んで再実装せず、観測側が既存admission APIを参照する互換層だけをSOURCE `1026c09b3645c1ef210def1f4678d79db2fda019` に固定した。単純適用で重なるhost dict/returnの扱いはHANDOFFに記載。e64へ統合済みとは主張しない。

## 高速化に着手する価値と残るUNKNOWN

最初は初回配置・required-host参加を済ませてからの通常runを計測するのが妥当。初回約235秒を後続配送の律速と混同しない。中央の検証保存と遠隔結果の採用はこの小さい人工結果では短く、ここを先に並列化する根拠は弱い。

relayはhostを順にpoll/status/submitする実装であり、後続配送の2.8〜5.0秒にはSSH起動やpoll周期も含み得る。ただし実H4の本体時間と、要求中jobの待機をまだ分離できない。次の承認済み通常runでB/C包絡、relay_response_to_submit、SSH各往復、全枠busy、dependency-waitを揃え、独立hostへの配送待ちが実際に探索を止めたときに限り、中継host並列化・batchingの費用対効果を判断する。今回高速化改修はしていない。

着手時の公開3cd8 HANDOFFは実H4 Q3完了0件。既知の同server processを観測した範囲では分散実H4 controller/relayは見つからず、旧R08は非分散runで今回の対象区間へ流用しなかった。未公開の別担当runが存在しないとの断定はしない。実H4ログの到着は待たずに公開する。

既存run・他担当worktreeはread-only。本番queue投入、controller/relay/worker再起動、新H4信号・compile・fit・実Q3はすべて0件。性能倍率、科学的な最安更新や価格資格の追加はない。ロードマップには「分散運用の計時経路が人工検証済み、実H4の内訳は次run待ち」という差分だけを渡す。本体は編集していない。

初期切り分けlogは `checks/INITIAL_312_AST_REJECTION.txt` と `INITIAL_INCOMPLETE_FIXTURE.txt` に保全した。最終PASSの正本は `REGRESSION_Q3.txt`、`REGRESSION_DISTRIBUTED.txt`、`REGRESSION_PREFIX_NATIVE.txt`、`TIMING_FINAL.txt`。
