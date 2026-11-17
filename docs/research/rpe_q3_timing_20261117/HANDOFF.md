# 元チャット・分散統合担当への引渡し

取り込むコード範囲は `3cd8b1769ffc4df71d85bdaefce0e337becf028e..1026c09b3645c1ef210def1f4678d79db2fda019`。
branchは `codex/rpe-q3-timing-20261117`。別担当のworktree、既存controller/relay/worker、本番queueは変更していない。
新規H4 Q3・信号・compile・fitは0件。保存実3ホストの18人工jobと、自分の独立人工spoolでの検証だけ。

## 取り込むcommit・通常入口の変更・集計コマンド

コードは次の3commit（後半は集計とoptional host観測の拡張）。`SOURCE.json`にfull SHAを固定。

```bash
git cherry-pick 19c43576c83ec9de24f6059f4cb39762458c6519 5897d26ab 1026c09b3645c1ef210def1f4678d79db2fda019
```

後続公開 `e64e377a8a27e6693231bb4cdf72f605a20965b5`（optional WSL）を最終照合した。
同版との統合では `rpe_optimizer_concurrent.py` のhost dict/required/return付近に同時編集があるため、単純なcherry-pickはconflictし得る。
元の `new_slots`、`reported_slots`、`host_admission`、optional除外、既存attempt回収を保持し、観測stamp/capacity/QUEUE_WAIT_STATE呼出しだけを重ねる。
`_timing_queue`は既存 `_host_admission_policy/_host_admission` があれば再利用し、任意hostの不参加をrequired待ちやbusyへ誤分類しない。
53 tests中の専用fixtureでこの参照を確認したが、e64への統合・WSL実機計時は未実施。
relayのWSL transportやparticipation_file処理も残し、計時を理由に上書きしない。

本HANDOFFと公開測定例は、このbranchの後続の成果物commitから参照する。
他担当branchへこちらから適用していない。統合担当が自分の採用SOURCEへ取り込む。

通常科学入口は従来の `scripts/run_rpe_schedule_search.py --problem ... --round ... --run-dir ...`。
新しい科学queue、候補選択器、承認・grantの仕組みはない。計時専用CLI flagの追加もない。
**稼働runへの途中適用や再起動は不要。次の採択済みSOURCE・新runで使う。**

1. controller/transport checkoutへ上のcodeを取り込み、次runのSOURCEを固定する。
2. 各hostの新transportと、実際に使用するadapterに `src/trottertracks/rpe_q3_timing.py` を配置する。
   新adapter scriptはhelperを自己checkoutから読み込む。manifestへhelperも含める。
   既存profileの `adapter_root` / `adapter_manifest` が旧固定adapterを指したままなら、そのhostのQ3 body計時は増えない。
3. **元の科学SOURCE・route_viewsと、新transport/adapter SOURCEを区別**する。旧科学rootを偽って書き換えず、既存host profileのadapter overrideを使う。
   別途固定された旧computation rootにhelperがなければ、その元manifestは元8filesを維持する。
   helperをimportする新rootでは9filesに含まれる。SOURCE/adapter/runtime guardは緩めない。
4. 新profileの科学入力/route/AST/runtimeについて、既存通常readiness/dry-checkを通す。
   配置は元の分散統合担当の範囲。今回こちらからremote配置や本番dry-check/Q3は行っていない。
5. 新しいrunでは結果artifactに時計domainとbody計時が同梱され、中央PROGRESSからattempt/host付きの採用時刻を読める。

通常run directoryを指定して1コマンドで集計する（元runは読取専用、出力は自分の場所へ）：

```bash
python3 -B scripts/summarize_rpe_q3_timing.py /ABS/PATH/TO/RUN \
  --output /ABS/PATH/TO/MY_OUTPUT/TIMING.json
```

稼働中にも読める。完了前/途中append、未受領、異なるattempt、欠けた時計・job対応は欠測として扱う。
SQL queueを開いたり科学OUTCOME/MSEをparseしたりせず、PROGRESSとLAUNCH、JOB_STAGES、TRANSPORT_TIMING、Q3_BODY_TIMINGだけを読む。

公開した同じ集計の再現例：

```bash
python3 -B scripts/summarize_rpe_q3_timing.py \
  docs/research/rpe_q3_timing_20261117/checks/artificial_run \
  --output /tmp/rpe-q3-timing-artificial.json

python3 -B scripts/summarize_rpe_q3_timing.py . \
  --timeline docs/research/rpe_distributed_q3/checks/REAL_THREE_HOST_TIMELINE.jsonl \
  --output /tmp/rpe-q3-timing-legacy.json
```

## JSONの読み方

- `rows[]`：job_id / attempt_id / hostを結ぶ全件。`by_host`は件数・初回群・後続群・統計と各件。
- A `A_queue_to_assignment_seconds`：allocation ENQUEUE→中央割当。物理candidate入力READYからの時間ではない。
  `queue_phase`は最初の割当前enqueue群/以後を分離。`host_phase`はhostへの初回割当かどうか。
  `A_observed_wait_seconds_by_reason`は予約slot使用中・required host参加待ちなどの観測済み区間。未観測は`A_unattributed_seconds`へ残る。
- B/C：同じboot clockなら`EXACT_SAME_CLOCK`、異なる時計ならworker内のbody spanと中央のturnaroundから作る`BOUNDED`、不足なら`MISSING`。
  B/Cの上限は**同じ非本体包絡を共有**し、加算不可。ホスト間の絶対時刻は引かない。これは時計rate校正付きのネットワーク保証ではない。
- 本体：`Q3_body_seconds`は既存Accelerator API（または明示人工body）のspan。process起動/dry-check/route envelopeを本体時間へ昇格させない。
  複数API呼出しは`Q3_body_calls_measured`と`Q3_API_call_seconds`に分ける。
- C補助：worker本体終了→bundle準備、SSH status/submit/ACK往復、relay poll応答後→各submit待ちを別々に保持。
- D：`D_receive_to_executor_adopt_seconds`と`D_receive_to_search_adopt_seconds`を分離。
  新`RESULT_RECEIVED`はcomplete RPCのJSON body parse後、検証・永続化前。
  旧`TRANSFER_COMPLETE`しかない場合の境界は`LEGACY_VALIDATED_PERSISTED_RESULT`と明示する。
- 未採用の先読みjobではDがnullのままになり得る。回収失敗やlogging failureとは区別して読む。
- `search_join`：新runは明示job/attempt。旧記録は同method/configurationの唯一のpending ADOPTだけを推定対応し、そのラベルを付ける。曖昧なら採用時刻はnull。
- job秒の合計と、同時計domain内の区間unionを別々に出す。
  `D_while_optimizer_waiting_union_seconds`は要求jobのdependency-waitとの交差。
  旧ログのようにwaitがなければnull。0秒との混同を避ける。
  これは観測した実待機の一部であり、relay改修で短縮できる量の因果推定ではない。

## 結論と改修の優先順位

詳細は [REPORT](REPORT.md)、全18件の値は [LEGACY_PER_JOB.csv](checks/LEGACY_PER_JOB.csv)、
機械出力は [legacy](checks/LEGACY_THREE_HOST_TIMING.json) / [新人工通常入口](checks/ARTIFICIAL_THREE_HOST_TIMING.json)。

保存実3hostの初回約235秒と、後続jobのqueue/配送を分離した。中央検証保存は平均約17〜24 ms、
遠隔の保存後→探索採用は平均5〜6 msと小さい。server2には採用順序待ちを含み得る最大5.98秒があり、転送と混同しない。
旧記録ではbodyと片道B/C、全適合枠busy、実際のdecision critical waitが欠測で、relay並列化の速度効果はUNKNOWN。
次の通常runで新計時を取り、独立hostへのsubmit待ちやpoll回収が要求中jobの進行を止めている場合に、host並列中継/batchingを検討する。
初回配置を済ませてから後続jobを比べ、H4の大きな結果bundleでvalidation/transfer時間が変わる可能性も残す。

## 検証と再現

53 Q3関連tests、7 distributed tests、元native環境の人工DP10 testsがPASS。
最終計時9 testsもPASS。新API計時を載せた人工DP2callは元結果exact一致、cache HIT、AST guard不変。
通常入口の1/2/3模擬hostは各18結果が保存参照とexact一致し、再開追加0件。切断回収も既存人工受入PASS。
実3台の新計時実測や実H4の性能倍率ではない。[人工受入](checks/ARTIFICIAL_ACCEPTANCE.json) / [native](checks/NATIVE_TIMED_DP.json)。

```bash
PYTHON=/home/AbeHiromu/projects/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/venv/bin/python
"$PYTHON" -B -m unittest discover -s tests -p 'test_rpe_q3_*.py'
"$PYTHON" -B -m unittest discover -s tests -p 'test_rpe_distributed*.py'
/home/AbeHiromu/projects/h6-environments/20261012/native-dependency-v2/venv/bin/python \
 -B tests/check_rpe_q3_timing_native.py
```

ロードマップ本体は変更していない。統合担当には「計時は人工検証済み、実H4内訳は次run待ち」の運用差分を渡す。
科学的な価格・精度資格や最安は変更せず、今回の計測作業は公開後に終了する。本番探索は止めない。
