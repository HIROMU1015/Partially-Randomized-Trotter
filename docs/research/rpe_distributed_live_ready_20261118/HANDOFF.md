# PR探索の後着READYと分散割当

RomuTailの通常入口に `e85dbe29263514aaa562970f9aad2cb6dc9c82b3` の `live_ready_refill` を統合した。固定controller SOURCEは `1b45204e0f32d2043a01984e08073e93537eb37d`、検証済みreleaseは `romutail-live-ready-1b45204e0`。RomuTail、server2、local0-wsl、PBS gatewayの配置照合はPASS。server1は停止扱い。

新releaseを明示したPR-Pauli／PR-DFの幅指定で機能が有効になる。Previewの `applied.live_ready_refill` で確認できる。共有currentは `romutail-auto-preparation-5b48aabbc`。固定された既存runのStatus・Resumeは各runの設定を読む。R08の固定入力と切替スクリプトは参照として使用し、R08や他の実runに切替を適用していない。

## 通常入口

```powershell
.\tools\local0-search\Invoke-RpeSearch.ps1 -Action Preview -Release romutail-live-ready-1b45204e0 -RequestFile .\SCOPE.json
```

```bash
python3 -B scripts/run_rpe_search.py --config "$client" --action preview \
  --release romutail-live-ready-1b45204e0 --request-file SCOPE.json
```

範囲、shot設定と工程別予算は通常の `SCOPE.json` で渡す。利用者が次の実探索範囲を承認し、Codexが§10.5の重複照合を終えた後は、同じreleaseと要求ファイルで通常Startを使用する。起動後は実際のrun IDでStatusとResumeを行う。

## 動作

既存の `METHOD_MINIMUM_PRIORITY_V2` が候補選択、shot提案、判定を担当する。入力・信号・価格を既存の固定条件で検査し、登録済み候補IDとrouteを探索器へ渡す。適格な空き枠が報告されれば、別のQ3の終了を待っている選択器も再選択できる。探索の判断に必要なQ3、必要な準備、先読み、任意の価格検証の順でキューを扱う。候補生成と資源・予算は各runの固定条件に従う。

既存の同一キュー、JobStore、worker、SSH mailbox、PBSのwalltimeとdrain、prefix cacheを使う。適格性と残りwalltimeの照合はauto枠の候補比較やcache affinityにも適用する。割当・成果受領・採用は既存のjob/attempt単位で扱い、再接続では元attemptを回収する。確認済み最安には既存の価格・identity・数値判定を通過した先読み結果も含まれる。

```bash
python3 -B scripts/summarize_rpe_q3_timing.py "$run_dir" --output "$my_output/TIMING.json"
```

`CANDIDATE_READY` と同じ候補への最初の `LAUNCH` を中央の時計で結ぶ。従来のENQUEUE→割当、RESULT_RECEIVED→探索採用と併記する。 `QUEUE_WAIT_STATE` はホストごとの空き枠、所有件数、仕事なし・適格な仕事なし・配送待ちなどを記録する。prefix HITは同じattemptの `DP_PREFIX_EVENTS.jsonl` から数え、ログのないattemptは計測対象外として明示する。

## 人工検収

WindowsとRomuTailの通常Previewは同じ `search-1e92e00ef38ac561b011` に解決し、両方式で `live_ready_refill=true` と検収用のQ3予算0を返した。回帰は123件PASS、7件SKIP、下記の旧AST比較1件。

- RomuTailとserver2の実機2台で通常schedule入口と既存SSH mailboxを通した。RomuTailのPR-Pauliが実行中に、server2でPR-DFの人工信号・価格準備が完了し、同じ候補IDの自動shot評価がserver2で開始・採用された。Q3はRomuTail 1件、server2 2件。READY→割当は0.569秒、結果受領→探索採用は0.033秒。このrunの完了後再開は追加起動0件。
- 独立したworker spool 2つのループバック検査で、開始後に接続を切り、元workerに保存された結果を同じattemptで回収した。割当・受領・採用は各1回。controller停止中にworkerが完了する別試験でも元attemptを1回回収し、再開時の二重起動は0件。
- ループバックと実機の配送fixtureは人工評価のためDPを呼ばず、prefix観測は欠測。別の固定native viewの人工DPでcache HITと逐次計算とのexact一致を確認した。新規実H4信号・実compile・実Q3は各0件。
- 旧AST比較テスト1件は、統合前からのQ3上限撤廃で `run_method` のループが変わったため旧基点との差を検出した。現current基点から `proposals`、`method_state`、`run_method`、`run_by_configuration` のASTが同一であることを別に確認した。

実機の配送と待ち時間の記録は [ACCEPTANCE.json](ACCEPTANCE.json)。原ログはRomuTailの `.rpe-central/acceptance/romutail-live-ready-1b45204e0/`。これらの待ち時間は人工試験の観測値。実H4の速度は未測定。
