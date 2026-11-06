# C6b：状態検証の高速化実装と適用境界

2026-11-07 JST。利用者の「高速化を行って」に対する技術修正。**高速化部品を実装し、人工4件と3-qubit/64入力の比較を確認した。現在の固定C6bジョブには未適用で、SOURCE `23725fc6f6d6d0c2bec93e31e32074711804e250`・sampling・回路・結果・既存STOPは変更していない。** 新しい分子sampling/build/compile/postcheckは0。

## 実装した差分

[BatchedStateActions](../../../src/trottertracks/track_d_state_action_acceleration.py)は既存 `StateActions` のopt-in subclassで、同じAer CPU/double statevector、同じ `simulation_circuit`、同じ入力順・全出力振幅・declared phase復元を使う。数値simulator、fermionic gate、compilerは作り直さない。

- `compose(copy=False)` で変換後に変更しないlocal操作を共有し、入力列ごとの不要なdeep copyを除く。元cost回路/P0回路は変更しない。
- 入力列を割当native並列数単位に分割し、同時に保持する回路metadataとnative resultを減らす。入力列や出力振幅を減らさず、全列を元の順序で返す。実分子でのpeak memory減少率は未測定。
- OpenMP予算を、Python呼出し元だけでなく**実際にAer native処理を呼ぶexecutor thread**で設定する。既存BLAS単一thread制御を保ち、worker内の設定をcall終了時に復元する。OS/共有venv/他processは変更しない。
- 固定環境のAer `_run_circuits` は、公開されたexecutor optionを `AerJob` constructorへ渡していなかった。option指定だけの試作では実並列数1が残ったため採用しなかった。最終版は既存 `AerJob` と同じ `_execute_circuits_job` callbackを明示executorで呼ぶ最小接続を使う。backend自体へのmonkey patchはしない。この接続は固定ライブラリ版に依存し、別版へ無条件移送しない。
- `stage_memory_estimates` はraw/Reference/direct各工程の**完了済みprocess peak**を分ける。瞬間RSSやSTARTED記録を根拠にした見積りを拒否する。観測見積りであり上界ではなく、dispatch時の実空きmemory確認は別に必要。実行中schedulerへ自動注入しない。

## 人工確認と限界

[test](../../../tests/check_track_d_state_action_acceleration.py)／[scalar結果](../../../artifacts/track_d_c6b_q8/2026-11-07/acceleration_technical/synthetic_checks.json)。4 testは、位相付き3-qubit circuitの全basis列と重複・重ね合わせ入力、actual inverse、既存state-action/denseとの一致、元circuit非変更、cache、入力schema、工程別memory evidenceを確認した。abs1e−12/rtol0、phase fittingなし。

別の3-qubit/3072 gate/64入力による人工比較では全出力差0を観測した。初期 `OMP_NUM_THREADS=1` と外側 `threadpool_limits(1)` の条件でも、最終版Aer metadataの `parallel_experiments=4` を確認した。最初のoption-only版は1だった。人工試作の失敗・修正は分子runのretryではない。

単発人工ベンチの速度比・wall/RSS/内部資源観測はローカルに保存し、C6b分子速度やmemory保証としない。周囲の同時負荷もあり、人工での速度比を残り時間へ乗算しない。元SOURCE・component abs1e−10/observable2e−10・検証列集合・law・P0・モデル/family/幅/seed/cutを緩和したものではない。

## 現ジョブへ適用する際の未解決な境界

[固定PLAN](PLAN.md)は「実行中taskの再起動・SOURCE hot-editなし」「失敗後の無断retry/resume、追加sampling/compile、seed/cut/model/family/幅変更なし」を定める。現在のrunnerに外部から安全に高速化部品を選択する入口はなく、pending taskを別workerから勝手に処理すると重複実行になる。今回の実装だけを理由にcurrent SOURCEを書き換えたり、稼働processへコードを注入したりしない。

現runに適用するための具体的候補は、**現在実行中のtaskを完了させる調整停止**と新SOURCEへの限定引継ぎである。元event32本・全256 occurrenceをそのまま固定、raw PASS/compiled QPY/postcheckの完了記録はhashで結び再実行しない。未着手raw、未compile wrapper、compile済みかつ未postcheckだけを新SOURCE側の明示残task集合にする。総compile上限128は旧新合算。元runはpartial/STOPとして保全、新run来歴とSOURCE変更を明示する。凍結モデル・cut4・P0・精度・family12・価格幅を保持し、全32が揃わなければ統計を作らない。

この引継ぎ制御・新実行は**未実装・未実施**であり、固定PLANの変更に対する明示許可が必要。許可前に正常jobへsignalを送らず、pendingを奪わず、再sampling/再compileしない。単に「高速化済み」として現runに適用したことにしない。

次の担当：ユーザーが、現runを維持するか、調整停止・固定SOURCE切替・残task限定引継ぎを許可するかを判断する。Codexは許可された場合だけ具体的な移管制御と新SOURCEを固定して適用する。どちらの場合もC6b結果整理・公開照合・mandatory STOPを維持し、追加q/追加calibration/N/全RPE/詳細科学レビューには進まない。
