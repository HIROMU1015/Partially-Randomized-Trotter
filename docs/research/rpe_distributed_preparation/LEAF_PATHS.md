# 通常入口からの準備工程と既存provider

この一覧は実装経路の確認であり、新しい実H4計算の実行記録ではない。
科学的な方式・候補・価格適用範囲・MSE判定は、従来の最適化器/providerの規則を保持する。

| 方式 | 通常探索の入口 | 信号 | 回路生成・compile | 監査・集計 |
|---|---|---|---|---|
| Full PF / B0I | `pf_candidate_selection.prepare` → `pf_optimizer_connection.complete_selected` → D `prepare_pf_candidate.py` | D `signals()` と既存signal provider | D `acquire()` の既存native child | D `acquire()`内の既存監査、`freeze()`、`finish()`。今回のstage wrapperがcompile後の監査を分ける |
| PR-Pauli | `WidthDriver.complete_points` / `prepare_for_search` → `Completion.complete` | `pauli_configuration`の必要P0と、`run_pr_pauli_candidate_signals.py` → A固定descriptor入口6e4 | `run_pr_pauli_candidate_prices.py --phase compile` → A固定02eの`run_pauli_split_cal.py` | `--phase audit` → 固定54bのA監査関数と、新SOURCEの`pauli_split_cal_audit_request.py` metadata adapter |
| PR-DF | 同じ`Completion.complete` | `run_pr_df_candidate_signal_jobs.py` → fefd固定signal provider。必要tail descriptorも元の既存導出入口 | `run_pr_df_candidate_prices.py acquire --stage calibration/holdout` | 同じrunnerの`freeze`と`compare-holdout`。既存wrapper/axis receipt照合を維持 |

Full PF/B0Iは選ばれた候補が必要とする既存D工程のみを実行する。保存証拠で足りる候補に準備計算を追加しない。
PRのsignalと価格枝は固定configurationを共有するが、互いの結果には依存しないので同時に共通queueへ進める。
同一physical scheduleの宣言違いは、SOURCE・辞書・law・評価契約を含む既存`physical_evaluation_contract`でsingle-flightにし、先行候補のREADY証拠を再利用する。
DF較正groupは既存`group_id`で直列照合し、先行候補の登録後に同じStepPriceRegistryへ再照会してから不足cellを取得する。

## portableな工程境界

`Completion.bind_stage_executor(DistributedPreparationBridge(...))`で既存`call()`の重い葉だけを共有`DistributedExecutor`へ送る。
metadata計画・configuration・候補選択・registry採用・Git publicationは中央に残る。
`prepare_for_search(..., stage_executor=...)`でも同じbindingを使い、再開中のCompletionも維持する。
各stageのqueue上限は`distributed_preparation.budgets`の明示値を使い、省略時0（HOLD）。Q3承認だけで準備の上限をUNLIMITEDへ広げない。
既存provider入力のpath数・compile数・SOURCE・one-shot/STOP条件は、これとは別に保持する。

入力は変更せずCASへ保存し、worker所有checkoutへ置く。動的なGit locatorsはbundleで元のcommitを保ったまま移送する。
compile成果のprivate回路・元receipt・元hashは監査側へ必要なものを移す。同じhashの既存CASを使い再転送を避ける。
`rpe_preparation_paths`はPython/workspace/private artifactの**ファイルアクセス先**だけを解決する。
SOURCE・Git locator・科学入力・原receipt bytesを新しいpathへ書き換える仕組みではない。
Pauli監査の派生metadataは`path`（現在のアクセス先）と`original_path`を残し、元のreceipt hashと回路hashを照合する。

Pauli価格runnerの従来の`phase=all`は互換のまま。`compile`はraw gateまでで戻り、`audit`は完了compile依存がなければ科学処理を始めずHOLDする。
完了済み工程の回収・二重採用防止は同じJobStore/attemptを使い、通信切断で別ホストへ再実行しない。

新しいJobStoreへ初めて入れる前には、元requestのworkspace/directoryと元のstage STARTED/STOPを直接読む。
cloneだけを見て元の未追跡markerを見失わないためで、元ownerが実行中・不明なら投入前の運用HOLDにする。
validな完了terminalは回路・receiptと一緒に既存providerの照合へ渡し、STOP/失敗は再実行しない。
既に保存した`.DISTRIBUTED.json`がある工程は新規投入せず、同じJobStore job/attemptを回収する。
この照合は既知の選択済みproducer出力と、既存registry/provider preflightの範囲である。
未知の別run・未接続host・未登録archiveを網羅するglobal重複排除を実証したものではない。

## 保存実入力の検査用locator

以下は保存入力を読むための例であり、コマンドを実行して信号・compileを始める指示ではない。
固定locatorのJSONをdry-checkへ渡す際も、その時点の工程SOURCE/runtime/profileと別途結び付ける。旧Q3の16 PASSを転用しない。

- PR-Pauli signal: `981c5efd3154014ef623eaaa5e6376c3b7d75321:docs/research/h4_pr_multiwidth_resume_20261117/run01/completion/candidates/15e782d4b0e09c9be929/PAULI_SIGNAL_REQUEST.json`
- PR-Pauli compile/audit: 同じprefixの`PAULI_PRICE_REQUEST.json`。
- PR-DF signal: `981c5efd3154014ef623eaaa5e6376c3b7d75321:docs/research/h4_pr_candidate_continuity_20261117/run01/completion/candidates/6fd6cc805556fafcf685/DF_SIGNAL_REQUEST.json`
- PR-DF compile/audit: 同じprefixの`df_price_groups/98621104fb2fde4e40d8/df_prices/INPUTS.json`。

着手時には上記candidate-isolation worktreeに進行中の既存科学runがあり、STATE・較正STARTED・監査work等を読取確認した。
そのworktreeのSOURCE・run・原結果・STOP・未回収/UNKNOWNを変更していない。
新しい分散準備機能と同等の後続実装は見つからず、981c5efd3の保存route照合・READY snapshot・descriptor修正だけを再利用した。
