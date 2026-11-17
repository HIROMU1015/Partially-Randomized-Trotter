# H4通常探索への価格モデル簡易検証

通常入口にモデル単位の検証記録を接続した。保存PR-Pauli LD2の独立実測を追加compile 0件で再利用し、未検証モデルの代表1点要求、共有・再開、待機中の探索継続、C通常同期から元実測へ戻る表示を人工ケースで検収した。新しい実信号・実Q3・sampling・実compile・fitは各0件。

## 起点と再利用

- 最新B2探索修正：`314524a913ba6e66b9069ab1514b1df765f9a5e2`。通常本体は `scripts/run_rpe_schedule_search.py`。独立・所有worktreeへ取り込み、旧runを変更しない。
- local0/分散割当：`9ad0260777a442a4ba2d1e983079941668e6cba6`。`run_rpe_local0_search.py`から同じmethod/schedule入口へ渡る。
- 分散準備：調査開始時の`91fe799dd`から、途中で公開された`7e0e1e666b7621fab717ce902f9a58a92e0025fb`へ進んだことを確認し、既存汎用準備workerを統合した。
- Cマップ・公開consumer：`3a9476130ec99566d54c5e38578b6444bb2f94af`。既存controller公開索引・通常syncを拡張した。
- PriceRules本体・凍結係数・既存科学kernelは再実装しない。旧DP結果・科学run・STOP・他担当worktreeには書き込んでいない。

## 動作

`make_executor`で通常探索に薄い診断serviceを接続する。PriceRulesとactual-step registryのSOURCE、係数、方式、分割・辞書、積公式、実step、piece/event recipe、compiler/lowering/profileをモデルIDへ結ぶ。shot配分・κのラベル・run・worker hostはIDへ入れない。適用先のq/Rは別に保存する。PFでは保存価格のSOURCEをたどり、canonicalのexact stepと既存H46 compilerのbinary64 T/q規則を区別する。

モデル初回使用時だけ、登録済み独立比較・較正・共有queue・既存公開索引を照合する。controller repositoryの共通Git領域に置く小さいSQLite記録を全runが共有する。これは二重科学投入を防ぐ実行状態であり、承認証明ではない。通常のmulti-host経路は同じcontrollerへ要求を集約する。別cloneで別controllerを起動する場合も、同じcontroller-owned store/共通queueを使用すること。分離されたstoreを同じ共有namespaceとみなさない。

独立実測がなければ、実schedule中の較正以外の点を選ぶ。見積wrapper規模が最小点の2倍以内の点を同程度の負荷と扱い、その中で現在提示された配分の加重寄与を優先する。配分が未提示なら既存capsを暫定選定重みとし、その事実を保存する。実測前に選定理由・回路・係数・予測を固定する。PRは独立2経路、各X/Yの計4wrapperが標準。係数fit、残差の全round適用、順位補正はしない。

投入は既存 `DistributedExecutor.submit_preparation` / `rpe_preparation_worker` のcompile stageを使う。通常Q3・通常準備・既存先読みを先にし、簡易検証は空き枠に置く。要求とjob IDをqueue投入前に永続化し、再開ではその同じ要求だけを照合する。待機・取得失敗は価格適用条件・Q3判定を変更しない。探索終了時にoptional検証を待たず、queueと開始済みattemptの所有者・部分成果を保全する。

## 保存実測の検収

[LD2保存データの表示例](index.html)／[機械可読VIEW_INDEX](VIEW_INDEX.json)。通常publication parserから読んだ2つの保存配分点と1モデルで、issues=0。新規科学callは0件。

PR-Pauli LD2のモデル：`b74e4e536392c2bbe94cd371436615fc56c0564f:docs/research/d_pauli_ld2_b1_cal_results_20261115/c2_freeze/run01/FROZEN_MODELS.json`。

独立比較：`705f59ea863e1b9fe90e023f1cb1057d00366b16:docs/research/d_pauli_ld2_b1_holdout_results_20261115/COMPARISON.json`。

m5/q32/R33は、各軸の独立2経路が68,228・68,300 RZ、保存平均68,264 RZ。凍結予測は6169957875465/90343469、符号付き相対差は2751307649/6167206567816（約+0.044612%、正が過大予測）。m6は1経路であり2経路とは数えない。経路・軸・元receiptは保存比較へのselector付きリンクを保持する。

Full PF/B0I p7の6基準点と粗幅Yoshida4についても、保存モデル・独立m6/m9比較のSOURCE接続をmetadataで確認した。新しい価格予測や実測は行わない。

## 検収範囲

| 項目 | 根拠 |
|---|---|
| 保存独立検証をcompile 0で再利用 | 通常`run_rpe_method_optimizer.py --round --run-dir`のQ3予算0実行、および通常executor入口のLD2保存資料テスト |
| 較正のみのモデルに代表1点 | 人工13round、複数配分・4同時要求で同じ選定点を共有 |
| 同時host・checkpoint・再同期で重複しない | 既存durable queueの排他claim、投入直後の通信不明を模擬した固定request再利用、Cの同一索引再同期 |
| 検証待ちでも探索する | 通常Q3を先にclaimし、optional失敗でも探索継続・終了。科学kernelは人工代替 |
| マップから予測と実測へ戻れる | 通常publication→C同期→INDEX/HTML。配分固有の検証round寄与率、適用q/Rと実測q/Rを別表示 |

最終固定SOURCEで人工・回帰96件PASS。旧executor総合scriptも実行したが、現行候補分離より前の全run停止ラベルを要求する2項は不一致（19 PASS/2旧assertion FAIL）。現行の候補分離・継続10項はPASSであり、ラベルを旧規則へ戻す修正は行わない。詳細はTESTSに保全した。

人工・回帰の具体的なcommandは[TESTS.txt](TESTS.txt)、CLIの固定記録は[ACCEPTANCE.json](ACCEPTANCE.json)。source変更前の途中技術失敗はlocal artifactへ保全し、最終検収と区別した。著者自身の検査とsub-agentの実装検査であり、独立した科学レビューではない。

## 限界・STOP

新規compile経路は人工ケースまで。Pauli/DFの自動入力組立ては既存normal-path取得関数を再利用するが、新しい対象での実compileは未検証。provider・seed出所・入力Git object・runtime/profileが不足すれば検証待ち/未完了とし、通常探索は続く。PFの保存holdoutは利用できる。fresh PFは既存providerのprepared PLANと固定実行profileが必要で、generic取得routeへ接続する。未設定のまま勝手に既存較正親runnerを起動しない。

Cの稼働中checkout・cron・他担当の実行SOURCEは切り替えていない。本branchの公開consumerを採用すれば、既存通常同期が共通controller索引から検証metadataを取り込む。hostごとの測定結果の手動転記は不要。

1点の独立実測の存在はall-law期待価格・長q外挿・CI・数値certificate・正式G/winnerの資格ではない。詳細なq/R検証は別指示。技術検収と公開後にmandatory STOPし、新しい科学実行へ進まない。
