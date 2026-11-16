# B2向け：候補評価順序の比較を再現する

2026-11-17 JST。branch `codex/h4-rpe-candidate-ordering-20261117`。実装SHAは [SOURCE.json](SOURCE.json)、結果・限界は [REPORT](REPORT.md)。実H4の新規信号・較正・compile・Q3、round05本番は未実行。

## 使えるもの

| 入口 | 用途 |
|---|---|
| `scripts/replay_rpe_candidate_ordering.py` | 同じW/V2候補menuからA/Bの順序だけを変え、選択・完了後に保存答えを開示する。未知要求でbranch打切り。 |
| `src/trottertracks/rpe_candidate_ordering.py` | 純粋な選択器。`Selector('ROUND05')` または `Selector('IMPROVEMENT_PER_TIME_V1')` を比較callerで選ぶ。archiveや実行権限へアクセスしない。 |
| `rpe_priority_search.priority_session()` | 内側V2のcounterを保持した候補frame。既定wrapper `run_priority()` は従来どおりAの候補を選ぶ。 |
| `scripts/summarize_rpe_candidate_ordering.py` | 完了した技術再生JSONから比較表・圧縮trace・標準matplotlib図を出す。科学archiveを読まない。 |
| `scripts/check_rpe_ordering_scenarios.py` | 人工時間・人工MSEの3case。準備再利用/較正待ちと未知scheduleが有利な反例。 |

通常roundへBを有効にするJSON設定は追加していない。既存roundの既定動作は変わらない。Bを実探索へ接続する際は下記APIを既存driverの候補選択部分に適用し、既存queue・mapper・価格・精度契約を使う。このhandoffはその科学実行許可ではない。

## 保存ログ再生

このサーバーでは、固定B2 worktreeにledgerと保存runのlocatorがある。原本を変更せず、出力は毎回新しいdirectoryにする。

```bash
cd /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h4-rpe-candidate-ordering-20261117
PYTHONDONTWRITEBYTECODE=1 python3 -B scripts/replay_rpe_candidate_ordering.py \
  --baseline-worktree /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/b2-rpe-split-preparation-20261116 \
  --context-cache docs/research/h4_rpe_candidate_ordering_20261117/cache/CONTEXT_ARCHIVE_V1.json \
  --output /tmp/h4_ordering_reproduction_new
```

主32枠/HYBRID16、1枠/1、LD5 κ≤64の3pairと、LD5のQ3予算4/8/16/32/64を再生する。`--budget` は長軌跡の模擬起動上限（既定400）。`--scan-budgets` を引数なしで末尾に付ければ追加走査を省く。400件完走を保証するものではなく、最初の未知候補で止まる。

公開cacheは初期C数値9件と26構成の準備metadataを持つ。4,783件の全答えを保存したcacheではない。実行用round/route登録として流用しない。cache経路ではこの入力のSHA256を出力し、headerの出所表示だけを内容検証と呼ばない。

cacheを再構築する場合は `--context-cache` に**まだ存在しない自分のpath**を渡す。固定d8のproblem/round/定数、route/価格registryと、固定round04 `KAPPA_POINTS.json` の準備metadataを使う。生のB2 round05/run01ファイルは拾わない。既存のSOURCE/input readerとmetadata生成を使い、新信号・較正・compile・実Q3は呼ばない。初期OUTCOMEは依然Cの9件だけである。

保存runの原本がない環境でも、公開 [saved_replay/COMPARISON.json.gz](saved_replay/COMPARISON.json.gz) と表は閲覧可能。再生そのものには原本のREQUEST/PROCESS_RETURN/OUTCOMEが必要で、原本欠測をPASSや代替平均時間にしない。

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B scripts/summarize_rpe_candidate_ordering.py \
  --comparison /tmp/h4_ordering_reproduction_new/COMPARISON.json \
  --output /tmp/h4_ordering_tables_new
```

`--plot` はmatplotlibがある場合だけ付ける。今回の図は既存 `/home/AbeHiromu/projects/h6-environments/20261011/venv/bin/python` の読取利用で生成した。共有環境の変更なし。

`COMPARISON.json` の `results` / `budget_scan` は [saved_replay/CACHE_EQUIVALENCE.json](saved_replay/CACHE_EQUIVALENCE.json) のとおり、固定metadataからの構築とcache再利用で一致した。実行器自身の処理時間やmetadata_inputの経路表示は一致判定から除く。中断した再生CLIは新しいoutputで初めから決定論的に再生する。これは実科学jobのresume手順ではない。

## 選択器との受け渡し

```python
selector = Selector('IMPROVEMENT_PER_TIME_V1', explore_every=4)
selection = selector.choose(actions, method_best=best_completed_cost,
                            tau=threshold, now=clock_now,
                            observations=completed_observations)
chosen = selection['chosen']
trace = selection['trace']
```

actionの必要項目は `id`、`kind` (`NEW_SCHEDULE` / `SHOT_Q3`)、既定順 `a_rank`。追加特徴は次を渡す。

| 項目 | 意味と未知時の扱い |
|---|---|
| `method`, `L_D`, `configuration_id`, `width` | 方式・構成・要求幅のidentity。widthはopaque、同一物理評価へのalias照合は既存mapper/契約側が担当。 |
| `cost` | 次に評価する**特定配分**の既知価格。不明ならNone。新scheduleの最終最安を渡さない。 |
| `price_floor` | 証明用の下界。scoreの予測適合価格にはしない。 |
| `parent_cost`, `parent_MSE_upper` | 既に完了した親の値。未知の変更後配分のMSEを渡さない。 |
| `preparation` | `signal_reuse`, `price_reuse`, `requires_calibration`, `seconds`, `source`。両再利用trueかつ較正falseのときだけ準備0秒。それ以外は測定/明示仮定、未測定None。 |
| `profile`, `q3_seconds` | 同方式・同profileの完了中央値に使うprofile。q3_secondsを明示すると仮定priorとして記録、実測とはしない。 |
| `untried` | 未知にも機会を残すための周期探索の対象。低scoreでも不適合にしない。 |

observationは `id, kind, method, configuration_id, profile, finished_at, status, MSE_upper, seconds`。`seconds` はSHOT_Q3ならQ3サービス時間、NEW_SCHEDULEなら準備込みの可能性があるため、Q3 legは明示 `q3_seconds` がある場合だけ利用する。`finished_at > now`、候補自身のheld-out情報、全archiveから計算した平均は渡さない。

`snapshot()` / `Selector.restore()` は周期・選択履歴を復元する。既存jobの再開・予算・完了結果のidentity確認は既存Executorの責務であり、この小さいstateで代替しない。

## V2 session

`session=priority_session(..., active_configurations=active_set)`、`frame=next(session)`、`frame=session.send(chosen_proposal)` で進める。frameは `eligible`, `default_choice`, `groups`, `step`, `since_explore`, `new_calls`, `method_min` 等を持つ。候補はeligibleに存在するものだけ送る。`send({'refresh': True})` は新scheduleのactivation後、手数を消費せず再分類する。返り値は `StopIteration.value` の従来result。

候補生成をpolicy別に作り直さず、同じsessionのeligibleを選択器へ渡す。W準備とshotの同時menuでもA rankはW提案順→V2既定順を保つ。既に起動済みのjobは無料採用できるが、予算が尽きた新規要求は外側へ伝播して止める。単に `max_steps_per_method=1` のV2を繰り返す方法は使わない。

## ΔPF担当との境界

着手時に確認した `codex/a-pr-step-width-search-20261117` は未公開の進行中作業だった。draftの `requested_widths`、`width_family_key` と requested/evaluation_configuration_id の区別に合わせた。幅mapper・較正・StepPriceRegistryはその担当の既存実装へ渡す。

別幅が再利用済みなら、既存幅のshot候補と同じaction menuへ `preparation={signal_reuse:True, price_reuse:True, requires_calibration:False}` を渡す。較正待ちなら、その準備状態・時刻根拠を別に渡す。requested IDで優先順位を決めても、evaluation IDの結果再利用は同一信号・価格・評価契約を既存側が確認してから行う。binary64 `.2` と文字列 `"1/5"` を同一化しない。

人工fixtureでは再利用0秒と較正60秒のケースを区別した。実 `"1/5"` のmapper受理、他方式、新幅の適合費用や実時間は未検証。

## 小さい回帰検査

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s tests -p 'test_rpe_candidate_ordering.py'
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s tests -p 'test_rpe_priority_session.py'
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s tests -p 'test_rpe_order_replay*.py'
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s tests -p 'test_rpe_throughput_entry.py'
PYTHONDONTWRITEBYTECODE=1 python3 -B scripts/check_rpe_priority_search.py --output /tmp/h4_ordering_priority_new
PYTHONDONTWRITEBYTECODE=1 python3 -B scripts/check_rpe_schedule_search.py --output /tmp/h4_ordering_wide_new
PYTHONDONTWRITEBYTECODE=1 python3 -B scripts/check_rpe_ordering_scenarios.py --output /tmp/h4_ordering_synthetic_new
```

上4つは11+9+20+13件、既存scriptは6+7 checks、人工比較3case。実H4 OUTCOMEの読取を禁止した小fixture、旧版default軌跡比較、未保存先読み・時刻欠測の打切り、採択外完了の最安包含、有限予算・無料採用、固定metadata/価格cache再利用を含む。

次の実探索で必要な計測はREPORT末尾に記した。今回の保存支持だけを根拠にBをround05既定へ変更せず、実探索の比較・科学実行は別範囲として扱う。
