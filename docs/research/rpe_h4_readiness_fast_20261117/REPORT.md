# H4次run準備：通常入口の待ち時間短縮

保存結果の読込み、次候補の選択、サーバー2の実入力dry-check、遠隔配布物の用意を、
通常のschedule入口から1コマンドで行えるようにした。
計画用contextと保存評価を再利用し、巨大なexact MSE配列の反復縮約・JSON二重出力を省いた。
候補とdry-check結果が揃った時点で `READY.json` を出し、CLIの報告保存完了とGit公開完了は別に記録する。
同じ固定codeの最終計測では、空cacheからREADYまで106.53秒、入力不変の再準備は6.47秒だった。
REPORT保存完了まではそれぞれ107.30秒・6.97秒。いずれも親process観測値で、科学計算は行っていない。

準備コードの固定SOURCEは `b5a8a60c57457e6888159031cfba719558aaf169`。
Q3用REQUEST・adapter・計算checkoutのSOURCEは従来の
`28d976b99cdf8329adcc1b79327e329e947ef774` を維持する。
旧公開版 `33928bf80cca18ce3f1fa3ddde88d27d7e2b7e5b` の遠隔入力、手順、出力path、
元科学run・STOP・UNKNOWN状態は変更していない。
新Q3、signal、sampling、価格fit・較正、compile、collect、本番queue投入は0件。
以前の実2台・3台と切断復帰の人工検収、42件の既存テストも再実行していない。

## 入口と処理の変更

実行例と再利用条件は[README](README.md)、固定codeは[CODE_SOURCE](CODE_SOURCE.json)を参照。
新しい出力directoryと所有するlocal cacheを指定する。

```bash
python -B scripts/run_rpe_schedule_search.py \
  --problem docs/research/rpe_distributed_h4_plan_20261117/inputs/PROBLEM.json \
  --round docs/research/rpe_distributed_h4_plan_20261117/inputs/FULL_HISTORY_PLAN_ROUND.json \
  --run-dir /absolute/path/to/new-readiness-output \
  --plan-only --prepare-readiness \
  --preparation-cache /absolute/path/to/local-preparation-cache
```

この追加経路は `--plan-only` を必須とし、`science_approved=false`、全方式の新Q3予算0、
`launch_disabled=true` で既存の `round_mode` とoptimizerを呼ぶ。
通常の科学実行・既定出力の経路は変更しない。

| 処理 | 再利用するもの | 毎回確認するもの |
|---|---|---|
| context準備 | 同じSOURCE・入力・registry・metadata providerで生成した候補と価格quote | 関係する入力・実装・view・runtimeの変更 |
| 保存評価の取込み | 既存ExecutorがID・route・配分・評価契約を検証したexact適合／不適合記録 | 元jobのmarker、追加・変更OUTCOME、未確定jobの現在markerと対応process、既存queueの読取状態 |
| dry-check | 同じSOURCE・route・入力・runtimeと一致する以前の結果 | fingerprintと元結果の整合性。以前のFAILもFAILとして再利用 |
| 報告保存 | best、pending、配分、費用、prior件数、needs、選択設定 | 保存直前の選択結果をcompactな投影へ変換 |

科学的条件が変わった入力を古いcacheで代用しない。
追加・変更のあった保存jobだけを再parse・再検証し、費用は現在候補の既存単価で再積算する。
UNKNOWNを未起動扱いにせず、応答がなくなっただけで再投入しない。

`round.plan_output="compact"` は明示的なscience=false・全予算0の計画だけで有効。
bestと次pendingの選択・配分・費用を保ち、数千件の `evaluated`・`resume_records`・traceに重複した巨大MSEを
報告のたびにdecimal化しない。`NEEDS.json` は `ROUND_RESULT.json` の該当箇所への小さな索引になる。
原exact値は元OUTCOME/catalogとlocal exact cacheを参照する。
compact出力をexact resume exportとして読み込むことは拒否し、元証拠から再読する。
cacheは自分がこの準備で生成したlocalファイルだけを使用し、private pickleを遠隔配布物や公開証拠へ含めない。

## 計測区間と最終cold/warm計測

旧計画はCLI内部254.179743秒、そのうち `SAVE_RESULTS` が163.129540秒であった。
4件のdry-check所要時間の合計は約5.2秒で、主な待ち時間はdry-checkそのものより保存結果処理にあった。
根拠は[TASK_TIMING](TASK_TIMING.json)。利用者が観測した「依頼から最終返答まで約50分」は
CLIの実行区間とは異なるため、この値からCLIの速度倍率を算出しない。

最終計測は同じ固定準備codeで、空cacheのfirst、続いて入力不変・同cacheのsecondを実行する。
親processのmonotonic時計を `Popen` 前から測り、Python起動時間を含めてREADY・REPORT・process終了を観測する。
計測runnerは[RUN_BENCHMARK.py](RUN_BENCHMARK.py)。両processはreturn code 0で完了した。

| 同一codeの実行 | command→READY | command→REPORT完了 | process終了 | cacheの条件 |
|---|---:|---:|---:|---|
| first | 106.529767秒 | 107.301847秒 | 107.956741秒 | context・保存評価・dry-check cacheなし |
| second | 6.473348秒 | 6.970234秒 | 7.035015秒 | 入力不変、同じlocal cache |

根拠：[first TIMING](benchmarks/first/TIMING.json)、[first親process計測](benchmarks/first/PROCESS_TIMING.json)、
[second TIMING](benchmarks/second/TIMING.json)、[second親process計測](benchmarks/second/PROCESS_TIMING.json)、
[METRICS](METRICS.json)、[CHECKS](CHECKS.json)。
内部stage時間と親process観測時間の起点は異なるため、混在させて差を求めない。
REPORTの保存完了後に行うGit commit/pushの時刻は[PUBLICATION_TIMING](PUBLICATION_TIMING.json)で別に記録する。
旧339版の遠隔bundleは既に公開済みであり、今回のREPORT公開を待って再配布する必要はない。

同じ最終runの内部指標は次の通り。firstでは旧339報告にexact配列を再出力していなかった
元OUTCOMEを一度読んで検証・cache化する。secondでも元jobのmarker走査と未確定状態の確認は省略しない。

| 内部処理・件数 | first | second |
|---|---:|---:|
| context準備 | 68.740515秒、MISS | 1.836072秒、HIT |
| 保存job走査 | 5372件 | 5372件 |
| REQUEST再parse | 5358件 | 0件 |
| 受入済みexact評価のcache HIT | 0件 | 4358件 |
| サーバー2 dry-check | 4件を並列実行、2.682638秒 | 4件を結果再利用、0.203129秒 |
| `SAVE_RESULTS` | 0.759262秒 | 0.493947秒 |

compactなROUND_RESULTは各1,970,853 bytes、NEEDS索引は各1,392 bytes。
巨大exact評価配列を新しいJSONへ再出力していない。
不適合結果も含む4358件の受入済み保存評価を再利用し、現在候補の範囲外等で受入れなかった記録を適合へ昇格させていない。

開発中の試行は最終計測から分ける。初回試行のREADY/REPORTは106.19/107.15秒、
入力不変での試行は20.45/20.95秒だったが、その後local exact cache復元時の巨大FractionのGCD再計算を除いた。
既に正規化済みの元Fractionを同じexact値で復元する変更で、古いlocal cacheとも互換を保つ。
これら開発試行の値を最終固定codeのcold/warm値として代用しない。

## 結果の同等性と残る科学上の不足

同じ4候補のREQUESTは旧339版とbyte一致する。remote assignment、PowerShell手順、ZIP、出力pathも旧版を維持する。
両runのconfiguration、job ID、X/Y配分、費用、条件付き改善額、dry判定の同値は
[EQUIVALENCE](EQUIVALENCE.json)で全4方式PASSを確認した。
今回のサーバー2結果はFull PF・B0I・PR-DFがdry-check PASS、Pauliが従来と同じassembly SOURCE不一致。
warmで再利用した結果はfresh PASSと呼ばず、元結果locatorと元statusを保持した `CACHED_RESULT_REUSED` とする。

Pauliの保存最安410199237 RZは26数値単価の通常入口へのbindingが未完であり、
通常入口内部best437758513 RZとの差は解消していない。
LD4候補の `Unsupported saved Pauli assembly SOURCE: entry code differs: src/trottertracks/pauli_signal_entry.py`
も解消していない。Pauli本番HOLD、遠隔8セルは今回実機未確認のままである。
準備が速くなったことを、最新4方式の科学的比較条件や3ホスト実行資格が新たに成立した根拠にはしない。
保存最安・次候補・条件付き価格の詳細は[前回REPORT](../rpe_distributed_h4_plan_20261117/REPORT.md)を参照する。

この実装を作成した各担当が、新規のcache 5、compact 6、readiness 3、保存追加→選択の接続1、
計15件の小さな検査を実施した。著者自身の検査であり、別担当による独立科学検証ではない。
保存追加の人工testは実 `HistoryCache` →既存 `METHOD_MINIMUM_PRIORITY_V2` → `O.optimize` を通し、
人工PASSの追加後にbest配分と次pendingが変わること、新Q3呼出し0件を確認した。
identity・route変更時の再検証、未確定marker更新、古いexact cache復元のexact一致、
既定出力維持、compact範囲外の拒否、dry-check FAIL保全も対象に含む。

開発中にstdout観測wrapperがJSON文字列をdictとして読んでREADY後に停止した例があったが、
そのCLI自体はREPORT保存を完了していた。最終wrapperはdictだけをeventとして扱い、
READY・REPORT完了・process終了をそれぞれ確認する。

## 実測範囲と次の改善

coldでは初回context生成68.74秒が依然として大きな割合を占める。
cacheなしでmetadata providerを準備する時間は残り、未知入力やSOURCE・catalog・価格規則の変更では
必要な差分の再確認が発生する。warmの時間を新しい科学条件へ無条件に外挿しない。
今回の改善は既存結果と同じ準備を速く返すためのものであり、MSE、科学優位性、実H4 Q3の速度や
新条件の妥当性を新しく認定していない。
