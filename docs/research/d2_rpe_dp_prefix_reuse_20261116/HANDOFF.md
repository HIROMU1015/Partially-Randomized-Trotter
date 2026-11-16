# B2への引渡し：同じ構成のshot変更時のDP prefix再利用

**m12だけが変わる2配分ではm0〜m11のtotal-state DPを再利用可能。** 元kernel/acceleratorは異なる配分間でその再利用をしていなかった。別branch `codex/d2-rpe-dp-prefix-reuse-20261116` にopt-inで実装し、元のMSE上下界・判定と一致することを人工検査した。

実装SOURCE：`f33595ef044a0b84dd543fbc63a1298ffdd94ad6`。起点：D2 scheduler公開 `1eda4ba3f468dfba89823941222843ef5cfe0402`。最終公開commitは完了報告のremote SHAを用い、このディレクトリのpathを指定する。[REPORT](REPORT.md)、[SOURCE](SOURCE.json)、[TESTS](TESTS.txt)、[人工計測](BENCHMARK.json)、[保存入力照合](SAVED_INPUT_CHECK.json)を参照。

新規実Q3・信号・sampling・回路build/compile・fitは0件。B2の既存run・process・checkpoint・実行SOURCE・旧STOPへの変更は0件。実装者の人工技術検査であり、実H4での新旧数値一致や科学的最安点の更新は行っていない。

## 再利用できる条件と保存するもの

同じconfiguration・signal packet・binding/target・評価契約・rawから生成する同じprefix count tables、同じh・target・最終M、同じfrontier/pair上限、同じ数値SOURCE/runtimeを要求する。LD・κやR/event数だけが一致する別信号へは流用しない。後続shotの変更でscope上限まで変更した場合はmissになる。

prefix境界には次を保存する。

- `(partial_unresolved_flag, binary64_phase_hex) -> integer_mass` の全frontier。
- denominator、累積transition pairs、最大観測transition超過。
- 残余のmass・loss・energy/moment区間・guard_failed・flagged・drop件数のexact会計。
- 完了round summariesと、その時のaccelerator observations。

状態辞書だけを再利用すると、pair予算や既切捨てmassの会計を誤る。残余の未来loss上界はM/h/targetに依存するが、後続のshot数には依存しないため、これらが固定なら復元できる。Mが変わると終端frontier処理も変わるので流用しない。元のsuffix入力検査・mass検査・transition allowance gate・最終MSE/moments・Q3上下界の組合せを実行する。

既定では最後の1つ前だけを保存する。13 roundならm11。別roundの変更も扱う場合は `prefix_checkpoint_rounds` を明示する。例えば `[5,11]` はm6以降の変更にm5まで、m12変更にm11までを使える。前段のtableが変わればその境界のkeyも変わり、合うcheckpointがなければ通常計算する。

## B2からの入口

既存round CLIのconcurrent executorに、次の2項目を追加する。承認済みの次run用に新しい所有cache directoryを指定し、既存B2 runの設定を途中変更しない。

```json
{
  "executor": {
    "scheduler": "concurrent",
    "max_concurrent_jobs": 18,
    "dp_workers": 8,
    "wait_for_running_jobs": true,
    "prefix_cache_directory": "/absolute/path/to/owned_next_run_cache",
    "prefix_checkpoint_rounds": [11]
  }
}
```

18/8は前の実行profileを維持した例で、今回の人工計測から実DPの最適profileを選んだ値ではない。新しい科学runは本指示では開始せず、次の採択済みroundに設定する。候補範囲・shot探索規則・予算・source・保存再利用・評価条件はそのroundの指定を維持する。

```bash
cd /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/d2-rpe-dp-prefix-reuse-20261116
NEXT_TASK_ROUND=/absolute/path/to/next_adopted_round_with_prefix_cache.json
NEXT_TASK_RUN=/absolute/path/to/new_next_run
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/run_rpe_method_optimizer.py \
  --problem configs/rpe_method_optimizer/h4_df10_problem.json \
  --round "$NEXT_TASK_ROUND" --run-dir "$NEXT_TASK_RUN"
```

設定は `ConcurrentExecutor.request` → `REQUEST.execution_profile` → `run_rpe_optimizer_q3_job.prefix_cache_context` → 元route supervisorのfork → 元Observed constructorに届く。保存Observedそのものを人工tablesで動かすfork検査を実施した。固定acceleratorを共有する4方式に共通の入口で、分割専用のpatchはない。今回4方式の実科学Q3を実行したわけではない。

libraryから呼ぶ場合の最小例（Pは固定SOURCE viewからロードした既存 `rpe_total_dp_parallel`）：

```python
from trottertracks import rpe_dp_prefix_reuse as C
store = C.PrefixStore('/absolute/path/to/owned_cache')
Engine = C.accelerator_class(P)
engine = Engine(store=store,
                cache_namespace={'configuration': fixed_configuration_locator,
                                 'signal': fixed_signal_locator,
                                 'evaluation': fixed_evaluation_contract},
                workers=8, outcome_batch=8)
result = engine(validated_tables, h=h, target=target, scope=fixed_scope)
```

SOURCE viewにhelperがない場合は自branchのhelperをmodule fileとしてloadする（実adapterがこの方法を使用）。直接libraryを使う担当はconfiguration/signal/evaluationをnamespaceに入れる。callごとに変わるlabel・allocation IDはnamespaceへ入れず、変化したshotはtableのkeyで区別する。adapterはnamespaceを自動構成する。

## 元SOURCEとの対応

|commit / path|対応|
|---|---|
|`710aa69b663ef3313439762d25c80a652a03948f` / `src/trottertracks/rpe_total_risk.py`|通常DP、残余半径・loss/moment会計・final gate。`total_state_dp` を参照|
|同commit / `src/trottertracks/rpe_deterministic_risk.py`|raw・各roundのNからcount tables、partial/coarse/closureを計算|
|`9e294778266c9c867a8b1b98e040c2e94b1cb635` / `src/trottertracks/rpe_total_dp_parallel.py`|元AST guard・outcome並列化・exact和。fileと期待hashは変更しない|
|実装SOURCE / `src/trottertracks/rpe_dp_prefix_reuse.py`|元acceleratorの生成処理にloop前のrestoreとmass gate後のsaveを追加|
|実装SOURCE / `scripts/run_rpe_optimizer_q3_job.py`|このworkerだけにcache contextを接続。既存supervisor/workerは維持|
|実装SOURCE / `scripts/run_rpe_method_optimizer.py`、`src/trottertracks/rpe_optimizer_concurrent.py`|既存roundからcache設定を伝達|
|実装SOURCE / `scripts/inspect_rpe_saved_prefix.py`|保存inputのmetadata照合のみ|
|実装SOURCE / `scripts/benchmark_rpe_dp_prefix.py`、`tests/test_rpe_dp_prefix_reuse.py`|人工一致・速度・memory・通常evaluate・worker経路の検査|

元acceleratorの `_kernel` byte hashと元DPのAST hash/置換数guardを実行する。prefix loop hookは新SOURCEとして明示し、期待hashの書換え・guard無効化はしない。Python 3.11.0rc1・固定SOURCE・Linux forkで検査した。3.12で旧AST guardを回避する処理、別kernel/spawn実行への自動移送はない。

## 保存資料と再利用限界

保存Full PF S2のm12=4→2の2配分は、configuration `11cf8ca6677a7ce651a00d827d0948e758510f95876acd24bf4d826c2bb6ca5d`、signal/target/contract・m0〜m11配分・frontier8192・pair上限141559036が一致する。固定locatorは [SAVED_INPUT_CHECK](SAVED_INPUT_CHECK.json) のinputsに記載した。

旧結果はround summary・MSEを保存しているがfrontier/denominator/res本体は保存していない。今回それらを再計算しておらず、旧結果だけからcheckpointは作れない。新しい承認済み配分評価の最初のcold計算で保存し、同じprefixの次配分で使う。保存Q3の科学的再解析やMSE再計算は今回0件。

同一配分の完全なOUTCOME再利用は既存機能を使い続ける。新配分の一部DPを再利用しても、Q3予算上は新配分の1評価である。coarse/partial DP、closure、lemma、count tables、最終上下界の組合せは再計算する。

## 検証結果と性能の範囲

新規prefix10、既存scheduler関連15、元accelerator人工7、計32件PASS。full Q3人工evaluateでMSE上下界・model判定を含む全record一致、DP全出力exact一致を確認した。frontier/pairの切捨てあり、zero/tie、巨大integer、signed zero、scope/h/target/M/namespaceの変更、invalid suffix、cache破損、disk再open、別fork childの保存Observed経路も検査した。

3反復の人工13-roundで、通常2回のDP合計中央値2.5915秒、disk cold＋warmは1.3538秒（対比較中央値1.91倍）。2件目だけは1.2939秒→0.0385秒（33.62倍）。cacheは41,463 bytes、各modeのprocess peak RSS中央値はいずれも46.0 MiBだった。これはcache overheadが0という意味ではなく、小さいfixtureを含むprocess peakの測定分解能内で差が見えなかったという結果である。memory版もBENCHMARKに併記した。

人工fixtureはprefix X/Y各4 shot、m12のみ各3、frontier256、pair上限10,000,000、workers1。実H4全Q3ではtable作成やmetadata・closure等が残る。実H4の大きな整数frontier、実DP worker数8でのIO/peak memory、探索全体の実時間短縮は未測定。

同時cold startは両jobがprefixを計算する場合があり、上のwarm速度を保証しない。元の採否順・lookahead・科学budgetを変えてcache hitを強制することはしない。directoryは所有するrun単位で管理し、異なるprefix数に比例するdisk容量を記録する。科学rawはcacheと別に保全する。

## 記録と失敗時

job出力の `DP_PREFIX_EVENTS.jsonl` にPID・UTC・HIT/MISS/SAVE・再利用round数・key・失敗理由を保存する。復元したaccelerator observationsはprefixが元々計算された時のworker数等を含むため、今回実行したroundとHITイベントを区別する。

異なる入力やscopeではmiss。壊れたcacheはREAD_REJECTEDとして捨てて通常DPへ戻る。write失敗はWRITE_FAILEDとして記録し、通常の数値結果保存を行う。元の数値gateが失敗した結果や未完了suffixをPASSにしない。checksum/atomic replaceは科学状態の再利用整合性のためで、承認の証明を行うものではない。

## 人工検証の再現

```bash
Q3_TEST_PYTHON=/home/AbeHiromu/projects/h6-environments/20261012/native-dependency-v2/venv/bin/python
PYTHONDONTWRITEBYTECODE=1 "$Q3_TEST_PYTHON" tests/test_rpe_dp_prefix_reuse.py
PYTHONDONTWRITEBYTECODE=1 "$Q3_TEST_PYTHON" scripts/benchmark_rpe_dp_prefix.py \
  --output /tmp/new_unique_d2_prefix_benchmark --repetitions 3
PYTHONDONTWRITEBYTECODE=1 python3 scripts/inspect_rpe_saved_prefix.py \
  --output /tmp/new_unique_d2_prefix_input_check.json
```

固定route viewを読取専用で使い、kernel/acceleratorを人工tablesで検査する。保存H4信号をkernelに渡さない。元のAST guardが必要とするPython 3.11.0rc1を維持する。

## 次回実行前の差分照合・ロードマップ

§10.5では、元と同じconfiguration・signal・評価契約・配分の保存OUTCOME/進行中jobを先に照合する。未評価配分だけを元schedulerで評価し、その内部で一致するprefixを再利用する。SOURCE/runtime/scopeを変えるならkey一致を期待せず、未対応acceleratorはguardで拒否する。既存実行中jobへhookを差し込まない。

共通基盤に「近い配分間のtotal-state DP prefix再利用」が追加された。科学結果、精度資格、価格、LD/κ・配分の選択規則は変更しない。B2が次の承認済み探索でopt-inにできる状態までを今回の引渡しとし、実科学での評価・追加探索は開始しない。
