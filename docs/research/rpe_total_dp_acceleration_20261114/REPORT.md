# Total RPE DPの整数集約・process並列化

2026-11-14 JST。独立branch `codex/rpe-total-dp-acceleration-20261114`、基点 `eecbcdaed88b626b00858c7e94b15f598258d5ec`。実行中のPauli固定2 jobのworktree・kernel・runner・processは変更していない。新しい実H4 risk/N/Gや旧DP再実行は0。

## 実装と効果

`rpe_total_dp_parallel.py`にopt-in acceleratorを追加。旧DP本体をそのまま再利用し、伝播・残余集計・終端モーメントの算術をexactな等価計算で高速化する。新decoder/risk controllerは作らない。

| 人工benchmark | 旧DP | 高速版1 worker | 高速版8 workers | 高速版16 workers |
|---|---:|---:|---:|---:|
| 144,918遷移、5round、frontier512 | 4.335秒 | 0.772秒（5.61倍） | 0.763秒（5.68倍） | 未測定 |
| 2,191,510遷移、5round、frontier2048 | 80.936秒 | 5.539秒（14.61倍） | 3.827秒（21.15倍） | 3.782秒（21.40倍） |

両caseとも、旧DPに対して**全output dictが一致**。MSE上下界・残余mass/loss・bias²/Var/mean・zero/tie・状態数・transition数・allowance診断を含む。これは人工条件の性能であり、実H4の所要時間や同倍率の短縮を保証しない。環境は既存Python3.11.0rc1/NumPy1.26.4。warmup/繰返しなしの探索的実測。元の固定2 jobも同時に走っている環境で測定。[BENCHMARK](BENCHMARK.json)／[再現script](BENCHMARK.py)。

**人工・回帰43件PASS**：新7、旧total11、旧refinement12、旧deterministic13。[TESTS](TESTS.txt)。全人工8分布、worker数1/2/4とbatch分割、部分scope、3000bit以上の整数重み、signed zero、mass不一致、UNKNOWN、既存Q.evaluateとの互換性を確認。主評価経路のMCなし。

## 高速化の内容と同値性

1. **整数質量の因数分解**。元のstate-major順では各前状態i・count事象jに対して`mass_i * weight_j`を計算する。同一j・同一出力(flag,binary64 phase)の前状態は先に整数和し、`weight_j * sum_i mass_i`を一度計算する。多倍長整数の分配則なのでexactで、浮動小数点へ変換しない。
2. **count事象でprocess分割**。前状態配列を各workerへ渡し、count事象のbatchを処理する。元の`T.step/_nearest`をそのまま使用し、出力はuint64 bit patternでgroup化する。+0/-0を併合しない。親はinteger massを合算し、元float.hex state keyへ戻す。
3. **約分を遅延**。profilingで多倍長`math.gcd`が大きな負荷と確認。残余と終端モーメントを、未約分の質量numerator/denominatorと元のexact coefficientで集計し、分母のLCMを保つ。値を読む時点でFractionへ一度約分する。量・loss cap・principal guard・bound式は変えない。
4. **科学scopeの選択順を保全**。元と同じ`(-mass, str(state_key))`順の前状態prefixだけを承認transition budgetで処理し、残りには元のdrop式を使う。全countの処理後、元と同じfrontier pruningを行う。worker/task順で追跡範囲を変えない。

元`rpe_total_risk.total_state_dp`のASTをSOURCE固定hashで確認し、3箇所だけを委譲する。残余の幾何式、mass gate、allowance、pruning、MSE/Varの式、UNKNOWN処理は元関数bodyから使用する。固定SOURCE `710aa69b663ef3313439762d25c80a652a03948f` が変わった場合は無断適応せず拒否する。

最初は伝播だけを変更して約1.3倍だったが、profiling後に有理数集計も改善し、上表の性能になった。記録するのは最終実装の測定である。workerを増やすだけでは改善が頭打ちになるため、算術量削減を併用した。

## 計算資源・利用API

```python
from trottertracks import rpe_total_dp_parallel as P
engine = P.Accelerator(workers=None, outcome_batch=8)
out = engine(tables, h=h, target=target, scope=fixed_scope)
```

`workers=None`はCPU affinityとtask数から選ぶ。小さい自動選択taskはprocess起動費を避けてserialで処理する。明示worker数も指定可能。CPU/RAMの共有設定・他ユーザー・他jobは変更しない。追加packageの導入なし。

既存の`Q.evaluate`/canonical callerへのopt-in接続は [HANDOFF](HANDOFF.md)。各process内の一時的patchだけで、外部workerへの注入・hot reloadはしない。科学実行時はbackend SOURCE/worker/batchを新しいprovenanceに記録する。

## 実行中の2件への適用境界

現2 jobは実行時に旧kernelを読み込み済みで、現在のcheckpointには再開に必要なDP状態を保存していない。今回の実装を別worktreeへ追加しても、現processは速くならない。固定SOURCE・各1回の条件を維持して現在の2件を継続している。

高速版で同じ2件を実行する場合、旧jobの途中成果を保全して停止し、新SOURCEの別namespaceで2件を再実行することになる。これはユーザーの科学指示§2の固定SOURCE、§8の各1回、§9の無断再試行禁止に関わるため、**切替・再実行を具体的に許可されるまでは実施しない**。新しいshot/ρ/追跡範囲・loss/target・decoderへは拡張しない。

次：ユーザーが現2件の継続／高速版への切替を判断し、Codex Bは指定範囲だけ実行する。技術成果をcommit/push・remote照合後、新しい科学計算へ自動進行しない。

## 技術公開後の切替・実行結果

2026-11-14：ユーザーが旧run保全停止と高速版での代替2評価を明示許可した。固定条件は同一で、各1件・計2件が約235秒で完了。[結果](../pauli_total_q3_accelerated_results_20261114/REPORT.md)。上記の「未適用・旧job継続」は技術公開時点の履歴。旧2 jobは未完了のまま別branchに保全されている。新しいshot探索・十分N/G・主推定量採択へ進まずSTOP。
