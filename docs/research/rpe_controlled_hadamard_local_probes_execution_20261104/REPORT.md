# 12局所probe：固定component不在で停止

2026-11-04 JST / LOCAL。**許可された固定SOURCEを1回実行したが、component存在検査でSTOPした。
12 probeの回路構築・state-action・Hadamard平均照合は全て未実施。**
結果は `STOPPED_WITH_ERROR`、native exit1。科学判定はUNDETERMINEDのままである。
これは実回路と理論signalの不一致を観測した結果ではない。

実行SOURCE：`76d25ac69c22da6efeec673dc36c5fe04b71533f`。
[固定実行範囲](../rpe_controlled_hadamard_runner_preparation_20261104/EXECUTION_SCOPE.md)、
[runner](../rpe_controlled_hadamard_runner_preparation_20261104/run_local_probes.py)、
[人工検証60 PASS](../rpe_controlled_hadamard_runner_preparation_20261104/IMPLEMENTATION_REPORT.md)を変更せず実行した。
結果専用branch `codex/rpe-controlled-hadamard-local-probes-execution-20261104`、
同SOURCEから分離した別worktreeを使用した。

## 固定条件と停止点

linear H4／隣接1.00 Å／STO-3G／保存legacy DF12、8 system qubit＋ancilla8、
B保存physical stateのnormalized ray・4電子sector70、Partial L_D3、δ=.02、K2、
extract_identity_phase、coefficient_atol=1e-12。
親m0,8,10,11,13,17の保存q/N/α/β/εを維持し、
r1,2,4,8,16,32の各q_probe1/all0、q_probe2/stress、計12本を予定していた。
親の大きいqを直接構築する計画ではない。

SOURCE/inputのbyte/schema照合を通過し、H/state load、固定DF preparation、support登録まで実施した。
prepare_jobs内の `select_fixed_components` が次を検出して停止した。

```text
ValueError: UNKNOWN/STOP: fixed component unavailable
```

保存された原結果：[RESULT.json](../../../artifacts/rpe_controlled_hadamard_local_probes/2026-11-04/run01/RESULT.json)。
SHA256：`7454b6f6342d5cc2430eb9ec6b767384cac061915e9204603eb69dd96095c028`、135097 bytes。
原RESULTのsupport_proofs、counts、error、versions、resource contextを保持した。
parameters.json／vectors.npzは、この停止点では未生成である。

## 静的に特定した実装上の原因

停止後は既存m17 `source_replay.json`のID文字列・cutoffだけを読んだ。
新しいH4 preparation・operator・signal・θ・参照値は計算していない。

| SOURCEで固定したID | 保存1e-12 exportの218 component内 |
|---|---|
| `df-fragment-10:Z0` | 存在しない |
| `df-fragment-11:Z0` | 存在しない |

保存exportにはfragment10/11をprefixに持つcomponent自体がない。
fragment_idの保存literalはdf-fragment-3〜9であり、先頭IDは`df-fragment-3:Z0Z1`。
これは保存metadataの転記で、別componentを新しい検証対象に選択したものではない。

前段では、旧Bのcutoff既定0と今回の固定1e-12が異なる点を認識していたが、
旧Bの選定IDを12probeの固定IDに使い、保存exportへのmembershipを事前に検査していなかった。
**この適合確認不足はrunner準備側の不備である。**
人工fixtureはIDを差し替えてgeneric coreを検証しており、実H4の固定selectorそのものの可用性を覆っていなかった。
cutoffがID欠落の唯一の数値的原因であることを新計算で分離したとはせず、
旧BのPASS・保存科学結果を遡及変更しない。

## 実施数と未実施数

原RESULTは開始数を記録する。下記の4操作は後続のselector検査まで進み、
preparation identity・support residual guardを通過している。

| 操作 | 今回 |
|---|---:|
| 保存H load | 1 |
| 保存state load | 1 |
| 固定DF preparation | 1 |
| support basis登録 | 1 |
| finite config/distribution parameter replay | 0 |
| 96-bit参照F/R・component・trajectory構成 | 0 |
| uncontrolled／controlled raw／boundary_optimized回路build | 全て0 |
| state-action／wrapper build／Hadamard ideal平均 | 全て0 |
| CDF／sampling／compile／quantum shots／長q回路 | 全て0 |
| 新H/state solve／SCF／DF再因子化／成功率・費用再解析 | 全て0 |
| retry／追加precision／別profile／別namespaceの科学実行 | 全て0 |

96 bitは予定precisionであり、この実行で96-bitのH4参照を評価したとは報告しない。
全12request・24軸は未評価であり、未評価をPASSや数値0で埋めない。

## 実行環境・資源・原本保全

Python3.11、NumPy1.26.4、SciPy1.14.1、Qiskit1.3.0、python-flint0.9.0/FLINT3.6.0。
起動前のCPU affinity32、利用可能RAM約51 GiB、GPU driver利用不可を確認。
1秒の/proc/stat観測ではCPU idle0、memory pressure avg10=0だったため、
自jobだけnice10で起動した。共有設定・他jobのpriorityは変更していない。
予定は6r群／6worker／worker BLAS1 thread、参照FLINT1 threadだったが、
worker pool起動前に停止したため、**今回spawnした科学workerは0**。
原RESULTのworkers=6は予定構成であり、実使用worker数ではない。
runner観測wall timeは約4.43秒。peak RSS、実CPU占有率、速度向上率は測定していない。

```bash
MPLCONFIGDIR=<owned-temp-dir> nice -n 10 <research-python> \
  docs/research/rpe_controlled_hadamard_runner_preparation_20261104/run_local_probes.py \
  --run --output-dir artifacts/rpe_controlled_hadamard_local_probes/2026-11-04/run01
```

SOURCE・固定入力・旧結果・grant/ledger/marker/STOPは変更していない。
別Codexの全18round包含、SERVER2 H8、他worktreeの処理を呼ばず、原RESULTのbyte/hashを保全する。
旧synthesisに今回の停止を追記するが、過去の判定や記録を書き換えない。

## 判断と次の担当

今回、Controlled RTE・Hadamard回路の新しい一致／不一致は確認できなかった。
既存Bの固定6requestの条件付き平均PASSはそのscopeに限って残り、
全18roundへのpreparation/selector/τ結合、r≥2回路、長q phase転送は引き続き未完である。
V1/V2/V3/R1、m17の条件付き結果、INCONSISTENT/u_num=null、
`RPE_tolerance_certified=false`、`scientific_status=UNDETERMINED`、U_exec=nullを維持する。

次の提案：**Codex**が別指示後に、保存exportで存在・sign・basisを確認できるselectorへ
対象profileを再設計し、実保存payloadへの静的membershipテストを追加した別SOURCEを準備する。
今回のSOURCEや固定対象を上書きせず、旧結果から次の科学実行へ自動移行しない。
**ユーザー**が変更後の対象/SOURCE/実行範囲を判断し、科学再実行を別途明示許可する。

科学実行はここでmandatory STOP。報告・commit/push・remote公開照合を完了して終了し、
新selectorの選定・実装・再計算・レビューはこの実行結果の後続として自動開始しない。
