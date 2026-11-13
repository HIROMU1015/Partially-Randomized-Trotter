# C → B：第1水準・第2水準のGに必要な入力と取込み手順

[REPORT](REPORT.md)／[有限価格検証](PRICE_VALIDATION_PLAN.md)。

**このbranchは、Bの本線にまだ適用していない。** 実データでの正式なG計算もしていない。

## 1. Bが受け取る入力（configurationごと）

| 入力 | 内容・出所 |
|---|---|
| identity | B canonicalのconfiguration ID。各roundのcanonical candidate、identity、law_binding。physical candidate（C/Dの`price_context`）。Pauliでは`split.tail_dictionary`（辞書のhash）を含める |
| 価格（round×X/Y） | `FULL_WRAPPER_MEAN_COMPILED_PRICE`の`mean.rz_count`。classification（`DIRECT_COMPILED_DETERMINISTIC`／`DIRECT_COMPILED_FINITE_SAMPLE`／`MODEL_PREDICTION`）、expectation、cohort、n_paths、law_context |
| DIRECTとMODELの区別・適格性 | 第1水準：`paper_nominal_cost`のtier。第2水準：Bの`assess`規則（qualified expectation、model資格） |
| 区間・残差 | sidecarの`mean_interval`（lowerとupper、qualification、coverage）。modelは`price_model_validation`の較正残差とholdout結果。共分散は未取得なのでnull |
| comparison policy | `id`、`source`、`profile`、`by_method[route]`（native_policy、rte_rotation_lowering、scalar_quantity）。routeは`FULL_PF`／`TRUNCATED_PF`／`PR`／`PR_PAULI_TAIL`／`FULL_H_RTE` |
| implementation evidence | sidecarの`implementation`（method、route、profile、measurement、native・wrapper・scalarのSOURCE、`rte_lowering_source`）と、固定commit JSONの`implementation_evidence` |
| N packet | 第1水準：`nominal_packet`（`PAPER_NOMINAL`）。第2水準：十分N（Z1＋T1/Q3が採択された後）。どちらも`signal_input`のbindingを持つ |
| UNKNOWN・未探索 | 欠測cell、`NOT_EVALUATED`、許容差を超えたholdout、外挿の範囲、探索していない幅・p・κ |

## 2. guardの再pin（Pauli routeを使う場合）

- Bのbinding（`codex/b-price-implementation-guard-20261114`の`rpe_price_implementation_guard.py`）は、C guardを固定`3a8b37f1a`のblobから実行している。そのため現状では`PR_PAULI_TAIL`を知らず、Pauli tailのPRは`PR`項目で照合される。
  - Pauli D2の価格にZZの発行元を付けても拒否される。
  - 逆に、DF用policyの`PR`項目でPauli価格を受理してしまうことは、Pauli sidecarが新ZZの発行元とIDを偽装しない限り起きない。
- Pauli routeを比較に含める前に、bindingの固定commitを、本branchの**Pauli bridgeのcommit**（`cfc0fae6`より後。DのLOWERING_IDとcanonical physical IDの検査を含む）へ更新する。その後、Bの`check_rpe_price_implementation_guard.py`と`cross_route_in_C_tree.py`を再実行する。`cfc0fae6`の暫定Pauli ID（`…_9C4006BF`）は廃止した。
- 旧guard（`3a8b37f1a`）の14件のテストは、本branchでも無変更でPASSする。既存4 routeの挙動は同じである。

## 3. 第1水準（原論文型G）の会計

**推奨**：Cの`paper_nominal_cost.paper_cost`を呼ぶ。
- guardを先に通し、`B.cost`・`B.select`は呼ばない。
- 出力の`G_paper`は`G_qualified`とは別fieldで、tierとstatusが付く。
- Bの`select`に混ぜない。比較は`paper_compare`（共通policyとtask、`qualified_winner=None`）で行う。

**B側にpathを設ける場合**：`evidence_kind='PAPER_NOMINAL'`を明示的に受ける別の入口を作り、結果は`cost`とは別の`paper_cost`fieldに置く。既存の`assess`・`cost`・`select`の資格判定を緩めない。

```python
from trottertracks import paper_nominal_cost as PN
packet = PN.nominal_packet(root, config, signal_packet, source=N_SOURCE)   # nominal N only
out = PN.paper_cost(root, config, packet, prices, cohorts=cohorts,
                    price_bindings=bindings, comparison_policy=policy)     # guard -> exact sum
menu = PN.paper_compare([out, ...])                                         # ordering, no qualified winner
```

## 4. 第2水準（MSEの資格付きG）

従来どおり、Bのguard付き入口を使う。
- `rpe_shot_allocation.compare(root=..., comparison_policy=...)`
- `R.cost`
- `R.select`

十分Nのpacket（`SCIENTIFIC`、error contractが外部で資格付けされたもの）が揃うまでGはUNKNOWN。controllerの`B.cost`・`B.select`を、科学packetで直接呼ばない。

## 5. 担当

| 担当 | 作業 |
|---|---|
| B | guardの再pin、第1水準を取り込む方式の決定、第2水準の十分N（Z1＋T1/Q3） |
| D | truncated builder・receipt、Pauli tail dispatch、Full-H P-1/P-2 |
| C | 価格の受入、sidecar、model検査（本件のAPI） |
| ユーザー | 比較policy・holdout許容差・各取得範囲の承認 |

各担当は、承認された範囲を超えて自動で進まない。

## 6. Pauli bridgeのB取込み（[PAULI_BRIDGE](PAULI_BRIDGE.md)§1）

Bのintake（`rpe_pauli_signal_intake.bind_packet`）が返す`C_plan_price_bridge='UNKNOWN_PAULI_PHYSICAL_ID_ADAPTER_REQUIRED'`は、次の呼出しで置き換えられる。

```python
from trottertracks import rpe_optimization_pipeline as I
view = result['signal_packet']
req = I.plan(root, result['configuration'], target=view['target'], signal_source=view['source'],
             implementation_sources=view['implementation_sources'])        # provider 'A_PR_PAULI'
I.provider_binding_check(req, view['means'][m])                             # per round; compare() also runs it
```

- configurationとIDはAとBのものをそのまま使う。Cは`B._validate`で検証するだけで、IDの生成器は複製していない。
- 実LD1/LD2では、保存carrierのphysical contextと26件すべて一致した（metadataのみ）。
- Bの実viewを`I.compare`に通す確認は、B側で取り込んだ後に再実行する（A12）。

## 7. 第1水準のN packetの形（BとCの対応）

| B `rpe_evaluation_levels`のpaper packet | Cの`nominal_packet` |
|---|---|
| `status='PAPER_EMPIRICAL_NOMINAL_ONLY'` | `kind='PAPER_NOMINAL_LEVEL1'`、`evidence_kind='PAPER_NOMINAL'`、`status='PAPER_NOMINAL_NOT_SUFFICIENT'` |
| `rigorous_sufficient_N=false`、`sufficient_N=null`、`global_RMSE_bound=null` | `sufficient_N=None`、`rmse_qualified=False`、`global_RMSE_bound=None`、round別の`sufficient_N_X/Y=None` |
| `rounds[{m, identity, law_binding, source, nominal_N_X, nominal_N_Y}]` | `rounds[{m, identity, law_binding, nominal_N_X, nominal_N_Y, log_B_used, nominal_status}]`。SOURCEはpacketの`source`と`signal_input`にある |
| `policy` | Cのpacketにはない。configurationの`rmse_policy`をBが付ける |

- どちらも、Bの既存`pr_rpe_rmse.nominal_axis_shots`の結果を使う。
- Bの形へ変換するadapterは、Bが第1水準の取込み方式を決めた後に作る（Cでは作らない）。

## 8. 新decoder version

[PAULI_BRIDGE](PAULI_BRIDGE.md)§4のinterface仕様を参照。
- Bが旧・新のconfigurationと、round対応の記録を発行する。Cはそれを受け入れ、価格を再束縛する。
- Cは新しいIDを推測・生成しない。
- PauliではphysicalなIDも変わるので、固定assertionの再発行か、guardのbridge経路が必要になる。
