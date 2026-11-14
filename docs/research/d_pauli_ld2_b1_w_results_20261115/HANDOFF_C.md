# Cへの引渡し：LD2 B1-W finite single path

観測pathのgateとX/Y compileがPASS。実行SOURCEは`1a7eed91dd96992cc6abcbfa1cd473a2044db16c`、raw公開SOURCEは`6c4709f85d0f28ce7c6957e9a70c85e378d8cad0`。branch `codex/d-pauli-b1-caller-prep-20261114`。[REPORT](REPORT.md)を参照。

| 公開commit `6c4709f85d0f28ce7c6957e9a70c85e378d8cad0`内のpath | 用途 |
|---|---|
| `docs/research/d_pauli_ld2_b1_w_results_20261115/run01/RESULT.json` | 物理request、event、D2、作用gate、元cohort/index/seed |
| 同dir `PRICES.json` | X/Y direct finite price |
| 同dir `IMPLEMENTATION_EVIDENCE.json` | guard証拠、selector `m0X` / `m0Y` |
| 同dir `RECEIPT_TRACE.json` | 元wrapper/compiled QPYの所在・hash |

公開後の[SIDECARS_PUBLISHED](run01/SIDECARS_PUBLISHED.json)を使う。[C_GUARD_CHECK](run01/C_GUARD_CHECK.json)は固定C `c628e9e8447c11c984064b922cbce6c52dc72d48`で両軸metadata PASS。未結合sidecarを受入済みとして使わない。policy名`B1_TECHNICAL_COMPATIBILITY_NOT_ADOPTED`は既存の技術確認policyであり、比較policyの正式採択ではない。

物理系は保存H4/1Å/STO-3G/DF10、LD2/S2/m0/T.1/q1/R2/K2/P0。configuration `d7dd4f52bece343126258ffa24b9c4d7e1b8d76af82d1f4f4004819e832aa4d3`、physical ID `RMSE:babca160daaf91bf567324e3e3ce75d7f2b6c6b9658634d5042e970d4ddc444d`、辞書SHA `dd986e52057aa01a330a83dd147316696dbae669dc8a51828ee2a54734016ea7`。入力H/state/snapshotと元normalizationは[PREFLIGHT](PREFLIGHT.json)。nativeはDF head one-body＋fragment0,1、tailはPauli D2 central rotation control、scalar `c+s_R`を総Tで一回。別候補のID・DF-tail価格へ付け替えない。

cohort `D_PAULI_LD2_B1-W_M0_PROPOSED_V1`、index0、seed `3788801539369037729`。eventは順にorder0 component34 `XZZXYZZY`とorder0 component40 `IIXZIZXI`。LD2 order2 product・非自明event位相は未観測であり、実LD2でのその項目はUNKNOWN。追加drawはしていない。

X/Y各RZ2236・CX960・depth1416、Qiskit1.3.0/rz,sx,x,cx/opt1/seed17、状態準備除外。`DIRECT_COMPILED_FINITE_SAMPLE`、`FINITE_COHORT_ONLY`、n_paths1、law_expectation=false。モデル較正用`C_INPUTS`は空。全law E[C]・CI・N/Gへ昇格しない。

将来B1-Vを採択する場合、この公開Wを元path0として元SOURCE/cohort/index/seedのまま使う（`reuse_wiring_source={commit:上記raw SHA,path:上記RESULT}`）。独立n2の新標本へ再標識しない。既存`ld2_wiring_gate`のmetadata読取はPASSしたが、今回VもCALも実行していない。CALのcell/cohort/許容差・model資格は別途判断する。

ロードマップへの変更候補はLD2初回Wの限定完了とC接続。残るorder2観測・統計/期待価格資格等はREPORTの通り。ロードマップ本体は無変更。**mandatory STOP。**
