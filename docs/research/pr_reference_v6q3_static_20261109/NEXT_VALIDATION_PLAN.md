# Q3-N：必要最小限の有限generator監査案

**未実行。別途の明示実行承認が必要。** [REPORT](REPORT.md)／[静的補題](ALL_SUPPORT_ACCEPTANCE.md)で残ったのは、固定literal poolの受入れmetadataと、完成済みgeneratorの共通格子幅である。Q2、旧M/B_norm、既存compiler資格を繰り返し実行しない。新しいtrajectoryやm17 whole回路を使わずに、この不足だけを埋める。

## 1. 解消する義務と解消しない義務

Q3-Nは、固定SOURCEの有限poolを網羅し、次を確認する1回の有限監査campaignに限定できる。

- providerが読む全basis parameterの型・finite性・shape・wire/classical条件。A1の全catalog族の証拠を再利用する。
- Dと中央RTEの完成済みbinary角、D global_phase、固定scalar、0/π/τのdyadic指数。basis対の厳密相殺により、basis角の指数解析は不要。
- 静的な丸め・剰余の格子閉包の条件の下で、半角込みの十分ℓが1022以下か。

Q3-Nは実m17 schedule callerを実装しない。受入れの有限監査が通っても、caller運用接続・未確認context・実機RPE・総Tは別義務として残る。旧算術モデルに対する条件付き保証と、実wheel全体の形式証明を区別する。

## 2. 固定SOURCE・入力

| 固定ref／入力 | 用途 |
|---|---|
| 元科学SOURCE／入力 `f2638a7845685a119619382b850441f5d8d33713` | partial S2／repeated／D ops／RTEと旧certificate。不変更 |
| provider `eaf70d886d42364a01b7e949ebc21f3b026bbe72` | 型判定・ratioの意味を再利用。科学laneへのprovider呼出しは不要 |
| Q2 `9560e310e9f848fe59a1dafce4b506d5a08aae76` | 既存RESULTS/CHECKS/STOPを参照。LITERALSを全support poolに見立てない |
| V6-P `5f672ce6884bd49231366875a9e24a539fcaec7a` | 保存M／B_normを引用。phase_boundやshot式を呼ばない |
| `artifacts/pr_reference_interference_pilot/2026-11-06/common.pkl.xz`（元科学ref） | 保存preps[3]、全D／RTE registryの記録gate、既存DF係数。新SCF／DF／ground solveなし |
| 同refの`A1_REFERENCE_PHASE.json`と旧STRUCTURAL_APPENDIX | 全D＋registryの族のcoverageと文法、half/full doubling等の既存証拠 |
| 同refの`artifacts/pr_reference_bprime/2026-11-07/V2_INPUTS_CONSTANTS.json` | anglesのr=32、order=0/2、actual_hexを再利用。保存all_angles_equal／exact_equalはSOURCE対応の既存証拠 |

common圧縮archiveの保存SHA256は`85d7e21a41e770596b92327d909f38f0ae091c875e2dad3651abeb660f26ae16`、解凍pickleは`68269aa0415f048a0220beb4054daa1bba715c6eb29fbbe9d5ba779baf0346f2`。[既存Q2入力manifest](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eaf70d886d42364a01b7e949ebc21f3b026bbe72/artifacts/pr_reference_v6q/2026-11-09/preparation/Q2_INPUTS.json)の来歴を再利用する。これは科学入力の同定であり、新しい承認証明は不要。今回このpickleをunpickleしていない。

固定点はm17/q131072/r32/bits88、lane step数n=q/2。fixed cacheの同一性、Qiskit1.3.0、binary64／旧算術モデル、genuine SOURCE族を維持する。旧Nやlawの確率は新入力として解析しない。

## 3. 科学callの有限範囲

### N1：受入れmetadata（全有限catalog、回路なし）

保存cacheをread-onlyで1回読み、preps[3]のdeterministic_blocksの全runtime_basis_operationsと、rte_preparation.basis_registryの全runtime_operationsを走査する。既存A1とregistryのID/hash対応を参照し、gate class/name、parameter数・実型・finite性、width、wire範囲・重複、clbit/condition、numericであることを確認する。

真正RZ／XX+YYのinverseは一次SOURCEの符号反転則で閉じるので、inverse回路を構築しない。必要ならparameterの符号反転値だけで同じ型条件を確認する。basisの局所Operator、matrix fingerprintの再生成、Gaussian decompose、QuantumCircuit、全trajectoryを呼ばない。既存登録済みmetadataを新たにregisterし直さない。

### N2：完成済みgeneratorと格子（有限5 D context＋固定scalar＋保存2中央角）

既存`track_c_h4.diagonal_primitives(block,time)`を読み替えず再利用する。これはops.pyのdataclass primitiveを返す関数であり、gateをappendする関数は呼ばない。

| 有限集合 | 固定した取得方法・対象 |
|---|---|
| D half 4種 | deterministic_blocksの各block、time=.01を各1回。rz/rzz/global_phaseの完成値 |
| D0 full 1種 | block0、time=.02を1回。既存boundary fusionの実生成値。fullをhalfの数学的2倍へ置換しない |
| scalar | repeated SOURCEと同じ左結合順の`-.02*n*(constant+extracted_identity)`を1回。n=q/2。完成値を監査し、新policyや再結合を使わない |
| 中央RTE θ | 保存anglesのr32/order0,2 actual_hexを読む2レコード。符号±とhalfをexact ratioで扱う。新atanやevent生成を呼ばない |
| phase generator | D完成phase、上記scalar、0、同じmath.piとnormalizerのbinaryτ。event符号は0/πに閉じる |

post-compiler lowering catalogはSOURCE global_phaseやprimitive生成途中を完全には保存していないので、これをD primitiveの代用品にしない。Q2の4 laneのliteralからpoolを推測しない。固定tailのidentity成分なしという既存条件を照合し、違う場合はscopeを広げずSTOPする。

各finite完成値について`as_integer_ratio`を取得し、既約分母が2^kであることとkを保存する。θ集合にはhalf分の+1を付け、phase generator集合と合わせて静的補題の十分ℓを算出する。basis角を除けるSOURCE条件と全catalogの受入れ条件は別に保存する。SOURCEの積で生じる内部値を任意に省略して0誤差とみなさず、今回の格子主張は**完成したSOURCE出力値**から始める。

g_F/g_Gの実到達値、S_F/S_G/S、RN64(S)、実phase誤差の数値評価は0件。m17 F/Gは生成しない。old phase_bound、qualify、campaign、events_for、sampling、DF preparation、basis registry再構築、科学runnerのinitialize／attach_cacheは呼ばない。

### N3：検算・証明への代入

同じ保存hexから、共通分母2^ℓに掛けた値が整数になることを独立の整数演算で確認する。θはexact半角で検算する。catalog coverage、対象5 D context、r32/order2値の保存witness、固定scalar式・period、未評価項がないことを確認する。providerや科学primitiveを再実行せず保存出力を検算する。

剰余の格子閉包は有限標本で検証するものではない。旧SOURCEのcorrectly-rounded add/sub/remainderモデルを使うことを明示し、弱いγ誤差上界だけで代用しない。SOURCE意味論への適用が受け入れられない場合は、数値が通っても条件付き／未資格としてSTOPし、環境全体の形式検証や別providerへ自動拡張しない。

## 4. 判定規則とSTOP

| 判定 | 必要条件・扱い |
|---|---|
| `PASS_IN_FIXED_ARITHMETIC_MODEL` | 有限catalogが完全、全parameter受入れ可・finite、指定primitive／scalar／保存中央角のcoverageが完全、basis対・正規化格子閉包の前提が成立、半角込みℓ≤1022。同じ保存Mから上側範囲も継承。**この算術モデル下の全support受入れ条件**に限る |
| `FAIL_INPUT_OR_SOURCE` | 入力不一致、欠損、unsupported型／族／shape、非有限値、固定poolとSOURCE対応の不一致。補完・gate置換・別入力へ切り替えずSTOP |
| `INCONCLUSIVE_LATTICE` | ℓ>1022、必要値が揃わない、basis対／normalizer閉包を適用できない。Sがsubnormalだと断定しない。追加min/max、整数到達性探索やtrajectory列挙を自動開始しない |
| 固定case反例 | 実際にfixed support内の不受入れを示す証拠がある場合だけ反例と呼ぶ。粗い十分条件の失敗・人工例は反例ではない |

科学入力を用いる監査はこの1有限campaignに限り、技術的な不具合があっても別SOURCE／入力／providerへ移って救済しない。部分結果と不一致を残してSTOPし、再実行の必要性はユーザーへ戻す。検算は保存出力の整数検査に限る。実装時に単なるimportで科学初期化が走らないことを確認し、既存read-only cache loaderを流用する。SOURCE／旧cache／結果／grant／ledger／marker／STOPを変更しない。

必要成果物は小さい監査SOURCE、入力参照、有限literal・metadata／k／ℓ、検算とSTOP、短いREPORTに限定する。CPU/RAM/workerへ恒常的な人工上限を置かず、有限集合を一度だけ扱い、共有環境・他利用者へ変更を波及させない。

## 5. 実装と承認の境界・次担当

新しい科学算法は不要。既存ratio/Fraction、read-only common loader、diagonal_primitives、保存angle witnessとA1を呼び出す**小さい有限監査adapterだけ**が必要となる。新provider・word policy・precision・和アルゴリズムは実装しない。今回はこのadapterも作成・実行していない。

ユーザーが必要と判断した場合に承認する範囲は、上記固定入力のread-only unpickle、5 primitive contextと固定scalarの数値評価、保存中央2角の有限指数解析、共通格子評価、受入れmetadata照合、保存結果の整数検算である。承認はチャットの明示指示で足り、grant等を新規作成しない。

科学call範囲外は、全trajectory列挙、m17 build／compile／simulation、新phase評価、Q2再実行、sampling、strict guard／合成、RPE／shot／18round RZ／T再会計、新periodやsubnormal policyの採択、他Track。判定・結果・検算を保存・公開したらmandatory STOP。

次担当は**GPT／ユーザー**：Q3-Nを実行するかと、明示した算術モデル・条件の扱いを判断する。**Codex**は承認された場合だけ有限監査を実施する。実schedule callerの接続は別途必要な後続作業であり、Q3-NのPASSから自動開始しない。SOURCE接続がP角だけの置換で済めば既存compiler資格を流用し、新compileを必須にしない。別contextを要すると判明した場合だけ、その最小範囲を改めて固定して判断へ戻す。
