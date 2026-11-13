# Z1＋T1/Q3のB統合とcanonical binding

2026-11-14 JST。Bの既存独立branch `codex/b-rpe-pauli-integration-20261114` を継続使用。基点 `a00d11c32d834e0ceda60eedecc46da0061efaf9`。技術実装・保存metadata照合・人工/回帰テストのみ。**新しい実H4/H6のMSE/N/G評価は0**。

## 完成した接続

1. 独立Codexの3 fileを実装SOURCE `710aa69b663ef3313439762d25c80a652a03948f` からbyteそのまま取込んだ。公開HANDOFFは `f3c84aa854583e78a88b9cbf260cce709325d342`。旧decoder・DP kernel・runner・controllerは変更していない。
2. `rpe_total_binding`を追加。既存Bの生成・検査・hash規則で、Z1＋T1とQ3の**未採択opt-in契約**を持つ新configurationを生成する。時間・R・K・辞書・回路law/profileは変えず、configuration/candidate/law-binding IDが変わることを明示する。
3. CのPauli planner/physical-ID bridgeと価格guardを、指定後の修正済みSOURCE `c628e9e8447c11c984064b922cbce6c52dc72d48` に固定した。`cfc0fae6`後の `bd7eec306` はPauli bridge、`c628e9e84` はD採択済みD2 conventionの保全・emitter照合強化。Cのplannerは複製していない。
4. 保存signalの新contract viewは、元configuration・元carrier/raw/target・numerical SOURCEを照合し、Cの元/新physical contextの差がcanonical物理IDだけであることを確認して作る。元packetを保持し、qualificationを上げない。
5. 価格は元IDで固定C guardを通してからdecoder-only bridgeで新契約に結合する。元receipt/assertion/source/cohort/価格値を保全し、新receiptは発行しない。正式入口は元証拠とbridgeを再検査してから既存B/W算術へ進む。bridgeなしの単純なID付替えは拒否する。
6. 第1水準は固定Cの`paper_nominal_cost`を再利用。人工データで`G_paper`を確認し、十分N・`G_qualified`へ昇格しない。実signalにnominal式を新適用していない。

人工/回帰 **110件PASS**（新bridge 8、取込み済み独立Codex 11、既存91）。[TESTS](TESTS.txt)と3 groupの完全出力を参照。人工の全count列挙、zero保持/tie argmin、解決履歴のbit一致、更新幅/principal guard/allowance、min上界、bias²/Var、元DP、C/B guardと会計、nominal/MSE分離を検証。旧MC/R3の人工回帰は保全しているが、新主経路はMCを呼ばない。

技術実装SOURCE `284a11c7da1b256cbd64f189a582cd431d29faba` の既存B wiring preflightもPASS（science_calls=0）。[WIRING_PREFLIGHT](WIRING_PREFLIGHT.json)。

## canonical対応（metadataのみ）

| | 元Pauli configuration | opt-in total/Q3 configuration |
|---|---|---|
| LD1 | `d905d2197d4b01df3074b3844c0f4a2b990c21b9c3e1a0e3b6ea13a488b1fd27` | `9fa3c47dbdef3b7d08d55cc02c9451cbb400ebbcf807b817db1abb72c5cd946c` |
| LD2 | `d7dd4f52bece343126258ffa24b9c4d7e1b8d76af82d1f4f4004819e832aa4d3` | `285bf4812e07a519f266ed88b83293bc4675493c5a9cc03985a6312363c6b8f0` |

[CANONICAL_BINDING_METADATA.json](CANONICAL_BINDING_METADATA.json)に全26roundの元/新candidate・law binding、時間/q/R/allocation、SOURCEとerror contractを保存。保存meanを使った確率・MSE/Nの算出はしていない。元targetのsolver proxy未採択・state/law/数値の未資格も保持する。

新contractは`decoder=consecutive_dyadic_nearest_branch_principal_energy_total_v1`とSOURCE・Z1/T1規則・`DIRECT_MODEL_MSE_Q3_OPT_IN`を含む。元Minkowski契約は参照としてそのまま格納する。既存controllerのpolicy familyを互換性のため保持するが、これは直接MSE契約の正式採択を意味しない。`scientific_status=PROPOSED_NOT_ADOPTED`、`risk_evaluation.adopted=false`。従って実科学N packetを自動で適格化する経路はない。

## risk engineの再利用方法

独立Codexの`Q.evaluate`は、旧decoder契約のconfigurationでcount表・閉包証明を作り、明示`options.estimator`でtotal推定量のriskを評価する設計である。これを変更せず、新adapterは検証済み元configuration/signalを**数値kernelの証明基底**として渡し、そのtotal推定量結果を新canonical契約へ結合する。`deterministic_input.kernel_input`に実際の旧kernel入力を保存し、新signal input・新error contract・bridge・estimator SOURCEも記録する。旧risk結果を新MSEと読み替える処理ではない。

新callerでは`state_refinement`（旧部分DP）を拒否し、任意の`total_state_refinement`だけを許す。新しい実H4計算は今回実行していない。小さい人工分布で独立Codexの元`Q.evaluate`とdiagnosticが完全一致することを確認した。

## 価格とCの境界

- C `plan`・`provider_binding_check`・`price_context`・`signal_input`・`compare`を、固定SOURCEのままB live moduleへ依存接続する薄い`rpe_c_interfaces`を追加。Cの科学provider loaderは呼ばない。
- `PR_PAULI_TAIL`はPauli D2専用route。DF ZZの`PR`、Full-H RZZ/ZZ、Full/Truncated PFのRTE非該当を区別する。D2 emitterと依存Pauli event emitterのSOURCE検査もCの既存処理を使用する。
- C側の元/新planはそれぞれcanonical IDを検査する。decoder versionをまたぐ保存signal/receiptの対応保証はBが担当し、Cに新IDを推測させない。
- C `compare`に、元receiptのまま新IDのviewを直接渡すと拒否される。decoder bridge付き正式入口はBの`R.cost` / `W.compare`（`canonical_binding=`）である。Cは新契約で発行された正規の価格sidecarなら従来経路を使える。Bが検証した元receiptを新C assertionと偽装しない。
- MODELの旧candidate domainを無条件に新domainへ認定しない。bridge viewでは未資格を明示し、外部の新domain証拠を待つ。有限cohort、未知CI/covariance、未検証外挿は従来の制約を維持する。

## 残るUNKNOWN

- Z1＋T1の正式主推定量としての採択と、Q3直接モデルMSEの主張水準。今回の接続はopt-inのみ。
- 新canonical error contractを満たす外部N資格packet。nominal packet・小さい上界・診断PASSだけでは十分Nにしない。
- target、raw数値allowance、実sampler law・IID・stateへの移送。binary64 transition allowanceはengineering allowanceであり認証ではない。
- 同一Pauli物理構成・law・loweringに対応した適格平均compiled価格。新MSE評価も価格再計算もしていない。
- 旧DFの`A_FIXED_GRID_510FBACD_V1`からのcanonical移行は別の固定adapter versionが必要。本接続は保存PauliのB-native scheduleを対象とし、旧DFを汎用ceil-gridへ黙って変換しない。元DFを旧canonical上のopt-in Q評価に使う独立CodexのAPI自体は保全している。

[HANDOFF](HANDOFF.md)にAPIとN/price binding、[FINITE_SCIENCE_PLAN](FINITE_SCIENCE_PLAN.md)に未承認の最初の有限2評価案を記載。元DP結果・raw・STOPは[PRESERVATION](PRESERVATION.json)のとおり変更なし。AのδEを直接raw-model riskへ再加算しない。

commit/pushとremote照合後、**mandatory STOP**。次の科学実行・shot探索・正式G比較へは進まない。
