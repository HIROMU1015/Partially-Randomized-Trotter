# A/C signal・D price → B whole-shot adapter

2026-11-13 JST。[REPORT](REPORT.md)／[有限科学案](SCIENTIFIC_PLAN.md)。本仕様は技術接続であり、loss/confidence/探索範囲の採択ではない。旧[source別signal/N仕様](../rpe_shot_evaluation_design_20261113/HANDOFF.md)を継続する。

## 1. 既存候補とsignal

`configuration`は既存V2正本、`means`/`target`は`rpe_decoder_statistics.prepare`の既存形式をそのまま用いる。A/Cの元physical IDとcanonical IDのSOURCE付きbridgeを保ち、Bは新physical ID・signal・normalizer・referenceを生成しない。

各meanは`m, identity, law_binding, source, raw{real,imag}, corrected, normalization{log_B,...}, target, qualification, bias, uncertainty`を持つ。全schedule/H/state/snapshot/split/finite K/共通κ/prior/decoder/target/price profileはcanonicalのまま。rawからだけ観測確率を作り、corrected/B/attenuationを再乗算しない。欠測・underflow・数値未資格は0や理想振幅へ置換しない。

Full PF、Truncated PF、PR、Full-H RTEの意味とretained-H/one-bodyを保持する。旧B3やB0を別方式へ読み替えない。新formula/K/grid/Mの候補生成は本adapterの仕事ではない。

## 2. allocationと統計契約

```python
from trottertracks import rpe_shot_allocation as W
from trottertracks import rpe_decoder_fast as F

allocations = [dict(
    allocation_id=external_allocation_id, source=allocation_definition_source,
    shot_policy='USER_SUPPLIED',
    shots=[dict(m=m, N_X=exact_N_X[m], N_Y=exact_N_Y[m]) for m in range(M+1)],
)]  # callerの有限menu。複数件を同じ形式で渡す
statistical_contract = dict(
    id=external_contract_id, source=contract_source, claim_level=declared_claim,
    simulation_policy=adopted_joint_IID_policy,
    allow_conditional_unqualified=declared_conditional_scope,
    summary_options=dict(unresolved_loss=adopted_loss_contract,
        iid_repetitions=declared_IID_repetitions, confidence=adopted_confidence,
        loss_support=adopted_support_or_none),
    # target/raw/lawの適用誤差・選択後被覆等の追加定義もそのまま保持可能
)
entries = W.prepare(configuration, means, allocations, target=target,
    statistical_contract=statistical_contract)
```

allocation ID/SOURCEはphysical configuration IDとは別。全round X/Yの正整数を順に一組で指定する。X/Y等数、共通scale、group、探索範囲の制約は供給menuの契約で定める。round独立最適化、Nの単調性、候補ごとの異なるlossを暗黙に仮定しない。NはPython exact int（輸送時decimal文字列なら丸めず復元）。MCのint64 domain超過は元backendが明示拒否する。

統計契約に共通`id/source/claim_level`が必要。これは科学的前提と適用条件の記録で、承認証明書ではない。loss/IID/条件付きscopeをadapterが既定採択しない。`diagnose`を使わない供給証拠経路ではsimulation設定を含まない契約も表現できる。

## 3. 明示的な診断と外部N資格

許可された科学範囲だけで、各entryに対し次を呼ぶ。

```python
diagnostic = W.diagnose(entries[j], repetitions=specified_trials, seed=specified_seed,
    source=result_source, batch_decoder=F.decode_sampled_counts)
```

`batch_decoder=None`は元scalar。raw binomial、exact count→binary64、全roundのbranch/alias/zero/tie、loss/CIを既存関数のまま使う。diagnosticは入力allocation/契約/SOURCE・全trial/summaryを含み、`n_packet=None`。欠測や既定gateでblockされた場合、simulation/summaryは未評価のまま。NOMINALや条件付き点RMSEを十分Nへ昇格しない。

資格付き評価を渡す場合は、既存N packetを次のenvelopeへ収める。

```python
evaluation = dict(configuration_id=configuration['id'], allocation=allocations[j],
    statistical_contract=statistical_contract, source=n_packet['source'],
    n_packet=n_packet, diagnostic=optional_diagnostic)
```

packetの`rmse_evidence.statistical_contract`にも同じ定義を含めること。これは実際のrisk評価の契約をproducerが記録するもので、過去NにBが新契約ラベルを貼って資格化する操作ではない。旧packetに情報がなければproducerへ返す。N packetのSOURCE、全round identity/law、exact Nがallocationと一致し、既存`assess`のpolicy/error contract・有限K/signed条件・全RPE/alias/target資格を満たす必要がある。

MC summaryのCIだけからこのpacketを自動生成する科学producerは未実装。conditional/empirical/L1診断は別欄のまま。選択に使う複数配分のrisk資格には、同時被覆又は独立確認等の採択済み根拠が別途必要。

## 4. Dの価格とprofile/区間sidecar

`prices/cohorts`は既存controllerのFULL_WRAPPER_MEAN_COMPILED_PRICE recordと選択cohort mapping。DIRECT/MODEL/外挿/有限cohort/全law資格は元のまま。`mean.rz_count`は測定を含むcontrolled full Hadamard wrapperのRZ/shot、P=0。SOURCEとfull identity/law/compiler/control/profileを既存assessが照合する。

loweringを旧profileへ歴史上書きしないため、D HANDOFFに沿い次を別添する。

```python
price_extension = dict(lowering_id=required_lowering_id, source=lowering_definition_source)
price_bindings = [dict(
    identity=price['identity'], law_binding=price['law_binding'],
    axis=price['axis'], cohort=price['cohort'], price_source=price['source'],
    source=binding_source, profile_extension=price_extension,
    mean_interval=interval_or_none,
) for price in selected_prices]
```

SOURCEは出典定義なのでprice本体・binding・loweringで異なり得る。`price_source`だけは親mean recordと厳密一致すること。新旧loweringは同compilerでも別extension。bindingの欠測/混在はpoint Gもblockする。価格・extensionが未取得なら`prices=[]/cohorts={}`とextension/bindingsの省略で不足を保持でき、架空のloweringを作らずG=UNKNOWNを返す。

区間を供給する場合、`mean_interval`は`quantity='FULL_WRAPPER_MEAN_COMPILED_RZ_INTERVAL', lower, upper, source, qualification, evidence_kind, domain, coverage`を持つ。endpointはexact int/有理数、unknown upperはnull。`domain`は元round candidate全体。資格付きとして使える種類は`DERIVED_PATHWISE_BOUND`又は`SIMULTANEOUS_MEAN_INTERVAL`。unqualified/未知は元sidecarに保持する。

`coverage`は共通`interval_contract`と全一致し、`source, configuration_id, simultaneous=True, scope='ALL_SELECTED_ROUNDS_AXES'`を含む。confidence、model/rare/law/numerical errorの包含とその根拠はDの定義/SOURCEに残す。adapterは全cellの対応を検査するが科学的被覆を証明しない。独立性や共分散0を仮定せず、供給された同時boxだけを伝播する。点meanがないbound-only sidecarは保存可能だが、現adapterはinterval-only最適化を実装せずGをUNKNOWNとする。

## 5. 比較・外側controller

```python
result = W.compare(configuration, allocations, evaluations, prices,
    cohorts=cohorts, statistical_contract=statistical_contract,
    price_extension=price_extension, price_bindings=price_bindings,
    interval_contract=qualified_box_contract_or_none,
    unexplored_scope=declared_unsearched_allocations)
```

これは供給実値に対する明示的算術入口なので、実N/G使用は別科学許可範囲。provider/decoder/MCは起動しない。価格をN評価のfallbackとして取得しない。`diagnose`から`compare`までを実行しても、診断しかなければN/G/winnerは未資格のまま。

返却：配分ごとの元N packet/診断/assess coverage/SOURCE・G状態、点最小/ties、価格区間/共有差分、pending ID、未探索域。未評価・未資格は捨てない。NOMINALは正式比較から除外する。人工と科学のN証拠は、不適格候補の除外に使う場合も同一menuへ混在させない。finite cohortは明示`price_policy='finite_cohort_scenario'`の場合だけ条件付き費用を示し、期待価格のstrict winnerを出さない。

`point_minimum`と`strict_interval_winner`を区別する。後者も供給N資格を条件とする価格優越であり、全科学claim/探索外最適性の新規認定ではない。全menu解決時だけ点最小の`outer_evaluation`を一つ返すので、外側で`B.select(unique_configurations, representatives)`へ渡せる。代表には選んだallocation全体・SOURCE・統計契約も保持する。これは元controllerの点比較への接続で、区間による正式選定policyの採択ではない。pendingがあれば代表はnull。`outer_selection`は単一候補のAPI互換確認で、4方式比較完了ではない。

normalization、B²、古典価格sample数は追加乗算しない。全roundのNを一組で比較し、shot総数だけのwinnerへ代替しない。N/risk/priceのunknown systematicやcovarianceはnull。
