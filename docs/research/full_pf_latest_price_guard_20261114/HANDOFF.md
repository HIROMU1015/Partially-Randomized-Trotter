# B/D/Aへの引渡し：最新Full PFとSOURCE付き比較guard

最新候補・coverageは [REPORT](REPORT.md) / [COVERAGE.json](COVERAGE.json)。A71f04be7の `shortlist_labels` だけを選択し、旧C d7c09候補は保全した。要求幅はS2 16h/7、Y4 16h/3、M5 16h。定常幅q校正に使える保存点はM5 m4だけで、S2/Y4 m3はearlyの別幅。新計画は [NEXT_PRICE_VALIDATION](NEXT_PRICE_VALIDATION.md)。

## 取り込む差分

- `src/trottertracks/price_implementation_guard.py`：metadata検査と薄いB.cost呼出し。
- `src/trottertracks/rpe_mean_price_model.py`：optional comparison_policyとsidecarの保持。
- `src/trottertracks/rpe_optimization_pipeline.py`：比較前検査・科学packetのpolicy未指定拒否・family policy一致検査。
- `tests/check_price_implementation_guard.py`：既存人工fixtureとB会計を使用。
- 既存coverage builderのoptional固定SOURCE/ラベルfilter、新studyの小caller。旧coverage/planは上書きしていない。

branch全体のmergeは不要。Bは稼働中科学SOURCEを変更せず、完了後の技術取込みとして必要file/差分だけを採用する。Cの固定B b9b9ffa6で確認したAPIであり、Bの最新SOURCEへ無断適用していない。

## policyとrecordの契約

`comparison_policy` はユーザーが別途選ぶscenarioのmetadataで、次を持つ。

```python
{
  'id': 'explicit_scenario_id', 'source': {'commit': '...', 'path': '...'},
  'profile': config['spec']['profile'],  # compiler/control/measured wrapper/P0
  'by_method': {
    # FULL_PF / TRUNCATED_PF: RTE非該当、元nativeとscalar quantity
    # PR / FULL_H_RTE: 同scenarioで同じ実RTE lowering。SOURCEはrecord側にも必須。
    'FULL_PF': {'native_policy': 'saved_native_policy',
                'rte_rotation_lowering': 'NOT_APPLICABLE_NO_RTE',
                'scalar_quantity': 'HAMILTONIAN_CONSTANT'}
  }
}
```

これはfield例であり実比較policyを採択したものではない。RTEの新旧IDはD S1をそのまま使用する。旧/新を別policyで個別に扱えるが、異なるRTE loweringを同policyへ入れたり、policyが違う結果をI.selectへ混ぜたりしない。異実装を比較可能とする証拠・追加APIは今回採択していない。

価格側bindingに `implementation` と `implementation_evidence` を供給する。元価格を変更する必要はない。implementationはmethod/profile、native_policy、rte_rotation_lowering、native/wrapper/scalar各SOURCE、RTE時のみrte_lowering_source、scalar_quantity/ONCE_AT_TOTAL_T、measurement、必要ならexpected/resolved scalar。B0/B0Iのscalar構成は元split/sourceを保持し、tail-onlyをtotalへ名前だけ付け替えない。

`implementation_evidence={commit,path,selector}` は固定Git JSON内の下のmetadata assertionを選択する。producerが実行/receipt SOURCEを確認した結果を使い、現在のdefaultを推測して作らない。

```python
{
 'price_source': original_record['source'], 'identity': canonical_row['identity'],
 'law_binding': canonical_row['law_binding'], 'axis': axis, 'cohort': cohort,
 'physical_context': bound_physical_context,
 'profile_extension': existing_extension,
 'implementation': source_checked_implementation_facts
}
```

assertionは価格/N/Gの資格証明ではなくSOURCE/domainの受入metadata。旧recordに欄がなければderived sidecarとして元ID/producer SOURCEを保持して添付する。元direct receipt・cohort role・mean資格は変更しない。本作業では実価格recordの受入や上の実sidecar生成による価格採択は行っていない。

## B consumerで必要な最小変更

Cの `PriceProvider(..., comparison_policy=policy)()` → `I.compare(..., **out)` はguard済み。I.compareは科学Nでpolicyが無い場合、正式算術前にpendingにする。既存SYNTHETIC_ONLY test fixtureは旧呼出しのまま動くが、科学packetをtest専用へ付け替えて迂回してはならない。

Bが直接 `rpe_shot_allocation.compare` を使う場合、その関数のB.cost行だけを置き換えるのでは遅い。前段の `_price_metadata` と同じ位置、**B.assess/費用会計より前**に次を入れる。

```python
from trottertracks import price_implementation_guard as G
checked = G.check(root, config, prices, cohorts=cohorts,
                  bindings=price_bindings, policy=comparison_policy)
# ここから既存W.compareの算術。結果へcheckedを保持する。
```

必要な追加引数は `root, comparison_policy`、既存のprices/cohorts/price_bindingsを再利用する。policy/source不明・不一致なら拒否又は明示pendingとし、0価格へ変換しない。guard後のN資格・価格law・interval・Gの算術は既存処理を維持する。

直接B.costだけを呼ぶ場合は、既存controllerを渡す薄い入口を使える。

```python
G.cost(root, B, config, n_packet, prices, cohorts=cohorts,
       price_bindings=price_bindings, comparison_policy=policy,
       price_policy='qualified_expectation')
```

B.selectへ直接渡すfamilyでも、全候補のchecked.policyが同じであることを選択前に確認する（CのI.selectと同じ条件）。B元cost/selectは変更しておらず、この取込みをしない直接経路はまだguardされない。Cの結果をもってB全経路を修正済みとしない。

テスト条件：同compiler旧Full-H＋新PR、同policy内の異lowering、unknown/矛盾SOURCE、PF/Truncated非該当、scalar/wrapper/axis/physical不一致でB.costをmockし呼ばれないことを確認する。同domainの人工値では既存のcostを得る。Dの科学取得/fit/N solverは起動しない。

## 保全・次作業

実価格の再集計・fit/予測・compile、N/G、X/Y片軸compile採択は0。SOURCE確認・人工テストを技術成果として公開しmandatory STOP。BのDP実行を中断・変更していない。次はBの技術取込みとユーザーによるscenario/価格精度/有限科学scopeの判断。許可がある場合にだけCが保存値分析/fit、Dが限定価格取得、Bが対応するN/Gを実施する。
