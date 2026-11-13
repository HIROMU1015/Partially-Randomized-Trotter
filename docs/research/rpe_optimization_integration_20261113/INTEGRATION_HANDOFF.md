# A/B/C/D統合の入出力

## 1. 取込みと依存

新module `rpe_optimization_pipeline.py` とtestを選択取込みする。C `86f37d25f` の2 baseline module・既存AST reader・固定Git objectsが必要。A/B/D branch全体のmerge、共通controller変更、他担当worktreeへの介入は不要。

`providers(root)` は固定A/BをCのprivate B/D namespaceへ接続し、`A,B,C,D,S,F,W` を返す。Dは公開済みphysical descriptor builderであり、並行中の価格取得adapterではない。metadata importではNumPy/SciPy/Qiskit/PySCFを起動しない。純RTE・元decoder・allowanceは実数値call時だけ読む。module中のsource constantsは出典で、承認証明ではない。

## 2. flow

```python
from trottertracks import rpe_optimization_pipeline as I
api = I.providers(root)
config = api.B.generate(fixed_spec, event_mapper=existing_mapper_or_none)
request = I.plan(root, config, target=original_full_H_target,
                 signal_source=fixed_numerical_source,
                 pr_physical=physical_by_round_or_none,
                 references=baseline_reference_descriptors)
price_requests = I.price_requests(root, request)  # metadata only, C=None

# 以下の実データcallは、別途承認された有限科学範囲でのみ実行する。
signals = I.produce_signals(root, request, fresh_output,
    context_factory=explicit_shared_factory,
    reference_provider=explicit_saved_reference_reader)
entries = api.W.prepare(config, signals['means'], allocations,
    target=signals['target'], statistical_contract=fixed_statistical_contract)
# 診断が許可された場合のみ。十分Nを返す関数ではない。
diagnostic = api.W.diagnose(entries[j], repetitions=n_screen, seed=seed,
    source=diagnostic_source, batch_decoder=api.F.decode_sampled_counts)

comparison = I.compare(root, request, externally_qualified_signals,
    allocations, external_N_evaluations, supplied_D_prices,
    statistical_contract=fixed_statistical_contract, cohorts=selected_cohorts,
    price_extension=D_lowering_extension, price_bindings=D_bindings,
    interval_contract=D_simultaneous_box_or_none)
selection = I.select(root, comparisons, formula_menu=explicit_formula_menu_or_none)
```

`comparison` と `comparisons` はそれぞれ1 configurationの結果とその有限list。callerがmenuを固定する。新しい候補生成/solver/最適化loopは本adapterにない。

## 3. signal側

- PR：Aの `metadata.plan` へ全physical recipeを渡し、元 `SignalBatch.run` を呼ぶ。factoryは `factory(config)`、referenceは `reader(job)`。source・B identity/law・physical context・数値statusを照合してB meanへcopy。保存inventoryの自動再解析/再試行はせず、既存A acceptanceへ返す。
- baseline：C `prepare/execute` をそのまま呼ぶ。factoryは `factory()`、referenceは `reader(descriptor)`。保存meanは元Cのexact domain再利用条件を守る。どちらもreference solverへfallbackしない。
- packetは `configuration_id,target,source,means,producer_result`。meanの `provider_binding` はphysical contextと元numerical/input/reference SOURCEを保持する。全meanは元H/state、q_RPEとqPF、signed structure、prefix・scalar・Full-H one-body/head0を保つ。
- production meanはUNQUALIFIED。局所数値gateと物理/RMSE資格を混同しない。別の受入資格が必要ならproducer側の出典付き処理で用意し、Cがフラグをtrueへ変更しない。欠測/数値失敗は残し、科学的FAILへ変換しない。

Aのgeneric context、Cの共有contextを明示callbackで再利用できる。A保存H4 factoryは同snapshot/sector、保存prefix λ/scalar、元reference pairの依存を要する。今回の最小private namespaceにはそのoptional factoryのrunner graphを追加していない。実行前に公開Aの完全なfactoryを渡すか、同じ保存SOURCEを使う共有factoryを固定する。実H4でのfactory動作確認は今回未実施。

数値型もidentityの一部。A physical recipeのbinary64 `T/.delta` と、B canonicalの厳密有理数は完全一致条件がある。`0.1` を `1/10` へ表示上の理由で変換しない。合わなければSOURCE/元入力規約を確かめ、IDを無断で付け替えない。

## 4. N側：B APIへの最小追加binding

allocationは元B形式 `allocation_id/source/shot_policy/shots[m,N_X,N_Y]`。複数allocationでもphysical configuration IDは不変。外部evaluationは元Bの `configuration_id,allocation,statistical_contract,source,n_packet` に、**そのrisk producerが実際に使用した** `signal_input=I.signal_input(signals)` を添える。bindingは元mean/target/sourceをcopyするだけで、新しいhash/承認機構ではない。

古いN packetへ今回のbindingを後付けし科学資格を作らない。情報不足ならpending。SOURCE/値が異なれば拒否する。Bのrisk/error contract、全round identity/law、exact N、finite-K・alias/target・truncation資格は引き続き元Bが検査する。

meanの物理資格/必要入力が未取得なら、供給N証拠を保存しても正式会計へ渡さない。nominal N、未資格RMSE、診断だけのN=NoneはG UNKNOWN。科学的N producerが完成したとはしていない。

## 5. D側：公開後に差し替える価格入口

`price_requests` は全round×X/Yのcanonical identity/law、元physical candidate/structure/H/split/profile、測定/単位、P=0、要求quantityを返す。mean・profile extension・sampler距離資格はnull。request自体は価格recordではない。`physical_context` は `I.price_context(request,m)` で、B N候補とD回路を結ぶ。

Dが返す価格は元Bのmean recordを維持する：`quantity=FULL_WRAPPER_MEAN_COMPILED_PRICE`、identity/law_binding、axis・measurement、cohort/n_paths、classification、expectation、mean.rz_count、source、model/domain、uncertainty。DIRECT/MODEL/外挿、有限cohort/全law、未知systematic/covarianceを元規約で保持する。

元B sidecar `price_binding` に `physical_context=I.price_context(request,m)` を追加する。その他は元B/Dの `identity,law_binding,axis,cohort,price_source,source,profile_extension,mean_interval`。`price_source` は親record、profile extensionは同一lowering/sourceと一致させる。unknown loweringを作らず、欠測ならGも欠測。新旧loweringをcompiler名だけで共用しない。

区間はDの同時box契約を元Wへそのまま渡す。UNKNOWN/未資格/forced rare/有限cohortを適格全lawへ昇格しない。Bは点最小と同physical配分間の区間優越を分ける。外側のcross-configuration同時区間・Nと価格のjoint confidence・科学的選定policyは未解決で、Cが新しい区間optimizerを追加していない。

D並行adapter公開後は、固定SOURCEと出力形状を照合して上のrecord/sidecarへ接続する。未公開branchを読み取らず、Cがsampling/compile/残差を代行しない。

## 6. 不完全menuとSTOP

供給recordは結果に残す。欠測/未資格allocationがあればWのouter代表はnullとなり、B外側menuもpending。利用可能部分の点最小を正式な有限menu最適性へ昇格しない。全有限allocationが不成立でも全てのNに関するphysical構成不成立とはしない。

Bの `scientific_optimum_adopted/cross_method_advantage_qualified/full_RPE_qualified` はfalseのまま。実行可能なAPIと科学的に適格な実比較を区別する。本技術成果の公開照合後mandatory STOP。
