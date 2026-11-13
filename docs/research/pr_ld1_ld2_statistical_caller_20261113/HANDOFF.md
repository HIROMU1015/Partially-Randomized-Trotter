# A保存signal → B明示target caller → C signal_input

2026-11-13 JST。[REPORT](REPORT.md)／[TESTS](TESTS.txt)／[次の科学条件](NEXT_SCIENCE_PLAN.md)。本納品は技術接続のみ。下記のdiagnoseを実H4で実行する許可は含まない。

## 読取専用の受入

C `5cc7c07e112ce97205948465c94441cba0042416`の`rpe_optimization_pipeline`と、本納品の`rpe_signal_statistics_caller`を同じPython packageから利用する。B branchにはCのprovider graphを重複取込みしていない。C側では新moduleの選択取込みで足りる。元controller/statistics/shot/fastを再実装・差替えしない。

```python
from trottertracks import rpe_optimization_pipeline as I
from trottertracks import rpe_signal_statistics_caller as K

accepted = K.accept_saved(repo, integration=I, bundle_source={
    'commit': '91349e703e0d09ef702ccd108745dfaf1acc29f7',
    'path': 'docs/research/pr_rpe_first_signal_results_20261113/B_MEANS_LD1.json',
})  # LD2も独立に同じ受入。MC/normalization再計算/compileなし。
request = accepted['request']
packet = accepted['signal_packet']
assert accepted['signal_input'] == I.signal_input(packet)
```

`accept_saved`はp.SavedArtifacts.read→I.plan→bind_bundleを使う。既にC requestと原recordを読んでいる場合は、`K.bind_bundle(config,bundle,records_by_m,bundle_source=...,request=request,integration=I)`だけでよい。records_by_mは各mean.sourceが指すraw JSONそのものを渡す。欠測や照合不一致はValueErrorで停止し、再計算やfallbackをしない。

| field | 保存する意味 |
|---|---|
| packet.configuration_id/configuration_source | Aの元canonical V2 ID/固定configuration SOURCE |
| packet.target | 元A target全体。solver参照・proxy=false・ground未資格を含む |
| packet.source/numerical_source | C request.signal_sourceに一致する元数値実装SOURCE |
| packet.bundle_source/raw_result_carrier | 公開bundleと数値carrierを別記 |
| mean.m/identity/law_binding | 元候補・round・law。全13roundのRe/Imを一組で保全 |
| mean.raw/corrected/reference/exact_tail | 元の値を無変換copy。観測はraw、補正meanを確率に使わない |
| mean.normalization/bias/uncertainty | finite-K、経験的allowance・数値観測・unknownを保全 |
| provider_binding.physical_context | C.price_context(request,m)とraw physical_contextの一致 |
| provider_binding.numerical_source | C既存名。**mean.sourceのraw carrier参照**であり実装commitと区別 |
| provider_binding.implementation_source/input_source/reference_source | 実装・元入力・referenceの来歴 |
| provider_binding.carrier_qualification | rawの辞書を保全。meanのUNQUALIFIED文字列と別schema |

accept_savedはI.produce_signalsやA.SignalBatchを呼ばない。原A bundleは変更せず、provider_bindingはcopy側だけに追加する。

## 明示evaluation_targetとwhole-shot準備

```python
# 下記は仕様の形。values/source/scope/vector/contractを採択するコードではない。
entries = K.prepare(request['configuration'], packet, allocations,
    evaluation_target=evaluation_target,
    statistical_contract=statistical_contract, integration=I)
```

evaluation_targetは`{kind,energy_ha,source,qualification_scope,original_target_source,binding}`。kindに評価値の意味、qualification_scopeにproxy/state/model等の限定を明記する。bindingはpacket.target.binding、original_target_sourceはpacket.target.sourceと一致すること。値は有限scalar。未指定/欠測/不一致ならprepare時にもdiagnose時にもMC前に拒否する。reference_solver_energy_haを自動転記せず、元targetにenergy_haを追加しない。

allocationsは既存W仕様の`allocation_id,source,shot_policy,shots=[{m,N_X,N_Y},...]`。全13roundの順序とX/Yのexact正整数を保持する。float/bool/0/負数/null/欠測は拒否。metadataは巨大整数を保持し、MCでは元NumPy binomialのint64 domainを超えるNを丸めず拒否する。roundを独立最適化しない。

statistical_contractは既存の`id,source,claim_level,simulation_policy,allow_conditional_unqualified,summary_options`を使用する。simulation_policyは元fresh IID binomial lawとの一致を要求する。summary_optionsのunresolved_loss・必要なsupport/confidence/IIDを明示する。callerは政策や科学的主張水準を選ばない。unqualified meanの条件付き診断には明示trueが必要で、falseなら既存NOT_EVALUATEDを返す。

## 承認後に限る実行入口

```python
# 将来、ユーザーが有限科学scopeを明示承認した場合だけ実行する。
from trottertracks import rpe_decoder_fast as F
evaluation = K.diagnose(entries[index], repetitions=repetitions, seed=seed,
    source=result_source, batch_decoder=F.decode_sampled_counts)
```

batch_decoder=Noneなら元scalar。R3を渡せば元scalar互換batchを使い、acceleration SOURCEは既存の全trialに記録される。乱数/count変換/branch/alias/prior/energy shift、未解決status、loss/CIの算術は元S/Fを再利用。全trialの保持形式も既存どおりで、元simulateが省くcount配列を新たに追加しない。

出力は元Wの`configuration_id,allocation,source,statistical_contract,diagnostic,n_packet`に、evaluation_targetと**実際に消費した**C signal_inputを追加する。`n_packet=None`、十分N/global bound/G/winner=null。診断やCIから正式Nを発行しない。後日原meanを更新した場合、過去evaluationに新signal bindingを付け直さない。

## C/Dへの接続と資格

C.compareへ同request/packet/allocationsとevaluationを渡せるshapeにしたが、今回のA meanはUNQUALIFIEDであり、Cのphysical_qualification gateは引き続きblockする。正式N packet producer・物理/数値/law/全energy risk資格は未完成。binding成立や条件付きMCを理由にgateを緩めない。

Dのphysical identity/law/compiler/lowering/full wrapper/P0・mean価格/区間/SOURCEは、既存C価格adapterとW.compareの検査を使う。価格未取得はG UNKNOWN。nominalや人工診断、有限cohort、C1 model=falseを正式最適化に使わない。今回、価格/会計/選択コードには変更を加えていない。

次はユーザーが科学条件と有限範囲を採択し、Bが別明示許可の範囲だけ実行する。C/Dの進行中研究を変更・停止しない。**mandatory STOP**。
