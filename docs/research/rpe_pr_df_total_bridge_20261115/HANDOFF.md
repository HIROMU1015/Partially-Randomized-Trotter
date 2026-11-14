# A/Cへの引渡し：固定grid total view

2026-11-15 JST。既存[total API](../b_total_q3_canonical_integration_20261114/HANDOFF.md)と[baseline受入](../rpe_required_n_cross_method_preparation_20261114/HANDOFF.md)に対する差分。旧資料の固定grid未対応の記述は公開当時の履歴で、本成果がその技術的不足を解消する。

## SOURCE付きmetadata受入

```python
from trottertracks import rpe_c_interfaces as C
from trottertracks import rpe_signal_statistics_caller as K
from trottertracks import rpe_total_binding as T

api = C.interfaces(str(root)).pipeline
bundle_source = dict(commit='714247619afb4714fd663eb15c0470ef32bdc7af',
    path='docs/research/pr_rpe_first_signal_results_20261113/B_MEANS_LD1.json')
bundle = T.p.SavedArtifacts(root).read(bundle_source)
accepted = K.accept_saved(root, bundle_source=bundle_source, integration=api)
old_config = accepted['request']['configuration']
bridge = T.create(root, old_config, original_source=bundle['configuration_source'])
view = T.bind_signal(root, bridge, accepted['signal_packet'])
T.validate(root, bridge, view['configuration'])
```

LD2はファイル名のLD1をLD2へ変更する。科学callは0。元configuration・carrier・target・allowanceとCのphysical contextを既存受入器が照合する。自作のraw packetをSOURCEなしで埋めたり、未資格をQUALIFIEDへ変更したりしない。

## identityと変更可能なmetadata

| 項目 | 扱い |
|---|---|
| 旧grid | `A_FIXED_GRID_510FBACD_V1`の生成器・固定step・early規則・元κを保持。generic ceil-gridへ変換しない |
| 新view | `A_FIXED_GRID_510FBACD_TOTAL_DECODER_VIEW_V1`。元specを`decoder_origin_spec`に保全し、厳密な旧生成器から同じround列を得る |
| decoder契約 | rootの`rmse_policy.error_contract`だけをtotal/Q3 opt-inへ拡張。元contractを`risk_evaluation.minkowski_reference`へ保持。`A_input`内の旧decoderは過去の生成入力として保全し、新推定量の指定とは区別 |
| canonical ID | configuration、round candidate、identity、law bindingのID部分は新しくなる。元/新両方をbridgeに保持 |
| physical ID | `round.physical_candidate`と`candidate.source_candidate_id`を保持。T、qPF、実幅、R、K、allocation/event数、lawの定義、λ/SOURCE、scalarを変更しない |
| signal/target | raw/corrected、finite normalization、reference/bias、数値uncertainty、target、source、元packetをそのまま保持 |
| 資格 | PROPOSED_NOT_ADOPTED、UNQUALIFIED、nullを保持。不明grid、元契約の改変、SOURCE/ID/LD混在を拒否 |

新しい候補やM/κを生成するbridgeではない。旧partial DPのMSE値・未解決lossを新推定量へ移送する処理はない。

## 将来のQ3入口

既存`T.evaluate(root, bridge, view['signal_packet'], allocations, evaluation_target=..., statistical_contract=..., source=..., options=...)`を再利用する。実データへの呼出しは**別の有限科学実行許可後だけ**。

Q3の既存support/lemma APIは元のpartial契約を閉包検査の基礎に使うため、kernelには元configuration・元signalを渡し、total推定量の結果を新canonicalへ結合する。`deterministic_input.kernel_input`と`canonical_error_contract`に両者を明示する。旧partial-state DPの再実行は禁止し、必要ならtotal版だけを指定する。kernelの科学的規則を修正していない。

全13round・X/Yを一組とするexact整数whole-shot vector、target/loss、測定law、model閾値、有限検査範囲、数値scopeは外部入力。今回、PR-DFの新しいvectorを選択・評価していない。

## Cの価格joinとの境界

C85557306のPR-DF coverageは**元physical ID**で[対応表](CANONICAL_MAPPING.json)と一致する。これはrecord/sidecarをguardで受理した意味ではない。P1/P2のcohort、native/lowering、full wrapper、scalar、SOURCE・law・価格区間の資格はC/Dの証拠のまま保持する。

価格receiptは元configurationでguard検査してから既存`T.bind_prices`によるdecoder-only viewへ結合する。fixed-gridではphysical IDを変えず、canonical identityだけを対応付ける。元receiptを新IDのreceiptとして再発行しない。今回のprice-view確認は人工fixtureだけで、保存価格値の移送・新平均・Gは実行していない。

原論文nominalはC.paper経路に留める。新decoderの十分N packetは、新canonical・error contract・allocation・実消費signal・科学的根拠が必要で、nominalや旧partial MSEから自動生成しない。

## 次に必要な担当作業

- **ユーザー**：主推定量採択、target/loss/model閾値、ideal IIDと実law、数値certificate範囲、N保証水準、有限科学scopeを判断する。
- **A**：Full PFおよび価格shortlistと同じphysical candidateのsignal coverageを公開する。新計算が必要かは共通指示§10.5で一次記録を確認してから判断する。
- **B**：許可後の固定risk評価。現段階はtechnical readinessのみ。Pauli caf1417の2件は再計算しない。
- **C**：新/旧canonicalと元physical IDの対応を利用し、同一候補の価格・law・cohort・SOURCEを照合する。C plannerや価格modelをBが再実装していない。
- **統合担当**：REPORTのロードマップ影響を、必要な箇所だけロードマップへ反映する。

**mandatory STOP**。今回は新科学計算の開始条件を満たしたとは扱わない。
