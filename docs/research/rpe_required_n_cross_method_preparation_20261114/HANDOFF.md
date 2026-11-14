# B1/B2 metadata adapterの引渡し

既存[total binding HANDOFF](../b_total_q3_canonical_integration_20261114/HANDOFF.md)との差分のみ。新risk kernel・N producer・controllerは作っていない。

## 保存baseline → opt-in total view

```python
from trottertracks import rpe_total_binding as T

config_source = dict(commit=pinned_commit, path=directory+'PLAN.json',
                     selector=['configuration'])
packet_source = dict(commit=pinned_commit, path=directory+'B_SIGNAL_INPUT.json')
config = T._selected(root, config_source)   # existing SavedArtifacts reader
original = T._selected(root, packet_source)
bridge = T.create(root, config, original_source=config_source)
view = T.bind_signal(root, bridge, original,
                     original_packet_source=packet_source)
# metadata only; science_calls == 0
```

固定locatorは[READINESS](READINESS.json)。`original_packet_source`はFull PF/Truncated PF/Full-Hに限定する。Pauliは既存A.accept_savedからの専用経路を維持する。PR-DF fixed-gridの拒否を回避するためにこの引数を使わない。

元packetとの完全一致を確認してから、新viewにcaller用`implementation_source`を追加する。単一SOURCE baselineで各mean.sourceとpacket.sourceが一致する場合のみ。元carrier・raw・target・normalization・qualificationは保持し、packet内に`origin_signal_packet`と`origin_packet_source`を保存する。既存callerへ渡す元canonical viewは`kernel_signal_packet`に明示する。意味の違うSOURCEを単に埋めない。

Cの同じ`plan/provider_binding_check/price_context/signal_input`を使い、元/新physical条件を比較する。C.prepareは固定c628e9e84の関数本体をそのままloadする。Cの数値factory・produce_signalsは呼ばない。新canonicalのconfiguration/candidate/law IDは別で、物理T/q/R/K・allocation・辞書・scalar・profileを変えない。

## 将来の数値入口（今回は呼んでいない）

```python
# Only after separate finite science permission and explicit contracts:
records = T.evaluate(root, bridge, view['signal_packet'], whole_allocations,
    evaluation_target=explicit_target, statistical_contract=explicit_contract,
    source=execution_source, options=explicit_Q3_options)
```

実行時は受入viewを再構成して改変を拒否し、既存Q.evaluateに元canonical/kernel signalを渡し、その結果を新契約へ結合する。`deterministic_input.origin_packet_source`にも元carrierを保持する。新SOURCE bridgeの入力保全を人工mockで確認しただけで、保存baselineの数値評価は0。全13round・X/Yのexact整数配分、lawに一致する測定モデル、明示target/loss/finite範囲が必要。旧partial `state_refinement`は禁止し、必要な場合はtotal版だけを使う。

`sufficient_N/n_packet/global_RMSE_bound/G/G_qualified`は自動生成しない。Q3のモデル閾値判定と元物理energy資格を別出力にする。正式N packetは新canonical/error contract/実消費signal/whole allocationの一致と外部科学根拠が必要。nominal packetはC.paper経路に留める。

価格は既存`T.bind_prices`とguard付き`R.cost`入口へ接続する。C/Dのreceipt・physical/law/実lowering/full wrapper/P0/cohort/mean intervalを保持し、同一性を確認してから算術へ進む。今回price reader・G算術・選択は実行していない。

## 担当と次の判断

- **ユーザー**：主推定量、target/risk/loss/数値scope、閾値τ、有限shot案と実行範囲を採択する。共通指示は追加後に11/14版を確認済みで、所在の照会は解消した。
- **Codex B**：採択後の固定範囲だけを実行。PR-DFでは旧grid保持の版付きbridgeが先に必要で、既存generatorを緩める作業は今回行っていない。
- **A/C**：Full PFの採択menuに対応する全round raw/SOURCEが不足。H_effだけでmeanを作ったことにしない。新spectral再構成も別科学scope。
- **C/D**：価格入力は別契約・別SOURCEのまま供給。新decoderのIDを理由にモデルdomainの未資格を解除しない。

**mandatory STOP**。新しいN/MSE/DP/価格/Gへ自動進行しない。
