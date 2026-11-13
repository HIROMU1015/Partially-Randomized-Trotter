# A → B：13round物理input契約

2026-11-13 JST。対象は固定H4 DF10・S2/L_D1/K2/ΔPF=.1・h=.1/M12・exactκ、保存state条件付きPRのみ。**SOURCE付き転記であり、新しい科学評価ではない。** 元N packetを上書きせず、この物理packetを参照する。

- canonical B：`3216d27a69ac8532d2a22e78f5f94491d202d238`。
- configuration：`587d20f7ed906d25841615dc9746674af872675b895bd438c06d3e74abf0e911`。
- 元A全13証拠：`169d3316dd008eb98dd9be56fe4ecefc5c1a7d60`、science c76ffe5d…。
- 新入力：[PHYSICAL_INPUTS.json](PHYSICAL_INPUTS.json)、schema `A_H4_PR_PHYSICAL_INPUTS_V1`。
- 元packetの`diagnostic_scope`は実行史を示す。新10round評価＋旧3round保存を、全13roundを同一SOURCEで再計算したことへ変更しない。

## Bが読む項目

| 項目 | パスと使用条件 |
|---|---|
| 全体identity | configuration_sourceのB正本とconfiguration_id。exactκ/M/全schedule/binding/profile/policyを照合 |
| roundの物理条件 | rounds[m].physical_candidate / structure / canonical_candidate_id / source_request_id / identity_source / law_binding |
| raw mean | rounds[m].diagnostics.signal.raw。complex real/imagの元値 |
| reference / corrected / exact-tail | 同signal内。比較診断用で、correctedをそのまま物理Bernoulli平均として使わない |
| normalization | diagnostics.normalizationとsignal.log_B。全interrogationのB、独立伝播のclosure、logB margin |
| bias・phase・radius | diagnostics.signalの3 bias成分、observed、conditional_error_disk / conditional_geometry。全energy誤差への変換は未取得 |
| 数値不確かさ | diagnostics.allowanceとsignal.allowance_evidence、reference_pair、bias_decomposition_residuals。すべてempirical |
| referenceの新旧経路 | reference_context。S1は旧中心・新pairの両方、旧S0/m1はdiagnostics_sourceのpairを参照 |
| SOURCE | diagnostics_sourceは公開resultのcommit/path/selector。mean_provenanceは元mean、scientific_sourceはその局所診断の実行SOURCE。read_sourcesは読取blobの既存方式SHA256 |
| shot数 | nominal_N_X/Yは保存配分のcopy。十分N_X/Y、energy_bias_ha、energy_RMSE_bound_haはnull |
| target/state | targetの保存solver energy/residual・元SOURCE、ground_state_certified=false、targetは保存DF-Hの同sector ground |

13 raw complex値、reference、局所診断、nominalはすべて揃っている。m3/m4の観測normは欠測でSOURCE gateに条件付けられる。rawを取得し直した、数値gateを新たに再判定した、または厳密intervalを生成したという意味ではない。

## 測定統計との対応（記号契約のみ）

既存`pf_pr_rpe_pipeline.MEASUREMENT`のX=cosine/Re(raw)、Y=sine/Im(raw)を再利用。Hadamard wrapperはYで最終Hの前にSdg、bit0を+1とする。

`p_X(bit0)=(1+Re(raw))/2`、`p_Y(bit0)=(1+Im(raw))/2`。

今回これらの実確率を計算していない。Bが承認された科学評価でmaterializeし、有限性・確率範囲・数値誤差を扱う。範囲外を黙ってclipしない。rawは既にattenuatedなので更にBで割らない。Bは正であり、正の共通scaleによる局所atan2の不変性はfinite-shot branchのRMSE保証にはならない。元nominalにはnormalization由来の負担を一回含む。

毎shotで独立なideal-law trajectoryとBorn outcome、X/Y・round間の独立性を仮定した場合にBinomial countへ接続できる。使い回したtrajectory、実RNG law差、control/compiled実装noiseの証拠は供給していない。`actual_implementation`内の未知量はnullのまま。

## RMSE契約で保持する境界

- 元priorとdecoderを保持し、E_stateでbranchを決めない。priorへのclipping等を追加しない。
- 保存referenceはfull-H **saved-state** mean。ground単一位相と同一視せず、保存energyは未資格のtruth proxy候補。
- `outer_PF_or_discard_bias=exact_tail−reference`と`finite_RTE_bias=corrected−exact_tail`はsignal空間の分解。phase/Tや単純和をεtrotへ代入しない。
- Minkowski成分の比較対象・coupling・未解決trial lossはまだ追加の判断が必要。common effective H、finite-Kと原論文モデルの対応、actual lawの資格も不足。
- 全13round nominal coverageと、十分N/global RMSE資格を別表示。`sufficient_N_qualified=false`、`global_RMSE_bound=null`、`common_effective_H_qualified=false`を保つ。
- B旧S0/m1専用validatorへ値を偽装して通さない。B側の新scope受付・decoder統計はBが担当し、Aでは再実装しない。

## 技術的入口と次の担当

```bash
python3 -B scripts/export_h4_pr_physical_inputs.py --out <fresh-physical-inputs.json>
```

保存JSONの読取・照合・copyだけで、科学実行optionはない。出力済みfileへの上書きを拒否する。既存physics/decoderコードは不変。

**B**：この物理packetを最新受付へ接続し、固定13roundのideal-law decoder科学計画を仕上げる。共通identity変更が必要なら元SOURCEのbridgeを示す。

**ユーザー**：target proxy、未解決trial loss、経験的数値allowanceの扱い、有限統計の精度・信頼度、B科学SOURCEとcall範囲を承認する。Aの追加mean取得は最初のideal統計評価に必要ない。

**A**：今回の公開照合後mandatory STOP。新物理action・保存再解析・decoder simulation・C/Gを起動しない。
