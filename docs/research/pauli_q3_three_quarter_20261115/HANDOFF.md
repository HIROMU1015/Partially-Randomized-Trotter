# A/CへのHANDOFF：Pauli Q3新2件／PF完成packet

[REPORT](REPORT.md)／[SUMMARY](SUMMARY.json)／[PF_FINITE_PLAN](PF_FINITE_PLAN.md)。B1はBのQ3であり、Dの価格B1-W/V/CALとは別。価格取得・E[C]/Gは今回0件。

## Pauli結果のconsumer入口

- 実行SOURCE **f54b41a2e311e44124c9e180e26b01fe11d4eabd**、[INPUTS](INPUTS.json)、`run01/RESULT_LD1.json`／`RESULT_LD2.json`。元controlcaf1417は別結果で再評価していない。
- 元Pauli source7142476、元→total bridge、total canonicalは同じまま。新allocation IDは`PAULI_LD1_USER_SCALE_3O4_01`／`PAULI_LD2_USER_SCALE_3O4_01`、両軸`[48,45,42,39,36,33,29,26,23,19,16,13,9]`。
- `record.diagnostic.deterministic_risk`は元kernelの出力をそのまま保存。`adopted_q3_assessment`に今回採択されたexact τによる判定を追加した。旧kernel内の`q3.limit=null`は未filterの原出力であり、追加の採択判定を否定するものではない。`adopted_evaluation`が現在のモデル評価契約。
- `record.deterministic_input`は実際に消費したsignal・evaluation_target・SOURCEとcanonical viewを保持する。既存N packetやqualificationを新発行していない。Cへ十分N packetとして渡さない。
- 2件とも`MODEL_BOUND_MEETS_ADOPTED_THRESHOLD`。これは限定されたモデル・binary64 allowance内の評価済みvector。`sufficient_N/global_RMSE_bound/G/G_qualified/winner=null`、ground/sampler移送未認証、数値certificateなし。
- `partial_decoder`の未解決診断、coarse/lemmaの大きな上界、totalのzero HOLD/tie argminを分離する。total DP残余はbounded lossで会計済み。AのδEを再加算しない。

## PF完成packetの入口

[PF_PACKET_ACCEPTANCE.json](PF_PACKET_ACCEPTANCE.json)はA1adb7e65のPACKET_INDEX・B_INTERFACE_CHECKとC0b4b2d94のALIGNMENTを参照する**metadata overlay**。AのPASS6件を再利用し、新しいintakeライブラリ・plannerを作っていない。

各`candidates`の`configuration`／`signal_packet`locatorをそのまま読む。raw carrier06a15ee4、統合producerfb9a1878と、`rounds.original_producer_SOURCE`を区別する。元13round signal、normalizer、target、scalar、signed schedule/actual幅、物理IDは原packetを正本とする。新total IDへ元price receiptを付け替えない。

次の評価には既存`rpe_total_binding.create`→`bind_signal(original_packet_source=...)`→`evaluate`を使える。新total-viewはAのB受入と一致済み。C候補表の旧signal/decoder待ち文字列は当時の状態として保持し、この新SOURCE付きviewを優先して状況を更新できる。価格列・cohort・qualificationは変えていない。

## 次の担当境界

- A：元signal・数値allowance・physical identityの正本を保持。今回はsignal再取得不要。
- B：提案PF6有限Q3の明示承認後に限り、各1vectorのモデルriskを評価。採択済みQ3方針は再決定不要だが、PF vector/有限範囲/科学call許可は次回必要。PR-DF/Full-Hの追加riskは別範囲。
- C：元configuration/m/axisからC ALIGNMENTのphysical ID・SOURCE・price roleを取得し、元→totalのsource-bound bridgeとguardを使用。今回の結果は十分N資格ではない。原論文型nominal/G_paperと混ぜない。
- D：既存の独立した価格研究を継続する際の参考情報。今回DのB1価格取得を依頼・起動していない。適格E[C]が揃っても正式Gに必要なN/誤差移送条件を別途保持。

ロードマップ影響はREPORT末尾。個別担当として統合ロードマップ本体を編集していない。新結果とPF metadataの公開・remote照合後**mandatory STOP**。追加shot探索・MSE・価格・Gへ自動進行しない。
