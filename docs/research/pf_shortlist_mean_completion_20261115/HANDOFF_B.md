# A → B：shortlist6候補の完成signal packet

科学SOURCE **fb9a1878b677b7ee116b52e804e5a01cdaa96004**、完成raw/packet carrier **06a15ee45beb177c018f6260ea4a5ac8cf66cfe6**。旧staging217eab647から保存30 meanを再計算せず再利用し、新48件を取得。全6候補13round完成、B9c0e12d87の既存intake6/6 PASS。[REPORT](REPORT.md)／[PACKET_INDEX](PACKET_INDEX.json)／[B_INTERFACE_CHECK](B_INTERFACE_CHECK.json)。

## 使うSOURCE

各候補のconfiguration descriptorはraw commitの`run01/<label>/PLAN.json:configuration`、packetは同`B_SIGNAL_INPUT.json`。固定descriptorはPACKET_INDEXに全件ある。`T.create(..., original_source=configuration_source)`→`T.bind_signal(..., original_packet_source=packet_source)`を無変更で実行した。

| label | 元configuration先頭 | B total-view先頭 |
|---|---|---|
| FULL_PF_S2_16h_over_7 | 11cf8ca6 | af18d37d |
| FULL_PF_Y4_16h_over_3 | a799697f | 833ca2c4 |
| FULL_PF_M5_16h | 70f7b317 | a4a5e71a |
| B0I_P7_S2_16h_over_7 | a6e55777 | e496358b |
| B0I_P7_Y4_16h_over_3 | defc5e88 | 95b2c417 |
| B0I_P7_M5_16h | 424eecbf | d1000f28 |

旧段階packetをriskへ渡さず、今回完成したpacketへ切り替える。Bのcode/interface変更は不要。保存値は元producerを`provider_binding.input_source`に保持したうえで、単一packet SOURCEへ束縛した。元結果の数値SOURCEと統合producer SOURCEを同一視しない。

## 信号契約

保存H4 DF10/sector(2,2)/同H/state/snapshot/共有target、P=0。全13roundで元configurationのformula/signed/actual幅/q/scalarを保持。Fullはc、B0Iはc+p7 tail identityの規則。raw=corrected、logB=0/B=1/attenuation=1。X/Yは同一complex meanのRe/Imであり、別の物理結果として複製生成していない。

新48件は保存direct stepのmatrix power/state内積、旧30件はそのままcopy。新reference/固有系/step生成0。数値gateはstep/power/norm1e−9、pair1e−10で通過。旧S2L4件は新directで取得し、旧legacy値を昇格していない。2つのm0共有roundは今回の明示方針に従って保存値を再利用した。

packet内のqualificationはUNQUALIFIED、state_error/law_distance等のunknownは保持。局所gateやB intakeを全RMSE/十分N保証へ昇格しない。今回BのMSE/risk/DP/Nは起動していない。PF/finite-Kなどの効果を別のbias値と二重加算しない。

次は、ユーザーの別承認範囲でBが固定推定器・誤差契約と全packetを結びつけてMSE/Nを評価する。Cへは元configuration/physical IDを保ったNと適用条件を渡す。Aは公開・照合後mandatory STOP。
