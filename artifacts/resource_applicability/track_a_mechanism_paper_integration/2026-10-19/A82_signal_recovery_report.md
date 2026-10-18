# A82原signalの限定回収・SOURCE/SHA照合

2026-10-19 JST。**原signalを回収。metadata同一性PASS、説明側数値検証は未実行。**

対象はA82_OLD_PR.PR、H8・ε=.05・B2_p7_q2_2nd_r4_K2、旧confirmation union n32、通常signed・factor1。
固定D0結果3d910ee1de4597b8895e8506500553dbad5a5085のcontract_v1.jsonにある
absolute_launch_command[9]から、そのinput_mapping_private.jsonのsignal_summary.json entryへ進んだ。
同契約がreceiptで固定するsignal・selection・confirmation・completion auditの4 fileだけを照合した。
projects全体、無関係private領域、agent履歴を横断する探索は行っていない。

signal_summary.jsonは1392287 bytes。D0 input receiptのSHA-256
`dac86904b9a9d5d72146956ce2ed7aa3f9da52545169f5bbecb2af45c8ae6706`と全byte一致した。
対象は92 signal中の`/signals/32`。candidate cellのprefix7/q2/r4/K2/outer2nd/R8をexact metadataで照合した。

signal科学SOURCEは`c31ef424175c02379eafebe8fc6c7548c1d9aabe`、D0解析SOURCEは
`20f89bc32b161ee22e4a1ed709f9e3414c19633a`。SOURCEを同一の役割として混同しない。
D0 bindingと元completion auditの入力snapshot/compilerが現在の固定A82入力に一致する。
auditの線形H8・隣接1 Å・STO-3G・tol-only1e-8/cutoff0・DF28・T=.8を確認した。
保存selectionにはこの候補のconfirmation replica0〜31が一意に32件登録され、fixed_before_confirmation=true、
exploration_confirmation_pooled=false。保存expected_signalは当該recordのcorrected signalと一致した。
保存precision endpointのshot欄も既存A82 payloadと同一。これらはmetadataと保存scalarの等値照合で、再計算ではない。

回収fieldはstep b、step数R、累積nominal log_B、empirical log_B_margin、total_signed.real/imag。
原recordとfield pointer、原本・mapping・receiptのSHA、2 fieldだけを加えた未実行入力をlocal専用監査領域へ保存した。
log_B_upper、headroom、shot bound、整数丸め、説明比を今回計算していない。

**0edd62403c35c5ee18b2a2a152158faca5c2ebd8のA82欠測statusは保持する。** 入力回収は説明成立の認定ではない。
既存core32、説明完了63 side／31比較、旧grant/ledger/marker/STOPを変更していない。
[bindingのmetadata投影](A82_recovered_input_binding_projection_DRAFT.json)と[未承認契約案](A82_reanalysis_contract_DRAFT.md)を参照。

初回metadata準備では固定payloadのTが文字列"0.8"であるschema差を検出した。
その失敗記録を保全し、文字列のexact照合へ修正した。科学解析のattempt/grant/retryは0。
限定回収調査はここで終了する。
