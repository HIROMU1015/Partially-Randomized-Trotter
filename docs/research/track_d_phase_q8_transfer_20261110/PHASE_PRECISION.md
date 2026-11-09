# 宣言global phaseの精度確認

[固定科学条件・SOURCE](REPORT.md)。新[phase ledger](../../../src/trottertracks/track_d_native_phase160.py)は旧処理を改変せず、既存gateの返すphase項と1Q runの補正項をside記録する。既存compilerのPHASE_BITS=160を再利用し、入力floatをas_integer_ratioでexact binary64値としてMPFRへ変換する。集約はMPFR160、最後に160-bitで評価した2πにより正規化し、一度floatへ戻す。

`RealMPFR(float,160)`の直接変換はdecimal表記を介するため、入力binary64値の同一性を保つ目的で整数比を使用した。独自の位相fitや価格／保存P0値による補正はない。

## q4の再現と原因

| wrapper | 旧phase差mod2π | ledger後の差mod2π | native／最終word |
|---|---:|---:|---|
| Strong X | 4.4945593913e−9 | 7.4718009557e−12 | 旧保存列と完全同一 |
| Strong Y | 4.4945593913e−9 | 9.4141361373e−12 | 同一 |
| Reference X | 4.5615067279e−9 | 6.3495875224e−12 | 同一 |
| Reference Y | 4.5615067279e−9 | 4.6536108300e−12 | 同一 |

差は新しい宣言phase−保存P0 phase。同じ原本・同じgate出力で旧値を再現した。native phaseの旧逐次値と全項math.fsumの差だけで、Strongは4.4965418056e−9、Referenceは4.5547494665e−9 rad。旧約4.5e−9のほぼ全量に相当し、巨大な負の全体位相へ逐次binary64加算する誤差が主因と確認できた。

旧circuit_chunk／sinkはmath.fsumを使うが、各回が二項の加算なので全履歴の補償和にはならない。新ledgerは同じ局所出力項を高精度で集約し、1Q反復のphase項も同様に加える。gate definitionや1Q合成器自身は元binary64のままである。

## q8のbefore／after

| wrapper | 旧方式を同時計算したphase差 | ledger後の差 |
|---|---:|---:|
| Strong X | −5.8413725412e−8 | 1.4590328945e−11 |
| Strong Y | −5.8442829243e−8 | 1.2894796342e−11 |
| Reference X | −5.7574510492e−8 | 1.8450685424e−11 |
| Reference Y | −5.7622465022e−8 | 1.6755485888e−11 |

旧方式を価格合わせのために後から変更したのではなく、予測時に旧running値とledger値をともに保存した。数値再実行やgate／rule変更は0。新旧phase項のprivate原本を保存し、公開は集約値だけとする。[q4結果](../../../artifacts/track_d_phase_q8_transfer/2026-11-10/run01/Q4_PHASE.json)／[q8比較](../../../artifacts/track_d_phase_q8_transfer/2026-11-10/run01/COMPARISON.json)。

## Globalとrelative、解決した範囲

q4の新旧命令列は全て同じで、修正対象は完成wrapper全体の宣言global phaseだけ。これはX/Yの確率・誘導測定演算子から代数的に消えるため、この修正自体は相対χや測定信号を変えない。P(χ)／signed rotation／actual adjointはgate列の中に残し、変更していない。

新4人工テストは、長いbinary64和で失う小項、exact binary64入力の保持、旧native列との一致と小回路の位相付き作用、全体global phaseと相対readout phaseの区別を確認した。最後は2qubit人工observableで、全体phaseだけの変更が不変、相対phase変更が信号を変えることを確認した。新H4全入力simulationではない。[テスト](../../../tests/check_track_d_native_phase160.py)／[log](../../../artifacts/track_d_phase_q8_transfer/2026-11-10/technical/phase_tests.txt)。

**解決したのは大きな逐次集約誤差の主因と、その最小修正。** 全工程MPFR160化や全qの誤差保証は未取得。残る約1e−11 radには局所binary64合成出力、π／float変換規約、合成経路等の寄与があり、今回厳密な寄与分離やenclosureは行っていない。既存のprecision閾値を変更したり、価格一致から新H4 correctness PASSを付けたりしない。長qでの相対phase／compile誤差の保証、RPE成功確率・Nへの接続は未資格のままである。
