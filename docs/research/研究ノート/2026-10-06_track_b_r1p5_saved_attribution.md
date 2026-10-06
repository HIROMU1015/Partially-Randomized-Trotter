# 2026-10-06：R1.5 saved-value attribution / GPTへSTOP

POSTHOC_ATTRIBUTION_DESIGN_INPUT。input R1 commitは`24bfeb84a4ce87b56985d174dd98d1d5e1702a2b`。
原result SHA256=`f726ad70cb2643533f0d037b518cde1b702724adb4e6571fea25e26e4bfdd61e`。
analysis script SHA256=`b26c3bdea5a42988c0534d45bd092d0cc28e50535e08dc75b4692ed601799911`。
no science rerun / no synthesis rerun / no new candidates。

利用者がR1保存値だけからprecision-envelope Pareto、G比の分解、angle/T-count帰属を許可した。
独立`track-b-r1p5-saved-attribution-20261006`をR1結果commitから作り、stdlib解析だけを行った。
新しいscalar score、resonance閾値、materiality、eta、precision、angle、contextは追加しなかった。

distinct-basis controlledのAは登録envelopeで、x=1/8では1e-4、x=1/4では1e-3/1e-4に残った。
全36 primary点を二符号別に比較し、座標exact equalityを確認したが、独立replicationとは扱わない。
216 exact G-factorizationsを保存し、1Qには既存Hadamard overheadを戻した。
126 saved angle keys、2,904 eventsのprobability加重T/1Q/error寄与を照合した。
追加circuit/compile/matrix/shot/synthesisは0。

gainの解釈はnormalizationだけでは不十分で、native費用とbias/shot側の相互作用が見える。
固定pygridsynthの離散列への依存も残る。限定診断はSUPPORTS_RA_RTE_DESIGNだが、
研究方向採択、新規性成立、R2/eta探索/DF接続承認ではない。
原R1 status/result/marker/source/authorization、R0/R0.5境界、旧証拠・共通API・Track Aを保持した。

[帰属報告](../../tracks/algorithm_codesign/r1p5_saved_value_attribution_v1.md)と
[GPT handoff](../../tracks/algorithm_codesign/r1p5_gpt_handoff_20261006.md)を公開してmandatory STOP。
次の数学設計・RQ・新規性・追加検証の必要性/範囲はGPT側へ戻す。
