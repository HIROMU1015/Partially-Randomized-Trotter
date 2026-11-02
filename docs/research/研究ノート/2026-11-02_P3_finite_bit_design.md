# 2026-11-02 P3有限bit law・角度誤差の非実行設計

ユーザーは公開Stage 4結果`a1d3f32aba6416fd1e200cd66a97d1a825b5dd73`の完了を確認し、有限bit law/角度を次の優先検証とした。compiled誤差と価格tailは後続に置き、科学的数値検証には別途明示許可を要すると指定した。

[設計案](../h4_p3_finite_bit_law_design_20261102.md)では、既存rte sampler/Stage 2・3のlaw接続/NumPy1.26.4のSOURCEを静的確認した。CDF整数区間とmoment平均の既存実装を再利用し、固定H4/案A/M_B・M_Cでbase/CDF/角度の各平均、計6複素信号を提案する。
保存CDF/角度表は未取得であり、将来SOURCEの丸めグラフから評価する。Stage 2 encoded operator上のlaw平均とliteral native Fullの実装平均を区別し、保存Gaussian・位相集約への接続は未取得項を残す。

差の未解像を0とせず、係数差からの上側評価候補と信号数値不確かさを報告する方針。bias/headroomへはgrid平均と保存referenceから直接接続し、旧nominal/splitの共有不確かさを再加算しない。
今回は文書と静的読取のみ、新しい科学数値・CDF/角度・平均・headroom・N/G・sampling/compileは未実行。全旧結果とSTOPを保全し、Stage 4 STOPを維持した。
次はユーザーが設計scope/精度政策と数値実行可否を判断する。許可後の実装・人工検査・限定実行担当はCodex。
