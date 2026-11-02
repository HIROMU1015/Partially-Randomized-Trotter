# 2026-11-02 P3有限bit law・角度の限定実行

ユーザーの「実行してよい」により設計commit`c52bbf2a76708079075f01f57af76676a6d5e7ff`の固定6平均を実行した。
科学SOURCE`d22ca967955a664260a3a9857df243b3f151610f`を人工9PASS後に公開・bytes照合し、18固定入力/16 SOURCEから1batchを約6.49秒で完了した。
[報告](../../../artifacts/df_pauli_matched_partition_validation/2026-11-02/finite_bit/run01/REPORT.md)。

H4/案A/M_B・M_C/T.8/S2/q8/r4/K4のbase/CDF/grid計6平均について直接差はRe/Imとも未解像、両bridgeはengineering整合。
CDF質量差と角度係数区間からbase基準のlaw/角度上側候補P9.35e-14/DF1.80e-13を得た。保存nominal基準はbridge不確かさ込みP1.43e-9/DF1.78e-9。
matrix/vectorはcomplex128であり、exact CDF count/scalar rotation区間を全signal厳密保証へ格上げしない。certified_gap_upper=null。

DFの35成分は53-bit CDFでcount0。全次数0/2/4は平均へ含む。grid-reference直接biasと保存精度operandによる20 headroomはnative bridge=0/compiled ρ=0という追加仮定で正。実装headroom・native一様接続・compiled ρ・価格tailは未取得のまま。
新sampling/native生成/compile/Gaussian/物理reference/finite Taylor bias高精度化/N/G更新0。旧結果/206manifest entry/全STOPを保全し、公開照合後mandatory STOP。
次の担当はユーザー：結果の利用範囲とnative接続差・compiled残差の次検証scopeを判断する。Codexは追加科学実行へ自動移行しない。
