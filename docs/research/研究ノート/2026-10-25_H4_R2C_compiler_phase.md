# H4 R2-C compiler・位相の限定証明

固定H4/1Å/STO-3G、Pauli184/prefix27、S2/T.8/q8/r4/K4、Qiskit1.3.0/opt1/basis rz-sx-x-cx/seed17を変更せず、R2の未解決辺を限定調査した。

[技術報告](../../non_df_h4_r2c_compiler_phase_proof_20261025.md)。判定PROOF_INCOMPLETE。
全orderのcontrol恒等式、windingを含むphase更新式、native後の有限rewrite回数、exact Eulerの特異点で発散しないoperator補題まで整理した。
実際のparameter binding・Complex64/libm・簡約・remainderには数値保証が不足するため、PROOF_PATH_SUPPORTEDや予算達成には進めない。
具体的な有限kernelへ課題が絞れたので、PROOF_ROUTE_BLOCKEDとも断定しない。

新3＋既存7の人工10PASS。実H4状態作用/Arb/scalar上界評価/compile/sampling/RNG replay/科学grant/N_Gは0。既存6局所結果と全lawの不足、母平均費用tail、C/L、旧native UNKNOWNを保全する。
次はユーザーが限定数値証明の継続か実装family再設計を選ぶ。今回成果保存・公開照合後mandatory STOP。科学scalar評価も別の明示承認が必要。
