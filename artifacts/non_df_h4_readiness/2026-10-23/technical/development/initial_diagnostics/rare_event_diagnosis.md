# H2保存済み希少Taylor eventの技術診断

H2 1.00A STO-3G; DF tol-only1e-8 rank3; T=.8 q4 delta=.2 S2; PR r2 K4; Pauli LD7/14 and DF LD1/3 plus one-body

保存された法則・集計・価格を照合した。新sampling・RNG replay・回路生成・compileは0。

|case|order|法則確率/event|観測数 / 512位置|保存された未観測確率|母平均費用寄与|
|---|---:|---:|---:|---:|---|
|pauli_jw_PR|0|0.9985477358749303|510|0.0|未同定|
|pauli_jw_PR|2|0.0014519118543097264|2|0.4752477401829402|未同定|
|pauli_jw_PR|4|3.5227076003660005e-07|0|0.9998196536034751|未同定|
|df_native_PR|0|0.9995230410312447|512|0.0|未同定|
|df_native_PR|2|0.000476921026921472|0|0.7832982001091581|未同定|
|df_native_PR|4|3.794183387274608e-08|0|0.999980573969364|未同定|

各caseの64標本はseedとcos/sinのRZ価格を保存しているが、trajectory別event列・order位置・component IDを保存していない。
したがってPauliのorder2×2件がどの価格に対応するかは、今回の保存値だけでは結合できない。
DFのorder2/4未観測をPauliの価格で補完しない。JSONに128標本の価格・seedと欠測joinを明示した。

名目SE/t95%は実現した64 trajectoryの標本分散の診断。未観測orderの大きな費用・母平均・厳密coverageは保証しない。
強制order0/2/4の代数的回路検査は保存されているが、cost MCへの寄与測定ではない。
既存C/N/G・順位・図は変更しない。追加情報取得案は別の未承認契約とし、実行しない。
