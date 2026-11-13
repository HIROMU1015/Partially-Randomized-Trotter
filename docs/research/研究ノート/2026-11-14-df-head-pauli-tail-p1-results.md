# 2026-11-14：DF head＋Pauli tailの限定P1

ユーザーがP0結果4220f3006を確認し、75fb0ba1 SCIENCE_PLAN P1のLD1/LD2二時点・DF固定R/full256対照を明示承認した。A既存branchを継続使用し、SOURCE05447d1d4でfresh run01を一回実行した。[REPORT](../df_head_pauli_tail_p1_results_20261114/REPORT.md)。

保存H4 DF10/Linear1Å/STO3G、S2/Δ.1/K2/h.1/M12/元exactκ、同H/state/target/scalarを維持。P0行列・辞書を再利用し、新Pauli commonκのRはLD1 m0/m6=3/171、LD2=2/66となった。4件にLD1/m0のfixedR2と同条件full256を加え、最大6mean/18complex・normalizer5uniqueが全局所数値gate通過。旧DF4rowはSOURCE/gate受入だけ、R不一致の4組でsignal差を計算しなかった。

B最新公開ce0517deのcanonical APIで辞書付き新IDを生成し、旧DF IDは流用していない。fixed-Rをκ scheduleへ混在させず、B mean packetは各m0/m6だけ取得済み・他11roundはnullとした。DF同Rのcorrected差2.7756e−17、full/36最大mean差3.4332e−16で、辞書依存のraw/Bは区別した。

人工6＋既存17＝23 PASS。新reference/DF mean/P0再計算/eigh/サンプリング/build/compile/RPE/N/MSE/C/Gは0。旧SOURCE/raw/KEEP/STOPや他担当環境は変更していない。十分N/global RMSE/winnerは未資格/null。公開・remote照合後mandatory STOPし、次の不足round・担当別評価はユーザーの別承認を待つ。
