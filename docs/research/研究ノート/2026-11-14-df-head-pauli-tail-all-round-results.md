# 2026-11-14：Pauli LD1/LD2の全13round signal完成

ユーザーがP1 e8641b84の未評価22件を明示承認した。A既存branchを継続使用し、SOURCE7084fbc9a/fresh run01でm1/2/3/4/5/7〜12×LD1/LD2を一回実行。[REPORT](../df_head_pauli_tail_all_round_results_20261114/REPORT.md)。全22局所数値gate通過、新66complex/normalizer22組。旧m0/m6の4件は再計算せずSOURCE付きで統合し全26row packetを完成した。

保存H4 DF10/Linear1Å/STO3G/同H/state、S2/Δ.1/K2/h.1/M12/元exactκ、P0辞書/scalar、P1 canonical ID/全scheduleを維持。数値providerはbyte不変。新rawはfce4e4f7eへ固定、旧4件はP1/数値05447d1d4、新22件はraw carrier/数値7084fbc9aへ個別接続した。

最大closure4.3729e−10は既定1e−9 gate内。m12 attenuationはLD1約.95969186/LD2約.95969517、signal bias約.05486044/.05473199でenergy RMSEへ変換しない。長時間allowance/normalizer感度はempiricalのまま保持。

新人工6＋P1回帰6 PASS。新reference/full256 mean/P0再計算/DF mean/eigh/sampling/compile/N/MSE/C/G0。旧SOURCE/raw/KEEP/STOP・他担当環境を保全。文書作成中に利用上限でtool実行が中断したが科学計算は再実行せず、ユーザーの継続指示後に未完了文書・公開確認を再開した。次はB受入とユーザーの次科学scope判断。Aは公開・remote照合後mandatory STOP。
