# H4/H6 標準Yoshida4の独立費用方向再現

[結果・4群統計・2比較・保存Full/探索・SOURCE・台帳・保全](../../artifacts/resource_applicability/track_a_h46_standard_y4_confirmation_recovery/2026-10-24/REPORT.md)。ε=.001で、保存独立S2 n25/23と新独立標準Y4 n45/32を同じ固定shot政策で比較した。Y4/S2のG比はH4=0.864959895、H6=0.862762409。両系で点方向を再現し、2比較familyのBonferroni正規近似区間も差の正側だった。

保存FullはS2/q8が低費用、独立PRはY4/q4が低費用。異なるqの固定構成の方向逆転を支持する。次数単独の因果効果、母平均順位・希少tail・厳密bias・全誤差の保証ではない。探索標本とはpoolしていない。

利用者の修正・再実行承認に基づき、runtime参照を修正し、実fork initializerを通す人工検査を追加した。新SOURCE a2a53698…で77 trajectory/154軸、exact2953 entry・会計9 eventを正常終了。25人工件PASS、全process exit0/reaped、569 sealed fileと旧停止64 fileを照合した。旧grant/STOP保持、旧strict BLOCKED・mandatory STOP。
