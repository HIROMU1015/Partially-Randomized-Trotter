# 2026-10-19 PR/PF絶対コスト図（保存値表示のみ）

利用者はv2を基準とし、5精度のうち描画可能な既存比較を表示するよう指示した。[説明と採否](../track_a_pr_pf_absolute_cost_v2_20261019.md)。

H4/H6/H8独立Pの6比較と、H2/H4/H6/H8の新A n8の20比較を採用。両端は独立P/探索の別パネル、中間3精度は探索のみ。H4新DF10を使い、legacy DF12、旧n32、FPの標本は混合しない。H6中間3精度のPFは検証済みmodelとして白抜き表示。H2はA完了の探索のみで、P未完了を維持する。

PNG/SVG/PDF各5枚、exactデータ表、採否表、caption、固定Git objectのSHAと照合scriptを専用bundleに保存。G/SE/shot/順位を再計算せず、分数の描画用変換だけを行う。v2のDRAFT状態、旧論文主表/CEM/原図、SOURCE/結果/ledger/marker/STOP、母平均未保証、H8公開監査の限界を保持する。

今回の許可範囲は表示とlocal commitまで。公開push、新しい科学stage、正式採択、freeze、詳細科学レビューへ進まない。次の作業なし。local commit後 **mandatory STOP**。
