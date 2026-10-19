# 図caption案

## H4絶対費用（allowance倍率1）

H4固定4候補における要求信号精度とshot込みRZ総費用。PF2候補とPR2候補を分け、S4修正後のNATIVE_MIXED理論期待費用とcompiled保存費用を比較した。PRは保存anchor n32。不適格点は補完しない。

## H6絶対費用（allowance倍率1）

H6固定3候補のshot込みRZ総費用。PF1候補とPR2候補について、S4修正後のNATIVE_MIXED理論期待費用とcompiled保存費用を5精度で比較した。PRは探索n8で、独立確認ではない。

## H4費用比（allowance倍率1）

H4固定4pairのPR/PF総RZ費用比。1未満はPR、1超はPFが低費用。PR p3/q8/r2/K4対PF p12/q1/S4のε=.05では、保存比が理論約0.872・compiled約1.039で優劣方向が異なる。有限標本・保存shotに限定した観測。

## H6費用比（allowance倍率1）

H6固定PR2候補とfull PF p19/q8/S4の総RZ費用比。NATIVE_MIXED理論とcompiled保存比を直接表示した。表示された条件では両方がPR低費用の方向だが、panel外の最適性・母平均順位を保証しない。

## H4/H6・allowance倍率10の4枚

各倍率1のcaptionに「allowance倍率10の保存条件による」を付ける。倍率1からshotを再計算した値ではなく、元から保存されている倍率10の総費用・比を描画した。倍率の異なる条件をpoolingしない。

## 全図共通の脚注

線形1 Å、STO-3G、T=.8、P=0、cos/sin full Hadamard wrapper。H4はlegacy固定DF12、H6はtol-only DF19。保存解析shot×RZの総費用で、状態準備やFTQC全体費用を含まない。

理論lower/upperは宣言モデルの保存期待費用端点で、CIではない。compiled PRは有限標本平均を用いた保存費用。母平均・rare次数の全support費用・旧compiler binaryの完全再現は未保証。横軸の5精度はカテゴリ、線は視認性補助。markerを同じεの左右へわずかにずらして表示した。
