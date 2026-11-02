# Figure 2：実データ版の仕様・caption

描画SOURCE `3de17acd912a28a1df9905b2f3c3e2e3e15aa0dc`。前回のpreview_mechanism_figure2.py、mechanism_paper.py、保存4 CSVを再利用した。
保存値転記・座標変換は承認範囲内。A82 science SOURCEとは役割を区別する。

## Figure 2A

H4/H6/H8の3列、上から軸別単価比R_C,a、軸別shot比R_N,a、総費用比R_G。
元規則のε=.05/.005/.0005×旧PR／新Aの18比較、90点。残12比較は主表32行・全64軸に保持する。
青は旧n32、橙は新A n8。cosine丸／sine四角。PF直接実測は塗り、modelは中抜き。
H6は.05/.0005のPFが直接、.01/.005/.001がmodel（主図の中間.005は中抜き）。
ε間を補間せずカテゴリ表示、比1の参照線と対数y軸。同じ行は3系で共通y尺度にする。

Caption案：「保存DF・有限時間signalの固定候補比較。linear1 Å/STO-3G、T=.8/P=0、通常signed/factor1、
Qiskit1.3.0/rz,sx,x,cx/opt1/seed17/no topologyのHadamard wrapper RZ。
H4はlegacy DF12、旧n32はfixed-anchor選択・推定、H6/H8旧n32は選択後confirmation union。
H4 Dと独立Pは別途完了したが、図のH4費用は元core32のn8/n32を保持し、新P標本は混合していない。
A82はcore点が存在し、説明側だけが保存結果で欠測。回収入力の数値検証は未承認。
modelと有限標本・経験的allowanceの条件を保持し、母平均優位を意味しない。」

## Figure 2B

H8独立固定n48の2比較を別図にする。ε=.05：B2_p8_q2_2nd_r4_K4対B0I_p11_q1_2nd。
ε=.0005：B2_p7_q16_2nd_r4_K4対B0I_p15_q4_4th。
上段はR_N、R_Ceff、R_G、下段は各軸のnominal norm²・empirical margin exp比・headroom²・alpha-log・丸め因子。
計26点。1に近いmargin／丸め因子は13有効桁の表示文字列も併記し、exact元文字列はplot_values.csvへ残す。

Caption案：「独立固定n48の保存値による説明。nominal normalizationとempirical log marginは別因子、
headroom因子は(h_PF/h_PR)²、丸めはkappa_PR/kappa_PF。marginはlog allowanceそのものの二乗ではない。
費用のexact恒等式とbinary64/log説明の精度を区別する。Ceffは各方法のshot構成に依存し、compilerの因果寄与ではない。
探索n8とpoolせず、高次event未観測・母平均優位未確定を保持する。」

両図とも保存exact fraction／decimal tokenからfloat座標と表示文字列へ変換しただけで、
新しい比・積・平均・SE・統計・順位・科学的分解を計算していない。PNG/SVG/PDFを保存した。
描画の目盛・layoutの算術は可視化処理で、科学会計ではない。
