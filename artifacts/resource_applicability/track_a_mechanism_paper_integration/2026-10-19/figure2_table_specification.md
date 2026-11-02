# Figure 2・主表のデータと表示仕様

2026-10-19 JST。`SAVED_DATA_READY_LAYOUT_SYNTHETIC_TESTED`。実データ図の完成描画は未実施。
元契約0a8ac1d2d03c5a9aef2acab0295226f8c88b38ceのFigure 2を、承認済みcore結果2bf5d83293b3a47f672805fae956ff2e50df2878と説明結果0edd62403c35c5ee18b2a2a152158faca5c2ebd8へ接続する追補案。
元契約・元結果・旧statusを上書きしない。[manifest](data_manifest.json)は32比較を保存順に固定する。

## 共通条件・表示の単位

線形H4/H6/H8、隣接1 Å、STO-3G、T=.8、P=0、通常signed、allowance倍率1。
H4 legacy DF12／元PF subset16、H6 tol-only DF19／PF136、H8 tol-only DF28／PF184。
候補IDにprefix・q・outer次数・r・Kを保持する。δ=T/qは説明用の定義だけで、新しいδ値を計算しない。
Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、backend/couplingなしのfull Hadamard wrapper RZ費用。
H8原profile bytes／旧compiler出力完全同一性等の旧限界を継承する。

新A探索n8、H4旧fixed-anchor n32、H6/H8旧confirmation union n32、H8独立固定n48を別系列にする。
同じ候補・標本を繰り返す精度点は独立標本数ではない。新旧から低い方を選ぶ線やpooled winnerを作らない。

## Figure 2A：系別3列×費用因子3段

列はH4/H6/H8。元表示規則どおり各系のε=.05/.005/.0005×旧PR／新Aの6点、計18比較を主panelとする。
残る12比較もdata/main_table.csvと全軸表へ保持し、削減率の大小で抜粋しない。
上段R_C,axis、中段R_N,axis、下段R_G。横位置は保存された精度・cohort区分のカテゴリ順とし、
ε間の補間や順位による並べ替えはしない。比1の水平線、比の対数軸、cosine丸／sine四角。
下段totalは丸。色と文字ラベルで旧PR／新Aを分け、直接PFは塗り、model PFは中抜きにする。

H6はε=.05と.0005がDIRECT_MEASURED、.01/.005/.001がVALIDATED_MODEL_ESTIMATE。
旧PR／新Aの両行に同じ保存ラベルを表示する。直接PFを使う点も全136候補の全実測最良を意味しない。
H4 panelには「legacy DF12; historical PRE_D_P core; D completed, P pending」を表示し、H4を補足へ格下げしない。
A82のcore点は存在する。説明因子欄だけを「原入力回収済み・数値検証未承認」とし、説明欠測を0で描かない。

## Figure 2B：H8独立n48の固定2例

探索n8とは別blockにH8_CONFIRM_1（ε=.05、B2_p8_q2_2nd_r4_K4対B0I_p11_q1_2nd）と
H8_CONFIRM_3（ε=.0005、B2_p7_q16_2nd_r4_K4対B0I_p15_q4_4th）を配置する。
各例の上段に保存R_N／R_Ceff／R_G、下段に軸別の保存nominal normalization²・margin exp比・headroom²・
alpha-log・rounding factorを並べる。continuous_shotとnormalization_upper_squaredは補足表に保持する。
全31比較の説明因子はdata/explanation_factors.csvの62軸行を使う。
高次event未観測、母平均優位未確定をcaptionへ記す。固定2対照はPF184全実測確認ではない。

## exact coreと説明因子の意味

保存exact式はR_G=R_N×R_Ceff、およびR_G=Σ_axis w_PF R_N,axis R_C,axis。
今回この積・和を再評価しない。core_axes.csvのweighted_termとw_PFも保存出力の転記である。
Ceffは各方法自身のshot構成で重み付けされており、純粋なcompiler効果の因果推定ではない。

説明因子の意味はexp[2(ell_nom,PR−ell_nom,PF)]、exp[2(du_PR−du_PF)]、(h_PF/h_PR)²、
log(2/alpha_PR)/log(2/alpha_PF)、kappa_PR/kappa_PF。式を説明するだけで今回評価しない。
margin exp比はduそのものの二乗ではない。upper normalization²とnominal×marginは代替表示で、
冗長に掛け合わせない。説明因子はbinary64/log診断、core費用恒等式は保存exact有理数という精度区分を保つ。
empirical allowanceと理想fresh-IID測定仮定をrigorous certificateへ格上げしない。

## 主表・補足表

data/main_table.csvは32比較全件。G_PR/G_PF、R_N、R_Ceff、R_G、保存削減率exact、保存SE、候補・n・cohort・
直接/model・説明status・元H4_stage・新D provenanceを別列に置く。保存SEはPR費用標本の転記のみで、
費用比のSE／差のSE／CI／p値やmodel不確かさへ変換しない。
data/core_axes.csvは64行の軸分解、data/explanation_factors.csvは62行の説明因子とlog因子。
H4_D_saved_endpoints.csvはD結果の保存10 winnerを転記する別表で、新しい最小化は行わない。

データの文字列表現は元CSV／canonical出力と同じ。実描画時の有理数→座標変換・表示丸めは将来の表示処理として明示し、
元exact列を保持する。新しい比・差・統計量・rankingの列を作らない。
現在のpreview CLIは人工fixture専用でreal-data入力経路を持たない。

## Caption案

「保存DF表現・有限時間・状態準備を除くHadamard wrapper RZに対する、固定候補の費用点比較。
shot総数と有効回路単価のexact分解は保存32比較から転記した。31比較の説明因子は通常signed・factor1の
empirical allowanceに条件付く。旧確認unionの選択後点推定、探索n8、独立固定n48は別系列。
H4 Dでは保存PF代表が10/10条件で不変だったが、元PRE_D_P coreを改訂せず、独立確認Pは未完了。
A82説明は元結果の欠測を保持し、原signal回収後の数値検証は未承認。
model PFを中抜きで示す。母平均優位性、厳密誤差保証、FT資源削減を意味しない。」
