# Track D：r16限定検証の記録（2026-11-10）

Linear H4／1.00Å／STO-3G／Nα=Nβ2・元H ground仮定/準備費除外、number-CDF8／元順LD2／S2/s1/r16/K0／h=.1。S1は全13roundで条件付き十分条件PASS（q4096 N=1359/軸、保存r32=910）。固定q4/cut2/C1 OFFの新1path・64event、raw/4P0/post/pairedχ PASS。Strong→ReferenceはCX5434→4874、depth7495→3568、RZ13042→13333、幅9→17。law母平均・全RPE総費用/actual成功保証は未評価。既存SOURCE/原本/STOP保全、公開照合後mandatory STOP。

r変更だけを対象とする利用者指示により、既存B1/B4の元H誤差・shot APIとC3/C6aのwhole-path/compile checkerを再利用した。PF/DFはr非依存としてSOURCEを照合し、RTE law・attenuation・十分Nをr16で新評価。別条件r16/s4・own-DF信号/Nは転用しなかった。

S1 PASSを受けて一つだけq4 pathを生成。必要Nが増える一方で単一pathのCX/depth削減は残ったため、総費用優位を採択せず、r16期待価格校正の要否をユーザー／GPTへ戻す。初期metadata不具合は科学開始前で、初回記録を保全してSymEngine表記だけ修正。科学的retry・条件変更なし。

詳細：[track_d_r16_limited_validation_20261110/REPORT](../track_d_r16_limited_validation_20261110/REPORT.md)。以降のsampling・価格model・全round総費用計算は未承認・未実施。
