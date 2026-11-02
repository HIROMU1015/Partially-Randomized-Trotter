# 固定証拠・科学条件・論文との境界

[機械可読索引](audit/source_catalog.json)のIDは固定commit/path/SHA/URLを指す。保存値を引用し、新しい費用・比・SE・CI・順位を算出していない。

|ID|固定commit|使う保存証拠と役割|
|---|---|---|
|ADOPTION/CEM/LEDGER/PAPER|09e71910ee49e0d9d3a2f2f410249f6623022c0d|条件付き独立費用方向の採用正本。CE_H4_STANDARD_Y4_FIXED_SHOT_DIRECTION、CE_H6_STANDARD_Y4_FIXED_SHOT_DIRECTIONへ参照|
|ORIGINAL/COMPARISONS/CONTRACT/END|4d117041469fb8eebf2ad55405c0f7d1b70dea7d|H4/H6独立4群・2比較・契約・終了監査。成功recoveryと旧停止履歴の保全は原監査を引用|
|PRIMARY/PRIMARY_COMPARISONS/SAME_Q/SAME_Q_COMPARISONS|eebdbfcbd0d5f5608d50bfd91c9cd526c68f0d23|精度別15方向/30比較、同一q4方向/8比較。前者は探索点、後者は補助対照|
|CNG/CNG_CSV|d2536c4f959bfb4edd7d2deeceb155585fb53eb0|保存C/N/G分解22比較。新独立Y4群の機構分解へ移送しない|
|COMMON_PF|f75a7c25d584b69d90de454731a7f343a421e059|共通PF588候補の既存採用索引。全PR/PFの普遍的最適性ではない|
|THREE_ENV|6dafed86cc82149d55b4e6395382953d5f3631d7|3環境の参照体系。別環境による今回結果の独立再現とは数えない|
|DISPLAY|1923fce8f7461513d1bc47b5061e87cd70917e87|固定主表・Figure2の基準。主表SHA 2d910a62451d6a0b3247751625e85c8aebd181b4dea6b493b94a469d4e49d86c|
|OLD_MATRIX/OLD_REPORT/OLD_CATALOG|501b4232b3e9ef9778e917f0a86f1d6c1d05f189|旧16主張・39文献の調査。旧判断は変更しない|

## 今回の高次系列

linear H2/H4/H6、隣接1Å、STO-3G、tol-only DF1e-8、rank3/10/19、T=.8、α_axis=.025、P=0、usual signed、
保存reference/signal・empirical allowance・analytic shot政策、symmetric-directional full cosine/sine Hadamard wrapper（状態準備除外）、Qiskit1.3.0/rz,sx,x,cx/opt1/seed17のcompiled RZ。

|系|Full prefix|PR prefix|S2→標準Y4のq|探索n S2/Y4|今回採用された独立n S2/Y4|
|---|---:|---:|---|---|---|
|H2|3|1|4→2|33/34|本高次比較の独立方向確認は採用なし|
|H4|10|3|8→4|48/33|25/45、ε=.001のみ|
|H6|19|5|8→4|33/33|23/32、ε=.001のみ|

PRはK4/ABS_TIME_PER_OUTER32。標準Y4は既存3 signed piece構成、K4はfinite Taylor cutoffであってPF次数ではない。
旧m5_best確認・P33の新4次formulaを標準Y4へ読み替えない。M5_MIN_*という候補IDの接頭辞だけでformulaを判断しない。
旧H6 K2/S2のε=.05/.0005独立採用とも別scope。

保存方向：H2は登録5精度でFull S2/PR Y4、H4はε=.005（弱い点順位）/.001、H6は.001で同方向差。
H4/H6は.0005ではFull/PRともY4。登録点の間の交差精度・漸近則は未推定。
同一q対照はε=.001のH4/H6に限り、q4でFull/PR双方Y4、q8で双方S2、方向逆転なし。

H4/H6独立結果は保存G比0.8649598948712324/0.8627624094690314、REPRODUCED_APPROXIMATE。
familyは2比較、α=.05、Bonferroni正規近似、母平均順位NOT_CERTIFIED。77は新Y4取得総数であり、family数や合算推定量ではない。
cohortをpoolせず、同じ標本を複数ε・索引で参照しても独立証拠は増えない。

H2は旧strict identity残条件を保つ探索的引用であり、旧strict COMPAREの解除ではない。
既存H6 Fullの旧親終了不明などは元監査の制約を保持し、新独立Y4の正常終了証拠で他runを遡及的に認定しない。

## 既存原稿と別成果

既存v0.4の中心候補は「scalar保持対照・実装単価・独立確認を通じたPRの適用条件」。
H4 legacy DF12の6距離/ordinary controlが中心で、今回の1Å/tol-only DF10/19・symmetric-directional高次系列とは異なる。
後続H468掲載packetは独立6条件主表、Figure2A legacy/探索・2B H8 n48という表示固定。全面統一済み原稿ではなく、複数時点のdraftと採用索引を区別する。

P33は決定論高次PFのenergy/指数演算数・RZ-layer depth研究。
本研究はその係数探索・新4次法・決定論数値を新規成果として再計上しない。
案A/Bを同時に採る場合も、同一高次PRデータに主成果としての掲載先を一つ割り当て、もう一方から参照する必要がある（未決定）。
