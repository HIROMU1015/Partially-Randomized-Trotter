# H4 DF10 m3 RZ費用の固定実行範囲

2026-11-10 JST。指定3ケースの数値検証はユーザーの明示許可済み。305439c01のaccount_subsetとPREPARED_INPUTS、固定D1 resultを再利用する。新しい会計器やN solverを作らない。既存branchを継続し、SOURCEをcommitしてから一括実行する。

対象はT=.8/m3のPF B0/S2/p7/qPF8（705/軸）、PR S2/p1/qPF8/R4/K2（977/軸）の探索n9と確認n33。各ケースaccount_subsetをRZだけで1回呼ぶ。X/Yの積和を保存binary64単価のexact-rational算術で独立照合し、元N/C・SOURCE/cohortを保存する。新しい平均やN/Cの再推定ではない。

h=.1/全13round、εE=.00159360010199040 Ha、αtotal=.05/方式、β=.4/.2、H4/1Å/STO-3G/保存DF10、保存数値ground条件付き/P=0を維持。m3以外12roundはmissing、full_rpe_resourcesと未知covariance/CI/actual Nはnull。n/Bを価格へ再乗算しない。比率・新順位・感度・残差・新N/prior/horizon/signal/trajectory/build/compile/fit/bootstrap/追加round/H6/全G/詳細科学レビューは実施しない。

新runnerは固定保存入力読取と既存会計呼出しだけ。既存14人工・回帰testは保存PASS記録を再利用する。旧SOURCE/入力/raw/result/grant/ledger/marker/KEEP/STOP、他担当branch/worktreeを変更しない。新しいapproval証明機構を作らない。新しい独立結果と簡潔なREPORTを公開し、remote SHA/tree/成果物照合後mandatory STOP。
