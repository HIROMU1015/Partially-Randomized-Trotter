# 2026-11-13 二時点RMSE pilotの固定条件選択案

条件決定支援として記号的κ整合を整理した。H4は(2,128)/(4,256)、H6は(11,16)/(11,128)/(16,256)が元mapper上で両立する。実λの代入・κ区間の数値評価や探索はしていない。初回の構成の単純さからH4/finite_Bと保存再解析限定S0を推奨するが、対象・M/κ・RMSE予算・合成則は未採択。

MはPR経験的horizonと指定QPE予算に従う決定式、κは保存再利用帯内の中点式を提案した。予算は既存chemical-accuracy規模を新RMSE候補として区別し、Minkowski/equal-half案とpaper quadrature案の根拠・未資格を記載した。十分N・全RMSE・Gを保証していない。

[REPORT](../pr_rpe_rmse_pilot_decision_20261113/REPORT.md)・実行計画・ユーザー決定事項を追加。既存code/test/template・SOURCE/raw/STOP不変、人工/科学数値評価なし。公開照合後mandatory STOP。次はユーザーが一組の条件と有限科学scopeを明示採択・実行承認した場合だけAが進める。
