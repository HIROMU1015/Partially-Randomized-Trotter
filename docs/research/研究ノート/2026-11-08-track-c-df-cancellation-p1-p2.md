# DF相殺P1→P2：固定近傍と保守的比較

## 17:25 JST — 確認済み・mandatory STOP

基点12e74dd、固定科学入力da8b2c3。許可済みP1→P2を独立namespaceで完了。P1はH6 m17・固定総時間の指定5点で条件付き資格を維持。P2は個別directed位相上界を加算し、H4はFull/Closed PRより低RZ、H6はFullより低RZだがPRより高い。有限cohortの結果で元combined保証は維持、mandatory STOP。

P1新4点は独立expm/Schur/native X/Yと時刻丸め追加会計を実施。P2は個別評価前に157点を固定し、154点がB資格。凍結後の独立照合と全18round decoder・十分N・α予算を確認。人工test13 PASS（3.64s）。最初のkeep呼出しはPLAN未完了で科学call前にFileNotFoundError、log保全後に完成済みPLANで実施した。

[REPORT](../track_c_df_cancellation_validation_20261108/REPORT.md)、[SOURCE・再現](../track_c_df_cancellation_validation_20261108/REPRODUCE.md)。条件は元H/exact ground/理想native、prep除外。連続h・別状態・全pool最適性は未確認。元SOURCE/raw/STOPを変更せず、新しいcacheは非公開。次：ユーザーが採否・後続検証を判断。Codexは資料公開・byte照合後STOPし追加科学計算を開始しない。
