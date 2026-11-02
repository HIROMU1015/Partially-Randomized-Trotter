# 2026-10-23 非DF監視基盤・保存診断・H4準備の技術記録

今回の利用者承認は堅牢化と人工/保存値検証、未承認H4契約案の準備に限る。科学実行の承認ではない。
旧H2のSOURCEを不変にするため、新supervisorを追加し、凍結済みlauncherを置換しなかった。
ready/終了監視を契約ごとに設定できる形にし、timeout既定None、専用親processのpidfd/subreaperで自分の子孫だけを管理する。
開発初回の人工deadline先行2件を保全し、意図した状態を観測する試験へ修正した。最終370 PASS、分子処理除外5・runtime skip1。
H2のevent-to-cost欠測はseedを再生せず欠測のままとした。DF全位置order0という集計からの推定は区別する。
H4のprefixや256標本は提案で、結果を見た選択・新統計保証の採択ではない。実rank/項数/L_Dは未生成で、固定規則を将来入力sealへ結合する案。
一括承認文案は実行権限を与えない。H4科学計算、希少event追加sampling、旧失敗retry、詳細科学レビュー、pushは実施しない。

[技術結果・SOURCE・保全・停止](../../non_df_h4_readiness_validation.md)。本記録は科学詳細レビューではない。
