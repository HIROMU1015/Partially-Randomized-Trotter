完了追記：探索512本・確認256本、計768本の科学計算と保存監査は完了。現在の正本はREPORT.md、EXECUTION_SUMMARY.json、STUDY_STOP.json。以下は計算中に保存した引継ぎsnapshotで、未完了記述は当時の状態。

# H6 Hybrid Phase B — 計算継続中の引継ぎ

固定4候補のみ。科学SOURCE887b69e449cb3c378286f09d0d7626599850fb67、解析SOURCEaf891fd45805454c6592bbee5441266f14235ffd。旧SOURCE・STOPは無変更。実行branchはcodex/h6-hybrid-fairness-extension-execution-20261027。

4 law/4 meanは成功済み。p78の3候補は全5精度適格、p684はε=.05だけ適格。新Hamiltonian/state/reference生成0。sampling_design.jsonを探索前に固定し、n_E=128×4候補、n_C規則・範囲64–1024も固定。実行中探索を再launchしない。

探索親、継続coordinator、報告書rendererが起動済み。coordinatorは探索native.jsonの正常終了・全回収を待ち、nominate→固定n_Cのconfirmation→finish→verify→packを一度だけ実行する。rendererはarchive_restore_verificationのPASSを待ち、REPORT/READMEを保存する。確認結果によるtop-upは行わない。科学STOPは完了時のSTUDY_STOP.jsonで記録される。

最初の探索投入時は実測空きCPUが約1coreだったため旧dispatchの低優先度1workerで進行。正常な科学workerの停止/再起動なし。待機だけのcoordinatorを確認開始前に置き換え、確認投入は同じprice workerのまま空きCPUへ追随させた。6ダミーjobで件数/回収を確認。

次のCodexは保存件数・stage native.json・confirmation_plan・STUDY_STOP・verification・archive_restore_verification・REPORTの順に確認し、実行中なら再起動せず待つ。失敗科学callがあれば保全して判断を戻し、無断retryしない。科学計算・結果保存完了後にGitHub公開、local/remote SHA照合を完了する。今回の一括承認はその範囲まで有効で、通常段階の再承認は不要。

未完了：残り探索、独立確認、最終比較、保存監査、最終結果のGitHub公開。新候補/H2/H4/純RTE追加探索/詳細レビューは承認範囲外。旧14/15は元保存集合の結果として維持する。
