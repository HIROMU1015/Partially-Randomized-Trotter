# 2026-10-11 Track A H6精度一致比較 v1：一回起動・資源設定STOP

利用者の公開12183db固定契約に対する認可を新grant公開cf52918で記録し、remote258 pathを照合後、source057ba97のrunnerを一回起動した。signal workerがcoordinatorのRLIMIT_AS hard2 GiBを継承し、固定8 GiBへ設定できず、数値port import前に失敗した。親terminalはH6_MATCHED_RESOURCE_STOP、原worker traceback・10 raw fileを保存した。

[結果・証拠索引](../track_a_h6_matched_accuracy_result_v1.md)と外部事後静的監査・全92候補の欠測一覧を追加する。source211・親入力33・旧pilot142 bytesは一致。原inventoryのsource_input_after_verified=falseとRSS peak0を改変せず、前者は外部照合、後者は実測ではないことを明記する。数値比較結果はなく、旧pilot partial証拠とSTOPを維持する。

一回grantは消費済み、retry/resumeせずmandatory STOP。H6_NOT_AUTHORIZED / DRAFT_NOT_AUTHORIZATION / next_stage_authorized=false、H8・追加探索未認可。修正や再実行は別source/seal・別認可とする。
