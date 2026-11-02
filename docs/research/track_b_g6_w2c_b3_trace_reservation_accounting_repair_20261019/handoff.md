# Q1–Q9と次担当

**Q1.** 固定旧SOURCEと外部原scriptで再現した。planned768、raw予約1、report reserved0、started0。固定独立checkerがGLOBAL_STATE_COUNTER_BINDINGを検出した。

**Q2.** 原因はstate更新とRAM会計の間のfallible trace。新versionではRAM counter/cursor/ticketを先に確立し、保存notificationが途中更新を中断しない。state保存自体が不明ならRAM intentとphysical evidenceを明示してSTOPし、非返却を維持する。

**Q3.** 同じwriter故障のS/P1はreserved1、started0、開始前取消1、completefalse。独立共通accountingは一致。trace欠損はINCOMPLETE_EVIDENCE。V2のfull adapter受入は固定V1 checkerの対象外でありローカルCodex待ち。

**Q4.** S/P1の108保存ケースと新41unitで、全trial状態共通経路、trace/ACK/phase、partial/flush/hash、state/journal、pair、summary/receipt/diagnostic、admission STOP、close、親/worker喪失、capを検証した。故障状態を不正な成功へ変換していない。

**Q5.** S/P1、元のcoalesced worker/F01/observerを保持。P1 drainは8開始・1FAIL・7COMPLETE。UNKNOWN/未ACK credit非返却、retry0/resumefalse、同一store one-shot/raceを確認した。

**Q6.** 今回の既存779＋新41＝820 tests PASS、最終FAIL/ERROR/SKIP0。別枠108 matrixは正常2・技術失敗40・証拠不完全66として期待分類を確認。初期試行のerrorは履歴として保持。最終人工受入に未解決の失敗はないが、物理保存不能の証拠欠損とV2の最終独立受入は残る。

**Q7.** 新V2 schemaはauthoritative RAM stateと不確かなphysical witness、trusted prefix、保存先別STOP成否、diagnostic/marker順序を明示する。byte creditは一度だけ課金し未ACK分を返さない。健康なtrial粒度/canonical/F01と科学条件を保持。将来正式SOURCEへ採る場合に技術契約差分の判断が必要で、今回は採択しない。

**Q8.** ローカルCodexへ最終固定公開HEAD、検証SOURCE45bf46b4、before原再現、108 raw/spec、41unit、固定checker原bytes、追加checker/比較結果を渡す。V2独立adapter/信頼境界、保存不能時RAM claimの証拠限界、歴史pendingが未決。production bridgeは未実装。

**Q9.** 実G7/F3 sampling/import、新科学計算/保存科学値再解析、科学SOURCE採択、実authorization/grant/claim発行・消費、quantum build/compile/synthesisは0。既存SOURCE/契約/raw/grant/ledger/marker/STOPと他worktreeは保持した。

次は**ローカルCodex**が新SOURCE/fixtureの最終独立受入を行い、**ユーザーとGPT**が技術契約差分と次作業を判断する。今回のサーバーCodexは公開・読み戻し照合後mandatory STOP。production bridge、科学release、pilot/Full-Nへ自動移行しない。
