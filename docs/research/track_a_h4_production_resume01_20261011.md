# H4 run10：高速化版へ完了分を引き継いで再開

利用者の「引き継いで」によりown run10 driverだけをpidfd SIGINTで停止、native6の残存0を2秒離した2回の監査で確認した。
[認可・実receipt要約・binding・起動資料](../../artifacts/resource_applicability/track_a_h4_production_resume01/2026-10-11/README.md)を一つの入口とする。

実行SOURCE `941749b8d4aedb43fd5f47b824116082493df393`、dataset/seed SOURCE `5aa6c3685afb9c3afcfbdc36f80244da0a7fe525`。
準備SOURCE11bba8fのconsumerが既存sealed `worker-log-first-stop.txt`を想定していなかったため、その既知名だけ許可し19限定mock PASSで新SOURCEへ固定。数値/科学/compile optionsは変更していない。
旧COMPLETE8150件・signal142件を全file bytes/SHA・ledger chain・seed/identity・外部completion digestで照合。実checkpoint/raw native proofはhome内private evidenceに保持しcommitしない。
未完了予約2件は再利用せず、費用を保持して再計算する。
carry actual7878 /4320481456 bytes /35474.12562095933秒、保証上限74784+7878-8150=74512≤74805、累積保存見積り13057453088 bytes（約12.16GiB）≤17GiB。
17GiB/74805/72h、worker4/32GiB、driver8GiB、observer AS256MiB/RSS64MiB、CPU2/4/5/6・driver16・observer18、pressure1/5/30/152.25、内部thread1と監視5秒はrun10から保持する。
72hはrun10を含む累積wallで、resetや失敗費用返却は行わない。回路完成単位を再開点とし、元directoryと原SOURCEをread-onlyに保持する。
新SOURCEはexecution provenanceへ別記録し、同じdatasetのseed/wrapper keyを保持して完了wrapperのbuild/compileを省略する。

SOURCE/profile/input/carry/実receiptと認可を結合し、新規独立output/controlとone-shot lockへsealする。
最終reviewは新2file・実receipt・最終bindingだけに限定、既存18回帰・Gaussian数学・旧benchmarkのcampaign反復0、追加人工compile0。
immutable artifact固定後にfresh CPU/memory/pressure/OOM/filesystem/inode/quotaと未使用rootsを検査し、全合格なら一度起動する。
実状態・PID・起動時刻はprivate STARTED receiptで別に記録する。MAP_COMPLETEまたはfail-closed STOP後は終了、次stageは認可しない。
PR研究の公開a87cf255/AX2bとはscopeを分け、H4 six-geometry追加mapの継続とする。accuracy/u/NG certification、H6認可の結論は出さない。

最終独立binding review：`PASS_READY_FOR_FRESH_ONE_SHOT_CONTINUATION`、blocking指摘0。実receipt/native/ledger/carryと新2file差分だけを確認し、旧campaignは再実行しない。
