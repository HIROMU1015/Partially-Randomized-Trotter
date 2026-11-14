# G36223：修正版への保存結果引継ぎ

ユーザーが「修正版に切り替えた方が早そうなら切り替えてもよい」と指示した。旧SOURCE887のcheckpoint wrapperでは内部fingerprintが旧writerへ戻る問題を人工で再現済みで、dfab06ee0の修正後は同じcanonical bytesのfast writerを全wrapper内部検査にも適用する。旧残検査は2時間以上続いているため、修正版で残検査を行う方針とした。実対象での速度改善率はまだ未測定。

## 固定入力・変更点

[FROZEN_COMPLETION.json](FROZEN_COMPLETION.json)はD7 G36223の同plan・physical ID・cohort・seed・compiler/profileと、d323fee1cのpath0 native checkpoint、887c3b9d0の完了path1 receipt/個別compile hashを固定する。path0のsampling/compile/既存7数値gateを再実行しない。path1は元receiptを検査付きで読み、全検査・価格・event列を再利用する。**追加sampling0・追加compile0、D7全体18 unique compileを維持する。**

既存P2のexecutor・record producer、SOURCE/input checks、phase/parity/XYを再利用する。childではsampler/compiler入口を禁止する。CODE変更はcheckpoint wrapperのfast writer適用、段階別progress callback、完了path再利用、保存済み入力からの制御付き引継ぎだけ。科学条件・閾値・Qiskit1.3.0/rz/sx/x/cx/opt1/seed17/P0/new ZZ/scalar once/測定wrapperは変更しない。

## 検証と移管条件

[TESTS.txt](TESTS.txt)：11人工test PASS。通常X/Y process並列、小toyの実transpile/QPY/7gate、checkpointの旧writer禁止・sampler/compile0・元event/fingerprint一致、完了path再利用の同一性・出所・hash確認、段階log、稼働worker/科学失敗からの再試行拒否。初回はtest fixtureのJSON tuple/list比較と排他的ファイル書込みで失敗し、fixtureだけを直して再確認した。H4追加試運転0。D1本人のlocal検査であり独立レビューではない。

旧managerとpath0 workerのUID/開始時刻/commandを照合し、保存path0 native・完了path1・旧jobのhashとgateを検査する。旧path0又は全体が先に完了した場合は切替えず既存結果を使う。未完了ならmanager→workerを一時停止して境界を再確認し、worker→managerの順で終了する。元のterminal STOP_CELL/exit2は計画的な切替として保全し、科学gate FAILと混同しない。旧SOURCE/raw/QPYを変更しない。

新SOURCEは別worktreeで固定し、一回だけ起動する。privateのPROGRESS.jsonlに段階開始・直前段階完了・秒数を記録し、STACKS.txtに120秒ごとにworker内部stackを記録する。観測のためにroot/sudo・OS設定・他processを変更しない。

切替の実績はrun01/TRANSFER*.json、実行job、console/time、最終G36223 receiptに記録する。現時点はSOURCE固定の準備段階。重複確認は既存df0e11fd1とDELTA_PREFLIGHT、今回の旧job・path1保存結果照合を継承し、受入可能な今回自身の4 compileを再取得から除く。

完了後は14軸の公開済み証拠へG36223の4軸を加え、Cへ固定locator付きで引き渡してmandatory STOP。C科学受入・fit/凍結、m5/m6 HOLDOUT、長時間外挿、law E[C]/N/Gは行わない。ロードマップ本体は編集しない。
