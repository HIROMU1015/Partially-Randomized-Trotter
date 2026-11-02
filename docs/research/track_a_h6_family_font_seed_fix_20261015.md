# H6 full-family：Matplotlib font Timer停止の修正

SOURCE `99abc20a29979f3b6b11d9c7127c1e8cf84c643f`。[診断・検査・最終binding](../../artifacts/resource_applicability/track_a_h6_family_font_seed_fix_v1/2026-10-15/README.md)。run01/02はともに科学予約全0/native wait2で停止、元証拠とcarryを保持した。run02はnative provider import中Threads2・affinity63不変を記録した。run01の曖昧な内訳は補完しない。

固定profileのMatplotlib font_manager.pyを確認し、空のMPLCONFIGDIRではFontManagerが5秒後の警告用threading.Timerを起動する経路を特定した。数値providerのtrotterlib初期importがplottingもimportするため、thread1監視に抵触する。installed libraryや科学providerを変えず、停止runの完成済み表示font一覧JSONをhash/version固定して新run/workerの専用MPLCONFIGDIRへbyte-only配置する。これはfont metadataだけで、科学のpartial結果・回路・transpile・JIT cacheは再利用しない。Numba cacheも専用tempへ隔離する。

人工49件PASS。新しいseed済み専用directoryから、numpy/qiskitを先にwarmせずにnative providerをimportする限定診断もPASS。5ms観測でthread1、H6配列/scalar/bias/shot/build/probe/compile0。主担当が新source closure・profile34273・provider230・input18・font bytes/SHA・carry/unused outputを照合した。独立reviewではなくsubagents0。

利用者の実行・停止後修正承認に従い、新grant/outputへ結合してfresh CPU/memory/pressure/OOM/FS/inode/quota検査後に一度起動する。driver63/workers62,61,58,53,52,50、max6/各thread1、AS32/RSS9GiB、wallnull/当stage累積CPU86400秒、metadata CPU1800秒、最大新compile18、旧6再取得0を維持する。盲目的retry・次stage自動進行なし。
