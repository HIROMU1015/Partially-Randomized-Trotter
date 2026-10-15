# H6 full-family 起動STOP・cache/監視診断修正

[停止・修正・検査・binding](../../artifacts/resource_applicability/track_a_h6_family_startup_cache_fix_v1/2026-10-15/README.md)。SOURCE `84d236cf9bbdd4f0b990dbb8f67c97d312a08ac7`。初回はNativeAdapter初期化中にEXTERNAL_AFFINITY_OR_THREAD_CHANGEDでfail-closed STOP。全科学予約0/native wait2/recorder0/driver退出、旧資料を保持した。

数値入力を読まない限定import診断で、Numbaが元provider側のcache pathへ書こうとする経路を確認した。新sourceはNUMBA_CACHE_DIRをrun/workerの専用tempへ設定し、thread count異常とaffinity異常を別理由で記録、library import/snapshotのphaseも分ける。thread1/5秒/PSI/OOM/AS/RSS制約は変更していない。旧ログは両者を区別できず、根本原因を今回の観測から逆算して補完しない。新startupの実動確認が残る。

人工46件PASS。主担当が差分・SOURCE/profile/input/carry/bindingを確認した（独立reviewではない、サブエージェント0）。既存venv/科学provider/math/compiler options/旧6結果は不変。失敗carryとUNKNOWNを保持しreset/refundなし。

利用者の実行・停止後修正指示を根拠に、新output/grantへ再結合。新9構成のscreen/適格な最大18compile、max6workers、driver63/worker62,61,58,53,52,50、AS32/RSS9GiB、wallnull/CPU24hを維持し、起動直前fresh gateで合格した場合に一度起動する。盲目的自動retry・次stageへの自動進行なし。
