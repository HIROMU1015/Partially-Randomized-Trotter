# V2再開準備：保存A/B・native混合・compiled残差

[報告・最小計画](../../../../docs/research/track_a_v2_residual_analysis_readiness_20261016.md)。既存理論結果7e416705…、実行SOURCE7e59df8a…、旧V2準備dd695e65…、V1成果整理63e5773b…を区別する。

- [入力台帳](input_manifest.json)：公開済み22ファイルのcommit/path/bytes/SHA。private pathなし。
- [再利用・schema/候補対応](reuse_inventory.json)：832理論行・10,480接続行・262 scope entriesのmetadataだけを照合。
- [A/Bとliteral nativeの対応](model_native_mapping.json)。A-prefix/B-tailの独立mixed期待値列は未取得。
- [数値stageの未承認計画](residual_execution_plan_DRAFT.json) / [未承認grant案](grant_UNAPPROVED_DRAFT.json)。承認false・SOURCE null・CPU[]・command null。
- [技術的な保全・公開確認](technical_validation.json)。既存結果再計算・残差・回帰・比・rankingは未実行。

既存832行と1,212 primitive digest・144 law・保存接続を再生成しない。最初の数値stage案はRZ単価のuniform A/B別、system/axis/来歴別の条件付き残差診断だけ。CX・mixed理論・shot総費用・PR/PF費用比更新・角分類・希少次数寄与は別判断。

精度/factorを独立標本として重複加重せず、探索8・確認32・PF再利用を区別。全接続は完全数値wrapper envelope未認証であり、critical来歴不明はfit不可。PR/PF平均signed相対残差差>=.05で即STOP。.05はV1の2%基準と別。

新しい計算runnerの実装/人工検証・final SOURCE seal・別利用者承認・fresh gateが数値実行の前提。今回新science/数値再解析/compile/新grant0。既存SOURCE/raw/grant/ledger、V1、H5/H7を保全。公開可能な文書・metadataのみcommit/push、remote確認後mandatory STOP。
