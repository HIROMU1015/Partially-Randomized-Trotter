# H4 D1：採用SOURCE基準のone-shot

[報告・固定条件・主張範囲](../../../docs/non_df_h4_d1_source_execution_20261024.md)。
基点f173b76c7、旧科学0c329b30、保存D1 SOURCE48d4d27a0。
承認発言と具体的範囲は[approval.json](approval.json)。
旧消費済みgrant/STOP・旧hashFAIL・元native UNKNOWNを保持する。

新controller/runnerは元adapter/compiler/ledger/witnessを再利用する。
新人工10＋既存17件PASS。SOURCE固定後にseal/active_contractを作り、新grantを一度だけ使用する。
full再生成0、既存QPY確認4、隔離/empty構築最大5、費用compile最大9。
Stage A/B/C、実際の消費数、終了はrun01内のstage/ledger/STOPを正本とする。
このSOURCE固定時点ではまだ科学起動していない。

実行開始後にSOURCE/入力/runtime/grant/markerを改変しない。
結果と新native監査を保存・commit/pushした後mandatory STOP。
