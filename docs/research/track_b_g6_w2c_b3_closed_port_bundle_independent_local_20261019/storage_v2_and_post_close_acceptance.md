# Storage V2・post-close

直接基点 `5e862a70f2d74defe72dd39205d38c2a4f8ebeaf`。統合公開HEAD `62967269e611b61e73b70abe20e998d9c7eaaf50`、人工測定SOURCE `deade78a19f5b23859d16816d3a7bfb37765dd80`。実施2026-10-19〜20 JST。ARTIFICIAL_ONLY・未採択・実科学入口閉鎖。

旧独立storage/trace consumerをそのまま利用し、RAM shadow、physical uint8 state、trusted prefix、journal、summary/condition receipts/global receipt/diagnostic/marker/delivery/callerを照合する。RAM intentを物理durabilityの証明にしない。

正常BundleにはSOURCE/input/runtime宣言、単一claim、全完了、Port proof、pair/F01、global state/counter、6phase、trusted trace、condition receipts、summary/global receiptのport_binding、diagnostic、marker、Admission STOP、close、delivery/caller、global byte予算、保存errorなしの結合を要求する。marker単独では成功にしない。

公開26件のINCOMPLETE_EVIDENCEを保持。部分write/flush/close、ACK欠損、state/trace保存失敗、未保存callerやreceiptを推測補完しない。新規missing caller/receipt/state/traceも証拠不足として拒否。

独立adapterはreadable Port証拠の不整合を拒否する一方、失敗した空/部分final documentを再parseしただけで元の保存不足を正常化しない。caller witness自身のclose、fsync/power-loss耐久性、完全な未保存atomic履歴は未確立。
