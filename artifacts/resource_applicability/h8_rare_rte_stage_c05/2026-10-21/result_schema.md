# 将来C-1結果schemaと実装接続

C-0.5は実科学結果を保存していない。以下は承認後の出力schemaであり、toy receiptと区別する。

|出力|単位・保存field|
|---|---|
|private task registry|condition/stratum/phase/replica/SOURCE/input/epoch、orderとcomponent step streams、全旧/新衝突検査、digest。実seedはprivate|
|parent preparation/law receipt|元snapshot/state/preparation/compiler hash、純粋復元call、元law hash、候補policy、canonical差分diagnostic。bit再現/費用bias certificateはfalse|
|task trajectory.json|順序付きq×r event列、order word、元係数/phase/normalization、event digest、cost_conditioning_only=true、quantum_weights_changed=false。rawはprivate|
|axis result|cosine/sine、同event digest、full measured wrapper fingerprint、compiler hash、整数RZ/CX/depth/RZ_depth/CX_depth/size、cache_hit|
|native receipt|task/PID/start/parent/UID、exit_code、same-parent wait/reap、所有adopted descendants、CPU/RSS（RSS合計は共有量を二重計上し得る）、未投入task、STOP/retry=false|
|ledger/reservations|事前予約のkind/count/unique key、nonrefundable=true。cache hitとcompile missを分離し、失敗分を戻さない|
|pilot_summary.json|層ごとのn/paired G mean/variance/SE。n0の推定値null、n1分散/SE null。未知母平均null、母平均順位保証false、all_support_upper_bound/CI null|
|independent numeric audit|同じ保存整数Gに対するraw moment/Fraction別実装照合。数値一致は科学的保証ではない|
|completion/STOP|全科学call・最終SOURCE/input/runtime gate・回収・失敗/未完了・旧証拠保全・次段階未承認|

平均/分散はFraction文字列、SEはDecimal100桁。pilot標本をproductionや旧Pとpoolしない。異なる候補・層・phaseの混入を拒否する。原S0 n48は別anchorで、将来の期待費用会計でその不確かさを含める。今回はS0再解析・母平均/期待費用の新順位判定0。

SOURCE接続はOriginalProvider.load_parent→Adapter.initialize_law→register_streams→run_registered。実permitはbegin_approvedで明示承認receipt・SOURCE/input/runtime/provider/library manifestを照合し、新markerを一度だけ消費した場合に作る。現在のPROPOSED契約はそれより前に拒否する。C-0.5で有効permit/grantは生成していない。

private manifest、module import pathsと実module hash、prefix/variance/footprint/validation選択、科学call整数予算を凍結するC-1起動bindingはユーザー選択後の作業。現在は実行CLIを公開せず、設計/preflight CLIのみを提供する。実seed/raw/内部path/有効grant本体は公開候補へ含めない。
