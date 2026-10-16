# M1-S γ依存V1：runtime修正と完了までの実施契約

2026-10-17。ユーザー最新指示「しっかり完了するまで行ってよい　途中で止まった場合は修正して行って」に基づく技術的継続。旧A1の再利用ではなく、別SOURCE S2・同一科学条件の再固定C2・専用authority A2・別出力を用いる。旧失敗・消費済みmarker・SOURCE・契約は変更しない。今回の指示は、技術的障害を修正して既に承認済みの解析を完了させる承認と解釈する。研究方針の新規採択ではない。

旧停止はPython 3.11.0rc1に`sys.set_int_max_str_digits`が存在しないためで、科学入力decode前に発生した。新namespace `m1s_saved_signal_gamma_v2`は科学V1の技術実装versionである。optional APIがある場合だけ25000桁を設定する。APIがない場合もrational 65536 bits、output 16 MiB、CPU160秒/wall320秒の既存上限を使う。25000桁は整数変換設定であり精度閾値ではない。実際のinterpreterで65536-bit整数の往復変換と、resource設定・起動・atomic checkpointの経路をmetadata stubで検査した。実科学入力はテストで開かない。

bounds/boundary/analysis/supervisorおよびpackage initializerは旧版と同一bytes。controlはoptional API呼出しへの置換のみ。spec/runnerのpathを別namespaceへ変更し、10ファイルをSOURCE identityへ含める。結果と契約schemaの意味論・bytesは維持する。新しい数学的誤差モデル、閾値緩和、結果依存fallbackは導入しない。

H4 linear1.00 Å/STO-3G/8 modes/DF rank12、G3/G6/G9、保存δ=1/8,1/16 Ha^-1、R=2,4、T=1,4 Ha^-1、zero込み108行・48対照を保持する。Task Aのraw無補正円周位相、全0<γ<π/2の共通正margin条件を扱う。保存raw-H biasにモデル差を再加算しない。

PER_METHOD_ROW、α=0.05、fresh-IID/Hoeffding/X/Y等配分、条件付き保存coverage、p=128、CPU160秒/wall320秒・CPU1worker/GPUなし・memory2GiB・各output/log16MiB・既存演算上限を維持する。32CPUと約48GiB空きmemoryを確認したが採択済み1workerを変更しない。小規模有理数カーネルとcheckpointが主経路であり、既存dyadic packとtrig cacheをそのまま利用する。速度向上率は主張しない。

同じ十分shot設計の連続marginと境界だけを比較する。絶対shot数、ceil後の整数削減数、新ランキング、native費用、多round RPE、物理shot保証、科学的GOは対象外。`CONDITIONAL_ENGINEERING_NOT_CERTIFIED`を保持する。各方式行95%は全72行同時95%ではない。連続margin改善はceil後のstrict削減を保証しない。G9絶対差未解像を相対改善で救済しない。

新SOURCEと同じ科学条件を再固定後に専用authorityを作り、科学入力・SOURCE・契約・interpreter identityとclean treeをguardで照合する。各試行のmarkerは排他的に一度だけ消費する。技術停止時は当該試行を保存し、最新ユーザー指示の範囲で必要な新version・新authorityを用いる。科学条件変更が必要になれば勝手に変更しない。解析完了後はmandatory STOPし、科学的解釈・研究採択はユーザーへ返す。

根拠は同directoryの新machine契約、user completion instruction、SOURCE静的検査、38件の登録外技術テストと後続の実結果・保全監査を参照する。新machine契約の科学条件は旧採択C1と同じで、変更項はSOURCE identityのみとする。
