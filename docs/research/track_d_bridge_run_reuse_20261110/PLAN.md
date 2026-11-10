# 同一ruleのrun内部再走査削減：限定PLAN

2026-11-10 JST。基点e6bd6c5dedf27601b0c3b50877653ad43d04cd17。同じリソース評価branchを継続し、旧SOURCE／入力／結果／STOPは変更しない。既存資産にrunをepoch間で保持する実装はなく、前回はbit-key cacheでも全命令を読む。

対象は保存q4_sample00/cut2、q8_sample00/cut4、Strong／whole-path Reference X/Yの8 wrapperだけ。Linear H4／1Å／STO-3G／Nα=Nβ2、元H／number-CDF8／元frame順LD2／S2/s1/r32/K0／h=.1、C1 OFF、元H ground仮定・準備費除外、既存P0条件・位相規約を維持する。前回run01/localのnative gzip／phase項とINPUTS.jsonを原本として再利用し、rawを再生成しない。

## Phase Aと限定設計

1Q flushの順序は非1Q命令のqubits tuple順と最終wire昇順。CX削除で以前の1Q runが結合するため、単純な位置cache／固定幅frontierは不可。任意CX連鎖へ完全worklistを適用するには依存graphと順序管理の拡大が必要なので、今回は一方式に限定する。

1Q runを不変の順序付きtupleとして保持する。単一runの評価結果をそのrunに記憶し、変更されないrun内部の分類／bit-key生成／再合成を省く。CX削除で複数runが結合したときだけ展開し、前回OneQStreamと旧公式oneq_runsを再利用する。各runを同じwire上のopaque barrierとして旧CX feedへ渡す。非空1Q runはCX相殺を常に遮るため、内部命令ごとのfrontier登録を省ける。新しい可換移動／相殺規則を作らない。空runはbarrierとして残さない。

全unit列は引き続き毎epoch読む。CX frontierの再構築そのものは残り、局所worklistによる全面的再走査除去とは称さない。既存rule・phase順を保った小さいrun単位の再利用が実測上有効か判定し、DAG/compiler再実装へ拡大しない。

## 検証と計測

まず旧人工／回帰と、局所run結合・多段CX連鎖・disjoint flush・classical blocker・signed zero等の差分テストを実行する。最終native bit署名／各論理roundのword／phase／round項目／MPFR160を一致条件とし、閾値は追加しない。

保存native8について、一回のtrace付き旧新差分を先に行う。各round word hashとphaseを比較し、旧保存P0全56値にも照合する。PASS後、同じ入力・同じ環境・共通fresh-process poolで8×旧新の一回の性能比較を行う。traceのための追加全word走査は計測から外す。native decode＋engine＋最終T0を計時し、raw/loweringやP0に対するend-to-end改善とは呼ばない。旧／新とも前回ObservedCXを使う。process peak RSS、1Q命令訪問とunit訪問、CX feed、合成／cache／memo、round数、node/frontier/cache保持を分けて報告する。

実空き資源を確認し、独立な16 taskを適切に並列化する。共有設定・他jobへの変更なし。新sampling／P0／whole-path生成／simulation／q16以上／fit／N／13-round／詳細科学レビューは0。技術不具合は初回SOURCEと失敗を保存して最小修正する。固定入力不一致／解決不能なword・phase不一致は該当処理STOP。raw/native角列/phase原本は非公開、公開はコード・科学要約・hash・scalar計測だけ。公開ref/tree/bytes照合後mandatory STOP、研究判断はユーザー／GPTへ戻す。
