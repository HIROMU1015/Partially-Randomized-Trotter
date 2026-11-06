# Claude独立レビュー追補：SOURCE・先行研究・資源指標

2026-11-06 JST。SERVER Codexによる静的技術調査。新しい独立科学レビュー、研究方針の採択、科学的数値再解析ではない。

- [REPORT.md](REPORT.md)：Q1–Q7への回答と判断の引渡し。
- [SOURCE_FORMULA_TRACE.md](SOURCE_FORMULA_TRACE.md)：finite RTE、signed stages、整数配分、二乗則の条件。
- [PRIOR_ART_MATRIX.md](PRIOR_ART_MATRIX.md)：一次文献との重複と残る差分候補。
- [METRIC_AVAILABILITY.csv](METRIC_AVAILABILITY.csv)：保存指標・candidate/cohort・不足データの台帳。
- [MINIMAL_NEXT_VALIDATION.md](MINIMAL_NEXT_VALIDATION.md)：別承認が必要なB1/B2の最小案。
- [SOURCE_INDEX.json](SOURCE_INDEX.json)：今回取得した固定Git blobと読取範囲。
- [VALIDATION.json](VALIDATION.json)：技術確認と実行境界。
- [validate_static.py](validate_static.py)：科学モジュールをimportしない文書・metadata検査。

基準は統合資料 `3380f6de0d86c0ac1db9aeb6d97ab5d328dd43bd`、台帳 `2779710a86d380bb484accc51d3dccba183370b8`、主要結果carrier `155529e20a8fac39e00786c40b57830b71203c4f`、可視化 `b3b194dbf405285239326fd25b671d30957b801c`。carrierと科学実行SOURCEは区別した。

Claudeレビューは、指示での `(1).md` 付き名称に対し、現在のサーバーで参照可能な同日・同題の `(1)` なしMarkdownを読んだ。両名のbyte同一性は確認不能。原本は公開せず、読んだファイルのSHA-256と照合節をVALIDATIONへ記録した。

専用の独立GitHub clone・新規branch内の本directoryだけを追加する。元checkout、既存SOURCE/input/raw/QPY/result/grant/ledger/marker/STOP、台帳と11図表は変更しない。LOCAL・別サーバーへのSSH・転送は行わない。公開後はremote SHAと各blobを照合し、mandatory STOP。次の科学判断はユーザーへ戻す。
