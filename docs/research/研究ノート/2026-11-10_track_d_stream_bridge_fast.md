# Track D 同一ruleの高速化・省メモリ化

2026-11-10 JST。[結果・科学条件](../track_d_stream_bridge_fast_20261110/REPORT.md)。基点817a1cd5、algo/準備SOURCE53f5d69e、native入力固定b5b1ad59。旧1Q/CX規則を再利用し、stream直結・exact run cache・不要な中間Counter省略を実装した。最終word／位相／roundの一致を優先し、worklistによる処理順変更は行わなかった。

計測keyword不備で旧optimizer前に停止した最初の記録とfast8 PASSを保持し、adapter SOURCE4c78a6c0で旧8だけを完了。source97bf9258のrun02で同じ16 process poolの比較を一度行い、主性能値とした。旧新の8wrapper×7資源、word、binary64／MPFR160 phase、各roundは一致。q8は約2.1倍高速、native-stage RSS約53〜54%減。63人工・技術回帰・固定54人工case差分PASS。late phase adapterの最初の人工期待値エラーも保全して修正し、本体ruleは変更しなかった。

raw QCとgate.definition cache、全列走査／可変履歴は残り、q16以上は実行しない。有限校正・整合性検査器として保全し、追加開発は再走査削減の費用対効果へ絞る。既存A/F/Sを使うPR＋RPE共同最適化へ重点を移す案も示し、正式採択はユーザー／GPTへ戻す。

新P0／sampling／whole-path／全入力simulation／q16以上／H・law変更／fit／N／13round更新／詳細科学レビュー0。旧SOURCE／raw／結果／grant／ledger／marker／STOPを保持し、公開ref／tree／bytes照合後mandatory STOP。
