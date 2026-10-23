# H4費用逆転：静的構造解析・引継ぎ

基点1caaba2516dd3a30475e2acbd69672810a9fc05a、2026-10-24 JST。

- [専用報告書](../../../docs/non_df_h4_cost_reversal_structure_20261024.md)：固定4経路、prefix/law、成分構造、仮説A〜E、Q1〜Q6。
- [静的照合](verification.json)：旧611 SHA、GitHub11file、固定SOURCE15file、封印runtime7file、Pauli保持label、DF保持basis、限定保存schema inventory。
- [checker](check_static.py)：標準libraryだけで照合。既存人工gate tests3件＋人工prefix1件を再利用／実行。
- [GitHub取得](github_verification.json)：固定file全文を取得、Git blobと固定refを照合。
- [D1〜D3非実行案](next_contracts_DRAFT.json)：優先1は4代表caseの部品隔離＋full（最大9 cosine compile）。優先2はPauli制御位置（2）、優先3は条件付きDF reuse（1）。現在すべて未実行・未承認、SOURCE/runtime/grant未seal。
- [終了](STOP.json)、[公開状態](publication_status.json)。旧native UNKNOWNや旧certificate欠測を今回のSTOPで置換しない。

保存平均Cは旧結果を引用し、今回の科学的再集計・H4数値評価・回路生成・sampling・compile・N/Gは0。
Compiled価格の成分配賦はnull。Pauli prefixはZ/ZZのみだが、DF basis単独への定量的帰属は未確定。
検証を再実行する場合はrepository rootで `python3 artifacts/non_df_h4_cost_reversal_structure/2026-10-24/check_static.py`。
これは新科学runを起動せず、verification.jsonを本調査directory内だけに保存する。

次：ユーザーがD1（必要ならD2/D3）の科学的範囲を選び明示承認。Codexが新replay adapter・人工位相検査・新SOURCE/runtime・契約・grantを準備する。今回はmandatory STOP。
