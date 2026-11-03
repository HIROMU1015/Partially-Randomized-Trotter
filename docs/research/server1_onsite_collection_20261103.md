# サーバー1原本からの追加収録

2026-11-03。main `c1dbc38ecd15d04062ba8bd3796a2621965c64a3` を比較基準とし、今回指定されたサーバー1の69要求だけを確認した。treeの子項目を分けた119項目の処置と、Git blob・mode・出典を照合した。

これは既存資料と再現用SOURCEの内容収録であり、枝全体の履歴merge、科学的な再採択、原稿配置案の採用、科学実行の許可ではない。
公開用commitの親は既存mainだけとし、非公開の元commitを履歴の親へ接続しない。元の完全bundle・受渡し物全体・私有パスの一覧は公開しない。

| 区分 | 個別項目数 | 扱い |
|---|---:|---|
| 同一内容が最新mainに存在 | 7 | 別path・既存保存版も含めてblobとmodeで確認。重複追加しない |
| 今回保存した公開可能な原版 | 99 | `.original` に原内容・modeを保持 |
| 私有保持 | 13 | ローカル専用報告、私有環境、保護対象の運用記録。公開しない |

項目数は研究結果数・独立実験数ではない。69要求のうちtree要求には複数の処置が含まれる。公開しない項目の実パスと要求行の対応は非公開報告に保持する。

[収録した原版と出典の対応](server1_onsite_collection_20261103_files.tsv)から、元snapshot・原path・blob・modeへ戻れる。

## 原版を探す

| 用途 | 保存原版 |
|---|---|
| 共通PFの過去の原稿配置草案 | [原版](../../docs/archive/server1_main_collection_20261103/fafefe97972047de8c0d3e3fa9659ef7a48d45b7/docs/research/track_a_h246_pf_publication_plan_20261023.md.original) |
| CNGの技術調査・適用範囲 | [原版](../../docs/archive/server1_main_collection_20261103/aa99e5e089c43fa15c88109da471218667435a64/docs/research/pr_pf_cng_technical_investigation_20261022/README.md.original) |
| CNG診断の原SOURCE | [原版](../../docs/archive/server1_main_collection_20261103/76d286ca9100776bbc840393d3ce4f8400316beb/src/trotterlib/cng_gaussian_diagnostics.py.original) |
| PRパラメータ境界の保存資料調査 | [原版](../../docs/archive/server1_main_collection_20261103/93fae93d451f3932e0c36087a1e08b33e51b9ac2/analysis/pr_parameter_boundaries_20261027/REPORT.md.original) |
| Track Bの過去の研究レビュー | [原版](../../docs/archive/server1_main_collection_20261103/343b847fc484745dbc98548cd026a41c4fb79ebb/Track_B_G6_RA_H4後_科学的価値と研究方針レビュー_2026-10-22.md.original) |
| 過去の研究結果マップ | [原版](../../docs/archive/server1_main_collection_20261103/a6f575ae85d9dad6703e8ee9052ae36dcea0ce45/RESEARCH_RESULTS_MAP.md.original) |

## 原版と現在の実行context

すべての新規原版を `docs/archive/server1_main_collection_20261103/<blob>/<原path>.original` に置いた。現在のライブラリや研究runnerへ自動採用していない。Pythonの原版はASTで確認し、import・実行はしていない。
原稿配置・追加科学契約・研究レビューの当時のDRAFT、未採択、欠測、条件付き判定、STOPは原文のまま残す。過去の共通運用指示も保存原版として扱い、現行のAGENTS.mdや個別契約を置換しない。
第三者SOURCEの保存原版は元copyrightを保持し、[Qiskit Natureの既存license](../../LICENSES/qiskit-nature-Apache-2.0.txt)を併読する。

UNCOMMITTEDと記した項目は受渡し準備時点のコピーであり、元checkoutのHEADに含まれていたとは記述しない。固定commitも閲覧・技術作業のsnapshotであり、実験の科学SOURCEは各原文のSOURCE/input identityを維持する。
元snapshot全体は非公開保存先に維持されるため、そのcommitのGitHub URLから全内容を取得できるとは案内しない。公開された保存原版と未公開の依存・環境記録を区別する。
旧マップの固定SHA・ローカル可用性・未照会URL・過去の保存案は当時の索引記録である。全リンクの現在のHTTP可用性や全rawの保全を保証しない。[現在の統一索引](RESEARCH_RESULTS_INDEX.md)と[先行するmain収録](main_server_results_collection_20261103.md)を優先する。
保存原版の相対リンク・manifest・SOURCE hashは書き換えていない。未収録の原データや別pathにある同一内容を参照するものもあるため、対応表と原文のscopeを併読する。パケット全体の完全性、現在のAPI整合、再現実行の成功は今回の確認範囲に含めない。

## 技術確認

公開差分の内容・Git mode・出典blob、Python AST、JSON/CSV/TSVの構造、新しい案内リンク、公開用commitの親と差分を確認した。既存mainの科学ファイルは上書きせず、新しい数値・判定を生成していない。
新しい科学計算、原結果の数値再解析、回路build/compile、元runner/testの実行は0件。原repo、原本、既存bundle、実行契約・grant・ledger・marker・STOP、jobの変更・停止は行わない。
