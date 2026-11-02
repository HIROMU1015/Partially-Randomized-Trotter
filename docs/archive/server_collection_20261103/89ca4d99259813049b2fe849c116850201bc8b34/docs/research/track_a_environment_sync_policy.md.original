# Track A 両環境の研究情報同期ルール（暫定初版）

2026-10-14。サーバー側の技術的保全ルール。ローカル側照合後に共通運用を確定する。科学方針やgrantを変更する文書ではない。

## 正本と共有対象

科学的結果の正本は**元の固定科学SOURCE・原本結果・input/compiler/seed binding・対応provenance**。
GitHubは軽量結果と来歴・研究ノート・レビューを両環境が参照する共有先。本[共通ノート](track_a_shared_research_status.md)は索引で、rawや固定値を上書きしない。
GPTの科学判断は該当範囲の承認済みレビューを参照し、実行承認・technical PASS・科学採択を別欄にする。

|GitHubで共有するもの|各環境に保持するもの|
|---|---|
|軽量JSON/CSV結果、条件・制約・coverage、source/result/input/profile SHA、public manifest|大容量NPZ、raw trajectories、checkpoint、cache、ledger原本、private runtime|
|研究概要、短い dated note、review、claim/evidence、実行契約の共有可能なscope記録|private venv・local activation/launch設定、credential/token/秘密endpoint、machine-specific caches|
|file path（repository相対）、commit、Git blob SHA、bytes/SHA256、公開状態|機微絶対pathの対応表、PID/start identity詳細、独立受領前の転送原本|

本体を共有しないことと来歴を隠すことは別。private fileは識別名・存在/bytes/記録hash・生成SOURCE・保存責任者を示す。
記録hashと今回実測hash、存在確認と独立再受領、metadata受理と原科学計算再現を分ける。

## SourceとResultの対応

各研究単位に `(repository, branch, execution_SOURCE, science_SOURCE, data/seed_SOURCE, result_commit, input_H/DF/state, compiler/environment, task/axis/phase/control, namespace)` を可能な範囲で付す。
文字列versionや同じfile名だけで同一結果としない。H4 ordinary/structuredとH6/H8 native、DF rank政策、PF対照、u、P、sampling/selectionの差を明記する。
sourceが変わる再開では旧COMPLETEの元SOURCEを保持し、許されたtransitionとmappingへ辿れるようにする。共有索引でbindingを新たに作り直したことにしない。
公開result commitは実行SOURCEとは異なる。後のreview/draft commitを結果の実行SOURCEと誤認しない。

## Branchと参照版

branchは可動参照、40文字commitは固定版。新しいCodexはまず`git ls-remote --heads origin <branch>`で現在のHEADを読み、共有ノートの基準SHAとの差を記録する。
同名remote branchがなくても、そのcommitが別公開HEADの祖先の場合は取得可能。branch未pushとcommit未公開を分ける。
HEADが異なる場合は祖先/後続/分岐/判定不能を記録し、勝手なpull/merge/resetや既存worktreeへのcheckoutをしない。
資料作業はcleanな独立branch/worktreeへ固定commitから開始し、変更fileを明示stage、non-force pushする。
新branch公開で既存実行worktreeやgrantのコードを更新しない。認証失敗時は設定を変えず手動pushと未公開状態を報告する。

## 研究ノートと競合処理

短い dated noteへ「何を確認/変更したか、理由、旧版、元証拠、stage状態、次の担当」を追記する。長い過去履歴を複製しない。
共通入口は現在の索引、旧レビュー/契約/結果は当時の記録として保持する。新方針は日付・由来・採否/認可を明記し、科学レビューを技術担当の判断へ置換しない。

競合は次のように扱う。

1. 同じscope/identityの数値差：両commit・blob・算術条件を並べる。解決前は一方へ上書きしない。
2. 科学条件差：別case study/namespaceとして索引化し、version一致を理由にmergeしない。
3. 進捗差：実行metadataの観測時点と公開起動snapshotを分ける。古い「準備/未実行」を現在状態の証明にしない。
4. GPT判断差：各レビューの範囲・採択・後続amendmentを確認。技術的な新しい文書だけで旧科学判断を無効にしない。
5. dirty/untracked：index/statusを読取保存して保全。staged deletionや原runtimeを掃除・復元・commitしない。

H8 snapshotは変化するため公開時点にsealしたfinal resultとは呼ばない。H8担当の原runner/observerが実行を管理し、この同期taskからsignal・停止・再起動・binding変更を行わない。

## ローカル側への引継ぎと初版の確定

[handoff](track_a_local_codex_sync_handoff.md)の固定branch/commit一覧をcleanな資料用worktreeで確認する。
file単位commit/blob/bytes/SHAを比較し、ローカルのみ・サーバーのみ・未push・public一致・未確認を区分した差分表を返す。
runtime環境・directory絶対位置を揃えることは研究情報共有の前提ではない。venvをinstall/upgradeや再作成で一致させない。
追加でrawが必要なら科学的再現目的、対象/bytes/hash/容量・取扱いを具体化して別途転送を設計する。
ローカルの確認完了と競合解決が記録された後にのみ、共通ノートの「サーバー側暫定初版」を統一版へ改める。

同期作業の終了時はbranch/commit/remoteとリンク・bytes/SHA監査、未確認事項を報告してSTOP。将来候補は実行認可と区別し、新科学計算を自動開始しない。
