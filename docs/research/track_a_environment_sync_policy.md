# Track A 正式共通研究情報の同期ルール

## 2026-10-16 結果追補の来歴

最新正式同期方針[6b78e5f3](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6b78e5f39917d92d6c60694d3e0b0d06394d7eb5/docs/research/track_a_environment_sync_policy.md)を固定参照する。本result branchは共通祖先4675730e…からの別系統であり、無条件merge/cherry-pickはしていない。旧本文・inventory・原reviewを保全し、共通ノートへ日付付きのH6/H8完了結果を追加した。

H8 runtime原本の誤った欄名`confirmed_PR_union8`は変更せず、byte/hashを残した公開projectionだけで`confirmed_PR_union6`へ訂正する。actualの入力はH8確認6候補/独立32で、H6のunion8と混合しない。[照合と訂正範囲](track_a_h8_native_b0i_full_family_results_20261016.md)。
公開対象は軽量scalar/費用/適格性/P表、元証拠hash、終了・監視の要約と研究ノート。NPZ/実runtime/checkpoint/cache/ledger/venv/credentialは収録せず、原稿・共有環境・他job・旧grantを変更しない。新科学計算・実行認可を同期に含めない。

以下は各時点の履歴として保全する。

2026-10-14 JST、研究情報共有 v1。server初版とlocal照合を利用者の指示で正式統合した。
[server元policy](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2298c4cdbc81d81c6268fdb7bd80f0b29d5b4e8c/docs/research/track_a_environment_sync_policy.md)を履歴として保全。科学方針・実行grant・raw・SOURCEを変更するルールではない。
**情報共有の統一であり、sourceや実験データの物理的な完全同期ではない。**

## 新しいCodexが最初に読むもの

1. [PROJECT_MAP](../../PROJECT_MAP.md)、[研究概要・現状](研究概要・現状.md)から入口を確認する。
2. [正式共通研究ノート](track_a_shared_research_status.md)と[科学review判断](track_a_scientific_review_decisions.md)を読む。
3. [条件対応](track_a_resource_comparison_conditions_20261014_v0_4.md)、[固定source/result参照](track_a_local_sync_reference_index.md)、対応する原review・結果・provenanceへ進む。
4. [server inventory](track_a_server_inventory.md)、[local inventory](track_a_local_inventory.md)、[差分](track_a_environment_sync_differences.md)で環境固有/未公開/未確認を確認する。
5. VALIDATION_STATUSとmachine manifestは該当branch・検証scopeに従う。dated note・古い起動文は当時の履歴であり、現在結果の代替ではない。

## 正本の関係と優先順位

原科学結果の正本は固定science/execution SOURCE、原本結果、input/state/compiler/seed binding、provenance。共通ノートはそれらの**現在の共有索引**であり、元値を書き換えない。
科学的採択は該当scopeのGPT review原本と、その日付・後続追補に従う。technical PASS、結果公開、科学的採択、計算明示認可は別欄にする。
新しい技術文書や共有ノートだけで古いGPT判断を無効化しない。原本未取得なら「レビュー原本未取得」、利用者チャット由来と固定資料由来を明示し、本文を推測で再構成しない。
R09 H4全面結果後reviewは利用者指定原本から取得し、exact copyのSHAで公開した。過去の「判断待ち」は時点の履歴として保全する。

各研究単位のidentityは `(repository, branch, science_SOURCE, execution_SOURCE, data/seed_SOURCE, result_commit, H/DF/state/input, compiler/environment, task/axis/phase/control, namespace)` を可能な範囲で保存する。
結果保存commit、review公開commit、execution SOURCEを混同しない。dirty bytesや同じ名前を固定sourceの代用にしない。
H4元family/B0+I、geometry/native、H6失敗/後続成功、探索8/fresh32、H8準備/時点snapshot/最終結果は別scope・identityで索引化する。

## Branchと固定commit

branchは移動する参照、40桁commitとfile blob/SHAは固定版。共通ノートへ基準commitと観測時点を記録する。
開始時に既存remoteのHEADと基準commitの祖先/後続/分岐をread-only確認する。同名branch未pushとcommitが全remoteで未公開であることは別。
HEADが違っても無条件pull/merge/rebase/reset、科学worktreeへのcheckoutをしない。資料変更はcleanな専用branch/worktreeで固定commitから行う。
公開はHIROMU1015対象repositoryだけ、明示allowlistでstage、staged diff確認、non-force push。quration等へcommit/pushしない。
push後は別のGit取得/固定GitHub file fetchでremote SHAと主要bytes/SHA、リンク・旧source保全を確認する。認証失敗時は共有設定を変えず未公開状態と手動push方法を報告する。

## 共有するmetadataと同期しない実体

|GitHubで共有するもの|各環境に保持するもの|
|---|---|
|軽量結果、coverage/欠測、公開契約、source/result/input/environment/selection hash、public manifest|大容量NPZ/raw/trajectory/checkpoint/cache/ledger原本、private runtime|
|共通ノート、日付付き判断履歴、取得済みreview exact copy、条件表・claim/evidence参照|private venv、activation/launch設定、資格情報/秘密endpoint、環境固有path|
|repository相対path、固定commit/Git blob/bytes/SHA、所在識別子・公開状態・責任側|絶対path対応表、機微native identity、独立受領前の転送原本|

server固有データはserverが原本を保持し、local固有ノート/ZIP/設定も原本を保持する。本文未同期を来歴不明や証拠不存在と同一視しない。
記録hashと今回実測hash、metadata存在とbytes受領、照合PASSと原科学計算の外部再現を分ける。venv再構築/upgradeで環境を揃えない。
大きいrawを移す必要があれば、目的・対象/bytes/hash/容量・private取扱いを固定する別作業にする。今回一括複製しない。

## 更新・競合・時点snapshot

新しい固定結果/review/承認範囲を得た場合は共通入口と該当条件/検証文書を更新し、dated noteに変更理由・旧版・原証拠・状態・次担当を追記する。
証拠inventory/statusが変わる場合は対応machine manifestを更新する。情報索引の変更だけで科学validation manifestや原実行statusを変更しない。
既存server/local inventory、差分、統合候補は時点履歴として保持し、追補は新しい記録として作る。

- 同一scope/identityの数値差：両commit/blob・算術条件を並べて未解決にする。一方へ黙って上書きしない。
- 条件差：別case study/namespace。更新日だけで統一せず、比較集合・DF/PF/control/数値/標本/Pの違いを示す。
- 新旧run：旧STOP・失敗・欠測・消費済みgrantを残す。後続成功へ遡及変更しない。
- review判断差：原reviewの対象と追補関係を確認。科学判断をCodexで独断解決しない。
- 未取得/未保存：UNVERIFIED/NOT_SAVED/MISSINGを維持し、0や成功で補完しない。
- dirty/untracked：HEAD/status・必要hashをread-only保存して保全。cleanup/一括commitしない。

H8 RUNNINGは観測日時付きsnapshotに限る。2026-10-14初版の203/592・8 workerを現在進捗や最終resultとしない。
原runner/observerの既存管理を尊重し、同期taskからserver process・SOURCE・grant・checkpointを操作しない。既存H8継続方針は新計算認可を含まない。

## 次の同期条件と終了

新しい原review/resultが公開された、H8最終結果が固定commitになった、明示承認scopeが変わった、未公開証拠が回収された、identity競合が発見された時に対象を絞って追補する。
server Codexは本正式版固定commitをread-only取得して原本/metadataを照合し、未取得事項を返す。全inventoryや大量hashを理由なく繰り返さない。
今回の情報統合は完了後MANDATORY_STOP。追加科学計算、詳細研究設計、source統合、原稿改訂は自動開始しない。
