# ローカル・サーバーA棚卸し統合と整理実行計画

作成：2026-10-26T17:48:25.409958+09:00。**PUBLIC_REDACTED_INTEGRATION／技術的な静的照合・将来計画のみ。削除・移動・branch整理の実行リストではない。MANDATORY STOP。**

Repository：`HIROMU1015/Partially-Randomized-Trotter`。入力branchは`codex/pr-environment-inventory-20261026`、取得HEADは`c38bef8aab03810d0fa0638eab2c1582ced6e30b`。指定済み`c38bef8aab03810d0fa0638eab2c1582ced6e30b`と一致し、親は`dc256b1e67433e27192457a2ed3882c61996b769`。元の8ファイルを変更せず、新規`inventory/integrated_local_server_a/`だけを追加する。サーバーA/BへSSH接続せず、サーバーBの実体・job・Git・cacheは調査/変更していない。

## 1. 結論と優先順位

425 worktree記録を統合したが、425件の独立研究や科学実験という意味ではない。元の保持分類を全件維持し、現時点で削除可能と最終確定したものは0件。サーバーAの12件は元の条件付き候補のまま、ローカルは元分類の削除候補0件のままである。

最初の確認対象は**SERVER_A-WT-0034 → SERVER_A-WT-0010 → 旧H4準備系10件**。12件すべてで、保持する代表HEADへのGit祖先到達を今回確認できた。現在の公開・clean・入力/raw/専用venvなし・限定control直接参照0といった既存事実は再調査せず、未記録依存・利用終了・外部custody責任と操作直前の差分へ確認を絞る。

次はローカルの**234件**を追加確認中心の計画群とした。内訳はKEEP_EVIDENCE 166件＋HOLD_UNKNOWN 68件。これは「234件削除可能」や保護解除の意味ではない。全件に具体的な残条件と担当を付け、SOURCE/controlがversionedであるという理由だけで作業用checkoutを永久保護しない一方、固定path契約やGit外原本があれば保全先行群へ戻す。

## 2. snapshotと元分類

| 元分類（変更なし） | LOCAL | SERVER_A | 合計 |
| --- | --- | --- | --- |
| KEEP_ACTIVE | 2 | 1 | 3 |
| KEEP_EVIDENCE | 302 | 21 | 323 |
| REMOVE_CANDIDATE_AFTER_APPROVAL | 0 | 12 | 12 |
| HOLD_UNKNOWN | 85 | 2 | 87 |
| 合計 | 389 | 36 | 425 |

ローカルsnapshotは2026-10-26 15:59–16:00 JST、サーバーAは同日16:05:57.818333 JSTに記録。元のGitHub一覧は各776 branch。今回のGitHub live観測は **2026-10-26T17:20:46.051686+09:00、779 branch**。776＋776を1552 branchと数えず、同一repositoryとbranch名で重複排除し、snapshot時のSHAと今回のSHAを別fieldに保持した。本計画用cloneと、その後に別Codexが作る環境は425件へ追加していない。

ローカルdirty31・detached22、サーバーA dirty1・detached1を引き継いだ。現在状態を再棚卸しした値ではない。操作する段階で、当該対象の状態差分を確認する。

### schema対応とUNKNOWNの意味

|統合概念|LOCALの元field|SERVER_Aの元field|解釈|
|---|---|---|---|
|worktree/common ID|worktree_id / git_common_directory_ref|worktree_id / git_common_directory|environment内のID。数字の一致を環境間の同一性に使わない|
|HEAD・公開状態|head_commit / publication_state|HEAD / remote_publication_state|HEAD、同名remote、他の公開tipへの到達を分離|
|dirty・untracked|working_tree_state、tracked_change_entries、untracked_group_entries|clean_dirty、status_entries付き文字列|directory内部の全file数ではない。元fieldを意味に対応するentry数へ変換|
|active|usage_status|run_status|未検出は利用終了の証拠ではなく、将来の再開予定もUNKNOWN|
|研究ID|study_classification、公開文書|research / series_ids|報告済み分類を保持し、既存indexの正確なID/commit bindingだけを追加。旧qc-c1と研究C C1は分離|
|SOURCE・結果・STOP|public_*_references_json|SOURCE_references / results_and_input_locations / STOP_references|公開参照がないものはUNKNOWN。manifestを置くcontainer SHAを科学SOURCE SHAと誤認しない|
|private raw|important_unversioned_data等|external_important_data等|unversioned pathあり＝raw内容確認済み、ではない。none identified＝不存在証明、でもない|
|backup|直接の確認fieldなし|private_raw_backup_status|両環境とも独立保管/復元性はUNKNOWNとして保持。公開HEADはrawのbackup保証ではない|
|絶対path依存|公開版は詳細非公開|control_absolute_path_reference_count|LOCALはUNKNOWN、Aの0は既存限定control範囲における観測値。全jobの非依存証明ではない|
|元分類|retention_classification|classification|original_classificationへ不変転記。新しい計画groupを別列に置く|

空の公開参照配列やUNKNOWNは、SOURCE・raw・STOPの不存在を意味しない。公開版で復元できないfieldは原本確認事項とした。全元fieldは入力commitで追跡でき、非公開原本の具体pathや内部環境値を補っていない。

## 3. Git同一性・参照重複・独立保全単位

**環境をまたぐ同一HEADのworktree組は0。同じbranch名をcheckoutしている組は1組だけ**：LOCAL-WT-0001とSERVER_A-WT-0002の`all-r-coherent-opt2-reoptimization`。SHAはそれぞれ`e098c54c78f589055082f9cfc2b13de50c90ca94`と`8b574c5431acf8d7fc2059ecc2c3b56ff19c37f0`で、後者は前者の22commit祖先（A固有0、LOCAL固有22）。同じbranch名だけで同じ状態としない。

同一HEADはLOCALで8群、SERVER_Aで1群、合計9群。以下のA/B/C/Dを分ける。

- **A：Git checkoutの重複候補**：SERVER_A-WT-0034と保持対象0030。同じhandoff HEADで、0034側には既存scopeで重要未保存データを検出していない。未記録依存とcustody責任の最後の確認が残る。
- **B：研究成果/資料の参照重複**：versioned pathとGit object identityが一致する参照838群。4021件のcommit:pathをGit metadataだけで照合した値で、独立rawやcohortの統合数ではない。今回比較した公開参照fieldでは環境横断の同一object参照群は0。既存study indexが複数環境から同じ成果を参照することと、各checkoutの直接参照が一致することも区別する。
- **C：独立に保全すべき証拠**：LOCAL-WT-0294/0298は同一HEADでも元棚卸しで別の実行出力namespaceが記録されている。SERVER_A-WT-0004の再構成confirmationは元入力探索STOPと別cohort、0005/0006の失敗pilotと回復も別履歴として保持。統計的独立性の新たな科学判定はしていない。
- **D：判定不能**：その他の同一HEAD群はdirty差、固定SOURCE path、別raw/STOP等の不足を残す。同一HEADを同じ科学結果という認定にしない。

| 群 | 区分 | worktree IDs | Git上の一致 | 保全上の意味 |
| --- | --- | --- | --- | --- |
| DUP-HEAD-01 | D_SCIENTIFIC_IDENTITY_UNDETERMINED | LOCAL-WT-0022, LOCAL-WT-0047, LOCAL-WT-0048, LOCAL-WT-0049, LOCAL-WT-0051, LOCAL-WT-0052 | b2e1bf65e21893b6c617223b42313623d3186f12 | 同一HEADでもdirty、SOURCE実行path、private出力の差がある/未確認。科学結果重複とは認定しない |
| DUP-HEAD-02 | D_SCIENTIFIC_IDENTITY_UNDETERMINED | LOCAL-WT-0042, LOCAL-WT-0043 | f98050e9402e8c65bb0f3dd27c7bd30d12fe7069 | 同一HEADでもdirty、SOURCE実行path、private出力の差がある/未確認。科学結果重複とは認定しない |
| DUP-HEAD-03 | D_SCIENTIFIC_IDENTITY_UNDETERMINED | LOCAL-WT-0050, LOCAL-WT-0202 | 054bb7b2b2ca788b47ebdc5c14c115f24c8b6379 | 同一HEADでもdirty、SOURCE実行path、private出力の差がある/未確認。科学結果重複とは認定しない |
| DUP-HEAD-04 | D_SCIENTIFIC_IDENTITY_UNDETERMINED | LOCAL-WT-0231, LOCAL-WT-0232 | 347bb0f34738613c0c58c1f638c8b17a0c71b1aa | 同一HEADでもdirty、SOURCE実行path、private出力の差がある/未確認。科学結果重複とは認定しない |
| DUP-HEAD-05 | C_DISTINCT_CUSTODY_EVIDENCE | LOCAL-WT-0294, LOCAL-WT-0298 | c7cde1868a6b38f9505ecc4f0858b98b49c506e8 | 元の非公開棚卸しで別実行出力namespaceを確認。HEAD一致で出力やSTOPを統合不可。統計的独立性は別途未確認 |
| DUP-HEAD-06 | D_SCIENTIFIC_IDENTITY_UNDETERMINED | LOCAL-WT-0341, LOCAL-WT-0343 | 3de42d3a3efe9ebc74dde8a1d0e1f1bdbadd607c | 同一HEADでもdirty、SOURCE実行path、private出力の差がある/未確認。科学結果重複とは認定しない |
| DUP-HEAD-07 | D_SCIENTIFIC_IDENTITY_UNDETERMINED | LOCAL-WT-0345, LOCAL-WT-0349 | 31e8391a56fbb1a0e64f477f78e242655329de99 | 同一HEADでもdirty、SOURCE実行path、private出力の差がある/未確認。科学結果重複とは認定しない |
| DUP-HEAD-08 | D_SCIENTIFIC_IDENTITY_UNDETERMINED | LOCAL-WT-0346, LOCAL-WT-0348 | 8192530b17d6bc4af195c734ff43a6e78c506942 | 同一HEADでもdirty、SOURCE実行path、private出力の差がある/未確認。科学結果重複とは認定しない |
| DUP-HEAD-09 | A_CHECKOUT_DUPLICATE_CANDIDATE | SERVER_A-WT-0030, SERVER_A-WT-0034 | c0a5b9692778878865cfa31d7df9a3bcc259f6c4 | 同一handoff HEAD。0034はclean・ignoredなし・観測直接control参照0。0030保持と未記録依存の最終確認が残る |

9群だけを対象にしても大幅削減にはつながらない。異なるHEADの準備用checkoutについて、公開refの保持と固有データの不在・path依存解消を条件に役割を集約する方が有望である。ただし「後続HEADの祖先」はコード履歴の包含であり、private raw/独立cohortの包含ではない。

## 4. 計画groupとローカル302/85件の絞り方

| 計画区分（現分類とは別） | LOCAL | SERVER_A | 合計 |
| --- | --- | --- | --- |
| Group 1：追加確認中心 | 234 | 12 | 246 |
| Group 2：保全・依存解消が先 | 135 | 15 | 150 |
| Group 3：当面維持する代表環境 | 5 | 8 | 13 |
| Group 4：B3/他環境の文脈確認待ち | 15 | 1 | 16 |

| LOCALの元分類 | Group 1 | Group 2 | Group 3 | Group 4 | 計 |
| --- | --- | --- | --- | --- | --- |
| KEEP_EVIDENCE | 166 | 118 | 3 | 15 | 302 |
| HOLD_UNKNOWN | 68 | 17 | 0 | 0 | 85 |
| KEEP_ACTIVE | 0 | 0 | 2 | 0 | 2 |

LOCAL Group 1はclean・公開HEADで、既存Git-status scopeでは科学的なGit外資産が未特定、独立cloneのGit本体削除や未知cache/専用環境の問題を切り離せるものを抽出した。共有案内へのsymlinkだけを持つものと、実venv/cache/editable SOURCE依存があるものを区別した。非公開の既存ローカル原本CSVを静的に参照しただけで、実worktreeやrawを再走査していない。

| LOCAL Group 1の研究単位 | 件数 | 確認対象の例 |
| --- | --- | --- |
| TRACK_B_ALGORITHM_HISTORY | 131 | LOCAL-WT-0054, LOCAL-WT-0055, LOCAL-WT-0056, LOCAL-WT-0057, LOCAL-WT-0058, LOCAL-WT-0059, LOCAL-WT-0060, LOCAL-WT-0061, LOCAL-WT-0062, LOCAL-WT-0063, LOCAL-WT-0 |
| RESEARCH_B_B2D_OUTER | 36 | LOCAL-WT-0271, LOCAL-WT-0272, LOCAL-WT-0273, LOCAL-WT-0274, LOCAL-WT-0275, LOCAL-WT-0276, LOCAL-WT-0277, LOCAL-WT-0278, LOCAL-WT-0279, LOCAL-WT-0280, LOCAL-WT-0 |
| QC_ALGORITHM_HISTORY | 18 | LOCAL-WT-0023, LOCAL-WT-0024, LOCAL-WT-0025, LOCAL-WT-0026, LOCAL-WT-0027, LOCAL-WT-0028, LOCAL-WT-0029, LOCAL-WT-0030, LOCAL-WT-0031, LOCAL-WT-0032, LOCAL-WT-0 |
| N1_N3_ALGORITHM_HISTORY | 16 | LOCAL-WT-0003, LOCAL-WT-0004, LOCAL-WT-0005, LOCAL-WT-0006, LOCAL-WT-0007, LOCAL-WT-0008, LOCAL-WT-0009, LOCAL-WT-0010, LOCAL-WT-0011, LOCAL-WT-0012, LOCAL-WT-0 |
| MANUSCRIPT_FIGURES_INDEXES | 7 | LOCAL-WT-0203, LOCAL-WT-0204, LOCAL-WT-0205, LOCAL-WT-0206, LOCAL-WT-0207, LOCAL-WT-0208, LOCAL-WT-0209 |
| RESEARCH_C_C4_CIRCUITS | 7 | LOCAL-WT-0319, LOCAL-WT-0320, LOCAL-WT-0325, LOCAL-WT-0326, LOCAL-WT-0327, LOCAL-WT-0328, LOCAL-WT-0329 |
| TRACK_B_G6 | 6 | LOCAL-WT-0109, LOCAL-WT-0110, LOCAL-WT-0111, LOCAL-WT-0112, LOCAL-WT-0113, LOCAL-WT-0114 |
| 既存アルゴリズム開発／中間実装・診断・停止履歴 | 4 | LOCAL-WT-0002, LOCAL-WT-0046, LOCAL-WT-0270, LOCAL-WT-0330 |
| RESEARCH_C_C1 | 4 | LOCAL-WT-0321, LOCAL-WT-0322, LOCAL-WT-0323, LOCAL-WT-0324 |
| REPRESENTATION_ALGORITHM_HISTORY | 3 | LOCAL-WT-0042, LOCAL-WT-0044, LOCAL-WT-0045 |
| DF_NATIVE_TRACK_A | 2 | LOCAL-WT-0020, LOCAL-WT-0021 |

優先するまとまりは、図表・文書用7件、旧QC/N1/N3の完了記録・preparation、研究Bのpause handoffが参照する中間作業など。次に件数が大きいTrack B履歴131件を研究用途ごとに確認する。旧G6のB3名を持つものはこの群から除外した。同じ担当者・同じ用途の利用終了をまとめて確認し、各IDのpath契約/未保存物を個別に閉じる。

LOCAL Group 2の135件では、独立clone Git本体56件、環境/cache等49件、Git外資産保全38件、未公開commit1件のgateがある（重複するため合計しない）。元KEEP_EVIDENCE全302件を永久保持とせず、確認不足の理由ごとに処理する。59独立cloneをlinked checkoutと同じ削除方法で扱わず、別ref/必要objectの保管を確認するまではGit本体を残す。

Group 3は削除候補ではない。当面の代表環境を以下に絞って明記したが、それ以外の元KEEP_EVIDENCE/HOLDの保護が解除されたわけではない。

| 当面維持するID | branch | 理由 |
| --- | --- | --- |
| LOCAL-WT-0001 | all-r-coherent-opt2-reoptimization | 現在の作業/共有Git本体、C1供給環境、または研究handoffの代表入口 |
| LOCAL-WT-0231 | codex/research-c-c1-h4-xdf9-cost-screen-preparation-20261026 | 現在の作業/共有Git本体、C1供給環境、または研究handoffの代表入口 |
| LOCAL-WT-0250 | codex/research-c-c1-integration-next-preparation-20261026 | 現在の作業/共有Git本体、C1供給環境、または研究handoffの代表入口 |
| LOCAL-WT-0252 | codex/research-c-c1-stage1-compatibility-20261025 | 現在の作業/共有Git本体、C1供給環境、または研究handoffの代表入口 |
| LOCAL-WT-0285 | codex/research-b-final-pause-handoff-20261026 | 現在の作業/共有Git本体、C1供給環境、または研究handoffの代表入口 |
| SERVER_A-WT-0001 | main | 使用中の端末・言語サーバーあり。AGENTS.md変更、未追跡実行領域・raw・報告・共有Git本体を保持。 |
| SERVER_A-WT-0002 | all-r-coherent-opt2-reoptimization | cleanだが専用venv、ignored ground-state cache、checkpoint・log・worker結果が残存。公開HEADは先行版の祖先でありraw保全とは別。 |
| SERVER_A-WT-0009 | pf-common-domain-execution-20261022 | 公開結果に加えてmain外部実行領域のraw・grant・STOPに依存。外部原本の独立backup未確認。 |
| SERVER_A-WT-0011 | pf-common-formal-adoption-20261022 | 採択済み共通PFの代表checkout。公開版とlocal-only採択commitを区別して保持。 |
| SERVER_A-WT-0012 | pf-common-h2-execution-20261022 | 既存controlからこのcheckout内の入力・保存結果への絶対パス参照が多数残る。private原本の供給元。 |
| SERVER_A-WT-0013 | pf-common-h6-metadata-execution-20261022 | H6中断・回復のSOURCE archive、ignored log、外部ledger/STOPを保持。中断前親processの終了履歴には未確定部分あり。 |
| SERVER_A-WT-0016 | GITHUB-ANON-001 | 既存23系列・2834参照・private保全索引を結ぶ代表worktree。新報告がこの索引を参照するため保持。 |
| SERVER_A-WT-0030 | track-a-h4-new-server-handoff-20261007 | 永続handoffの代表checkoutとしてPERSISTENT_PATHSが指定。bundle・元入力・run05停止記録との依存を保持。 |

## 5. サーバーAの12条件付き候補

全12件の公開HEAD到達、clean、重要untrackedなし、ignored raw/入力/専用venvなし、本人process cwd未検出、既存限定control直接参照0は**確認済みの元事実**として引き継ぐ。新しい全件raw hashや同一の広範調査を要求しない。全12件のHEADが保持予定0009/0011または0030のHEADから到達することを今回Git metadataで確認した。branchは別途残し、固定SOURCE・結果commitの保持refを確認する。

| 確認順位 | SERVER_A ID | branch | HEAD（完全SHA） | 保持予定の後続HEAD | 残る固有確認 |
| --- | --- | --- | --- | --- | --- |
| 1 | SERVER_A-WT-0034 | DETACHED | c0a5b9692778878865cfa31d7df9a3bcc259f6c4 | SERVER_A-WT-0009, SERVER_A-WT-0011, SERVER_A-WT-0030 | SERVER_A-WT-0030を保持し、同一handoff HEADの参照・bundle原本の保管責任が0034側にないこと、別jobの固定pathに0034がないことを確認 |
| 2 | SERVER_A-WT-0010 | pf-common-domain-preparation-20261022 | 3728a618d7f09e67a070632ded1f27dab54b21a4 | SERVER_A-WT-0009, SERVER_A-WT-0011 | SERVER_A-DEP-0017はmain側custodyの外部参照。候補本体のraw有無を再調査せず、main供給元の保持と未記録依存・利用終了を確認 |
| 3 | SERVER_A-WT-0021 | track-a-h4-geometry-contract-20261006 | 7c1a3d43f61c5501a9e79206b7c60933f94b1077 | SERVER_A-WT-0009, SERVER_A-WT-0011, SERVER_A-WT-0030 | 既存control範囲の参照0は確認済み。未記録の他job/他host依存と再開予定だけを確認。並行して保持代表へのGit到達と元branch保持を使う |
| 4 | SERVER_A-WT-0022 | track-a-h4-geometry-contract-v2-20261006 | b662dbd72e49fa713a25c716f323843e547e973b | SERVER_A-WT-0009, SERVER_A-WT-0011, SERVER_A-WT-0030 | 既存control範囲の参照0は確認済み。未記録の他job/他host依存と再開予定だけを確認。並行して保持代表へのGit到達と元branch保持を使う |
| 5 | SERVER_A-WT-0023 | track-a-h4-geometry-input-generation-authorization-20261006 | 5245a29ca26cad7421640410934907647459b822 | SERVER_A-WT-0009, SERVER_A-WT-0011, SERVER_A-WT-0030 | 既存control範囲の参照0は確認済み。未記録の他job/他host依存と再開予定だけを確認。並行して保持代表へのGit到達と元branch保持を使う |
| 6 | SERVER_A-WT-0024 | track-a-h4-geometry-input-generation-cpu-launch-proposal-20261006 | 7c641ea8a2610e40667a311ebc48682702cc3a82 | SERVER_A-WT-0009, SERVER_A-WT-0011, SERVER_A-WT-0030 | 既存control範囲の参照0は確認済み。未記録の他job/他host依存と再開予定だけを確認。並行して保持代表へのGit到達と元branch保持を使う |
| 7 | SERVER_A-WT-0025 | track-a-h4-geometry-parallel-source-20261006 | 88461f3930b9fef511739f91edae88231c33a3f5 | SERVER_A-WT-0009, SERVER_A-WT-0011, SERVER_A-WT-0030 | 既存control範囲の参照0は確認済み。未記録の他job/他host依存と再開予定だけを確認。並行して保持代表へのGit到達と元branch保持を使う |
| 8 | SERVER_A-WT-0026 | track-a-h4-geometry-resource-observer-fix-20261006 | a1b0ba5e1ae14c2fd7c345a4d979b85b1eff2538 | SERVER_A-WT-0009, SERVER_A-WT-0011, SERVER_A-WT-0030 | 既存control範囲の参照0は確認済み。未記録の他job/他host依存と再開予定だけを確認。並行して保持代表へのGit到達と元branch保持を使う |
| 9 | SERVER_A-WT-0027 | track-a-h4-geometry-resource-server-prep-20261005 | c2ab34fed49bb1fb104d39fe83b36858a2c92c2a | SERVER_A-WT-0009, SERVER_A-WT-0011, SERVER_A-WT-0030 | 既存control範囲の参照0は確認済み。未記録の他job/他host依存と再開予定だけを確認。並行して保持代表へのGit到達と元branch保持を使う |
| 10 | SERVER_A-WT-0028 | track-a-h4-geometry-source-20261006 | 4d5eba454dda06bc2735730cf7c3f132e456db29 | SERVER_A-WT-0009, SERVER_A-WT-0011, SERVER_A-WT-0030 | 既存control範囲の参照0は確認済み。未記録の他job/他host依存と再開予定だけを確認。並行して保持代表へのGit到達と元branch保持を使う |
| 11 | SERVER_A-WT-0029 | track-a-h4-input-generation-storage-review-20261006 | 2094c07eb445463ea2dbcd574a6124a1d90f86a8 | SERVER_A-WT-0009, SERVER_A-WT-0011, SERVER_A-WT-0030 | 既存control範囲の参照0は確認済み。未記録の他job/他host依存と再開予定だけを確認。並行して保持代表へのGit到達と元branch保持を使う |
| 12 | SERVER_A-WT-0031 | track-a-h4-signal-compile-plan-review-20261006 | 8cbbf0285cc42be2daf6cf7054e6c048dcd2cd72 | SERVER_A-WT-0009, SERVER_A-WT-0011, SERVER_A-WT-0030 | 既存control範囲の参照0は確認済み。未記録の他job/他host依存と再開予定だけを確認。並行して保持代表へのGit到達と元branch保持を使う |

確認順位1の0034は既存0030を保持でき、具体的なcheckout重複候補として最も説明しやすい。順位2の0010は外部custodyがmain側のSERVER_A-DEP-0017であり、候補checkout内のrawがないという観測と分ける。外部原本を動かさずmain供給元を保持できれば、候補解除だけのためにその原本の再移送・全hash・新backupを一律に要求しない。固有データや固定pathを失う経路が判明した候補に限り、必要な保全措置を先に具体化する。

残る10件は旧H4準備・計画・SOURCE構築履歴。公開履歴は保持できても、別jobが固定pathとして使う可能性と再開予定は未確認である。12件とも現在削除可能と保証せず、元の条件付き分類を保持する。

## 6. 未公開commit・raw・SOURCE・STOPの保全リスク

| 環境 | branch/参照元 | 保護commit | 保護場所・役割 |
| --- | --- | --- | --- |
| LOCAL | track-b-cross-scientific-review-record-20261019 | 3bd4eb9fadb1be69e39f48903293391acdac931f | LOCAL-WT-0053; SNAPSHOT_UNREACHABLE_UNIQUE_1 |
| SERVER_A | codex/pr-pf-cng-technical-investigation-20261022 | c4cb5f3c9007f84b8d3798be17398ee2ea498e0f | SERVER_A-WT-0018; SNAPSHOT_LIVE_HEAD_UNREACHABLE_API_404 |
| SERVER_A | codex/pr-pf-cng-technical-investigation-20261022 | f5ae5d6d3fd228f862f8be3eea6c1fb4c0f0ca37 | SERVER_A-WT-0018; SNAPSHOT_UNPUBLISHED_SOURCE_API_404 |
| SERVER_A | pf-common-formal-adoption-20261022-local | 60b76576263bb0c31b4a3433a306747d62dc5789 | NONE_CHECKED_OUT; SNAPSHOT_LOCAL_ONLY_API_404 |
| SERVER_A | pf-common-paper-publication-draft-20261023 | 8f0ff62b2bfc6676c31773261b160e3935ffcdae | SERVER_A-WT-0014; SNAPSHOT_LIVE_HEAD_UNREACHABLE_API_404 |

保護対象は**5 distinct commit（LOCAL 1＋SERVER_A 4）**。Aの4件にはcheckout HEADだけでなく0018のSOURCE commitと、checkoutのないlocal-only採択branchを含む。同じsubjectの公開採択commitと置換しない。元snapshotの非到達/API404を記録したもので、全世界のclone/backup不存在や将来も非公開という保証ではない。今回これらの内容を取得/公開/pushしていない。

保存依存は以下を別扱いにした：公開済みversioned参照、未公開commit、Git外raw/control、archive/原本、SOURCE/input/grant/STOP、external custody、shared Git object store、venv/cache/editable SOURCE、絶対path依存、保全状態UNKNOWN。各worktreeのstorage_dependency_categories_jsonと残条件へ対応付けた。

- AのH2転送記録はSERVER_A-DEP-0306に`MANIFEST_ONLY_NOT_TRANSFERRED`、archive_created=false、transfer_executed=falseとあり、417行のstorage表には対応するcontrol記録SERVER_A-DEP-0264もある。両IDは同一成果物だと推定せず、A担当が原本で関係を確認する。manifest存在は転送完了・backupではない。
- Aの9 custody rootは存在が報告されるが、独立backupと復元性は未確認。過去台帳のfile/byte数を今回の再計数や削減容量へ流用しない。
- Aの独立Track B clone `SERVER_A-GIT-0002` / WT0036は、主Gitのobject store（DEP-0304）をalternateとして借用する。独立clone＝独立backupではない。35 worktreeの共有本体GIT-0001とそのobjectsを保持する。
- Aの共有venv DEP-0294、外部editable SOURCE DEP-0296、WP11専用環境、temporary領域のhandoff/controlは別依存。別repositoryの内容棚卸しや共有env変更を今回行っていない。
- 長時間研究の外部locator DEP-0291やPauli最新rawはAで所在未確認。ローカルに同じSOURCE/result参照があることだけで、その原本が存在するとは補わない。各実行担当のprivate台帳で保管責任を確認する。
- 旧H4 run01初期化失敗、run02 INPUTS_FROZEN_STOP、run03 MONITOR_STOP_NO_AUTOMATIC_RETRY、run04/05 MONITOR_INTERVAL_FRESHNESS_STOP、H6中断/回復を保持。後続成功や新しい集計でSTOP・grant消費・未確定の親process終了履歴を置換しない。
- LOCALのdirty31、detached22、ignored科学path、SOURCE凍結環境、refなしbare object packも保護。Gitの公開HEADやcleanだけでは、これらの復元性や再開条件を保証できない。

temporary領域には消失リスクがあるが、今回データ移動・保管先変更・再圧縮・追加backupは実行していない。容量は記録が不足するため削減byte数を推測しない。

## 7. 研究成果・branchへの接続

| 研究単位 | 代表入口branch/環境 | 固定参照と役割 |
| --- | --- | --- |
| DF・Pauli統合 | codex/df-pauli-evidence-paper-plan-20261026 | [RESEARCH_SYNTHESIS](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6e3436a16b13e539b0faa2a0a3286f6ab3549543/RESEARCH_SYNTHESIS.md) / [EVIDENCE_MATRIX](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6e3436a16b13e539b0faa2a0a3286f6ab3549543/EVIDENCE_MATRIX.csv)。科学的採択を更新しない |
| DF長時間・高次PF/PR | codex/h46-long-time-pf-pr-20261025 | [REPORT](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/REPORT.md)。科学SOURCE・result・探索/確認cohortを別扱い |
| Pauli-JW | codex/pauli-jw-research-summary-20261026 | [総合資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b6c899698645a4ccdaa65c0c5e8d706fd9d1cc2b/docs/research/pauli_jw_resource_evaluation_summary.md)。新規計算・結果再解析なし |
| Track A研究索引・既存CEM/ledger | codex/track-a-data-systematization-20261024 | [索引README](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/731b3d6f3572c6aa7950a579b68251fec9a5218b/docs/data_collection/2026-10-24_track_a_research_data/README.md)。study_index、source_catalog、既存CEM v3・ledger v5への固定参照を再利用 |
| DF共通PF・原本供給 | SERVER_A-WT-0009 / 0011 / 0012 / 0013 | 公開共通PF、採択、H2供給元、H6来歴を分離。HEAD公開と外部raw保管を混同しない |
| 研究B・B2-D/outer | LOCAL-WT-0285 / codex/research-b-final-pause-handoff-20261026 | 既存成果とpauseのhandoff。サーバーBのDF-native B3停止の根拠にはしない |
| 研究C C1/C4 | LOCAL-WT-0250 / 0252、各preparation/science/results branch | C1の現行integrationとvenv供給元を当面保持。旧qc-c1等とは別の研究単位 |
| Track B/G6および旧アルゴリズム | SERVER_A-WT-0036、LOCALのTRACK_B_G6/ALGORITHM_HISTORY群 | 既存PROJECT_MAP、公開SOURCE/control、停止済み履歴を参照。G6のB3名をDF-native L_D=0 B3と同一視しない |
| PR2・N1/N3・QC履歴 | LOCAL-WT-0022 / 0005 / 0032の入口と各固定HEAD | PR2 baselineのdirty、N3 closure、QC strategy handoffを役割別に保持。後続commitへの包含はGit履歴と未保存資産を別に確認 |
| 旧H4永続handoff | SERVER_A-WT-0030 | 同一HEADのtemporary copy 0034と区別して保持。元入力・bundle・run05 STOPの責任を残す |

43行のEVIDENCE_MATRIX、58行のstudy_index、80参照のsource_catalogをmetadataとして再利用した。研究条件、勝敗、係数、比、SE、順位は再計算/再評価していない。既存CEM・ledgerの採択状態は固定参照として残し、新しい科学claimの採否を判断していない。SOURCE manifestのURLを置くcommit、実科学SOURCE、result、cohortは別field・別scopeで保持する。

BRANCH_RECONCILIATION.csvはlocal branch529行、remote-tracking3531行、symbolic HEAD62行、その他ref51行、共有GitHub branch779行、保護commit5行を区別する。529などは環境/common単位のref数で、固有branch数ではない。その他refとsymbolic HEADをGitHub branch数へ足さない。各branchに代表入口、SOURCE/STOP参照、実行中の報告、未公開固有commit、private/Git外保全、中間preparation、後続保持HEADの祖先、複数環境、B3文脈/他環境確認待ちの区分を複数付与する。

内部hostを含むbranch名3件はGITHUB-ANON-*へ置換し、対応表は非公開。Aで伏せられた2branchは、元LOCALのGitHub観測で同じSHAを持つbranch名が一意だったことと現在のlive refを静的照合して同じ共有行へまとめた。匿名名を実際のGit ref文字列や環境間のworktree IDとして使わない。公開URLにhost成分がある参照は省略し、存在を推測で補わない。

**remote branchの最終削除候補は0。** ローカル・AのcheckoutがなくてもサーバーBで利用され得る。GitHub branch整理は三環境照合後の別計画とし、mainへの一括mergeは不要。削除前には、重要なcommit→保持する永続ref→Git到達を確認する。固定SHAを文書に記しただけでGitHub上の将来保持を保証しない。今回、タグ追加、ref移動、科学branchのmerge/rebase/cherry-pickは行っていない。

## 8. サーバーA担当Codexへ渡す追加確認

1. **12候補だけの残条件**：上表の利用終了/再開予定、未記録job・他host参照を原本で確認する。snapshot済みのclean/ignored/control検査はやり直さず、操作する段階の差分確認に限定。0034は0030の保持とcustody責任、0010はmain/DEP-0017供給元の保持を確認する。
2. **未公開4commit**：c4cb5f3…/f5ae5d6…、60b7657…、8f0ff62…に必要な保持ref/原本保管を既存GitとSOURCE catalogueで確認する。GitHub公開可否はユーザー判断。既存証拠があるので世界中を再探索したり科学SOURCEを再実行しない。
3. **H2の転送未完了**：DEP-0306とDEP-0264の原本対応、転送対象と原本供給元、現在の受領/保管状態を既存manifestで確認。未完了なら未完了として維持し、この依頼を転送開始の許可としない。
4. **path依存**：A0001/0012/0020/0032/0033/0035等の既存control参照を使い、12候補や集約対象へのincoming参照だけを対象化。fixed contractがpath保持を要求する場合は、checkoutを保持するか別途契約変更の判断へ戻す。
5. **共有object/env**：GIT-0002→GIT-0001 alternate、DEP-0294/0296、WP11とtemporary領域の依存が残ることを前提に、代表環境を削らない計画を確認。共有設定変更・object切離し・venv作直しは今回認可しない。
6. **長時間/Pauli原本**：既存locatorを実行担当へ照会し、実在/保管責任/STOP所在だけを確認。公開集計からrawの復元を仮定せず、数値再解析や追加samplingを開始しない。
7. A旧棚卸しは2026-10-25版共通指示を未取得・未読と報告している。本統合作業ではローカル実在版の§8.4・§10等を確認して適用した。過去A報告を読了済みへ書き換えない。

## 9. サーバーB完了後の照合事項

DF-native **L_D=0 B3（H4/H6、現在実行中）** と、旧 **Track B/G6/W2CでB3と呼ぶ開発系列** は別文脈である。現在のbranch名だけで同じ研究と認定しない。両方を必要に応じて保持し、B担当が自身の棚卸しで使用branch・HEAD・実科学SOURCE・input/cohort・raw/control・shared store/venv依存を報告した後で照合する。

今回のlocal/A資料から現行DF-native B3の実際のbranch、SOURCE SHA、保存先、実行状態を追加推定しない。全GitHub branchのB利用は未調査。Bの実行中job、cache、raw、SOURCE、grant/ledger/marker/STOP、Git管理領域には触れず、後日の確認事項は調査の提案であって現在のB停止・再開・移動の指示ではない。

## 10. 実際に整理する段階の安全な順序（今回未実行）

1. **各環境担当**：CLEANUP_CANDIDATESのGroup 1から、既存確認を引き継いで残条件を閉じる。まずA0034、A0010、旧H4準備10件。次にLOCALの文書/図表・過去の中間実装を用途単位で確認する。
2. **各環境担当＋ユーザー**：固有データがある対象だけ、SOURCE/input/raw/失敗履歴/STOP/grant/ledger/markerと必要Git object・環境の保全先/復元単位を具体化する。実際の追加保全・移動/保管先変更は別途指示後。原本をredacted public copyで置換しない。
3. **統合担当**：候補ID、環境、原本で解決した完全path、保持ref、HEAD、branch保持、SOURCE path契約、依存、必要保全、操作方式を具体的な対象表にする。匿名IDの数字だけで実行先を選ばない。
4. **ユーザー**：対象別の実際の登録解除/削除・集約範囲を判断して許可する。今回の文書承認や計画groupを削除許可として扱わず、承認証明書・認証基盤・専用ledgerは新設しない。
5. **該当環境担当（その許可後のみ）**：当該対象のHEAD/status/利用/参照の差分を確認し、保持予定環境・原本を先に確認する。変化があれば対象を保留し、別対象へSTOPを迂回しない。共有Git本体や他worktreeのデータを一括処理しない。
6. **該当環境担当**：許可されたcheckoutだけを個別に整理し、残したref・SOURCE・raw/controlと依存環境が維持されたことを確認する。独立cloneの削除はlinked worktreeの解除と別判断。無関係job/環境を操作しない。
7. **統合担当（Bを含む三環境照合後）**：GitHub branch整理案を別途作成する。重要commitの保持ref到達を実確認し、科学SOURCE/独立cohort/未公開commitを保護する。branch削除・mainへのmergeを作業数削減の代替にしない。

### 各worktreeに付けた解除条件

| 確認ID | 不足条件を閉じる内容 |
| --- | --- |
| USE_ENDED | 対象所有者が利用終了・再開予定・別Codexの利用を確認。限定process一覧の未検出だけでは閉じない |
| PRIVATE_ASSET_SCOPE | 非公開原本の既存索引で、このcheckout内と外部custodyの所有・参照を区別。private rawの有無は推測しない |
| SOURCE_PATH_CONTRACT | 固定SOURCEの絶対path条件・grant/ledger/marker/STOPの参照を既存契約で確認。SOURCE/controlの公開だけでpath依存解消とはしない |
| KEEP_GIT_STORE | 共有Git本体・objects・他worktree登録を保持。checkoutだけの整理とGit本体削除を切り離す |
| RETAIN_REFS | branch/SOURCE/result commitを保持するrefと祖先到達を確認。SHAを文書へ記すだけでは永続保管にならない |
| PRESERVE_UNTRACKED | dirty・untracked・ignored科学資産と失敗/STOP履歴の保持単位・既存保管先・復元条件を確定。移動/追加backupは別の実行指示後 |
| COMMIT_CUSTODY | GitHub未到達commitと未公開SOURCEの保持方針を決定。公開かprivate独立保管かはユーザー判断 |
| ENV_DEPENDENCY | 共有venv/cache・editable SOURCE・symlinkの参照元と復元条件を確認。参照先環境や外部checkoutは変更しない |
| CLONE_OBJECT_CUSTODY | 独立cloneの全ref/必要object/未公開履歴の保管を確認。clone自体の削除とlinked checkoutの解除は別作業 |
| OTHER_ENV_SCOPE | 他環境がこのcheckout/固定SOURCEを実参照しないか確認。branchを他hostが利用することとlocal directory依存を区別 |
| B3_DISTINGUISH | DF-native L_D=0 B3と旧Track B/G6 B3を別研究としてSOURCE・cohort・使用refを照合。名称だけで同一視しない |
| BEFORE_ACTION_DELTA | 具体的な操作の承認後にHEAD/status/使用/参照の変化だけ再確認。既存確認済みの全調査をやり直さない |
| REPRESENTATIVE_RETAIN | 当面の代表環境/供給元。依存先と保管責任が解消するまで保持し、同一HEADを理由に一括整理しない |

## 11. 未確定事項・検証・成果物

未確定なのは、利用終了、非公開rawの保管同一性/独立復元、未記録path依存、固定契約に必要なSOURCE checkoutの存続、未公開commitの最終保管方針、Bの実参照、ディスク削減容量。これらをUNKNOWNのまま明示した。全425件を同じ強さで永久保持する判断も、Group 1全件を削除可能とする判断もしていない。

既存CSV処理・公開化処理のCSV I/O、ID置換、公開参照の限定という方法を再利用し、schema差の対応と計画列だけを追加した。固定study-index生成コードも静的に確認したが、entry pointや科学runnerをimport/実行せず、原本の再生成を行っていない。今回の新コードは独立した非公開作業領域に置き、公開するのは次の4ファイルだけである。

- [INTEGRATED_INVENTORY.csv](INTEGRATED_INVENTORY.csv)：425 worktree。元分類・元根拠、SOURCE/result/STOP参照、保存依存、UNKNOWN、提案group・解除条件を分離。
- [CLEANUP_CANDIDATES.csv](CLEANUP_CANDIDATES.csv)：425件すべてに将来の処置・担当・確認順序・保全・リスクを付与。Group 3/4は現在の整理候補外も含む。削除実行リストではない。
- [BRANCH_RECONCILIATION.csv](BRANCH_RECONCILIATION.csv)：環境/common別refと共有GitHub一覧、5保護commitの対応。各時刻のSHAを保持し、同じGitHub branchを二重計上しない。
- [INTEGRATED_REPORT.md](INTEGRATED_REPORT.md)：本報告。元[LOCAL](../local/LOCAL_INVENTORY_REPORT.md) / [SERVER_A](../server_a/SERVER_A_INVENTORY_REPORT.md)は不変。

公開前に425件の一意性・分類不変・12候補の網羅・branch種類/重複排除・JSON/CSV構造・公開情報範囲を検査し、元8ファイルの内容不変と新規directoryだけの差分を確認する。private原本・対応表、絶対パス、内部host/IP、認証情報、非公開raw/環境構成はGitへ追加しない。今回の検査は科学的性能の再検証やraw全件byte保証ではない。

**専用branchへ通常のfast-forward push・local/remote SHA照合後mandatory STOP。実際のworktree/branch削除・prune・gc・merge・移動・科学計算・サーバーA/B操作は行わない。**
