# ローカル75件：サーバーB依存確認・B3非干渉調査

2026-10-27 JST。静的照合snapshot：2026-10-27T00:33:10.186224+09:00。PUBLIC_REDACTED。

**75件の静的調査票を作成した。非公開パス対応表は未受領のため、実パス照合完了は0/75件、主判断は全75件 `PATH_IDENTITY_UNRESOLVED`。物理依存の確定0件、物理依存なしの確定0件。先行64件のサーバーB条件は未解消、旧SOURCE未解決11件は別枠で維持する。**

## 1. 入力・正式指示・既存資産の再利用

正確な対象集合は固定commit `e5651ab7b7f1682dac51a756e6b277f4a2d9c217` の [SERVER_B_75_DEPENDENCY_REQUEST.csv](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e5651ab7b7f1682dac51a756e6b277f4a2d9c217/inventory/local/pre_cleanup_75_20261026/SERVER_B_75_DEPENDENCY_REQUEST.csv) の75行である。連番やbranchから対象を追加していない。ID一意性、75件のreadiness表・A結果とのID集合、branch・完全HEADの対応、64件計画との集合関係を確認した。LOCAL-WT-0283/0286は追加しない。

同commitの [統合報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e5651ab7b7f1682dac51a756e6b277f4a2d9c217/inventory/local/pre_cleanup_75_20261026/LOCAL_75_INTEGRATED_REPORT.md)、LOCAL_75_CLEANUP_READINESS.csv、LOCAL_64_CONDITIONAL_CLEANUP_PLAN.csv、batch_cleanup_75・archive_followup_68・path_and_historical_source_followupの既存報告・CSV/manifestを取得・確認した。manifestは2120行（図表実コピー2052＋他68件の当時の計画行）、8284行（後続の実コピー索引）を区別し、再コピー・payload再hashはしなかった。

サーバーA固定commit `5eef8de248d454cc2497ef9631bfc171b1a78aae` を独立cloneへ取得し、[A結果2資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/5eef8de248d454cc2497ef9631bfc171b1a78aae/inventory/server_a/local75_actual_path_reconciliation_20261026)が今回固定入力内の同資料と差分なしであることを確認した。Aの71件＋論理参照4件は過去のA scopeとしてCSVへ保持し、Bの判定へ転用していない。

正式指示 `research_project_instructions_unified_2026-10-25.md` の実在ファイルを確認し、特に§8.1、§8.4、§10を適用した。承認証明・grant発行・承認原文保存を新設していない。既存個別科学契約・SOURCE lock・one-shot・retry/resume・STOPは変更しない。

既存CSV、保全manifest、研究索引、既存SOURCE hash宣言、入力設定、既知の実行記録を再利用した。取得したinventoryには実行可能な照合helperは含まれず、Aの非公開helperも未受領だったため、必要なCSV join、選択済みテキストの照合、既知metadataの読取り、報告生成だけを標準ライブラリの小さな非公開scriptへ実装した。科学moduleのimport・実行は0。新しい全環境棚卸しや研究pipelineの再実装は行っていない。

## 2. B3の現在状態と区別

対象はlinear H4/H6・1Å・STO-3G・tol-only DF10/19、全二体DF項をrandom側へ置く `L_D=0` のDF-native B3である。旧Track B/G6/W2Cの同名B3とは別系列として扱った。科学的性能・費用・順位の評価は本調査の対象外である。

本人UIDのprocess snapshotと既知のuser serviceのread-only照会で、引継ぎ後の取得controllerと終了監視controllerがactive/running、科学Python workerが34本稼働していた。旧controllerのserviceはinactiveであり、旧起動記録のservice名を現在の実行主体と取り違えていない。既存のfinal_statusは `SCIENTIFIC_ACQUISITION_RUNNING`。これは一時点の稼働確認であり、全試行成功・最終終了・科学的受入れを示さない。既存statusの数値を再集計していない。

既存記録には中断・SIGBUS・固定seed復旧・高速化引継ぎの履歴がある。今回それらを実行・再開・変更していない。既存の「1回の復旧、後続失敗はSTOP、recursive retryなし」という条件を維持した。現在job識別子・process番号は非公開原本のみへ保存した。

## 3. SOURCE・provider・input・共有環境

- 実行scriptの所在地はB3専用worktree。観測HEADは `811a93cee53f16d47b6e8ce185741f1fffcfdf85`。checkout HEADと科学SOURCEを同一視しない。
- 既存provenanceはbase `155529e20a8fac39e00786c40b57830b71203c4f`、実行/復旧SOURCE `ae3d301e2318a3f9f1331c047c68d310771b6849`、provider SOURCE `c31ef424175c02379eafebe8fc6c7548c1d9aabe` を記録する。providerは `.git` を持たない保存snapshotであり、架空のprovider HEADを付与していない。
- B3のsetupは既存設定のprovider `src` をimport検索先へ先頭挿入し、保存H4/H6 snapshot・receiptを参照する。既存のprovider/runtime sealed-file索引とinputの宣言hashを再利用した。5つのinput、設定、provider、B3/基準private保存先等、既知12参照先の存在・realpath/file metadataを確認した。数値配列はロード・再hashしていない。
- `new_source_files` の10本と現行最適化の3本、計13本の小さいscriptについて、現在bytesのSHA-256が既存宣言と一致した。これを科学的正しさや全runtimeの完全性の証明へ拡張しない。
- B3は共通Git管理領域を利用する。観測した共通storeに `objects/info/alternates` はなかった。対象75件と同名のローカルbranch refはこのBの共通storeでは検出されなかったが、commit/objectの不存在や物理checkout不使用を意味しない。
- 使用venvの既知 `.pth` / editable metadata 3件を静的確認し、別研究リポジトリへのeditable参照を検出した。設定済みproviderの優先順位とfallback環境は区別した。実行中processへのコード注入やmodule照会はせず、実際にロード済みの全module `__file__` は未検証である。
- 自分と既知B3 processの可視mount metadata、既知参照先のrealpathを非公開に保存した。これだけでLOCALとのexport/bind/mount同一性や未記録aliasの不存在は証明できない。

これらはB3が必要とするB側の参照を確認した結果であり、非公開対応表が未受領のLOCAL75実体との非依存認定ではない。SOURCE・input・基準結果・共通Git・venv/editable供給元・private保存先はいずれも保持対象である。

## 4. B_STATIC_1の範囲と一致の意味

既知の研究索引、validation manifest、B3関連script、直接関係するprovider/input設定、既存中断/provenance記録の38ファイル、約3.26 MBを静的に確認した。固定75件のbranch・完全HEADおよび12桁HEAD prefixの参照は、この選択済みテキスト内では検出されなかった。対象IDごとの結果は [CSV](SERVER_B_LOCAL75_DEPENDENCY_RESULTS.csv) に保持した。

参照元と文脈を保持した物理path候補は非公開に保存した。文字列一致、同じ実体へのアクセス、現在/再開時の必須依存の3段階を区別する。今回はLOCAL正規pathがないため、その候補との物理path照合は実施していない。**文字列不一致0件という結果すらまだ得ていない。** branch/SHAへの参照非検出や、B3と異なる研究系列であることから、物理非依存を推定していない。

本人の閲覧可能な研究processのcwd/引数は限定snapshotとして保存した。他所有者・別namespace・未記録scheduler・他Codexの未完了作業・手動再開予定の不存在は未確認。process環境変数の取得は行わず、将来利用終了の所有者方針は確定済みとして再質問しない。

## 5. 件数と64件・11件への判断

| 項目 | 件数 |
| --- | ---: |
| 固定対象・静的調査票作成 | 75 / 75 |
| 正規実パス照合完了 | 0 / 75 |
| PHYSICAL_DEPENDENCY_CONFIRMED | 0 |
| NO_REQUIRED_DEPENDENCY_IN_VERIFIED_SCOPEという主判断 | 0 |
| LOGICAL_REFERENCE_ONLYという主判断 | 0 |
| HISTORICAL_REFERENCE_ONLYという主判断 | 0 |
| PATH_IDENTITY_UNRESOLVEDという主判断 | 75 |
| PRESERVATION_OR_CONTRACT_GATEという主判断 | 0 |
| VERIFICATION_DEFERRED_FOR_B3_SAFETYという主判断 | 0 |
| 対象branch/HEADの明示参照非検出（限定静的scopeのみ） | 75 |
| mount/aliasの副条件未解決 | 75 |
| active/restartの副条件未解決 | 75 |
| 旧SOURCE保全問題（重複） | 11 |
| 整理可能確定・今回削除許可・削除実行 | 0 |

主判断は排他的、副条件は重複する。B3安全性のため未実施にした調査範囲は§7に記載する。物理path未受領が全75件の直接の未完了理由であり、主判断を安全性保留へ置き換えていない。具体的な物理保持義務がないと確認した件数は0であり、依存の確定0を非依存75へ反転させない。

先行64件は図表7、QC18、Research B34、N1/N3保全検証済み5。新たな具体的依存IDは特定できていないが、サーバーB条件は解消していない。`PREPARED_PENDING_SERVER_B` を維持し、本結果によって候補を削除可能へ進めない。B3本体と旧Research B系列の名称の類似を根拠に、対象34件へB3依存を割り当てていない。

旧SOURCE未解決11件はLOCAL-WT-0005、0009〜0016、0018〜0019。既存の3期待SHA-256・55宣言を引き継ぎ、Bの既知SOURCE/provider保存索引に明確な期待hash候補があるかだけを確認した。今回の索引scopeでは明確候補0、回収0、55宣言は未解決のまま。新しいGit全object探索、全worktree/全archive探索は行っていない。旧SOURCE未発見は物理checkout永久保持の根拠にならず、現存資料の保全も旧SOURCE回収の代用にならない。11件は先行64件へ混入させない。

## 6. ユーザーから必要な非公開ファイルと続行点

**必要：本人所有のローカル75件最小パス対応CSVまたは同等のJSON。未受領。ユーザーがBへ直接提供する。** Aが受領済みという過去記録をBの受領と扱わず、自動接続・転送・path推測はしない。

最小fieldは `worktree_id`、`branch`、`head_commit`（完全40桁）、`canonical_absolute_path`、`path_flavor`、`environment_scope`（非公開の環境識別）、`observed_at`。`verified_checkout_aliases` は既知のcheckout別名がある場合のみ追加し、未調査と確認済み空集合を区別する。共通Gitを照合する場合だけ `common_git_directory` を追加する。外向きguideリンクはcheckoutの別名にしない。field名が異なっても既存仕様との意味対応を確認できれば利用可能である。

受渡し時は保存先とファイル形式、本人所有である旨を知らせてもらう。SHA-256が既に提示される場合は受領bytesと照合するが、承認証明や新たな認証手続きは要求しない。credential・秘密鍵・private raw本文・不要な保存先一覧は含めない。既知の共有実体・mount/alias関係が分かる場合は、その事実だけ別の非公開注記で補足できる。

後続指示後のCodexは本人所有/権限・regular file、75件のID集合・重複・branch・完全HEAD、提示されたdigest、最小fieldと機密情報混入を検証する。その後、今回保存したpath候補・参照元・roleへ境界付き照合し、同じ文字列・同じ実体・必須依存を別々に判定する。未確認実体や役割はUNKNOWNを維持し、変化したB3状態と具体的なjob/契約差分だけを必要に応じ更新する。静的調査全体のやり直しや科学runは不要。

## 7. 保全状況・未実施範囲

本調査の書込み先は本人専用の新しい非公開領域と、その中の独立公開cloneのみ。B3のprocess・service・資源条件・SOURCE/input・raw/途中結果/最終結果・checkpoint/cache・grant/ledger/marker・STOP・既存worktree/branch/共通Git/venvには書込みを行っていない。SOURCE13本は観測時点で既存宣言と一致。稼働中のraw等が本来のjobによって更新されることまで凍結・比較しない。全結果の完全性・科学的PASS・終了は認定していない。

B3の安全性を優先し、raw/compiled cache/checkpointの再帰走査・payload hash・trace全件解析・runtime実行検査・科学verifier・巨大archive展開を行っていない。これらは確認済みとして数えない。未知の共有mount、別namespace、他所有者job、未記録の再開計画、全private custodyと別障害域backup、未公開commitの全object探索は今回scope外。旧SOURCEの広域再探索も禁止範囲へ拡張していない。

SOURCE hashes・既存入力の宣言hashと来歴は保持した。既存文書の旧承認・grant条件を削除/変更したり、過去STOPを解除していない。新規科学計算・compile・sampling・数値再解析は0。B3停止・再開・retry・signal・process変更は0。他hostへのSSH/SFTP/scp・自動ファイル転送は0。

## 8. 公開・次の担当・mandatory STOP

開始時remote HEADと独立公開clone HEADは固定入力 `e5651ab7b7f1682dac51a756e6b277f4a2d9c217` と一致した。独立cloneはB3の共通Git/object storeを共有せず、alternatesも使用しない。今回の新規directoryのCSVと本報告だけを追加する。既存 `inventory/local/`、`inventory/server_a/`、`inventory/integrated_local_server_a/` は変更しない。公開前に内部path・host/IP・mount詳細・private raw配置・credential・非公開process/job識別子の混入を確認する。

push直前にremote HEADを再確認し、通常fast-forwardだけで公開する。並行更新があれば他者の追加を保持する。公開commit SHAとlocal/remote一致の実測結果は最終通知と非公開PUBLICATION_RESULTへ記録する。本報告の事前記述だけでpush成功を認定しない。

次の担当：**ユーザー**が必要最小限の非公開75件対応表をBへ直接提供し、未完了照合の続行を指示する。**Codex（後続指示後）**が受領検証と残る物理照合だけを実施する。**ローカルCodex（その後の指示後）**がB結果を64件計画へ統合する。実際の整理は残条件解消後、**ユーザーの具体的なバッチ実行許可**による別作業である。

**今回の静的調査と公開後はmandatory STOP。実パス対応表の受領待ち。全75件のworktree・branchを維持し、削除・登録解除・prune・gc・main mergeを行わない。**
