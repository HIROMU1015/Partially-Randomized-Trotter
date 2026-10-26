# ローカル139件：サーバーB実パス依存照合

2026-10-27 JST。PUBLIC_REDACTED。照合snapshot：2026-10-27 03:05 JST。

**非公開139件対応表の受領bytesは提示SHA-256と一致し、139件すべてのID・branch・完全40桁HEADを固定CSVと照合した。正規pathの限定照合139/139件、主判断は全139件 `NO_REQUIRED_DEPENDENCY_IN_VERIFIED_SCOPE`。必須物理依存の確認0件、主判断としての保留・未解決0件。ただし、未知alias・別namespace・未記録job/再開契約・private custody・将来利用の副条件は残る。これは全環境の非依存認定や整理実行許可ではない。**

各IDのbranch・完全HEAD、研究単位、元の131件/8件保全状態、主判断・副条件は [SERVER_B_LOCAL139_PATH_RESULTS.csv](SERVER_B_LOCAL139_PATH_RESULTS.csv) に保存した。前回75件の結論を転記せず、今回の正確な139件で新たに照合した。

## 1. 固定入力と受領検証

固定commitは [`96cf3dae3bd03d6f64f384fefedc6e88e782eef6`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/96cf3dae3bd03d6f64f384fefedc6e88e782eef6/inventory/local/next_group1_139_preparation_20261027)。NEXT139_PRESERVATION_STATUS.csv、NEXT139_ARCHIVE_MANIFEST.csv、NEXT139_CLEANUP_PLAN.mdを取得し、保存bytesのGit blob一致を確認した。開始時にGitHub整理branchのHEADも同commitと確認した。139件のID集合は固定CSVから取得し、branch名や連番から生成していない。

ユーザー指定の本人所有regular CSVをBで直接受領した。139行・139固有ID、固定集合との差分0、branch不一致0、完全HEAD不一致0。原本を移動せず、本人専用非公開領域へ同じbytesを保全した。

提示値・受領値のSHA-256はともに **`670472a2d8b00fb3fc1935e8abd64a9a61fcac0aeaaca8130c90933b5279d01b`**。実際のbytes照合に基づく転送前提示digestとの一致であり、承認証明書を新設したものではない。

受領fieldは `worktree_id`、`branch`、`head_commit`、`canonical_absolute_path`、`recorded_absolute_path`、`host_scope`、`path_flavor`、`verified_checkout_aliases_json`、`observed_utc` 等。全139件はLOCALのPOSIX絶対pathで、正規pathと記録pathは一致した。列挙されたcheckout別名は0件だが、alias scopeは新規別名探索を行っていないと明記している。空集合を全別名の不存在証明にはしない。

外向きguideリンクをcheckout別名に含めていない。今回表には外向きリンク先の詳細とLOCAL共通Git directoryのfieldがないため、それらの今回直接照合を完了扱いにせず `NOT_SUPPLIED` として保持した。前回75件のLOCAL共通Git値や68本のリンク一覧を139件に補完していない。

日時・path形式・公開SOURCE/result/STOP locatorの形式を確認し、既知credential/private-key形式は検出しなかった。実path・host・process/job・mount・private rawの配置を公開CSVへ転記していない。LOCAL/AへのSSH/SFTP/scp・自動転送は0。

## 2. 方法と再利用範囲

前回の [B_STATIC_1](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/34015c1f58ceda0e0409d858d0f606831e178115/inventory/server_b/local75_dependency_check_20261027) と [B_PATH_1](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/33fd56d573c3d6022609b9141daf2429d19c75b4/inventory/server_b/local75_actual_path_reconciliation_20261027) の非公開索引・標準ライブラリhelperを再利用した。今回は前回helperの読取り・path正規化・成分境界照合・B namespaceのmetadata確認関数を直接呼び、139件入力と研究単位・保全条件へのjoinを追加した。前回scriptの75件main処理は実行していない。

`B_PATH_139` の対象は、保存済み2,023 path occurrence、2,146 SOURCE/runtime宣言、既知12 B3参照先、runtime metadata、既存process/controlの保存文脈と現在の限定metadata。前回静的38ファイルの全読み直しはしていない。固定資料と今回対象についてのみ新規照合した。

正規path・記録path・列挙済みcheckout別名を成分境界付きで照合し、隣接名の部分文字列一致を除外した。ancestor一致も物理保持義務へ自動昇格させない。空白を含むpath、隣接名、ancestor関係の人工チェックを行った。branch・完全HEAD・12桁prefixは保存済み参照文脈に別枠で照合し、Bの対象同名Git refも読取専用で確認した。

現在のB3では既知2 service、本人UIDのcontrollerのcwd/exe/引数、既知bindingの存在・正規化先・device/inode、runtimeの既知3 metadata、B共通Gitのalternates、可視mount表を確認した。本人UIDの可視project Python/strace processのcwd/exe/引数も限定確認した。他所有者のprocess、環境変数、実行中moduleのロード元全件は取得していない。

## 3. 字句一致・実体・物理的必要性

今回の正規path/別名に対する保存参照・既知binding・限定processの一致は0件。保存文脈での対象branch・完全HEAD・12桁prefix一致も0件、B共通Gitの対象同名local branch refも0件だった。同commitのobjectが存在しないという意味ではない。

供給された139 path文字列は、Bの観測namespace内の同名位置では存在を検出しなかった。Bで不在であることはLOCAL実体の不存在や全環境の非依存を示さない。LOCALのpathをBで無条件realpath解決して同一視せず、host間のdevice/inode一致を同一実体の根拠にしていない。

B3の既知12参照先は前回と同じB内device/inode、runtime metadataは前回保存値と一致した。B共通Gitにはalternates fileを検出せず、現在の可視mount表は前回と同じで、観測controllerと調査processのmount表も一致した。これらは既知B側実体の継続確認であり、未記録export/bind/aliasや別namespaceからLOCAL実体へ接続する可能性の不存在証明ではない。

今回、対象LOCAL checkoutをSOURCE/provider/input/runtime、実行workdir、固定path契約、保存結果・共通Gitの必須物理保管先として指定する具体的bindingは確認範囲内で見つからなかった。文字列一致と役割・必要性を分離し、見つかっていない保持義務を推測で認定していない。逆に、未確認の契約やcustodyの不存在も認定していない。

## 4. 件数

| 項目 | 件数 |
| --- | ---: |
| 固定ID・branch・完全HEAD・受領SHA検証 | 139 / 139 |
| 正規実pathの限定照合完了 | 139 / 139 |
| PHYSICAL_DEPENDENCY_CONFIRMED | 0 |
| NO_REQUIRED_DEPENDENCY_IN_VERIFIED_SCOPE | 139 |
| LOGICAL_REFERENCE_ONLY | 0 |
| HISTORICAL_REFERENCE_ONLY | 0 |
| PATH_IDENTITY_UNRESOLVED（主判断） | 0 |
| PRESERVATION_OR_CONTRACT_GATE（B主判断） | 0 |
| VERIFICATION_DEFERRED_FOR_B3_SAFETY（主判断） | 0 |
| 将来利用・未知alias・別namespace等の副条件 | 139 |
| ローカル追加保全条件を維持 | 8 |
| Research Cの今後/別Codex利用未確認 | 9 |
| 今回の削除・登録解除・branch削除・移動 | 0 |

主判断としての未解決0と、scope外の副条件未解決139を区別する。今回のB照合が完了しても、139件の所有者判断やローカル保全を全面的に完了させたことにはならない。

## 5. 研究単位と優先群

| 研究単位 | 件数 | 今回B scopeの必須物理依存 |
| --- | ---: | --- |
| TRACK_B_D_E_HISTORY | 16 | 未検出 |
| TRACK_B_DF_CORRECTION | 19 | 未検出 |
| TRACK_B_M1_M1S | 13 | 未検出 |
| TRACK_B_PARENT_ALLOCATION | 15 | 未検出 |
| TRACK_B_HANDOFF_AND_DIAGNOSTICS | 9 | 未検出 |
| TRACK_B_RA_D0 | 7 | 未検出 |
| TRACK_B_BF_R1_SP_EARLY_HISTORY | 14 | 未検出 |
| TRACK_B_G_SERIES | 23 | 未検出、7件のローカル大容量条件は維持 |
| TRACK_B_G6 | 6 | 未検出、現在のDF-native B3と別系列 |
| RESEARCH_C_C1 | 4 | 未検出、将来利用未確定 |
| RESEARCH_C_C4_CIRCUITS | 5 | 未検出、将来利用未確定 |
| DF_NATIVE_TRACK_A | 1 | 未検出 |
| REPRESENTATION_ALGORITHM_HISTORY | 3 | 未検出 |
| 既存アルゴリズム開発／中間実装・診断・停止履歴 | 4 | 未検出、1件のローカル大容量条件は維持 |

Track Bは今回候補116件だけを扱い、別資料のTrack B131件全体へ結論を拡張しない。優先72件（D/E16・DF correction19・M1/M1S13・parent15・handoff9）も限定非依存だった。既存のSOURCE・独立cohort・失敗・diagnostic・grant/ledger/marker/STOPは元bindingとして保持し、科学的包含・再実行可・契約解除を推定しない。

Research C9件について、今回の既存索引と限定可視job metadataでは対象LOCAL pathの利用を検出しなかった。現在のprocess非検出だけで将来利用終了や別Codexの予定なしとは認定しない。研究担当による研究単位の将来利用・再開予定判断は残条件として維持する。旧G6等のB3という名称を現在のDF-native B3へ結合していない。

## 6. 131件と大容量保全未完了8件

固定資料の `ARCHIVED_VERIFIED_CLEANUP_CANDIDATE` 131件と `PRESERVATION_ACTION_REQUIRED` 8件を変更していない。公開保全manifest8,352行のentry_type内訳は、新規実コピー5,889、既存archive再利用514、コードGit参照のみ981、sparse未materializeのGit参照のみ953、未コピー大容量binding15。これは保存metadataの再利用であり、Bで5,889 payloadを再取得・再hashした結果ではない。

8件は **LOCAL-WT-0002、0095、0096、0097、0099、0100、0101、0103**。この8件もB scopeのLOCAL checkout必須依存は未検出だったが、ローカルの15 binding・9固有大容量checkpoint/resultに対する追加保管判断は継続する。B側がその実ファイルを保管すべきと明示した具体的bindingは今回確認範囲で見つからなかったが、全custody責任の不存在を証明していない。

GitHub ref保持、選定scopeのarchive、追加実ファイルのcustody、独立障害域backupは別条件である。Bへraw/checkpointを転送・コピーせず、ローカルの未コピー条件を解消扱いにしていない。

## 7. DF-native B3の状態と非干渉

保護対象はlinear H4/H6・1Å・STO-3G・tol-only DF10/19、one-bodyを保持するDF-native `L_D=0` B3。03:05 JSTのread-only観測では取得と終了監視serviceがactive/running、既存statusは `SCIENTIFIC_ACQUISITION_RUNNING`。前回の状態を流用せず、今回実際に照会した。全取得終了・科学的性能・結果受入れを示すものではない。

既存13 SOURCE fileの宣言hash一致結果は、前回観測時点の証拠として再利用した。今回再hashは0で、現在時点の13本一致を新規認定していない。SOURCE/provider/inputの数値payloadは読み込まず、科学module/runner/verifier、compile/sampling、数値再解析は実行していない。

controller/workerへのsignal、B3停止・再開・再起動・retry、service/scheduler/cgroup/mount変更は0。SOURCE・provider・input・raw・checkpoint・cache・結果・grant・ledger・marker・STOP、B3 worktree/branch/共有Git object store/venv/editableは変更していない。既存one-shot・復旧回数・後続失敗時STOPの条件を保持した。他ユーザー・他jobのprocess/file/設定への変更も0。

書込みは本人専用非公開領域と科学環境から独立した公開cloneだけ。正式共通指示2026-10-25版の§8.1・§8.4・§10を実在ファイルで確認して適用した。承認proofや新grant発行の仕組みは作っていない。

## 8. scopeと未確認条件

前回の候補索引は特定形式の引用された絶対home系pathが中心であり、任意の相対path・動的path・一時path・未記録metadata・他namespaceを網羅しない。この限界を今回も維持し、非検出を全参照の不存在へ拡張しない。

raw/cache/checkpointの再帰走査、全Git object/履歴探索、全trace解析、大容量payload展開、実行中processへのコード注入・実module import確認は行っていない。具体的な新規物理依存の兆候は今回検出しなかったため、広域再調査へ拡張していない。139件のprivate custody・独立backup・未記録SOURCE/path契約・未記録再開/手動利用の全確認は残る。

## 9. 保存・公開とローカルへ返す事項

非公開原本には受領CSV・提示SHA照合・固定blob対応、ID別path/候補/参照元/実体metadata・判断根拠、B3/runtime/Git/process/mount観測を保存した。今回公開するのは本directoryのCSVと本報告の2ファイルのみ。旧LOCAL/A/B資料や計画は上書きせず、139件の利用終了・ローカル131/8区分を変更しない。

開始時と公開直前に整理branchのHEADを確認する。A側の並行更新があればそれを保持した親commitへ今回2ファイルだけを追加し、通常fast-forwardで公開する。公開後はGitHubから2ファイルを読み戻してbytesとcommit SHAを確認する。成功とlocal/remote一致は実測後に最終通知・非公開公開記録へ残し、この事前文だけで認定しない。

次の担当は、**ローカルCodex**がB限定結果を131件・8件の計画へ統合し、研究単位ごとの契約・custody・他環境・直前差分を残条件として扱う。**ユーザー/研究担当**が139件の研究単位ごとの将来workspace利用と再開/別Codex予定を判断する。**該当研究担当/ローカルCodex**が8件の大容量実ファイルのcustodyと追加保全範囲を別途判断する。いずれも今回のB調査が代行認定したものではない。

**匿名化公開後はmandatory STOP。139件の削除・登録解除、branch削除、原本/archive移動、gc/prune、main merge、新科学実行を開始しない。B3は既存jobのまま継続する。**
