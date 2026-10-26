# ローカル234 worktree：残条件の追跡確認報告

PUBLIC_REDACTED_FOLLOWUP。確認期間：2026-10-26T19:37:40.639654+09:00〜2026-10-26T19:54:15.417080+09:00。対象は固定統合commit `da117e6f683eb5cf71bb5e2fb45021f6602801fc` のLOCAL Group 1のみ。元KEEP_EVIDENCE166／HOLD_UNKNOWN68を全件維持し、新判断を別fieldへ追加した。公開作業の開始HEADは最新確認した`4217e2f43ba363c9687a390657128bd5716d0074`で、サーバーA追跡commit `4217e2f43ba363c9687a390657128bd5716d0074`を含む。

## 1. 結論と確認完了数

234件のローカル技術確認を実施した。確認できない所有者・他環境・保管責任はUNKNOWNとして残した。「確認完了234」は全gate完了や削除可能234件を意味しない。

| 新判断状態 | 件数 |
| --- | --- |
| TECHNICALLY_CLEAR_PENDING_OWNER | 0 |
| TECHNICALLY_CLEAR_PENDING_OTHER_ENV | 214 |
| PRESERVATION_ACTION_REQUIRED | 0 |
| KEEP_REQUIRED | 7 |
| UNRESOLVED | 13 |
| 合計 | 234 |

- 技術的に整理可能性が高い計画群は214件。ただし利用終了・歴史保持役割の所有者判断と他環境の実参照確認が残る。
- ユーザー判断**だけ**が残る件数は0。OTHER_ENV_SCOPEを閉じる証拠がなく、process未検出だけで終わった扱いにしていない。
- 他環境の確認を主な残条件として分類したのは214件。他の状態を含め、OTHER_ENV_SCOPEが残るのは全234件。
- 維持必須7件、未解決13件。今回、新しい固有Git外資産や未公開変更によるPRESERVATION_ACTION_REQUIREDは0件だった。外部raw/backupの不存在・復元性を保証しない。
- 最初の検討群は文書・図表7件。次はQC18/N1N3 16件、Research Bの34件、Track Bの116件を研究用途単位で確認する。

## 2. 再利用と実際に行った差分確認

元のLOCAL4資料、統合4資料、非公開の詳細原本・匿名ID対応、既存control/source/studyの索引を再利用した。既存CSV I/O、Git read-only helper、ID変換/公開化処理はentry pointを実行せずに再利用し、実パス参照の役割解釈だけを最小限追加した。2026-10-25版共通運用指示の§8.1・§8.4・§10/10.3を実在ファイルから確認して適用した。

234件について現在のHEAD/branch、status、untracked/ignoredと環境pointerの差分を確認。既存metadata索引からSOURCE/control/STOP/契約/handoff/文書の実パストークンを照合し、過去snapshotと実依存を分けた。大きい既存control/ledger93ファイルは対象234パスの文字列だけを追加照合し、一致はなかった。数値fieldの解析・集計、科学SOURCEのimport/実行、raw全件走査/hash、archive展開は行っていない。

metadata読取りで利用可能なCPU範囲に応じて独立チェックを並列化し、共有設定・他jobは変更しなかった。独立の公開Git領域以外へGit書込みを行わず、read-only操作はGIT_OPTIONAL_LOCKS=0、暗黙fetch抑止、fsmonitor抑止を適用した。speedupは測定/主張していない。

## 3. 差分・資産・利用状況

全234件が実在し本人所有、元HEAD/branchと一致、tracked clean、未追跡科学資産の新規検出なし、ignored領域の追加なしだった。候補内の専用venvとnon-guide symlinkはなく、既存の研究環境pointerから候補へのeditable参照も未検出。案内symlinkとPython等の技術cacheを科学rawと区別した。

Track Aの2件にあったplotting cacheは浅いmetadata確認でfont metadataのみと識別した。科学的なsource/inputやprivate rawを持つと推測せず、追加保全を一律に要求しない。キャッシュの再生成や環境変更も実行していない。

可視processで対象cwdを検出しなかったが、可視namespaceの限界があり、別job・他Codex・再開予定の不在は確認できない。USE_ENDEDは全件OWNER_NOT_CONFIRMED。ユーザーの利用終了判断は取得・捏造していない。

対象内の現在のGit外資産が未特定であることと、外部custodyや過去の原本のbackupが確認されたことは別である。既存index外の情報や別hostにある物理pathを推測で埋めていない。索引候補の2136ファイルは現在読取り不可であり、過去HEADでの削除等を含み得るため、それだけで原本欠落やデータ不存在と判断していない。93大型controlは限定トークン検索で補完したが、全世界の契約/データの不存在を保証しない。

## 4. 原稿・図表7件と完了済み中間作業

0203〜0209は全件TECHNICALLY_CLEAR_PENDING_OTHER_ENV。0208の2件のpath参照は過去の文書出版manifestの作業pointerと確認した。その他6件は既存索引scopeで実パス一致なし。公開された図表/原稿/索引は保持refから再取得可能で、専用checkoutを継続するかは所有者の用途判断へ返せる。

7HEAD中5件は既存DF/Pauli統合snapshotの祖先だが2件は違う。異なる履歴を無理にmergeせず、それぞれのbranch/refを保持する。QC18件とN1/N3の16件も全件ローカル技術確認終了。後続handoff/closureのGit関係はCSVへ記録し、別cohort/独立SOURCE/STOPの包含とは解釈しない。

## 5. Research B 36件・Track B 131件・他系列

Research Bは34件がローカル技術確認終了、0283/0286はreadiness後の科学SOURCE供給役割が未解決。全36HEADが最終成果/pause handoffの祖先であることを確認したが、それはコード履歴の証拠。rawや固定SOURCE環境の復元保証ではない。

Track Bは116件が技術確認終了、7件維持、8件未解決。過去worktree一覧・baseline/preservation・文書出版の記録が多数存在するが、全てを実行環境の必須参照と扱わなかった。一方、明示的SOURCE/provider、凍結契約、source/runtime付きone-shotの束縛は今回解除しなかった。

| 研究系列 | 計 | 他環境確認待ち | 維持必須 | 未解決 |
| --- | --- | --- | --- | --- |
| 原稿・図表・索引 | 7 | 7 | 0 | 0 |
| QCアルゴリズム履歴 | 18 | 18 | 0 | 0 |
| N1/N3履歴 | 16 | 16 | 0 | 0 |
| Research B B2D Outer | 36 | 34 | 0 | 2 |
| Track Bアルゴリズム履歴 | 131 | 116 | 7 | 8 |
| 旧Track B G6 | 6 | 6 | 0 | 0 |
| Research C C1 | 4 | 4 | 0 | 0 |
| Research C C4回路 | 7 | 5 | 0 | 2 |
| DF-native Track A | 2 | 1 | 0 | 1 |
| 表現・アルゴリズム履歴 | 3 | 3 | 0 | 0 |
| その他アルゴリズム・中間検証 | 4 | 4 | 0 | 0 |

Research C C4では0319/0328の出力namespace責任が未解決。C1 4件とC4の残5件はlocal技術確認終了だが、現在の研究・別Codex予定について所有者/環境担当の確認が必要。Track Aは0020のraw namespace責任が未解決、0021は技術確認終了。

## 6. 維持7件と未解決13件

| ID | 研究系列 | 新判断 | 具体的な障害 |
| --- | --- | --- | --- |
| LOCAL-WT-0020 | DF-native Track A | UNRESOLVED | 公開済み実行記録のraw_namespaceが候補内を指す。現時点でGit外資産は未検出だが、原本/namespace保管責任の終了は未確認 |
| LOCAL-WT-0093 | Track Bアルゴリズム履歴 | KEEP_REQUIRED | 既存source_preparation記録が、このcheckoutと同じ固定SOURCE commitをproviderとして直接指定。物理SOURCE参照の責任を未解除 |
| LOCAL-WT-0102 | Track Bアルゴリズム履歴 | UNRESOLVED | 保存observer source内にこのROOT定数が残る。凍結監視コードの再使用・歴史保持・path責任の扱いが未確定 |
| LOCAL-WT-0162 | Track Bアルゴリズム履歴 | UNRESOLVED | source review/authorization準備記録が物理worktreeを保持。後続記録との役割移行が明確でない |
| LOCAL-WT-0168 | Track Bアルゴリズム履歴 | KEEP_REQUIRED | 凍結実行契約・guardがcompile入力/出力rootを直接指定し、fixture変更・reconstructionを認めない。既存path条件を維持 |
| LOCAL-WT-0175 | Track Bアルゴリズム履歴 | UNRESOLVED | exclusive B worktreeという既存保護記録がある。専用開発環境としての役割を終える判断は未確認 |
| LOCAL-WT-0176 | Track Bアルゴリズム履歴 | UNRESOLVED | 複数の既存READMEがB worktreeとして指定。共有venv実参照は未検出だが代表開発環境の役割終了は未確認 |
| LOCAL-WT-0185 | Track Bアルゴリズム履歴 | KEEP_REQUIRED | R1の固定SOURCE/runtimeと実行worktreeをinvocation/終了記録が直接束縛。結果後STOPと固定pathの保持責任を維持 |
| LOCAL-WT-0187 | Track Bアルゴリズム履歴 | UNRESOLVED | 旧SOURCE/契約の修正待ち記録の保管責任と後続reviewへの移行が未確定 |
| LOCAL-WT-0188 | Track Bアルゴリズム履歴 | UNRESOLVED | 独立SOURCE review準備worktreeという記録が残る。旧レビュー用環境の役割終了が未確認 |
| LOCAL-WT-0193 | Track Bアルゴリズム履歴 | KEEP_REQUIRED | SOURCE_FROZENのmanifestがこのworktreeを指定。直接child・固定SOURCEの契約を今回解除しない |
| LOCAL-WT-0196 | Track Bアルゴリズム履歴 | KEEP_REQUIRED | SP0.5のone-shot実行記録に固定SOURCE/runtime・実行worktreeが残る。停止・grant消費と固定参照を維持 |
| LOCAL-WT-0198 | Track Bアルゴリズム履歴 | KEEP_REQUIRED | SP1のSOURCE/review/direct-childと実行worktreeを既存記録が束縛。固定参照・STOPの保管責任を維持 |
| LOCAL-WT-0199 | Track Bアルゴリズム履歴 | UNRESOLVED | 旧SP1契約proposalがworktreeを記録。提案の再使用・自己拘束の扱いは所有者判断が必要 |
| LOCAL-WT-0200 | Track Bアルゴリズム履歴 | KEEP_REQUIRED | FROZEN_SOURCE_PREPARATIONのmanifestがこのworktreeを指定。旧STOP・source identity条件を維持 |
| LOCAL-WT-0201 | Track Bアルゴリズム履歴 | UNRESOLVED | design review pendingの契約がworktreeを記録。設計提案/履歴を保持する役割が未確認 |
| LOCAL-WT-0283 | Research B B2D Outer | UNRESOLVED | Exact-R readinessが候補の物理worktreeを参照。最終handoff後も固定SOURCE供給元として必要か未確認 |
| LOCAL-WT-0286 | Research B B2D Outer | UNRESOLVED | Full DF-PF readinessが科学SOURCEと候補worktreeを記録。最終handoff後のSOURCE供給責任を未確認 |
| LOCAL-WT-0319 | Research C C4回路 | UNRESOLVED | 既存実行scopeの出力namespaceが候補内を指定。現状Git外成果は未検出だが出力/STOP保持責任を未確認 |
| LOCAL-WT-0328 | Research C C4回路 | UNRESOLVED | Rustiq実行scopeの出力namespaceが候補内を指定。役割終了・将来再使用と他環境の責任を未確認 |

KEEP_REQUIREDは現在記録される固定SOURCE/path・実行条件の保持責任を尊重した判断であり、物理checkoutを永久に保持する新規契約ではない。固定refが公開済みでも既存path条件を勝手に消さない。未解決13件は、準備文書・旧proposal・監視helper・代表開発環境・科学SOURCE供給・出力namespaceの役割を所有者/原本で確認する必要がある。

## 7. Git履歴・未公開commit・SOURCE/STOP

233 HEADは現在のGitHub tipと一致、1 HEADは現在tipの祖先。対象の新しいlocal-only HEADは検出しなかった。今回、対応controlから明示される8 SOURCE commitについて保持refと公開tipからの到達も確認した。現在の保持状態を記録しただけで、将来の永久保管を保証しない。

既存統合資料のLOCAL 1＋A 4の未公開候補commitはこの234件の整理許可へ含めず、既存保護状態を引き継ぐ。別SOURCE・失敗・旧STOP・grant消費・独立cohortを後続成功で置換せず、SOURCE/結果/STOPの元bytesや科学条件を編集していない。

## 8. shared Git・環境・他環境の条件

234件はいずれも対象外の主Git本体を共有するlinked checkout。共通objectsのalternateファイルは今回の主Git領域で未検出。これは別環境/独立cloneのalternate依存解消の確認ではない。共有Git本体と元の代表環境を残す方針を前提にし、checkout登録解除とGit本体・独立clone削除を別操作として扱う。

候補の専用venv・shared cache・editable sourceの実参照は既存pointer scopeで未検出だったが、環境は変更していない。現在のライブラリimport/科学runnerを実行して接続を試すこともしていない。未記録設定はUNKNOWN。

A追跡結果はKEEP_REQUIRED3（SERVER_A-WT-0026/0029/0031）・UNRESOLVED9。今回Aを再調査せず参考として保持し、公開匿名資料からLOCAL候補の物理参照不在を推定しない。A/B担当が必要とするのはbranch名の共通性ではなく、候補directory/固定SOURCEへの実参照の有無。今回はSSHしない。

サーバーBで申告上実行中のDF-native L_D=0 B3と、旧Track B/G6/W2CのB3を別文脈として扱う。B側worktree・SOURCE・raw・cache・process・Gitは変更していない。B3終了の推定や再開指示も行わない。

## 9. ユーザー判断と今後の実行順序

所有者判断を研究単位へまとめた[GROUP_DECISIONS](LOCAL_234_GROUP_DECISIONS.md)を使う。最初は図表7件、次にQC/N1N3、Research Bの34件、Track Bの116件等の214件について、利用終了・再使用予定・履歴保持の役割を判断できる。13未解決と7維持は役割/契約の扱いを先に確認する。

他環境担当の実参照確認後、ローカルCodexが対象IDを非公開原本の完全pathへ解決し、保持branch/原本/shared Gitとcheckout登録解除を分けた具体的操作案を作る。実際のworktree解除/削除、branch削除、privateデータ移動/保管先変更、新規科学実行は別途明示許可が必要。承認証明書・新grant・approval ledgerは作らない。

**BEFORE_ACTION_DELTAは全234件NOT_DONE_FUTURE_ONLY／完了false。** 今回の現在状態チェックは、将来の具体的整理承認後の直前チェックの代わりにならない。

## 10. 公開化・保全・停止理由

公開するのはこの新規directoryの次の3ファイルだけ。元LOCAL4、SERVER_A全資料、統合4、A追跡2、および非公開原本は変更しない。

- [LOCAL_234_FOLLOWUP.csv](LOCAL_234_FOLLOWUP.csv)：234件の元分類、現在HEAD、各gate、source/raw/STOP、後続関係、新判断、必要な追加確認と将来処置。
- [LOCAL_234_GROUP_DECISIONS.md](LOCAL_234_GROUP_DECISIONS.md)：研究単位の候補群・維持群と所有者判断。
- [LOCAL_234_FOLLOWUP_REPORT.md](LOCAL_234_FOLLOWUP_REPORT.md)：本報告。

絶対path、内部host/IP、private raw/非公開ファイル名・配置、process/job識別子、credential・token、内部環境設定は公開しない。詳細referenceのcontextとID逆引きは独立した本人所有の非公開領域へ保存する。CSV中のevidence IDはその非公開原本への対応であり、公開URLや実行指示ではない。既存public SOURCE/結果/STOP参照のみ公開可能な範囲で再利用する。

確認不能項目を閉じるための科学再計算や全raw hashは行わなかった。ローカル技術確認が終了した214件を繰り返し再調査せず、所有者/他環境の判断待ちとしたことが今回の停止理由。7維持・13未解決は具体的な阻害要因を残す。

**専用branchへの通常fast-forward pushとlocal/remote SHA照合後mandatory STOP。実際のworktree削除・登録解除・branch削除・merge・データ移動・科学SOURCE/raw/grant/ledger/marker/STOP変更・科学計算・他環境操作は行わない。**
