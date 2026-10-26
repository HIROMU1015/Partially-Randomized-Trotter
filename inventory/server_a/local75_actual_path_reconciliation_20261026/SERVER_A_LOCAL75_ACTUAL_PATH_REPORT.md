# ローカル75件：サーバーA実パス照合結果

2026-10-26。既存A準備commit `3b95bab223427405caa779c01194671f06de1b54` と、ローカル後続公開commit `3fc92cad436c400d2f850cdb533d77d18a8ee648` を参照した。ユーザーが直接配置した非公開最小対応表を受領し、以前の4,043候補・参照元索引・照合仕様・helperを再利用した。前回の2,135資料検索は繰り返していない。

## 受領・完全性

本人所有のregular file、owner-onlyの保存先とファイル権限を確認した。受領後の権限制限でCSV bytesは変わっていない。ユーザーが提示した元SHA-256と受領bytesのSHA-256は完全一致。digest値・ファイル名・実パスは非公開原本に記録し、Public GitHubへ載せない。

75/75件のID・branch・完全HEAD SHAが固定資料と一致し、重複ID・正規pathの重複はない。受領fieldは必要なpath/identity情報、既知の外向きguideリンク、共有Git情報、公開SOURCE/result/STOP参照だけで、raw本文fieldやcredential等の既知パターンは検出しなかった。公開SOURCE参照95件・STOP参照95件（result参照0件）は以前の固定入力と全75行で一致した。checkout HEADと記録済み科学SOURCEを混同していない。

## ID別結果

| 主判定（相互排他的） | 件数 |
| --- | ---: |
| PHYSICAL_DEPENDENCY_CONFIRMED | 0 |
| NO_REQUIRED_DEPENDENCY_IN_VERIFIED_SCOPE | 71 |
| HISTORICAL_REFERENCE_ONLY | 0 |
| LOGICAL_GIT_REFERENCE_ONLY | 4 |
| ACTIVE_OR_RESTART_DEPENDENCY_UNRESOLVED | 0 |
| PATH_OR_MOUNT_IDENTITY_UNRESOLVED | 0 |

主判定はR1内で観測した関係を分類する。次の副条件とは重複するため、上表の未解決0を未検証条件の解消と読まない。

| 別fieldに保持した結果・副条件 | 件数 |
| --- | ---: |
| 正規path・受領済み確認別名を使った照合完了 | 75 |
| 有効な正規path／配下ファイルへの文字列参照一致 | 0 |
| R1内で元checkoutの必須物理依存を検出しなかったID | 75（上の71＋論理参照4） |
| ACTIVE_OR_RESTART_DEPENDENCY_UNRESOLVED | 75 |
| PATH_OR_MOUNT_IDENTITY_UNRESOLVED | 75 |
| 旧SOURCE3版の保全未解決（上記と重複） | 11 |
| 削除実行を許可 | 0 |

論理参照4件はLOCAL-WT-0203〜0206で、以前のSOURCE/result・図表文書に記録された公開commit参照をそのまま保持した。これはLOCAL元checkoutの物理保持義務を示さない。他71件は以前の具体的メタデータ参照非検出を維持し、今回受領したpathでもR1に必須束縛は検出しなかった。取得済みGitのHEAD保持参照74件と固定公開履歴内ancestor保持1件は、Git保存参照として別fieldに残した。単なる保持refの存在を研究jobの物理依存に数えていない。

## R1：照合方法と誤検出の除外

R1は、受領正規path・明示された既知checkout別名、既存Aの抽出済み候補と参照元field、既知A symlink別名の字句展開、閲覧可能な本人UID processのcwd/引数スナップショットを対象とする。過去の候補索引を現在の全filesystem・全jobの調査結果として扱わない。

前回の4,043候補にはJSON pointerが混在していた。元の候補原本は変更せず、今回の参照元field確認により、JSON pointer由来6,106 occurrenceと、それだけからなる1,385候補を物理path照合から除外した。残る候補は2,658。これは既存キャッシュの文脈整理であり、全資料の再検索ではない。

既存helperの一次結果は75件の弱いancestor候補だったが、すべてJSON pointerのルート表記であり、filesystemの参照ではなかった。元の正規pathや配下ファイルへの有効一致は0件。本文・fieldの役割を確認せず文字列一致だけで物理依存へ格上げしていない。

受領表の既知checkout別名は0件。外向きguide symlink68記録（34 ID）は、checkoutから案内先へ向くリンクであり、元checkout自身の別名ではない。案内先が別の対象75 checkout配下へ向く対応も検出しなかった。これを逆向きのAからLOCAL checkoutへの依存として数えず、ローカル側の可搬性確認事項として保持する。

既知A symlink別名4件から展開可能な参照表現5件も照合し、元checkoutへの一致は0件。受領したLOCAL共有Gitのpathも既存A候補と同一文字列の一致はなく、Aの共有Git/alternate/venvをLOCALの同一実体と認定していない。LOCALのrealpathをAで解決せず、case fold・Unicode変更・環境変数展開・branch名からのpath生成は行っていない。

閲覧可能な本人UID processのcwd/引数に対象path一致はなかった。これは一時点の限定結果であり、取得できなかったprocess、他所有者・別namespace・scheduler・手動の再開計画、未記録aliasを含む不使用確認ではない。process環境変数の取得や停止・変更は行っていない。

## 未確認の実体同一性・実行条件

受領表はLOCAL host scopeで確認されたpathと既知リンクの最小情報であり、Aとの共有filesystem/export/bind/mount同一性を確定する情報は含まない。A側で既に観測したsymlink/mountだけでも、別hostとの共有実体の同一性や不存在は証明できない。

したがって全75件にPATH_OR_MOUNT_IDENTITY_UNRESOLVEDとACTIVE_OR_RESTART_DEPENDENCY_UNRESOLVEDを副条件として残す。字句不一致を確認範囲外の依存不存在へ一般化せず、R1で必須参照がなかったという結論に限定する。主判定71件を全環境で依存なし・削除可能と読み替えない。

## ローカルへ返すID別判断とB確認

CSVの各行は同じID・branch・完全HEADに対する返却結果である。全75件について、AのR1内で元checkoutを維持しなければならない具体的束縛は検出されなかった。4件の公開Git/SOURCE等の参照は保持し、branchや公開SOURCEを使うという理由だけで元checkoutを永久保持する判断はしない。

ローカル側は既存保全scope・保存ファイルの参照性・固定SOURCE/input/result/STOP契約、現在/再開job、未記録共有実体・別名をこの返却結果と統合する。所有者の全75 workspace利用終了方針は確認済みで、再承認は求めない。未確認条件や物理的に必要な入力が具体的に見つかった対象だけ、そのIDの根拠を追加する。実際の整理には後日の具体的対象別指示と直前確認が必要。

旧SOURCE3版の未解決11件はLOCAL-WT-0005、0009〜0016、0018〜0019。今回のpath照合で旧bytesを回収したことにはならず、保全問題を独立して維持する。その未解決だけで11 checkoutの物理保持を自動必須とせず、path一致や不一致だけで旧SOURCE問題を解消しない。科学SOURCE・失敗・STOP・実行条件を変更していない。

サーバーBについては全75件を後日確認待ちとする。現在のDF-native L_D=0 B3のSOURCE/provider/input・job・再開契約と共有path/別名を、B担当が非干渉の方法で照合する。旧Track B/G6/W2CのB3とは別であり、今回は他hostへ接続していない。

## 保全・公開境界

公開前の状態検証中に、対象外のworktree登録1件と研究branch追加1件を観測した。既存worktree登録の変更・削除は検出せず、mainの状態も保持されていた。追加を戻したり新しい研究作業へ介入したりせず、差分を非公開で記録した。R1のsnapshot以後の新しい作業も、実行・再開未検証の副条件に含める。

Aの従来12候補はKEEP_REQUIRED 3・UNRESOLVED 9を維持する。非公開対応表・受領digest・絶対path・host識別・mount/alias根拠・参照元の具体的場所・process詳細・private原本の配置・内部pathの単純hashはPublic GitHubへ追加しない。詳細は本人所有の非公開原本へ保存した。

10月25日版正式運用指示はAでは引き続き未取得・未読。ローカル報告の読了記載をAの読了と扱わない。今回の受領ファイルに同指示本文は含まれず、既存のユーザー指示と非公開照合仕様を適用した。

公開は本ディレクトリの新規CSV・本報告2ファイルだけとし、独立cloneで最新remote HEADを確認して既存成果物を保持したfast-forward更新を行う。非公開CSVの本文はcommitしない。worktree削除・登録解除・branch削除、研究branchの切替/rebase/merge、原本移動、共有Git/venv/cache変更、科学SOURCE/raw/grant/ledger/marker/STOP変更、新規科学計算は行わない。完了後mandatory STOP。
