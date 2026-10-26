# ローカル75件：サーバーA実パス照合準備と旧SOURCE追跡

2026-10-26。固定入力 `d908f578a41c8ee235ef5bcbb75af6435f473278`。前回A調査 `f8e6cee8d08804095be8ba296726da91b9c8f21b` の75件・4件の論理参照・残71件の検索結果を保持した。[ローカル68件保全報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d908f578a41c8ee235ef5bcbb75af6435f473278/inventory/local/archive_followup_68_20261026/LOCAL_68_REPORT.md)の57件保全scope検証済み・11件旧SOURCE未解決も取り込み、今回Aで保全コピー自体を再検証した結果とは区別する。所有者の全75専用workspace利用終了方針は確認済みで、再質問しない。

## 結果

| 項目 | 件数 |
| --- | ---: |
| ID別の照合準備・結果記録 | 75/75 |
| ローカル実パスを受領し照合可能になったID | 0 |
| 具体的な物理依存を確認 | 0 |
| 確認範囲内で物理依存なしと確定 | 0 |
| 実パス・別名・host対応が未確認 | 75 |
| 実行・再開契約の確認保留（上記と重複） | 75 |
| 旧SOURCE3版が未解決のID（上記に含む） | 11 |

75件の主判断は従来のPATH_VERIFICATION_NOT_POSSIBLEを維持する。CSVのPREPARED_MAP_NOT_RECEIVEDは照合の準備状態であり、物理照合完了や削除許可を意味しない。物理依存あり0・なし0は、未確認75を解消していない。ローカルパス情報の作成予定を受領済みとは扱わない。

## 確認範囲 P1

前回2,135資料のbranch/SHA検索を繰り返さず、抽出済み2,599 locatorを再利用した。参照元・JSON pointer/lineを補うため、control/handoff等134件と前回一致資料17件の計151メタデータだけから追加の明示path・埋込み候補・未展開templateを抽出した。限定抽出の読取りエラー0。

非公開で保全した候補は4,043、参照occurrenceは11,505、未展開template記録は2,022。これは依存件数・job数・確認済み実ファイル数ではない。4,043候補のうち以前の2,599 locatorを継承し、追加1,444候補を抽出した。既存locatorの過去の存在観測を現在の状態と混同せず、全候補の現存・利用を再認定していない。A内で実測した既知symlink別名4件と、既知保護pathに対応するmount情報4件も非公開で記録した。hostをまたぐ共有実体の同一性は未検証。

4件の論理参照はLOCAL-WT-0203〜0206で変更なし。0203/0204は固定公開図表資料への参照、0205/0206は記録済み公開SOURCE/result・文書入力commitの共通参照であり、物理checkout依存とは区別する。他71件も、前回の検索範囲で具体的照合キー一致を検出しなかったという限定結果を維持した。

## 非公開照合の最小条件

現在、LOCAL75用に承認されたパス対応表の受渡し方法・宛先は確認できておらず、対応表も未受領。別目的の既存private転送方針は流用しない。新しいSSH・共有storage・外部転送先・公開GitHubを情報交換手段にしていない。

必要な最小入力schema、正確な75 ID/branch/完全HEADの入力ひな形、非公開candidate catalogue、参照元解決表、保守的matcherを作成し、未受領状態で75行を生成した。matcherは文字列候補を返すだけで、物理依存・不存在・整理可を自動確定しない。公開資料には内部pathのhashも載せない。

- U1 / Q1：ID・branch・完全HEADに結び付く元worktreeの正規path・path flavor・host scope・確認日時を、合意済み非公開手段で受け渡す。承認された経路がなければ、ローカル側で同じ条件を照合してID別の非機密結果を返す方法を合意する。credential・秘密鍵・raw本文・全home一覧は不要。
- U2 / Q2：symlink/mount別名と共有実体を示す根拠、SOURCE/provider/input/result・実行/再開契約を照合する。path component境界で照合し、case fold・Unicode変更・環境変数展開・他hostのrealpath解決は自動で行わない。親成分を含むpathはsymlinkをまたぐ可能性があるため要確認とする。同じ文字列でも別hostなら別実体、異なるrealpathでも共有実体である可能性を残す。
- U3：一致した参照元が現在の必須入力なのか、過去の履歴・文書・公開Git参照だけなのかを確認する。実行中job・再開予定、動的path、未展開template、未確認aliasを残したまま不使用とは判定しない。Git branchだけの利用を元checkoutの永久保持理由にしない。
- H3 / Q3（11件）：旧SOURCE3版の保存由来・期待hash・過去の宣言との対応を別条件で確認する。Aの限定探索では未発見で、現行SOURCEのコピー・再構築・編集・再実行で代用していない。詳しくは[旧SOURCE探索結果](SERVER_A_N3_HISTORICAL_SOURCE_RESULT.md)。
- B1（全75件）：サーバーB担当が後日、現在のDF-native L_D=0 B3のSOURCE/provider/input・job・再開契約と元worktreeのpath/別名を照合する。旧Track B/G6/W2C B3とは別。今回Bへ接続せず、稼働環境へ介入していない。

## 保全と作業境界

旧SOURCEの探索は既存A索引・既知保管物・A独自履歴に限定した。ローカルで探索済みの公開Git全履歴・export検索は繰り返さず、対象版の再構築・科学runner実行はしていない。今回完全一致bytesを回収できなかったため、SOURCE保全コピーは0件。

サーバーAの従来12候補はKEEP_REQUIRED 3・UNRESOLVED 9を維持。main・共通Git・SOURCE provider・private raw/custody・H2/H4/H6のinput/result/STOP・永続handoff・共有venv/cache・alternate object store・未公開commitを保護した。今回研究結果・研究判断の変更は行っていない。

`research_project_instructions_unified_2026-10-25.md` は既知A領域・固定入力branch・最新Research B参照の既知pathで確認できず、GitHub mainの同pathも404だった。取得・読了済みとは扱わない。今回添付の保護・read-only・STOP指示を適用した。

詳細原本・物理参照・環境別別名・参照元資料の場所・mount情報・照合手順・helperはAの本人所有owner-only非公開領域に保存した。公開コピーは内部絶対path・host/IP・private配置・非公開ファイル名・秘密情報・内部pathの単純hashを含めない。公開SOURCE相対pathと期待SHA-256は既存公開宣言のファイルbytesに対する値であり、内部pathのhashではない。

専用branchへの本ディレクトリ3ファイル追加だけを公開対象とする。公開直前に最新remote HEADを確認し、既存成果物を保持したfast-forward更新だけを行う。研究worktree/branchの切替・rebase・merge・削除・登録解除、原本移動・上書き、共有Git/venv/cacheの書込み、SOURCE/raw/grant/ledger/marker/STOP変更、科学計算は行わない。本調査は整理実行の承認を与えず、完了後mandatory STOP。
