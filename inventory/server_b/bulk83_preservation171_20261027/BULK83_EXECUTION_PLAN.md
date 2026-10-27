# SERVER_B 83件の一括削除実行承認用計画

**今回は準備のみ。worktree remove/prune、branch削除、merge、force、原本移動は実行していない。下記83件に対する実削除の明示指示を受けた後の計画である。**

元の固定83件に対する利用終了判断は今回のユーザー指示で確認済み。同じ質問を繰り返さない。171件から追加された86候補はこの集合に含めない。

| 研究単位 | 件数 | 正確なID集合 |
| --- | ---: | --- |
| BASELINE_OR_OTHER_UNRESOLVED | 1 | SERVER_B-WT-0344 |
| DF_H2 | 1 | SERVER_B-WT-0054 |
| DF_H46_PF_PR | 5 | SERVER_B-WT-0133, SERVER_B-WT-0134, SERVER_B-WT-0135, SERVER_B-WT-0343, SERVER_B-WT-0345 |
| DF_H5_H7_OPEN_SHELL | 7 | SERVER_B-WT-0143, SERVER_B-WT-0144, SERVER_B-WT-0145, SERVER_B-WT-0146, SERVER_B-WT-0191, SERVER_B-WT-0192, SERVER_B-WT-0194 |
| DF_H6 | 10 | SERVER_B-WT-0155, SERVER_B-WT-0164, SERVER_B-WT-0168, SERVER_B-WT-0173, SERVER_B-WT-0174, SERVER_B-WT-0175, SERVER_B-WT-0178, SERVER_B-WT-0180, SERVER_B-WT-0181, SERVER_B-WT-0182 |
| DF_H8 | 28 | SERVER_B-WT-0195, SERVER_B-WT-0196, SERVER_B-WT-0198, SERVER_B-WT-0199, SERVER_B-WT-0200, SERVER_B-WT-0201, SERVER_B-WT-0202, SERVER_B-WT-0203, SERVER_B-WT-0204, SERVER_B-WT-0205, SERVER_B-WT-0206, SERVER_B-WT-0207, SERVER_B-WT-0208, SERVER_B-WT-0209, SERVER_B-WT-0212, SERVER_B-WT-0218, SERVER_B-WT-0219, SERVER_B-WT-0222, SERVER_B-WT-0226, SERVER_B-WT-0227, SERVER_B-WT-0229, SERVER_B-WT-0230, SERVER_B-WT-0231, SERVER_B-WT-0232, SERVER_B-WT-0246, SERVER_B-WT-0247, SERVER_B-WT-0248, SERVER_B-WT-0259 |
| DF_NATIVE_TRACK_A | 18 | SERVER_B-WT-0004, SERVER_B-WT-0007, SERVER_B-WT-0250, SERVER_B-WT-0252, SERVER_B-WT-0254, SERVER_B-WT-0261, SERVER_B-WT-0324, SERVER_B-WT-0329, SERVER_B-WT-0331, SERVER_B-WT-0332, SERVER_B-WT-0335, SERVER_B-WT-0336, SERVER_B-WT-0348, SERVER_B-WT-0349, SERVER_B-WT-0354, SERVER_B-WT-0358, SERVER_B-WT-0359, SERVER_B-WT-0360 |
| PAPER_EVIDENCE_PRESENTATION | 4 | SERVER_B-WT-0139, SERVER_B-WT-0172, SERVER_B-WT-0210, SERVER_B-WT-0330 |
| TRACK_B_G6_HISTORY | 5 | SERVER_B-WT-0023, SERVER_B-WT-0025, SERVER_B-WT-0027, SERVER_B-WT-0028, SERVER_B-WT-0030 |
| TRACK_B_OTHER_HISTORY | 4 | SERVER_B-WT-0361, SERVER_B-WT-0372, SERVER_B-WT-0374, SERVER_B-WT-0375 |

実行順序は[READY83_IDS.csv](READY83_IDS.csv)のsequence。実path/branch/HEAD/common Git/status bindingは本人専用PRIVATE_83_EXECUTION_BINDINGS.csvに保持。対象は固定83件からの合格集合で、連番やbranch名で拡張しない。一括承認・同一の実行で83件全てを順に処理できる計画であり、恣意的に数件ずつの追加承認へ分割しない。

## 保持する資産

83件のHEADを保持するlive GitHub tipとlocal refを確認。tracked clean・untracked0・ignored0で、元の限定依存監査と同HEADを再利用し、新規SOURCE/control差分にも該当する具体的外部依存を検出しなかった。必要なtracked SOURCE/code/文書/result/図/失敗/STOPは同じGit treeとその履歴から取得できる。非公開binding表から保持refと復元commitを辿る。Git branchや共通Git/object storeは削除しない。

管理外資料がcheckout内にないことと、Git外の全研究原本の保全は別である。既知の外部provider/input/private raw/checkpoint/archive/runtimeは対象外の原位置に残し、今回コピー・移動・削除しない。以前のworktree外参照4077件、B側4474件、editable設定、既知Git alternate、本人UIDのcwd/argv/fdを使用した限定scopeの判断であり、未知の将来jobや全環境不在の証明ではない。

## 削除直前preflight

1. 明示指示のID集合と上記83件を完全一致させ、非公開表の実path/branch/HEADと登録backlinkを再確認する。絶対pathを推測しない。
2. 各対象のHEAD・branch・tracked差分・untracked・ignored・lock・現在の使用をfreshに確認する。元HEADから変化したIDや新規資産・具体的依存を持つIDは保留する。
3. 各HEADを保持するGitHub refまたは到達性とlocal保持ref、共通Git/object/alternateが候補外で存続することを確認する。Git管理外資料が新たに出たIDは保全完了まで保留する。
4. B3 SOURCE/baseline/provider/input/private/runtime/cache/Git、旧grant/ledger/marker/STOPを対象外に保つ。B3状態を判断する作業をこの整理へ混ぜない。他環境への接続やprocess操作はしない。
5. 合格IDへ、所属する保持Git管理領域から通常の `git worktree remove` を1回ずつ実行する。失敗時にforce、手動削除、clean、reset、prune等で拒否を回避しない。

## 保留と全体STOP

HEAD/branch/status変化、未保全原本、個別lock/利用、固定SOURCE/input/control不整合、Git通常remove拒否はそのIDを保留し記録する。共有Git/object/alternateの破損・保持先欠落、対象IDと実path対応の共通異常、B3保護領域との交差はバッチ全体をSTOPする。他chatの無関係なHEAD/status変更だけで全研究を凍結しない。

## 削除後の監査と復元

成功したIDだけ登録と物理checkoutが無くなったことを確認し、branch/重要commit/共通Git/外部private原本/保全archiveが残ること、保留IDとB3関連資産へ変更していないことを確認する。復元時は非公開表の保持branch/commitから新規checkoutを作り、必要なprivate archiveを別の空directoryへ安全に戻す。固定path契約を変える復元・再実行はその契約の条件と権限を別に確認する。今回の計画は科学再実行やSTOP解除を許可しない。

実際の解放容量は未計測。過去の容量を現在の削減効果として提示しない。
