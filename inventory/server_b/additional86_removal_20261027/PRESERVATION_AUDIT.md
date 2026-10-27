# 追加86件の固有情報・Git履歴保全監査

2026-10-27 JST。対象は[固定86 ID](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/50612696d8ef7db322729fd3946b2033d3d39e01/inventory/server_b/bulk83_preservation171_20261027/NEW_TECHNICALLY_READY_IDS.csv)のみ。元171件からの候補86件の利用終了を今回の指示に基づきWORKSPACE_END_APPROVEDとし、以前のPENDING_OWNER_DECISIONは旧資料のまま保持した。新しい承認証明書・grant・認証手順は作成しない。

## 管理外資料の保全判定

既存94 archive・3 Git bundle・SHA member manifest・47小型copy・stage/working/indexの復元検証を再利用した。171件全体のarchive作成・全展開・無関係な大容量原本の再コピーはしていない。対象86件に対応する43個の元archiveと3個のruntime補足archive、および25 HEADのbundle bindingをIDで照合した。

86件のfresh statusと列挙したsource-leafを以前のmanifest/snapshotへ対応させた。77,994 fileは保存時のsize/mtime/inode等が一致し、検証済み保存SHAを再利用。runtime等の4,550 fileは現在のSHAが保存member SHAと一致した。literal symlink 12件とempty directory 11件も対応を確認した。追加保全が必要な新しい差分は0だった。

除外した再生成可能なcacheは、SOURCEのあるPython bytecode 2,763 file、pytest cache 11 file、matplotlib font cache 6 file。各pathの現在の種類と再生成条件を再確認し、科学SOURCE/raw/result/checkpoint/STOPの固有fileをcacheとして破棄していない。source-byteの照合であり、数値の再解析やSOURCE実行ではない。

tracked/staged変更を持つ対象2件では、現在のstaged/unstaged binary patchとindex-entry列が、既存の検証済み復元資料と一致した。元3件の差分保全を再利用し、対象外1件のworktreeは保持。原worktreeでadd/stash/reset/cleanは行わない。必要な差分・index・working fileはcheckout外の保全領域から再構成できる。

## 単一forceを使用する条件

69件は、linked checkout・固定HEAD/branch/index・ローカル資産の現状、固有情報の保全、source/control/runtime依存、lock/本人UIDの使用なし、common Git ownerでないこと、archive/bundleが対象外にあることをIDごとに確認した。各削除直前にstatusだけでなくignored領域とruntime配下のleaf集合も再列挙したため、ignored directory内の新規fileも保全確認をすり抜けない。人工の小さいfileでこの検出を確認した。

条件を満たした69件だけ通常removeに単一の--forceを指定した。lockやGit異常を回避する目的には使用しない。double force、clean、reset --hard、prune、gc、手動再帰削除は0。clean17件はforceなしの通常removeを使用。全件exit0と物理directory/登録消失を確認した。

[ID別結果](ADDITIONAL86_REMOVAL_RESULTS.csv)にはforceの回数、固有情報の保全、cache除外数、stage/working状態、既存[保全manifest](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/50612696d8ef7db322729fd3946b2033d3d39e01/inventory/server_b/bulk83_preservation171_20261027/PRESERVATION_MANIFEST.csv)のID、bundle対応を記録した。詳細な実path/branch/完全HEAD・member原本・復元手順は非公開領域へ保持する。保全後の新しいsource-leaf変化や具体的依存は検出していない。

## 削除後

元local branch/tag refの変更0。全86 HEADとそのtracked SOURCE/code/文書/結果/失敗/STOPのcommit/blobを、保持common Gitから削除後にも取得可能と確認。live tipで保持を閉じなかった25 HEADについては、以前検証したfull-ancestry bundleの保存refを削除前に照合し、3 bundleは削除後も保持した。Git履歴と管理外原本の保全を混同しない。

94 archive・3 bundleとmember manifest・復旧索引の存在/size/mtime/inodeに削除前後の変化なし。今回使用したmember manifestのSHAも一致した。元のprivate raw/input/result/checkpoint/失敗/STOPは、保持Gitと検証済みarchive・copyで取得可能。旧archiveを上書きせず、対象外の外部原本は移動/削除しない。

**archive/copy/bundleは同じfilesystem上にあり、独立障害域バックアップではない。** 秘密情報やprivate原本、内部配置は公開していない。
