# Batch 2〜4：57 worktreeの整理実行結果

2026-10-27 JST。PUBLIC_REDACTED。事後確認：2026-10-27T02:34:53.073076+09:00。
固定入力：Batch 1完了commit `c6e7623c95243b9ff1d35be575e75763ad2aaa16`、3環境最終計画commit `45c78a67992cc2534bd1a9bb85a2fc1985cdf2eb`、139件原集合commit `2caa236fa7332872b5a346f0744318d48d59f6ad`。

**57件すべてが最終preflightに合格し、通常のgit worktree removeを各1回実行した。削除成功57、最終保留0・失敗0。Batch 1の削除済み7件は再操作していない。** 個別結果は[BATCH2_4_EXECUTION_RESULTS.csv](BATCH2_4_EXECUTION_RESULTS.csv)。

| バッチ | 固定CSVから抽出した対象 | 合格 | 削除成功 | 最終保留/失敗 |
| --- | ---: | ---: | ---: | ---: |
| Batch 2：QC | 18 | 18 | 18 | 0 |
| Batch 3：Research B | 34 | 34 | 34 | 0 |
| Batch 4：N1/N3検証済み | 5 | 5 | 5 | 0 |

固定LOCAL_64_BATCH_READINESS.csvの正確なID集合を非公開原本のpath/branch/完全HEAD/Git common/backlinkへjoinした。匿名IDやbranchから実pathを推測していない。正式運用指示2026-10-25版の§8.1・§8.4・§10を実在ファイルで確認した。今回の明示許可は57件のcheckout/登録だけの除去と139件の保全準備であり、承認証明を新設していない。

## preflightと実操作

本人所有・正規path・主checkoutやarchiveとの分離、linked登録とadmin/gitfile backlink・common Git・HEAD/branch・lock、既知mount/aliasのLOCAL実体、tracked/untracked・ignored全leaf、可視本人UID process cwd/引数、直近control/provider/input metadata差分を各操作直前に確認した。ignored内の案内symlinkは記録先との一致、再生成可能なpyc/pytest cacheは既存leaf一覧との一致を確認し、科学raw/結果/STOPと区別した。新しい不明ignored資産は検出していない。

57件の実行側namespaceで元のLOCAL directory device/inodeとGit identityが一致することを確認し、target/配下mountなしを確認した。通常の主checkout配下にあるlinked worktreeも、主checkout自身や他linked checkoutと取り違えないよう登録関係で区別した。最初の6件ではpreflight helperの入力準備とmain配下の包含判定を整えるまでremove未実行で保留し、修正後に全条件を再確認した。拒否をforceで回避した例ではない。

既存206 control metadata・12既知editable pointerの差分と、前回以降にHEADが変化したcheckoutの最近のcontrol/doc 65件を限定照合し、57 pathの新参照0・読取エラー0。利用中のjob/他Codex・固定SOURCE/pathの新しい具体的束縛は確認scope内で未検出。A_R1/B_PATH_1は既存75件の結果を再利用し、A/BへSSHや全件再調査をしていない。未知の全namespace・未記録再開の不存在までは保証しない。

各IDで通常removeを1回行い、directoryと登録の除去、branch/完全HEAD・commit objects、archive/manifest、他の既存登録維持を確認した。force、prune、clean、reset、gc、手動強制再帰削除、lock解除、branch削除、復旧操作は0。

## 事後保全と登録数

対象共有Gitの登録数は**338→281**。今回57 IDのdirectory/登録消失を個別確認し、数の差だけで成功認定していない。観測区間内の新規登録は0件。別研究worktreeのHEAD前進1件は、旧HEADの祖先保持とbranch等の登録field維持を確認し、今回のremoveと別の並行変更として記録した。各remove前後では他の既存登録に変更なし。並行作業を巻き戻していない。

57件自身のlocal/remote保持branch、重要commit・SOURCE/result/STOP参照を維持した。0003の固有HEADをclosureだけで代替せず、Research B 0283/0286や旧SOURCE11件を対象に含めていない。主checkoutのHEAD/status・main、common Git objectsを保持し、研究branch/科学SOURCE・契約を編集していない。

57件分の既存archive **27,793版別path**はmetadata不変。過去の全SHA-256/Git blob検証を再利用し、regular/readable・source inode独立性・旧manifestを確認した。再コピー・無条件全hashは0。対象68件全体の既存8,284実ファイル／511,618,500 bytesと保存先内hardlinkの保全記録を維持する。コピーは同じローカルdiskであり独立backupではない。科学runtimeの移設・完全復元は保証しない。

既存原本・manifest等137 metadata filesと過去inventory49 filesは不変。SOURCE/input/result・STOP/grant/ledger/markerのbytes・one-shot/retry/resumeを変更していない。対象外139件・Group1保護20件・旧SOURCE11件はdirectory/登録を保持。サーバーA/B環境・DF-native L_D=0 B3には接続/転送/job操作していない。旧Track B/G6/W2CのB3と同一視していない。

## 非公開記録・公開・次の範囲

本人所有の新規非公開領域へ、各IDの元path・branch/HEAD・Git管理/backlink・archive/manifest・preflight・remove開始/終了・事後確認・通常Gitによる再checkout参考情報を保存した。復旧自体は行わず、元ignored cache/guide/runtimeまで完全復元できるとは主張しない。削除済みIDを公開失敗時に再実行しない。

公開は本directoryのCSVと報告2資料だけ。内部絶対path・host/IP・private raw/原本配置・credential・保全file本体を載せない。独立公開cloneで最新remoteを確認し、Task Bの新規3資料と合わせて既存inventoryを保持した通常fast-forward更新を行う。commit/remote一致・読み戻しは非公開公開記録と最終通知で報告する。

**Task Aの削除は終了。Task Bの139件は保全・準備のみで削除0。旧SOURCE11件・Group1保護20件・他環境やGitHub branch整理へ進まず、公開後mandatory STOP。**
