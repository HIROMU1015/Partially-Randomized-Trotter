# H2 PR資源評価：専用環境と新規チャットへの引継ぎ

2026-10-19（Asia/Tokyo）。環境の隔離準備だけを完了。H2のPhase 0科学検証、F/FP、D、A、S_STATS、P最終sealはいずれも未実施。

## 作業場所と分離範囲

- Repository：`HIROMU1015/Partially-Randomized-Trotter`
- H2 branch：`codex/h2-pr-resource-preparation-20261019`
- H2 worktree：`/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h2-pr-resource-preparation-20261019`
- 固定checkout基点：`d728d088c010089c6b672eef0521261791b2d209`
- H2公開用準備資料・結果root：`artifacts/resource_applicability/track_a_h2_pr_resource/2026-10-19/`
- H2私有root：`.h2_pr_resource_private/2026-10-19/`（Git対象外）

実質的なH2評価は、現在進行中のH4検証チャットから独立した新規Codexチャットで行う。この準備ではチャットの新規作成は行っていない。引継ぎ先は共通AGENTS.md、本worktreeのAGENTS.md、PROJECT_MAP.md、本書を読む。

Git object storeは通常のlinked worktreeとして共有する。H2のbranch、HEAD、index、作業ファイル、準備記録・将来の結果領域は別である。新しいH2 branch/worktreeのGit管理情報を追加した。H4のSOURCE・契約・grant・ledger・marker・STOP・結果・実行プロセスを変更する操作は行っていない。H4の現在の実行状態や科学的正しさを再監査したとは主張しない。

checkoutに含まれるH4等の履歴・成果物は基点commitの参照資料であり、H2の成果ではない。参照元の可変worktreeへsymlinkして実行することや、H4のgrant・cohort・入力・保存先をそのまま使うことを避け、固定Git objectまたはH2側snapshotとして来歴を結合する。継承したH4専用runnerは直接実行せず、H2専用adapterと契約の準備後に接続する。

## 固定参照

| 対象 | commit |
|---|---|
| 共通プロトコルv1 | `9036d8a87ad1c38e83b9ff654b412dde7041e9d3` |
| H4 Phase 0 | `80652ab6b42cf0c11ee97485214027f507b1bf0a` |
| H4 D完了 | `406f70570323366da185355550a3900388717fbb` |
| H4 A50完了 | `85cb5b4d1808434165c421115ed09dab3b63dd00` |
| H4 S_STATS完了 | `d728d088c010089c6b672eef0521261791b2d209` |

前の読取確認で上記commitをGitHubから参照し、今回ローカルのcommit object存在も確認した。結果commitと実際の科学SOURCEは区別して追跡する。基点checkoutに存在しない別branch上の文書・実装は固定SHAから読む。H6/H8の独立確認・native DF・並列化・seed管理・one-shot・監査の参照も、引継ぎ先で必要な固定SOURCEと対象fileに絞って追加する。

## 引継ぎ先で構築する評価

linear H2、結合距離1.0 Å、STO-3G、DFはtol-only 1e-8でrank自動、T=.8、状態準備費用P=0。精度はε=.05/.01/.005/.001/.0005、通常signed bias、allowance倍率1、full Hadamard wrapperの両軸compiled RZ×必要shotを合計する。

1. Phase 0／入力・F/FP：既存H2資料のSOURCE・入力・compiler同一性を監査し、再利用を確定する。新規生成・実分子検証が必要なら具体的な科学契約を先に提示する。
2. D：full PF、B0、B0I、S2/S4をH2に適した集合で事前登録し、承認後に5精度の登録内最良PF対照を取得する。
3. A：H2用PR候補を事前登録し、有限標本n8を基本として探索する。D/Aとも縮退、重複、不適格、欠測、未確定を区別する。
4. S_STATS：保存D/A結果を入力としてε=.05/.0005の独立確認nを設計する。モデル・感度条件・入力・数値解析予算を固定し、別途承認を得る。
5. P最終seal：候補、PF対照、採用n、H2 SOURCE、独立seed、科学call予算、実行契約を固定してmandatory STOP。Pのsampling・compileは実行しない。

H4のrank10、D190、A50、n1108/78を移植しない。PR不利や適格候補なしも正当な結果として記録する。既存H2データの再利用可否・実rank・候補・n・seed・予算は今回未判定であり、数値を埋めていない。

## 承認境界と保全

静的調査・SOURCE準備・人工テストと、実分子による入力生成／signal／sampling／compile／S_STATS等を分ける。各科学stageはSOURCE・入力（生成時は生成規則と既存入力）・call予算を固定してユーザーの明示承認を得る。one-shot実行後にmandatory STOPし、次stageへ自動移行しない。自動retryを行わず、異常や契約不成立時は証拠を保存してSTOPする。P科学実行は今回の到達目標に含めない。

H4/H6/H8のSOURCE・raw結果・既存契約・grant・ledger・marker・STOP・cohortを変更・混合しない。他worktreeのHEAD/status変化は観測情報であり、H2が実際に使う封印対象以外を凍結条件にしない。共有環境を変更せず、実空き資源に応じた計算プロセスの並列化・高速化を行う。過去のCPU/RAM/worker上限を新規実行の恒常的上限へ移さない。

成果はH2 branchにcommitし、pushは公開承認の対象を具体的に確認できた範囲だけで行う。今回の隔離準備ではpushしない。

## 前の指示確認で挙げた未採択の補足案

- 片方／両方の主endpointで適格PR候補がない場合に、該当S_STATS/Pを適用外として終了する例外の扱い。
- S_STATS終了後の提案n・希少event限界の受入と、P最終nのユーザーによる採用手順。
- 新規入力生成Fと実分子検証FPの承認単位。

これらはCodexが提案した補足であり、今回の分離指示によって採択されたとは扱わない。依存しない静的準備を進め、関係する契約を具体化した時点で判断事項を明示する。

## 今回の確認と次の担当

今回の確認は新規worktreeの初期clean状態、固定commitの存在、H2保存先の分離、準備metadataのschema/承認flag、差分範囲、private領域のGit除外を対象とする。科学module import、入力生成、signal、sampling、compile、S_STATS数値解析、新seed、有効grant、ledger、科学markerは作成・実行していない。独立科学レビューではない。

次はユーザーが新規H2チャットをこのworktreeで開始し、Codexが本書を引き継いで静的な再利用監査とH2契約準備を進める。各科学実行は別途ユーザーの明示承認を得る。
