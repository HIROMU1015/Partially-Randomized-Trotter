# 3日以上前のbranch：結果保全確認後の追加削除（2026-11-09）

**対象137本を確認し、73本を追加削除、64本を保持した。削除操作で全remoteは296→223本。その後、別作業の新設1本を反映した最新一覧は224本。保持64本の分類はKEEP33・HOLD31。** 前回の1週間条件による1本削除と区別し、この年齢条件拡張による追加削除は73本である。

対象は開始時点の72時間前 `2026-11-06T18:45:50+09:00` 以前の先端committer日時を持つ枝。枝名の日付を使わず、最後のref push日時と同一視もしない。以前削除された候補を再対象化せず、既知の再出現枝も既存HOLDを継承した。3日未満の枝は対象外。

[対象137本の全件処置・保全先・SHA・結果blob](branches.csv)、[最新224本の全件分類](remaining_branches.csv)、[個別根拠・実行集計JSON](results.json)を参照。

## 削除した系列

| 系列 | 追加削除 |
|---|---:|
| H8 / H46 | 3 |
| O2 / A1 | 11 |
| O2 / other | 10 |
| Evidence / publication | 3 |
| Other / shared | 2 |
| RPE / measurement / CS | 44 |

## 結果が統合整理されていることの確認

各候補の先端commitの差分から結果・報告を抽出し、削除対象外の後続保全先の同pathに同一Git blobが残ることを確認した。SOURCEやresource logだけの一致で済ませず、Markdown報告またはRESULT/SUMMARY/VALIDATION等の結果・判断資料を枝ごとに確認した。準備・不成立・停止等の結果も元の意味を保持し、科学的なPASSへ読み替えていない。

元tipが保全先の祖先であり、未統合commit数は各73本すべて0。したがって元SOURCE・入力・結果・失敗・契約・grant/ledger/marker・STOPの旧版は、元full SHAとpathから取得できる。保全先の最新SOURCEを元科学SOURCEへ付け替えていない。すでに存在していた結果・報告と履歴を確認したもので、削除条件を作るための新規科学統合・再計算は実施していない。

| 残した保全先 | 元枝数 | 削除前後に保持確認したSHA |
|---|---:|---|
| `codex/h8-g04-o1-evidence-integration-20261103` | 1 | `879a28940392a1f2ea58f32f1a320d4b6822a6dc` |
| `codex/h8-stage-c-execution-20261105` | 2 | `eb07156a37b1938ed3aed95c13abbb545466c573` |
| `codex/o2-vacuum-h4-preparation-20261108` | 16 | `781e155d0907a82c962e41ab8e89f985735bc49b` |
| `codex/pr-paper-claim-storyline-20261106` | 7 | `3fbf8cac2503722f1e329e079da74a493afdbc3a` |
| `codex/track-d-consolidated-20261109` | 3 | `141fa7624d74aa5deacbd752d09df516cad9206f` |
| `codex/rpe-input-kernel-preparation-20261105` | 6 | `ccbd7aa974eda1e017138066a70dc8ac35221907` |
| `codex/rpe-measurement-backend-contract-preparation-20261105` | 38 | `655bf6cb84ab161706baf843a8ae255da39bf7a6` |

この7本の保全先は削除集合に入っていない。保全先そのものに未統合の独立成果がある場合も、その枝は保持した。新しいarchive tagや管理branchを作成していない。旧archive tagを含む103 tagは全SHA不変。

## 64本を保持した理由

- mainと保護対象、現在の実在checkoutが使う枝は維持。現ホストの主repositoryと既知の別6 repositoryから登録を再取得し、削除73本に現checkoutの結合がないことを直前確認した。対象137本中43本に実在checkoutがあり、全て保持した。
- 研究Cの既存HOLDは解消しない。今回の年齢対象に入る12本も保持した。旧Cの削除56本、Track Dの削除38本、別作業のTrack C/O3削除21本、前回の追加削除1本は再削除対象にしていない。
- 前回からの具体的HOLD9本を維持。LOCAL未公開差分、H46のlocal/remote分岐、pr2固定入力context、Track Bのbranch名を実照会する検証コード等を、年齢だけで解除しない。
- 保全先が自身の枝である独立履歴、結果・報告の統合を今回確認できない枝は保持。保全先への新しいmergeを自動実施していない。

理由は重複するため、理由別件数の足し算は保持64本と一致しない。個別理由はCSVに記録した。削除直後の223本の分類はKEEP131・HOLD92である。

## 実行と事後照合

削除は `2026-11-09T18:53:24.023283+09:00`〜`2026-11-09T18:53:26.987841+09:00`。既存 `retire_archived_remote_branches.py` のremote照会・origin確認・期待SHA付き削除引数・Git実行ヘルパーを再利用した。旧固定352候補の実行runnerを使い回していない。

直前のGitHub APIで全対象のprotected=false、open PR0・ruleset0を確認。実GitHub refで各対象SHAと各保全先SHAを再確認し、明示73 refだけをatomic pushと個別期待SHA leaseで削除した。終了コード0、73/73で不在と元tipの祖先到達を確認。削除操作による対象外refの変更は0件、全tag不変。

GitHubのremote ref削除だけを実施した。元local branch・worktree・未追跡/ignored原本・SOURCE・入力・raw・結果・既存契約/STOP・jobは変更しない。共有設定、他ユーザー、広域prune、gcを操作していない。科学計算・数値再解析・回路compile・詳細科学レビュー0件。外部のLOCAL/SERVER1/SERVER2全原本や未追跡rawを新たに完全検証したとは主張しない。

本追補は既存 `codex/research-c-branch-cleanup-20261109` の直系子commitとして文書だけ公開する。CSVの整理branch SHAは公開直前snapshotであり、公開による当該refの更新だけを最終照合で別記する。過去の297本・296本の分類表とSTOPは当時の履歴として維持した。

## 次の担当とSTOP

- ユーザー：保持64本も整理する場合、実在checkoutの継続用途、旧9本の未公開差分・固定参照、研究C HOLD、独立tipを保持するか統合するかを具体化する。3日条件の削除許可自体を重複して出す必要はない。
- Codex：上記の事実または必要な参照変更・統合の範囲が定まった後、今回の年齢・結果保全条件の許可内で再判定する。固定科学契約やSOURCEの変更、新しい統合・科学実行へ自動進行しない。

**公開後の文書bytes・実remote・保全先を照合してMANDATORY STOP。今回確認できた73本の追加削除は完了。**


## 公開後照合で観測した新設ref

別作業が `codex/pr-reference-v6q-q2-20261109` を `ca19707b5c0c1cefa338d47987c27f338cb4cf67` で公開した。先端日時は2026-11-09 18:54:17 JSTであり、今回の3日条件の対象外。現ホストの実在worktreeとHEAD一致を確認しKEEPとした。この監査による科学branch作成・科学実行ではない。

最新一覧は224本（KEEP132・HOLD92）。削除した73本の不在、保全先7本のSHA、全103 tag不変を確認し、追加refを一覧へ反映した。公開による整理branchの直系更新を別に照合してmandatory STOP。
