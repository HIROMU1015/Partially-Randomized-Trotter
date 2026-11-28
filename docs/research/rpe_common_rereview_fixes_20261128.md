# H4・H6・H8 共通版 再レビュー修正・検証記録（2026-11-28）

対象: Claude の16:07 JST再レビュー N01–N43、および保存物 S01–S14。公開共通ブランチ `codex/rpe-common-hchain-fleet-20261126` の確認済み基点は `d458cde3688cd8068c8075d2ecbce668990c5ff0`。直前の `bf5b23570b` と基点commitが既に導入した保存物改善は重複実装しない。作業ブランチは `codex/rpe-common-rereview-fixes-20261128`。実行中の固定版・既存run・科学SOURCE・PBS予約・現物の保存データは変更しない。登録上の中央currentは `romutail-pbs-independent-f37e1ebd0963`（公開ソースとは別）。H4の登録run `search-538c35a68be6425b1b4c` は `romutail-h4-independent-continuation-ca27a9ff3a59` 配下。RomuTailでH4 PID 665284およびworker PID 669005の継続を読み取りで確認（`rpe-common-pbs-storage-a6ba498f82` とH4固有overlay）。H6/H8の実worker固定版はPBS接続不能のため現物照合できない。再照合でも公開branchは同じSHAだった。PBSへの接続・数値startup・新規科学計算は実施していない。

判定は以下を区別する。**対応**は指摘の具体的な修正を実装し人工検証の対象にしたもの、**既存**は最新基点に実装済みで再利用したもの、**部分**は修正または既存機能を確認したが改善案の一部が残るもの。**実機待**は人工検証済みの機構について実機での成立確認が残るもの。部分・実機待を完了扱いしない。実機性能向上倍率・実データ回収量は今回の結果ではない。

## N01–N43

| ID | 判定 | 実施内容・残る確認 |
|---|---|---|
| N01 | 対応（代替） | 固定kernel・startup依存の静的照合とsaved exportのStart時SHA・内容照合。数値startup実行後の不一致をPLAN SHA・stage付き診断とCONSUMED_STARTUPで分類し、旧traceback互換も維持。真の起動前不一致のみ既存prelaunch回復経路の対象。実行済みstartupをpre-scienceへ再分類する提案は科学契約と矛盾するため採用しない。 |
| N02 | 対応 | `HCHAIN_INPUTS_EXHAUSTED` を科学完了と区別して閉じる。未回収ownerがいると閉じない。無関係なCENTRAL_SHORTLIST更新を監視から外した。 |
| N03 | 対応 | 固定blob byte予算、cycle再利用、同一cycle carrier/aliasの限定owner照合、固定locator単位schedule key cacheを実装。既存ownerを検出すると通常照合に戻る。人工cache/隔離/route回帰成功。実H4の時間短縮は未計測。 |
| N04 | 対応 | 有限準備予算は固定providerのselect_casesと実batchesを再利用してphase境界込みでstage別batch件数を数える。共有anchorは独立childごとの実作業分を数え、架空の再利用を仮定しない。正の需要に最低1件。人工7葉phase分割と独立H6/H8 fixture成功。 |
| N05 | 対応 | quote/publishはcandidate snapshotをlock外で処理。価格registry取込を専用writer lockへ分離し、既存候補lockを外して取込後に候補・registryを照合。変更時は上書きしない。人工回帰成功。 |
| N06 | 対応 | 候補差分journalに加え新規runのcompact_family_metadata opt-inでREADY固定configuration/templateをSHA参照へ分離。旧snapshotはそのまま、Resumeで検証付き復元、公開時に従来形式へ戻す。人工round-tripと改変拒否。 |
| N07 | 対応 | STATE比較でworker時刻・remaining_seconds・scheduler.resources_usedという揮発値を除外、実状態の変化は保持。 |
| N08 | 対応 | PBS終端skip、native終端資源集約、観測ACK永続化、未変化SLOT_STATE/RESOURCE_OFFER書込省略。新規専用mailbox spoolの `mailbox_indexed_spool: true` でactive attemptのみ走査、終端resource/prefix/未ACK観測を永続集約。旧spoolはstat-bound索引で互換を維持、混在writerは拒否。人工再起動・重複ACK回帰成功。 |
| N09 | 対応 | コピー候補の存在を先に調べ、既存背景測定でprefixの使用量を分離して非cache実使用量を引き、監視cache間の予算を配分。実機のchurn・計測時間は未測定。 |
| N10 | 対応 | group差分max/件数/key件数、journal追記、receipt読取りをstate.cv外へ移動。既存ownerの確実なmemory予約維持に必要な確定区間のみlockを保持。人工3,000件で先頭100中央値0.401ms、末尾100中央値0.396ms、全再計算と一致。実機GIL計測ではない。 |
| N11 | 対応 | 明示した互換source組だけ暫定初期値を転用し、cold高優先classに排出待ち期限を設定。異なる固定科学SOURCEを無断で同一視しない。実機稼働率は未測定。 |
| N12 | 対応 | streaming中のwaveに全方式を含め、prepare未完時のREADY再走査に最小間隔、入力probe中はidle rotation対象外。別unitのLOCK_NB・入力保留・handoffを事前確認。人工所有競合試験成功、PBS実稼働率は未測定。 |
| N13 | 実機待 | `prepare_rpe_partition_handoff.py` で既存prepare/register入口・lockを再利用した明示的移管。gatewayに到達できることが必要。接続不能時の所有移管を推測で許可しない。 |
| N14 | 対応 | provider dir/module/priority/phase policy/固定kernel/保存export path・key prefix・hash fieldを宣言可能。旧H6/H8互換の既定分岐を維持。新providerが明示したkernel名は固定profileと一致し実Core観測も一致する場合のみ受理する。架空分子・追加kernelの人工検証成功。 |
| N15 | 実機待 | qsub応答欠損時に正の一意な所有job名証拠だけでID復元、空・不明・複数ならHOLD継続。keeper側書込み失敗や実scheduler履歴の検証は残る。 |
| N16 | 対応 | 転送timeout後に次回のchunkを半減し既存retryを再利用。受信中にSHAを計算するCAS一回書込み、全体SHA照合後のみlink。人工timeout/再試行/改変拒否確認。実回線性能はPBS待ち。 |
| N17 | 対応 | 完了timeout/backoff、採用済みownerのexact bundle digest照会で再送省略、archive cache列縮小、5秒CPU窓、照合済みsupervisor peak統合。旧REQUESTは安定statと固定SHAに束縛したconfiguration_idを再利用、exact owner待機時にRESERVE/eligibility/host offerの無関係変化を再計画理由にしない。他の入力待機の必要triggerは維持。 |
| N18 | 対応 | 既定OFF機構を凍結する方針。R10容量差分台帳、R43 persistent poolを新規に有効化せず、R30/R31/R38等はPBS実検収までOFFを維持。来歴と復元互換性のため削除しない。 |
| N19 | 対応 | fairness予約をwindow末尾に限定、完了laneのstandby viewをdetachして他laneを妨げない。既に所有された仕事は取り消さない。 |
| N20 | 対応 | 走行中RSSは下限のみに使い、終端観測だけを測定済みとする。旧schemaの混入も再構築。2回目pollで追加過剰枠を出さない回帰。 |
| N21 | 対応 | gateway capabilityを確認、legacy非対応時はUNKNOWN、fence失敗は該当ownerだけHOLD、未回収のまま再照合。 |
| N22 | 対応 | 新SHA helperが旧overlayにない場合も厳密なfile・Git blob hash比較を従来方法で行う。固定SOURCEを変更しない。 |
| N23 | 対応 | advisory判定は無関係な通知で破棄せず、実際のclaimは厳密照合のまま。待機のbusy loopを抑える。 |
| N24 | 対応 | `_merge_pf` をcopy-on-write、公開時にconditionでroutes/expectedを切替。旧selectorの辞書を破壊しない。 |
| N25 | 対応 | DF ready通知をkeyの集合からrevision単位へ変更し、同一候補の更新も通知。 |
| N26 | 対応 | 固定generated configuration本文のmemoを共有して、候補の可変部分のcopyを残す。 |
| N27 | 対応 | 明示exact replay ownerが未完なら固定cycleを消費せず受理待機し、通常work-conserving OFFでも回収経路を維持。STOPは保持。人工owner状態の検証と広域回帰。 |
| N28 | 対応 | 低水位の有効な過去sampleを測定遅延中に限定再利用し、現在の物理空きは再確認。圧力時は保留。 |
| N29 | 対応 | 単発cold probeは従来経路、反復viewだけ30秒stat-bound cache、runtime package情報はpollを跨いで再利用しない。人工cache再利用・変更検知。 |
| N30 | 対応 | 新graph全体PLAN SHAをrequest契約へ伝え、物理host＋PLAN単位のstartup holdで区別。旧graphは変更しない。 |
| N31 | 実機待 | successorの状態を先に照合し終端なら新しいgenerationを許容。STOP時には正確な前任予約に紐付いた未接続の後継だけをreleaseし、現在の予約は維持。人工fixture成功、実PBS検収は残る。 |
| N32 | 実機待 | opt-in lease v2でtoken変化をworkerのmonotonic時間、gatewayの受信時刻で判定。旧lease互換維持。異なるclock値と重複tokenの人工fixture成功。実host時計差・実移管は残る。 |
| N33 | 実機待 | durable JSON再読のdigestを使い、不一致の消費ACKを認めず他の応答を処理。実PBS再送・消費ログ照合待ち。 |
| N34 | 実機待 | SUBMIT.lock下にqsub開始・job・session開始が存在しないことを確認し、明示releaseされた未投入予約を終端扱い。実scheduler検収待ち。 |
| N35 | 対応 | 新規固定PLANのbatch size明示と宣言startup_prefixesの新provider interfaceを追加、full startup照合と旧providerはそのまま。人工prefix選択・exact schedule check成功、実数値startup未実施。 |
| N36 | 対応 | ENOSPC/EDQUOTの診断保存失敗でも実測を失わず、新規取得を停止して整理経路を残す。科学受理の厳密書込みは緩和しない。 |
| N37 | 対応 | 未実体化の予約memoryを現在のMemAvailableから差し引き、共存プロセス分も二重使用しない。人工境界検証。 |
| N38 | 対応 | native/PBSでUNKNOWN検証再試行の間隔を抑制、同じstate/detailを再送・再保存しない。 |
| N39 | 対応 | numpy/scipy.linalg両importの実kernel metadata照合、READY再告知の独立retry。RomuTail .venv311でHASWELL/NEHALEMのimportのみ各2個の一致Core行を確認。固定H8 scoped数値はN01の未許可範囲。 |
| N40 | 対応 | inodeに加えsize/mtime/ctimeを確認、読取りの再試行を有限化。 |
| N41 | 対応 | 選択関数のAST比較を確認済み固定基点 `512978edcb` に整合し、4関数を比較対象から除外しない。 |
| N42 | 対応 | S01–S14の既存実装・追加実装・隔離移行/復元の回帰を統合。実データの物理整理・稼働版への導入は今回の利用者指示に従い行わない。 |
| N43 | 対応 | 既存回帰にphase境界、alias owner、旧helper欠損、index再開、価格lock、転送timeout、静的メタデータ、graph hash、保存export、diagnostic archiveを追加。広域人工回帰成功。全テストの修正前反証を再現したとは主張しない。 |

## S01–S14

| ID | 判定 | 最新共有版と今回の扱い |
|---|---|---|
| S01 | 対応 | 既存compact/restoreを利用。停止runの旧版向け明示オプションと通常Start/Resume前の復元、旧Gateway用の明示復元入口を追加。Git blob・ファイルmode、固有source・証跡・固定configを保持する人工往復。可変workspace共有は採用せず、共通Git objects・sparse-index・bundle再利用を維持。 |
| S02 | 既存 | 停止sessionのRPCを外部mountへ退避して元の再送pathを保持、検証された大ファイル選別と消費receipt機構あり。T7実データは今回移動しない。 |
| S03 | 既存 | 停止runの完全一致event三重複をhardlink、`install_files` の共通event実体共有を人工検証。 |
| S04 | 対応 | 新規runのsplit_eligibility opt-inでdeferred_hosts/host_decisions/decision_sha256を小さな別表へ分離、claim/read/Resumeは完全decisionを復元。永続入力とhost fingerprintはjobs側に維持。完全一致はwrite transactionなし。人工fixtureで大きなbase行・updated_epoch不変とclaim・復元・イベント完全一致を確認。 |
| S05 | 対応 | 受理REQUEST hardlinkに加え既存queue ownerと一致するreadinessのrequest参照化、新規runのcompact_preparation_graph opt-inで共通SOURCE/path/runtimeをSHA参照化。人工graph復元・改変拒否。停止済みH4 runの読取標本100件ではroute.configurationsが6種類（生計555,984 byte・重複除外31,590 byte）、templatesも6種類（187,992 byte・重複除外10,773 byte）。最新4b1057のcompact_candidate_requestsを再利用し、既知PF routeでは当該configurationとfirst-matchで有効なtemplateのみに縮小（expectedが同一の場合のみ）。新規run向けの正規REQUEST削減で、台帳を別の非互換形式へ変えず重複入力を減らす。 |
| S06 | 既存 | H-chainのresultをcompact JSONで保存、同一JSONの人工fixtureを検証。差分結果保存は将来改善。 |
| S07 | 既存 | local workerと中央CASで検証済み同一内容を共有。人工fixtureでinodeとSHAを確認。 |
| S08 | 対応 | 負回収防止と再pack抑制を再利用、pack前後件数・byte・index存在を集計。明示opt-in `--multi-pack-index` で既存保守lock内のGit MIDX write/verifyを追加（pack削除なし）。2 packの人工fixtureで検証成功。稼働repoのrepack/index変更なし。 |
| S09 | 既存 | settled DBだけ `VACUUM INTO`、行・連番・索引の検証後に置換する機構を人工検証。実台帳には未適用。 |
| S10 | 対応 | publication cloneの参照inventoryと、全Git object（未参照objectを含む）を保存する独立copy・移行先alternate照合を実装。既存cloneとalternateを一切変更せず、人工cloneで旧依存先が無くてもGit参照・commitが成立することを確認。現物切替は行わない。 |
| S11 | 対応 | `evidence-service` で新規run向けに固定SOURCEを別Git viewへ配置し、その内側へ証跡を保存。旧releaseの設定と通常Git locator方式を再利用し、固定service入力をbyte一致で移す。人工Prepare、pin参照、Git locator、Statusと旧release不在時の解決を確認。既存runの固定pathは不変。 |
| S12 | 対応 | 試験成果物の検証済みcold copy、外部symlink/alternateの明示、linked worktreeも対象にした独立bare object storeのstageと全object検証を実装。人工worktreeでalternateの切替後に旧planning無しでもGit fsck成功。現物移行・実processの変更なし。 |
| S13 | 既存 | 最新基点の親死亡監視を再利用。既に孤立している実processを終了していない。 |
| S14 | 対応 | 受理済みjobとbyte一致するtransfer診断だけを検証済みJSONL.xzへ集約・除去し、固定旧版の読み取り前に復元。未settled/異なる診断・科学receiptは保持。7万桁超の人工HCHAIN_RESULTをraw byteのままcold圧縮/復元。新規runの共有版gzip proofも再利用。実データには未適用。 |

## 検証

- すべて新しいworktreeの人工データ・モックのみ。通常の運用Start/Resumeは呼ばず、専用のpytest一時領域を使用。
- 初回広域回帰701成功・8失敗を修正。第2回716成功・2失敗（54 subtests）を確認。screened streamingで変化がないのに再waveが起動する回帰を修正し、H-chain/PBS pilotの48テストを再実行して成功。
- 失敗した人工controller再起動を独立の専用一時領域で再検証: `PASS`、scientific_H4_calls=0、同一attemptを回収、実行1回、重複完了の冪等性、最終resumeで重複起動なし。前回のtimeoutについては再現せず、時間依存の間欠性を否定しない。
- worker/offline/maintenance/completionの追加対象検証117成功（7 subtests）。
- 第3回広域554成功・7失敗。基点ですでに非同期化されたqsubに対して古いsynthetic schedulerのmockが子processへ引き継がれていなかったため、fixtureを同一process内の人工submitへ修正。該当10件成功。第4回広域561成功、42 subtests成功。
- 追加検証: storage/rotation 129成功（4 subtests）、PBS successor/managed gateway 34成功、Completion/障害連続性/幅完成63成功（3 subtests）、memory learning8成功、旧overlay helper欠損回帰1成功。各集計には重複があるため合算しない。
- 最終広域回帰: **606 passed、42 subtests passed（117.42秒）**。最後のsnapshot journal分離確認後、候補分離・Completion・route bindingの49件（21 subtests）を再検証して成功。alias所有者の追加回帰1件も成功。
- `git diff --check` 成功。今回の変更は未commit/未push、既存稼働版の切替なし。

## 残る検収・運用

部分対応は完了とはしない。PBS回復まではN13/N15/N32/N33/N34の機構を人工検証に限定し、既定OFFの機構は有効化しない。旧固定PLANの数値startupやprefix-levelの科学契約変更、旧runの予算・再配送条件の変更は独立の利用者許可を得た固定scopeで行う。既存の保護SOURCEやraw結果を修正しない。実データ収縮・性能測定を今回の実装テストから推定しない。共有ブランチへの統合と稼働版への導入は別の操作であり、今は隔離した修正候補だけを作った。

## 継続修正（実行環境回復後）

- 公開共通branchを再照合し d458cde3688c と同一。今回も隔離worktreeのみ。
- 最終広域人工回帰 **655 passed, 1 skipped, 45 subtests passed (119.92s)**。最後の終端index profile digest最適化は個別2件成功。保守fixture4件成功。
- 人工memory 3,000件: 先頭100中央値0.401 ms、末尾100中央値0.396 ms、総1.394秒、全再計算一致。実稼働速度の計測ではない。
- 実環境参照inventory: systemd設定41ファイル、参照10,782、読取エラー0。publication 2件とも参照あり。削除対象とは判定しない。
- HASWELL/NEHALEM import-only: それぞれ2件の一致Core行。数値startup・科学計算は行っていない。
- 稼働版・実保存データの変更なし、commit/pushなし。部分対応の内容は上表で明示。

### 新規release向けの選択機能（既存runは無変更）

- `ROUND.json` の `compact_family_metadata: true`：固定family metadataのSHA参照。公開時復元。
- `ROUND.json` の `executor.distributed.compact_preparation_graph: true`：共通準備inputのhash参照。旧graphはそのまま。
- `ROUND.json` の `executor.distributed.split_eligibility: true`：一時適格性を別表へ分離。旧版と途中で切替しない。
- これらを現在の稼働版や固定runで有効化していない。旧コードを新形式のrunへ混在導入しない。
- 停止run用の `--compact-provider-sources`、`--pack-diagnostics`、`--restore-diagnostics` は人工fixtureのみ実行。現物の移動・消去は未実施。
- 追加確認後の公開registry currentとH4 controller/worker PIDは前回と同じ。空き容量59GiB。
- 途中の広域回帰で、テスト実行中に修正SOURCEを書いたためSOURCE照合で1件失敗。最後の検証ではファイル変更を停止して再実行。

### 作業中の公開共有版更新の追随

- GitHub比較で4件の追加を検出。最新SHA `4b1057acfacbfa9ab09b2bd9f770e4fa7b6ce6a6` (`74071dda33`, `4d4e9821bc`, `a3f144bcef`, `4b1057acfa`)。隔離候補をこの基点へ前進、手元の修正と3-way照合し、3ファイルの競合を両方の機能が残る形で解消。commit/pushしていない。
- 新版のactive lineage保護、local result manifest、bundle/shared input bases、compact_candidate_requests、PFランキング共有、不要なtelemetry書込み省略を再利用。新規のcompact/diagnostics入口も `protect_references=True` に統一。
- 最新inventory（同一利用者のprocess・unit・symlink参照込み）：13,462参照、読取エラー0、publication 2件とも参照あり。
- 更新前の固定回帰679成功・1skip・45 subtests成功（121.17秒）。4b1057基点統合後の広域回帰744成功・1skip・49 subtests成功（124.50秒）。

- さらに公開コミット `ebe099c37d`（exact owner照合後の古いattempt通知抑制）を検出・統合し、隔離branch基点を `ebe099c37dfd36c84e9bd6d9a3e2e072e6483be2` へ前進。GitHub比較でも同一を確認。worker/controller/人工Resume回帰109成功・8 subtests成功（57.36秒）。最後の宣言kernel・saved export path照合追加後にH-chain/preparation回帰76成功・5 subtests成功（5.90秒）。各集計は重複を含むため合算しない。
- 前回27対応済みから今回42対応済み（N34＋S既存6＋新規2）へ前進。部分対応9（N01/N08/N42/S01/S08/S10/S11/S12/S14）、PBS実機検収待ち6は維持。N42は保存物全体の総括。全57項目完了とはしない。
- 部分対応のうちN01は固定契約の科学startup/予算条件、N08は旧writer混在時のpath発見互換、S01/S08/S10–12/S14は旧固定版の復元・Git locator/alternate・現物整理の条件が残る。旧稼働版の保護条件は緩めない。

## 最終統合に向けた追加修正

- N08：専用spoolでactive attemptの索引をASSIGNMENT公開前、終端索引をTRANSFERRED永続化後に更新。metadata-only mailbox status/resources・submit/ackの経路を接続。既存mixed spoolは移行しない。
- N01：数値startup後の失敗と数値未実行のdry-check失敗を区別。PLANに対応する診断が壊れていても起動前失敗には分類しない。数値計算なし。
- S01：停止runの明示source復元を追加。S08：既存packに限定したMIDX作成・検証を任意選択化。S10/S12：新alternate単独のobject存在確認を追加し、欠落・外部alternate依存をHOLDとする。いずれも人工fixtureのみ。S11は既存Git locator契約と外部証跡配置の整合が残る。S14は旧固定版が参照する転送証跡の削除を行わず、既存のarchive/restoreを維持。
- 追加修正を含む広域人工回帰：750 passed, 1 skipped, 49 subtests passed (132.32s)。
- 追加の公開共有版 `6948e237471ac3ad3e24c139d8007f0cd987ea2c` に含まれる履歴cache、Git bundle再利用、provider sparse-indexを統合する。
- この段階の分類は45対応（代替・既存再利用を含む）、部分対応6（N42/S01/S10/S11/S12/S14）、PBS実機待6。未対応部分を実データの破壊的変更で埋めない。稼働固定版、PBS予約、実データ、固定科学契約は変更しない。
- `6948e23747` へのrebase後の実装commit `8f9dd3ba2b218565e54e4e0eaae4af6a43540cf7` で広域人工回帰 **770 passed, 1 skipped, 53 subtests passed (123.59s)**。Git履歴cacheの追加回帰も含む。`git diff --check` 成功、H4 controller/workerの既存PID 665284/669005を読み取り確認。実機PBS・新規科学計算なし。

## 保存データ6項目の隔離環境での最終対応

- 対象はN42、S01、S10、S11、S12、S14。今回の利用者指示は隔離環境での実装・技術検収に限定し、稼働版・現物の保存データを変更しない。
- S01/S14：旧版source復元と診断重複整理を既存停止run保護・固定入力照合に接続。新Gatewayでは通常Start/Resume前に保存layoutを復元する。旧Gatewayでは `--restore-saved-layout --run-id` による明示復元が必要（旧固定版を書換えない）。人工試験で、復元先の衝突/改変・保護対象をHOLD、正常経路でpreflight→launchの順序を確認。
- S10/S12：`scripts/rpe_saved_data_migration.py` の `copy-git` はsourceの全object（到達不能objectも含む）、refs/index/reflogsとファイルを保持した新しいcopyを検証する。`stage-object-store` はlinked worktreeも含む独立bare object storeを作る。`copy-tree` は外部参照を明示して試験成果物をbyte照合コピーする。いずれもsource削除・実alternate変更・timer/process変更は行わない。
- S11：`evidence-service` は独立object storeから別の固定実行/evidence viewを作成し、従来の `run_evidence_root` とGit locator規則を使う。元releaseのsparse scope・固定SOURCEを保持し、内部の固定service入力ファイルをbyte一致で複製。その他の明示provider/runtime依存はそのまま残す。登録・採択は別の操作。旧版に対するretention inventoryは引き続き参照あり/未確認を削除許可と見なさない。
- S14：`--pack-diagnostics` のあと `--retire-diagnostics` はsettled runのjob側と同一の転送診断だけを対象にする。scientific receipts・異なる転送証跡は保持。raw resultのcold copyは `compress-file` / `restore-file` のSHA照合で実行し、旧版のraw JSON byte表現を変えない。
- 追加の共有版 `f585a0f5dd396d846e90f5b7939a806a73965122` のH-chain exact leaf再利用、中間report削減、gzip proof保存を統合する。新規runのgzip proofと、旧版rawのcold copyは別用途。
- 隔離fixtureではS01/S10/S11/S12/S14の正常復元・改変拒否・source不変を確認。対象回帰26件成功。
- **技術対応51件（採用された代替策・既存再利用を含む）、PBS実機待ち6件（N13/N15/N31/N32/N33/N34）**。実データの回収量・本番運用の成立や性能はこの検証では主張しない。
