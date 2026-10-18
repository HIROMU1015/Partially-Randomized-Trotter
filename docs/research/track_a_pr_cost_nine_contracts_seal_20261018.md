# Track A：A_COST・既存B/D会計の9契約 binding・技術seal

2026-10-18。**準備のみ。実科学call・preparation復元・native compile・科学再会計・B数値診断は0。**
実行は未認可であり、技術sealを実行承認と扱わない。

資料入口はこの文書。機械可読の計画・無効grant・検証記録は
[9契約bundle](../../artifacts/resource_applicability/track_a_pr_cost_batch_seal/2026-10-18/)。
前段の[人工40検査・科学条件・来歴索引](track_a_pr_cost_integration_readiness_20261018.md)は保存し、
同じ検証campaignや科学レビューを繰り返していない。

## 固定版と今回の変更

- branch：`track-a-pr-cost-batch-seal-20261018`
- 新SOURCE：`31ef62a12407010e9c1a07f9b37facedb8dff5fe`
- 起点REVIEW：`91d204fcac1cdaf4cb4ee9e5f3e9c0e5e31329eb`
- 再利用する元SOURCE：`33c7231fc7c8f3940106bd8fe473e3db775fbcb0`
- private batch digest：`6f327252a234b1fbccd19a9fef054fafe3c165305f705b1b369875f2cd540f8d`
- 作業専用clone：`/home/AbeHiromu/projects/track-a-pr-cost-batch-seal/20261018/repo`
- 封印済みprivate領域：`/home/AbeHiromu/projects/track-a-pr-cost-batch-seal/20261018/private/sealed03`

既存worktreeを登録・checkout・編集せず、独立cloneを使った。SOURCEは4つの新規file
`scripts/pr_cost_batch.py`、`scripts/pr_cost_batch_science.py`、
`scripts/prepare_pr_cost_batch.py`、`tests/test_pr_cost_batch.py`の追加。
旧scientific sourceの変更はない。元native sampling/control/compile・paired会計を再利用する。
H6の未export p3/p4 hashは推測せず、保存p19と元prefix_view sourceに結合して実行時に記録する。
この報告・監査・索引はSOURCEを変更しない後続REVIEW_BUNDLE commitへ分離する。

## 9契約の状態と提案call数

以下の「実行候補」は、選択承認・fresh資源gate・quota方針・有効な個別grantが揃った場合の候補。
現在の全grantは `approved_by_user=false`、`runtime_authorization=false`、`allowed_cpus=[]`。
各contractのtechnical_digest・SOURCE・input/runtime・output・lockは個別に固定した。

| 個別scope | 技術状態 | 別途承認する計算範囲 | 新規wrapper compile |
|---|---|---|---:|
| H4_A_COST | READY | 固定16候補×探索n8、trajectory/evolution各128、occurrence584、event2448、probe64、event oracle16。元native preparation p2/3/4を各1、最大3復元 | 256 |
| H6_A_COST | READY | 固定16候補×探索n8、trajectory/evolution各128、occurrence736、event3328、probe64、event oracle16。保存p5＋p19から元p3/p4 view | 256 |
| H8_A_COST | READY | 固定16候補×探索n8、trajectory/evolution各128、occurrence736、event3584、probe64、event oracle16。元保存exportのp6/7/8復元 | 256 |
| H4_D_ACCOUNTING | READY | 保存480行、旧8決定論＋8直接実測追加、元PR8候補×確認n32。既存単価の保存shot会計を1回 | 0 |
| H6_D_ACCOUNTING | READY | 保存2880行、旧32/強化136決定論を別比較、元PR確認union8×n32。p11/q4 S4と既存p7実測優先、探索参考1440行は別保持 | 0 |
| H8_D_ACCOUNTING | READY | 保存3800行、旧32/強化184決定論を別比較、PR確認union6×n32。主要6軸反映済み会計へ追加D2構成の実測を優先 | 0 |
| H4_B_ANALYSIS | READY_PARTIAL_DIAGNOSTIC | 既存2候補・4条件付き群×n8の元law/費用診断1回。無条件平均は未確定 | 0 |
| H6_B_ANALYSIS | READY_PARTIAL_DIAGNOSTIC | 既存2候補・2条件付き群×n8の元law/費用診断1回。K4 s2_multi未取得、無条件平均は未確定 | 0 |
| H8_B_ANALYSIS | READY_PARTIAL_DIAGNOSTIC | 既存2候補・4条件付き群×n8の元law/費用診断1回。無条件平均は未確定 | 0 |

Aを全3系承認する場合の合計はtrajectory/evolution各384、wrapper compile768、
occurrence2056、event9360、probe192、event oracle48。これは提案値であり実績ではない。
各D/Bは保存scalar処理を1 pass。signal/reference・SCF/DF/state生成・fit・量子shotは全契約0。
native controlは新しい科学signal panelの取得と区別する。A先頭16軌道でcontrol/oracleを確認後、
同じA内の残り112を並列取得する。各AからD/Bやwinner確認samplingへ自動進行しない。

## 入力・preparation・SOURCE・seed・compilerの結合

全系linear 1Å/STO-3G/T=.8/P=0。H4はlegacy DF12の保存状態、H6はtol-only DF19、
H8はtol-only DF28。H4 tol-only DF10・新状態、高次PF研究、理論リソース比較は混合しない。

既存private configとハッシュ付き入力を専用content-addressed storeへbyte copyし、
元sourceを固定Git blobから独立snapshotへ抽出した。科学arrayのdecodeは行っていない。
全SOURCE blob、各入力bytes/SHA、provider source、Python binary、元runtime closureを検証し、
**9契約すべてbyte-only gate PASS**。旧configそのものも来歴として保存し、旧grantを再使用しない。

- H4：元A_SIGNAL SOURCE `79758d28d89ccaf2f942dc10ad724e70cf513089`。
  archive未結合のため、元builder SHA `7528ebae754e14ee623d28212a668e4edc237575775a78d0d2bf4d34dc53c293`
  による最大3 prefix復元をA契約へ明示した。今回は復元0。
- H6：元A_SIGNAL SOURCE `c507ec6486b2c16f383364598a80f046a1d544ad`。
  保存archive SHA `ad52f9f6a3150981d7be00191d8b3a9f9af992c37c9102cba6d94fcffc8e3010`。
  faithful:p5は保存hashを保持し、p3/p4は同じ保存p19から元prefix_viewで取り出す。
  元p3/p4 hashが未exportであることを、過去にhash確認済みだったという表現に置換しない。
- H8：元A_SIGNAL SOURCE `6197e97d7ff0f0104387ca244a1e03de701cd19d`。
  保存export SHA `771137828552b858361b6e2b25d2b675fb83f908aad6209f757e46464b817420`、
  元p6/7/8 preparation hashを保存receiptに結合した。
- seed：元通常n32・探索・条件付きpilotの保存master/派生seed台帳を結合し、
  新SOURCEを含むsemantic keyから新A各128 masterと元APIのstep seedをmetadataとして導出。
  旧/新・新/新の衝突0。**RTE event抽選・trajectory生成は0**。
  新Aは探索n8のまま、元確認n32へのpooling・独立確認済みへの昇格は禁止。

準備中のmetadata seal初回はPython `-P` のmodule探索で入口が停止した。
bootstrapを修正し、最終SOURCEでmetadata sealが正常完了した。科学attempt・消費・旧STOPは発生していない。
旧準備領域も削除していない。

共通runtimeは34,273 file / 1,503,620,794 bytesを重複排除して一度だけstreaming SHA照合し、
missing/mismatchとも0。その後はdevice/inode/size/mtime_nsが同じ葉の既存照合を再利用し、
変わった葉だけ再hashする。既存private Python/venvのinstall・upgrade・設定変更は0。
Qiskit1.3.0、rz/sx/x/cx、optimization1、seed17、topologyなし、symmetric_directionalを保持する。
compiler hashは `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。
H4/H6 profile実file SHAとH8の既存profile識別子は区別する。
旧compilerとの全output同一性は未確認のまま。[profile/runtime監査](../../artifacts/resource_applicability/track_a_pr_cost_batch_seal/2026-10-18/runtime_profile_audit.json)参照。

## D会計の独立性・既存実測の保護

Dは**今回のA_COST新結果を消費しない**。既存のD実測と元保存shot/PR確認n32だけを使う。
通常signed/経験的非相殺×5精度×allowance倍率1/10を保存済みの値のまま結合し、
将来承認後だけshot重み付きRZ・paired SE・旧集合/強化集合の比較を行う。
H4の元phase `fixed_anchor_panel`、H6探索/確認、H8探索/確認の由来を保存する。
内部の共通n32会計ラベルへ正規化しても、原cohortを新しい独立確認として扱わない。

H6では既存p7/q1 S2の実測単価を保持し、V1 modelへ後退させない。
旧V1探索1440行は確認union8のランキングへ混ぜない。
H8は主要6軸反映済みの既存会計を起点にする。直接実測と固定modelを明示し、
model残存条件で全候補の実測最良を主張しない。
元PR32軌道のpaired costから再現した平均が保存費用と一致しなければ当該DはSTOPする。
保存shot・適格性・bias・normalizationを再最適化しない。

## Bの実行可能範囲と証拠不足

元native parentのexit0・reap receipt、条件付きn8のtrajectory/event digest・paired費用を照合し、
各群の元 `original_distribution` / `log_stratum_mass` をそのまま結合した。
今回log質量を指数化したり、条件付き平均・層確率・新費用を計算したりしていない。
通常n32のseed/event/paired wrapper来歴はH4/H6各256軌道、H8 192軌道分を別々に結合した。

**不足は、各対象候補の通常軌道eventによる層分類の結合と、完全で重複しない元law質量receipt。**
今回のsnapshotでは未結合であり、原証拠が保存されなかったと断定していない。
H6 K4 s2_multi費用は未取得という別の不足である。K2の結果で代用しない。
zero/single/multi/s4の全support質量・費用が揃うまで、無条件平均/SEはnull。
再正規化、欠測0補完、条件付きn8を通常n32へ混合する処理はない。
3件のBはこの限定診断の承認候補であり、**完全なlaw-weighted無条件費用比較3件は実行不可**。
不足した診断だけを未確定として終了し、他のA/Dを止めない。

## 並列batch・資源計画

科学的依存DAGは9独立nodeでedgeなし。選択されたscopeだけを別session/process・個別grant・
個別output・個別one-shot lock・個別ledger/STOPで起動する。
入力不合格のscopeは起動前STOPとし、独立した正常scopeは起動できる。
一つのchildの非zero exitから他scopeへの一括停止・retryは行わない。
batch driver自体の中断時はown identity/sessionを確認し、TERM→必要時KILL→wait/reapを行う。
cleanupの一対象でerrorが出ても後続対象を回収し、errorを成功扱いしない。

候補はAのH4/H6/H8を8/8/12 worker＋B/D各1 process、計34 compute slot。
A coordinatorは自系CPU集合内で待機し、preparationをprefixごとに一度復元してfork/COW共有する。
cosine/sineでevent/evolutionを共用し、元fingerprint reuseを維持する。
worker内部thread1で入れ子並列を避ける。
CPU番号・worker数は固定認可しておらず、承認後に既知の並行jobのCPUを除いてfresh選定する。
余裕が変われば実行前に同じ契約内のworker数を調整できる。これらの数を恒常上限にしない。

memory admissionの初期見積りはH4/H6/H8のworker/parent各2/6/24GiB、
上の並列案はparent込み385.5GiB＋追加host余裕案32GiB。
これは既存同系native実行を踏まえた起動判断用の保守的見積りであり、RSS/AS hard capや実使用量の予測保証ではない。
固定wall capは設けない。実利用・OOM・ディスク不足は観測し、継続不能な当該scopeをSTOPする。
PSI・CPU競合・無関係worktree更新だけでは科学workerを強制終了しない。

output/log/journal/scratchはA各2/2/8GiB、B/D各128MiBの見積り、全9で12.75GiB、inode25,344。
保存するのはevent trace、予約台帳、metrics、native receipt等であり、QPYやwrapper全体を一括保存しない。
元private input byte storeは重複排除し、runtime照合結果も共用する。
実際のmemory・disk・inodeが選択scope合計の見積り以上あることをbatchと個別起動で再確認する。
[観測時刻・CPU/memory/PSI・filesystem](../../artifacts/resource_applicability/track_a_pr_cost_batch_seal/2026-10-18/resource_observation_and_batch_proposal.json)は起動許可ではない。

`/home`はext4/rw,relatime、観測時の空きは約477GiB（df -h表示）、free inodeは約2.19億。
quotaコマンドがなく、quotaの有無・残量は**未確認**。空きfilesystem容量を個人quotaと同一視しない。
将来のgrantでは、quota残量確認、quotaなし確認、または**利用者が未公開quotaを了承し
ENOSPC等で当該scopeをfail-closedする方針**のいずれかを明示する。
現在その了承は発行していない。共有設定やquota設定を変更して解決しない。

## 統合承認案（未承認）

利用者は1回の明示指示で、例えばA_COST3件だけ、D3件だけ、または限定Bを含む任意のsubsetを列挙できる。
承認対象は上表のscopeとcall範囲、新SOURCE、input/compiler/preparation、one-shot/no-retry、
fresh CPU/memory/disk/inode・quota方針である。
無指定scopeを暗黙に実行せず、既存準備依頼を実行認可へ転用しない。

承認後のCodexの作業は、同じbatch_digestへ選択一覧と利用者指示receiptを結合し、
個別有効grantを作成してfresh gateを通過したscopeだけ一括起動すること。
いずれもretry・失敗返却・自動再実行なし。完了またはSTOP後はmandatory STOP。
次段階・新sampling・A結果を使う追加再会計は含めない。
現在の[統合案](../../artifacts/resource_applicability/track_a_pr_cost_batch_seal/2026-10-18/integrated_approval_INVALID.json)と
個別9 grantは全て無効。絶対launch commandは承認までnull。

## 検証・保全・公開

新しい承認ゲート・subset選択・独立scope失敗・quota方針・CPU除外・own cleanup等の人工21検査PASS。
追加で9個別CLI＋batch CLIの無効grant拒否10件PASS。新synthetic compileは0。
実動作cleanup fixtureは最小child1つだけでTERM・reap・二重cleanupを確認した。
旧人工40検査は固定receiptを再利用し、旧benchmarkや科学campaignは再実行していない。
これは作者の技術確認であり、別担当の独立レビューではない。AI subagentは0。

元準備repoはHEAD/statusとも開始時と一致。高次PF/Codex 2・理論資源比較・他worktree・
旧SOURCE/raw/grant/ledger/marker/STOPを変更しない。共有環境、既存venv、他jobへの変更0。
新規公開対象は新source/testsと軽量な計画・無効grant・hash監査・資料索引のみ。
private config/raw/NPZ/QPY・runtime34,273ファイルそのもの・有効grant・認証情報はstageしない。
公開の実成否・REVIEW_BUNDLE完全SHA・remote SHAはcommit後の最終報告に記録する。

次の担当：**ユーザー**が実行するscopeとquota方針を一括指定して明示承認する。
**Codex**はその指示後に選択scopeだけfresh gate・個別grant・一回実行を行う。
今回の準備成果公開またはpush待ち報告後はmandatory STOP。
