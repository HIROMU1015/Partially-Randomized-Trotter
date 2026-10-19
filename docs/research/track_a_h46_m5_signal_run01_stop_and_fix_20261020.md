# H4/H6 m5：signal run01のSTOPとfull-tail端点の修正

2026-10-20。**承認された一回実行を起動したが、H4/H6ともprefix contextの初期化中に`INVALID_GENERATION_PREFIX`でSTOPした。候補signal評価・予約は0件。原因を修正し、人工18件とmetadata37項目を確認した。修正版は別worktreeの新SOURCEへ保存し、再実行は未承認・未起動のままmandatory STOPする。**

元の[実行準備報告](track_a_h46_m5_signal_readiness_20261020.md)に対するユーザーの「その方針で進めてよい」を、固定88候補（最大82新平均signal＋6raw再利用）、保存reference2件、当該runでの経験的allowance案の使用に対する承認として記録した。追加PFの主比較採用AD03、正式shot政策、sampling・compile・費用会計は承認に含めていない。

## 実行範囲と終了証拠

元SOURCE `480ca4507394e35185ec535c9f488bd90538de25`、準備commit `d09497961a9da5092bec0a075a32cb84295abfc3`。新承認receiptと一回grantを作り、元draft・旧grant・旧STOPを保持した。承認plan SHAは`f0002d08d8cc1efd40bb9e281b235c20f8f3ada640b9ac73ad08a5d6868480af`。

対象は保存線形1 Å/STO-3G、tol-only1e-8・rank自動決定のH4観測10/H6観測19、T=.8/P0、通常signed bias・倍率1、q1/2/4/8、PR anchor p3/p5・outer32/K4である。m5決定論的対照のprefixと3公式PRを含む88候補集合は変更していない。経験的allowanceは保存reference精度に条件付け、reference絶対誤差上界はnullのままである。

| 検査・処理 | 記録 |
| --- | --- |
| 起動直前seal | SOURCE/input/runtime/承認receipt、2,047 filesを確認 |
| fresh資源 | 空きCPU約61.30 core相当、実効空きRAM 502,893,977,600 bytes、disk 478,966,071,296 bytes |
| dispatch | H4/H6の2 process、候補thread予算は各30 |
| 実際の到達点 | 両系ともbase context復元後、prefix context初期化でSTOP。候補thread poolへは未到達 |
| 候補ledger / 候補結果 | 0 / 0。保存6rawの候補出力も0 |
| science worker | 両方がSTOPを保存しexit0、launcherが双方を回収 |
| campaign | `STOP`。workerのexit0を科学PASSと扱っていない |
| launcher | native exit1、外部wait4で回収確認 |
| 経過時間 | launcher内26.0849305秒、外部supervisor29.4120602秒 |
| 新compile / sampling / reference生成 | 0 / 0 / 0 |
| 費用再会計 / ranking / 自動retry / 次Phase | 全て未実施 |

ホストPSI・CPU競合や他worktreeの更新による停止ではない。今回の失敗は入力化学データやPF係数の不一致でもなく、新adapterのfull-prefix端点処理によるものだった。並列signalの実効速度は今回まだ測れていない。

## 原因と修正

`h46_new2_core.matrix_only_tail_coefficients`は打切りprefix向けの既存関数で、`0 <= prefix < n_blocks`を要求する。新adapterは`prefix == n_blocks`のfull PFにもこの関数を呼んでいたため、両系のfull contextを作る段階で拒否された。

以前の人工テストではtail係数関数をmockしてfullに(0,0)を返していた。このため実関数の拒否条件を検出できなかった。今回、実関数を置き換えずfull contextを作る回帰と、full時に打切り専用関数を呼ばない回帰を追加した。修正前は1 failure＋1 errorで同じ原因を再現し、修正後はPASSした。

修正は新adapterでfullの場合だけ**空tailのidentity係数とλを正確に0とする**もの。full preparationは既存のものを再利用し、打切りprefixでは従来どおり元関数とpreparationのidentity/λ整合検査を使う。既存のsignal関数・係数・signed順序・scalar政策・allowance・候補domain・数値閾値は変えていない。fullのtailが空であることを、未知費用の0補完と混同しない。

元SOURCEとrun01を変更しないため、修正は独立worktreeへ移した。

- worktree：`partially-randomized-trotter-worktrees/h46-m5-signal-empty-tail-fix-20261020`
- branch：`track-a-h46-m5-signal-empty-tail-fix-20261020`
- full-tail修正commit：`43b8ad30df3fe09981191c3cadbf27dcdcc9a046`
- 最終修正SOURCE：`bec018c64e7a8458dbf13b1c0e9655d628292857`

別worktreeで保存pilotの来歴を検証する際、元contractが元worktreeの絶対SOURCE pathを要求することも確認した。これを新pathで代用したり検査を緩めたりせず、元sealから一意に特定した元contract・Git bytesを確認し、元rootのまま独立したstdlib-only processで元検査を実行するmetadata adapterを追加した。元seal・入力・seed検査は省略しない。この準備中のmetadata拒否では有効grantも科学workerも追加していない。

## 検証・保全・新しい未承認契約

人工18件が固定Python3.12.3でPASS。追加4件はfull端点2件と元root検証2件であり、元14件（3公式×4q×4手法のtoy積照合、負時間、paired平均、cache共有、one-shot・片系STOP分離など）も再実行した。既存の科学SOURCE各fileは元worktreeとbyte一致を確認した。著者自身の検査であり、独立レビューではない。

metadata37項目と公開資料の構造24項目では、元2,047 seal・grant消費・STOP・candidate0件・exit/reapと、修正版2,064 seal・同じ入力/runtime/6raw再利用・同じ科学scopeを確認した。元2,899保護fileと停止済みrunの18 fileは全てSHA不変だった。修正後の実H4/H6 context・signalは未実行である。

修正版の新plan SHA：`19e0022bb72c220ab53db382117f4688aface219c1e1357160b4be3eb20c575f`。これは未承認設定であり、grantは無効。`--execute`/`--supervise`の双方が、入力復元前にexit2で拒否し、新run root・出力・markerを作らないことも確認した。元run01のgrantは消費済みのままで再使用しない。

[証拠artifact入口](../../artifacts/resource_applicability/track_a_h46_m5_signal_run01_stop_and_fix/2026-10-20/README.md)にfresh gate、両STOP、completion、外部exit、修正前失敗・修正後18件PASS、metadata、旧保全、新無効grantとSOURCE closureを保存した。公開用ファイルは非実行用projectionである。公開pushは行っていない。

## 次の担当

- **ユーザー**：元run01のSTOPを維持したまま、修正版SOURCE `bec018c6…`で同じ88条件を新しい一回契約として実行するか判断する。現在の固定契約は`retry=false`のため、今回の承認を自動再起動へ流用しない。
- **Codex**：新しい実行承認があれば、修正版の新承認receipt・grant・fresh gateを結合して実行し、完了またはSTOP後に終了監査する。候補signal・compile・会計の結果を先取りしてPASSにしない。

今回の作業は修正・人工検証・再実行準備まで完了し、local commit後mandatory STOPする。
