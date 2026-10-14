# Track A H8 D0：保存値one-shot再解析結果（2026-10-15）

## 完了と資料入口

**H8 D0を一度実行し、旧precision944件・旧費用map3,576件の完全一致後、5精度×factor1/10×6指標・paired SE・候補再選択・P感度を完了した。native exit0、mandatory STOP。** H4/H6 D0は再実行していない。

確認32 trajectoryのε=.0005/P=0/RZでは、PR点推定は取得済みB0/B1決定論集合の最良費用の **0.761667659倍（23.8332%低減）**。探索8 trajectoryは **0.761833256倍**。これは固定DFモデル・候補domain・compiler・経験的allowance・有限標本に条件付けた費用benchmarkである。native B0+Iはまだ未測定であり、最良決定論的方法全体や母平均winnerの保証ではない。

- Branch：`track-a-h8-d0-saved-value-one-shot-20261015`
- SOURCE：**`20f89bc32b161ee22e4a1ed709f9e3414c19633a`**（修正後独立監査SOURCE、その後source変更0）
- GRANT commit：`b613db106abd2e1cadb7bd5a29a765f4f257fcc5`
- 旧科学結果：`681ec23c93fdd1090d325908f4b155a79115afdf`
- Frozen D0/shot SOURCE：`fa75c7997e135da7b7a0c4e23949e47252bd7a2e`（不変）
- run ID：`h8-d0-run01-20261015`、driver PID1247365、launcher PID1247364（ともに退出）
- [公開結果・監査bundle](../../artifacts/resource_applicability/track_a_h8_d0_saved_value_execution_v1/2026-10-15/README.md)。[科学レビュー原本](track_a_h8_d0_integrated_scientific_review_20261015.md)、[独立監査](track_a_independent_technical_audit_20261015.md)、[実施研究ノート](研究ノート/2026-10-15_track_a_h8_d0_saved_value_one_shot.md)。

利用者の「つぎのさぎょうに進んで大丈夫」を、直前に確認したH8 D0 one-shotの実行承認として結合した。科学レビュー原本の未承認状態はレビュー時点の履歴として保持し、新grantに最新承認を記録した。H6や後続stageの承認には拡張しない。

## 固定条件とPhase A

対象はlinear H8/1.00 Å/STO-3G、tol-only DF28、保存Hamiltonian/state、T=.8、native symmetric-directional、固定compiler。B0 prefix7/14/21、full PF prefix28/S2・S4、B2 prefix7/14/21・r1/2/4・K2/4、q1/2/4/8の既存92候補を保持。δ=T/q、split `L_D` は各prefixであり、新しいdelta windowやtrajectoryを追加しない。確認は事前固定union6のB2＋det20。既存8/32標本を分け、同一seed/event/SOURCE/compilerのpaired cosine/sine費用を使う。

6入力は合計6,518,976 bytes。起動前と出力後にstreaming bytes/SHA一致、SOURCE Git object/disk bytes、既存Python3.12.3 profile/runtime hashesを確認した。private入力4JSONと固定旧結果のCSV/metadata2fileを参照し、NPZ科学arrayを読まない。

| Scope | 旧precision完全一致 | 旧6metric費用完全一致 | 結果 |
| --- | ---: | ---: | --- |
| 探索92 | 736 | 2,856 | EXACT PASS |
| 確認union6＋det20 | 208 | 720 | EXACT PASS |
| 合計 | **944** | **3,576** | 両scope PASS後にPhase B |

旧axis統計・method/phase・normalization/bias・top2 RZ unionも固定driverで検査した。許容差変更・不一致fallback・source差し替え・retryは0。独立監査F01の異SOURCE paired axis拒否に実入力は抵触しなかった。

## Phase B：精度・metric・標本

再解析domainはε=.05/.01/.005/.001/.0005、allowance factor1/10、RZ/CX/RZ_depth/CX_depth/depth/size。探索920、確認260、計1,180 precision recordsを保存した。適格656／不適格524、適格6metric費用3,936行。MISSING費用0であり、不適格を費用0へ置かない。family選択は120scope/ε/factor/metric cases、3familyのsummary360行。

以下は**確認32、factor1、P=0、RZ**の点推定。deterministicは同じ保存候補集合のB0/B1を比較する。SEはpaired標本SEで、信頼区間や選択後winner保証ではない。

| ε | PR点選択（共通prefix7/r4/S2） | 取得済det最良 | PR/full PF | PR/取得済det最良 |
| ---: | --- | --- | ---: | ---: |
| .05 | q2/K2 | B0 | .457076 | .605218 |
| .01 | q4/K4 | B0 | .356301 | .472820 |
| .005 | q4/K4 | B0 | .356164 | .472664 |
| .001 | q8/K4 | B0 | .342413 | .438795 |
| .0005 | q8/K4 | full PF S4/q4 | **.761668** | **.761668** |

新ε=.0005ではB0 prefix21/q8の費用324,137,634,018,224に対し、full PF S4/q4は186,504,990,295,360となり、取得済det最良がB0からfull PFへ切り替わる。確認PRは142,054,819,336,237.5、paired SE420,613,978,705.5152。探索PRはprefix7/q8/r4/K2、142,085,703,973,281.25、SE410,101,833,135.4219。

K2/K4の細かな点順位は母平均順位と認定しない。新εでPR/full PFの比が.001より大きくなるため、「高精度ほどPRが有利」は支持しない。全domainはq≤8であり、q16以上やε=.0001を追加していない。

P=0の探索metric-selected B2は全60casesで旧union6内にあった。ただし、次のP envelopeでのcoverageは異なる。確認後の点選択が含む選択効果、経験的u、未観測event費用の限界も保持する。[全20RZ precision summaries](../../artifacts/resource_applicability/track_a_h8_d0_saved_value_execution_v1/2026-10-15/precision_summary_v1.csv)と[360metric summaries](../../artifacts/resource_applicability/track_a_h8_d0_saved_value_execution_v1/2026-10-15/metric_selection_summary_v1.csv)にはfactor10と探索を分離保存した。

## Phase C：共通1-shot準備費用P

`G_j(P)=G_j(0)+N_total,j P`、P≥0を既存Fraction規則で計算し、120曲線を保存した。P=0のRZ代表を固定するpair交点と、全適格候補の再選択lower envelopeを区別する。Pは仮想共通overheadであり、state preparationの実compile値ではない。各metricの単位を混ぜず、depth-shotを実機時間へ変換しない。

ε=.0005/factor1/RZのlower envelopeは次の順に切り替わる。

| Scope | 初期PR | full S4/q4へ | full S4/q8へ |
| --- | --- | ---: | ---: |
| 探索8 | prefix7/q8/r4/K2 | P≈108,293.821 | P≈15,315,669.258 |
| 確認32 | prefix7/q8/r4/K4 | P≈108,369.817 | P≈15,315,669.258 |

この新εでは両envelopeのB2は旧union内である。一方、**探索60曲線のうち36曲線には旧union6外のB2が現れ、計140のsegment–cell出現（12固有候補）**を記録した。これは追加trajectory数ではない。例としてε=.05/factor1/RZではP≈83,871.702でprefix14/q2/r1/K2へ移り、さらにunion外候補へ切り替わる。これらは探索8本のP感度であり、確認32本の結果へ混合しない。[P coverage監査](../../artifacts/resource_applicability/track_a_h8_d0_saved_value_execution_v1/2026-10-15/P_confirmation_coverage_audit_v1.json)に区間・候補を保存した。

確認外候補を主要claimにする判断や追加確認は行わない。保存された全曲線・exact numerator/denominatorが正本であり、表のPは表示丸めである。

## 希少次数・未保証事項

旧saved aggregate Taylor histogramは探索0/2/4次＝5,024/15/1、確認＝3,580/4/0。確認4次の0観測は費用0ではなくUNOBSERVED、population layer costはnullのまま。確率law・sampling・条件付きpilot・母平均上界を追加していない。aggregate観測が各候補の未観測層を埋めるとは解釈しない。

結果は保存DFモデルのfinite-time complex signalに対する条件付き会計である。厳密u certificate、未知分子のtruth-free選抜、energy/QPE/RPE、routing/noise/FT費用、state-preparation実測、native B0+I familyを含む全面対照・一般size scalingには拡張しない。

## 資源・終了・保全

CPU1のown-run affinity、1process/thread1、Python -P -B、既存venv不変。承認STOP ceilingはAS512MiB/RSS256MiB、CPU240秒/wall300秒、output32MiB＋terminal reserve1MiB、monitor5秒。/homeのuser/group/project quotaはread-only Q_GETINFOで全ESRCH（無効）を確認し、設定変更0。

- launcher wall **2.9709500800818205秒**、監視上のpeak RSS **54,878,208 bytes（52.3359 MiB）**、CPU **2.792849秒**。
- 監視12回、最大start-to-start interval **.5031271446496248秒**、観測所要時間最大 **.006435232236981392秒**。host PSI最大0、memory/own-cgroup PSI/OOM/filesystem gatesは全PASS。
- native driver exit0、同じforeground shellのwait receipt、driver/launcher identity absent、scientific children0、artifact FD/timer cleanup完了。正確なexit timestampは未取得のnullを保持。
- output7result＋terminal＋native exitをbytes/SHA照合。durable one-shot markerはprivateに残し、使用済みoutputを再利用しない。
- SOURCE/profile/input・元科学結果・旧grant/ledger/runtime/checkpoint・shared venv/他job変更0。carryは既存unknownを含む履歴を保持、reset/refund0。
- **H8保存値再解析1回、新SCF/DF/state/signal/trajectory/native build/compile0、H4/H6 D0再実行0。** 75人工testsやbenchmarkを再実行しない。

長C/IO中の監視の一般保証は今回の完了実績から推定しない。[完了監査](../../artifacts/resource_applicability/track_a_h8_d0_saved_value_execution_v1/2026-10-15/completion_audit_v1.json)とraw terminal/native metadataを正本とする。

## 公開とmandatory STOP

本branchには固定grant・科学レビュー原本、軽量な派生accounting/selection/P/coverage/rare outputs、byte/SHA監査、研究ノートだけを含む。private input mapping・one-shot marker・実NPZ/runtime/checkpoint/cache/credentialsはcommitしない。[公開receipt](../../artifacts/resource_applicability/track_a_h8_d0_saved_value_execution_v1/2026-10-15/publication_receipt_v1.json)でremote確認・認証失敗を区別する。

one-shotは完了して停止した。自動retry、H6 pilot、追加確認sampling、次stage起動を行わない。H6 native B0+Iは採択済みの次候補だが、実行承認は別途必要。今回結果を研究ノートへ固定し **mandatory STOP**。
