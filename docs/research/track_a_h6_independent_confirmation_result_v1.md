# Track A H6：独立32 trajectory確認結果

探索592 group後に固定した8構成の独立確認は、256/256 group・512 wrapperを保存して正常終了した。確認32 trajectoryでも、選択集合内の最良PRの測定込みRZ費用の点推定は、full PFより各精度で約54–63%低い。ただし希少Taylor次数の未観測と経験的数値allowanceが残り、母平均の正式なwinnerや最終エネルギー推定の総費用は認定しない。

[機械可読結果・CSV・536行算術照合・来歴の入口](../../artifacts/resource_applicability/track_a_h6_confirmation_summary_v1/2026-10-12/README.md)。本報告は保存済みJSONの集計のみで、新しい科学計算・transpile・trajectory・GPU・grantを実行していない。

## 比較条件と標本

linear H6、原子間隔1.00 Å、STO-3G、12 modes、alpha3/beta3のsector400、保存state、T=0.8。DFはtol-only 1e-8・rank19・cutoff0、signed generation orderとweighted Hermitian projectionを保持する。探索のprefixは5/10/15/full19、q=1/2/4/8、delta=0.8/0.4/0.2/0.1。今回の選択8構成はすべてprefix5で、q1/2/4はr4、q8はr2、各K2/K4である。

epsilonは有限時間のcomplex signal精度で、エネルギー精度ではない。各精度の探索B2上位2候補の和集合を事前固定し、各構成32本の新seedを取得した。探索8 trajectoryは確認平均に混ぜない。cosine/sineは同じtrajectoryの対なので標本数は32であり、64ではない。残り64のB2構成は独立確認未実施で、除外や母平均劣位を認定していない。

Qiskit1.3.0、basis rz/sx/x/cx、optimization1、seed17、backend/coupling/layout/routing=None。同じsymmetric_directional full Hadamard wrapperを数える。state preparationは除外したP=0。測定数は各軸alpha=0.025のHoeffding会計による解析的予算で、実shotの実行やエネルギー/RPE実験はしていない。数値allowance uは経験的なままである。

## 精度別の主要結果

`G_RZ = N_cos * mean(RZ_cos) + N_sin * mean(RZ_sin)`。下表のPR最良は確認済み8構成内、PF最良は保存済みfull-rank S2/S4の適格集合内である。B0はprefix-discard S2。決定論的B0/B1の保存費用を再利用し、再計算していない。

| epsilon | PR構成（prefix5） | PR G_RZ | 最良full PF | full PF G_RZ | PR/PF | RZ費用低減 |
|---|---|---:|---|---:|---:|---:|
| 0.05 | q1 r4 K4 | 4.03423200e8 | full19 S2 q1 | 8.77049350e8 | 0.459978 | 54.00% |
| 0.01 | q2 r4 K2 | 1.9549158739e10 | full19 S2 q2 | 4.8918110661e10 | 0.399630 | 60.04% |
| 0.005 | q4 r4 K2 | 1.0183633918265625e11 | full19 S2 q4 | 2.736762519e11 | 0.372105 | 62.79% |
| 0.001 | q8 r2 K2 | 5.146731715972781e12 | full19 S4 q2 | 1.3298858970625e13 | 0.387006 | 61.30% |

PRのpaired G_RZ標準誤差は順に3.883686e6、1.670236e8、4.632327e8、9.532269e9。これは有限標本の記述統計であり、tailを含む母平均の信頼保証ではない。PR総shot予算は30,400 /752,048 /1,987,665 /53,687,969、full PFは23,350 /658,413 /1,859,100 /60,083,125。

最良discard B0はprefix15・S2のq1/2/4/8で、PR/B0比は0.575357 /0.501027 /0.465310 /0.436845、約42–56%低い。探索8標本時のPR/full PF比0.454501 /0.395175 /0.366888 /0.387811と方向は一致するが、確認は別標本として保存する。

探索では4精度ともK4が点最良だった。確認ではepsilon0.01/0.005/0.001でK2へ点順位が変わり、q/r/prefixは維持された。4精度すべてでK2とK4の差は `2*sqrt(SE_K2^2+SE_K4^2)` より小さい。これは記述的な比較で、K2優位の検定・同等性の証明・同時信頼区間ではない。

## 他の費用指標と感度

CX、depth、sizeのnative mapも保存した。各指標で適格候補を別々に点順位づけし、epsilon0.01のdepth最良PRはK4、同精度のRZ/CX/size最良はK2だった。RZ順位をすべての資源の順位へ一般化しない。RZ_depth/CX_depthは512回路CSVに生の整数値として残すが、native shot-weighted mapの指標はRZ/CX/depth/sizeの4つである。

経験的uを10倍にした既存感度会計でも4精度の最良IDと適格候補数は変わらず、PR/full PF比は0.459978 /0.399630 /0.372105 /0.387006で、表示桁では同じだった。細かいshot切上げの差はJSONに保持する。u10倍は厳密な数値certificateを与えない。

共通state preparationのRZ費用をP/shotとすれば `G(P)=G(0)+N_total*P`。表の固定したPR/PFペアの交点Pはepsilon0.05/0.01/0.005で約67,181 /313,654 /1,336,599 RZ/shot。epsilon0.001はPRのshot予算も小さく、非負Pで交点がない。候補をPごとに再最適化した全体比較ではなく、state preparation費用自体は未測定である。

## 希少Taylor次数と結論の限界

確認32標本のoccurrence数はR*32で、trajectory数やshot数と区別する。

| q/r/K | order0 | order2 | order4 |
|---|---:|---:|---:|
| 1/4/2 | 128 | 0 | 対象外 |
| 1/4/4 | 124 | 4 | 0 |
| 2/4/2 | 254 | 2 | 対象外 |
| 2/4/4 | 253 | 3 | 0 |
| 4/4/2 | 511 | 1 | 対象外 |
| 4/4/4 | 511 | 1 | 0 |
| 8/2/2 | 512 | 0 | 対象外 |
| 8/2/4 | 512 | 0 | 0 |

3/8構成はnonzero次数を一度も観測しておらず、4つすべてのK4構成でorder4は未観測。保存された有限分布におけるorder4確率はq1/2/4/8で約4.59e-5 /2.94e-6 /1.85e-7 /1.85e-7（occurrenceあたり）。未観測を確率0や費用0とは扱わない。

q1 K2では1本のtrajectoryが未観測次数を含む確率は約6.38%、q8 K2/K4では約1.67%。この有限分布確率は保存lawからの診断で、未観測eventの回路費用を制約するものではない。native `cost_mean_tail_bound=null` と `population_mean_certified=false` を保持する。固定32本は完了したが、population winner、全72 B2の独立順位、厳密numerical certificate、最終state-preparation込みRPE/ground-state総費用は未認定である。

## 実行・高速化・会計

元4worker runから利用者指示で159完了groupを引継いだ。元runはUSER_STOP_REQUESTで終了し、全163 dispatchedを回収、partial4は結果標本に混ぜず予約を保持した。新run `h6-confirmation-fast-handoff-v1` は8worker・内部thread1で残97を取得した。

新runのnative coordinator PID39637、exit0、観測終了2026-10-12 23:27:25 JST。`H6_CONFIRMATION_FAST_COMPLETE_MANDATORY_STOP`、reason/first_stop_reason=null、全97 exit0/reaped、cleanup_errors=[]、pending0、log/journal closed。次stage・自動retry・H8を認可しない。

新97 groupのsupervisor wallは972.793405秒（16分12.793秒）、aggregate peak RSSは5,340,696,576 bytes（4.973911 GiB）。元runのsupervisor wallは2,908.576816秒、両supervisor合計は3,881.370221秒（64分41.370秒）。この合計は引継ぎ準備時間を除き、元runのpartial作業は含む。全研究の歴史的wall時間ではない。

新97 workerの保存hash統計では582 calls中291 final-digest hits、operation bytesは9,254,610 hits /649,095 misses、hit率93.4459%、eviction0。実際に再利用が働いたことを示す。引継ぎ前後のqや完了範囲が違い、hash検査・回路build・transpileなどを含むため、全体の高速化倍率は算出しない。古典wall/RSSの改善は量子資源の改善とは別で、hash最適化は回路条件を変更しない。

known callsはcompile1700、trajectory834、control_probe360、occurrence3136、primitive1470、reference_matvec400。compile予約20・trajectory予約10を加えた1720/844は明示承認上限内。歴史的unknown actualはnull、予約をactualと呼ばず、失敗分返却・budget resetなし。

## 固定source・監査・公開状況

実行SOURCEは `87b6b9f46b7e8ba9abe132ee21478c114b6f46f7`、科学SOURCEは `5a903aea2492c2d211c1e15e7737df0d50694c12`。元159のSOURCE `73018cbcc39803890974c7ee51401063e6c8daf6` と新97の来歴を分ける。旧science224/旧runtime245は不変、hash/fast追加を含むclosure251・profile・compiler・元task/seedの照合を保存した。

既存private venv・Python3.11.0rc1/Qiskit1.3.0等の固定profileを変更しない。旧host/compilerとのoutput bit完全同一性は未検証であり、以前の丸め差受理条件を保持する。profile SHAは `4e32bb9ba57c38280562390100ebf916e9858218534504ead1519625a609bd75`。新hash実装の101限定人工testsは以前の合格記録を参照し、今回は繰り返していない。

今回の局所集計は256完了groupを元task/seed/event/compiler/両軸/sourceで再受理し、512行digestと完了receiptを照合した。保存整数回路値からFraction平均とpaired SEで536費用行を照合し、別CSVの算術・保存signalのshot会計も確認する。科学arrayのdecode、入力再生成、実compileは0。これはlocal metadata監査で、独立科学reviewやCIのPASSではない。

軽量報告・JSON・CSVだけをlocal commitへ固定し、実runtime/checkpoint/cache/NPZは追加しない。利用者がpushする方針を保持し、agent push/認証設定変更は行わない。remote公開済みは未主張。現在のH4計算・共有環境・他jobは変更していない。
