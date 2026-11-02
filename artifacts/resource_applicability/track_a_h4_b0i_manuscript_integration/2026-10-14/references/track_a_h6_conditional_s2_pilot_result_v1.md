# H6 s2限定pilot：完了・平均費用への寄与

2026-10-13。利用者の実行承認により、固定SOURCE **843265b624eac8fbb9acea47ef21cab1207e1880** でB2_p5_q8_2nd_r2_K2のs2条件付き8 trajectory・両軸16 full-wrapper compileを一度実行した。**8/8・16/16完了、native coordinator exit0、全worker reap/cleanup0、残存0、mandatory STOP**。[結果・CSV・byte監査・別算術の入口](../../artifacts/resource_applicability/track_a_h6_conditional_s2_pilot_result_v1/2026-10-13/README.md)。追加science stage・自動retryは起動していない。

## 固定条件・来歴

linear H6/1.00 Å/STO-3G、12 modes、alpha3/beta3 sector400、保存state、T=0.8。tol-only DF1e-8/rank19/cutoff0/signed generation order/weighted Hermitian projection。既存grid prefix5/10/15/full19、delta0.1–0.8を保全し、対象はprefix5/q8/delta0.1/r2/K2・whole trajectory R16だけ。

s2は16 occurrenceのうちorder2が1個以上。元finite lawに対してBinomial(16,p2)|M>=1・一様subset・原component確率を用いた。symmetric_directional/raw concatenation、測定込みfull Hadamard wrapper、P=0、signal/numerical allowance/B/normalization/shot会計を維持。探索8・独立確認32・新条件付き8はpoolしない。8本のmaster/order/step seed、phase、namespace、grantは既存と分離する。

科学SOURCE5a903aea2492c2d211c1e15e7737df0d50694c12の224件・旧runtime251件は不変、新closure257/input33 byte SHAを事後も照合した。既存roundoff受理archiveを再利用し、新SCF/DF/state/input/signalを生成しない。Python3.11.0rc1/Qiskit1.3.0/NumPy1.26.4/SciPy1.14.1、basis rz/sx/x/cx/opt1/seed17/backend/coupling/layout/routing=Noneを維持。旧compilerとのoutput bit完全一致は未検証。

実行grant commit b3aa49a57d422cc3e28ab0b2b21f8d36d9c34eca、grant byte SHA f91dc3d1921d0c74cae74599699221527d1c31be77ea2b343974a4fca3df625a。coordinator PID103016、run ID h6-conditional-s2-pilot-v1-20261013。専用出力・one-shot lockを取得したscience runは一回だけ。

## 条件付き費用と点推定の変化

全8本は各1個のorder2を含み、全16 wrapperのcosine/sine RZはpairedで同数だった。s2のRZ/shot平均 **96,285.25**、標本SD916.4116、SE324.0004、範囲94,866–98,079。既存確認32本は全てs0で、RZ/shot平均95,863.78125。条件付きs2平均の増加は約0.4397%に留まった。

保存確率pi2=0.01669690613968732、pi0=1-pi2を使い、`mean = pi0*mean(s0 confirmation32) + pi2*mean(s2 pilot8)` とする。両軸の保存整数shot N_cos/N_sinで各trajectoryのDを先に合算し、paired標本SD/SEを計算する。独立標本を仮定した記述的SEは `sqrt(pi0^2*SE0^2+pi2^2*SE2^2)`。選択後推定と有限標本の留保を維持し、正式confidence intervalやpopulation upper boundとは扱わない。

ε=0.001では保存shotはcosine24,402,829・sine29,285,140、full PF B1_p19_q2_4thのG=13,298,858,970,625。新層別G点推定5,147,109,530,245.307（記述的SE9,377,608,596.667）。旧確認Gから**+0.00734086%**。PR/PF比 **0.3870055113→0.3870339209**、RZ削減は **61.2966079%**。このpilotの観測範囲では大きな費用差の点推定はほぼ変わらない。

以下は同じ対象q8/K2を4精度の保存shotで評価した値であり、4精度それぞれの最良候補を置き換えた表ではない。

| epsilon | 対象q8/K2の旧PR/PF比 | 層別点推定PR/PF比 |
|---|---:|---:|
| 0.05 | 1.6503600773 | 1.6504812279 |
| 0.01 | 0.7592461713 | 0.7593019064 |
| 0.005 | 0.5612104721 | 0.5612516698 |
| 0.001 | 0.3870055113 | 0.3870339209 |

旧ε=.05/.01/.005の点winner（q1/K4・q2/K2・q4/K2）と事前選択K4 winnerは変更しない。q8/K2は確認後のε=.001の点最小候補であり、今回の解析を事前winnerの独立再確認へ読み替えない。従来の約54–63%の点削減との関係は、この対象構成の補足費用評価として整理する。

## 実資源・停止・累積会計

worker1 CPU20/thread1/AS32GiB/RSS STOP9GiB、coordinator22/23/ASsoft2-hard32/RSS reserve2、aggregate40/jobRAM64。既定1秒観測・5秒所要/欠測guard、pressure/OOM/headroom/output/call/instruction guardを維持。実runのSTOP理由null、全8worker exit0/reap、cleanup errors0、journal/log閉鎖、coordinator native wait/reap。読み取りによるpost-exit残存0も別記録した。

supervisor wall689.093秒、native起動から終了まで698.894秒（約11分39秒）、aggregate peak RSS0.956207GiB。出力411 file、logical7,908,446 bytes、file block allocation9,277,440 bytes。これは今回8標本の観測値で、全supportや大規模campaignの資源上界ではない。H4/他job/共有環境/既存venv/GPUへ作用しない。

known callsはcompile1716/trajectory842/occurrence3200/control360/primitive1470/reference400。旧予約compile20/trajectory10/occurrence64/control4を保持し、known＋予約は1736/852以内。unknown historical actual=null、予約をactual総数と呼ばず、返却/resetなし。新規実行はcompile16/trajectory8/occurrence64/event128だけ。

初回の起動helperが子spawn前にCPU22/23へ限定したためfresh CPU gateで拒否された事実を保存した。science worker/output/one-shot claim/callは全て0。helperのaffinityを子spawn後へ移し、固定SOURCE/grant/予算を変更せず未開始の一回runを起動した。科学run開始後のretry/resumeは0。最初のpreflight exit1と今回のnative exit0/時刻は別の証拠として保持し、旧証拠を置換しない。

## 検証と残る限界

source/input/phase/seed/order/vector probability/原finite law/両軸compiler/event digest/整数metrics/累積会計/native終端/cleanupをlocal metadataで照合した。公開CSV/JSONだけを使う別scriptの40算術検査も合格。旧人工58 testsは再実行せず、今回は認可した16実compileを完了した。CIや独立GPT科学レビューのPASSではない。

s2のM>=2確率は約0.7873%、8本でM>=2が未観測になる確率は約93.87%。実際には全8本がM=1で、複数order2・希少component列の費用を測定し尽くしていない。元lawから引いたs2標本としてmeanを推定できるが、母平均tail上界・正式winnerを保証しない。未観測層を0へ補完せず、K4 s4や他構成の費用を今回のK2 pilotから外挿しない。

今回のpilotは、観測したs2費用が逆転目安約9,189,927 RZ/shotより大幅に小さいという追加証拠を与えた。全supportの平均上界の証明にはならない。追加の条件付き費用取得・H8科学計算・新比較契約は未認可。結果をGPTへ戻して、留保付きbenchmark整理か、未観測費用をさらに取得する実験設計かを判断する。自動的に次の科学stageへ進まない。

軽量metadata/CSV/監査/解析scriptだけをcommit・non-force公開し、remote bytes照合後mandatory STOP。NPZ/native runtime/checkpoint/cache/credential/内部SSH情報は追加commitしない。
