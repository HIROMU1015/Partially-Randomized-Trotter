# H6精度一致・測定込み資源比較：探索結果 v1

状態：**探索592/592完了、独立確認未実施**。この報告は保存されたJSON scalar・wrapper費用の再集計であり、新signal・trajectory・回路build・transpileを行っていない。

[一つの資料入口](../../artifacts/resource_applicability/track_a_h6_exploration_summary_v1/2026-10-12/README.md)。

探索点推定では、PR（B2）の最良候補は4精度すべてでfull PF（B1のS2/S4を合わせた最良候補）よりshot込みRZ費用が約55–63%低い。discard baseline（B0）との比較でも約43–56%低い。これはP=0、経験的数値allowance、各PR候補8 trajectoryの条件下の結果であり、母平均winnerや一般的PR優位を認定するものではない。

## 対象・指標・coverage

linear H6、等間隔1.00 Å、STO-3G、12 modes、alpha3/beta3 sector400、保存state、T=0.8。DFはtol-only1e-8/rank19/cutoff0、signed generation orderと採用weighted Hermitian projectionを保持。splitはgeneration-prefix L_D=5/10/15とfull19。q=1/2/4/8、delta=0.8/0.4/0.2/0.1、B2 r=1/2/4・R=qr・K=2/4。要求精度は有限時間複素signalのepsilon_sigで、化学的energy精度ではない。

B0 discard-S2 12候補、B1 full-S2/S4 8候補、B2 partial-random 72候補、合計92。各方法の最良qを共通の有限集合から比較する。B0のprefix5・4候補は全精度で不適格であり、費用欠測や技術失敗には置き換えない。

| 保存・受理項目 | 件数 |
|---|---:|
| signal / B3 normalization組 | 92 / 84 |
| 全登録探索group | 592 / 592 |
| B0/B1 deterministic cost group | 16 |
| B2 cost group | 576（72候補×8） |
| primary measured wrapper | 1,184（X/Y各592） |
| ordinary感度wrapper | 4 |
| wrapper合計 | 1,188 |
| 独立確認 | 0、未実施 |

来歴は旧449＋server Phase Aの447＋前run保存7＋resume01新135。TASK_COMPLETE・source・整数seed・event・compiler・両軸・必要control validationを再受理し、原index集合0–591の欠落/重複なしを検査した。partial456/458や旧pilotは完成標本へ混ぜていない。

主指標はstate preparationを除くfull measured Hadamard wrapperのRZ。controlはsymmetric_directional、cosine/sineは同じtrajectoryを共有して実compileした。Qiskit1.3.0、basis rz/sx/x/cx、optimization1、seed17、backend/coupling/layout/routing指定なし。

`G_RZ(0)=N_real*mean(RZ_cosine)+N_imag*mean(RZ_sine)`。軸headroom `epsilon/sqrt(2)-abs(bias)-u`、各軸alpha=0.025の既存Hoeffding shot会計を使用する。Nは解析上の量子shot予算であり、量子shotを実samplingしていない。

## 精度ごとの探索点推定

下表はP=0、allowance1倍。費用はshot込みRZ総数で、古典wall時間ではない。各精度・family内のG_RZ点推定最小を示す。

| epsilon_sig | B0最良 / G_RZ | full PF最良 / G_RZ | PR最良 / G_RZ | PR/PF | PR/B0 |
|---|---|---|---|---:|---:|
| 0.05 | `B0_p15_q1_2nd` / 7.011705e+08 | `B1_p19_q1_2nd` / 8.770494e+08 | `B2_p5_q1_2nd_r4_K4` / 3.986200e+08 | 0.4545 | 0.5685 |
| 0.01 | `B0_p15_q2_2nd` / 3.901817e+10 | `B1_p19_q2_2nd` / 4.891811e+10 | `B2_p5_q2_2nd_r4_K4` / 1.933121e+10 | 0.3952 | 0.4954 |
| 0.005 | `B0_p15_q4_2nd` / 2.188569e+11 | `B1_p19_q4_2nd` / 2.736763e+11 | `B2_p5_q4_2nd_r4_K4` / 1.004086e+11 | 0.3669 | 0.4588 |
| 0.001 | `B0_p15_q8_2nd` / 1.178159e+13 | `B1_p19_q2_4th` / 1.329886e+13 | `B2_p5_q8_2nd_r2_K4` / 5.157441e+12 | 0.3878 | 0.4378 |

最良PRはいずれもprefix5。epsilon=0.05/0.01/0.005ではq=1/2/4、r4・K4、epsilon=0.001ではq8・r2・K4。これは探索の順位で、確認用候補集合をsealしてはいない。K2の候補も各精度で近接するため、K4の母平均優位をこの順位だけで認定しない。

full PFは粗い3精度でS2（q1/q2/q4）、epsilon=0.001でS4 q2が最小。S4を除いた比較ではない。各orderの結果もJSONに保存した。

| epsilon_sig | 適格B0/B1/B2候補数 | PR N_total | full PF N_total | PR Gのpaired相対SE |
|---|---:|---:|---:|---:|
| 0.05 | 8/8/72 | 30,400 | 23,350 | 1.446% |
| 0.01 | 3/7/54 | 752,081 | 658,413 | 1.745% |
| 0.005 | 3/7/54 | 1,987,674 | 1,859,100 | 1.165% |
| 0.001 | 2/5/36 | 53,687,901 | 60,083,125 | 0.190% |

SEは同一trajectoryのX/Yをpairedにした8標本の記述統計。16独立標本ではなく、同時信頼区間・選抜後の母平均保証ではない。deterministic n=1のSEはnullを保持する。

## 希少order・数値幅・準備費用の限界

72 PR候補のうち58候補では、探索8 trajectory中のnonzero Taylor order観測が0だった。未観測orderの費用を0とみなさず、母平均tail boundはnullのまま。既存order診断に加え、保存されたfinite-law確率からlog1p/expm1を使う未観測mass表示を保存した。p0≈1の桁落ちを避ける表示上の再会計であり、分布や標本を変えていない。

| epsilon_sig | PR最良のorder2/order4観測 | 未観測orderを含む1 trajectoryの確率 |
|---|---:|---:|
| 0.05 | 1/0 | 0.01835% |
| 0.01 | 0/0 | 3.2996% |
| 0.005 | 0/0 | 1.66998% |
| 0.001 | 1/0 | 0.000295581% |

epsilon=0.01/0.005の最良候補ではnonzero order未観測であり、上表の約3.30%/1.67%は未観測orderを含む1 trajectoryの確率。この可能性を小さいsample SEが解消するものではない。全72候補の観測count・既知mass・全zero標本確率はorder JSONを参照する。instruction guardは母平均費用の上界ではない。

数値allowanceは経験的uで、厳密なcertificateではない。保存precision rowsを既存のprecision_rows関数で再会計し、92候補×4精度×2感度の元記録と完全一致を検査した。u/log-B marginを10倍にしても、今回の適格候補数とfamily最良IDは同じ。最良PRのG変化は最大約0.000347%だった。ただしuの10倍感度は旧compiler bit同一性やroundoff policyの新しい科学的検証ではない。

共通state preparationの費用Pを含めるなら `G(P)=G(0)+N_total*P`。次の交点はP=0最良同士の固定pairの比較であり、Pに応じて全候補を再最適化した境界ではない。

| epsilon_sig | PRとB0の交点P（RZ/shot） | PRとfull PFの交点P |
|---|---:|---:|
| 0.05 | 42,908.9 | 67,862.3 |
| 0.01 | 210,126.5 | 315,869.9 |
| 0.005 | 920,451.7 | 1,347,610.4 |
| 0.001 | 1,902,919.4 | 非負交点なし |

粗い3精度ではPRが多くのshotsを使うため、十分大きいPで固定PF baselineとの優劣が逆転し得る。epsilon=0.001では最良PRは最良full PFよりshotsも少なく、この固定pairには非負交点がない。最良B0には非負交点がある。Pを未測定のまま、state preparation込みの最終総費用優位を認定しない。

## 補助結果

CX・depth・sizeのshot込みmap、RZ/CX depthを含む全wrapper raw metric、全92候補の適格/不適格・bias・B・shots・費用・SEをJSON/CSVに保存した。secondary metricの最適候補がRZ最適候補と一致するとは仮定しない。

ordinary感度のpaired replica0では、ordinary/primary RZ比はB1 S4 q1で2.16446、B2 prefix10 q2 r1 K2で2.05845（X/Yとも同値）。限定2anchorの結果であり、全familyのcontrol比較ではない。

B3は84 normalization組のみ。log_B範囲は0.982281–20.213363。B3 signal/full-wrapper費用は取得していないため、family全体の科学的排除をしない。

## 実行・環境・資源・来歴

実行SOURCE `78a39051d8b6d41c33d592ecfe172f8824adfaeb`。科学SOURCE `5a903aea2492c2d211c1e15e7737df0d50694c12`。report作成後もruntime240 source closureは実行manifestのhashと一致。旧science224を変更していない。

候補環境は既存private venvのPython3.11.0rc1、Qiskit1.3.0、NumPy1.26.4、SciPy1.14.1、Numba0.60.0、OpenFermion1.6.1、PySCF2.7.0。固定profile SHA `4e32bb9ba57c38280562390100ebf916e9858218534504ead1519625a609bd75`、compiler hash `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。

旧449結果は元host/sourceのまま。server Phase A・丸め7・resume135はsource別に記録した。利用者承認済みのroundoff bridgeは旧native basis/係数の完全bytes同一性を解除し、元discrete choice/order/sign/seedと限定float許容を維持したもの。事前診断の最大float差2.7755576e-17、追加演算差見積り4.8653727e-12はその限定診断の値で、全592結果での旧compiler output bit同一性証明ではない。旧host/compilerの完全output同一性は未検証のまま。

今回のresume01は2026-10-12 20:10:47 JSTにcoordinator exit0を観測。wall15,460.684819秒（約4時間18分）、aggregate peak RSS11.1797 GiB。135 workerをすべてwait/reap、own cleanup検証済み、pending0、cleanup error0、journal/log閉鎖済み。これは新135取得runの時間・peakであり、旧449やPhase A/前runを含む全研究の古典時間ではない。

worker4 CPU20/21/24/25・thread1、coordinator CPU22/23。worker AS32 GiB・sampled RSS9 GiB、aggregate RSS40 GiB、job RAM予算64 GiB、wall cap=null。host RAMの予約を主張しない。全runは既定の監視下で完了・mandatory STOPとなった。H4/venv/共有環境/他job/GPU変更0。

完成・診断を含むknown callはcompile1188/control360/trajectory578/occurrence2176/primitive1470/reference_matvec400。過去未完了actual totalはnullを保全し、独立予約compile12/control4/trajectory6/occurrence48は実数と呼ばない。失敗分返却・budget resetなし。

## 未完了の研究工程・公開状態

独立32 trajectory確認は0。結果前規則は「各精度B2 point G_RZ上位2、ID辞書順tie、union最大8を固定後、独立namespaceで各32」。この報告では探索順位を表示するだけで、候補seal・確認task/grant・runnerを作成/起動していない。探索8と確認32をpoolしない。H8/追加探索も行わない。

次は保存mapに対する研究判断と、必要なら原規則による確認準備・別実行認可。母平均winner、数値精度certificate、ground state、chemical accuracy、最終量子総費用・科学的GOは未認定。

本報告はlocal保存metadataの再受理・scalar算術検査であり、独立レビュー/CI/外部再現ではない。source/testsは変更せず、軽量report資料だけをlocal commitへ固定する。利用者のpush方針を維持し、agent pushは行わない。raw runtimeとNPZはcommitしない。全groupの外部worktree path/hash参照はprovenance JSONに記載し、Git公開後の完全再現には外部rawの別受領が必要。
