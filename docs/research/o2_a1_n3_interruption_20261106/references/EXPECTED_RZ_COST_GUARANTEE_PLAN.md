# 期待compiled RZ費用の保証計画

2026-11-06 JST。**設計案／未採択／実科学・実compile・保存数値再解析は実施しない。**
base `a167effd4a1192d9806ea0bde8e0f7af71a1ef55`、採択済みN3契約
[03928ce5…/EXECUTION.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/03928ce55bc5a1bfd070aa9f0a6bf8a990a0ff25/docs/research/o2_a1_n3_science_20261106/EXECUTION.md)
を参照する。対象条件と結果判定は[判定方針](N3_RESULT_DECISION_FRAMEWORK.md)を併読する。
以下の追加取得、n/J/信頼水準/完走率閾値/価格上界の数値は正式採択していない。

## 1. 推奨する順序と、現在保証できないこと

最小の入口は、同じselected full価格providerを使った**独立価格較正**と、
全supportの価格上界の根拠を用意することである。それだけで凍結済みproduction計画の期待費用区間を得られる。
しかし学習・pilot・calをやり直す手法全体の期待総費用には、ランダムなshot数とSTOPを積分する必要がある。
次段階として、独立したtrain/pilot/cal履歴だけを反復し、productionの測定を省いた条件付き期待値で評価する方法を推奨する。
まず静的な保守的包絡で比較が確定しないか調べ、足りない場合だけ履歴反復を提案する。

現在は実λ/価格/variance/N/期待値を取得しない。実現campaign差、pilotの点平均、
保存済みの経験的proxyを期待費用改善の保証へ読み替えない。

## 2. 確率空間と、比較する費用を先に定義する

対象は保存linear H4/1.00 Å/STO-3G/legacy DF12、8qubit/全4電子sector/N2保証ground、
L_D3、m0/q1/δ=.02/r1/K2/coefficient_atol0、X/Y、N1 actual Q/selected policy、
固定exact engine・理想uniform53-bitモデル、同Qiskit1.3.0 opt2 no-prep full RZに限定する。
別の物理系、PF誤差係数C、energy/RPE保証、native noiseには接続しない。

固定compiler/profileでpath ω・axis bの整数価格を c_b(ω) とする。compiler乱数seed17は固定し、
平均はcompiler seedの平均ではない。確率はpath/測定wordと、それが決めるfit/cal/配分/STOPに対するもの。
実PRNGを数学的に独立と証明した意味ではなく、N3と同じ理想word仮定を明記する。
固定seedの一回の決定論的出力そのものに確率分布を後付けせず、独立wordによる反復policyの期待値を対象にする。

| 記号 | 意味・主張できる範囲 |
|---|---|
| C_m^obs | 1campaignで実現したtrain/pilot/cal/prodの全RZ。平均の保証ではない |
| T_m(H) | 履歴Hとquotaを凍結した後、fresh productionを行う条件付き期待総費用。過去費用は実現値のまま |
| Θ_m^plan=E_H[T_m(H)] | 同じ方針で学習からやり直す計画仕事量の期待値。production途中STOPを省いた場合はその旨を明記 |
| Θ_m^stop=E[C_m^stop] | methodを単独で開始し、採択されたguard/cap/STOPで終える1attemptの期待全費用。返却/中断を共に含む |
| E[C_m^stop | return] | 完走選別された期待値。上の無条件期待値の代用にはしない |

研究上の比較候補は Θ_A1^stop と Θ_B1-OL^stop に加えて、各方式の返却確率と返却精度を同時に報告する方式。
同じ返却サービスを達成しない安い中断方式を資源削減成功としない。要求完走率も後日の採択事項とする。
N3の全体controllerはA1 trainの失敗でB1まで中断するので、その実験内のB1費用はA1の成功に依存する。
controllerのE[方式別費用]自体は定義できるが、独立に配置するA1/B1の期待費用比較とは違う。
**単独方式としての履歴反復は将来の別プロトコル案**であり、実行中N3の順序を変更しない。
再試行を組み込んだ「成功までの期待費用」も新方針になる。C^obsやΘ/p_returnで勝手に代用しない。

## 3. 既存実装の再利用表

以下はbase内のSOURCEの静的調査。既存SOURCEの変更・実分子関数の呼出しは行わない。

| 接続・既存関数 | 再利用できるもの／追加が必要なもの |
|---|---|
| `a1_n3_connection.ConditionalSampler` | actual整数countと5層の直接条件付きlaw。price-only stageに渡す独立domainの薄いwrapperは後日必要。既存N3のstageやseedを流用しない |
| `a1_n3_connection.SavedPrices.price` | 保存selected logical full列、X/Y、phase/制御/ancilla/測定、固定compiler。価格だけの取得へ使用可能。現load経路のengine保持からBorn/stateを評価器へ漏らさない |
| `a1_connections.selected_full_prices` | live request用の同価格規約。保存payloadでは再preparationせず上のSavedPricesを優先 |
| `rte_compiled_cost.transpile_and_measure_cost` / `TranspiledCircuitCostCache` | 実compile metric、fingerprint・compiler/実回路binding、private SQLite cache。hitでも展開先shotは毎回課金 |
| `a1_control_variates.fit_and_freeze/FrozenCV`、`a1_n3_connection.freeze` | 固定3control、exact凍結係数、range、rank。新feature/clip/再fitなし |
| `calibrate/sufficient_shots/choose_allocation/plan_production` | 独立cal、variance上界、外向き十分N、J8の同じpolicyを履歴反復で呼ぶ。policyを解析用に単純化しない |
| `TrialCost/account_costs` / `summarize_costs` | 全段階実現価格、UNKNOWN、重複trial拒否。既存関数自身は期待費用保証を返さない |
| `h8_rare_tail_selection.bounded_simultaneous_radius` | 全support上界を前提としたHoeffding＋family unionの式。別H8 law・grant・runnerは流用しない。float値は人工検査用、最終certificateは外向き丸めへ接続 |
| `df_correction_comparison.intervals` | rational/interval log/sqrt/ceil・外向き端点の再利用。未評価の実価格上界を生成する関数ではない |

旧connected-cluster/affine proxy、nominal label確率、別opt_levelの価格をrigorousな代替価格にしない。
selected fullの一体compile価格はevent価格の単純和ではない。

## 4. Path/stratum価格lawと全supportの上界

5層をh、actual質量をp_h>0、条件付きlawをQ_hとする。

\[
F_{h,b}(x)=\sum_{\omega\in h}Q_h(\omega)1\{c_b(\omega)\le x\},\quad
\mu^c_{h,b}=E_{Q_h}[c_b(\omega)],\quad
\mu^c_{Q,b}=\sum_h p_h\mu^c_{h,b}.
\]

B1-Oの価格meanも、該当5層のexact質量をorder質量で割った混合で得る。
A1/B0は同じμ_Q、B1-OLは同じμ_hを使えるので、価格poolはmethodごとに重複取得する必要がない。
X/Yを同価格と仮定せず、全10cellで同時に保証する。

必要な前提は **a_hb≤c_b(ω)≤u_hb が全Q_h supportに成立**すること。
安全な下限は0でよい。sample最大値、観測したn2だけの最大、engineのL_maxや未compile sizeをuへ格上げしない。
「opt2なのでgate数は増えない」という推測も証明にならない。有限support・固定seedだけでは数値uを取得したことにならない。

上界取得の候補は次の順に考える。

1. **同じ固定compiler passの構造的上界**：保存gate alphabet/最大列長、制御分解、basis translation、opt2の各passと反復、最終RZ数までを結ぶ検査可能な根拠を用意する。engineの列長証明は入力に再利用できるが、compile後の上界は別に必要。これは現時点で未取得。
2. **厳密な有限部分和＋残余上界**：事前に選んだ有限path集合S_hの価格を列挙し、exact Q_h massで加重する。残余質量r_hに対し `[Σ_{S_h}Q_h c +r_h a_hb, Σ_{S_h}Q_h c+r_h u_hb]`。n0全列挙を優先し、n2残余が残る限り上界を要する。結果を見た集合追加は別scopeにする。
3. **全有限supportの価格列挙**：最後の代替。positive label数をsとするとn0/n2のordered pathはsとs³、両軸で最大2(s+s³)件。全価格が得られれば期待単価は厳密和になり、価格CI不要。ただし最小scopeではなく、今回も将来も自動実行しない。

どの取得でも、上界の根拠が閉じなければ**保証経路STOP**。点推定の表示は保証未取得と明示する。
価格をclipした平均は別estimand。閾値超過時STOPを追加する案も元の平均価格を保証せず、新algorithm条件になる。
経験的外れ率やbootstrapは未観測価格tailの保証を代替しない。

## 5. 独立価格較正と必要compile数

将来の案では、N3 train/pilot/cal/prodと独立な `price_cal` domainで、各(h,b)に事前固定n_hb本をQ_hから取得する。
測定bit・Born・state作用は不要で、path→selected full build→既存compile/cache metricだけを取得する。
実入力のRNG/build/compileなので別の明示許可を要する。N1再生成、state solve、engine再認証は含めない。
pathが同一でも独立抽選のmultiplicityを標本に残し、unique価格だけの平均にしない。

Δ_hb=u_hb−a_hb、δ_hb>0、Σδ_hb≤γ_priceを先に固定し、

\[
r_{hb}=\Delta_{hb}\sqrt{\log(2/\delta_{hb})/(2n_{hb})},\quad
[L_{hb},U_{hb}]=[\max(a_{hb},\bar c_{hb}-r_{hb}),\min(u_{hb},\bar c_{hb}+r_{hb})].
\]

同時確率≥1−γ_priceで全μ_hbを被覆する。必要半幅t_hbに対する十分数は
`n_hb ≥ ceil(Δ_hb² log(2/δ_hb)/(2 t_hb²))`。定数価格が全supportで証明されればCI不要。
取得中のword cap等で止まればCIを返さず、失敗trialを捨てて再抽選しない。
有限取得guardを伴う場合の主張は `P(CIを返し、かつ被覆失敗)≤γ_price` であり、完走条件付きcoverageではない。
範囲正規化したHoeffdingを用い、最終端点は既存interval helperで外向きに計算する。
この標準不等式と既存variance較正の出典は
[Maurer–Pontil (2009), v1 Theorem 1・10](https://arxiv.org/pdf/0907.3740v1)。
本計画の費用合成とSTOP補正は、以下で明示する条件付き期待値・非負有界性による。

全5層×X/Yなら標本数・compile要求数はΣ_hb n_hb、fresh cacheのmiss数はそれ以下。
price-onlyには量子shotを実行しないので、それ自体の実現量子RZ消費は0で、将来の1shot価格を取得する古典評価費である。
一方、deploymentに独立**量子bit**較正を加えるなら、そのshot価格を総RZへ加える必要がある。
研究評価用compile費と、配布後algorithmの量子費・古典費を別々に報告する。

独立poolは選択バイアスを避ける簡潔な十分設計である。数学的に常に独立poolだけが可能という意味ではない。
元pilotも事前fixed-n/IID/同じ全support上界と全候補同時CIが揃えば利用を検討できるが、
選ばれた方式/完走者だけの価格平均、calとの無根拠な積、N3の未完了データでpoolを代用することは本案ではしない。
price CIで配分を変更するなら別policyになる。外部評価用poolは既存policyを変えない。
標本数は事前固定を基本とし、任意の中間結果で追加せず、逐次化するなら別途confidence sequence等を採択する。

## 6. Calibration依存Nと全期待費用

Hをtrain/pilot/calと凍結λ/配分/STOP情報までの全履歴、A_m(H)をその時点までの**全実現RZ**、
N_mhb(H)をproduction quota（計画が成立しない履歴では0）とする。
A1のproductionはQなので、5層表示では重み `w_A1,hb(H)=p_h N_Ab(H)`、
B1-OLでは `w_OL,hb(H)=N_OL,hb(H)` と置く。
fresh productionが全quota完了し、未来pathがHから独立で指定lawに従う場合、

\[
T_m(H)=A_m(H)+\sum_{hb}w_{m,hb}(H)\mu^c_{hb},\qquad
\Theta_m^{plan}=E_H[A_m(H)]+\sum_{hb}E_H[w_{m,hb}(H)]\mu^c_{hb}.
\]

Nとcal/pilot価格は一般に依存する。`E[N×cal価格]=E[N]E[cal価格]`とはしない。
上式で真μとE[N]が分離できるのは、**未来の1shot価格lawがHと独立で固定**だからである。
λ、N、pilotの間の相関を捨てた近似ではない。
凍結Hに対して `[A_m(H)+ΣwL, A_m(H)+ΣwU]` は条件付き期待費用の区間だが、
Hをやり直した無条件期待値の保証ではない。

所定全段階が無条件に実行され、途中guard/word中断がない設計なら、production前の期待費用は

\[
E A_{A1}=\sum_b\{64\sum_h\mu^c_{hb}+512\mu^c_{Q,b}\},\quad
E A_{OL}=\sum_b\{64\sum_h\mu^c_{hb}+512\sum_h\mu^c_{hb}\}.
\]

これはconditional train、pilot、全calを含む。完走者だけに条件付けた標本でこの等式を使わない。
現N3の早期STOPを含む費用は到達indicatorと価格の依存を扱う必要があり、
実行されなかった固定費まで単に足すことも、失敗historyを捨てることもしない。

最初の保守的経路は、全履歴でのquota/段階数上界とuを使うΘ_A1上界、
必ず実行する段階とaによるΘ_OL下界で差を挟む方法。範囲が交差すれば**未判定**とする。
成功時だけのN≥1は、中断込み無条件期待値の下界にはそのまま使えない。

## 7. 無条件期待総費用：production測定を省く履歴反復案

全量が必要で保守的包絡が足りない場合、将来の別実行でJ個の独立履歴を学習から取得する。
同じ固定train/pilot/cal/OLS/rank/range/variance/N/J8 policyを使い、成功履歴も失敗履歴も保存する。
production **bit/Bornは取得しない**。各履歴の全pre価格A_mjと凍結quotaから

\[
Y^-_{mj}=A_{mj}+\sum_{hb}w_{m,hb}(H_j)L_{hb},\quad
Y^+_{mj}=A_{mj}+\sum_{hb}w_{m,hb}(H_j)U_{hb}
\]

を構成する。price poolと独立な履歴にする。途中STOPを再試行してquotaを埋めない。
既存のN3データを独立replicateとして増幅/再解析せず、fresh独立campaign historyだけを使う。
新policyの単独反復であり、実N3一campaignとは区別する。

全履歴での段階上限とuから**決定論的** B_m≥Y^+_mj≥Y^-_mj≥0 を証明する。
production上界は各axisで `8192 max_h u_hb`、preは全許可段階のquota×該当uの和でよい。
guard/worker中断時に回収されるin-flight trialも上限内に含め、未価格を0にしない。
固定Jについてρ_m=B_m sqrt(log(2/δ_out,m)/(2J))と置けば、price同時被覆と合わせて

\[
\Theta_m^{plan}\in
[\max(0,\bar Y^-_m-\rho_m),\min(B_m,\bar Y^+_m+\rho_m)]
\]

を得る。両端に一側Hoeffdingを適用してunionする。両方式へΣδ_out,m≤γ_outerを配る。
同じprice intervalを全履歴に使うことによる相関は、poolに条件付けて処理でき、Jごとにprice失敗予算を重複加算しない。
この条件付けはpoolの実現値を固定する証明上の操作であり、被覆成功者だけを選ぶデータ選別ではない。
`J ≥ ceil(B_m² log(2/δ_out,m)/(2s_m²))` はouter半幅s_mの保守的十分数。
幅が大きければ保証未判定でSTOPし、勝つまでJを増やさない。

### productionの途中STOPを省かないための補正

上のTはquotaを全て実行する**計画量**。有限word cap/production中断を無視したままΘ^stopと呼ばない。
同じ無限独立word列上の仮想完了列と実行prefixをcoupleでき、
production中断で省かれる非負RZが常にB_m^prod以下であり、
production途中中断確率に無条件上界η_mを持つなら

\[
\max(0,\Theta_m^{plan}-B_m^{prod}\eta_m)
\le\Theta_m^{stop}\le\Theta_m^{plan}.
\]

pre段階STOPは既にA_mj/N=0に反映する。η_mは追加のproduction STOPだけに対応する。
word capはconditional整数棄却の全許可集合countから解析的に上界化するか、
独立履歴のfuture path/wordだけを所定quotaまで生成して中断indicatorを取り、固定JのBernoulli上界で評価する。
後者も新たな科学RNG取得なので追加許可範囲へ含めるが、production Born/bit/全価格compileは不要。
中断とpathの依存があっても、上のB×確率の補正なら条件付き価格meanのすり替えを避けられる。
新規callを停止した後に回収するcallを含め、実費が仮想完了費以下というcouplingが必要。
compile/環境失敗を確率0と仮定して数値補正を付けない。そのモデル・上界・couplingが閉じなければΘ^stop保証は保留する。

pre段階の中断確率も全historyからBernoulli上界を作り、上のηを足してp_stop上界を出す。
返却精度の信号失敗確率と併せ、`P(中断または精度違反)≤p_stop+α_signal` としてサービスを評価する。
同じ精度でもp_stopが大きい安い方式を、同じ成功率の資源改善として採択しない。

## 8. 比較保証と失敗確率配分

両方式の**同じestimand**の区間をI_A=[L_A,U_A]、I_B=[L_B,U_B]として

\[
\Theta_A-\Theta_B\in[L_A-U_B,U_A-L_B].
\]

U_A<L_Bなら期待費用減少の保証、L_A>U_Bなら逆向きの保証、それ以外は未判定。
L_B>0のときだけ比を `[L_A/U_B,U_A/L_B]` で示す。
要求する返却率/精度・全段階会計が同等に成立することを資源削減claimの別gateとする。
凍結quota同士なら、共有価格の差係数d_hb=w_A−w_Bを使い、
正係数に[L,U]、負係数に[U,L]を掛けると共通項を相殺した、より狭い差区間を作れる。
全履歴版へはouter誤差とSTOP補正も含める。price poolの共用は共通signal測定の採択を意味しない。

新しい保証familyをγ_costとし `γ_price+γ_outer+γ_abort≤γ_cost` を先に固定する。
各price cell/方式/中断indicatorにその内訳を配分し、固定familyのunion boundを使う。
cross-method/cell間の独立性はunion boundには不要だが、各標本meanのIID、
price poolとouter履歴、calとproductionの独立性は、それぞれ使う式の前提として必要。

既存N3のα_diag=1/20は信号精度family専用であり、価格保証へ転用しない。
信号と費用を一つの同時claimにすると失敗確率上限はα_diag+γ_cost（必要ならavailability分も明示）。
同じ総予算1/20の中に新保証も入れるなら、新配分と必要Nを採択する**別実験**が必要で、N3の保証を遡及変更しない。
deterministicに認証済みのengine誤差は座標bias枠であり、根拠なく確率budgetに入れ直さない。

### 認証engine上の期待費用と、理想量子測定上の期待費用

履歴反復が直接評価するのは保存exact engineのbit law上の期待費用である。
座標biasが小さいだけで、閾値/ceilを含むNの期待値まで同じだとはしない。
理想logical量子測定によるdeploymentの期待費用へ移すには、別の明示的な移送幅を付ける。
同じ全pathで一様なraw bit mean差がb_engine以下なら、bitのBernoulli確率差はb_engine/2以下。
同じpath/wordから二つのpre履歴をcoupleし、最大pre bit数をM_m^preとすると、
不一致確率は `ζ_m=min(1,M_m^pre b_engine/2)` 以下にできる。
価格・制御方針が同じで、全費用が[0,B_m]なら
`|Θ_m^ideal−Θ_m^engine|≤B_m ζ_m`、中断確率差はζ_m以下。
両軸・全conditional train/pilot/calをMに含め、production outcomeが配分を変えない前提を使う。
値やbitに依存する追加STOPがある場合は、必要なproduction bitもcoupling数へ含める。
これはb_engineを費用へそのまま加算する操作ではなく、履歴不一致を介した決定論的な保守幅である。
所定law/全support一様性が確認できない場合はこの移送を主張しない。
大きすぎて差の符号が確定しなければ理想deployment改善は未判定とし、勝手にengine精度を上げない。
上の価格/outer/STOP区間の後にこの移送幅を加えてから、理想量子費用の差を判定する。
native合成誤差や実機noiseの保証はこの移送にも含まれない。

## 9. 次回に提案する最小scopeとSTOP

| 段階 | 追加量の提案・承認対象 | STOP |
|---|---|---|
| P0：価格support根拠 | 同じ保存law/selected列/固定compilerの構造的上界。新分子計算なし。uと証明範囲を事前固定 | all-support compiled RZ上界が未閉鎖、profile/依存差、別compilerへの変更が必要 |
| P1：独立価格較正 | 全5層×X/Y、Σn_hb path、full build/compile要求≤Σn_hb、missはunique key数以下。Born/bit0 | 価格欠測・bound違反・identity/stream衝突。clip/再抽選・標本追加なし |
| P1だけでの結論 | 静的quota包絡による無条件区間、または将来承認された凍結計画に対する条件付き区間。今回N3結果は使わない | 区間重なりなら未判定。条件付き区間を手法全体の期待値へ昇格しない |
| P2：必要な場合のみouter | J独立A1/B1-OL単独pre履歴、X/Y。最大pre bit/価格要求 `2J[(5·64+512)+(5·64+5·512)]`。production bit0。N1/N2/proof再計算0 | guard/capによるhistory中断を記録。全実験は技術的不整合・契約外call・scope超過でSTOP。history失敗のやり直しなし |
| P2の任意補足 | ηの解析bound、またはfuture production path/word-only trace（各履歴method/axis≤8192）。新RNGのみ、Born/bit/全prod compile0 | 有効なSTOP coupling/上界なし、stream再利用。ηを未取得のまま0にしない |
| 公開 | SOURCE/入力/price上界/family/n/J/全失敗history・UNKNOWN/保証区間・返却率を保存 | 公開照合後mandatory STOP、次の計算を自動開始しない |

P2のcompile要求はP1要求＋上記pre path要求が保守的上限で、専用cacheのhitにより減る。
全support列挙は別代替案であり、この最小案のscopeに自動追加しない。
独立履歴の並列化・保存bundle共有・既存cacheを使えるが、空き資源と他者非干渉に合わせる。
CPU/RAM/workerの恣意上限を新設せず、実行中N3や共有環境を操作しない。

未解決なのはuの根拠/実際の幅、必要n/J、価格とNの分布、完走率、STOP coupling、理想測定への移送幅、計算実費である。
今回の文書だけで、A1の期待費用改善・実行可能な数値予算・科学的優劣は確定しない。
次に必要なのは、選ぶestimand、返却率と失敗確率配分、P0/P1（必要ならP2）の限定scopeに対するユーザーの採択・明示許可。
新しい承認証明書/grant発行は要求しない。詳細科学レビューは開始しない。

**本作業は設計文書・人工検査の公開照合後mandatory STOP。**
