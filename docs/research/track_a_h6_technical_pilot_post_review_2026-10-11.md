# Track A：H6技術pilot v2 STOP後の独立科学レビュー
## 部分証拠の採用、精度一致資源比較の設計、H6からH8への研究判断

- 作成日：**2026年10月11日（日本時間）**
- レビュー開始承認：利用者の「レビューを開始して」。直前に提示したレビュー論点を対象とする。
- Repository：`HIROMU1015/Partially-Randomized-Trotter`
- 対象branch：`track-a-ax2b-h4-post-review-20261010`
- 固定結果commit：`eca6bfdab85fa7c41cbe5dcdec770cb1cd95c9e2`
- 実行science source：`0b04886869efb9d08b07d6517300da2bc0123f4a`
- 実行seal：`fc97d46db9dc84d255eaf317d1fea75fd1d1c015`
- 原実行status：`H6_TECHNICAL_PILOT_STOP`。本書は原status・契約・grant・rawを変更しない。
- レビュー状態：**本書に記載する範囲で完了。新しい科学計算の実行認可は付与しない。**

## 0. 結論

**研究方針は維持する。H6の7 cellと保存32 wrapperは、登録範囲の技術・経験的な部分証拠として採用し、H6の精度一致資源比較を設計する段階へ進む。**

未保存のB3 replica1・4 wrapperの補完を、次の研究設計の一律前提にはしない。旧pilotは32/36のまま保存し、36/36 COMPLETEへ変更しない。これは不足を隠す判断ではなく、今後の研究判断に対する追加情報価値を区別した判断である。

次に優先するのは、B2と二次・四次の決定論PFを、**同じqではなく同じ信号精度で、normalization・bias margin・実際のwrapper費用を含めて比較すること**である。B2のKやRだけを増やすこと、B3の同じ不利な点で費用標本を増やすこと、直ちにH8へ移ることを第一案にはしない。

**次の担当はCodex。** H6主検証の結果前契約、費用取得の効率化、数値・統計の扱いをまとめて具体化する。実際の追加計算は新しい範囲・実行認可の下で行う。H8の独立評価は、その前にH6の問い・比較規則・予測を固定してからとする。

| 判断対象 | 本レビューの判断 |
|---|---|
| H6入力の工学的完成 | 既存の完成記録を継承する。Hermitization議論を最初からやり直さない |
| 7/7 correctness | 検査した入力・state・scheduleでの技術的整合性を支持 |
| 保存32 wrapper | 個別・paired回路費用として使用可能。random母平均の認定ではない |
| 旧pilot全体 | 未完了STOPを維持 |
| B3残り4 wrapper | 本検証設計への一律必須条件から外す。補完は目的を限定した任意の技術作業 |
| PRの一般的優位 | 未確定 |
| 有望な次の科学的問い | B2の回路短縮が、四次PFとの精度一致後も残る領域があるか |
| RQ-R / RQ-P1 | RQ-R主軸、RQ-P1補助を維持 |
| H6主検証 | **計画具体化へ進める。即時の無制限launchではない** |
| H8 | まだ参照性能を開かない。H6後の独立検証として残す |

## 1. 根拠の読み方と、今回行ったこと

### 1.1 証拠区分

本書では次を区別する。

- **保存事実**：GitHubの報告・JSON・sourceが実際に記載する条件、数値、欠測、実装。
- **レビュー内算術**：保存されたscalar・gate countから本レビューが計算した比、normalization、条件付き比較量。
- **数学的導出**：明示した仮定の下で成立する式。保存数値の厳密な丸め認証とは別。
- **解釈・提案**：どの検証に価値があるか、次の対象・完了条件をどう定めるかという研究判断。

### 1.2 確認範囲

結果索引、準備文書、保存結果要約、paired metadata監査、実行中のsetup portと継承元のsignal/oracle/cost処理を読んだ。primaryのsymmetric_directionalについて、保存された**16 wrapper（8 replica groups × 2 axes）を直接取得**し、個別費用を確認した。ordinary側は代表5 wrapper（08/16/20/24/28）を直接取得した。残りordinary記録については、公開済み監査・索引の対応を根拠にし、本書で全件の費用を独立再集計したとは扱わない。[S01] / [S02] / [S03] / [S08] / [S09] / [S10] / [S11]

7 cellの中心数値は保存JSON要約を読み、実装上の計算式と照合した。全cellの大きいstate traceを一件ずつ別実装で再計算したものではない。全206 science source・142 rawのSHA-256をGPTが独立に総当たりで再計算したわけでもない。

レビュー用の算術は、取得したJSON fieldを明示的に転記した入力に対する標準Pythonの算術である。転記入力を元rawのbyte-exact copyとは称さない。補助コードはリポジトリのmoduleをimportしない。

**非実施**：SCF、DF再分解、state生成、solver、Hamiltonian時間発展、回路build/compile、trajectory生成、旧runner再起動、referenceの再fit、GitHubへのwrite。レビュー中に新しい分子科学データを生成していない。

### 1.3 今回の主目的

単なる「残り4件を埋めるか」という工程管理ではない。H6への接続によって得られた証拠を使い、PRの資源研究のどこを次に検証すべきかを判断する。sourceの全面的なセキュリティ監査や、全先行研究の網羅的な新規性再調査は対象にしていない。

## 2. 原runの到達点と、部分結果の採用

原reasonは`PHASE_WALL_CAP:wrapper_cost`。parent wallは3,913.9031868656166秒、worker exit -9。correctnessは7/7、wrapperは32/36。最後の保存boundaryは`wrapper_31.json`後で、B3 replica1へ入る前のprogressにseed 2416534881が記録されている。[S01] / [S02] / [S03] / [S06]

欠測は次の7件である。

`cost_summary.json`、`worker_terminal.json`、`H6_B3_K6_q2_R4_rep1_trajectory.json`、`wrapper_32.json`～`wrapper_35.json`。

最後の保存counterはcompile32、control_probe192、occurrence6、primitive735、reference_matvec400、trajectory3。これは**最後の保存境界のcounter**であり、その後の試行・途中処理がゼロだったという証明ではない。未完了の4件を推定して補わない。

今回のSTOPから数値gate不一致、PRの反証、四次PFの不具合は結論できない。一方、worker最終確認と全出力の完成には達していないので、成功runに再分類しない。

本レビューは正常に保存された個別記録を、scopeを明記した部分証拠として使用する。後日集計表を作る場合も「STOP後の保存値再集計」として別成果物にし、原runに存在しなかったcost_summaryを原出力へ追加したことにしない。

## 3. DF入力問題は、今回の主な停止要因ではない

親入力完成記録では、保存raw19 fragmentへのweighted Hermitian projection、係数再構成、指定stateとsnapshotの保存まで工学的に完了している。旧無重みgateの違反15–18は保全されている。[S12]

- 追加projectionのN12三角和評価：約9.9025×10^-13 Ha。
- N6評価：約2.4920×10^-13 Ha。
- 新政策のdecision gate：9.9×10^-11 Ha、追加予算10^-10 Ha。
- solver energy：−3.2360662799087647 Ha。
- 保存residual：約6.1060×10^-15。
- snapshot SHA：`99440a59d903dfd6330e786d84a956f1dd8a687a592295a1cd771c839769b005`。

これらは工学的評価であり、provider truncation、raw-to-integralの係数上界式、projection、state residualを一つの保証値へ混ぜない。ground-state/representation/total-uの厳密認定は未成立のままである。

ただし、今回のfinite-time taskは採用したDF Hamiltonianと指定保存stateに対するものなので、真の基底状態の完全証明を得ることを、この技術pilotの結果を使用する一律前提にする必要はない。新しい問題がないのにSCF・DF生成からやり直す必要もない。

## 4. H6で確立した技術的な範囲

入力はlinear H6、1.00 Å、STO-3G、12 modes、α3/β3、sector400、19 fragment。T=0.8、mid-prefix10、B3もone-bodyは決定論側に残る。[S01] / [S08]

referenceは400次元sector matrixのbinary64 expm/eigh。全400 occupation-columnとの構成照合の最大差は約3.3312×10^-15、expm/eighのstate差は約1.5220×10^-15、signal差は約3.4509×10^-16。[S02]

245個の「physical primitive ID・実時間」の組に3 probe、計735作用を実施し、最大差は約5.9723×10^-15。245は異なる時刻だけの個数ではない。局所probe、sector構造、full-vector作用、独立occupation構成の役割を分ける。

native Horner経路と独立forward Taylor/occupation経路のsignal差は、7 cellでおよそ10^-15～3.5×10^-14。これは指定stateと登録scheduleに対する経験的な一致であり、全状態上のoperator誤差上界や化学Hamiltonianに対する精度保証ではない。[S09] / [S10]

cost用の8 complete replica groupについて、ordinary/symmetric両branchとcosine/sine wrapperの検査記録が保存されている。最大値はmetadata上約2.4944×10^-14。これも有限probeでの検査である。[S03]

### 4.1 全回路とsignalの接続について残す限定

継承元`control_and_measurement()`は、ordinaryのcontrol=1 branchから得たzを測定wrapperの比較先とし、ordinary/symmetricの一致も見る。したがって、主としてbranch/control/測定の自己整合性を検査する。別経路の有限平均と、任意のsampled full-trajectory回路の全operator関係をこの検査だけで再証明したわけではない。[S11]

次の主検証runnerでは、既に計算するdeterministic full-wrapperのzを、対応するdeterministic signal recordと照合・保存すると説明が明確になる。randomでは個別trajectoryとensemble平均を直接等置せず、既存のevent/確率/phase/normalization検証を引き継ぐ。この記録改善のためだけに全36 wrapperを再compileすることは求めない。

## 5. 保存signalの科学的な読み取り

| 構成 | total signal差 | signed Re差 | signed Im差 | log B |
|---|---:|---:|---:|---:|
| B0 / S2 / prefix10 / q2 | 0.0264142608896 | -0.0135590967632 | -0.0226685701646 | 0 |
| B1 / S2 / q1 | 0.0112836805341 | 0.00622410788554 | 0.00941179724737 | 0 |
| B1 / S2 / q2 | 0.00269773709837 | 0.00143360415419 | 0.00228529310615 | 0 |
| B1 / S4 / q1 | 0.00343597838414 | -0.00179697042444 | -0.00292862506136 | 0 |
| B1 / S4 / q2 | 0.000236742718454 | -0.000124264170141 | -0.00020150814068 | 0 |
| B2 / prefix10 / q2 / R4 / K2 | 0.00269773715901 | 0.00143360447699 | 0.00228529297524 | 0.000205719741307 |
| B3 / prefix0 / q2 / R4 / K6 | 0.00220807406831 | 0.00117429487879 | 0.00186992583511 | 20.2133626291 |


表は[S02]の保存値。エネルギー誤差ではなく、採用DF・指定state・T=0.8に対する複素signal差である。epsilon_signal=0.001は原pilotの診断ラベルで、全7 cellがその精度を達成したという意味ではない。

### 5.1 B2：finite cutoffではなくouter-PFが支配的

B2ではouter-PF差が約2.6977372×10^-3、finite-RTE差が約3.4391×10^-10。比は約1.2748×10^-7。B1 S2 q2とのcorrected signalの距離も約3.4834×10^-10である。

これは当該prefix10/q2/R4/K2で、tailのfinite cutoffをさらに詰めることがtotal差の主因へ作用しないことを示す。Kを増やしたり、q・partitionを固定したままRを増やしてfinite tailを正確にしても、極限として残るouter-PFの差は消えない。

数学的には、同じouter-stepの中央に置いたexact tailのmicrostepは、同じH_Rの指数なので合成してそのouter-stepのexact tailになる。Rの変更だけでは、前後の決定論分割のqや順序は変わらない。有限Kの誤差相殺が偶然変わる場合はあり得るが、今回のtiny finite成分を主たる改善余地と見なす根拠はない。

**次の比較で優先する自由度はqとpartitionである。R/Kはnormalizationとfinite-tail許容を満たすために選ぶ。** これは当該条件からの研究設計判断であり、PR全条件でKが不要という主張ではない。

### 5.2 S2/S4：次数の効果は見えるが、漸近則の認証ではない

q1→q2のtotal差から得る有限窓の見かけ次数は、S2で約2.0644、S4で約3.8593。二次・四次らしい低下と整合するが、2点だけから漸近領域・大q・長時間・別状態への外挿を保証しない。

S4 q2の差約2.3674×10^-4は、B2 q2より小さい。ただし、回路費用とshot予算を合わせるまでは方式の資源winnerではない。

### 5.3 B0：qを詰めるだけではdiscardの差が残る

B0 prefix10/q2ではdiscard差約2.90034×10^-2、残した部分のPF差約2.58944×10^-3で、両者は複素量として一部相殺してtotal約2.64143×10^-2になる。

当該prefixを固定したままq→∞とすればexact truncated-H signalへ近づくので、discard差そのものを消すことにはならない。B0の全prefixを否定する結果ではないが、当該点でqだけを増やし続けることを優先しない。

### 5.4 B3：finite平均が近いことと、測定が安いことは別

B3ではouter-PF差約2.20807×10^-3、finite-RTE差約3.20371×10^-9。B2と同様、当該点のtotalはfinite cutoffよりouter分割に支配される。

一方、保存log B=20.213362629147213から、レビュー内算術で

\[
B\simeq6.005537049\times10^8,\qquad B^2\simeq3.606647525\times10^{17}
\]

となる。raw signalが10^-9程度でも、corrected signalはO(1)である。raw側の小さい絶対oracle差だけを見て「補正後も同じ桁で保証される」としない。現在はcorrected経路も別に照合している。

## 6. 保存wrapper費用：何が有望か

以下はprimary `symmetric_directional`。今回直接読んだ16記録では、同じreplicaのcosine/sineで列挙した6 metricsが一致した。一般の将来回路でも同じと仮定する根拠にはしない。

| 構成 / replica | RZ | CX | RZ depth | CX depth | total depth | 直接取得したwrapper |
|---|---:|---:|---:|---:|---:|---|
| B0 / S2 / prefix10 / q2 / rep0 | 39,847 | 18,188 | 5,567 | 7,846 | 16,009 | [02](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eca6bfdab85fa7c41cbe5dcdec770cb1cd95c9e2/artifacts/resource_applicability/track_a_h6_technical_pilot_v2/2026-10-10/launch_v1/wrapper_02.json) / [03](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eca6bfdab85fa7c41cbe5dcdec770cb1cd95c9e2/artifacts/resource_applicability/track_a_h6_technical_pilot_v2/2026-10-10/launch_v1/wrapper_03.json) |
| B1 / S2 / q1 / rep0 | 37,561 | 16,960 | 5,251 | 7,292 | 15,041 | [06](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eca6bfdab85fa7c41cbe5dcdec770cb1cd95c9e2/artifacts/resource_applicability/track_a_h6_technical_pilot_v2/2026-10-10/launch_v1/wrapper_06.json) / [07](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eca6bfdab85fa7c41cbe5dcdec770cb1cd95c9e2/artifacts/resource_applicability/track_a_h6_technical_pilot_v2/2026-10-10/launch_v1/wrapper_07.json) |
| B1 / S2 / q2 / rep0 | 74,297 | 33,660 | 10,387 | 14,496 | 29,817 | [10](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eca6bfdab85fa7c41cbe5dcdec770cb1cd95c9e2/artifacts/resource_applicability/track_a_h6_technical_pilot_v2/2026-10-10/launch_v1/wrapper_10.json) / [11](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eca6bfdab85fa7c41cbe5dcdec770cb1cd95c9e2/artifacts/resource_applicability/track_a_h6_technical_pilot_v2/2026-10-10/launch_v1/wrapper_11.json) |
| B1 / S4 / q1 / rep0 | 111,107 | 50,480 | 15,569 | 21,798 | 44,726 | [14](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eca6bfdab85fa7c41cbe5dcdec770cb1cd95c9e2/artifacts/resource_applicability/track_a_h6_technical_pilot_v2/2026-10-10/launch_v1/wrapper_14.json) / [15](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eca6bfdab85fa7c41cbe5dcdec770cb1cd95c9e2/artifacts/resource_applicability/track_a_h6_technical_pilot_v2/2026-10-10/launch_v1/wrapper_15.json) |
| B1 / S4 / q2 / rep0 | 221,341 | 100,636 | 31,015 | 43,472 | 89,149 | [18](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eca6bfdab85fa7c41cbe5dcdec770cb1cd95c9e2/artifacts/resource_applicability/track_a_h6_technical_pilot_v2/2026-10-10/launch_v1/wrapper_18.json) / [19](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eca6bfdab85fa7c41cbe5dcdec770cb1cd95c9e2/artifacts/resource_applicability/track_a_h6_technical_pilot_v2/2026-10-10/launch_v1/wrapper_19.json) |
| B2 / prefix10 / q2 / R4 / K2 / rep0 | 43,457 | 19,204 | 6,002 | 8,127 | 16,982 | [22](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eca6bfdab85fa7c41cbe5dcdec770cb1cd95c9e2/artifacts/resource_applicability/track_a_h6_technical_pilot_v2/2026-10-10/launch_v1/wrapper_22.json) / [23](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eca6bfdab85fa7c41cbe5dcdec770cb1cd95c9e2/artifacts/resource_applicability/track_a_h6_technical_pilot_v2/2026-10-10/launch_v1/wrapper_23.json) |
| B2 / prefix10 / q2 / R4 / K2 / rep1 | 43,301 | 19,156 | 5,990 | 8,117 | 16,958 | [26](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eca6bfdab85fa7c41cbe5dcdec770cb1cd95c9e2/artifacts/resource_applicability/track_a_h6_technical_pilot_v2/2026-10-10/launch_v1/wrapper_26.json) / [27](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eca6bfdab85fa7c41cbe5dcdec770cb1cd95c9e2/artifacts/resource_applicability/track_a_h6_technical_pilot_v2/2026-10-10/launch_v1/wrapper_27.json) |
| B3 / prefix0 / q2 / R4 / K6 / rep0 | 13,678 | 3,856 | 1,694 | 934 | 3,754 | [30](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eca6bfdab85fa7c41cbe5dcdec770cb1cd95c9e2/artifacts/resource_applicability/track_a_h6_technical_pilot_v2/2026-10-10/launch_v1/wrapper_30.json) / [31](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eca6bfdab85fa7c41cbe5dcdec770cb1cd95c9e2/artifacts/resource_applicability/track_a_h6_technical_pilot_v2/2026-10-10/launch_v1/wrapper_31.json) |


RZは`rz_count`というnative gate数。T count、physical runtime、state preparation込み総費用ではない。randomの2行/1行は個別cost trajectoryであり、母集団平均が既知という意味ではない。

### 6.1 B2はS2に対する実装上の利益を示す有望な点

B2の2標本RZは43,457と43,301、単純標本平均43,379。B1 S2 q2は74,297。比は0.5838594、すなわち**この標本平均では約41.61%小さい**。CX平均19,180もB1の33,660より小さく、深さ指標も小さい。

B2のB²は約1.000411524、同qのsignal差もほぼ同じなので、この点は「回路短縮が測定負担で全て失われた」例ではない。精度一致比較を続ける合理的な理由になる。

ただし、n=2から期待費用の41.61%改善や一般的PR優位を認定しない。各axis/controlで同じtrajectoryを共有しているので、4 wrapperを4個の独立標本とも数えない。

### 6.2 S4は強い対照だが、qの選び方も重要

S4 q2のRZ221,341はB2標本平均の約5.1025倍。同時にbiasは大きく小さいため、statistical marginによるshot減少がこの差を埋めるかが研究上の比較点になる。

保存点だけを見ると、S4 q1はS2 q2よりRe/Imの絶対biasも、今回の各native費用も大きい。同様にB0 prefix10/q2はS2 q1より両軸biasと各費用が大きい。この**固定point同士**の比較は、全S4 familyや全discard familyの排除ではない。

### 6.3 control改善を一律の係数にしない

直接確認したcosine側の例：

| 構成 | ordinary RZ | symmetric RZ | 削減率（レビュー内算術） |
|---|---:|---:|---:|
| B1 S2 q2 | 159,961 | 74,297 | 約53.55% |
| B1 S4 q2 | 479,441 | 221,341 | 約53.83% |
| B2 rep0 | 89,127 | 43,457 | 約51.24% |
| B2 rep1 | 88,971 | 43,301 | 約51.33% |
| B3 rep0 | 13,728 | 13,678 | 約0.364% |

B3では決定論backboneが小さく、RTE tailは同じordinary-controlled処理に残る。観測された小さい削減はその構造と整合するが、gate-category別の因果寄与まで今回計測したわけではない。

主検証は対称controlをprimaryとして維持し、ordinaryは限定paired感度とする。一方だけに既知のcontrol簡約を与えて比較しない。

## 7. 測定込み資源の評価式と、事後のnominal診断

### 7.1 原pilotのN/Gはnullのまま

原runはshot予算を計算せず、N/Gnull、accuracy UNDETERMINED、u未認定で止まっている。本節は**本レビューが追加した事後の条件付き算術**である。原runの認定結果へ混ぜない。[S01] / [S02]

以前の対称軸split・Hoeffding型の会計に沿うと、軸aについて

\[
h_{c,a}=\epsilon_{\rm sig}/\sqrt2-|\Delta z_{c,a}|-u_{c,a},\qquad
N_{c,a}=\left\lceil\frac{2B_c^2\log(2/\alpha_a)}{h_{c,a}^2}\right\rceil
\]

であり、h>0が必要になる。総費用は\(G_c=\sum_a N_{c,a}\mu_{C,c,a}\)。\(\mu_C\)は期待wrapper費用で、n=1/2のsample meanそのものではない。

同じ失敗確率、**仮にu=0**、整数ceilを無視し、random Cへ保存sample meanを代入した比較量として

\[
J_c(\epsilon;u=0)=B_c^2\sum_{a\in\{\Re,\Im\}}
\frac{\overline C_{c,a}}{(\epsilon/\sqrt2-|\Delta z_{c,a}|)^2}
\]

を計算した。共通の2log項を省いた**比較用proxy**であり、採用済みのshot budget、formal confidence、operational予測ではない。新しい実験結果でもない。

### 7.2 nominal適格境界

| 構成 | nominal対称軸splitの境界 √2 max(|ΔRe|, |ΔIm|) |
|---|---:|
| B0 / S2 / prefix10 / q2 | 0.0320581993664 |
| B1 / S2 / q1 | 0.0133102913135 |
| B1 / S2 / q2 | 0.00323189250472 |
| B1 / S4 / q1 | 0.00414170128088 |
| B1 / S4 / q2 | 0.000284975545478 |
| B2 / prefix10 / q2 / R4 / K2 | 0.00323189231957 |
| B3 / prefix0 / q2 / R4 / K6 | 0.00264447447664 |


これは対称軸配分の十分条件における、u=0の代数的境界。複素誤差normの最小達成限界でも、energy/chemical accuracyでもない。数値幅を正に戻せば境界は変わる。

### 7.3 固定7点・保存標本だけに対する比較量

各epsilonで最小の有限Jを1とした比を示す。「—」は仮のu=0でもpositive headroomにならないことを表す。

| ε_sig | B0 q2 | B1 S2 q1 | B1 S2 q2 | B1 S4 q1 | B1 S4 q2 | B2 q2/R4/K2 | B3 q2/R4/K6 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 4.282 | 1.293 | 1.712 | 2.64 | 4.618 | 1 | 1.1140e+17 |
| 0.01 | — | — | 1.712 | 3.211 | 2.845 | 1 | 9.9450e+16 |
| 0.005 | — | — | 1.712 | 9.003 | 1.036 | 1 | 7.0851e+16 |
| 0.001 | — | — | — | — | 1 | — | — |


**ここから得るべき研究判断はwinner認定ではなく、次の情報価値である。**

- epsilon=0.01ではB2の回路短縮が有望な比較点になる。
- epsilon=0.005ではB2とS4 q2のproxy差は約3.59%に縮まり、母平均・数値幅・q最適化を確認する価値がある。
- epsilon=0.001では、固定7点の中でpositive nominal headroomを持つのはS4 q2だけ。B2/PRというfamily全体の不適格ではない。
- B3の現点はB²負担が極端に大きい。n=1の費用推定を精密化することより、R/partitionによる負担の変化を先に見た方が情報価値が高い。

今回のepsilon群は既存計画のanchorを用いたレビュー内感度である。pilot時に同精度比較として事前登録されていた、とは言わない。探索後の仮説としてH6主検証を設計し、H8で固定後の独立確認を行う。

### 7.4 回路最安・bias最小・総資源最小は異なる

共通のstate preparation/readout費用をPとすると、\(G_c(P)=\sum_aN_{c,a}(C_{c,a}+P)\)。Pは共通でもNが違えば寄与は同じではない。P=0のnative結果とP感度を分ける。

また、例としてB=1、一軸、C=c q^a、bias=A q^-pという単純モデルなら、\(G\propto q^a/(e-bias)^2\)の内部stationary点は

\[
\frac{bias}{e}=\frac{a}{a+2p}
\]

になる。これは一般最適性の証明ではなく説明用モデルだが、「最初に適格になる最小q」を常に採用すべきではないことが分かる。qを少し増やして統計誤差の余地を増やす方が安い場合がある。実際のbias曲線・B・compiled costで検証する。

## 8. B3の巨大normalizationは単にHoeffdingが粗いからか

宣言されている同じcanonical RTE測定法で、各fresh whole-trajectoryのHadamard出力をX_a∈{−1,+1}、補正推定量をY_a=B X_aとすると、

\[
\operatorname{Var}(Y_a)=B^2-(\mathbb E Y_a)^2.
\]

これはX²=1からの恒等式である。意図したfinite平均との接続が成立し、独立試行として平均するという測定契約を前提にする。今回のcorrected平均はO(1)、B²は約3.6×10^17なので、この**現在の推定量の分散**もB²にほぼ等しい。

従って、現点の測定負担をHoeffdingの保守性だけで説明することはできない。ただし、これは任意のestimator・分散低減・別RTE law・Track Bの別法に対する情報理論的下界ではない。実際の量子測定を今回行ったという意味でもない。

Rを増やせばmicrostepのtau=λ_R T/Rを減らせる。登録されたfinite lawでは、Kが偶数で

\[
b_K(\tau)=\sum_{k=0,2,\ldots,K}\frac{|\tau|^k}{k!}
\sqrt{1+\frac{\tau^2}{(k+1)^2}},\qquad B=b_K(\tau)^R.
\]

K≥2、小tauならb=1+tau²+O(tau⁴)で、log B≈(λ_R T)²/R。この漸近式を現B3の大tauへ無条件に代入せず、実際のfinite式で安価に確認する。Rを増やすと回路費用も増えるため、normalizationだけで採択しない。[S10] / [S11]

## 9. 残り4 wrapperの扱い

### 9.1 第一推奨：主検証設計への必須条件から外す

保存済みには、全7構成のsignal、全7構成について少なくとも一つのcomplete paired cost group、B2の2標本、B3の1標本がある。今回残っているのは、まだ一度も接続できていない新しいmethodではなく、B3同cellの2本目の費用trajectoryである。[S03]

その2本目を得ると、登録pilotの技術的coverageは埋まるが、n=2でも母平均の精密評価にはならず、現在のB²負担も変わらない。よって、**研究として次に必要な判断のために、必ず4件を先に埋めるとはしない。**

この判断は旧36-wrapper契約の達成を認めるものではない。新しい作業計画に「旧pilotの部分証拠を採用、B3 replica1は未取得のまま主検証設計へ進む」と明記する。

### 9.2 補完する場合の正しい位置付け

再現性・回路生成のstress確認などの目的で補完すること自体は妥当である。その場合は、保存seed・cell・input・compiler/phase意味論に結合した**欠測分だけの新しい実行**にする。消費済みgrantのresume、32件の全面再compile、旧runのcounter書換えをしない。

worker terminalは失われた過去の実行終端なので、後から再生成して本物として埋められない。新しいpost-hoc監査を作ることと区別する。

### 9.3 欠測による統計上の注意

B3の1標本を、無条件にランダム欠測後の不偏推定としない。停止・compile所要時間とevent構造の関係は未確定で、後半の高費用eventが失われる可能性を一般には排除できない。保存済み1標本は個別費用として表示し、未保存trajectoryの代替値・分散推定を作らない。

## 10. 今後の時間上限なし方針と、計算の実現可能性

ユーザーの「今後は時間の上限をつけなくてよい」は、今後新規に準備・個別認可する計算でphase/total wall capを設けない方針として記録済みである。本レビューはこれを維持する。新しい任意の30分・60分capを再導入しない。[S13]

旧runの起動時capとSTOPは変更しない。新runner/契約でnullを正しく扱うようにする。ログ・進捗監視・明示中止手段・メモリ/出力/call/instruction上限・科学gate・mandatory STOPは別であり維持する。

### 10.1 7.09 GBは「8 GiBまでまだ安全」という意味ではない

保存最大RSSは7,092,899,840 bytes、約7.093 GB＝6.606 GiB。AS上限8 GiBは仮想address spaceの制限で、RSS上限や空きRAM予約ではない。両者を引き算して利用可能なメモリを断定しない。

今回の停止理由は時間であり、OOMを記録したrunではない。`ru_maxrss`は過去最大なので、大きい値が後続wrapperでも続くことだけからメモリleakやlive保持量を断定しない。

### 10.2 時間無制限でも、全候補のheavy compileは合理的でない

7 cellの保存cell wall合計は約252.78秒。ただしreference/probe/setup/全wrapper評価を含む総時間ではない。wrapper_cost phaseはsampling、回路build、control/statevector検査、hash、transpile、書出しを含むので、65分全体を「32回transpileだけの時間」としない。[S10] / [S11]

一方、sourceではordinary/symmetricのevolutionを同時に保持し、各replicaで複数probe・両axisを検査してから4 wrapperをcompileしている。主検証の全点でこの技術pilotと同じ重複を課す必要があるかをCodexが検討すべきである。

改善候補は、primary cost取得とcontrol感度検査の分離、不要な参照寿命短縮、cell/replica単位のprocess分離、同じ固定構造の再利用である。どれが実メモリ・時間の主因かはprofileで確認し、改善量を推測して保証しない。

今回cosine/sineのcountが同じだったことだけで、将来の一方のcompileを無条件に代用してはいけない。意味論と費用同値を検証し、新契約で許可した場合に限る。compilerの変更で得たcostを同じ実装の値と混ぜない。

## 11. 次の科学的検証：H6の精度一致資源比較

### 11.1 中心RQ

> 同じ採用DF Hamiltonian・指定state・finite-time coherent-signal taskの下で、PRの回路短縮は、強い決定論PFと比べて、どの要求精度・partition・outer-step条件で測定込み資源の利益として残るか。

H6の現7 cellは技術pilotであり、最適化比較ではない。その結果を受け、H6をdevelopmentとして実験設計を具体化する。

### 11.2 まず比較する自由度

**q。** B1 S2/S4とB2を、それぞれ同じ精度を満たすqで比較する。現在のq1/q2をanchorとし、q4/q8などの追加を含む有限の候補生成規則を結果前に定める。具体的な最大qは費用とメモリ実現性を踏まえてCodexが計画化する。ここでは全qの実行を認可しない。

**partition。** prefix10は一つの中間点で、最良prefixとは限らない。共通の少数prefixをB0/B2で用い、例えば生成順の四分位に対応するprefix5/10/15等を出発案にできる。これは新しい提案で、既存pilotへ追加済みの条件ではない。B0/PRで不公平なpartition集合にしない。

**R/K。** qとR=qrの整合、finite-law normalization、finite-RTE差を満たす有限候補を設定する。各prefixのλ_Rから計算可能なBを先に確認できる。低Bだけを目標にRを際限なく増やさない。qを変える際、R固定が不可能な整数条件も明示する。

**要求精度。** 既存のepsilon_sig={0.05,0.01,0.005,0.001}を中心に維持する。表示点を増やすより、0.005付近と0.001を満たす構成を実際に持つことが重要。0.0001や長時間を直ちに全直積へ追加しない。

### 11.3 全候補をいきなりcompileしない

推奨順序は、固定候補生成 → signal/bias/Bと数値検証 → 精度会計・候補coverage確認 → 必要なactual primary-wrapper費用 → 重要比較の確認標本、である。

ただし、予測costだけで都合のよい候補を選び、その部分集合から全method最適性を主張してはいけない。厳密な除外ができないheuristic shortlistingは探索的と明記し、何を測っていないかを残す。全候補集合と実際にcostを取得した集合を別々に管理する。

実装上のcap超過はscience不適格・真の高費用と同じではない。取得不能な候補を費用0や成功集合から無言で削除せず、technical unavailableとして記録する。

## 12. 数値幅、shot会計、費用統計の完成条件

### 12.1 厳密certificateの全面完成を一律前提にしない

経験的な数値検証に基づくresource studyは可能である。その場合は、指定target、参照構成、solver/stateの選び方、演算順序/精度比較、適用範囲を明記したempirical uを使用し、保証付き上界とは区別する。今の最大oracle差だけをそのまま普遍的uにすることは避ける。

数値誤差はepsilonだけでなくhとの比で見る。h≈0の候補は、小さい差がNを大きく変える。正負・境界が不確かならUNDETERMINEDを許し、winnerを強制しない。

### 12.2 費用のn=2を完成条件にしない

B2の2標本が近いことは一つの観測であって、rare eventを含む母平均の保証ではない。KとRによるfinite support、Taylor order・componentの確率、event長・basis遷移のばらつきを踏まえ、必要なら層化と残りtailの評価を使う。

全候補に機械的に大量標本を課す必要はない。探索用の費用標本と、主要比較を確認する独立標本を分け、精度・標本追加・停止の規則を結果前に固定する。優位が見えた候補だけ都合よく追加取得して決めない。

sourceのinstruction capは、評価した回路のguardである。全分布の最大compiled costがそのcap以下だという証明なしに、母平均tailの上界へ使わない。cap超過eventの確率をゼロにしてはいけない。

### 12.3 Quantum shotsとcost trajectoriesを分ける

costのpaired Re/Im/controlは同じtrajectoryに対する相関を持つ。独立量子shot用のfresh whole-trajectory lawとは別。固定された少数trajectoryを無限回測れば元の平均に対する同じshot保証が得られる、としない。

state preparation、T synthesis、routing/architecture等を含まないnative RZ/CX/depthの結果は、そのscopeで報告する。物理実行時間・fault-tolerant総資源・chemical-accuracy QPEを主張する段階は別の完成条件とする。

## 13. H4、H6、H8とRQ-P1の扱い

**H6はdevelopment。** 今回見たsignal/costを使って設計を変更するため、同じH6からの追加結果を独立held-outと呼ばない。

**H8はまだ先。** H6で比較の意味と有望/非有望領域を整理し、候補生成、DF/state policy、confidenceと費用統計、モデル、failure処理を凍結してから、Track Aの参照性能が未観測なH8条件へ進む。単に分子名がH8というだけで独立性を保証しない。既存exposureを台帳で確認する。

**H4のlegacy rank12とH6のtol-only rank19を、無条件に同一政策のsize scalingへ並べない。** サイズによる移送を主張する際には、旧H4と共通target生成政策を結ぶbridge、または相違を明示した限定比較が必要である。旧210候補の全面再計算を自動的に要求する意味ではない。

**RQ-P1は補助。** 旧FEWを今回の16 primary費用で再fitして移送成功と称さない。order/control/sizeに対応しないfeaturesは適用範囲外または外挿診断とする。H6の費用説明モデルを新しく作る場合もdevelopment fitであり、H8前にfreezeする。Operationalなbias/shot/総資源予測の完成をRQ-Rの必須条件にしない。

## 14. Track Bとの関係

Track Bで参照した、同じ有限平均に対してnormalization・sampling law・合成費用・十分shot予算を区別する考え方は参考になる。しかし、明示small-providerのnative TとTrack AのDF-native RZは同じ量ではない。Track Bのreturn法やG9の削減率をH6へ移用しない。

本レビューではTrack Bの新しい結果を再評価・採用せず、両研究の統合を実行条件にしない。将来接続する場合も、同じtarget mean、provider access、control phase、誤差と費用を再確認する別の研究判断が必要である。

## 15. 先行研究と、論文化の着地点

今回、一次資料の書誌・abstractを再確認した。PR原論文は既に、量子化学benchmarkに対する詳細なsingle-ancilla phase-estimation資源見積もりを行っている。arXiv 2503.05647はv2が2026-07-10、書誌上はPRX Quantum 7, 020332 (2026)となっている。[L1]

対称Trotterizationのcontrolled任意角回転削減も既知であり、本研究の新しいcontrol原理とはしない。[L2] また、利用者らの高次PF研究は、形式次数だけで資源効率を決められず、四次formulaの選択自体が結果を変える例を扱っている。ただし、そのenergy-error/QPEの結論を今回のfinite-signal taskへ直接転用しない。[L3]

本研究の候補貢献は、**指定DF-native実装において、直接のsignal/bias、finite-RTE normalization、actual wrapper、数値・統計の範囲をそろえたとき、どの単純化が資源量や方式選択を変えるかを説明すること**である。

現状はまだ独立論文の十分性・新規性を確定する段階ではない。H6で同精度mapを作り、主要な境界と理由を確認し、未使用条件で説明を検証できれば、単なる実装記録を超える成果を目指せる。

比較対象がS2とglobal Yoshida S4に限られるなら、その限定を明示する。「高次PF一般よりPRが最良」と主張する場合には、別の有力formulaも同じtaskで比較する必要がある。今直ちに全8次・10次等を追加することは求めない。

否定的結果でも、比較が公平で、制約の原因と適用範囲を明らかにできれば意味がある。PRが勝つまで候補を追加し続けることを完成条件にしない。

## 16. 代替進路の比較

| 進路 | 情報価値・問題点 | 判断 |
|---|---|---|
| 同じ36 wrapperを最初から再実行 | 完了済み7cell/32wrapperの重複が多い | 採用しない |
| B3残り4件を必ず先行 | 技術coverageは埋まるが資源の中心問題は解けない | 一律必須にしない |
| Kを増やすだけ | 現B2/B3ではtiny finite成分を削るだけになりやすい | 第一案にしない |
| B2のq/partitionと決定論を精度一致比較 | 回路短縮とstatistical marginのどちらが支配するかを判断できる | **優先する** |
| B3のRを安価なnormalization会計から検討 | family全体を誤って排除しない | 限定診断として残す |
| 直ちにH8へ拡大 | H6で未解決の比較設計とheavy cost取得を拡大する | 今は採用しない |
| FEW追加fitを主目的化 | n/条件が乏しく、RQ-Rの中心問いに直接答えない | 補助に留める |
| Track Bの新lawを直ちに取り込む | task/provider/費用の対応が未確定 | 採用しない |
| 全数値を厳密認証するまで停止 | 現在の経験的resource studyの完成条件を過剰に広げる | 一律前提にしない |

## 17. Codexへ渡す次の作業単位

**H6の精度一致資源比較・実行設計をまとめて準備する。** 細かなmodule、tests、logging、cache、process分割はCodexが判断してよい。

1. 旧STOPとpartial範囲を保全し、本レビューの採用範囲・B3補完非必須という決定を新契約へ記録する。
2. 同じ採用H6 target/state/Tを維持した比較集合、q/partition/R/Kの有限な生成規則、epsilon anchor、強い決定論対照と共通指標を固定する。
3. signal/Bの先行評価と、actual primary-wrapper費用取得を分離し、主張する集合・未評価集合・技術失敗を管理する。
4. empirical uと保証付きu、shot法則、費用標本の追加・確認規則を明記する。N/Gを計算する新しい解析scopeを旧pilotと区別する。
5. future wallなしを新runnerへ反映し、メモリ/出力/call/instruction制限・progress・明示中止手段・failure監査を維持する。
6. primary controlを維持し、ordinaryと全probeの重複を必要な診断へ限定できるか技術的に検証する。実装やcompilerを変える場合は旧costとのidentityと比較可能性を記録する。
7. source/plan/input/envを固定・commit/push・remote確認し、具体的な追加計算の明示認可へ進む。今回のレビュー承認を新しいlaunch認可として流用しない。

**次の主要なGPTレビュー**は、H6の精度一致資源mapと主要比較の確認結果が揃い、H8で何を独立検証するかを決める時点。通常の実装修正やsynthetic追加だけのために、同じ科学方針の全面レビューを繰り返さない。

## 18. GO / STOPと研究の終了条件

### 今回GOとする範囲

H6主検証の研究設計・準備。7cell/32wrapperの部分証拠を利用すること。B3残り4件の取得を主検証設計の必須前提から外すこと。時間上限なしという既存ユーザー方針の新runner反映。

### 今回認可していない範囲

新しい科学runnerの即時launch、候補無制限追加、入力DF/state/Tの変更、H8参照性能への接触、Track B統合、旧結果の再分類、消費済みgrant再利用。

### 前倒しでGPTへ戻す条件

新しい数値矛盾、target/phase/estimatorの意味論変更、比較を不公平にする除外、想定外の情報漏洩、必要なbaselineを外す方針変更、今回合意した範囲を超える研究対象変更。

### 主研究を閉じる判断

PR有利領域が出ることだけを継続条件にしない。公平な比較集合・十分なcoverage・不確かさの評価で、利益が残る条件または失われる条件を説明できれば成果としてまとめる。逆に、同じ限定点の工程完了だけが増え、RQ-Rへの新しい情報が得られない場合は規模拡張を止め、限定resource studyとして整理する。

## 19. 主張できること／できないこと

| 主張 | 状態 |
|---|---|
| 採用H6 DF/stateで7 cellのstate-action経路が独立binary64 oracleと整合 | 登録probe・signal範囲で支持 |
| 全7方式について少なくとも1 complete paired cost groupを取得した | 保存証拠で支持 |
| B2の2標本RZ平均が同qのS2より約41.61%小さい | 保存値の算術として支持 |
| B2が期待総資源で41.61%一般的に優れる | 未支持 |
| S4 q2だけが固定7点中epsilon=0.001のnominal対称軸余裕を持つ | u=0の事後診断に限定 |
| S4が全PR構成より最良 | 未支持 |
| 現B3のcanonical補正量は大きなB²を持つ | 保存logBと推定量恒等式から支持 |
| B3というfamily全体を排除できる | 未支持 |
| H6からH8へ傾向が移送する | 未検証 |
| chemical-accuracy QPEや物理runtimeの改善 | 対象外・未支持 |
| 原pilotが36/36完了した | 不成立。STOPを維持 |

## 20. 最終判断

**本研究は、入力・正しさの接続を確かめる段階から、H6における精度一致の資源比較へ重心を移すべき段階に来た。**

B2には、二次PFとほぼ同じsignal差を、少ないnative RZ/CXで実現できる有望な点がある。四次PFには、小さいbiasにより統計予算を確保できる強みがある。これらの競合がepsilonに依存することを、保存値の事後診断が示唆している。

次に価値が高いのは、B3の不利な同じ点の2本目を必ず埋めることではなく、**q・partition・normalizationと測定予算をそろえた、限定的だが説明可能な資源mapを得ること**である。

レビューは完了。次の担当はCodex。研究方針はRQ-R主軸のまま、新しい計算は具体化された契約と明示認可で進める。

---

## 付録A. 証拠リンクと確認範囲

| ID | 資料 | 固定ref |
|---|---|---|
| S01 | [H6技術pilot結果・証拠索引][S01] | `eca6bfdab85fa7c41cbe5dcdec770cb1cd95c9e2` |
| S02 | [保存結果要約][S02] | `eca6bfdab85fa7c41cbe5dcdec770cb1cd95c9e2` |
| S03 | [保存partial task/event metadata監査][S03] | `eca6bfdab85fa7c41cbe5dcdec770cb1cd95c9e2` |
| S04 | [保存run bytes監査][S04] | `eca6bfdab85fa7c41cbe5dcdec770cb1cd95c9e2` |
| S05 | [実行後source/input/environment監査][S05] | `eca6bfdab85fa7c41cbe5dcdec770cb1cd95c9e2` |
| S06 | [親terminal][S06] | `eca6bfdab85fa7c41cbe5dcdec770cb1cd95c9e2` |
| S07 | [全raw hash・欠測inventory][S07] | `eca6bfdab85fa7c41cbe5dcdec770cb1cd95c9e2` |
| S08 | [H6実行準備v2][S08] | `eca6bfdab85fa7c41cbe5dcdec770cb1cd95c9e2` |
| S09 | [実行中のH6 setup port v2][S09] | `0b04886869efb9d08b07d6517300da2bc0123f4a` |
| S10 | [継承元H6 signal/oracle/cost port v1][S10] | `0b04886869efb9d08b07d6517300da2bc0123f4a` |
| S11 | [継承元sampling/control/wrapper cost][S11] | `0b04886869efb9d08b07d6517300da2bc0123f4a` |
| S12 | [保存DF受理・state/snapshot完成結果][S12] | `eca6bfdab85fa7c41cbe5dcdec770cb1cd95c9e2` |
| S13 | [今後のwall-time方針][S13] | `eca6bfdab85fa7c41cbe5dcdec770cb1cd95c9e2` |
| S14 | [参照計算・入力identity][S14] | `eca6bfdab85fa7c41cbe5dcdec770cb1cd95c9e2` |
| S15 | [全primitive probe記録][S15] | `eca6bfdab85fa7c41cbe5dcdec770cb1cd95c9e2` |
| S16 | [固定pilot manifest][S16] | `eca6bfdab85fa7c41cbe5dcdec770cb1cd95c9e2` |
| S17 | [source freeze][S17] | `eca6bfdab85fa7c41cbe5dcdec770cb1cd95c9e2` |


[S04] / [S05] / [S06] / [S07] / [S14] / [S15] / [S16] / [S17]は索引・既出取得記録から追跡できる証拠。今回の中心確認はS01–S03、S08–S13、個別wrapperと保存scalarであり、リンクを掲げた全rawを同じ深さで独立再認証したという意味ではない。

primary16 wrapperのリンクは§6の表にある。ordinaryの直接取得例は
[08](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eca6bfdab85fa7c41cbe5dcdec770cb1cd95c9e2/artifacts/resource_applicability/track_a_h6_technical_pilot_v2/2026-10-10/launch_v1/wrapper_08.json)、[16](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eca6bfdab85fa7c41cbe5dcdec770cb1cd95c9e2/artifacts/resource_applicability/track_a_h6_technical_pilot_v2/2026-10-10/launch_v1/wrapper_16.json)、[20](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eca6bfdab85fa7c41cbe5dcdec770cb1cd95c9e2/artifacts/resource_applicability/track_a_h6_technical_pilot_v2/2026-10-10/launch_v1/wrapper_20.json)、[24](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eca6bfdab85fa7c41cbe5dcdec770cb1cd95c9e2/artifacts/resource_applicability/track_a_h6_technical_pilot_v2/2026-10-10/launch_v1/wrapper_24.json)、[28](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eca6bfdab85fa7c41cbe5dcdec770cb1cd95c9e2/artifacts/resource_applicability/track_a_h6_technical_pilot_v2/2026-10-10/launch_v1/wrapper_28.json)。

## 付録B. 外部一次資料

- [L1] Günther et al., *Phase estimation with partially randomized time evolution*, arXiv:2503.05647v2。今回は公式arXivの書誌・abstractを再確認。詳細QPE見積もりが存在することを根拠とし、44ページの全数値を今回再監査したとはしない。
- [L2] Simon and Love, *Halving the Cost of Controlled Time-Evolution*, arXiv:2511.13855v1。対称Trotterizationに対する既知のcontrolled compilation改善。
- [L3] Abe et al., *Evaluating higher-order product formulae for molecular ground-state energy estimation*, arXiv:2605.30967v1。energy/QPE taskの研究であり、本pilotと同じtaskの結果ではない。

文献の新規性比較はこの範囲に限る。今回の部分pilotから独立論文のpriority・一般最適性を認定しない。

## 付録C. 算術の再現と限界

補助bundleに`analyze_saved_scalars.py`と`saved_scalar_review_calculations.json`を含む。前者は固定資料から転記した7行のsigned biasとprimary costを明示する。後者はB/B²、finite-window apparent order、cost ratio、nominal J表を含む。

Python標準libraryだけで実行できる。インターネットアクセス、リポジトリimport、分子input、quantum libraryを必要としない。原データのbytesを独立に複製したという主張はなく、取得したJSON fieldの明示的転記である。

再現されるのは本レビューの算術であり、元H6実行の再現や精度certificateではない。

[S01]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eca6bfdab85fa7c41cbe5dcdec770cb1cd95c9e2/docs/research/track_a_h6_technical_pilot_result_v2.md "H6技術pilot結果・証拠索引"
[S02]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eca6bfdab85fa7c41cbe5dcdec770cb1cd95c9e2/artifacts/resource_applicability/track_a_h6_technical_pilot_execution_v2/2026-10-10/result_summary_v2.json "保存結果要約"
[S03]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eca6bfdab85fa7c41cbe5dcdec770cb1cd95c9e2/artifacts/resource_applicability/track_a_h6_technical_pilot_execution_v2/2026-10-10/saved_partial_bytes_audit_v2.json "保存partial task/event metadata監査"
[S04]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eca6bfdab85fa7c41cbe5dcdec770cb1cd95c9e2/artifacts/resource_applicability/track_a_h6_technical_pilot_execution_v2/2026-10-10/saved_run_audit_v2.json "保存run bytes監査"
[S05]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eca6bfdab85fa7c41cbe5dcdec770cb1cd95c9e2/artifacts/resource_applicability/track_a_h6_technical_pilot_execution_v2/2026-10-10/execution_audit_v2.json "実行後source/input/environment監査"
[S06]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eca6bfdab85fa7c41cbe5dcdec770cb1cd95c9e2/artifacts/resource_applicability/track_a_h6_technical_pilot_v2/2026-10-10/launch_v1/terminal_status.json "親terminal"
[S07]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eca6bfdab85fa7c41cbe5dcdec770cb1cd95c9e2/artifacts/resource_applicability/track_a_h6_technical_pilot_execution_v2/2026-10-10/execution_evidence_inventory_v2.json "全raw hash・欠測inventory"
[S08]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eca6bfdab85fa7c41cbe5dcdec770cb1cd95c9e2/docs/research/track_a_h6_technical_pilot_preparation_v2.md "H6実行準備v2"
[S09]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0b04886869efb9d08b07d6517300da2bc0123f4a/src/trottertracks/resource_applicability/ax2b_h6_pilot_port_v2.py "実行中のH6 setup port v2"
[S10]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0b04886869efb9d08b07d6517300da2bc0123f4a/src/trottertracks/resource_applicability/ax2b_h6_pilot_port_v1.py "継承元H6 signal/oracle/cost port v1"
[S11]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0b04886869efb9d08b07d6517300da2bc0123f4a/src/trottertracks/resource_applicability/ax2b_molecular_ports_v3.py "継承元sampling/control/wrapper cost"
[S12]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eca6bfdab85fa7c41cbe5dcdec770cb1cd95c9e2/docs/research/track_a_h6_saved_df_completion_parallel_result_v2.md "保存DF受理・state/snapshot完成結果"
[S13]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eca6bfdab85fa7c41cbe5dcdec770cb1cd95c9e2/artifacts/resource_applicability/track_a_h6_technical_pilot_execution_v2/2026-10-10/future_wall_time_policy_v1.json "今後のwall-time方針"
[S14]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eca6bfdab85fa7c41cbe5dcdec770cb1cd95c9e2/artifacts/resource_applicability/track_a_h6_technical_pilot_v2/2026-10-10/launch_v1/input_reference.json "参照計算・入力identity"
[S15]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eca6bfdab85fa7c41cbe5dcdec770cb1cd95c9e2/artifacts/resource_applicability/track_a_h6_technical_pilot_v2/2026-10-10/launch_v1/primitive_validation.json "全primitive probe記録"
[S16]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eca6bfdab85fa7c41cbe5dcdec770cb1cd95c9e2/artifacts/resource_applicability/track_a_h6_technical_pilot_v2/2026-10-10/launch_v1/frozen_pilot.json "固定pilot manifest"
[S17]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eca6bfdab85fa7c41cbe5dcdec770cb1cd95c9e2/artifacts/resource_applicability/track_a_h6_technical_pilot_preparation_v2/2026-10-10/source_freeze_v2.json "source freeze"
[L1]: https://arxiv.org/abs/2503.05647v2
[L2]: https://arxiv.org/abs/2511.13855v1
[L3]: https://arxiv.org/abs/2605.30967v1
