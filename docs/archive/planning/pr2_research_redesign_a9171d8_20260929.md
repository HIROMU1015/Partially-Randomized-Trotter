# PR-2研究方針の再設計：部分ランダム化を使う条件を、同じ精度の資源比較で明らかにする

作成日：2026-09-29  
基準repository：HIROMU1015/Partially-Randomized-Trotter  
基準branch：pr2-v4-s2-parallelization-20260928  
基準commit：`a9171d8bea93441afc9b17c7ffeab79af8dbcc95`  
位置付け：S2結果を踏まえた研究設計の提案。repositoryへの書込み、追加simulation、compile、量子shotは実施していない。旧事前登録・旧結果・旧statusは変更しない。

## 0. 結論

**PR-2を、rank 6の優位性を示す試験から、「残差を捨てる／決定論で保持する／ランダムに補完する選択を、同じ信号精度と実装費用で比較する研究」へ再設計する。**

仮題：**二重因子化Hamiltonianにおける部分ランダム化の資源損益分岐――残差廃棄、近似精度、状態準備費を含む比較**。

新しいprefix法、一般最適化アルゴリズム、新しいfinite-RTE定理の発明は目的としない。新規性候補は、既知の部分ランダム化を具体的な制御回路・有限RTE・同一信号タスクへ接続したとき、どの設計上の判断が変わるかという実証・資源評価上の知見である。

次の論文の中心を「rank 3が勝つ」に置かない。rank 3は有力な探索的証拠だが、強いdiscard baseline、反復数を調整した決定論baseline、random端点の探索境界が未確認である。これを埋めずにheld-outでrank 3の成功を探すと、再び弱い対照に対する改善を研究成果と取り違える。

進行は、**既存S2の原因分解 → 同精度比較の不足を一巡で埋める → 固定した設計の確認用geometryで検証 → 原稿へ収束**とする。correctnessの小gateごとに主題を再選定しない。

## 1. 現在の結果をどう解釈するか

### 1.1 確認したS2の範囲

固定H4 1.00 Å、STO-3G、rank-12 DF target、T=0.8、delta=0.1、q=8。評価量は、状態準備を除く測定付きHadamard wrapperのcompiled RZと、補正付き信号の解析的shot数の積である。[R1,R2]

| 候補 | shots | mean RZ/shot | no-prep RZ work | 証拠の役割 |
|---|---:|---:|---:|---|
| B0：rank 6 discard | 40,641 | 82,230 | 3.342e9 | primary baseline |
| B1：rank 12 deterministic | 14,073 | 159,046 | 2.238e9 | primary baseline |
| B2：rank 6、r=1,K=2 | 14,073 | 約87,897 | 1.237e9 | primary partial |
| B3：L_D=0、r=32,K=4 | 36,032 | 約32,722 | 1.179e9 | one-bodyを残すtwo-body-random端点 |
| B2：rank 3、r=1,K=2 | 14,858 | 約46,324 | 6.883e8 | fixed-setting control、探索的証拠 |

rank 9のfixed-setting controlのno-prep workは約1.792e9である。[R1] B3は全Hamiltonianのfull-randomではない。one-bodyは決定論側に残る。[R2]

### 1.2 rank 3の低コストを説明する、現在分かっていること

保存済みのrank 3 recordとrank 6 reportからの算術では、rank 3／rank 6の比は次である。

- mean RZ/shot：約0.527。
- shot数：約1.0558。
- no-prep work：約0.5564。

rank 3のlambda_Rは約0.5833、rank 6の約0.02208より大きい。rank 3のattenuationは約0.9732、normalization multiplierは約1.0275で、shot数増加はこの比較では約5.6%に留まる。rank 3のcorrected signal biasは約1.068e-4、finite truncation biasは約1.58e-7であり、登録されたaccuracy条件を満たしている。[R1,R3]

**現時点の説明は「1 shotの回路削減がshot増加を上回った」である。** deterministic部分とtail部分それぞれの厳密な削減寄与、basis融合の寄与まで個別に実測したという意味ではない。full compileは非加法的になり得るため、部品costを足した値とfull-wrapper costを同一視しない。

### 1.3 10% materialityと統計的不確実性を混同しない

rank 6 B2／B3のengineering ratio intervalは約[1.0354,1.0632]である。現行のintervalの範囲ではB3の方が安い。両者が10% material frontierに残るのは、差が事前の実質差基準より小さいからであり、両者の点順位が分からないからではない。[R1]

またmean±2SEはformal confidence intervalではない。10%は研究上の採用目安であり、統計的有意差・数学的同値性・新規性の定義ではない。

### 1.4 pairwiseの状態準備損益分岐も、全候補の最適切替ではない

既存のrank 6対B3では、共通prep cost Pを加えるとpoint crossingが約2,638 RZ相当/shotにある。[R1]

しかし、現在のrank 3 controlは、選択済みB3よりno-prep workもshot数も小さい。その二つの固定設定だけの点推定では、P>=0の全域でrank 3がB3より安い。rank 3対rank 6のpairwise crossingは、保存済み値から約6.99e5 RZ相当/shotとなる。

これはposthocの算術であり、rank 3を旧primary winnerへ昇格するものではない。**約2,638という交点だけを「研究全体で最適法が切り替わる条件」として中心結果にしてはいけない。** 他の設定・rank・discardを加えると、最安候補はさらに変わり得る。

## 2. 今回の重要な修正：splitだけでなく「同じ精度での設定」を比べる

### 2.1 固定q=8の比較は正しいが、最安資源の比較ではない

S2では全方式のqを8に固定した。[R2] これは同じ時間刻み・同じouter-S2の比較として有効であり、旧結果を撤回する必要はない。

ただしB1のcorrected complex biasは約1.07e-4、目標complex errorは0.05である。[R1] 反復数を減らしても精度を満たす可能性がある。実際にどこまで減らせるかは未計算で、漸近q^-2だけを使って可否を断定しない。

**資源的な採用条件を論じる次段では、同じ物理時間Tで、決定論法にも部分ランダム法にもqを選ぶ機会を与える。** qを変更するときはdelta=T/qとし、物理時間を変えない。

### 2.2 discardはrank 6だけでは足りない

S2のprimary B0はrank 6である。[R2] 新しい問いは「残差を捨てるか、補完するか」なので、rank 3/6/9それぞれのdiscardを同じtargetへ比較する必要がある。

rank 9の残差が小さいとしても、signalとPF biasと回路費用を評価する前に勝敗は決めない。旧pilotの別snapshotのenergy差を、新系列のsignal受理に転用しない。

### 2.3 B3は探索上限にある

選択されたB3はr=32,K=4で、登録gridの上端である。[R1,R2] 「登録されたgrid内の選択」と「random端点全体の最適値」を区別する。

境界確認は全候補へ公平な規則で一段だけ許す。B3だけ弱い上限に閉じ込めてpartial優位を主張しない。一方、全r/Kを無限に探索することもしない。後述の一段拡張後も境界未解決なら、その範囲に結論を限定する。

## 3. 主RQ・副RQ・目標主張

### 主RQ

> 固定DF Hamiltonianのcoherent signalを同じ精度で推定するとき、残差廃棄、決定論的保持、ランダム補完のどれを選ぶべきか。決定論側へ置くfragment数と、outer反復・finite-RTE精度を合わせて選ぶと、その採用領域はどう変わるか。

### 副RQ A：機構

> 1-shot circuit workの削減が、normalizationによるshot増加、PF・finite-cutoff biasによる統計予算の減少を上回る条件は何か。lambda_Rや1-step costだけによる判断は、どこまで役に立つか。

### 副RQ B：条件付き移送

> developmentで選んだ設定と、費用を支配する項の説明は、事前固定された同一H4 familyの別geometryにも成立するか。共通の状態準備費を足すと、固定設定の採用領域はどう変わるか。

独立した研究候補を三件並走させるのではなく、主RQに答えるための機構説明と確認として使う。

### 完成時に目指す主張

> 二重因子化Hamiltonianの部分ランダム化について、残差を捨てる対照、反復数を調整した決定論対照、random-dominant端点と同じ信号タスクで比較した。中間splitが有効になる、または不要になる条件を、1-shot費用と必要shotの交換関係として説明し、固定した確認条件でその適用範囲を検査した。

これは達成済みの結論ではない。特定のrank、特定の改善率、partialの勝利を約束しない。

## 4. 何が新規性・価値の候補か

| 一次研究 | 既知のこと | 本研究で候補となる追加知見 |
|---|---|---|
| Günther et al. [W1] | partial PFとsingle-ancilla phase estimationの詳細資源評価。状態準備と回路反復数の関係も議論済み | finite-RTE・DF-native full-wrapperの固定実装において、discard、q選択、実測compiled costをそろえた比較が何を変えるか |
| Hagan–Wiebe [W2] | Trotter/qDRIFT合成、partition、error/cost trade-off | 合成を新発明とせず、目的signal・制御回路・有限設定をそろえた利用条件を調べる |
| SPRINT/GRADE [W3] | factorization、近可積分PF、残差のqDRIFT/RTE等による処理、誤差対費用での選択 | その一般的発想の再提示ではなく、明示した比較条件・不足しやすい費用項の影響を定量化する |
| RC-DF [W4] | 圧縮DFと資源低減 | 現在は非prefix RC-DFを実装していない。prefix切断を新しい圧縮法とは称さない |

「Pを足すと順位が変わる」「rとrankを最適化した」「full wrapperを数えた」だけでは独立した方法論の新規性は弱い。候補となる貢献は、これらを公平に接続したことで**既存の簡略評価から得られる設計判断が、どこでどう変わるか**という具体的な知見である。

逆に、新しい一般定理や漸近次数を必須にしない。既知理論の利用、実装、再現可能な資源比較で非自明な適用条件を示す研究も候補になる。

現段階では完全な新規性監査・投稿可能性を確定していない。最も近い文献の同じtask・input・cost scopeの比較を、原稿のrelated-work表で一度完成させる。タイトル検索だけで「既存にない」と判定しない。

## 5. 共通のestimandと目的関数

### 5.1 維持するtarget

\[
z_H(T)=\langle\psi_{12}|e^{-iH_{12}T}|\psi_{12}\rangle.
\]

H12は保存済みrank-12 DF Hamiltonianを表し、12原子水素鎖を表す記号ではない。以降の文書では混乱を避けH_refと呼んでもよい。targetは元分子の厳密Hamiltonianではなく、この固定DF表現である。

初期状態は同じ保存済みsector state。厳密な基底状態を参照として用いるので、oracle-informedなHamiltonian simulation subroutineのbenchmarkである。未知energyをclassicalに知っていることを利用して、時間発展を単一位相gateへ置換しない。全QPE・状態準備・真の量子優位を示したとは言わない。

### 5.2 設計変数

候補j=(mode,k,q,r,K)。modeはdiscard / deterministic / partial。kはdeterministic DF prefix長。qはouter repetition、delta=T/q。rは各tail occurrenceのRTE細分、Kはpaired cutoff。

partialのL_D=0端点でもone-bodyと定数位相を保持する。これを全Hamiltonianのfull-randomとは呼ばない。

### 5.3 正しい平均とshotモデルを維持する

raw meanをmu_j、全normalization multiplierをB_jとし、補正後meanはnu_j=B_j mu_j。

\[
 b_{j,a}=|\nu_{j,a}-z_{H,a}|,
 \quad s_{j,a}=\epsilon_a-b_{j,a},
 \quad
 N_{j,a}=\left\lceil\frac{2B_j^2}{s_{j,a}^2}
 \log\frac{2}{\alpha_a}\right\rceil .
\]

s<=0ならその候補は不適格。数値biasの誤差上限が判明しているときはsを安全側に減らす。証明できないoracle推定値はbenchmark数値と明記する。

主設定T=0.8、epsilon_C=0.05、epsilon_axis=0.05/sqrt(2)、alpha_axis=0.025を維持する。これはHoeffdingで設計した十分shotであり、必要shotの情報理論的最小値や実際に量子測定した成功確率ではない。

### 5.4 目的関数

\[
 G_j(P)=\sum_{a\in\{\mathrm{Re},\mathrm{Im}\}}
 N_{j,a}\{\mathbb E[C_{j,a}]+P\}
 =A_j+N_jP.
\]

primaryはP=0。secondaryはP>=0の固定設定間の下包絡線とbreak-even。Pは同じstateを同じ手順で再準備する一回の費用であり、Pを変えてoverlapやstate品質まで変えるモデルではない。

Pの有利な値を結果後に選び「勝った」としない。交点は全て表示し、実装可能なPの実測値がない間はhardware上の結論を出さない。prep回路とevolution回路を共同最適化した費用でもない。

### 5.5 分解する量

同じ比較j/kについて

\[
\log\frac{G_j(0)}{G_k(0)}
=\log\frac{N_j}{N_k}
+\log\frac{\overline C_j^{\mathrm{shot\ weighted}}}
{\overline C_k^{\mathrm{shot\ weighted}}}.
\]

さらにnormalization、biasによるstatistical allowance、backbone/tail/basis-change/ancillaの診断を分ける。後半の回路部品はfull compilationで相互に簡約され得るため、部品別の和を厳密なfull-cost分解と呼ばず、同条件ablation/診断と表記する。

## 6. 公平なbaselineと範囲

| 系列 | 必須の比較 | 注意 |
|---|---|---|
| discard | k=3,6,9それぞれ | 全て同じH_refのsignal誤差を使う。energy差で代用しない |
| full deterministic | rank 12 S2 | 同じTでqを選べること。旧q8一点だけで最安決定論を代表させない |
| partial | k=3,6,9 | G/Wは同一なので二重計上しない。同じq/r/K選択機会 |
| random-dominant | k=0、one-body deterministic | 同じfinite precision grid。r上限による不利を隠さない |

qの新しい基本候補は{1,2,4,8}、Tは一定。これは精度要求のsweepではなく、同じ精度を満たすための実装設定の比較である。追加する場合は今回のS2後改訂として明記し、旧結果を遡及的に変更しない。

rの基本grid {1,2,4,8,16,32}、K={2,4}は維持。選択または費用下界上の競争候補がr=32にある全方式へ、r=64の一段確認を同じ規則で与える。これを超える自動拡張はしない。K=4やq=8の境界も記録し、許した集合内の結論と集合外の最適性を区別する。

高次deterministic PFは、S2のmatched second-order実装比較では必須でない。一方「既存の有力deterministic PFより有利」と主張する段階では、既存コードで検証済みの代表的高次PFを少なくとも一つ、同じ精度・scopeで比較する必要がある。大きな高次PF探索はしない。初期主張をS2 class内に限定する選択も明示的に許す。

## 7. 実行計画：新しいテーマ選別pilotを増やさず、本研究の不足を埋める

### M0：既存S2から説明と不足表を作る（read-only）

新signal・compileなし。既存28 signalとcost recordから、全候補のcorrected/raw mean、bias、B、shots、cost、engineering interval、境界を一つの表にする。

rank 3/9は旧S2ではcontrolのまま、新研究のdevelopment情報へ利用する。旧primaryの再判定はしない。既存q8/rKの値と新計算を合流させる際も、元artifact、source、seed、役割を保持する。

成果物は「現在の証拠」「何が未比較か」「各不足で結論がどう変わるか」の表。再度のhash回復・新しいcanonicalization研究は行わない。

### M1：同精度の比較を一巡で完成させる（結果前に凍結）

**目的**：rank 3の見かけの利益が、過剰に細かいdeterministic qや弱いdiscard対照に依存していないかを検査する。

1. 保存済みdevelopment snapshotのみを使用。新規SCF/DFは不要。
2. discard k=3/6/9とfull deterministicについてq={1,2,4,8}のsignal/biasを評価。既存一致cellは再利用。
3. partial k=3/6/9とk=0について同じqと既定r/Kのmean・normalization・shotを先に評価。
4. 候補を精度不適格・適格へ分ける。compileが高価なため、全候補へ128 trajectoryを最初から与えない。
5. 事前固定したcost-screening規則で直接compileする候補を選ぶ。代理指標で捨てた候補を「実測で支配された」とは記載しない。
6. 境界r64は上記の同一規則に従って一段だけ検査。下降が続く場合は境界未解決を残し、global optimumを主張しない。
7. 新しいcostによる順位だけでなく、q・discard・normalization・shot・basis実装のどれで旧判断が変わったかを記録。

**規模の目安**：基本集合はrandom 4 split ×4 q ×6 r ×2 K =192 mean設定、deterministic/discard16設定。既存28設定はidentity条件が一致する場合再利用できるため、基本集合で最大180の追加mean設定。これは180個のMonte Carlo compile試験を意味しない。条件・精度・Hamiltonianを増やす探索ではなく、固定した比較問題内の候補評価である。

**compile制限案**：直接compileするrandom cellを最初に16以下へ制限する。方法ごとの同じ規則でscreening上位候補と別proxyが選ぶchallengerを残す。初回32、結論を変え得るcellのみ追加96、計128まで。既存cellの不確かさが十分なら再compileしない。

この16という数は科学的定数ではなく計算上限の提案。最終preregistrationで、選出規則・prioritization・上限時の扱いを一意にする。安全なcost下界がない場合、未compile候補を除いた集合での比較に限定するか、method選択の不確かさを報告する。都合のよい候補だけを見て「全grid最適」としない。

**必ず一度止める地点**：M1完了時。ただし通常のcorrectness通過や一つのrankが不利だったたびにRQを再定義しない。ここで「公平な対照にも残る知見」「完成に必要な確認」「縮小すべき主張」を判断する。

### M2：設計を固定してからheld-outを一度だけ開く

現時点では未実行・未承認。M1後に以下を固定する。

- 主に問うのはdevelopmentで選んだ具体的な設定のtransfer。
- P=0のpartial代表、必要ならP下包絡線を形成する別partial代表、best discard、full deterministic、random-dominant端点を固定する。
- 各代表のrank、q、r、K、compiler、accuracy、shot計算式を固定。
- 全baselineを同じ入力へ適用し、held-outで設定をretuneしない。
- held-outでは同じ規則から必要shotを再計算する。旧developmentのshot数をそのまま移すことではない。
- transfer失敗時もそのまま報告。新しい設定をheld-outで選び直した結果を、未使用評価の成功へ読み替えない。

これは「凍結した構成の移送」であり、held-outで各methodを最適化した最安資源比較ではない。将来、固定された選択アルゴリズムを各geometryに適用する別の研究を行う場合は、その目的とoracle情報を別契約にする。今は両者を混ぜない。

### M3：原稿と図へ収束する

M1/M2が主張に必要な証拠を満たせば、H12・別分子・長RPEを自動追加しない。追加するのは、主張を無効にし得る具体的な対照・反例への確認だけ。

## 8. 統計・資源モデルで固定する注意点

1. 旧S2のmean±2SEはengineering intervalのまま保持。新campaignでも同じ表記を守り、adaptive samplingや候補選択をした後の95%保証とは呼ばない。
2. 10% materialityを維持して旧結果と比較するが、小さい差を0差とはせず、統計の不確かさとも区別する。
3. 決定論costは指定compilerの固定出力。random costはtrajectory expectationの推定。量子shotとcost較正trajectoryは別のsample。
4. Re/Imで共通trajectoryを使うとcost推定は相関する。全workのSEを出すならその相関を扱うか、axis区間を保守的に合成する。
5. 結論を左右する選択済みrandom候補だけ、必要なら独立なstreamで確認する。選択時の最小値に都合のよいノイズが含まれる可能性を記録する。
6. runtime/FT resourceを主張しない限り、compiled RZをその単位のまま報告。CX、depth、sizeは保存済みsecondaryで整合を確認する。RZはすべて同じT費用ではない。
7. compiler tolerance、極小係数、identity phase、basis fusionの規約を全候補で共通にする。少数qubitで一般制御合成が支配していないかは、既存命令・cost breakdownから先に確認する。
8. 不確かな結果はuncertainであり、scientific negativeへ自動変換しない。implementation failureはさらに別。

## 9. 新しい定理を要求しない、機構説明の具体形

RTEのsampling overheadがB_j^2として現れること、状態準備を毎shot負担すること、partitionにcost trade-offがあることは既知である。[W1,W2]

本研究で調べるのは、それらの既知因子のどれがこの実装・taskで設計を変えるかである。例えば、小さい局所時間ではlog B_K(tau)=O(tau^2)なので、均一q,rのnormalization増加は概念的にlambda_R^2 T^2/(qr)のscaleを持つ。これは診断用のleading見積りであり、finite-Kの正確なBやsourceの定義を置き換えるcertificateではない。

対照的に、決定論側のworkはおおむねqと保持fragmentに増えるが、full compilationのbasis融合があるため単純線形を普遍則とはしない。

その二つを並べることで、次を結果として説明できる可能性がある。

- tail normを小さくし過ぎてbackbone費用を払い過ぎる領域。
- 少ないbackboneでよいが、randomization/shot overheadが支配する領域。
- residual discardだけで精度が十分な領域。
- 固定qの利得が、全候補のq調整で消える領域。
- no-prepとprep込みで採用候補が異なる領域。

単なる「rank 3が最小」という表より、以上のどこを新しく定量化できるかを評価する。

## 10. 着地点と完了条件

### 最小の着地点：限定した実証・資源study

次がそろうことを完成の判断基準とする。

- 同じtarget・state・T・accuracyで、discard／deterministic／partial／random端点を比較。
- 主要な比較器に同じ実装精度の選択機会がある。
- 主張する改善・失敗を1-shot workとshot要因へ分けて説明。
- strong baseline、境界、oracle情報、compiler scopeの限界を記録。
- 凍結した代表設定の確認条件で、成立または破綻を報告。
- 近接研究との差を「具体的に何が新しく分かったか」で説明。

二つのH4 geometryから一般分子・大系・化学精度の全RPEへ結論を広げない。この範囲では、まず再現可能なcase study／technical resource studyとして閉じる。投稿可能な独立論文に十分かは、残った知見の非自明さと先行研究との差に依存し、現時点で保証しない。

### 目標の着地点：使い分けに使える説明と確認

既存の簡略評価では判断を誤る、または不明だった採用領域を、具体的な実装費用と誤差配分から説明できること。さらに固定確認条件へ移るか、その破綻がどの仮定によるかを明らかにすること。

新しい一般最適性定理、全PFへの勝利、必ず10%以上の改善は共通必須条件ではない。

### 科学的negativeとして残す条件

公平な対照と十分な対象範囲の下で、partialの利得が消える理由が定量的に説明できれば成果候補になる。ただし一つの小系で勝たなかっただけでは一般no-goや独立論文の価値を保証しない。新しい説明が既存評価の直接的再現に留まるなら、technical noteへ縮小して完了する。

## 11. 結果に応じた分岐

| 結果 | 扱い |
|---|---|
| q調整・discard対照を戻しても中間partialが有力 | その機構と適用条件を主成果として、固定transferへ進む |
| q調整だけで旧利得が消える | 固定q比較と最安資源比較の違いを原因として整理。partial全般の否定にしない |
| rank 9等のdiscardが強い | residualを補う必要がある条件／ない条件を整理。discardを後から除外しない |
| random端点の境界が結論を左右する | 一段確認後も不明ならfinite-grid限定・boundary未解決。勝利を作るための無制限拡張はしない |
| prep費を足したときだけ候補が変わる | 条件付き感度結果。Pの実現性なしに実機での優位を主張しない |
| 全体に差が小さい／不確か | 科学的negativeと混同しない。追加精密化が主張を変える場合だけ限定追加、そうでなければ限定結論で閉じる |
| held-outで移らない | 旧developmentを取り消さず、固定設定のtransfer失敗として報告。held-outでretuneしない |

新しい条件ごとに研究題目を付け替えない。主RQは一貫して「部分ランダム化をいつ使うか」であり、勝敗はその答えの一部である。

## 12. 論文の図・章の構成

中心図は次の四種類で十分。

1. **比較設計図**：discard／deterministic／partial／random端点、共通estimand、含める費用。
2. **developmentの同精度比較**：固定q8とq調整後を並べ、許した候補集合・未評価境界を明記。
3. **費用差の説明**：1-shot work、B^2、bias/statistical allowance、shot数。P下包絡線は同図の別panelまたは別図にできる。
4. **固定設定の確認**：1.30 Åへ移した結果と、成立・失敗理由。

原稿は、問題設定 → 既知手法とscope → matched-task評価方法 → 結果と機構 → 確認条件 → 限界、で構成する。

全historyのgate一覧やテスト数は補足へ回す。手続きの厳密さは成果の信頼性を支えるが、論文の中心内容ではない。

## 13. 直近のCodex作業

まずM0と新campaignの比較契約・計算計画を作成する。新契約をcommitし、予算・候補選出規則・停止点を固定してからM1を実装する。

旧S2は`S2_TRANSFER_CANDIDATE_AWAITING_REVIEW`のまま保持し、rank 3 controlの遡及的primary昇格、held-outの先行開封、旧pilotと新snapshotの混合はしない。

今回GPT側で行ったのは、S2文書とauthorization、artifactの関連recordの読解、一次文献の重点再確認、保存値からの比・交点の算術、研究計画書作成である。artifact全体のhash再計算・独立再実行はしていない。GitHub原本のプログラム用一括取得はネットワーク制限でできなかったため、connectorから必要な範囲を直接確認した。新しい科学計算を実行したとは扱わない。

## 14. 出典

### 固定repository一次資料

- [R1: S2結果文書](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a9171d8bea93441afc9b17c7ffeab79af8dbcc95/docs/pr2_v4_s2_development_validation.md)
- [R2: authorization v5](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a9171d8bea93441afc9b17c7ffeab79af8dbcc95/docs/research/pr2_v4_s2_development_authorization_v5.md)
- [R3: S2 artifact](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a9171d8bea93441afc9b17c7ffeab79af8dbcc95/artifacts/pr2_v4_s2_development/2026-09-29/pr2_s2_development_resource_result_parallel_v1.json)：B2_candidates、deterministic、rank3_rank9_controls、task等の関連recordを確認。
- [R4: 研究概要・現状](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a9171d8bea93441afc9b17c7ffeab79af8dbcc95/docs/research/研究概要・現状.md)
- [R5: PROJECT_MAP](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a9171d8bea93441afc9b17c7ffeab79af8dbcc95/PROJECT_MAP.md)

### 公開一次文献と今回の確認範囲

- W1. J. Günther et al., *Phase estimation with partially randomized time evolution*, PRX Quantum 7, 020332 (2026), arXiv:2503.05647。公開PDF v2の導入、Sec. IV–Vへの接続、状態準備と反復費用の議論を確認。https://arxiv.org/abs/2503.05647
- W2. M. Hagan and N. Wiebe, *Composite Quantum Simulations*, Quantum 7, 1181 (2023)。出版ページの構成・partition・既知貢献を確認。今回は全文定理監査ではない。https://quantum-journal.org/papers/q-2023-11-14-1181/
- W3. P. A. M. Casares et al., *Theory and practice of Trotter product formulas for quantum chemistry*, arXiv:2606.30741v1。Fig.1を画像確認し、Sec. IV Step 7のremainder戦略・cost/errorの記載を確認。https://arxiv.org/abs/2606.30741
- W4. O. Oumarou et al., *Accelerating Quantum Computations of Chemistry Through Regularized Compressed Double Factorization*, Quantum 8, 1371 (2024)。出版ページでcompressed DFの既知の方法・資源目的を確認。https://quantum-journal.org/papers/q-2024-06-13-1371/

この重点調査は同一研究の不存在証明ではない。主張の最終版では、同じestimand、候補選択、費用scopeとの比較を原稿に明記する。
