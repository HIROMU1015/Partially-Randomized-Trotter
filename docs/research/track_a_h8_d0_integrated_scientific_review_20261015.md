# Track A：H8・D0統合科学レビューと追加検証方針

- 作成日：2026-10-15（JST）。
- 開始承認：利用者の「レビューを開始して」。
- 対象：`HIROMU1015/Partially-Randomized-Trotter`。
- レビュー状態：**完了。研究方針・追加検証の科学的採否を以下に記録する。**
- 実行状態：**新しい実験・D0 one-shot・grant・サーバー操作の実行承認は含まない。**
- 利用者の現方針：原稿化は後回し。研究ノートで引き継ぐ。追加geometryは1.00 Å固定、系サイズはH4〜H8、H9以上は対象に加えない。
- 直前に利用者が見送った「H6 runnerと汎用quota receiptの追加準備」は、今回のレビューを理由に自動再開しない。
- 既存判断：R01〜R09を引き継ぎ、特にR09のH4限定benchmarkの採用を取り消さない。本レビューはH8完了・D0完了後の追加判断である。[S1]

## 0. 結論と採否

**研究は継続する。目的は、固定DF実装におけるPRの資源低減が、強い決定論的対照、要求精度、時間、系サイズ、費用指標を変えてもどこに残るかを調べることとする。準備・監査・厳密compiler上界の作成そのものへ主目的を移さない。**

|対象|本レビューの科学的判断|実行・結果についての限定|
|---|---|---|
|既存H8・D0結果|固定モデル・候補・compiler・標本・経験的allowanceに条件付けた資源benchmarkとして採用|母平均winner、実機実測、最終energy費用ではない|
|H8 D0|**採択。直近の次作業とする**|修正SOURCEに契約を再bindingし、別途実行承認後に旧値一致→5精度・6指標・P解析。H8のε=.0005結果はまだ未知|
|H6/H8 native B0+I|**強化対照として採択**|H6の3構成6軸で実接続を確認。native全体の比較には各系12構成24軸を基本完成範囲とする。pilot後の拡張は別実行承認|
|H5/H7|**中性・開殻の追加対象として採択。H5を先行し、H7はH5の正しさ・実資源を確認後**|ROHF共通空間軌道、DF sector stateを使う。charged tripletの既定helperは使わない。各40 compile案はcorrectness/feasibility用で、全面資源比較完了ではない|
|T依存性|**H6の限定比較を採択。主比較はδ=.2固定、T=.4/.8/1.6、q=2/4/8**|旧固定q2案からの明示的変更。新Tの48 compileという静的数は同じでも、PR occurrence数は128→320。旧予算・grantを流用しない|
|より厳しいε|`.0005`を保存値評価domainに採用|`.0001`やq16を自動追加しない。未適格・q上限は結果として残す|
|H4のDF政策全面再統一|**現時点では必須にしない**|H4 legacyは別case。H5/H6/H7/H8の新比較政策を揃える。全H4〜H8で統制されたscaling則とは主張しない|
|希少次数|**全support証明・一律sampling増量は現時点で要求しない**|未観測層の費用は未保証。強化対照後の費用差と主張目標に照らし、必要時のみ条件付きpilotを選ぶ|
|追加監査・汎用quota機能|独立した次の研究工程としては見送る|採択された実行に不可欠な接続だけを、実行指示・承認のある段階でまとめて行う|

以上は科学的な選択であり、新しい科学計算を起動する包括承認ではない。実装の修正、入力受領、資源予算の具体化はCodexが担当するが、既存のmandatory STOP・one-shot・retry制限は維持する。

## 1. 資料取得、確認範囲、未実施事項

### 1.1 固定資料

本レビュー開始後にGitHub接続から以下を実取得した。

- 最新独立監査branchのHEAD：`40a4f55464dba80ab2192d777f7d69d2f551676f`。remote HEAD一致を確認。
- 修正SOURCE：`20f89bc32b161ee22e4a1ed709f9e3414c19633a`のH6 bridge該当部分、及び契約再binding索引。
- H8結果：`681ec23c93fdd1090d325908f4b155a79115afdf`の結果報告、4精度主要CSV、完了監査、希少次数JSONの該当部分、確認CSVのschema・一部行。
- D0結果：`8dbdb68293c988a999645e21a82b53966f9ef6cd`の結果報告と30行のprecision summary。
- 追加検証設計：`2ccc79502f043b4885e2337ffa98ff45f1875ae4`の開殻、DF政策、T、対照、指標の該当部分。
- 最終技術handoff：`2ec66aae5e17c9329fcea8ad962be27130980d31`。これは修正前実装に対応する履歴であり、現SOURCEとして使わない。
- 科学レビュー判断索引：`4675730e7e40d25ba3618334a4a0e28b4bd2f145`。
- H6希少次数の既存解析：`eee1343f6e99aa7661892ddfbe1067e974e2b74a`の層分解・費用・限界。

添付された2026-10-12共通指示と2026-10-14 H4科学レビュー原本も参照した。現レビューで必要な主要結果・監査・数理規約は取得できた。サーバー固有のraw NPZや全runtimeを取得していないことは、下記の限定を伴う。[S1–S12]

### 1.2 本レビューで行ったこと

資料にある数値と定義を読み、比較の公平性、統計的意味、追加実験の情報価値を評価した。コードは科学的意味に関係するH6 scalar/allowance経路と独立監査の所見を重点確認した。

付属の標準ライブラリのみの小さな検算では、公開されたH8の4行について`N × mean RZ = G`、費用比のshot比×回路費用比分解、既報のP固定pair交点を確認した。H8の公開2法則について、whole-trajectory層の確率と仮定付き逆転目安を計算した。これらはレビュー中の代表スカラー検算であり、H8 D0の新精度map、H4/H6 D0全体、sampler、compilerを実行したものではない。

### 1.3 実施していないこと

75人工テストの再実行、全wrapperの再compile、全rawの独立hash監査、新signal、SCF/DF/state、sampling、H8 ε=.0005の実値算出、サーバーのlive状態確認は行っていない。入力原本の全受領や実native接続について、Codexの人工PASSだけから成功を認定しない。

原稿・図のレイアウトや画像の目視レビューは今回の対象外。外部一次文献はPR研究のabstract・書誌、SPRINTの導入と方法の位置付け、PySCF公式SCF/spin資料を限定確認した。網羅的な新規性・優先権監査ではない。

## 2. 研究全体の到達点と主題

初期H4のmatched-accuracy資源比較、geometry展開、B0+Iによる対照強化、H6独立確認、H8独立確認、D0の精度・指標・P拡張が揃った。R01のRQ-R主・RQ-P1補助という位置付けを維持し、未知分子で参照誤差を知らずに候補を選べるRQ-P2を達成済みとはしない。[S1]

主対象は、保存されたDF Hamiltonianとstateに対する有限時間複素signalである。

\[
z(T)=\langle\psi|e^{-iH_{\rm DF}T}|\psi\rangle.
\]

ここから、候補のbias、finite-RTE normalization、解析的shot数、実compile済みのfull-wrapper費用を接続する。主費用は

\[
\widehat G_{j,m}=N_{j,c}\overline C_{j,c,m}+N_{j,s}\overline C_{j,s,m},
\]

であり、`m=RZ`を主指標、CXや各depth等を補助とする。ハットは有限標本による回路平均を含む点推定という意味である。[S2,S5]

今回の成果は「PRの回路が小さい」という一つの観測より進んでいる。近似誤差がshot負担を増やすこと、軽い回路がその負担を補うこと、対照の定数項だけで優劣が変わり得ること、指標・状態準備費用により候補が切り替わることを、保存結果で追跡できるようになった。

ただし、その分解式自体は恒等式であり、新しいアルゴリズムや物理法則ではない。研究価値は、具体的な回路と公平な対照でこの競合を定量化し、どの結論が比較条件に依存するかを明らかにする点にある。

## 3. H8結果の採用範囲

### 3.1 比較の分母を二種類に分ける

H8結果はlinear H8/1.00 Å/STO-3G/DF28/T=.8、native symmetric-directional、固定Qiskit条件、経験的u、P=0である。探索92候補、PR探索72×8、確認union6×32を分離している。[S2,S4]

|ε_sig|PR構成（共通p7/r4/S2）|full PF|PR/full PF|PR/B0|full PF比低減|B0比低減|
|---:|---|---|---:|---:|---:|---:|
|.05|q2/K2|q1/S2|.45708|.60522|54.29%|39.48%|
|.01|q4/K4|q4/S2|.35630|.47282|64.37%|52.72%|
|.005|q4/K4|q4/S2|.35616|.47266|64.38%|52.73%|
|.001|q8/K4|q4/S4|.34241|.43879|65.76%|56.12%|

上表は公開CSVの表示丸め。4点とも保存B0最良はfull PF最良より低い。したがって、**現在取得済みのB0/B1を合わせた決定論集合との比較では39.48–56.12%が該当し、54.29–65.76%はfull PF限定の比較**である。[S3]

どちらも有効な結果だが、後者を「最良の決定論的方法すべてに対する削減」と書かない。native B0+Iはまだ未測定であり、その追加後に全体分母を更新する。

### 3.2 高精度での利益は主に回路費用がshot増を補ったもの

ε=.001では、PRのshot数は60,710,094、full PFは36,390,063。PRは約1.668倍のshotを必要とする一方、平均RZ/shotは253,005.8125対1,232,704、約0.2052倍である。[S3]

\[
\frac{\widehat G_{\rm PR}}{G_{\rm PF}}
=\frac{N_{\rm PR}}{N_{\rm PF}}
 \frac{C_{{\rm PR},\mathrm{eff}}}{C_{{\rm PF},\mathrm{eff}}}
\simeq1.6683\times0.2052=0.3424.
\]

一方ε=.05の同じ比較では、PRはshot数比約.738、回路費用比約.620で両要因が有利。**PRは常にshotを減らす方法でも、常にshotを増やして回路を軽くするだけの方法でもない。** 方法・精度ごとの分解を維持する。

### 3.3 独立32本が確認したもの

32 trajectoryは、新しいseedによる平均回路費用の確認である。signalは固定ensemble計算に基づく。32回の実機測定、32個の独立signal精度証明、全候補の母平均順位の保証ではない。[S2]

確認対象union6の事前固定は、探索標本だけで自己評価する問題を緩和する。しかし確認32本の中から再び点最良を選ぶため、最終表示の選択後効果は残る。K2/K4の細かな順位を認定しない。

研究ノートには、事前に選んだ候補を固定した確認値と、確認後の点最小を区別して残す。これを理由に全候補を新たな32本で再取得することは要求しない。既存H6には、この区別を保存値から評価した先例があるが、その数値をH8へ移さない。[S11]

## 4. D0結果の意味：高精度と費用指標

### 4.1 ε=.0005を評価domainへ採用する

D0は旧precision1,088件・旧費用map2,659件を完全一致させた後に、新εと6指標・Pを再会計した。新signalやtrajectoryを追加したものではない。[S5]

|系列|ε=.001のPR/full PF|ε=.0005のPR/full PF|ε=.0005のPR点最良|
|---|---:|---:|---|
|H4 native fixed16|.51051|.42966|p3/q8/r2/K2|
|H6探索92|.38781|.49737|p5/q8/r2/K4|
|H6確認union8|.38701|.49633|p5/q8/r2/K2|

H4とH6で精度を厳しくした時の比の変化は同じではない。H6確認ではPR費用が約6.34倍、full PFが約4.94倍に増えている。これは公開2行の比の検算であり、新しい全domain解析ではない。[S6]

**「高精度ほどPRが有利」という一般則は採用しない。** 現在のq≤8領域でε=.0005でも利益が残る、という記述を採用する。PRの点最良がq8端にあるため、q16以上を含む最適性は分からない。rやKを増やすだけでPF biasが消えるとも考えない。

ε=.0001やq16の一律追加は見送る。まずH8でも同じ5精度の保存値評価を行い、適格性・費用・domain端の状態を揃える。

### 4.2 独立確認のcoverageはmetricやPに依存する

H6探索のε=.0005ではCX depthの点最良p5/q8/r4/K2が旧union8外になる。またPのlower envelopeにも同候補が中間で現れる。[S5]

これは実装異常ではなく、**RZとP=0を基準に選んだ確認集合が、別指標・別Pの最良候補を必ず含むとは限らない**ことの具体例である。

今はRZを主指標のまま維持する。探索8本に基づく別metric最良は探索結果として表示し、確認32本の最良と混ぜない。確認外候補をRZ主結論に不要なまま一律に追加確認することは要求しない。後にCX depth又はP込み境界を主要claimにする場合は、その候補を事前固定した追加確認を設計する。

### 4.3 P感度を正式な補助結果として採用する

\[
G_j(P)=G_j(0)+N_{{\rm total},j}P,\quad P\geq0.
\]

同じstate preparationでもshot数が違えば、準備費用は相殺されない。Pは共通1-shot overheadの仮想モデルであり、状態準備回路の実測ではない。[S2,S5]

H6探索ε=.0005のRZでは、P≈58,354でPR r4/K2へ移り、P≈179,714でfull S4 q4へ移る。P=0のPR/PF固定pair交点≈172,574とは異なる。旧確認union8ではこの中間PR候補を評価できず、そのenvelopeは別集合の結果である。[S5]

H8 ε=.001の固定pair交点約1,212,917 RZ/shotも、全候補再選択後の切替点とは限らない。H8 D0でそれを区別する情報価値がある。P_RZ、P_CX、P_depthの単位を混ぜず、depth-shotを実機時間・T-depth・shot並列時のcritical pathへ換算しない。

## 5. H8 D0：直近の実施対象として採択

H8にも旧4精度、6指標、Pの保存情報はあるが、H4/H6と同じ新ε=.0005・再選択envelopeの解析は未実施。必要6入力6,518,976 bytesとadapter/driverが準備されている。[S9]

**H8 D0を科学的に採択する。新しい物理モデルの導入ではなく、既存の固定モデル・標本について評価軸を揃える作業である。**

別途実行承認後の一単位は、修正SOURCE `20f89bc32b161ee22e4a1ed709f9e3414c19633a`を用い、探索92と確認union6＋det20の両scopeについて旧944 precision行・3,576費用行を先に完全一致確認し、その後5精度×factor1/10×6指標・paired SE・Pを計算するものとする。[S9,S10]

H4/H6 D0は再実行しない。新signal/trajectory/compileも不要。旧値不一致なら止まり、既存の許容差を後から広げて通さない。H8に異なるresume SOURCEのpaired axesがあれば、監査F01の正しい拒否を回避せず、来歴を調査する。[S8]

本レビューはH8 ε=.0005でPRが有利になることを認定しない。不適格・費用逆転・未確認候補の選抜も有効な結果である。

現在の資源案AS512MiB/RSS256MiB/wall300秒/CPU240秒/output32MiBは未承認STOP ceiling。必要量・完走時間の予測ではない。契約には修正前SOURCE/hashが残るため、候補binding索引に従い新しい固定契約へ結合する。変更のない解析規則について科学レビューをもう一巡させない。[S9,S10]

## 6. Native B0+I：採択の根拠と測定範囲

### 6.1 なぜ必要か

H4では既知scalar保持だけで比較結果が変わった。native H6/H8のB0はdiscarded tailのidentityを保持していないため、同じ改善余地がある。ただしH4の費用差0をnativeへ移植してはいけない。[S1,S7]

追加対照は真のenergyへfitしたものではない。元Hamiltonianの既知係数から決まる定数項を保持する。非identity tailを捨てたままの安価な決定論対照と比べて、PRによる補完がどこで有効かを調べる。

### 6.2 数学的定義を採択する

spin-orbital上のHermitian行列Gに対しA=dΓ(G)、fragment=λA²とする。full-Fockのnormalized traceは

\[
2^{-m}\operatorname{Tr}_{\cal F}A^2
=\frac14\{(\operatorname{tr}G)^2+\operatorname{tr}(G^2)\}.
\]

Gを対角化した基底ではA=Σ_i g_i n_iであり、full-Fockの一様traceではE[n_i]=1/2、E[n_i n_j]=1/4 (i≠j)。これを展開すると上式になる。固有基底への変換でtraceは不変である。

したがって、採用する規約は

\[
C_{\rm tail}=\sum_{\ell>p}\frac{\lambda_\ell}{4}
\{(\operatorname{tr}G_\ell)^2+\operatorname{tr}(G_\ell^2)\}.
\]

ここでGは既存native providerが渡すspin-orbital行列である。空間軌道行列のままspinの重複を落として同じ式に代入してはならない。signed λを絶対値へ変えない。固定電子数sectorのtraceや最適fit scalarへ変更しない。[S8,S12]

元B0の近似発展をU0とすれば

\[
U_I(T)=e^{-iTC_{\rm tail}}U_0(T),\qquad
z_I=e^{-iTC_{\rm tail}}z_0.
\]

full-H referenceは回転しない。Hadamard testではこの位相は制御枝間の相対位相であり、無制御回路全体のglobal phaseとして捨てられない。retained block、核、one-bodyのscalarは二重加算しない。これらの規約は採用可能と判断する。[S8,S12]

### 6.3 H6 pilotと全面比較の役割を分ける

直近のH6実測候補は既存案どおりp5/q1、p10/q2、p15/q8、S2/T=.8、両軸計6 compileとする。これは接続・phase・scalar・実費用の検査であり、native B0+I family全体の最小値を決める実験ではない。

H6/H8について対照を含む全面資源比較を述べるなら、各系3prefix×4q×2軸=24軸を基本的な完成範囲とする。H6のpilot6を再利用できれば追加18、H8は24が標準案。これは科学的な測定範囲の方針であって、pilotからの自動拡張許可ではない。[S7]

適格性を独立に確認して費用取得が不要と判定できる候補は、根拠を残して省ける。ただし「勝ちそうにない」「旧B0が不適格」だけでB0+Iも省略しない。identity保持でbiasが変わるためである。

matched旧B0が存在しない場合、Δmetricはnullとしてよい。目的がB0+I自体とPR/full PFの費用比較なら、Δを埋めるだけの旧B0再compileを自動要求しない。同一環境でのmatched deltaを研究成果として主張する必要がある時だけ、最大6軸の別scopeを明示承認する。

B0は消さない。強化後の決定論分母は、同じH/state/control/compilerと候補domainで取得したB0、B0+I、full S2、full S4の最小値である。別inputの結果や未取得費用を混ぜない。

## 7. 数値allowanceの科学的評価

### 7.1 真の保証と経験的diagnosticを分ける

独立監査75件PASSは制御・静的/人工数学の証拠であり、経験的uが真の誤差を上から抑える証明ではない。監査で直した6件も、将来の実行経路の保護不具合であって、既存H8/D0の数値結果が汚染されたと示したものではない。[S8]

bridgeは保存B0 signalとfull-H referenceを分け、signal側の軸誤差を回転し、reference側は回転せず加える。θ=-TCとすると、真の成分誤差を仮にuで抑えられる場合、形式上

\[
u'_{R}\le|\cos\theta|u^{(0)}_R+|\sin\theta|u^{(0)}_I+u^{(\rm ref)}_R+u^{(\rm phase)}_R,
\]

という三角不等式型の伝播になる。ところが元のuが経験的なら、伝播結果も経験的なままである。[S12]

**差の誤差だけを抑えた量から、二つの被減数それぞれの真の誤差を抑えたとは一般には言えない。** 共通の誤差が差で相殺する場合がある。本bridgeは旧uを各diagnostic contributionの保守的proxyとして重複使用すると明示している。これを本研究では経験的診断政策として採用できるが、厳密上界とはしない。

### 7.2 保存binary64モデルと元化学Hamiltonian

`stored_model_coefficient_error_upper=0`は、保存binary64係数を対象モデルそのものと定義することに対応する。係数が元化学Hamiltonianへ完全一致する、DF誤差がゼロ、SCF誤差がゼロという主張ではない。

Fractionによるidentityのexact算術とfloat算術との差を求める処理は、保存係数モデルの算術誤差の評価である。libm phase回転に使う`64ε`とfloorのレシピは、別の経験的項として扱う。この区別を維持する限り、pilot用の数値政策として採用する。[S12]

### 7.3 実行で確認する最小項目

新しい実験を際限なく追加せず、承認されたpilot内でscalar符号・相対phase・full reference不回転・実native経路を確認する。uとheadroomの比、factor1/10の感度、proxyが候補の適格性を支配していないかを記録する。差がuの選び方に依存して変わるなら、精度の物理的結論を保留する。

h=d-u>0かつu/h≤.01なら、同じBを固定しceilを除いた形式上のshot比はN(u)/N(0)=(1+u/h)²≤1.0201である。これは**採用したuの影響が小さいことの計算上の指標**であり、真の誤差がu以下という証明ではない。B_upperのmargin効果と整数ceilは別に確認する。

## 8. 希少次数：追加samplingを今すぐ必須にしない理由

### 8.1 未観測なのは4次だけではない

H8確認のq4/r4/K4は512 eventすべて0次、q8/r4/K4は1024 eventすべて0次。したがって2次を含む非ゼロ層の費用も当該確認標本では未観測である。[S2,S13]

保存lawのp0,p2,p4とR=qrに対し、層をs0=全0、s2=4なし・2あり、s4=4ありと分けると

\[
\pi_0=p_0^R,\quad
\pi_2=(p_0+p_2)^R-p_0^R,\quad
\pi_4=1-(p_0+p_2)^R.
\]

これは既存H6の層分解をH8の公開確率に適用したレビュー算術であり、分布・seed・費用標本を変更しない。[S11,S13]

|H8候補|π2|π4|非ゼロtrajectory確率|32本全てs0となる確率|
|---|---:|---:|---:|---:|
|p7/q4/r4/K4|約.0346709|約1.29936e-5|約3.4684%|約32.32%|
|p7/q8/r4/K4|約.0175284|約1.62949e-6|約1.7530%|約56.78%|

公開p2,p4の十進値からp0=1-p2-p4を置いた近似算術。元samplerのbit単位replayではない。上表から全0観測が不自然とは言えないが、未観測費用が無視できるとも言えない。

なお次数0でもcomponent/basisはランダムである。「全0次だったからB0と同じ回路を測った」という解釈はしない。[S11]

### 8.2 仮定付き逆転目安

同一候補・精度でD=NcCc+NsCsとし、s0平均を保存点推定G0へ固定する。非ゼロ層の平均をμ+とすれば

\[
E[D]=(1-\pi_+)G_0+\pi_+\mu_+.
\]

決定論費用GDと等しくなる平均の比は

\[
\frac{\mu_+}{G_0}=1+\frac{G_D/G_0-1}{\pi_+}.
\]

保存B0を分母とすると、q4/K4・ε=.005で約33.17倍、q8/K4・ε=.001で約73.96倍となる。full PFに対しては約53.12倍、110.55倍。ただし**s0平均を真値扱いした仮定付き感度**であり、未知component、s0推定誤差、compiler最大値を覆う保証ではない。B0+I追加後のGDは未取得で、この閾値をそのまま使えない。[S3,S13]

この結果から「希少層は絶対に無害」と結論しない。優先順位として、まず安価な強化対照を測り、その後に残る差をどの強さで主張するかを判断する方が合理的である。

### 8.3 今回の採否

全supportのcompiled RZ/CX上界、汎用compiler証明、一律32→128/1024の通常sampling増量は必須にしない。部分のinstruction上界、実行size cap、観測最大値を母平均上界へ読み替えない。[S11]

追加診断が必要になった場合、まず1 caseの非ゼロwhole-trajectory層を候補とする。q4/r4/K4では2次側のmassが4次側より大きいため、4次だけのpilotを自動優先しない。8 trajectory/16軸は記述的なpilot規模であり、母平均上界や必要信頼精度の標本数ではない。

conditional law、複数2次、4次との混合、component分布を正しく保持することが条件。既済H6 q8/r2/K2 s2 pilotを繰り返さず、その局所0.00734%補正をH8やK4へ移さない。

## 9. H5/H7：中性開殻の追加検証を採択

### 9.1 何を対象にするか

**中性linear H5/H7、1.00 Å、STO-3Gを採用する。** 旧helperのcharge+1/multiplicity3は別モデルなので流用しない。[S7]

|系|電荷|SCFで指定する開殻候補|Nα,Nβ|determinant sector次元|
|---|---:|---|---|---:|
|H5|0|doublet用のROHF|3,2|C(5,3)C(5,2)=100|
|H7|0|doublet用のROHF|4,3|C(7,4)C(7,3)=1225|

これは電子数・Msの選択であり、得られたDF solver stateのS²=3/4を測定前に認定するものではない。状態のS²・必要なspin symmetry・残差を確認してからdoubletという結果表現を用いる。

### 9.2 ROHF共通空間軌道を選ぶ理由

今回はUHF全面一般化ではなく、**ROHFで共通空間軌道を作る方法を採択**する。現実装の単一MO行列とspin-independent積分規約を保ち、電子数の偶奇に加えてspin別軌道・積分・DF政策まで一度に変える交絡を避けるためである。ROHFの共通軌道という性質はPySCF公式資料で確認できる。[S7,L2]

UHFが不正だという判断ではない。完全な積分変換と全空間計算では軌道表現の違いに対応できるが、現在のDF threshold/prefix/回路費用は表現に依存する。今回はROHFをbaseline orbital policyとし、UHFを別の追加研究にしない。

ROHF determinantは軌道取得・solver初期値のために使う。評価signalのstateを、既存DF sector solver stateからROHF determinantへ黙って変更しない。

### 9.3 必須の技術的修正と科学的条件

NaとNbが違うoccupation基底・固有値・変換を別々に構成する。H5で10×10、H7で35×35となっても、shape一致を理由にalphaとbetaの基底を同一視しない。[S7]

DF平方primitiveのphaseは(eα+eβ)²を含み、cross termを落とさない。spin数、JW ordering/sign、sector invariance、full-state native制御枝を対応させる。S²の診断はfull-space意味と整合する演算を使い、ROHF determinantのspin純度だけでDF stateのspinを代用しない。[S7,L3]

H5を先行する。最大40 compileの6構成案はcorrectness/feasibilityとして採用できるが、q1一つで5精度全域の資源比較を完了したとは呼ばない。精度に不適格だったことと実装が誤ったことを区別する。

H7はH5のinput/sector/reference/native correctnessが成立し、実資源を見て別予算を固定した後に進める。H5でPRが勝ったことはH7へ進むための必要条件にしない。H5だけでは奇数鎖一般の性能は分からないため、有限サイズの適用範囲を調べる第二の奇数点としてH7に意味がある。

## 10. T依存性：固定δを主比較に選ぶ

### 10.1 旧案から変える点

設計D3の旧案はq=2固定、T=.4/.8/1.6、δ=.2/.4/.8であった。これは固定回路ブロック数で時間を延ばした時の限界を調べるには妥当である。[S7]

今回、資源評価の主比較には**δ=.2固定、q=2/4/8**を選ぶ。同じ刻みでより長く発展するときの費用・bias・normalizationの競合を調べたいからである。固定q案を誤りとして消すのではなく、別の問いとして後順位へ置く。

|T|q|δ=T/q|扱い|
|---:|---:|---:|---|
|.4|2|.2|新規|
|.8|4|.2|既存値をidentity照合して利用|
|1.6|8|.2|新規|

対象はH6保存H/DF/state。B0/B0+Iはprefix15、full S2/full S4、B2はprefix5/r4/K2という既存5構成案を保つ。全構成で同一qを使う限定panelであり、各Tの全family optimumを認定する設計ではない。

### 10.2 この選択で何を調べられるか

同じDF tail、r、K、δなら

\[
\tau=\lambda_R T/(qr)=\lambda_R\delta/r
\]

は一定となる。単一eventのfinite-RTE lawは同じでも、R=qrが増えるため

\[
\log B=R\log b(\tau,K)
\]

はTとともに増え、normalizationのshot penaltyは蓄積する。局所lawを保っているからといって、総signal biasや費用比が一定とは限らない。[S7]

これは既存式からの推論であり、新Tの数値結果ではない。新Tではreference、candidate signal、u、全wrapper費用を実際に検証する。旧event列を長さだけ変えて新独立標本と呼ばない。

### 10.3 予算変更を明示する

新2T×[det4構成×1＋PR1構成×8]×両軸でprimary compile数は48、PR trajectoryは16のまま。ただしr=4なのでoccurrence数は

\[
8(2\times4+8\times4)=320,
\]

となり、旧q2固定案の128とは違う。回路の大きさ・wall/RSSも同じではない。旧4時間capや旧grantで実施してはいけない。

T=.8/p15/q4のB0+Iが未取得なら、anchorの2軸を別途計上する。native I全24軸が先に揃えば再利用できる。control/reference probes、失敗・予約carryも48に隠さず別計上する。

この変更を反映した候補・計算規模・資源案の具体化はCodexに任せ、実行は別承認とする。全H4〜H8×全T×全candidateの直積実験は採用しない。

## 11. 系サイズ比較とDF政策

H4 geometryには1.00 Åがなく、nativeともcontrol/loweringが違う。H4 native anchorは1.00 Åだがlegacy DF12・固定16候補、B0/B0+Iなし。H6/H8はtol-only DF19/28、探索92と独立確認を持つ。これらを一本の統制されたsize-scaling curveへpoolしてはいけない。[S5,S7]

**現在は設計案Aを採用し、H4を政策差のある別caseとして保持する。** 新H4を直ちに再生成する案Bは見送る。理由は、まず既にあるnative結果の対照を揃える方が、PRの利益の有無という問いに直接答えるためである。

H5/H7の新入力にはH6/H8と同じtol-only1e-8、cutoff0、signed generation order、weighted Hermitian projection、sector-state検証の政策を適用する。rankを他系と一致させるために切り落とさない。prefixはceil(L/4)、ceil(L/2)、ceil(3L/4)を共通規則とし、同じrank割合が同じtail normを意味しないため、λ_R・scalar・b/B等も記録する。[S7]

同じtolは元化学Hamiltonianに対する同じoperator誤差ではない。結果は各保存DFモデルを対象に解釈する。後に全H4〜H8の統制比較を主要claimにするなら、共通政策の新H4を別identityで追加する必要がある。既存H4を事後的に同政策だったことにしない。

H4〜H8は、1.00 Å/STO-3Gの有限サイズ適用範囲を調べる範囲として妥当である。ただし漸近指数、一般分子・大規模系・強相関距離全域への性能法則を示す範囲ではない。H9/H10以上は今回要求しない。追加geometryも要求しない。

## 12. 検証を増やすべき場合と、増やさない場合

### 12.1 今回埋める価値が高い不足

第一はnativeでの強い決定論対照、第二はH8の精度・指標・Pの整合した保存値評価。その後に開殻とTの限定実験を置く。新しい系の数を増やすだけより、何が原因で資源差が生じたかを分離できる。

### 12.2 今は必須にしないもの

- 旧H4 geometryの全面再計算、旧H4/H6 D0の再実行。
- 全候補の独立32追加、全希少supportのcompile。
- すべてのPF次数、別compiler、routing、FT synthesisの全面比較。
- 実state-preparation、energy/QPE/RPE、未知分子のtruth-free選抜。
- H9以上、複数距離の追加。
- 科学的採否前の汎用runner/quota/監査campaign。

これらは一般に無価値ではないが、現在の研究の条件付きbenchmarkを完成させる必須条件ではない。比較範囲を拡張して主張したい場合に、目的に沿って個別に採択する。

### 12.3 追加実験の成功条件

H8 D0は旧値一致と正しい再会計、native Iは正しいphase・費用・provenance、H5/H7は正しいtargetとsector/native接続、T比較は固定条件と新Tの正確な測定が成功条件である。

**PRが勝つことを成功条件にしない。** 費用逆転、全候補不適格、最良候補がdomain端、追加対照で利益が消えることも研究結果である。勝つまでprefix/q/r/K/Tを無断で増やさない。

## 13. 新規性・代替方針・研究としての着地点

PR原論文は、既に単一ancilla位相推定の量子化学資源推定と水素鎖のサイズ傾向を扱う。2026年のSPRINT研究も、near-integrability、randomization、symmetry protection、QROM等を統合した具体的な資源最適化を扱っている。[L1,L4]

したがって、PRの初提案、最初のH-chain資源評価、既知scalar保持の初提案、最良PF一般への勝利を本研究の新規性とはしない。外部論文のlarge-system/Toffoli/energy結果を、本研究の小さいDF-model/RZ/signal結果の欠測へ移さない。

独立した数値資源研究としての候補貢献は、**実compileした測定wrapper・固定signal誤差会計・標本選抜・強化対照を接続し、見かけの利益と維持される利益を分離すること**である。D0は精度・metric・Pで候補が切り替わることを、H4は対照に含める既知phaseが結論を変えることを、H6/H8は別native系でも費用差が存在することを示す。

ただし追加サイズを埋めるだけでは研究の新規性は自動的に増えない。統制した比較条件と、負の結果も含めた利益の適用範囲が必要である。採択や優先権を保証する判断は行わない。

代替案として、現状を限定case studyとして終了することも可能である。今回は利用者が広いresource評価の不足を埋めたいと明示しているため、上記の限定拡張を選ぶ。逆に全support保証の開発や大規模QPEのend-to-end化へ研究を転換する案は、現在のデータと目的からは採用しない。

原稿化は保留する。現段階の成果物は、固定結果・数式・比較scope・未保証事項・決定履歴が揃う研究ノートで十分である。

## 14. 次の担当・作業順序・再レビュー条件

### 14.1 直近の次作業

**サーバー側Codex：H8 D0 one-shotの実行承認を受け、修正SOURCEへの契約再bindingとfresh実行条件確認を行い、旧値一致→5精度/6指標/P解析→公開→mandatory STOP。**

新grantは本レビュー中には発行していない。旧draftのSOURCEをそのまま使わない。元H4/H6再解析を付け足さない。H8旧値不一致があれば技術原因を切り分けるが、旧shot規則や許容差の意味を変える場合は科学判断へ戻す。[S8–S10]

### 14.2 その後

1. H6 native B0+I 6軸pilot。実行を選んだ段階で必要最小限のrunner接続をまとめる。
2. 科学scopeを変えず、別実行承認の下でH6/H8各24軸の対照coverageを完成させる。既存PRは再取得しない。
3. H5中性ROHFの入力/sector/native pilot。
4. H6固定δのT比較。H5と実装上独立なら、個別承認・資源がある範囲で並行可能。
5. H5の正しさ・実資源が揃った後、H7へ。H5/H7の全面mapはpilot40軸の完了だけで自動認可しない。

この順序は情報価値に基づく優先順位で、全項目を完了するまで何も判断しないという意味ではない。native Iで利益の適用範囲が大きく変われば、その時点で次の範囲を再判断する。

### 14.3 GPTへ戻す条件

- 同一科学identityの数値不整合、reference/phase/allowanceの意味変更が必要になった。
- native Iを加えるとPRの主な利益が消え、研究の中心的解釈が変わる。
- H8 .0005や新metric/Pで、確認外候補を主要claimにする必要が生じた。
- H5/H7のspin/state検証が想定と違い、target・orbital/DF政策の変更が必要。
- T比較で誤差・normalizationの競合が想定と異なり、範囲の拡張又は縮小を決める必要がある。
- 希少層について、経験的benchmarkを超える母平均保証を研究目標に加える。

通常のsource差分、test追加、契約欄の結合、公開確認だけで同じ科学レビューを再実施しない。科学レビューが必要な場合も、明示的な開始承認を得てから行う。

## 15. 直前の提案からの変更点

- 「さらに最後のrunner/quota準備を行う」という提案は実施しないという利用者判断を維持する。採択した実行に必要な接続と、汎用準備campaignを分ける。
- 待ち状態だったH8 D0、native B0+I、H5/H7のtarget/ROHF、T比較の主軸について、科学的方針を今回決定した。
- Tは旧q2固定の推奨から、δ=.2固定を主にする判断へ変更した。理由と増えるoccurrence数を第10節に明記した。
- 以前の「H4の追加計算を必須にしない」は、旧H4限定benchmarkについて維持する。新しいnative対照や開殻・Tの検証へ無条件に禁止を広げない。
- 「H8までで一般的scalingが分かる」「高精度ほどPRが有利」「小さい未観測確率なら問題ない」は採用しない。
- 原稿編集を直近の次作業に戻さない。研究ノートと固定結果を優先する。

## 16. Codexに引き継ぐ最小決定表

```text
review_status = COMPLETE
review_date_JST = 2026-10-15
new_runtime_authorization = false
new_grant_issued = false
next_primary_work = H8_D0_ONE_SHOT_AFTER_EXPLICIT_EXECUTION_APPROVAL
implementation_source_candidate = 20f89bc32b161ee22e4a1ed709f9e3414c19633a
audit_bundle = 40a4f55464dba80ab2192d777f7d69d2f551676f
H8_D0_domain = existing4_eps + 0.0005; factors1/10; sixmetrics; P
H8_D0_old_equality_required = both_scopes_before_new_analysis
native_B0I_scientific_convention = FULL_FOCK_PAULI_IDENTITY_FROM_STORED_SPIN_ORBITAL_G
native_B0I_pilot = H6_p5q1_p10q2_p15q8_S2_T0.8_6axes
native_B0I_completion_scope = H6_24axes_and_H8_24axes_subject_to_separate_execution_approval
matched_B0_auto_execution = false
odd_chain_target = neutral_H5_then_H7_STO3G_1.00Angstrom
odd_chain_orbitals = ROHF_common_spatial_orbitals
odd_chain_state = DF_sector_solver_state_not_automatically_ROHF_determinant
T_primary_panel = H6_delta0.2_T0.4_0.8_1.6_q2_4_8
T_new_primary_compiles_static = 48
T_new_PR_trajectories_static = 16
T_new_PR_occurrences_static = 320
T_counts_exclude_control_and_missing_T0.8_B0I_anchor = true
new_H4_strict_DF_policy_required_now = false
all_support_certificate_required_now = false
automatic_more_sampling = false
H9_plus = out_of_scope
additional_geometry = out_of_scope
manuscript_edit = deferred
generic_runner_quota_campaign = not_restarted
mandatory_STOP = preserved
```

これは科学判断の索引であり、上記文字列をauthorizationとしてrunnerへ渡してはならない。

## 17. 参考資料・固定証拠

以下のリンクは読取・来歴追跡用。branchの最新性ではなく指定commitの内容を基準とする。

[S1] 科学レビューR01–R09判断索引：
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4675730e7e40d25ba3618334a4a0e28b4bd2f145/docs/research/track_a_scientific_review_decisions.md

[S2] H8結果報告：
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/681ec23c93fdd1090d325908f4b155a79115afdf/docs/research/track_a_h8_resource_map_result_v1.md

[S3] H8主要精度CSV、Git blob `650af7849f0cc31726b500be5b83fbdda6da7331`：
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/681ec23c93fdd1090d325908f4b155a79115afdf/artifacts/resource_applicability/track_a_h8_resource_map_result_v1/2026-10-14/precision_summary.csv

[S4] H8完了監査、Git blob `2ca6d3cb0d8c6705b178479b97125f6c7ce931d9`：
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/681ec23c93fdd1090d325908f4b155a79115afdf/artifacts/resource_applicability/track_a_h8_resource_map_result_v1/2026-10-14/completion_audit.json

[S5] D0結果報告、Git blob `1ff7387330f529ddc65e28c79b0dc97420affbba`：
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8dbdb68293c988a999645e21a82b53966f9ef6cd/docs/research/track_a_d0_saved_value_reanalysis_result_20261014.md

[S6] D0 precision summary、Git blob `54512ddabd659283153a56337f1a4121a5a13f54`：
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8dbdb68293c988a999645e21a82b53966f9ef6cd/artifacts/resource_applicability/track_a_d0_saved_value_reanalysis_result_v1/2026-10-14/precision_summary_v1.csv

[S7] 追加検証設計、Git blob `e76b2157eb759f6a6328f1296fa628ffe3295856`：
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2ccc79502f043b4885e2337ffa98ff45f1875ae4/docs/research/track_a_resource_evaluation_expansion_design.md

[S8] 独立技術監査、Git blob `65f280c2f7c5edf4eb105a207bd77dc7b21c620f`：
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/40a4f55464dba80ab2192d777f7d69d2f551676f/docs/research/track_a_independent_technical_audit_20261015.md

[S9] 最終技術handoff（修正前SOURCEに関する履歴）：
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2ec66aae5e17c9329fcea8ad962be27130980d31/docs/research/track_a_final_pre_review_handoff_20261015.md

[S10] 修正SOURCEへの候補binding更新索引、Git blob `e713ea2f8b7ddce0b08248151c348107a11d99da`：
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/40a4f55464dba80ab2192d777f7d69d2f551676f/artifacts/resource_applicability/track_a_independent_technical_audit_v1/2026-10-15/candidate_binding_update_v1.json

[S11] H6希少次数既存解析：
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eee1343f6e99aa7661892ddfbe1067e974e2b74a/docs/research/track_a_h6_rare_order_cost_reliability_v1.md

[S12] 修正SOURCEのH6 bridge（identity・signal・error policy部分を重点読取）：
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/20f89bc32b161ee22e4a1ed709f9e3414c19633a/scripts/h6_native_b0i_bridge_v1.py

[S13] H8希少次数保存law、Git blob `44c18a6dcb6252fd8c67d8aa901cb1845bc30266`：
https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/681ec23c93fdd1090d325908f4b155a79115afdf/artifacts/resource_applicability/track_a_h8_resource_map_result_v1/2026-10-14/rare_order_observation.json

[L1] Günther et al., *Phase estimation with partially randomized time evolution*, arXiv:2503.05647v2（2026-07-10改訂）、PRX Quantum 7, 020332（2026-05-19公開）、DOI 10.1103/ynxb-p2xq。abstract・書誌を確認。
https://arxiv.org/abs/2503.05647v2
https://arxiv.org/abs/2503.05647 （取得できたabstract・版履歴）
https://journals.aps.org/prxquantum/abstract/10.1103/ynxb-p2xq

[L2] PySCF公式、*Self-consistent field (SCF) methods*。ROHF/UHFの軌道規約を参照。現公式文書を固定インストールのbinary同一性の証拠とはしない。
https://pyscf.org/user/scf.html

[L3] PySCF公式、`pyscf.fci.spin_op`。S²診断の実装資料。
https://pyscf.org/_modules/pyscf/fci/spin_op.html

[L4] Casares et al., *Theory and practice of Trotter product formulas for quantum chemistry*, arXiv:2606.30741（2026-06-29）。導入・SPRINT/GRADEの位置付けを限定確認。全比較条件の独立再現ではない。
https://arxiv.org/abs/2606.30741

---

本レビューの採択は将来の実行承認を代行しない。科学的な判断と、固定SOURCE・実行契約・利用資源・STOP条件を結び付けたユーザーの実行承認を分離する。研究ノートとして本書を保全し、原稿化、無関係な技術準備、既済計算の再実行へ自動的に進まない。
