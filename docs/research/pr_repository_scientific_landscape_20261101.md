# Partially-Randomized-Trotter：研究全体の科学的到達点と証拠の地図

2026-11-01 JST。Repository：`HIROMU1015/Partially-Randomized-Trotter`。Codexによる既存資料の技術調査であり、中心claim・新規性・次の研究方針の正式採択や詳細科学レビューではない。対になる資料は[主張・証拠・未解決課題対応表](pr_repository_claim_evidence_gap_matrix_20261101.md)。

## Executive Summary

- **本来の中心課題**は、DF Hamiltonianのどこを決定論的に保持し、どこを有限RTEで補うと、同じ精度・失敗確率の位相／エネルギー推定が安くなるか、である。単回回路の短さだけでなく、PF bias、finite-RTE bias、normalizationによる減衰、必要shot数、全roundの回路価格を接続する目的を持つ。[元の研究目的][ORIGIN]
- **DFで得た主要な実証**は、固定されたHamiltonian・状態・候補集合・compilerにおける有限時間複素信号の資源地図である。B2が有利な条件はあり、geometry移送と別seed確認もある。一方、B0/B0+I・高次PFを強くすると優位が消える条件もある。「DF部分ランダム化は常に勝つ」は支持されない。[M1][M1]、[強化対照][AUDITA]、[6確認比較][A6]
- **最新DF長時間研究**ではH4/H6、T=.8〜2、5精度の取得域でB2が決定論PF envelopeより低い点費用を33/40セルで示し、その33比較は別標本でも方向を再現した。後続B3はB2・決定論PFより低い費用が0/40だった。B3は二体DF fragmentを全ランダム化しても**一体項を保持する**。Pauliの全非identity項をランダム化する純RTEとは別方式である。[長時間B2][LONG]、[B3][B3]
- **RPE接続は「未着手」でも「最終実証済み」でもない**。短4段の信号・分枝復元、round別schedule、全18roundのモデル費用再最適化まである。最終モデルのH4 partialはfullより点推定4.858%低いが、全採用不確かさ区間が重なる。直接compile domainを越える長round移送、目標energy成功条件まで含む最終保証は未完成である。[4段][RPE4]、[M06-F][RPEMODEL]
- **Track Bの成果**は、return集約・局所生成・有限bit会計の構成、固定syntheticでのnative T/CX/shot交換、保証と価格による順位変化である。後続H4 RA pilotも3方式各1024pathまで完成しているが、tail-onlyのlogical観測であり、認証N・全wrapper native総資源優位は未取得。新アルゴリズムとしての独立性も未確立。[Track B横断表][BSURVEY]、[後続claim表][BLATEST]、[H4 C1][C1]
- **Pauli-JWで新しく得た観測**は、最新有限候補・保存価格標本・仮想compiled残差ρ=0の条件で、純RTEがHybridより14/15精度セルで低Gとなること（H4 ε=.05のみHybrid低G）。H6 Hybridの大event追加後も維持する。ただし5精度は価格標本を共有し、15独立実験や母平均winnerの証明ではない。[11/01統合][PAULI]、[H6追加][H6HYB]
- **低費用の説明で確かなのはNとCの競合**である。H2/H4では純RTEの大きいNを低いCが補い、H6厳精度では小bias・広いheadroomでNも下がる。prefix、tail、sampling law、Gaussian基底、control、compilerを一意に因果配分した証拠はない。DF対Pauliの比較はこの内部比較と分ける。[機構調査][MECH]、[H4表現比較][PX]

新しい科学計算・保存値の数値再解析・再ランキング・sampling・simulation・Hamiltonian/state/reference生成・回路build/compileは全て0件。既存の70研究カード、Track A横断監査、Track B横断調査を再利用し、その後の長時間・B3・Pauli・混在辞書の結果で時点差を補った。過去の人工テストPASS、公開、採択記録を新しい科学的保証へ格上げしていない。

## 1. 元の研究目的と比較単位

DFでは概念的に `H = scalar + H_one + Σ_l H_l` を用い、rank済み二体fragment列のprefixをH_D、残りをH_Rへ置く。H_oneを決定論側に残す。fragment順位用proxy normと、回転基底のI/Z/ZZ展開後にRTE lawへ使うexact normは異なる。rank、生成順／norm順、identity policyを跨いで同じL_Dと呼ばない。[方法][METHOD]

| DF方式 | 操作と研究上の問い |
|---|---|
| Full PF（旧B1） | 一体項と全DF fragmentを保持。全保持の同精度対照 |
| B0 | prefixを保持しtailをdiscard。短い回路のbiasが許容精度に収まるか |
| B0+I | discard tailのidentity成分をscalarへ一回戻す。B0とは位相biasが違い得る |
| B2／Hybrid PR | 一体項＋非空prefixを保持し、tailをfinite paired-Taylor RTEで補う |
| B3 | 二体prefix L_D=0。一体項は保持、全二体fragmentがrandom tail。全H純RTEではない |

この命名は旧実験のB0 anchor取得stageや「Phase B2」、Track Bの比較class B2と同名でも別物である。PauliのB0/B0Iはconstantを常に保持する実装の同prefix/qでは同等で、DFのB0/B0+Iと重複関係が異なる。[Pauli定義][PAULI]

元の目標は `G_total = Σ_round,axis N × E[C_full_Hadamard,no-prep]`。現在の主要な実測比較は固定Tにおける `G = N_cos C_cos + N_sin C_sin` である。Cは通常compiled RZ、Nは解析的十分shot会計、nは古典trajectory価格標本数。状態準備P=0は研究範囲の仮定であり、候補間のNが違えば共通Pも相殺しない。RZ、Clifford+TのT、CX、depth、古典秒は別指標である。

## 2. DF研究の展開と現在有効な結果

### 2.1 誤差係数・finite RTE・価格モデルの基礎

初期DF screeningには誤ったground-state入力を受けた失効結果があり、現行DFの肯定証拠へ使わない。一方、その後のPF係数検証・finite-RTE検証は別の証拠系列である。H4のD6摂動係数と支配固有位相係数、H2〜H5のstate-actionとdense参照、H6のPF行列を作らない運用係数が検証された。H6旧係数はDF rank11/L_D5の窓限定で、後年のtol-only DF19の資源研究とは異なる。C_useは実行delta窓内の経験的envelopeであり厳密上界でもH12の係数でもない。[係数検証][COEFF]、[失効・欠落][STATUS]

有限RTEでは打切りによる平均信号誤差と正規化によるattenuationを分けた。正scalarは位相を変えなくても信号半径・shot数を変える。Hadamardで読む平均振幅演算子とrandom-unitary channelを混同しない。[finite RTE][FINITE]

回路価格では、event単価の和はwhole compileを再現しないこと、same/different基底境界、pair/triple/connected-cluster、controlled反復のaffine modelが条件付きで役立つことを確認した。H4/H5の未使用列長holdoutの成功も、L_D・snapshot・compiler変更への無条件移送ではない。希少orderをほぼ観測しないIID runを、全lawの価格確認から外した履歴も重要である。[価格検証統合][COST]

### 2.2 PR-2からTrack Aのmatched-accuracy地図へ

PR-2初期pilotは圧縮＋残差補完を有望候補としたが、旧S0は入力再現不一致で停止した。別snapshot系列のV1〜V4/S2、M1、M2が後続しており、旧S0の成功への書換えではない。同時期のPR-3は固定2qubitのtail-only外挿でbiasを96.96%下げたが、分散とbackbone価格を戻すと通常PR最良の7.557倍で停止した。bias改善を総費用改善としない負結果である。[初期pilot][PR3]、[M1][M1]

H4 linear1.00Å/STO-3G/DF12/T=.8のM1-B1では、全210cell（後続PM-1を含めると218）を扱い、full測定wrapperの費用を取得した。登録域の6指標point ParetoはB2 rank3/q1の2候補。旧16cell selectorはその1件を落とした。候補探索と最良候補の独立確認を分ける必要性を示す実例である。

同じDF12のH4 1.30Åへdevelopmentで固定した5構成だけを移したM2はTRANSFER_SUPPORTED。B2のr4/r8がpoint Paretoに残った。これは移送先で全候補を再最適化した結果ではなく、より大きいPでB1が費用envelopeへ入る。近接B0 rank4/5の追加後も元精度でB2の点利得は残ったが、PM-2の302精度点は同じ価格標本を共有し、全点のengineering区間が他候補と重なる。[M2][M2]、[横断カードPR-01/EVAL-01][CARDS]

候補選択の周辺研究では、P-Aのblind run改善は残るが非退化30taskで一区間対照との差がなく、interval DP固有claimを停止した。P-BはH4 DF12/L_D0〜11・6δ・q1/2/4のsignal-aware診断で実用的な選択差なし。P-Cは局所geometry支持の後、stretchで予測破綻。P-Dは強い対照と選択が一致し、finite補正固有の利益は未成立。R3はquantum-specificな方法差が固定できず、数値pilot未実行のまま停止している。[OPT/CIRカードと原証拠][CARDS]

### 2.3 Geometry・分子数・強い決定論的対照

| 系列・固定条件 | 得られた観測と現在の限界 |
|---|---|
| H4 geometry、legacy DF12、追加6距離 .70/.80/.90/1.10/1.40/1.60Å、S2/T=.8、218候補 | 元6×302保存精度点でB2点最小。後続B0+Iの120構成/240軸を加えるとB0I397点・B21415点へ。比率は独立実験の勝率ではなく、同じ32価格標本に基づく有限地図。geometryごと最適化と固定5構成transferを区別。[geometry][GEOM]、[B0I][B0I] |
| H5/H7中性open-shell pilot、1Å/STO-3G、DF14/23、q1 | 固定6構成・各PR n8。粗精度でPR低費用例があるが、H7 ε≤.005は全候補不適格。H6/H8 singlet系列とのscaling同定ではない。[19系列監査][AUDITA] |
| H6 tol-only DF19、S2/S4強化136対照、T=.8 | 旧Full対照の大きなPR利得は強化対照で縮小。ε=.05のB0I p7/q1を実測するとPRより低費用。残る精度の当時の最良対照にはmodel単価が残る。[実測pair][H6STRONG] |
| H8 tol-only DF28、強化184対照、T=.8 | 主要6軸実測反映後、通常signedのPR削減率は5精度で−8.0230/33.2373/31.4549/22.9613/−50.8125%。全184を実測した意味ではない。非相殺診断も厳密bias保証とは異なる。[最新置換会計][H8STRONG] |
| H2/H4/H6、tol-only rank3/10/19、共通588決定論候補、q≤8、S2/Yoshida4/m5 | H2は全5精度でPF低費用。H4/H6は粗い3精度でPF、厳しい2精度で保存PR低費用。ε=.0005のPR削減率は−25.1409/+37.3736/+40.9592%。PR側は保存探索集合であり、この全比較の独立確認とはしない。[共通域完成][PF588] |

H8希少RTEの後続では、上記独立S0各48本とS2_SINGLE条件付き各43本を層別に扱い、残るS2_MULTI/S4_PRESENTの費用感度まで完成した。ユーザーは10/21にその限定scopeを採択して終了し、追加sampling不要と決定している。母平均保証は未知のままだが、古い「希少層は全部未取得」を現在の状態にしない。[後続採択政策][H8TAIL]

tol-onlyはrankを10/19/28へ強制する指定ではなく、保存された返却rankの観測である。H4 legacy rank12とtol-only rank10、H6のweighted-Hermitian completion、入力生成・state solver・正規化差を保持する。同じ分子名だけで表を結合しない。

### 2.4 独立確認と長時間・B3の後続結果

10/19統合の主表6比較は、linear1Å/STO-3G/tol-only1e-8、H4/H6/H8 rank10/19/28、T=.8、P=0、full wrapper、Qiskit1.3/opt1/seed17の保存別seed確認である。

| 系 | ε=.05 | ε=.0005 | 確認の意味 |
|---|---|---|---|
| H4 DF10 | PF低費用、探索方向NOT_REPRODUCED、n1108 | PR低費用、REPRODUCED、n78 | legacy DF12の別P成功と混ぜない |
| H6 DF19 | PF低費用方向をREPRODUCED、n48 | PR低費用、REPRODUCED、n104 | REPRODUCEDはPR勝利の意味ではない |
| H8 DF28 | PR低費用、n48 | PR低費用、n48 | Mechanismの同一cohortのalias。独立証拠を二重計上しない。詳細公開監査の制約あり |

一次表は[固定6比較][A6TABLE]、監査とSOURCEの対応は[統合資料][A6]。さらに10/22には固定prefix1/3/5・K4・R_outer32・ε=.0005で、高次PR対S2の3比較を新環境118trajectoryで確認し、保存削減率46.51/53.11/63.13%を得た。旧全seed除外の原資料未回収、S4未観測などの限界を維持する。この確認は強化PF全domainへの優位の確認ではない。[高次独立確認][HIGHCONF]

10/25〜26の長時間研究はH4/H6 rank10/19、T=.8/1.2/1.6/2、ε=.05〜.0005、S2/Y4/m5、q/r/K/prefixを広げた別研究である。11,632signal構成に対し価格取得は642候補。全signal候補をcompileしたのではない。取得域のB2対最良PFは33/40点優位で、その33方向を別標本でも再現した。高次対S2ではH4 T=1.6・ε=.005のm5探索優位が確認で反転し、単一方式の一様優位は支持しない。[主報告][LONG]、[確認CSV][LONGCSV]

後続B3は3432signal構成、価格192候補、探索1728＋確認1553trajectory。取得域でB2より低い費用0/40、Full PFおよびB0/B0I込みPF envelopeより低い費用も0/40。ε=.0005の公式別nominee確認ではm5がS2より低い条件0/8。**B2の長時間利得と、DF二体全ランダム化B3の負結果を同時に保持する**。B3にも大きいq/R・K拡張はあるが、未計測域・境界外最適性・全law母平均を認定していない。[B3主報告][B3]、[SOURCE来歴][B3PROV]、[一体保持のSOURCE][B3SRC]

B3の公開refは`ebe9e2aa…`。local最終`f794a2aa…`は今回GitHub contentsで404、remote branchも前者だった。主要比較表・scientific_provenance等はlocal最終版と同一だが、圧縮費用データと一部終了来歴などには差がある。本文では公開版へリンクし、全rawのlocal/remote同一性は主張しない。詳細は対応表の可用性欄。

## 3. RPE/QPE接続の到達点

| 構成要素 | 理論・実装・数値確認の到達点 | 現在も分離すべき限界 |
|---|---|---|
| PF近似 | D6、支配位相、state-action、実行窓C_useを検証 | 経験的係数から全δの厳密上界は得られない。H12未決定 |
| finite RTE | 打切り、B、attenuation、補正平均とraw信号の接続を検証 | law／精度／基準信号半径の成立域。正Bとbiasは別 |
| 半径→位相→shots | 正の基準半径とfresh IIDの下でHoeffding・軸別α・union配分を実装 | 半径1はexact groundから自動的には出ない。Nは必要最小ではなく十分設計 |
| 短RPE | H4 DF12/L_D3/δ=.1/r4/K2、q1/2/4/8の4段、1572shot会計と分枝復元をlocal検証 | 10万復元の失敗0は全round成功率の証明ではない。[4段][RPE4] |
| energy horizon | CA/10、δ=.1ではq_max32768。旧固定r4/K2はPF予算とattenuationで不適格 | q8短roundの設定を単純反復できない。[方法][METHOD] |
| round別schedule・全round集計 | δ=.01/.0125/.02の行列確認、後続WP/M06-Fで配分・r/K・整数shot再最適化まで実施 | 中央block proxy、full wrapper proxy、直接q≤32、長q外挿を分ける |
| 最新coherent opt2 | H4 DF12/L_D3対12、両者δ=.02、全18roundのモデルGが1.263314e12対1.327822e12、13588対11162shots | partial点利得4.858%だがlocal5%・per-r・transfer区間が全て重なる。物理prep未測定。[M06-F][RPEMODEL]、[全round保存JSON][RPEJSON] |

したがって「総費用を一度も集計していない」は現在の全証拠を表さない。一方、上記のモデル集計、後年の単一TのC/N/G、Clifford+T toy予算を合わせても、目標energy精度・成功確率と全long-round実装の公平な最終費用実証にはならない。最新Pauliや長時間DF報告自身も全QPE/RPE評価からscopeを分けている。

M1-S後続にも、visibilityとphase biasを合わせた単一時刻Task Aの十分shot診断、π/3開端点の整数witnessがある。G3/T4の8対照で局所shot削減witnessが得られたが、native単価・energy prior・全dyadic roundを含むTask Bは未接続。これは初期RPEの不足を全面解消した系列ではない。[M1-S][M1S]

## 4. Track B・アルゴリズム共設計の成果

既存[Track B利用可能性表][BSURVEY]と[資源構造整理][BSTRUCT]を基礎に、後続v1.2/C1まで追跡した。以下のnativeは主に加法的Clifford+T primitive会計であり、Track Aのwhole compiled RZと同じ価格ではない。

| 系列 | 研究課題・得られた成果 | 現在の状態・研究上の意味 |
|---|---|---|
| BF/BM・高次係数/合成 | BFの固定H4 DF12/5-stage family・同評価予算で追加decision valueなし。BM adapterは既知compact BCHと同値 | 現family負結果／方法差なし。高次PF一般の不可能性ではない |
| SP0.5/SP1 | 安い離散回転とweight momentの交換。strict primitive witnessと短wrapperのT利益、長wrapper反転/cap | syntheticの限定成果。角度と本数も変わり、単一因果・分子優位ではない |
| R0/R1/R1.5 | 同finite P3 meanでidentity/odd配分・IS・native価格を比較 | normalization最小がnative最小とは限らないという限定反例。一般最適sampling未確立 |
| RA-D0、G1〜G5 | P3係数多面体・有限bit law・固定classの分離を確認。ただし強いmatched CTSが得たlawはJ1をT/CX/1Qで下回った | 限定数学成果を保持し、同固定辞書の優位主線を閉じた。全RAのno-goではない。[G4][G4] |
| G6 | 正有理p、0<x≤1、奇数m、Hermitian involutionのformal return集約、非負係数、Green局所query、有限bit補正 | 有効な構成成果。既知Green/Euler/ISを超える結合構成の独立性・一般資源利益は未確立。[監査][G6]、[kernel][G6SRC] |
| G7/G8 | Ordinary envelope、zero-fill、局所生成とon-demand合成へ接続 | whole-trial予算・取得費用が必要。生成器完成とnative利益を同一視しない。[生成SOURCE][G7SRC] |
| G9/G10 | 固定3system-qubit synthetic、m3/5/7 within-degree比較。Closed P5のT利点とCTSのCX利点、general Fullが強い特殊化に負ける例 | 同target・同情報の強い対照が必要。異なるmを同exponential精度の順位へ並べない。[G9][G9] |
| D1〜D3 | finite P5、τ5/7、r1/2の6方式。literal inverse取消、保証×価格4caseで選択・順位が変わる | Taylor保証・price precision・native bias allowanceを分離した診断成果。新生成法の独立性ではない。[D3][D3] |
| Parent allocation/S4 | 訂正KKTを含む70law、固定3qubit/P5/r1でcommon対noncommonのbest-to-best T利益約14/19%、N/CX増 | 配分・proposal・角度が変わる。KKT単独因果や分子PR改善ではない。同一入力探索は区切り済み。[S4][PARENT] |
| DF sparse correction/E/T | 局所支持と補正構成、固定generic whole planの3条件でwhole下界がPauli上界より高い、1条件未確定 | 限定費用負結果。全Gaussian/whole合成の不能ではない |
| Gaussian積生成／interstep相関 | 既知法対応・候補固有規則不足。marginal保存だけではcoherent meanを保存しない | 性能で敗れた実験と未定義の案を分け、停止を維持 |

### 4.1 RAの後続実分子接続を反映する

C1はlinear H4/1Å/STO-3G/DF12、generation-prefix3、13basis/324labels、T4/5・q1/r4/m5で、Ordinary/General Full/Closed P5各1024whole-pathが完成した。度0parentでもchild rotationが各pathに4個あり、無料identityではない。logical差は小さく、zero-path削減は観測されなかった。

この完成は、古い「syntheticしかない」を更新する。ただしscopeはTAIL_ONLY_PLUS_SCALAR、価格はlogical計数・古典生成観測で、deterministic halves・full coherent estimator・strict native価格・共通認証Nが未取得。exact R1 dyadic modelから物理DF exponentialへのbindingも未閉鎖。CPUの並列構成・cache・計時境界が異なり、最適化済み算法速度順位を示さない。C1は追加資源研究を一時凍結している。[H4資源観測][C1]、[v1.2 claim表][BLATEST]

### 4.2 周辺理論・負結果・表現研究

- **FR**：phase不変な正scalarと物理半径を分離。非一様4×4で同情報strict gainはあるが、固定予算の片側認証差なし。後続のユーザー提供root技術記録snapshotはTECHNICAL_NOTEとして整理完了、T3のfinite-RTE制約付き到達可能性は未解決とする。docs/researchに残る古いDRAFTを唯一の現在状態にしない。今回は当該保存snapshotの帰属を記し、新規定理の再採択をしていない。[後続記録][FR]
- **M1-S**：shared ± labelのC-sec有限次元理論、secular減衰・共鳴floorとH4有限step信号を保持する。高visibilityと小phase biasは同義でなく、同draw数と同native費用も別。新primitiveの強い主張、全RPE総資源改善は未確立。[統合][M1S]
- **N1/N2/N3**：N1 selectorはbias込みでexact fallback。N2のweighted entry-L1訂正と回路費用増を分離。N3は非負density・固定粒子数・一外部mode射影のSchur/OCC証拠、理想族同値・SOS包含、指定query分離を保持するが、元精度で追加H縮約0。現benchmark主線は終了。精度保証能力と最終task利益の違いを示す。[N3][N3]、[REPカード][CARDS]
- **QC C1/C2/C3/C4**：macrocycleは固定catalogで資源利益なし、branch外挿・self-energyは安い具体access未成立、cost-aware whole-path law/DP自体は既知法へ還元。未知normalizer・range・生成取得費を省略しない。[C4][C4]、[QCカード][CARDS]
- **QC A2**：RF0/RF1/EW2の有限step相対保証と独立checkerは成立。H4/DF12 pilotでは強いcentered対照の十分stepが大きく、現実用化経路の追加投資は見送り。定理の誤りではなく現保証の資源負結果。[成果と限界][QC]
- **mixed/THRIFT、core吸収・physical pair、SOS-PF、H1〜H4/exact encoding**：限定step改善や恒等式はあるが、同情報既知法への還元、追加算術費用、basis/central価格、具体生成則不足を別々に記録している。これらを全部「未検討」に戻さない。[全体map][ALL]

## 5. Pauli-JW研究の位置付けと低費用の二つの問い

### 5.1 DFから引き継いだもの／新しく比較したもの

引き継いだのは、有限RTEのcoherent mean・位相規約、bias＋normalization＋shot＋full-wrapper価格の会計、探索と別seed確認の区別である。新しいのは、Pauli-JW項を分割単位としてFull PF、B0/B0I、Hybrid、L_D=0純RTEを同じ分子内の保存H/raw ray/referenceで比べたこと。DFのL_D/rank/一体保持をPauli prefixへ数値で移した研究ではない。

最新はlinear H2/H4/H6・1Å/STO-3G、非identity項14/184/918、S2/T=.8、central-rotation control、cosine/+Im sine、Qiskit1.3/opt1/seed17、P=0。主会計はρ=0仮定で、実compiled残差と母平均価格tailは未保証。

| 系 | 最新保存比較 | 旧結論を更新した証拠 |
|---|---|---|
| H2 | 純RTEがHybridより全5精度で低G | 有望な既存s4候補の追加独立確認。旧Hybrid優位を更新。大s探索による改善ではない。[H2][H2PAULI] |
| H4 | ε=.05のみHybrid、残る4精度は純RTE低G | PR141構成探索、確認集合minと事前nomineeを分離。旧B0対固定PRの粗精度優位を最新PR集合へ移さない。[H4][H4PAULI] |
| H6 | 大event Hybrid追加後も全5精度で純RTE低G | r≤4→r8→純RTE s512→Hybrid r32まで追跡。旧巨大倍率を最新比較へ使わない。[event][EVENT]、[Hybrid追加][H6HYB] |

Full PF/B0の保存family最良値も統合されており、PR familyの低費用はHybridそのものの利点とは限らない。確認後集合min・逐次延長を含むため、全prefix/q/r/Kの最適性・選抜後同時coverageを認定しない。[四方式・cohort表][PAULI]

### 5.2 問い1：Pauli内部の純RTE対Hybrid

保存費用は候補固有のB・bias・allowanceから得るNと、保存Cの積で説明できる。H2粗精度、H4厳精度では純RTEのNが大きくてもCが低い。H4粗精度はHybridのshot節約が勝つ。H6厳精度では純RTEのBが大きくても小bias・広headroomによりNが小さく、Cも低い。

eventを増やすとB/Nが下がってもCが増え、H4は保存s64、H6はs512付近で取得域の低Gを示す。Hybridにもr32→64で再上昇がある。同q/r/KのH4 L0/L1等は既にあるが、prefix変更はtail・law・biasも変える。**保持回路だけの独立因果効果を測った対照ではない**。[機構調査][MECH]

### 5.3 問い2：Pauli純RTE対保存DF-native

旧H4固定比較では同MO積分からのfull-H一致診断（差約8.46e-15）と共通raw stateがある。しかしPauli184項/prefix27対DF10fragment/prefix1＋one-bodyで、分割・event・lawは異なる。後続Pauli L0最良と、DF B2/B3の最良はさらに候補域・状態／reference来歴・精度保証政策が違う。「Hamiltonian表現が違うから安い」という一因子説明にはならない。[H4比較][PX]

D1後続9/9測定はgeneric Pauli/DFの保持・event費と非加法性を実測し、D2は同Pauli bodyのgeneric→central controlでPF255500→15028、PR index0 9652→1145 RZを実測した。これはcontrol構成とcompiler応答の合成効果であり、最新純RTE対Hybridの効果量でも表現だけの効果量でもない。DFにはGaussian基底の共有が既にあり、「毎event必ず全基底往復」も正しくない。[D1][D1]、[D2][D2]

補助的に4mode合成混在辞書F2ではPP＜SP＜PS＜SSの単回期待費用と、Gaussian往復／Pauli局所Cliffordという具体構造差が確認された。ただしK0/r2・別fixtureで、N加重比較は未取得。分子純RTEの低費用原因へその差を転用しない。PS48のsingleton置換は一律RZ/CX削減を支持せず、この独立研究方向は一時終了済み。[F2機構][MIXCAUSE]、[PS48][PS48]、[終了記録][MIX]

## 6. 実回路・実装検証の到達点

| 証拠層 | 到達した内容 | その層だけでは言えないこと |
|---|---|---|
| 実装・人工検査 | DF/Pauli adapter、signed stage、phase、actual adjoint、control、scalar-once、law・会計・cache | 科学的費用優位、分子accuracy、全law保証 |
| AX実分子correctness | H4 v5のnative DF・global S4・symmetric control、8correctness/28wrapperなど登録scope | n2の価格から母平均winner・認証Gを得ない。[AX][AX]、[native SOURCE][AXSRC] |
| native/compiled取得 | H4/H6/H8のfull wrapper、長時間DF、Pauli両軸価格 | 原理的最小回路、実機routing/noise/FT価格 |
| 局所包含 | H4 PauliのPF/PR0/PR1×両軸6局所Arb包含 | 最新全候補・全lawの実ρ。R2-CはPROOF_INCOMPLETE。[Arb][ARB]、[R2-C][R2] |
| 理論対compiled | 固定S2のC置換会計、20条件のG00/G01など | 独立N_circuit未取得。H6/H8補助unweighted-G診断不合格と新H8 cohortを保持。[S2再計算][S2CNG]、[会計][CNG] |
| 科学的費用比較 | 経験的DF／ρ条件付きPauliのN×C | 全energy推定総費用、実量子shot・physical state-prep・実backend成功率 |

過去のnative終了UNKNOWN、local correctness、SOURCE静的対応、保証付き包含、有限標本平均は別の軸で記録する。後続完了がある工程を一律未完成とせず、個別局所PASSを全体PASSへ格上げしない。

## 7. 時系列で修正された主張

| 以前の読み方 | 後続の証拠 | 現在の扱い |
|---|---|---|
| 「小norm・低PF誤差なら低総費用」 | B0/B0I、R1、D3、Parent、PauliのN/C交換 | precision適格性と測定込み低資源性を分ける |
| 「Full PFより安いので強い決定論対照より有利」 | H4 B0I、H6 p7、H8強化184、共通588 | 比較集合を明記。粗精度や厳精度の反転を保持 |
| 「高次はRTE負担増で不利」 | 高次固定確認と長時間B2で利益例 | 高次を一律棄却しない。一方B3のm5は0/8低費用 |
| 「P-A blind成功はinterval DPの独立寄与」 | 一区間baselineとの非退化30task差0 | run-level改善は残しinterval固有claim停止 |
| 「P-C局所geometry予測はstretchにも移る」 | 1.40/1.60Åで破綻、trackingがprefixを変えず | 局所結果を保持しcurrent family主題停止 |
| 「P-D finite補正が独自の選択差を生む」 | 強いB1b/B2/B4が同候補 | weak対照だけの差を独立寄与にしない |
| 「FRは理論監査待ちだけ」 | 後続root技術記録のTECHNICAL_NOTE整理 | 独立新定理未確立・T3未解決と区別 |
| 「G6はsynthetic接続だけ」 | H4 C1の3方式1024path完成 | 分子logical接続は完成、native/Nは未取得 |
| 「D1は停止したので全部品価格未測定」 | 別SOURCE基準Bの9/9完了 | 初回STOPと後続完成を両方保持 |
| 「Pauli H2/H6ではHybridが勝つ」 | H2追加確認、H6双方のevent拡張 | 最新14/15純RTE低G。旧範囲の倍率を流用しない |
| 「旧Cgs表はどこにも残っていない」 | 後続GAP探索で無効化37ファイルのhistorical identity確認 | 修正済みcanonical入力はなお未確認。旧結果は失効のまま |

上表は新しい科学的判定ではなく、固定文書間の更新関係の整理である。[研究map][ALL]、[Track B][BSURVEY]、[Pauli][PAULI]、[GAP後続][GAPS]

## 8. Coverage、未取得証拠、今回の限界

mainの6入口資料（PROJECT_MAP、研究概要、目的、方法、VALIDATION_STATUS、manifest）、最新4591系の入口、既存横断3調査を使用した。local remote-tracking refsは872本を名前で確認したが、全branch全artifactの全文独立監査や全GitHub branchの再列挙はしていない。以前の調査が集めたbranch/tree/70カードのcoverageを明示して再利用した。[旧coverage][COVER]

| 既存カタログの系列 | 今回の扱い・後続更新 |
|---|---|
| REP-01〜09 | grouping/DF、失効screening、UWC、N1/N2/N3、mixed、Pauli対DF、core/pairをcoverage。Pauli/H4以降を11/01まで更新 |
| PF-01〜07 | Yoshida/Morales・processed・係数・finite-time校正/responseを歴史的背景として把握。対象repoの高次PR/共通PF/長時間を直接追跡。別PF repo固有の第一/第二研究は旧カードの二次参照に限定 |
| CIR-01/02、OPT-01〜05、ERR-01 | run-sharing、P-A/P-B/P-C/P-D/R3、RPE配分、FRの成功・負結果・停止を区別 |
| PR-01〜03、EVAL-01〜03 | PR-2/PR-3、未実施初期案、Track A各分子/geometry/精度/rareを把握し、独立確認・高次・長時間/B3で更新 |
| RB-01〜18 | BF/BM、SP、R/RA-D0/G1〜10、DF補正、D1〜3、Parent、M1-S、finite-bit/live API、C1、相関案をcoverage。後続v1.2/C1を直接確認 |
| QC-01〜14 | macrocycle、branch/self-energy、cost-aware、one-body、SOS、相対保証、covariance、H1〜H4/exact encodingを既存カード/重複監査で整理。重要なA2/C4/N3は最終本文まで確認 |
| EXT-01〜08と別repo専用PF | 元研究との関係を旧横断mapで把握。bus/CAR・集団遷移・条件付き測定・unknown-gap等を今回新たに科学審査していない。対象repoの資源比較への直接証拠には使わない |
| 10/23以降の追加系列 | 混在辞書F1/F2、rectangular PS48/終了、DF長時間B2/B3、Pauli4方式/機構を追加。旧70カードを70独立成功例と数えない |

主要な未取得・未保証は次のとおり。

- UWC原JSON/診断rawは後続限定探索でも未発見。旧DF Cgsは無効化前37ファイルを発見したが修正済み入力・科学run来歴は未解消。旧表の値を推定で埋めない。[後続探索][GAPS]
- H8独立確認の詳細公開監査、旧H4 event/未丸めnormalization/親exit、M1-S旧full-H/HD、一部旧review原本は公開証拠の限界として残る。新cohortの成功で補完しない。[A6][A6]、[M1-S][M1S]
- 長時間DFのprivate event/seed/QPY/checkpoint全量、他host原本は今回読解・転載しない。B3の一部event再構成は保存fingerprint等の一致確認であり、失われた元byteの同一性の証明ではない。[B3][B3]
- 理論・十分N・実ρ・母平均tail・state-prep・energy/RPE全体の未実証は、単に「どこかのファイル未取得」とは異なる。全てを追加計算の必須backlogにしない。

## 9. 研究全体として分かったことと次の判断

固定taskでは部分ランダム化が強い決定論対照より安い条件を得られる。一方、norm、次数、random化割合、表現の単純な単調則は支持されず、bias/headroom・normalization・単回価格を同時に扱う必要がある。既知法と同情報の強い対照を加えると、初期の独立寄与や優位が消えた例が複数ある。これらの負結果も研究成果として保持する。

ユーザー/GPTが科学レビューで判断する材料は、(a)現行有限taskの実証研究として何を中心claimにするか、(b)Pauli低Gを会計上の説明までとするか因果機構へ進めるか、(c)DF/Pauli比較とRPE/energy接続のどちらに研究上の追加価値があるか、(d)RA等の構成成果を独立方法論としてどこまで主張できるか、である。本資料はその優先順位を正式採択しない。限定的な追加対照・既存資料だけで済む確認は[未解決対応表](pr_repository_claim_evidence_gap_matrix_20261101.md)に分けた。

作業branchは`codex/pr-repository-scientific-landscape-20261101`、起点4591b8d7。変更は新規2文書だけ。既存main/研究概要/manifest/論文本体、元SOURCE/input/raw/result/grant/ledger/marker/STOP、元研究branchは変更しない。主agentだけで実施し、今回の文書検査を独立科学レビューと呼ばない。文書専用の既存`one`・`source_rows`・`check_links`を再利用し、保存行の文字列と引用pathを確認した。参照辞書83件は固定GitHubファイルを取得でき、local Git blobと一致した。科学runnerや数値validatorは実行していない。

**公開・remote SHAと本文照合後mandatory STOP。次の担当はユーザー/GPT：中心claim、重要な未解決事項、必要なら詳細科学レビューや追加実行の範囲を判断する。Codexの次の実行作業なし。**

<!-- 固定commitの参照。科学実行SOURCEは本文の対応表を参照。 -->
[ORIGIN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0babed07006c4cfc34b2b4191f4f0c8a9e9bceaf/docs/research/研究目的・研究課題.md
[STATUS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/VALIDATION_STATUS.md
[METHOD]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/research/研究方法・解析手順.md
[FINITE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/finite_rte_signal_validation.md
[COEFF]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/pf_c_system_size_validation.md
[COST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/rte_compiled_cost_validation_summary.md
[RPE4]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/rpe_four_round_phase_validation.md
[RPEMODEL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/research_direction_full_opt2.md
[RPEJSON]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/research_direction_full_opt2/2026-09-25/wp11_all_r_opt2_coherent_analysis_20260925_065827.json
[AUDITA]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/research/track_a_h4_h8_cross_system_evidence_inventory_20261017.md
[A6]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/research/track_a_h468_paper_evidence_integration_20261019.md
[A6TABLE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/resource_applicability/track_a_h468_paper_evidence_integration/2026-10-19/main_table_proposed.md
[CNG]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/analytic_circuit_cng_validation.md
[S2CNG]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/recomputed_s2_theory_compiled_validation.md
[PX]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/non_df_h4_comparison_synthesis_20261025.md
[D1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/non_df_h4_d1_source_execution_20261024.md
[D2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/non_df_h4_d2_execution_20261024.md
[ARB]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/non_df_h4_central_arb_pf_pr0_sine_20261025.md
[R2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/non_df_h4_r2c_compiler_phase_proof_20261025.md
[PAULI]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/research/pauli_h2_h4_h6_paper_evidence_synthesis_20261101.md
[PR3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/docs/pr2_pr3_minimal_pilot_validation.md
[M1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e3531956af6a6d359e2f1fb320e9a5179f41eb98/docs/pr2_matched_accuracy_m1_b1_result_validation.md
[M2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e3531956af6a6d359e2f1fb320e9a5179f41eb98/docs/pr2_matched_accuracy_m2_transfer_result_validation.md
[GEOM]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7c4483378047b57f514c47dad98905e3f2a9a756/docs/research/track_a_h4_geometry_results_20261013.md
[B0I]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/98cdb35cd08469714f20d6cc3b982d9c80908f46/docs/research/track_a_h4_b0i_map_results_20261014.md
[H6STRONG]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f89c40af73289af9f4c2b70ed0b3966d95d47b06/docs/research/track_a_v1_p7_measured_reaccount_results_20261016.md
[H8STRONG]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/96e69a01d1fae7a402aac2c4efa570a59f8cc7a0/docs/research/track_a_h8_six_axis_reaccount_run01_results_20261017.md
[H8TAIL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/862d732fba0a5f353c522a9219dec78ae2f21a4b/docs/research/h8_rare_rte_closure_policy_20261021.md
[PF588]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/docs/research/track_a_h246_pf_common_final_20261022.md
[HIGHCONF]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/docs/research/track_a_h246_reconstructed_confirmation_results_20261022.md
[LONG]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/REPORT.md
[LONGCSV]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/155529e20a8fac39e00786c40b57830b71203c4f/artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/independent_comparisons.csv
[B3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ebe9e2aa2c83cd71a6da0a885b78f32274bfb6ed/artifacts/resource_applicability/track_a_h46_long_time_b3/2026-10-26/REPORT.md
[B3PROV]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ebe9e2aa2c83cd71a6da0a885b78f32274bfb6ed/artifacts/resource_applicability/track_a_h46_long_time_b3/2026-10-26/scientific_provenance.json
[B3SRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/638e0a989b4471a075549ccb58617b1030e6748f/scripts/h46_b3.py
[ALL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a9bdcf1d44061daceffe896637e2c92fcb3353b4/docs/research/pr_algorithm_improvement_landscape_20261023/RESEARCH_MAP.md
[CARDS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a9bdcf1d44061daceffe896637e2c92fcb3353b4/docs/research/pr_algorithm_improvement_landscape_20261023/STUDY_CARDS.md
[COVER]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a9bdcf1d44061daceffe896637e2c92fcb3353b4/docs/research/pr_algorithm_improvement_landscape_20261023/COVERAGE.md
[GAPS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8dfdfa6460f9e7d25a9595132a672e8de2a5dc7e/docs/research/pr_algorithm_improvement_landscape_20261023/local_environment_search_20261023/SEARCH_REPORT.md
[BSURVEY]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aafa0f4a42fc0b5780e0716c4bd7c7710bfa4d14/docs/research/track_b_paper_claim_survey_20261018/results_usability_map.md
[BSTRUCT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aafa0f4a42fc0b5780e0716c4bd7c7710bfa4d14/docs/tracks/algorithm_codesign/track_b_resource_structure_and_bottlenecks_20261015.md
[G4]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aafa0f4a42fc0b5780e0716c4bd7c7710bfa4d14/docs/tracks/algorithm_codesign/g4_results_and_gpt_handoff_20261009.md
[G6]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aafa0f4a42fc0b5780e0716c4bd7c7710bfa4d14/docs/tracks/algorithm_codesign/g6_independent_mathematical_audit_20261010.md
[FR]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aafa0f4a42fc0b5780e0716c4bd7c7710bfa4d14/docs/tracks/algorithm_codesign/inputs/fr_research_technical_record_root_snapshot_20261005.md
[G6SRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/28cfabb1d47e0e1824bce1e154a9c289738fa2b9/src/trottertracks/algorithm_codesign/return_aggregation.py
[G7SRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trottertracks/algorithm_codesign/g7_generator.py
[BLATEST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f750a23f023d75c8c4fbcabc6937d6953ce61f2/docs/research/track_b_g6_v1_2_draft_review_integration_20261022/CLAIM_EVIDENCE_MATRIX.md
[G9]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78cb65be373b0808641b5a5e44d9b69913bf84e9/docs/tracks/algorithm_codesign/g9_v2_results_and_gpt_handoff_20261010.md
[D3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f0df9a3b36bc3540f0c83941871fa0d3110ddbf7/docs/tracks/algorithm_codesign/d3_v2_results_and_gpt_handoff_20261014.md
[PARENT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/53c48317d16851333c0e0354b0dfd8ef91e1d524/docs/tracks/algorithm_codesign/parent_allocation_s4_final_comparison_evidence_packet_20261015.md
[C1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/185ec4e9d6591f4ee8d7b047c2697bed7be7bc2e/docs/research/track_b_c1_freeze_synthesis_20261021/C1_RESOURCE_EVIDENCE_AND_LIMITATIONS.md
[M1S]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f079dbe67bb35eff35613857b169fdf812af4b8/docs/research/m1s_integration_20261017/integrated_research_summary.md
[AX]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/docs/research/track_a_ax2b_h4_pilot_v5_result.md
[AXSRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d6510db9326e9335bedd03d0d07c490561e23112/src/trottertracks/resource_applicability/ax2b_native_df_v5.py
[N3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ccc575ee339be091fd4031272c38bee13866f097/docs/research/n3_saved_attribution_v1_claim_and_observability_20261013.md
[QC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f7e103bf3a0ffcee6162bd4b8d244f9a2f4502c5/docs/research/qc_a2_current_route_handoff_20261018/achievements_and_limits.md
[C4]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/60da5738d48e0faaedd3de125b0350301935dc3a/docs/research/qc_c4_finite_rte_method_efficiency_technical_audit_v1_20261014.md
[MIX]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/69f8e1cac23a4924ab81aebe399657a705024f6a/docs/research/pr_gaussian_singleton_closeout_handoff_20261024.md
[MIXCAUSE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3a782bd79967d23c47f27e75907027e2c2e87bf4/docs/research/pr_algorithm_improvement_phase2d_f2_resource_causes_20261024.md
[PS48]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1e20458d742a8fac62ce0d0264188b1a7772ee28/docs/research/pr_rectangular_givens_ps48_execution_20261024.md
[MECH]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5432857b94c3714ea9f4d6031e493e8dfc308a91/docs/research/pauli_pure_rte_cost_mechanism_inventory_20261101.md
[H6HYB]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/00958ae1945c780718b2deef07d5dab1096a5cd8/artifacts/h6_hybrid_fairness_extension/2026-10-27/REPORT.md
[H2PAULI]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4aea054b3369709bc0d35ff081071e62b5f77cbe/docs/h2_pauli_rte_validation_20261026.md
[H4PAULI]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d3fdfad880902b7ff35b710d22e1c9759f7087c2/docs/non_df_h4_pauli_pr_search_20261025.md
[EVENT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/10cb3e5cca7834ebf75301bd91bd093ca51d4a6b/docs/pauli_rte_event_extension_20261026.md
