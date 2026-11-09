# B. 改善候補の比較

2026-11-09 JST。これは既存証拠を踏まえた**未採択の候補**であり、新しい優越・新規性・実行GOの認定ではない。[A：証拠とSOURCE](EXISTING_RESULTS_AND_OVERLAP.md)の固定出典を使う。[C：候補別の最小検証・承認・STOP](MINIMUM_VALIDATION_PLANS.md)と組にして読む。

## 1. 共通の目的関数

状態準備を除いた全round／X,Yの

\[
G_j=\sum_{m,b}N_{m,b}\,\mathbb E[C^{\rm no\text{-}prep}_{j,m,b}]
\]

を資源jごとに比較する。RZ、CX、native DAG depth、serial depth上界、Tは別指標。現Closedの主比較はRZなので、CX/depthだけの改善からRZ2.308629倍の差が縮むとはいえない。主目的を変える場合はユーザーが判断する。

Closedの測定変数は全path weight×Hadamard outcomeであり、現在は認証rangeで規格化した半径rhoと残余位相betaに対するdirected Hoeffding十分Nを使う。Nは真の最小必要trial数ではない。lawを変える、誤差を減らす、安い回路を選ぶ、統計界を変える、という操作は全て最後に同じN×価格へ戻す。旧bottleneckの反実仮想感度を実現可能な節約率としない。

Fullに加え、H4では既存Truncated PFのA/Bを強対照に残す。各候補に同じ合法control／fusion／統計／state情報の機会を与え、PR内部の改善と対照との差の改善を別ablationで示す。

## 2. 独立した問いの一覧

番号は識別子であり、期待削減率の順位ではない。既知原理の新条件への接続と、新しい改善機構の仮説を区別した。

| ID | 具体的な変更 | 主に変える量 | 既存との差／現時点の状態 |
|---|---|---|---|
| K1 | 異basis runの**actual joint completion間のrelative Gaussian橋**を直接合成し、往復basisを置換 | RTE単価、D/R境界、追加native disk | 同basis共有の先。Dに既存relative実装あり。Cのpartial completionへの接続は未検証。既知合成原理の拡張 |
| K2 | 最新Closed P3＋joint/fusionを**vacuum Reference干渉**で読む | control単価、depth、qubit、χ/disk経由N | gadgetは実装済み。最適化済みC-H4 weighted lawとの組合せが未資格。既知法の統合 |
| K3 | 同じ元Hを**決定論D／RTE残差R／誤差予算内の省略Hdrop**へ三分割 | λ/range、RTE event価格、PF/drop誤差、N | 現CのPRは全tail保持、TruncatedはRなし。両端間の三者配分の総費用は未検証。既知法の組合せ |
| K4 | 元H groundを保ったまま**Z再配置／core＋厳密残差／既存DF候補**を接続 | D/R交換、辞書・basis、λ、model誤差、N | RC own-DFやM2 operator-onlyからの未接続部分。汎用rank gridではなく保存候補一組の適合検証 |
| K5 | 同じ測定lawに**range以外の認証情報を使う有限confidence会計**を適用 | 十分N、X/Y・round配分 | 現CはHoeffding。B moment/range・旧短段二項検証あり。新lawの発明ではなく統計保証の拡張。利益未評価 |
| K6 | D内部のh²相殺の先で、**次のground energy補正も制御する組成自由度**を検討 | PF位相余裕、許容h、D仕事量、N | 既存1係数のh²相殺・h⁴観測との差。新PF構成の仮説段階で、まず成立条件の検討が必要 |

K1/K2/K3には直接の既存回路・費用証拠があるが、組合せ後の効果は未取得。K4にはpromise差、K5/K6には新保証・新条件という障害がある。全候補を同じ成熟度と扱わない。

## 3. K1：joint completion間のsequence合成

**変更するもの。** 現`track_c_closed_p3.build_events`は最大same-source-basis runごとに、選ばれた全8mode completionのinverse、中央対角列、completionを連結する。隣接する異basis runの出口／入口を、actual completionの積に対する一つのphase-correct relative Gaussianへ置換する。最初はcatalogueとsupport mappingを固定し、completionの新最適化まで同時に行わない。

**総費用への経路。** law、ordered applications、weight、P3 meanを保ったままbasis価格を減らせれば、RTE総RZ9.599285356e11の構成へ作用し得る。新loweringの誤差でrho/betaが悪化する場合はNを再評価する。理想演算が同じというだけで旧Nを無条件流用しない。

**既存との差・還元。** same-basis run共有、singleton／joint QR、GraySynth、compiler Aの既存inverse相殺はbaselineに含まれる。Dの`StrongBuilder.transition`とC1はfull frameでこの原理を実装済み。ここで異なる問いは**Cのsupportごとに異なるpartial completionを接続した後にも追加費用差が残るか**。新しいGaussian原理・独立新算法という主張ではない。

**利益を消すもの。** relative行列がdenseになり、独立二回の疎partial合成より高価となり得る。列mapping、spin構造、analytic gauge、vacuum phase、compiler A区間資格が必要。多数のpair取得費もある。known fallbackを保持し、SOURCE命令数の減少だけでnative利益を認定しない。

**証拠・負結果。** jointはRZ減・depth増を既に観測し、always-partialは共有を壊して失敗した。D C1実H4一pathの利益は両方式共通でReference固有差は増えなかった。従ってbasis最適化がPRだけを有利にする保証はない。[A §3–5](EXISTING_RESULTS_AND_OVERLAP.md)

**再利用。** Cの`flattened_runs/build_events/catalogue/emit`、Dの`transition/vacuum_phase`とCAR/exterior・actual inverse checker、C compiler A。C1の固定H4/block guardを削除してそのまま一般化したことにせず、actual completionとの必要差分だけを対象にする。最小検証とSTOPは[C-K1](MINIMUM_VALIDATION_PLANS.md#k1)。

## 4. K2：Referenceを最新Closed回路へ接続

**変更するもの。** 旧canonical C-H4やD-H4で検証したReferenceを、最終Closed P3の符号・全path weight・D負時間・joint completion・対角合成へ接続する。uncontrolledな時間発展を実装し、既存split／actual adjoint／vacuum χ／CSWAPで同じ複素測定平均を得る。

**総費用への経路。** evolutionのcontrol負担を追加registerと固定干渉費へ交換する。長roundで単価とdepthを減らせても、新しいχ・native/synthesis diskでNが増えれば相殺される。全trial weightは元のまま掛ける。Referenceをnormalization除去法として説明しない。

**既存との差・還元。** 新A・D・O2のgadgetは既知SE型の同じ系列。新実装を作る理由はない。残るのは**Cの既に安いjoint／diagonal／Closed lawの後にも増分があるか**という統合課題。最新強Full／TruncatedにもReferenceが成立するなら同機会を与える。

**利益を消すもの。** 追加8qubit、CSWAP/reset、cut境界の融合損失、vacuum phase、負時間のactual inverse、control済みnativeからの不正な位相移送。新Aの強B′比RZ増加は警戒材料で、RZ目的ではCX/depthの実績だけを利益根拠にできない。joint completionのgaugeはsource値に結ぶ。

**証拠・負結果。** 新Aは条件付き18round CX/depth上界改善、RZ増。D q64有限8本ではRZも.987%減るが、law・split・精度・compilerが違う。CT-4Bの2 pathのT低下を全roundへ拡張しない。[A §3](EXISTING_RESULTS_AND_OVERLAP.md)

**再利用。** 新A Reference/transfer、D physical segment/reference/checker、C Closed weighted builder／qualify／round policy。C1の未実施人工案を再発行しない。最小検証とSTOPは[C-K2](MINIMUM_VALIDATION_PLANS.md#k2)。

## 5. K3：D／R／dropの三者配分

**変更するもの。** 保存元Hとstateを固定し、H=H_D+H_R+H_dropと分ける。小さい／高価格の成分を全てRへ入れるのではなく、一部を誤差予算内で省略し、残るRだけをClosed P3で扱う。Dへの移動と省略を区別する。第一検証は既存の保持集合一つに限定し、自由な全subset探索を始めない。

**総費用への経路。** random poolから成分を外すとλ、return確率、basis出現、range、必要resolutionが変わる。省略誤差は元H energyへの位相／半径diskを消費しNを増やし得る。安くなったRTEと増えた誤差負担を同じ全round会計で比較する。

**既存との差・還元。** 既存Truncated PFはRを完全に無くす端点、現Closedはdrop=0の端点。その中間の検証であり、truncation／hybrid PFそのものの新規発明ではない。既存TruncatedがH4で安いことを認め、PRを残す必然性を事前に置かない。

**利益を消すもの。** 長時間TによるHdrop誤差の累積、元groundとHkeep固有分枝のずれ、dropとPFの相殺依存、λ低下より小さい回路節約、retention更新費。R=0の強対照が常に勝つなら、PR中間点の研究を続ける理由はない。

**証拠・負結果。** H4のTruncatedは保守BでもFull/Closedより安い。H6ではcombinedと保守BでPRとの順位が変わる。M2の微小residualはoperator-only。純RTEの固定点では分割bias減よりnormalization増が勝つ。いずれも三者方式の成功証拠ではない。[A §3–4](EXISTING_RESULTS_AND_OVERLAP.md)

**再利用。** Cのretention/元H signal・directed drop/PF分離、Closed generator、既存D/native、cost-sensitive policy、M2の残差・identity保持。新Hや新stateを作らず、同じ元Hへ誤差を戻す。最小検証とSTOPは[C-K3](MINIMUM_VALIDATION_PLANS.md#k3)。

## 6. K4：元H ground条件での表現・partitionの再接続

**変更するもの。** 新fit/rank探索を先行せず、保存RC候補の線形Zをheadへ移す表現、又はM2のcore＋保持residualを一つ選ぶ。元H exact groundへ作用させ、元Hとの差を認証予算へ戻す。Z移動が各候補内でHamiltonianを保つことと、候補間の近似Hが違うことを分ける。

**総費用への経路。** 高いsampling質量を持つ線形項を安いone-body Dへ移し、又は安いcoreに高精度残差を添えることでRTE rangeとbasis価格を変える。D増、PF mixed bias、model error、N増まで含めて初めて利益となる。

**既存との差・還元。** UWC型の表現／split／資源共同最適化、DF regularization、Z再配置、collected Pauliは既知又は実装済み。新しい点は**own-DF groundを使わず同じ元H state・energyを保証した条件で利益が残るか**。最適化済みSOURCEの再作成を避ける。

**利益を消すもの。** fit残差とstateの違い、spin/gauge／κ／scalar、D circuitの増加、Fullにも同じDF表現が効くこと、元H whole-law保証不足。M2の物理model許容を今回採択したことにしない。

**証拠・負結果。** RCの保存点値と名目統計は有望だがC-H4へ移せない。A-C0ではattenuationが単価利益を打ち消した。M2は小残差mapperが成立してもPF/RTE/N未取得。主張できるのは「未接続の問いがある」までである。[A §2–3](EXISTING_RESULTS_AND_OVERLAP.md)

**再利用。** RC export/J/U/再配置、M2 retained mapper/CAR、元H/state adapter、Cの誤差・price・trial会計。min検証とSTOPは[C-K4](MINIMUM_VALIDATION_PLANS.md#k4)。K3とは省略の有無と入力表現の変更が異なり、別々のablationが可能。

## 7. K5：同じlawで十分trial数の保守性を減らす

**変更するもの。** samplingも回路も変えず、認証可能なweight二次moment／outcome構造を使う有限confidence界、又はbounded X/Yの直接confidence regionを検討する。現Cのrange-only Hoeffdingを置換できるかという問いで、単なるαの再配分とは別。Bernstein等は既知手法であり新しい統計法の発明ではない。

**総費用への経路。** rho/betaを保ちNが小さくなれば回路単価を変えずGが下がり得る。Fullの±1 outcomeにも同じ改良を与えた後にPR/Full比が縮むかは不明。Closedのほぼconstant weightではM2の追加情報が弱い可能性もある。

**既存との差・還元。** Track Bはmoment/range/biasを会計済み。旧RPE短roundでは厳密二項失敗率を確認済み。一方、最新Closedの全path weightと36軸・branch/energy保証への代替界は、今回確認した`rigorous_shots/qualify`には実装されていない。CV／ISの新研究ではない。

**利益を消すもの。** 真の未知位相・分散をoracleとして使うこと、学習／pilot費、有限bit weightの非二項性、range項、multiple testing／optional stopping、整数ceil。価格MC標本分散はHadamard outcome分散ではない。保存meanの点分散だけでNを減らさない。

**証拠・負結果。** O2のS2はM2/rangeの増加で単価減が相殺された。SCVは学習費込み利益が未確立。これらは新confidence界のno-goではないが、推定器だけ変えれば必ず勝つ根拠にもならない。[A §3](EXISTING_RESULTS_AND_OVERLAP.md)

**再利用。** B finite-bit moment/range、C directed会計／decoder／α、旧short-round二項チェック。まず同情報での十分条件を整理し、成立前に新Nを算出しない。最小検証とSTOPは[C-K5](MINIMUM_VALIDATION_PLANS.md#k5)。

## 8. K6：ground-specificな次補正の制御

**変更するもの。** 現D_vの一係数によるleading h² energy補正相殺を基準に、追加のD-only対称組成自由度で次のground補正を制御できるか調べる。R負時間を増やさず、finite-RTEを含む実信号で評価する。単に同じv/hの格子を細分する案とは分ける。

具体的な検討対象の例は、既存D対称S2をS_Dとして、net時間を保つ既知の5段対称族

\[
D_{a,b}(t)=S_D(at)S_D(bt)S_D((1-2a-2b)t)S_D(bt)S_D(at)
\]

を同じD/R sandwichへ置くもの。二自由度でleadingと次のground energy補正を制御できるかが問いで、係数を解いた・解があるという主張ではない。次補正には非零H2の非対角混合に由来する寄与も含める必要があり、H4の期待値だけを消せば足りるとは仮定しない。その情報を元ground供給の仮定だけで安価に得られるかも未解決である。

**総費用への経路。** PF位相余裕を増やせれば大きいouter hでD呼出しを減らせる可能性がある。ただしRTE event数・減衰が支配する現在、PF精度だけの改善はGへ効かない場合がある。追加D段数・負D時間・係数感度も価格へ戻す。

**既存との差・還元。** 対称合成、BCH、状態依存error設計は既知。現研究でh²が消えてh⁴傾向を確認した先の限定仮説であり、新しい一般高次作用素公式を得たものではない。既存外側5段族のa=1/4下限を破ったと主張しない。

**利益を消すもの。** 追加段でDが重くなる、RTE位相と振幅は残る、誤差の偶然相殺やwrapped終点位相への過適合、強Fullにも同自由度が効くこと。未知E0でdecoderを補正しない。古典設計に必要なstate期待値アクセスの範囲も事前に決める。

**証拠・負結果。** 動機は既存D内部h²相殺のみ。D-only Y4／混合2MN／固定processingは既存負結果。次補正の消去可能性、安い係数、科学的独立性は未確立で、現時点では**数値実行を推薦できる完成候補ではない**。

**再利用。** `track_c_d_*`のBCH展開・schedule・nonresonant検査・finite mean・D価格、研究Bの高次／Exact-R対照。まず有限の組成と満たす条件を記述し、旧法と同値なら終了。最小の次作業とSTOPは[C-K6](MINIMUM_VALIDATION_PLANS.md#k6)。

## 9. 改善機構と区別して残す事項

| 区分 | 対象 | 次作業を提案する場合の条件 |
|---|---|---|
| 既存パラメータの再探索 | h/resolution/αの更なる細分、未価格116条件、D順序追加 | 現共同最適化済み。未検証セルがあることだけを新機構にしない。限定拡張を選ぶなら保存SOURCE・新training/holdoutを固定し別許可 |
| 既存familyの拡張 | Closed P5/K4、r/K共同最適化、Full/PR混合round | P5とP3は別finite target。G7/G9と旧K gridがある。m5移植・分子native資格は未完でも、次数変更そのものを独立新算法と呼ばない。全体比較が必要 |
| 目的資源の変更／再選択 | joint catalogueをdepth/CX目的で再選択、Clifford+T精度配分 | RZ policyと別。C2-Bの約.514%差はFullの有限pool。新Aの有限T例も全18roundTではない。主資源を先に判断 |
| 価格資格の補完 | D長q構造、q≥128、全support期待価格、T0→P0 bridge | 科学的改善機構の実証と別。q64完了やT0人工PASSだけで旧13roundを更新しない。既存resource-eval系列を優先 |
| 現状HOLD：whole-path sampler | previous basis/support等の有限状態scoreで安いpathへlawを寄せ、全path likelihood weightでmean保持 | 既知IS/weighted automataへ還元され得る。最新O2はcheap conditional summary未特定。同じ提案の実装推薦を繰り返さず、安価なscore部分和と実生成が具体化した場合のみ再判別 |
| 現状HOLD：CV/LCU/instrument | Born noise／測定accessの変更 | 最新O2は新実装推薦0。新たな有料測定accessの構成と強対照との差がない限り再開しない |
| 周辺候補・目的変更 | QSP-PE等RPE以外の推定器 | finite weighted PR平均とideal unitary queryのgap未解決。「PR＋RPE」のままの改善と分け、family変更の採否をユーザーへ戻す |

## 10. 候補の独立性と組合せの順序

- **K1、K3、K5**はそれぞれ回路・近似分割・confidenceに作用するため、別の固定namespaceと担当で並行に問いを判別できる。ここでの並行研究は将来の研究計画であり、今回sub-agentやworkerを起動した意味ではない。
- **K2**は独立にcontrolを扱うが、K1とbasis／boundary費が重なる。単独2×2のablationを済ませてから統合し、削減率を足さない。Full／Truncatedへの共通利益も同時に見る。
- **K4**はK3と違い表現／oracle bindingを変える。元H接続を閉じるまではK3の固定H結果やK1のcompiler diskへ移さない。Z移動とresidual方式も別armにする。
- **K6**は先に組成の成立性が必要。K1–K5の未検証結果を前提にしない。新PFでsignalが変われば旧Nも再資格となる。

追加効果を最小に判別しやすい案はK1の固定2-run接続、又はK3の保存retention一組である。これは作業の具体性に基づく提案で、期待利益の順位でも正式採択でもない。K2は主目的がCX/depthなら別線として自然だが、RZ目的では既存のRZ増加証拠を先に扱う。どの候補でも強Full逆転を前提にしない。
