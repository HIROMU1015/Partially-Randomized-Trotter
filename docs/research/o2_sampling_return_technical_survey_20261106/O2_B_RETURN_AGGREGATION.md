# O2-B：Return Aggregationを全RPE費用へ結ぶための技術照合

2026-11-06 JST。静的SOURCE読解、既存結果の引用、既知構成との対応整理。新しい計算・再集計・回路生成・合成・実装・科学レビューは行っていない。比較候補の研究採択、新規性、論文化可能性の最終判断も行わない。統合は [候補比較](O2_CANDIDATE_COMPARISON.md)、統計側は [O2-A](O2_A_SAMPLING_STATISTICS.md)。

**残る問いは、returnの構造から得られる係数集約・生成アクセス・回路相殺が、既知の低次数閉形式と最適化baselineを認めた後にも、必要shot数を含むfull-wrapper費用を減らすかである。** G6の構成、G7〜G10の限定native会計、C1のlogical接続は既に存在する。再実装を次の課題にしない。現在、分子の全RPE費用改善を確定する証拠はない。

## 1. 比較する対象・費用・情報

元の物理Hamiltonianの厳密基底状態 \(H|E_0\rangle=E_0|E_0\rangle\) を全方式で利用する。状態準備は費用から除く。\(H_D\) や近似DF Hamiltonianの基底状態へ置き換えず、真のenergy・gap・完全なstate vectorを無料の入力に加えない。厳密な初期状態でも、PF/finite-RTE/native実装のbias、normalization、半径低下、RPEのalias/branch失敗は残る。

\[
G_{\rm total}^{(a)}=\sum_{m,b\in\{X,Y\}}N_{m,b}\,
\mathbb E[C_{m,b}^{\rm full,no\text{-}prep,(a)}].
\]

RZ、CX、T、depthは個別に比較する。古典setup/online時間、memory、追加qubitも別座標とし、換算規則なしに加算しない。ここでのdepth合計は総wall timeではない。新しい分子、精度、成功確率、compiler、主資源指標は選んでいない。

G6 Fullのzero-fillでは、既存SOURCEの `N` はzeroを含む全planned classical estimator trialの数である。実行する量子interrogationはaccepted trialだけで、両者は区別する。上式へ全trialを使うなら、完全に事前判定できたzeroの量子価格を0、既に実行したprefixがあるならその実費を残す。accepted条件付き価格を使う場合は、その受理数の期待値と対応させる。zeroを分母から除いて無補正平均を作ること、受理率を二度掛けることはできない。`WholePath.assemble` / `AllTrials.add,evidence` と `rte_development.accounting.whole_cost` がこの区別の既存入口である。[C45]、[D-SOURCE]

## 2. G6〜G10・API・C1の到達層

以下は保存結果の引用であり、今回のtest・科学実行の結果ではない。固定実行SOURCEと結果carrierは別の役割として記す。

| 系列／固定条件 | 確認済みの到達点 | その結果では確認できないこと／負材料 |
|---|---|---|
| **G6**：正有理 \(p_i\)、Hermitian involution \(Q_i^2=I\)、odd cutoff、\(0<x\le1\)、有限 \(P_m\)。分子・geometry・basis・DF rank・\(L_D\) 非該当 | 形式returnの集約、even-suffix/child pairing、ordinary-envelope rejection、zero-fill、有限bit補正の数理構成。人工形式testが係数・位相・語順・supportを確認 | nativeの未実装という当時の記述は後続G7〜G10で部分更新。G6単独は物理回路・分子性能を示さない。Green再帰、Euler pairing、zero-fillは既知部品。[G6]、[G6-ART] |
| **G7**：P3 control \(p=(3/7,4/7),x=2/5,m=3\)、P5 development \(p=(1/5,3/10,1/2),x=5/7,m=5\)。Re/Im許容誤差各1/200、16軸familywise失敗上界1/20、H160/K256、\(\eta=\rho=10^{-12}\)、strict RZ誤差 \(10^{-6}\)。分子条件非該当 | helper＋exact controlled-Qという条件付きprovider IR、finite-bit moment/rangeによる十分N、取得RZのT費用、未知provider価格を非負変数として残したaffine比較。固定P5ではFullのT会計が登録3対照より小さい | controlled-Qの物理価格・誤差は条件。P3ではclosed P3がFullより小さい。全回路compileでなくIR加算。hard trial capと古典生成費はFullで増え得る。[G7] |
| **G8**：同じ既知P3/P5、同bit/RZ精度。provider誤差 \(\delta=10^{-6}\) は仮定、16軸0.049＋8資源行0.001の失敗配分 | 非列挙local生成→on-demand strict RZ miss取得→actual adjointの接続。有限cold/warm trace、conditional accepted-call cap。保存RZ表を用いた既知cost-aware ISの補助診断も存在 | traceは全support、全angleの取得成功・終端保証でない。G7と同backend/seedの20 sequence再利用を独立確認に数えない。P3のFull/closed差はI2列挙IS診断で消えたという既存記録。未知Bをbudgetへ無料に入れない。[G8] |
| **G9 v2**：上記P5、既知3-system-qubit synthetic provider、11行22軸、共通familywise0.05、同H/K/primitive精度 | direct nativeとhelper診断、matched CTS、全登録eventのweighted operator meanとphase検査、native T/CX/1Q、十分N。closed P5が登録direct集合のT切片とprep/readout係数で小さい | CTSはCXで強い。全資源支配・最良compiler・任意IS最適性・分子結果ではない。Fullとclosed P5は同じideal accepted ensembleで、固有の量子event単価差を示す比較ではない。[G9] |
| **G10 v3**：\(p=(1/5,3/10,1/2),x=5/7\)、同synthetic provider、m=3/5/7別、17行34軸、各軸誤差1/200、familywise0.05、同finite-bit/strict native精度 | degree別の有限 \(P_m\) 内でnative価格・moment/range・N・受理率・資源上限の会計完了。m5の6行/19keyはG9由来 | **異次数間で同exponential精度の比較ではない。** m3でFullはclosed P3より、m7でFullはclosed P5+tailよりT費が大きい保存表。fixed-dictionary lowerは実行可能な最適law取得ではない。旧RSS/JSON停止は技術失敗として保全。[G10] |
| **独立人工adapter受入／512 correctness／限定16,896 pilot**：固定synthetic P5、実Core/API、H160/K256等の各元契約 | 人工受入、固定実APIの採択invariant・codec・保存・有限workload費用という別の技術層。512と16,896は別scope | 512はsynthetic入力で**実API**を呼んだ有限correctnessであり、分子全体・IID・native性能保証ではない。reportの `artificial_only=True` とtrialのfalse/PINNED_G7_F3の矛盾を保持。件数だけで証拠を格上げしない。[B3-512]、[B3-PILOT] |
| **C1**：linear H4、1.00 Å、STO-3G、legacy DF12、R1 generation-prefix \(L_D=3\)、8system qubits、13basis/324labels、\(T=4/5,q=1,r=4,m=5,h=1/5,\sigma=+1\)、H2048/root4096、\(\eta,\rho,\tau_{angle}=10^{-12}\)、angle root256/terms64 | R1入力・basis packet、R2の固定6event correctness、3方式各1024 whole logical paths。Ordinary/Fullは旧runの完了substage、closed P5は別P5R完走。`TAIL_ONLY_PLUS_SCALAR` の探索的記録 | 元physical Hへのrigorous binding、strict native全support、共通認証N、D halves、readout/full wrapperは未完了。zero回避なし、degree0主体、logical差小、古典費用あり。**正式一時凍結を維持。**[C1-R1]、[C1-R2]、[C1-P5R]、[C1-PAUSE] |

G9/G10のproviderは \(Q_0=Z_0\)、\(V_1=R_{XX01}(\pi/4)\)、\(V_2=R_{XX12}(\pi/4)R_{ZZ01}(\pi/4)\)、\(Q_i=V_i^\dagger Z_iV_i\)。明示Pauli情報を安く取得できるI1 contextであり、一般DFでの同じアクセスを取得した結果ではない。G9 SOURCEは `0ef2b92738750a3c0d187743dd4b7c9b927db802`、G10 SOURCEは `b9ed01455351628c9073748f5ba5751aa794b789`。結果carrier `4301c0f87a16be45b620304c3fa935d5aa2d721a` に収録されたSOURCE本文と指定ファイルを照合し、今回読んだ `g9_p5.py/g9_native.py/g10_generator.py` にbaseとの差分がないことを静的に確認した。旧runの再現・実行互換性を確認した意味ではない。

## 3. return集約が変更する処理

| 構造／SOURCE接口 | 消える・減り得る処理 | 追加で必要な処理／全費用への境界 |
|---|---|---|
| `reduce_word`、`ReturnKernel.word_series/mass/parent` | \(Q_i^2=I\) によるreturnを係数側で集約し、冗長なsigned word寄与を減らす | 形式語の係数query、正確な符号・位相・積順序。物理辞書の追加関係を全て使う構成ではなく、Pauli collectionより常に強いわけではない。[KERNEL] |
| even-suffix parent＋child rotation | 複数次数の寄与を一つの回転とreduced parent積へまとめる | parent別の角度、child分布、root・atan/合成、\(i^{|u|}\) 位相。odd語全体をinvolutionとして回転する構成ではない。[KERNEL]、[G7-SOURCE] |
| `FullReturnGenerator.sample` | 非reduced raw proposalまたはcoin rejectを量子実行前のzeroにできる | 全trial分母、zero生成・判定の古典費、ordinary envelopeに由来するrange/N。raw語をそのreduced語へ再割当するとproposalを二重計上する。`None` は全returnが恒等演算という意味ではない。[G7-SOURCE] |
| `P5Closed`、`ClosedP5Tail` | 低次数の全return係数を群化し、Fullの局所棄却・保守的normalizerアクセスを避ける | P5群normalizer、completion DP、群selector、finite-bit weight。群構築はO(L²)だが現linear scanもあり、全sample O(L)という保証ではない。angle群だけでwhole-path basis相殺価格は決まらない。[P5-SOURCE]、[G10-SOURCE] |
| 同じreturn／parentを共有するevent | 同じ回転key・basis descriptor・係数queryを古典cacheへ再利用できる | **独立shot間の量子gateが一回で済むことにはならない。** 同じideal returnでもchild/phase/order/control/精度が違えばnative keyや回路は違う。cacheのcold/hot条件をbaselineにも対称に与える |
| DF basis sharing | 隣接する同一basisの実逆対、reduced signed-Qとchild rotation、event間のbasis境界を相殺できる場合がある | 全体のordered IRを見た比較が必要。G9 `simplify` は隣接inverseのliteral規則、`native_ir` は指定3qubit provider。DFのfull/support-runやD境界へ無条件移植しない。[NATIVE-SOURCE]、[C1-BASELINES] |
| controlled rotation | G9 direct loweringではbasisを無制御にし中央操作だけ制御できる。generic helperより安い場合がある | 全方式に同じdirectを認める。helperのCQ二回＋RZ二回/CX二回とdirect DFの費用境界は異なる。actual adjoint、global scalar由来ancilla位相、active/inactive blockを保持する。[PROVIDER]、[NATIVE-SOURCE] |
| finite-bit／native synthesis | 正確なdyadic proposalに対するrational weightで係数を補正し、bounded biasを会計できる | root midpoint・atan・native誤差が残る。`q*w=coefficient` はideal irrational係数と厳密同一という意味でない。全supportの誤差被覆、weight増幅、反復積の非unitary normを扱う。[KERNEL]、[D-SOURCE] |
| deterministic PFとfull wrapper | 同じsplit/step/scheduleの中央RTE表現だけを変更する場合、Dのtarget自体を変えずに接続する入口がある | D half-step、step間・外側境界、ancilla X/Y読出しの価格・誤差を戻す。中央単価差をfull価格の差へそのまま足せない。r/step/splitまで変更するならPF bias・D価格も変わる。[C1-BASELINES]、[DF-BUILDER] |

元のPR finite meanが同じでも、その分解を集約するとevent unitary family・係数質量・proposal/weightが変わる。O2-Aの「同じpath集合・同じ目標平均・range-only十分N」の限定障害を、表現まで変わるRA一般の不可能性に拡張しない。一方、同じ補正meanだけではweighted ±1/zeroの測定分布・variance・十分Nは一致しない。同じraw meanを持つ**非加重±1単一測定**ならmarginal分布も決まるが、G6のweight/zeroを含むestimatorはその限定条件の外にある。

## 4. 統計と全roundへの接続を省略できない理由

既存 `g7_generator.budget` は既にmoment/rangeを用いるBernstein型十分Nを持ち、未知のexact signalやFullの全normalizerを無料で使わない。Fullではordinary envelope BとUから \(M_2^+\) を作り、closed構成では自身のBを使う。これはvariance-aware会計の実装入口であり、分子全RPEに適合済みという意味ではない。[G7-SOURCE]

D1〜D3は固定 \(r\) 内の同じ \(P_5(-i\tau R/r)^r\) について、finite meanの非unitarityを許した積誤差、weight積、moment/range、十分N、native価格の接続を既に行った。`whole_budget` は独立stepのmoment/range積、`whole_cost` は全pathの事前zero判定と未相殺加法価格を前提にする。`cross_step_fusion=False` を消してwhole-wrapper最適化済みと読まない。D系列はr=1/2の固定時間toyで、全round RPEのenergy成功条件ではない。[D-ART]、[D-SOURCE]

必要な将来の接続は、同じ元Hの各roundについて、PF/finite-RTE/finite-bit/native誤差を控除した統計残余、正のsignal半径下界、X/Y別Nと失敗配分、共通energyへのbranch/alias条件を満たした価格を使うことである。元Hの厳密状態でも近似PFの固有状態とは限らない。raw半径によるnormalizationのshot増加とcorrected weight rangeによる増加を二重に数えない。[RPE-COSTMAP]

`rpe_resource_accounting.py` の `_hoeffding_shots/evaluate_rpe_round_candidate` はbounded ±1の既存入口であり、weighted/zero-fill lawへ式だけ流用できない。条件付き18round成功・energy接続の既存結果は保持するが、RAへそのまま移した保証でも最終費用優越でもない。[RPE-SOURCE]、[RPE-COSTMAP]

## 5. 強いbaselineと既知手法との差

| 対照 | 同条件で認める改善機会 | 今回の位置付け |
|---|---|---|
| 通常PR/ordinary finite RTE、returnなし | 元paired sampling、同じstrict phase・precision、合法なbasis共有・cache・selector・compiler最適化 | 非列挙ordinaryを全word展開実装で代用しない。RAの古典優位を人工的に作らない |
| partial return、closed P3＋tail、closed P5/fullまたは＋tail | 同じ有限target、同じ情報、必要normalizer取得費、低次数fast path | P3＋tailはtargetをP3へ緩めた対照ではない。小mのFullの追加費を明らかにする既存の強対照 |
| Pauli collected CTS/SCU、既知return吸収 | 対象Hamiltonianで可能なcollection、layered/Markov生成、同じfinite target、位相・identity correction・取得費 | CTSを常に全support展開が必要な弱対照としない。G9のCTS CX優位を保持。H4のPauli取得費を無料にしない。[G6-ART]、[G9] |
| resource-optimal IS、whole-path IS、Parent/S4 | 同じevent/price/moment情報と前処理予算、同じfinite-confidence規則 | \(p/\sqrt c\)、Euler配分、訂正KKTは既知。未取得のglobal price oracleを新法の貢献にしない。[PARENT] |
| native回路最適化 | actual adjoint・relative phase・direct中央control・basis共有・D境界相殺、採択した同じcompiler条件 | 通常RZ/CXモデルとstrict T/native実費を別に比較。candidateだけに最適化を与えない |
| 決定論symmetric PF、適合する既知高次PF等 | 同じ元H・state・energy精度・成功確率、同じ合法な最適化・誤差情報 | 同finite \(P_m\) 対照とは別に、最終exponential/energy taskで必要になった時の強対照。今回その新価格・step数は求めない。[C1-BASELINES] |

G6のfree-product Green式はAomoto–Katoの既知構造の \(\mathbb Z_2\) 特殊化、pairingは既知Euler構成、zero-fillとcost-aware ISも既知である。ここは [G6の既存一次資料照合][G6-ART] を再利用し、新たなpriority監査・独立同値性証明を実施しない。本調査で残すのは、既知構成の適用・改良としての取得可能性と全費用接続である。

Parent/S4はcommon配分が係数L1を最小化し、noncommon配分はnormalizationを増やしながらnative角度を安くし得るという既存交換を扱う。best-to-bestの利益にはproposal変更も入り、allocation単独の効果とは限らない。訂正ordinary KKTは同一proposalでT改善を示しておらず、online費用と多step実証も不足する。同fixture追加探索は既に区切られている。状態準備除外はS4の未知full readout/D/backbone価格を無料にする条件ではない。[PARENT]

## 6. C1で研究判断を変え得る不足を選ぶ

**C1は再開しない。** 以下は旧UNKNOWNの全解消計画ではなく、将来ユーザーが別scopeを選ぶ時に必要な判断材料である。

| 不足／既存証拠 | 判断への影響と最小の確認対象 | 今は不要な拡張 |
|---|---|---|
| physical DF binding：R1の新basis/324 exact-dyadic係数、数値再構成支持、`MODEL_BINDING_UNCLOSED` | 同じ元Hのenergyを主張する前提。保存DFとR1の定義・scalar・D/tail・stateの対応、既存norm allowanceが必要精度へ使えるかを限定照合。数値残差をrigorous certificateへ昇格しない | 旧係数135差や旧basis来歴の全復元を無条件の必須作業にしない。旧失敗原因を閉じるための再実行もしない。[C1-R1] |
| strict native全support：R2は6event、C1資源はlogical | logical差が小さい時、T合成価格・basis/control相殺が結論を変える可能性を判別する対象。将来許可するなら、同一物理provider/精度でfull supportへ被覆できる既存primitive境界か、適格な期待価格取得法を先に選ぶ | 観測angleだけのnative価格を全law期待値へ代用しない。全support列挙、未取得key一括合成を既定の必須作業にしない。[C1-R2]、[C1-BASELINES] |
| 共通N：C1の1024は探索path数、経験的weight²/maxは認証moment/rangeでない | shot増加が小単価差を消すかを決める。canonicalとFullに共通target/総bias/finite-confidenceを適用し、実行可能情報だけでNを比較する必要 | 保存1024から新N・variance推定・rankingを作らない。旧G7/G9の十分NをH4へ移さない |
| D half-step・readout・full wrapper：`TAIL_ONLY_PLUS_SCALAR` | D価格で中央差が希釈されること、D/basis境界相殺が符号を変えることを判別。元H厳密stateで同じfull scheduleへ戻す必要 | state-prepを再価格化する必要はない。`WholePath.circuit(before,after)` のhooksがあるだけでD実装/価格取得済みとはしない。[C45] |
| 生成・normalizer・return取得費 | quantum差が小さい時の実行可能性を決める。Full全microstep packet費、P5群setup、selector、cache miss、必要bit精度を対称条件で比較する必要 | 旧Ordinary/FullとP5RのCPU/wall境界が異なる保存値から新しいspeedup・償却点を計算しない。監査serialization費を量子gate費へ混ぜない。[C1-LIMITS] |

固定C1では全方式でzero pathを観測せず、degree0が大半で、**毎pathの4 child rotationsは残った**。empty parentはno-opではない。logical差が小さく、Fullの局所packet費とP5 setup費が無視できないという既存の負材料は、新native結果がない現在も候補継続の根拠を制限する。別method bit keyの有限観測でありpaired superiorityを認定しない。旧P5 workerの直接exit原因UNKNOWNや512 metadata矛盾を閉じても、それ自体は全RPE費用改善の証拠にならない。[C1-LIMITS]

## 7. 残す具体的な問い・最小検証・終了条件

これは実行しない検証案であり、新しいsampler、定理、数値条件、研究契約の正式設計ではない。最終候補の分類は [候補比較](O2_CANDIDATE_COMPARISON.md) にまとめる。

| 候補 | 改善機構と統計／RPE条件 | 未解決点・必要なら別途行う最小検証 | 技術的に継続する条件／終了を検討する条件 |
|---|---|---|---|
| **R1：構造化returnの集約をfull-wrapperへ接続**（独立方向） | 冗長係数とparent積をまとめ、event単価・normalization/momentの交換を改善する既知RAの適用。元finite meanとphaseを保ち、全roundのbias・半径・X/Y Nへ戻す | まず入力に存在するreturn構造と既存low-order fast pathで処理済みの範囲を静的に区別する。将来科学許可が得られた場合、ユーザーが選ぶ一つの同target条件で、ordinary・適合closed baseline・Fullのcommon NとD/readout込み価格を同時に検証。代表eventだけなら技術接続までとし全law主張はしない | strong baseline後にもN×full価格の削減が残り、取得費が実行可能なら続行材料。closed方式と同じensemble/価格でFullのaccess・N負担だけが増える、又はbackbone/shot費で差が消えるならその候補を終了する材料 |
| **J1：return状態が非加法path価格を安く参照できるか**（Samplingと組合せ） | parent/child/basis境界の構造を、既知whole-path ISまたは既存allocationへ渡す。安いpathを増やすだけでなくrange/moment/有限confidenceと価格の積を比較 | `ReturnKernel` の局所queryは全path価格oracleではない。有限状態/群で、normalizer・価格情報・実生成が得られるかを最小に照合。未取得IR価格を無料とせず、十分Nへ使う情報とclassical preprocessingを共通に固定する必要。実際の再配分/保存再会計/compileは別許可 | 既知IS＋closed RAに還元される部分は応用として扱う。追加の取得可能性が確認でき、全費用の交換を説明できれば検証価値の材料。全support価格取得が前提で古典負担が支配する、又は同情報baselineで差が消えるなら終了材料 |

pre-quantum zeroの全path回避はR1/J1で利用し得る**既存部品**であり、独立した新algorithm候補に数えない。全microstepを先に生成し真のzeroが分かった場合だけfull circuitを省略できる。streamingでDやprefixを先に実行した場合は回避できた部分だけ数え、全古典trialの費用とweight積を残す。[C45]、[D-SOURCE]

追加検証の初手は「旧C1を再度同じ1024pathで回す」「新degreeや新分子を探す」「全UNKNOWNを埋める」ではない。既存負材料とlow-order対照を見て、**どの不足が費用差の判断を変えるかをユーザー／GPTが選ぶこと**である。同C1条件の延長を支持する新しい具体的理由は、本調査では取得していない。

## 8. 再利用するSOURCE・test・pipelineとSTOP

| 資産／実読した関数・test | 再利用できる範囲と保留する境界 |
|---|---|
| `return_aggregation.py`：`reduce_word/ReturnKernel/root_interval/dyadic_distribution/rational_event_weight` | formal return係数、有限rational packet、support/phaseの接口。Hamiltonian・native価格・full normalizerは生成しない。[KERNEL] |
| `g7_generator.py`：`CanonicalGenerator/FullReturnGenerator/budget`、`g9_p5.py`：`P5Closed`、`g10_generator.py`：`ClosedP5Tail` | 非列挙productionとreference列挙の区別、既存moment/range予算、closed群。samplingは今回実行せず、大L/分子適合は未確認。[G7-SOURCE]、[P5-SOURCE]、[G10-SOURCE] |
| `g7_provider.py`、`g9_native.py`：`controlled_event_ir/native_ir/CQ/crot/simplify/cost` | 条件付きhelperと指定synthetic direct providerのphase/費用接口。一般DFへproviderを偽装せず、strict nativeと通常compiled価格を区別。[PROVIDER]、[NATIVE-SOURCE] |
| `rte_development/accounting.py`：`step_bounds/whole_budget/whole_cost` | 独立step・非unitary有限mean・全trial/zero・未相殺加法会計。D系列の後続改善は各固定SOURCEへ戻し、固定toy契約を新taskへ強制しない。[D-SOURCE]、[D-ART] |
| `parent_allocation.py`：`reweight_events/event_statistics/error_accounting/native_plan`、`parent_allocation_native.py`：`native_accounting` | explicit-table ISでcheap global samplingを主張しない。位相を保持したnative接口、誤差不足はUNKNOWN。S4訂正と同proposal比較を再利用し、閉鎖したtoy探索を再開しない。[PARENT-SOURCE]、[PARENT] |
| `whole_path.py`：`WholePath/AllTrials`、C1 R1/R2/P5R保存文書 | whole trialの積・all-trial分母・prefix実費・before/after hooks。現版の `SYNTHETIC_ONLY/confidence_certified=False` をproduction保証へ変えない。[C45] |
| `test_g6_return_aggregation.py`：`test_accept_only_average_changes_mean/test_reducing_raw_proposal_would_double_count/test_digital_packet_support_weights_and_l1_budget` | zero分母、proposal、有限bit、語順の既存人工対照。今回実行していない。[G6-TEST] |
| `test_g9_p5_and_native.py`：`test_same_ideal_Green_coefficients/test_physical_CQ_direct_phase/test_phase_corruption_detected`、`test_g10_degree_preparation.py` のdegree/下界対照 | ideal ensemble同一、synthetic controlled phase、lowerと実行lawの区別の既存test SOURCE。実分子full-supportの検証に読み替えない。[G9-TEST]、[G10-TEST] |

指定の3環境統合5資料の関連行・profileと、PR＋RPE文献調査6資料を入口にし、上記の原固定文書・SOURCEへ戻った。統合台帳の129行は実験件数ではなく、全143資産群を再実行・全文精読した意味でもない。固定CSVのID/SOURCE/path/statusの抽出、Git本文差分の確認だけを行い、rawや科学値から新しい集計・順位・予測を作っていない。[INTEGRATED]

**STOP**：C1一時凍結、G6〜G10・D1〜D3・Parentの旧STOP、SOURCE/raw/result/grant/ledger/marker/one-shot条件は保持する。この文書は再開許可ではない。次の担当はユーザー／GPTで、R1/J1の問いを採択する必要性と最小対象を判断する。重要科学レビューの開始許可と、新実装・新科学計算の許可はそれぞれの範囲で別途必要。Codexは本調査の公開・取得照合後にmandatory STOPし、自動で次段階へ進まない。

## 固定参照

本文中のSOURCEは静的読解、結果は保存記録の引用である。公開済み固定pathだけを記す。過去レビューへのリンクは既存判断の帰属であり、新レビューを実施した意味ではない。

[INTEGRATED]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/3571477093441ac4852166fe829f8ccc5db5ec4f/docs/research/pr_algorithm_improvement_three_environment_integration_20261105
[G6]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/28cfabb1d47e0e1824bce1e154a9c289738fa2b9/docs/tracks/algorithm_codesign/g6_results_and_gpt_handoff_20261010.md
[G6-ART]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/28cfabb1d47e0e1824bce1e154a9c289738fa2b9/docs/tracks/algorithm_codesign/g6_prior_art_and_method_delta_20261010.md
[G7]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a689694080f4b7600fe67cf77d841d1cbbf04503/docs/tracks/algorithm_codesign/g7_results_and_gpt_handoff_20261010.md
[G8]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/g8_results_and_gpt_handoff_20261010.md
[G9]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78cb65be373b0808641b5a5e44d9b69913bf84e9/docs/tracks/algorithm_codesign/g9_v2_results_and_gpt_handoff_20261010.md
[G10]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/g10_v3_results_and_gpt_handoff_20261010.md
[B3-512]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/22ef92cef31f98e3e31f27a209405f81f5c4a0e4/docs/research/track_b_g6_w2c_b3_correctness_512_authorized_run_20261020/handoff.md
[B3-PILOT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f27e63b78518c9e807b91f0028fd29078439ec4f/docs/research/track_b_g6_w2c_b3_limited_pilot_16896_authorized_20261020/handoff.md
[C1-R1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6b48bb3f4f7b080baca15d6fa45711619880d64d/docs/research/track_b_c1_1_r1_new_partition_exact_model_20261021/handoff.md
[C1-R2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/068f0ae8771fccf28c529bda12a84abe8b7e7648/docs/research/track_b_c1_3_r2_reference_recovery_c1_4_20261021/handoff.md
[C1-P5R]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bb9059bacb697e1105e63c1b3826b0c34886ad15/docs/research/track_b_c1_5a_p5_worker_recovery_20261021/handoff.md
[C1-PAUSE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/185ec4e9d6591f4ee8d7b047c2697bed7be7bc2e/docs/research/track_b_c1_freeze_synthesis_20261021/C1_PAUSE_AND_RESTART_CONDITIONS.md
[C1-LIMITS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/185ec4e9d6591f4ee8d7b047c2697bed7be7bc2e/docs/research/track_b_c1_freeze_synthesis_20261021/C1_RESOURCE_EVIDENCE_AND_LIMITATIONS.md
[C1-BASELINES]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/068f0ae8771fccf28c529bda12a84abe8b7e7648/docs/research/track_b_c1_3_r2_reference_recovery_c1_4_20261021/C1_4_NATIVE_BASELINES.md
[C45]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/af75af03c0b334c29b5a4311bd2b4578a999460c/docs/research/track_b_c1_4_5_c1_5_technical_preflight_20261021/whole_path.py
[KERNEL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/28cfabb1d47e0e1824bce1e154a9c289738fa2b9/src/trottertracks/algorithm_codesign/return_aggregation.py
[G7-SOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ab2549f41b3546fb3940342a2162dd9ee93699c4/src/trottertracks/algorithm_codesign/g7_generator.py
[P5-SOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0ef2b92738750a3c0d187743dd4b7c9b927db802/src/trottertracks/algorithm_codesign/g9_p5.py
[G10-SOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b9ed01455351628c9073748f5ba5751aa794b789/src/trottertracks/algorithm_codesign/g10_generator.py
[NATIVE-SOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0ef2b92738750a3c0d187743dd4b7c9b927db802/src/trottertracks/algorithm_codesign/g9_native.py
[PROVIDER]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ab2549f41b3546fb3940342a2162dd9ee93699c4/src/trottertracks/algorithm_codesign/g7_provider.py
[D-SOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trottertracks/algorithm_codesign/rte_development/accounting.py
[D-ART]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d3a232a446062e6baadef2b69eab87faa0285363/docs/tracks/algorithm_codesign/d1_d3_prior_art_claim_evidence_and_gpt_handoff_20261014.md
[PARENT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/track_b_parent_allocation_research_closure_20261015.md
[PARENT-SOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4ccc1832f63cd323b23fc05623f7e0b58cf34f37/src/trottertracks/algorithm_codesign/parent_allocation.py
[DF-BUILDER]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/src/trotterlib/df_rte_qiskit.py
[RPE-SOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/src/trotterlib/rpe_resource_accounting.py
[RPE-COSTMAP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/research/pr_rpe_end_to_end_literature_survey_20261106/END_TO_END_COST_ANALYSIS_MAP.md
[G6-TEST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/tests/tracks/algorithm_codesign/test_g6_return_aggregation.py
[G9-TEST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/tests/tracks/algorithm_codesign/test_g9_p5_and_native.py
[G10-TEST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/tests/tracks/algorithm_codesign/test_g10_degree_preparation.py
