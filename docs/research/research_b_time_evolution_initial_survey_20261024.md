# 研究B：時間発展アルゴリズム改善の初期技術調査

2026-10-24 JST。区分：既存資料・静的SOURCE・一次文献の限定調査。

**次に検討する価値があるのは、B4のエネルギースケール分離を使う乱択PFと、B2のCorrected／Processed PFのcoherent signalへの接続である。** 前者には平均unitaryを直接評価した一次文献の中間結果があり、後者には再利用できるprocessor・kernel・費用会計がある。ただし、量子化学DFでのアクセス費用、有限RTEを含む保証、同精度の総資源利益、独立した算法的新規性は未確立。本書は追加調査の提案であり、正式採択・科学的GO・実装／計算の承認ではない。

## 1. 目的、取得資料、固定起点

目標精度を維持した量子化学時間発展の資源削減を、PRの改善とPRを置き換える原理の双方から調べる。新しい生成則の設計、擬似コード、係数探索、実装、科学計算、保存値の再集計・再ランキングは行わない。外部論文は保証とアクセス方式の関連箇所だけを確認し、網羅的レビューや全定理の独立再証明はしない。

Repositoryは `HIROMU1015/Partially-Randomized-Trotter`。独立branchは `codex/research-b-initial-survey-20261024`、worktreeは `/tmp/prt-research-b-initial-survey-20261024`、起点は `7aa911bf929d8d1469d27749bf1d83132a36a00e`。変更対象は本Markdown一件だけで、他研究branchへのmergeは行わない。

以下の必須資料はGitHub接続から指定commitの本文を実際に取得した。

| 固定資料 | commit | 今回の用途 |
|---|---|---|
| [Gaussian singleton終了・引継ぎ][HANDOFF] | `69f8e1cac23a4924ab81aebe399657a705024f6a` | 一時終了の正式採択、F1/F2の限界、旧STOPを保持 |
| [次期改善機構調査][MECHANISMS] | `7aa911bf929d8d1469d27749bf1d83132a36a00e` | IS・辞書・normalizerの既知部分と研究A/Cとの境界 |
| [研究横断地図][MAP]、同commitの[研究カード][CARDS] | 同上 | PF、B-F/B-M、THRIFT、相関法等の既存状態を追跡 |
| [過去の高次PF研究・reviewer response][PFOLD] | `20d64c2fea1ed2d2188777df4f3d962ee94df486`、別repository | Moralesの版、processed構成、有限時間誤差、費用scope |

ユーザー指定の正式共通指示 `research_project_instructions_unified_2026-10-23.md` は元workspaceのファイルを全文確認した。SHA256は `7ce342dbf7086424f1afd837d8c274687bc28e4838601f5ad96f5b626d95667f`。未追跡の原本をコピー・公開・改変していない。

作業起点の `PROJECT_MAP.md`、`docs/research/研究概要・現状.md` 全文、研究目的、関連 `VALIDATION_STATUS.md` と `artifacts/validation_manifest.json` を確認した。起点の概要にはsingleton終了の採択前記述が残るため、今回の終了状態には別固定commitの[正式引継ぎ][HANDOFF]を用いる。これを揃えるために共有概要・manifestを変更したり、終了branchをmergeしたりはしない。

## 2. 比較する対象と保証の境界

PRで必要なのは、ordered pathのunitaryを位相ごと保持した補正済みfirst momentと、controlled HadamardのX/Y信号である。以下は既存[相関監査][CORR]と次期機構調査の規約を再掲したもの。

\[
A_K=\mathbb E_\omega[W_\omega V_\omega],\qquad
z_K=\langle\psi|A_K|\psi\rangle.
\]

実装平均と共通finite targetの差、finite targetと物理時間発展の差、測定統計誤差を分ける。Bの候補がfinite target自体を変更する場合、既存の同mean ISとは別armであり、同じ物理信号精度へ戻す比較が必要となる。

| 評価対象 | 保証から読めること | 自動的には読めないこと |
|---|---|---|
| unitary／平均unitaryのoperator norm | 位相規約を固定した演算子近似。任意の固定状態の線形信号へ接続する基盤 | 基底energy biasの最適係数、全RPE成功率、実回路の価格 |
| 平均conjugation channelのdiamond norm | ancillaを付けた入力にも適用できるchannel距離 | 実現unitaryの位相を固定した `E[V]`。system channelへのancilla追加と、各実現をcontrolled化する操作は別 |
| coherent signalの複素誤差 | 指定状態・時刻・control・normalizationでのX/Y誤差 | 信号半径が小さい場合の良好な位相精度、他時刻、energy全体 |
| PFの固有位相・固有値 | 分枝を同定したeffective Hamiltonianのenergy誤差 | 元入力状態との重なり、survival phase、weighted random meanの単一位相性 |
| ground-state energyの同精度総資源 | 状態準備、推定器、全round、成功確率まで揃えたtask | operator誤差や一回路価格だけからの順位 |

`U` と `−U` は同じsystem channelを与えてもcontrolled相対位相が異なる。この既存の区別を維持する。一方、channelを主対象とする論文でも、証明に位相を固定した平均unitary評価があれば、それを別途確認して接続可能性を判断する。「channel論文だからPRへ使えない」とも断定しない。

共通費用には、processor／corrector、正負時間、inner approximation、state preparation、controlled evolution、readout、必要shot数、採用・棄却を含む全試行、normalizer／probability／weight取得、setupとonline古典処理、compileと償却単位を含める。prep除外の既存指標を用いる場合も、shot数変更によるprep感度を別に残す。RZ・CX・depth・native T・logical qubit・古典時間は、根拠なく一つの数へ加算しない。

有限ビットlaw、角度、normalization、合成・compileに伴うbiasを戻さず、理想実数の式から認証Nを算出しない。`C_use` は実行delta窓の経験的envelopeであり、厳密上界ではない。H12係数、最終全round総費用、PRの最終優位性は未決定のまま。

## 3. B1〜B5の候補比較

表の順序は分類順で、性能順位ではない。既知法の応用価値と独立した新算法の可能性を分ける。

| 方向 | 資源削減機構・数学的条件 | PRへの接続と強いbaseline | 既存状態・追加判断点・新規性 |
|---|---|---|---|
| **B1 構造適応PF** | 高次化、係数／順序／分割／multirate、commutator相殺で誤差または反復数を減らす。内部block誤差とfinite-time remainderも必要 | 決定論部分またはouter split。S2、Yoshida4、既存m5、published Morales、同情報compact BCH・同cache・公平再最適化 | 係数比較は局所検証済み。B-Fは限定negative、B-Mは既知再帰と同値。構造を使うだけでは新規性にならず、異なる生成原理か追加情報取得差が必要 |
| **B2 Corrected／Processed／MPF** | correctorで誤差成分を除去、processingで反復kernelと端点費を交換、線形結合で低次誤差を相殺。二分割access、corrector合成誤差、係数conditioningが条件 | coherent operator／signalを保つかを先に確認。既存YP8m8、強い非processed PF、well-conditioned MPF／通常Richardsonも対照 | processed資産あり。CPFのDF-PR接続、端点償却、shot込み利益は未検証。既知CPFの適用と独立算法は別 |
| **B3 Interaction picture** | 可解Aを外し、残差の時間順序付き発展をTHRIFT／Dyson／Magnus等で生成。必要なoracleが方式ごとに違う | `H_D` の可解性は未保証。THRIFTはmixed指数、Dysonはtime-indexed block-encodingを要求。元frameのPFと同アクセス費で比較 | H4の現mixed-access templateは終了。別oracle／生成原理なら論点は残るが、安いaccessの具体根拠が必要 |
| **B4 Randomized simulation** | 順序・挿入時刻・相関block・補正unitary等を変え、近似演算子そのものの誤差を平均で相殺。channel／operator／observable保証を区別 | 新しいenergy-scale乱択PF、qDRIFT、qSWIFT、TETRIS、PRHS、qSHIFT、SPRINT。同精度canonical PR／CTS等にも同じ合法最適化を与える | 2026年乱択PFにfirst-moment評価を確認。旧相関監査の重複、signed weight・生成費・有限bit・controlが残る。確率変更だけの研究Aとは分離 |
| **B5 非PF** | Taylor/LCU、QSP・qubitizationで低次数PFの精度依存を回避。block-encoding正規化とoracle精度・費用が前提 | PRのtail交換または全time evolutionの代替。DF等を用いた実装済み文献のqubitizationを強い対照に置く | 既知原理の直接適用が中心。今回同task費用は未評価。query優位をnative／physical優位とせず、安い新encoding等がなければ独立新算法とは呼ばない |

## 4. 各方向の技術的な読み方

### B1：係数再探索を始める前に、既済結果との差を固定する

[Morales等のpublished版][MORALES]、[commutator scaling理論][COMMUTATOR]を既知の基盤とする。可換構造を残す誤差評価と、同じstage数で係数を変えること、実行頻度を変えるmultirateは異なる自由度である。評価式の整理・cache・既知optimizerへの特徴量追加を、そのまま時間発展の新原理には数えない。

[過去PF研究][PFOLD]では、legacy arXiv-v2の `8th(Morales)` とpublished `8th(Morales-Y8m10b)`、processed `8th(Morales-YP8m8)`、published tenth m17を区別している。旧m8／m16だけを対照にして改善を主張しない。短時間の誤差係数と有限時間の零交差・選択時刻では順位が一致しない場合があり、漸近fitの良さは運用時間での資源順位ではない。今回その保存曲線を再採点していない。

既存のPR高次PFに関する[次数逆転監査][ORDER]には、保存順位の逆転候補とallowance・native/runtime結合の未解決がある。因果説明や科学的比較の完成とは扱わない。決定論PFの良い係数を採る場合も、負stageがRTEのnormalization・total absolute time・shot負担を変えるため、決定論部分だけの勝敗をPRへ移さない。

**次の限定調査で変わり得る判断：**既存と違う誤差成分／生成列を狙い、同じ情報・探索budget・cacheを持つ強い対照から得られない差が具体的に残るか。現時点で追加係数探索を優先する根拠は得ていない。

### B2：相似変換、kernelの設計、線形結合を区別する

[CPF論文のv4][CPF]はsymplectic、symmetric、compositeのcorrectorを扱い、二分割のexact evolutionを基準にcommutator誤差を除く。perturbed系のProposition 1は短時間の収束条件下で、CPF2の局所誤差を `O(α²|λ|³ + α|λ|^(2k+3))` とする。correctorの項数kは通常PF次数と別。高次再帰やcorrectorの型により、費用増は端点への加算にも各stepの乗算にもなる。§5はcorrectorの指数列への合成とその誤差を扱う。これは実DF回路の費用保証ではない。

既存[processed実装][PFSOURCE]は `P K^r P^-1` を構成し、[費用helper][PFCOST]は反復kernel費とprocessor-pair数を別に受け取る。YP8m8は17個のS2-block kernelと片側20個のprocessorを持つ。これらは列長の記録で、native gate数ではない。[検証runner][PFVALIDATOR]はkernelの固有値とfull processed回路の重なり評価を分けている。

**PRへの技術的含意：**固定したunitary Pによる相似変換はkernelの固有値自体を変えない。一方、固定入力状態のcoherent signalは状態の向きにも依存する。従って「processorで固有位相を改善する」とは一括して言えず、processed精度を持つkernelの設計と、端点処理によるoperator／overlapの改善を分ける。決定論部分だけを処理すると残差との境界も変わるため、full PRを処理したことにはならない。これは既存実装の意味論からの適用上の整理で、今回新しいPR構成を作ったものではない。

同一Pが隣接stepで相殺できるか、roundや独立shotごとに何組必要かは実際のwrapperに依存する。sample・step幅でPが変わる場合に一組だけと数えない。残差やcorrector内の指数をfinite RTEへ置くなら、そのbias・正負時間・normalizationを別途支払う。

[well-conditioned MPF][MPF]はPFの線形結合による高次化で、一本のPF積とは異なる。coherent LCUによるoperator実装、signedな線形信号の後処理、conjugation後のobservable外挿は、費用と近似対象が違う。PRのX/Y信号に用いる場合も、係数の絶対値質量・weight range・必要shot数、各armのbackbone反復を無視しない。PR-3の固定toyでtail bias改善がshot/backbone費に負けた既存記録は、全MPFへのno-goではない。[研究カード PR-02][CARDS]

**次の限定調査で変わり得る判断：**既知CPF／YP8m8／MPFそれぞれが、元の物理信号を何の追加accessと端点費で近似するか。新規性はその対応付けの完成と別に判断する。

### B3：現THRIFT templateの救済ではなく、accessの差を調べる

[THRIFT][THRIFT]は `H=A+αΣB_l` の可解部を利用し、`exp[-is(A+αB_l)]` 等のmixed accessからinteraction-picture PFを構成する。[Low–Wiebeのtruncated Dyson方式][DYSON]はtime-indexed Hamiltonian encodingとtime-orderingを使う。後者は小さい残差のnormalizationを利用できる一方、時間register、SELECT／encoding、準備・uncomputeを要する。同じ「interaction picture」という名前だけでアクセス費を共有しない。

既存[H4終了記録][MIXED]の条件は、linear H4 1.00 Å、STO-3G、実RHF canonical MO、4 spatial／8 spin modes、N=4、補正one-body＋DF全16 factors保持、切捨てなし、PRの `L_D` は非該当。T={1,4}、r={1,2,4,…,1024}、operator作用誤差目標εU=.001、固定7法・登録費用modelである。ideal PFのstep削減は保持するが、bitwise可変角回転・occupation算術templateではknown費用に正の追加budgetがなく、詳細backend投資を見送った。ground-energy 1mHaや実compiled費用の敗北を示した結果ではない。

**別候補として残る条件：**Aが本当に安くexactに実装でき、残差のtime-dependent encoding、またはmixed指数が旧templateと違う費用構造を持つこと。`H_D` は複数の非可換DF fragmentを含むため、「決定論側」というだけでは可解Aにならない。単一可解fragmentをAにすると残差が大きくなり得る点は、研究Cとの論点になる。既知Dyson方式の導入自体は新算法ではない。

Magnus／CFQM型も候補名として残るが、時間依存した和の指数を実装可能と仮定しない。旧H3 frame監査と同様、安い生成accessが特定されない段階では実装・再検証へ進めない。[研究カード QC-12][CARDS]

### B4：2026年乱択PFと、既存のsigned／history法を分ける

**エネルギースケール分離型の新しい一次文献。** Kim–García-Pintos、2026-08-07、[2608.07720v1][RPF2026]。比較可能なnormのA/Bと小さいαに対し、挿入時刻を乱択する。標準accessでのTheorem 1は固定次数2kについて平均channel誤差 `O(α² t^(2k+1)/r^(2k))`。強いmixed accessのTheorem 2はα³と期待O(L)指数blockを与える。標準高次法の「2倍」は同次数決定論PFに対するprimitive指数列の深さで、native gate／shot総費用の倍率ではない。

**保証の重要な確認点：**主定理のdiamond normを逆向きに読み替えたのではない。補足S2.1 Eqs.(S14)–(S17)、S2.2 Eq.(S23)は平均unitaryのoperator normを直接評価する。S3.2.4 Eqs.(S91)–(S92)も同型。ここがcoherent first momentを調べる根拠である。Propositions 1/2のdeterministic障壁は、固定有限stage・Hamiltonian等に依存しない係数に対するexistentialな最悪例で、特定DFや適応的係数・processor全体の不能定理ではない。

**PRへ向けた今回の推論：**標準access側を先に照合する価値がある。旧THRIFTのmixed指数を要求しないためである。ただしA/Bのexact指数をnested PFや有限RTEへ置換した際の誤差、recursion内で共有する乱数とstep間独立性、負stage、連続時刻の有限bit化、scalarとcontrolled位相、長roundの信号半径を別に扱う必要がある。混合channelが良いことだけでは完了せず、平均unitaryの中間評価も「現在のfinite targetを保存する」という保証ではない。randomized THRIFTには可変長補正のtailとmixed accessが残るため、初手の同等候補にしない。

その他の重要法は次のとおり。query／orderの改善をPRの改善倍率へ移さない。

| 方法 | 生成原理と確認した保証 | PRへの残る接続、既存成果との差 |
|---|---|---|
| [qDRIFT][QDRIFT] | term normに比例した単一指数の独立sampling。平均channelを基準とする既知法 | 標準baseline。term数依存を減らしてもλと精度の負担は残る。現finite Taylorの同mean法とはしない |
| [qSWIFT][QSWIFT] | 高次補正をstochasticに実現するchannel構成。diamond norm保証とobservable推定手順 | 補正回路・ancilla・shotを含む対照。今回v2 HTMLは本文変換不完全で、詳細algorithm／全定理は未確認 |
| [TETRIS][TETRIS] | involution回転をPoisson時刻順に生成し、既知attenuationで複素振幅を補正。離散化biasと回路長・shotを交換 | main Eqs.(9)–(11)は二つの独立pathを使うobservable手順。有限期待長は固定最大長ではない。controlled一側信号と背景Dへの接続費を別に確認する |
| [PRHS][PRHS] | correlated signed block。Liouvillian／diamondのbias評価、形式word条件で係数を構成 | 符号のある係数は正の確率とsign weightへ分ける必要があり、normによるshot増加を含める。元finite RTEの同meanではない。既存相関監査に取得・重複整理あり |
| [qSHIFT][QSHIFT] | history条件付きsigned係数でobservableのTaylor係数を合わせる。本文の全体scalingには線形誤差累積の仮定がある | 係数アクセスはL^r規模。量子回路のshallow化と古典無料化を混同しない。既存監査の版・表示問題、weightと実測shotの会計を引き継ぐ |
| [SPRINT][SPRINT] | near-integrability、processing、symmetry protection、fragment順序乱択、QROMを組み合わせた量子化学PF | §III.5／Appendix Fはspectral broadeningも扱う。強い化学baselineとして重要。GRADEやQROMを含むためCと分担し、PR全体への効果と各要素の効果を混同しない |

SPRINTはB1/B2/B4にまたがる追加の重要候補である。単独の「順序乱択を加える」案を新規とせず、既存のnear-integrable／processed構成にどの成分を接続できるかを問う。今回は元論文の化学taskの倍率を引用せず、手法と保証範囲の入口だけを確認した。

### B5：query削減と、実量子化学回路の価格を分ける

[Taylor/LCU][TAYLOR]は有限Taylorのunitary和をcoherent SELECT・ancilla準備とrobust oblivious amplitude amplificationで実行する。PRのように古典samplingして測定weightを補正する実装とは異なる。両方に「Taylor」があっても、ancilla・成功処理・depthとshotの会計を同一にしない。

[Low–Chuangのqubitization][QUBIT]は正規化したHamiltonianのencodingとQSPによる多項式変換を使う。同論文のLCUモデルでは、係数質量aに対しquery上界 `O(a t + log(1/ε))` が得られるが、oracle単価は別である。化学では[DFを用いたvon Burg等の構成][CATALYSIS]が、同じ表現情報を利用する強い既知対照となる。今回その大型資源見積りを再現・PRへ換算してはいない。

PREPARE／SELECT、QROM、Gaussian basis回転、reflection、rotation精度、workspaceとuncompute、controlled wrapperを含む費用が必要。PRのtailだけを置換する場合も、block-encodingのancillaを適切に返し、どの線形信号を保証するかを明示する。energy推定ではqubitized walkの位相を利用する方式もあるため、常にQSPで物理時間発展を作る方式だけをbaselineにしない。

2025年の[Low等のspectral amplification][SPECTRAL]も、ground-energy taskの強い対照として追記する。sum-of-squares表現と低energy領域を使ってencodingのnormalization負担を減らす既知法であり、任意状態の時間発展を一様に安くする保証とは異なる。古いDF-qubitizationだけを現行最強とせず、同じ表現・情報取得費で適用できるかをCと確認する。既知SOS構成の応用と、過去の独自SOS-PF案の非採択を混同しない。今回同法の資源値を再評価していない。

2026年の[Paganelli等のSparse QSVT][SPARSEQSVT]も追加候補に記録する。dominant項を決定論的に残す点はPRに似るが、stochastic encodingの誤差は多項式変換中に伝播する。論文のrandom Hamiltonian ensembleでのcrossoverは、現DF・ground-energyの一般閾値ではない。安いencodingや保証の具体差がなければ、既知法の適用研究として扱う。

## 5. 既存成果・限定negative・不足をどう引き継ぐか

| 既存系列 | 確認した状態と条件 | 今回の扱い |
|---|---|---|
| Morales／Yoshida・processed PF | 別PF repositoryの[固定response][PFOLD]とSOURCEに係数版・kernel／processor・局所検証がある | 基盤を再利用。決定論unitaryの固有値誤差とPRのfull signalを混ぜず、保存値の再fitなし |
| 有限時間PF第一研究S0–S4 | [固定最終統合][FINITE]は機構・信頼性解析を保持し、固定二状態診断の資源利益は `complete_no_benefit` | 未完了の救済課題としない。有限時間校正全般のno-goにも広げない |
| B-F | [closure][BF]：H4 linear1Å／STO-3G／DF12／generation-prefix L_D=3、T=.8、q1/2/4/8、Rbudget5/10/20/40/80、K2＋事前trigger K4。同5-stage対称4次family・同32評価budget | 保存cellの公平cross-scoreでfinite objective固有のdecision valueなし。旧中断INCONCLUSIVEと事後recoveryを区別。追加family探索の承認にしない |
| B-M | [BM0.5監査][BM]：固定nested列の形式三次係数を同情報compact BCH再帰と比較。分子計算・finite signal検証ではない | 既知再帰と同値、BM-1新手法実行なし。cacheをbaselineにも認める。multirate一般を否定しない |
| THRIFT型mixed access | §4 B3の固定H4・全16factor・T/r／作用誤差task、現occupation算術templateは `CLOSED_CURRENT_TEMPLATE_FEASIBILITY_SERIES` | 現構成の再検証・単価救済scanは開始しない。別アクセス原理の調査余地と区別 |
| 時間相関／PRHS／qSHIFT | [既存一次資料監査][CORR]は本文取得済み、単純二step高次化の既知法対応、ordered mean・signed weight・有限confidenceを整理 | 「未検討だから新規」としない。旧監査で指摘したqSHIFT v3の表示問題は、本書では訂正版を認定しない |
| Gaussian singleton／混在辞書F1/F2 | [正式引継ぎ][HANDOFF]に従い一時終了。4 mode・2粒子・occupied(0,1)・合成density-square H・L_D=1・残差2fragment。分子geometry／AO basis／分子DF rankは非該当。F2はT=1/5、q1、δ=1/5一点、r2、h1/10、K0、固定Qiskit1.3.0／opt1／seed17／rz,sx,x,cx／free topologyのfull wrapper | 登録数値検査と1-shot費用の証拠はあるが、γ・統計資格・認証N・同精度総費用は未成立。[status][STATUS]／[manifest][MANIFEST]の `source_present_no_current_ci` を外部再現へ格上げしない。F1/K2の結果をF2/K0へ混用しない |

過去の各費用優位・負結果は、その保存task、モデル、accuracy／confidence、providerと情報budget内の結論である。未実装、資料未取得、保証未成立、性能negative、終了判断は異なる。旧grant・marker・STOPを流用せず、同じ研究を新しい名前で自動再開しない。

## 6. 一次文献：版と今回の直接確認範囲

著者は原著の表記に従い、長い著者一覧は筆頭著者＋et al.とした。版日はarXivのsubmission historyで識別し、HTML生成日／title-page日付と混同しない。主要な科学的記述は論文・公式出版物に限り、二次紹介記事には依拠しない。

| 一次文献・著者・年／版 | 実際に確認した箇所と限界 |
|---|---|
| Günther, Witteveen, Schmidhuber, Miller, Christandl, Harrow, *Phase estimation with partially randomized time evolution*, 2025初出／[2503.05647v2、2026-07-10][PRPAPER]、PRX Quantum 7, 020332 (2026) | 今回は公式metadata・abstract。finite-RTEの具体意味論は既存SOURCE／監査を使用。v2全本文を新たに読了とはしない |
| Morales, Costa, Pantaleoni, Burgarth, Sanders, Berry, *Selection and improvement of product formulae for best performance of quantum simulation*, [2210.15817v3、2025-01-10][MORALES]、QIC25,1 | 版・abstractと固定repositoryのpublished係数対応。論文全係数の独立照合はしない |
| Childs, Su, Tran, Wiebe, Zhu, *A Theory of Trotter Error*, [1912.08854v3、2021-02-03][COMMUTATOR]、出版題 *Theory of Trotter Error with Commutator Scaling*, PRX11,011020 | 公式metadata・abstract。汎用commutator基盤の位置付けに限定、DF固有certificateを新作していない |
| Bagherimehrab et al., *Faster Algorithmic Quantum and Classical Simulations by Corrected Product Formulas*, [2409.08265v4、2026-07-28][CPF]、PRX Quantum7,033018 (2026) | §2のcorrector三種・Prop.1／Thm.2、§3–5の関連説明とTable1、§6／8。v3と混同しない。PDFも取得したが全37頁の精読はしない |
| Low, Kliuchnikov, Wiebe, *Well-conditioned multiproduct Hamiltonian simulation*, [1907.11679v2、2019-09-20][MPF] | 本文のMPF定義・conditioning・LCU成功処理の説明。PR接続の証明／実装は未実施 |
| Bosse, Childs, Derby, Gambetta, Montanaro, Santos, *Efficient and practical Hamiltonian simulation from time-dependent product formulas*, [2403.08729v3、2024-06-24][THRIFT]、Nat.Commun.16,2673 (2025) | arXiv §2／3のaccessとMagnus/Ferの入口。出版社サイト/PDFは認証redirectで取得失敗。arXiv本文を代替取得し、publisher版全体との一致は未監査 |
| Low, Wiebe, *Hamiltonian Simulation in the Interaction Picture*, [1805.00675v2、2019-06-06][DYSON] | §III input model、Dyson truncation、§IV／Cor.4と必要oracle。誤指定v3の404後に公式historyのv2を取得 |
| **Kim, García-Pintos, *Randomized product formulas beyond optimal deterministic scaling*, [2608.07720v1、2026-08-07][RPF2026]** | 主文Thm.1/2・Prop.1/2、SM S1/S2.1–2、S3.2.4、S4の費用scope。first-moment中間評価を直接確認。全証明の再証明・数値再現なし |
| Campbell, *A random compiler for fast Hamiltonian simulation*, [1811.08017v2、2019-06-25][QDRIFT] | 公式metadata・abstract。正確なchannel／control移送は既存監査を使用し、元論文の倍率は転記しない |
| Nakaji, Bagherimehrab, Aspuru-Guzik, *qSWIFT: High-order randomized compiler for Hamiltonian simulation*, [2302.14811v2、2024-06-19][QSWIFT]、PRX Quantum5,020330 | 公式metadata・abstract。HTMLはIntroduction以降の変換不完全。詳細定理のPRへの移送は未確認 |
| Granet, Dreyer, *Continuous Hamiltonian dynamics on digital quantum computers without discretization error*, [2308.03694v2、2023-11-17][TETRIS] | 主文Eqs.(1)–(11)、Poisson生成・algorithm・background Hamiltonianとshot説明。時間発展unitaryの生成分布と、二path observable手順を区別 |
| Cugini, *Pathwise Random Hamiltonian Simulation*, [2608.29756v1、2026-08-30][PRHS] | 主文Eqs.(2)–(8)、Appendices A–Cの構成・norm・費用の関連箇所。formal uniquenessを全sampler最適性へ拡張しない |
| Lee, Choi, qSHIFT／*Adaptive sampling based quantum simulation*, [2604.26263v3、2026-09-02][QSHIFT] | §III／III.1、Appendix B.2/B.3の関連箇所。公式absの題は *qSHIFT: An Adaptive Sampling Protocol for Higher-Order Quantum Simulation*、同v3 HTML題は上記別題。題・版を無言で統一しない |
| Casares et al., *Theory and practice of Trotter product formulas for quantum chemistry*, [2606.30741v1、2026-06-29][SPRINT] | §III.2／III.5、IV、V.4、Appendix Fの関連説明・入口。GRADE／QROMを含む論文全体の再監査ではない |
| Berry, Childs, Cleve, Kothari, Somma, *Simulating Hamiltonian dynamics with a truncated Taylor series*, [1412.4687v1、2014-12-15][TAYLOR]、PRL114,090502 (2015) | §§I–IIのSELECT・準備・robust amplificationの説明 |
| Low, Chuang, *Hamiltonian Simulation by Qubitization*, [1610.06546v3、2019-07-11][QUBIT]、Quantum3,163 | Thm.1、Table1、§§3/6、Cor.16。queryとoracle単価を分離 |
| von Burg, Low, Häner, Steiger, Reiher, Roetteler, Troyer, *Quantum computing enhanced computational catalysis*, [2007.14460v2、2021-03-03][CATALYSIS]、PRResearch3,033055 | 今回は公式metadata・abstractのDF資源構成。82頁補足の精読・資源値再検証はしていない |
| Low, King, Berry, Han, DePrince III, White, Babbush, Somma, Rubin, *Fast quantum simulation of electronic structure by spectrum amplification*, [2502.15882v1、2025-02-21][SPECTRAL]。出版題は *Fast Quantum Simulation of Electronic Structure by Spectral Amplification*, PRX15,041016、2025-10-31、DOI `10.1103/pb2g-j9cw` | arXiv公式metadata／abstractと出版社abstract。SOS／低energy・normalizationの機構を確認した範囲。全定理・資源表の監査なし |
| Paganelli, Grossi, Giachero, O’Brien, Kiss, *When is randomization advantageous in quantum simulation?*, [2604.07448v1、2026-04-08][SPARSEQSVT] | §III／Vの入口とabstract。ensembleのcrossoverを一般分子へ移さない |

**資料上の保留：**CPF v4 Eq.(9)は、今回取得したHTMLとPDF抽出textの両方で中央指数が `αλB/2` と表示され、Eq.(2)との整合を要する。紙面画像取得はcache missだった。本書はその表示を新しいS2規約として採用せず、著者の理論全体の否定にも使わない。将来使う式はpublisher版／紙面との照合対象とする。qSHIFTの旧監査が記録したv1/v3表示問題も、今回修正済みとは認定しない。これらは追加計算の理由にはしない。

## 7. 再利用可能なSOURCE・検証資産

以下は静的に参照した接口と既存検査であり、今回import・呼出し・test実行をしていない。単にpathが存在することと、新候補への適合性を確認したことを区別する。

| 固定SOURCE／資産 | 再利用できるもの | 適合性の未解決点 |
|---|---|---|
| 別PF repository `20d64c2…` の [product_formula.py][PFSOURCE] | `morales_2025_yp8m8_kernel_list`、`morales_2025_yp8m8_processor_sequence`、`_get_kernel_s2_sequence`、`_get_processor_s2_sequence`、`_get_s2_sequence` | 現PR branchの同名moduleは版が異なる。名前だけで実装済み扱いしない。丸め・列順・actual inverseとDF-PR接続が必要 |
| 同commit [processed_cost.py][PFCOST] | `ProcessedCostComponents.total`、`morales_yp8m8_hchain_costs`、明示的なprocessor-pair会計 | Pauli-rotation／RZ-layerの既存表は新しいfull controlled native費用ではない |
| 同commit [Morales検証runner][PFVALIDATOR] | direct eigenvalueとfull processed overlapの比較構造、published／legacy識別 | H2 grouped Pauliの処理をDF／finite RTEへ無条件流用しない。`tests/` directoryのGitHub照会は404で、同repoの専用unit testsを取得済みとはしない |
| 起点 `7aa911b…` の [`rte.py`][RTESOURCE] | `RTEEvent`、`event_unitary`、`finite_taylor_operator`、`finite_rte_corrected_operator`、`finite_rte_operator_moments`。補正meanとattenuated meanを区別する基盤 | paired cutoff Kはdegree K+1。新乱択法のtarget／lawを既存finite Taylorと同一としない。保存入力の新評価は別承認 |
| 同起点 [`finite_rte_signal_validation.py`][SIGNALSOURCE] と `tests/test_finite_rte_signal_validation.py`、`pf_delta_validation.py`／対応tests | sector action、operator／signal／radius、PF分枝・conditioning、較正とholdoutの分離 | 新生成法・新state・長qを既存PASSで代替できない。runnerやscience testsは今回実行しない |
| 同起点 [`pr_dictionary_validation/circuits.py`][CIRCUITS]、[`gates.py`][GATES] | `build_event_sequence`、`actual_adjoint`、`hadamard_wrapper`、`controlled_gate`、`compiled_wrapper`。ordered phase・scalar・非制御basis＋中央control・同frame共有を保持 | 現合成fixture／compilerでの接口。新processor・Dyson／block-encoding・連続時刻samplingのadapterは未実装。費用の既存共有を候補固有効果へ再計上しない |
| `4301c0f…` の[B-F][BF]／[B-M][BM]／[相関監査][CORR] | 同情報・同budget・同cacheの強対照、negativeのscope、signed weight／range／shot会計 | 旧SOURCE・grant・STOPを新しい候補の認可として使わない |
| [THRIFT終了記録][MIXED] と保存decision／費用scope | 旧mixed-accessの前提とknown-budget障害、比較契約 | 安い新oracleの存在や、新native費用を供給する証拠ではない |

公開済み報告のstatus・根拠箇所を読む作業に限定し、raw CSV／NPZ／QPYを開いた科学的再解析や証拠の全件再認証はしていない。固定commitの参照は実行SOURCEの同一性を自動的に意味しない。

## 8. 次に提案する限定調査：2件

性能上の順位や正式採択ではなく、少ない追加作業で適用可能性の採否を変えられる候補を挙げる。

### 候補1：energy-scale乱択PFの標準accessとPR接口の対応付け

- **支持する根拠：**2026年論文のfirst-moment中間評価があり、旧mixed-access templateを前提としない。分布だけでなく挿入時刻と指数列を変えるので研究Bの対象となる。
- **最大の未解決点：**DFの決定論／残差を、論文が必要とする安いA/B指数へ対応させられるか。有限RTEのinner bias・normalizationを戻すと利益が残るかは未検証。
- **強いbaseline：**同じA/B accessのS2・最適化済み高次PF・CPF。物理taskではcanonical PR、適格な同finite targetの既存CTS／Closed P5等にも、同じ価格情報・合法共有・IS最適化機会を与える。異なるfinite targetなら同じ物理精度で比較する。
- **Codexへ次に依頼する価値のある最小作業：**論文のprimitive、乱数依存、正負時間、first-moment評価の適用条件を、上表の既存関数／未実装接口へ一枚で対応付ける。新しいsplit・sampler・係数・保証を作らず、値を評価しない。exact A/Bが未供給なら、その時点で返す。
- **その後に採否を変える証拠：**適用可能なアクセスと制御位相の説明、有限bit／inner approximationを含む誤差会計、同精度shot×full-wrapper費用。取得には新しい設計・実装と科学実行の別承認が必要。本書では取得していない。
- **新規性：**論文法のPR接続は応用候補。接続できたことだけで独立算法の新規性を認定しない。

### 候補2：CPF／processed PFで何の信号を改善するかの限定整理

- **支持する根拠：**published YP8m8の係数、processor／kernel分離、費用helper、direct／overlap検査を再利用できる。CPFは係数探索以外の誤差項処理を提供する。
- **最大の未解決点：**改善対象がoperator／overlapか、kernel設計を含む固有位相かを固定でき、端点費・corrector合成・残差との境界を含めても実益があるか。
- **強いbaseline：**published Y8m10b／YP8m8、既存強い低次PF、既知CPF、well-conditioned MPFと同taskの通常外挿。候補だけにexact state／E0や無料processorを与えない。
- **Codexへ次に依頼する価値のある最小作業：**一次式の表示を照合したうえで、既存の実際のsequence・inverse・round／shotごとのprocessor-pair支払い単位を静的に対応付ける。Dだけの処理、full evolutionの処理、線形結合の3ケースを混同しない。新しいcorrector、擬似コード、cost値を作らない。
- **その後に採否を変える証拠：**同じ物理信号に対するcorrector込みbias／radius、端点償却可能性、native価格とN。未実施のsynthetic対照も今回自動生成しない。
- **新規性：**既知CPF／processingの利用価値と独立算法差を別々に評価する。単なる `P K^r P^-1` の接続を新原理としない。

B3は具体的に異なる安いaccessが示された場合、B5は同taskの強い全体baselineが必要な場合に限定調査を提案する。SPRINTは上の2件の近接先行研究・化学baselineとしてCと連携する。これらの優先提案は、旧B-F／B-M／THRIFTやGaussian singletonのSTOPを変更しない。

## 9. 研究A・Cとの連携と統括チャットへ返す判断

| 相手 | 共同で確認すべき具体点 | 研究Bの担当範囲 |
|---|---|---|
| **研究A：law／statistics** | 新unitary poolでのsign・normalization、M2と一様range、finite-bit bias、fresh-IID shot、有限confidence、state preparationを含むN | 時間発展近似のtargetとordered meanを渡す。ISだけの改善をB固有の新算法としない |
| **研究C：representation／compiler** | 安い可解A／mixed指数／block-encoding、DF rank／basis／splitの同一性、corrector合成、phase保持control、QROM／native価格とworkspace | どのoracleが必要で、なぜ旧templateと違うかを明示。compiler共有・表現変更の利益をBと二重計上しない |
| **統括／ユーザー** | 既知法の応用改善を成果とするか、独立算法差を必須とするか。候補1／2の静的接口調査を依頼するか。共通task・metric・情報予算・confidenceを将来どこまで揃えるか | 調査の根拠・未解決点を渡す。正式採択、科学的GO、詳細レビュー開始、科学実行の代行判断はしない |

共有すべき判断事項は次の5点である。

1. B4新論文は主定理のchannel保証だけで除外する対象ではなく、平均unitaryの評価が接続候補になる。ただし現PRの保証・資源利益は未取得。
2. B2は資産再利用度が高いが、processingで変わるoverlapと不変なkernel固有値を分離する必要がある。correctorの名前だけでは有利にならない。
3. B-F／B-M／現THRIFTの結果は既済で、同構成の再検証や救済探索を標準次作業にしない。新規性のない通常接続に終わる可能性も報告対象とする。
4. PRHS／qSHIFT／SPRINT／Sparse QSVTは2026年の近接先行研究として外せない。既知法対応、保証target、古典取得費、shot負担を落とさず比較する。
5. 本調査では独立新算法も、PRにおける同精度総資源の優位も確立していない。未取得・未検証を推測値で埋めず、次の担当と承認を分ける。

## 10. 保全、公開、mandatory STOP

今回の新科学計算・simulation・保存値の数値再解析／再ランキング・Hamiltonian生成・回路build／compile・PF最適化・sampling・sampler／新算法実装・science tests・新grant発行はすべて0件。資料取得と静的text読取り、文書作成・技術的な差分／リンク／bytes確認だけを行う。CPU／RAM／GPU／workerへ人工上限を設定する計算契約は作成していない。

固定SOURCE、raw／result、grant／ledger／marker、旧STOP、科学契約・評価基準、元workspaceのdirty／untracked、他worktree・共有環境・他ユーザーを変更しない。既存保護物全件を新たにhash監査したとは主張しない。文書一件の追加であることを固定起点との差分で確認する。共有研究状態ファイルはユーザーの今回の指定に従って更新しない。本提案は既存研究判断の変更ではない。

本Markdownのみを専用branchへcommit・pushし、remote branchのcommit SHAと固定commit本文を再取得して確認する。最終SHA・成果物URL・照合結果はチャットへ返す。文書の公開検査を科学検証PASSやCI再現へ格上げしない。

**公開確認後mandatory STOP。** 次の担当は**研究統括／ユーザー**で、候補1／2の追加静的調査を依頼するか判断する。**Codex**は新しい明示指示後だけ、その限定調査を行う。新しい設計・実装、保存値の再解析、build／compile、sampling／simulationは別途の対象限定承認を要する。詳細なGPT科学レビューも今回開始せず、必要と判断した場合に別の開始承認を受ける。停止中に次段階へ自動進行しない。

[HANDOFF]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/69f8e1cac23a4924ab81aebe399657a705024f6a/docs/research/pr_gaussian_singleton_closeout_handoff_20261024.md
[MECHANISMS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7aa911bf929d8d1469d27749bf1d83132a36a00e/docs/research/pr_next_resource_mechanisms_survey_20261024.md
[MAP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7aa911bf929d8d1469d27749bf1d83132a36a00e/docs/research/pr_algorithm_improvement_landscape_20261023/RESEARCH_MAP.md
[CARDS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7aa911bf929d8d1469d27749bf1d83132a36a00e/docs/research/pr_algorithm_improvement_landscape_20261023/STUDY_CARDS.md
[PFOLD]: https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/20d64c2fea1ed2d2188777df4f3d962ee94df486/review_response/README.md
[PFSOURCE]: https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/20d64c2fea1ed2d2188777df4f3d962ee94df486/src/trotterlib/product_formula.py
[PFCOST]: https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/20d64c2fea1ed2d2188777df4f3d962ee94df486/src/trotterlib/processed_cost.py
[PFVALIDATOR]: https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/20d64c2fea1ed2d2188777df4f3d962ee94df486/review_response/validate_morales_y8m10b_h2.py
[FINITE]: https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/6a1e54d5e830a20b8817791f9f1f65729815fc12/PF_first_study_final_synthesis_20260925.md
[BF]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/bf_current_hypothesis_closure_20261005.md
[BM]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/bm05_equivalence_and_method_delta_audit_v1.md
[CORR]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/interstep_correlated_randomization_prior_art_audit_20261013.md
[MIXED]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7af83266f96b714bb3fa2f1ef5e4bc89faf37db3/docs/research/qc_mixed_access_feasibility_v1_closure_20261013.md
[ORDER]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e3531956af6a6d359e2f1fb320e9a5179f41eb98/docs/research/track_a_pf_order_reversal_existing_evidence_audit_20261023.md
[STATUS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7aa911bf929d8d1469d27749bf1d83132a36a00e/VALIDATION_STATUS.md
[MANIFEST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7aa911bf929d8d1469d27749bf1d83132a36a00e/artifacts/validation_manifest.json
[RTESOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7aa911bf929d8d1469d27749bf1d83132a36a00e/src/trotterlib/rte.py
[SIGNALSOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7aa911bf929d8d1469d27749bf1d83132a36a00e/src/trotterlib/finite_rte_signal_validation.py
[CIRCUITS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7aa911bf929d8d1469d27749bf1d83132a36a00e/src/trottertracks/pr_dictionary_validation/circuits.py
[GATES]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7aa911bf929d8d1469d27749bf1d83132a36a00e/src/trottertracks/pr_dictionary_validation/gates.py
[PRPAPER]: https://arxiv.org/abs/2503.05647v2
[MORALES]: https://arxiv.org/abs/2210.15817v3
[COMMUTATOR]: https://arxiv.org/abs/1912.08854v3
[CPF]: https://arxiv.org/html/2409.08265v4
[MPF]: https://arxiv.org/html/1907.11679v2
[THRIFT]: https://arxiv.org/html/2403.08729v3
[DYSON]: https://arxiv.org/html/1805.00675v2
[RPF2026]: https://arxiv.org/html/2608.07720v1
[QDRIFT]: https://arxiv.org/abs/1811.08017v2
[QSWIFT]: https://arxiv.org/abs/2302.14811v2
[TETRIS]: https://arxiv.org/html/2308.03694v2
[PRHS]: https://arxiv.org/html/2608.29756v1
[QSHIFT]: https://arxiv.org/html/2604.26263v3
[SPRINT]: https://arxiv.org/html/2606.30741v1
[TAYLOR]: https://arxiv.org/html/1412.4687v1
[QUBIT]: https://arxiv.org/html/1610.06546v3
[CATALYSIS]: https://arxiv.org/abs/2007.14460v2
[SPECTRAL]: https://arxiv.org/abs/2502.15882v1
[SPARSEQSVT]: https://arxiv.org/html/2604.07448v1
