# PR総資源削減：次期改善機構の事前調査

2026-10-24 JST。**調査・技術比較による候補選別であり、研究方針の正式採択、新算法の設計、科学検証ではない。**

最優先の検討候補は、**同じfinite targetを保つwhole-trajectory importance samplingについて、有限confidence・有限bit・価格取得費を含む実行可能性を見極めること**。既知の最適確率式を新法とせず、PRのordered pathを安価に扱える追加構造が残るかを入口で判定する。次点は辞書とlawの共同選択、補助候補はRAのnormalizer／有限bit生成費の削減。**独立新規性・同精度総資源改善を確立した候補はない。実装・実験開始を推奨確定したものでもない。**

## 1. 範囲・固定起点・調査方法

- Repository：`HIROMU1015/Partially-Randomized-Trotter`。
- 固定起点：[`ab3b2c30df6333ceefd408f59e48c5ae0de4563f`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/ab3b2c30df6333ceefd408f59e48c5ae0de4563f)。専用branch：`codex/pr-next-mechanism-survey-20261024`。独立worktreeで本Markdownのみ追加する。他branchへmergeしない。
- 正式共通指示：ユーザーが明示指定した `research_project_instructions_unified_2026-10-23.md` を全文読了。元workspaceの未追跡ファイルを読取り、SHA256は `7ce342dbf7086424f1afd837d8c274687bc28e4838601f5ad96f5b626d95667f`。起点に同名のtracked fileはなく、コピー・commit・改変しない。
- 起点の `PROJECT_MAP.md`、`研究概要・現状.md` 全文、研究目的、関連validation status／manifestを先に確認。10月23日の横断地図・Phase 2A／2Cの監査を再利用し、関連する旧Track B報告を固定commitのGit objectから読む。既存レビューの新規性評価を無条件に追認しない。
- 新しい科学計算、保存数値の再集計・再ランキング、Hamiltonian生成、sampling、simulation、回路build／compile、SOURCE import／変更、算法実装は行わない。外部論文は関連箇所のみ照合し、網羅的な再レビューを繰り返さない。
- Gaussian singleton終了整理の並行branch、元workspaceの変更、既存SOURCE／raw／grant／ledger／marker／STOPは操作しない。共有の概要・索引・manifestも編集しない。本書の提案は既存statusを変更しない。

## 2. Track B成果と先行研究の対応

以下の「支持」は各固定契約内の既存証拠に限る。旧資料の取得不足、技術停止、性能負結果を区別する。[横断地図][MAP]と[Phase 2A重複監査][OVERLAP]を入口に、[D1–D3監査][DART]、[Parent closure][PARENT]、[相関監査][CORR]、[G6整理][G6]、[C1整理][C1]を優先した。

| 系列 | 既存成果／現状 | 既知法との対応と今回への制約 |
|---|---|---|
| ordinary paired finite RTE | involution、Euler pairing、ordered first moment、normalizationと信号補正が基盤 | 原PR、Wanらの次数pairingが先行。Pauli以外のframe involutionも原PRに記述がある。一般involution接口を新発明としない |
| R0／R1、G6 Return Aggregation | 平均を保存する係数再構成、formal returnの集約、局所queryと有限bit生成の構成成果 | RAは構成の枠組み、General Fullはその生成方式、Closed P5は内部特殊化。Euler、Green再帰、identity吸収の既知部分を除いた独立性は未確立 |
| G7–G10 | 固定synthetic providerの同次数・同finite targetに条件付きT利益。一部CTS対照にはCXの利点 | native価格、normalizer取得、zero-fill、十分Nに依存。全座標優越、任意IS最適、分子PR全体の利益ではない |
| D1–D3 | finite target、bias／confidence、native価格、全方式共通のliteral境界相殺を接続 | 同精度のshot×回路費会計自体は既知。保証や価格の変更で構成順位が変わる。加法価格＋局所相殺はwhole-wrapper compileの代用ではない |
| Parent allocation／S4 | common配分のfamily内L1最小性、配分変更による角度・価格・weightの交換、訂正KKTの限定比較を保持。現在の数値探索は終了整理済み | Euler＋線形分割、cost-aware IS、標準KKTの部分は既知。r=1・既知development入力の結果を多stepへ移せず、同fixtureの配分／proposal追加を次期独立機構としない |
| C1のH4移植 | logical差は小さく、empty parentでもchild rotationが残った。native・共通十分N・完成coherent estimatorは未取得 | return数やparent短縮だけでは高頻度child／basis費を消せない。古典時間には実装・cache・計測境界の交絡があり、方式の最適CPU順位ではない |
| G6 finite-bit／B3 live接続 | 限定workloadで生成・角度・検査・保存のlive経路を観測済み | 古典費用が全く未計測なのではない。同budgetのClosed P5／CTS、全N償却、大規模scalingとの比較は未確立。[既存handoff][G6LIVE] |
| M1-S shared ± labels | 適用域付き理論と局所信号witnessを保持。visibilityとphase biasの交換、whole-cost／多round RPEは未確立 | 元finite RTEと異なるcoherent信号を扱う。単なる同mean ISのarmに混ぜず、方向制御qDRIFT等への既存対応を維持。[統合整理][M1S] |
| 混在辞書F1／限定F2 | 全S／混在SP・PS／全Pを同じ物理分割で扱う接続を検証。F1の登録正しさ、F2の全件数値検査とactual 1-shot集計が完了 | 既知の表現変更＋canonical RTE＋共有＋有限選択。独立生成機構は未確認。F2のγ／統計資格／認証Nは未成立 |
| 時間相関／history sampling | marginal保存だけではordered mean保存にならないこと、whole-path再重み付けの条件を既存監査が明示 | resource-optimal IS、PRHS、qSHIFTが先行。channel／observable matchingをcontrolled operator meanの保証に移すには追加条件が必要 |
| 固定辞書G5等、Gaussian積、P-A | 指定classのCTS比較は負結果、Gaussian積は追加手順未定義、P-A interval固有差は強対照で消失 | 全samplerのno-goではない。閉じた探索の単なる拡張・改名や、既存共有による効果の再計上を避ける |

### 比較に直接使える最新証拠と、その限界

限定F2は、**4 mode・2粒子、occupied(0,1)、合成density-square Hamiltonian、L_D=1・残差2fragment、T=1/5、q=1、δ=1/5の一点、r=2、h=1/10、K=0**。分子geometry／AO basis／分子DF rankは非該当。SOURCE `9468c7089727f5648d9f59cddbaebe0a42f5f0a0`、Qiskit1.3.0／opt1／seed17／rz,sx,x,cx／topology-free、同じfull Hadamard wrapperである。

保存報告の1-shot期待RZはSS／SP／PS／PPの順に **70.825335／45.161578／53.950984／35.445140**、全計測回路指標の小さい順は **PP < SP < PS < SS**。一方、SSはλとBtotが小さい。これらは[実行報告][F2]の転記であり、本調査で比率・N・順位を計算していない。有限bit lawと全caseの費用は保存されているが、数値PASSをγのcertificateへ格上げしない。[要約JSON][F2JSON]、[資格確認][QUAL]、[validation status][STATUS]、[manifest][MANIFEST]。

F1は同じ合成fixtureでもq=2、δ=1/10、r=2、h=1/20、K=2であり、F1のliteral価格を上のF2 actual価格へ混ぜない。いずれもcommitted local evidenceで、CI／外部再現ではない。F2は準備X0X1・readout・measure込みで、プロジェクト主目的のprep除外費用とはscopeが異なる。compilerの非加法性があるため、共通診断回路の数を単純に引いて換算しない。[F1報告][F1]、[F2報告][F2]。

C1の引用条件は**linear H4、1.00 Å、STO-3G、DF rank12、R1 generation-prefix L_D=3、13 basis／324 labels／8 system qubits、T=4/5、q=1、r=4、m=5、microstep=1/5の一点**、`TAIL_ONLY_PLUS_SCALAR`。各方式1024 pathは探索標本数で、必要shot数ではない。経験的weight moment／最大観測weightから母moment／一様rangeや新Nを作らない。分子H4の証拠を上の4-mode合成F2へ移さない。[C1資源・限界][C1]。

## 3. 何が解決済みで、何が残るか

| 区分 | 本調査で引き継ぐ内容 |
|---|---|
| 理論・構成として既知／既存 | 確率変更とlikelihood補正による同mean化、費用×二次momentの最適law、Euler pairing、commonのL1最小性、RA内部のClosed P5、合法な隣接逆対相殺 |
| 限定的な肯定結果 | G9／S4等の固定native価格における資源交換、F1／F2の登録数値的正しさ、F2の1-shot費用差。総資源の一般的改善とは区別 |
| 負結果／区切り | 固定辞書G5比較、generic Fullの登録対照への不利、Parent同development探索、P-A interval固有差。C1追加計算の凍結も維持 |
| 未解決の技術問題 | 安いwhole-path確率／normalizer／costアクセス、丸め後のsupportとweight、compiled実装まで含むγ、range／confidence／ceilの資格、cold／online古典費用 |
| 未確立の科学的主張 | 強い既知法を超える独立機構、他入力への成立条件、同精度総資源利益、分子PR／全RPE優位性。H12係数や最終全round総費用は今回も未決定 |

## 4. 同じtask・精度・費用へ戻す条件

### 4.1 finite targetとsampling law

固定H、scalar、D/R、順序、signed time、δ、q、r、Kに対する補正済み有限演算子を `A_K` とする。比較するのはそのcontrolled first momentとX/Y信号であり、conjugation channelだけの一致では足りない。Uと−Uはsystem channelが同じでもcontrol後の相対位相が異なる。

既存[相関監査][CORR]の記号で、ordered trajectoryをω、canonical正確率をp、全normalizationをB、実行unitaryをVωとすれば、

\[
A_K=B\sum_\omega p_\omega V_\omega,
\qquad W_\omega=Bp_\omega/q_\omega,
\qquad \mathbb E_q[W_\omega V_\omega]=A_K.
\]

ここでproposal `q_ω>0` はpのsupport全域で必要。実測axis outcome Y∈{−1,+1}に対し、

\[
M_2=\mathbb E_q[(WY)^2]=B^2\sum_\omega p_\omega^2/q_\omega,
\qquad R_\infty=B\max_\omega p_\omega/q_\omega.
\]

これは**既存式の再掲**で、新しい導出ではない。実分散はM2からaxis平均の二乗を引く。exact path expectationの分散だけで量子測定雑音を落とさない。shot間fresh IIDと、同じshot内のlabel間の独立性は別条件である。

既知ISの `q*∝p/√c` は、正の既知whole-path cost cに対する `E_q[c] M2` の最適lawであり、原定理は非加法cを排除しない。**「全trajectory費用を使う」だけでは新規性はない。** そのnormalizer・sampling・価格の取得が安いこと、range制約・bias・有限confidence・integer N・古典費用込みの最適性は別問題。zero-cost経路には境界の扱いが必要で、勝手なfloorを置かない。[Cugini–Atif–Subaşı §II Thm.1／§III Thm.2][IS]、[既存対応監査][CORR]。

同じpath実装を単に再重み付けするISは既存の実装biasを消さない。一方、角度や分解まで変えれば別の生成法になり、同finite targetを改めて照合する。qDRIFTの任意分布法、qSHIFT／PRHSの高次化を、そのまま現finite RTEの同mean対照にしない。

### 4.2 finite-bit bias・shot数・総費用

F2の既存資格整理では、共通finite target zKと実装平均μの差を覆う `|μ−zK|≤γ`、残余 `s=η−γ>0`、補正outcomeのrange `R∞`、fresh IIDが必要。二axisをunionする既存Hoeffding policyは

\[
N_X=N_Y=\left\lceil
\frac{4R_\infty^2}{(\eta-\gamma)^2}
\log\frac4{\alpha_{\rm arm}}
\right\rceil.
\]

十分数であって最小必要数ではない。4arm同時保証と各arm個別保証を区別し、η／α／γやscopeを今回採択しない。M2目的に有利なproposalでもrange主体の式で有利とは限らず、既知Bernstein等を使う場合も条件と配分の採択が要る。[F2資格・式の根拠][QUAL]、[既存Bernstein会計][CORR]。

γには実law、係数／weight、角度、Gaussian分解、control／compileなどの適用誤差を対象に応じて戻す。binary64をexact有理数として読むこと、P0数値閾値内、90／128桁の一致だけでは、実装平均と共通targetの差の保証にならない。F2の既存未資格を維持する。

量子費用は各metric jについて `G_j=Σ_(round,axis) N E_q[c_j(ω)]`、古典費用は辞書生成、前処理、価格／合成取得、確率構築、online sampling、cache miss／保存を別に記録する。T・CX・RZ・depth・CPU秒を単位なしに加算しない。同一runtimeへ換算できる根拠がある場合だけ共通目的にする。状態準備除外を主条件にする場合も、Nが変わる準備費感度を別に残す。準備・readoutを含む文献値／F2値を混用しない。

### 4.3 強い対照・情報取得条件

- canonical ordinary／partial、同finite targetのP3+tail／General Full／Closed P5／collected CTSのうち、当該入力で実装可能なもの。m／r／Kを変える場合はfinite target変更とphysical近似誤差を別途戻す。
- 同じ辞書pool・price情報・likelihood取得を与えた既知whole-path／cost-aware IS。候補だけに全compile表やexact state／E0を与えない。
- 同じactual adjoint、非制御basis＋中央control、support/full run共有、合法なD境界融合、phase保持compiler、同profile。ordered非可換項を費用の都合でsortしない。
- 全列挙oracleは小系の参照上限／下限の用途とし、運用時にはその構築・記憶費を払う。既存保存表は開発入力として再利用できるが、独立holdoutや無料production oracleではない。新lawへ旧IID較正proxyを無条件に流用しない。

## 5. 絞った候補3件

### A. 有限confidence下で実行可能なwhole-trajectory IS（最優先の検討候補）

**改善原理・資源：** 同じA_K・同じ回路poolを維持し、高価なordered pathの頻度を下げる。その代償で増えるM2／range／十分Nと価格・likelihood取得費を戻して、量子総費用と古典費用が両立する条件を調べる。

**既知法で済む部分：** 上記change of measure、最適law、既知のregularization／mixture、有限confidence不等式。Markov化・switch削減・確率変更そのものは独立新法ではない。過去の相関監査は既にこの点とmean破壊の反例を扱った。

**PRで追加する意味：** ordered product、child、frame run、D挿入の構造を利用して、非列挙で確率・weight保証と費用情報を取得できる追加差があるか。**具体的な新生成規則は現在なく、PR固有性も未確立。** 既存ISをPRへ接続する応用価値と、新アルゴリズムの価値を分ける。

**再利用・最小次段階：** [相関監査][CORR]のp/q・moment/range式、[Parent SOURCE][PASRC]の`digital_proposal`／`reweight_events`／`event_statistics`、F2保存law・case費用・phase保持wrapperを関数／artifact単位で対応付ける。旧Parentの加法event価格接口がwhole-pathに適合するとは仮定しない。次は**値を評価せず**、既存接口が何を供給でき、何が欠けるかを1枚にまとめる段階。

**追加証拠・リスク：** 安価なfull-support samplingとnormalizer、同confidenceの保証、γ、実whole-path価格との対応、cold／online費用が必要。range悪化、長列での費用集中、取得費が利得を消す可能性がある。費用が加法かつ独立に集中する既知の漸近議論を、非加法PRへの一般no-goにも改善保証にも使わない。新規則が既知ISの通常実装に留まれば、独立算法路線は支持しない。

### B. 演算子辞書・回転配分とlawの共同選択（次点、Aと重複実験にしない）

**改善原理・資源：** 同じH_Rと補正finite polynomialを異なるPauli／Gaussian-frame辞書、または既存RA内の適格回転配分で生成する。basis／child／parity費と係数mass・B・range・Nを共同で交換する。単なるAの確率変更と違い、実行する演算子／角度pool自体が変わり得る。

**既知法で済む部分：** 元PRの残差Pauli化／異frame involution、cost-awareな可換fragment割当、SPSの辞書設計とL1／残差最適化、Parent allocation、canonical有限RTEとIS。S/P二値selectorとrun価格DPは[既存方法差監査][DELTA]で既知構成の組合せに分類され、反可換pairへの置換も既知unitary partitioningへ対応した。新名称で再提出しない。

**PRで追加する意味：** 個々のatom単価では決まらないordered全回路境界と、辞書で変わるcanonical event law・normalizationを同時に扱う実装可能な成立条件。**現在の4arm選択には独立した量子生成中核は確認されていない。** SPSのsystem process一致だけからcontrolled相対位相が保証されるとしない。

**再利用・最小次段階：** F1／F2の保存B、`prototype.canonicalize_arm`、`make_tail`、typed `circuits.py`、`pricing.expected_literal_price`、独立`reference.py`を再利用候補にする。literal価格DPはactual compiler価格ではなく、F1/K2とF2/K0を混同しない。まず既存4arm＋同pool既知選択＋同IS機会の比較条件をAの接口表へ併記する。

**追加証拠・リスク：** 全armのγとconfidence、全体Pauli collect／辞書構築費、粒子数非保存atomのfull-space実装、価格取得／memoryが必要。現F2では混在armはPPより1-shot費用が高く、混在固有の総利益を示す証拠はない。最小λ・最少項数だけを選ぶ方法や、候補を増やすだけの探索は優先しない。RA側の再探索・新角度合成も別採択事項。

### C. RAのnormalizer・有限bit生成アクセスの改善（補助候補）

**改善原理・資源：** 理想return ensembleを維持し、局所packet／normalizer・離散selector・角度情報の取得と保証の費用を減らす。既知normalizerを安く取得できればzero-fill等の統計会計を変え得るが、**単にcacheを速くする同law実装では量子Nは減らない**。

**既知法で済む部分：** Green再帰、Closed P5、累積整数／alias型sampling、memoization、on-demand合成、既存finite-bit enclosure。これらの通常実装は新しいPR算法ではない。General FullとClosed P5のlaw／normalizerアクセスの違いは既存成果として保持する。

**PRで追加する意味：** 許されたDF／local-query情報だけで、列挙せずnormalizerやweight保証を取得できる追加の構造・bit／memory保証があるか。これは未確立。全列挙を隠した「安いlocal query」を認めない。

**再利用・最小次段階：** [G6生成接口][G6SRC]の`GeneratorSession.parent_packet`／`sample`、[G7生成接口][G7SRC]の`FullReturnGenerator`／`upper_normalizer`、[後続live接続記録][G6LIVE]、G8のcold/warm記録、C1のsetup／online／cache記録を再利用候補とする。初期generatorと後続live APIの版を一致と仮定しない。次は既存計測の境界・欠測・bit依存を静的に対応付けるだけ。過去の実行承認・grantは移さない。

**追加証拠・リスク：** C1には古典費用が記録されているが、公平な最適実装比較やscalingは未取得。現実に支配的か、改善が証拠出力削減だけかも未確立。A／Bに必要な取得が律速と分かった場合に限定して検討する。独立RQがなければ補助実装に留め、C1凍結を解除しない。

## 6. 優先判断・見送る方向・次の担当

**技術調査としての優先順はA → B、Cは必要時のみ。** Aは既存native回路poolを変えず、Bより少ない変更要因で「shot×actual費用」の改善可能性を識別できる。既知最適lawがあるため、それとの差が取得・保証に残るかを早く判定でき、SOURCE／保存費用の再利用度も高い。これはAの性能が良いとの予測ではない。現F2のγ未成立により、今すぐ認証Nを算出する段階にはない。

「確率を変えれば新算法」「Bが小さければ総費用最小」「GaussianならPauliより安い」「returnでparentが空なら回路不要」は採用しない。Gaussian singletonの選択器、同development Parent探索、閉じたgeneric Full探索は再開候補にしない。新しいsigned/history高次化は、既存PRHS／qSHIFTとの重複とcoherent保証の未接続が大きく、具体差のないまま今回の上位候補には置かない。

| 担当・段階 | 具体的な作業 | 承認・終了境界 |
|---|---|---|
| **ユーザー／GPT：次の判断** | Aの追加差を調べる限定technical調査に価値があるか、既知法の応用実装を成果として認めるかを判断。Bはその比較条件、Cは依存課題として扱うかを決める | 本書は採択しない。重要な新規性／保証／研究方向の詳細レビューは、必要と判断した場合に別の開始承認。既存レビューの反復を自動要求しない |
| **Codex：次に提案する最小作業** | A／B共通の「p・q・weight・R∞／M2・γ・whole-wrapper価格・取得費」を既存関数／保存artifactへ対応させ、欠落接口と既知対照との差を1枚に固定。新値・新law・擬似コード・実装を作らない | 今回のSTOP後は自動着手しない。追加のread-only調査として依頼された場合に行う。新規則／新保証が必要なら未確立として返す |
| **ユーザー／GPT：計算前の科学条件** | 共通finite targetとphysical taskの区別、metric／prep scope、η・α・同時保証範囲、γの根拠・配分、既知対照と情報予算を採択 | 未解決のままNやwinnerを認定しない。探索済み表はdevelopment扱い。独立性の単位も先に定める |
| **Codex：将来の実装・検証** | 具体的な追加機構が成立すると判断された場合のみ、適合する既存コードを再利用して新SOURCE・限定比較案を準備 | 実装の承認、数値再解析／build／compile／sampling等の科学実行承認は今回に含まれない。旧grant／STOPを流用しない |

最小作業の結果が「既知IS／既知辞書選択の通常接続のみ」であれば、そのまま報告して算法新規性の探索を止める。性能の識別に保存値再解析が必要でも、それを今回の資料読取りに含めない。新しい方法を無理に作らず、応用／実装改善として継続するか研究を区切るかをユーザー／GPTへ返す。

## 7. 一次文献の確認範囲

既存[D1–D3監査][DART]・[相関監査][CORR]・[Phase 2C監査][PRIOR2C]を再利用した。今回の直接取得は以下の箇所に限定し、全文精読・全定理再証明・世界的priority調査とはしない。別taskの改善倍率は転用しない。

| 一次資料 | 今回直接確認した箇所／採用する限定的な対応 |
|---|---|
| [Cugini, Atif, Subaşı, Resource-Optimal Importance Sampling, 2603.13495v1][IS] | §II Eqs.(4)–(10)、Thm.1、§II.1、§III Thm.2、§V.1関連箇所。全回路parameterのcost×weight二次moment、bias不変。finite-confidence最適性とはしない |
| [Campbell, A random compiler, 1811.08017v2](https://arxiv.org/abs/1811.08017v2) | 公式abstract／版情報。係数normに比例したqDRIFT samplingは既知。原論文の倍率をPRへ移さない |
| [Kiss, Grossi, Roggero, Importance sampling for stochastic quantum simulations, Quantum 7, 977 (2023)](https://quantum-journal.org/papers/q-2023-04-13-977/) | 出版社abstract／summary。任意sampling・bias／fluctuation・実装費・composite channelの同時考慮が先行。詳細定理のPR移送は未確認 |
| [Günther et al., PR, 2503.05647v1](https://arxiv.org/pdf/2503.05647v1) | §VII.A–Bの残差Pauli化、異軌道involution、基底費。直接照合はv1。v2のpairing等は既存D1–D3監査に帰属。HTML v2取得は404で、今回v2全文を読了とはしない |
| [Mukhopadhyay–Wiebe–Zhang, npj QI 9, 31 (2023)](https://www.nature.com/articles/s41534-023-00697-6) | Abstract／Introductionのcontributions・Algorithm 1の説明。可換群と非Clifford費用を結ぶ割当は既知。Algorithm詳細対応は既存Phase 2C監査を継承。後の再取得エラーを初回取得成功と区別 |
| [Koczor, SPS, 2402.15550v2](https://arxiv.org/html/2402.15550v2) | §II.1–2、Statements 1–2とdesign matrix。process辞書・L1／残差と測定負担の交換。controlled-space embedding／取得費は別条件 |
| [Peetz–Smart–Narang, SCU／CTS, 2407.21095v2](https://arxiv.org/html/2407.21095v2) | Abstract、Supplementary Note 4.1の総誤差分配とsample×CNOT比較。同精度総資源会計そのものを新規としない。matched CTSの詳細は既存監査を再利用 |
| [Cugini, PRHS, 2608.29756v1](https://arxiv.org/html/2608.29756v1)／[Lee–Choi, Adaptive sampling, 2604.26263v3](https://arxiv.org/html/2604.26263v3) | 公式本文の版／構成を直接取得。式の版差、signed block、保証taskの詳細は既存相関監査に帰属し、今回再導出・再検証しない |

## 8. 再利用SOURCE・証拠の固定参照と完了境界

起点の `src/trottertracks/pr_dictionary_validation/` では、`prototype.py`のtyped atom／collect、`circuits.py`のbasis共有／central control、`pricing.py`のliteral期待価格、`f2_execution.py`のfinite-bit正規化と診断扱い、`f2_statistics.py`の条件付きshot gateを静的に確認した。SOURCEはimportしない。旧G6／G7／Parentは下記固定版の公開関数・関連箇所を確認し、実行・変更しない。

本文中リンクの関連文書を直接読むことと、過去監査が参照した全raw／全SOURCEを再監査することを区別する。今回はCSV／rawを開いて再解析せず、保存JSONのstatus／countsとmanifest登録、報告の既存値を確認した。旧private rawの保全を実ファイル全件hash検査したとは主張しない。

本変更は**専用Markdownの追加のみ**。固定起点との差分・文書内ローカル参照・空白を確認し、専用branchへcommit／push後、remote refと固定commitの本文を再取得してGit blob／bytes／SHA256を照合する。最終commit・照合結果はチャットに返す。文書検査は科学的正しさや新規性のPASSではない。

**公開照合をもってmandatory STOP。** 新算法の設計・実装・科学検証、数値再解析、γ／N算出、Gaussian singleton終了整理への介入、他branchへのmergeへ移行しない。

[MAP]: pr_algorithm_improvement_landscape_20261023/RESEARCH_MAP.md
[OVERLAP]: pr_algorithm_improvement_phase2a_20261023/STRONG_BASELINE_AND_OVERLAP_MATRIX.md
[PRIOR2C]: pr_algorithm_improvement_phase2c_design_20261023/METHOD_DELTA_AND_PRIOR_ART.md
[DELTA]: pr_algorithm_improvement_phase2c_method_delta_20261023/PHASE2C_BASELINE_DECOMPOSITION.md
[F1]: pr_algorithm_improvement_phase2d_f1_rerun_20261024/EXECUTION_REPORT.md
[F2]: pr_algorithm_improvement_phase2d_f2_resources_rerun_20261024/EXECUTION_REPORT.md
[QUAL]: pr_algorithm_improvement_phase2d_f2_qualification_20261024/README.md
[F2JSON]: ../../artifacts/pr_algorithm_improvement_phase2d_f2_resources/2026-10-24-r1/RESULT_SUMMARY.json
[STATUS]: ../../VALIDATION_STATUS.md
[MANIFEST]: ../../artifacts/validation_manifest.json
[IS]: https://arxiv.org/html/2603.13495v1
[DART]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/d1_d3_prior_art_claim_evidence_and_gpt_handoff_20261014.md
[PARENT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/track_b_parent_allocation_research_closure_20261015.md
[CORR]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/interstep_correlated_randomization_prior_art_audit_20261013.md
[G6]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f750a23f023d75c8c4fbcabc6937d6953ce61f2/docs/research/track_b_g6_v1_2_draft_review_integration_20261022/handoff.md
[C1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/185ec4e9d6591f4ee8d7b047c2697bed7be7bc2e/docs/research/track_b_c1_freeze_synthesis_20261021/C1_RESOURCE_EVIDENCE_AND_LIMITATIONS.md
[PASRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/src/trottertracks/algorithm_codesign/parent_allocation.py
[G6SRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/af75af03c0b334c29b5a4311bd2b4578a999460c/src/trottertracks/algorithm_codesign/g6_f3_generator.py
[G7SRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/af75af03c0b334c29b5a4311bd2b4578a999460c/src/trottertracks/algorithm_codesign/g7_generator.py
[G6LIVE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a48b3c8244fb5d932e0d6bf4a5de6bb573ce35ed/docs/research/track_b_g6_w2c_b3_limited_pilot_16896_authorized_20261020/handoff.md
[M1S]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f079dbe67bb35eff35613857b169fdf812af4b8/docs/research/m1s_integration_20261017/integrated_research_summary.md
