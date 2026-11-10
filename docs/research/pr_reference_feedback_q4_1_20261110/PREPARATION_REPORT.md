# Q4-1：H/E/Cの実装・人工検証・preflight

2026-11-10 JST、SERVER 2、primary Codex。研究階層はアルゴリズム改善→新A→Adaptive Feedback Readout→Q4-1。

**実装と人工検証を完了し、科学campaignの明示承認待ちでmandatory STOPする。** H/E/Cの新しい科学数値結果は存在しない。強X/Yの最小shot数、matched二項CIの順位、native費用の新しい改善率は今回計算していない。

基準は[`b159d721cc9e7cee39c246a2b55425e63796b249`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/b159d721cc9e7cee39c246a2b55425e63796b249)。同commitのQ4-0 REPORT／BASELINE_COMPARISON／PRIOR_ART_AND_NOVELTY／NEXT_NUMERICAL_VALIDATION_PLANを確認し、最後の文書を数値scopeの正本とした。既存Q branch `codex/algo-improve-new-a-total-cost-20261109`を再利用した。

## 1. 実装した範囲・再利用

| SOURCE | 実装内容 |
|---|---|
| [pr_reference_feedback_q4.py](../../../src/trottertracks/pr_reference_feedback_q4.py) | private 192-bit directed context、Hoeffding／等尾Clopper–Pearson CI、端点の内外包含、rational polygon＋annulusの角度union、非直交native軸、periodを残したresidual集合、Q用の単軸CI adapter |
| [pr_reference_feedback_q4_certify.py](../../../src/trottertracks/pr_reference_feedback_q4_certify.py) | 全countのinteger-box certificate、物理的二点witness、連続αの区間排除、prefix-only allocation、連続中心のlookup／境界padding、上限・下限・gap、Q-E全count検査、energy復元charge、Fraction会計 |
| [pr_reference_feedback_q4_check.py](../../../src/trottertracks/pr_reference_feedback_q4_check.py) | optimizerを呼ばないcertificate replay。count latticeの隙間／重複、同一中心での全配分排除、α全域、真の軸とCI内側でのwitness、CP rootの別経路exact rational tail検査 |
| [run_pr_reference_feedback_q4.py](../../../scripts/run_pr_reference_feedback_q4.py) | metadata-only preflight、承認後のH→E→条件付きC、fresh science namespace、途中結果保存、STOP、standalone checker入口 |
| [人工テスト](../../../tests/test_pr_reference_feedback_q4.py) | 14件のanalytic／mock fixture。H4の保存数値でテストを代用していない |

同一のjoint optimizer／outward CP inverseは既存Q0–Q3に無かったため、その不足部分を新規実装した。既存Q0 scalar逆写像を二軸jointに改名して流用することはしていない。

再利用したSOURCEは、Q0の`iv/exact/epsilon`（固定commitから純粋な関数定義だけAST取得）、Q1 `AxisCounts`、Q0 `inverse_endpoint/asin`の数式、Q3-Dの`pair/add/scale/subtract`（同じく純粋定義のみ）。既存科学runnerのtop-level campaignやbuilderは呼ばない。Clopper–Pearsonの既存scipy点APIは参考に留め、CI端点を点値のままcertificateへ採用しない。

## 2. 幾何と数値包含の根拠

### 共通潜在信号を保持する

同じ`z=ρ exp(i(ψ+u+b))`へ両軸CIを同時に課す。native軸は`(0,a)`、`a=float.fromhex('0x1.921fb54442d18p+0')`で、π/2への置換は無い。Q4-0 §3の共通coherence補題と既存Q3-Cのcore/readout metadataを前提とする。別noiseや別compilerへの保証ではない。

nativeのcos(a),sin(a)を192-bitで囲み、各成分のrational midpointをnormalにする。そのnormalの誤差をeとすれば、physicalな`|x|,|y|≤1`上でstripを各側eだけ広げた**有理外側polygon**は真のnative parallelogramを含む。この外側化は数値包含のためであり、X/Yに別々の半径を与える緩和ではない。witnessは外側polygon内という理由だけで採用せず、実際のnative cos応答がCIの**内側端点**内に入ることを再検査する。

### Polygon–annulusの全角度成分

有理half-planeの交点からconvex polygonをexact Fractionで作る。頂点のnormと辺の最短点による完全な内円／外円外の排除はexact。円と辺の判別式も有理数で判定し、非負の場合の交点を区間で囲む。

ρL>0のcompact可行集合で、可行方向の境界はpolygon頂点のray又はpolygon辺と内外円の交点のrayに現れる。円弧の途中は角度射影が局所的に開いており、新しい境界ではない。退化した点／線分も頂点・円交点で扱う。これらのcritical angleを区間で覆い、turn 0/1を加える。critical boxは安全側に残す。間のgapには方向の可行性が変わる点が無いため、midpointで**共通ρの区間交差が確実に空**と証明されたgapだけを除く。判定不能ならそのgapを残す。

atan2前にquadrantを分け、負の実軸を跨ぐboxを[-π,π]の大きな誤った枝へ潰さない。全成分をturnのunionで残し、Bを加え、ψを引き、periodを保持したままpriorと交差して最後にhullを測る。空集合は`INCONSISTENT_COUNTS`。prefix成功とCI coverage上では真値が残るので、この拒否は既存の末尾failureに含まれ、無料のpost-selection／retryではない。

### CIとcoverage

Hoeffdingは旧`ε=sqrt(2 log(2/α)/n)`を再利用。CP下端は`Pr_p(K≥k)=α/2`の増加tail root、上端は対称なrootから求める。tailは正の有限和を192-bit outwardで評価し、根は常に[0,1]の包含bracketを保つ。k=0/n、p=0/1は別処理し、0除算を避ける。bracket分解能は2^-160、評価がrootと重なって進めなければ広いbracketと`resolved=False`を残す。広い外側CIで幅が証明できた場合は安全側の十分証拠になり、狭いと推定して採用することはない。

exact二項tailの単調性と等尾反転により、真のCP区間のcoverageは固定nで≥1−α。数値外側区間はその集合を含む。QとX/Yで同じCI規約を使う。X/Yのunion boundはαX+αY≤A、QはA全部、未測定軸は[-1,1]。prefixで条件付けた後のfresh IID ±1という旧前提を変えない。固定nのCIを逐次停止へ流用しない。

### 予算・lookup・丸め

上限budget Nに対し、各中心が**実際の総数nX+nY≤N**から選べる。下界には、一つの同じψでN以下の全整数配分と全αを排除する必要がある。「各exact Nが別々のψで失敗」だけではこの下界にならない。checkerと人工testもこの差を検査する。

α boxの上端をXへ、逆側の上端をYへ同時に与える、より狭いCIへの楽観的緩和を使える。そのCIにも同じ二点が入れば、そのboxの全合法αがFAIL。PASSには特定の合法αと全count・中心cellの外向き幅が必要。grid PASSやL2用最良αで連続最適化を代用しない。

lookupは中心cell全体＋数値中心幅のpaddingを資格化する。selectorは観測可能な区間の有理midpointで選び、未知の真中心を選ばない。periodを跨ぐ入力も保持する。Q3-B energy中心をturnへ囲むadapterは、保存κとalias由来の`|t17 Ec|<2^20`を確認し、幅不足なら拒否する。この粗い包絡は`|hE|<π`とprefix誤差cから`|t17 Ec|≤q17 π+2c<2^20`で導け、新しい物理系条件ではない。

最終判定は数式上の位相幅だけでなく、`phase_width+t17(κ+ωE)≤4/5`まで要求する。Q-Eのfeedback算術を0にせず、旧Q専用ωQ0を新inverseの証明に流用しない。新端点の外向き誤差は新hullそのものに含まれる。旧PF／finite-bit／native diskはρL/Bに一度だけ保持する。

## 3. 実施した検証とpreflight

専用既存venv（mpmath1.3.0、Python3.12）で**人工14テストPASS**。[ARTIFICIAL_TESTS.log](../../../artifacts/pr_reference_feedback_q4_1/2026-11-10/preparation/ARTIFICIAL_TESTS.log)に全文を保存した。共通半径の不可行例、分裂、接線、period、native非直交角、人工2×2 readout恒等式、n0、CP端点／tailのexact照合、連続α排除、予算量化順序、丸め、lookup境界、mock native価格相関、checkerのoptimizer不呼出しを含む。

初回人工テストではFractionをmpmath区間へ直接渡す経路で変換例外が出た。明示的な外向き変換へ修正し、同じfixtureで確認した。既存科学結果を変更する修正ではない。

[PREFLIGHT.json](../../../artifacts/pr_reference_feedback_q4_1/2026-11-10/preparation/PREFLIGHT.json)は固定Git objectの入力12件のhash／schema、旧資格status、純粋helperのSOURCE一致と新SOURCE hashを記録する。**Arithmetic／model_for／CI／optimizer／価格会計を呼ばないmetadata-only経路**である。H4数値bindingの実計算と実データでの計算性能は承認後のscopeであり、preflight PASSがH/E/Cの科学PASSを意味しない。

全検査は著者自身による。checkerの制御経路とCPのexact算術経路は分けたが、幾何・CIのprimitiveは共有する。別研究者・別AIによる独立科学レビューとは称しない。

## 4. 実行準備状態と残る限界

実行入口・SOURCE・固定入力・出力namespace・判定・STOPは[EXECUTION_SCOPE](EXECUTION_SCOPE.md)へ固定した。承認後にH→E→条件付きCを起動できるコードはあるが、実H4での数値binding／広い範囲の性能試験は未実施。最適解や短時間完了を保証していない。

特に全count、連続α、連続中心は重い可能性がある。count boxの枝刈り、配分内CI cache、同じ中心でのallocation cache、独立taskのprocess並列化を使う。witness探索は候補生成のheuristicであり、見つからなければFAILとはしない。2^-128の連続leafや192-bitの包含で結論が閉じなければUNRESOLVEDとgapを報告する。可行解しか無いのに最適値と呼ばない。実行を速めるための候補範囲・精度・CIの無断変更はしない。

Cでは保存trajectory期待単価を維持し、prefix履歴に関するworst-caseと区別する。共通prefixを差から厳密に消し、QとYの同じ価格変数を使う。可行policy、費用下界、最適性gapを分ける。新しい価格目的の探索、未知の費用の0埋め、native指標の独自合算はしない。

## 5. 今回の終了

新科学数値campaign／保存科学値再解析／H4 build・compile・simulation・sampling／価格campaign／CT・総Tは**0件**。旧Q0全n探索、Q3-B count監査、Q3-D campaignは再実行していない。旧SOURCE・input・raw・result・grant・ledger・marker・STOP・KEEP、他Track／worktree／共有環境は不変更。新しい承認管理基盤やgrantは作っていない。AI subagentなし。

[STOP.json](../../../artifacts/pr_reference_feedback_q4_1/2026-11-10/preparation/STOP.json)は今回の準備完了だけを記録する。H/E/C RESULTSは作成しない。commit/push・remote照合後**mandatory STOP**。

次の担当は**ユーザー**：本公開SOURCEとEXECUTION_SCOPEに限定したH/E/Cの科学数値実行を許可するか判断する。**Codex**はその承認後に限り固定campaignと検算を実施する。正式な新規性・主要claimの判断は**GPT／ユーザー**に戻す。
