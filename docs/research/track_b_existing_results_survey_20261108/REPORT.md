# Track B既存成果・公開差分の調査（SERVER 1）

2026-11-08 JST。**K2＝有限P3にはG6のm=3／Closed P3を数学的な接続候補にできる。G9のClosed P5をそのまま使うとtargetが変わる。H4でnormalization低下を認証shot数・総native資源改善へ結んだ保存結果は未発見。** 新しい手法採択・科学的GO・研究再開は行っていない。

調査は既存文書・SOURCE・保存JSONのfield・Git object・archive bytesの読取照合のみ。研究moduleのimport、旧test再実行、新sampling／simulation／compile／synthesis／solver／N算定は0。固定SOURCE・raw・STOPは無変更。既存local evidenceをimmutable CI／外部再現へ昇格していない。SERVER 2等への接続・転送・メッセージ送信はしていない。公開可能な所在・hashは[evidence_inventory.json](evidence_inventory.json)、非公開原本の詳細は公開対象外のtask-local `PRIVATE_LOCATORS.json`へ分けた。

## 1. 公開済み／ローカルのみ／未発見

公開mainは実照会時 `546bca7ed2cece59b714b986a264d7b0b0871a01`。repositoryのvisibilityはpublic。旧branch名の消失・mainへの未収録を「非公開」とは扱わない。

| 区分 | 今回確認した内容 | 範囲・限界 |
|---|---|---|
| 公開済み・mainに存在 | G6数理／finite-bit監査、G7/G9/G10報告、F1–F3/W2、C1報告・SOURCE manifest・小summary、研究C sampling-law調査 | 代表10 pathのGitHub readback blobをlocal Gitと照合。主なSOURCE実装も一致 |
| 公開済み・固定履歴に存在 | G9 v2の大きな`result_v1.json`、C1のraw `.partNNN`、G6 v1.2草稿など、main未収録の証拠 | 保存carrier [5f750a…][HIST]。live公開tag `archive/research-map-20261102/33424d205aa9` のOID `33424d205aa918498e1ef54beb94ab67612bdec8`にその履歴が含まれる。上記raw result／archive index／草稿もGitHubから取得確認 |
| SERVER 1のローカルcopy | `/tmp/track-b-g6-v1-2-draft-review-integration-20261022` | HEAD=5f750a…、sparse checkoutはclean。新しい未公開科学成果とは認定しない |
| ローカルのみ・非公開 | ユーザーのレビュー原本、G6草稿作成時の管理用取得／保全／公開照合記録、親checkoutの未commit資料索引 | レビュー原文・private入力・fingerprintは新たに公開しない。v1.2には既に限定解釈の公開記録がある。別の完走済み科学結果は今回の調査箇所では特定できなかった |
| SERVER 1で未発見 | 文書が指す旧Track B実行worktree／runtime原本 | `/home/AbeHiromu/projects/partially-randomized-trotter-worktrees`、`/home/abe/Project/prt-worktrees`はこのhostに存在しない。Gitにあるraw archiveは取得可能。元の原本消失やSERVER 2側不存在は意味しない |
| 証拠未発見／未確立 | H4の共通認証N、全support strict native価格・誤差、全round total-cost比較、未提供のG6/G7レビュー付属selfcheck SOURCE | 「未実施」と記載されたものと、別環境での所在未確認を混同しない |

歴史carrierの関連path集合3,462件をmain treeと照合すると、同一blob2,864、相違2、未収録596。相違2件は`docs/tracks/algorithm_codesign/README.md`と`src/trottertracks/algorithm_codesign/df_correction_comparison/README.md`で、調査集合内の実装差は見つからなかった。件数は研究数・成功数ではない。集合の定義・個別113 pathのbytes/SHA256/blobはJSONに記録した。親main checkoutの未commit変更は保全し、公開main起点の専用worktreeでこの報告だけを追加した。

## 2. 関連成果とSOURCE→結果

各表のSOURCEは実行用固定点、結果commitは保存点。統合carrierを実行SOURCEと呼ばない。一般式は既存監査からの整理であり、今回新しく証明・評価していない。

| 系列 | 固定SOURCE／結果 | 成果と読み方 |
|---|---|---|
| G6 | 監査固定点`28cfabb1d47e…`／[数理監査][G6]・[finite-bit][BITS] | 奇数有限P_mのreturn集約・正値性・Green局所query・zero-fill。16既存focused testsは形式恒等式。native成功・独立新規性の認定ではない |
| G7／G8 | G7 S=`ab2549f41b35…`、結果`a689694080f4…`／[G7][G7] | G7はexact controlled-query/helper仮定下の費用会計。m5の登録対照への利益とm3のClosed P3対照への不利を保持。G8はon-demand RZ miss／cold-warm traceで、分子全support保証ではない |
| G9 v2 | S=`0ef2b9273875…`、A=`5ed597d8163a…`、結果`c95736fd2990…`／[報告][G9]・[契約][P5] | Closed P5の少数群化、Ordinary／Partial+tail／P3+tail／Full／P5／matched CTSの同target・matched-native会計。旧v1の停止を置換しない |
| G10 v3 | S=`b9ed01455351…`、結果`fcd3ea6217bc…`／[報告][G10] | m3/5/7の次数内比較。Fullの追加価値が強い低次数対照に対して不足した固定例。次数間の同exponential精度ランキングではない |
| F1–F3／W2 | [F3][F3]、[W2b][W2B]、[W2c][W2C] | supplied-precision preflight／角度enclosure、十分Nの比較条件、setup＋全trial online＋量子費用。Full認証をcanonicalへ転用しない |
| C1-3-R2 | S=`2be82f7bff78…`、A=`324634a37538…`、結果`068f0ae8771f…`／[correctness][C1CORR] | R1 exact入力とcontrolled adapter、固定6 event・13 basisのbyte復元検査。全support/native認証ではない |
| C1-5A | S=`900a5278c604…`、A=`079ba4282574…`、結果`b022f2ea4216…`／[凍結要約][C1] | Ordinary／Full各1024 path完走。旧P5のworker突然終了により全jobはtechnical STOP |
| C1-5A-P5R | S=`2fbd83a44a0d…`、A=`bf32b3b3455e…`、結果`bb9059bacb69…`／[費用・比較][P5R] | 別SOURCE／authorization／markerのP5-only完走。累積整数selector、one producer→fresh exec provider、原法則を保持。旧runのretry／原因解明ではない |
| R0/R0.5/R1/R1.5 | R1 S=`d43d64a821a0…`、結果`24bfeb84a4ce…`／[保存帰属分析][R15] | 2-qubit有限P3、Euler再配分・共通角とnative価格。B²低下でもbiasによりNが増える保存例あり |
| Parent allocation S4 | S=`4ccc1832f63c…`、A=`71cf1b76492e…`、結果`fe0f3f077342…`／[70-law packet][S4] | parent identity mass配分＋angle変更＋canonical／IS。訂正20＋既存control50を保持。旧raw-C proxy20は正式集合外 |
| D1–D3／SP | [D3][D3]、[SP1][SP1] | D系列は有限P5・r1/2、Taylor保証とnative価格、literal境界取消の保存例。SPは離散primitiveとweightの交換。短wrapper利益／長wrapper反転・cap等を保持 |

今回C1の3固定SOURCE manifestをGit objectから照合し、それぞれ540／549／560 fileでhash不一致0、各Aの直接親=Sを確認した。重複依存を含む件数であり、科学的独立再現数ではない。既存testコード・logは資料として読み、実行しなかった。

## 3. 手法の数理・sampling・normalization・認証範囲

共通表記：`H_R^0=λ_R R`、`R=Σ p_i Q_i`、`p_i>0, Σp_i=1, Q_i²=I`、`x=λ_R |h|/r`、符号σを別保持。G6/P5の保証は明示正有理p、正有理`0<x≤1`、奇数m、固定`P_m(-iσxR)`。Identity・κ・scalarはtail massと分離する。分子geometry／basis／DF rank／L_Dを使わないsynthetic検証は、その旨を明示して読む。

| 手法 | proposal／weight／normalization | 平均・bit・価格の条件 |
|---|---|---|
| Ordinary even/odd Euler | 偶数l、`b_l=√[(x^l/l!)²+(x^(l+1)/(l+1)!)²]`、次数`b_l/B_o`、labels IID p、`B_o=Σb_l`。理想補正outcomeは`B_o Y` | `E[V]=P_m/B_o`。回転角`atan(x/(l+1))`、parent位相・順序を保持。実lawの丸め・native errorは別 |
| Partial-return／Closed P3 | 低次数returnをroot・parent係数／angleへ吸収。G9の「+tail」は未処理次数をOrdinaryで残す | G9のP3+tailは**同じP5 target**。R1の2-qubit P3実験とは異なる。exact group質量・実proposal・補正weightは各固定SOURCEに従う |
| G6 General Full | 同じ形式reduced wordへの有限Taylor係数を集約。偶数parent uの係数a、全child和s、`d_u=√(a²+s²)`、`tanφ=s/a`。Ordinary envelopeでraw語を提案し、非reducedはzero、有効語を`d_u/(b_l p(u))`で受理、childを`a_iu/s`から選ぶ | 全trial分母、受理時`B_oY`／拒否時0。`B_RA=Σd_u, Z=B_RA/B_o, M2=B_o B_RA, range=B_o`。global B_RA未取得で生成できるが未知Zをshot予算へ無料挿入しない。一般odd語をinvolution扱いせず、回転generatorは単一Q_i |
| Closed P5 | root、長さ2 ordered pair、長さ4 first-label群の高々L²+1群を構築。canonical群lawからparent／childを生成し、理想weight=B5、acceptance=1 | m5の同じp/x/σ・係数・pairingではFullの**理想受理条件付きevent law**と一致。digital丸めは別。O(L²)は群構築の算術数で、bit／selector／native取得費用を含む無料化ではない |
| CTS／Pauli collection | finite targetをPauli情報から再集約し、even correctionとodd Eulerをsampling | 非列挙layer samplingも既知。Gaussian回転Q_iを単一Pauli wordと扱えない。展開・取得・normalizer・controlled phaseの費用条件が必要 |
| Parent z／cost-aware IS | `Σz_i=a`、`c_i=√(z_i²+b_i²)`、`φ_i=atan2(b_i,z_i)`。辞書／係数固定ならproposal q、outcome weight=`c_e/q_e` | `M2=Σc_e²/q_e, range=max|c_e/q_e|`。正価格固定のproxyでは`q∝|c|/√C`。range・ceil・bias・setup込みの総最適性ではない。zero価格は別扱い。非common zはangleと価格を変え、B低下を保証しない |
| 相関／whole-path IS | joint proposalへ元ordered-path確率/qを掛ける、またはjoint operator平均の恒等式を別途満たす | 各microstepの周辺law一致だけでは積平均を保存しない。積weight、support、条件付き確率、shot間IIDを別に確認する |

G6 finite-bitでは実dyadic proposal `π=h_l∏h_j h_A h_(i|u)`と`W̃=α̃/π`を保持し、`πW̃=α̃`はexact rational。再現するのは近似係数のmeanであり、元P_mへの無条件exact一致ではない。既存上界は`bias_coeff≤ρ B_o`、`|W̃|≤(1+ρ)B_o/(1−η)^(m+2)`。角度enclosure／provider／native合成・相対位相誤差は別計上。[finite-bit監査][BITS]、[F3][F3]

理想Fullの既存Bernstein十分条件は、残余幅s>0、`ℓ=log(2/α)`で `N≥ceil{ℓ[2 B_o² Z_+/s²+4 B_o/(3s)]}`。未取得Zには1、または取得済みUから作る**認証済みdigital上界**を使う。有限bit後は実moment/range/biasへ戻し、ideal式を無条件使用しない。Nは十分trial数で、必要最小shot数ではない。量子呼出しの期待値・高確率cap・hard attemptsも別量。[W2b][W2B]

**normalizationだけでは総資源を判定できない。** v1.2の限定解釈では、同じ理想accepted-event平均量子価格cに対し、Fullの`M2=B_oB`・量子価格`(B/B_o)c`とcanonicalの`M2=B²`・価格cは、`M2×価格=B²c`で相殺する。これは実分散・finite-bit・range項・ceil・保守的予算・古典費用を含む総cost定理ではない。[v1.2 §7.3][V12]

## 4. normalization→shot→総資源の判断に使える保存データ

**SyntheticのG9/G10／R1.5／S4には、係数・実proposal・weight・moment/range/bias・十分N・native単価・総費用を結ぶ保存表がある。H4 C1にはその完全な鎖がない。**

- G9 v2：`artifacts/track_b_g9_p5_native_result/2026-10-10/v2/`のraw result、11-row CSV、manifest、consumed marker、STOP、affine比較。G10 v3は17-row／34-axis policy。保存probability／moment／strict sequence／価格／Nを追えるが、実量子shotsは0。
- R1.5：`artifacts/track_b_r1p5_saved_attribution/2026-10-06/`のfactorization、angle／precision、Pareto資料。2-qubit `p=(3/4,1/4)`、P3、x=1/8・1/4、σ±、distinct basis、native ε=10^-3/10^-4/10^-6。例えばx=1/8・ε=10^-4ではB²が低くてもbiasが増えNが増えたという保存診断がある。[R1.5][R15]
- S4：`artifacts/track_b_parent_allocation_review_packet/2026-10-15/s4/law_comparison_index.csv`、v3/v4のresource map・chunk・inherited rows。係数mass／acceptance／M2／range／bias／残余幅／N／T・CX・1Q／prep slopeまで対応する。同fixtureのD＋E終了判断は維持する。
- B3 correctness512／pilot16896：event／bit／zero・reject・accept／angle・codec／setup・online CPU等の技術データ。固定3-label `p=(1/5,3/10,1/2), x=5/7, m=5`、H160/root256、η=ρ=10^-12。分子／DF／geometry／basis／L_Dは非該当。n256等のprofile長は必要shot数ではない。Pilot compactのoriginal digestから省略全文を復元できるとはしない。[pilot][PILOT]

G9の固定native例は3-system-qubit synthetic、`p=(1/5,3/10,1/2), x=5/7,m=5,σ+`、`Q0=Z0, V1=RXX01(π/4), V2=RXX12(π/4)RZZ01(π/4), Qi=Vi†ZiVi`、state000、Rz ε=10^-6、H160/root256、η=ρ=10^-12、provider exact model。geometry／basis／DF rank／L_D／PF δ窓は非該当。G9の22-axis失敗配分とS4の単complex task、G10の34-axisを横断して新順位を作らない。保存G9表ではClosed P5が登録direct方式のT切片と準備／読出し呼出し係数で最小だった一方、CTSはCXで有利であり、全資源支配ではない。

**C1の数値引用条件**：直鎖H4・1.00Å・STO-3G・DF12・R1 generation-prefix L_D=3・13 basis/324 labels/8 system qubits。T=4/5、q=1、r=4、m=5、microstep1/5のみで時間窓掃引なし。H_prob2048/root4096、η=ρ=angle τ=10^-12、angle root256/64項。TAIL_ONLY_PLUS_SCALAR、各方式1024 path。[凍結報告][C1]

| C1保存指標（原summaryの引用） | Ordinary | Full | P5R |
|---|---:|---:|---:|
| empirical weight二乗平均 | 1.1141049662602474 | 1.1141049662602474 | 1.1105279232736538 |
| valid zero path | 0 | 0 | 0 |
| logical instructions/path | 285.125 | 284.6494140625 | 284.7666015625 |

これは母分散・一様range・必要N・native価格ではない。degree0はempty parentであり各pathに4 child rotationsが残る。異なるmethod乱数keyなので差をpaired効果としない。Fullのsampling/checks/Evidence CPU、P5の群setup／selector／cache／worker構成は保存されているが、計測境界が異なり公平な最速実装順位には使えない。

今回、既存C1 archiveを**メモリ内でのみ**開き、part／連結archive／全memberのbytesとSHA256をindexに照合した。元ディレクトリへの展開・上書き・trajectory再集計は0。

| 公開archive | member数 | 利用できる情報 |
|---|---:|---|
| Ordinary | 229 | exact path proposal/weight、event／angle Evidence、logical費用、batch、monitor／log |
| Full | 246 | 同上。zero分類・raw/reduced length・局所query保存 |
| 旧P5部分資料 | 38 | setup／途中資料・障害記録。完走結果や0費用へ補完不可 |
| 別P5R | 406 | 完走batch、sampling/event/angle、exit／CPU／RSS/PSS、journal／log |

全919 memberのhash不一致0。JSONにはarchive SHA256と全partの公開locator、先頭batchの**field名だけ**を記録した。これらには測定実行flag／coherent estimator未完成／deterministic halves・native未取得のscopeも残る。H4の共通認証N・十分shots・全law strict価格は保存資料で未確立。後からweight標本二乗平均や観測maxだけで認証を埋めることはできない。

## 5. STOP・負結果を保持する

| 過去の状態 | 後続と混同しない点 |
|---|---|
| RA-D0旧run | `UNCERTIFIED_NUMERICAL_POINT`でbudget生成停止、比較0。性能negativeではない |
| G9 v1 | Fraction→mpfのAPI TypeError、native比較0 row。v2は別SOURCE／markerの完了 |
| G10旧run／訂正 | 旧停止／保存後会計／v3成功を別証拠として保持。v3を独立再現回数へ水増ししない |
| C1-3 | state hash定義差で停止。R1は別のreference bytes不一致で6 event比較未到達。R2のsigned-zero復元修正と限定成功は新しい来歴 |
| C1-5A | P5 BrokenProcessPool、直接exit/signal・根本原因UNRESOLVED。P5R完走でも旧jobの失敗を解除しない |
| P_U | L2、p=(1/2,1/2)、x1、m7の既存反例は**Full上界Uの普遍比較命題**への反例。真のRA normalizerやRA一般の不可能性ではない |
| G5／BF/BM／DF correction | 固定辞書の強対照、既知BCH対応、限定generic加法planの負結果。全生成器／whole compilerのno-goへ一般化しない |
| S4／研究C law調査 | D＋Eの探索終了、`NO_DISTINCT_FREEDOM`は同種の追加機構が特定できなかった判断。一般の改善不能定理でも、今回の研究再開許可でもない |

## 6. Track C K2/P3への接続条件と確認依頼

`src/trotterlib/rte.py`のKは最大even orderで、**K2=P3、K4=P5、K0=P1**。mainの既存研究C law調査にはK0/r128の文脈もあるため、今回依頼のK2と取り違えない。[研究CのSOURCE整理][C-LAW]

1. **同じ有限P3を保持する候補**はG6 m3またはClosed P3。既存式は`χ=Σp_i²`、`a_empty=1−χx²/2`、`a_i=p_i[x−x³(2χ−p_i²)/6]`、相異なるj,kのparentに`a_jk=(x²/2)p_jp_k`、`s_jk=(x³/6)(1−p_j)p_jp_k`、`tanφ_jk=x(1−p_j)/3`。これは[G6 §9][G6]の転記。Track C候補固有のλ、p、x、B、χは今回評価していない。
2. **P5は別target**。K2へClosed P5や「P3+tailのP5比較表」をそのまま移せない。K4採用ならTaylor bias／r／round schedule／attenuation／価格を新たに整合させる必要がある。K0の積`P1^r`を単一P3/P5と扱うこともできない。
3. **入力binding**：改善後J/U・DF rank/order・L_D・head/tail・identity抽出・κ/scalar・符号・basis gaugeを固定し、正有理pと`x≤1`を結ぶ。C1のDF12/LD3モデルと研究Cの保存CDF8/元順LD2は異なり、C1 R1は旧PR-2とも135係数hexが異なる。係数をexact dyadicと呼ぶことは分子operatorへのrigorous bindingではない。
4. **controlled実装**：ordered parentの回路時間順、単一child回転、`(-iσ)^|u|`、負回転のactual adjoint、inactive/active blockとscalar相対位相を保持する。既存C1 adapterは参考にできるが、G9のsynthetic native価格やH4選択6 eventのPASSを新J/U全supportへ移さない。
5. **統計API**：現`RTEEvent`とsample mean／accuracyはcanonical unweighted ±1を前提にする。q変更だけでは元係数を保存しない。Full／ISを接続するなら元target係数・実proposal・weightを分離し、全trial分母・zero／technical failure・path weight積・M2/range/biasを実装する。従来Hoeffdingをweighted outcomeへそのまま転用しない。
6. **multi-step／RPE**：fresh独立occurrenceでordered meanを保存し、有限非unitary meanの積誤差を戻す。canonicalなら変更後Bの全microstep attenuation、Fullならweighted全pathのmean／confidenceを対応させる。PF係数C、DF誤差、RTE／finite-bit／合成／law bias、統計誤差を別ledgerに置く。signal半径・alias／分枝復元・各round failure配分も必要。
7. **価格と総費用**：研究Cの同じ合法basis共有／boundary規則によるwhole controlled kernel価格が必要。少数event加法価格・logical差・小angleでは代用しない。古典費用は`setup+N E[online/trial]`、量子費用は`N E[1_accept C_complete]`を基本に、状態準備／readoutのscope、期待値・cap・hard上限を揃える。CPU秒とT/CX/RZを無換算で加算しない。

**SERVER 2に確認してほしい事項（問い合わせ・転送は未実施）**：

- 最新のTrack C K2/P3を使う固定SOURCE、J/U／DF入力hash、rank policy、L_D、h/r、各round、λ/p／scalar規約、既存law/synthesis認証laneの所在。K0/r128の公開記録から推定しない。
- 旧C1／P5R／B3 original runtime、raw_preserved、元stdout/stderr、入力NPZ／basis／SOURCE archiveの現存とhash。公開archiveと独立な原本保存かを区別する。原本・private入力を無断pushしない。
- 2026-11-02の持込inventoryにある`track-b-…-local-evidence`／`-local-content-record`等の現HEAD・未公開差分。例：correctness512 `814bae76f4d2385a7fa530d577cd883138422d5e`、partial-sum audit `48f004727f494098138f16f6bf5f687972a8188e`。本調査JSONの19候補は**過去inventory**で、現在の存在／公開状態の認定ではない。
- G9 v2 raw、C1全partの公開URLと元archiveの対応、未提供selfcheck SOURCE、canonicalの一様finite-bit certificate、候補固有のstrict native／共通N結果が別保存されていないか。

**別途科学検証を検討する場合の具体対象（今回すべて未実施）**：対象K2の同一P3についてOrdinary／Closed P3／General Fullのoperator meanとdigital bias、全support proposal・weight・range／momentの認証、改善後J/Uのcontrolled provider、同精度・同failure配分の十分N、whole-kernel native費用、setup償却／online費用、全round信号／分枝復元を順に結ぶ。P5を含める場合は別の同exponential精度比較として設計する。まずSERVER 2の既存証拠の有無を照合し、未取得部分だけを対象にする。これは実行契約・再開許可ではない。

調査成果のcommit/push・remote照合後、**mandatory STOP**。既存の研究凍結・採否・科学的資格と検証manifestのstatusは変更しない。

[HIST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/5f750a23f023d75c8c4fbcabc6937d6953ce61f2
[G6]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/28cfabb1d47e0e1824bce1e154a9c289738fa2b9/docs/tracks/algorithm_codesign/g6_independent_mathematical_audit_20261010.md
[BITS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/28cfabb1d47e0e1824bce1e154a9c289738fa2b9/docs/tracks/algorithm_codesign/g6_finite_bit_and_access_audit_20261010.md
[G7]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a689694080f4b7600fe67cf77d841d1cbbf04503/docs/tracks/algorithm_codesign/g7_results_and_gpt_handoff_20261010.md
[G9]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c95736fd2990f5ef6dd1cb5866421fd78bb28687/docs/tracks/algorithm_codesign/g9_v2_results_and_gpt_handoff_20261010.md
[P5]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/73a7abd6547830d8c09c95637c978855af83a01e/docs/tracks/algorithm_codesign/g9_p5_matched_native_contract_20261010.md
[G10]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fcd3ea6217bc00b667180cec149a70102d75f07e/docs/tracks/algorithm_codesign/g10_v3_results_and_gpt_handoff_20261010.md
[F3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1aea9b8c228c5b940f2fcba0f94493b3ffdec46e/docs/research/track_b_g6_f3_preflight_angle_20261018/handoff.md
[W2B]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ed26d1c85cf7ad80625f7abbc662b70abd690a97/docs/research/track_b_g6_w2b_certified_trial_budget_20261018/handoff.md
[W2C]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/864097c1285f1f06399040cf0f7c0f42334d9252/docs/research/track_b_g6_w2c_total_cost_conditions_20261018/handoff.md
[C1CORR]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/068f0ae8771fccf28c529bda12a84abe8b7e7648/docs/research/track_b_c1_3_r2_reference_recovery_c1_4_20261021/CORRECTNESS_REPORT.md
[C1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/185ec4e9d6591f4ee8d7b047c2697bed7be7bc2e/docs/research/track_b_c1_freeze_synthesis_20261021/C1_RESOURCE_EVIDENCE_AND_LIMITATIONS.md
[P5R]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bb9059bacb697e1105e63c1b3826b0c34886ad15/docs/research/track_b_c1_5a_p5_worker_recovery_20261021/comparison.md
[R15]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/af3d014d0a0cfcbbd25bb544f6544652fec92942/docs/tracks/algorithm_codesign/r1p5_saved_value_attribution_v1.md
[S4]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/53c48317d16851333c0e0354b0dfd8ef91e1d524/docs/tracks/algorithm_codesign/parent_allocation_s4_final_comparison_evidence_packet_20261015.md
[D3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f0df9a3b36bc3540f0c83941871fa0d3110ddbf7/docs/tracks/algorithm_codesign/d3_v2_results_and_gpt_handoff_20261014.md
[SP1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9d2bb1fa439748b02084bd9fbc9b10a705328f8a/docs/tracks/algorithm_codesign/sp1_one_shot_result_validation_20261006.md
[V12]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f750a23f023d75c8c4fbcabc6937d6953ce61f2/docs/manuscripts/track_b_return_aggregation_native_resource_note_v1_2_draft.md
[PILOT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f27e63b78518c9e807b91f0028fd29078439ec4f/docs/research/track_b_g6_w2c_b3_limited_pilot_16896_authorized_20261020/handoff.md
[C-LAW]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/546bca7ed2cece59b714b986a264d7b0b0871a01/docs/research/research_c_sampling_law_freedom_survey_20261030.md
