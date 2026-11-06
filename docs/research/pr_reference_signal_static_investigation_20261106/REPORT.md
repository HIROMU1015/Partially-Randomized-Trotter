# 新研究候補A/B：限定技術調査・GPT科学レビュー準備

静的調査の結論は、**Aはtrajectoryごとのreference作用と単一shot信号の対応、Bは固定入力の保証と既知importance samplingとの差分を先に判断する必要がある**、である。実装採択、性能改善、一般的新規性の認定は行っていない。比較表は[CANDIDATE_COMPARISON.md](CANDIDATE_COMPARISON.md)。

2026-11-06 JST、SERVER 2（GPUA100-gleap02）。専用branch：`codex/pr-reference-signal-static-investigation-20261106`。基準treeは`602ca4b67b1037e221e910a2733444630c5add62`。

## 1. 範囲・確認深度・保存境界

今回実施したのは既存資料・SOURCEの静的読取、一次文献の限定照合、依頼された2文書の作成・公開である。科学SOURCEのimport、runner/test実行、新しい数値解析・sampling・simulation・回路build/compile・価格取得・合成は行っていない。新しいsampling式、optimizer、CSWAP回路も実装していない。

適用AGENTS、PROJECT_MAP、`研究概要・現状.md`全文、VALIDATION_STATUSとmanifestの関連箇所、worktrees親の正式共通指示`research_project_instructions_unified_2026-10-25.md`を確認した。主agentだけで調査し、共有設定・環境・他ユーザーのファイルやプロセスを変更していない。

| 固定資料 | 利用した範囲 |
|---|---|
| [3環境統合 / 3571477][integration] | REPORTと関連reuse記録。全asset・全rawの再監査ではない |
| [文献調査 / e8b19ff][survey] | RESEARCH_OPPORTUNITIESのO5・O2、O6の適用境界、SOURCESの関連locator。全文献の再読ではない |
| [Track Cプレフライト / 4b625e5][preflight] | REUSE_AND_BASELINESのPF/RTE・phase/control・Parent・RPE会計・test入口 |
| [C2限定調査 / 6f6c084][c2] | 元H・状態・情報・総費用の規約、旧負結果の適用限界 |
| [C2-B強baseline / 602ca4b][c2b] | REPORTの固定条件・実行可能解と下界の区別、最適性未確定、STOP |
| [過去のPRHS/qSHIFT/IS対応監査][audit] | signed coherent mean、joint law、binary測定分散、既知ISへの還元。原論文の再監査へ格上げしない |

C2-A（`ce84fa8e2d2f1d487df3f05e63d519ebb03f83ad`）、C2-B pilot（`78004943ed5b718c12841e80f55fddf1d1644364`）と強baseline、未着手C2-Cは一時保留のままである。Aの長round・全18round優越は未資格、Bの強baseline優越は`OPTIMALITY_UNDETERMINED`。本調査の新A/Bとは区別し、C2、C-N/C-S/C-P、P-D、BF/BM、PR-3等のSTOPを維持する。過去のSOURCE・raw・result・grant・ledger・markerは変更していない。

## 2. 共通の目的と比較規約

元の物理Hamiltonian Hの厳密基底状態を供給し、状態準備費を除くPR＋位相推定の総回路資源を、同じenergy精度・成功確率で比較する。今回は物理系、数値条件、round数、科学的実行契約を新たに固定しない。過去のH4・18round条件は再利用候補であり、自動的な採択ではない。

厳密状態の供給は、真のE₀、gap、古典的ground vector、controlled状態準備oracle、全pathの価格表を無料供給する仮定ではない。追加referenceの準備・reset・routing・gate、情報取得と古典samplingの費用も明示する。回路資源と古典費用は別に記録する。

将来の会計の入口は、各資源kについて

\[
G_k=\sum_m\sum_{b\in\{X,Y\}}N_{m,b}\,
 \mathbb E[C^{\rm no\text{-}prep}_{m,b,k}],
\quad k\in\{\mathrm{RZ,CX,T,depth}\}.
\]

各指標を分け、追加qubit数も別記する。shot加重depthは壁時計時間ではない。candidateだけに精度・alpha・round構成の最適化機会を与えない。短時間価格やprimitive数を全費用の改善証拠にしない。

信号への接続には、参照信号の半径下界ρに対し、同じ正規化で表した全deterministic disk η<ρを確認する。円板幾何では半径下界ρ−η、位相ずれ上界arcsin(η/ρ)となる。残った位相予算をX/Yの統計精度へ渡し、実際の測定rangeで整数shotを評価し、全round・両軸の失敗確率を総予算内に配る。値は今回計算していない。特にweighted測定のrangeは元の±1とは限らない。

## 3. 新A：PR × controlled evolution削減

### 3.1 一次文献で確認した範囲

[SE-QPE v2][seqpe]の§II、Eqs. (1)–(9)、compute–uncompute節、§III Eqs. (14)–(19)を限定確認した。

| 文献上の性質 | 適用境界 |
|---|---|
| CSWAP gadgetの2分枝は一般のcontrolled-Uとは異なる | 共通固有基底を持つU_A、U_Bなら連続gadgetの固有成分確率を保ち、既知位相のreferenceで位相差を補正できる |
| targetは固有状態の重ね合わせでも扱う | compute–uncompute置換のcanonical QPE同値性は別条件。controlled状態準備も必要 |
| 粒子数保存系ではvacuumをreferenceにできる | 定数phaseを補正する。追加register・CSWAP層・任意のfan-out/resetを課金する |
| uncontrolledな2 laneへevolutionを分ける | control費とdepthの利益は分解・実装依存。RPEのshot加重費用は同文献のcanonical QPE式だけでは決まらない |

PRへの定理や本SOURCEの利益は同文献から認定しない。以下は**Codexの静的な代数的対応・検証仮説**であり、GPTによる正式レビューと実回路検証が未了である。

### 3.2 canonical QPE条件とfresh-shot RPE条件を分ける

文献Eq. (2)の分枝をそのまま仮定する。U=U_AU_B、referenceをr、入力をψとし、通常のcontrolled-UのX+iYが⟨ψ|U|ψ⟩となる測定規約を選ぶと、分枝の内積は

\[
z_{\rm gadget}
=e^{i\vartheta}
 \langle\psi|U_AU_B|\psi\rangle
 \langle r|U_B^\dagger U_A^\dagger|r\rangle
=e^{i\vartheta}z_\psi(U)\overline{\langle r|U|r\rangle}.
\]

これはEq. (2)を展開した単一干渉blockの式であり、非可換なrandom halvesにcanonical QPEの連続gadget保証を与えるものではない。実装のlane順序、出力swap、Yの符号との対応は将来の確認対象である。

各shotにψとrを供給するRPEで、各trajectory U_ωについて
`U_ω|r⟩=exp(iφ_ω)|r⟩`かつφ_ωが既知なら、`ϑ=φ_ω`によるpath単位の信号復元を検討できる。この単一blockの内積計算にはU_AとU_Bの可換性は使っていない。一方、状態を保持したまま異なるrandom blockを連続させるcanonical/iterative方式への同値性は別途必要である。

したがって「random halvesが共通固有基底を持たない」だけでfresh-shot RPEを一括否定せず、「H全体が粒子数保存」だけでpath単位の成功も認定しない。元H groundは近似PFや各random trajectoryの固有状態とは限らない。

### 3.3 SOURCEとの静的対応

以下のコードは[基準tree][base]で静的に確認した。共通APIが任意の登録basisを受け付けることと、特定のphysical SOURCEの対称性を区別する。

| SOURCE / 処理 | 確認した内容 | 新Aへの推論・未確認 |
|---|---|---|
| [track_c_source_guarantee.py][physical] `basis` | physical basisの許容familyはRZ、XXPlusYY。未知familyは拒否。native gateはsector投影しない | この理想familyは粒子数保存。登録済みphysical basisの範囲ではN=0 sectorが1次元であることを利用できる。generic DF registry全般への保証ではない |
| [df_rte_tail.py][tail] と [df_rte_qiskit.py][builder] | signed diagonal Z/ZZをbasisで共役。product、最後のrotation、application順を保持 | このphysical basis＋Z/ZZの完全なblockは粒子数保存。完全trajectoryにもvacuum固有性が成立する見込み。actual phaseの復元は未実装・未検証 |
| [df_partial_s2.py][partial] | deterministic blockのbasis逆列→diagonal→basis、constant/extracted identity phaseを区別 | 負時間も対称性自体は変えない。PF次数・新しい登録basisへ既存証明を無条件に移さない |
| [rte.py][rte] `_make_event / event_unitary` | paired Taylorのorder符号、component符号、product順、rotationを保持。有限Kはdegree K+1。normalizationとattenuated meanを区別 | vacuum phaseは物理Hのvacuum energyだけでは足りず、各pathのorder/product/rotation/scalar全体から追跡する必要がある |
| [Pauli中央制御][pauli] `central_control_body` | X/Y/Z productとPauli rotation、集約phaseを一度だけancillaへ移す | 一般のX/Yを含む個別因子はNを保存しない。total Hの対称性からPauli trajectoryのvacuum固有性は導けない。特定pathが偶然戻る場合と全supportを分ける |
| [phase保持wrapper/compiler接口][gates] | controlled blockをphase fittingなしで比較。X/Y、full IR、register順とglobal phaseを扱う | 検査接口は再利用可能。新2-register gadgetのcorrectnessやT価格は未資格 |

`_append_basis`は実際のgate列をreverseし、各gateの`inverse()`を使う。新Aで必要になるU_A†も、同じsampled trajectoryの**actual adjoint**を使う必要がある。inverse側を独立に再samplingした平均や、逆向きPFを別生成したものに置き換えない。

既存builderはcontrolled時のevent/identity phaseをancilla-relativeに、uncontrolled時はglobal phaseに保存する。reference干渉ではU→e^{iχ_ω}Uのscalarはzとreference overlapの積から相殺される。従来の信号に戻すにはχ_ωも含む補正を一度だけ入れる必要がある。既存phaseを捨てたり、event符号を二重追加したりしない。

### 3.4 適用不能となる条件・平均での代替条件

次は**単純なpathごとの位相補正方式**の反証条件である。

- 非零の対象信号を持つpathで`|⟨r|U_ω|r⟩|<1`なら、phase-only補正ではそのpathの半径を復元できない。未知のpath依存reference phaseも未補正biasになる。
- 必要なinverseがactual adjointでない、scalar/符号/順序が不明、または共通referenceを準備できない場合、上の対応を主張できない。
- canonicalの連続gadget条件が崩れている場合、その既知保証は使えない。単一shotの式をそのまま連続使用の保証にしない。
- native近似で粒子数sectorから漏れる場合、理想SOURCEのvacuum固有性だけでは不十分。両lane・補正・CSWAPのfull-space誤差を含む新しい保証が必要となる。

pathごとに成立しなくても、平均信号について
`Σ p_ω W_ω e^{iϑ_ω} z_ω conjugate(r_ω) = Σ p_ω W_ω z_ω`
が認証できれば別の接続余地はある。しかしSOURCEはこの相関を認証していない。`E[UρU†]`というchannelの一致や`E[U]`だけから、reference overlapとの積の平均を決められない。逆reference overlapによる重み補正はrange/variance増加・零点・生成費を伴う別方式であり、今回設計・開始していない。

### 3.5 強baseline・費用への経路・最小検証案

強baselineはgenericな全gate controlではなく、既存の**basis非制御・diagonal/中央rotationだけ制御・同basis共有・phase保持**を使う。Pauli中央制御も元の資格範囲を明記して対照に含める。Full/tail-free方式も同条件で比較し、candidateだけに有利なPFや合成精度を与えない。

費用利益の経路は「残っている制御gateの差 − reference/CSWAP/補正等の追加gate」が各shotで有利になり、かつ信号半径・bias・必要shot数が悪化しないことである。CX、RZ、T、depth、追加qubitは別々に比較する。basisの制御は既に除去されているので、その全量を新しい節約に数えない。CSWAPの標準分解や古い合成wordだけから新回路の実Tを断定しない。fan-outは任意の追加案であり今回採択しない。

**将来、別承認を得た場合の最小検証案**：

1. 選んだSOURCEの全factor familyについてreference作用・scalarとactual inverseの記号的対応を確定する。Pauli反例を含め、対象supportを明示する。
2. 小さい人工例の非可換path・非固有target・符号/identity/負時間を用い、既存controlled wrapperと単一blockのX/Yを照合する。これは実装を伴うため今回は実行していない。
3. 同じ元H groundへのphase bias、半径、compiler/合成disk、整数shotを接続する。正半径・統計予算・alias分枝を満たせなければSTOP。
4. 以上が成立して初めて、強baselineと同じ価格取得機会・精度でfull回路費用を調べる。これは追加承認事項であり、今回価格は取得しない。

## 4. 新B：PR × signal-aware randomization

### 4.1 一次文献で確認した性質と対象の違い

[平均誤差最適化 v1][average]のEqs. (1)–(3)、(16)–(32)、Appendix A (35)–(48)とdiscussionを確認した。独立qDRIFT stepで`exp(-iδH_j/p_j)`を使い、generatorの平均をHに保つ方式である。

文献の保証は、`D_p=ΣH_j²/p_j−H²`を用いるHaar平均infidelityと**coherent operator meanからの複素信号bias**の非漸近上界である。後者は`t² Tr(D_p)/(2Nd)`以下で、正規化HS normに比例するlawがその上界を最小化する。個別Pauli項では絶対係数lawと一致する。groupingによるRZ層の比較はfull Clifford+T費用の比較ではない。

Appendixのtrace norm上界は入力平均を取る前にもある。この点を無視して「channelの結果しかない」とは扱わない。ただしHaar平均の`1/d`を固定groundへ無料で移せない。有限shotの統計誤差も別である。

**Codexの対応推論**：文献Eq. (35)に`|⟨ψ|A|ψ⟩|≤||A||op≤||A||1`を適用すれば、固定入力にも`min(2, t² Tr(D_p)/(2N))`という保守的なbias上界を読める。Haar式の因子dを失い、非自明性・PRでの有用性は未確認である。Nを保存する部分空間への制限も、各factorの不変性と、その空間の平均か単一groundかを区別する必要がある。これは新しいstate-specific改善保証ではない。

またgeneratorの不偏性は、任意の有限stepで`E[V]=exp(-itH)`という等式でも、既存finite-PR polynomialの維持でもない。qDRIFTの`H_j/p_j`への変更を、そのまま既存PRのlaw変更として扱わない。

### 4.2 PRのcoherent meanを維持する条件

[PR原著 v2][pr] Appendix A、Eqs. (A18)–(A29)の有限化前の分解とnormalizationを限定確認した。実装の有限化は[rte.py][rte]のpaired Taylor、exact finite distribution、corrected/attenuated operatorを使う。原著の無限和等式を有限Kで厳密なexpと呼ばない。

既存pathの係数をc_ω、unitaryをU_ωとし、固定有限targetを
`Q=Σ c_ω U_ω`と書く。phaseをU側に含める現行表現でproposal qを変えるなら、既知change-of-measureとして`W_ω=c_ω/q_ω`、`E_q[W_ωU_ω]=Q`を保つ必要がある。元の正規化meanをtargetとする場合はcをそのnormalizationで割り、単位を揃える。これは新しいsampling lawの提案ではない。

| 保持・確認するもの | 変更時に失ってはいけないもの |
|---|---|
| 全supportと順序付きjoint probability | marginalが同じでもcorrelation変更で非可換wordの平均は変わり得る。履歴依存ならjoint weightが必要 |
| order・component・scalar/sign、finite K/r、実際の角度 | 正負/複素係数を確率として直接使わない。phaseをgate側とweight側に二重配置しない |
| 正規化と`M₂=Σ|c_ω|²/q_ω`、`R=max|c_ω/q_ω|` | range、有限bit近似、support不足によるbias、生成/normalizer取得費を無視しない |
| X/Y測定と全試行の分母 | 複素weightはX/Yを混合するので、元の±1測定のconfidence式を無条件に流用しない。discard/postselectionも受理率とbiasを課金する |

固定groundのpath期待値やcommutator期待値を利用するなら、その情報をどう取得し、誤差・shot・古典計算費をどう課金し、baselineにも同じ情報を与えるかが未決である。厳密state供給だけでは古典記述、E₀、gap、pathごとのexact signalは得られない。

### 4.3 既知法への還元と残る具体的仮説

[既知resource-optimal IS][is] §II Eqs. (7)–(13), Theorem 1はwhole-path価格Cに対する`E_q[C] E_q[W²]`型の最適化を既に扱う。非加法なCでも`q∝p/√C`という式だけに新規性はない。有限confidence、integer shot、効率的なlaw生成までの最適性はこの式だけでは決まらない。

| 既知研究・既存実装 | 新Bが実質的に同じとなる条件／残る境界 |
|---|---|
| 絶対係数sampling、平均誤差HS則 | 単一Pauli、または全空間で`P²=I`の同規格involutionを個別samplingするなら、HS normは絶対係数と一致する。登録DFのsigned共役Z/ZZもこの場合に含まれる |
| [Parent/S4・ALLOCATION][parent] | 同じ有限eventを係数/価格で再配分しweightで同じmeanへ戻すだけなら既存IS/配分問題。`digital_proposal / reweight_events / event_statistics / error_accounting`を再利用する。訂正済みS4・common-angleの既知境界を保持する |
| PRHS / qSHIFT | [既存一次資料対応監査][audit]を継承。channel・observableの高次一致をcontrolled coherent meanへ移さない。PRHSのsigned係数の確率表記、qSHIFTのhistory係数の未解決表記を推測で補正しない。今回原論文全文の再照合は未実施 |
| 非加法native path価格 | 既知whole-path ISの目的に含まれる。未検証なのは巨大supportでのnormalizer・生成・価格取得を含む実行可能性と有限RPE利益。加法proxyをexact native価格と呼ばない |
| signalを重視する配分 | 候補だけがexact ground signalを知る方式は不公平。同じbinary Hadamard法・同じmean・実weightなら`Y²=1`なので`E[(WY)²]=E[W²]`。条件付き期待値のばらつきだけを小さくしても測定の量子分散を除去できない |

最後の行は[過去の分散対応監査][audit]の再利用である。X/Yの各座標の平均を固定したまま同じ測定・weightだけを使う限り、「signal-aware」という名称で追加の分散自由度が生まれるわけではない。測定、control variate、grouping、target biasまで変える場合は別の仮説と保証が必要で、今回それらを設計していない。

具体的に残る問いは、**取得可能でbaselineにも共有できる情報**に基づく構造が、既存配分の変数を超えて、coherent bias/range/normalizationと生成費を含む総費用を改善し得るか、である。未検証というだけでは新規性を認めない。

### 4.4 強baseline・費用経路・最小検証・STOP

強baselineはcanonical finite PRに加え、適合する既知IS、訂正済みParent/S4/ALLOCATION、同一groupingに対する既知HS則/絶対係数則、whole-path価格を扱う既知ISとする。PRHS/qSHIFTは保証対象・oracleが対応した範囲だけ比較する。UWCは[固定文献調査][survey]のround別資源最適化として参照し、旧未検証rawを資格付きbaseline価格にしない。

利益経路は、law/構造の変更→認証されたmean bias・M₂/range・回路価格→正半径と統計予算→整数shot→全round資源、である。平均誤差、RZ層、価格期待値のいずれか単独の低下を総T改善と同一視しない。

**将来の最小確認案**は、まず式と情報アクセスをGPTが判断し、既知法への非自明な差分が具体化した場合だけ、別承認の下で最小人工supportのordered coherent mean・weight・実binary分散を確認すること。その後にのみ固定groundの保証と実行可能なlaw取得費を接続する。新optimizerや大規模gridを先に作らない。

次の場合はその仮説の範囲でSTOPする：HS則が現行lawと同じ／既知ISの変数だけ、必要なstate情報が取得不能またはbaselineに非共有、coherent meanが不明、range/normalizationが制御不能、biasで位相予算を使い切る、価格・normalizer・生成費が欠損。欠損を0で埋めず、既知への還元と未資格を区別する。

## 5. 再利用資産と資格

| 入口（基準tree、Parentは訂正固定版と同一blob） | 再利用する機能／限界 |
|---|---|
| `rte.py`：`finite_rte_distribution / finite_rte_operator_moments / compose_finite_rte_occurrences` | 有限law、corrected/attenuated meanとnormalization。新lawの等式を自動保証しない |
| `df_rte_qiskit.py`：`build_event / build_sequence`、`df_partial_s2.py` | signed順序、basis共有、relative phase、実inverse。新reference回路は未実装 |
| `pr_dictionary_validation/gates.py`：`controlled_gate / wrapper_signal / signal_gate / compiled_wrapper` | phaseをfitせず検査する接口。新構成のcorrectness認定やcompile許可ではない |
| `algorithm_codesign/parent_allocation.py`：上記配分関数、`native_plan / adjoint_plan` | 有理数proposal、M₂/range、誤差欠損拒否、phase/order/adjoint。provider callbackは実回路証明ではない。基準版と`4ccc1832…`のblob一致を確認 |
| `algorithm_codesign/rte_reallocation/{model,native,accounting}.py` | [既存reuse map][preflight]からnative非加法価格・confidence接口を再発見可能。今回providerや旧runnerは呼んでいない |
| `rpe_resource_accounting.py`：`evaluate_rpe_round_candidate / RPERoundCostProvider / build_rpe_resource_summary` | phase/radius/shot/providerの接口。unit-radius/S2・fresh iid等の元仮定を確認してから使う。weighted測定や非単位半径への無条件移植は禁止 |

関連test入口は`tests/test_df_rte_qiskit_builder.py`、`test_df_partial_s2_repeated.py`、`test_finite_rte_signal_validation.py`、`test_rpe_resource_accounting.py`、`test_rpe_resource_accounting_guards.py`、`test_rpe_allocation_sensitivity_validation.py`、`tests/tracks/algorithm_codesign/test_parent_allocation{,_native}.py`。固定treeで所在を確認し、検査対象は既存プレフライトの対応表を再利用した。今回testは起動していない。

旧compiler A・strict合成recordは、元SOURCE・角度・構成の資格を保持する。新CSWAP/register/controlや新lawの角度へ保証を移さない。誤差を再接続する際は、compiler diskと合成diskを**同じ信号単位**へ揃えて加算後にphase/radiusへ変換し、normalizationや同じ誤差を二重計上しない。新Bのweighted測定ではshot ceilingも適合性を見直す必要がある。

## 6. GPTへ戻す科学的論点と次の担当

| 担当 | 必要な判断・作業 |
|---|---|
| GPT | A：単一干渉blockの対応をfresh-shot RPEへ接続できるか。canonicalの共通基底条件との境界、vacuum phaseの全path追跡、native誤差をどう保証するか。既知reference干渉の応用を超える差分はあるか |
| GPT | B：固定groundに対して有用な保証・公平に取得可能な情報があるか。既知IS/Parent/HS則への還元を超える具体的機構があるか。なければ実装・数値実験を進める理由はあるか |
| ユーザー | GPT判断後、進める候補・科学条件・強baseline・主資源・最小検証範囲を選び、必要な実装/科学実行を別途明示承認する。採択せず保留する判断も可能 |
| Codex | 今回は2文書の公開照合で終了。追加指示・必要な実行承認がある場合だけ、選択された差分を既存資産へ接続する |

A/Bを統合した未検証の優越、改善率、正式な新規性GO/NO-GOは提示しない。O6の代替推定器を新Cとして開始しない。

## 7. 静的検査とmandatory STOP

文書の固定commit/path、相互リンク、追加対象が2文書だけであること、差分の空白を確認する。検査は文書整合性に限定し、科学的性能・回路correctness・CI成功の検証とは呼ばない。公開ref/tree/blobの照合結果は最終チャットで報告する。

**この限定調査の公開照合後mandatory STOP。次の担当はGPT／ユーザー。科学レビュー・新規実装・科学計算は自動開始しない。** 既存STOPとC2保留を維持する。

[integration]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3571477093441ac4852166fe829f8ccc5db5ec4f/docs/research/pr_algorithm_improvement_three_environment_integration_20261105/REPORT.md
[survey]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/research/pr_rpe_end_to_end_literature_survey_20261106
[preflight]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4b625e5bef06ac51f2b05cafe66f709df2f8bddb/docs/research/track_c_o3_technical_preflight_20261106/REUSE_AND_BASELINES.md
[c2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6f6c084bdfaf04ba951fd5d8ea0c521313ceb0b3/docs/research/track_c2_limited_investigation_20261106/REPORT.md
[c2b]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/602ca4b67b1037e221e910a2733444630c5add62/docs/research/track_c2_b_strong_baseline_20261106/REPORT.md
[audit]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/tracks/algorithm_codesign/interstep_correlated_randomization_prior_art_audit_20261013.md
[base]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/602ca4b67b1037e221e910a2733444630c5add62
[seqpe]: https://arxiv.org/abs/2604.14921v2
[average]: https://arxiv.org/html/2609.36694v1
[pr]: https://arxiv.org/abs/2503.05647v2
[is]: https://arxiv.org/html/2603.13495v1
[rte]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/602ca4b67b1037e221e910a2733444630c5add62/src/trotterlib/rte.py
[tail]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/602ca4b67b1037e221e910a2733444630c5add62/src/trotterlib/df_rte_tail.py
[builder]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/602ca4b67b1037e221e910a2733444630c5add62/src/trotterlib/df_rte_qiskit.py
[partial]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/602ca4b67b1037e221e910a2733444630c5add62/src/trotterlib/df_partial_s2.py
[physical]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/602ca4b67b1037e221e910a2733444630c5add62/src/trottertracks/track_c_source_guarantee.py
[pauli]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/602ca4b67b1037e221e910a2733444630c5add62/src/trotterlib/h4_d2_pauli_control.py
[gates]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/602ca4b67b1037e221e910a2733444630c5add62/src/trottertracks/pr_dictionary_validation/gates.py
[parent]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4ccc1832f63cd323b23fc05623f7e0b58cf34f37/src/trottertracks/algorithm_codesign/parent_allocation.py
