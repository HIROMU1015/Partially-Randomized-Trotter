# 研究C：C2-A ZSXX全到達領域・whole-P0誤差保証の技術検証

2026-10-27 JST。起点`fd6820e9f922cb935e61334533898a969ad4f363`。

**全runのSOURCE由来のdomainを、RZ/SX/Xのdeterminant 4 cosetとrun長依存のnear-unitary tubeへ絞った。条件付きの逐次積・whole-P0 composerと、一部のuniform translation規則を実装した。未観測Mに対する実Rust ZSXX operator imageは未資格。** 新規32＋既存回帰161＝193 tests PASS、旧順序依存test 1件は引き続き明示除外。新規実H4科学callは0。

[新契約](research_c_pr_aware_df_zsxx_uniform_contract_draft_20261027.json)は`BLOCKED_UNIFORM_RUST_IMAGE`／`executable=false`／科学call上限null。未知値を0にせず、前回のcaptured-pair証拠・旧科学SOURCE・結果・STOPは変更していない。今回の到達点は「全law保証の完成」ではなく、SOURCEから得られる制約・条件付き評価式と、stock Rustで残る具体的な数値保証の境界の確定である。

## 1. 対象、再利用、差分

将来対象は保存直鎖H4・1.00 Å・STO-3G、通常number CDF rank8のBaseline/Proposed、Nα=Nβ=2/sector36、system8+ancilla1/full512、元frame順LD2、partial symmetric S2 δ=.1/s4/h1/40/r16/K0、m0..12、own-DF ground oracle／状態準備除外。元g補正κ・scalar・Gaussian/P0政策、保存C1/C2-B/Bridge証拠を保全する。今回、保存J/U/NPZ/QPYの数値decode、実H4 operator/reference/build/compile、sampling、shots、総資源、phase-room再評価はない。

| 再利用する資産 | 今回の用途 |
|---|---|
| `c_pr_aware_df_zsxx_enclosure` | exact dyadic、Arb gate意味論、balanced reference積、outward norm、RLock、runtime条件を再利用 |
| `c_pr_aware_df_c2a` | raw grammar、P0/check_graph、monotone_budget、symbolic_templates、trace/旧回帰、uniform_previewの条件を維持 |
| `stage_c2_audit`／`c2_bridge` | SOURCE/compiler経路、CAR/law/referenceの分離、全round unitary telescopeを確認 |
| `c1_h6_compiler_precision` | 既存MPFR160 marker、最後のbind/castという実経路を確認。Compilerは変更しない |
| B2 `SharedQiskitCalls` | 人工definition/translation生成の既存process-local lock |
| 既存ZSXX／Stage B/C/C2 test | 旧SOURCEを変更せず回帰161件を実行。旧例外順序の2代替testも継続 |

新規は[uniform adapter](../../src/trottertracks/c_pr_aware_df_zsxx_uniform.py)、[固定人工fixture](../../src/trottertracks/c_pr_aware_df_zsxx_uniform_artificial.py)、[passive/artifical-only entry](../../scripts/check_research_c_pr_aware_df_zsxx_uniform.py)、[tests](../../tests/management/test_research_c_pr_aware_df_zsxx_uniform.py)。既存にはuniform run長依存式・det coset tube・明示phaseモデル・capturedとcompiler imageを分けるcomposerがなかったため、opt-inで追加した。新ZSXX compiler、Gaussian emitter、optimizer、数値計算ライブラリは作っていない。

## 2. SOURCEから得た全到達domain

[到達領域資料](../../artifacts/research_c_pr_aware_df_zsxx_uniform/2026-10-27/technical/reachable_domain.json)／[外部SOURCE identity](../../artifacts/research_c_pr_aware_df_zsxx_uniform/2026-10-27/technical/source_inventory.json)。固定Qiskit 1.3.0のEuler、gate_matrix、operations、DAG/CircuitData、BasisTranslator search、Cargo.lock、対応するnum-complex 0.4.6を調べた。Git blobと参照先を保存した。Installed wheelのbuild flags/toolchainと公開SOURCEの完全なbinary対応まで認証したという意味ではない。

固定P0、basis閉包、target/error_mapなし、ZSXX only、parameter-free cancellation、calibrationなし、finite bound numeric・unitary grammarを前提とする。

1. Raw GaussianはRZ/XXPlusYY、中央制御・scalar等はCRZ/controlled RZZ/Pへ。Official unroll後はu/rz/sx/x/cx、初期basis translation後の1Q runは**rz/sx/x**。CXはrunを区切る。元event境界は区切りにならない。
2. ZSXXが生成したrz/sx/xは後続runへ再入力される。初回raw alphabetの有限観測だけを以後の全入力集合にしない。
3. 理想runのgate部分Gでは、det(RZ)=1、det(SX)=i、det(X)=−1なので、**det(G)∈{1,i,−1,−i}**。Compilerの宣言phaseは回路global phaseへ移され、次のrunの2×2積には入らない。この区別が必要である。
4. RZの初期angleは安全なsupersetとして任意のfinite binary64を許す。一部Gaussian/event angleのSOURCE上の制約はあるが、未decodeの全raw diagonal係数を人工例のmin/maxへ制限しない。Rewrite後のfinite RZ callbackはmod2πでほぼ[−PI_f64,PI_f64]へ縮約されるが、それだけで最初のRZ行列評価の全domainを置換しない。
5. Ideal RZ/SX/X envelopeはEuler polesもdet-arg cutも含む。4 cosetに絞れても、det≈−1のdiscはargの±π境界を横切り得る。Branchを都合よく除外できない。

### 初期node数の構造的な上限

SOURCEの固定definition topologyを前提に、実H4回路を作らず上限を導いた。`U→U3→3RZ+2SX`は5-node witness。固定BasisTranslator searchはnative gate cost1の加算costを使うため、同じequivalence graphではUの翻訳cost≤5。これは数値phaseの正しさの証明とは別である。[Basis search](https://github.com/Qiskit/qiskit/blob/1.3.0/crates/accelerate/src/basis/basis_translator/basis_search.rs)、[equivalence定義](https://github.com/Qiskit/qiskit/blob/1.3.0/qiskit/circuit/library/standard_gates/equivalence_library.py)。

| Primitive | 固定official unrollの最大topology | 最終basis node上限 |
|---|---|---:|
| XXPlusYY／inverse | 7u+6rz+2cx+sx | 44 |
| 閉control CRZ | 2rz+2cx | 4 |
| 1-control RZZ | 2 CCX+CRZ。CCXは9u+6cx、合計18u+14cx+2rz | 106 |
| P | u | 5 |
| RZ/CX | そのまま | 1 |

XX definitionはparameter-independentな命令列、inverseは同topology。RZZ controlは一つの追加controlでCX→CCX、RZ→CRZ、closed control・use_basis_gates=falseの経路。MCXの2-control dispatchとCCX definitionを確認した。Equivalence searchはgate名・qubit数で経路を選び、数値angleでcost graphを変えない。人工symbolic traceはこのSOURCE議論の回帰であり、有限traceだけを全parameter証明にはしない。

既存固定raw grammarから、40 Gaussian×(8 RZ+28 XX)、head6×(8 CRZ+28 controlled RZZ)、tail16×(2 RZ+4 CX)、scalar P≤7。従って

`N ≤ 320 + 1120×44 + 48×4 + 168×106 + 96 + 7×5 = 67731`。

これは**固定K0 kernel grammarと上記official経路を前提とするSOURCE構造上限**である。実H4 node数の測定でもnative費用でもない。別head/readout/actual-adjoint wrapperまでこのNで覆ったとはしない。新control、target/plugin/calibration/opaque synthesis、equivalence変更には適用しない。`grammar_node_bound`はcoverage前提なしではnull。数値合成誤差の資格はこの整数上限から得られない。

## 3. Run長依存の条件付き誤差

[算術・branchモデル](../../artifacts/research_c_pr_aware_df_zsxx_uniform/2026-10-27/technical/arithmetic_and_branch_model.json)。Rust `matmul_1q`は、各matrix entryを二つのcomplex積の和として**逐次**更新する。num-complex 0.4.6の通常multiplyはreal componentごとに4 real productと3 add相当、各termのrounding pathは最大3段。Arb referenceのbalanced積でこの実順序を置き換えない。

仮定はbinary64 RN、gradual underflow、finite intermediates、reassociationなし、同じ式のseparate演算またはcorrectly-rounded FMA contraction。Installed wheelの全該当条件を認証したものではなく、返す状態は`CONDITIONAL_ENCLOSURE`。FMAの有無を隠れて固定しない。

u=2⁻⁵³、h=2⁻¹⁰⁷⁵、γ₃=3u/(1−3u)。各gate理想行列がunitaryで、その数値行列評価に全domain共通のoperator誤差aがあると仮定する。直前の積の誤差をeとすると、component誤差は

`4γ₃(1+a)(1+e)+7h(1+u)^3`以下。

8 real componentのFrobenius上界を保守的にまとめると、一回のmatrix積roundingは

`16γ₃(1+a)(1+e)+28h(1+u)^3`以下。

p=(1+a)(1+16γ₃)−1+28h(1+u)³とおくと、1+e_next≤(1+p)(1+e)。従ってLp<1で

\[
\|G-M\|_2\leq\eta_L=(1+p)^L-1\leq\frac{Lp}{1-Lp}.
\]

後半はbinomial係数≤L^jによる保守的な有理上界。巨大なLのmatrix計算やexpansionを不要にし、underflow項を0へ落とさない。Lp≥1ではこの小誤差式は使えずnullにする。計算資源の人工的なrun長上限ではない。

このときSOURCE上の理想Gに対する条件付き到達集合は

\[
D_L=\{M:\exists G,\det G\in\{1,i,-1,-i\},\ G^*G=I,\ \|M-G\|_2\leq\eta_L\}.
\]

`||M* M−I||2≤2η+η²`、η<1ならσ_min(M)≥1−η。また`|det M−det G|≤2η+η²`。後者はdet(I+E)=1+trE+detE、|trE|≤2η、|detE|≤η²から従う。ここでのdefectはoperator2であり、旧verifierのFrobenius入場guardとは区別する。

**実際のaは未知。** [RZ行列SOURCE](https://github.com/Qiskit/qiskit/blob/1.3.0/crates/circuit/src/gate_matrix.rs#L314)は`exp(±i*0.5*theta)`を使い、[num-complex](https://github.com/rust-num/num-complex/blob/num-complex-0.4.6/src/lib.rs#L239)はexp(re)とcos/sinへ進む。SX/Xのdyadic matrixはexactでも、すべてのRZ evaluationをa=0とはできない。人工資料のa=0はexact-node仮定／SX-X-only部分集合の算術診断に限る。RewriteにRZが入れば、その部分集合は閉じていない。実H4のG→M boundには採用しない。

## 4. 未観測入力とRust image

| 方式 | 今回の成果・限界 |
|---|---|
| A：validated input-domain evaluation | Arb192で**理想template** RZ([−1,1])→SX→RY([0,4])、phase[−.2,.2]の全連続parameter boxを囲う。未観測parameterも含むが、Rust compiler(box)のimageではない |
| B：Rustモデル＋解析評価 | 上記run式、predicate widening、phase式、whole composerを構成。RZ complex-exp／arg／atan2等のuniformモデルがなく、実Rust ζ(D_N)はnull |
| C：SOURCE domain制限 | RZ/SX/X、det4 coset、N/L/rewrite/visit上限、条件付きnear-unitary tubeへ絞る。Euler poles、mod/arg cutを除外する証明にはならない |

`theta_branch_cover`はstrict abs<atolとbinary64 subtractionのoutward誤差を含め、θ≈0/π/2/πの両経路を残す。複数branchを横切る区間も扱う。Generalのlambda/phi flip、RZ emit/omit、SX pair/X、endpoint clampは両側を保守的に保持する。**全branchの論理supersetは、出力operator imageの囲い込みではない。** 返り値に`compiler_interval_image_enclosed=false`を固定する。

旧captured-pair、held-V box、ideal template imageをuniform Rust proofとしてcomposerへ入れると拒否する。精度192を増やしてもこの不足は解消しない。角度自体の一様Lipschitzだけに頼ってpoleを消す方法は採用していない。

Installed `_accelerate`のdynamic importにはatan2/cos/exp/fmod/sin/sincosを確認した。glibc 2.35の[一次資料](https://github.com/bminor/glibc/blob/glibc-2.35/manual/math.texi#L1296)は、一般math関数のcorrect roundingを目指しておらず、一部は精度目標にも違反すると明記する。目標の「few ULP」やtest用ULP表を全入力保証にしない。atan2/arg、RZ exp、remainderの固定wheel全call経路を含む有効な一様誤差モデルは未取得。有限high-precision比較も代用しない。

現在、意味のある実Rust一様operator誤差上界は構成できていない。Unitary間距離≤2という自明な包絡を有用な全law資格として採用しない。

## 5. Initial translation・phase・actual inverse

[規則と伝播](../../artifacts/research_c_pr_aware_df_zsxx_uniform/2026-10-27/technical/translation_and_propagation.json)。Exact unitary identityとparameter丸めを分離した。

- **閉control CRZ**：official definitionはRZ(theta/2),CX,RZ(−theta/2),CX。すべてのfinite binary64 thetaでhalving誤差≤h。Control0はexact cancellation、control1はRZ(2fl(theta/2))。よってideal operator距離≤|fl(theta/2)−theta/2|≤h。Normal域の半分はexact、subnormalのtieを含める。Parent回路phase等を含めた全translation誤差ではない。
- **非制御RZZ**：CX,RZ(theta),CXはparameterをそのまま保持し、ideal gate semanticsでexact。ここだけのbound0は未知誤差の穴埋めではない。実kernelのgeneric controlled RZZ全体を0とはしない。
- **RZ inverse**：finite parameterの符号反転はexact。Global phase normalization、SXdgを含むinverse全体、inverseを再compileする場合は別項。

一方、XXPlusYY/controlled RZZ、U/U3 parameter式、binary64 PIと数学的πの差、全global phase補正、MPFR160 expression演算、最終cast、head/X/Y shellのuniform numerical coverageは未完了。Topology上限はこれらの誤差を消さない。

### Phaseで新たに明示した項

Rust DAG/CircuitDataはFloat phaseを`rem_euclid(2*PI_f64)`で縮約する。ParameterExpressionはその段階のFloat縮約を避けるが、最終cast後には再び同処理を通る。MPFR160 markerはこれを数学的な2πへ変更しない。

Exact phase atomの絶対値和をA、accumulator演算数をn、precisionをp bitsとする。正しいrounding・有限exponent・全演算coverageを仮定すればγ_n,p Aで累積を囲え、最後のcastはu64(A+γA)+hで囲える。さらにT_f64=2PI_f64、rがある整数qに対して`r=p_cast−q*T_f64+error_rem`を満たすなら、**真の2π周期**との違いが必要である。

`epsilon_phase ≤ γA + epsilon_cast + epsilon_rem + Q|T_f64−2π|`、unitary phase距離として上限2。

Qはcast前の絶対値上限から保守的に囲う。`period_gap`はArbのπからoutwardに求め、約2.4492935983e-16（規範値はexact rational）。この項は観測最大値ではない。複数normalizationがあれば各回を別に計上し、この一回用式を全wrapper保証へそのまま使わない。

Exact fmodを仮定すれば、rem_euclidの負remainder+Tのaddに対してepsilon_rem≤u64*T+hという条件付き式を使える。ELFにfmod importがあることだけを全実装の証明とはしない。現wrapperの全MPFR演算とremainderモデルは`verified=false`。Unknownのままならphase項はnull。

人工例では、phase=−最小subnormalをsetterへ渡すとstored phaseがT_f64となり、真のphaseとの距離が約2.449e-16生じた。これはcaptured setter例でありuniform libm boundではない。別の人工marker cancellation例では[1e16,.125,−1e16]×128のbind結果16を確認したが、その成功だけで全160-bit処理を保証しない。

## 6. Whole-P0接続

R≤N、採用入力ΣL_j≤6N、各L_j≤Nを再利用する。Run項η_L=Lp/(1−Lp)より

\[
\sum_{j\in accepted}\eta_{L_j}\leq\frac{6Np}{1-Np},
\quad
\epsilon_{P0}\leq\epsilon_{translation}
+\frac{6Np}{1-Np}+N\zeta(D_N)+\epsilon_{phase}.
\]

ζ(D_N)は**位相込みの実Rust M→Vを全到達入力・全branchで覆う一様operator2上界**。有限sample最大ではない。pにはRZ等のmatrix評価aを一度だけ入れる。G→V直接boundを別に加えない。不採用rewriteは元runを変えず、そのcandidate計算誤差は加算しない。Rewritten gateの再入力は、新ノードを含むΣL≤N+5Rのamortizationで扱う。

`uniform_whole_bound`は同一norm／exact unitary target／phase-sensitive／全domain・全initial path・全phase normalizationというproof scopeを要求する。条件や入力不足ではnull、captured boxやmax-entry-only proofは拒否する。外部proofの数学的正当性をschemaが認証する機能はなく、正常時も**CONDITIONAL_WHOLE_P0_PROOF_INPUT_BOUND**。実H4/all-lawの数値欄はnullのまま。

既存`uniform_preview`とBridgeのexact rational全round接続は、有効なwhole-word/raw/reference/shell入力が揃った後だけ使える。C2-B law、CAR bridge、Gaussian/rawの数値誤差は別項であり、今回のcompiler composerへ重複加算しない。m12の有用性、H4のphase room、CA/shots/totalを評価していない。

## 7. 人工・回帰・保全

[人工domain出力](../../artifacts/research_c_pr_aware_df_zsxx_uniform/2026-10-27/technical/artificial_domains.json)／[集計](../../artifacts/research_c_pr_aware_df_zsxx_uniform/2026-10-27/technical/tests.json)。新規[32 PASS](../../artifacts/research_c_pr_aware_df_zsxx_uniform/2026-10-27/technical/uniform_tests.log)、旧ZSXX52＋旧回帰109＝[161 PASS](../../artifacts/research_c_pr_aware_df_zsxx_uniform/2026-10-27/technical/regression_tests.log)、計193。旧順序依存test 1件はdeselectし、既存の両順序代替testを維持した。旧非決定性は未修正で、旧SOURCE/test/初回FAIL log/STOPは保全。

新規はrun長1〜2²⁰のsymbolic budget、underflow、missing model、invalid/FMA/FTZモデル拒否、branch境界両側・複数branch区間、未観測ideal template parameter、長さ1024相殺／768非可換run、norm/phase/unitary条件、captured image拒否、translation/subnormal/max finite angle、phase true-period、MPFR marker、全path/null接続、SOURCE/input不一致・science execute option不在を確認した。

逐次積の人工replayはSOURCE通りのcomplex2x2式であり、private Rust optimizerから内部Mを直接取得したと偽らない。Arb exact referenceはbalanced積。非可換順序を逆にした故障はroundingではなく意味論の不一致として検出する。これらの有限試験は解析式の診断・回帰であり、未知Rust libmのuniform証明を追加するものではない。

新規testの初回[log](../../artifacts/research_c_pr_aware_df_zsxx_uniform/2026-10-27/technical/uniform_initial.log)はCLI negative testの期待順序だけ1 FAIL。Argparse required-group検査が先に出るため、passive optionを与えてからunknown `--execute`を検査するようtestを修正した。Production/source数値処理の変更や科学retryではない。旧例外競合を単なるrerunで解消済みとしたわけでもない。

[Passive preflight](../../artifacts/research_c_pr_aware_df_zsxx_uniform/2026-10-27/technical/passive_preflight.json)はSOURCE116／保存input-provenance29／外部Qiskit Python SOURCE22／package11をread-only照合。NPZ decodeと新数値entryを遮断した状態でPASSし、fresh科学outputは未作成。[保全](../../artifacts/research_c_pr_aware_df_zsxx_uniform/2026-10-27/technical/preservation.json)は旧1889件中1887件byte不変、2 READMEだけ入口追記。旧科学SOURCE/raw/grant/ledger/marker/STOPに変更なし。

Own worktree/venv/cacheを使用し、依存は前回環境をread-only参照。Arb192 context lock、既存Qiskit mutexを再利用した。2独立suiteを並列実行し、2×2/小人工回路のためGPUや巨大密行列は不要。環境の空き資源・pressureを[記録](../../artifacts/research_c_pr_aware_df_zsxx_uniform/2026-10-27/technical/environment.json)し、CPU/RAM/worker/wallの人工上限は設けない。実分子でのspeedupや資源優位は測定していない。

## 8. Q1〜Q5・最小の次作業と契約

**Q1：全runの構造をsource由来のsupersetへ絞れた。** rz/sx/x、det4 coset、条件付きnear-unitary tube、R≤N／ΣL≤6N、固定kernel grammarでN≤67731。最後の整数は条件付き構造上限であり実回路測定ではない。Euler pole/cutは残る。

**Q2：実Rustの未観測入力を含む有用な一様operator誤差上界は未完成。** 全parameterに対するideal template image、CRZ等のcomponent規則は成立。実Rustはcaptured-pairまで。Mode A/B/Cを比較したが、RZ complex-exp評価aとEuler/branch imageの全域モデルを得ていない。

**Q3：run長と回数の条件付き接続は実装した。** SOURCEの実逐次順序、underflowを含むp、ΣLのamortizationから6Np/(1−Np)を導いた。Unknown aを0にせず、観測最大値×6Nも使わない。

**Q4：translationの一部とphaseの不足を具体化した。** CRZ/RZZ/RZ inverseの規則、MPFR/cast/remainder/真の周期誤差の分離、条件付きwhole composerまで。全initial translation、phase normalization、raw/reference/shell/actual inverseが完成したわけではない。

**Q5：実H4科学契約を実行可能には固定できない。** [新草案](research_c_pr_aware_df_zsxx_uniform_contract_draft_20261027.json)はSOURCE、旧入力、P0、precision192、classification、fresh output、retry/resume0、STOPを固定したが、A1/A2/A3 call上限はnullのまま。許可証明や新grantが不足しているのではなく、数値評価法・coverageが不足している。

次に必要な最小technical作業は、(1) fixed RZ complex-expの一様a、(2)同じD_N上のarg/atan2・simplification/modを位相込みoperatorとして囲う方法、(3)選んだ実験に必要なtranslation/phase規則の残り、である。Stock libmのtest ULP表や追加有限probe、Arb精度増加だけでは(1)(2)を解決しない。具体的な検証済みplatformモデルを得るか、compiler政策／保証目標の変更が必要なら研究Cで実質判断する。今回のscopeでP0を変更しない。

最小科学実験は、前提が成立した後のA2 local whole-P0、または独立reference evaluatorが完成して情報価値がある場合だけA1 finite Gaussian/phase/referenceを選ぶ。全trajectory列挙・大量compile・追加samplingを既定にせず、未実装のA1へ架空のcall ceilingを割り当てない。A3は有効な全入力が揃った後である。

Branch：`codex/research-c-pr-aware-df-zsxx-uniform-preparation-20261027`。
[SOURCE116](../../artifacts/research_c_pr_aware_df_zsxx_uniform/2026-10-27/technical/SOURCE.json) SHA256：`7cabc12d9847b87af21cd3fce9ad08ff8eadfbc82c7c4f991fe9425829370368`。
[技術STOP](../../artifacts/research_c_pr_aware_df_zsxx_uniform/2026-10-27/technical/STOP.json)：実H4科学call0。

次は**研究C／ユーザー**がこの条件付き成果と残るplatform/保証方針を確認する。**Codex**は次の具体的な指示後に不足へ絞ったtechnical作業を行い、実行可能な科学契約を固定できた場合にのみ別途明示許可後に実H4へ進む。公開・remote/SOURCE/成果物照合後は**mandatory STOP**。新規科学計算、C1再解析、sampling、shots/総資源、別分子、GPT詳細科学レビューへ自動進行しない。
