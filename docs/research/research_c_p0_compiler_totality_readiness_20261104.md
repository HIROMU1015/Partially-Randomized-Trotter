# 研究C：P0 compiler全word適用性・技術準備

2026-11-04 JST。**全word成功保証は未成立。`all_q_words_compile_success_qualified=false`を維持する。** 失敗wordの存在を発見したという判定ではなく、固定P0の全到達入力に対する正常終了・有限native出力の根拠がまだ足りないという判定である。既存の構造上限と人工PASSを全word保証に昇格しない。新実H4 build/compile/sampling、保存結果の科学的再解析、Stage P/C、期待bias校正、actual CA判定、重要科学レビューは0。

Branch：`codex/research-c-p0-compiler-totality-readiness-20261104`。起点は`bd560a45410409573ae849ca5aaf4050343910bf`、作業場所は独立worktree `/tmp/prt-c-p0-compiler-totality-readiness-20261104`。正式共通指示2026-10-25原本の§8.1・§8.4・§10を確認した。承認証明や新grantを作らず、技術準備と未承認科学実行を区別した。

## 1. 固定条件・参照と再利用

対象はLinear H4、隣接1.00 Å、STO-3G、number CDF8、元順LD2、partial ordered symmetric S2/s1/r32/K0、h=δ=.1。Nα=Nβ=2、sector36/system256/controlled512、m0..12、own-DF ground-state oracle・状態準備費用除外。PR endpoint SHA256は`a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`。P0はQiskit1.3.0/opt1/seed17、basis rz/sx/x/cx、approximation1、initial-zero false、backend/target/coupling/layout指定なし。既存MPFR160 declared phase、Gaussian、head内relative fusion、tailの同一basis共有を保持する。Samplingは各eventで独立な53-bit IID-grid categoricalという数学モデルで、実PCG64のIID資格とは別である。

| 資料 | 固定結果commit | 今回の扱い |
|---|---|---|
| [B1](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/69c9a764aee66cf2ca79ab29bd3145a027385bcc/docs/research/research_c_p0_uniform_method_readiness_20261103.md) | `69c9a764aee66cf2ca79ab29bd3145a027385bcc` | node/rewrite上限・SOURCEの再利用 |
| [Stage R](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1c0e1d1b4967efecb08dd3d8e7a747eba6c83039/docs/research/research_c_population_range_schema_execution_20261104.md) | `1c0e1d1b4967efecb08dd3d8e7a747eba6c83039` | 保存population metadata・条件付きrangeの引用のみ |
| [Head bridge](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/369e2dfad9a4251ba5eb2fda28f183d559a791cc/docs/research/research_c_head_bridge_execution_20261104.md) | `369e2dfad9a4251ba5eb2fda28f183d559a791cc` | sector reference接続。compiler成功資格は供給しない |
| [固定4-word](research_c_fixed_word_pilot_execution_20261104.md) | `3649b75af191916daf44abb91e14d34c43b83e8f` | 有限wordの数値PASS。全word/IID標本として転用しない |
| [期待bias準備](research_c_iid_grid_expected_bias_readiness_20261104.md) | `bd560a45410409573ae849ca5aaf4050343910bf` | 統計・伝播・qualification engineを再利用対象として確認 |

[固定入力索引](../../artifacts/research_c_p0_compiler_totality/2026-11-04/technical/fixed_inputs.json)に固定commit/path/digestを記録。Stage R/Hはbranch全体をmergeせず、固定Git objectと公開本文を読んだ。NPZ/QPYのdecodeや旧runnerの再実行はしない。

既存`c_p0_uniform_method.rewrite_budget`、C2-A `compiler_coverage/check_graph/trace_artificial`、ZSXXのdomain・composer、`compile_checked/compiled_wrapper`、`StrongBuilder.transition`、`gaussian`、`build_event_sequence`、`SharedQiskitCalls`を確認・再利用した。Atom/event型のsign/support/angle検査、53-bit境界用`grid_index`、`zero_failure_tail_limit`も再実装しない。既存compiler、サンプラー、校正・accuracy qualificationのdefaultと科学SOURCEは変更しない。

## 2. 入力領域・例外経路

| 段階 | 成立する構造・既存チェック | 全word主張に不足する条件 |
|---|---|---|
| Word/event | 正qの216カテゴリ、length32、K0は各eventにrotation1個。supportは整列したZ/ZZ、係数は非零signed rational | 保存catalogのID順とCDF/count順、全frame/basis_keyと実builder入力のbinding。新adapterのschema一致だけでprovenanceを証明しない |
| Raw kernel | `partial_kernel`はevent数・時間・unsigned stored angleを確認。既存event型は有限phase/angle/sign、順序を検査 | 原料tensor→kappa/eigh→Gaussian/diagonalの全有限性・各例外条件を固定入力について閉じる必要。今回は新buildをしない |
| Gaussian | 有限square、unitarity residual≤1e-10、有限ループのGivens、有限transfer残差≤1e-10。8 RZ/最大28 XX+YYという上限 | 単一particleチェックはfull512誤差包囲ではない。固定9 basisとheadの4 directed relative transitionの全成功を示すことは有限前処理課題 |
| Sharing/fusion | 同じbasis_keyの隣接runをまとめる。head transitionは`V_next† V_previous`、analytic vacuum phase。tail異basisはfull-frame往復、head/tail間relative融合なし | Frame順・basis_key整合が必要。tailの6×6 frame pairだけをcompileしてもwhole-P0の全word検査にならない |
| Definition/unroll | 空EquivalenceLibraryを使いofficial definitionをbasis+uへ再帰展開。XX+YY、CRZ、controlled RZZ、P等に既存templateあり | Gate定義の非循環、全param式binding・phase処理、unsupported/opaque/annotated gate排除。未知definitionは例外経路でSTOP |
| Basis translation | 固定graphのUnitarySynthesis/HLS/BasisTranslator。通常grammarはu/nativeのみでopaque unitaryやHLS操作を導入しない | `u`の到達パラメータ・翻訳phase・basis閉包の入力ごとの成立と全domain保証は別。再帰・translator内部をnode数だけで認証しない |
| 1Q/ZSXX | RZ/SX/Xの2×2 f64行列積、arg/abs/atan2、有限分岐、mod2π。ZSXXのみ、target/error-mapなしなら受理rewriteはgate数短縮、出力≤5 gates | 全到達行列のfiniteを閉じる根拠、実装のmatrix取得・DAG更新の前提、phase加算/bindingの例外排除。Accuracyには更に全branchの誤差包囲が必要 |
| Event境界 | Whole-kernel P0にはevent間fenceなし。CX削除が次の反復で長い1Q runを露出させ得る | Event単独・atom単独・frame pair単独の成功は全word成功を含意しない |
| Opt loop | Depth/Sizeの固定点を検査。全受理rewriteとinverse cancellationは固定basis内でnode数を減らす | **1000反復以内の固定点到達は未証明**。上限到達は正常native出力でなくPassManagerError |
| Phase/output | MPFR160 markerを使い、最後にmarker=-1をbindしてfloat cast。未束縛、native basis、qubit/clbit数、非identity layoutを既存wrapperが検査 | Marker gradient消失・bind/cast/remainder例外、native角度/phaseのNaN/Infは別検査。Markerは全内部数値演算の包囲ではない |

K0 central角は固定catalogの`α=atan(λ_R h/32)`とatom signから決まり、finite λ_Rなら|α|<π/2。Native central emitterはその±αを用いる。Wordを変えても新しい任意係数や任意supportを生成しない。Gaussianのfiniteな中間値から出るRZはargの範囲[-π,π]、XX+YY角は`2 atan2(|s|,Re c)`の保守的範囲[0,2π]、そのbetaはarg(s)−π/2の範囲[-3π/2,π/2]にある（inverse時の符号反転も有限）。これは中間値がfiniteという条件付きの範囲で、その条件自体の全入力検証を代替しない。

Head diagonalとscalar角は固定modelの係数に±h/2等を掛け、同一basis runでは既存`combined_diagonal`が`fsum`する。固定SOURCEでの具体的角度の最大値は今回新たに数値評価していない。Rawの有限集合の角度だけを列挙しても、translationとcross-event再合成が作る角度・phaseは更に覆う必要がある。Adapterに科学的根拠のない角度cutoffは設けない。

Rust一次SOURCEは[Qiskit1.3.0固定commitのEuler/optimizer](https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/crates/accelerate/src/euler_one_qubit_decomposer.rs)。`optimize_1q_gates_decomposition`内の`collect_1q_runs`、matrix取得、Euler basis変換にはunwrapがある。その存在だけから合法wordでpanicすると断定しない。標準bound gate・有効DAG・ZSXXの各前提を全入力で満たす必要がある。ZSXXの実際のangle経路は`params_u1x→params_zyz`であり、別basisの`params_xzx`にあるsqrt(det)除算を今回の失敗条件に取り違えない。arg cut/θ≈0,π/2,πは通常の分岐で、それ自体は例外や失敗の証拠ではない。

単に「有限角度だから全積がfinite」「seed17だから全wordで決定的」「人工回路に例外がなかったから例外不能」とはしない。数値successの証明には、標準gate行列評価と各到達積の有限性を示すことが必要で、近unitary性や小誤差までを同時に要求する必要はない。逆にfinite native出力を得ても、literal targetに近いという証明にはならない。

Active graphにはrouting/layout searchがなく、ZSXX候補も一つなので、その範囲のseed依存探索は除かれる。一方、DAG run順、definition/equivalence経路、共有状態、同値なphaseの加算順、wheel/libm等の全決定性はseed固定だけで認証しない。統計実験前に、固定環境でdeterministicなword→outcomeを対象にするか、compiler内部乱数も含むjoint lawを対象にするかを明示する。Build/compile/actual inverseの共有Qiskit状態は既存の**同一jobで共有するmutex**を使用し、並列化時は必要に応じてprocess分離する。

## 3. B1構造上限と停止性

B1の保守的なpost-translation上限は1Q=100914、CX=6608、総node N=107522。最大run長≤100914、受理rewrite≤107522、受理gate visits≤645132、phase update≤107522。これは固定standard definition、basis閉包、targetなし、calibration/control flowなし等に条件付けた構造的主張である。

毎回少なくとも1 node減るという議論は、有限回での固定点到達上限N+1=107523を与える。しかし固定P0のguardは1000である。既存B1の`successful_output_iterations_max=min(1000,N+1)`は**成功出力経路に限る上限**であり、成功の存在証明ではない。[DoWhileController](https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/qiskit/passmanager/flow_controllers.py)は条件が残ったまま上限へ達すると例外を返す。[固定opt1 graph](https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/qiskit/transpiler/preset_passmanagers/builtin_plugins.py)の変更やguard増加は今回行わない。

全word成功を目指す次の技術課題は、実grammarでのCX削除・1Q短縮・新run露出を含む、1000以下のより鋭い反復上限の導出である。抽象状態をまとめる場合は、同じ状態にまとめた全wordが各数値branch・phase・DAG経路で安全であることを示す必要がある。Node数だけ、frame pair数だけ、captured行列boxだけの同一視では不足する。新P0政策でloopを変えるなら元P0保証とは別scopeになる。

このtaskで確認したのは条件付き構造上限・固定graph・人工例の挙動までである。上限に達するwordの反例を得たわけではなく、その確率も未評価。全word216^32の実compileは実施しない。

## 4. 最小adapterと技術テスト

[新adapter](../../src/trottertracks/c_p0_compiler_totality.py)は、供給wordの正grid support、標準raw gate grammar、bound/finite parameterとphase、unitary register、closed control、native output、layoutを検査する。`compile_once`は既存`compile_checked`を既存共有call context内で**一回だけ**呼ぶ。例外はそのまま伝播し、retry・fallback・再抽出はない。Opaqueなgate名だけの偽装も拒否する。

既存`compile_checked`はmeasure/barrierを許す汎用経路で、全paramのfiniteチェックも含まない。今回のunitary kernel scopeの追加検査はopt-inであり、旧SOURCEを修正しない。`require_complete_sample`は全N結果を要求し、失敗/unknownや縮んだ標本を旧all-success校正経路へ渡すことを拒否する。これは将来のfailure-aware統計法の正式採択・実装ではない。

[人工test](../../tests/management/test_research_c_p0_compiler_totality.py)はinvented 2-mode frame/3-qubit回路、signed Z/ZZとlength32イベント列、同一basisと異basis、逆順relative transition、0/±π/2π/1e-300/閾値近傍角、cross-event 1Q、controller上限例外/即固定点、nonunitary/非有限frame、invalid support/parameter/opaque gate、native出力NaN、compiler例外で次wordが始まらないこと、support216/count総和2^53/右境界、有限PASSからの資格昇格禁止を対象とする。保存NPZ/QPYのdecode、実H4 sampling/build/compileは行わない。

SOURCE・非実行契約の照合、Stage P scopeの契約流用拒否、数値library importを禁止したpassive subprocess、未承認scienceの拒否も確認した。**新47件＋関連回帰188件＝235 PASS、既存2件を除外**。除外2件は起点bd560a4でも旧`c_pr_integrated_performance.py`のfrozen SOURCE digest不一致で失敗することを再現した。旧契約を書き換えず、履歴の制約として保持した。[テスト記録](../../artifacts/research_c_p0_compiler_totality/2026-11-04/technical/tests.json)と[起点再現ログ](../../artifacts/research_c_p0_compiler_totality/2026-11-04/technical/parent_known_failures.txt)に範囲・理由を記す。人工PASSは科学的GO・全word資格・小synthesis誤差を意味しない。

## 5. 全word保証が未成立の場合の三案（未採択）

| 方法 | 必要な科学的・技術的条件 | 計算量・可能な結論 |
|---|---|---|
| 1. 元P0の全word成功証明を継続 | 全入力binding、finite native出力、definition/phase例外排除、1000以内収束、固定環境・共有状態 | 静的証明自体は科学call0。固定basis等の新数値確認を要するなら別許可。成功すれば全q-wordのcompiler domainだけを資格化し、誤差保証は別途 |
| 2. Fresh IID-gridの失敗確率 | Nと失敗事象・δ_compile・実行環境・cache・判定法を事前固定。各trialの全wordを保持、replacementなし。N件を完了する設計が必要 | Build/compileはO(N)。有限confidenceでp_fail上界、p_fail=0の証明ではない。小さな上界には相応のNが必要 |
| 3. 失敗を含む別algorithm | 失敗で実験停止、または科学的に定義したfallbackを返す。返却作用・費用・確率の意味を採択し直す | 停止scopeはabort確率を別表示。Total fallbackはその作用・誤差・全終了性の検証が必要。成功条件付き再抽出はqを変えるため現契約へ導入不可 |

方法2のcompile失敗事象は、raw build不能、compiler例外、guard到達、unbound/nonfinite/native/register/layout不適合まで含む全pipeline failureと、rawが適格なときのP0単独failureを分けて登録する。観測された例外だけの評価なら、非停止/hangを無視した「全failure」の評価にはならない。今回、新wall capやtimeoutを科学的failure定義として採択しない。非停止の証明・判定不能なtrialを残す場合は固定N推論を完成扱いしない。

固定N、k失敗の一側exact binomial上界p_Uは、k<Nなら`Σ_{j=0}^k binom(N,j) p_U^j(1-p_U)^(N-j)=δ_compile`を解く。k=Nならp_U=1。k=0では`p_U=1−δ_compile^(1/N)>0`、目標ε_compile以下を0失敗で得る十分Nは`ceil(log(δ_compile)/log(1−ε_compile))`。これはbinomial tail反転からの導出である。[NISTのexact binomial interval式](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm)の上側tailを一側δに置く。既存`zero_failure_tail_limit`を人工条件で再利用検証したが、新H4 outcomeやconfidenceは計算していない。k>0用の数値root solverは未実装。

**即STOP契約と固定Nのk失敗推論は同じ実験ではない。** 最初の失敗でSTOPしたprefixを予定Nのbinomial結果として扱わない。登録したzero-failure採否設計なら、最初の失敗で「認証せずSTOP」とし、全N成功した場合だけ上式を用いることは可能。k>0も集計する設計なら、各失敗trialを保持して次trialへ進む許可と、stop-on-failureからの契約変更を科学実行前に定める。今回はどちらも正式採択せず、科学runnerも追加しない。

実compilerが同じwordでも変動する場合、p_failはqと内部変動のjoint lawで定義するか、環境を固定してdeterminismの前提を立てる。Unique seedや有限の反復一致は、その前提の証明にならない。費用測定用や固定4-wordの観測をfresh IID failure標本として流用しない。

## 6. 期待bias校正・population rangeへの接続

次の資格は互いに独立して扱う。

1. **Literal D(w)の定義・norm**：Stage Rは全正q・length32のliteral targetに対するpopulation normを包囲した。
2. **Compilerの正常終了**：今回も全wordでは未資格。成功した有限wordだけの作用C(w)は全域関数を定義しない。
3. **成功native列の数学的作用**：有限実数角の理想RZ/SX/X/CX列はunitary。その存在・norm1とD(w)への近さは別。
4. **Per-word enclosure**：compiled→raw、raw→literalのfull512、phase、actual inverse、shellを包む必要がある。成功回路のbinary64残差だけではこの資格を得ない。Stage Pの一word証拠が後で得られても全wordへの移送は禁止。
5. **Fresh期待値・統計資格**：q上で全域に定義したCまたは別algorithmが必要。失敗wordを無視した平均は一般に`E[·|success]`で、元の`E_q[·]`とは異なる。

Stage Rの保存norm `a`とrange `M`は[固定metadata](../../artifacts/research_c_p0_compiler_totality/2026-11-04/technical/fixed_inputs.json)にexact rationalをそのまま引用した。a>1・M>2を1/2へclipしない。`range_qualification=conditional_on_total_compiler_domain`のままであり、今回も条件は解消していない。Head bridgeはsector reference間の差を接続するが、compiler domain・per-word native誤差を資格化しない。

別total fallback Fを採択し、全wordでnorm≤1を示せるなら、拡張C_Fに対して`||C_F−D||≤1+a=M`を使える可能性がある。これは**新algorithm**への条件付き移送で、旧P0の全word保証ではない。成功側誤差の非条件付き寄与を校正し、失敗側はM·p_failで保守的に評価する方法を検討できる。成功条件付き標本平均だけを旧mean_upperへ渡すことはできない。Identity fallbackなどを今回実装・採択しない。

Compile時STOPだけなら、無条件native signalは欠け、abortを含む出力空間が必要になる。将来T回compileする計画では、固定pの下でunion上界min(1,Tp_U)、独立trialなら1−(1−p_U)^Tを別に扱う。Tは2^mだけではなく、複数round・shot・X/Y・cache/build方針を含む実際の要求に合わせて固定する。停止・再抽出・cacheを都合よく混ぜない。

δ_compileは失敗率上界を得る**校正の失敗確率**で、p_failは**運用時のcompiler failure確率**である。δ_calと既存RPE測定α=.01も別。校正成功eventに条件付けたRPE十分条件と、無条件の失敗/abort確率を分け、必要ならunionで加算する。δ_compile+δ_cal+.01だけで運用abortまで消えるわけではない。既存α=.01への再配分、N、δ_cal、failure配分は今回は固定しない。

## 7. 未承認契約・calls・資源

[非実行設計契約](research_c_p0_compiler_totality_contract_unapproved_20261104.json)は`executable=false`、`maximum_logical_units=null`、method/N/δ_compile/δ_cal/ε_compile/failure policy/future scientific SOURCEはnull。Technical SOURCE一覧は再現性用で、実行許可を証明するものではない。[入口](../../scripts/prepare_research_c_p0_compiler_totality.py)はpassive照合のみで、実行flagもsampling入口も持たない。許可flagだけ渡しても未完成の科学設計は起動できない。

将来のcompile-failure-only案は、setup binding S_bind、Gaussian cache miss G_basis、relative transition miss G_relativeを別計上し、各trialを「word抽出・保持／raw build／P0一回／native finite-register-layout判定」の4 logical units、最終summary1とすれば、**S_bind+G_basis+G_relative+4N+1**。これは処理区分の設計式であり、実行上限ではない。Setupの入力group・model/catalog/support照合とcache政策を確定するまで数値を入れない。内部既存builder callbackのGaussian呼出しをN回buildへ隠さず、wordごとに0回とは仮定しない。

Per-word synthesis enclosureや期待bias校正も必要なら、full512 raw/compiled records、compiled/raw certificate、raw/literal bridge、actual inverseとshell、集計を追加で固定する。既存Stage Pの41、固定4-wordの86、Stage Rの256を新実験の上限へ流用しない。Whole-wordを網羅するなら216^32 build/compile規模であり、今回の方法として提案しない。全domain静的証明の科学call自体は0だが、固定basis・角度の新数値検証には別明示許可が必要。

将来のsubstantial run前に実CPU affinity/quota・RAM・GPUと既存timingを確認し、独立trial/process、immutable basis cache、列batch等を選ぶ。Qiskit共有状態は同一job mutex/process-local、BLAS/Rayon/workerの多重並列を調整する。今回の小さい人工回路・scalarテストにGPUは不要。CPU/RAM/worker/wallの恣意的cap、共有venv/cache/OSの変更はない。速度比較を実施しておらずspeedupを主張しない。

Identity/source/nonfinite/unsupported/例外時は依存投入を止め、失敗wordと部分結果を保持しmandatory STOP。新科学実行時は、採択済み固定SOURCE/input/call定義、fresh output、retry/resume0を別に確定する。科学的に定義した統計trial継続を採択しない限り、失敗後の続行・再試行はしない。

## 8. 保全と引渡し

旧library・科学SOURCE/raw/契約/grant/ledger/marker/STOPは無変更。Stage Pのworktree `/tmp/prt-c-stage-p-execution-20261104`は読み取りのみで、固定契約・technical SOURCE・P実装2 fileを基準commit369e2dfと照合した。Cache/lock/processに対する操作は0。[保全照合](../../artifacts/research_c_p0_compiler_totality/2026-11-04/technical/preservation.json)に最終結果を記録する。別Codexが生成するStage P出力の変化は本taskの生成物と混同しない。

今回は未資格statusを変更する新科学結果を得ていないため、研究概要・科学manifest・dated科学ノートを重複更新せず、技術報告と関係索引だけを追加した。公開後はremote SHA/treeと報告・契約・SOURCE・test本文を固定Git blobに照合する。

次の担当は**ユーザー／研究C**：三案の採否と、必要なら失敗率の目標・confidence・failure/abortの扱いを選ぶ。**Codex**：採択された案について、鋭い停止性証明または不足する数値前提・fixed-N実験設計を具体化し、科学実行前に固定SOURCE・callsを完成させる。この提案は新科学計算や重要科学レビューの許可ではない。

**MANDATORY STOP。新実H4計算、Stage P再実行、期待bias校正、Stage C、actual CA判定、重要科学レビューへ進まない。**
