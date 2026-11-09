# Q3-S：証明義務・SOURCE・不足条件

[REPORT](REPORT.md)の固定科学条件・commitを使用する。`PROVED_STATIC`は下記で明示する数学・runtime条件下のSOURCE上の命題、`CONDITIONAL`は適用条件付き、`UNQUALIFIED`は固定m17全supportへの証拠不足を表す。固定条件内の反例は今回確認していない。

## 1. 固定SOURCEへの対応

| SOURCE | 追跡した経路 |
|---|---|
| [provider eaf70d88](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eaf70d886d42364a01b7e949ebc21f3b026bbe72/src/trottertracks/pr_reference_v6q.py#L18) | _binary_input／_lane_sum／_round_once／_verify_nearest_even、round selector、旧coreへの委譲 |
| [assemble f2638a78](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2638a7845685a119619382b850441f5d8d33713/src/trottertracks/pr_reference_transfer_qualification.py#L156) | 同じevent列、uncontrolled F/G、q≥2の二分、旧phase callsite |
| [partial S2](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2638a7845685a119619382b850441f5d8d33713/src/trotterlib/df_partial_s2.py)／[repeated](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2638a7845685a119619382b850441f5d8d33713/src/trotterlib/df_partial_s2_repeated.py#L575) | 記録basisの前後、D half/full、scalar集約、D境界融合、compose |
| [D primitives](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2638a7845685a119619382b850441f5d8d33713/src/trotterlib/df_trotter/ops.py#L226) | 完成済みbinary global_phase／RZ／RZZ、数値型変換、exact-zero pruning |
| [RTE event](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2638a7845685a119619382b850441f5d8d33713/src/trotterlib/rte.py#L1594)／[RTE builder](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2638a7845685a119619382b850441f5d8d33713/src/trotterlib/df_rte_qiskit.py#L278) | order0/2、sign、product→rotation順、basis run、0/π加算とglobal_phase |
| [basis registry](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2638a7845685a119619382b850441f5d8d33713/src/trotterlib/df_rte_tail.py) | 有限の記録operations、basis ID/hash、wire/size、metadata照合 |
| [旧M・B_norm](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2638a7845685a119619382b850441f5d8d33713/src/trottertracks/pr_reference_transfer_error.py#L99) | phase_boundのprefix／normalization、保存算術仮定 |
| [reference core](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2638a7845685a119619382b850441f5d8d33713/src/trottertracks/pr_reference_interference_pilot.py#L120)／[compiler A](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2638a7845685a119619382b850441f5d8d33713/src/trottertracks/track_c_compiler_a.py) | F.inverse／G／P／CSWAP、fixed inverse cancellation、definition lowering |

Qiskit 1.3.0の固定一次SOURCEはcommit `896d1c8b72888a52a46081a6d1f4de8c56db8aea`。既存の[global_phase／copy／compose／inverse監査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2638a7845685a119619382b850441f5d8d33713/docs/research/pr_reference_n3b1a_emitter_static_20261108/EMITTER_SOURCE_AUDIT.md)を再利用し、今回は下記のgate inverse・parameter処理・setterを一次SOURCEでも限定照合した。wheel全体の形式証明ではない。

## 2. Q3-S1：受入れ文法と有限literal

### 2.1 読取算術 — `PROVED_STATIC`

_binary_inputはfloat継承型の実際のC doubleを`float.__float__`で取り、`as_integer_ratio()`をFractionへ移す。builtin intはbinary64へexactに写せる場合だけ受け入れる。bool／Decimal／Fraction／complex／ParameterExpression／それ以外のnumpy型を便宜的にcoerceしない。NaN/Infinityは拒否する。numpy.float64の受入れがQ2で観測されたことは、他のnumpy scalar全部の受入れを意味しない。

_lane_sumはgから始め、命令順に全parameterを検査し、RZ/RZZだけFractionでθ/2を引く。両laneの和もFractionのまま。物理演算や元Hから理想角度を再構成せず、SOURCEが生成した値を対象とする。floatへの変換は合計の最後の1回であり、読取中にlaneやgateを変更しない。

### 2.2 生成経路 — 条件を満たす有限poolに対する `CONDITIONAL`

| 要素 | SOURCEの生成／変換 | 全supportで必要な条件 |
|---|---|---|
| D角・Dのphase | _as_real／_real_or_symbolicが数値入力をPython float化。one-body、DF二乗の積・和を既存順で評価し、RZ/RZZとglobal_phaseへ | 固定5 primitive contextの実値がfinite。有限operandだけでは積のoverflow防止まで自動証明しない |
| D／RTE basis | 保存runtime_operationsをそのままappend。inverse時は逆順にgate.inverse。registry照合はID/hash/operations/sizeを要求 | 全D＋全registryの真正RZ/XX+YY、shape、wire、全parameter型・finite性。registryの一般APIだけでこの狭い族を強制してはいない |
| RTE rotation | _make_eventでa_o=float(atan(τ_r/(o+1)))、o∈{0,2}。unsigned_rotation_angleはsignを適用し、RZ/RZZは±2a_o | 固定r32の保存角witnessを利用可能。RTEEventの数値finite検査を保持。成分選択で別の角の大きさを作らない |
| RTE product/sign | uncontrolled productはZ、負符号とevent phaseは0またはbinary math.piを位相和へ | component supportは1/2-site、eventの順序と符号規則を不変更 |
| identity | 一般builderのidentity productは0/π、rotationは−unsigned_angle | 固定tailは旧構造資料でidentity成分なし。新identity policyを追加しない |
| scalar | repeated builderの`-.02*n*(constant+extracted_identity)`。uncontrolled global_phaseへ | 完成した固定n=q/2の値がfinite。積の再結合で置き換えない |
| g_F/g_G | numeric constructor/setter、global_phase加算、copy/compose。numeric Floatはfloat周期のrem_euclid | symbolic branchなし、finite更新、旧算術モデル。保存gの生成誤差はB_normとして残す |
| 幅・順序・条件 | uncontrolled lanesは8 qubit、q≥2では完全なstepを前後半に分ける。RTE application順は保存順。clbit/conditionを付与するpathなし | 同じassembly・canonical lawとpool、追加gateやcontrolなし |

[Qiskit Gate.validate_parameter](https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/qiskit/circuit/gate.py#L256)はint/floatを保持し、numpy integer/floatingに.item()を使い、ParameterExpressionも許す。このAPIだけではproviderのより狭い型条件・finite性を証明しない。SOURCE数値経路と保存poolとの対応が要る。

既存[A1](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2638a7845685a119619382b850441f5d8d33713/artifacts/pr_reference_interference_pilot/2026-11-06/A1_REFERENCE_PHASE.json)は、[stage_a1](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2638a7845685a119619382b850441f5d8d33713/src/trottertracks/pr_reference_interference_pilot.py#L260)で全D／全registry basisを走査した保存結果で、basis_familiesはrz／xx_plus_yy。これをQ2の4 trajectoryより広い**固定basis catalogの族の証拠**として再利用する。ただし全parameterの元の型・finite literal域を記録した新監査の代用にはしない。

したがって文法の条件付き閉包は整理できるが、固定m17全supportのprovider受入れを丸ごと認定するには不足がある。**実型／finite poolの完全な適合は `UNQUALIFIED`。** 回路の種類を増やさず有限metadata監査で確認可能である。dense Gaussian fallback、support-restricted basis plan、control付きF/G、symbolic phaseは今回の固定assemblyには `NOT_APPLICABLE`。

## 3. Q3-S2：exact和とnormal域

### 3.1 対になったbasisの消去補題 — `PROVED_STATIC`（指定族・完全なrunの条件下）

gを取得済みSOURCE global_phaseとし、真空位相を

\[
S_L=g_L-\tfrac12\sum_{j\in L,\,RZ/RZZ}\theta_j
\]

とする。固定RZ inverseは[−θのRZ](https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/qiskit/circuit/library/standard_gates/rz.py#L135)である。finite binary64の符号反転はexact。XX+YY inverseは[第一角だけ符号反転、第二角を保持](https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/qiskit/circuit/library/standard_gates/xx_plus_yy.py#L207)し、両方向とも|00⟩を位相0で保つ。

Dの_append_blockは同じruntime_basis_operationsのinverseとforwardでdiagonalを挟む。RTEはbasis runへinverseで入り、切替／identity／終了で同じbasisのforwardを必ず置く。same-basis共有はこの一対の内側を延長するだけで、閉じていないbasisを残さない。D融合も完全なblockを使う。m17の二分は完成したstepの境界であり、両laneはそれぞれ完全である。

よってbasis RZの角は各runで±θの対となり、exact Fractionの和では0である。基底上の行列近似やcompiler後wordのactual adjointを仮定したものではなく、**未compileのgenuine SOURCE gate**の命題である。SOURCE global_phaseを追加するcustom basisや、独立に再合成したinverseには適用しない。

これによりnormal下界の格子に必要な角はD対角／中央RTEだけに縮約できる。providerはXX+YYの第二parameterも含め全parameterを検査するため、basisの受入れmetadataを省けない。旧naiveの途中加算誤差は相殺されるとは言えず、旧Mの相殺を新削減へ二重計上しない。

### 3.2 共通格子の十分条件 — `PROVED_STATIC`な条件付き補題

Λ_ℓ=2^-ℓ Z、ℓは非負整数とする。以下は新しい量子化policyではなく、**既存literalの所属集合の証明**である。

1. finite binary64 xについて、分母が2^kの既約ratioならx∈Λ_k。整数と0はk=0としてよい。exactなθ/2はΛ_(k+1)に入る。全finite binary64一般ではk≤1074にすぎず、half後の1075という格子ではnormal下界は得られない。
2. x,y∈Λ_ℓなら符号反転・exact加減算・整数倍はΛ_ℓを保つ。**RN64も有限結果の範囲でΛ_ℓを保つ**：その大きさのfloat間隔が格子幅以下ならexact格子点自体が表現可能であり、間隔が格子幅より大きければ丸め先の両端点が格子点だからである。
3. x,τ∈Λ_ℓなら数学的剰余x−kτもΛ_ℓに入る。正しく丸められたこの剰余と、負の剰余へのτ加算は2により格子を保つ。[Qiskit numeric setter](https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/crates/circuit/src/circuit_data.rs#L848)はrem_euclid(2*f64::PI)を使う。従来SOURCEが明記するcorrectly-rounded add/sub/remainderモデルの下でこの閉包を使用する。境界0とτを両方含める。
4. D global_phaseの完成済みfloat、完成済みscalar、0/π、周期τがΛ_ℓに入り、必要なD／RTE半角もΛ_ℓに入るなら、既存global_phase更新・compose・copyを逐次帰納してg_F,g_G∈Λ_ℓを得る。eventの種類／順序／反復数は格子を細かくしない。basisを3.1で除くとS_F+S_G∈Λ_ℓである。

2をHamiltonian係数の**積**へ適用してはいけない。積で分母は細かくなり得るので、対象はops.pyが既存順で計算した完成済みprimitive値である。また、単なる剰余の絶対誤差γ上界は3の格子閉包を含意しない。boundedな任意摂動が粗い格子を保つとは限らない。剰余の正しい丸め／同じ格子を保つ実装という意味論を条件として区別する。このSOURCE照合は実wheel・libmの形式検証ではない。

より具体的には、global_phase生成用有限pool P（D phase、scalar、0/π/τ）と、basisを除いたRZ/RZZ angle pool Θについて

\[
\ell=\max\bigl(0,\max_{x\in P}k(x),\max_{\theta\in\Theta}(k(\theta)+1)\bigr)
\]

を十分値として使える。0の半角で1を使っても安全側であり、削減目的の最適化は不要。ℓ≤1022と閉包が確認できれば

\[
S\ne0\ \Longrightarrow\ |S|\ge2^{-\ell}\ge2^{-1022}.
\]

この条件は十分条件であり必要条件ではない。ℓ>1022でも、SOURCEの整数係数制約等で実Sがnormalになる場合はある。その場合今回の粗い証明は不成立だが、RPE失敗や実subnormal反例を意味しない。

### 3.3 固定m17への適用 — `UNQUALIFIED`

実primitive poolのk(x)・ℓを新計算していないため、上の条件を固定m17で満たすとはまだ言えない。g_F/g_Gの実到達値やSのmin/maxも探索していない。_is_effectively_zeroのatol=0はexact-zero pruningであり、_as_realのimaginary tolerance 1e-12から非零角の下限を仮定できない。

一般のbinary64だけからnormal域を示せないことは、例えばRZ角2^-1022、g=0の人工laneの真空位相が−2^-1023となることでも分かる。これは数学的な人工例であり、固定H4/m17のtrajectoryでも新実行結果でもない。**固定caseの `COUNTEREXAMPLE` には分類しない。** Q1のsubnormal拒否テストを再実行する必要もない。

## 4. Q3-S3：上側範囲・M・B_norm

### 4.1 旧Mとoverflow — `CONDITIONAL`

phase_boundのexact_prefix_absolute_upper=Mは、同じSOURCEの各laneのexactな逐次prefixを覆う。D drift、中央r回／step、開いているbasisの最大寄与、local diagonal、正規化gの周期幅を含む。S_FとS_Gは最終prefixなので、それぞれ|S_L|≤M、三角不等式で|S|≤2M。

保存Mの適用条件は、同じfixed cache／Dとbasis catalog、r32／q131072、同じevent文法・max order2・support、D0のexact binary doubling、同じscalar集約・global_phase更新数と算術モデル。これらの定数の[旧導出](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2638a7845685a119619382b850441f5d8d33713/docs/research/pr_reference_bprime_20261107/STRUCTURAL_APPENDIX.md)を再利用する。

保存値M=16375.943495244212は2Mがbinary64 max finiteから十分内側にあることを示す既存certificateで、今回は値の再評価をしない。従って旧certificateと有限primitive生成が適用される限り、最終変換のoverflowは排除できる。ただし最終Sの上界だけで、未検査のprimitive生成途中の積のoverflowまで証明したと扱わない。共通の有限pool監査と旧生成モデルが必要である。

### 4.2 返却値のRNとunderflow — `PROVED_STATIC`／適用は `CONDITIONAL`

前提はPython整数／Fractionのexact算術、binary64 format／隣接値／RN-evenのruntimeモデル。_round_onceはradix2、mantissa53、max_exp1024、double8 byte、rounds1を確認する。|S|>max finiteと0<|S|<min normalを拒否してからfloat(S)を1回行い、隣接floatのexact中点とsignificand偶奇で最近接偶数丸めを検査する。検査失敗なら返却しない。

S=0では誤差0。正常に返却したnormal入力では

\[
|\Phi-S|\le u|S|\le2uM,\quad u=2^{-53}.
\]

Fraction半角は入力が小さくても中間underflowしない。最終非零subnormal exact和のrejectが全supportで起きないことは3.3の未資格事項であり、返却値RN検査の正しさとは別である。runtime条件が破れても誤差を黙って増やす設計ではない。

### 4.3 旧B_normの移送 — `CONDITIONAL`

旧B_normはglobal_phase生成時のfloat加算・float周期と真の2πの差・剰余／正規化とcontrolled SOURCEの相対位相の差を保守的に覆う。phase_boundは20nのlane更新、RTE内部更新、scalarの項を含む。providerは完成したgを読んでからexact加算するだけなので、**これらの上流誤差は残る**。

F/G builder、copy/compose、scalar、signed event、Qiskit setter、controlled側と比較するSOURCEを不変更とする条件で、旧B_normを同じ位置に保持できる。F.inverseのglobal_phase再正規化のうち回路全体に共通なscalarはcoherenceに影響しないが、これを理由にrelative vacuum Pや旧B_norm全体を除去しない。compiled wordのinverseへ置き換える資格ではない。

同じ旧算術モデルと上記条件の下でδ_B1≤B_norm+2uM。古い逐次vacuum加算誤差はこの最終RN項で置換し、B_normや他diskを二重加算／削除しない。builder変更、別のperiod、独立の再合成inverse、別compilerなら、この継承を停止する。Qiskitのnumeric phaseとParameterExpressionの別経路も混同しない。

## 5. Q3-S4：opt-in・coherence・価格

### 5.1 Selectorと実caller — 個別判定

`phase_for_round(..., mode='legacy')`は旧vacuum_phase(F)+vacuum_phase(G)を使う。exact_binary_m17は`type(round_m) is int and round_m == 17`のときだけ選択できる。この選択規則と旧coreへの委譲は `PROVED_STATIC`。

一方、実assembleは依然として旧phase和を返す。新wrapperはmを回路から推定せず、q/r/bitsを検証しない。Q2はq1/q4の局所fixtureでdirect providerを検証したもので、m17本番callerではない。**m17↔q131072/r32/bits88のbinding、m0〜m16のlegacy運用、callerが同じF/Gを渡すことは `UNQUALIFIED` な未実装接続条件。** 今回新callerを実装・使用しない。

必要な最小接続は保存scheduleを参照する一つのcallsiteでm17だけ新modeを選び、その他はlegacyとし、F/G・event列・canonical lawを再生成／変形しないこと。m引数の値だけを全support資格の根拠にしない。

### 5.2 Relative P — `CONDITIONAL`

genuine SOURCE族のvacuum rayとactual SOURCE inverseを前提に

\[
z_{\rm Ref}=e^{i(\Phi-S_F-S_G)}\langle\psi|FG|\psi\rangle.
\]

旧coreはCSWAP cascade、F.inverse()、G、ancilla P、CSWAP cascadeの順。新wrapperはこの順・wire・2 register・X/Y wrapperを保持する。F/Gを読んで生成するPhiだけが変わり、Pは信号のrelative phaseである。[Q2のancilla Z negative witness](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9560e310e9f848fe59a1dafce4b506d5a08aae76/docs/research/pr_reference_v6q_q2_20261109/REPORT.md#3-有限correctness結果)はこの区別の有限根拠として再利用する。再simulationやphase fittingは行わない。

### 5.3 Compiler Aとnative RZ — `CONDITIONAL`

固定SOURCEの真正gate族・parameter形状・ordered wires・gate順・contextが同じ、Pの角だけを変更、追加equivalence／Target／calibrationなし、controlはlower前、同じpass順という条件を置く。

1. 最初のInverseCancellationの固定対象はCX/X/H/Z、S–Sdg、SX–SXdg、T–Tdg。PとRZは対象外。角0でもreference_coreはPを省かない。
2. UnrollCustomDefinitions／BasisTranslatorは固定family/contextの同じsymbolic templateへ束縛する。既存`p_symbolic`はP(φ)→RZ(φ)の1個、理想nativeでは共通joint phaseを除いて解析的に正しい。φを新floatへ変えてもgate数は変えない。
3. 最終InverseCancellationはCXとXだけで、RZを消さない。数値commutation・angle cutoff・1qubit数値合成・反復最適化のpassはない。周囲のP以外のSOURCEも同じならRZのwhole個数が保持される。

これは[既存context／価格資格](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2638a7845685a119619382b850441f5d8d33713/docs/research/pr_reference_bprime_20261107/STRUCTURAL_APPENDIX.md)の再利用条件であり、局所Pの観測だけから任意consumerを資格したものではない。actual m17 callsite/contextの不変更は未実装接続の確認が必要。別contextを要するなら今回はcompileせず、その対象だけを別の限定scopeへ戻す。CX/depth/Tや実機時間への新しい不変主張はしない。

## 6. 現在の証拠境界

Q2の4 fixture PASS、Q1の人工20テスト、旧M/B_norm・symbolic価格資格は保持する。今回は新テストや科学再解析をしていない。normal域、literal pool、callerが揃うまで、V6-Pの626 shots／総RZ利益は条件付きのまま。CT、domain、emitter、shell、terminal、総Tの旧未資格とSTOPも保持する。

残る有限監査は[NEXT_VALIDATION_PLAN](NEXT_VALIDATION_PLAN.md)へ限定した。実normal和の反例を確認していないので`COUNTEREXAMPLE`は付けない。証拠不足を回避するためにsubnormal reject・precision・和アルゴリズム・phase policyを変更しない。
