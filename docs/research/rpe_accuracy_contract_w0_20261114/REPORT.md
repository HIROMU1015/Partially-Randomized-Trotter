# RPE精度評価契約の整理（W0）：decoder定義・unsafe loss・精度資格

2026-11-14 JST。Claude Code（主agentのみ。独立レビューではない）。

- branch：`codex/rpe-accuracy-contract-w0-20261114`。基点は`3a27f99feef2073341dfd0b980ae6b6a23906e9f`（Bの`ce0517de6`とI1/I2/E1技術修正を含む）。
- 内容：既存SOURCE・契約・decoder・risk評価器の静的調査、条件整理、人工検証のみ。

**実施していないこと**：実H4/H6の新しい科学計算、保存risk・MSE値の再集計、DPの実行、MC、decoder・loss・資格規則・Minkowskiの変更、shot最適化、N・価格・Gの評価、Bへの介入。src・testsは無変更。

詳細は次を参照：[数理メモ](MATH_NOTES.md)／[人工検証script](SYNTHETIC_CHECKS.py)・[結果](SYNTHETIC_CHECKS.json)。

**参照資料について**：指示書が挙げる「PRレビュー.md」（11/14）はこのサーバーで見つからなかった。代わりに次を基準にした：一次SOURCE、11/13 Claude統合レビュー§5.1・付録A5/A7/A8、11/14横断監査`51d324ea`のS2（更新幅補題の原案）、A調査`97146481`（直接riskとMinkowskiの接続）。Claudeの提案は採択済み契約として扱っていない。

**根拠の表記**：【実装】固定codeの読取、【既存記録】公開結果の引用、【数学】証明、【人工】今回の人工検証、【提案】未採択の案。

## 0. 要旨

1. **現行の推定量は部分関数である。【実装】** どこかのroundでX=Y=0、または距離差≤1e−12のtieが起きると、energyはNoneになる。そのため全事象のMSEは定義されず、未解決分にloss family `[0,L_U]`（L_U≈1127.77 Ha²）を当てて上界だけを求めている。
2. **現行loss familyには構造的な下限がある。【数学＋既存記録】**
   - どの上界も`U≥P(any zero)·L_U`を満たす。
   - `U≤ε_E²`を示すには`P(any zero)≤ε_E²/L_U≈2.25e−9`が必要。
   - 公開済みの制御ベクトルではP(any zero)≈1.0e−7で、これを満たさない。
   - したがって**Bの状態集約DPの結果にかかわらず、この配分ではモデルMSE≤ε_E²を示せない**。tieも同じ形で効く。
3. **全関数版（zeroは保持、tieはargmin）は、未解決履歴でだけ現行と出力が異なる。【数学＋人工】**
   - 解決済み履歴ではbit一致する。
   - Bの既存上界・下界は、全関数版のMSEに対しても有効なまま使える。
   - DPの状態遷移（Markov性）とも両立する。
4. **更新幅補題`d(θ̂_m,θ̂_(m−1))≤π/2^m`は、現行の選択規則で成り立つ。【数学＋人工】**
   - binary64では1ulp程度の超過を観測した。slackを明示する必要がある。
   - 初回unsafe roundがfのとき、最終誤差は`R_f=ρ/2^(f−1)+π(2^(1−f)−2^(−M))`未満。
   - H4の殻条件では、f=2..12のloss capは`L_U`の0.38〜1.4e−7倍になる（幾何定数のみ）。
   - **この上界が有効なのは全関数版に限る**。現行decoderでは、後で未解決になる履歴にL_Uが残るため改善しない。
5. **直接モデルMSEとMinkowskiは別の量を保証する。【数学】**
   - rawに対する直接riskは、PF・finite-Kの位相ずれを既に含む。δEを加えると二重計上になる。
   - δE（固有値シフト）と有限shot推定量のbiasは同じではない。
   - Minkowski経路では、PRの共通H_effが未確立であること（不存在の証明ではない）が未解決のまま残る。
6. **zero/tie問題は、どちらの資格契約を選んでも解決が必要である。** ε_qpe項もMSEが定義されていることを前提にするため。
   - 推奨【提案】：全関数版を別versionの推定量として定義する。
   - DPに全関数版の遷移と、残余に対する更新幅上界を加える（小差分）。
   - そのうえで、直接モデルMSEを主たる資格の候補として、ユーザーが採否を判断する。

## 1. 現行RPE精度評価契約の整理

### 1.1 decoderの定義【実装】

正本は`pr_rpe_rmse.decode_supplied_coordinates`（A wrapper）と`rpe_four_round_phase_validation.reconstruct_rpe_phase`（scalar）。決定論riskとDPは同値な`rpe_decoder_fast._nearest`（R3、M≤12で一致を検証済み）を使う。

| 項目 | 現行の規則 |
|---|---|
| 1. X/Y countから位相 | `count_mean`：`(2k−N)/N`をexactなFractionで作ってからbinary64にする。X=Re(raw)、Y=Im(raw)。`φ_m=math.atan2(Y,X)`（numpyではなくmath）。principal値は`wrap(φ)=(φ+π)%2π−π` |
| 2. 候補とbranch | m=0は`θ̂_0=wrap(φ_0)`（branchなし）。m≥1は`c_k=wrap((p_m+2πk)/2^m)`。前推定との円周距離`|angle(exp(i(c_k−θ̂)))|`が最小の候補を選ぶ。priorは選択に使わない（`prior_applied_to_decoder=False`） |
| 3. tie | 選択はargmin（完全同値ならlowest branch ID）。wrapperは、どれかのm≥1で`|d_best−d_2nd|≤1e−12`（abs_tol、rel_tol=0）なら`UNRESOLVED_BRANCH_TIE`にする。tieかどうかは前状態に依存する。旧vectorized経路はargsortが不安定で、別規則 |
| 4. X=Y=0 | どれかのroundで両座標が0.0なら、復号前に`UNRESOLVED_ZERO_COORDINATES`（tieより優先）。両軸のNが偶数の場合だけ起こり得る |
| 5. binary64 | count→float→`math.atan2`の順序を固定。count_meanは+0.0だけを返す。wrapは端点で`+π_f`を返し得る【人工】。DPは主値`[−π,π)`外をUNKNOWNとする。M≤12・q≤2^52の範囲だけを検証済み |
| 6. 戻り値とstatus | `DECODED_SUPPLIED_COORDINATES_NOT_RMSE_QUALIFIED`（`Ê=−θ̂_M/h`、`rmse_guaranteed=False`）／上記2種のUNRESOLVED（energy=None）／`NOT_EVALUATED_MISSING_COUNTS`。範囲外・非有限・型不一致はValueError |
| 7. 定義域 | `D_O`＝zeroがなく、実際の復号経路で全tie gap>1e−12となるcount集合。経路依存なので直積集合ではない。値域は`[−π_f/h,π_f/h]` |
| 8. lossとtarget | `E*`=保存solver proxy `−2.1663874486347634 Ha`（`SAVED_FULL_DF_H_SOLVER_CONDITIONAL_PROXY`、ground未認証）。loss=`(Ê−E*)²`をexact rationalで計算。未解決は`BOUNDED_INTERVAL [0,L_U]`、`L_U=max((b−E*)²,(−b−E*)²)`、`b=nextafter(π/h)`。scopeは`RAW_MODEL_DIAGNOSTIC_LOSS_FAMILY_ONLY` |

### 1.2 事象の区別【実装＋数学】

| 事象 | energy | 現行coarse上界でのloss | 状態集約DPでのloss |
|---|---|---|---|
| E1 全round good（安全集合内） | 有限。閉包証明によりoracle値`c_M` | 実loss（J_M項） | 実loss |
| E2 unsafeだが解決（正branch・誤branchとも） | 有限 | L_bad（一律に最大） | 実loss（保持した状態について） |
| E3 zeroによる未解決 | None | L_bad | `[0,L_U]`（吸収） |
| E4 tieによる未解決 | None | L_bad | `[0,L_U]`（吸収） |
| E5 数値的に不適格 | — | ValueError / UNKNOWN（MSE=null） | 同左（質量不保存、非有限遷移、範囲外など） |

- unsafe・誤branch・未解決は別の概念である。
- 包含関係は`P(any zero)≤P(未解決)≤P(any unsafe)`。good遷移ではtieが起きないことが閉包証明で保証されるため。
- reference-oracle tie（targetから見たtie）は、実際の経路でのtieと同じではない。

### 1.3 target・raw・MSE・誤差予算【実装＋既存記録】

| 項目 | 現状 |
|---|---|
| raw | 保存されたideal finite-K Hadamard平均。PF headの効果、finite-K補正、減衰Bを含む。`p=(1+raw)/2`をexactに扱い、IID二項分布とする（`EXACT_AFFINE_OF_BINARY64_RAW`） |
| 直接評価している量 | `E_model[(Ê−E*)²]`の上界（coarse／DP、exact rational、MCなし）。claimは`RAW_MODEL_DIAGNOSTIC_ONLY` |
| 予算 | `ε_E=1.59360010199040e−3 Ha`、`ε_qpe=ε_trot=ε_E/2`、`error_rule=minkowski`、`finite_B_diagnostic`。canonical configの`rmse_policy.scientific_status=PROPOSED_NOT_ADOPTED`で、目標値であり達成証拠ではない |
| 予算の現在の使われ方 | `pr_rpe_rmse.evaluate`の適用性検査（`qpe+evo≤target`、`Cmax·π/T_max≤ε_qpe`）と、論文経験式のnominal N（`ceil(B²(11+4(M−m)))`）だけ。十分Nの資格ではない |
| 公開済み結果 | LD1/LD2のcoarse上界は約0.2399 Ha²。このうちall-good項は約4.71e−7 Ha²で、残りのほぼ全部がP(any unsafe)×L_U【既存記録】 |

## 2. zero/tie事象への対応案

| 案 | 数学的定義 | m=0 | 後続round | 現行との出力差 | DPとの互換 | 既存結果への影響 |
|---|---|---|---|---|---|---|
| Z0 現状維持＋family | 未解決にloss `[lo,hi]` | — | 吸収 | なし | 現行DP | なし。ただし§0-2の構造的下限が残る |
| Z0' 配分の制約：各roundで一方の軸のNを奇数にする | zeroは確率0（exact）。decoderは不変 | — | — | なし | 現行DP | 新しい配分なので新しい評価が必要。tieは残る。Bの[FINITE_PLAN](../rpe_mse_state_refinement_20261114/FINITE_PLAN.md)の偶奇対照と同じ考え方 |
| **Z1 保持** | zeroなら`θ̂_m=θ̂_(m−1)` | `θ̂_0=+0.0`（priorに依存しない）。prior中心を使う案もあるが、prior非適用の原則を変える | 通常どおり。保持した直後のroundは誤branchが増え得る（MSEはDPでexactに評価可能） | zero履歴だけ | 状態は不変で遷移。Markov性を保つ | 既存上界はTに対しても有効（命題2） |
| Z2 φ=atan2(+0,+0)=0とみなす | zero判定を外す | 同上 | 架空の位相0の候補へ動く（最大π/2^m） | zero履歴だけ | 可 | 同上。架空の観測を入れるので推奨しない |
| **T1 argmin採用** | 許容差内でも計算上の最小を採る（完全同値はlowest ID） | — | 選ばれた値から継続 | tie履歴だけ | DPは既に`_nearest`の値を計算済み | 同上。1e−12付近の判定はlibm・numpyのulp差に依存し得る |
| T2 明示規則 | 許容差内では、例えば反時計回り側を選ぶ | — | 同上 | tie履歴だけ | 可 | 環境差に頑健だが実装が増える |
| T3 tieで保持 | `θ̂_m=θ̂_(m−1)` | — | 同上 | tie履歴だけ | 可 | 精度向上をそのroundだけ見送る |
| Z3 未解決に点lossを置く | `l=u` | — | — | statusは同じ | 現行DP | MSEは定まるが推定量ではなく、lossの約束事にすぎない。資格には推奨しない |

**成立条件【数学】**：Z1/Z2/T1/T2/T3はいずれも更新幅`≤π/2^m`（+slack）を満たすので、§3の補題が使える。

**現行との差【数学＋人工】**：差は未解決履歴だけに出る（命題1）。人工8例・7,136のDECODED履歴でbit一致を確認した。

**既存結果の保全**：
- 新しい推定量は別ID（例`consecutive_dyadic_nearest_branch_principal_energy_total_v1`）と別statusにする。例えば`DECODED_TOTAL`に、zeroを保持したround・tieを採ったroundのflagを付ける。
- 元のdecoder・status・Bのrun01は変更しない。
- Bの上界・下界は全関数版に対しても有効な（緩い）範囲として残る（命題2）。

**推奨【提案】**：Z1＋T1を最小案とする（m=0のzeroは+0.0）。Z0'は「decoderを変えない対照」として併記できる。

## 3. unsafe loss上界の改善可能性

1. **安全集合**：ρ_m=1.0 rad（全m）、閉包証明は`VERIFIED_FINITE_BINARY64_SUPPORT`（公開済み2件）。初回unsafeの確率`p_f`は、既存`_bound`の`first_unsafe`がexactに出している【実装】。
2. **DPが保持する情報**：解決中のbinary64状態（float.hex）と整数質量、吸収した未解決質量（zero/tieは区別しない）、残余のscalar質量。初回unsafeのroundも、残余を落とした時点の状態も保持していない【実装】。
3. **更新幅と初回unsafeによる上界**：[MATH_NOTES N3/N4](MATH_NOTES.md)。
   - 主値guardは`|θ*|+R_f<π`（H4ではf≥2で成立）。f=0は前推定がない、f=1は`R_1≥π`なので、どちらもL_Dのまま。
   - 円周距離と直線上のenergy誤差はguardの下で一致する。guardを外すと過小評価になることを、人工の負の対照で確認した。
4. **zero/tieの影響**：補題はZ1/T1でも成り立つ。現行decoderでは未解決にenergyがないので、補題を当てられない。人工の完全DPでは、8例中5例で補題の上界がfamily上端を下回った。つまり現行decoderに置き換えるのは不正。
5. **binary64**：補題の超過は最大1.29e−15 rad【人工】。slack τを明示する。DPでは`best≤π/q+τ`を実行時にassertして確認することもできる。

**H4殻条件での幾何loss cap**（h=0.1、ρ=1、M=12、θ*≈0.2166から計算した幾何定数。確率は掛けていない）：

| f | 0 | 1 | 2 | 3 | 4 | 6 | 8 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|
| R_f/h（Ha） | — | — | 20.70 | 10.35 | 5.169 | 1.287 | .3159 | .07322 | .03278 | .01255 |
| ℓ_f/L_U | 1 | 1 | .38 | .095 | .024 | 1.5e−3 | 8.9e−5 | 4.8e−6 | 9.5e−7 | 1.4e−7 |

この表を保存済みの`first_unsafe`と掛け合わせて新しい上界を作る計算は、保存値の再解析にあたるため**行っていない**。

**DPとの接続条件**（[N5](MATH_NOTES.md)）：
- 全関数版のDPでは、残余を落とした時点の状態sで`ℓ^res(s,m)`を使って囲む。
- 互いに素な分割を守る。coarse（F分割）とDP（状態分割）の項は足さず、完成した上界どうしの`min`だけを取る。
- 未解決確率の下端は上界に使わない。閉包証明が不成立なら補題上界を使わない。

**現行最大lossを残す事象**：
- f=0、f=1。
- guardが不成立のf。
- 現行decoderでの未解決分（契約の`hi`）と残余（将来の未解決確率の上界がない限り）。
- 数値UNKNOWNは上界を出さない。

## 4. 精度資格の選択肢

### 4.1 何を保証するか【数学＋実装】

| 量 | 意味 |
|---|---|
| Minkowski（現行の目標） | `‖Ê−E*‖₂≤‖Ê−Z0‖₂+‖Z0−E*‖₂`（またはE_effを基準）に分け、各項≤ε_E/2とする。各項の資格には、基準Z0またはE_effの定義と、couplingまたは定数基準が要る（A97146481 §5.2） |
| 現行の決定論risk | rawのIIDモデルの下で、**E*に対する総riskの上界**。PF・finite-K・state混合による位相のずれを含む。ε_qpe単独でもε_trot単独でもない |
| rawが含まないもの | target proxyの誤差（DF10 Hの基底か、DF前の分子Hか）、rawの数値allowance、実samplerのlaw、IIDからの逸脱 |
| δEとbias | δE=E_eff−E*はunitary stepのスペクトルの性質で、PFでは得られている。biasは`E[Ê]−E*`で、δE、信号のずれ（混合・finite-K）、有限shotのdecoder bias（誤branch・atan2・打切り・zero/tie規則）を含み、Nに依存する（[N7](MATH_NOTES.md)）。両者を同一視しない |

**二重計上の規則**：
- E*に対するrawの直接riskに|δE|・δE²を加えない。
- E_effに対するriskとして求めた場合にだけ、|δE|を加える。
- `paper_quadrature`は交差項が0であることを仮定しており、有限shotでは保証されない。

### 4.2 三つの選択肢

| | Q1 Minkowskiを維持 | Q2 直接モデルMSEで資格判定 | Q3 Q2を主、b²＋Varは設計用、Minkowskiは診断 |
|---|---|---|---|
| 判定式 | `‖Ê−E_eff‖₂≤ε_qpe`かつ`|δE|≤ε_trot`（または結合した基準Z0） | `U_model(Ê;E*)≤ε_E²`。移送項を加えるならA97146481の`sqrt(R⁺+UΔ)+d_target≤ε_E` | Q2で合否を判定。設計はDPの`E[Ê]`・`Var`で行う。PFのH_effは分解の診断に使う |
| PRへの適用 | **共通H_effが未確立**（corrected stepは非unitaryでR依存）のため、ε_trot項と基準の定義が未解決 | H_effは不要。rawにすべて含まれる | 同左 |
| zero/tie | ε_qpe項のMSEに全関数化が必要 | 全関数化が必要（または点loss） | 同左 |
| 保守性 | ε_qpe=ε_E/2のため、終端roundのNは約4倍（11/13レビュー§5.1の桁評価） | 予算を自動的に共有する | 同左 |
| 新たな仮定 | Z0・coupling・E_effの分枝の資格 | モデル（IID、rawの正確さ）に条件付いた主張。transfer項を別に扱う | 同左＋設計値は資格ではない |
| 比較可能性 | 現行計画と連続 | 全方式（PF・PR・Full-H RTE）を同じ契約で評価し直す必要がある | 同左 |

**直接モデルMSEで資格判定するための必要条件**：
1. 全関数の推定量（または点loss）。
2. targetの採択（proxyと狭い包含区間）。
3. lawの範囲（IID、ideal finite-K）の明示。実samplerへの移送は別項。
4. rawの数値誤差の扱い（認証されたηがあれば`Δ≤ΣNη/2`。経験的なuは代用しない）。
5. exactな決定論上界（DP、coarse補題、そのmin）。
6. 全方式に同じdecoder・loss・targetを適用すること。

**b²＋分散を設計に使う条件**：
- 全関数版のDPで1次・2次モーメントをexactに得る（r=0）か、区間で囲む。
- `b(N)`はNに依存するので、δEで代用しない。
- 漸近biasは`E_sig−E*`（rawを雑音なしで復号した値）で見積もれる。
- 設計値は資格ではなく、最終判定は上界で行う。

**既存結果との比較可能性**：
- Bの上界・下界は全関数版に対しても有効（命題2）。
- 旧S3の解決条件付きRMSE（標本値）は、全事象MSEとは比較できない。
- nominal NはMinkowskiの目標から導いたもので、Q2では同じ意味を持たない。

## 5. 既存コードへの最小変更案（採択後。今回は実装しない）

**再利用するもの**：
- `count_mean`、`binomial_axis`、`round_distribution`、`verify_good_transitions`
- `_bound`のfirst_unsafe・J_M・L_D、`_refine_states`の分割・質量保存
- `rpe_decoder_fast._nearest`/`reconstruct_batch`
- `_sqrt_display_upper`、`D.compare`の`MODEL_MSE_FILTER_ONLY`

| 変更 | 内容（目安の規模） |
|---|---|
| decoder | `rpe_decoder_fast.decode_coordinates_batch_total(coords,*,h,zero_rule='HOLD',tie_rule='ARGMIN_LOWEST_ID')`を新設する。元関数は不変（約30行） |
| risk options | `options['estimator']∈{'ORIGINAL_PARTIAL'(既定),'TOTAL_HOLD_ARGMIN_V1'}`。`deterministic_input`に記録され、既存のN packet照合にそのまま結び付く |
| coarse | 全関数版のときだけ`U_lemma`を計算し、`min(coarse,U_lemma)`。guard・slack・πの上側有理数を使う（約20行） |
| DP | 全関数版ではzero/tieを吸収せず遷移させる。残余は`Σw·ℓ^res(s,m)`で囲む。`E[Ê]`を累積してbias・Varを出す（約25行） |
| identity | canonical configの`error_contract.decoder`を変えると、configuration・候補IDが変わり、bundleや価格の結合が切れる。診断段階ではevaluationのoptionとして扱い、正式なN packet（`B.assess`のerror_contract一致検査）では新しいconfig version＋bridgeが必要。**採否はユーザー** |
| 任意（現行decoderを残す場合） | 残余から将来未解決になる確率の上界（zero分はexact、tie分はsup_s）。優先度は低い |

**最小の人工テスト**：
1. 全関数版は解決済み履歴で元とbit一致（zero/tieの多いケースを含む全数列挙）。
2. 全遷移で更新幅≤π/2^m+τ（大きなq、構成したtie中点）。
3. 全関数版DPの完全値＝全数列挙のMSE・E[Ê]・Varで、質量保存。
4. Bの旧上界・DP下界でMSE_totalを挟めること。
5. `U_lemma`≥MSE_total（各点・期待値）、guardを外すと破れる負の対照、f≤1でL_D。
6. 小さな状態上限での残余上界の有効性。
7. 既定経路の既存57件が無回帰。
8. 奇数Nでzero=0、m=0のzero、±π・signed zero。

今回の[SYNTHETIC_CHECKS.py](SYNTHETIC_CHECKS.py)は1・2・4・5・8を、検証用oracle（src外）で先行確認したもの（59/59 PASS）。

## 6. ユーザーが正式に判断すべき事項

| # | 判断 | 利点 | 新たな仮定・代償 | 既存結果の保全 | 必要な追加科学実行 |
|---|---|---|---|---|---|
| D1 | 推定量：Z0維持／Z0'（奇数N）／**Z1＋T1を別versionで追加** | MSEが定義され、§0-2の下限が消え、補題が使える | 新しい推定量の採択。m=0の規則。near-tieの環境依存 | 別ID・別出力。Bの上界は有効なまま | なし（定義の採否のみ） |
| D2 | 上界の方法：補題＋全関数版DP（残余の状態上界・モーメント）の実装可否 | O(M)の上界と、DPのexact値・bias・Var | binary64のslack（認証ではない）、guard | 既定経路は不変 | 実装後、同じLD1/LD2 2ベクトルでの評価（新しい科学計算） |
| D3 | 資格契約：Q1／Q2／Q3 | Q2/Q3は共通H_effなしにPRを評価できる | モデル条件付きの主張。全方式の再評価 | Minkowski結果・nominalは履歴として保持 | 全方式を同契約で評価（範囲は別途指定） |
| D4 | transfer項：target包含・raw η・lawのΔを主張に含めるか | 物理的な主張に近づく | 証拠の取得が必要（A97146481） | — | Aのtarget包含などは別作業 |
| D5 | 順序：Bの実行中DPの結果を待ってからD1/D2を決めるか | 実際のtie質量と、残余の支配度が分かる | — | — | Bの承認済み範囲内 |

## 7. Bの状態集約DPとの関係

**現行DPで既に評価できる範囲**：現行decoder＋familyの下で、
- `η`（保持した解決状態の実loss）
- `u_0`（吸収した未解決）、`r`（残余）、P(any zero)、tieの範囲
- 上界`η+u_0L_U+rL_bad`と下界`η`

r=0ならfamilyの両端がexactに求まる。

**現行DPでは解消できない問題**：
- `u_0·L_U≥P(any zero)·L_U`の下限（§0-2）。
- 未解決にenergyがないこと。
- 残余を状態なしのscalar質量でしか持たないこと。

**全関数版で変わる入出力**：
- 入力：`options.estimator`。
- 出力：吸収未解決の代わりに、zeroを保持した確率・tieを採った確率。r=0ならMSE・E[Ê]・Varのexact値。残余は状態別の上界。

**Bの結果を保持したまま比較する方法**：
- 同じ配分について、旧（B run01）と新（全関数版）を別ID・別出力で並べる。
- 新しいexact MSEがBの`[η,U]`に入ることを、相互整合の検査として使える。

**観測**：作業時、Bの`run_h4_dp_risk_fixed.py execute`（2 worker）は実行中だった。process・worktree・branch・結果には触れていない。結果の先取りもしていない（§0-2は、どのDP上界も満たす不等式として述べたもの）。Bの結果次第で不要になり得る実装（D2）は行っていない。

## 8. 検証と限界

**人工検証**：[SYNTHETIC_CHECKS.json](SYNTHETIC_CHECKS.json)、59/59 PASS、約11秒。
- 環境：`h6-environments/20261011/venv`の既存python（3.11.0rc1、numpy 1.26.4）。
- 入力：Bのtest fixture（h=10、人工loss[0,20]）。
- 方法：既存の`D.evaluate`・`round_distribution`・`decode_counts`を使い、8例・11,748履歴を全数列挙した。大きなqは各round 220,000遷移。
- 保存raw・target・結果の読取はmockで禁止。MCは0。

**限界**：
- binary64の結論は工学的なslackによるもので、区間演算による認証ではない。
- H4の幾何表は確率を含まない。
- 全関数版のH4でのMSE・改善幅は未評価（新しい科学計算が必要）。
- target・law・数値の移送は未取得のまま。

**保全**：固定SOURCE・input・raw・結果・grant・ledger・marker・KEEP・STOP、Bの作業、共有環境は変更していない。新設したのはこのdirectoryと索引の各1行だけ。

**mandatory STOP**：契約の変更・新しい科学計算・Bへの介入は行わない。次はユーザーがD1〜D5を判断し、必要な範囲を明示的に許可した場合だけ進める。
