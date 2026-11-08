# 新A V6：Native vacuum位相保証の静的成立性調査

2026-11-08、SERVER 2。基準commitは **f2638a7845685a119619382b850441f5d8d33713**、関連設計は **c9d5b55729064882afaa9ae5d816283902f1f53c**。今回の成果はSOURCE読解、既存証拠の照合、条件付き数理整理と次の検証案である。**科学数値計算・保存値の新規再解析・phase計算器実装・回路生成／compile・shot再計算は実施していない。**

[候補の前提と証明義務](PHASE_IMPROVEMENT_OPTIONS.md)／[次の限定検証案](NEXT_VALIDATION_PLAN.md)。以下の新しい不等式・候補評価はCodexによる静的整理であり、既存結果や独立科学レビューによる採択とは区別する。

## 1. 結論と研究上の位置

1. **現行SOURCEを変えない保証改善の余地は残るが、鋭い全support上界は未完成。** 現行の一律prefix最大値と減算数上界を、実順序に沿うprefix／block別上界に置き換える候補がある。ただし現行証明もbasis／inverseの数学的相殺とD反復構造を既に使っている。同じ相殺を追加利益へ計上できず、実際の丸め残差の取消は証明されていない。
2. **算術変更では、同じbinary入力を正確に集約し、F/G合計を最後に一度だけbinary64へ丸める案の証明条件が明確。** 既存normalization項を残し、逐次和の項だけを一回の丸め上界へ交換する条件付き構成である。`math.fsum`は変更量の小さい別候補だが、runtime条件と二つの和の最終加算を別に扱う。いずれも未実装・未採択であり、SOURCE変更後の成功資格を今回認定しない。
3. 次の科学診断は、**m17の保存上界だけを使う条件付き上界・整数ceil・総native RZの判別**を最小案とする。新位相方式の実装に先立ち、仮に一回丸めの保証を実現しても整数Nと総RZに利益が出る余地があるかを判別する。現行SOURCEの改良証明が先に閉じれば、それを別armとして同じ比較へ投入できるが、未完成の証明を数値評価へ持ち込まない。詳細・call範囲・STOPは次計画に固定した。**今回、その診断は実行していない。**

これは既知の数値和技術を既存PR＋RPE保証へ接続する設計調査である。新規性、RPE成功、shot削減、native RZ削減、総T削減を認定しない。

## 2. D0が示した対象

以下は[固定D0 REPORT](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2638a7845685a119619382b850441f5d8d33713/docs/research/pr_reference_cost_structure_d0_20261108/REPORT.md)と[保存RESULTS.json.xz](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2638a7845685a119619382b850441f5d8d33713/artifacts/pr_reference_cost_structure/2026-11-08/D0_run01/RESULTS.json.xz)の既存結果の引用である。今回、分解や順位を再計算していない。

| 保存結果 | 意味 |
|---|---|
| B′基準の単価差寄与 約−103.61億RZ、追加shot寄与 約＋157.40億RZ、直接総差 約＋53.79億RZ | 保存十分shot条件ではReferenceの単価節約を実行回数増が上回る |
| m0〜m16でN一致、m17のみB′626／Reference630 shots/axis | 位相保証の候補を調べる第一対象はm17。ただし両値とも必要最小shot数ではない |
| 共通RTE basis/frame/inverse単価に、追加shotから約＋92.39億RZ | Nの改善はvacuum Pだけでなく全roleの繰返し費用へ作用する |
| 前17roundの差はReference有利、m17の差はReference不利 | 長roundの保証と整数ceilが全18round比較に効く。新しいNを導く証拠ではない |

資源の主指標は同じ成功条件下の全18round総native RZ。CX・native DAG depth・追加qubitは別々に保持する。回路や単価が同じでもNが減ればgate実行総数とdepth実行量は変わり得るが、1-shot depthやqubit数はそれだけでは減らない。総Clifford+T T-countは別の未資格目標である。

## 3. 固定条件と読解したSOURCE

Linear H4／1.00 Å／STO-3G／DF12、元H厳密ground供給、状態準備費除外、Partial S2／LD3／h=0.02、finite RTE K2、ordinary `diag(I,V)`、canonical independent finite-bit lawを保持する。Qiskit 1.3.0／compiler A／`rz,sx,x,cx`、B′の既存4-CX controlled-RZZ、ReferenceのF/G・actual SOURCE inverse・vacuum P・2 register・CSWAP、D融合・basis共有・scalar/event相対位相・X/Y wrapperを変更しない。保存18roundのq/r/bits、精度1.5936e-4 Ha、失敗確率0.05、位相cap 0.4、各axis α=1/720、保存Nも不変。E0・gap・groundの古典記述を補正位相に追加使用しない。

SOURCEリンクはすべて基準commitへ固定する。

| SOURCE／既存資産 | 静的に確認した役割と再利用範囲 |
|---|---|
| [pilot: vacuum_phase / reference_core / wrapper](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2638a7845685a119619382b850441f5d8d33713/src/trottertracks/pr_reference_interference_pilot.py) | global_phaseからRZ/RZZ半角をSOURCE順に逐次減算。F.inverse()・G・ancilla P・二つのCSWAP cascade、X/Y測定の符号規約を保持 |
| [transfer: assemble](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2638a7845685a119619382b850441f5d8d33713/src/trottertracks/pr_reference_transfer_qualification.py) | 同じ保存eventsからcontrolled／uncontrolled tailを作り、q=1の特別分割、q≥2の等分割、D融合、F/Gの実際の時間順序を定義 |
| [phase_bound / qualify](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2638a7845685a119619382b850441f5d8d33713/src/trottertracks/pr_reference_transfer_error.py) | 逐次和とnormalizationを別に上界化し、native joint誤差とshared diskへ一度ずつ合算。importやqualify呼出しは今回行わない |
| [B′ error_and_shots](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2638a7845685a119619382b850441f5d8d33713/src/trottertracks/pr_reference_bprime_accounting.py) | 旧compiler diskの置換、B′ kernelの解析資格、外向き端点の保存後に整数Nを算出。全runner再実行ではなく、将来は純粋な算術部分だけ再利用する |
| [finite-RTE build_sequence](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2638a7845685a119619382b850441f5d8d33713/src/trotterlib/df_rte_qiskit.py) | event・signed identity・central phaseを集約。controlledではrelative P、uncontrolledではglobal_phaseへ渡すため、その上流丸め／正規化の差はvacuum和の改善だけでは消えない |
| [repeated Partial S2](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2638a7845685a119619382b850441f5d8d33713/src/trotterlib/df_partial_s2_repeated.py) | 物理scalar、boundary fusion、same-basis sharing。`math.fsum`が現れるのは返却metadataのrte_relative_phaseであり、実回路global_phaseやvacuum_phaseの和を置き換えていない |
| [B′ STRUCTURAL_APPENDIX](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2638a7845685a119619382b850441f5d8d33713/docs/research/pr_reference_bprime_20261107/STRUCTURAL_APPENDIX.md) | 336・49・24・20のSOURCE由来、binary scaling、Qiskit context、symbolic P→RZ、既存q1/q4 witnessを再利用。有限witnessを全support証明の代用にしない |

適用AGENTS.md、PROJECT_MAP.mdと正式共通運用指示2026-10-25版を確認した。旧共通資料と固定研究契約に従い、別worktreeの3文書だけを追加する。AI subagentは使わず、他Track・他利用者・共有設定に変更を加えない。

## 4. 七つの位相要素を分離する

| 要素 | 固定SOURCEでの意味／V6で残す課題 |
|---|---|
| 数学的vacuum位相 | 実際のbinary gate literalを実数とみなすと、RZ/RZZは−θ/2。XX+YY、Z、P、idはvacuum rayを保ち、このSOURCEのvacuum作用で加算不要。未対応gateはrejectを保持 |
| global_phase生成 | 物理scalar、D、event、central等の寄与を上流でbinary64集約・composeする。保存global_phaseはその処理後の値で、理想Hamiltonianの解析位相と同一視しない |
| 逐次減算 | `phase -= float(theta)/2`の順序を保持する限り、各更新の丸めが残る。半角のexact binary scalingと、和のexactnessは別 |
| 正負／inverse相殺 | basisと逆の角度和の数学的相殺はprefix上界に既使用。離れた減算の丸め残差は自動相殺しない |
| normalization | Qiskitの数値global_phase setterはfloat τを用いる。τと真の2πの差、正規化演算、controlled SOURCEとの集約差を旧別項が囲む |
| controlled相対位相 | uncontrolled側globalとcontrolled側Pの差は干渉信号に現れ得る。平均channel一致やcommon joint scalar不変性で除去できない |
| 最終vacuum P | 二つのvacuum和をbinary64で足し、ancilla relative Pに渡す。片側の和だけ改善しても最終加算、上流normalization、native loweringの扱いが必要 |

Qiskitの数値setterにおけるfloat周期の剰余は[公式pinned Rust SOURCE](https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/crates/circuit/src/circuit_data.rs#L848)でも限定照合した。既存runtime監査の対象commitと同じである。このSOURCE確認は共有runtimeの変更やwheel全体の形式証明を意味しない。symbolic pathも新たに資格化していない。

理想的にF|0⟩=e^{iφF}|0⟩、G|0⟩=e^{iφG}|0⟩で、SOURCE actual inverseの関係を維持する場合、既存X/Y規約のcoherenceは

\[
z_{\mathrm{Ref}}=e^{i(\Phi-\phi_F-\phi_G)}\langle\psi|FG|\psi\rangle.
\]

従ってΦの誤差は干渉の相対位相誤差である。F/Gを交換したり、SOURCE inverseをcompiled wordの独立近似に置き換えたりしてよいという式ではない。unconditional joint全体に共通なscalarは測定不変だが、このancilla Pの誤差はそれに当たらない。

## 5. 現行上界の何を改善するのか

旧SOURCEは片側の減算数L、exact prefixの絶対上界Mについてγ_L Mを使い、二つのlaneと最後の加算を課金する。γ_j=ju/(1−ju)、u=2⁻⁵³。L=(336+49r)n+24、n=max(1,q/2)である。MにはD反復のnet drift、RTE最大回転、basis往復の途中振幅、global phaseの範囲が入る。basis／inverseを全回数の絶対和で足す証明ではない。

[STAGE3_ALL_SUPPORT](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2638a7845685a119619382b850441f5d8d33713/artifacts/pr_reference_transfer_qualification/2026-11-06/STAGE3_ALL_SUPPORT.json)のm17に保存された上界は次の通り。新計算ではない。

| 保存フィールド | 値 |
|---|---:|
| naive_accumulation_upper | 0.00045372585145842106 |
| normalization_vs_controlled_source_upper | 5.172131038537467e-08 |
| phase_error_upper | 0.00045377757276880644 |

この保存結果では逐次和の上界が大きい。これは実際の誤差測定値でも、丸め誤差がこの量に達した証拠でもない。normalizationを0にしたり、phase全体をjoint scalarとして消したりしない。

SOURCE不変更案は実prefix順の細分化、exactな更新の証明、反復blockの安全な集約である。新しい全support証明が必要で、同じ数学的相殺の重複除外、有限trajectoryの誤差観測、worst-case符号の期待的相殺では採択できない。算術変更案は[候補文書](PHASE_IMPROVEMENT_OPTIONS.md)に分離した。

## 6. Raw diskから整数Nへの接続

旧誤差規約のまま、位相以外のshared SOURCE・finite-RTE・event・finite-bit proposalとnative joint誤差をD_otherと置き、δをvacuum位相の上界とする。

\[
D=D_{\rm other}+\delta,\quad
\rho=\rho_0-D,\quad
\beta=0.4-\phi_{\rm PF}-\arcsin(D/\rho_0),\qquad
N=\left\lceil\frac{4\log(1440)}{\rho^2\sin^2\beta}\right\rceil.
\]

δはradの位相誤差を|e^{iδ}−1|≤|δ|でraw diskへ入れる旧規約。native operator誤差は既存の2√2変換後に足す。旧compiler diskは置換済みで、二重加算しない。0≤D<ρ0、ρ>0、0<β<π/2が必要である。実際のshot判定では旧SOURCEどおりρ、βの外向き端点を保存した後、192-bit directed intervalでceilを取る。

δの上界が下がればこの領域のρとβは悪化しない。しかしNは整数である。整数kについてρ²sin²β≥4log(1440)/kを保証できる境界を越えなければ、保存630からの減少はない。今回、k・必要δ・新Nを計算していない。B′626を代入せず、626到達だけを成功基準にしない。位相以外のdiskは残る。

仮に同じ単価のままNが下がれば、各axisの全roleにそのNを掛け直す。RTE basisも対象である。全18roundの比較は他17roundを保存値で固定し、m17だけを更新できるが、これには別途科学実行許可が必要。総RZが依然不利なら、shotだけ改善しても全体改善とは呼ばない。CX・depth・qubitや総TへRZの利益を無条件に移さない。

B′にも共通するscalar生成・normalization等まで変える候補なら、同じ入力情報・保証改善機会を両armへ与える。Reference固有のvacuum和だけを交換する候補ではB′回路は変わらないが、その範囲を明記し、B′誤差やNを不利に固定する共通改善を隠さない。

## 7. 重複確認と未資格事項

既存[横断整理](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/654b8501c1348cd0694daeefd19391df6b928354/docs/research/pr_rpe_cost_refocus_20261108/REPORT.md)と[比較表](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/654b8501c1348cd0694daeefd19391df6b928354/docs/research/pr_rpe_cost_refocus_20261108/COST_IMPROVEMENT_MATRIX.md)を再利用した。全branchや他利用者の作業を横断収集していない。

| 既存項目 | 今回確認できること／境界 |
|---|---|
| [B′時点のV6レビュー案](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2638a7845685a119619382b850441f5d8d33713/docs/research/pr_reference_bprime_20261107/CLAUDE_REVIEW_RECEIVED.md) | fsum／閉形式位相の提案は既存。そこでのshot改善の見通しは採択結果ではない。[当時REPORT](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2638a7845685a119619382b850441f5d8d33713/docs/research/pr_reference_bprime_20261107/REPORT.md)も方式変更を未実施とする |
| 既存vacuum／sign／scalar／inverse correctness | pilotとtransferの検証を再利用。新しい加算法のall-support保証にはならないため、今回再実行しない |
| repeated builder中のfsum | metadataだけ。実vacuum providerは依然逐次減算であり、同じ改善がSOURCE採用済みとは扱わない |
| [N5-B1 joint scalar因子分離](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2638a7845685a119619382b850441f5d8d33713/docs/research/pr_reference_n5b1_scalar_20261108/JOINT_SCALAR_FACTORIZATION.md) | CT wordをjoint適用した後の共通scalarの不変性。SOURCE scalar/event/vacuum relative Pの誤差を消す結果ではない |
| [emitter SOURCE監査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2638a7845685a119619382b850441f5d8d33713/docs/research/pr_reference_n3b1a_emitter_static_20261108/EMITTER_SOURCE_AUDIT.md)・[runtime評価](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2638a7845685a119619382b850441f5d8d33713/docs/research/pr_reference_n3b1b0_runtime_20261108/EMITTER_BOUND_ASSESSMENT.md) | numeric global_phaseの剰余、copy/inverse等の既存情報は再利用可能。CTの特定word検証をnative vacuum和の新保証へ移さない |
| fixed evidence外での同じ案の実装／負結果 | **UNKNOWN**。一般的な未実施・新規性・他Trackの進捗を断定しない |

m17 strict CT、D3 inverse defect、basis全体のactual-adjoint化、新composite guard、新合成、43-key runtime全面資格、新PF/law/H、全18round RPE最適化、総T評価、他Track追加実験は引き続き保留。V0/V1・S1/S2等の負結果、domain・central・emitter・shell・terminalの未資格は変更しない。

## 8. 完了・STOP

変更はこのV6用の3文書のみ。元SOURCE・raw・結果・grant・ledger・marker・STOPと科学契約を変更していない。科学runner／科学テスト・新しい数値phase検証は実行せず、文書の差分・参照・公開内容だけを技術確認する。新しい管理基盤や承認証明は作らない。

**commit/push・remote SHAと3文書の内容一致を確認後、mandatory STOP。** 次担当はGPT／ユーザー：SOURCE不変更案の追加証明を先行させるか、次計画の条件付き数値診断を許可するかを判断する。Codexによる次の実装・科学計算は未着手であり、固定scopeへの新しい明示許可が必要である。
