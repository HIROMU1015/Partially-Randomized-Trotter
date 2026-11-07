# 新A N3-B1b-1：固定43 word・有限辞書provider・strict誤差・T価格上界

2026-11-08 JST、SERVER 2。基準 **fbe7ac8b53acd5c795e94939c5c5e400ced5679a**、branch `codex/pr-reference-n3b1b1-dictionary-20261108`。

**固定43/43基本wordのstrict取得、固定vacuum角域を覆うprovider実装、誤差certificateと局所検証を完了した。** 実採用wordの価格上界は

\[
B_{\rm dict}=6072,\quad B_{\rm cache}=156,\quad
B_P=\max(B_{\rm dict},B_{\rm cache})=6072<173790.
\]

N1の保守的十分条件に要求されたvacuum capの不等式を満たす。これはcompiler条件C・共通word policy・他roleの資格を前提とする結果であり、**全support CT RPE成功／全回路T優越／18round総Tを認定しない**。指定範囲完了により**mandatory STOP**。

## 固定条件とSOURCE

Linear H4・1.00 Å・STO-3G・DF12、元H厳密ground供給・準備費除外。Partial S2、LD3、h=.02、finite RTE K2、ordinary diag(I,V)、canonical independent finite-bit law、q128/r1/m7・保存73 bits。最終native RZ strict Frobenius10^-11、b36、基本指数-36…6、43 key、基本word strict Frobenius10^-13を維持。

pygridsynth2.0.0／seed17／up_to_phase=False、Qiskit1.3.0/compiler A、B′4-CX controlled-RZZ、Reference F/G・actual SOURCE inverse・vacuum P・2 register・8-CX CSWAP、D融合・basis共有・scalar/event・X/Yを変更していない。保存199 shots/axis、energy1.5936e-4 Ha、failure.05、phase.4、alpha1/720も不変更。

正本は[N3-B0 FIXED_PARAMETERS](../../../artifacts/pr_reference_n3b0_numeric/2026-11-08/FIXED_PARAMETERS.json)と[ERROR_BUDGET](../../../artifacts/pr_reference_n3b0_numeric/2026-11-08/ERROR_BUDGET.json)。SOURCE・INPUT・ENVIRONMENTの入口は[成果物README](../../../artifacts/pr_reference_n3b1b1_dictionary/2026-11-08/README.md)。科学条件は[INPUTS.json](../../../artifacts/pr_reference_n3b1b1_dictionary/2026-11-08/INPUTS.json)にも保存した。

実装は[provider](../../../src/trottertracks/pr_reference_dyadic_dictionary.py)、[準備・取得runner](../../../scripts/run_pr_reference_n3b1b1.py)、[検証runner](../../../scripts/validate_pr_reference_n3b1b1.py)。旧acquire／strict_guard／exact_word_matrix／save_json／CT-0 metadata validatorは変更せず、必要な関数ASTを再利用した。旧科学moduleやrunnerの起動を避け、OUTだけをfresh領域へ分離している。旧lower_nativeとCT-1 actual adjoint／word診断は[N3-B1b-0資産](../pr_reference_n3b1b0_runtime_20261108/REPORT.md)経由で再利用した。新しい合成アルゴリズム／emitter／integer-W adapterは作っていない。

## Stage A/B：43 keyの取得

取得直前にN3-B0の43 literal key・inventoryが一致することを確認。保存3 archiveをmetadataで照合して再利用0/43を確認した。legacy method未記載recordを補完・再guardしていない。

43 keyそれぞれ1回だけ既存acquireへ渡し、**43/43 PASS、missing0、failed0、retry0**。各wordはexact cyclotomic matrixと実binary64 RZ targetに対する外向きFrobenius guardで10^-13以下を満たした。結果は[BASIC_WORDS_MANIFEST](../../../artifacts/pr_reference_n3b1b1_dictionary/2026-11-08/BASIC_WORDS_MANIFEST.json)と[basic_records.tar.xz](../../../artifacts/pr_reference_n3b1b1_dictionary/2026-11-08/basic_records.tar.xz)。

literal angle、spec、sequence、hash、W、guard／方式、T+Tdaggerと内数Tdagger、route、backend／数値precisionを元recordのまま保全した。負角のbackend取得はなく、全体actual adjointを使用する。

既存acquire内のClifford候補確認・必要時の同一wordに対するguard precision昇格は元SOURCEの処理であり、別key／seed／backendへの再取得ではない。20000文字の既存拒否条件は維持した。今回の成功をpygridsynthの任意入力停止定理と扱わない。

### 資源・監視

実affinity64、load約4、空きメモリ約499GB、観測時PSI0、GPU未使用を確認。43独立processで取得し、約2.09秒で完了。key別cache／lock／record、排他的attempt record、atomic保存を用い、1 keyの重複起動を防いだ。

監視期限は過去3 poolの最大取得時間約1.61秒の1000倍と1時間の大きい方、今回は1時間。研究的な停止性保証ではなく、無期限待ちを防ぐ十分な余裕の運用上限である。worker失敗／破損／深刻なメモリ不足でも自分の新process groupだけを停止する設計。PSI単独では停止せず、他者processや共有設定を操作していない。[SCOPE](../../../artifacts/pr_reference_n3b1b1_dictionary/2026-11-08/SCOPE.json)に取得前固定した規則を保存した。

## Stage C：有限辞書provider

[DICTIONARY_PROVIDER_SPEC](DICTIONARY_PROVIDER_SPEC.md)に入力／順序／certificateとemissionを定めた。

- 入力は有限binary64で、保存域 **|theta| <= 82.49247654978845**。
- 同じabsolute literal keyが固定C0にあれば、元の299採用wordを優先する。
- missでは整数比から最近接ties-to-evenでnを求め、n*2^-36へ丸める。浮動theta*2^36の丸めや角度mod縮約を行わない。
- 選択bitのwordを**j昇順のwritten product**で連結する。近似wordの可換性を仮定しない。
- 負角はcomplete positive word全体のactual adjoint。emitterへ渡すのはpositive sequenceで、旧lower_nativeが負角の全体inverseを行う。
- named gate／exact Wを保持し、joinはinplace=True。QPY／copy／to_instruction／再compile／後置controlを本番経路へ追加しない。

基本backend recordとcomposite certificateは別のprovenanceである。compositeをpygridsynthの新しい取得recordへ偽装していない。online providerからbackendは呼ばれず、43の有限bit処理と取得済み文字列だけで終了する。

## Stage D：strict誤差とcorrectness

[STRICT_ERROR_BUDGET](../../../artifacts/pr_reference_n3b1b1_dictionary/2026-11-08/STRICT_ERROR_BUDGET.json)にFraction正本を保存した。固定割当を緩めず、

\[
Q_{\rm up}+43\cdot10^{-13}
+\frac{45}{140737488355328}<10^{-11}
\]

を検算した。実guard和を使う上界も固定割当上界以下である。個別certificateは実入力の正確な量子化距離・選択した実guard和・emitter上界を一度ずつ加える。

全域保証は、正確な丸め、N_bによる43-bit包含、全基本wordの外向きguard、unitary telescoping、固定runtime／inplace経路のemitter上界から成立する。有限例のPASSだけを全域証明へ格上げしていない。1-qubit Frobenius保証の埋込みはspectral normで扱い、joint Frobeniusへdimension係数なしで流用しない。

| 確認 | 結果・範囲 |
|---|---|
| 43基本record | key／precision／sequence hash／W／T,Tdagger／取得時guardが全一致 |
| 全43基本wordの正負 | exact whole-adjoint identity、旧emitter順序／named逆gate／W phaseの有限診断PASS |
| 丸め | zero／±tie／隣接grid値／subnormal／両端、Fraction参照と一致 |
| 入力域 | NaN/Infと域外直近値をreject |
| 実providerのcomposite | 固定9入力の正負、certificate、actual adjoint、物理T/Tdagger出現が一致 |
| 最大43 chunk | 実43 wordを全て連結した人工fixtureの正負PASS。全bit1の角は域外なのでpublic providerはreject |
| C0優先 | 全299 keyで元sequence・価格を保持。record viewはimmutable |
| C_phi | 実validation processで定数／SOURCE／runtime binary・丸め条件を再照合 |
| C0誤差 | 全299で保存guard+64u <10^-11、u=2^-53。新guard再実行ではない |

43 chunk fixtureは上界構造の人工検査であり、域外の新しい科学targetの取得・資格ではない。域内の42 chunk入力も検査した。検証は53独立taskを53 process、各BLAS1 threadで約5.23秒。thread数は同時task数から決めた一時設定で、恒常的な上限ではない。

数値Operator tolerance1e-10は診断専用で、strict残余のcertificateにしていない。Operator変換自身の再正規化というN3-B1b-0の留保を維持する。[COMPOSITE_CORRECTNESS](../../../artifacts/pr_reference_n3b1b1_dictionary/2026-11-08/COMPOSITE_CORRECTNESS.json)と[CHECKS](../../../artifacts/pr_reference_n3b1b1_dictionary/2026-11-08/CHECKS.json)に実結果を保存した。

## Stage E：実価格とN1条件

単位は**採用物理T+Tdagger**。Tdaggerを二重加算しない。再compileや新fusionをしない単純連結なので、各基本wordを高々1回使う価格は実43 word価格の和で覆える。

| 量 | 結果 |
|---|---:|
| B_dict=sum(t_j) | **6072** |
| B_cache：固定C0全299 wordの最大実価格 | **156** |
| B_P=max(B_dict,B_cache) | **6072** |
| N1 V_threshold | **173790** |
| threshold − B_P | **167718** |
| B_P < V_threshold | **true** |

各jの実価格と保存N1値は[T_PRICE_BOUND](../../../artifacts/pr_reference_n3b1b1_dictionary/2026-11-08/T_PRICE_BOUND.json)。K=160222、d0=106、d2=114を保持した。C0の既存wordを辞書のwordへ置換して価格を下げた結果ではない。

このcapは全vacuum入力域に対する保守的上界であり、最小T・実到達最大値・canonical期待値ではない。N1のcompiler条件Cと共通policy等が満たされる場合に、求められたpointwise十分条件の価格側を満たす。167718を無条件の全回路削減量や18round改善率へ読み替えない。今回、期待分布・199-shot積・18round総T・T-depthを計算していない。

## 未資格・STOP・次担当

残るのは、他roleの未資格wordと完全な共通policy接続、compiler条件Cの適用範囲、全位置の誤差disk／全support CT RPE成功、総RPE／FT費用である。N2 common frame／event key、CT-3B/SKは照合・取得・再開していない。辞書の角域が似ているという理由で、それらを完了扱いしない。

新PF／gadget／sampling law、Hamiltonian生成／ground solve、native PF build/compile、物理simulation／sampling、b・precision・seed再選択、shot／alpha／fusion最適化は0。新規科学レビューも行っていない。正式共通運用指示2026-10-25、親／worktree AGENTS.md、PROJECT_MAP、固定N3-B1b-0／B1a／B0／N3-A／N1／CT-4B／CT-5B／CT-2・3Aを適用し、旧SOURCE／raw／result／grant／ledger／marker／STOP、共有環境、他Track／他ユーザーを保全した。

[STOP](../../../artifacts/pr_reference_n3b1b1_dictionary/2026-11-08/STOP.json)のとおり限定scope完了でmandatory STOP。**次担当はGPT／ユーザー**：このvacuum capとstrict providerを踏まえ、残るrole／共通policy／RPE資格のどの最小検証を別途指示・承認するか判断する。Codexは追加取得や次stageへ自動進行しない。
