# H4 XDF9：native費用・楽観的shotsスクリーニング限定科学実行

2026-10-26 JST。**83/83 logical units COMPLETE、correctness FAIL／inflight／retry／resume0。両sとも、合成誤差0を仮定した楽観的総RZが保存XDF8／CDF8の費用を上回った。登録した固定Hoeffding処方・serial実装の数値スクリーニングでは、どちらのbaselineも厳密には下回れないという結果である。** full512 correctness・実合成誤差・最終CA・実際の合成誤差込みshots・最終総費用は未評価。公開照合後mandatory STOP。

- 固定科学SOURCE：`7328ecc9a0602524810e2055cf99c2b1e07729db`、tree `040bb7d96ef4b74cc3150bbf4816512749d7f38d`。library／runner／契約は無変更で実行した。
- Branch：`codex/research-c-c1-h4-xdf9-cost-screen-execution-20261026`、独立worktree `/tmp/prt-c1-h4-xdf9-cost-screen-execution-20261026`。
- [固定83-call契約](research_c_c1_h4_xdf9_cost_screen_contract_unapproved_20261026.json)、scope `C1_H4_XDF9_NATIVE_OPTIMISTIC_COST_SCREEN_V1`。
- 契約SHA256：`fbcb81abf3569d3ace07293ab69d2085e27880a55e2ee6973e7e8a04fb1def7e`。
- SOURCE manifest SHA256：`d736404b6329d8745246b2a608710398d440750e35700f523ddb8b02b9436e08`。
- [機械可読summary](../../artifacts/research_c_c1_h4_xdf9_cost_screen/2026-10-26/execution_summary.json)／[raw48・QPY4・SOURCE・journal・STOP](../../artifacts/research_c_c1_h4_xdf9_cost_screen/2026-10-26/run01/)。
- [native metrics](../../artifacts/research_c_c1_h4_xdf9_cost_screen/2026-10-26/native_metrics.csv)／[全52軸の楽観的shots](../../artifacts/research_c_c1_h4_xdf9_cost_screen/2026-10-26/optimistic_shots.csv)／[全52軸の1測定serial費用](../../artifacts/research_c_c1_h4_xdf9_cost_screen/2026-10-26/serial_per_shot_costs.csv)／[4比較](../../artifacts/research_c_c1_h4_xdf9_cost_screen/2026-10-26/screen_comparisons.csv)。
- [preflight](../../artifacts/research_c_c1_h4_xdf9_cost_screen/2026-10-26/execution_preflight.json)／[postflight・原本hash](../../artifacts/research_c_c1_h4_xdf9_cost_screen/2026-10-26/execution_postflight.json)／[技術準備・単調性・人工97 tests](research_c_c1_h4_xdf9_cost_screen_preparation_20261026.md)。

## 固定条件と再利用

直鎖H4・隣接1.00 Å・STO-3G、Nα=Nβ2／sector36、保存OpenFermion1.6.1 eigh由来の元順prefix9、通常number表現。元g補正κ、signed λ/G、核scalar一回、元frame順・spin／bit conventionを維持した。全frame ordered symmetric S2、s=1主対象／4対照、h=.1/s、δ=.1、m0..12、n=s·2^m、PR非適用。own-XDF9 ground oracle、状態準備費用除外、ideal IID Born測定を仮定する。

CA=.00159 Ha、DF reserveCA/8=.00019875 Ha、β_totalπ/6、β_statπ/12、候補単位failure.01、各round／axisは.01/26のまま。実CAのenergy判定はこの契約では呼ばない。新しいDF分解・fit・rank探索・DF／PF誤差の再計算・別分子は0。

Gaussian=`nature_jw_exact_zero_lapack_atan2_v1`、P0 phase=`qiskit_P0_mpfr160_declared_phase_v1`。Qiskit1.3.0／opt1／seed17／rz,sx,x,cx／approximation_degree1／qubits_initially_zero=false、backend・target・coupling・layout等の追加指定なし。隣接完全同一basis共有・fsum角度結合、head／kernel／tail別compileとall-wire fenceのserial policyを維持した。一般相対Gaussian融合や長round再compileは導入していない。

元packet SHA256は`8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50`。保存XDF9のDF／ground／PF結果は科学SOURCE9a4cda44…／公開347bb0f34…、共通precision shellとbaseline nativeはSOURCE862a97d…／公開f4de9f62…、保存baseline総費用はSOURCE18045333…／公開3a8356a…に由来する。これらのJSON・SOURCE・STOP・契約・CSV・QPY byte identityを照合した。保存ground／generator／PF-kernel NPZと共通shell QPYはhashのみで、新たな数値decodeはしていない。

ユーザーの明示実行許可後、fresh run01を一度だけ実行した。historical unapproved契約名／metadataは変更せず、承認証明書・原文保存・新grantは作成していない。**run01は消費済みで、再起動・retry／resumeは不可。**

## 構造correctnessと未資格範囲

| Gate | 結果 |
|---|---|
| SOURCE56／保存input24／原packet／依存9版 | PASS、実行前後hash一致 |
| 保存DF／PF26／own-ground／alias／元SOURCE・STOP | PASS、再計算なし |
| 保存shell／共通precision・P0／共有・境界policy／baseline総RZとCSV | PASS、baseline再build・再集計なし |
| packet κ／scalar／signed λ/G／元順prefix9 | PASS。κ max-entry差2.7755575615628914e-17、scalar2.29310124732を一回 |
| 10 Gaussian basis request／single-particle transfer | PASS。最大残差1.2237134026861224e-15 ≤1e-10 |
| native modelのrank／block順／scalar／precision | PASS |
| raw kernel2のinstruction itinerary／controlled相対位相 | PASS。cached basis inverse、中央diagonal I項、basis、核scalar ancilla pの登録順・時間と一致 |
| P0 kernel2のregister／finite parameter・phase／既知unitary gate class | PASS、measurement_count0 |

上記は**固定native実装と構造gateに条件付けた費用**の資格である。raw／compiledのfull512 operator作用、global phaseの実作用、actual adjoint、X/Y、全operatorの数値同値性は今回測っていない。個々のGaussian gateをsectorへ射影する代用検証も行っていない。structural PASSをfull-operator PASSに昇格させない。

## native metrics：単一h-step kernel

| s | h | RZ | RZ depth | CX | CX depth | total depth | size |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | .1 | 15720 | 6786 | 8764 | 7314 | 14822 | 30208 |
| 4 | .025 | 15720 | 6786 | 8764 | 7314 | 14822 | 30208 |

raw2／compiled2の計4 QPYを保存した。両sのcount／depth一致は今回の観測であって事前仮定ではない。s4では1 roundあたりのcoherent反復nが異なるため、1測定や13round総費用の一致を意味しない。CDF8／XDF8の回路・metricsは新規生成していない。

## 楽観的shots：13round×X/Y

保存PF誤差をそのまま使い、A=1、決定論RTE／reference law discrepancy0、**counterfactualの合成signal誤差σ=0**で既存Hoeffding solverを呼んだ。実際の合成誤差はnullのまま。各軸に同じNを別々に割り当て、実量子outcome抽選・測定は0である。

| m | s1：X＝YのN | s4：X＝YのN |
|---:|---:|---:|
| 0 | 511 | 511 |
| 1 | 512 | 511 |
| 2 | 512 | 511 |
| 3 | 512 | 511 |
| 4 | 512 | 511 |
| 5 | 512 | 511 |
| 6 | 513 | 512 |
| 7 | 514 | 512 |
| 8 | 516 | 512 |
| 9 | 520 | 512 |
| 10 | 529 | 513 |
| 11 | 548 | 514 |
| 12 | 589 | 516 |
| **13round×X/Y合計** | **13600** | **13314** |

全26 axis-pairが`OPTIMISTIC_FORMULA_ESTIMATE`。INFEASIBLE／UNRESOLVEDは0。固定PF・β・failure配分に対し、非負σを加えるとρ=1−PF−σと正の統計coordinateは非増加、実数shots式とceil後のNは非減少になる。これは登録十分shots処方の下限であり、情報理論的な普遍的必要shots下界ではない。式・ceilの導出と人工境界検証は技術報告に保存している。

## 条件付きserial総RZ下限と4比較

既存serial helperでn=s·2^mと各axisのhead／tailを一回ずつ数え、既存rational楽観集計器でN(0)を加重した。整数RZをdecimal stringで保存し、比較も整数で実行。`aggregate_totals`のCA gateを偽装せず、baseline総費用は保存CSV／JSONの値を引用した。

| s | XDF9 C_opt | 保存XDF8 | 保存CDF8 | 登録数値screen |
|---|---:|---:|---:|---|
| 1 | 144805110560 | 128445454592 | 113219727909 | 両baselineへのstrict改善を除外 |
| 4 | 530107748679 | 470217718215 | 413834761607 | 両baselineへのstrict改善を除外 |

4比較すべて`EXCLUDED_STRICT_IMPROVEMENT_FIXED_MODEL_NUMERICAL`、両candidateは`NO_STRICT_IMPROVEMENT_VS_XDF8_FIXED_MODEL_NUMERICAL`。C_opt≥保存totalという事前登録判定を用い、結果に合わせた閾値・候補・solver・compiler条件の変更はない。

この下限は、**固定した保存PF上界・Hoeffding十分処方・非負追加signal誤差・固定native count・serial policyの数値モデル**の範囲である。sin／log／ceil境界、保存PF reference等の厳密な実数丸め囲いは未資格。実XDF9 full-operator correctnessも未取得であり、無条件の除外、XDF全体の最適性、他実装・誤差相殺・改良PF bound・別solver・新fusionに対する不可能性は示さない。

`actual_synthesis_error`／`actual_total_cost`／`CA_achieved`／`rigorous_numerical_lower_bound`はnull、`full_operator_correctness=false`を保持した。未知σを実際に0と認定してはいない。今回のC_optをXDF9の最終CA適格費用や保証付きshots費用と呼ばない。

## 科学call・計算資源・保全

| call | 実績／上限 |
|---|---:|
| saved_input_load／saved_identity／packet_gate／native_model | 各1＝4/4 |
| gaussian_build／model_structure | 10＋1＝11/11 |
| kernel_build／raw_structure／compile／compiled_structure／metrics | 各2＝10/10 |
| optimistic_round（X/Y一組） | 26/26 |
| round_serial_cost（X/Y一組） | 26/26 |
| optimistic_aggregate | 2/2 |
| screen_compare | 4/4 |
| **合計** | **83/83 COMPLETE** |

journal attempt83／COMPLETE83／FAIL0／inflight0を照合した。Gaussian cache hitも登録basis requestとして計上し、内部diagonalization・primitive構造確認・X/Y展開は固定call定義を維持。CSV・N件数・結果要約は今回保存値の報告処理だけで、solver／compile／費用集計を再起動していない。

新fit／optimizer／rank探索／DF-PF再計算／full Operator／state-action／actual合成誤差／CA／quantum sampling／baseline再build・再集計はすべて0。科学SOURCEとrunnerは無変更で、同SOURCEの人工27＋既存回帰70＝97 PASSの技術証拠を再利用した。追加の科学実行・retry／resumeは0。

開始前affinity32 CPU／空きRAM約52.8 GiB、CPU／memory pressure avg10=0を確認。見えたcgroupのmemory.max／highはmax、CPU quota数値は閲覧範囲に現れず、無制限とは断定していない。既存resource_planで独立s taskをworker2にし、raw inverse／構造の生成をP0前に完了、P0は既存mutexで排他。local BLAS1で過剰並列化を避けた。恣意的CPU／RAM／wall capや共有環境・他ユーザーjobの変更はない。

runner wall10.006123878993094秒、process CPU14.159464611秒、peak RSS469988 KiB。read-only既存Python3.11.1／固定9 package／旧lockを使い、task内cache／tmpを使用した。新しい環境のインストールはしていない。

raw48 files／4,329,205 bytes（QPY4、26 round packet、2 candidate、構造gate、SOURCE／契約／journal／STOP等）を保存。SOURCE56＋saved input24＝protected80件、原packet、契約／manifestの実行前後hashが一致した。旧H2／H4／H6のSOURCE・raw・FAIL履歴・grant・ledger・marker・STOPと、旧約27%／新約12%比較は変更していない。公開されたローカル実行証拠であり、外部再現・CI実行とは区別する。

## 次に必要な担当・判断・mandatory STOP

事前登録の判断規則では両sとも保存XDF8よりstrict改善不能という数値screenになったため、**この固定処方・実装内でXDF9のfull512追加評価を費用競争目的で優先する理由は弱くなった**とユーザー／研究Cへ返す。これはXDF9一般の科学的NO-GOや、将来の別実装を排除する正式採否判断ではない。

**ユーザー／研究C**が本scopeの資格と制約を確認し、XDF9追加評価の優先度・継続要否を判断する。**Codex**は次の具体的依頼まで停止し、full512・actual合成誤差・最終CA・shots・総費用・重要な科学レビューを自動開始しない。必要になった場合のfull512再利用案は準備報告に残してあるが、未実行・未承認である。

今回の83-call scopeに未完了計算・技術blockerはない。commit／push／remote SHA・SOURCE・input・成果物照合後、**mandatory STOP**。
