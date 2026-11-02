# C1：H4 XDF9 native費用・楽観的shotsスクリーニングの技術準備

2026-10-26 JST。起点 `347bb0f34738613c0c58c1f638c8b17a0c71b1aa`。独立branch `codex/research-c-c1-h4-xdf9-cost-screen-preparation-20261026`。

**XDF9の2 kernelと、保存PF26セルからの楽観的shots／serial費用スクリーニングを接続した。人工27＋既存回帰70＝97 tests PASS、SOURCE56／保存input24／原packet／環境のbyte preflight PASS。未承認の科学call上限は83。実H4の数値decode・build／compile・shots／費用評価はすべて0で、公開照合後mandatory STOPする。**

本書は技術実装・人工検証・実行条件の固定であり、重要な科学レビュー、実XDF9の費用結果、最終CA判定ではない。

- [最小adapter・runner](../../src/trottertracks/c1_h4_xdf9_cost_screen.py)／[CLI](../../scripts/run_research_c_c1_h4_xdf9_cost_screen.py)／[人工test](../../tests/management/test_research_c_c1_h4_xdf9_cost_screen.py)
- [未承認83-call契約](research_c_c1_h4_xdf9_cost_screen_contract_unapproved_20261026.json)／[SOURCE・test logs・passive証拠](../../artifacts/research_c_c1_h4_xdf9_cost_screen/2026-10-26/technical/)

## 問いと比較条件

直鎖H4・隣接1.00 Å・STO-3G、Nα=Nβ2／sector36。保存OpenFermion1.6.1 eighの元順XDF prefix9、元g補正κ・核scalar一回・signed λ/G・元frame順を維持。全frame ordered symmetric S2、s1／4、h=.1/s、δ=.1、m0..12、n=s·2^m、PR非適用。own-XDF9 ground oracle、状態準備費用除外、ideal IID Born測定。CA=.00159 Ha、DF reserveCA/8=.00019875 Ha、β_totalπ/6、β_statπ/12、候補failure.01、axis.01/26を維持する。

Gaussian=`nature_jw_exact_zero_lapack_atan2_v1`、P0 phase=`qiskit_P0_mpfr160_declared_phase_v1`。Qiskit1.3.0／opt1／seed17／rz,sx,x,cx／approximation_degree1／qubits_initially_zero=false。backend・target・coupling・layout等を新指定せず、隣接完全同一basis共有・fsum角度結合と、head／kernel／tail別compile・all-wire fenceのserial費用定義を維持する。一般相対Gaussian融合、別compiler、rank探索は導入しない。

研究上の問いは、同じ保存PF誤差を使う登録Hoeffding処方と固定native実装の下で、XDF9に保存XDF8より安くなる余地があるか、である。full512資格付けを最初から行う代わりに、固定実装のP0費用と楽観的Nによる条件付きスクリーニングを先に取得する設計とした。

## 固定証拠と再利用

| 対象 | 科学SOURCE | 結果公開revision・用途 |
|---|---|---|
| XDF9 表現／ground／PF | `9a4cda44a45675c2518b2252a98e78eeeb320942` | `347bb0f34738613c0c58c1f638c8b17a0c71b1aa`・[95-call結果](research_c_c1_h4_xdf9_pf_execution_20261026.md) |
| CDF8／XDF8共通precision native | `862a97d4213121749448ca716cde820a44b99772` | `f4de9f62c2ff1cbaaa75b6cdaa07217d00dfd957`・[native結果](research_c_c1_h4_native_fair_execution_20261026.md)、共通shell metrics／QPYの来歴 |
| CDF8／XDF8 CA／費用 | `18045333a1438740160d7b9e44e78139ad59f2b5` | `3a8356a7843d3cb46bb8069955640381b490ac4c`・[CA／費用結果](research_c_c1_h4_ca_resources_execution_20261026.md)、保存総RZと元CSV |

上記3資料を固定GitHub refで取得し、Git blobとローカル起点が一致した。[reuse evidence](../../artifacts/research_c_c1_h4_xdf9_cost_screen/2026-10-26/technical/reuse_evidence.json)。既存native／resource／PFのSOURCE manifestに含まれるファイルも全件不変だった。新module／script／test以外の計算SOURCEを変更していない。

元P-H4 SHA256=`8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50`。λ/G・κ・核scalarをこのpacketの元順prefix9から取得する。将来の読込は元packetだけを数値decodeし、保存XDF9 ground／generators／PF-kernel NPZと、共通shellのcompiled QPY3件は来歴のbyte identity確認に留める。PFは既存result JSONの26行を使用し、ground eigensolve・DF norm・PFの再計算を行わない。

保存DF・ground／alias・PF PASS、各元SOURCE／消費済みSTOP／契約、共通Gaussian／P0／route／serial policy、shell測定範囲、保存baseline totalsとCSVの一致を接続gateで確認する。XDF8／CDF8の回路は再buildしない。保存baselineは次の値に固定し、今回再集計しない。

| s | XDF8総RZ | CDF8総RZ |
|---|---:|---:|
| 1 | 128445454592 | 113219727909 |
| 4 | 470217718215 | 413834761607 |

## native接続と最小correctness

既存`prepare_xdf`へ保存one／λ[:9]／G[:9]／cを渡し、既存`PrecisionBuilder`でbase＋9 frameのGaussianを準備する。固定Gの内部対角化は表現変換であり、新しいDF分解やfitではない。`NativeBuilder.ordered`でh=.1／.025の2 kernelだけをbuildし、既存MPFR160 wrapperとP0 compile mutexを使用する。

metrics取得前のgateは次の通り。

1. packetのfinite、exact spin lift、元g由来κ（max-entry1e-10／rtol0）、scalar、rank／frame順を確認する。
2. 全10 basis requestのsingle-particle transfer reportが共通precision版かつ残差≤1e-10であることを確認する。cache-hitも同じblock coverageとして記録する。
3. raw kernelの全instructionを、cached basis inverse → 既存中央diagonal emitter → basisの登録run順と比較する。recording sinkは追加の全kernelを作らず、既存emitterのgate objectを記録するだけである。frame／h／共有規則／diagonal I項のcontrolled phase／核scalarのancilla p一回を保持する。rawの不意なglobal phase・位相追加・順序変更はFAIL。
4. P0のregister・ancilla-last・全parameter／declared phaseの有限性、classical condition／非unitary操作の不在、既知のRZ／SX／X／CX gate classを確認してから既存`circuit_metrics`を呼ぶ。gate名だけの偽装やresetも拒否する。

これは**構造gateと固定実装に条件付けた候補費用**である。実H4のfull controlled operator、raw／compiledの数値同値性、global phaseの実作用、X/Y、actual adjointはこの初回契約で測らない。例えば有限のcompiled global phaseを変えた人工回路は構造gateを通り得ることをtestで明示した。構造gateをfull512証明に昇格させない。個々のGaussianをsectorへ射影する処理もない。

raw／compiled QPYを各2、計4件保存する。RZ count／depth、CX count／depth、total depth、sizeを既存の定義で取得する。両sのkernel費用が同じとも、rank9がrank8より高いとも事前には仮定しない。

## 楽観的shotsの下限性：式とceil

保存PF誤差をp、仮に加える非負の合成signal誤差をσとする。今回の固定条件ではA=1、決定論RTE／reference law discrepancy=0である。既存`assess_round`の処方は、

\[
\rho(\sigma)=1-p-\sigma,\qquad
c(\sigma)=\rho(\sigma)\frac{\sin\beta_{\rm stat}}{\sqrt2},\qquad
N(\sigma)=\left\lceil\frac{2\log(2/\alpha_{\rm axis})}{c(\sigma)^2}\right\rceil.
\]

適用域はρ>0かつasin(p+σ)+β_stat≤β_total、PF／target／failure配分が同一の場合。σ≥0を増やすとρと正のcは非増加、正の実数shots式は非減少であり、**ceilも順序を保つ**。したがって適用可能なσに対しN(0)≤N(σ)。途中でphase／radius十分条件を満たさなくなれば、その処方には有限の適格Nがないので「少ないNで実現可能」とは扱わない。

これは単なる標本検査ではなく固定式の単調性である。実装では既存`assess_round`→`rpe_resource_accounting._hoeffding_shots`をそのまま呼び、integer ceilを変更しない。X/Yは同じ式・同じαなので1回のsolver結果を両軸の別々の予定Nへ割り当てる。outcomeの共有や実量子測定は行わない。

仮定σ=0は`optimistic_synthesis_signal_error`としてのみ記録し、`actual_synthesis_signal_error`はnullを維持する。普遍的／情報理論的な必要shots下界ではなく、**固定PF上界を保つ登録十分shots処方の楽観的最小値**である。PF上界の改良、誤差相殺を使う別solver、別回路／compiler／fusion／law、状態準備込み条件には、この除外を拡張しない。

binary64で計算した保存PF、rho、sin／log、ceil境界は厳密な実数丸めenclosureではない。同一binary64定数・通常の順序保存演算における非負追加誤差を人工検証し、ceil境界の隣接floatも確認したが、これを超越関数や保存reference誤差の厳密証明とは扱わない。`rigorous_numerical_lower_bound=null`、`floating_point_rounding_enclosed=false`を維持する。

## serial RZ下限と比較ロジック

固定した非負のnative countに対し、各axisの1測定費用は

\[
R_{m,a}=s2^m R_{\rm kernel}+R_{\rm head}+R_{{\rm tail},a},\quad
C_{\rm opt}=\sum_{m=0}^{12}\sum_{a=X,Y}N_m(0)R_{m,a}.
\]

Nの単調性と非負のRから、同じkernel／shell／fence policyで非負σを含めた処方の費用はC_opt以上になる。headとtailは各axis・各測定に一回、kernelはn回。未観測の境界融合・literal/shared削減・長round再compileを加えない。depthも明示serial scheduled sumとして記録し、global最適depthとは呼ばない。

費用展開は既存`serial_round_cost`／`compose_serial_cost`、加重集計は既存`cost_only.aggregate_optimistic`を再利用する。`aggregate_totals`は実精度のgateを持つため、このcounterfactual screenへ`same_accuracy=True`を与えて迂回しない。既存楽観的集計器のrational totalsから整数RZを取り出し、比較は整数／decimal stringで行う。2^53超の値もfloat表示へ丸めて順位を決めず、overflowや欠測をclip・0埋めしない。

| 条件（同じs） | 出力・意味 |
|---|---|
| C_opt ≥ 保存XDF8 | `EXCLUDED_STRICT_IMPROVEMENT_FIXED_MODEL_NUMERICAL`。この固定処方・serial実装・数値資格の下でXDF8を厳密には下回れない |
| C_opt ≥ 保存CDF8 | 同じ資格でCDF8を下回れない。XDF8との比較結果は別に残す |
| C_opt < 保存XDF8 | `COMPETITIVENESS_NOT_EXCLUDED`。候補に残るだけで、資源優位・CA PASSではない |
| 欠測／数値range／未資格集計 | `UNRESOLVED`。除外や0補完を行わない |
| optimistic phase／radius未達 | `INFEASIBLE_OPTIMISTIC_PRESCRIPTION`。登録十分処方の未達であり物理的不可能性ではない |

等号は「strictly beating」を除外する。deterministic回路のcount supportは当該固定回路のsingletonであり、accuracy資格とは別である。数学的処方下限と、実入力を厳密に囲っていない数値スクリーニングを区別し、無条件の除外・総資源優位は主張しない。片方のsが不利でも両sの全13roundを実行する。

## 人工・回帰検証

| suite | 結果・範囲 |
|---|---|
| 新XDF9 cost screen | **27 PASS**、13.20秒。人工2空間軌道・9 frameのe2e、2 kernel／QPY4、全83-call計上、26 axis-pair、4比較、fresh／STOP |
| 既存H4 native fair／H4 CA resources／H6 precision／H6 phase | **計70 PASS**、28.91秒。builder／phase／係数／旧scalar経路を無変更で回帰 |
| 合計 | **97 unique tests PASS**。外部再現・CI結果ではなく、ローカル人工／回帰証拠 |

[新test log](../../artifacts/research_c_c1_h4_xdf9_cost_screen/2026-10-26/technical/artificial_tests.txt)／[既存回帰log](../../artifacts/research_c_c1_h4_xdf9_cost_screen/2026-10-26/technical/regression_tests.txt)。

人工e2eでは、独立CARから元signed Gのsquareを構築したcontrolled全32成分referenceと、raw／P0の両方を比較した。tiny Gaussian角・det符号・phase・第9frameを含み、実H4のNPZ／QPYは使っていない。位相追加、frame逆順、時間違い、Gaussian transfer異常、compile後reset、未知gate class、SOURCE／入力／P0／saved rank／PF target／shell measurement／baseline値の不一致を拒否した。

shotsは非負合成誤差の系列とceil境界隣接floatで単調性を確認。費用は独立した整数式でhead／X／Y／n／Nの計上を検証し、2^53超の正確な総RZ・等号判定も確認した。競争可能、競争不能、unknown、十分条件未達、correctness FAILの分岐を検証。未知σとactual totalsはnullのまま、CA gateの迂回・実molecular PF／DF／fit／energy／full Operator呼出しは例外mockで遮断した。

開発中に大整数testの独立期待式でshell合計を11と誤記して1 FAIL／25 PASSとなり、正しい12へ修正した。[初期log](../../artifacts/research_c_c1_h4_xdf9_cost_screen/2026-10-26/technical/artificial_development_initial.txt)も保全。旧科学結果や科学callのretryではない。最終SOURCEで上記97 testsの適合範囲がPASSしている。

## 未承認契約・83 logical units

Scope=`C1_H4_XDF9_NATIVE_OPTIMISTIC_COST_SCREEN_V1`。過去95／305／119の残枠を使わず、今回の処理から独立に算定した。

| call | 最大 |
|---|---:|
| saved_input_load／saved_identity／packet_gate／native_model | 各1＝4 |
| gaussian_build（10 block request）／model_structure | 10＋1 |
| kernel_build／raw_structure／compile／compiled_structure／metrics | 各2＝10 |
| optimistic_round（各s×13round、X/Y一組） | 26 |
| round_serial_cost（同じ26組、内部でX/Yを別展開） | 26 |
| optimistic_aggregate | 2 |
| screen_compare（各s対XDF8／CDF8） | 4 |
| **合計** | **83** |

内部の固定G diagonalization、Gaussian 1／2-qubit gate transfer、recording sinkでの既存diagonal emitter、XYの二つのserial展開は各複合call内と明記した。batch化で計上を減らさない。rawのinverse／構造確認を全て終えてからP0 workersを開始し、既存mutexを保つ。

| 固定対象 | Identity |
|---|---|
| 契約SHA256 | `fbcb81abf3569d3ace07293ab69d2085e27880a55e2ee6973e7e8a04fb1def7e` |
| SOURCE manifest SHA256 | `d736404b6329d8745246b2a608710398d440750e35700f523ddb8b02b9436e08` |
| SOURCE／保存input | 56 files／24 files。原packetは別途同一hashを確認 |
| fresh output | `artifacts/research_c_c1_h4_xdf9_cost_screen/<execution-date>/run01` |
| fit／DF-PF再計算／full512／actual synthesis／CA／quantum sampling／baseline再build・再集計 | 0 |
| retry／resume | 0／0 |

SOURCEはこの公開revisionのmanifest集合に固定し、起点347bb…や旧PF科学SOURCE9a4c…と区別する。将来のrunは実行HEADをSOURCE.jsonへ保存する。[passive preflight](../../artifacts/research_c_c1_h4_xdf9_cost_screen/2026-10-26/technical/passive_preflight.json)ではNumPy／Qiskit／SciPy／ffsim未importのまま、input hash・packagesを確認した。実NPZ／QPYの数値decode、新科学call、new run01は0。

既存read-only Python3.11.1、NumPy1.26.4／SciPy1.14.1／OpenFermion1.6.1／openfermionpyscf.5／PySCF2.7.0／Qiskit1.3.0／Nature.7.2／SymEngine.13.0／threadpoolctl3.6.0と既存lockを使う。新インストール・共有環境・他ユーザーjobの変更なし。56 SOURCE／24 inputには旧資料をbyteで固定したものがあり、旧code・raw・grant・ledger・marker・STOPを改変していない。

実行規模は9qubit kernel2 build／P0 compile2、10 basis request、4 QPY、scalar26組。512×512のOperator／state-actionや長round回路は作らない。利用可能affinity／quota／RAM／pressureを実行直前に確認し、独立s task2とcompile排他、local BLASを調整する。worker2は登録独立task数に由来し、恣意的な資源capではない。raw/basisの逐次部分はcacheとinverse／compile競合を避けるため。実H4のwall時間やspeedupは未測定で推定値を固定しない。

## 競争可能性が残った場合のfull512案（未実装・未契約）

対象はXDF9 s1／4。必要になった場合だけ後続契約を固定する。

1. 今回のraw／compiled QPY4件を、source／target／h／共有・境界／phase／P0／QPY byte identityが不変の場合に再利用する。不要な再build／compileを避ける。変更が必要なら公平な比較への影響を別判断する。
2. 既存独立CAR referenceからbase＋9 frameの正しいfull-Fock targetを準備する。保存sector36のgeneratorやPF kernelをfull512 coverageの代替にしない。全controlled512入力／出力でcontrol0 identity／control1 targetとraw／compiledを比較する。
3. 既存`numerical_segment`、`c1_h4_full_synthesis.segment`、actual adjoint、head／X／Y、Frobenius、両行列・shellのnorm-growth guardを使う。finite probeやbasis gateのsector射影で代用しない。旧shellとCDF8の保存native／合成結果は、同じtarget／precision／P0／route／time／serial policyとbyte identityを確認できる範囲で再利用する。
4. 保存XDF9 PF26を一回だけ接続し、13round telescopingを評価する。数値資格Bと厳密rounding enclosureを区別する。合成誤差込みのactual prescription N、固定N margin、CAと最終serial費用は、その後の必要な固定範囲・別許可で評価する。

screenで両sとも改善不能なら、固定モデル内でfull512追加評価の費用競争上の優先度が低いと研究Cへ返す。可能性が残れば後続技術設計を提案する。どちらも全XDF9実装の不可能性や普遍的優位を意味せず、今回の準備から新科学計算へ自動進行しない。

## 残る資格・次の担当・STOP

技術的blockerはない。式の単調性と整数化／exact serial集計は確認したが、実入力の丸めenclosure、実XDF9 full operator・compile数値同値性・actual合成誤差・最終CA・保証付きshots・最終総資源は未資格／未評価である。

次は**ユーザー／研究C**が本公開SOURCE・未承認83-call契約の限定科学実行を許可するか判断する。**Codex**はその別明示許可後だけ、fresh run01で2 kernelと楽観的スクリーニングを一度実行し、両sの結果を公開してSTOPする。承認証明書・原文保存・新grantは不要。重要な科学レビューも今回開始していない。

今回の技術準備はcommit／push／remote SHA・SOURCE・保存input・成果物照合後**mandatory STOP**。実H4のbuild／compile・shots・費用評価を開始しない。
