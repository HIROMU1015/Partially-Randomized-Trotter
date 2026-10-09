# G5 既存sourceの情報アクセス棚卸し

2026-10-10 JST。G4公開commit `a221588f42ef3e58f63373915de95607a4ca36be` のsourceをtextだけで監査した。
新input取得、source module import、回路構成、合成・sampling・matrix評価は行わない。
[機械可読記録](../../../artifacts/track_b_g5_fixed_dictionary_closure/2026-10-10/static_access_audit.json)に14 source SHAを固定する。
下記は実装上の事実とMISSINGを分けるための記録であり、新しい研究候補や実行contractではない。

## 入力から費用表まで

| 層 | 既存sourceから確認した事実 | 一般化するとき未取得のもの |
|---|---|---|
| p | [model.py L11](../../../src/trottertracks/algorithm_codesign/rte_reallocation/model.py#L11) の明示exact表(3/4,1/4)。eventsで全tupleの積確率を作る | 実problemのp取得・preprocessing・有限bit sampler費用。toyはp oracleだけを与えた実装ではない |
| Q | [native.py L73](../../../src/trottertracks/algorithm_codesign/rte_reallocation/native.py#L73) は名前付きPauli contexts、distinct_basisのindex0/1、又は明示Pauli string。Q0=ZI、Q1=V† IZ V | 任意involution実装・制御Qアクセス・許可された入力記述のcontract。black-box任意Q APIはない |
| basis | [basis_xx L40](../../../src/trottertracks/algorithm_codesign/rte_reallocation/native.py#L40) はH/CX/RZ(pi/8)の既知2-qubit V記述。uncontrolled VとV†で中央controlled作用を挟む | 別basisを取得・近似・compileする一般費用。Vが与えられることと無料であることは別 |
| rotation | [pauli_rotation L57](../../../src/trottertracks/algorithm_codesign/rte_reallocation/native.py#L57) は明示parityとCRZ分解、Angleのatan比率。involutionでは同じV conjugatorを使用 | Q又はcontrolled Qだけのoracleから任意角rotationを無料で得る証明はMISSING。rotation accessは追加の実装記述が必要 |
| phase/odd | [lower_event L97](../../../src/trottertracks/algorithm_codesign/rte_reallocation/native.py#L97) はodd complementのextra Q、sign反転、phase変更を保持。controlled S/Z/Sdgはancilla相対位相 | これらの費用を一般problemでも取得する方法。system global phaseをcontrolledで捨てることはできない |
| word support | [events L86](../../../src/trottertracks/algorithm_codesign/rte_reallocation/model.py#L86) はpure wordでL^k、rotation付きでL^(k+1)をproduct列挙。現L=2,m=3の評価用materialization | order→IIDで一つのwordを生成できる数学的構成と、全supportの費用評価は別。大きなL,mでの非列挙取得法はMISSING |
| Pauli情報 | modelの[collection L72](../../../src/trottertracks/algorithm_codesign/rte_reallocation/model.py#L72) は登録Pauli controlsのみ。distinct_basisもV明示なので実際にはPauli展開可能 | Pauli情報を使わなかったことからI0 acquisition advantageは結論できない |
| 合成列/error | [numeric L41/L52/L73](../../../src/trottertracks/algorithm_codesign/rte_reallocation/numeric.py#L41) はstrict Frobenius guard、up_to_phase=false、epsilon/4合成、保存列/key/count/guard照合 | 既存126 R1列や12 G4列は既に取得されたlocal evidence。一般Qの未知列・runtimeを無料情報として使えない。G5ではguardを再評価していない |
| native cost | [accounting L11](../../../src/trottertracks/algorithm_codesign/rte_reallocation/accounting.py#L11) は保存列のT/1Q、CX、Clifford、joint errorの加算とIR hash。[canonical_profile L28](../../../src/trottertracks/algorithm_codesign/rte_reallocation/accounting.py#L28) は全events/circuitsを受け取る | whole-circuit再合成最適値やmachine physical costではない。conditional E sqrt(C)、error集計を安く取得する手順はMISSING |
| saved table | [table.py L25](../../../src/trottertracks/algorithm_codesign/ra_d0/table.py#L25) は固定R1 SHA、sign pair、phase、IID law、21 columns/x、重複O0を照合して抽出。prepare runnerはschemaを作る | saved tableのreadが安いことは元のtable取得が安いという証拠ではない |
| IS/dyadic | [g3_finite_law L54/L79](../../../scripts/tracks/algorithm_codesign/g3_finite_law.py#L54) は全event poolの係数とsqrt(cost)からproposalを作り、zero-cost mass/mixを有限候補化。[L41](../../../scripts/tracks/algorithm_codesign/g3_finite_law.py#L41) は2^60 largest-remainder配分 | full-support qの数学的記述は保持するが、スケール可能なサンプラ実装・classical acquisitionの有利性はMISSING。量子測定もしていない |
| known return | [g3_return L30/L60](../../../scripts/tracks/algorithm_codesign/g3_return_comparator.py#L30) はchi=sum p²=5/8を用い、unequal wordsへconditioning、旧O2費用を再利用 | R0の[直接CDF式](rte_reallocation_r0_independent_proof_v1.md)は理論上の分布恒等式。実有限bit汎用サンプラや再allocationとの統合は未検証 |
| CTS | [g4_cts_specialization L64/L80](../../../scripts/tracks/algorithm_codesign/g4_cts_specialization.py#L64) は明示R=3/4ZI+c/4IZ+s/4XY、R²の9積と集約後R³の6積、quartic scalar algebraを使用。real negative correctionを保持 | このtoyでCTS acquisitionが困難とは言えない。別problemでのPauli acquisitionコスト比較・不可避下界はMISSING |
| CTS lowering | [g4_matched_cts L31/L46](../../../scripts/tracks/algorithm_codesign/g4_matched_cts.py#L31) は集約Pauliを直接lowerし、全6event費用からproposal。G4で12 synthesis/28 conditions取得済み | CTSへ後付けでV費用や架空classical penaltyを加えない。違う辞書へ同native費用を転送しない |

## 係数算術と情報取得を分離する

[R0 A2](rte_reallocation_r0_independent_proof_v1.md)のO(m)は(x,m)から隣接次数係数を作る算術量。
precision bit complexity、Q/pの取得、word生成、制御・basis・odd-event費用、conditional費用/誤差表の取得を含まない。
IID word生成の理論構成は全wordの列挙を要求しないが、現resource evaluatorは列挙する。
現pilotの小さな表・保存値再利用を、general DF scaleでのclassical acquisition削減と呼ばない。
importance sampling最適proposalもcomplete per-event costを仮定している。非列挙で同情報を取得できることは未証明。

## 共通DF codeの静的境界

[src/trotterlib/rte.py L567](../../../src/trotterlib/rte.py#L567) のRTEEventはodd orderを拒否し、phaseをpaired evenの(-1)^(order/2)へ制限する。
[df_rte_circuit.py](../../../src/trotterlib/df_rte_circuit.py) はZ/ZZ、basis記述、controlled identityのancilla相対位相を定義する。
[df_rte_qiskit.py L92](../../../src/trotterlib/df_rte_qiskit.py#L92) のevent phaseは±1に限る。
既存even DF wrapperの存在は、Track Bのodd/complement/reallocated ensembleの統合検証ではない。
本監査ではshared source/APIを変更・コピー・importしていない。DF入力や分子NPZへ触れていない。

## 未解決事項とSTOP

一般I0 routeの入力/access契約、non-enumerative cost/error acquisition、sampler/range contract、
同accessの強い既知対照との実classical費用比較、Pauli取得の不可避下界、一般DF odd controlled semanticsはMISSING。
どのMISSINGを次の科学的課題として採択するかはGPT/利用者の判断。新candidateを選ばずmandatory STOP。
