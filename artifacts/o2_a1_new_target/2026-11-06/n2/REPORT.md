# A1新finite target：N2限定検証結果

2026-11-06 JST、SERVER1 GPUA100-gleap-01。
**総合判定：部分確認／mandatory STOP。stateは保証できたが、総engine誤差はUNKNOWNであり、N3へ進める状態ではない。**

対象は保存linear H4／1.00 Å／STO-3G／legacy DF12、8 system qubits、**全4電子number sector（70次元）**。
L_D=3、m0/q1/δ=0.02/r1/K2、coefficient_atol=0.0、X/Y、理想gate・保存DFモデル限定のN2診断である。
一次targetはN1 actual finite-grid Qとselected logical full interrogationのattenuated複素meanのまま維持した。

## 固定と実行した科学計算

| 項目 | 固定先・回数 |
|---|---|
| N1結果 | `744e1a09661ebba6841fe5faf456f88c41c47780` |
| N1科学SOURCE | `b6dbec9f80da35a88fba28795ad02ca74c9adfa0` |
| 新finite-target契約 | `2b2e6c1bc3b2d201d1a6d1238172187d68aea6ab` |
| **N2科学SOURCE** | [`a4141795bd3e23670612d0f0d4a43678457a1a37`][SOURCE] |
| 同じ保存NPZのload | 1 |
| 既存ground solver | 1（全Hをそのまま入力、既存Python matvec、deterministic all-ones v0、eigsh tol1e-10） |
| exact sector matrix組立 | 1 |
| 全70次元spectral検査用eigh | 1（solver retryとは別の全sector検証） |
| N1 partition/preparation/CDF再計算 | **0** |
| RNG sampling／Hadamard shots／実分子full build／compile／variance・N・費用評価 | **すべて0** |

[SOURCE.json](SOURCE.json)、[CALL_COUNTS](CALL_COUNTS.json)、[実行ログ](RUN.log)、[STOP](STOP.json)を保存した。科学retryはない。
[実行仕様・command](../../../../docs/research/o2_a1_new_target_n2_20261106/EXECUTION.md)と[環境lock](PROFILE.json)を入力前にcommit・公開照合した。
Python3.12.3／NumPy1.26.4／SciPy1.14.1／Qiskit1.3.0／mpmath1.3.0、SOURCE536ファイルを固定し、実行時に全profile一致を確認した。
共有venv/packageは変更していない。CPU/RAM/workerの人工上限は設けず、[実資源観測](RESOURCE_OBSERVATION.json)と自分専用cacheで非干渉を維持した。

## State：全保存DF Hの一意groundを確認

既存cache129件の[header・identity調査](../../../../docs/research/o2_a1_new_target_n2_20261106/STATE_INVENTORY.json)には、全4電子number sectorの対応stateがなかった。実行時の全H/sector照合でも適格candidateは0件で、旧spin sectorや旧opt2 stateへ変更していない。
`solve_df_ground_state`は名前のdocstringにH_Dとあるが、関数は入力されたDFHamiltonianをそのままmatvecへ渡す。今回は全12 fragment・one-body・constantを持つN1 Hを入力した。

[STATE_RESULT](STATE_RESULT.json)の主な保証は次のとおり。

- H identity：`56e4df83655aa2f2f8132126f2635996516dc2cfb1bf4cb8d95490184ad631e5`（N1と同じ）。
- sector identity：`7ce082a7442726829c4ccf4f4a77dfc50909e71aea18a666dc245c224c4f1ca3`。8 modeの全4電子70 indexで、S_z sectorへ狭めていない。sector外成分はexactに0。
- 新state payload identity：`405636138c2d5ccf6835beeb6e08337523fc04b513732110d3d0125874c49de3`。
- 保存raw vectorのnorm²−1は約2.24e-16。保証対象の数値stateは**保存vectorをexact normで正規化したray**と定義する。
- 全sectorのground区間と第一励起下界が分離し、一意groundを確認した。exact endpointとgapはengine用auditへ保存した。
- 任意の同じideal unitaryに対するX/Y座標state誤差上界は、**7.81e-7以下**。これは保存DF groundとの差の一様上界であり、総engine誤差ではない。

保存binary64係数をexact rationalとして、既存fermionic excitation tableからH=cI+one_body+ΣλA²を厳密に組み立てた。余分な1/2、threshold、symmetrizeによるモデル変更はない。
全eigensystem候補Q,dに対し、Gram defectとH−Q diag(d)Qᵀをexact Fractionで検査し、polar orthogonalizationとWeylの関係からordered eigenvalue区間を作った。
このmatrix error上界は約3.48e-14で、単なる浮動小数点固有値や小さいresidualを下界として扱っていない。
既存directed interval helperで平方根を外向き評価し、草案§6の `(r_U−l0)/(l1−l0)` と `2 sqrt(excited weight)` を用いた。厳密endpointが正本で、小数は表示用である。

完全vectorとqubit permutationは[engine_internal/STATE_VECTOR](engine_internal/STATE_VECTOR.json)、全eigensystemとenclosure witnessは[engine_internal/SPECTRAL_WITNESS](engine_internal/SPECTRAL_WITNESS.json)に保存した。
E0・gap・vector・path signal・真varianceをCV学習や配分へ渡す処理は起動していない。[N3_HANDOFF](N3_HANDOFF.json)はstate identity・一様誤差上界・N1 identityとUNKNOWN事項だけを渡すwhitelistである。
元未近似Hamiltonianへの保証や状態準備費を認定していない。

## Engine：規約を確認、総数値保証は未閉鎖

[ENGINE_RESULT](ENGINE_RESULT.json)は、確認済み事項と未取得事項を分けている。

| 項目 | 確認・保証範囲 |
|---|---|
| qubit ordering | OpenFermionはmode pをindex bit 7−p、Qiskitはbit pへ対応。既存bit reversalを実stateに適用し、全indexの逆変換一致を確認。sector・full-space embeddingも一致 |
| controlled規約 | 既存wrapperのordinary diag(I,U)、ancilla最後、二重controlなしを人工回路で確認 |
| scalar・identity phase | 既存`_append_step_phases`のcontrolled ancilla PhaseGateを人工入力で確認 |
| event phase/sign | 既存RTE builderのn=2の−1、負identity符号、rotationを人工eventで確認。SOURCEのphase累積箇所も確認 |
| X/Y | 既知phaseを持つ人工1-system-qubit回路で、Xは実部、Yは虚部、Y readoutはSdg→H、bit0→+1／bit1→−1を確認 |
| state項 | exact-normalized保存stateについて一様座標上界≤7.81e-7 |
| measurement bit-grid項 | **独立uniform U=k/2^53とU<p_hatが実装される場合**、座標mean誤差≤2/2^53=1/4503599627370496。grid endpointsを含む解析・人工確認。実RNG/bit生成は未実行 |
| selected gate適用の数値誤差 | **UNKNOWN** |
| Born確率計算・norm/range guardの実装binding | **UNKNOWN** |
| 総b_engine | **UNKNOWN**。未取得項を0として加算していない |

人工回路の数値一致は規約とSOURCE処理の確認であり、全実分子pathの厳密operator誤差証明ではない。
N1のlaw/preparation/policyはそのまま参照するが、保存selected操作を復元してstateへ作用させる実measurement engineとのend-to-end対応とuniform arithmetic boundは今回閉じていない。
既存`fixed_signal_evaluation`は再preparationやalgebraic referenceを使う経路を含むため、その旧bias/allowanceを保存selected targetの保証へ移していない。実分子full wrapperのbuild/evaluationはN2の禁止範囲なので実行していない。
ideal IID word/Born modelは仮定であり、seed名やPCG64の独立性を証明したとは言わない。

草案のε_axis=1/4、b_engine≤ε_axis/8（候補1/32）は**未採択の参照値**である。state項だけを比較して総engineが受理されたとはしない。新しいε/α、測定予算、pilot数、controlを採択していない。
N1のg2/g3の希少性、特にν3≈1.8921e-9を維持し、control・pilotの自動変更は0件。

## Correctness・保全・次の条件

新規N2人工15件＋既存A1人工40件＝[55件PASS](../../../../docs/research/o2_a1_new_target_n2_20261106/TESTS.json)（[log](../../../../docs/research/o2_a1_new_target_n2_20261106/TESTS.txt)）。
独立FermionOperatorの小模型照合、zero-residual励起状態や縮退groundの棄却、spectral reconstruction error課金、phase/readout/order、禁止callを確認した。
実科学取得後は再solve・再eigh・再preparationをせず、保存ファイルのidentity/counts/UNKNOWNを[静的照合](../../../../docs/research/o2_a1_new_target_n2_20261106/CORRECTNESS.json)した。
これはlocal科学取得と人工correctnessであり、外部再現やCI実行の証拠ではない。

[保全記録](../../../../docs/research/o2_a1_new_target_n2_20261106/PRESERVATION.json)：元mainのHEAD/statusとdirty/untracked 2,136ファイルのbytes、N1/旧A1 artifact40ファイル、固定SOURCE536ファイルを照合した。旧SOURCE・結果・STOPを保全し、C1は凍結のままである。

N3には、少なくとも次が必要である。

1. N1 identityと今回state identityを固定したまま、保存selected logical gate列を使うengineの実装binding、exact normalization、確率range guardとuniform gate/Born arithmetic boundを閉じるための追加検証scopeと明示許可。N2 state solveを無条件に繰り返す必要はない。
2. state＋数値演算＋確率＋bit-grid（必要ならcompile）を含む総b_engineの受理。UNKNOWNを残したまま測定を始めない。
3. ε/α、独立stream、条件付き取得法、train/pilot/cal/prod予算・range guard・価格指標を採択し、N3の測定・必要なfull build/compile/価格評価を明示許可すること。旧opt2のN・価格・biasで補完しない。

**N2の結果保全・公開照合後にmandatory STOP。N3・18round展開・C1解凍・重要科学レビューは開始しない。**

[SOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/a4141795bd3e23670612d0f0d4a43678457a1a37
