# Full-H κ2 P-1：保存run02の再検証（方式A、再計算なし）— PASS

2026-11-14 JST。利用者の承認（方式A）に基づく作業である。実施したのは主agent（Claude Code）で、作者自身の検証であり、独立した査読ではない。

## 結論

| 区分 | 状態 |
|---|---|
| 原run02 | **`STOP_WIRING_CHECK`のまま**。[run02/P1_RESULT.json](../run02/P1_RESULT.json)の`ok=false`・`scalar_once=false`は書き換えていない（`cd56422a8`の原本とbyte一致を確認） |
| 再検証 | 修正SOURCE `8258b2200`による事後再検証で**PASS**（23条件すべて）。記録：[REVALIDATION.json](REVALIDATION.json)、commit `6d188defd` |
| 価格 | X/Y各1件の価格record・sidecar・implementation evidenceを発行した。**Cの固定guard（`c628e9e8`のblobを無変更で実行）が2件とも受理した**：[GUARD_CHECK.json](GUARD_CHECK.json) |
| 新規の科学計算 | sampling 0、回路build 0、statevector 0、compile 0、retry 0 |

**STOPが旧判定の偽陰性だと判断した根拠**

1. ancillaに作用する命令は581個で、X/Yとも次の内訳だった。この並びが、保存したeventから固定provider規則で導いた期待列と、名前・qubit・角度（exact一致）まで完全に一致した。
   - crz 568（各eventの中心回転）
   - cz 9（order-2 eventのproduct）
   - event位相のp 3
   - scalarのp 1
2. event位相は`p(π)`の3個で、位置はrequest 0・3・7の末尾（命令番号7039・28625・56898）だった。
   - 各requestはorder-2 event（位相−1）を1個だけ含む（request内のevent番号は7・68・64）。
   - product符号−1とidentity applicationは0個なので、provider規則による累積はπになる。
3. scalarの`p(0.2651822507334529)`はちょうど1回、制御進化の最後の命令（56899）にある。
   - 角度は−scalar·T＝−(−0.33147781341681615)×0.8とbit一致する。
   - scalar値は保存Full-H preparationの値（定数2.29310124732＋抽出identity −2.6245790607368162）と一致する。
4. X/Yのwrapperは、同じ制御進化（56,899命令）を命令単位で同一に含んでいた。
5. 旧判定（ancillaの`p`がちょうど1個）をX/Yの両方に再適用すると`false`になり、原STOPを再現した。
6. 作用検証は原run02ですでにPASSしていた。
   - 独立occupation oracleとの差は1.18e−14で、このoracleはevent位相とscalarを含む。
   - compiled作用は≤9.9e−12。
   - 位相の構造に誤りがあれば、これらの検査は通らない。

## 1. 修正した判定（`SCALAR_EVENT_PHASE_ANCILLA_SIGNATURE_V1`）

SOURCE：`scripts/run_full_h_kappa2_p1.py`（commit `8258b2200`）。旧SOURCE `72d4f4ff2`との差分は[GATE_SOURCE_DIFF.patch](GATE_SOURCE_DIFF.patch)。

**期待列の作り方**（保存event、またはlive eventの`to_dict()`から作る）

- request内の各eventについて、applicationの順に次の命令を並べる。
  - product：`cz(anc, q)`
  - 1-qubitの回転：`crz(anc, s)`
  - ZZ回転：新ZZでは`crz(anc, s1)`（前後のcxはsystem側なので期待列に入らない）
  - いずれも角度は2·`unsigned_rotation_angle`
- request末尾のevent位相：固定provider c31ef424 `_build`と同じ順序で位相を累積する。
  - まず全eventの`_event_phase_angle`を足す。
  - 次にapplicationごとに、identityなら`_identity_application_phase`、それ以外なら中心命令の返り値（productは符号−1でπ）を足す。
  - 合計が0.0でなければ`p(合計)`を1個置く。
  - providerの関数を直接呼んでいる。中心命令の返り値を再現した部分は、providerの`_append_central_application`と一致することをtestで確認した。
- 全requestの後に、composeが付けるscalarの`p(−scalar·T)`を置く。

**PASSの条件**

- scalarが保存preparationの値と一致する。
- 実回路のancilla命令列が期待列と完全に一致する。これにより、欠落・重複・順序違い・request取り違えを検出する。
- scalarの`p`が制御進化の最後の命令である。
- X/Yでは、上記に加えて、両軸の制御進化が命令単位で同一である。

**注記**

- 位相の比較は、既存SOURCE（`instruction_parity`、旧`scalar_once`）と同じexactなfloat一致で行う。許容差は導入しておらず、他のgateも緩めていない。
- 同じrequestの区間内で、ancilla位相とsystem側だけのgate（基底変換）との前後は区別しない。両者は別のqubitに作用するので可換である。
- 旧`scalar_once`は診断値として残し、合否には使わない。

## 2. test（すべてPASS、新しいH4計算なし）

| file | 件数 | 内容 |
|---|---:|---|
| `tests/check_full_h_kappa2_p1_circuits.py` | 11 | 既存5件はすべて保全。追加は下記の6件 |
| `tests/check_full_h_kappa2_p1_prep.py` | 4 | 価格recordがBの種別になったこと。この木のguardとCの`c628e9e8` guardの両方で受理されること。旧ZZ・scalar不一致・price SOURCE不一致を拒否すること |
| `tests/check_full_h_kappa2_s1_caller.py` | 7 | 回帰 |

既存5件は、control=0・high-control・leakage・Hadamard・新ZZ・parity・one-body・provider byteを確認する。

追加した6件：

- order-0だけのtoyでPASSする。
- order-2を含むtoy（request別の位相3π・2π。eventの−1にproduct符号−1が加わる）で、event位相とscalarが共存してもPASSする。旧判定はfalseを再現する。
- providerの位相規則と一致する。
- scalarの重複・欠落・event位相より前・event位相がscalarの後、およびscalar値の不一致を拒否する。
- event位相の欠落・重複・他requestへの移動・request間の入替え、event記録の符号違い・requestの入替えを拒否する。
- X/Yで位相が違う場合、およびsystem gateが1個違う場合を拒否する。

さらに、order-2を含むtoyでpilotをend-to-endで実行し、既存gateがすべて保たれることを確認した。

## 3. 原本照合（読取専用）

- **public**：`run02/SOURCE.json`、`run02/P1_RESULT.json`、`RUN02_SCALAR_ONCE_DIAGNOSIS.json`は、`cd56422a8`のblobとbyte一致した（SHA-256は記録に保存）。
- **private**：job、child、H4_context、events、result、X/Yのcompile record・wrapper QPY・compiled QPYの11 fileがすべて存在し、SHA-256を記録した。再検証の前後で変化はない。
- **対応関係**（23条件。詳細は[REVALIDATION.json](REVALIDATION.json)の`conditions`）
  - SOURCE `72d4f4ff2`と、そのFull-H・native・wrapper・provider・lowering
  - configuration `54ac3eb3…`
  - candidate `FULL_H_4de6130f0d3a66e5d695`（q8/R71/s568/K2/T0.8）
  - 8 seed（eventとSOURCEの両方）
  - 8×71 eventで、order0が565、order2が3
  - child.jsonと公開P1_RESULTの一致、result.jsonと公開trialの一致
- **作用検証7件**：誤差＜許容、`phase_alignment=False`。
- **その他の一致**
  - 新ZZ：wrapper上で`crzz` 0、`crz` 568
  - parity：PASS。wrapper中の制御進化の命令数56,899が、parityの新回路の命令数と一致した
  - one-body：eventから数え直して205で、記録と一致
  - scalar・preparation hash・provider 9 moduleのSHA-256：記録と一致
  - **wrapper fingerprint**：QPYから`canonical_qiskit_circuit_fingerprint`で再計算した値が、file名・compile record・公開診断の値と一致した
  - **compiled価格**：compiled QPYからproviderの`_gate_counts`・`_filtered_depth`・depth・sizeで数え直した値が、compile recordとtrialに一致した。9 qubit、basisは`cx/rz/sx/x`（＋measure）
  - compiler hash `1239304c…`：nativeの`COMPILER_HASH`と一致

request別の対応表は[REQUEST_PHASES.json](REQUEST_PHASES.json)にある（request、event数、order、位相−1のevent番号、provider規則による角度、X/Yの命令番号と角度）。

## 4. 価格record・sidecar・C guard

**価格record**：[PRICES.json](PRICES.json)。X/Yとも、全compiled RZ 92,448、CX 27,719、RZ depth 26,079、CX depth 13,996、total depth 55,352、size 176,866。

**分類（Bの種別、`D_PRICE_RECORD_B_KINDS_V2`）**

| field | 値 |
|---|---|
| classification | `DIRECT_COMPILED_FINITE_SAMPLE` |
| expectation | `FINITE_COHORT_ONLY` |
| n_paths | 1 |
| law_expectation | false |
| price_status | `DIRECT_FINITE_COHORT` |
| law_context | 保存cohortのsampler（`PCG64_numpy_float64_CDF_saved_cohort`、`ideal_sampler_distance_qualified=false`） |
| diagnostic | 旧label（`DIRECT_SINGLE_PATH_WIRING`など）、期待値への昇格禁止、model点・G入力としては使わない |

**値の出所**：検証済みのcompile record（＝compiled QPYから数え直した値＝trial）から復元した。手入力はしていない。

**provenance（`source`）**

- run02のSOURCE（`72d4f4ff2`、Full-H・native・wrapper・provider・新ZZ）
- 原記録`run02/P1_RESULT.json`（commit `cd56422a8`）
- privateのcompile record（path、SHA-256、fingerprint、compiler）
- 受入（acceptance）：commit `8258b2200`、`scripts/revalidate_full_h_kappa2_p1_run02.py`、gate ID、`REVALIDATION.json`、原状態`STOP_WIRING_CHECK`、再検証PASS

**sidecar**：[SIDECARS.json](SIDECARS.json)。method `FULL_H_RTE`、native `H46_LONG_TIME_NATIVE_COMPOSE`、lowering `CONTROLLED_ZZ_CX_CRZ_CX_V1`（`01e068af`）、scalar quantity `HAMILTONIAN_CONSTANT_PLUS_EXTRACTED_FULL_H_IDENTITY`（Full-H `9b83`、`ONCE_AT_TOTAL_T`、resolved＝expected）、measurement、physical context。

**implementation evidence**：[IMPLEMENTATION_EVIDENCE.json](IMPLEMENTATION_EVIDENCE.json)。公開commit `6d188defd`に固定し、sidecarの`implementation_evidence`がそれを参照する。commitの仮置き値は残っていない。

**C guard**

- `c628e9e8`の`price_implementation_guard.py`を無変更で実行した。依存する`baseline_signal_pipeline.py`が同じbyteであることも確認した。
- policyは`D_FULL_H_KAPPA2_NEW_ZZ_P1_IMPLEMENTATION_ACCEPTANCE_V1`（FULL_H_RTE、新ZZ、上記のnative・scalar）。**P-1の実装受入の範囲に限るもので、比較scenarioとして採択したものではない。**
- 結果：`IMPLEMENTATION_POLICY_CHECKED_NOT_SCIENTIFIC_QUALIFICATION`、2件（3X、3Y）。
- 発行した実recordで負の対照を確認した。次の6件はいずれも拒否された（[EXECUTION.txt](EXECUTION.txt)）。
  - sidecarを旧ZZにする
  - scalarを不一致にする
  - evidenceのcommitを違うものにする
  - price SOURCEを不一致にする
  - axisのmeasurementを入れ替える
  - policyを旧ZZにする

引渡しは[HANDOFF_C.md](HANDOFF_C.md)。

## 5. 実行件数・資源

| 段階 | 内容 | wall | peak RSS |
|---|---|---:|---:|
| metadata preflight | 原本の存在とbyteの照合 | 数秒 | — |
| `--revalidate` | 1回。子processがQPY 4個を読込み、fingerprint 2件、数え直し2件、判定を行う | 3分24.7秒 | 7.64 GB（子process。QPY・fingerprintの一時dataによると推定） |
| `--issue` | 1回。価格・sidecar・C guard | 1.6秒 | — |

- 新しいsampling、回路build、statevector、compileはいずれも0。
- 本番実行の前に、同じ読取専用の子process計算を、scratchpadへの出力でdry-run確認した（同じSOURCE bytes、科学計算なし）。

## 6. 残るUNKNOWNと範囲外

- **価格の位置付け**：単一trajectory・単一seedのdirect価格である。Full-Hの期待価格\(\mathbb E[C]\)、order-2 eventの出方による価格のばらつき、law平均は未評価。P-2の較正点、G、方式比較には自動では使わない。
- **compile時間**：axisごとのcompile時間は、原run02で記録されておらずUNKNOWN。
- **sampler**：実samplerと理想lawの距離は未資格（`ideal_sampler_distance_qualified=false`）。
- **比較policy**：比較scenarioとしてのpolicyの採択は、ユーザーの判断事項である。
- **mandatory STOP**：P-2、S-2、N/MSE/G、価格modelのfit、run03へは進まない。
