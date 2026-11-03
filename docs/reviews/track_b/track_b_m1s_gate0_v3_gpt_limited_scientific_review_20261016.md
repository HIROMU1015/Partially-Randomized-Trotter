# Track B／M1-S Gate 0：v3数値精度契約のGPT限定科学レビュー

**日付**：2026-10-16（JST）  
**実施者**：GPT-6  
**レビュー区分**：Level 2（既採択の科学scope内での品質条件・数値coverage・runtime契約の限定判断）  
**開始承認**：ユーザー「レビューを開始して」  
**結論**：`V3_AS_IS_NOT_ADOPTED_RECOMMEND_ABSOLUTE_ORACLE_QUALITY_GATE_AND_CONDITIONAL_ENGINEERING_COVERAGE_REVISION_NO_SCIENCE_EXECUTION_MANDATORY_STOP`  
**権限**：このレビューは科学的推奨であり、ユーザーによる実験契約採択、実施authorization、grantやmarkerの作成ではない。新規科学数値計算・保存数値の再解析・数値correctnessテスト・GitHub書込みは行っていない。**mandatory STOP継続**。

---

## 0. 結論と実務上の判断

**現在のv3契約候補をそのまま正式採択することは推奨しない。** ただし、既存の限定Gate 0科学scope（H4 1.00 Å／STO-3G／DF rank12、保存生成順G3・G6・G9、非冗長Pauli-tail、保存数値oracleでの\(W,v,\rho\)）を見直す必要はない。実施に向けて**契約の小規模修正**を行う方針を推奨する。

- **Oracle残差品質**：未決の\(\eta_{\rm oracle}\)を恣意的に数値化せず、当該保存state生成SOURCEに記録された**絶対残差受理基準\(10^{-8}\,\mathrm{Ha}\)**を、*数値oracleに対するprovenance整合の品質gate*として再利用する契約案を優先する。\(\varepsilon_{\rm state\ residual}=10^{-8}\,\mathrm{Ha}+B_{\rm res}\)を検討する。ただし、旧solverと新referenceの算術誤差差・再構成差を別途記録し、**厳密固有状態への距離・ground性・非縮退性の保証にはしない**。
- **誤差coverage**：8種類の事前登録式は、凍結binary64 inputと指定算術モデルをtargetにする**条件付きengineering error envelope**として原則利用可能。ただしIEEE演算・複素BLAS・`math.fsum`・underflow/FTZ等の前提が実環境で確認されていないため、**rigorous numerical certificateとしては不受理**。条件付きのstatus/ledgerと失敗時STOPを必須とする。
- **17項目の品質**：既存8 scalarを「構造チェック／数値品質／報告分解能」に分類して保持できる。特に\(\varepsilon_W=\varepsilon_v=2^{-24}\)、\(\varepsilon_\rho=2^{-12}\)は**報告分解能**であって、研究の有意性・総資源改善の閾値ではない。8 formulaは**条件付き採用候補**。残差品質gateを修正したv4候補の整合確認後に一括採択する。
- **runtime/resource**：CPU1 worker・各種thread1・GPUなし、4 GiB virtual address、result/log各256 MiB、\(M=65536\)、contribution cap\(10^6\)は、**安全側の技術的制約案として概ね妥当**。ただしPython 3.11.0rc1／NumPy 1.26.4の実import・BLAS経路・制限の実効性・初回数値correctnessは未検証。capは「確実に完走できる」証拠ではない。**失敗したら条件を変えず保存・STOP**する。
- **科学結果**：\(W,v,\rho\)がすべて得られない可能性もある。その場合は`PRECISION_UNRESOLVED`等の**技術的不確定**であり、M1-Sの資源改善余地がないという結果ではない。\(\rho\)単独の科学GO／STOPは引き続き禁止。

**正式採択の推奨順序**：Codexが上記残差gateの設計変更と文書・SOURCE・静的検査を行う → ユーザーが17項目・条件付きcoverage・runtime/capsを含む最終契約を採択する → **別のユーザー明示承認**により、初回数値correctnessから始めるGate 0 one-shotを限定実施する。これらを本レビューだけで自動実施しない。

## 1. 固定証拠・取得状況

Repository：`HIROMU1015/Partially-Randomized-Trotter`。

| 資料 | 固定ref／取得 | 役割 |
|---|---|---|
| [v3実験契約候補](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e749bf8844f5d78817f18f738ade87269c2d70e1/docs/tracks/algorithm_codesign/m1s_gate0_headroom_execution_contract_v3_candidate_20261016.md) | commit `e749bf8844f5d78817f18f738ade87269c2d70e1`、全文取得 | 契約・guard・未採択事項 |
| [誤差算定式・coverage設計](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e749bf8844f5d78817f18f738ade87269c2d70e1/docs/tracks/algorithm_codesign/m1s_gate0_preregistered_error_formula_design_20261016.md) | 同commit、全文取得 | 8 formulaと残差品質未決条件 |
| [machine contract v3](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e749bf8844f5d78817f18f738ade87269c2d70e1/artifacts/track_b_m1s_gate0_formula_contract/2026-10-16/v1/execution_contract_v3_candidate.json) | 同commit、全文取得 | `adopted=false`、coverage/η未採択、SOURCE/cap |
| [未決事項記録](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e749bf8844f5d78817f18f738ade87269c2d70e1/artifacts/track_b_m1s_gate0_formula_contract/2026-10-16/v1/unresolved_items.json) | 同commit、全文取得 | 8 formula候補＋品質係数1件 |
| [budget.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e749bf8844f5d78817f18f738ade87269c2d70e1/src/trottertracks/algorithm_codesign/m1s_gate0_v3/budget.py) | 同commit、全文取得 | `Fraction`、予算式、`global_limits/law_limits`、gate前計算 |
| [science.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e749bf8844f5d78817f18f738ade87269c2d70e1/src/trottertracks/algorithm_codesign/m1s_gate0_v3/science.py) | 同commit、必要全処理を静的読解 | 再構成・残差・W/v/ρ・区間・STOP |
| [control.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e749bf8844f5d78817f18f738ade87269c2d70e1/src/trottertracks/algorithm_codesign/m1s_gate0_v3/control.py) | 同commit、schema/permit関連を静的読解 | 契約adoptionと別実行authorizationのguard |
| [既存runtime契約候補](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4019d045cbfad94ba9333b544f7950a6d10d467d/docs/tracks/algorithm_codesign/m1s_gate0_runtime_and_resource_contract_candidate_20261016.md) | 固定commit `4019d045…`、取得 | caps・runtime・一回限り実行 |
| [H4保存生成SOURCE](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c644925b50587072784846df09bf02c39e8453e1/src/trotterlib/pr2_s0_s1_validation.py) | commit `c644925b…`、行823–837を確認 | H4 1.00 Å生成に`tol=1e-12`を使用 |
| [旧sector solver SOURCE](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c644925b50587072784846df09bf02c39e8453e1/src/trotterlib/df_hamiltonian.py) | 同commit、行635–750を確認 | `converged and residual_norm<=max(100*tol,1e-8)` |
| [限定科学scope採択記録](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e749bf8844f5d78817f18f738ade87269c2d70e1/artifacts/track_b_m1s_gate0_precision_contract/2026-10-16/v1/gate0_scientific_scope_adoption_record.json) | 同commit、全文取得 | 対象・law・oracle診断が既採択、計算は未承認 |
| [Checkpoint 2 GPT原本](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/62554ee127a2bec1206367cf1754df2ff49fa000/docs/research/track_b_m1s_g0t1_checkpoint2_gpt_independent_scientific_review_20261016.md) | 以前に確認・ローカル原本参照 | small-step headroom診断の科学的境界 |
| [2026-10-16統合共通指示](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e749bf8844f5d78817f18f738ade87269c2d70e1/docs/research/research_project_instructions_unified_2026-10-16.md) | 同commitと提供ローカル本文を確認 | 独立レビュー承認・計算の別承認・STOP |

GitHubのbranch `track-b-m1s-gate0-formula-contract-20261016`のHEADが固定commit `e749bf8844f5d78817f18f738ade87269c2d70e1`と一致することを確認。関連の公開receiptを取得し、その記録上のstatic PASSと計算0件を区別した。**公開全ファイルのSHA256を今回独立に再計算・再検証したものではない**。科学SOURCE・NPZ数値・実行機器の直接動作検証は未実施。

## 2. 研究目的と受理済み科学scope

今回の目的は、凍結H4数値stateに対し、固定G3/G6/G9の非冗長Pauli-tailから
\[
W=1-\langle h^2\rangle_{\psi},\quad
v=\mathrm{Var}_P\langle P_{\rm signed}\rangle_\psi,\quad
\rho=\frac{W+v}{2W},\qquad h=H_R/\lambda_R
\]
を求め、**small-stepの先導visibility headroom**を診断すること。保存数値vectorに対するoracle診断であり、真の全H基底状態、全Fock非縮退性、共鳴のない(SP)条件、finite-step RPE／shot／native資源の保証ではない。この区別はS0・Checkpoint 2と整合する。

保存状態の品質を評価する残差と、保存状態上の\(W,v,\rho\)の*数値算術誤差*は別物である。残差が小さくても、gap情報なしには\(\psi\)から真の特定固有状態\(|k\rangle\)への距離を上界化できない。従って\(\eta_{\rm oracle}\)をどのように採択しても、それだけで定理の適用条件をcertifyしない。

## 3. \(\eta_{\rm oracle}\)と残差gateの判断

### 3.1 現v3の式と本質的な限界

v3は
\[
\varepsilon_{\rm state\ residual}=\eta_{\rm oracle}S+B_{\rm res},\qquad
S=|c|+\sum_{pq}(|\Re h_{pq}|+|\Im h_{pq}|)+\sum_l|\lambda_l|G_l^2
\]
とし、\(B_{\rm res}\)は保存binary64参照からの演算誤差、\(\eta_{\rm oracle}S\)は別の数値状態品質の許容残差を表す。後者は丸め解析から決定できないため、null維持は適切だった。

**独立した科学的指摘：残差品質基準\(\eta S\)は定数エネルギーシフトの取り方に依存する。** 正規化状態のRayleigh残差を\(r(H,\psi)=\|(H-\langle H\rangle_\psi)\psi\|/\|\psi\|\)とすると、任意の実数\(\kappa\)について
\[
r(H+\kappa I,\psi)=r(H,\psi)
\]
である一方、\(S\)の\(|c|\)項は\(c\mapsto c+\kappa\)で変わる。固定H4入力なら一意に評価できるが、\(\eta S\)を物理的に普遍な「状態品質」として解釈するのは不適切である。\(B_{\rm res}\)が\(S\)に依存することは、丸め誤差がエネルギーoffsetやcancellationに依存し得るので別問題として許容される。

### 3.2 既存の正本から得られる客観的な品質基準

H4 snapshot生成SOURCEは`solve_df_ground_state(..., tol=1e-12)`を明記する。元solverの返却`converged`の条件は`converged and residual_norm <= max(100*tol,1e-8)`。従って**元の保存状態を受理する生成時の数値的な基準は\(10^{-8}\,\mathrm{Ha}\)という絶対残差ノルム**である（元solverの計算路と定義におけるもの）。これは新たに任意の\(\eta\)を作るより、provenanceに基づく科学的根拠が強い。

**推奨する改訂案**：状態品質の部分を\(r_{\rm oracle,quality}=10^{-8}\,\mathrm{Ha}\)という絶対固定値にし、将来の新しい参照／残差の丸め誤差\(B_{\rm res}\)を独立計上する。
\[
\boxed{\varepsilon_{\rm state\ residual}=10^{-8}\,\mathrm{Ha}+B_{\rm res}.}\tag{3.1}
\]
この変更には`budget.py`、machine契約・式ID・採択記録の*新しい版*への更新が必要。既存v3へ後付けで値を注入しない。元solverの残差は異なるmatrix-free演算経路によるため、旧残差評価そのものの算術誤差を厳密に押さえたcertificateではない。**(3.1)は数値oracleの品質screenであり、既存source受理基準を再掲したengineering policy**である。

代替として既存v3形式を維持するなら、\(\eta_{\rm oracle}:=(10^{-8}\,\mathrm{Ha})/S\)を、*結果を見る前・凍結入力尺度から決定する導出式*とする必要がある。ただし\(S\)を元資料に対して評価する工程自体は、別途科学実施承認前に開始しない。科学的な意味が明確なのは(3.1)の絶対品質gateである。

### 3.3 残差と\(W,v,\rho\)の報告区分

- 残差quality gate PASS：数値solver由来のgross residualについて指定基準内という情報。**全Hの真のground/nondegeneracyやstate距離、真のgroundの\(W,v,\rho\)誤差を証明しない**。
- 残差quality gate FAIL：新しい参照演算路・保存入力との不一致、state品質、数値丸めの可能性を分け、証拠保全後STOP。基準を自動緩和して継続しない。
- 保存stateに対する\(W,v,\rho\)の算術精度PASS：あくまでそのvectorと凍結operatorをtargetとするconditional engineering品質。eigenstate headroomの物理的認定ではない。

もしユーザーが真の固有状態headroomの定量保証を要求するなら、gapや適切な固有空間の別認証が必要で、これはGate 0の初期scopeを変更するため別の科学判断・承認に戻す。

## 4. 条件付きroundoff coverageの検証

### 4.1 受理できる範囲

v3は通常のunit roundoff \(u=2^{-53}\)、\(\gamma_m=mu/(1-mu)\)（\(mu<1\)）を使い、fractionで予算を算定してbinary64へ上向き変換する。数値許容値は観測された\(W,v,\rho\)、残差、再構成差を使わず、凍結inputの演算尺度とcanonical Pauli lawのみから、評価の**前**に決定する。構造としては結果依存の事後的閾値調整を避ける設計であり、受理できる。

条件は、複素BLASが保守的\(\gamma_{8k+32}\)以内、`math.fsum`が\(\gamma_{m+32}\)×入力絶対値和以内、正しいbinary64/丸めmode、未会計のunderflow/FTZなし、one_bodyとgの**exact Hermitian入力**、SOURCE/runtime/hash同一性である。この条件が成り立つなら、式群は*安全側を狙ったengineering誤差管理*として合理的な出発点である。SOURCEの静的検査PASSとhash固定のみでこれらのcoverageを**証明したことにはならない**。Higham–Maryの\(\gamma_m\)記法は標準的だが、個別のPython/NumPy/BLASコード経路全体を自動的にcertifyする定理ではない。

条件付きerror ledgerを**`CONDITIONAL_ENGINEERING_NOT_CERTIFIED`**相当のstatusで必ず出力し、成功値に対して「verified exact error bound」「true-eigenstate interval」のラベルを付けない限り、**exploratory oracle headroom診断用として条件付き採択を科学的に推奨する**。数学的に厳密な誤差certificateを要求する実験なら現v3は不十分であり、独立の丸め検証・interval法・runtime認証が必要となる。

### 4.2 特に注意すべき前提とfailure意味論

- `science.prepare`が`np.array_equal(matrix,matrix.conj().T)`を要求するため、元gまたはone_bodyが浮動小数点レベルで非厳密Hermitianなら、式経路は**冒頭でtechnical STOP**する。これは科学否定ではなくsource入力モデル不成立を示す結果である。観測後に勝手にHermitian射影、閾値緩和、別lawへ変更しない。
- \(W\)・\(v\)の前に予算を解決し、\(e_W>\varepsilon_W\)または\(e_v>\varepsilon_v\)ならその分割は\(W,v,\rho\)未評価になる。保守的算定式により**全3分割が未評価で終わる可能性**がある。これはprecision infeasibilityであり、改善余地の有無の証拠ではない。
- `budget.py`のFractionは*予算式自体の丸め不足*を制御する実装であり、\(\gamma\)の前提やbudget導出の網羅性までは保証しない。
- `science.py`は既存参照とJWの別実装を比較する設計だが、双方の概念的前提が一致していること、誤差モデルが演算路を覆うことを、静的検査だけで無条件に認定できない。最初の数値correctness gateは不可欠。
- Silent underflow/FTZ、BLASによる算術順序・中間精度、入力scaleの見積りの十分性などは**未検証仮定**として報告し、これに条件付けた数値を証明済み精度へ格上げしない。

## 5. 17項目の採択候補としての位置付け

| 区分 | 対象 | 判定 |
|---|---|---|
| 構造上の厳密確認 | `eps_basis=0`, `eps_hermiticity_factors=0` | 候補維持可。ただし厳密配列一致を要求するので、元入力が完全対称でなければSTOP。値0が一般的roundoff保証という意味ではない |
| scalar品質条件 | `eps_state_norm=2^-40`, `eps_prob=2^-46`, `eps_imag_expectation=2^-40` | 候補維持可。結果を見た閾値修正は禁止。quality gateと真値の誤差certificateを区別 |
| 報告分解能 | `eps_W=eps_v=2^-24`, `eps_rho=2^-12` | 候補維持可。科学的headroom閾値でも達成保証でもない。厳格すぎて未評価が出ればprecision unresolvedとして保全 |
| 条件付き算定式8件 | `eps_H`, `eps_hermiticity_H`, `eps_phase`, `eps_imag_coeff`, `eps_imag_energy`, `W_error_budget`, `v_error_budget`, `W_floor` | **条件付きengineering用に採択を推奨**。forward model未実証、少分母／one-body/g Hermiticityを別技術gateにする |
| Oracle残差品質1件 | `eps_state_residual` | **v3の\(\eta S+B_{res}\)はそのままの採択を推奨しない。** 元SOURCEに基づく絶対\(10^{-8}\,\mathrm{Ha}+B_{res}\)へ改訂推奨 |

\(\rho=(W+v)/(2W)\)では、\(W\)の不確かさが小分母で増幅する。SOURCEの\(W,v\)矩形区間とrho区間は、入力予算自体がcoverageを持つ場合の**engineering感度**である。外向き`nextafter`は端点の浮動誤差を軽減するが、元予算のcoverage不足を解決しない。\(\rho\)区間が目標分解能を満たさない場合は`RHO_PRECISION_UNRESOLVED`に分類し、得られたraw数値を科学的に解釈しない。\(W=0\)・\(\lambda_R=0\)・区間が0を含む場合は`NOT_APPLICABLE`／`UNDETERMINED`を区別する。

## 6. Runtime/resource設計の科学的評価

提案：Python **3.11.0rc1**（pre-release版であることを明記）、NumPy **1.26.4**と主要build hash、CPU worker=1、BLAS/OpenMP等1thread、GPUなし、memory `RLIMIT_AS=4 GiB`、result/log各256 MiB、context4 MiB、Pauli terms cap65536、contributions cap1000000、科学wall-time上限なし。SOURCE設計の規模（8 mode／256状態／G3,G6,G9）に対して過度な並列化を避けるのは合理的である。

**判定**：上限の*技術的保護条件*として条件付き推奨。**実行成功を保証する数値的performance予測ではない**。特にPython pre-release/NumPy実import、BLAS build経路、`RLIMIT_AS`が実環境でどう作用するか、native log captureとoutput limit、例外時のresult保存は未検証。初回実行の技術監査として、開始前に固定構成を照合し、failしたら結果や実行contextを保全してSTOPする必要がある。

`wall_time_seconds=null`は今回のスコープで直ちに誤りではないが、無期限の科学処理を安全と証明する意味ではない。運用時の監視・中断手順、one-shot失敗後に自動retryしない条件を固定する。Runtimeやcapを都合よく実行後変更しない。未承認のpilot／benchmarkで先に値を調整しない。

## 7. 研究結論とexperiment/technical STOPを分離

| 状態 | このGate 0で意味すること | 許されない読み替え |
|---|---|---|
| 入力／SOURCE／scope／authorization/hash不一致 | `TECHNICAL_STOP` | 研究方法の一般的no-go |
| 入力のexact Hermiticity条件FAIL | `INPUT_MODEL_NOT_APPLICABLE_TECHNICAL_STOP` | Pauli/involution理論の反例 |
| 新旧Hamiltonian再構成FAIL | `RECONSTRUCTION_UNRESOLVED`＋STOP | 項内分散改善がないという結論 |
| 保存state residual FAIL | `ORACLE_QUALITY_UNRESOLVED`＋STOP | 厳密固有stateの不在、M1-S no-go |
| 予算\(e_W/e_v\)超過・小分母・rho区間幅超過 | `PRECISION_UNRESOLVED`または`NOT_APPLICABLE` | \(\rho=1\)・改善余地ゼロ |
| 数値的に\(W,v,\rho\)取得 | `HEADROOM_DIAGNOSTIC_ONLY` | RPE／native資源／true-groundの優位証明 |

Gate 0の成功・失敗いずれでも**3分割全行**を記録し、分割・入力・閾値を変更して再試行しない。成功値だけを抜き出してposteriorな研究GOを宣言しない。\(\rho\)のみでM1-S資源研究全体のSTOPを自動化しない。Checkpoint 2 GPT判断を維持する。

## 8. 具体的な次作業・承認条件

### 8.1 次の担当：Codex（技術契約の限定改訂）

1. v3原本・採択済み科学scope・旧solver SOURCEを保全し、**新しいv4契約候補**として残差gateの品質部分を`absolute_original_solver_acceptance_v1`等へ改訂する。提案式\(10^{-8}\,\mathrm{Ha}+B_{res}\)を採る場合、元solverとの数値算術経路差・旧残差と新評価の意味の違い、reference誤差台帳を記載する。
2. 既存の8 formula、8 scalar、rate/rhoの報告範囲、runtime/caps、one-shot guard、SOURCE/contract/auth三段階を維持する。`coverage_model_adopted`はユーザー採択までfalseとし、将来は`CONDITIONAL_ENGINEERING`という証拠区分を実際の結果へ残す。
3. SOURCE、contract、machine candidate、authorization schema、結果台帳、静的チェックを**新version／新namespace**で整合させる。実データのPauli係数生成、数値correctness、run、回路、compileは実行しない。
4. Runtime smoke／science correctnessを今回の静的検査で実施した扱いにせず、別実施承認後の初回gateへ残す。成果をcommit・pushし、GitHub bytes/SHAと親worktree・旧結果を保全してmandatory STOP。

### 8.2 次の担当：ユーザー（契約採択）

Codexのv4資料で、以下を明示的に確認・採択する。

- 科学scopeは既採択どおり、保存数値oracleの品質gateは**旧solverの絶対残差基準を用いる**こと。結果と定理のground-state保証を混同しないこと。
- 誤差算定式8項目を**条件付きengineeringモデル**として採択し、BLAS等が未認証でも科学的certificateと呼ばないこと。実際の条件不一致・精度不足は技術STOPで保全すること。
- 8 scalarの構造／品質／報告分解能、全17項目、runtime・上限、全3分割の出力、retry・STOPを最終実験契約として採択すること。

**採択するだけでは科学計算を開始しない**。

### 8.3 次の担当：ユーザーの別実施承認後のCodex（Gate 0）

固定SOURCE・採択済み契約・個別実施authorizationを結び、**初回数値correctness（入力、JW／bit-reference再構成、input Hermiticity、state/residual、precision coverage）を最初のgate**として実行する。合格・precision-resolvableな分割に限り\(W,v,\rho\)を報告し、全対象をstatus付きで残してmandatory STOP。結果が`UNRESOLVED`なら条件を事後緩和せず、次の科学判断へ戻す。

**追加のGPT詳細レビューは、Codexが上記の採択可能なv4へ忠実に具体化しただけなら原則不要。** 改訂により評価量・科学scope・誤差の意味・基準の合理性が変わる場合にのみ再判断する。今後の研究継続／T1c／technical noteはGate 0の結果を踏まえ別途判断する。

## 9. 本レビューの範囲と監査

- 固定GitHubのv3契約・算定式全文、候補machine JSON、未決一覧、S0後の採択記録、SOURCEの主要数値／guard処理、旧state生成／solver sourceを読解した。
- 算定式の各定数があらゆる内部BLAS最適化・rounding modeで数学的に厳密なerror enclosureを与えるとの証明は行っていない。SOURCEの静的整合と数値correctnessを区別した。
- 独立して確認した新しい論点は、\(r(H+\kappa I,\psi)=r(H,\psi)\)に対する\(\eta S\)のshift依存、および旧solver `tol=1e-12`・絶対残差受理`1e-8 Ha`の出典である。
- 新規科学数値計算・NPZ科学配列処理・予算算定式の実数値評価・eig/gap・sampling／simulation／circuit／compile／benchmarkは**0件**。GitHub SOURCE・grant・marker・原結果・採択済み契約へ書き込んでいない。
- **mandatory STOPを維持**。本レビューの科学的推奨を実験契約採択／実施承認と扱わない。

## 10. 参考文献

- Higham & Mary, *A New Approach to Probabilistic Rounding Error Analysis*, SIAM Journal on Scientific Computing **41**, A2815–A2835 (2019), https://doi.org/10.1137/18M1226312 。古典的な\(\gamma_n\)の背景と確率的な緩和を区別。今回のcoverageは確率的独立性を仮定して\(\sqrt n\)へ置換する方式ではない。
- NumPy, `finfo`（`eps=2^-52`、roundoff\(u=2^{-53}\)との区別）https://numpy.org/doc/1.26/reference/generated/numpy.finfo.html 。
- Python documentation, `math.fsum`, https://docs.python.org/3.11/library/math.html#math.fsum 。浮動小数点和の実装依存性に関する注意。
- GitHub固定一次SOURCEと契約・監査：§1の各リンク。

---

**最終結論**：`LIMITED_SCIENTIFIC_REVIEW_COMPLETE_V3_AS_IS_NOT_ADOPTED_V4_RESIDUAL_POLICY_REVISION_RECOMMENDED_NO_NUMERICAL_EXECUTION_MANDATORY_STOP`。
