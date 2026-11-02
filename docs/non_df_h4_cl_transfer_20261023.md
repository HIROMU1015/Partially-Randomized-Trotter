# H4 C/L誤差会計・literal wrapper保証転送・40軸接続

2026-10-23系列の技術作業。基点branch `codex/non-df-h4-local-diagnostic-20261023`、commit `de94417e7e1a0b984b93b81aba673b5ef94fdf98`から、新branch `codex/non-df-h4-cl-transfer-20261023`に分離した。**新技術SOURCEは`0880f4af98b125d62599e970db89fc33634d1b61`。C/Lの会計・literal転送規則・40軸×14 gateの接続を実装し、人工・回帰202件PASS。実H4数値評価・新grant・compile・正式受理・N/Gは0。C/LはCOMBINATION_BLOCKED、旧compilerはMISSING_EVIDENCEのままmandatory STOP。**

## 固定資料・SOURCE・再利用

GitHubから基点commit、主資料・Cの40軸・normalization・SOURCE/契約・K調査・実装等28 fileを取得した。大きいgate_bindings/C sealのcontentsが空だった2件は固定blobを取得し、返却された全UTF8 bytesのGit blob SHAを計算してローカルと照合した。[取得とSHA照合](../artifacts/non_df_h4_cl_transfer/2026-10-23/github_verification.json)。取得したことと数学的証明を分離する。

固定科学SOURCEはC `e80d7510880f8db993b777c627bb8e95b05a070d`、L `eb195bf2dedd0aed9d80ad7f84948b59ea53d854`、元H4 `0c329b30a81accb35c212be8defa66fee0dd0f12`。Linear H4/隣接1.00 Å/STO-3G、8 system qubit/Nα=Nβ=2、Pauli184項/PR prefix27、DF tol-only1e-8/rank10/PR prefix1＋one-body、S2/T=.8/q8/r4/K4、既存5精度と両軸を変更していない。H2 S2/Yoshida4系列とは独立。

最新版運用指示2026-10-23 §8.4/§10と共通AGENTSを適用した。[運用指示SHA](../artifacts/non_df_h4_cl_transfer/2026-10-23/instruction_receipt.json)。次を再利用した。

| 既存資産 | 今回の使用 | 変更 |
|---|---|---|
| h4_evidence_binding.py | EvidenceStore、exact endpointの構文検査、40軸のSOURCE/target/fact結合とverifier | なし |
| h4_certificate_gates.py | SHA/path/JSON、phaseを含むword等価性、control/basis/Hadamard/readoutの形式規則 | なし |
| L raw/CSV・coverage | 148 forward/148 inverse/11 basis/324 supportとPF/PR一様上界のSHA・方向・順序・scope接続 | 数値再評価なし |
| C保存結果・SOURCE | bias_distance/semantic/normalization/law/referenceの定義と保存tokenの接続 | 数値再集計なし |
| T67＋binding33＋L48 | 既存人工・回帰テストをそのまま使用 | なし |
| C/Lの旧監査・grant・STOP | SHA保全、旧native状態とmtime失敗履歴の維持 | なし |

既存word checkerは角度係数の多項式や異なる誤差辺の会計を表現できないため、必要な差分だけを新規追加した。[h4_transfer_rules.py](../src/trotterlib/h4_transfer_rules.py)はoccupation projectorのbit²=bitによるexact形式多項式、typed coverage、条件付きmeasurement transferを担当する。[h4_cl_binding.py](../src/trotterlib/h4_cl_binding.py)は既存40×14構造を保ったoverlayを作る。NumPy/FLINT/Qiskit/科学accountantをproduction adapterからimportしない。

## A：C/L primitiveを同じ名前だけで同一視しない

[primitive SOURCE追跡JSON](../artifacts/non_df_h4_cl_transfer/2026-10-23/check03/primitive_coverage.json)、[対応CSV](../artifacts/non_df_h4_cl_transfer/2026-10-23/primitive_coverage.csv)に関数・行・SHA・数学的対象・不足命題を記録した。

Cの`semantic_bound`は、経験的なfitや仮置きallowanceではなく、次の解析的な候補をArbで包含する実装だった。

- `basis_reuse`：`N*d*c^(2N)`。`d=max ||J_b F_b-I||_F`、`c=max(1,||F_b||,||J_b||)`として、丸めbasisの隣接打消しを扱う。
- `phase_accumulation`：`gamma_m*(1+A)+m*2^(-1074)`、`gamma_m=m*u/(1-m*u)`、`u=2^(-52)`。有限straight-line演算を仮定したworst-case候補である。
- `pi_rounding`：PRでは`2*q*r*|binary64_pi-exact_pi|`、PFでは0。
- これらをさらにnorm増幅で覆い、`_tier`が`state_norm*abs(B_saved)`を掛けてsemantic allowanceとして保存する。
- C軸totalは保存`bias_distance`上端＋semantic allowance上端を既に含む。Exporterの`semantic_error_bound_exact='0'`という宣言から、実際の保存allowanceが0とは推論できない。

一方Lは、丸めたRZ/XXPlusYY局所行列と、**同じ保存binary64角度をexact realとした理想gate**の差、forward/inverseのtelescoping、`f+i+fi`、PF/PRの一様basis伝播を保証する。Cの`JF-I`とLの`F-U`/`J-U†`は異なる演算子差である。

したがって、旧ledgerが共有する`basis_reuse`という名前は保守的な衝突警告であり、同じ数学的誤差が二度含まれたという証明ではない。逆に、名前が違うだけで非重複とも言えない。**演算子の始点/終点、operator normとsignal、stateの正規化、B scale、event familyを合わせる必要がある。** 異なる連続辺をtriangleで結ぶ場合は同じ物理的原因が現れても条件付き上界として成立し得るが、実H4の辺の対応はまだ証明できていない。

さらに、Cの`physical_bias=(S_encoded-reference)/state_norm`に対して、semanticは`state_norm*B*delta`として加えられる。Operator差からraw期待値への転送にはnorm係数nがあり、同じrayで正規化するとnは打ち消される。保存式のn係数が安全に支配する条件は別に必要である。**今回は保存nの大小・誤差の大きさを評価せず、n≥1や無害性も仮定しない。実H4で不適合が起きたと確定したものではなく、未証明の適用条件として残す。**

### 会計verifierと4つの方法

[比較・十分条件・失敗条件](../artifacts/non_df_h4_cl_transfer/2026-10-23/coverage_design.json)。Verifierは`PROVED_DISJOINT / PROVED_COVERED / OVERLAP_UNRESOLVED / MISSING_PROOF / NOT_APPLICABLE`を区別する。形式的triangle-chainの`PROVED_DISJOINT`は別会計辺の整合性を意味し、物理noiseの独立性を主張しない。`PROVED_COVERED`は同じ辺・同じbound参照を重複して記録した場合のdedup規則までで、CがLを支配するという数値証明ではない。

| 方法 | 必要な十分条件 | 現在 |
|---|---|---|
| CがLを既に包含 | 同一演算子差・observable・state/scale/familyでC boundの支配を証明 | 未証明 |
| 非重複primitive分解 | 各辺の上界と被覆の完全性を証明し、単位とnorm/scaleを揃える | Cの個別数値項は未保存、辺対応も未証明 |
| Joint enclosure | 保存`bias_distance`を直接再利用し、C encoded mean→理想family→literal readoutの追加辺だけを証明 | 条件付きの技術案。研究上の採用・数値実行は未承認 |
| 分離保持 | 元C total/L上界を保全し、結合しない | 今回適用：COMBINATION_BLOCKED |

Joint案ではC totalからsemanticを引く必要がない。独立に保存されている`bias_distance`を境界として使う。ただしfamily mapping、正規化、literal phase/control、B_saved、最終予算を証明するまでは、新しいtotalを作れない。今回は減算・相殺・単純加算・40軸の予算比較を行っていない。

## B：Literal wrapperへ転送できた範囲

[11項目のSOURCE・phase/control/B対応表](../artifacts/non_df_h4_cl_transfer/2026-10-23/diagonal_control_phase_B_map.csv)、[formal lemma](../artifacts/non_df_h4_cl_transfer/2026-10-23/check03/formal_lemmas.json)、[固定Qiskit source対応](../artifacts/non_df_h4_cl_transfer/2026-10-23/literal_runtime_source_map.json)。

実装・人工検証で確認したexact/conditional規則は以下。

1. `n_k=(I-Z_k)/2`からone-body diagonalのglobal phase/RZ、および`lambda*(sum eta_k*n_k)^2`のglobal phase/RZ/RZZを形式多項式で証明した。DF fragmentをPauli involutionへ置換しない。
2. Controlled global phaseをancilla相対位相へ保持する。Zero branchとone branchを区別し、位相を落とす誤ったcontrol証明を拒否する。
3. Ordered basisの逆順・adjointを既存word checkerで確認し、丸めforward/inverseは別々のL誤差として扱う。
4. PR order0/2/4の`(-i)^k`、product sign、signed rotation/identity phaseを保持する。SOURCEの`unsigned_rotation_angle=coefficient_sign*rotation_angle`との対応を明記する。
5. 固定QiskitのCRZ定義は`RZ(theta/2),CX,RZ(-theta/2),CX`、RZZは`CX,RZ(theta),CX`であり、その理想exact-real phase/parity恒等式を確認した。ただしbinary64半角生成・generic controlled lowering・実際に選択された分解は未証明。
6. `H-CU-H`と`H-CU-Sdg-H`の`p0-p1=Re/Im`規則を再利用し、人工入力でも位相符号を確認した。Ancilla初期化・normalized preparation・ideal Born readout・実際のcontrolled U対応は仮定/別obligationとして保持する。
7. 両branch誤差eta0/eta1に対し、cross期待値の差を`eta0+eta1+eta0*eta1`で覆う条件付き規則を実装した。Zero branchをexact Iと証明できる場合だけeta1へ簡約する。

**これらはideal operator/形式規則と限定SOURCE定義の対応であり、実H4 literal wrapper全体への保証転送は未完成。** Cで使用した丸めbasisを無条件にunitaryとしない。過去の有限probability一致も一様上界に格上げしない。

### B_savedとnormalization

Nominal B、saved B、保存`log_B_upper`、implemented CDF massesを別の証拠として接続した。Bitを`B_saved*(1-2*b)`と記録すれば±B_saved支持となるが、それはencoded eventのnorm1やencoded meanとの一致を意味しない。

Normalized rayと正しいoperator-transferが証明できれば、測定scaleへの転送は記号的に`B_saved*delta`となる。実H4の新しいB_scaled上界は計算していない。Cのnominal log_B包含が、実際のsaved Bまで同じupperで覆うことは自動的には従わず、別のscale bindingが必要。IID uniform words/fresh independent shotsの従来の条件付き仮定を維持し、PCG64の無条件独立性証明を新しい必須条件にしていない。

## C：既存40軸×14 gateへの接続

[560行のmatrix](../artifacts/non_df_h4_cl_transfer/2026-10-23/check03/gate_matrix.csv)、[各軸の14 node graph・fact・target・SOURCE](../artifacts/non_df_h4_cl_transfer/2026-10-23/check03/gate_connections.json)、[分類集計](../artifacts/non_df_h4_cl_transfer/2026-10-23/gate_summary.json)。元のschema/checker/判定を変更せず、新しいoverlayとして保存した。

| 分類 | 接続数 |
|---|---:|
| 既存の範囲限定命題・条件付き規則・静的来歴が成立 | 200 |
| 保存証拠はあるが追加照合/数学的証明が未完了 | 180 |
| 新しい技術的実装/証明が必要 | 80 |
| 新しい保証付き数値証拠が必要 | 60 |
| 旧program等の歴史的証拠が不足 | 40 |
| 固定条件への不適合が証明済み | 0 |

200件はcertificate全体の成立数ではない。各gateの限定scope・仮定・元判定を保持し、**40軸すべてでall_scientific_gates_verified=false、受理0、N/G nullのまま**。Cの保存40軸TARGET_MET/64 bitは再計算せず、total/bias/normalization tokenをそのまま参照する。

DFのL上界はmethodを合わせ、両軸・5精度へ再利用できる宣言scopeだけを接続した。Pauli-JWへDF局所証拠を流用しない。Forward/inverse、basis順序、component ID、別case/axis、SOURCE/SHA、encoded/literal/compiledを取り違える変更を拒否する。Lが得た近似unitarityの上界から、丸めencoded eventのexact norm1とは推論しない。

## 残存課題・最小追加契約案

[9つのproof obligation](../artifacts/non_df_h4_cl_transfer/2026-10-23/proof_obligations.json)、[最小診断対象](../artifacts/non_df_h4_cl_transfer/2026-10-23/minimal_numeric_targets.json)、[実行不可の契約案](../artifacts/non_df_h4_cl_transfer/2026-10-23/minimal_numeric_contract_DRAFT.json)。

DFの既存primitive listを構造的に列挙すると、PF379＋PR46＝425項。次回の候補予算はこの425項、PR rotation parameter3、pi1、scalar phase trace family2、uniform phase family2、PRのsaved B upper binding2の計435論理callまでを提案する。これは保存構造から作った候補であり、数値評価ではない。128 bit/Arb-Acbを提案するが、実行承認や成功保証ではなく、無断の精度昇格もしない。

この案は**実行準備未完了**である。SOURCE/DAG・同じrayの会計・入力の回収可能性を先に確認し、専用数値runnerを実装・人工検証して新SOURCEを固定した後、その具体的なone-shot契約にユーザーの明示承認が必要。今回固定したSOURCEは保存証拠checkerであり、将来の数値runner SOURCEを代用しない。

40軸full-H reference/law mean/biasやL148/148/11/324の再実行は既定案に含めない。Cの保存bias_distance、Lの独立上界、nominal/saved normalization、runtime/input SHAを再利用する。Pauli側literal rotationの追加要否は、まずexact dyadic角度生成の技術的証明を確認し、必要な場合だけ別の最小scalar診断を提案する。このDF案だけですべての40軸gateが閉じるとはしない。

## Kとの境界

旧program/pass traceを今回新たに回収していない。保存Kの限定的なSOURCE恒等式・候補passの未証明事項を接続しただけで、既知4case外への広域探索も新compileも行っていない。旧compiler全体はMISSING_EVIDENCE。Literalの技術的な前進をcompiler保証へ流用せず、新compiled系列を採用しない。研究の評価対象・保証方針の変更はユーザー判断である。

## 検証・SOURCE固定・保全・公開・STOP

[最終テスト](../artifacts/non_df_h4_cl_transfer/2026-10-23/tests02.json)：T67＋binding33＋L48＋新CL54＝**202 PASS、FAIL/ERROR/skip0**。初回198 PASS後、forward/inverse・basis逆順・component raw・literal/compiledの取り違え拒否4件を追加した。実H4行列/確率/上界の再評価を人工検証に混ぜていない。主agentによる検査で、独立科学レビューではない。

[新技術SOURCE・runtime・入力](../artifacts/non_df_h4_cl_transfer/2026-10-23/SOURCE.json)、[封印後CLI](../artifacts/non_df_h4_cl_transfer/2026-10-23/seal_verification.json)。Code9・既存private runtime1,050・入力673のSHAを封印し、同じruntimeで静的CLIを実行、前後SHAを確認した。別に[原SOURCEと同一のDF diagonal source](../artifacts/non_df_h4_cl_transfer/2026-10-23/SOURCE_static_appendix.json)1 fileの読取照合を付記した。旧C/L SOURCE・旧sealを変更していない。

[保全監査](../artifacts/non_df_h4_cl_transfer/2026-10-23/preservation_after.json)では既知の旧3,177 fileをSHAで確認する。Lのmtime失敗履歴を含め、旧raw/grant/ledger/marker/STOP、元H4 native未証明/原因UNKNOWNを保持する。新しい時刻順序証明や旧native PASSは主張しない。今回のCLI exit0は技術検査の終了だけである。

静的joinと小さい形式恒等式を対象に、既存file cacheとcheckerを流用した。CPU/RAM/workerへの人工上限、共有環境・他ユーザーへの変更は0。実H4処理の速度向上は測定していない。

専用commitに保存し、共通AGENTSのcommit/push許可に従い新branchだけ公開を試みる。[公開記録](../artifacts/non_df_h4_cl_transfer/2026-10-23/publication_status.json)で成否・remote SHAを別記録する。**mandatory STOP。新科学grant0、execution_authorized/grant_valid/dispatch_enabled=false、正式受理0、N/G0。**

次の担当は**ユーザー**：joint会計案、normalized-ray適用条件、literalの最小診断を進める範囲を判断する。**Codex**：明示承認された技術範囲で専用runnerとproofを準備し、数値実行は固定SOURCE・契約への別途明示承認後に限る。正式受理A・N/G・新compile/系列へ自動移行しない。
