# 研究C：F2 κ Hermiticity STOP・限定診断の技術準備

2026-10-27 JST。起点 `acf61e9f1a4d5fd8948b517b978c65994b797bc5`、独立branch `codex/research-c-F2-kappa-diagnostic-preparation-20261027`。

**技術準備のみ。新規実H4科学callは0。** κの非Hermiticityの大きさ・原因は今回新たに測定していない。元h/g/κ、旧F2 SOURCE・契約・raw・STOP、旧B1/F1/C1の資格は変更しない。F2再実行・F3・重要科学レビューは行わない。新しいCAR targetも採択しない。

## 固定失敗証拠と再利用

[旧F2報告](research_c_pr_aware_df_fixed_F2_polar_execution_20261027.md)と[原本](../../artifacts/research_c_pr_aware_df_fixed_car/2026-10-27/F2-polar-run01/)をJSON／bytesで照合した。`input_arrays`、`arm_replay`、`trajectory_load`はCOMPLETE、`CAR_reference`が`_hermitian_input(kappa)`で`EXACT_HERMITIAN_INPUT_REQUIRED`。4/最大7 attempts、retry/resume0。CAR block・exp・polar bridge・raw QPY decode・raw/CAR・triangleは未実行である。残3 callsは消費可能な診断枠ではない。

旧科学SOURCE manifest SHA256は `9d077123674137b23a9d050f013eb5984f2d57f3ea2e29c4d23f02b16225ca34`。今回のSOURCEはその既存135ファイルinventoryを保全し、新規の診断module・CLI・人工testだけ追加する。既存コードの修正は0。

| 既存SOURCE | 今回の用途 |
|---|---|
| `c1_stage1.corrected_kappa` | 将来の診断で元のNumPy補正式をそのまま呼ぶ |
| 同`operation_lengths/allowance/finite_arrays` | κに限るStage 1式の再現、有限値確認 |
| `c1_stage1_execution.sha/read/save/journal`、`c1_stage3_integrated.Calls` | SOURCE/input照合、exclusive保存、call/first-failure/partial STOP |
| `stage_b.prepare_arm`、Stage A/B2/F2 runner | 静的にκのdataflowを照合。診断ではarm/CAR/nativeを生成しない |
| `fixed_car._hermitian_input` | 人工回帰で旧厳密条件が維持されることを確認。診断値をCARへ渡さない |
| 既存Arb/acb verifier | K0/K1の前提を静的調査。今回の診断は有限dyadic加減算だけなので、標準`Fraction`による厳密oracleの方が小さく直接的 |

新しいHamiltonian生成器、fitter、CAR代数、Gaussian emitter、optimizerは作らない。診断は[専用module](../../src/trottertracks/c_pr_aware_df_kappa_diagnostic.py)・[CLI](../../scripts/run_research_c_pr_aware_df_kappa_diagnostic.py)・[人工test](../../tests/management/test_research_c_pr_aware_df_kappa_diagnostic.py)に閉じる。

## Q1：入力とκ生成SOURCE、厳密判定との差

元保存入力は直鎖H4・隣接1.00 Å・STO-3G、Nα=Nβ=2。元生成commit `86b8b002704aced8ca239414080f53793de81658` の `src/trotterlib/h4_molecular_inputs.py::build_h4_inputs` は、PySCF RHFの実MO係数を用い、`one = mo.T @ mf.get_hcore() @ mo`、`eri = ao2mo.restore(1, ao2mo.kernel(molecule, mo), 4)`を保存する。SOURCE／既存schema上、hはspatial one-body、float64/4×4、gは**chemist** `(pq|rs)`、float64/4×4×4×4。ここで記すdtype/shapeは既存schemaと生成SOURCEの確認で、保存payloadの新decodeではない。

実軌道の理想積分にはhの実対称性、gのpair内交換・electron pair交換の8重対称性がある。複素軌道へ一般化する場合は `h_pq*=h_qp`、`g_pqrs*=g_qpsr`、`g_pqrs=g_rspq` を用い、実軌道のpair内交換をそのまま仮定しない。保存binary64がこれらを厳密に保持するかは別問題で、今回は判定していない。

ffsim v0.0.83／SOURCE `558d3517385778c733f1aa226bb61b8f30f4b0f3` の [MolecularHamiltonian](https://github.com/qiskit-community/ffsim/blob/558d3517385778c733f1aa226bb61b8f30f4b0f3/python/ffsim/hamiltonians/molecular_hamiltonian.py) は `H2 = (1/2) sum g_pqrs a†_pσ a†_rτ a_sτ a_qσ`。同commitの [DoubleFactorizedHamiltonian.from_molecular_hamiltonian](https://github.com/qiskit-community/ffsim/blob/558d3517385778c733f1aa226bb61b8f30f4b0f3/python/ffsim/hamiltonians/double_factorized_hamiltonian.py) は `h - .5*einsum("prrq", g)` を使用する。両ファイルはローカル固定ffsim SOURCE／既存hash inventoryと照合した。現研究Cのspin順は2p=α、2p+1=β。ffsim内部のblock spin配置と混同せず、spatial h/g/κの段階で接続する。

`E_pq = sumσ a†_pσ a_qσ` とすると、CARから `E_pq E_rs = δ_qr E_ps + sumστ a†_pσ a†_rτ a_sτ a_qσ`。したがってnumber-density表現のone-body補正は

`T_pq=sum_r g_prrq`、`κ_pq=h_pq−T_pq/2`。

この縮約に追加の複素共役を入れない。Hermiticityはh/gの対称性から導かれる。OpenFermionへは既存`spin_lift_unthresholded`が `eri.transpose(0,2,3,1)` をspin liftし、`InteractionOperator`のtwo-bodyへ1/2を掛ける。すでにtranspose済みのOF tensorをchemist gとして再縮約することは許さない。補正式は変更しない。

| 経路 | κの生成・利用 |
|---|---|
| Stage A `prepare_worker` | 元packetのone_mo/eri_chemistをrenameしてinput.npzのh/gへ保存。元gを再分解tensorへ置換しない |
| Stage A `rank_assessment` | `SectorCAR.one(corrected_kappa(h,g))`を使用 |
| Stage B1/B2 | `stage_b.prepare_arm`→`corrected_kappa`→一般Kのspin lift。one-body nativeは`spin_adapted_eigh`から作る |
| F2 | `prepare_reference`で同じprepare_armを再生後、`corrected_kappa(h,g)`をもう一度呼びCAR入口へ渡す |

現行補正は `np.einsum("prrq->pq",g)` の既定評価とbinary64減算であり、対称化を挟まない。`optimize`・縮約順・dtypeの変更を今回導入しない。ffsimの暗黙出力`prrq`は残る添字p,qの出力に対応する。

一般Kのone-body metadataはHermiticityの別gateを要求する。native側のspin helperはcomponent絶対許容1e-10を通して`np.linalg.eigh`へ渡し、Hermitian solverの前提を置く。これは元κの要素完全一致を証明する経路ではない。近似Hermitian入力に対するsolverの選択三角の意味と、literal κを数学的targetにすることは区別する。

F2はfiniteかつ`np.array_equal(κ,κ.conj().T)`を要求する。これは丸め後の要素の**完全一致**であり、符号付きゼロを区別するbyte列同一性ではない。Stage 1の数値残差条件との不一致はこの入口にある。SOURCE誤り・丸めが実際の原因と確定したわけではない。

## Q2：Stage 1の既存PASSの意味

既存の[保存fit評価専用Stage 1結果](../../artifacts/research_c_c1_stage1_eval_only_science/2026-10-25/run01/result.json)を引用する。保存P-H4・上記物理条件、通常number CDF8／保存XDF10、split LD・時間窓はStage 1では非適用。

| 旧gate | 旧error | 旧allowance | 判定 |
|---|---:|---:|---|
| κ_Hermitian | 8.660473972585522e-17 | 3.518379690624931e-11 | PASS |
| κ_original_g | 0.0 | 1.8829801476580426e-11 | PASS |
| saved_XDF_κ | 6.723512808976018e-17 | 2.5246248268644952e-11 | PASS |

これは**旧保存結果の転記**で、新F2 κの再解析ではない。旧Stage 1 κと新Stage A/F2返却κのbit同一性は今回認定しない。

Stage 1は `D=κ−κ†` のflattened Frobenius norm、絶対条件 `error≤limit` を使用する。rtolとの混合ではない。n=4、m_base=8(2n)²+32=544、`B=|constant|+2 sum|κ_pq|`、u=2^-53、γ_m=mu/(1−mu)として、

`limit = sqrt(n) * 8 * [2 γ_m B + 32 n u ||D||F]`。

既存`operation_lengths/allowance`をそのまま再利用する。scale依存の工学的allowanceで、κそのものの相対誤差閾値でも、厳密丸めenclosureでもない。保存PASSから言えるのはこの旧条件で数値残差が許容内だったこと。厳密Hermitian、F2入力条件の充足、非unitary増幅の厳密無視は言えない。hやTへ同じ閾値を流用しない。

## Q3：最小診断と原因分類

将来取得するbinary64観測はh/T/κ各々のdtype/shape/finite、Δ=M−M†のmax-entry・Frobenius・spectral2、Mの同じ絶対scale、exact一致と不一致index、κAの相対Frobenius scale、κに限る上記Stage 1比較である。Frobeniusにはstable `hypot`も併記し、旧Stage 1の`np.linalg.norm`値とは分ける。零scaleは相対値null／UNRESOLVEDとし、未知の保証を0にしない。これらのspectral normは数値観測でありvalidated上界ではない。

独立exact oracleは保存floatの実部・虚部を標準`Fraction`でexact dyadicへ変換する。n=4の有限和に新たな任意精度数値libraryは不要。T*を保存gの厳密縮約、T̂をNumPy返却値、K*=h−T*/2、K̂を既存corrected_kappa返却値とし、

`E_T=T̂−T*`、`E_K=K̂−(h−T̂/2)`、

`ΔK̂=Δh−ΔT*/2−ΔE_T/2+ΔE_K`

を有理数で検証する。E_Kには1/2倍のsubnormal丸めも含む。各行列のexact entries・非零index・Frobenius²・entryの実虚絶対値和による保守的spectral上界を保存する。exact κH=(K̂+K̂†)/2、κA=(K̂−K̂†)/2も**診断recordだけ**に保存し、入力やCARへ渡さない。κHをfloat64へ再丸めした行列と同一視しない。

| 分類 | 診断から言えること／限界 |
|---|---|
| A | 保存hのΔh≠0。元SCF/MO計算のどこで生じたかまでは未証明 |
| B | 保存gを厳密縮約したΔT*≠0。g全体のどの対称性の破れが生成されたかの一意な原因ではない |
| C | ΔK*=0だがΔK̂≠0なら、保存値の正確な補正式に比べ演算が対称性を失わせたと判定可能。混合ケースではE_T/E_Kの寄与を併記し、丸めだけと断定しない |
| D | SOURCE/hash/schema/元h/g/cの同一性不一致ならSTOP。対称性残差だけから添字規約違反を断定しない |
| E | 元積分の生成誤差、NumPy reduction tree・各内部演算への原因帰属は未解決として保持 |

A/Bは同時に成立し、その非対称成分がκでは相殺する可能性もある。人工検証にこの対照を含めた。Cの人工例は大きい数と小さい数を異なる順で縮約するもので、分子gの全対称性を持つfixtureとは主張しない。各rの縮約summandが対称位置で厳密に等しい人工対照では、同じ評価経路で対称性を保持した。したがって「binary64だから自動的にκが非対称になる」とは説明しない。

元packetとStage A保存inputのh/g/cを将来element-wise照合する。元packetのstate/energy/DF/JUは読まない。入力の来歴はhash・生成SOURCEで確認するが、元の真の積分精度をこの診断だけで証明しない。

## Q4：K0／K1とCAR接続の条件（いずれも今回新規採択しない）

| 方針 | 成立する比較／追加条件 |
|---|---|
| K0：返却literal dyadic κを維持 | 非Hermitianなら非unitary mapへのphase-sensitive validated距離という意味は保てる。独立acb referenceの全column差とtriangle不等式自体はunitarity不要。ただし旧`compare`の`min(2, u+bridge)`や、全factorがunitaryというnorm伝播・信号モデルはそのまま使えない |
| K1：数学的exact κHを新理想targetへ | 原入力を変更せずκ差を別model bridgeとして囲う。旧「exact κ＋polar Q」のtargetと異なる。exactなκHのacb組立を新SOURCEで実装・検証し、ユーザーによるtarget採択が必要 |

K1設計の候補式：spatial `a=||(κ−κ†)/2||2`、full8-mode Fockでは粒子数上限8なので、one-body liftのanti-Hermitian部分のnormは `μ≤8a`。任意の実tに対しHermitian Hとanti-Hermitian Aについて、logarithmic-norm/Duhamelから

`||exp(−it(H+A))||2 ≤ exp(|t| μ)`、

`||exp(−it(H+A))−exp(−itH)||2 ≤ exp(|t| μ)−1`。

κAの数値spectral推定だけをμの厳密上界にしない。将来は今回準備したexact entry-L1上界、または独立validated spectral評価を使う。sector36だけの粒子数4の係数をfull256/controlled512へ流用しない。scalarが実ならこの成長へ寄与しない。

完全trajectoryはfactorごとのtelescopingで、`||A_j−B_j||≤e_j`、norm上界a_j/b_jを用い、`sum_j e_j prod_{k>j} a_k prod_{k<j} b_k`のように非unitary側の増大を含める。既存scheduleのbaseは左右それぞれh/2で現れるため、片方だけの補正では不十分。κ以外のliteral U eventもunitaryと仮定しない。

二重計上を避ける一案は、**元κ/UのB0 → exact κH/元Uの中間map → exact κH/polar Qの理想map**という二つの独立bridgeを組むこと。κ bridgeにはliteral Uのnorm増大、既存U→Q bridgeにはκHを固定してその前提を検証する。旧U→polar Qの数学的定義は変えない。acbのreference半径は一度、κ bridgeとU bridgeはそれぞれ一度、旧compiled/raw距離も一度だけ加える。

現時点の推奨は**先に6-call診断を行ってから選択すること**。診断後、厳密Hermitianなら旧target条件に沿う別の固定F2を検討できる。非Hermitianなら、unitaryな理想targetが目的の場合K1は検討候補、literal implementationの比較が目的ならK0が候補だが、値を見ずに採択しない。必要な情報はκAのexact量とscale、source/rounding寄与、登録allowance、要求するtargetの意味、将来のbridge上界の実用性である。

## Q5：新しい未承認6-call科学契約

[契約](research_c_pr_aware_df_F2_kappa_diagnostic_contract_draft_20261027.json)のscopeは`C_PR_AWARE_DF_KAPPA_HERMITICITY_DIAGNOSTIC_V1`、statusは`UNAPPROVED`、technical executableはtrue。チャットで別途実行許可が必要で、承認証明・原文保存・grantは要求しない。

| 順序 | logical call | 上限／対象 |
|---:|---|---|
| 1 | input_load | 1：Stage Aと元packetのh/g/scalar限定loadを一組として計上 |
| 2 | input_identity | 1：dtype/shape/finite、元packetとの値同一性 |
| 3 | kappa_generation | 1：既存corrected_kappaと同じeinsumのT |
| 4 | binary64_diagnostics | 1：h/T/κのnorm batch・κの旧allowance式 |
| 5 | exact_dyadic_diagnostics | 1：小さい有理数縮約／誤差分解／H-A分解batch |
| 6 | classification | 1：観測の切分け、未資格/null保持 |

**最大6 logical units。** 内部scalar加減算／各normを別callにはしない定義を事前固定した。raw/CAR・operator生成・exp・JU/trajectory/QPYのdecodeは含めない。Fresh outputは `artifacts/research_c_pr_aware_df_kappa_diagnostic/2026-10-27/diagnostic-run01`、現在未作成。retry/resume0、fit0、F2/F3実行0、mandatory STOP。

入力はStage A `run01/prepare/input.npz` SHA256 `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db` と、元P-H4 snapshot SHA256 `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50`。保存場所と生成SOURCE、旧F2 manifest・失敗・STOP・Stage 1引用証拠を契約へhash固定した。

元snapshotは現在の起点treeにはないため、歴史commit `86b8b002704aced8ca239414080f53793de81658` の `artifacts/non_df_h4_pilot/2026-10-23/run01/input_run/input_generation/data/snapshot.npz` blobを**byte列のまま**[新診断専用入力](../../artifacts/research_c_pr_aware_df_kappa_diagnostic/2026-10-27/inputs/P_H4_original_snapshot.npz)へ回収した。2,157,448 bytes／上記SHA256一致で、数値decode・科学再解析ではない。旧snapshotやStage A inputを変更していない。

NumPy1.26.4／Python3.11.1、FE_TONEAREST、現行float64補正を固定。exact oracleは保存値の有理数評価で、元SCFやDFの再計算ではない。n=4・縮約64項程度の逐次依存処理で、並列process立上げが支配する規模のためNumPyベクトル化＋小さい有理数処理を使う。worker/CPU/RAM/wallの人工capはない。passive資源記録とtask専用venv/cacheを用い、共有venv/cache/OS/他jobを書き換えない。

## 人工検証・保全・STOP

人工テストは厳密Hermitian、微小／大きい非対称性、h/T由来、演算由来、相殺、real/complex、exact H/A、norm、subnormal、exact対allowance、source/input/schema不一致、null/UNRESOLVED、fresh禁止、partial/STOPを検証する。旧F2厳密gateも人工入力で回帰し、緩和しない。

[技術証拠](../../artifacts/research_c_pr_aware_df_kappa_diagnostic/2026-10-27/technical/)にSOURCE、tests、passive preflight、引用JSON、source audit、preservation、技術STOPを保存する。実科学callsは0。人工PASSを実κの原因確定・F2 gate PASSへ昇格しない。

最終テストは**125 distinct tests PASS＋29 subtests PASS**（新診断29、既存F2/CAR/Stage B/Stage 1が83、旧一般K13＋29 subtests）。旧一般Kは既存のimport隔離条件に従い別のfresh Python processで実行した。ログは`technical/tests.log`と`general_k_tests.log`。人工でoptimizer iteration・実分子decodeを行っていない。

新SOURCE manifest SHA256：`408fe5114bd32deef82279aae7d9c14461e5e4de097d6473846a29d0b84b178c`（旧135＋新3）。未承認契約SHA256：`ffeda0e5ffaf5e0b08d3a94818e824218a6570429e590117176fcbb63242e7ae`。公開commit identityは最終チャットのcommit URLを参照し、manifest内で自己参照hashを作らない。科学SOURCEファイルの差分は新3ファイルだけで、既存codeの変更0。

SOURCE/input/version/finite/固定shape・元packet一致・exact分解不整合は依存処理を停止。実行開始後は完了したbinary64 record等のpartialを保存する。非Hermiticity自体は診断対象の観測であってdiagnostic correctness FAILではない。元入力・演算履歴の原因が一意に決まらない場合はUNRESOLVEDを保つ。補正式やCAR targetの変更が必要になっても自動変更・追加計算しない。

## Q6：診断後の新F2に必要な条件と次担当

1. **ユーザー／研究C**：この新6-call診断契約とSOURCEを確認し、実行を別途明示許可するか判断する。
2. **Codex**：許可された場合だけfresh診断を一度行い、κA・入力／演算寄与・Stage 1比較を保存してSTOPする。旧F2は再起動しない。
3. **ユーザー／研究C**：診断結果と研究targetの意味からK0/K1等を判断する。必要な重要科学レビューは独立の許可対象。
4. **Codex**：採択後にだけ、新targetのnorm伝播・必要なbridge・allowance・新しいSOURCE/契約・fresh output・call枠を準備する。新F2実行も別の明示許可後だけとする。旧残3 calls・旧run01/STOPを再利用しない。

今回の公開・remote照合後は**mandatory STOP**。実κ診断、F2/F3、sampling、C1再解析、CA/shots/総資源、別分子、詳細科学レビューへ自動進行しない。
