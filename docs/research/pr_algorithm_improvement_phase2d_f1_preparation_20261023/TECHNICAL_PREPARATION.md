# F1実行前準備の技術記録

対象はP0の4 mode・2粒子・occupied (0,1)、決定論one-body block 1（L_D=1）、残差density-square 2 fragmentの合成fixture。分子geometry・AO basis・分子DF rankは非該当。SS/SP/PS/PPを同じH、同じfinite corrected targetで比較する。T=1/5、q=2、delta=1/10の一点、r=2、h=1/20、K=2。full16／controlled32、JW little-endian、ancilla-last、faithful_identity_in_tail、scalar・paired phase・負符号・products→child順を維持する。catalogは保存4label、c=-1/31、h=T=1/20、q=r=1、K=2。

## 保存入力と固定資料

B結果起点は `32c76bc6392ba277ce9fdd8d4bc54ffb5339b767`、B実行SOURCEは `3de42d3a3efe9ebc74dde8a1d0e1f1bdbadd607c`。P0 `b8d54708f90ccef275eda3757143100b2d0c677e`、I `bfaab3831042f64a7112296437c0001e34920209`、2C `0c94a3d973b0ba0d24b3d277ab8316829739b278` の固定文書／SOURCEを確認した。GitHubの固定pathも読み戻した。

private handoffにあるrawからphysical/catalog、4arm、5law、input-only bundleをread-only再利用する。原本全40fileはB公開hash台帳と一致。保存bundleのliteral_prices／score_intervalsは空のまま保持する。実行時もBの係数map／lawを再生成せず、binary64 hex値とtupleを復元して既存RTE dataclassへ渡す。新しい価格・検証結果は別のF1出力へ保存する。

正式共通指示は作業中にユーザーが追加した2026-10-23正式版を取得して読んだ。SHA256は `7ce342dbf7086424f1afd837d8c274687bc28e4838601f5ad96f5b626d95667f`。§8.4・§10に従い、固定原本を保全し、独立venv／作業場所だけを変更し、科学callを増やさず既存SOURCEを再利用する。§10.3に従い、F2にしか要らない形式資格や一般的な管理条件をF1の新しいblockerにしない。原workspaceへユーザーが追加した本書はそのまま保持し、こちらから編集・移動・公開していない。

## 環境と再現

Bで構築済みのCPython 3.11.9基底とNumPy wheelを再利用し、別venvへ50個の固定wheelをoffline/hash必須で導入した。Bのvenvは変更していない。主要依存はREADMEの5packageに加え、既存trotterlibのeager importが要求するnumba 0.60.0、matplotlib 3.9.2、PySCF 2.7.0、OpenFermion 1.6.1、OpenFermion-PySCF 0.5、Qiskit Aer 0.15.1、Qiskit Nature 0.7.2。NumPy 1系列との依存整合のためCirq-core 1.4.1、Qiskit Algorithms 0.3.1を固定した。全依存とwheel hashはlockを正とする。projectのeditable install、共有環境の変更、sudoは行っていない。

Linux x86_64、glibc 2.35。通常importは必要11packageすべてPASS。Qiskit transpileのkeyword signature、依存整合もPASS。QuantumCircuit／Operator／Statevector／interval contextは生成していない。研究SOURCEは環境smokeにimportしていない。依存lock、wheel、Python／stdlib／venv／native packageのfile hashはprivate保存し、公開manifestには再現用lock／wheel identityとprivate receipt hashだけを記録する。

B環境の既存receiptに記載されたbase 3,471file、venv 925fileのhash一致を確認した。専用環境の動作確認は科学的正しさの検証ではない。科学SOURCEの動的な接続、Gaussian recipe、controlled operator、compileの実動作は別途承認されたF1の登録attemptで初めて確認する。

## 追加SOURCEと既存コードの関係

| 追加path | 理由／既存処理の再利用 |
|---|---|
| `src/trottertracks/pr_dictionary_validation/f1_saved_plan.py` | stdlibのみのID／ordinal／依存計画。Bで確定したLにより既存16familyを固定。event payloadは生成しない。 |
| `src/trottertracks/pr_dictionary_validation/f1_saved_execution.py` | IのRegisteredF1を継承。旧law()の入力再bindingをB値の復元に置換し、既存router／gates／price DPを呼ぶ。 |
| `scripts/run_pr_dictionary_f1_saved.py` | 旧verify_packet／claim_packetを再利用し、固定plan一致・専用Python・HEAD・全local Python closure・fallback境界を追加照合。science importは認可消費後。 |
| `tests/management/test_pr_dictionary_f1_saved_plan.py` | ID／件数／依存順と未承認draft拒否のみ。研究import・数値処理を伴わない2件。 |

旧execution.pyは全LAW recordごとに独立lawを再計算し、増加中の確率listをfsumし、全SOURCEを再hashする。特にprefix fsumはO(E²)。追加adapterでは独立lawをarm×precisionでcacheし、確率massを最後に一度fsumする。全Eのmetadataと重複検査は維持する。SOURCEはゲート時・各handler phase境界・partition終了で再照合する。

独立referenceのCAR・exterior-minor・Pauli作用・factorial polynomial・Taylor/Pade・event作用は既存reference.pyを再利用。ctx／元の独立H・D・R・frameをprecision別に保持し、旧reference_caseの繰返し構築を避ける薄いreference adapterを加えた。candidateのcanonical Rや回路matrixを独立reference生成へ流用しない。90/128評価数、精度間比較、rtol=0、phase fitting禁止を維持する。component productのHermiticity／involutionと、保存S/P fragment対独立CAR fragmentの検査を既存COMPONENT／H-ID record内に明示した。新規の別fixtureやevent探索はない。

旧SOURCEには手を入れていない。新SOURCEはASTと参照helper名の静的照合のみ。科学branchは未実行であり、追加実装に不具合がないという数値保証はまだない。

## 価格・intervalの境界

既存recipe_prices／literal_rz_count、直接run scan、EMPTY/SINGLE/MULTI DP、selector／既知四候補を用いる。basis full/singletonのentry+inverse、central、identity、負product、event／scalarの潜在phase reserveを既存の同一grammarで計数する。literal countはactual phase集約後やcompiled costの予測・上界ではない。

価格table 5個はF1承認後の最初のprice caseで各一回bindする。各nonidentity atomにfull／singleton provider request各一回、basis keyとsupportが同じものは既存cacheを利用する。内部basis生成は登録component／motif／catalog buildとprice処理に含まれる。別のcontrolled probeや追加compileを作らない。catalogの68直接price、DP1、比較1は旧SOURCEのまま。DPと比較の90/128 scalar集約は各case内の事前固定処理で、506個のmatrix reference評価とは別に記録する。

P0では価格の事前bindingを承認条件に置いていたが、今回はそのbuildを実行してよい承認がない。**bindingをF1の登録price caseへ移す案をユーザーの次の承認判断へ戻す。独断で採択していない。**

interval providerの形式資格検査は今回追加しない。P0／2Cの既定fallbackを採る実行範囲を提案する。4armのliteral期待値を計算しても、score_enclosureは未資格を拒否し、ARM_PRICE／SELECTOR／EXHAUSTIVEはUNPRICED_FALLBACKとする。N・score・認証された優位候補は出さない。SHOT_COUNT_UNRESOLVEDを保持する。将来資格providerを採択する場合はそのSOURCE／env／検査callを含む別の承認が必要。今回のSOURCEは資格statusの偽装を拒否する。

この範囲でP0の16familyは省略しないが、正常な認証価格選択やshot境界の科学的検証を済ませることにはならない。全gate済みの `F1_VALIDATED_WITHIN_REGISTERED_SCOPE` は自動表示せず、正しさ結果と価格資格の不足を分けてレビューへ渡す。

## 登録計画と資源

件数の正本はmanifestと保存BのF1_COUNT_DESCRIPTORS。L=SS14／SP58／PS26／PP51は保存metadataから引用。

| family | 件数 | 単位／内容 |
|---|---:|---|
| H-ID | 4 | logical case、full H／sector／保存fragment |
| LAW | 348,232 | 全event metadata、回路buildなし |
| COMPONENT | 447 | product／rotation0／rotation2 controlled build＋check |
| MOTIF | 96 | 4arm×8motif×raw／optimized／actual inverse |
| WRAPPER／COMPILE | 各64 | X/Y wrapper、raw／compiled signal、full wrapper限定compile |
| MEAN | 4 | actual component control1 block由来のone-step／全体mean |
| CATALOG EVENT／MEAN | 136／2 | raw／optimized各68buildと各平均 |
| PRICE ENUM／DP／COMPARE | 68／1／1 | literal直接scan、DP、独立集約比較 |
| ARM PRICE／SELECTOR／EXHAUSTIVE | 4／1／1 | 同じ4arm価格cacheと未資格fallback |
| REFERENCE | 506 | 2×(4+32+68+149)、90／128 precision evaluations |

異なる単位の行を科学call総数に加算しない。ID／ordinalだけのprivate展開planは52,958,909 bytes、SHA256はmanifestに固定した。実際のevent／trajectory payloadはgrant消費後だけ遅延解決する。旧authorizationはこのID計画を普通のcase_inventoryとして検証するため、新しい認可方式は作っていない。

CPU affinity 32、RAM空き約52GiB、disk空き約348GiBを準備時に確認した。full32行列の小系なのでGPU pathは追加しない。独立5dataset（4arm＋catalog）をspawn ProcessPoolへ分け、worker=min(5,利用可能affinity)。同じdataset内ではreference／basisを再利用し、LAWをstreamする。5は科学入力のpartition数であり総CPU capではない。BLAS/OpenMP/Numba/Qiskit子processは各1、Rayonはaffinity/worker数へ配分してoversubscriptionを防ぐ。wall time・RAM・総科学run時間のcapは置かない。

参考行列とcomponentはlast-use後に解放。IR完全一致のcompile cacheは旧gate実装を維持する。call requestは64のまま、cache hit aliasもrecordを残す。arm間のcacheは独立で、冷／再利用のtimeをjournalとprice reportへ保存する。小さい補助basisのnative合成、90/128独立reference、compileが残る候補bottleneck。dataset間のLの偏りによる待ち時間は残る。実測speedupは主張しない。

失敗時は自分の新しいoutput内のSTOP_REQUESTEDを共有し、他partitionは新しいcase attemptを始めない。既にattempt済みの処理は結果／失敗を残して停止する。再試行・置換・増精度・追加samplingはしない。

## 既存20testの取扱い

旧test moduleは読解のみでcollect／実行していない。prepared()/arm_data()はfixtureとlawを再生成し、truncation toleranceもB固定値ではないので、pytestをそのまま追加実行しない。重複する内容は以下の登録handlerで確認する。独立した汎用edge testを実行済みとは表示しない。

| 旧test（test_接頭辞省略） | F1への対応／範囲 |
|---|---|
| exact_serializer_roundtrip_and_noncanonical_fraction_rejection | saved spec復元・固定hash。一般invalid文字列caseは追加実行しない。 |
| context_time_meaning_is_not_q_r_swap | preflightの固定context。F2 context検査は対象外。 |
| exact_jw_multiplication_matches_independent_car | H-IDの独立CAR／保存S/P fragment照合。個別annihilator 4個の別計算は追加しない。 |
| all_four_dictionaries_reconstruct_original_h_and_sector | H-ID4件。 |
| pauli_collect_exact_cancellation_without_screening | B入力をhash固定し再collectしない。1e-30等の新規edge fixtureは対象外。 |
| zero_tail_is_exactly_certified_only | Bの全armは非empty。新規empty-tail生成は対象外。 |
| physical_pauli_not_disguised_as_df_support | 保存typed component復元／catalog任意長Pauli。 |
| scalar_interval_binary_endpoint_decoding_and_ceil_refusal | 未資格fallback／未決Nを保持。新規scalar資格testは対象外。 |
| actual_component_product_rotation_control_blocks | COMPONENT447。 |
| support_completion_preserves_columns_and_conjugation | 既存support_completion内の列／unitarity検査＋actual component gate。 |
| n0_n2_sign_order_identity_scalar_fusion_and_actual_adjoint | MOTIF96＋LAW metadata。 |
| hadamard_xy_readout_sign_and_state_prep | WRAPPER64。 |
| component_derived_corrected_mean_uses_actual_blocks | MEAN4、CATALOG MEAN2。 |
| literal_dp_matches_independent_finite_event_scan | 保存catalog lawのPRICE68／DP／COMPARE。旧toy確率は使わない。 |
| unknown_recipe_prices_do_not_become_zero | price validate、unknown recipeはSTOP。 |
| selector_and_exhaustive_share_information_and_tie_policy | 4unique priceを共有。未資格fallbackのみを結果化し、認証tie改善branchのPASSは主張しない。 |
| selector_unqualified_interval_returns_explicit_fallback | SELECTOR／EXHAUSTIVEの明示fallback。 |
| two_independent_precision_paths_are_registered_without_retry | REFERENCE506のpaired比較。 |
| compiled_full_wrapper_retains_operator_and_signal | COMPILE64。 |
| f1_authorization_rejects_p0_or_i_before_scientific_import | 同じ旧gateの純管理拒否testを追加management testで確認済み。 |

## 数値条件・出力・停止

P0の閾値を固定：H／Hermiticity／involution 1e-11、controlled／unitarity 1e-10、mean／signal 2e-10、compiled 5e-10、p／mass／normalization L1 1e-12、B／coefficient相対尺度1e-12×max(1,reference)、angle 1e-12、DP double 1e-10×max(1,price)、高精度price 1e-40×max(1,price)、90/128 reference 1e-60。rtol=0、phase合わせなし、NaN/Infは失敗。compilerはQiskit1.3.0／opt1／seed17／rz,sx,x,cx／topology-free／approximation_degree1／qubits_initially_zero=False。

将来の新規outputは `artifacts/pr_algorithm_improvement_phase2d_f1/<実行日>/<承認時の一意ID>`。旧B namespaceを使わない。5partitionのappend-only journalにATTEMPTED／COMPLETED／FAILED、timing、数値残差、LAW metadataをstreamする。回路はQPY、actual block／meanはNPY、506referenceは精度付き文字列表現で保存する。巨大event matrixは作らない。resources、summary、出力hash台帳、source/env/inputの承認時snapshot、STOPを保管する。成功時も次の科学段階には進まない。

未承認・未binding・SOURCE／input／env hash不一致はscience import前に拒否。科学処理開始後のoperator／phase／sign／mean／signal／price不一致、case逸脱、原本保全違反はFAILED_MANDATORY_STOP_NO_RETRY。新runでも同一失敗の自動retryはしない。F2、統計／bias認証、新規性、資源優位性の権限は発生しない。
