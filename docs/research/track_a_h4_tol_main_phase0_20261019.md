# H4 tol-only rank10 主比較：Phase 0技術準備（2026-10-19）

**PHASE0_COMPLETE_MANDATORY_STOP。科学実行は未承認。** 保存F/FP入力の再利用監査、D/Aのadapter・契約案、Pの条件付き設計、人工テストを完了した。新しいSCF・integral・DF・保存state・reference・signal・shot・sampling・科学compile・統計解析・n数値計算・順位・費用比は実施していない。

実行実装SOURCEは `84db3909639b34ec521004c3d8f7ef9b5e6ad851`。独立branchは `codex/h4-tol-only-main-phase0-20261019`、作業場所は同名の専用worktree。成果物は[Phase 0 artifacts](../../artifacts/resource_applicability/track_a_h4_tol_main_phase0/2026-10-19/)、状態は[completion audit](../../artifacts/resource_applicability/track_a_h4_tol_main_phase0/2026-10-19/completion_audit.json)を正本とする。実行中の科学runはない。push・最終evidence freezeは行っていない。

## 1. 固定資料と保存入力

以下のcommitから実際のSOURCE・結果・契約・終了記録を取得し、関連する私有入力をbyte SHAで照合した。主な参照先は[来歴](../../artifacts/resource_applicability/track_a_h4_tol_main_phase0/2026-10-19/SOURCE_provenance.json)と[政策SOURCE一覧](../../artifacts/resource_applicability/track_a_h4_tol_main_phase0/2026-10-19/policy_comparison.json)。

|対象|固定commit|今回の扱い|
|---|---|---|
|F実装|`dfef38fc8107dd76f5d71dcdeca31f8d3dc39f64`|rank10生成コードと保存receiptを確認。再生成しない|
|FP追加準備|`64e0fc91cd273cff02ae8cffb729ee8e40c302a9`|入力・compiler・signal・実測軸の来歴|
|F/FP完了|`97d25cc431516d720ce91c7cb4f148e061d8ea34`|public gzip、展開bytes、私有resultのSHA一致|
|legacy rank12 D/P|`4e8ad1936363266d1b55bb689c36846e6c3dfcdb` / `8a18b277663bf2f8998f79e9b8c0a9e61d2b184c`|実装・保全基盤の参照。科学数値はrank10へ移さない|
|H6/H8 A_COST|`fc6bd039dc7171d5674ac04f0b3ad427f1b6495b`|完了資料と実際のkernel SOURCE `31ef62a12407010e9c1a07f9b37facedb8dff5fe` の15ファイルを照合|
|総費用比較|`64c85f7a833b5950ef2ccae623708b73081edcd6`|費用定義・探索範囲・実測/model区分の参照|
|H8 P SOURCE/結果|`745a674546f2cb6258526fbfde1d9a00601689f8` / `e6eea5387a6e84d1e39a9adb3b61975acc74e200`|事前固定・独立seed・paired会計方式の参照。n48は移さない|

H4はlinear・隣接1.00 Å・STO-3G、system 8 qubits、Nα=Nβ=2のsector36。保存入力の識別子は以下のとおり。

|対象|SHA/hash|
|---|---|
|Hamiltonian|`c1a6c4281f11bba90df2fd18a998bdec96bed00379116537d22faf548c525ea0`|
|state|`9bcc0b0ecf44440c4411bcb76d652d78ff0595f036144580660c56c47c78e80b`|
|snapshot|`12219b7d14dc2a275853314269e5e8628005ed72b42d16bc4d3831c8ea515c96`|
|compiler|`1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`|

[F保存metadata](../../artifacts/resource_applicability/track_a_h4_tol_main_phase0/2026-10-19/F_saved_input_metadata.json)には保存済みarray hash・shape・dtype・残差などを転記した。NPZは今回decodeしていない。F/FPのnative exit0、wait4完了・same-parent回収、final gate PASS、COMPLETE_MANDATORY_STOPを確認した。

既知のF/FP出力、元入力、SOURCE、rank12 D/Pの固定spec/grant/marker/result/terminal等の**774レコード**で原本と私有copyのSHAを照合した。これは対象を限定した保全監査のレコード数で、他研究の全ファイルを再走査した件数ではない。実行に使うSOURCE closure17ファイル、私有provider236＋A kernel15＝251ファイル、既存runtime34,273ファイルを固定した。新規のvalid grant・科学ledger・ONE_SHOT・科学outputは作成していない。

## 2. H4/H6/H8の生成政策と比較上の差

|項目|H4 tol-only rank10|H6 tol-only rank19 / H8 tol-only rank28|判定|
|---|---|---|---|
|SCF・MO|同じOpenFermion-PySCF helper、RHF、conv_tol=1e-9、max_cycle=50、Angstrom、symmetryなし、canonical MO返却順で追加sortなし|元SOURCEと保存integral receiptで同じ設定・規約を確認|政策一致|
|integrals・spin|spatial→spinorb、偶数alpha/奇数beta、two-bodyに1/2|同じhelper・保存convention|一致|
|DF|OpenFermion `low_rank_two_body_decomposition`、threshold=1e-8のみ、final_rank指定なし|同じ分解方針。H8の省略metadataは元SOURCEのkwargsで確認|政策一致、実rankは異なる|
|順序・cutoff|生成順を保持、signed係数保持、後続cutoff=0|同じ|一致|
|Hermitian projection|係数重み付きtriangle/Frobenius評価、full-Fock N=8|同じ定義、N=12/16|定義一致。9.9e-11はengineering判定、区間保証なし|
|定数・scalar|元constant＋抽出tail identityを一度だけ、referenceは回転しない|同じnative制御・scalarの規約|同じ処理を再利用|
|sector・状態目標|Nα=Nβ=2、sector36の最低固有状態|Nα=Nβ=3/4、sector400/4900の最低固有状態|物理sectorの定義は共通|
|状態solver|完全sector行列のdense `eigh`、最低vector|`eigsh` SA、HF初期vector、k1、tol1e-12、ncv40、maxiter1000|**実装差あり**|
|正規化・位相・残差|eighの正規化vector＋最大振幅の位相固定。保存後再正規化なし、残差gate1e-10|一度のbinary64正規化＋同じ位相規約、absolute1e-9とrelative1e-10の残差判定|**数値手順・gate差あり**|
|compiler|Qiskit1.3.0、rz/sx/x/cx、opt1、seed17|宣言設定は同じ。H8の歴史的profile識別子は異なる|過去のbinary出力の同一性までは認定しない|
|探索・PF費用証拠|今回D190/A50の案|既存A16、強化PFの探索領域やmodel利用範囲も異なる|サイズだけの因果効果と解釈しない|

DF生成政策は揃っているが、「全入力生成手順が同一」とは認定しない。保存stateのsolver・正規化・残差判定まで厳密に揃えるには新しい入力生成・状態計算の科学承認が必要で、今回の技術修正の対象外である。保存入力を固定してD/Aを進める案と、この差を解消する追加計算を要求する案は、次の科学実行前にユーザーが判断する。

技術的に統一したのは、rank10用candidate/full指定、入力hash、signed/factor1 endpoint、compiler確認、cohort表示、call会計と拒否gateである。legacy rank12のhard-coded prefixやnを移植していない。H6/H8の過去のCPU・thread・worker・メモリ条件は来歴として保持し、新runの恒常的な資源上限にしていない。

## 3. 既存実装・FPの再利用

|既存実装・証拠|再利用範囲|新しいadapterが担当する点|
|---|---|---|
|`h4_tol_only_input.load`、完成Fのsnapshot/receipt|保存bytes・array/Hamiltonian/state identityの検査。新SCF/DF/state生成なし|rank10 bindingと私有copyへの接続|
|`h4_native_science`、`h4_followup_science.compile_task`|forward/native・独立oracle、control4 probe、full wrapper、paired axes、fingerprint・compiler照合|保存full-H referenceの再利用、rank10のgrid、signed/factor1だけを新規会計|
|`h6_native_b0i_bridge_v1`|tail identityのexact算術と解析phase overlay|rank10のB0→B0I、reference無回転|
|完成FP|4決定論候補n1の8軸、PR1候補n32の64軸、対応signal・shot・保存mean/SE/n|入力・候補・軸・SOURCE・compiler・既存fingerprint一致を確認したものだけ採用|
|original A `pr_cost_native.steps/compile_task/control`|finite-RTE law、原seed hierarchy、各軌道のcosine/sine共通event、独立event oracleとcontrol|rank12/H6/H8用loaderを避け、rank10 preparation/contextを渡す|
|`h4_dp_contract`|exact paired費用・SD/SE/分散・同一候補endpoint間共分散、seed検査|新規n8 cohortの会計。FP rawから再集計しない|
|`h8_key_compile_parallel.map_owned`|fork/COW、1 worker 1 thread、own CPU pin、native exitと同一親回収、fingerprint再利用|承認時のfresh CPU集合を使用。PSIは観測のみ|
|H8/H4 P|事前固定、独立seed、固定n、共通endpointのpaired費用、no pooling/no optional stopping|最終P契約はD/A後。現runnerからのP実行は明示拒否|

FP固定5構成は `B0I_p8_q4_2nd/4th`、`B1_p10_q4_2nd/4th`、`B2_p3_q4_2nd_r4_K4`。保存signalにはoverlay前のB0 2構成もあり、Dで6 signal recordを再利用する。[FP再利用証拠](../../artifacts/resource_applicability/track_a_h4_tol_main_phase0/2026-10-19/FP_reuse_evidence.json)の25 main endpoint行は保存値の転記で、新しい順位・費用比を付けていない。

FPのPRは **SAVED_FP_PANEL_N32_ANCHOR** のまま保持する。新Aの49候補n8に混ぜてnを読み替えず、raw費用を集計し直さない。A50候補全体を今後評価するまではFPだけを全探索結果と扱わない。Aのcohort間のnの違いと選択効果は残り、A自体も独立確認ではない。FPに保存されていない共分散が必要なら、その数値解析を別途承認対象にする。

## 4. 共通評価条件と事前登録案

T=.8、P=0、通常signed bias、allowance倍率1、ε=.05/.01/.005/.001/.0005。費用はfull Hadamard wrapperのcompiled RZと保存shotの積をcosine/sineで足す。compilerは上記固定設定、backend/coupling/layout/routing=None。q=1/2/4/8/16なのでδ=T/q=.8/.4/.2/.1/.05。厳密bias certificate・母平均保証は付与しない。

**D：190候補。** B0/B0Iはprefix1〜9、q5点、2nd/4thを全組合せで180候補。full PFはB1 prefix10、同じq・次数で10候補。rank12のfull指定は使わない。最低費用は全登録signalの適格性が確定し、適格候補の実測費用が揃った後に、exact費用・同値ならcell_id辞書順で選ぶ。不明・欠測を不適格として除外しない。q16等の登録境界winnerから自動的に探索を拡張しない。

**A：50候補。** `center=floor(rank/4)=2`、prefix=center±1を含む1/2/3、q5点、r4、K2/4で30候補。さらにcentral prefix2でr2/8、q5点、K2/4の20候補を追加する。新規49候補は原finite lawによるEXPLORATION_N8、同一条件のFP1候補は保存n32 anchorである。量子分布・係数・normalizationは変更しない。全候補のsignalが確定後、少なくとも1 main endpointで適格の候補だけ固定nの費用を取得する。適格候補の欠測費用はselectionをblockする。

このgridはrankに対する構造規則で提案し、新しいrank10 signal/costに基づく候補選択をしていない。H6/H8より探索領域が広い点を含め、登録内最良をglobal optimumや純粋なサイズ依存性と呼ばない。[全候補](../../artifacts/resource_applicability/track_a_h4_tol_main_phase0/2026-10-19/candidate_registrations.json)。

## 5. 承認対象となる新規科学call予算

以下は候補・測定対象に由来する**科学scopeの最大数**で、CPU/RAM/worker/wallの制限ではない。不適格が確定した候補はcompileしない。将来実行時にはsignal・overlay・build・軸compileの予約と各workerのresult/exit/reapを残す。

|段階|新signal / 解析overlay / main shot endpoint|新軌道・native evolution / wrapper compile|control / oracle / preparation / primitive|
|---|---|---|---|
|D|96 / 88 / 920|最大186 / 372軸|control744、独立forward96、打切りexact reference88、preparation10、primitive1095|
|A|49 / 0 / 245|最大392 / 784軸|control196、独立event oracle49、独立forward49、tail matrix最大49、preparation3、primitive120|
|S_STATS|未実施・別承認|新sampling/compileなし|入力は受入済みD/Aの保存mean/SE/n・PF費用・保存rare情報だけ|
|P|PENDING_D_A_RESULT|n、軌道数、compile予算は未確定|最終候補とnの固定後に確定|

Dの新規compileからFPの4候補8軸を、AからFPの32軌道64軸を差し引いている。SCF・integral・DF・保存state生成・full-H referenceは両段階とも新規0。Dの打切りreference、各signalの独立forwardやprimitive検証は新規科学範囲に含む。full-H保存referenceの再利用と、打切り対照の新oracleを区別する。[全call予算](../../artifacts/resource_applicability/track_a_h4_tol_main_phase0/2026-10-19/proposed_calls.json)。

primitiveの365/40 structural time pair×3 probe＝1095/120は、元の純stdlib scheduleを使った列挙である。負のYoshida時間・directional符号・merged PFを含む。Hamiltonian/stateのdecode、量子action、signal計算はしていない。[構造的coverage](../../artifacts/resource_applicability/track_a_h4_tol_main_phase0/2026-10-19/structural_primitive_coverage.json)。

## 6. 契約・SOURCE・seed・資源・STOP

[D](../../artifacts/resource_applicability/track_a_h4_tol_main_phase0/2026-10-19/D_contract_DRAFT.json)はTECHNICALLY_BOUND_PENDING_APPROVAL、[A](../../artifacts/resource_applicability/track_a_h4_tol_main_phase0/2026-10-19/A_contract_DRAFT.json)はPENDING_D_RESULT_AND_SEED_INVENTORY。[S_STATS](../../artifacts/resource_applicability/track_a_h4_tol_main_phase0/2026-10-19/S_STATS_contract_DRAFT.json)と[P](../../artifacts/resource_applicability/track_a_h4_tol_main_phase0/2026-10-19/P_contract_DRAFT.json)はPENDING_D_A_RESULT。すべてINVALID grant、approved_by_user=false、runtime_authorization=false。Phase 0承認receiptを科学実行receiptとして使用しない。

runnerは最初に科学実行の明示承認を要求し、その後にstage、one-shot、SOURCE commitの実bytes、契約digest、実際に使う入力・metadata・runtime・provider、fresh資源receipt、未使用output/lockを照合する。Aでは受入D結果とseed registry/exclusion/receiptも入力として固定する。私有providerに未登録ファイルが加わった場合も拒否する。他worktreeのHEAD/status・資料更新は実行gateにしない。

新A seedは専用namespace、SOURCE、plan digest、input binding、cell_id、replicaからSHA256でmasterを導出し、原direct_master_to_step_seeds_v1でstep seedを作る案。cosine/sineは同一軌道、候補間CRNは使わない。旧H4・geometry・conditional pilot・FP・rank12 P等のseedとの完全な衝突照合は**Aのseal時に実施する未確定作業**。今回は新しい実行用seed値・registry・衝突ゼロreceiptを発行していない。人工seedの衝突テストだけを実施した。衝突ゼロは確率的独立性の証明ではない。

将来の重い処理は既存owned poolの構成/軌道間並列、fork/COW、prefix preparation・小行列spectraのcache、paired evolution共有、fingerprint再利用を使う。1 worker 1 threadでnested並列を避け、worker数は実行直前の許可CPU・quota・メモリ・既知の他job割当から決める。今回worker_cpus=[]、resource admission=false。歴史的なCPU数、メモリ、deadlineは適用しない。OS/cgroup/共有package/他processは変更しない。PSI等の監視失敗や一時的競合だけで正常workerを止めず、実際の入力破損・OOM・自分の異常・科学的不一致ではSTOPする。

各承認stageはone-shotで終了し、異常時も予約とSTOPを保持する。自動retry・次stage移行はfalse。修正再実行には元記録を残した新しいSOURCE/attemptと、その範囲の承認が必要である。

## 7. Pと標本数設計の未確定事項

主確認条件はε=.05/.0005を提案する。受入済みD/Aの結果を固定した後、各精度でcomplete registered Aの最小point費用候補、exact同値は辞書順を代表として固定し、同じ候補ならdeduplicateする。PFもcomplete D内の最小費用対照を使う。適格PRがない場合はNO_ELIGIBLE_PR、PFが不足ならINCONCLUSIVEとしてPを開始せず、候補差し替え・endpoint拡張は行わない。

n設計案は保存SE×sqrt(n)からのSD、検出差η×PF費用、family alpha=.05、power=.8、主2 endpointを使う近似normal fixed-gap planning。SD倍率1/2/4/8、相対差.01/.02/.05を感度条件、基準案を倍率4・差.02とする。共通候補のnは対応するendpointの必要nの最大とする。**このモデルと基準は提案であり、実データのn数値計算をまだ実行していない。** H8 n48、rank12 P n4185やrank12分散を流用しない。保存rare情報を確認し、未観測S4の費用やpopulation meanがこの式で保証されるとは解釈しない。

S_STATSのモデル・入力・scenario数をD/A後に最終固定し、数値解析だけの承認を求める。raw再集計・bootstrap・不足共分散の取得はこの承認に自動包含しない。別途必要なら最小限の独立承認対象とする。S_STATS後にPの候補・PF・n・SOURCE・master/step seed・予算・保存形式を最終固定し、科学実行を別承認にする。現runnerはP/S_STATSのfinal contractがないため明示拒否し、P adapterの完成や実科学での動作確認を認定していない。

判定案は固定したPF−PR平均の方向が探索時の固定方向と一致するか。ゼロはINCONCLUSIVE、同じ候補の複数精度は同じ固定標本で共分散を保存する。no pooling、no optional stopping、候補再選択なし。これは有限標本の点比較で、selection-adjusted coverageや母平均の優位保証ではない。

## 8. 人工テスト・metadata終了監査・次の作業

新規32件と再利用基盤42件、計74件PASS。新規は候補・rank/scalar/compiler binding、保存FP mean/SE/nの非再集計、cohort/replica分離、paired反相関・共分散、欠測・同値判定、seed衝突、source/input/provider/runtime改変拒否、claim後STOP、D/A mock統合、保存reference再利用、structural primitive coverageを検証した。再利用42件には人工Hamiltonianだけのnative/compile8軸が含まれ、実分子標本を使っていない。再利用testのimport path不足は専用provider pathで解消し、共有環境は変更していない。現SOURCEの新32件と、それ以前に成功した変更のない42件を分けて記録した。

[人工報告](../../artifacts/resource_applicability/track_a_h4_tol_main_phase0/2026-10-19/artificial_test_report.json)、[実DRAFTのdirect/witness拒否8経路](../../artifacts/resource_applicability/track_a_h4_tol_main_phase0/2026-10-19/grant_refusal_report.json)、[保全receipt](../../artifacts/resource_applicability/track_a_h4_tol_main_phase0/2026-10-19/preservation_receipt.json)、[artifact SHA](../../artifacts/resource_applicability/track_a_h4_tol_main_phase0/2026-10-19/artifact_manifest.json)。実DRAFTの拒否検査では科学module import・output・lock・子process作成前の拒否を確認した。監査は著者であるCodexの確認で、別AI/reviewerによる独立レビューではない。

実分子を用いるD/A/P、実負荷での速度・peak RSS、Aの最終seed照合、Pの選択・n・最終SOURCEは未検証/未確定である。準備完了を科学実行のPASSと扱わない。

次の作業：

1. **ユーザー**：状態生成の実装差を明記して既存rank10保存入力を使うか判断し、D190候補・最大372新規軸等の具体的scopeを確認して、Dの科学実行を別途承認する。
2. **Codex（その承認後）**：fresh資源とDのSOURCE/input/approval receiptをsealし、Dだけ実行・保存・監査してSTOPする。
3. **Codex／ユーザー（D受入後）**：A50候補のcohort・最大784新規軸・seed照合を最終固定し、A実行の別承認。D/A受入後にS_STATS、さらにPの最終契約と別承認へ進む。

現在はmandatory STOP。追加科学実行・統計解析・公開pushを自動開始しない。
