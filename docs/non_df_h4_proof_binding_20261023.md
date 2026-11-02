# H4非DF/DF：実証拠接続と旧compiled programの保証可能性

2026-10-23。専用branch `codex/non-df-h4-proof-binding-20261023`。基点commit `36d0c12de9cec2a4fbb3abc0d8ab59c7f890cb2a`、T SOURCE `a66cefb497cb83c13158c9d4ab926202f791309f`、C科学SOURCE `e80d7510880f8db993b777c627bb8e95b05a070d`、元H4科学SOURCE `0c329b30a81accb35c212be8defa66fee0dd0f12`。**今回の技術SOURCEは `14c0f8b0adda6160d507807334b82da0531a0e35`**。今回の実装・静的実行・保存は科学certificateの生成・正式受理を含まない。

**実証拠adapterを実装し、40軸×14 gateの証拠値・SHA・対象命題・scopeを接続した。旧compiled program本体とpass実行receiptは、調べた4caseの保存領域で確認できない。** 固定compilerのSOURCEにはTの閉じた規則では覆えない分解・rewrite・数値kernelがある。したがって、旧compiled program全体の厳密保証は未成立であり、新しいcompileで旧証拠を置換していない。

## 対象・取得・実施範囲

固定対象はlinear H4/隣接1.00 Å/STO-3G、8 system qubit、保存状態Nα=Nβ=2、Pauli-JW184項/PR prefix27、DF tol-only1e-8/rank10/PR prefix1＋one-body、S2/T=.8/q8/r4/K4。4方式×ε=.05/.01/.005/.001/.0005×cosine/sineを維持する。別系列のH2 S2/Yoshida4解析と混合しない。

最初にGitHubの基点branch/commit、T SOURCE、C結果commit `86b8b002704aced8ca239414080f53793de81658` と指定資料を取得した。**基点branchが36d0c12deを指すこと、31ファイルのbytes、8 directoryのGit objectを実照合した。** C sealは1MiB超のためcontents APIのcontentが空だった。これを取得済み扱いせず、固定blob SHA `96adbc360a7a12849da1fadf4c610ed4e0b64a4e` を使って取得し、全bytesをGit blobと照合した。[GitHub照合記録](../artifacts/non_df_h4_proof_binding/2026-10-23/github_verification.json)。既に公開された基点commitは再pushしていない。

今回は静的SOURCE・JSON型・保存値の同一性・SHA・scope照合、人工/mock/T回帰テストだけを実施した。Endpointは整数比の**文字列として型と一致を確認**し、大小・幅・符号・包含・target充足を新しく評価していない。Reference/law mean/bias、誤差上界、shot/Gを再計算せず、sampling/RNG replay/新compile/区間計算も0。

## A：実証拠adapterとproof verifier

[library](../src/trotterlib/h4_evidence_binding.py)、[封印済みCLI](../scripts/verify_h4_evidence_binding.py)、[テスト](../tests/test_h4_evidence_binding.py)、[26資料の固定manifest](../artifacts/non_df_h4_proof_binding/2026-10-23/evidence_manifest.json)を新規追加した。Tの3実装file、TのSOURCE・結果、旧科学SOURCE・結果を上書きしていない。

adapterは保存C model/resultと旧resultを接続する。Model全体のbinding hash、caseのmetadata/expected_metadataと旧model、Hamiltonian hash、preparation hash、state hash、旧normalization、旧signed bias、C contract/sealのSOURCE・input manifest・routingを照合する。Pauliのpreparationは既存label順序等のmetadataとHamiltonian hashで結び、DFのpreparation hashを無断でPauliへ補わない。

各gateのrecordは、対象case/精度/軸のtarget digest、数学的命題、scope、証拠file/SHA/SOURCE、JSON pointerと値のdigest、source参照、lemmaと前提、不足obligationを持つ。`binding_status` と `proof_status` を分離する。**binding_status=VERIFIEDは値と来歴の接続完了であり、数値命題の新しい数学証明を意味しない。** 保存ラベルは `saved_claim` として保持し、SHA一致やTARGET_MET/VERIFIED/CERTIFIEDだけからproofを成立させない。

producerの出力に対しverifierが同じ静的recordを再構成し、scope/target/証拠値の改変、別case・別軸への流用、欠測gate、重複proofを拒否する。すべての実行/正式受理/N_G flagはfalseを要求する。科学certificateの受理APIは提供しない。

### 接続できた命題と不足

[40軸×14 gateのCSV](../artifacts/non_df_h4_proof_binding/2026-10-23/check01/gate_matrix.csv)、[完全な証拠pointer・命題・blocker](../artifacts/non_df_h4_proof_binding/2026-10-23/check01/gate_bindings.json)に各行を保存した。既存gate ID/scope/状態語彙は維持する。

| gate・対象 | 今回のproof_status | 確認範囲・不足 |
|---|---|---|
| 数値包含/full-H reference/signed bias | EVIDENCE_EXISTS_UNCHECKED | 保存64 bit enclosure・target・bias・state defect・semantic allowanceを同じSOURCE/入力へ接続。数値的不等式を再検証していない |
| PF normalization | VERIFIED | 固定deterministic構造のB=1/log_B=0。literal回路との対応は別 |
| PR nominal log_B | EVIDENCE_EXISTS_UNCHECKED | 保存包含値・上界flagと同じtail/normalizationを接続。新しいlog/exp・endpoint比較なし |
| binary measurement range | VERIFIED | 理想Born二値測定と有限非負B_savedという明示条件で、Y=B_saved(1−2b)の支持集合は±B_saved。DF行列normを証明したものではない |
| Pauli encoded event norm | VERIFIED | signed Pauli componentのlabel/sign型とexact tensor-Pauliのinvolution。有限P5 meanやliteral角度/compiled回路全体のunitarityへ拡張しない |
| DF encoded event norm | MISSING_EVIDENCE | 丸めforward/inverseと理想gateの差、basis/control合成が未証明。falseを数値逸脱と解釈しない |
| PF/PR law | PF VERIFIED / PR ASSUMPTION_DEPENDENT | PRは保存effective massへの接続とIID uniform-word仮定。CDF/massを再計算していない |
| phase/control/basis/literal wrapper | MISSING_EVIDENCE | 定義・作用順・抽象lemmaを接続。実gateの角度/符号/phase、control両branch、全eventへの一様転送誤差が不足 |
| compiler transfer | MISSING_EVIDENCE | 旧compiler設定・価格join・有限state診断は存在。旧program/実行pass traceと全変換の保証が不足 |
| conditional statistics | ASSUMPTION_DEPENDENT | fresh independent shots、PRのIID uniform wordsを明示。無条件PCG64独立性を新必須条件へ追加しない |
| provenance | VERIFIED | 指定されたcode/input/model/preparation/state/runtime identityの保存記録を接続するscope。原H4 native終了証明とは別 |
| B_saved→shot上界 | PF VERIFIED / PR EVIDENCE_EXISTS_UNCHECKED | PRは固定測定scaleと保存log_B upperのscalar bindingが残る |
| combined error / acceptance | EVIDENCE_EXISTS_UNCHECKED / MISSING_EVIDENCE | 新transferとの結合・固定guard比較・正式受理は後段。今回は計算・受理しない |

従来matrixのnumerical項VERIFIEDは「既存候補証拠が保存されている」というscopeだった。今回の `proof_status` は、**ここで新しい数値検証を行わないことを表すためEVIDENCE_EXISTS_UNCHECKED**とした。旧 `saved_claim` と数値結果を変更したわけではない。保存された40/40 TARGET_MET/64 bit、正式受理0、20条件N/G=nullは保持する。

### 演算子・位相・作用順の対応

[SOURCE位置一覧](../artifacts/non_df_h4_proof_binding/2026-10-23/source_map.json)と[encoded→literal→compiled対応表](../artifacts/non_df_h4_proof_binding/2026-10-23/guarantee_relations.json)に、固定関数とSHAを示した。

- C referenceは保存binary64から構成したfull-Hと共通state上のψ†exp(−iTH)ψ。Same-ray normalized biasとencoded signalは別fieldで保持する。真の分子積分・厳密ground stateのcertificateへ拡張しない。
- C `basis_operators` は記録順のAを左から掛けてforwardを作り、保存inverseを右から掛ける。Literal builderはinverse時にoperationsを逆順にして各gate.inverseを適用する。Cのnative因子はU D Uiであり、Uiをexact U†と同一視しない。TのB†DB記号との向きは、Bをliteralの最初に作用する基底変換と定義した場合の対応で、同名記号の見た目だけで同一性を判定しない。
- 固定finite RTEはorder0/2/4、積を先に作用させ最後にrotationを適用するためmeanはΣπ_k(−1)^(k/2)R_k M^k。DF fragment λA²そのものをPauli/involutionへ置換せず、symbolic signed Z/ZZ/I componentと区別する。保存mass/angle/支持集合を接続するが、新しいmeanを計算しない。
- Pauliのgeneric controlはsystem global phaseも相対位相へ変える。DFではbasisを非controlとし、diagonal/control branch、constant/identity/event signのancilla phaseを保持する必要がある。control=0側のbasis cancellationにも理想gateと丸め行列の違いが残る。
- Literal wrapperはancilla n、H、cosineはH、sineはSdg-H、p0−p1=Re/Imという定義。T lemmaはnormalized state/exact CU/ideal readoutの前提付きで接続する。phase/branch/basisの実装全体の保証を、このlemmaだけで成立させない。

### 誤差項の二重計上防止

Cの保存totalには `bias_distance` と `circuit_semantic_allowance` が既に入る。今回、その2項をprimitive coverageとして1個のsaved_C_total recordへ接続し、同じsemantic項を追加するとverifierが拒否する。新しい合計・差・u/headroom/guardを評価していない。局所Lの結果を後で加える場合、既存semantic項のbasis reuse/phase roundingと何が重複するかを先に証明し、未被覆部分だけを扱う必要がある。

## B：旧compiled programを優先した保証可能性

[詳細調査JSON](../artifacts/non_df_h4_proof_binding/2026-10-23/check01/historical_compiler.json)。既知4caseの15/525/15/525ファイル、計1,080 file名を調査し、QPY/QASM/QASM3/DAG候補は0。保存物はresult/validation/event/trajectory/price join等であり、調べた保存集合に完全なpostcompiled program・そのhash・pass callback traceを確認できなかった。全filesystemに存在しないとは主張しない。

旧H4呼出は `verify_compiled=True`、cost_cacheは未指定。Evaluatorはretain_transpiled_circuits=Trueのdefault cacheを作るが、persistent_pathはNoneで、回路はin-memory返り値に保持する。価格のpersistent payloadも回路本体の保存ではない。`wrapper_fingerprints` はpretranspile wrapped inputのfingerprintであり、output program hashとして使えない。SOURCE/seed/設定が一致することから旧program完全一致を推論しない。

4方式共通でQiskit1.3.0、basis rz/sx/x/cx、optimization1、seed17、backend/coupling_map/layout/routing指定なし。旧の514軌道/1,028 wrapperに対する固定state probability差は、Pauli PF 2.4444890556196697e-12、Pauli PR 1.3922196728799463e-13、DF PF 5.0616733027197824e-12、DF PR 2.2315482794965646e-13と保存される。既存event/価格/hash joinを参照接続したが、これらの経験的最大差を厳密上界へ昇格しない。

PFは各1軌道で固定program familyが小さい一方、PRは各256軌道の有限標本にとどまる。保存production order4は両PRで観測0、DFで観測したcomponentは160/324。Forced order0/2/4の固定例と、全event familyの証明は別である。方式間で状態診断とsymbolic structureは異なるが、旧postcompiled program/receipt欠落と全compiler保証不足は共通する。

### 固定runtime SOURCEから分かる範囲

C sealの既知legacy runtimeから、10個のcompiler関連fileのSHAを実確認し、Python SOURCEを静的に読んだ。Qiskitをimportせず、pass managerを生成・実行していない。これは10 fileの照合であって、今回全legacy runtimeを新しく再検証したという主張ではない。

| SOURCE上の候補経路 | Tで扱える範囲 | 不足するもの |
|---|---|---|
| level1 init/optimizationのInverseCancellation | 隣接するexact symbolic inverseの恒等式 | 全gate pair、照合規則・許容差・DAG書換え、旧pass instanceとの対応 |
| Optimize1qGatesDecomposition | RZとそのexact inverse等の限定恒等式 | accelerated Euler分解、角度/phase、数値丸めを含む一様誤差証明 |
| HighLevelSynthesis/controlled decomposition | 抽象diag(I,U)とphase保持 | 実際の分解・ancilla・controlled branch・全入力gateの適用条件 |
| BasisTranslator/equivalence library | 一部のexact word equality | equivalence選択、DAG変換、全basis gate規則の網羅 |
| UnitarySynthesisを含むtranslator構成 | 対応proofなし | 旧programでの実呼出有無、synthesisと近似/丸めの保証 |
| global phase更新 | pre-control phaseを保持する記号的命題 | 実phase蓄積、source branch、後段変換との整合性 |
| routing/layout | coupling/backendなしの設定を確認 | 全旧実行passを観測したreceiptはない |

これらは**固定SOURCEから得た候補経路**であり、旧各programで実際に実行されたpass列はUNKNOWN。人工word equalityをcompiler全体へ拡張しない。`inspect_pass_proofs` は未対応pass・重複proof・未登録passを拒否し、支持するword lemmaを全部渡した場合でも旧IRとのbinding未証明を保持する。

旧programがなくても、固定compiler全体の変換・分岐・数値誤差を証明する道は理論上排除しない。ただし今回その証明は成立しておらず、「SOURCEがあるので保証可能」とは判定できない。ユーザーが特定できる旧保存先・backup・pass receiptがあれば、その限定範囲を追加調査できる。現在の不足を新STOPや新receiptで事後補完しない。

## C：局所診断L・compiler K・受理Aの最小案

[DF局所scope確認](../artifacts/non_df_h4_proof_binding/2026-10-23/DF_local_scope.json)で、既存registryとmodel SHA、148局所hash、11 ordered basis、PR324 componentの構造を再照合した。Forward/inverseの保存値と記述は存在するが、理想RZ/XXPlusYYとの差、inverse/control合成の厳密保証はまだない。今回は行列積/norm/angle/新しい誤差上界を計算していない。

[非実行の最小契約案](../artifacts/non_df_h4_proof_binding/2026-10-23/minimal_next_contract_DRAFT.json)に、対象命題・保存入力SHA・方法・予算・監査・STOPを記録した。

1. **局所Lの実装準備**：新SOURCE/runtime、解析的2×2/4×4と外向き誤差の人工oracle、term coverageを実装・sealする。科学call0でSTOP。
2. **承認後の局所L診断候補**：128 bit固定・自動昇格なし。forward最大148、inverse最大148、basis合成11、symbolic支持集合324。上界を両軸・5精度で再利用し、SCF/FCI/DF/state/reference/law/bias/sampling/RNG replay/compile/N_Gは0。精度で閉じない場合も勝手に増やさずSTOP。実装SOURCEを固定する前は実行不可。
3. **旧program優先K**：SOURCE-levelの限定証明と、明示された旧保存物の回収可能性調査を別に進める。全compiler proofかprogramが得られなければMISSING_EVIDENCEを維持する。
4. **代替の新program系列**：ユーザーが選ぶ場合のみ別契約。PF4 wrapperは最小候補にすぎず、PRにはlaw family保証と有限workloadの確定が必要。新program/trace/hashを保存しても旧programの同一性を回復したことにはならず、旧資源評価を置換できるかは未確定。今回はcompile予算0、生成しない。
5. **受理AとN/G**：数値/transfer/scale/budget gateが揃った後、固定IID等の条件付きでAを別承認する。A後にもN/Gを自動会計せず、さらに別承認・別STOP。

将来の科学診断は、新namespace・新grant・call前予約/非返却ledger・owned native wait/reap witness・partial/receipt/STOP保存を要求する。旧grant再利用/retry/resumeは禁止。科学SOURCE/input/runtime変化・未対応gate・非有限/証明不能・固定条件不充足では保持してSTOPする。B/allowance/guard/lawを緩めない。

## 検証・保全・終了

**T既存67＋新規33＝100件PASS、FAIL/ERROR/skip0**。[テスト記録](../artifacts/non_df_h4_proof_binding/2026-10-23/tests01.json)。SOURCE/SHA/scope、別case/軸への流用、人工証拠の混入、丸めDFのunitarity、未対応pass、二重計上・欠測・重複、未承認flagを検査した。Phase/control/basis順序の誤りは既存T回帰でも拒否する。主担当自身の検査であり、独立科学レビューではない。

新技術SOURCE固定後に封印済みCLIを実行し、[静的接続verification](../artifacts/non_df_h4_proof_binding/2026-10-23/check01/verification.json)を保存した。Exit0は静的検査完了であり科学的PASS/正式受理ではない。実H4科学call・新数値上界・compile/sampling/RNG replay・certificate生成/受理・N/Gは0。

元H4 native終了未証明/監視原因UNKNOWNと、既存C run native exit/reap PASSを分離する。今回CLIの正常終了を旧native証拠へ代用しない。2,402既存ファイルを保全し、T/C/元SOURCE・raw・grant・ledger・marker・STOPを変更しない。資源は現行共通AGENTSに従い、任意の低いCPU/RAM/worker capを加えず、共有環境や他ユーザーへ変更を行っていない。指定統合指示fileは既知のroot候補とtracked file listingで見付からなかったため、その事実を[資源観測記録](../artifacts/non_df_h4_proof_binding/2026-10-23/resource_observation.json)に残した。

成果物は専用commitへ保存し、共通AGENTSのcommit/push許可で新branchをpushする。[公開記録](../artifacts/non_df_h4_proof_binding/2026-10-23/publication_status.json)と最終チャットに実際の公開状態を記録する。公開失敗時には再実行コマンドを示す。基点が現在公開済みという事実と、過去のpush失敗記録は時点を分けて扱い、過去記録を上書きしない。

**mandatory STOP。局所L・新規compile K・正式certificate受理A・N/Gは開始しない。** 次は**ユーザー**が旧program回収先やSOURCE証明/局所L実装の範囲を判断・承認し、**Codex**がその範囲だけを実施する。全40軸の再計算や旧資源評価の置換は承認済み扱いしない。
