# 研究C・C1：H2・H6の決定論S2検証準備

2026-10-25 JST。起点公開commit `6202f036457940532aeaa2873cd6d9e6f123c0ca`。
Branch `codex/research-c-c1-h2-h6-preparation-20261025`。

**H2は新入力・rank選定からnative／CA・shots・条件付き総費用まで、H6はsector400のDF／PFまでの限定経路を準備した。人工14＋既存回帰166 tests・29 subtests PASS、両moleculeのbyte preflight PASS。実分子の新fit・計算・保存結果再解析・実回路build／compileは0。新しい実行条件は未承認案であり、公開照合後mandatory STOP。**

[実装](../../src/trottertracks/c1_molecule_extension.py)／[CLI](../../scripts/run_research_c_c1_h2_h6.py)／[人工test](../../tests/management/test_research_c_c1_h2_h6.py)／[SOURCE・試験・資源metadata](../../artifacts/research_c_c1_h2_h6_technical/2026-10-25/)／[未承認実行案](research_c_c1_h2_h6_contract_unapproved_20261025.json)。

## 対象と準備完了範囲

直鎖H2／H6・結合距離1.00 Å・STO-3G・中性singlet、通常CDF number表現とexplicit XDF。決定論ordered symmetric S2、s=1／4、δ=0.1、m=0..12、CA=0.00159 Ha。各DFのsector ground oracleを仮定し、状態準備費用を除外する。PRのsplitは使わず、全outer frameの順序・original-g補正κ・scalarを保つ。

| 項目 | H2 | H6 |
|---|---|---|
| spatial／spin orbitals | 2／4 | 6／12 |
| Nα=Nβ／physical sector | 1／4次元 | 3／400次元 |
| full／controlled | 16／32次元 | 4096／8192次元 |
| 初回の実行範囲案 | DF・一般K同値性、PF、full native・P0、13round数値CA・shots・総費用 | DF・一般K sector同値性、ground／alias、4候補×13roundのsector PF誤差 |
| native／CA・費用 | 接続済み・人工検証済み | 初回scopeから除外、合成／shots／費用はunknown |
| CDF fit最大 | rank1..3、最大3回 | rank1..21、最大21回 |
| 科学call最大 | 196 logical units | 129 logical units |

一つのコマンドは一つの分子だけを実行し、H2完了後にH6へ自動移行しない。両方を明示許可する場合の上限合計は325 unitsだが、出力・STOPは分子別fresh runとする。H4のrank8／rank10・shots・結果・grant・STOPは流用も変更もしない。

## 調べた既存資産と適合性

| 資産 | 再利用／今回使わないもの |
|---|---|
| [過去H-chain PF資料](../pf_c_system_size_validation.md)、`pf_c_system_size_validation.py`／runner、H2/H6の係数artifact | 1Å/STO-3Gの計算経路は存在するが、config rank・partial-S2 split・delta窓・state-action摂動係数が今回と異なる。数値をdecode／再解析せず、CDF/XDF同CAの証拠には転用しない |
| `scripts/run_uwc_grouped_alpha_h2_h6.py` | grouped UWC／PF係数用。今回のordinary CDF・native S2比較とは異なるのでdispatchしない |
| `h4_molecular_inputs.py` at `0c329b30` | PySCF RHF・ao2mo・MO orthonormality・chemist tensorの生成workflowを静的に確認。新しい積分計算器は作らず、geometryとorbital数だけ一般化する。元のFCIやH4独自計算は実行しない |
| `c1_stage1.spin_lift_unthresholded/physical_sector/corrected_kappa` | 元h/g/cの物理reference、元g由来κ、OpenFermion疎JWとbit順序を再利用 |
| `df_hamiltonian.PhysicalSector/_build_excitation_table` | sector sparse CARの既存符号・spin規約を再利用。新しいfermionic bit-action実装は作らない |
| `df_hamiltonian.df_linear_operator`・既存matrix-free ground solver | square-block専用で、一般Jのframeをそのまま渡せない。一般Kをauxiliary squareへ分割してS2のframe順を変える流用は避け、既存excitation tableと疎行列を使う |
| `c1_ffsim_compat.double_factorized`、固定ffsim0.0.83 | 既存の通常CDFとSciPy logm bridgeをそのまま利用。新fitter／optimizer／初期化器は作らない。H4の固定max_vecs8 runnerを再起動せず、新rank引数を独立entryで渡す |
| 一般K adapter、Stage2 `NativeBuilder`、P0 wrapper／metrics | H2のnative Gaussian・RZ/RZZ・controlled scalar・basis共有・角度結合を再利用。両armへ同じ規則を適用 |
| `c1_stage3_reference/accuracy/connection/deterministic_accuracy` | 独立CAR・S2・数値残差／unitary telescoping・RPE horizon／Hoeffding／DF一回計上・CA-gated費用集計を再利用 |

参照した固定treeのファイル名・関連入力文書の範囲では、元h/gを含む今回のH2/H6 packetは見つからなかった。全サーバー・全履歴での不存在証明ではない。既存PF artifactや古い高次PF係数ファイルは存在するが、新しい元Hamiltonian入力の代替にはならない。元packetが後から回収できた場合は来歴・geometry・tensor規約を照合して新しい入力契約へ固定する必要があり、黙って差し替えない。

## 新しい入力とrank選定案

将来の科学実行では、分子ごとに1回のcanonical RHF（提案conv_tol1e-12）とAO→MO積分生成を行い、`h/g/constant/mo_coeff`をpickleなしNPZへ保存する。gはchemist `(pq|rs)`、RHF orbital orthonormality gate1e-10。FCIを追加生成しない。同じh/g/cをCDF／XDFへ共通入力とし、数値packetのSHA256をfit前に保存する。今回は生成自体を実行していない。

rank policy・DF予算・数値guard・XDF分解方式は**科学実行前に採択が必要な提案**である。

- DF reserveはH4と同じCA/8=0.00019875 Haを推奨する。CDFとXDFの実誤差を完全一致させず、各armで物理sector operator norm＋guardが同じ予算を満たす条件を使う。tensor residualだけで選ばない。
- n spatial orbitalsに対して有限rank ladder1..n(n+1)/2（H2:1..3、H6:1..21）。これは対称なpair空間から決める初回探索範囲で、fitの成功やDF予算達成を保証するものではない。
- CDFは各requested rankで既存ffsim通常C-DFを1回だけ呼ぶ。tol1e-8・L-BFGS-B・既存Cholesky初期化、number表現、元g由来κを保ち、最初の合格rankを採る。rank間の誤差単調性を仮定しない。optimizer不収束はSTOPで、restart／best-of-seedsをしない。
- XDFはffsimのnonoptimized explicit modified-Cholesky分解を最大rankで1回行い、元の順序のprefixを評価する。CDFと同じ元g由来κで、最初の合格prefixを採る。明示分解の短いexportなら実rankまでで止める。以前のH4 OpenFermion-eigh rank10と異なる新baseline algorithmであることを明記する。
- rankごとのrequested／actual rank、物理sector誤差、guardを保存する。CDFとXDFの誤差・rankは別々に報告する。XDFはprefixをcacheして過去frameを再構築しない。
- 提案guardは`sector_operation_scale_guard8_proposal_v1`：`8*eps64*(n^4+R_actual*n^3)*(d+1)*max(1,||Hphys||F+Σ||Hblock||F)`。宣言したoperation scaleに基づくengineering値で、厳密enclosureではない。H4の既存guard versionをこの別式で上書き・拡張しない。

初回ladderで両armがDF予算を満たさなければ`DF_BUDGET_NOT_MET`でSTOPし、rank拡張・tol変更・fit再試行を自動実行しない。この場合は研究上の物理的不可能性ではなく、登録条件の不適格である。

## H6の計算規模と安全なsector処理

complex128の1行列はsector400で2,560,000 bytes（約2.44 MiB）、full4096で256 MiB、controlled8192で1 GiB。複数full Operator、積、SVD workspaceを持つ巨大密行列経路は実装から除外した。行列の単体容量だけで現実的と判断しない。

H6では既存excitation tableからspin-preserving bilinearを疎行列として作り、h／κ／回転density／generalK frameの**完成した因子**をsectorへ制限する。各因子がNα・Nβを保存するので、これらの積の制限は正当化される。個々のGaussian elementary gateの途中をsectorへ射影する処理は行わない。raw J/U density積と独立の一般K I/Z/ZZ展開をsector400で照合する。

物理referenceは既存OpenFermion JWの疎4096 operatorをsectorへ制限してから400だけをdense化する。4096／8192のdense配列を作らず、full CARの12²×4096² stackも作らない。selected generatorsは最大各22×400² complex128（各arm約53.7 MiB）、400行列用のSVD／eigen workspace・候補並列を含む資源見積りを既存resource-planで行う。

PFは4つのarm／sごとにsector内でordered S2 kernelとDF固有分解を用意し、13 roundへ反復二乗法で接続する。最大2^12の回数だけ行列積を逐次実行せず、mが1増えるごとに一度だけ二乗する。各roundのPF spectral operator errorを直接測り、既知PF部分のphase-room適否も記録する。古いstate-action摂動係数からoperator誤差を推定しない。全spaceより小さい400 sectorでも実測timingは未取得で、速度や完走時間は断定しない。

既存[SciPy expm_multiply](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.expm_multiply.html)は行列指数の作用を計算でき、既存H6 state-action経路でも使われている。ただしstate1本の作用誤差は全sector operator norm上界ではないため、今回のoperator誤差評価を無断で置き換えない。さらに大きい系へ進むなら、matrix-free作用とnorm qualificationを別途設計する必要がある。

H6 nativeのbuild／compileだけなら巨大Operatorは不要であるため、技術的な次候補にはなる。しかしcompiled合成誤差・13roundのCA資格を同時に検証するには、別のmatrix-free／state-actionまたは解析的合成検証が必要となる。初回H6ではnative・shots・総費用をすべてunknown／対象外にしてSTOPし、DF／PF結果と実timingを見て研究Cが後続範囲を判断する。

## H2のStage1～3接続と誤差の区別

H2はsector4／full16／controlled32を使い、物理DF normと一般K表現同値性を別gateにする。selected CDF／XDFを既存native builderへ接続し、各armのs1／s4 kernel4件と共通head／X／Y3件を、同じP0でbuild・compileする。full controlled scalar、κ、Gaussianの向き、元frame順を保持し、raw対独立CAR、raw対P0、unitarityを既存max-entry1e-10／rtol0／phase fitなしで照合する。

合成はFrobenius残差の数値代用値をunitary telescopingで`n=s*2^m`へ伝播し、X/Y readoutを含めてraw複素signal誤差へ変換する。DF誤差（Ha）はここへ加えず、PF／合成をsignal→phaseへ接続し、最後に旧`assess_energy`でDFを一回だけ計上する。Nは新分子の誤差に対して既存Hoeffding式で算定する。H4の保存shotsを機械的に持ち込まない。

同じCAを数値pilot条件で満たした候補だけ、X/Y・13roundのNとnative kernel／readout費用を旧集計関数で接続する。主指標は条件付き総RZ、副指標はRZ/CX depth・CX count・total depth・circuit size・shots。費用はfenceのあるstaged serial和で、長い回路をglobal recompileした最適depthではない。未観測のliteral/shared追加削減は仮定しない。

unitary telescoping、norm2≤Frobenius、条件付きHoeffding／Weyl関係は解析的関係である。一方、SCF／DF／PF／CAR／expm／Operatorの数値残差と新guardのrounding enclosureは未資格であり、数値CA PASSを厳密CA保証へ昇格させない。state preparation・実hardware noiseは除外／未評価で、無条件の総資源優位は出さない。

## 科学call案・実行順序・STOP

| logical unit | H2最大 | H6最大 |
|---|---:|---:|
| `scf` | 1 | 1 |
| `spatial_integrals` | 1 | 1 |
| `physical_sector` | 1 | 1 |
| `sector_context` | 1 | 1 |
| `explicit_XDF` | 1 | 1 |
| `CDF_fit` | 3 | 21 |
| `rank_assessment` | 6 | 42 |
| `representation_gate` | 2 | 2 |
| `ground_alias` | 3 | 3 |
| `sector_kernel_reference` | 4 | 4 |
| `PF_round_assessment` | 52 | 52 |
| `full_CAR_context` | 1 | 0 |
| `full_generators` | 2 | 0 |
| `full_kernel_reference` | 4 | 0 |
| `native_model` | 2 | 0 |
| `gaussian_build` | 8 | 0 |
| `circuit_build` | 7 | 0 |
| `compile` | 7 | 0 |
| `operator_decode` | 14 | 0 |
| `segment_assessment` | 7 | 0 |
| `metrics` | 7 | 0 |
| `round_accuracy` | 52 | 0 |
| `energy_assessment` | 4 | 0 |
| `cost_aggregation` | 4 | 0 |
| `cost_comparison` | 2 | 0 |

callは既存のjournalを再利用し、composite unitの内部範囲を契約の`call_units`へ列挙した。H2のfit最大3、XDF分解1、native build／compile各7、operator decode14、13round評価各52。H6はfit最大21、XDF分解1、rank assessment最大42、PF52で、native／shots callは存在しない。rankが早く合格すればfit・rank callは上限より少なくなる。

SOURCE／環境／geometry preflight→新しい共通input snapshot→XDF／CDFのrank ladder→両arm DF予算→selected adapter gate／physical・own-DF ground残差・alias確認→PF。H6はここでSTOP。H2はnative/P0→数値CA・shots→適格候補だけ費用・同じsでCDF/XDF比較→STOP。ground／alias不適合を無視してRPEへ進まず、候補探索を追加しない。

最初のfit不収束、contract／schema不一致、数値correctness FAILで依存する新規callを停止し、開始済みworkだけ回収・partial evidenceを保存する。retry／resume0。CA十分条件を満たさない候補はCA-qualified総費用へ入れない。共通policyでも両armのDF実誤差が異なる点は報告し続ける。

```bash
/path/to/eval-python scripts/run_research_c_c1_h2_h6.py \
  --molecule 2 --contract docs/research/research_c_c1_h2_h6_contract_unapproved_20261025.json
```

既定はmetadata／byte preflightのみ。科学許可後だけ`--execute --fit-python /path/to/fit-python --output <fresh>`を追加する。分子6は別コマンド・別fresh outputで、初回scopeはDF／PFだけ。承認はチャットの許可で足り、承認証明・承認記録・新grantは不要。

## 人工検証・環境・保全

新規14 tests PASS。人工h/g/J/UによるH2 end-to-end、H6人工sector400、sparse CAR／full16 reference／独立generalKの一致、rank不適合と有限探索STOP、失敗時partial保存、phase・scalar・spin順序、反復二乗法、単一fit dispatchのmock、PySCF／AO→MO mock、venv Python symlink保全、passive importを確認した。H6 dispatch試験のPF roundはmockで、実400-sectorのCAR／一般Kは人工行列で検証したことを区別する。

既存回帰は独立processでStage3 112、Stage2 27、一般K13＋29 subtests、ffsim bridge14、計166 tests PASS。新規と合計180 tests＋29 subtests。ffsim初期値生成は人工入力でoptimizerをmock遮断した旧試験を再利用した。旧venv起動testに必要な`.venv-fit`参照はtask内symlinkだけを作成し、既存venvは変更していない。新しいfit経路も`Path.absolute()`を使い、Python symlinkを`resolve()`しない。

評価環境は既存専用venvのNumPy1.26.4／SciPy1.14.1／Qiskit1.3.0／Nature0.7.2／OF1.6.1／OF-PySCF0.5／PySCF2.7.0。fitは別processでffsim0.0.83／JAX0.4.38／Qiskit2.4.2／PySCF2.14.0の既存専用venvをread-onlyで再利用する。既存lockを参照し、共有環境のupgradeやライブラリ本体のpatchはない。fit-workerは固定版をmetadataで確認する。

現在のmetadata probeはaffinity32 CPU・空きRAM約52 GiBで、H6 planは候補4 workersを選択した。fit ladderはfirst-passing条件のため逐次、独立PF候補はresource-aware並列、BLASはprocess内でworkerと協調する。利用可能資源に対する必要配列見積りを使い、人工CPU／RAM／wall capを加えない。今回の人工timingは保存するが、実H6の速度や所要時間の予測に流用しない。

今回の実分子fit／SCF／operator／PF／RPE／shots／総費用の新計算・保存値再解析・実回路build／compileはすべて0。H4のSOURCE・結果・grant・ledger・marker・STOPに差分なし。SOURCE33点を固定し、contract SHA256 `967173d6cb8a2ac695a55e5fd7338776322cef929cb496a33802c33c65517537`、SOURCE manifest SHA256 `380bb20f69c8c0b92d00069a7e2ed33d42f28ff5d1bf500894a536e571cc2a0b`。人工・技術証拠はlocal実行で、CIや外部独立再現ではない。

## 次の担当とmandatory STOP

ユーザー／研究Cが、H2先行の196-call条件、rank ladder／CA/8予算／新guard／explicit-XDF baselineを採択し、H2科学実行を別途許可するか判断する。Codexは許可された分子とscopeだけ実行する。H6は別途129-callのDF／PF限定許可が必要で、nativeへの進行はその結果と負荷を確認してから判断する。必要な重要科学レビューも別の明示許可後に行う。

今回のSOURCE・実行案・人工結果をcommit・push・remote照合後、**mandatory STOP**。新規科学計算へ自動進行しない。
