# H4 保存順XDF8：表現・ground/alias・PF再資格付けの技術準備

2026-10-26 JST。起点 `e3c8b3e6550b2866b6e370a13780a0ee41ebefc8`。独立branch `codex/research-c-c1-h4-xdf8-requalification-preparation-20261026`。

**第1段階の最小runnerと人工検証を完成した。実H4の数値decode・科学評価・回路生成は0。公開照合後mandatory STOP。** ユーザーは保存OpenFermion元順prefix8を追加比較候補に採用したが、その科学実行は今回の許可に含まれない。

- [評価SOURCE](../../src/trottertracks/c1_h4_xdf8_pf.py)／[passive既定CLI](../../scripts/run_research_c_c1_h4_xdf8_pf.py)／[人工test](../../tests/management/test_research_c_c1_h4_xdf8_pf.py)
- [固定未承認93-call契約](research_c_c1_h4_xdf8_pf_contract_unapproved_20261026.json)
- [SOURCE manifest・テスト原本・byte preflight](../../artifacts/research_c_c1_h4_xdf8_pf/2026-10-26/technical/)
- [前段のDF予算測定](research_c_c1_h4_xdf_prefix_budget_execution_20261026.md)／[旧CDF8・XDF10精度費用結果](research_c_c1_stage3_deterministic_accuracy_execution_20261025.md)

## 1. 対象と来歴

直鎖H4・隣接1.00 Å・STO-3G、Nα=Nβ=2、sector36。元のnumber表現、元g補正κ・核scalar・spin/bit/sector順・外側frame順を保持する。新XDFは保存OpenFermion1.6.1 eigh由来λ/Gの**先頭8成分だけ**。全frame ordered symmetric S2、s=1/4、δ=0.1、m=0..12、PR split非適用。CA=.00159 Ha、DF reserve CA/8=.00019875 Ha、own-XDF8 sector ground oracle／状態準備費用除外。

| identity | 固定値・扱い |
|---|---|
| 元P-H4 packet SHA256 | `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50` |
| 元入力生成SOURCE／公開commit | `0c329b30a81accb35c212be8defa66fee0dd0f12`／`86b8b002704aced8ca239414080f53793de81658` |
| 前段DF予算科学SOURCE | `f7bc730d12ad704cdb72cc8a82d2bd973d01b2e5`、消費済み356-call run01を保全 |
| 保存prefix8 DF誤差 | norm `9.533753411107529e-5`、allowance `1.480439843662034e-9`、保存sum `9.533901455091894e-5` Ha、PASSを引用。再計算なし |
| DF allowance | 前段の `binary64_operation_scale_guard8_v1` を再利用。厳密enclosureへ昇格しない |
| CDF8 fit SOURCE／export SHA256 | `da2edb465d5ae6432439fbd1df8e57512ee59c11`／`c33f782d3224aa80968e1e6ea70f5ae3f764b872b5bce545c38875c83f3c4e08`。今回byte確認のみ |
| 旧CDF8／XDF10 | 表現評価SOURCE `bec41ac7dc7e034bd664817acfdcfed858ddadd0`、精度評価SOURCE `002d0dc2e1dafd8cb71735c43253fa054e675f6d`。約27%の旧条件付き総RZ削減は履歴として保持 |

新評価SOURCE manifest SHA256：`4f4d3261411176bebdade53d3a621badfb120839df49e4ac9f59396cad9489cd`。

未承認契約SHA256：`39af588fee2d879cfa1bf38073dac2bc295cc6f8f610a6be7ef6c105f26fa63c`。

今後の実行commitと上記manifestで新評価SOURCEを識別する。旧fit／元入力／前段DF測定SOURCEのidentityを新評価SOURCEと混同しない。元packetは旧git objectから新worktreeのignored `.external/P-H4.npz`へbyteコピー・hash照合した。旧結果は変更していない。

## 2. 再利用した実装と最小差分

| 既存資産 | 今回の利用と適合性 |
|---|---|
| `c1_h4_xdf_prefix_budget.context/prefix_model/prefix_matrix` | H4の既存PhysicalSector、OF→Qiskit permutation、保存prefixモデル、36 matvecをそのまま呼ぶ。旧9rank runnerは再起動しない |
| `c1_stage1_execution.load_input` | 固定P-H4のh/g/c/MO・保存κ/λ/Gのみwhitelist decode。旧ground/energy/circuitは読まない |
| `c1_stage1.corrected_kappa/sector_indices/bit_reverse_indices` | 元gのκ確認を1回、従来H4 basisで照合。rank別κ再補正なし |
| `c1_molecule_extension.SectorCAR` | 既存疎CARの完全one-body／density-square因子。H6 runnerの分子・rank・guard・aliasはコピーしない |
| `diag_hermitian`、`c1_general_k_adapter` | 固定された各4×4 Gを内部対角化して一般Kへ表現変換。新しい二体DF分解やffsim fitterは呼ばない |
| `component_check` | 既存max-entry／finite／shape gate。Frobeniusは診断値。DF sector誤差の再評価に使わない |
| `MolecularBackend._ground` | XDF8自身のsector eigensolverを呼び、groundと全eigenpairを保存。旧CDF/XDF10を構築する`prepare()`は呼ばない |
| H4 `MolecularBackend._state` のalias条件 | 既存H4の[-10,10] Ha windowとβ_totalを採用。full-state構築部分は今回不要 |
| `c1_molecule_extension.PF_rounds`、`c1_stage3_reference.reference_kernels`、`ordered_schedule` | 既存S2、expm、operator 2-norm、初期powerと反復二乗。XDF8固有eigenpairを両sで再利用 |
| `integrated.Calls/resource_plan` | 既存のthread-safe call計上・first failure・資源選択。承認証明の管理機構は追加しない |

追加は新しい薄いadapter／CLI／人工testだけ。旧計算SOURCE・契約・raw・grant・ledger・marker・STOPは無変更。新CAR、PF法、ground solver、fitter、Gaussian合成器は実装していない。

## 3. 第1段階のcorrectnessと判定

モデルは `H8=cI+κ̂+Σ(l=0..7) λ_l Ĝ_l²`。κは元gからの保存補正値、cは核scalarを1回だけ含むbase。各Gのspin構造が**厳密に**`kron(G_spatial,I2)`であることをgateとし、微小off-spin成分も落とさずFAILする。完全なone-body／square因子がNα/Nβを保存するため、その積とsector制限は整合する。個別Gaussian gateをsectorへ射影する検証ではない。

独立照合は以下の3経路で行う。

1. 保存DFモデルの既存行列作用を36基底へ適用したXDF8 sector target。
2. 既存CARからbaseと8個の`λ Ĝ²`を構築したraw経路。
3. `G=U diag(η) U†`から`J=2λ ηηᵀ`とし、既存number adapterの`(1/2)ΣJij NiNj`をI/Z/ZZで展開する一般K経路。

内部G対角化は外側frameのreorderingや再分解ではない。CARと保存matvecのOF順を既存H4 permutationで同じQiskit順へ変換し、9因子をraw/K間、両sumを保存targetと照合する。max-entry≤1e-10、rtol0、finite mandatory。物理Hamiltonianとの差のDF normは求めない。

ground residual≤1e-10、正規化誤差≤1e-12、有限36次元eigenpair／basisを確認する。ground energyはXDF8固有で、XDF10の状態を流用しない。物理系との関係は、同一入力・sectorの保存DF誤差を使うWeyl型energy対応に限定し、physical ground overlapは未測定のnullとする。

aliasはH4既存window `E0∈[-10,10]` と `max(abs(window))*0.1+π/6<π`。window外は`UNRESOLVED_GROUND_ALIAS`としてPFへ進まない。residual・正規化等の不正はcorrectness FAIL。

各sについて `W_s=S2(h=0.1/s)`、`n=s*2^m`、`t_m=0.1*2^m` とし、

`E_m=||W_s^n−V exp(−i t_m E)V†||₂`

を既存関数で測る。全26セルについて登録十分条件 `E_m≤sin(π/12)` を用い、事後guard・閾値変更はしない。これは既存H4の**数値pilot判定**で、rounding enclosureではない。決定論なので補正RTE差0・attenuation1は構造上の値であり、未知合成誤差の0認定ではない。

| 分類 | 意味・後続 |
|---|---|
| correctness FAIL | SOURCE／入力／finite／basis／表現／ground不正等。依存する新規callを止め、開始済みを回収してpartial evidence保存 |
| `INFEASIBLE_REGISTERED_PHASE_ROOM` | 13roundのいずれかが登録PF十分条件を満たさない。物理的不可能性ではなく、他の登録セル・候補は評価を継続 |
| `UNRESOLVED` | 必要な証拠がない。nullを保持し、0補完しない |
| `POTENTIALLY_FEASIBLE` | 保存DF予算・表現・ground/alias・全13 PF条件がPASS。native評価を提案できるが、最終CA・資源優位や次段階実行許可ではない |

## 4. 固定科学callと実行規模

| call分類 | 上限 |
|---|---:|
| saved input／DF gate／sector context／input identity | 1ずつ＝4 |
| 保存prefixモデル／実matvec | 1＋36＝37 |
| raw factor／K model／表現gate／sum gate | 9＋1＋9＋1＝20 |
| own ground eigenpair／ground-alias | 1＋1＝2 |
| S2 kernel／PF round／candidate summary | 2＋26＋2＝30 |
| **合計** | **93 logical units** |

各composite callに含む内部処理は契約の`call_definitions`に記載した。例えばK model内の固定G内部対角化8件、各S2 kernel内の17 factor expmを含む。過去のcall残枠や旧runnerを再利用しない。

dominant workloadは36×36 operatorの26個の2-normと2 kernel。全eigenpairを一度だけ求め、既存反復二乗と2個の独立s jobを利用する。resource_planが実CPU affinity／cgroup／空きRAMに応じてworkerを選び、複数job時はprocess-local BLAS1でoversubscriptionを防ぐ。CPU/RAM/worker/wallの恣意的な上限、共有環境の変更は追加しない。今回の第1段階にfull256/512行列や大きなSVDはない。

fresh outputは`artifacts/research_c_c1_h4_xdf8_pf/<execution-date>/run01`。既存output、run02、retry／resumeを拒否する。保存予定はcontract/SOURCE、入力identity、DF引用、9因子とsector indices、ground/spectrum、representation、2 kernel、26セル・2候補、call journal、resource plan／実測、failure／partial／STOP。正常・未適格・FAILのいずれでもmandatory STOP。

## 5. 人工・回帰検証とpassive preflight

**新規22＋既存92＝114 tests PASS**。実分子の人工置換を使い、実H4の数値decodeや再解析は行っていない。[機械可読検証結果](../../artifacts/research_c_c1_h4_xdf8_pf/2026-10-26/technical/validation.json)。

| 範囲 | 結果 |
|---|---|
| 人工E2E | 実loader／prefix matvec／CAR／K／ground／PF経路で26セル・93 attempt/COMPLETE、独立dense chronological S2／expmと短・長roundを照合 |
| 規約 | 正負λ、非対角G、prefix8のみ、未使用末尾λ不変性、元frame順、核scalar1回、非zero gのκ確認1回、Qiskit占有順とOF permutation |
| 分岐／STOP | PF PASS／NOT_MET／UNRESOLVED、alias外、非finite／負PF誤差、spin項、κ、Hermiticity、K係数、basis/permutation、ground不正、SOURCE／input drift、fresh output拒否 |
| 既存回帰92 | prefix DF budget29、H6 saved PF21、H2/H6接続14、Stage3 feasibility16、deterministic accuracy12 |
| passive | 実packet／保存結果のbyte hash、21 SOURCE files、依存8版を照合。CLI defaultはBYTE_PREFLIGHT_PASS、NumPy未import、分子科学call0、run01未生成 |

開発初期の人工FAILはcomplex zeroのdtypeとOF/Qiskit変換不足によるものだった。新adapterだけで修正し、初回8 FAILと次回5 FAILの原ログも保存した。最終人工22件・関連92件のFAILは0。

既知の旧Stage1 full256人工回帰1件のSciPy gesdd／32-thread異常は[前段報告](research_c_c1_h4_xdf_prefix_budget_preparation_20261026.md)に保全しており、今回再実行・修正していない。新sector36経路は当該旧full256 SVDを使わない。114 PASSでその旧問題が解消したとは扱わない。

評価環境は既存の独立`/tmp/prt-c1-stage2-preparation-20261025/.venv/bin/python`を**literal pathで**読取利用（Python3.11.1）。NumPy1.26.4／SciPy1.14.1／OpenFermion1.6.1／Qiskit1.3.0等は契約と既存requirements.lockを再利用した。venv・SciPy・ffsim本体・共有cacheは変更なし。テスト時のBLAS1は小行列／worker協調のためで、科学契約の資源capではない。

許可前の確認コマンド（科学計算をしない）：

```bash
/tmp/prt-c1-stage2-preparation-20261025/.venv/bin/python \
  scripts/run_research_c_c1_h4_xdf8_pf.py \
  --contract docs/research/research_c_c1_h4_xdf8_pf_contract_unapproved_20261026.json \
  --input .external/P-H4.npz
```

将来、固定公開SOURCE／契約／元入力／fresh outputの別明示科学許可を得た場合のみ、同CLIへ`--execute --output artifacts/research_c_c1_h4_xdf8_pf/<execution-date>/run01`を付ける。承認証明・原文保存・receipt／ID／新grantは不要。

## 6. 第2段階の接続案：公平なnative・合成誤差（未採択・未実行）

**推奨はCDF8／XDF8双方を同じprecision／P0で再資格付けすること。** XDF8だけ精度修正して旧CDF8 metricsと混ぜる比較は採用しない。

`c1_h6_native_precision.PrecisionBuilder`のGaussian処理はmode数から構築し、H6粒子数を固定していない。同じtask-local `_basis`差替えをH4の両armへ接続できる。既存native `prepare_cdf`には保存CDF8 J/U、`prepare_xdf`には保存κ・元順λ/G[:8]・scalarを渡す。`c1_h6_compiler_precision.compiled_wrapper`もqubit数非固定で、同じP0 gate処理のdeclared phase加算のみを安定化する。新Gaussian法や一般相対Gaussian融合を追加しない。

両armにQiskit1.3.0／opt1／seed17／rz,sx,x,cx／approximation_degree1／qubits_initially_zero=falseの同一profile、同じ同一basis共有・角度結合・serial fenceを適用する案。旧CDF metricsの無条件再利用はしない。target・全ordered IR（gate/angle/qubit/global phase）・compiler profileの一致と新precision経路のcorrectnessが別途確認できる場合に限り、対応する旧metricsを再利用できる。IRが変わる場合は両armを再測定する。

最小案は**CDF8/XDF8×s1/s4の4 kernel＋共通head/X/Yの3 shell＝7 raw build／7 P0 compile／7 metrics**（必要Gaussian cache・reference／検証callは別計上）。literal/sharedの両経路を比較するなら8 kernel＋3 shell＝11件へ増えるため、研究Cが必要性を判断する。未観測のliteral/shared差を0や削減量と仮定せず、route・厳密IRの確認範囲を将来契約で固定する。

局所base＋8 frameの向き・時間・κ/scalar・controlled相対位相、actual adjoint、X/Y readoutを既存NativeBuilder／raw CAR／H4 full referenceで検証する。CDF8の保存full CAR9行列はidentity確認のうえ再利用できる。XDF8は将来、元λ/Gから既存`xdf_car_generator`でfull9因子を用意する（新fit／DF norm不要）。今回のsector generator・eigenpair・PF2kernelを後続へ保存する設計とした。

合成誤差の選択肢は次の通り。**H4の小ささからAを推奨するが、今回正式採択しない。**

| 案 | 規模・科学的範囲 |
|---|---|
| A：既存H4 full operator | controlled512（complex128行列約4MiB）。4 raw/P0 kernelsと3 shellを独立referenceと比較し、raw-reference＋P0-rawのFrobeniusを測る。全512入力を覆える。full SVDは不要で、旧大行列gesdd問題を避ける |
| B：H6 ColumnStream再利用 | ancilla×sector36＝72入力列、出力はfull512。4kernelで288列、leakageを残差へ含む。完全な入力部分空間の資格Bであり、full Fockのglobal boundではない。個別Gaussian gate射影や有限random probesで代用しない |

raw-reference、compiled-raw、head、X/Y tailをそれぞれ1回含め、既存`round_synthesis`／`repeat_bound`等でn=s2^mの13roundへtelescopingする。数値unitarity欠損があれば既存reference_infoとα growthを保持する。浮動小数Frobenius上界は数値pilotで、CAR/expm／回路行列復元／normのroundingを囲った厳密enclosureではない。有限probeのmaxを全operator上界へ昇格せず、未資格項はnullとする。

後続call概略は、7又は11件のbuild/compile/metricsに、必要な局所因子、full reference、raw/P0 action／operator、adjoint／X/Y、4×13又は8×13合成profileを加える。案Bなら288又は576の全入力列actionを別計上する。coverage・route・precision条件の採択前に科学call上限を過去からコピーしない。次の技術準備で契約を固定する。

## 7. 第3段階の接続案：共通CA・RPE・条件付き総資源（未実行）

既存`AccuracyTarget`／`assess_round`／`assess_energy`／Hoeffding solver／固定N検証、`compose_serial_cost`／`aggregate_totals`／`compare_totals`を再利用する。共通CA=.00159 Ha、β_total=π/6、β_stat=π/12、候補failure=.01・各round/axis=.01/26、ideal IID Born、own-DF ground oracle、状態準備除外、13round X/Y、同P0のserial費用モデルを維持する。

接続順序は、PF operator誤差＋合成complex signal誤差を同じsignal単位で1回ずつ扱う→位相margin→統計座標精度と整数shots→RPEのphase-to-energy換算→最後にDF norm＋guard（Ha）を1回だけ加える。DF誤差をcomplex signalへ直接加算しない。決定論tailのRTE/law追加差0は構造上の条件に限り、未知合成／reference enclosureはnullのまま。

XDF8自身のDF・PF・合成証拠で新shotsを算定し、同一precisionで資格付けしたCDF8と比較する。XDF10のground／PF／shots／cost／CAは一切流用しない。CDF8の保存PF／DFはtarget identityが同一なら再利用できるが、新precisionで合成marginが変わり得るため、両armに同じshots solverを適用する案を推奨する。旧CDF shotsを固定再利用するなら、新marginでそのNが適格か検証する別条件が必要。

概略は4候補×13＝52 round精度計画、4 energy判定、104 axis serial費用行、4 totals、同sの2比較。主指標は条件付き総RZ、副指標はRZ depth/CX count・depth/total depth/size。両armが同じCA条件で適格の場合だけ比較へ進める。depthはserialモデルの和であり全round再compile最小値ではない。旧約27%削減はCDF8／XDF10という登録候補間の歴史的結果、新CDF8／XDF8は独立した追加結果とする。厳密CA・無条件優位・状態準備込み資源へは拡張しない。

## 8. 次の判断・mandatory STOP

第1段階の技術blockerはない。実H4でXDF8の表現／own-ground／alias／26 PF条件が成立するかは**未評価**であり、人工PASSでは認定しない。

次は**ユーザー／研究C**が公開SOURCE・93-call契約を確認し、第1段階だけの科学実行を別途明示許可するか判断する。許可があれば**Codex**は固定SOURCEを変更せずfresh run01を一度実行し、全gate・26セル・2候補・call実績・STOPを公開する。第2段階のcoverage／route／共通precision採択、第3段階のshots方針、baseline公平性と主要claimは研究Cの別判断。

今回の新fit・新DF分解・実H4数値decode／DF norm／ground／PF・回路build/compile・合成・CA・shots・総費用はすべて0。旧H2/H4/H6・失敗履歴・SOURCE・grant／ledger／marker／STOPは保全。remote照合後mandatory STOPし、第1段階の科学計算も第2・第3段階も自動開始しない。
