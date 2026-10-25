# 研究B：A0高精度検証・B0 Native Pilotの実行準備

2026-10-25 JST。**実装・人工検証・read-only preflight完了。A0高精度演算・B0実回路build／compileは未実行。別々の科学実行許可が必要。公開照合後mandatory STOP。**

## 1. 起点・固定SOURCEと保全

起点は[調査設計3d40b053d](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3d40b053d78357bdf4377aaabd18afcfb7571413/docs/research/research_b_outer_numeric_uncertainty_native_cost_design_20261025.md)。科学SOURCE・原本・STOPは旧22行 `26e8734aff3969dc8077cb058826d12394be28e1`／結果公開 `33715848bfb83d49ec4e54d6eef481012a7441e4` のまま保全する。旧16行・36行の科学結果・契約・grant／ledger／marker／STOPも変更しない。

独立branchは `codex/research-b-outer-a0-b0-ready-20261025`。SOURCEを先に2段階でcommitし、独立clean checkoutを作った。本書・索引は後続文書commitに置く。

| 実行単位 | 固定科学SOURCE候補 | Clean checkout |
|---|---|---|
| A0 | **`1e3921ba16e5d531e9a3010cffa9ff717609a93a`** | `/tmp/prt-research-b-outer-a0-science-20261025` |
| B0 | **`8d0cb96df9d1f6229f207550634c7e11aff2851c`** | `/tmp/prt-research-b-outer-b0-science-20261025` |

B0はA0 commitを祖先に持つが、A0は先の固定checkoutで実行する。共通IOはA0 commitで固定した版をB0も使用する。A0/B0の実行許可・入口・namespace・scopeは独立しており、一方の許可を他方へ流用しない。

## 2. 実装と再利用

| 部品 | 今回の接続 |
|---|---|
| [`a0_reference.py`](../../src/trottertracks/b2d_evaluator/a0_reference.py) | 独立数学列・metadata graph・90/128桁参照・snapshot lift・比較記録。DenseBackendの数値演算を呼ばない |
| [`b0_native.py`](../../src/trottertracks/b2d_evaluator/b0_native.py) | 固定representative・native renderer・独立CAR full-Fock参照・event mean・compiler／費用接続 |
| [`followup_io.py`](../../src/trottertracks/b2d_evaluator/followup_io.py) | 既存 `_check_source`、`ScopeLedger.call`、`Evidence`、失敗保存／STOPのA0/B0用接続。新しい汎用実行／認証frameworkではない |
| [`reference.py`](../../src/trottertracks/pr_dictionary_validation/reference.py) | 既存 `precision_context`／`exact` をA0で再利用。旧fixture専用のoperator生成を今回の保存DF入力へ無理に移植しない |
| [`df_partial_s2.py`](../../src/trotterlib/df_partial_s2.py) | `append_block_evolution`を追加し、既存 `_append_block` のGaussian往復・diagonal primitiveを明示時間で再利用。partial-S2 requestにouter全列を偽装しない |
| [`rpe_hadamard_interrogation.py`](../../src/trotterlib/rpe_hadamard_interrogation.py) | 終端X/Y規約を `append_hadamard_readout`へ共通化。別入口 `build_prepared_hadamard_wrapper`でmode0準備・measurementを含む。旧RPE q=1/2/4の契約は維持 |
| [`gates.py`](../../src/trottertracks/pr_dictionary_validation/gates.py) | 既存 `compiled_wrapper` のofficial-definition展開・phase保持compile・実QCを返すcacheを再利用 |
| [`rte.py`](../../src/trotterlib/rte.py)／[`df_rte_qiskit.py`](../../src/trotterlib/df_rte_qiskit.py) | 既存config、合法event factory、event operator、finite moments、exact weighted single-step mean、native sequence builderを利用 |

新しい参照が必要だった理由は、旧dense evaluatorはcomplex128に固定され、既存高精度operator helperは別fixtureのCAR／辞書仕様を入力とするためである。B0追加の理由は、旧partial-S2反復interfaceが単一時間／configを反復し、既存Hadamard費用経路はstate preparationを明示的に除外するためである。

`TranspiledCircuitCostCache`はmetricのみのhitでは正当性用QCを返さず、標準transpileへの接続も固定されている。B0では既存phase保持compilerのlocal circuit cacheを用い、実compiled QCを保持して検査する。`CompiledMetricAccumulator`のSEを決定論的代表trajectoryへ適用しない。`plan_compiled_cost_workload`のwork-item／X/Y二回路の考え方はscopeに対応させたが、科学package importを伴う旧plannerをmetadata preflightから呼ばない。費用値は既存 `f2_execution.circuit_metrics`→`rte_compiled_cost`のgate／filtered-depth定義をそのまま使う。旧研究の価格・許可・科学的資格は移送しない。

共通 `Evidence`への変更はprecision-status表示を実行単位に合わせられる1行のみで、旧既定表示は不変。AGENTS.md・共通運用指示・共有環境は変更していない。

## 3. A0：固定8行・入力target・独立性

S1（SABA2/q4/A2/r2K2）、F1（refined-S2/q2/f8/A2/r2K2）、B3（Yoshida4/q2/A2/r2K2）、R4（Yoshida4/q2/A3/r2K4）、各T=±0.18。計8行×90/128 decimal digits=16 case-precision。n_D=1、N=1／mode0、LD2 weight-ranked、既存synthetic2-mode one-body＋3DF。分子geometry／AO basis／分子DF rank探索は非該当。新grid・係数・物理条件を追加しない。

保存元は `artifacts/research_b_outer_rte22/20261025/analytic-once/` のinputs.json／inputs.npz／binding.json／graph.json／result.json。Preflightで結果公開commit33715848bのGit blobと現ファイルbytesを照合する。これは科学入力同一性の管理で、承認証明ではない。

実行時にだけNPZを読み、complex128／2×2／state2要素・finite値・key集合を検査する。`binary_fraction`は `float.as_integer_ratio` を使い、各complexの実部・虚部を正確な二進有理数としてliftする。Scalarは保存Fractionを保持する。時間はcaseのFraction、PF／SABA係数は既存float由来のFractionを保持し、理想的な1/√3等へ置換しない。

保存H_physをD+R+cIへ上書きせず、両者の差を診断に残す。DF再分解・再対角化・別fixture生成はない。Primitiveから高精度で和・積を行う数学targetと、旧complex128の浮動評価を比較する。これは保存binary64モデルの数値診断であり、元の十進literalや化学モデルへの保証ではない。

参照はmathematical productを右積で構成する。D内部は独立したD0/2→D1/2→D2→D1/2→D0/2の数学列に既存S2係数を適用する。Candidateの `append_operations`／DenseBackend／finite moment helperを高精度参照へ流用しない。Outer係数・既存published係数の取得とscope metadataだけは既存descriptorから再利用する。

90桁はmpmath `expm(...,method='taylor')`、128桁は`method='pade'`。既存 `precision_context`のcloneを使い、global精度を変更しない。Installed mpmath1.3.0の `matrices/calculus.py` を静的に読み、Taylorのscaling／ctx.eps収束判定・精度復元、Padéの別 `_exp_pade` 経路を確認した。独立アルゴリズムを使うが、同じライブラリと凍結入力を共有する経験的確認である。

Operator 2-normは `ctx.svd(matrix,compute_uv=False)` の最大singular valueを使う。Installed `eigen_symmetric.py`でcomplex SVD／引数／返却仕様を確認し、未実装の `mnorm(p=2)` やFrobenius normへ置換しない。今回はexpm・SVDを実行していない。

## 4. A0：scopeと保存指標

Metadata graphは**3,400 node**。旧22行1239 dispatch／274 keyを流用していない。

| 登録境界 | 固定値 |
|---|---:|
| 既存8行×2精度 | 16 case-precision |
| 5対象の高位request／signal | 各80 |
| Shared matrix exponential | 412（精度間共有なし） |
| Explicit product multiply | 2,348 |
| Polynomial recurrence term／add | 各88 |
| Operator residual 2-norm | 96 |
| D／outer unitarity診断node | 40 |
| D診断の行別収集 | 16 |
| N_j／stage Taylor bound | 各24 |
| 全体N／bound合成 | 各16 |
| Scalar phase／phase差診断 | 各4 |
| Scalar×matrix | 44 |
| Raw生成divide | 16 |
| Identity／input数値診断 | 各2 |
| Snapshot読込み／参照初期化 | 各1 |
| 90/128・旧値比較 | 8 |

Phase差診断は同一scalar・Tでのcmath binary64 phaseとの比較を含む。Snapshot初期化に各精度の正確なarray lift、D和・R形成を含む。Polynomial termは一回のmatrix積とscalar演算、各expm／SVD／Taylor scalar級数のlibrary内部は対応する高位境界内に含める。親子・入れ子境界を加えて独立科学call総数とは呼ばない。登録されたkind／identity外・重複call・未完了scopeは拒否する。最大event記録は6,852。

各case-precisionで5対象matrix／complex signalを全桁文字列で保存する。Physical対各近似、D近似、finite-RTEのoperator残差、stage別・合成Taylor bound、N、H identity、tail norm、D/R Hermiticity、state norm、D／outer unitarity、scalar phase差も保存する。

128桁の比較段階で90桁との差、旧signal・bias・operator残差・nominal boundとの差を保存する。90/128間のsignal／residual norm／N／合成boundの差≤1e-40を経験的な安定目安にする。これをu_numの保証値にせず、**u_num=null／certified=falseを維持**する。Threshold緩和・追加精度・新grid・A1包含実装はない。

安定性未達または高精度nominal boundとの不一致が残れば、結果を保存して `INCONCLUSIVE_MANDATORY_STOP`。安定して整合した場合も `EMPIRICAL_CONSISTENCY` でありcertifiedではない。保存入力のnorm／Hermiticity前提・boundの丸め方向・積／expm／normの寄与は診断として残し、理論破綻や実装バグを都合よく断定しない。

## 5. B0：固定candidate・representative・renderer

B3、D1、R2、R4、B4、F1の6候補×両T=12 case。全て既存2-mode、N1／mode0、LD2、n_D1の同じrecipeである。各case4代表で48 trajectory。確率標本ではなく、費用最適候補の採択でもない。

| 代表 | Event descriptor |
|---|---|
| 0 | 全short-step order0、最初のcomponent |
| 1 | 全short-step order0、componentを循環 |
| 2 | 最初のR short-stepだけ最大保持次数K、componentを循環 |
| 3 | Tと逆符号のR stage（複数ならappend順の最初、なければ最終R）の最終short-stepだけorder2、他order0 |

T<0でもYoshida4中央の符号反転を検査する。「最負stage」を単純選択して中央を落とさない。各eventは既存 `_make_event`を使い、K／order／component index／component sign・rotation/product順・signed configを保持する。実行時に当該orderとevent probabilityが正であることを検査し、support外ならSTOPする。人工event lawや新samplerは作らない。

Native rendererは既存 `OuterPlan.body` のappend順を使い、DStageの実現block列をpublic化したnative block primitiveで描画し、R occurrenceには固有r/K/configとevent列を渡す。同一context／family／tauのD kernel反復の合法融合を既存 `fuse_d_stages` で保持し、異tauの近似Dを時間和へ置換せず、R障壁を越えて融合しない。F1は16実step。

Gaussian basis往復を制御せず、中心diagonalだけをcontrolledにする。Scalar／extracted identityは全Tで一回、controlled時はancilla phaseとして入れる。RTE sequence builderのevent phase・identity-relative phaseを維持し、負時間meanを正時間のinverseへ置換しない。A4／processorは対象外。

Full wrapperはmode0の対応qubit X→ancilla H→controlled evolution→（YのみSdg）→H→ancilla measurement。X/Yは同一trajectoryから生成する。旧state-preparation除外costをfull-wrapper費用として返さず、旧RPE roundの型・q制限を迂回する偽のresultも作らない。

## 6. B0：独立full-Fock参照・phase検査・compiler

独立参照は2-mode CARから4×4 one-body／DF平方演算子を形成する。Candidate回路のOperatorを独立参照の入力には使わない。RTE componentは元DF行列の独立orbital projectorとZ/ZZから形成し、既存event operator関数でsigned product／rotationを組み立てる。D・outerはmathematical productを右積で作り、rendererのappend列をそのまま数値化する参照にはしない。

Qiskit little-endianではmode0をqubit0、full-state index1とする。OpenFermion sector `[1,2]` とQiskit full indices `[2,1]` の対応を明示し、保存2×2入力とfull-Fock参照のsector射影を比較する。H／tail identity、selected DF0,1とresidual2、3 retained components、stateの対応が不一致なら比較前にSTOPする。Gaussian basisの実数値整合は未検証で、人工PASSから補完しない。

Control-zeroはI4、control-oneはU4の8×8 block diagonal参照と比較する。Wrapperの独立参照は明示的なH／Sdg／mode0-XのKronecker積で作り、measurement前のfull 8×8 operatorと比較する。System sectorだけの一致・global phase fitでcontrolled相対位相を代替しない。

Compiler profileはQiskit1.3.0、basis `rz/sx/x/cx`、optimization1、seed17、approximation_degree1.0、qubits_initially_zero=False。Backend／target／coupling map・layout／routing指定なし。既存 `compiled_wrapper` のofficial-definition phase-preserving loweringを使用し、共有EquivalenceLibraryを変更しない。

実QCを返すcacheを今回のrun内だけに作る。96はcompile要求数で、実transpileはcache miss数。実際のmiss数・hit数・cache keyを結果へ保存し、metric-only hitからcompiled operator PASSを推定しない。Compiled circuit本体は正当性確認までメモリに保持する。全gateはabsolute operator／signal誤差≤1e-12、rtol0、phase fitなし。これは経験的correctnessで、γ certificateではない。

## 7. B0：固定scope・correctness先行

Metadata inventoryから次を確定した。

| 登録境界 | 固定値 |
|---|---:|
| Candidate／trajectory | 12／48 |
| System U／controlled-U build | 各48 |
| X/Y full-wrapper build／compile要求 | 各96 |
| System／controlled operator gate | 各48 |
| Raw wrapper／compiled wrapper operator gate | 各96、operator gate計288 |
| Raw／compiled X/Y expectation診断 | 各96、計192 |
| Single short-step mean | 24 logical要求、event項1692 |
| Full-space analytic 5対象mean確認 | 12 case |
| Unique RTE config | 76 |
| Unique event descriptor／independent event matrix | 各1740 |
| Native RTE sequence build | 492（system／controlledを区別） |
| Native D block build | 372（system／controlledを区別） |
| Shared reference exponential | 206 |
| Shared finite moments | 16 |
| Cost measure | 96、全correctness後のみ |
| Fixture／split／DF preparation／full input／input gate | 各1 |

1692は24 logical single-stepのevent項参照数で、unique event1740には別の代表trajectoryに必要なeventも含む。Node／event共有で物理occurrenceを削除しない。各caseで最初とT逆符号stage（なければ最終）を選び、single short-step exact weighted meanをfinite polynomialと照合する。有限全trajectoryのexact列挙や48代表の単純平均によるmean認定は行わない。

Full analytic meanの12境界には、D/R product・N・scalar、5対象signalと保存sector結果への対応確認を含む。Event operator境界にpaired product／rotationを含み、各native block境界にGaussian往復・diagonal primitivesを含む。Library内のelementary arithmetic・BLAS内部を別の独立callと数えない。親子境界を合算した「全job独立call数」は定義しない。最大event記録11,414。

入力→event support／single-step mean→full mean→system／controlled→raw wrapper→compiled wrapper／signalのgateを満たしてから、96回のcost measurementへ進む。途中gate失敗時は費用表を完成結果として返さず、完了済みcorrectness・部分結果・理由・STOPを保存する。

RZ countを主指標とし、既存定義のCX count、RZ/CX depth、total depth、circuit size、measurement countを併記する。Source／input identity、config／distribution、component order、signed representative descriptor、gate誤差、raw／compiled expectation、compiler／cache実績を保存する。代表trajectoryの観測費用であり、**expected_cost=null、shots=null、gamma=null**。期待費用MC、shot加重G、Clifford+T合成、全RPE費用の認定はない。

## 8. 保存・失敗・STOP・実行境界

既存 `Evidence`でbinding.json、inputs.json、graph.json、events.jsonl、partial_results.json、result.json、status.json、STOP.jsonを保存する。inputs.npzは旧snapshotをread-onlyで参照するため新規複製しない。入力元commit・5ファイルblob・versionをbindingへ保存し、SOURCE／snapshot identityを結び付ける。未登録call・重複・未完了scope、input不一致はSTOP。

専用namespaceはA0 `artifacts/research_b_outer_a0/`、B0 `artifacts/research_b_outer_b0/`。排他的fresh mkdir、上書き／resume／自動retry禁止。正常、科学的FAIL、INCONCLUSIVE、技術例外・保存失敗でmandatory STOP。旧 `Evidence`の16 MiB総枠・各JSON1 MiB・event64 KiB・STOP reserve256 KiBを使用し、今回のmetadata graph JSONはA0 296,614 bytes／B0 558,004 bytesで枠内。これらは出力制御で、CPU／RAM／worker／wall-time制限ではない。

Import・`--inspect`・read-only preflightでは科学packageをimportせず、array／operator／回路を作らない。実行は個別の `--execute --source-commit … --output …` だけ。ユーザーの許可はCodexの運用で確認し、承認ファイル・ID・原文保存・certificate／grant／packet／receiptは要求しない。

## 9. 人工検証結果

**既存119＋A0/共通10＋B0 9＝138件PASS、最終FAIL0。** ローカル人工検証であり、CI／外部再現・科学的correctness・高精度保証の証拠ではない。

- A0/共通：8行／2精度・80 request、正確なbinary lift規約、signed polynomial／PF metadata、独立Taylor／Padé呼出し、SVD接口、snapshot bytes、call拒否、全経路のstub保存、失敗／STOP／fresh競合、CLI誤起動防止。
- B0：6候補×両T・4代表、r/K・component順、中央符号反転、scope全件、actual rendererと独立参照の走査をopaque valueで確認、全gate後の費用測定、gate失敗で費用0、shared block／readout関数をstub circuitだけで確認。
- 旧119：adapter15、evaluator14、input_preflight16、stage1_launcher15、execution_ready4、simple_execution8、outer_preparation19、outer_execution14、outer_rte22 14。旧36／16／22行の技術回帰PASS。

```bash
/usr/bin/python3.11 -I -B -S tests/management/test_b2d_followup.py -v
/usr/bin/python3.11 -I -B -S tests/management/test_b2d_b0_native.py -v
```

全てCPython3.11.0rc1、stdlib／opaque mock／人工scalarとscientific import barrierを使用した。保存科学arrayを演算に使わず、実mpmath context／matrix exponent／operator／signal、Qiskit circuit／Gaussian build／compile／shotを起動していない。Snapshot bytes照合はread-only、旧result数値との比較は未実行。共有libraryの追加glueはASTから関数定義だけを抽出してstubで検査し、科学packageはloadしていない。

## 10. 固定SOURCEのread-only preflight

SOURCE commit後にA0／B0それぞれ独立detached clean checkoutでPASSした。HEAD／SOURCE・dirtyなし、保存入力5ファイルのblob、固定case／metadata scope、依存version／module origin、B0 profile、予定command、fresh output非存在を確認した。Scientific loaderを禁止するimport barrier下で、**両方とも科学module load0、実科学output作成0**。

既存環境 `/tmp/pr-phase2df1pre-private-20261023/venv/bin/python`、CPython3.11.9を使用。A0：mpmath1.3.0／NumPy1.26.4。B0：NumPy1.26.4／SciPy1.14.1／Qiskit1.3.0／Qiskit Nature0.7.2／OpenFermion1.6.1／OpenFermion-PySCF0.5／PySCF2.7.0。環境のinstall・設定変更はない。

32 CPU・空きRAM約52.7 GiBを確認した。A0は精度別412 exp共有とD／outer graph共有、B0は372 block・492 RTE sequence・206 exponential・16 momentの共有cacheを用意した。Full-QCを保持する96要求の限定pilotで、古典compile時間・memoryを記録する。今回は速度倍率を測っておらず、MC用の大規模並列基盤は追加しない。将来の実行直前に空き資源・非干渉を再確認し、人工的な低いCPU／RAM／worker上限は設けない。

## 11. 許可後だけ使う個別command

**今回はどちらも起動していない。** 次は研究BでA0の実行を先に判断する。A0の許可はB0の許可を含まない。

A0：固定8行×90/128桁、保存値との比較・bound／operator／signal診断を明示許可された場合だけCodexが一度実行する。

```bash
/tmp/pr-phase2df1pre-private-20261023/venv/bin/python -I -B \
  /tmp/prt-research-b-outer-a0-science-20261025/scripts/run_b2d_outer_a0.py \
  --execute \
  --source-commit 1e3921ba16e5d531e9a3010cffa9ff717609a93a \
  --output /tmp/prt-research-b-outer-a0-science-20261025/artifacts/research_b_outer_a0/20261025/independent-once
```

B0：固定48代表trajectoryのfull-Fock／controlled／wrapper検査・96 compile要求と費用測定を**別途**明示許可された場合だけCodexが一度実行する。

```bash
/tmp/pr-phase2df1pre-private-20261023/venv/bin/python -I -B \
  /tmp/prt-research-b-outer-b0-science-20261025/scripts/run_b2d_outer_b0.py \
  --execute \
  --source-commit 8d0cb96df9d1f6229f207550634c7e11aff2851c \
  --output /tmp/prt-research-b-outer-b0-science-20261025/artifacts/research_b_outer_b0/20261025/native-pilot-once
```

## 12. 未確認事項・次の担当・STOP

A0の高精度収束・残差・bound整合・識別力、保存matrixの厳密な理論前提、B0のDF basis／full-Fock／component sign／相対phase／compiled operator正当性・費用は未検証。人工PASS・metadata整合・旧科学PASSから今回の科学的成功を補完しない。重要な問題が見つかっても未承認の追加計算で解決しない。

A1包含、B1期待費用MC、shot数の数値評価、ε=1e-8／1e-9の保証付きG比較、新物理系・新grid・全RPE費用は今回SOURCE準備と将来のA0／B0許可に含まない。u_numとγを0で埋めず、費用最小や科学的GOを自動認定しない。

次の担当は研究Bのユーザーで、まず固定A0の科学実行可否を判断する。B0は別途判断する。Codexは本報告・SOURCEを公開してremote内容を照合した後にmandatory STOPし、科学実行を開始しない。
