# 研究C・C4：Q13限定adapter実装・科学実行準備（2026-10-24 JST）

状態：**IMPLEMENTATION_PREPARED_SCIENCE_NOT_RUN**。
typedな中央rotation adapterと非科学的テストを追加し、次回の4caseを固定した。
実際のQuantumCircuit生成、Rustiq合成、transpile、operator／signal評価、資源比較は今回すべて0件。
operator同値性・位相保持・資源削減の科学的PASSを意味しない。

起点は `1dd39865e6b5598d3dfac5f65e434d7db6777788`。
独立branchは `codex/research-c-rustiq-adapter-preparation-20261024`。
元workspace・既存SOURCE・raw結果・grant／ledger／marker／STOP・正式共通指示は変更しない。
共通指示SHA256は `7ce342dbf7086424f1afd837d8c274687bc28e4838601f5ad96f5b626d95667f`。
§8.4・§10の既存コード再利用、人工的な資源上限を置かない方針、共有環境保護を適用した。

## 成果物の入口

| 役割 | ファイル |
|---|---|
| opt-in adapter・typed plan・既存builderへの再接続 | [rustiq_adapter.py](../../src/trottertracks/pr_dictionary_validation/rustiq_adapter.py) |
| stdlibのみのunit／mockテスト | [test_research_c_rustiq_adapter.py](../../tests/management/test_research_c_rustiq_adapter.py) |
| 次回の実行契約案・37 SOURCE hash・7 input hash・case／領域／call数 | [research_c_rustiq_limited_contract_20261024.json](research_c_rustiq_limited_contract_20261024.json) |
| 将来の結果保存先 | `artifacts/research_c_rustiq_limited/<execution-date>/<fresh-run-id>/`（未作成） |

今回は準備だけで、新しい科学runnerや汎用承認基盤を増やさない。
以下に既存関数への具体的な呼出し対応を示す。既存F1全件runnerを起動して本scopeを代用しない。
他研究資料を保全する明示指示に従い、既存の索引・研究概要・validation manifestも編集していない。
実行後の証拠登録と索引への追加は、新結果を公開する段階の作業になる。

## 固定資料と再利用の判断

- [接口監査（固定起点）](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1dd39865e6b5598d3dfac5f65e434d7db6777788/docs/research/research_c_rustiq_interface_audit_20261024.md)
- [研究C初期調査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c0924b1fdcc3a28d9a65062b4feaf9d0db944328/docs/research/research_c_hamiltonian_circuit_survey_20261024.md)
- [Gaussian singleton独立研究の一時終了](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/69f8e1cac23a4924ab81aebe399657a705024f6a/docs/research/pr_gaussian_singleton_closeout_handoff_20261024.md)

| 既存資産 | 今回の利用と境界 |
|---|---|
| `circuits.py` | `BasisProvider`、`append_central`、`append_deterministic`、`MixedEvolutionResult`をそのまま呼ぶ。B0は既存`build_evolution(optimized=True)`へ直接委譲。actual adjointとfull X/Y wrapperも既存関数をそのまま再利用する |
| `gates.py` | 既存の絶対誤差検査、90/128比較、X/Y検査、full QPY cache key、位相を保持する公式definition展開と固定compilerを将来呼ぶ。今回は呼んでいない |
| `reference.py`、`SavedF1.reference_case` | 独立CAR／exterior reference、独立law、90/128桁を将来再利用。候補回路からreferenceを作らない |
| `f1_saved_execution.py`、`cases.py` | 保存binary64 lawのdecodeと元のmotif定義を再利用する準備。今回は保存JSONの静的metadataを読んだだけで、`law`／`resolve_case`／`motif_events`は呼んでいない |
| `f2_execution.circuit_metrics` | 既存のRZ/CX count/depth、total／quantum-only depth、full size、measureの定義を将来呼ぶ。F2の実行承認・mean・統計処理は使わない |
| `df_rte_qiskit.py`、`df_partial_s2.py`、`df_partial_s2_repeated.py`、`df_rte_tail.py` | typed順序、符号、basis ID/hash、scalar／identity、D境界融合の規約を照合。Hamiltonian／DF／lawを再生成しない |
| 2978版basis-plan、WP06-b／f29版run、ae743版Pauli primitive | 監査で確認したfull/support・run共有・boundary機会を確認。今回の混在辞書baselineが既に持つcommon-run／singleton support補完／D融合を全routeで揃える。別basis選択器やrectangular研究を移植しない |

旧builderには中央emitter callbackがない。このため新版ではtyped run走査と
短いevolution組立だけを追加し、basis分解・support補完・中央literal emitter・D emitter・
wrapper・compiler・referenceを複製しない。旧ファイルの編集やproduction monkeypatchはない。

## Adapterの実装内容

`Occurrence`に通し番号、event/application位置、atom ID／role、元word／support、
符号付き角、coefficient sign、basis ID/hash、frame、run IDを保持する。
`SequencePlan`はsystem `(0,...,m-1)`、ancilla `m`、phase記録、route／API／core／options／
compiler／SOURCE／input来歴とfingerprintを持つ。scalarはevolution traceに記録する。
FRAME hashは保存basis identity、PHYSICALのhashは局所basis recipeのhashであり、
新たに数値行列のhashを計算したという意味ではない。

先に旧規則と同じ最大同basis runを確定し、**productを含む元run長**からsingletonを決める。
その後で連続rotation領域を切る。product fenceで区切った後の領域長をsingleton判定へ
流用しない。identity・異frame・異local-basis・macro境界を越えない。
重複occurrenceも元順序も残し、非可換演算子の移動は行わない。

FRAME_Zと、既存local basisが明確なPHYSICAL_PAULIに対応した。
どちらも既存basis inverse → 中央Z列 → basis forwardを保ち、中央部分だけを置換する。
PHYSICALのX/Yを直接広いnetworkへ展開する経路ではない。
同basis中央Z列は可換だが、full演算子の異basis間順序はfenceで保存する。

PRの`exp(-i a P)`に対してQ13へ渡す角は`a,-a`。
`controlled_terms`は元wordの`word[p]`をqubit `p`に対応させ、nonidentity文字と
index列を同順に出す。中央regionでは既存supportをZへ写す。
たとえば中央support `(1,3)` は `('ZZ',[1,3],a)`、
`('ZZZ',[1,3,4],-a)`。ancilla込みの5qubit出力を直接composeし、二重controlしない。
既存の`event.unsigned_rotation_angle`は名前に反して符号付きなので、再度signを掛けない。
0やClifford近傍を含めて角度を丸めず、occurrenceを集合化しない。

Q13 optionsは`optimize_count=True,preserve_order=True,upto_clifford=False,upto_phase=False,
resynth_clifford_method=0`へ固定。実行時にQiskit 1.3.0を要求する。
Q13の符号付き共役・角度挿入・末尾literal actual inverseをそのまま使う。
独自の探索器・DAG・Clifford合成器は実装していない。
上流固定版は[Qiskit SOURCE](https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/qiskit/synthesis/evolution/pauli_network.py)、
bundled coreは0.0.10で、外部R425／C011は接続していない。

productは元`append_central(angle=None)`、identity rotationは元と同じancilla phaseへ残す。
Taylor phaseと負productのπを旧順序で加算した後、identity rotation分を引く。
scalarも従来式のancilla phaseへ残す。Q13へall-Iを渡さない。
Dの最初／最後halfと内部boundaryの融合、Gaussianのactual inverse、
`actual_adjoint(result)`、測定付きfull Hadamard X/Yの規則を保つ設計である。
これらの**数値的な同値性は未検証**。

既存entry pointの既定動作は変わらず、新adapterも`route='B0'`が既定。
B1／R1は明示opt-in。`compile_identity`は構築routeとの一致を検査し、
route／options／SOURCE／input／evolution fingerprintを既存QPY cache keyへ渡す。
B1／R1のfingerprintにはordered events、atom records、run/basis/phase planも含む。

## 今回の技術確認

既存のF1専用CPython 3.11.9 venvをread-only利用。新規package導入・共有venv変更なし。
Qiskit 1.3.0をimportし、対象関数の公開signatureが監査と一致することを確認した。
返値annotationの確認だけでQuantumCircuitを生成していない。
installed `pauli_network.py`のSHA256は
`02a90e0f62a83bbb52da225ea812a7dcde5da66be0c295421a3ca3b015086ea2`。

Qiskitの非extra直接依存9項目は、`importlib.metadata`のversionと宣言rangeを照合した。
rustworkx0.18.1、NumPy1.26.4、SciPy1.14.1、SymPy1.14.0、dill0.4.1、
python-dateutil3.0.0、stevedore5.9.1、typing-extensions4.16.0、symengine0.13.0が範囲内。
venvは既存の`--without-pip`構成のため`pip check`は利用できず、packageは追加していない。
これは全transitive依存の新しい動作試験ではない。旧50wheel lockを保全し、次回も照合する。

```bash
PYTHONDONTWRITEBYTECODE=1 /tmp/pr-phase2df1pre-private-20261023/venv/bin/python \
  -m unittest discover -s tests/management -p test_research_c_rustiq_adapter.py -v
```

**21 tests PASS**（CPython3.11.9、最終実行0.020秒、ローカル技術確認）。
input validation、疎word/index、signed pair、重複保持、product/identity fallback、
frame/run境界、元runからのsingleton、phase記録、stale plan、route/cache context、
既定B0委譲、mock再接続、D融合／scalar／result metadata、契約とSOURCE hashを検査した。
各testの終了時に`numpy,qiskit,mpmath,trotterlib`が未importであることも確認する。
公開Q13へのdispatchは人工moduleとMockで検査し、実関数は呼んでいない。
最初の破損入力testではMixedAtom自身のconstructorが先に拒否したため、
外部レコードを模したSimpleNamespaceへ修正した。科学条件の変更はない。

既存のcompile位相回帰testや通常DF testsは実回路・行列処理を含むため実行していない。
本test PASSは、Operator同値・compiled phase・Q13の実合成・費用低減のPASSではない。
API確認のimportと、科学SOURCEの実動作確認を区別する。

資源のread-only観測ではCPU affinity相当の`nproc=32`、物理RAM約67.1 GB、
MemAvailable約56.3 GBだった。GPU照会はdriverへ接続できず利用可否未確認。
科学処理は未投入なので高速化率・実使用量は主張しない。
次回はその時点の空きCPU／RAM／pressureを再確認し、CPU実装のQ13／compilerについて
独立probe／wrapperを並列化し、BLAS/OpenMP/Numba/Qiskit/Rayonとprocess数を調整する。
CPU／RAM／worker／GPU／wall-timeの人工上限は契約に置かない。

## 次回の最小caseと対象領域

固定モデルは保存F1の4mode・2粒子・occupied `(0,1)`、density-square合成Hamiltonian。
分子geometry／AO basis／分子DF rankは非該当。
split `L_D=1`、残差2fragment、`T=1/5,q=2,δ=1/10,r=2,h=1/20,K=2`の一点だけ。
scalar `1/37`、Dとrational frameは契約のspecificationにそのまま保存した。
Hamiltonian、DF、law、finite targetは変更しない。新たなmean／target評価も行わない。
F2のq1/K0・認証N・全round総資源の条件とは混ぜない。

保存bundle SHA256は`9d87488f673551ace8e13456bc64cb73a5384ba56c42cac28f345fb270c6e4e0`。
private原本の7fileのhashを旧F1契約と照合した。再生成・sampling・イベント構築は行っていない。
契約JSONには正確なatom ID、product→child列、2×2 macro列、10領域のIDとoccurrence番号を記載。

| case ID | 検査する内容 | 適格領域 |
|---|---|---|
| `SS_motif_2` | Taylor order2、負product2個、異frame product fence、後続同frame rotation共有 | macro0 `run2:region0`、macro1 `run0:region0` |
| `SS_motif_3` | identity product／rotation、負product、Taylor phase、異frame、fallback | macro0 `run1:region0`、macro1 `run0:region0` |
| `SS_motif_4` | 同frameの重複rotation、負角、両macroの共有とD融合 | 各macroの`run0:region0` |
| `PP_motif_0` | 物理Pauli `Z3` と`X2 X3`の交互順序、負角、異local-basis fence | 各macroの`run0:region0`と`run1:region0` |

SSの`u=s=S:F0:IIZI`、`v=S:F1:IIIZ`はともに負係数。
この保存入力ではM1とM4が同一列になるためM1を省く。
M5の重複productはmockで保持を確認し、M6の同frame／異frame境界は選定4caseに含まれる。
M7のall-identity fallbackはmockで確認し、科学検査のidentity/scalarはM3と全caseで担う。
全motif／全armは実行しない。

PPでは`u=P:collect:IIIZ`と`v=P:collect:IIXX`が1箇所で非可換なので、
静的Pauli wordから反可換を確認できる。SSの異frame同士のcommutatorは**未確定**で、
数値評価していない。Rustiqに渡す各中央regionのZ列自体は可換であり、
非可換なfull operator間順序は別runのfenceとして検証する。

## 比較・gate・call契約案

B0は既存の強いcommon-run Qiskit baseline、B1は同じadapter領域分割で旧中央emitter、
R1はその中央だけQ13。全routeで既存basis共有・singleton support/full・D融合・
scalar/identity/Taylor/product signと測定付きfull X/Y scopeを揃える。
native basis `rz,sx,x,cx`、opt1、seed17、approximation_degree1、qubits_initially_zero=false。
現行compilerの公式definition展開を含めて同一とし、profileのvalidateを緩和しない。

gateは順に、局所絶対operator → full `diag(I,U)`とunitarity → actual adjoint →
raw X/Y複素信号 → compiled full-wrapper operator／X/Y → PASS caseの資源比較。
max-entryの絶対許容差はlocal/full/adjoint/compiled/unitarityすべて`1e-10`、
複素X/Yは`2e-10`、独立90/128 reference間は`1e-60`、rtol0。
phase fitting、global phaseだけを無視する比較、事後の許容差緩和を認めない。
max-entryに加えてFrobenius／spectralも記録する。

| fully passing runの論理request | 数 |
|---|---:|
| 保存motif復元／保存law arm | 4／2 |
| 局所literal build（B0/B1共通emitterを共有）／局所R1 build | 10／10 |
| full evolution（4case×3route）／full measured wrapper | 12／24 |
| actual circuit adjoint | 12 |
| Q13 synthesis（局所10＋full中10）／入力rotation triples合計 | 20／60 |
| compiler cold／warm request、実transpile | 24／24、24 |
| reference bundle（local10＋full4、それぞれ90/128） | 28 |
| Operator materialization（local20＋full12＋adjoint12＋compiled24） | 68 |
| Statevector signal evaluation（raw24＋compiled24） | 48 |
| X/Y pair gate（raw12＋compiled12） | 24 |
| precision matrix／signal agreement | 34／4 |
| compiled資源metric | 24 |
| retry／sampling／law生成／mean／認証shot | 0 |

数は登録request単位。内部QC constructor、gate append、reference内部の行列代数は
独立caseとして追加しない。これを理由に追加評価を許可しない。
paired full referenceはH,D,R,U,wrapperX,wrapperYを各caseで比較するため24 matrix比較、
局所10を足して34。以前のF1全件評価やstandalone scientific testは含めない。

RZ/CX count、RZ/CX depth、total depth、quantum-only depth、full-wrapper size、measureを保存。
adapter変換、Q13合成、builder、transpile cold、compile-cache warmを区別し、wall/CPU/RSSを記録。
adapter traceに変換とregion別合成のwall/CPU計時を用意したが、まだ実測していない。
基底cacheはcase/routeごとにfresh providerを作り、内部の合法共有は全routeへ与える。
compiler cacheはroute別でcold24 miss／同一IR再要求warm24 hitを予定し、warm追加transpile0。
Q13 synthesis cacheは実装していない。局所検査後のfull合成はprocess-warm/cache-disabledと記録し、
新しいwarm synthesis実験は追加しない。T count、認証shot、全round資源へ外挿しない。

## 承認後の関数接続とpreflight

1. ユーザーの新しい明示承認を、この公開commit SHA・契約JSON SHA256・
   `C4_Q13_LIMITED_V1`・fresh output/run IDへ結び付ける。旧grantを再利用しない。
2. 独立clean worktree、37 SOURCE hashと7 input hash、固定lock／version／API、
   その時点の共有資源をread-only検査する。新しいpackage・共有設定変更なし。
   SOURCE identityは公開commitと`digest(source_files)`、input identityは`digest(input_files)`。
3. 記録を開始した後にだけ`SavedF1(verified)`を作る。`verified`へ検証済みinput byte列と
   `contract.fixture_bundle_path='handoff/f1_bundle.json'`を与える。
   選定caseを`resolve_case`、`macros`で復元し、契約のliteral列と領域ID／occurrenceをassertする。
   `SavedF1.law`は保存法則のrestoreで、新規law生成へfallbackしない。
4. 各full referenceは`reference_case(reference_kind='motif',digits=90/128)`を呼び、
   90結果をengine.cacheへ保存して128の`paired_reference`に指定する。
   局所referenceは同じ独立lawのtauとevent orderから高精度角を得て、中央Z積を
   左から時系列に掛ける。`controlled_operator`でcontrol0=Iも明示する。
5. 局所literalは新しい5qubit circuitへ元`append_central`をoccurrence順で追加。
   局所R1は`region_network`を`_q13`へ渡すだけとし、basis／product／identityは入れない。
   これが最初の実回路によるoptimized-path検査となる。
6. 全local gateの後、各routeの`build_evolution`、既存`actual_adjoint`、
   `hadamard_wrapper`を上表の数だけ呼ぶ。B0は旧builderへ委譲される。
   compilerは`gates.compiled_wrapper(wrapper,CompilerProfile(),cache=route_cache,
   identity_context=compile_identity(result,route=route))`。
   wrapperのIRがaxis/prep/readout/phaseまで含むためX/Yも別keyになる。
7. 各gate順序を保って独立probe／compileを並列化する。利用可能資源と実pressureに合わせて
   workerと各library threadを調整し、先行failure後は新しい依存作業を投入しない。
   起動済み独立taskの完了も登録上限内で記録する。
8. 絶対operator／signal PASSしたcaseだけ`circuit_metrics`へ進む。
   QPY／NPY／高精度reference、ordered typed入力とangle.float.hex、phase、route/options、
   source/input/profile、cache、timings、call台帳をfresh outputに保存する。

非有限、reference不一致、case/region/source/input/profile drift、位相／operator／signal FAIL、
cache/count不整合、実際の資源不足・共有環境干渉でSTOP。retry/resume0。
partial結果を残し、case置換・許容差変更・追加route・未登録testで救済しない。
JSONには4096行のjournal上限と16×16／32×32の出力scopeを置き、CPU/RAM/timeの上限とは分けた。

## Mandatory STOPと担当

公開内容とremote SHAの照合後、今回の作業を終了する。
新規grantは発行・消費していない。C1／外部Rustiq／QuCLEAR／Gaussian singletonへ進まない。

次の提案は、ユーザーが研究C・統括側で上記4case／比較／call契約を確認し、
**新規科学実行を別途明示承認すること**。
承認後のCodexがその固定契約へapprovalとSOURCEをbindし、preflightと登録実行、
結果・索引・証拠台帳の更新を担当する。今回はその科学実行の承認を得たとは扱わない。
