# 研究C・C4：RustiqのPR controlled回路への接口監査

2026-10-24 JST。**推奨状態：`READY_FOR_LIMITED_IMPLEMENTATION_PROPOSAL`。**

順序・符号・末尾Clifford・位相を保持する技術的経路を、固定SOURCEから具体化できた。推奨する最初の経路は、**Qiskit 1.3.0に既にある角度付き `synth_pauli_network_rustiq` と、既存PRのtyped情報・位相保持wrapper検査を接続する小さいadapterの提案**である。指定Rustiq Python固定版の骨格出力を直接使う経路も説明できるが、角度の挿入処理まで新たに接続する負担があるため、最初の実装候補から分ける。

これは接口監査による次段階の推奨である。**演算子同値性PASS、資源改善、科学的GO、実装承認、研究方針の正式採択ではない。** C1は中期中心候補のまま今回の対象外、Gaussian singleton研究の一時終了も維持する。

## 0. 起点・取得範囲・保全

- Repository：`HIROMU1015/Partially-Randomized-Trotter`。起点：`c0924b1fdcc3a28d9a65062b4feaf9d0db944328`。
- 専用branch：`codex/research-c-rustiq-interface-audit-20261024`。独立worktree：`/tmp/prt-research-c-rustiq-audit-20261024`。他branchをmergeしない。
- 必須の[研究C初期調査][SURVEY]は指定commitからGitHub connectorで実取得し、特にB4/C/D2/D3/E2/Fを確認した。[Gaussian singleton終了記録][CLOSE]、起点の `PROJECT_MAP.md`、概要・関連status/manifest・SOURCEも読取参照した。
- 正式共通指示は既読の `research_project_instructions_unified_2026-10-23.md` を適用し、元workspaceのSHA256が `7ce342dbf7086424f1afd837d8c274687bc28e4838601f5ad96f5b626d95667f` と同じことを確認した。起点へ複写・commitしない。
- 外部実装はGitHub connectorから静的に実取得。Rustiq Pythonは指定commit、coreは依存宣言と対応release tagのcommit、Qiskitはtag `1.3.0` をcommitへ解決した固定版を区別する。現行IBM APIは説明の照合に使い、旧環境へ機能を推測移送しない。
- 変更は本Markdownだけ。install、SOURCE import/変更、Rustiq実行、Hamiltonian生成/分解、回路build/compile、QPY decode、operator/signal評価、sampling/simulation、保存数値の再集計・性能測定、新grant発行は行わない。

前回B4の「Qiskit 1.3.0に同APIがあるとは未確認」は、**今回の固定SOURCE取得で解消した**。前回文書を遡及変更せず、この新証拠と結論の適用版を本書に残す。実際の研究venvでimport可能か・wheelが期待版と一致するかは実行していない。

## A. SOURCE・API対応表

### A1. 三つの実装を分ける

| 識別 | 固定SOURCE・今回確認した箇所 | 入出力・役割 | PRへの接続判断 |
|---|---|---|---|
| **R425：指定Python wrapper** | `qiskit-community/rustiq@425000222e21a7a28d70a08c75be968f16fd6903` の[Python公開接口][RPY]、[Rust binding][RIF]、[utils][RUTIL]、Cargo/pyproject、example | `pauli_network_synthesis(paulis, metric, preserve_order=True, nshuffles=0, skip_sort=False, fix_clifford=False, check=False)` → `list[(gate_name, qubit_indices)]`。入力に角度はない。出力はClifford骨格で、RZの角度やrotation occurrence位置は返らない | 骨格だけでは元の回転列を実装しない。入力の角度・符号・occurrenceを保持し、符号付き共役追跡・非可換依存関係に従ってrotationを挿入し、末尾を閉じる処理が別に必要 |
| **C011：R425のcore側** | Cargo宣言 `rustiq-core="0.0.11"`。release tagを解決した `dbe33fa8f03cf46134402a75432a5b289494b813` の[synthesis][C11SYN]、[PauliDag][C11DAG]、[PauliSet][C11SET]、[CliffordCircuit][C11CLI]、rotation extraction・PauliLike | `greedy_pauli_network` は順序保持DAGか順序自由法を選び、Clifford gate列を返す。`fix_clifford` はinverse tableauのisometry再合成を追加。`check_circuit` は各入力axisがどこかでsupport≤1へ到達することを検査 | 符号・角度を持つ完全unitaryやcontrolled枝の検査ではない。release SOURCEの確認であり、未導入crateの実build identityを検証したわけではない |
| **Q13：Qiskit同梱Rustiq** | Qiskit tag `1.3.0` → `896d1c8b72888a52a46081a6d1f4de8c56db8aea`。[公開Python API][QPY]、[Rust実装][QRS]、[HLS plugin][QHLS]、evolution export、Cargo.lock、[既存test][QTEST] | `synth_pauli_network_rustiq(num_qubits, pauli_network, optimize_count=True, preserve_order=True, upto_clifford=False, upto_phase=False, resynth_clifford_method=0)` → `QuantumCircuit`。`pauli_network` は角度付き疎triple列 | **最小候補**。符号付きrotation挿入・commutativity DAG・末尾actual inverseが実装済み。PR側に新しいnetwork compilerを自作せず使える。下記のnonidentity入力・位相条件へ限定 |
| **C010：Q13のcore依存** | Q13 Cargo/lockはcore `0.0.10`。対応tag commit `cbd8db823c0ab55f9e1522b0487e8494acefc837` の[synthesis][C10SYN]も確認 | Q13のRust adapterはcore骨格へ角度を挿入してQiskit circuitを作る | R425/C011と同一版ではない。Q13成功を指定R425の成功と記録しない。Q13公開APIへ外部R425骨格を渡す引数もない |
| **現行IBM API** | [PauliEvolutionSynthesisRustiq公式説明][IBMAPI] | HLS `PauliEvolution.rustiq`、順序・phase・末尾Cliffordのoptions | optionの意味は照合できる。今回の実装存在の根拠は現行説明だけでなくQ13固定SOURCE |

R425の `fix_clifford` とQ13の `upto_clifford` は名前・極性・処理が異なる。前者をtrueにすることと、後者をfalseにすることを「同じphase保持設定」と扱わない。

### A2. 入出力規約と安全な範囲

| 論点 | 固定SOURCEから確認できた内容 | 必要なadapter条件 |
|---|---|---|
| R425のPauli入力 | 同じ幅の文字列を `PauliSet.from_slice` が文字位置順に読む。各入力の初期signはfalse。`insert` はX/Y/Z以外を特別なPauliとして解釈しない | `I/X/Y/Z` のみ、同一幅・登録範囲内を検査。`-XYZ`、`iXYZ` のprefixを渡さず、符号を角度／既存phaseへ分離。重複axisも異なるoccurrenceとして保持し、set化しない |
| R425の出力gate | `CNOT`、`CZ`、`H`、`S`、`Sd`、`SqrtX`、`SqrtXd` とqubit index列。coreの `to_vec` を確認 | CNOTのcontrol/target、S↔Sdg、SqrtX↔SX/SXdgを正確に写す。実gate列のinverseを取る場合、同じgate conventionで反転してphaseも打ち消す |
| Q13の入力 | 例 `("XY", [1,3], θ)` はXをqubit1、Yをqubit3へ配置。`num_qubits`を別に指定できる | support・文字の長さ一致、indexの重複/範囲、実数有限角、idle qubitを明示。Q13内部の `expand_pauli` はzipで配置するので、adapterが不一致を拒否する |
| 角度 | nonidentity tripleの第3要素は `R_P(θ)=exp(−iθP/2)` のrotation角。Q13の `qiskit_rotation_gate` は共役後のminus signなら角度を反転する | PRの `exp(−iaP)` を渡すならθ=2a。`event.unsigned_rotation_angle` は名前に反してcoefficient signを含むため二重に掛けない |
| qubit文字順 | Rustiqのfull stringは左からqubit0,1,…。PRの `MixedAtom.word[p]` も物理mode p。Qiskitの通常のdense Pauli labelは右端がqubit0 | 直接stringをQiskit `Pauli` labelへ渡す場合は向きを明示変換。Q13 sparse tripleなら文字とindexの対応を保持し、安易なreverse・全register bit順変更をしない。ancilla-lastを保つ |
| 順序保持 | C011のDAGは非可換pairの元順序へedgeを置く。Q13のrotation挿入も非可換pairの先行条件を持つ | `preserve_order=True` を明示。可換項の合法な交換まで禁止する設定ではない。元event/occurrence順はprovenanceとして保存し、非可換の相対順序を変えない |
| HLSの入口 | Q13 HLSは `PauliEvolutionGate` の `ProductFormula.expand` を呼ぶ。optionによって `algo.preserve_order` も設定する | 固定PR列をH=sumへcollectしてHLSへ投げる経路は最初に使わない。再Trotter化・collect・reps変更を避け、既存の明示ordered triple列を低水準APIへ渡す |

**all-I入力は初期adapterから除外する。** Q13 Rust実装はsupport0を専用のglobal-phase分岐で処理する。nonidentity回転のhalf-angle規約と同じと思ってidentityを投入せず、既存PRのidentity/scalar phase処理をそのまま使う。これはSOURCEから安全な入力域を絞る判断であり、今回all-Iの数値的成否・bug認定をしたものではない。

### A3. option、末尾Clifford、global phase

| 経路／option | SOURCE上の意味 | 本監査の推奨 |
|---|---|---|
| Q13 `preserve_order=True` | 非可換順序を保持、可換性に基づく自由度は利用 | 必須。falseの回路は一般に元列と同じoperatorでなくなる |
| Q13 `upto_clifford=False` | 最後の補償Cliffordを付ける | 必須。trueでCliffordを無料省略しない |
| Q13 `upto_phase=False` | Rust adapterが `resynth_clifford_method` を0へ固定し、骨格の `circuit.dagger()` の実gate列を末尾に付加する | 必須。比較条件の明確化のためmethod0も明示。SOURCE上、全phaseを戻す経路がある |
| Q13 method1/2 | 末尾Cliffordをtableau等から再合成。docstringはglobal phase非保存を説明 | 初期比較では使わない。小さいCX数のために `upto_phase=True` へ変えない |
| R425 `fix_clifford=True` | C011では骨格inverseのtableauをisometry合成し付加。global-phase返却値はない | **完全phase保持の根拠にしない。** 共役作用を復元してもcentral scalar phaseが残り得る |
| R425 `fix_clifford=False` | 未閉鎖の骨格を返す | 角度挿入後、同じ骨格のactual inverseを付ける既知の閉鎖経路を検討できる。Q13の既存処理が先例。ただしR425側へそのままcallできる完成接口ではない |
| R425 `check=True` | すべてのaxisがsupport0/1としてhitすることを確認 | full matrix、rotation角・順序・末尾phaseやcontrol0=Iを検査するものではない |
| R425 `nshuffles`／`skip_sort` | 前者はqubit permutation候補の探索、後者はheuristic内sort省略。Python docはskip_sortに収束上の注意を記載 | 初回は既存defaultの0／falseを固定する提案。科学seedとcompiler探索の乱数を混同しない。これは計算worker資源の制限ではない |

R425骨格から完全unitaryへ戻す場合は、**入力occurrenceごとのsigned angle、符号付きPauli共役、非可換の先行関係、挿入位置、actual inverse**がすべて必要である。骨格には角度がないので入力情報を捨ててから復元することはできない。一方、元角度と実gate列を保持する経路なら、未知のglobal phaseを数値fitで推定する必要を避けられる。静的な構成可能性と、完成adapterの正しさは分ける。

### A4. `extract_rotations`と依存関係

`extract_rotations`を最初の接続入口にはしない。

- R425 Python wrapperはgate listと角度を分け、Rust側はrotation axis/signと**末尾Cliffordの符号付きtableau**を返す。tableauのsignはPauli共役のsignで、元unitaryのglobal scalarではない。入力回路の `global_phase` を渡す引数もない。
- `utils.convert_circuit` は数値RZがπ/2の倍数に近いかを `np.isclose` で判定し、S列へ置換する分岐を持つ。その分岐にRZとSの間のscalarを返す処理は見当たらず、完全位相同値の入口として採用できない。また `np.divmod` の数値 `mult` を `for _ in mult` で反復する記述があり、通常の数値入力に対する実行上の問題が疑われる。**今回は実行しておらず、例外の再現済みとはしない。**
- `optimize=True` はrotation最適化と末尾tableauの変更も行う。falseへ変えるだけでは、前段のnumeric変換・元global phaseの不足は解消しない。空入力・idle register幅を `max(gate qubits)` だけで推定する制約もある。
- SOURCEのexampleは `Metric.COUNT/DEPTH` を使うが、確認したRust bindingは文字列を受けるconstructor `Metric("count")`／`Metric("depth")` を公開し、Python wrapperにはそのclass属性定義が見当たらない。将来はexampleの動作を前提にせず、固定版の実公開接口へbindする。

依存関係は次のとおり。

| 対象 | 静的に確認した内容 | 実行前に残ること |
|---|---|---|
| R425 | pyprojectはPython≥3.9、numpy/networkx、setuptools-rust build。Python package version1.0.0、Cargo package0.0.8、PyO3 0.24.1、core0.0.11。固定treeのCargo.lock取得は404 | 表示version文字列だけでidentityを決めず、commit・resolved crate・wheel/環境を記録。transitive依存やABIの実動作は未確認。既存3.11.9環境での動作保証ではない |
| Q13 | 公開Python export・HLS plugin登録とも存在。Cargo.lockのcore0.0.10、Qiskit pyprojectはPython≥3.9。Rust build要件は別に記載 | Q13低水準APIを使うだけならSOURCE上Qiskit更新・外部Rustiq installは必須でない。研究環境の期待wheel/API実在確認は別承認後 |
| 既存PR環境 | 保存 `CompilerProfile` はQiskit1.3.0、opt1、seed17、basis `rz,sx,x,cx`、approximation_degree1、qubits_initially_zero=falseを厳密要求 | 共有venvを変更しない。将来version変更するなら新profile/新SOURCEを準備し、固定profileのvalidateを緩和しない |

## B. PRの意味論と技術的制約

### B1. 再利用SOURCEと差分の位置

本書の起点SOURCEは `c0924b1f…`。下表の別固定版を同じSOURCEとして混ぜない。

| PR SOURCE・関数 | 既存の意味論／最適化 | Rustiq接続で保つこと |
|---|---|---|
| [circuits.py][PRC]：`BasisProvider`、`build_event_sequence`、`append_central` | typed FRAME_Z／PHYSICAL_PAULI、同basis連続run共有、singleton support補完、basis inverse→中央control→basis forward | **採用済みbasis planを先に固定**し、Gaussian gate列の同定不能なflatten/re-extractをしない。run内の明示rotation領域だけが最初の対象 |
| 同：`build_evolution`、`append_deterministic` | D halfと次stepの同一D halfを融合し、scalarをancilla位相へ反映 | 既存D境界融合を候補・対照で共通化。Rustiqのためにraw D halfへ戻さない。D側の新合成は初期scope外 |
| 同：`actual_adjoint`、`hadamard_wrapper` | actual circuitの `.inverse()`、prep・H・controlled evolution・X/Y readout・ancilla測定 | 完成candidateからactual inverseを作る。referenceのoperatorでcandidate adjointを代用しない。terminal measureとprep/readoutは同じ位置 |
| [gates.py][PRG]：`matrix_gate`、`controlled_gate`、`wrapper_signal`、`signal_gate` | phase fitなしの行列照合、full controlled block、X/Yを複素信号へ接続 | control0・control1、full wrapper、actual adjointを独立referenceと照合。単一入力状態のsignal一致だけでoperator PASSとしない |
| 同：`compiled_wrapper`、`compile_cache_key` | 公式definitionを独立EquivalenceLibraryで展開して固定Qiskitへ渡し、角度/global_phase/registerを含むIR hashを作る | 旧位相修正を残してRustiq合成後のfull wrapperにも適用。新pipeline・options・core版をcache contextへ追加し、旧価格のcache hit扱いをしない |
| [df_rte_qiskit.py][PRDF]：`_validate_event_application_order`、`_append_central_application` | product→child、sign、signed angle、Taylor phaseを保持。中央Z/ZZだけcontrol、basis非制御、同basis ID/hashだけ共有 | 真のapplication順を引き継ぐ。各componentをsortしない。basis順・ancilla位置も維持 |
| [df_partial_s2.py][PRS2]、[df_partial_s2_repeated.py][PRREP] | deterministic diagonal primitive、scalar／extracted identity、D_0境界half融合とstep位相集約 | full physical split・time・r/q/K・normalizationを変えない。一次basisの差とcompiler差を交絡させない |
| [df_rte_tail.py][PRTAIL]：`DFBasisRegistry`、`exact_df_diagonal_coefficients`、identity policy | basisのID/hash、I/Z/ZZ support、coefficient signと抽出identityを分離 | 既存抽出結果を入力に使い、Hamiltonian・辞書・lawを再生成しない |
| [2978版builder][B2978]・[repeated版][R2978]、[P-Aの強いrun対照][PABASE] | `basis_plan`によるfull/support選択、同一runのsupport unionと既存boundary最適化 | 起点の古いDF builderだけを対照にしない。将来の新SOURCEでは必要な既知最適化を双方へ同じ条件で与える。今回移植・mergeはしない |

`build_event_sequence` は中央emitterのcallbackを現在公開していない。`BasisProvider`の差替えだけではRustiq中央合成を挿入できない。また、flat native circuit中のRZを見ただけではGaussian由来か中央由来か分からない。したがって将来の小さい変更は、**typed run/role/basisの境界を保つ中央emitter接口**に置くのが妥当である。旧frozen SOURCEを直接編集・monkeypatchせず、新版adapterで既存run走査・basis provider・wrapper・検査を再利用する。

### B2. controlled operatorと位相

既存の意味は `C(U)=|0><0|⊗I+|1><1|⊗U`。ancilla-last／little-endianの行列表現では `diag(I,U)` である。もしsystemで `U'=exp(iφ)U` を同じとみなすと、control後はancillaのrelative phaseがφだけ変わる。system channel同値・`Operator.equiv` のみでは受理しない。

| 位相／順序の由来 | 既存処理 | adapter条件 |
|---|---|---|
| Hamiltonian scalarと抽出identity | `−cT`等を `qc.p(...,ancilla)`、またはDF builderのstep集約で実装 | そのまま保持し、system global phaseとして捨てない。二重加算しない |
| Taylor phase | paired eventの `(-1)^(n/2)` をancilla相対位相へ | K0だけのtestでは網羅できない。K2を含む既存motifも将来対照に使う |
| product sign／identity product | 負係数productのπを相対位相へ。nonidentity product本体はCZ等 | productをrotationへ無条件変換しない。最初はproduct gateをそのまま境界として保持 |
| rotation sign | `RTEEvent.unsigned_rotation_angle = coefficient_sign * rotation_angle` | 実際に渡されるsigned aを一回だけ使う。coefficient magnitudeをrotation角へ足さない |
| identity rotation | systemの `exp(−iaI)` をancilla phase `p(−a)`へ | all-I tripleへ変換せず、既存phase記録とcallを保持 |
| Gaussian basis位相 | 基底のactual adjoint対が中央を挟み、control0枝で打ち消す | Gaussian recipe・support completionを今回変えない。異frameを一つの同basis runとして束ねない |
| 末尾Clifford | Q13 full-phase経路はactual inverse | control1枝だけで未知phaseを残さない。末尾を省略して期待値測定へ吸収する方式は初期対象外 |

既知のcontrolled Pauli回転の同値式で接続できる。nonidentityのHermitian Pauli Pについて、通常のrotation角θを使うと

\[
C(R_P(θ))=R_{I_a\otimes P}(θ/2)\,R_{Z_a\otimes P}(-θ/2).
\]

右辺2項は可換である。PRの `exp(−iaP)` なら入力tripleの角は **aと−a** になる。これは既存controlled回転分解と整合させるための標準恒等式の再掲で、新しいPRアルゴリズムの設計ではない。system Pのsupportとancilla追加supportを明示し、**既にcontrolを含むこのnetworkをさらにcontrolしない**。basis非制御・中央controlという元の構造を保てる。

この式の向き・符号を用いたadapterの実動作は未検証である。P=Iは前記のphase専用経路、Pauli productの±Pは既存gate専用経路とする。Qiskitに渡すのは入力Pauli軸と角度であり、`exp`の時間係数や新しいHamiltonian和ではない。

### B3. 最初に適用する領域とfull-wrapperへの戻し方

1. **FRAME_Z run内の連続rotation領域**：選択済みGaussian basis対をそのまま残し、中央Z/ZZ回転だけをancilla込みPauli networkへ渡す。product、identity phase、basis切替、D/R境界は既存位置で保持する。中央が対角であることは利用できるが、異frameの演算子順序を無視する理由にはならない。
2. **PHYSICAL_PAULIだけのrotation領域（限定案）**：既存atomが明示する物理Pauli wordを利用できる。ここには非可換列もあり得る。ローカルH/S基底を含む同じ演算子へ戻すことを独立referenceで確認する。これはGaussianの巨大Pauli展開ではない。適用を含めるかは将来のcase・adapter仕様で固定する。
3. **未対応領域**：Gaussianを跨ぐnative RZ列の無差別抽出、全controlled evolutionを末尾phase省略で置換、prep/readout/measurementを含むwhole-circuit extractionは最初に採用しない。未対応blockは元の既存emitterに戻し、最適化済みと偽装しない。

局所networkを正しく置換しても、局所gate数の減少がそのままfull wrapperのcompiled減少とは限らない。basis・隣接D・scalar・readoutとのtranspiler相互作用があるため、**全体は既存の位相保持Qiskit経路で一体compileして比較する**。原wrapperのfingerprintを変えずに中身だけ差し替えず、candidateの新fingerprintへ親入力・event順・basis plan・route/optionsを結び付ける。

## C. 最小接続案と再利用範囲（実装はしていない）

### C1. 推奨案：Q13既存角度付きAPIへの小さいadapter

必要な構造は次の四つであり、network探索・DAG・tableau合成・回路費計測を新たに実装しない。

| 接続部分 | 新たに必要な構造 | そのまま再利用する処理 |
|---|---|---|
| typed領域の入口 | ordered occurrence、元atom ID/role、run/basis hash、signed angle、system/ancilla mapを保持。対象範囲を判定し、不適格領域は既存処理へ戻す | `RTEEvent.application_sequence`、`MixedAtom`、既存run選択と`BasisProvider`。support union policyも対照と共通 |
| rotation binding | nonidentityだけのQ13 sparse triple。controlled半角・sign・idle qubitを保持。angleを近傍Clifford値へ丸めない | Q13 `synth_pauli_network_rustiq` のrotation挿入／符号付き共役／DAG／actual末尾inverse |
| 合成後の再接続 | 同じregister・basis境界・scalar phaseへ戻す。actual adjoint・wrapper metadataを新pipeline identityで作る | `actual_adjoint`、`hadamard_wrapper`、`compiled_wrapper`、独立reference。旧SOURCEは変更せず新版へ限定差分 |
| 証拠・cache | API/core/option/adapter版、input/ordered IR/basis plan、末尾復元方式、native profileを記録。cold/warmの区別とcandidate namespace | `compile_cache_key`、`matrix_gate`、`controlled_gate`、`wrapper_signal`、`signal_gate`、`f2_execution.circuit_metrics` |

Q13が既に存在するので、**version変更はSOURCE上の必要条件ではない**。既存環境へ新packageを追加する前提も置かない。active環境の実API・依存動作は今回importしておらず、限定実装・環境確認の承認後に検査する。

### C2. 指定R425を直接使う場合の追加負担

R425そのものを必須にするなら、Q13 public APIをcallしただけでは満たさない。R425へPauli文字列を渡すだけの薄いwrapでは不十分で、返った骨格へ角度を挿す処理が必要になる。

既存Q13の `inject_rotations`、commutativity DAG、`qiskit_rotation_gate`、actual inverse閉鎖の実装が再利用元になる。ただしQ13の公開Python接口は**外部骨格を受け取らない**。R425と接続するには適法な既存コードの限定移植または別の既存完成adapterが必要で、それを今回発見済み／完成済みとはしない。転用時はlicense・convention・core版の差を保ち、同じ処理を独自方式として再発明しない。

その追加負担を払う目的が「core0.0.11を指定して比較する」ことなら、新版比較として別に固定する。Q13経路だけでC4の初期可否を確認する方が変更要因は少ない。R425の骨格単独または `fix_clifford=True` による結果を、完全controlled operatorの完成回路と報告してはならない。

### C3. 強いbaseline・version効果・古典費用

| 条件 | 意味 |
|---|---|
| B0：固定Qiskit1.3.0 baseline | 現行の合法basis/run共有、support/full選択、boundary融合、phase保持loweringを適用した既存wrapper |
| B1：同じ環境・Rustiqなし | adapterの同じ領域分割・入力binding・IR表現を使い、networkを通常の正確な合成へ戻す対照。controlled回転分解だけを変えた効果をRustiq固有利益に数えない |
| R1：同じ環境・Rustiqあり | 同じ領域へQ13 APIを適用し、末尾inverse込みで比較。B0との差は全pipeline効果、B1との差はnetwork合成側の効果として記述 |
| version変更する場合 | 旧1.3.0、使用環境のRustiqなし、同環境のRustiqありを分離。異core・異Qiskitの変化を一つのRustiq倍率としない |
| 既存Rustiq baselineの意味 | Qiskit1.3.0に関数が存在することと、既存PRの通常 `transpile` が既にその関数へ同じ明示列を渡していることは別。読んだ `compiled_wrapper` にRustiq明示callはない。ただし将来は**既存Q13 Rustiq接続自体を強い既知法対照**として扱い、独立新compilerと呼ばない |

`CompilerProfile.validate()` は固定値を厳密要求する。Q13の合成前処理を加える場合も、旧terminal compiler profileを維持した上で新route identityを明示する。Qiskitを更新する場合は旧profileを書き換えず、新契約のprofileを作る。

cache keyはfull IRを含むが、それだけで全実験の来歴が説明できるわけではない。`identity_context`へroute、Qiskit/core、adapter/options、入力の順序・angle・basis・control・scalar、許されたcache scopeを結び付ける。異pipelineの旧価格・同じ見掛けのevent IDをcache hitとして流用しない。

古典費用は入力metadata復元、typed領域判定、IR変換、Rustiq合成、末尾復元、full-wrapper build、lower/transpile、verification、cache I/Oを区別する。既存timingと同じくwall・CPU・cold/warm・cache missを分け、検証用高精度referenceの費用を運用時の必須費用へ無断加算しない一方、無料とも扱わない。今回その測定やspeedup計算はしていない。

## D. 将来の最小科学検証案（未承認・未実行）

### D1. 再利用する既存test・独立reference

| 資産 | 再利用する検査 |
|---|---|
| [tests/test_df_rte_qiskit_builder.py](../../tests/test_df_rte_qiskit_builder.py) | control0=I／control1=ordered U、順序・cross-event共有、異basisを消さないこと、中央だけcontrol |
| [tests/test_df_partial_s2.py](../../tests/test_df_partial_s2.py)、[test_df_partial_s2_repeated.py](../../tests/test_df_partial_s2_repeated.py) | scalar／identity・one-body、反復ordered product、D boundary最適化とcontrol block |
| [tests/test_df_rte_tail.py](../../tests/test_df_rte_tail.py) | I/Z/ZZ抽出、identity phaseとbasis ID分離。新H生成を伴うtestは今回実行せず、将来も限定caseから必要箇所だけ再利用 |
| [compile phase回帰](../../tests/management/test_pr_dictionary_compile_phase.py) | absolute operator、global phase、Gaussian primitive／controlled回転、固定native basis。許容差緩和なし |
| [reference.py][PRREF] | CAR・exterior frameに基づく独立operator、event・trajectory、control、wrapper X/Y。candidateの回路やRustiq結果をreference生成へ流さない |
| [Q13既存test][QTEST] | plugin存在、順序・parameter、operator比較、`upto_phase`の意味を確認する参考。upstream testを今回実行した、あるいはPRでの適合性を確認したとはしない |

旧test一式を一律実行する提案ではない。分子生成・compile・simulationを起動し得るため、将来の実行契約で必要なcaseに限定する。今回testは静的に読むだけで、新しい人工operator検査も0件。

### D2. 小さいcase集合の候補

[既存cases.py][PRCASE]と[保存F1のID計画][F1PLAN]にはM0～M7の短いmotif、raw／optimized／adjoint、X/Y wrapperの対応が既にある。[初期調査D2][SURVEY]の条件を、その資産へ対応させる。

| 必要な論点 | 既存motifの候補と確認事項 |
|---|---|
| 順序・異basis・非可換 | M0のu/v交互列、候補arm PP。**異basisという条件だけでは非可換を保証しない。** 将来、保存word/supportを照合して実際の非可換pairの存在とordered列を固定する |
| 負角度 | M1の負coefficient atom sを含む列。eventのunsignedという名前に惑わされずsigned angleを確認 |
| Taylor phase・product sign | M2のorder2 product→child、M3のidentity／負product。rotation-only最適化の外に残すgate・phaseまでfull wrapperで確認 |
| 同frame共有 | M4の同じuの連続、M5の同じproduct pair。SSを候補にしてbasis/run共有を壊さないことを確認 |
| run／反復境界 | M6のu,u,v,vとq=2のD境界。固定plan・boundary融合を候補と対照で共有 |
| identity/scalar・空の最適化領域 | M7のidentity列と共通scalar。Rustiq対象が空なら既存回路を保持し、phase省略や誤った空回路化を検出 |

開始案は上記の小さいmotif subsetで、全arm×全motif、全F2、長いtrajectoryや別分子へ広げない。候補arm・case ID・対象block・実際の非可換coverageは**今回確定していない**。同じu/vが可換なら非可換test済みとはせず、別の既存reference付きcaseを探すか、必要な追加人工caseを明示提案して承認へ戻す。自動置換・追加科学callはしない。

F1候補の固定モデルは4 mode・2粒子・occupied(0,1)、density-square合成H、`L_D=1`・残差2fragment、`T=1/5,q=2,δ=1/10,r=2,h=1/20,K=2` の一点。分子geometry／AO basis／分子DF rankは非該当。これは再利用候補のscope記述で、新しい検証結果ではない。F2のK0・q1条件を混ぜず、認証N・同精度総資源は対象外とする。saved referenceのprivate原本は今回回収していない。

### D3. 実行前に固定するcall範囲とgate

| 段階 | 将来の限定検査 | 承認前に固定するもの |
|---|---|---|
| 環境・adapter準備 | Q13 APIと期待wheel、nonidentity triple・register/bit対応・options・cache identity。R425を使うなら別依存版 | 新SOURCE、再利用範囲、依存・install/importの可否。今回はその準備実行も許可しない |
| 局所operator | 選定領域の元実装／Rustiq実装をfull matrixで照合。末尾inverse・negative angle・idle bit・Y・符号を含む | build数、operator評価数、独立reference数、precision、許容差、対象block。identityをnetworkへ入れないこと |
| full controlled evolution／actual adjoint | full `diag(I,U)`、control0=I、control1の完全ordered U、actual inverseのabsolute operatorを照合 | raw/candidate/adjoint別の件数とreference。system phase fitやsector制限だけでPASSにしない |
| full wrapper X/Y | 共通prep/readoutを含め、measurementを外したunitaryと独立wrapper referenceを比較。X/Y信号を個別・複素で比較 | axis別build／operator／signal評価数。statevector計算も科学callとして明示。shot samplingは不要 |
| compile後gate | 正しさ通過後だけ同じ位相保持lowering＋native compile。operatorとsignalを再照合 | compiler環境×case×axisごとのcompile要求／unique／alias、検査call、cache scope。旧grantやcacheを実行許可にしない |
| 資源・古典費用 | RZ/CX count・各depth・total/quantum depth・size、末尾復元込みのfull wrapper。変換／合成／compileの古典費 | 主metricと比較条件、単純なcase比較のscope。保存law全体へ新しい期待資源を集計するなら別の科学再解析として承認 |

同じ既存P0 scopeを使う場合、既存の `rtol=0`、phase fitting禁止と、control/operator 1e−10、signal 2e−10、compiled 5e−10の閾値が再利用候補である。新しい受理幅を本書で採択しない。90/128桁referenceの保存値を使えるか、新規に評価する必要があるかも実行前に判断し、そのcallを無料で省略・追加しない。

上記は将来の契約に必要な**検査単位**で、今すぐ実行できる件数表やgrantではない。case ID一覧・各段階のbuild／compile／operator／signal／reference call数と依存順序は、実装準備後・科学実行承認前に固定する。科学的同値性のgateを先に置き、失敗したcaseを資源改善として数えない。

必要な処理が重い場合だけ、実際の空きCPU/GPU/RAMと既存timingを調べ、独立caseのbuild/compile/reference等を適切に並列化する。恣意的なCPU/RAM/worker・phase/total wall上限を新設せず、共有環境や他ユーザーのprocess・設定を変更しない。科学call・出力・log・instruction制限とSTOPは別に維持する。

## E. 技術的結論、未解決条件、次の担当

### E1. 推奨状態と根拠

**`READY_FOR_LIMITED_IMPLEMENTATION_PROPOSAL`を推奨する。**

- ordered Pauli軸を渡せる既存SOURCEがあり、Q13に符号付き角度挿入と非可換DAGがある。
- full-phase設定は単なるAPI上の文言だけでなく、実際のinverse gate列を追加するSOURCEへ追跡できた。
- explicit controlled rotationをnonidentity Pauli列として扱い、Gaussian／identity／scalar処理を保持する接続域を説明できる。
- 既存のoperator・actual adjoint・full-wrapper X/Y・位相保持compile・metric/cacheを再利用できる。
- Qiskit1.3.0の固定SOURCEに必要APIがあるため、version更新・新compiler作成を前提とせず小さく始められる。

判定対象は**上記の入力域と設定に限定した接続提案の準備可能性**。指定R425とQ13/C010は別版である。指定R425だけへ限定するなら追加のrotation挿入adapterが必要で、そのまま使える完成Qiskit adapterを確認できたという意味ではない。

`NEEDS_ADDITIONAL_INTERFACE_EVIDENCE`を全体へ付けるほどの未取得接口は残っていない一方、**科学的同値性と資源差は未検証**である。`DEFER_RUSTIQ_ROUTE`やRustiq一般のNO-GOを認定する静的証拠もない。末尾を正確に閉じる費用が利益を消す、短い対角列では既存Qiskitと差がない、またはadapter費が勝る可能性は残る。存在するAPIを接続するだけの価値は既知法の適用改善であり、独立新アルゴリズムのclaimではない。

### E2. 未取得・未確認の範囲

| 区分 | 残るものと影響 |
|---|---|
| 実環境 | 既存venvのQ13公開API・core/wheel identity、R425のbuild/ABIは今回import/installしていない。SOURCEと実環境の一致は将来の限定準備で確認 |
| 指定R425の完全adapter | 角度挿入・signed共役・actual inverseまで統合した外部完成adapterは未確認。Q13内部実装が再利用の根拠だが、異版へ移植済みではない |
| phase情報の不足する経路 | R425 `fix_clifford=True`／tableau-only extractionから元global phaseを返す接口は見当たらない。数値phase fitで穴埋めせず、actual inverse経路へ限定 |
| case原本・coverage | 保存F1 motif定義とID計画は確認したが、private QPY/reference全件は未回収・未decode。非可換pairの実coverage・確定ID・新call数は未固定 |
| 正しさ・性能 | PRへの接続、位相、X/Y信号、full compiled資源、古典時間は新規未評価。γ・認証N・全round同精度総資源も今回扱わない |
| 文献・取得限界 | Rustiq論文は公式abstractと§2のrotation／Pauli-network説明を確認。後半全benchmarkの再監査や倍率のPR移送はしない。docs.rsの指定SOURCE URLは取得不能だったためGitHub固定core SOURCEで代替。QuCLEARは前回固定調査の比較境界のみ再利用し、今回の独立実装調査・検証を追加しない |

### E3. 承認とmandatory STOP

1. **ユーザー／研究Cチャット・研究統括**：最初の対象を「Qiskit1.3.0同梱Rustiqの限定接続」とするか、「指定R425/C011の追加adapter」まで必要かを判断する。実装・科学的GOは本書で採択しない。
2. **Codex（別指示・承認後）**：選ばれた経路の最小adapter／新SOURCE・限定case一覧を準備する。既存code/test/referenceを再利用し、build/compile/operator評価の必要数と実行契約を具体化する。
3. **ユーザー**：install/importを含む実装準備の範囲、および回路build/compile・operator/signal/reference評価などの科学実行を別途明示承認する。旧one-shot grant・ledger・marker・STOPを流用しない。

本書追加以外のSOURCE・共通研究文書・概要・validation status・manifest・結果・grant・STOPは変更しない。独立worktreeの起点との差分、文書参照、空白、元workspaceのtracked差分と主要保護ファイルhashを照合する。旧raw全件の再hash・科学的再現を実施したとは主張しない。

専用branchへ本Markdownだけをcommit/pushし、remote refと固定commit本文・Git blobを再取得照合する。最終SHA・URL・公開照合の結果はチャットへ返す。**公開と照合が完了した時点でmandatory STOP。** 新adapter実装・compiler実行・科学計算、C1やGaussian singletonへの拡張、他branchへのmergeは自動開始しない。

## 固定参照と実際に確認したSOURCE

本書の相対PRリンクは起点に含まれる版。外部は以下の固定commitの対象fileを実取得し、別固定PR版はローカルGit objectの対象箇所を静的確認した。

- R425：`python/rustiq/__init__.py`、`src/interface.rs`、`python/rustiq/utils.py`、`examples/pauli_network_synthesis.py`、`Cargo.toml`、`pyproject.toml`。
- C011：`src/synthesis/pauli_network/synthesis.rs`、`greedy_order_preserving.rs`、`greedy_pauli_network.rs`、`src/structures/pauli_set.rs`、`pauli_dag.rs`、`clifford_circuit.rs`、`pauli_like.rs`、`src/routines/rotation_extraction.rs`。tagをcommitへ解決して確認。C010はsynthesis.rsの関連処理とtag identityを確認。
- Q13：`qiskit/synthesis/evolution/pauli_network.py`、同`__init__.py`、`product_formula.py`、`suzuki_trotter.py`、`crates/accelerate/src/synthesis/evolution/pauli_network.rs`、`qiskit/transpiler/passes/synthesis/hls_plugins.py`、`test/python/transpiler/test_high_level_synthesis.py`の関連class、`test/python/circuit/library/test_evolution_gate.py`関連箇所、Cargo/lock/pyproject/requirements。test実行はなし。
- PR：B1/D1のSOURCEに加え `rte.py` のapplication順とsigned angle、`cases.py`、`f1_saved_plan.py`、`f1_saved_execution.py`、`f2_execution.py`、`f2_resources.py`、`reference.py`、`fixture.py`、関連文書・保存JSONのstatus/metadataを確認。SOURCEはimportしない。

[SURVEY]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c0924b1fdcc3a28d9a65062b4feaf9d0db944328/docs/research/research_c_hamiltonian_circuit_survey_20261024.md
[CLOSE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/69f8e1cac23a4924ab81aebe399657a705024f6a/docs/research/pr_gaussian_singleton_closeout_handoff_20261024.md
[RPY]: https://github.com/qiskit-community/rustiq/blob/425000222e21a7a28d70a08c75be968f16fd6903/python/rustiq/__init__.py
[RIF]: https://github.com/qiskit-community/rustiq/blob/425000222e21a7a28d70a08c75be968f16fd6903/src/interface.rs
[RUTIL]: https://github.com/qiskit-community/rustiq/blob/425000222e21a7a28d70a08c75be968f16fd6903/python/rustiq/utils.py
[C11SYN]: https://github.com/qiskit-community/rustiq-core/blob/dbe33fa8f03cf46134402a75432a5b289494b813/src/synthesis/pauli_network/synthesis.rs
[C11DAG]: https://github.com/qiskit-community/rustiq-core/blob/dbe33fa8f03cf46134402a75432a5b289494b813/src/structures/pauli_dag.rs
[C11SET]: https://github.com/qiskit-community/rustiq-core/blob/dbe33fa8f03cf46134402a75432a5b289494b813/src/structures/pauli_set.rs
[C11CLI]: https://github.com/qiskit-community/rustiq-core/blob/dbe33fa8f03cf46134402a75432a5b289494b813/src/structures/clifford_circuit.rs
[C10SYN]: https://github.com/qiskit-community/rustiq-core/blob/cbd8db823c0ab55f9e1522b0487e8494acefc837/src/synthesis/pauli_network/synthesis.rs
[QPY]: https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/qiskit/synthesis/evolution/pauli_network.py
[QRS]: https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/crates/accelerate/src/synthesis/evolution/pauli_network.rs
[QHLS]: https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/qiskit/transpiler/passes/synthesis/hls_plugins.py
[QTEST]: https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/test/python/transpiler/test_high_level_synthesis.py
[IBMAPI]: https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.transpiler.passes.synthesis.hls_plugins.PauliEvolutionSynthesisRustiq
[PRC]: ../../src/trottertracks/pr_dictionary_validation/circuits.py
[PRG]: ../../src/trottertracks/pr_dictionary_validation/gates.py
[PRDF]: ../../src/trotterlib/df_rte_qiskit.py
[PRS2]: ../../src/trotterlib/df_partial_s2.py
[PRREP]: ../../src/trotterlib/df_partial_s2_repeated.py
[PRTAIL]: ../../src/trotterlib/df_rte_tail.py
[PRREF]: ../../src/trottertracks/pr_dictionary_validation/reference.py
[PRCASE]: ../../src/trottertracks/pr_dictionary_validation/cases.py
[F1PLAN]: ../../src/trottertracks/pr_dictionary_validation/f1_saved_plan.py
[B2978]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2978e2fea672b7a1ff20cac74269ec9a610159dc/src/trotterlib/df_rte_qiskit.py
[R2978]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2978e2fea672b7a1ff20cac74269ec9a610159dc/src/trotterlib/df_partial_s2_repeated.py
[PABASE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/src/trotterlib/research_direction_joint_synthesis_mechanism_validation.py
