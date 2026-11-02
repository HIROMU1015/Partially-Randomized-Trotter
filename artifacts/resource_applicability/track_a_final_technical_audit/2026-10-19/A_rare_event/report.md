> 公開用projection。private内部パス、原入力・回路・運用記録、ローカル補助scriptは含まない。本文に記す照合件数は原調査の報告値であり、未公開原資料を再公開した意味ではない。

# A：希少event上界の静的検証

2026-10-19 JST。判定は **L1は固定profileの条件付きコード事実として確認、L2は一部確認・全体未確認、Tier Rの科学的成立は未認定**。新compile・回路生成・sampling・上界数値計算・保存科学結果の再集計は行っていない。

## 取得した固定資料と対象

Git object storeから `64c85f7a833b5950ef2ccae623708b73081edcd6` の研究概要、PROJECT_MAP、native理論モデル・接続検証、実compiled-cost caller等を取得した。A_COST結果は指定 `fc6bd039dc7171d5674ac04f0b3ad427f1b6495b`、希少層結果は指定 `cf8e93eef50333bef41f6d75803f6b819d679343` からも取得した。読んだreview原文は `../evidence/レビューpr_2.md` §4.3–4.6（L1/L2は§4.5）。主張の出所をレビューとコードで分離した。

公式 [Qiskit 1.3.0](https://github.com/Qiskit/qiskit/tree/1.3.0) のtag参照を取得し、commit **`896d1c8b72888a52a46081a6d1f4de8c56db8aea`** へ解決してarchiveを取得した。source.tar.gzとtag/API receiptは `qiskit/`、取得記録は `acquisition_receipts.json`。展開した全ファイルを読了とは扱わず、使用したファイルだけ `source_manifest.json` に `read_scope` を付す。

Qiskit Nature 0.7.2の [BogoliubovTransform公式source](https://github.com/qiskit-community/qiskit-nature/blob/0.7.2/qiskit_nature/second_q/circuit/library/bogoliubov_transform.py) も取得した。SHA-256 `b3956e8469dfe22ed57a5bfe6ce341f34dfda6a22bef7082855371c157b181f2` は保存接続receiptと一致する。同receiptのQiskit gate-definition 7項目すべてと取得公式sourceのSHAが一致した。公式 `qiskit/compiler/transpiler.py` のSHA `749fcca54c3508e9ed0da83aa5e7f0a15074f73938b20fb9b5a34468a890ccef` は固定compiler profileの値と一致した。さらに固定runtime receipt（digest `831e2b98…`）の期待値6項目と、実installedのlevel1/builtin_plugins/common/generate_preset_pass_manager/add_control/standard equivalence libraryの6ファイルを実際に読み直してSHA比較し、すべて公式sourceと一致した（`runtime_source_hash_comparison.json`）。これは選択した6ファイルの対応証拠であり、実験runtime全closureやRust binaryの再同定ではない。

## L1：level 1のpass意味論

固定profileの公開原本は `87b6b9f46b7e8ba9abe132ee21478c114b6f46f7:artifacts/resource_applicability/track_a_h6_server_preparation_v2/2026-10-12/environment_compiler_profile_v2.json`。basis=`rz,sx,x,cx`、optimization_level=1、seed_transpiler=17、backend/coupling_map/layout_method/routing_method=null、Qiskit1.3.0、settings_hash=`1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。`transpile_and_measure_cost` はこの設定をkwargsに渡し、backend引数は指定時だけ付ける。H8のprofile識別子 `622ce4a8…` はH4/H6原profile file bytesと同一とは記録されていないので、H8まで同一file hashと断定しない。

[公式level1.py](https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/qiskit/transpiler/preset_passmanagers/level1.py) と `builtin_plugins.py` / `common.py` / `generate_preset_pass_manager.py` を照合した。

1. backend=None、coupling_map=None、target未指定なら `_skip_target=True` でtargetを構成しない。layout/routingはNone。通常のdirection補正も入らない。
2. initは `InverseCancellation`。このprofileでは3q unrollの追加条件に該当しない。Rust実装は該当nodeを除去する。
3. translator stageは **UnitarySynthesis → HighLevelSynthesis → BasisTranslator(SessionEquivalenceLibrary)**。翻訳前の高水準命令数をTとは呼べない。Tを測る概念上の位置は、このstageを完全に終えた直後。
4. level1最適化loopは `Optimize1qGatesDecomposition → InverseCancellation → GatesInBasis/必要時再翻訳 → depth/size固定点検査`。level2/3のCommutativeCancellation、ConsolidateBlocksをlevel1へ混ぜない。
5. Pythonの古い `_substitution_checks` 読解だけでは不十分。実際の `run` はRust `optimize_1q_gates_decomposition` を呼ぶ。[Rust実装](https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/crates/accelerate/src/euler_one_qubit_decomposer.rs) の1055–1239行付近で、target=None時は old_error=(run長,run長)、new_error=(新列長,新列長)。完全基底内なら `outside_basis=false` なので置換は長さ減少または空列の場合。`compute_error_list` もerror_map=Noneなら長さを返すが、実runの根拠は前者。
6. したがって、**完全にbasis内で、target/error mapなし、routing・direction・scheduling追加なしのこのpipeline**では、翻訳後の量子gate総数をlevel1最適化は増やさず、最終RZ数≤翻訳直後gate総数というコード上の関係を支持する。measureはwrapperに1命令として別扱いにするか、Tへ一貫して含めてより粗くする。Tをnative命令数・理論RZ数と同一視しない。

**条件外の反例構造**：Rustには `outside_basis || new_error < old_error` のguardがある。未翻訳gateが残ればgate数増加を許す。hardware Targetがあれば誤差改善を優先し、gate数増加を排除しない。coupling/routing/direction補正でも増加しうる。よって「Qiskit1.3.0 level1一般でgate数非増加」は不成立な一般化。ここでの反例は実コードの分岐構造の同定であり、新しい数値compile反例を作ったという意味ではない。

RZ個数そのものの単調減少は確認しておらず、総数の非増加から主張していない。任意のnative gateを1命令と数えた値へのL1適用も不適切。

## L2：native gate、制御、library、加法性

native builder `ax2b_native_df_v5.py` はprovider SOURCE `217c971d…`（H4）、`87b6b9f4…`（H6）、`c31ef424…`（H8）から実際に取得し、**3版のファイルSHAはすべて `7b8b13483066bb39cc4aad1c36f6e5da6fbc679ecb8bae9c08a0b326648a36fd`** と一致した。common library `df_trotter/ops.py`、`df_rte_qiskit.py`、`df_partial_s2.py` の関連関数も静的に照合した。元実行SOURCEとprovider SOURCEを混同しない。H8継続SOURCE `87f1f0d528478ad212ba98159db06eec3c78433e` の `pr_cost_native.py` も取得した。

- square-Uのnumber-conserving Gaussian producerはRZとXXPlusYYをemitする。basis inverseは逆順に `gate.inverse()` をappendし、XXPlusYYのinverseは同gate familyのtheta符号反転。Gaussian全体を制御する経路ではない。
- native S2/S4 deterministic blockのbasisはuncontrolled。ordinary diagonalはCRZとnumeric `RZZGate(angle).control(1)`、directional diagonalはCX/RZ parity列。global scalarはancilla Pを1回追加。S4の負timeもここへ流れる。
- ordinary-controlled RTEのproductはZ supportの各qubitにCZをappendする。rotationはCRZまたはcontrolled-RZZ、identity/符号位相はaccumulated ancilla P。reuseは組立時のactive basis状態に依存するため、event単体のtを単純加算するのではなく、**組立後の実gate列Q(ω)** が対象。
- wrapperはH、任意Sdg、H、任意measure。state preparationは費用scope外。

`native_gate_static_table.csv` にgate family、制御、literal templateと未解決事項を保存した。表の整数はソース中の局所template長のmetadataであり、研究回路の集計・rare上界ではない。

### 確認できた加法性

`BasisTranslator`は(source name,num_qubits)ごとの置換mapを合成し、各nodeへtemplateをcopy、parameterをbindしてappendする。Rust `compose_transforms.rs` と `basis_translator/mod.rs:apply_translation/replace_node` では、数値角度をbindするこの段階でtemplate gate列を最適化・相殺する処理はない。**同じmapが固定され、特殊gate/HLSの前処理も固定されたとき、翻訳後template長を各nodeに足す加法性**はコード上確認できた。

ただしdefault translatorは単一 `_define` を機械的に展開するだけではない。`basis_search.rs` はSessionEquivalenceLibraryの複数候補をDijkstra探索し、gate数に基づく重みでruleを選ぶ。RZZ、CRZ、H、SXdg、CCX等は代替ruleを持つ。固定basis・無変更libraryでは定型的な展開に還元できる見込みがあるが、本調査では**全native familyの選択ruleと最終basis総数t(g)の完全表、およびその全経路の証明**は完成していない。

### controlled RZZの重要な相違

numeric RZZの`control(1)`は `RZZGate.control → Gate.control → add_control`。`add_control.py` の `_unroll_gate` は内部でUnrollCustomDefinitionsとBasisTranslatorを使う。RZZのCX–RZ–CXを制御すると **CCX–CRZ–CCX** を生成する（mcrzの単一control・use_basis_gates=FalseはCRZをappend）。その後CCXをさらに翻訳する。

理論モデル `explicit_rz_sx_x_cx_v1` のCRZZは **CX–CRZ–CX** のparity構成で、これは同じunitaryを表す別実装。前者のt(g)へ後者の「2RZ+4CX」を代入できない。XXPlusYYでも理論モデルのRXX/RYY経由とQiskit等価libraryの14命令templateは別経路。理論カウンタのunitary接続検証PASSを、実compilerの展開数一致PASSへ変換しない。

### L2の未確認・条件・反例となる構造

- **全native t(g)：未確認**。特にcontrolled-RZZ→CCXを含む再帰closure、XXPlusYYのRY/SXdg等、選択ruleをすべて固定した総数表はない。Qiskit sourceのliteral展開の存在と、実compileで採用される展開の同定を区別する。
- **library状態：仮定付き**。SessionEquivalenceLibraryはmutableなsession wrapper。取得した関連project sourceでは追加ruleを見つけなかったが、全runtime全経路を調べた証明ではない。H8原profile file bytes・Rust binary・import時libraryの完全同一性も本調査で再認定しない。
- **全input alphabet：source構成上は標準gateに閉じる条件とH8復元whitelistを同定、全保存preparationに対する認定は未確認**。H8の別固定adapter `scripts/h8_key_compile_adapter.py:109` は復元factoryをRZ/XXPlusYYに限定する。これはH8全A inputの復元経路を全部証明したという意味ではない。native `validate_blocks` は任意numeric unitary Gateを受理し、basis gate名のwhitelistを強制しない。opaque/custom definitionやcontrolled Gaussianを含む任意API inputに同じt表を一般化できない。
- **一般のL2は不成立となる構造がある**。同名custom Gateでもdefinitionが違い、UnitaryGateはmatrix依存のsynthesisを受ける。一般control処理はcontrol数、open-control state、数値parameterによる分岐やglobal phaseで列が変わる。これらは実装上の反例構造であり、意図した固定standard-gate subset内の実測反例を発見したとは言っていない。
- **loop/control-flowを数える定義**が必要。現native evolutionはGateのみ、numeric、classical bitなしであり、wrapperのmeasure以外のdynamic control flowを想定しない。一般Qiskit dynamic circuitへ式を拡張しない。

## Tier Rと論文記述の扱い

既存 `cf8e93ee…` のRARE_SENSITIVITYは保存標本のμhat0/μhat2_singleと未知μ2_multi/μ4による条件付き逆転境界であり、未知費用・母平均保証はnullを維持している。レビュー§4.5のT/C≈3〜4および「中間窓が閉じる」は仮定値の未計算試算。今回その数値を実計算・認定へ格上げしていない。L1の静的コード事実だけから母平均優位や中間窓の厳密頑健性は出ない。

論文に使える記述案：

> 固定されたQiskit1.3.0のbackend/couplingなし・rz/sx/x/cx・level1 profileでは、完全な基底変換後の最適化が量子gate総数を増やさないことをsourceで確認した。rare軌道のcompiled RZを基底変換後総数で抑える方法は候補となるが、制御ZZを含む全primitiveの実translator展開と加法的会計の結合、および期待値上界の評価は未完了であり、母平均に関する保証は付与していない。

ここで「sourceで確認」は技術調査であり、GPT/Claudeによる新しい科学レビュー・数学的定理認定ではない。

## 別承認を要する最小作業

1. Tier Rの**数値**を必要と判断した場合に限り、担当Codexが、固定Q(ω)文法・全native t(g)または正当化した上界重み・compiler/library/Rust identityを先に固定し、元lawの有限DP期待値を新しい数値解析として別承認範囲で評価する。今回実施0。
2. 静的closureだけで解決せず局所compileを必要と判断した場合、担当ユーザーが対象gate/control/boundary類と少数の合成compile範囲を別承認する。実科学trajectory sampling・独立確認はその承認へ含めない。今回合成compileも0。
3. 現在の有限標本・条件付き感度として論文化するなら、Tier Rの認定値を必須とせず、未観測rare/選択効果を明記して進めることは技術上可能。科学的採否はユーザーの判断。

## 実行境界・保全・STOP

専用 `A_rare_event/` への新規文書・取得コピー・metadata receiptのみ作成。project Git objectの`show`読取と公式source取得のみ。Qiskit import0、回路生成0、transpile/compile0、科学サンプル/新signal/SCF/DF/state生成0、保存回路の新数値集計0、新科学レビュー0、SOURCE/raw/grant/ledger/marker/既存STOPの変更0、他job操作0、push0。テストは実行せず、source/hash/metadata照合で検証した。共通最終STOPは親担当の統合時に実施し、本担当はAの報告提出後に追加stageへ進まない。
