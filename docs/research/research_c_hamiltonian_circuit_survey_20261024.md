# 研究C：Hamiltonian表現・量子回路合成改善の初期技術調査

2026-10-24 JST。**read-only技術調査と候補の推奨であり、研究機構の正式採択・新アルゴリズム設計・実装・科学検証ではない。**

**推奨はC1（native回路費まで接続する圧縮DFの適用可能性）とC4（順序・位相を保持したRustiq接続）。** 中心的な研究候補はC1、先に小さな技術的可否を識別しやすいのはC4である。いずれもPRの同精度総資源削減や独立した新規性を確立していない。C2の一般的Gaussian合成は残すが、終了済みsingleton選択・P-A区間分割の再開は推奨しない。

## 0. 作業範囲、固定資料、読み方

- Repository：`HIROMU1015/Partially-Randomized-Trotter`。
- 起点：`69f8e1cac23a4924ab81aebe399657a705024f6a`。専用branch：`codex/research-c-hamiltonian-circuit-survey-20261024`。独立worktree：`/tmp/prt-research-c-survey-20261024`。元workspaceのdirty変更を取り込まず、本書だけを追加する。
- 必須資料の[Gaussian singleton終了・引継ぎ][CLOSE]と[次期改善機構調査][NEXT]は、指定された固定commitの本文をGitHub connectorから実取得した。後者は別branchの成果として読み、起点へmergeしない。
- 正式共通指示 `research_project_instructions_unified_2026-10-23.md` は元workspaceから読取り。SHA256：`7ce342dbf7086424f1afd837d8c274687bc28e4838601f5ad96f5b626d95667f`。起点にtracked copyはなく、複写・公開・改変しない。
- 起点の `PROJECT_MAP.md`、[研究概要・現状](研究概要・現状.md)全文、[研究目的](研究目的・研究課題.md)、[VALIDATION_STATUS](../../VALIDATION_STATUS.md)、[manifest](../../artifacts/validation_manifest.json)の関連登録を確認した。9月の近似検証・dirty証拠と、10月の固定公開結果を混同しない。
- 既存[横断調査][LANDSCAPE]、[Phase 2A一次資料案内][SOURCEGUIDE]、[強いbaseline対応][BASELINES]、[Phase 2C方法差][METHOD]を再利用した。下記の別commit実装はローカルGit objectから対象箇所を静的に読んだもので、すべてを今回GitHubから再取得したという意味ではない。
- 外部一次資料は関連節・公式API・著者実装を確認する限定調査。文献全体の再審査・全branchの不存在証明・網羅的priority調査は行わない。

新しいHamiltonian生成、分解・数値再解析、sampling、simulation、回路build／compile、QPY decode、SOURCE import、性能測定、compiler等の実装・installは行っていない。JSONのstatus／来歴の読取り、Git／文書の整合検査と、科学値の再集計は区別する。

### 名称の衝突

本書の**C1～C5は研究C内の5方向**である。旧Track Bの「C1 H4移植」や「C4 importance sampling」、旧研究候補P-Cとは別の識別子である。[NEXT]のwhole-trajectory ISは別系列の推奨で、本書はその採択・実装・継続許可を変更しない。

## A. 既存成果と証拠の整理

### A1. 再利用する固定証拠、負結果、未検証事項

| 確認した資料・証拠 | 検証済み／保存された範囲 | 負結果・制約・未検証事項 |
|---|---|---|
| [CLOSE]、[NEXT]、起点の概要・status・manifest | Gaussian singleton独立研究の一時終了がユーザー採択済み。辞書・law・実回路費・統計負担を接続する論点を引継ぐ | 合成全般の不可能性でもPR全体のNO-GOでもない。具体的新機構は未採択。既知法の組合せを新算法と認定しない |
| [F1実行報告][F1] | 固定合成fixtureで混在SS/SP/PS/PPの登録正しさ検証。未資格価格のfallback経路も確認された | 最適selector・認証価格・認証shot数の成功ではない。F1とF2はr以外の時間・cutoff等も異なり、価格を混用しない |
| [修正版F2報告][F2]と[保存要約JSON][F2JSON] | 登録されたraw／compiled operator・signal・finite mean検査が許容差内。保存statusは `F2_RESOURCE_NUMERICAL_PASS_MANDATORY_STOP`、SOURCEは `9468c7089727f5648d9f59cddbaebe0a42f5f0a0` | 1-shot指標はPP < SP < PS < SS、SSのnormalizationは小さいという交換。JSONの `N` はnull。γ・統計／interval資格・N加重費用は未成立。旧compile位相欠落FAILは保全され、修正版の成功で削除されていない |
| [F2寄与分析][ATTR]、[PS48報告][PS48]と[保存要約JSON][PS48JSON] | 寄与の記述的分解と限定Gaussian置換の同値性。PS48は `PS48_CORRECTNESS_PASS_LIMITED_CASE_COMPARISON_MANDATORY_STOP`、SOURCE `770e53addda63a03fa8c6c5d5168f7062067c572` | 一律置換によるRZ/CX削減は支持されず、局所改善はcase依存。PS48の等case比較は全F2の確率加重改善率ではない。一般化するcompile前選択規則は未確立 |
| [P-A機構検証][PA]、[事前登録][PAPRE]、[固定JSON][PAJSON] | 下記H4条件でrun内区間分割は一区間の強い対照と同じplan・compiled指標。run単位full／support-union選択の既存成果は保持 | 区間分割固有の主張を停止。任意のGaussian共同合成への一般no-goではない。閾値やprofileを事後変更して再開しない |
| [反復回路文書](../df_partial_s2_repeated_compiled_cost.md)、[event API](../df_rte_event_circuit_api.md)、旧[回路費検証](../random_circuit_cost_validation.md) | 中央部だけのcontrol、scalar相対位相、隣接同basis共有、反復境界の同一D block融合を既に実装・限定検証。費用の非加法性を既知の注意点として再利用 | event別compile費の単純和をfull wrapperの価格としない。古いcost proxyを新DF・新compilerへ無条件移送しない。最終全round総費用の完了を主張しない |
| [UWC実装記録](../../notes/uwc_current_implementation_and_results.md)、[不足資料一覧][GAPS]、status／manifest | 粒子数shiftの実装・単体test資産。manifestの `uwc_grouped` は `prose_only_missing_raw_artifacts` | prose表の小改善や悪化訂正を性能の成功・負結果と認定できない。比較JSON・sector診断・sweep・来歴が今回も未取得 |
| status／manifestの旧DF screening登録 | `df_partial_randomized_screening` は `invalidated_pending_regeneration` | stale screeningからrank・L_Dの科学的順位を採らない。入力表を新計算で埋め直すことも今回の許可外 |

**上表で数値的順位・負結果を引用する条件：**

- F1/F2/PS48のHamiltonianは4 mode・2粒子、初期占有(0,1)、`L_D=1`・残差2fragmentの固定density-square合成モデル。分子geometry・AO basis・分子DF rankは非該当、rank探索なし。F1は `T=1/5,q=2,δ=1/10,r=2,h=1/20,K=2`、F2/PS48は `T=1/5,q=1,δ=1/5,r=2,h=1/10,K=0` の一点。F2/PS48は同一full Hadamard wrapper（準備X0X1・readout・測定を含む）、Qiskit 1.3.0、basis `rz,sx,x,cx`、opt1、seed17、free topology、approximation_degree=1、qubits_initially_zero=false、位相保持修正版compiler。順位は1-shot期待回路指標だけである。
- P-Aはlinear H4、1.0 Å、STO-3G、8 qubit、DF rank12、`L_D=3,δ=0.02` の一点、全eventをTaylor order2へ固定したsource-basis runの構造検証。identityはextract、順序保持、Qiskit1.3.0／opt1／seed17／同native basis／routingなし。full wrapper・自然発生頻度・最終総資源は対象外。

これらは固定公開された**ローカル証拠**で、immutable CI・外部独立再現ではない。本調査では旧raw全件のhash監査や科学的再現をしていない。

### A2. SOURCEの再利用地図

以下の相対リンクは本書起点の版を指す。別版の強いbaselineは明示した固定URLを使い、同名モジュールの版を混ぜない。

| SOURCE／既存接口 | 既に達成していること、再利用先 | 最小の差分／未検証条件 |
|---|---|---|
| [df_hamiltonian.py](../../src/trotterlib/df_hamiltonian.py)：`DFHamiltonian`、`df_hamiltonian_from_integrals`、`PhysicalSector`、`df_linear_operator` | `constant + one_body + Σ λ_l A_l²`、rank/tol・one-body correction・truncation metadata、sector上のmatrix-free参照。C1/C3の元H・表現同定に使える | RC-DFの一般結合行列は直接格納できない。OpenFermionのtruncation戻り値をscalarへ足さない既存規則を維持。新表現の検査を古いground-state検査で代替しない |
| [df_trotter/](../../src/trotterlib/df_trotter/)：`df_decompose_from_integrals`、`DFModel`／`DFBlock`、`U_to_qiskit_ops_jw_givens`、`df_squared_diagonal_primitives`、`append_diagonal_primitives` | integrals→square fragment、Gaussian実装、I/Z/ZZ角度とglobal／control相対位相を分離。C1/C2の共通下流資産 | 空間／spin軌道の並び、chemist/InteractionOperator・1/2係数、転置conventionを照合。general diagonal係数を受ける入口は別途必要。既存Givens helperの再実装は不要 |
| [df_partial_s2.py](../../src/trotterlib/df_partial_s2.py)：`prepare_df_partial_s2`、`QiskitDFPartialS2CircuitBuilder` | 順位付きprefix・分割hash、one-bodyとscalar、決定論half sweep、中央RTEとの接続。C1/C2共通 | 分解変更でfragment数・順位・tailが変わる。同じ数値のL_Dを同じH_Dとみなさず、物理split固定対照と再最適化対照を分ける |
| [df_rte_tail.py](../../src/trotterlib/df_rte_tail.py)：`DFBasisRegistry`、`exact_df_diagonal_coefficients`、`extract_df_diagonal_tail`、`extraction_to_symbolic_rte_tail` | basis ID/hash、I/Z/ZZの係数・符号・identity policy、thresholdとλを追跡。多体行列なしの抽出を再利用可能 | `exact_df_diagonal_coefficients(eta,lam)` はsquare専用。RC-DFの一般Z行列をこの引数へ押し込まない。scalar／one-body補正、spin展開、full-space係数の接続が必要 |
| [df_rte_qiskit.py](../../src/trotterlib/df_rte_qiskit.py)：`QiskitDFRTEEventCircuitBuilder.build_event/build_sequence` | application順序保持、中央だけcontrol、Taylor／符号／identity相対位相、同basis共有 | 起点の版は強い最新baseline全体ではない。[2978版][BUILD2978]は`basis_plan`を受けfull／support方式を選べる。C2/C4ではこちらの最適化機会も対照へ与える |
| [df_partial_s2_repeated.py](../../src/trotterlib/df_partial_s2_repeated.py)：`QiskitDFPartialS2RepeatedCircuitBuilder._build_boundary_optimized` | 同一D_0の隣接halfを融合、basis往復を削減し、各step scalarを相対位相へ集約。非可換順序は保持 | [2978版][REPEAT2978]も確認。異frameの一般transition最適化はこの同一block融合の成功に含まれない。C2の比較はraw連結だけをbaselineにしない |
| [混在辞書circuits.py](../../src/trottertracks/pr_dictionary_validation/circuits.py)：`BasisProvider`、`support_completion`、`build_event_sequence`、`actual_adjoint`、`hadamard_wrapper` | full basis run共有、singleton support補完、typed Pauli/frame dispatch、actual adjoint、full wrapper。C1の抽出後・C4のcompile接口に使える | 現fixture専用のD側と汎用分子DF経路を同じとしない。C1は新分解入力のadapterが必要。singleton選択研究は再開しない |
| [gates.py](../../src/trottertracks/pr_dictionary_validation/gates.py)：`compiled_wrapper`、`matrix_gate`、`controlled_gate` | 位相欠落への公式definition展開、full matrix照合、compiler/cache identity。C4の基準 | 新compilerを入れた経路への保証はない。compiler版／options／IR・角度・位相をcache keyへ戻す必要。古いcacheで新価格を代用しない |
| [UWC](../../src/trotterlib/uwc.py)：`preprocess_qubit_hamiltonian`、`number_sector_shift_operator`、`_optimize_bliss_theta` | `θ(N−N_target)^power`、power1/2、sector check、hash分離。l1／lambda_rを目的とする1D最適化。C5へ再利用 | `estimated_total_cost` とorbital optimizationは `NotImplementedError`。一般BLISS全自由度の実装ではない。`simple_shift`は物理保存の対照に使わない |
| [grouped_uwc_comparison.py](../../src/trotterlib/grouped_uwc_comparison.py)：`build_grouped_hamiltonian_data`、`fit_grouped_trotter_alpha`、`compare_grouped_uwc_pf_qpe` | grouped対照、delta Z項のclique再配置、sector推定、PF係数とrotation／RZ-layer／T-depth proxyの会計 | full controlled PR native費用ではない。grouped-PFのL_Dやproxy T-depthをDF fragment／実T countと同一視しない。旧数値原本の欠損は残る |
| [P-A固定SOURCE][PASRC]：`preserved_columns_unitary_completion`、`make_interval_union_basis_plan`と[機構検証SOURCE][PAMECH] | 任意support union補完とrun単位full/support比較、区間DPが一区間へ退化する限定証拠。C2の強い対照 | 基底選択・境界効果を新方式だけへ許さない。既知DPを再実装して新規機構と呼ばない |

**既存test・runner・artifactの読取り範囲：** `tests/test_df_hamiltonian.py`、`test_df_partial_s2.py`、`test_df_partial_s2_repeated.py`、`test_df_rte_tail.py`、`test_df_rte_qiskit_builder.py`、`test_uwc_preprocessor.py`、`test_grouped_uwc_comparison.py`の関連検査を静的に確認した。特にcontrol0=I／control1=U、signed angle・scalar、別basisを相殺しないこと、反復境界同値性、sector・cache分離を再利用する。将来は既存 `tests/management/test_pr_dictionary_compile_phase.py` も回帰入口にする。

入口は `scripts/run_df_ground_state.py`、`scripts/run_grouped_uwc_pf_qpe.py`、`scripts/run_pr_dictionary_f2_resources_saved.py` の呼出し箇所・実行guardを確認した。`df_partial_s2_repeated_cost.py`／`df_rpe_hadamard_compiled_cost.py` の既存compiled-cost接口は将来の費用記録資産であり、今回呼び出していない。testの存在・過去PASS・今回の静的確認を、今回のtest実行PASSと書き換えない。

## B. C1～C5の比較

| 方向 | 改善機構・削減対象 | 主要先行研究／既知の成果 | 強いbaseline | PRへの接続・再利用SOURCE | 難易度・期待される価値 | 主なリスク |
|---|---|---|---|---|---|---|
| **C1 DFとnative費の共同評価** | fragment／内側rank・対角係数・frameを交換し、basis往復・中央RZ/CX・depthを減らし得る。normは別指標 | [Mottaら][DFPAPER]の低rank DF、[RC-DF][RCDF]の正則化fit、[SCDF][SCDF]の対称性併用。文献の測定／qubitization利益は確立した各対象内に限定 | 同じ元H・近似誤差枠のX-DF、適用可能な既存C-DF/RC-DF、collected Pauli。各表現へ同じbasis共有・boundary融合・compilerを許す | A2のDF生成・抽出・partial-S2・wrapperを共用。general diagonal入力にgapあり。分解変更後はlaw／PF入力も更新が必要 | **中～高**。まず既知法のPR適用・表現比較の価値。native構造に関する既知法との差が後に残れば独立改善の研究余地。現状は新算法なし | fewer fragmentsでも中央がdense化。fit費、H近似、順位/split、normalization、shot変化。文献norm改善率のPR移送不可 |
| **C2 複数fragment Gaussian共同合成** | 隣接する異frameの相対basis変換、中央対角部との合法な融合、block／反復境界の簡約。RZ/CX/depthを狙う | 既存Givens・低rank Trotter、[ffsimのrelative-basis実装][FFTRANS]。相対basisを直接合成する着想自体も既知 | equal-frame共有、run単位full/support-union、中央だけcontrol、D_0境界融合、phase保持whole-wrapper compile。既存合法transitionも公平に開放 | A2の2978版、P-A、typed builder、df_trotter Gaussianを再利用 | **中～高**。実装改善の価値はあるが、強いbaselineを超える具体差が未確立で優先を下げる | 異basisを単純に打消せない。dense化・再合成費、operator順序、control0枝。P-A／singleton停止済み範囲との重複 |
| **C3 軌道回転・局所化** | 項／係数の疎性、近似分解の構造、frame合成を変更。Pauli項・RZ/CXやnormの候補 | 原PRの軌道選択は既存監査に帰属。[Kronenbergerら][ORBITAL]は軌道・順序・PF誤差の関係を調査 | 同じactive space・sector・元物理Hのcanonical／適用可能な局所軌道、各基底で同等に最適化したDF/Pauli | integral→DFとbasis registry・参照検査を再利用。UWCのorbital予約接口は実装ではない | **高**。C1に組み込む補助自由度として有用。独立新機構は未特定 | exact rank不変性とthreshold疎性を混同。PF bias・stateの基底・scalar・初終端変換費が変わり、純compile比較に留まらない |
| **C4 既存compiler適用** | Pauli rotation列全体のparity network、Clifford抽出・再合成によるCX/depth削減。角度融合があればRZにも影響 | [Rustiq論文][RUSTPAPER]・[著者実装][RUSTSRC]、[QuCLEAR][QUCPAPER]・[著者実装][QUCSRC]。各benchmarkの改善はPR未検証 | 既存basis/boundary最適化＋位相保持Qiskit。追加対照に現行Qiskitの同等最適化機会。版更新効果と合成法効果を分離 | A2の固定ordered IR／中央control／wrapper／operator検査／compiler cacheへ小さいadapter | **中**。短期可否を判定しやすい。主価値は既知compilerの安全な適用と実装改善。compiler自作・独立算法claimは不要 | `upto_phase`・末尾Clifford省略、無断reorder、角度/Pauli endian、Qiskit版差、古典compile費、Tとの混同 |
| **C5 BLISS・対称性・grouping** | 対象sectorで意味を保ち、full-spaceの係数やspectral rangeを変更。可換group／反可換unitary partitionで辞書と実装費を交換 | [BLISS][BLISSPAPER]、SCDF、既存Phase2Cのgrouping／unitary partitioning対応 | 元H、θ=0、適用可能な最適化BLISS、同じgrouping・既知unitary partition、同じcompilerと誤差枠 | UWC・grouped比較・typed辞書／参照を再利用。ただしPauli経路とDF経路は別接口 | **中～高**。既知法の適用価値はあるが、旧実績不足と意味論の検証負担で当面後順位 | sector外作用・途中のsector漏れ、scalarのcontrol相対位相、finite-PF/RTE差、unitary生成費、旧prose値を成果として扱う危険 |

### B1. C1：X-DF、RC-DF、SCDFの違いが実装範囲を決める

[RC-DF一次本文][RCDF]のAppendix Aでは、X-DFの各対角係数行列はouter product、C-DF/RC-DFはそのrank-one制約を外した一般対称行列である。正則化fitの既知の利点は、近似精度と係数質量等の制御にある。**basis数減少と、PRで実行する中央ZZ回転の数・価格の減少は同じではない。**

現行 `DFBlock(U_ops,eta,lam)` とsquare専用抽出は一般行列の直接入口を持たない。既知の同frame内のsquare分解へ写す場合も、内部rank・fragment ID・同frameのまとめ方・prefixの意味が変わる。一般対角blockとして保持する場合は、既存のI/Z/ZZ primitiveとtyped atomの下流を再利用できる一方、係数抽出とHamiltonian参照接口の追加が必要になる。ここではdata表現の適合性を指摘するだけで、新しい分解・fit目的関数は設計しない。

[SCDF][SCDF]はRC-DFの単なる同義語ではない。本文§2.2/2.4・Appendix Bではfit中のrank-one制約とfragment別symmetry shiftを使い、shift後の対角行列は一般にrank twoになる。したがってsquare型SOURCEとの親和性がある部分と、shift・scalar・同frameの追加成分を処理すべき部分を分ける。qubitizationのToffoli利益を、PRのRZやshotの削減率に読み替えない。

利用可能な外部既存接口として、[ffsim 0.0.83の公式API][FFDF]は `diag_coulomb_mats` と `orbital_rotations`、最適化付きDF生成、非零位置指定、元分子Hamiltonianへの変換を公開している。これはgeneral diagonal表現・通常の圧縮fitの再利用候補であり、**そのままRC-DF／SCDFの正則化・対称性目的を実装済みとはしない**。APIのtensor許容差はPRのenergy／signal誤差保証ではない。SCDFの[著者配布アーカイブ][SCDFDATA]は存在を確認したが、download・展開・内容監査は未実施。

PR上の利益の経路は、主として「D側で反復するframe数」「中央回路の疎性」「R側で現れるframeとsupportの分布」「normalizationと統計負担」の交換である。RTEは辞書の全項を各shotで必ず実行する方式ではないため、総fragment数・総Pauli項数だけで回路長は決まらない。新しい分解で別のH近似・splitを作るなら、元のfinite targetとの一致を無条件に要求する比較と、同じ物理精度を要求する比較を区別する。

### B2. C2：残る余地は異frameと中央部を含む実際の列

同一frameの閉じる／開く対を消すこと、反復のD halfをまとめることは既済である。[ffsim公式SOURCE][FFTRANS]も連続するorbital frame間のrelative rotationを保持してTrotter列を作る。**異frame間transitionを直接合成すること自体を新原理とはしない。**

中央対角演算を跨いでGaussian演算を自由に動かすことはできない。basis非制御・中央制御の構成であっても、control0枝の全体がIになることとcontrol1枝の全ordered演算子を確認する必要がある。異frame列を安くするには、その列の中央部・相手frame・初終端条件まで含む差が必要だが、今回読んだ証拠には強いbaselineを超える一般的な正条件がない。

P-Aの限定的負結果、PS48の混在した費用差、Gaussian積closureで追加手順が未定義だった事情を区別する。後者の詳細は既存[NEXT]の帰属を継承し、今回独立に全実装を再監査したとはしない。singleton selectorを新名で再開する理由は得られなかった。

### B3. C3：単なる共通軌道回転でrankや内部transitionが減るとは限らない

以下は既知のunitary変換の性質とSOURCE構造からの**技術的推論**で、新計算結果ではない。完全なactive-space内で共通のunitary軌道変換を施すだけなら、exactな二体tensor行列のrankや、対応するone-body factorの固有値は変わらない。共通frameの変更だけではfragment間の相対basis関係も自動的には簡単にならない。

変わり得るのは、閾値を伴う疎近似、再fit・再grouping、qubit mapping上のPauli展開、実際の合成表現や初終端費用である。active-space自体を変える場合は別のモデル近似であり、単なる基底回転として扱わない。[軌道変換とPFの一次論文][ORBITAL]も、順序条件と軌道条件を分離する必要を示す関連資料であるが、その分子結果をDF-PRの保証には移さない。

したがって「localize→rank低下→総資源低下」という連鎖は採らない。C1に追加するなら、何の近似・どのframe費が変わるかを先に特定し、PF誤差係数Cの変化は純粋なcompiler改善と別の将来検証に戻す。旧P-Cのgeometry追跡結果も局所化の性能証拠ではない。

### B4. C4：Rustiqを先に、QuCLEARは抽出と吸収を分ける

外部SOURCEはGitHub connectorで固定commitの本文を取得し、実行せずに確認した。

| 実装・固定版 | 確認した接口・長所 | PRで使う条件／不足 |
|---|---|---|
| Rustiq `425000222e21a7a28d70a08c75be968f16fd6903`：README、`python/rustiq/__init__.py`、`src/interface.rs`、`examples/pauli_network_synthesis.py`、`utils.py` | `pauli_network_synthesis` は順序保持とcount/depth目標、`fix_clifford`を公開。`extract_rotations`も存在。個々の回転を別々に分解するより列全体の構造を扱える | 順序保持を明示し、回転角・符号・occurrence・末尾Cliffordを保持。raw接口のcheck／Clifford復元だけでglobal phase完全一致が保証されたとはしない。位相を落とす補助抽出や初期状態に依存する簡約を無条件に使わない |
| [現行QiskitのRustiq HLS API][IBMAPI] | `PauliEvolution.rustiq`、`preserve_order`、`upto_clifford`、`upto_phase`を公開 | 既存Qiskit1.3.0に同APIがあるとは確認していない。意味論として順序保持、末尾Clifford保持、phase保持が必要。高levelのH=sumを再Trotter化して固定ordered列を置換しない |
| QuCLEAR `b4e8684fc32f34a750510aa418d02114598df2ed`：README、`src/CE_module.py`、`src/CA_module.py`、requirements | `CE_recur_tree_seq`は各入力を単独order setとして扱う入口。抽出後回路と末尾Cliffordを返す。`CE_recur_tree`は可換set内の並替え、`update_observables`は観測量更新 | extraction経路の正しさ・位相はPRで未検証。末尾Cliffordを実際に復元するoperator-preserving比較を先に置く。requirementsはQiskit1.1.1等を指定し、既存1.3.0環境との互換は未確認。専用環境が必要 |

QuCLEARの**測定吸収による利益**は、量子回路と古典後処理を組み替える意味を持つ。systemの末尾Cliffordを、system上の通常observable測定と同様にcontrolled evolutionの内部から削除してよいことにはならない。全wrapper末尾へ合法に移し観測量を更新できる特別な構成の可能性まで否定しないが、それは別のcoherent-signal／測定意味論の検査を要する。短期候補は吸収を前提とせず、末尾Clifford込みの合成改善に限定する。

任意Gaussian gateを大きいPauli展開へ戻してからcompilerに入れる方針も最初から採らない。まず既存中央Pauli列や既にあるnative回転列の局所領域を対象にする。Gaussian部分をそのまま保つ対照と、抽出・再合成の古典費用を含めて比較する。文献benchmarkの最大削減率は一切PRへ転用しない。

### B5. C5：sector保存とfinite controlled演算子保存を分ける

[BLISS][BLISSPAPER]は対象blockの作用を保ちながらLCU分解等を改善する既知の前処理である。現UWCはその限定的な粒子数shiftで、一般BLISS全体の最適化ではない。full-space operatorは変わり、shiftが対象sectorで0ならそのsectorのexact evolutionは保たれるが、**再分解した個々のPF因子・RTE atomが各々そのsectorを保つとは限らない**。途中のsector外作用を含むfinite ordered product／coherent meanは改めて確認する必要がある。

既知scalar補正が必要な変換なら、energy原点だけでなくcontrol相対位相へ戻す。full-space同値を主張する試験と、対象sectorに制限した意味論を採用する試験を区別し、sector試験だけでF2のfull-space gateを通過したことにしない。

可換groupingは同時対角化の費用を、反可換unitary partitioningは群を実装する追加回転の費用を伴う。VQE測定数・group数・係数normの低下を、PRの生成回路・全shot資源の削減と同一視しない。既存Phase2C監査の「既知表現変更・cost-aware辞書の応用」という位置付けを保持する。旧UWC表は資料不足で成否不明であり、BLISS一般の失敗例にもしない。

### B6. THCの追加価値と今回の位置

RC-DF／SCDFの比較でTHCは有用な参照であり、さらに候補を増やすなら1件だけここに置く。compact tensor表現・qubitization資源の既知の利点がある一方、非直交・非unitary factorとPRのinvolution／controlled生成接口の接続はsquare DF以上に大きい。[RC-DF][RCDF]・[SCDF][SCDF]内の比較範囲を確認したに留まり、THC原論文と実装の独立調査は今回行っていない。C1の既存DF入力で改善が見込めない場合の再検討先とし、上位の軽い限定検証へは追加しない。

## C. 優先候補と、その位置付け

| 観点 | C1：圧縮DFのnative接続 | C4：Rustiqの順序・位相保持接続 |
|---|---|---|
| 改善機構の具体性 | frame往復数・対角疎性・tailの分布を変える既知の表現自由度がある。ただし一般係数行列の接口gapがある | 既存ordered rotation列のnetworkを直接対象にでき、Hamiltonian・sampling lawを変えずに検査できる |
| 既存研究との差 | 固定X-DFのL_D探索・S/P辞書選択とは自由度が異なる。RC-DF/SCDFそのものは既知 | 既存Qiskit＋basis/boundary最適化に対する追加の合成機会。Rustiq／QuCLEAR自体の再発明ではない |
| 再利用・負担 | tensor来歴、scalar、sector、抽出、D/R、wrapper・費用記録を再利用。fitと新H近似の検査が必要になり得る | fixed wrapper、phase修正、operator／signal検査を再利用。古いQiskitとのadapter・末尾Clifford／位相処理が主な負担 |
| 失敗時の情報価値 | 少数frameの利益がdense中央部・normalization・fit費で消えるか、そもそも既存型で表せないかを切り分けられる | 順序・位相を保持すると文献の利益が残らない、または変換費で消えることが分かれば早く区切れる |
| 現在の主張 | **中期の中心候補**。既知法のPR応用として推薦し、独立新規性は未確立 | **短期の技術確認候補**。最初はRustiq、QuCLEARは末尾Clifford保持の補助対照 |

C2はC1/C4双方のbaselineに含めるべき既知合成を多く含み、今回独立主題へ押し上げる具体差がなかった。C3は表現・PF条件への影響が広い。C5は旧数値の欠損とsector／finite演算子条件が残る。このため優先を下げるが、C3/C5の一般的な価値を否定しない。**この順序は検証価値の推奨であり、資源の性能順位ではない。**

既知法の通常適用だけでも、保存された来歴と公平なnative比較を伴う実装改善には価値がある。独立アルゴリズムを目指すなら、安い表現・生成・合成アクセスや、既存法を超える成立条件が必要であり、未定義の新機構を今回作って埋めない。

## D. 次段階の最小検証案（未承認・未実行）

### D1. C1：まず表現接口、次に1条件の圧縮DF対照

| 項目 | 提案内容 |
|---|---|
| 次に確認すること | **第1段階はread-only接口監査**：既存保存integral／H snapshot、X-DFの来歴と、候補実装の出力型・scalar・spin順序・error定義を対応付ける。ffsimの通常圧縮fitとRC-DF/SCDFを混同せず、実際に入手できる1方式だけを選ぶ。保存済み出力がなければ生成をせず欠測として返す |
| 最小の将来科学scope | 採択された場合に限り、既存linear H4・1.0 Å・STO-3Gの同一元integralを出発点とし、baseline X-DF rank12と既存の圧縮方式1件を比較する案。初期baselineはL_D=3、δ=0.02の一点・短いordered列。候補側rank policy、D/Rの意味、r/K、誤差枠と件数は実行前に固定し、旧P-A/F2の入力・grantを流用したことにしない。H4の結果を一般分子へ移送しない |
| 有望とする条件 | 元Hに対する採択済み近似誤差を満たし、scalar／operator／sector／control意味論が適合し、強いbaselineに対してactual full-wrapperの選定metricが改善する。basis／中央／law変化の区別と前処理費を残す。これは次段階へ進む局所証拠で、総資源勝利の認定ではない |
| 終了・後回し条件 | 出力をsquare型へ戻すことで圧縮利益を失う、同精度でframe減少を中央dense化が相殺、位相／scalar条件が閉じない、normだけが低下、または既存C-DFの通常適用以上の問いを定められない場合。C1全体のno-goへ一般化しない |
| 再利用範囲 | `DFHamiltonian`／元integral来歴、`df_trotter` Gaussian・diagonal primitive、registry・tail・partial/repeated-S2、typed wrapperと独立operator検査、既存cost記録。新fitterを自作せず既存方式を優先 |
| 新実装・build／compile・計算 | 接口監査だけなら不要。科学pilotには必要最小限のgeneral diagonal等のadapter、表現再構成・H近似検査、既存方式によるfit、短い回路build／compileが必要になり得る。新ground state・sampling・PF係数fitを必要なら別項目として明示し、自動追加しない |
| 別途承認 | ユーザー／研究C・統括チャットが方式と同精度の比較契約を採択。Codexへのadapter実装指示と、分解／再解析／build／compile等の科学実行承認を対象限定で別に与える。旧grant・STOPは解除しない |

このpilotでdecompositionを変えれば、旧finite targetと一般には一致しない。PF bias、finite-RTE誤差・normalization、実装誤差、統計予算を同じ物理taskへ戻して初めて総資源を比較できる。H近似のtensor残差やCCSD(T) energy heuristicだけでその予算を満たしたとはしない。

### D2. C4：固定された短いcontrolled列のRustiq適合性

| 項目 | 提案内容 |
|---|---|
| 次に確認すること | **第1段階はread-only接口監査**：固定Rustiqのordered-network出力、角度／符号のbinding、末尾Clifford・global phase、Qiskit adapterの互換を仕様として対応付ける。新compilerを設計せず、既存回路をimport／buildして確認する操作は科学承認後にする |
| 最小の将来科学scope | 既存の独立reference付き短い合成controlled wrapperから、非可換順序・負角度／product sign・identity相対位相・同frame／異frame境界を含む小さいcase集合を実行前に固定する案。Gaussian singleton方式選択を比較軸にしない。先にoperator検査、その後だけ同一wrapperをcompileして資源記録。既存caseでの確認はdevelopment testと呼ぶ |
| 有望とする条件 | 既存許容差でphase fitなしのfull operator、control0=I／control1=U、X/Y信号、必要ならactual adjointが一致し、固定lawを保つ。同じnative basis・routing・精度条件で、強いbaselineに対するCX/depthまたは事前選定metricの改善が、末尾Clifford込みでも残る |
| 終了・後回し条件 | 効果が非可換reorder／phase省略／Clifford吸収を要する、角度・符号・bit順序を安全に復元できない、同等Qiskit最適化で差が消える、あるいはcompile／IR変換費が用途に合わない場合。位相の問題を許容差緩和で通さない |
| 再利用範囲 | fixed `circuits.py`のordered IR・中央control・actual adjoint・wrapper、`gates.py`の位相保持lowering／matrix・signal検査とcache、2978版のbasis-plan／反復最適化。費用metric収集の再実装は不要 |
| 新実装・build／compile・計算 | 小さいadapterと回帰test、隔離環境での依存準備が必要になり得る。新Hamiltonian・新分解・新sampling lawは不要。実際のoperator評価、build／compile・測定は新科学実行として扱う。Qiskit更新が必要なら旧版対照、更新版のみの対照、更新版＋Rustiqを区別する |
| 別途承認 | ユーザーが限定adapter／環境準備と、固定caseのbuild／compile／operator・signal検査を別途明示承認する。QuCLEARの測定吸収、長列、全F2、認証N評価は含めない |

同じlaw・同じunitaryを保てれば統計的targetは変わらないが、実装誤差の扱いは維持する。F2のγが未成立のまま認証Nを算出してよい、という許可にはならない。新しい分布をsampleせず保存caseを使う場合も、その保存caseの読取り・build／compileの実行範囲を固定する。

### D3. 共通の費用・誤差条件

| 区分 | 記録し、混同しないもの |
|---|---|
| 表現 | 外側DF fragment数、内側rank／rank policy、非零対角係数、Pauli項数、basis数、L_Dと物理splitの定義 |
| normalization | Pauli l1、各文献のLCU λ、PR tail λ_R、finite normalization／attenuation。名前が同じλでも定義を照合 |
| 誤差 | 元物理Hの近似、PFの決定論的bias、finite-RTE truncation、有限bit／angle／compile実装誤差、QPE/RPE統計誤差を分離 |
| 回路 | RZ count、CX count、各depth、total depth、1-shot full-wrapper期待費用。準備を含むF2 scopeと主研究のno-prep scopeは別。非加法compile費から共通gate数を引いてscope換算しない |
| fault tolerant | T count／T-depth、Toffoli、合成精度とその誤差配分。RZ減少やCX減少だけでT利益を認定しない。synthesis前RZ数を実T countと呼ばない |
| 統計・総量子資源 | 必要shot数の保証条件、confidence、range、実装bias allowance。最終的に同task・同精度でshot×期待回路費を比較。今回新N・総資源順位は計算しない |
| 古典費用 | integral／fit・辞書生成・IR変換・価格取得・build／compile・samplingのsetupとonline、cache cold/warm、償却対象。同じ情報・cache機会を対照にも与える |

未来の重い処理は、承認されたscope内で既存timing・静的SOURCEから律速を見積もり、実際の空きCPU/GPU/RAMを調べて並列化・高速化する。CPU/RAM/worker・phase/total wallへ恣意的な低い上限を設けない。科学call・出力・log・instruction制限とmandatory STOPは維持し、共有環境・他ユーザーの設定やprocessを変更しない。これは将来の運用案で、今回の計算許可ではない。

## E. 取得範囲、未取得資料、残る判断

### E1. 外部一次資料の実確認範囲

| 資料 | 今回直接確認した範囲 |
|---|---|
| [Mottaら、DF][DFPAPER] | arXiv公式abstract／版情報。低rank化とtruncationの位置付け。詳細な資源定理の新規再検証なし |
| [Oumarouら、RC-DF][RCDF]、[出版社ページ][RCPUB] | 2212.07957v3の導入、Appendix AのX-DFと一般対称Zの違い・fit、近似／normのscope。文献数値の再現やPR倍率への変換なし |
| [Roccaら、SCDF][SCDF] | 2403.03502v1の§2.2–2.4、Appendix B関連箇所、結論・配布情報。rank制約とshiftを確認。ACS掲載本文は403、存在しないv2 HTMLは404。取得できたv1と出版社版を同一全文読了扱いにしない |
| [Kronenbergerら、軌道とPF][ORBITAL] | 2602.18913v1のabstract、軌道変換／順序依存の関連説明。全分子結果・全補足を再審査しない |
| [Rustiq論文][RUSTPAPER]・[固定実装][RUSTSRC]・[IBM API][IBMAPI] | 論文は公式abstract、実装はB4記載の固定file、APIはoption意味を確認。数学的正しさ・controlled位相・依存互換の実証はしていない |
| [QuCLEAR論文][QUCPAPER]・[固定実装][QUCSRC] | 2408.13316v2のClifford extraction／absorptionと順序の説明、B4記載のSOURCE。v3 HTMLは404。論文のbenchmark倍率をPRへ転用しない |
| [BLISS][BLISSPAPER] | 2304.13772v3の導入・対象block保持／LCU costの説明。旧UWC runの再現性とは別 |
| [ffsim API][FFDF]・[transition SOURCE][FFTRANS] | 表示版0.0.83のgeneral diagonal DF、fit optionsとrelative-basis Trotter実装。導入・import・実行はしていない |
| [既存Phase2A/2C][SOURCEGUIDE] | 原PR、grouping、unitary partitioning、SPRINT/GRADE等の対応は既存レビューを再利用。これら全一次文献を今回新たに全文取得したとはしない |

### E2. 未取得と未検証を区別する

- **未取得原本：** 旧UWC比較JSON、H3 sector/time-grid診断、theta sweepとrun来歴。既存[GAPS]を継承し、今回全filesystemや他hostを再探索していない。prose値から成功／失敗を決めない。
- **未展開資料：** SCDF配布アーカイブはmetadataのみ。RC-DF/SCDFをそのまま使える著者fit実装・保存分解出力の完全な再利用可否は未確定。ffsim通常fitの存在でその不足を埋めない。
- **未取得scope：** 他研究のprivate raw全件、原PR等の全版・全補足、THCの独立した実装・一次論文全体。今回の結論に不要な回収は広げていない。
- **未検証事項：** 圧縮DFが同じ誤差枠でnative PR回路を減らすか、Rustiq/QuCLEARが固定controlled位相を保つか、cold/online古典費を含む実益。資料を取得するだけでは科学的PASSにならない。
- **将来の科学判断：** C1の分解方式とrank/誤差契約、C4のmetric／compiler版、強いbaselineに与える同等の最適化機会、holdout単位。これらは今回正式採択しない。
- H12のPF係数は未決定のまま。`C_use`は実行delta窓の経験的包絡で厳密上界ではなく、H4/H6からH12へ係数を外挿しない。最終全round総資源評価の完了も主張しない。

## F. 保全、公開、引継ぎ、mandatory STOP

本書は候補選別用の新規Markdown **1件**であり、既存の研究概要・status・manifest・索引・他研究文書、SOURCE、固定raw／結果、grant、ledger、marker、STOPを更新しない。新しい研究決定や数値結果は作っていないため、共通文書へ本書の推奨を採択状態として反映しない。

完了確認は本書の文書・リンク・Git差分に限定する。起点との差分が本ファイルだけであること、既存tracked fileの変更がないことを確認し、専用branchへcommit／pushする。その後remote refと固定commitの本文を再取得し、Git blob・本文bytesを照合する。最終commit SHA・成果物URLと公開照合の成否は最終チャットに返す。

元workspaceのtracked差分と概要・status・manifest・共通指示のhashを技術的に照合する。保全の根拠は独立worktree・単一追加差分・対象への書込み不実施であり、他研究のprivate raw全件を再hashしたという主張ではない。

**次の担当と承認：**

1. **ユーザー／研究Cチャット・統括チャット**：C1/C4のどちらの限定確認を採るか、既知法の応用改善を目的として進めるかを判断する。本書はそのための材料で、正式採択を代行しない。
2. **Codex（別指示後）**：選ばれた候補のD1/D2のread-only接口監査、または別途承認された最小adapter準備だけを行う。必要なSOURCE・入力・比較契約・科学call・STOPを固定してから実行承認へ戻す。
3. **ユーザー**：新実装と、分解・保存値再解析・build／compile・simulation等の科学実行を対象ごとに明示承認する。旧grantや過去の実行契約を新候補へ流用しない。

**remote公開と本文照合を確認した時点でmandatory STOP。** 次機構の正式採択、新アルゴリズム設計・実装、回路build／compile、科学計算、Gaussian singleton選択研究の再開、他branchへのmergeへ自動進行しない。

## 固定参照

[CLOSE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/69f8e1cac23a4924ab81aebe399657a705024f6a/docs/research/pr_gaussian_singleton_closeout_handoff_20261024.md
[NEXT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7aa911bf929d8d1469d27749bf1d83132a36a00e/docs/research/pr_next_resource_mechanisms_survey_20261024.md
[LANDSCAPE]: pr_algorithm_improvement_landscape_20261023/RESEARCH_MAP.md
[SOURCEGUIDE]: pr_algorithm_improvement_phase2a_20261023/PRIMARY_SOURCE_GUIDE.md
[BASELINES]: pr_algorithm_improvement_phase2a_20261023/STRONG_BASELINE_AND_OVERLAP_MATRIX.md
[METHOD]: pr_algorithm_improvement_phase2c_design_20261023/METHOD_DELTA_AND_PRIOR_ART.md
[GAPS]: pr_algorithm_improvement_landscape_20261023/EVIDENCE_GAPS.md
[F1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b0dad8cc2f7a4cb821f53185c4a65f3a42e8f381/docs/research/pr_algorithm_improvement_phase2d_f1_rerun_20261024/EXECUTION_REPORT.md
[F2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dc7378ab34015287215bcf4ce52147e46d94ecd9/docs/research/pr_algorithm_improvement_phase2d_f2_resources_rerun_20261024/EXECUTION_REPORT.md
[F2JSON]: ../../artifacts/pr_algorithm_improvement_phase2d_f2_resources/2026-10-24-r1/RESULT_SUMMARY.json
[ATTR]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/41c4e79f1c58af0f2be634f46dbeba8dcd36acc5/docs/research/pr_algorithm_improvement_phase2d_f2_attribution_20261024.md
[PS48]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1e20458d742a8fac62ce0d0264188b1a7772ee28/docs/research/pr_rectangular_givens_ps48_execution_20261024.md
[PS48JSON]: ../../artifacts/pr_rectangular_givens/2026-10-24/RESULT_SUMMARY.json
[BUILD2978]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2978e2fea672b7a1ff20cac74269ec9a610159dc/src/trotterlib/df_rte_qiskit.py
[REPEAT2978]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2978e2fea672b7a1ff20cac74269ec9a610159dc/src/trotterlib/df_partial_s2_repeated.py
[PA]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/docs/research_direction_joint_synthesis_mechanism_validation.md
[PAPRE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/docs/research/pa_joint_synthesis_mechanism_validation_preregistration.md
[PAJSON]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/artifacts/research_direction_joint_synthesis_mechanism_validation/2026-09-25/pa_forced_support_order2_mechanism_validation_v1.json
[PASRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/src/trotterlib/research_direction_joint_synthesis_pilot.py
[PAMECH]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/src/trotterlib/research_direction_joint_synthesis_mechanism_validation.py
[DFPAPER]: https://arxiv.org/abs/1808.02625v2
[RCDF]: https://arxiv.org/html/2212.07957v3
[RCPUB]: https://quantum-journal.org/papers/q-2024-06-13-1371/
[SCDF]: https://arxiv.org/html/2403.03502v1
[SCDFDATA]: https://zenodo.org/records/10836682
[ORBITAL]: https://arxiv.org/html/2602.18913v1
[RUSTPAPER]: https://arxiv.org/abs/2404.03280
[RUSTSRC]: https://github.com/qiskit-community/rustiq/tree/425000222e21a7a28d70a08c75be968f16fd6903
[IBMAPI]: https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.transpiler.passes.synthesis.hls_plugins.PauliEvolutionSynthesisRustiq
[QUCPAPER]: https://arxiv.org/html/2408.13316v2
[QUCSRC]: https://github.com/revilooliver/QuCLEAR/tree/b4e8684fc32f34a750510aa418d02114598df2ed
[BLISSPAPER]: https://arxiv.org/html/2304.13772v3
[FFDF]: https://qiskit-community.github.io/ffsim/api/stubs/ffsim.DoubleFactorizedHamiltonian.html
[FFTRANS]: https://qiskit-community.github.io/ffsim/_modules/ffsim/qiskit/gates/double_factorized_trotter.html
