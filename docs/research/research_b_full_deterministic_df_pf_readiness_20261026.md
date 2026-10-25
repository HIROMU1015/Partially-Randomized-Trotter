# 研究B B-Next 4：Full deterministic DF-PFの再利用・実装・科学実行準備

2026-10-26 JST。起点commitは`58502f5549d350a8da8cbc23a6e674c4670feb9b`。
**TECHNICALLY_PREPARED／SCIENCE_NOT_EXECUTED／MANDATORY_STOP。**
科学SOURCE：[8811ef089d588a529aca001f9e40cef1b2f6e1c7][SOURCE]。

## 1. 目的と固定比較条件

Full Hamiltonianを決定論PFで近似する4 caseを、研究Bの強い対照として実行準備した。今回は旧SOURCE・関数・test・pipelineの静的確認、最小adapter、人工検証、read-only preflightまで。**実fixtureの新Hamiltonian／operator／signal、実basis／native回路生成、compile、費用評価、保存数値の新比較は行っていない。**

固定入力はsynthetic 2-mode、one-body＋3 DF因子、N=1／mode0 occupation、同じphysical Hとstate。T=±0.18、ordinary control、mode0準備・readout・measurement込みのX/Y full wrapper、native-v5。分子geometry／AO basis／分子DF rankの新設定は非該当。q=2は固定点比較であり同資源・最適精度調整済みの比較ではない。Full PFには内部n_Dを導入しない。

## 2. 既存資産の取得状況

**必要な公開結果・スクリプトは取得できた。** ただし起点checkoutには`resource_applicability/` package自体がなく、AX-2A／AX-2Bを直接importできなかった。指定公開commit `31cdff47f3282d2898c153a1af22f90e500be4c6` のgit objectから原本を取得し、必要な定義だけを研究B側の固定抜粋へ置いた。欠落を別の新PF実装で埋めていない。

| 確認した固定資産 | 実際に確認したもの | 今回の取扱い |
|---|---|---|
| [H2/H4/H6保存報告][H246]、commit8613b3ec | 条件・取得SOURCE、決定論部分集合、公開input provenance／CSVの存在 | 報告・schemaの静的確認だけ。費用値を現Bへ移送しない |
| [AX-2B native provider][OLDNATIVE]、commit31cdff47 | Full S2／Yoshida4、ordinary／directional、DF primitive、scalar、instruction bound | 9定義を本体変更なしで再利用 |
| [AX-2A control plan][OLDPLAN]、同commit | S2のforward/reverse、Yoshida4 signed3 piece、q反復 | 3定義を本体変更なしで再利用 |
| [AX-2B人工test][OLDTEST]、同commit | 複素Gaussian toy、独立occupation-energy参照、ordinary embedding | 5 helper定義を変更なしで再利用 |
| H4/H6取得SOURCE5a448d94 | [cost adapter][OLDCOST]、[native compose][OLDCOMPOSE]、signal adapter | 実際の呼出経路を静的確認 |
| H2取得SOURCE4ec26921 | [runtime][H2RUNTIME]とpipeline | H4/H6経路再利用とH2固有phase guardを確認 |
| 現B0／Exact-R／22行 | 同じfixture・CAR・state／sector、wrapper／v5、保存比較snapshot | 同条件接続に再利用 |

旧H4/H6の`h46_m5_native.compose`はsigned S2 pieceごとに`build_deterministic_native(...formula='2nd',scalar=0,control_policy='symmetric_directional')`を呼び、最後にscalarを一度付ける。H2はH4/H6の取得処理を再利用しつつ固有compiler guardを使う。**この取得pipelineが、今回参照した31cdff47の全provider closureとbitwise同一だったとは認定しない。** Private run全体を回収・再監査することは今回の必要条件にしていない。

## 3. 再利用できなかった部分と理由

旧package一式・診断stream fingerprint・取得launcher・grant管理は復元しない。Full PFに必要な旧関数が現在のDF primitiveを呼ぶ固定抜粋を使用し、依存importだけを接続した。旧SOURCEは変更していない。

旧`build_native_hadamard_wrapper`にはmode0準備がないため、現在のB0 prepared wrapperで置き換える。旧分子state、P=0費用、symmetric-directional、compiler位相guard、旧sampling／grant／STOPは新研究へ移さない。

旧stream fingerprintの代わりに現ライブラリの既存canonical circuit fingerprintをimportする。これは回路の生成式やphaseを変更せず、旧結果のfingerprint同一性を主張するものでもない。Stage moduleでは科学importを避けるため既存`_baselines`の同じS2/Yoshida係数とPython-intの固定scope検査へimportを接続する。

## 4. Full DF-PFの数学的定義

`H_phys = c0 I + H_one + λ0 F0² + λ1 F1² + λ2 F2²`。
Block順はone-body、DF0、DF1、DF2の4つ。全3因子と元の係数・符号・Gaussian basisを含め、truncationしない。

`H0=H_one, H1=λ0F0², H2=λ1F1², H3=λ2F2²`とすると、数学的S2は

`S2(h) = E0(h/2) E1(h/2) E2(h/2) E3(h) E2(h/2) E1(h/2) E0(h/2)`、`Ej(t)=exp(−it Hj)`。

Yoshida4は`S2(a h) S2(b h) S2(a h)`、`a=1/(2−2^(1/3))`、`b=−2^(1/3)/(2−2^(1/3))`。既存binary64列をそのまま使い、新しい係数探索・再正規化はしない。各macro時間はT/qで、q=2を反復する。

## 5. Scalar／identity／phase

Full fragmentを元のまま使うため、全体scalarは**元のc0だけ**を公称Tに対し一度付ける。旧分割用`c0+c_ext`、Exact-Rのstage別`+c_ext*t_R`をFull PFへ持ち込まない。

入力準備はB0を再利用し、共通H／sector／stateを旧snapshotへ結合する。そのD-only spec3個と、Exact-Rで使用した既存`prepare_tail` helperの**uncentered fragment spec**1個を合わせる。D/R metadataは入力同一性確認のためだけに残り、Full evolutionにD/R outer plan、RTE config、event、normalizationは使わない。

独立CAR側のfragment2は`R+c_ext I`として元へ戻し、`c0 I+Σ四block = H_phys`をspectral normで検査する。Native側はDF二乗primitiveそのものを用いる。この恒等式検査はscalar二重計上・factor欠落を検出する。Full-Fock4次元の構成を使い、N1の2×2指数を無料の回路primitiveにしない。

Control=0はidentity、control=1はscalarを含むPF作用。Gaussian basisは既存actual inverseを使い、global phase fitによる補正をしない。

## 6. S2／Yoshida4のstage規約

Controlled evolutionは旧`build_deterministic_native`を直接再利用する。旧control planのIRはforward UNCONTROLLED／reverse DIRECTIONALを表せるが、builderのordinary分岐は**全blockをORDINARYへ上書き**する。新結果のappend_scheduleにも実mode=ORDINARYとsource_plan_modeを分けて保存する。

旧providerは各S2で4 blockのhalf forward＋half reverseを保ち、中央の同block half2つも未融合。q2でS2は16 block描画、Yoshida4は48描画/case。非制御system側には既存`build_df_trotter_circuit`／`iter_pf_steps`を使い、中央・piece境界の既存融合を保持する。数学的なPFは同じだがliteral列は異なるため、独立参照に対して双方を検査する。

Native費用はcontrolled evolutionから作る8 full wrapperだけを測る。短い非制御列の価格をcontrolled価格に流用しない。旧instruction boundは実blockのbasis長と固定q／piece数から導出して渡し、恣意的なCPU／RAM上限にしない。

## 7. Prepared X/Yへの接続

非制御systemは既存DF-PF circuit builder、controlled systemは旧AX-2B builderを使用する。B0の`wrapper`／`wrapper_reference`／`expectation`／`compile`／`measure`を継承し、Mode0準備、ancilla2、cosine/sine終端、measurement1を揃える。

CompilerはQiskit1.3.0、basis rz/sx/x/cx、optimization1、seed17、approximation_degree1、qubits_initially_zero=False、free topology、phase保持native-v5。旧AX-2B wrapperや旧H2 compiler guardへ置き換えない。共有compiler設定・SessionEquivalenceLibraryは変更しない。

## 8. 最小adapterの差分

新規は[full_df.py](../../src/trottertracks/b2d_evaluator/full_df.py)、[入口](../../scripts/run_b2d_full_df_pf.py)、[schedule固定抜粋](../../src/trottertracks/b2d_evaluator/_full_df_schedule.py)、[native固定抜粋](../../src/trottertracks/b2d_evaluator/_full_df_native.py)、人工test2件。

新しい処理は4 case whitelist、元c0と全blockへの結合、独立CAR数学参照、既存prepared wrapperへの接続、scope登録に限定した。S2／Yoshida4の生成原理、DF二乗・basis・control primitive、native compiler／metric、fresh output／失敗保存を再実装していない。12定義をgit原本とAST比較して一致を確認した。

## 9. 人工検証

| 検証 | 件数 | 内容 | 結果 |
|---|---:|---|---|
| 新opaque／CLI／IO | 6 | Signed全block順、q/T/method/control/n_D拒否、scope、correctness先行、失敗・STOP、read-only | PASS |
| 新Full DF toy | 14 | 異なる複素Gaussian toy、one-body＋3DF、正負時間、全block、system/control、X/Y、c0二重計上・欠落検出 | PASS |
| 旧Exact-R opaque回帰 | 10 | 旧scope／renderer／IO／CLI／失敗条件 | PASS |
| 旧Exact-R toy回帰 | 9 | Basis／DF二乗／control／phase／prepared X/Y | PASS |

**計39件のローカル人工検査PASS。** 旧AX-2Bのtoy helper5定義も原本とAST一致。実fixture・保存数値の新評価、native transpile／metric／samplingはtestで使用していない。新toyではcompile、NPZ読取り、RTE configに失敗stubを置き、samplingは呼ばない。

実行したfocused testsは`tests/management/test_b2d_full_df.py`、`tests/management/test_b2d_exact_r.py`、`tests/test_b2d_full_df_toy.py`、`tests/test_b2d_exact_r_toy.py`。Stdlib suiteは`python3 -B -S`、toy suiteは既存venvの`python -I -B`とpytestのcache無効で実行した。B0／Exact-R／共通libraryは差分0。人工PASSは実候補correctness・費用優位の証拠ではない。

## 10. Read-only preflight

固定SOURCE専用worktree `/tmp/prt-bnext4-full-df-science-20261026` はclean。`--inspect`はstdlib metadataのみ、`--preflight`はSOURCE・fixture literal・旧22行snapshot・Exact-R比較snapshot・共通input identity・依存version・fresh pathを読取り確認する。

実preflightは**科学package importを拒否するbarrier下でPASS**。Status=`PREFLIGHT_ONLY_NO_SCIENCE`、予定outputは未作成。NumPy1.26.4／SciPy1.14.1／Qiskit1.3.0／Qiskit Nature0.7.2／OpenFermion1.6.1／OpenFermion-PySCF0.5／PySCF2.7.0。共有venvの変更・bitwise環境監査はしない。

Explicit `--execute`なしに科学処理を起動しない。Flagは誤起動防止で、許可の証明ではない。Certificate、approval ID、原文保存、receipt、packet、grantは要求しない。

## 11. 将来の4 caseと8 wrapper

| Case | Full方式 | q | T | Wrapper |
|---|---|---:|---:|---|
| FD1 | DF-S2 | 2 | +0.18 | Prepared X/Y |
| FD2 | DF-S2 | 2 | −0.18 | Prepared X/Y |
| FD3 | DF-Yoshida4 | 2 | +0.18 | Prepared X/Y |
| FD4 | DF-Yoshida4 | 2 | −0.18 | Prepared X/Y |

参照はB3/B4 finite-r2-K2の両Tと、[Exact-R E1〜E4][EXACT]。旧結果のセルをそのまま保存し、再計算・MC統計再集計をしない。対応physical signalの整合も確認する。

## 12. 独立correctnessと将来の評価

入力は同じHamiltonian／state／sector／bit mapping、全4 blockと元c0の恒等式を先行確認する。Block probeはGaussian／DF diagonal回路を、独立CAR行列の`expm(−itHj)`と比較する。Full参照はbuilderのstage列を辿らず、canonical S2数学積と既知3-piece Yoshida4を構成する。

System4×4、controlled8×8=diag(I,U)、明示H/Sdgとoccupation embeddingによるprepared wrapper、compiler後operatorとX/Y expectationを別々に検査する。全caseのgate・保存physical整合を通過した後だけ、signalと費用を解放する。

Precision complex128、operator spectral 2-norm／scalar絶対誤差、atol1e-12／rtol0／phase fitなし。主指標`physical_complex_coherent_signal_absolute_error`、Full-Fock／N1 operator誤差、X/Y・半径を別保存。NativeはRZ/CX count、RZ/CX depth、total depth、circuit sizeを測る。Measurement count／quantum-only depthは既存metricの診断値として残す。

## 13. 科学call・出力・STOP

| 登録境界 | 件数 | 意味 |
|---|---:|---|
| fixture／split／prepare／full_inputs／input_gate | 各1 | 既存B0入力closure。分割は入力結合だけ |
| full_prepare／definition_gate | 各1 | 追加fragment2のGaussian1、全H／c0恒等式 |
| block build／block gate | 各48 | 6 signed時間×4 block×control有無。Probeはcache共有 |
| 独立exponential | 32 | CAR block・canonical中心時間・physical±Tを共有 |
| system／controlled／reference | 各4 | 旧Full builderとlibrary system builderを使用 |
| system gate／controlled gate | 各4 | 独立数学積との比較 |
| wrapper build／reference | 各8 | 4 case×X/Y |
| raw gate／expectation | 各8 | Phase・control・X/Y |
| compile／compiled gate／expectation | 各8 | 同じnative-v5 |
| saved comparisons | 1 | 保存finite-RTE4＋Exact-R4、physical整合 |
| signals／measure | 4／8 | 全correctness通過後のみ |

224登録境界、72 operator検査、16 expectation、最大event480。Probe cacheとFull builder内部のblock描画は別で、後者がprobe circuitを再利用すると偽らない。小さい固定4 block/q2の展開を各build境界に含める。旧Evidenceの総16 MiB／JSON1 MiB／event64 KiB／STOP reserve256 KiB／最大9 fileを再利用する。

専用fresh outputを排他的作成。`binding.json`、`inputs.json`、`events.jsonl`、`partial_results.json`、成功時`result.json`、`status.json`、`STOP.json`にSOURCE・入力参照・stage／scalar／basis来歴・数値条件・全gate・全結果・実績を保存する。失敗時には例外・部分case・数値差、operator FAIL時の既存matrix／QPY診断も残す。

H／state／scalar不一致、gate FAIL、識別不能、例外、保存失敗、scope未完了でSTOP。許容差／precision／条件の緩和や自動retry／resumeをしない。起動直前のCPU affinity・load・MemAvailableを記録し、既存cache・vectorized小行列・native-v5を利用する。最初のFAIL後の追加compileを避けるcase順を維持し、不要な大規模worker基盤や恣意的なCPU／RAM／worker／wall capを追加しない。

## 14. 固定SOURCE・保全・公開

科学SOURCEは`8811ef089d588a529aca001f9e40cef1b2f6e1c7`。Branchは`codex/research-b-full-df-pf-ready-20261026`、作業worktreeは`/tmp/prt-research-b-full-df-pf-ready-20261026`。本書・索引の後続commitは実装を変えない。

旧PR／B0／B1／Exact-R／shotのSOURCE・raw・契約・grant／ledger／marker／STOP、R2-CS終了、AGENTS.md、共通指示、元dirty workspace、共有環境は変更しない。新科学結果を得ていないので旧科学本文・manifestの資格も変更しない。公開後にremote SHA／SOURCE／成果物を照合してmandatory STOPする。

## 15. 科学実行許可後の単一command

**未実行。ユーザーがこの固定SOURCE・4 caseの科学実行を別途明示的に許可した後だけCodexが起動する。**

```bash
/tmp/pr-phase2df1pre-private-20261023/venv/bin/python -I -B \
  /tmp/prt-bnext4-full-df-science-20261026/scripts/run_b2d_full_df_pf.py \
  --execute \
  --source-commit 8811ef089d588a529aca001f9e40cef1b2f6e1c7 \
  --output /tmp/prt-bnext4-full-df-science-20261026/artifacts/research_b_outer_full_df_pf/20261026/full-df-once
```

Read-only確認は`--execute`を`--preflight`へ置換する。特別な承認ファイル保存や端末操作は不要。

## 16. 残る問題と次の判断

技術的blockerは現在見つかっていない。実候補のfull-H identity／PF／controlled／compiled correctness、signal精度、native単価はすべて未検証であり、preflightや人工PASSで代替しない。

q2固定点は方式別最小資源や同精度最適比較ではない。S0条件付きRTE単価・B0代表を母平均にせず、未観測rare費用を0に置かない。Shot/G、A1／γ、精度保証、全RPE、実分子一般化を評価しない。独立算法研究の継続・終了、詳細科学レビューを自動採択・開始しない。

次担当は研究Bの**ユーザー**で、固定4 caseの実行可否を判断する。**Codex**は許可後だけ上記commandを一回実行し、完了／FAIL／識別不能／技術停止でmandatory STOPする。

[SOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/8811ef089d588a529aca001f9e40cef1b2f6e1c7
[H246]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8613b3ecda2030a677b1690fe7aa3da23dec4873/docs/research/track_a_h246_saved_cost_results_20261021.md
[OLDNATIVE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/31cdff47f3282d2898c153a1af22f90e500be4c6/src/trottertracks/resource_applicability/ax2b_native_df_v5.py
[OLDPLAN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/31cdff47f3282d2898c153a1af22f90e500be4c6/src/trottertracks/resource_applicability/ax2a_control_plan.py
[OLDTEST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/31cdff47f3282d2898c153a1af22f90e500be4c6/tests/tracks/resource_applicability/test_ax2b_native_df_v5.py
[OLDCOST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5a448d94ea389ada6e2b2755de6a215ed2034679/scripts/h46_m5_cost_native.py
[OLDCOMPOSE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5a448d94ea389ada6e2b2755de6a215ed2034679/scripts/h46_m5_native.py
[H2RUNTIME]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4ec26921c6872ed6e8667f167074e1e405cf1a3f/scripts/h2_outer32_runtime.py
[EXACT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/58502f5549d350a8da8cbc23a6e674c4670feb9b/docs/research/research_b_bnext3_exact_r_execution_20261026.md
