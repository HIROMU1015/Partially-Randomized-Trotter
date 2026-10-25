# 研究B B-Next 3：Exact-R最小pilotの実装・人工検証・科学実行準備

2026-10-26 JST。Repository：`HIROMU1015/Partially-Randomized-Trotter`。
作業起点：[4a8ef63081011c890d68bfd1cab78515a3d01f02][BASE]。
**状態：TECHNICALLY_PREPARED／SCIENCE_NOT_EXECUTED／MANDATORY_STOP。**

## 1. 目的と今回の到達点

単一DF残差を直接指数化するExact-Rを、既存B0のD列・prepared Hadamard wrapper・native-v5へ接続した。既存finite-RTEのSOURCE／挙動／結果は変更していない。新しい実装はadapter、薄い入口、必要な人工テストに限定する。

新人工19件と旧B0人工回帰11件の計30件PASS。固定SOURCEでread-only preflightもPASSした。**実fixtureの新operator／signal評価、実basis／native回路生成、compile、費用測定、保存数値の新比較は実施していない。** 科学的correctness・識別力・費用優位は未検証である。

科学SOURCE：[e76c0474806dca42598fa92a6564cded10896f98][SOURCE]。本書と索引の後続commitは実装を変更しない。実行にはこのSOURCE専用のclean worktreeを使用する。

## 2. 固定条件と4 case

既存synthetic 2-mode、one-body＋3 DF因子、N=1／mode0 occupation、L_D=2 weight-ranked prefix。Dはone-body＋因子0,1、Rは因子2のidentity分離後。D内部はA2 Yoshida4、n_D=1、q=2、T=±0.18。分子geometry／AO basis／分子DF rank探索は非該当。

| 新case | Outer／D内部 | Tail | T | 対応する保存finite-RTE |
|---|---|---|---|---|
| E1 | Outer Yoshida4／A2 | Exact-R | +0.18 | B3／+0.18、r2 K2 |
| E2 | Outer Yoshida4／A2 | Exact-R | −0.18 | B3／−0.18、r2 K2 |
| E3 | SABA2／A2 | Exact-R | +0.18 | B4／+0.18、r2 K2 |
| E4 | SABA2／A2 | Exact-R | −0.18 | B4／−0.18、r2 K2 |

候補／時間／mode／内部PF／q／n_Dの変更は拒否する。既存4 caseは[22行の固定結果][R22]を参照し、再実行しない。Full deterministic DF-PF、n_D探索、他fixture・sector・時間、PF係数探索は含めない。

## 3. 数学的定義・basis・phase

`F2 = Σ_pq g2[p,q] a†p aq`、`H_fragment = λ2 F2²`、`R = H_fragment − c_ext I`。
実装すべき指数は `exp(−itR) = exp(+it c_ext) exp(−it H_fragment)` である。λ2は既存表示では+0.22だが、adapterは表示値を独立SOURCEにせず、固定fixtureのliteralと既存input identityから取得する。

既存`build_df_blocks(df_hamiltonian_to_model(h.select_blocks((2,))))`で因子2のGaussian blockを作り、`DFDeterministicFragmentSpec`へ結合する。既存`append_block_evolution`がactual inverse → DF squared diagonal → forwardを描画する。基底の転置・共役規約を新たに定義し直さない。

二乗対角項は既存の `Σ ηj² nj + 2 Σj<k ηj ηk nj nk` のI/Z/ZZ primitiveを使用する。N=1で二体項が消える特殊性に依存した2×2 oracleではなく、**4次元Full Fockの物理primitive**である。古典CAR/expmは独立correctness参照であり、その価格を無料の量子回路価格へ置換しない。

各R stageのsigned時間をそのまま渡し、補正phaseは`+c_ext*t_R`。非制御ならglobal phase、制御ならancillaのP gateにする。Basis往復は非制御で行い、control=0では対角作用がidentityとなって往復が相殺する。

全体scalarは既存B0と同じ`c0+c_ext`を公称Tに対し一度だけ`−(c0+c_ext)T`として付ける。R内の正補正と二重計上しない。列の係数を再正規化せず、実signed R時間和とappend順を結果へ保存する。Phase fitting／出力operatorからの位相合わせは行わない。

Exact-R descriptorにRTEのr/K・event・attenuationはない。`tail_mode="exact-r"`を固定し、normalizationは1。旧outer planはD/Rの順序・時間を継承するためだけに使い、R slotの旧RTE payloadをconfig／event生成へ渡さない。

## 4. 再利用したSOURCEと新規差分

| 既存資産 | 再利用する範囲 | 今回の差分 |
|---|---|---|
| [B0 Native][B0CODE] | Fixture literal読取り、DF preparation、CAR参照、snapshot identity gate、D block描画 | SubclassでR描画・参照・結果収集だけ追加 |
| [outer.py][OUTER] | A2、Yoshida4／SABA2、signed順序、D境界の既存融合 | Case whitelistとExact-R tail差替え |
| [df_partial_s2.py][DF] | FragmentSpec、basis actual inverse、arbitrary signed block描画 | D-only preparationを変更せず残差因子のspecを別に追加 |
| [df_trotter/ops.py][OPS] | `build_df_blocks`、DF二乗primitive、control／phase | 新しいGaussian分解・回転合成を作らない |
| [df_rte_tail.py][TAIL] | 既存identity extraction・basis registry | RTE config／event sampling／finite momentsは起動しない |
| [Hadamard helper][WRAPPER] | Mode0準備込みordinary controlled X/Y、measurement | 旧wrapperと同じ関数を使用 |
| [native-v5 compiler][COMPILER]／既存metric | `compiled_wrapper(...stable_phase=True)`、compile cache、RZ/CX/depth/size | Compilerや共有equivalence設定を変更しない |
| [followup_io.py][IO] | SOURCE／snapshot metadata、scope recorder、排他的fresh output、失敗・partial・STOP保存 | Studyを新namespaceに結合するだけ |

`df_trotter/circuit.py`の非制御full PF builderは今回のtail差替えには不要だった。公開AX-2B provider `31cdff47…`も静的に照合したが、旧wrapperは状態準備・symmetric-directional等の範囲が現B0と異なるため流用せず、現在のB0 ordinary-control経路を採用した。[B-Next 1 §6/7][BNEXT1]

新規ファイルは[exact_r.py][ADAPTER]、[run_b2d_exact_r.py][RUNNER]、[opaque tests][MOCKTEST]、[toy tests][TOYTEST]。独立参照を追加した理由は、旧B0参照がRTE event列を参照するためである。既存D/outer係数を使いながら、RはGaussian回路とは独立のCAR行列指数で評価する。

## 5. 人工検証

| 検証 | 件数 | 入力・実行内容 | 結果 |
|---|---:|---|---|
| 新opaque／CLI／IO | 10 | 科学import barrier下のstub、4 caseのscope・D/R順序・signed phase・拒否・失敗保存・STOP | PASS |
| 新複素toy | 9 | 固定fixtureと異なる2-mode入力、負λ、複素Hermitian g、非zero identity。CAR/expmとQiskit Operatorで照合 | PASS |
| 旧B0人工回帰 | 11 | 既存mock renderer／scope／representative／wrapper／failure tests | PASS |

Toyのtail時間は±0.137、λ=−0.37、c_ext=0.081。固定fixtureは呼ばない。Full composition検査は固定schedule metadataだけを使い、DもRも独立toyへ置換した。ToyでのQiskit回路生成と小行列評価は人工検証に限定し、native transpile／費用関数とExact-R内のRTE config／event描画を失敗stubに置いて未使用を確認した。

検査したのは、複素basisのactual inverse、二乗primitive、signed時間、control0 identity／control1の相対位相、identity補正符号、scalar二重計上検出、数学積とappend積、準備込みcosine／sineの符号である。Normはspectral 2-norm、atol=1e-12、phase fitなし。

Opaque testsでは全correctnessをsignal／費用より先行させ、9種類の入力／gate／compile中断に対して費用0・partial診断・STOPが残ることを確認した。Source不一致・候補/T/mode変更・重複call・同じoutputへの再起動も拒否する。Read-only入口に承認証明用の入力はない。

実行コマンドは以下。科学suite全体は起動していない。

```bash
/usr/bin/python3 -B -S tests/management/test_b2d_exact_r.py -v
/usr/bin/python3 -B -S tests/management/test_b2d_b0_native.py -v
/tmp/pr-phase2df1pre-private-20261023/venv/bin/python -I -B -c \
  "import sys; sys.path.insert(0, '/tmp/prt-research-b-bnext3-exact-r-ready-20261026/src'); import pytest; raise SystemExit(pytest.main(['-q','-p','no:cacheprovider','tests/test_b2d_exact_r_toy.py']))"
```

**人工PASSは実fixtureのExact-R correctness・小さいsignal差の識別・費用優位を証明しない。**

## 6. 将来の独立correctness gate

| 対象 | 回路側 | 独立参照／基準 |
|---|---|---|
| 入力 | 同じfixture・DF split・preparation | 22行の固定H／sector／state／tail identity、N1 snapshot、Full-Fock CARのH恒等式 |
| 非可換性 | 同じD/R | CAR行列のDR commutator有限かつnorm>1e-12。不成立ならSTOP |
| R単独 | Gaussian＋二乗対角＋identity phase | 元g／λから作るFull-Fock CARの`expm(−itR)`、全signed unique時間 |
| Controlled R | 同じprimitiveの制御 | 明示8×8のdiag(I4,U_R)、off-diagonal branchと位相も含む全operator |
| Full outer | 既存D append列とExact-R | 既存係数の未融合数学積、Dは独立CAR block指数、Rは上記指数 |
| Prepared X/Y | 既存wrapper | 明示bit embedding、H/Sdgとmode0準備による独立8×8参照 |
| Compile後 | Phase-preserving v5 | Rawと同じ独立wrapper operator、およびancilla-Z expectation |
| 保存ideal-tail | 回路から独立に作ったfull参照 | 保存22行のideal_tail／physical signalとatol照合。新回路oracleの代替にはしない |

Precisionはcomplex128。OperatorはFull-Fock spectral 2-norm、expectation／複素scalarは絶対誤差、atol1e-12、rtol0、phase fitなし。全case／axisのgateと保存signal整合が通過するまで性能比較・単価測定を開始しない。

H定義・分割・state不一致、非可換性不成立、gate FAIL、保存signal不一致、数値識別不能、例外、保存失敗、scope未完了はいずれもSTOP。許容差・precision・候補を緩和せず、自動retry／resumeしない。誤差や順位が小さく見えることを科学的GOへ変換しない。

## 7. Signal・費用比較の意味

主指標は`physical_complex_coherent_signal_absolute_error`。Exact-Rのsignalはnormalization1のまま、同じphysical H／mode0 stateの`exp(−iHT)`へ比較する。X、Y、半径、それぞれの直接誤差も保存する。D内部とouterの近似誤差は残る。

Operator誤差はFull-FockとN1 sectorを別フィールドで保存する。旧22行のoperator値はsectorであり、Full-Fock値と同じ量として比較しない。保存finite-RTEのcorrected／raw／ideal_tail／physical outputsは原値のまま対応付け、新しいfinite-RTE meanやnormalizationを計算しない。

Nativeは同じprepared X/Y full wrapperでRZ count、CX count、RZ depth、CX depth、total depth、circuit sizeを保存する。既存metricのmeasurement count／quantum-only depthも診断として残る。Qiskit1.3.0、native basis rz/sx/x/cx、optimization1、seed17、approximation_degree1、qubits_initially_zero=False、backend/target/topology/layout/routing指定なし、native-v5を固定する。

今回準備した新単価は決定論的Exact-R wrapperの単価である。[B0][B0]の代表費用は期待値ではなく、[B1][B1]の既存推定はS0条件付きでrareが未観測だった。B1のsampleを再集計せず、本pilotで新しい費用比・母平均順位・shot/Gを計算しない。旧[shot評価][SHOT]の資格不足も変わらない。

## 8. 固定科学call scope

`scope()`はmetadataだけで構成し、各callのidentity・回数を既存FollowupLedgerへ登録する。新しいauthorization frameworkは作らない。

| 境界 | 登録数 | 主な内部処理 |
|---|---:|---|
| fixture／split／prepare／full_inputs／input_gate | 各1 | 既存B0入力closure。Prepare内はone-body Gaussian1＋全DF3＋tail1 |
| tail_prepare／tail_identity_gate | 各1 | 因子2の追加Gaussian1、FragmentSpec結合／定義・非可換性 |
| D block build | 84 | Signed block/controlキーで共有。4 caseのliteral描画を重複生成しない |
| Exact-R tail build／tail gate | 各12 | 6 unique signed時間×非制御／制御 |
| 独立exponential | 50 | D／R／physicalの4×4 CAR行列指数、共有cache |
| system build／controlled build／reference | 各4 | E1〜E4のみ |
| system gate／controlled gate | 各4 | Full operator照合 |
| wrapper build／wrapper reference | 各8 | X/Y各1/case |
| raw wrapper gate／raw expectation | 各8 | 独立参照への照合 |
| compile／compiled gate／compiled expectation | 各8 | Native-v5と位相保持の確認 |
| saved baseline／signals | 各4 | 保存4 case整合／新Exact-R直接誤差 |
| measure | 8 | 全gate後の決定論的wrapper単価 |

Operator gate総数36、expectation16。境界は入れ子のclosureを含むので、各内部操作数や派生metricを独立job数として加算しない。Scope最大event記録546、旧Evidenceの必要十分な出力枠（総16 MiB、JSON1 MiB、event64 KiB、STOP reserve256 KiB、最大9 file）を再利用する。

Scope外は新sampling、RTE event生成、finite-RTE再評価、追加MC、rare費用、shot数／G、A1包含／γ、全DF-PF、n_D／係数探索、分子計算、全RPE、Stage自動移行である。R2-CSの再開も含めない。

## 9. 出力・失敗保全・一回実行

専用fresh outputを排他的に作成し、既存outputには一切追記・上書きしない。将来の出力は`binding.json`、`inputs.json`、`events.jsonl`、`partial_results.json`、成功時`result.json`、`status.json`、`STOP.json`。旧snapshotのinputs.npz等は固定commit／blob参照をbindingへ保存し、新しい承認ファイルを要求しない。

結果にはSOURCE、入力binding、case／append順・signed時間、scalar／tail basis／η／λ、compiler／依存version、全gate値、X/Y／複素signal／operator誤差、8 wrapper単価、cache実績を保存する。Partialは進行caseと失敗境界・例外・診断を残す。Operator FAIL時は既存B0のactual/reference行列とQPY診断も保持する。

失敗・未完了・保存失敗でもSTOPし、再起動・resume・新namespaceでの自動再試行は行わない。新しい技術修正や再実行が必要なら、その状況を返す。本準備はその追加実行許可を与えない。

## 10. 実行前確認と資源方針

固定SOURCE専用worktree `/tmp/prt-bnext3-exact-r-science-20261026` はclean。`--inspect`はstdlib metadataのみ、`--preflight`はSOURCE・固定snapshot bytes／条件・依存version・fresh pathを読取り確認する。両者とも科学処理やoutput作成を起動しない。明示的`--execute`なしには実行しない。このflagは誤起動防止であり、ユーザー許可の証明ではない。

SOURCE e76c04748の実preflightは**科学package importを拒否するbarrier付きでPASS**し、status=`PREFLIGHT_ONLY_NO_SCIENCE`、予定output未作成を確認した。依存はNumPy1.26.4／SciPy1.14.1／Qiskit1.3.0／Qiskit Nature0.7.2／OpenFermion1.6.1／OpenFermion-PySCF0.5／PySCF2.7.0。共有venvを変更していない。

将来の起動直前にaffinity CPU、load、MemAvailableを再取得しbindingへ保存する。4小行列case／8 wrapperでは既存signed block・exponential・compile cacheとnative-v5を使用する。最初のFAILで後続compileを開始しない順序を保ち、不要な大規模worker機構を追加しない。CPU／RAM／worker／wall-timeに人工的な上限を設けず、速度倍率も主張しない。共有環境・他ユーザーの設定やjobは変更しない。

## 11. 実行前に残ることと判断境界

技術的な未解決blockerは現在確認されていない。ただし実fixtureのbasis／Exact-R operator／full wrapper／compiled operatorは未検証で、complex128の微小差の識別力も未資格。これらは将来の限定科学実行で初めて確認する。人工PASSやpreflight PASSをその代替にしない。

特にatol1e-12は実装correctnessの受理値であり、signal誤差のu_num上界・γ保証ではない。丸め水準の順位や改善を断定せず、識別不能ならそのまま記録してSTOPする。Exact-Rが安い・高精度・総資源優位という結論を本書で出さない。独立算法研究の継続採択や新しい詳細科学レビューも行わない。

## 12. 許可後に使用する一つの実行command

**以下は未実行。ユーザーが固定SOURCEと4 caseの科学実行を明示的に許可した後だけCodexが起動する。**

```bash
/tmp/pr-phase2df1pre-private-20261023/venv/bin/python -I -B \
  /tmp/prt-bnext3-exact-r-science-20261026/scripts/run_b2d_exact_r.py \
  --execute \
  --source-commit e76c0474806dca42598fa92a6564cded10896f98 \
  --output /tmp/prt-bnext3-exact-r-science-20261026/artifacts/research_b_outer_exact_r/20261026/exact-r-once
```

Read-only確認は同じcommandの`--execute`を`--preflight`へ置換する。別途approval ID、原文保存、certificate、receipt、packet、grant、特別な端末操作は不要。

## 13. 保全・公開・次の担当

Branch：`codex/research-b-bnext3-exact-r-ready-20261026`。独立作業worktree：`/tmp/prt-research-b-bnext3-exact-r-ready-20261026`。新実装・テストと本書・関連索引だけをcommit/pushし、remote SHA・公開内容を照合する。

旧SOURCE、raw結果、科学契約、grant／ledger／marker／STOP、R2-CS branch、AGENTS.md、共通指示、元dirty workspaceは変更していない。旧4報告の最新状態は[統合][INTEGRATED]および以下の固定版を参照する。

公開・照合後は**mandatory STOP**。次担当は**ユーザー**で、このSOURCEのExact-R4 caseを実行するか判断する。**Codex**は明示的な科学実行許可後だけ上記commandを一回起動し、成功／FAIL／識別不能／技術停止でSTOPする。研究統括への送信や他の研究段階へ自動進行しない。

[BASE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/4a8ef63081011c890d68bfd1cab78515a3d01f02
[SOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/e76c0474806dca42598fa92a6564cded10896f98
[BNEXT1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4a8ef63081011c890d68bfd1cab78515a3d01f02/docs/research/research_b_bnext1_prior_art_and_strong_baselines_20261026.md
[INTEGRATED]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3cfbac1d4f7df751e972170e54ad21aa947ceff1/docs/research/research_b_integrated_outcomes_and_future_directions_20261025.md
[R22]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/33715848bfb83d49ec4e54d6eef481012a7441e4/docs/research/research_b_outer_rte_22row_execution_20261025.md
[B0]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0db9b3ed189eb6cd9d66e070993dbd0e72d29104/docs/research/research_b_outer_b0_recovery_20261025.md
[B1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bef63edbd8cc564444b44e4e8cf23baa90f05fe5/docs/research/research_b_outer_b1_expected_native_cost_20261025.md
[SHOT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e80cf902cbacbfa3d120d630c8d7ad06d5d53a6d/docs/research/research_b_outer_shot_weighted_execution_20261025.md
[B0CODE]: ../../src/trottertracks/b2d_evaluator/b0_native.py
[OUTER]: ../../src/trottertracks/b2d_evaluator/outer.py
[DF]: ../../src/trotterlib/df_partial_s2.py
[OPS]: ../../src/trotterlib/df_trotter/ops.py
[TAIL]: ../../src/trotterlib/df_rte_tail.py
[WRAPPER]: ../../src/trotterlib/rpe_hadamard_interrogation.py
[COMPILER]: ../../src/trottertracks/pr_dictionary_validation/gates.py
[IO]: ../../src/trottertracks/b2d_evaluator/followup_io.py
[ADAPTER]: ../../src/trottertracks/b2d_evaluator/exact_r.py
[RUNNER]: ../../scripts/run_b2d_exact_r.py
[MOCKTEST]: ../../tests/management/test_b2d_exact_r.py
[TOYTEST]: ../../tests/test_b2d_exact_r_toy.py
