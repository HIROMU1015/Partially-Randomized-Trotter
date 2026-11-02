# H2 PR原証拠回収・identity判定・COMPARE準備

**保存PR単価の現行契約での再利用成立は0/134軸。原証拠は67 trajectory・134軸とも回収できたが、厳格identityは未成立のためCOMPAREはBLOCKED。新規科学実行なしでmandatory STOP。**

専用branch `codex/h2-pr-original-identity-20261023`。基準commit `e7a63b2b06a124f9305483aa39bcf1ba8cafdd0d`、技術監査SOURCE `9d413d7fb9cbe302ff064a4078ff2a747a8f3db2`。[実行SOURCEのbyte結合](SOURCE_binding.json)。旧COMPARE SOURCE `46a3b33e3717c4fadf11846e922caf17d10156d9`、契約digest `a65f3e90b7ed70701f57bcf1519e1d74acdfd1c45c320a5fe91d00d4d44d9ef2`は変更していない。

対象はlinear H2/隣接1 Å/STO-3G/tol-only DF1e-8/rank3、T=.8、ε_sig=.0005、α_axis=.025、P=0、system4/ancilla4。Full S2 `B1_p3_q4_2nd` とFull Yoshida4 `B1_p3_q2_4th` はprefix3。PR S2 `H2O32_c7ec590a32a859bc`/q4とPR Yoshida4 `H2O32_8a2ecdd0294ffb74`/q2はprefix1/K4/outer32。共通QR native・経験的allowance・旧FP reference条件付き再利用の既存政策を保持。今回は費用・誤差・順位の新しい科学的数値結論を出していない。

## 原本と回収

`GPUA100-gleap02`上でユーザー指定のprivate領域を直接読んだ。起点は `/home/AbeHiromu/projects/h2-higher-pf-outer32-execution/20261021/private/authorization_run02`。その `stagepacks/H2_PR_EXPLORATION/config_APPROVED_PRIVATE.json` は1,331,184 byte、SHA256 `f011e12b5df6c534fd610899abd2f0f8a17dd05e1cfefaf418a7cb09c0c9a8c7` が完全一致。補助O00001も期待SHAと一致した。[回収集計](locator_and_recovery_summary.json)。原本をコピーしていない。

固定Gitから既存回収toolと受領metadataを再利用し、指定された2,554ファイルを照合した。[全回収結果](recovery_verified.json)はmetadata原本2,085件、圧縮/伸長raw SHA一致402件、全文event内容一致67件。未取得0、対象requestのbyte/content不一致0。archiveはwrapper QPY134・compiled QPY134・保存vector NPZ134。vectorsは圧縮/伸長SHAだけ確認し、数値配列としてdecodeしていない。

原event全文は67ファイル/6,400 event/336 signed piece。6,404 component applicationの順序・component・phaseを含むcontentを既存resultのdigestへ結合した。全文やseedはGitへ保存していない。[67 event判定](event_67.json)。

初回の[recovery.json](recovery.json)では67 eventを `EVENT_COMPONENTS` として誤拒否した。旧toolがcomponent IDを整数と仮定していたためで、固定SOURCEのRTEEventは非空文字列を要求する。toolと人工fixtureを修正し、同じ原本で全67件一致を確認した。原eventの不一致として数えていない。

## 134軸のidentity

[全134軸判定](axis_identity_134.json)、[67 trajectory判定](task_identity_67.json)、[既存16受入fieldの判定](identity_field_matrix.json)を保存した。S2は33 trajectory/66軸、Yoshida4は34 trajectory/68軸。元sample/cohortとcosine/sineを保持し、平均・u/N/G・費用・順位を再評価していない。

| 証拠 | 今回成立したこと | 限界 |
|---|---|---|
| H/state/DF/prefix/preparation/time/law | 原sealed input/resultと既存DIAGNOSTICSのbinding、prefix1 prep SHA、signed scheduleに一致 | setup/preparationの再計算なし |
| 原event | 全文のdigest/schema/順序/component/phase、6,400 eventの結合 | 新sampling・RNG replayなし |
| Gaussian rotation | 全6,404 applicationの登録basis記述をdecimal repr→同一binary64 hexで共通receiptと照合。134 wrapperでdeterministic forward群のgate/hex/wire一致 | 全inverse・basis-reuse境界・deterministic diagonal・global phaseを含む完全stream証明とはしない |
| central/control | 原eventから指定されるcentral操作の順序・support・符号付き角度・controlを全134保存wrapperと照合。保存controlled RZZ定義/base/bodyも確認 | 新common PR回路の生成なし |
| scalar/wrapper | scalarは共通診断と一致、最後のancilla PhaseGateの角度・H/Sdg/H/measurementとwireを照合。cosine/sineの保存evolution構造digest一致 | operator行列の評価・phase fitなし |
| compiled raw/単価 | 全134 raw SHAを原manifest→cost→accepted resultへ結合。保存RZ/CX/sizeとの一致だけを検査。134 guardとcompiled QPY内metadataも一致 | 新しい費用の推定・平均・再compileなし |
| compiler/fingerprint | guard/settingsと主要SOURCEの既存byte join、原fingerprint receiptを原rawへ結合 | Qiskitのlazy gate定義を生成するrecursive fingerprint再計算はしない。新しいdata構造digestとは区別 |
| correctness | 134軸の原CT01/SINGLE_SAVED_INPUT_STATE・strict <1e-7受入receiptを回収、保存resultとの一致を確認 | 数値誤差を再評価せず、全状態/operator証明や新しい閾値へ昇格しない |
| runtime | 下記の実原本・現採択環境・固定SOURCEを照合 | 完全な歴史的worker実load identityは未成立 |

QPYは新規のstdlib parserで保存binaryをdataとして読んだ。Qiskit/numpy等をimportせず、回路object・gate定義・inverse・matrixを生成しない。QPY v13/Qiskit1.3.0のnumeric endian差と、controlled custom gateのbase headerを固定schemaとして扱い、未対応layout/condition/calibration等を拒否する。これは新回路生成や科学実行ではない。

## runtimeで判明した差と不明点

[原runtime封印と対応ファイル](runtime_compiler_identity.json)、[実使用証拠・ABI差の補足](runtime_evidence_supplement.json)。元runtimeの2,012封印項目は現在2,011一致、1差分。差分は `liblzma.so.5.4.5` で、保存SHA `696e868d…` に対し現SHA `26b37d25…`。今回の前後hashは同一であり、今回の作業による変更ではない。元workerがこのlibraryを実loadしたか、その差がcostに関係するかは確認できない。全zlib archive/rawの照合は成功しており、この差だけで特定axisの回路不一致を主張しない。共有libraryを修復・変更していない。

元と採択環境はともに固定SOURCE上NumPy1.26.4/Qiskit1.3.0。対応するQiskit830ファイルは830/830 byte一致。NumPyは同名408一致、`__config__.py` 1差分、ABI名が異なる19 binary。SymEngineは34一致、ABI名が異なる1 binary。cp312→cp311の対応binary20件は全て実在しbyte差がある。Python実行binaryも3.12と3.11で異なる。これは版名だけの不一致判定より具体的な証拠であり、科学的不等価性の証明ではない。

原SOURCEの`install`はnumpy/qiskitのversion・封印済みorigin・provider originを検査する。これは実使用の部分証拠である。一方、封印inventoryのruntime/imported役割はpre-stage templateから継承されており、全workerの実module/native-library load観測とは異なる。指定runのprewarm/fresh gate/completion/task completion/end audit/compiler audit/supervisor entered/exitを確認したが、完全なworker別load manifestは回収できなかった。native logは0 byte。現時点のinventoryやSOURCE一致を、過去の完全観測記録として作り直さない。

したがって、**保存artifactと確認対象fieldの結合は134/134軸で成立、現行strict再利用条件は0/134軸、確認範囲内で特定された科学的回路不一致は0軸**。残る不明は完全stream conformanceと実runtime/correctness scopeの新系列への結合であり、「不一致がないから完全identity」とは判定しない。

## COMPAREと代替契約案

[metadata-only preflight](COMPARE_preflight.json)は封印済み旧launcher/SOURCEをそのまま使い、technical binding確認後 `BLOCKED_PR_IDENTITY`、execution_ready=false。完全candidate witness2件は未作成。O00002発見だけでBLOCKEDを解除していない。

[新契約の準備案](COMPARE_contract_DRAFT_BLOCKED.json)は非実行のDRAFT。既存科学条件・候補・政策・exact call予算を写した準備資料であり、実行可能schemaの新契約やgrantではない。将来のCOMPARE予算はbootstrap1/setup1/prepare2/bundle_load1、Full native_validation2/compose2/lowering10、wrapper4/compile4/transpile4/simulate12、allowance4/accounting4/ranking1。diagnose/path_signal/reference_rebind/sampling/request_factory/rte_event/occupation_trajectoryは各0。実消費は全0。

現行条件を維持する限り、異なるruntime byteと過去の観測欠測を新規compileで修復することはできない。[代替契約案](alternative_equivalence_contract_DRAFT.json)は、原compiled回路をPR費用の固定対象とし、Hamiltonian/preparation/event/control/scalar等の科学的同等性と、出力へ関係するcompiler/runtime closureの証拠を要求する別政策。全inverse/reuse/diagonal/global-phaseの構造conformance、ABI差の影響、原single-state correctness scopeを明示する必要がある。保存回路をcanonical cost objectにできるかは研究上のユーザー判断であり、今回は採択・実装・受入条件変更をしていない。

## 最小追加検証と承認

[最小追加案](minimal_additional_validation_DRAFT.json)。今回の証拠から全134軸の再compileが必要とは言えず、科学call0の別政策・構造証明を先に検討する。追加科学callの必要最小数や全cohortを保証する範囲は未確定で、欠測を「0件で解決」とも扱わない。

runtimeの出力感度probeが別途必要と判断された場合だけ、S2 sample0とYoshida4 sample4の2保存軌道/4軸を候補にする。後者は保存order2の経路も含む。192 event/194 component application、bootstrap1/setup1/prepare1/archived_request_load2/event_metadata_deserialize192/occupation_trajectory2、native_validation2/compose2/lowering10/wrapper4/compile4/transpile4/simulate12、sampling/request_factory/診断/path/reference/allowance/会計/順位は0。RNGなし復元の新SOURCE・人工検証・exact契約と別承認が必要な未実装DRAFTである。2例の成功では全67 taskのidentityや当時のloadを証明できない。probe後に自動で全軸へ広げない。

[次に必要な判断](next_approvals.json)：担当ユーザーがstrict条件維持か別の同等性政策かを選ぶ。担当Codexが別作業でその条件の証拠・SOURCE・人工検証を完成させる。追加科学probeが必要なら担当ユーザーが具体的対象/新SOURCE/exact予算/unused grantについて別途承認する。完全受入とmetadata preflight成立後だけ、担当ユーザーがCOMPAREの科学実行を別途承認する。GitHub公開も範囲の別承認が必要。現時点で科学実行承認や公開承認を求める段階へ自動移行しない。

## テストと保全

[人工技術テスト200件PASS](test_results.json)、[端末完了summary](technical_tests.txt)。新26件はnumeric QPY/schema/拒否、control/base/body、wire/angle/order、wrapper/scalar、Gaussian exact match、科学importなし等を検査。既存174件もPASS。実分子回路を作るテストではなく、Codex主担当の技術チェックであり独立科学レビューではない。

[原本の前後保全](preservation_audit.json)は16,804 pathで変化0。元config/event/archive/結果、SOURCE/input/runtime、DIAGNOSTICS/COMPAREの実使用seal、旧grant/ledger/marker/STOPを保持した。元DIAGNOSTICS SOURCE35444654…、契約f71e52ce…、completion SHA `cc7ebfee979b2e34eb4327aa8a8c6e20ea53bc46871f97c83d0391db5e7ec90f` は不変。過去sealとのliblzma差と、今回の前後保全を区別する。

[終了監査](final_verification.json)と[新成果物inventory](file_inventory.json)で固定SOURCEと成果物SHA/リンク、旧manifest内容、変更範囲、非実行状態を確認する。原本・共有環境・他worktree/jobを変更せず、恣意的な資源上限を加えていない。旧compiler契約内のnum_processes等は歴史的固定設定であり、新規の一般資源制限とは扱わない。

[STOP](STOP.json)：新sampling/signal/simulation/回路生成/compile/u/N/G/費用再評価/順位/科学import/live grant/詳細科学レビュー/最終freeze/pushは全0。専用branchへlocal commitし、mandatory STOP。
