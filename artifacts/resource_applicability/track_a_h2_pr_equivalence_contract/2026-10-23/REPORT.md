# H2 PR代替同等性契約：静的構造検証・runtime依存性・終了監査

**134/134軸で保存QPYの全instructionを構造grammarへ結合できた。代替契約Aの12条件中11、Bの3条件中2が全134軸でVERIFIED。独立した共通diagonal角度証拠とcross-runtime費用同等性はUNRESOLVED。現行strict認定0/134、代替契約の正式認定0/134、COMPAREはBLOCKED。科学実行なし、未採択DRAFTのままmandatory STOP。**

基準 `588d19544a2afdfb55ec4b40475a58114bd47b0b`、専用branch `codex/h2-pr-equivalence-contract-20261023`。技術実装SOURCEは[SOURCE.json](SOURCE.json)のimmutable commit。現行COMPARE SOURCE `46a3b33e3717c4fadf11846e922caf17d10156d9` と契約digest `a65f3e90b7ed70701f57bcf1519e1d74acdfd1c45c320a5fe91d00d4d44d9ef2` を変更していない。

対象はlinear H2/1 Å/STO-3G/tol-only DF1e-8/rank3、T=.8、ε_sig=.0005、α_axis=.025、P=0、system4/ancilla4。Full S2 `B1_p3_q4_2nd`/q4、Full Yoshida4 `B1_p3_q2_4th`/q2はprefix3。PR S2 `H2O32_c7ec590a32a859bc`/q4、PR Yoshida4 `H2O32_8a2ecdd0294ffb74`/q2はprefix1/K4/outer32。元RTE law、symmetric-directional、usual signed bias、経験的allowance、旧FP reference条件付き政策、元cohort/費用/shot政策を維持する。新しい総費用・u/N/G・順位の結論はない。

## 原証拠とSOURCE

GPUA100-gleap02で、指定されたO00002原本を直接読んだ。`/home/AbeHiromu/projects/h2-higher-pf-outer32-execution/20261021/private/authorization_run02/stagepacks/H2_PR_EXPLORATION/config_APPROVED_PRIVATE.json`、1,331,184 byte、期待SHA256 `f011e12b5df6c534fd610899abd2f0f8a17dd05e1cfefaf418a7cb09c0c9a8c7` が完全一致。

[再回収監査](recovery_recheck.json)で前回の固定request2,554件を再照合した。metadata原本2,085・archive/raw402・全文event67が全て一致。原eventは6,400 event/6,404 application/336 signed piece。旧67 trajectoryのS2 33/66軸、Y4 34/68軸を保持し、private config/event/seed/QPY/raw/archiveをコピーしていない。保存vectorは必要なarchive SHAまたはNPZ headerまでで、数値payloadをdecode・評価していない。

[科学SOURCEの16依存](SOURCE_joins.json)も現在の原fileで封印SHAを再確認した。新しい検査SOURCEと人工testsのみ追加し、旧runner/library/科学SOURCEは無変更。

## 代替契約：AとBを分離

[代替契約案](alternative_contract_DRAFT.json)は`H2_PR_ALTERNATIVE_EQUIVALENCE_DRAFT_V1`。正式採用・科学実行承認・grant/ledger/marker/run_rootはいずれもない。各条件に必要証拠、受入基準、既存資料、追加静的検査、科学計算との境界、未確認時のSTOPを記載した。

Aは意図した科学回路へのconformance、Bは原compiled費用を新Full費用と比較できること。**Aの成立をBの成立として扱わない。** 現行完全byte identityへのattestationではなく、旧strict契約を保持した独立DRAFTである。

| 分類 | 全134軸で成立した条件 | 残る条件 |
|---|---|---|
| A：科学的回路conformance | input/prep/cohort binding、event/law、Gaussian forward/inverse、basis reuse、diagonal形/mirror、central/control、relative phase、scalar/wrapper、全instruction消費、serialized global phase、歴史的single-state correctnessの11条件 | 共通preparationに対するabsolute deterministic diagonal角度の独立binding |
| B：費用測定の同等性 | 原compiled raw/費用receiptのidentity、compiler SOURCE/settings/phase guardの2条件 | compilerに関係するruntime差の出力・費用同等性 |
| 宣言scope外 | all-state/operator保証、新runtimeでの新statevector correctnessはNOT_APPLICABLEとして明示 | historical acceptanceの拡張に使わない |

A全条件充足0/134、B全条件充足0/134。各13個の個別条件は全134軸でVERIFIED、2個は全134軸でUNRESOLVED、2個はNOT_APPLICABLE。今回の回路field検査でMISMATCHとなった軸は0。これは未確認項目の同等性を認定する意味ではない。

[134軸の条件表CSV](axis_conditions_134.csv)、[JSON判定](axis_evidence_134.json)、[67 trajectory表](task_evidence_67.json)、[各軸の技術証拠packet](operation_stream_evidence_134.json)を保存した。条件のevidence_sha256はその軸packetのcanonical JSON digest。preflightはmetadataのschema・固定cohort・scopeを検査するもので、未採択packetや人為的なVERIFIEDラベルを科学実行権限に変える入口はない。

## 全operation streamの静的検査

既存`h2_pr_saved_qpy.py`、原event/SHA検査を再利用し、`h2_pr_stream_conformance.py`に保存instructionを順番に消費するcursorを追加した。Qiskit等をimportせず、実H2回路object・diagonal primitive・rotation・matrix・native loweringを生成しない。既存パラメータの符号/hex/wire/順序の関係を検査する。

- Gaussian：共通receiptの6登録basisと全event記述を結合。固定Qiskit1.3 SOURCEのPhase/XXPlusYY inverse規則に従い、保存gateの逆順・第一角度の符号・第二角度・wireを厳密照合する。
- basis reuse：各pieceの先頭/末尾、同一basis継続、basis切替、identity flushを原SOURCEの状態遷移に結合。余分・不足・順序違いを拒否する。
- deterministic diagonal：固定H2のnonzero pattern（one-body 4 RZ、fragment 6 RZZ＋4 RZ）を読み、対応するdirectional CX/RZ列の全support・符号・順序と一致を確認する。両側の角度が同じだけ変化した場合、absolute bindingは未証明のままになることも人工testで確認した。
- event/control：全保存eventのcentral操作、product、control metadata、controlled RZZのbase/body/control state/global phaseを照合する。
- 位相：event/product/identity位相を元SOURCEの順序で集約し、保存ancilla PhaseGateとbinary64で完全照合。deterministic側の保存ancilla RZからの位相関係と保存global phaseも134/134でexact match。新しいHamiltonian由来の位相を求めたものではない。
- wrapper：one scalar、cosine/sineのH/U/Sdg/H/measure、ancilla4とclbit0、全instructionの消費を確認。同じ保存evolution digestを要求し、sineはSdg 1個だけ多い。

「保存streamの全操作をgrammarへ割当てた」と「共通Hamiltonian/preparationから独立に計算された全角度が同じ」を区別する。後者のdiagonal値は元QPYには存在するが、共通DIAGNOSTICSの独立target listはない。[既存bundleの保存scope](common_saved_receipt_scope.json)で、2 prep JSON/4診断JSON/4 NPZのkey/headerを確認した。共通receiptはGaussian operationsを保存し、NPZはprimary/independent vectorとbasis indexで、独立diagonal primitive receiptは回収できなかった。数値vectorからの新逆算は行わない。

初回auditはcosine/sineのinstruction数まで同一と仮定して技術STOPした。[元技術STOP](attempts/attempt01/technical_stop.json)とSOURCE/人工ログを保持。Sdg 1個を考慮するpair検査と回帰2件を追加して修正した。原科学証拠のMISMATCHではない。

## runtime差とcompiler費用への関係

[静的SOURCE/import/ELF調査](runtime_relevance.json)ではSOURCE11ファイルとELF4個をdataとして読んだ。[状態別runtime表](runtime_conditions.json)。`readelf -d`のDT_NEEDEDは直接依存だけであり、実load観測やtransitive非依存の証明とはしない。

| 差/一致 | native生成への関係 | compiled費用への関係と未確認 |
|---|---|---|
| Python3.12/3.11 | interpreter/extension ABI・numeric orchestration | 版差のみで不等価とはしない。seed固定や同一SOURCEだけでも出力同一とは認定しない |
| NumPy ABI/build config | preparation/diagonal/Gaussian数値処理。保存Gaussian field一致は既存証拠で確認済み | Qiskit numeric synthesis/array経路に関係し得る。新common diagonalやcounterfactual compileの差は未評価 |
| SymEngine ABI | numeric final QPYだけでは無関係と判断できない | scalar guardはParameter/exact_zero_scalarを作り、ParameterExpressionはSymEngineをimportする。元runtime内でのguard同一stream保証を新ABIへ外挿しない |
| Qiskit1.3.0 830ファイル一致 | inverse/control/serialized gate semanticsとRust acceleration byte一致 | 強いSOURCE証拠だが、transitive library/numeric input/ABI全体の出力感度は未証明 |
| liblzmaの歴史的seal差 | 調査したcompiler/native pathに直接の関係は未確認 | 検査したQiskit/NumPy/SymEngine ELFの直接NEEDEDにlzmaはなく、stdlib _lzmaにはある。過去workerで未使用とは断定しない。zlib archive/raw SHA一致は保存証拠の保全を支える |

歴史的inventoryは完全worker実load manifestではない。部分的version/origin/完了receiptを証拠として保持し、歴史的loadを新しい再現結果から作らない。既知byte差はruntime表でMISMATCH、科学的な出力・費用への影響はUNRESOLVEDと分けた。追加の全134軸compileを自動で必要とは判定していない。

## correctnessのscope

[保証範囲](correctness_scope.json)。134個の元CT01 compiled-vector receiptのSHA、PASS、strict <1e-7、phase_alignment=falseを元resultと結合した。native/occupation/Hadamard <1e-9の元保存scopeも保持する。今回errorの数値を再評価せず、複数状態・all-state/operator・新runtimeのcorrectnessへ昇格しない。静的gate関係の一致も、compiled operator全状態保証とは異なる。

## metadata-only preflightと必要な追加作業

[代替preflight](alternative_preflight.json)と[CLI一致](alternative_preflight_CLI.json)は `BLOCKED_ALTERNATIVE_UNADOPTED`、execution_ready=false、science_authorized=false。[旧strict preflight](strict_preflight_preserved.json)も `BLOCKED_PR_IDENTITY` のまま。代替契約をtrueにする、scope/閾値や旧SOURCE/digestを書き換える、不完全な証拠を渡す、未承認executeを試す操作は拒否する。科学backendやlive grantは接続していない。

[最小追加案](minimal_additional_validation_DRAFT.json)は以下の二つを分離する。**今回は計画のみで、いずれも未実装・未承認・未実行。**

Aの不足には、prefix1 preparation 1回と、保存済み3種類のsigned half-time×2 deterministic blockのprimitive receipt 6件だけを独立取得する案を示した。bootstrap1/setup1/prepare1/diagonal_primitive_evaluation6/archived_axis_parameter_compare134、build/lowering/compile/simulation/sampling/会計は0。134比較は既存軸のparameter field検査であり、134新回路ではない。独立input/prep/Gaussian/diagonal hex/phase一致が受入条件で、失敗時はSTOP。これだけでBやcompiled全状態保証は満たさない。

Bの出力感度にはS2 sample0とY4 sample4（保存order2も含む）の2軌道4軸を候補にする。新H2 buildやrequest復元をせず、元wrapper QPYをそのまま新runtimeでcompileする案。bootstrap1/saved_input_state_load1/archived_wrapper_QPY_load4/compile4/transpile4/simulate8/saved_receipt_RZ_equality_check4、setup/prepare/lowering/wrapper/sampling/会計/順位は0。元SHA/settings/guardの不変、選んだ4軸の保存/new RZのexact equalityとsaved-state strict <1e-7が条件。2代表が成功しても全134軸の費用同等性や過去実loadを証明せず、自動拡張しない。

両案とも新しい別SOURCE、内部callを含むexact契約、人工検証、private input/runtime seal、新しい未使用grant/rootとユーザーの別途明示承認が必要。現SOURCEには実行入口がない。科学callの真の最小数が確定したとは主張しない。

追加科学計算なしで現DRAFTを満たせるとは認定できない。保存compiled回路自体を費用対象とするより限定的な研究上の比較政策を選ぶ可能性はあるが、その科学的意味とruntime差をどこまで許すかは別判断である。[実行承認前の残条件](execution_readiness_conditions.json)に整理した。

## テスト・保全・終了

[248人工tests PASS](test_results.json)、[実行記録](technical_tests.txt)。新48は原SHA改変、全instructionの欠落/wire変更、inverse/reuse/diagonal/phase/control/scalar、runtime未知、cohort/候補/receipt混入、correctness scope/閾値拡張、契約変更、不完全packet、未承認executeを検査。既存200のfile bytesは不変で、その同じ作業内のPASSログを保持して再利用した。初回246 PASSと最終新48を二重加算していない。Codex主担当の人工技術検査であり、独立科学レビューではない。

[保全監査](preservation_audit.json)は16,804保護pathを前回監査・今回開始・終了で照合し変化0。元SOURCE/input/runtime、原結果、旧契約/consumed grant/ledger/marker/STOPを保持した。liblzmaの過去seal差は今回の変更と区別する。共用venv/OS/scheduler/他job・他worktreeに書き込まず、恣意的な資源上限を追加しない。元compiler契約のnum_processes1等は変更しない。

[最終監査](final_verification.json)と[技術成果物inventory](file_inventory.json)で旧科学namespace/manifestと新成果物を照合する。これは最終科学evidence freezeではない。

次の担当ユーザー：原PRの歴史的費用と新Full費用をどのruntime政策・correctness scopeで比較するか科学的判断。担当Codex：別作業で必要な最小witness SOURCE/人工検証/封印を準備。担当ユーザー：その具体的計画の科学実行と、代替契約の正式採用をそれぞれ明示承認。担当Codex：全受入条件が成立し正式採用された後だけ、別SOURCEのCOMPARE接続・metadata preflightを準備。COMPARE本実行はさらに別の明示承認が必要。

[mandatory STOP](STOP.json)。新sampling/signal/simulation/実H2 build/native lowering/compile/transpile/u/N/G/費用再解析/順位/科学import/live grant/詳細科学レビュー/最終evidence freezeは全0。添付指示の公開範囲別承認条件を保持し、今回はローカルcommitのみ、pushなしで終了する。
