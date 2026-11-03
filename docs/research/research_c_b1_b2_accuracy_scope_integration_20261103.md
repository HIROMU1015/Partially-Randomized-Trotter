# 研究C：B1/B2統合schema・精度評価の残件（B2公開未確認）

2026-11-03。**独立して進められる統合準備は完了。B2結果に依存する統合判定は保留した。**
開始時のGitHub branch検索と`git ls-remote`では、予定B2 branchの公開を観測しなかった。
B2の完了状態・固定commit・SOURCE・契約・STOPはUNKNOWNであり、その数値評価や実装を重複実施していない。
バックグラウンドの待機・監視も行わず、今回の公開後mandatory STOPする。

[検証scopeの比較・推奨案](research_c_accuracy_validation_scope_options_20261103.md)を判断資料として用意した。
新しい保証水準・sampling model・科学的主張は採択していない。新実H4計算、raw再解析、Stage B/C、重要科学レビューは0。

## 固定参照と公開状態

- B1結果 `69c9a764aee66cf2ca79ab29bd3145a027385bcc`／[Whole-P0報告](research_c_p0_uniform_method_readiness_20261103.md)。
  実装SOURCE `5fc9e6a18e016bfa0725d18edd1c7fc4589a0075`、[未承認設計](research_c_p0_uniform_method_contract_unapproved_20261103.json)、
  [SOURCE/STOP](../../artifacts/research_c_p0_uniform_method/2026-11-03/technical/)。Adapter GO、一様誤差未成立、非実行、252 PASSの資格を保持。
- Stage A結果 `ae9a4e13d1ab6f48fc4d38cbbcd0fddca30e9c53`／[38-call結果](research_c_iid_grid_stage_a_execution_20261103.md)。
  科学SOURCE `ddd8333488e07a3b3af9095b5cb193ef9f90b3c3`、[実行SOURCE/契約/STOP](../../artifacts/research_c_iid_grid_all_law/2026-11-03/stageA-run01/)。再実行0。
- 既存all-law adapterの固定参照 `3a784888a188db946681a6143c5e18479ddf284a`、IID-grid接続は上記Stage Aまでの既存SOURCE。
- B2予定branch：`codex/research-c-reference-numerical-uncertainty-readiness-20261103`。
  [開始時の公開照会](../../artifacts/research_c_b1_b2_scope_integration/2026-11-03/technical/publication_check.json)は検索結果0・exact remote head一致0、照会自体は成功。
  「未観測」を未完了・不存在の確定と同一視せず、別名の未観測成果や将来更新は推測しない。

B1/Stage Aの固定公開本文を取得し、blob identityをローカル固定資料と照合した。
正式共通指示原本 `research_project_instructions_unified_2026-10-25.md` の§8/§8.4/§10を確認した。
科学SOURCE・raw・契約・grant・ledger・marker・STOPと元dirty checkoutを保全する。

## 共通の科学targetとcoverage

対象endpoint SHA256：`a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`。
Linear H4・隣接1.00 Å/STO-3G、Nα=Nβ=2、sector36、number CDF rank8、元frame順LD2、partial ordered symmetric S2、
s/r/K=1/32/0、h=δ=.1、m0..12、CA=.00159 Ha、DF reserve=CA/8、own-DF ground oracle、状態準備費用除外。
保存P0/Qiskit1.3.0/opt1/seed17/rz,sx,x,cx、MPFR160 declared phase、既存Gaussian/relative fusion、
actual inverse、controlled full512、head/X/Y shell込みfenced serialを変更しない。

モデルは`IID_UNIFORM_53_BIT_WORDS_COUNTERFACTUAL`。各event・coherent occurrence・measurement shotで独立な一様53-bitを仮定する。
実PCG64 joint streamの数学的IID保証とは区別し、その差を0にしない。

| 項目 | 統合で確認する内容 |
|---|---|
| Candidate/SOURCE | Endpoint、元PR科学SOURCE、Stage A科学SOURCE/結果を一致させる。B1/B2の実装commitは役割別に保存し、互いが同一commitであることは要求しない |
| 条件・政策 | s/r/K/h、m0..12、compiler/native/phase、CA/DF、α/β政策とmodelを一致させる |
| Norm | 四つの追加誤差はraw complex signal。Operator2/max-entry/Haをそのまま同じslotへ入れない |
| Native空間 | Synthesisは最終whole-P0/full512、actual inverseと準備H/X/Y readout shellを覆う。決定論half-headはkernel内 |
| Reference空間 | Bridge/uncertaintyは同じown-DF/sector36 referenceへのraw signal boundとして整理。Sector証拠をnative full512へ昇格しない |
| Categorical | Same exact trajectory mapに対するIID-grid lawの信号差。特定sectorや実PCG64の保証と同一視しない |
| Scope | Native全word、固定有限family、確率的期待値を別に記録。Referenceはmodel内の全roundという別domain |
| Qualification | 観測、precision sensitivity、estimate、qualified numerical bound、rigorous enclosureを区別。Schemaは証明の正しさを認証しない |

## 誤差partitionと現在の状態

[機械可読partition](../../artifacts/research_c_b1_b2_scope_integration/2026-11-03/technical/current_error_partition.json)は保存fieldの引用とmetadata分類だけで、新しいnorm/TV/accuracyを計算していない。

| 成分 | 現在の分類 | 加算・残件 |
|---|---|---|
| PF | 保存numerical pilot取得済み | 既存reference項としてA倍。Law/referenceへ重複加算しない |
| Corrected finite-RTE | 保存numerical pilot取得済み | 同上。新規再評価0 |
| Synthesis | 方法のみ／上界未成立 | B1構造上限・composerはあるが、実H4の全domain誤差はnull |
| Categorical p→q | IID-grid条件付き取得済み | Stage Aの正確な格子TVとcoupling。実stream保証ではない |
| Reference bridge | 数値観測のみ | Saved-atom平均とCARの対応を観測。観測距離を上界へ変更しない |
| Reference numerical uncertainty | UNKNOWN・B2公開待ち | 方法・資格・契約の可否を含めて未判定 |
| PCG64 joint stream | 今回のモデル外／未資格 | 差はnull、0ではない |
| DF | 保存numerical pilot取得済み | 既存energy評価で一回だけ。Raw信号に混ぜない |
| RPE statistics | 既存式・政策あり | 新しいshots/energy/actual CA評価は未実施 |

Stage Aの保存引用：TV約3.42775e-15、m12条件付きraw上界約8.98564e-10、保存raw信号再生の最大残差約6.02110e-15。
いずれも上記固定H4/CDF8/LD2/s1-r32-K0/h.1/m0..12の範囲であり、bridgeの厳密上界や全native誤差を取得したという意味ではない。

誤差会計は既存どおり

`delta_m = A_m(eps_PF,m + eps_corrected_RTE,m) + eps_synthesis,m + eps_categorical,m + eps_bridge,m + eps_reference,m`。

四追加項はraw単位なのでAを再乗除しない。Native Gaussian→saved-atom、atom→CAR、reference演算の所属をB2公開後に照合する。
同じ丸め・normalization差をbridgeとreference uncertaintyの双方へ入れない。

## 最小統合adapterと既存accuracyとの互換性

[新統合層](../../src/trottertracks/c_b1_b2_scope_integration.py)は既存`c_iid_grid_accuracy.qualify/assess`を呼ぶだけで、
`RoundEvidence/assess_round/assess_energy`、shot式、PF/RTE/DFの会計を再実装していない。
共有の`c_all_law_signal_ca.py`、B1/Stage A固定moduleも無変更。

科学target/αβ/phase/space/13round/raw単位を照合し、`included_error_terms`の重複とPF/RTE/DF等の混入を拒否する。
これは申告されたpartitionの技術検査であり、意味的な誤差の重複がないことは出典と方法を別途照合する。
有限・確率的scopeは現在の全word資格へ自動変換せず、別のqualification設計が必要と返す。
数値estimateやprecision sensitivityをrigorous boundへ変更しない。

公開B2 pinまたは必要boundが不足すれば、既存accuracy自体を呼ばず`NOT_READY_NO_STAGE_C_CALLS`。
人工入力がすべて揃った場合だけ既存IID-grid gateへ接続する。Actual-law packetの`error_evidence`はnullを保持する。
旧ideal、IID-grid条件付き数値、実PCG64、保証付き精度、actual CAの五区分を維持した。
必要入力が将来揃った場合の最大16-call adapterは残るが、今回Stage Cは0 callである。

## B2の独立性と保留する判定

B2が未観測なので、その結果がsensitivity・estimate・validated bound・rigorous enclosureのどれかは判定していない。
その科学契約が実行可能か、必要calls・precision・出力schemaが何かもUNKNOWN。
公開後はcandidate/科学SOURCE/Stage A input、meanか各trajectoryか、sector/full-space、phase、13round、normと既包含丸めを確認する。

一般にreference限定の数値不確かさ評価は、native全word一様boundとは独立して実施できる可能性がある。
B1がNOT_READYであることだけをB2限定検証の停止理由にはしない。一方、実際のB2契約を未読のままGOにせず、
B2だけでStage C全体がPASSになるとも説明しない。今回B2作業の再実装や未公開worktreeの探索はしていない。

## 設計状態・検証・公開・次の担当

[未承認比較設計](research_c_b1_b2_accuracy_scope_design_unapproved_20261103.json)は`PENDING_B2_PUBLICATION`、非実行。
Uniform/finite/probabilistic/B2独立案を分け、未決のfamily/sample数・精度・信頼度・科学call上限はnull。
旧38/16-callや人工テスト件数を新科学予算へ転用しない。実行許可だけで未完成設計を起動しない。

[テスト記録](../../artifacts/research_c_b1_b2_scope_integration/2026-11-03/technical/tests.json)：**新32＋既存252＝284 unique PASS、旧2件deselect**。
Identity/SOURCE/政策、raw単位、二重計上、unknown、finite/probabilistic scope、IID/PCG64、estimate/rigorous、
coherent model、校正confidence schema、partial round STOP、B2不在時のStage C誤起動拒否を人工入力で確認した。
旧2件はStage A実行前fresh専用とC2並列例外順序の既知case。消費済み拒否と両順序の既存代替testを含め、旧SOURCE/testは無変更。
**B2固有testsは未取得であり、B2との実データ統合検証PASSとはしない。**

- 実装SOURCE commit：`8516bd166f95aa87200baf84b4f7d891c9b5cb9e`。
- [SOURCE166 file](../../artifacts/research_c_b1_b2_scope_integration/2026-11-03/technical/SOURCE.json) SHA256：`ce16b59e28ee8e0049b630d2cc019749dd59b5c67fda0b05ad0c8747371d57d5`。
- 設計SHA256：`0aee98eee9bb2b821f175c7cbf27335aa1bfc81ae615d2adb5dd9d330cd8c76d`。
- [Passive preflight](../../artifacts/research_c_b1_b2_scope_integration/2026-11-03/technical/passive_preflight.json)：数値library importを遮断してPASS。保存input/metadata33 fileを照合。
- [保全・静的確認](../../artifacts/research_c_b1_b2_scope_integration/2026-11-03/technical/static_verification.json)／[STOP](../../artifacts/research_c_b1_b2_scope_integration/2026-11-03/technical/STOP.json)。

CPU affinity32・available RAM約52GiB。既存resource_plan/ColumnStream/immutable入力共有とBLAS調整を将来も再利用する。
今回の人工suiteはtask-local cache、共有venv/OS/global cache/他jobは無変更。新H4 science/array decode/native build/compile/sampling/Stage A再実行/Stage C/重要レビュー0。
科学的な新結果や資格変更はなく、旧Evidence Map・validation manifest・研究概要の科学記録を変更しない。
独立branch `codex/research-c-b1-b2-accuracy-scope-integration-20261103` で公開し、remote SHA/tree・本文・主要blobを照合する。

**Mandatory STOP。** 次はユーザーがB2完了報告と固定公開commitを渡す。Codexはその後、B2の実際の方法・資格・独立契約を照合する。
有限/確率的scope、推定対象と校正confidenceの正式な採否はユーザー／研究Cが判断する。
新科学実行はscope・SOURCE・callsを固定した別の明示許可が必要で、今回は開始しない。
