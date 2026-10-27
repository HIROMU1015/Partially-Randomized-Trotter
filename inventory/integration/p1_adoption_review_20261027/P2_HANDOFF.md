# P2 Pauli-JWへの具体的引継ぎ（実装未開始）

## 固定入口と最新版の区別

既存アーキテクチャ `1b649896393a4a0c5f2e3bcf41b2853090ef2053` のP2を継承し、代表成果入口は
`codex/pauli-jw-research-summary-20261026` @ `b6c899698645a4ccdaa65c0c5e8d706fd9d1cc2b`。
着手時remote tipは同一だった。[summary](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b6c899698645a4ccdaa65c0c5e8d706fd9d1cc2b/docs/research/pauli_jw_resource_evaluation_summary.md) と[source索引](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b6c899698645a4ccdaa65c0c5e8d706fd9d1cc2b/docs/research/data/pauli_jw_summary/source_references.json)を再利用し、既存の科学比較は再集計していない。

公開metadataと正式handoffで確認した後続：

|役割|branch / 固定remote tip|今回の確認範囲|
|---|---|---|
|event拡張|codex/pauli-rte-event-extension-execution-20261026 / `10cb3e5cca7834ebf75301bd91bd093ca51d4a6b`|旧計画の固定providerとして維持|
|H6公平性監査|codex/h6-hybrid-search-fairness-audit-20261027 / `f7757c4320ba34dd36a64de4250c29a69bf38390`|公開refと既存計画|
|H6準備|codex/h6-hybrid-fairness-extension-readiness-20261027 / `bcc2eb6b201e1434bf47560775f5afb3e14d3ca8`|正式READINESS_REPORTの版付き参照|
|H6追加段階|codex/h6-hybrid-fairness-extension-execution-20261027 / `00958ae1945c780718b2deef07d5dab1096a5cd8`|公開CONTINUATION完了追記/README、tree/blob metadata|

[正式継続handoff](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/00958ae1945c780718b2deef07d5dab1096a5cd8/artifacts/h6_hybrid_fairness_extension/2026-10-27/CONTINUATION.md)の冒頭は追加段階完了と、現在の正本REPORT/EXECUTION_SUMMARY/STUDY_STOPへの参照を記す。
その後ろの「実行中/未完了」とPhase A文書は当時のsnapshotであり、現状記述へ転記しない。
科学SOURCE `887b69e449cb3c378286f09d0d7626599850fb67`、解析SOURCE `af891fd45805454c6592bbee5441266f14235ffd` はhandoffにある別版として保持する。
旧アーキテクチャ/summaryのH6大r Hybrid未取得記述は固定版のscopeであり、P2で新たに重複開発して埋めるべき欠落ではない。
本作業は新結果の内容監査/再ランキング、runtime/job/worker/controllerへの調査をしていない。完了はhandoffの記載で、独立科学認証ではない。

## 使用すべき既存実装とadapter境界

以下はsummary固定commitの実source/ASTを確認したAPI。上の最新handoff branchとの10選定pathのblobは一致した。
新H6専用adapterのbodyは今回未調査で、readinessの正式対応表から既存資産として引き継ぐ。

|段階|再利用module / 正確なAPIまたは入口|P2の責務と注意|
|---|---|---|
|Pauli入力|`pauli_signal_adapter.PauliHamiltonian`; `from_terms`; `PauliSignalAdapter`|既存保存terms/constant/bit順/label順/入力identityを受ける。from_spatial_integrals/from_openfermionを自動生成fallbackにしない|
|method/law|`PauliSignalAdapter(hamiltonian,method,ld,step_time,r,K)`、`log_B_step/trajectory/operators/metadata`|PF/B0/Hybrid/pure RTEとprefixの適用域を研究adapterが分ける。trajectory/operatorsは実scienceなので今回未実行|
|H4比較|`scripts/run_h4_pauli_b0.py:inputs/verify_source/prepare/compile_axis/target`|h4_b0_execution_private namespaceと固定旧runnerを保持。scriptをlibraryとして丸ごとimportしない|
|Hybrid対RTE|`scripts/run_pauli_hybrid_rte_comparison.py:source_check/h4_trajectory/worker/aggregate`|hx_study内の固定moduleを現在のtrotterlibへ暗黙差替えしない|
|event拡張|`scripts/run_pauli_rte_event_extension.py:check/model_ref/ham/target_worker/price_worker/proposal/confirm`|event_run namespace、prefix0専用範囲、探索/確認の別seed/cohortを維持|
|H6大r Hybrid|`src/trotterlib/h6_hybrid_fairness_extension.py`（正式readiness記載）|既存4候補用adapterを再利用候補にする。旧r≤8やprefix0 wrapperを変更せず、新H6 pathを二重実装しない。最新body/APIの限定引継ぎ確認はP2実装前に行う|
|mean/包含|`pauli_vector_tight_enclosure.evaluate`; `sparse_mean_action`|law固有mean/Arb-L2/raw normは別adapter。今回数値評価なし|
|wrapper/価格|`fixed_signal_evaluation.hadamard_wrapper(evolution,axis,measure=True,already_controlled=False)`; `rte_compiled_cost.transpile_and_measure_cost`|+Im sine、central/generic control、global/relative phase、state prep範囲を保持。H4 DF wrapperを流用しない|
|会計|`fixed_signal_accounting.fixed_signal_shots(*,epsilon_sig,bias,allowance,log_B_upper,evidence_kind,evidence_ref,epsilon_axes,alpha_axes)`|旧H4 shots簡略式やphase RPEへ置換しない。Nと価格標本数n、仮想rho/未資格/guardを分ける|
|保存比較|`pauli_hybrid_rte_comparison.saved_class_minimum/hybrid_over_rte`|保存表の用途を保持し、新composerのimport/planから呼ばない。新たなwinner/比の再評価は科学許可が必要|
|人工I/O|`fixed_signal_io.load_artificial_input/source_provenance`|人工入力は実保存snapshotの復元済み証拠ではない。real loaderと分離|
|結果入口|`scripts/document_pauli_jw_saved_summary.py:assemble/export/check`、既存source_references|SOURCE/result/code/summary版を別locatorにし、旧表/STOPを上書きしない。今回generator/評価関数は未実行|

10 pathの同blob確認は上記の主要既存6 moduleと4 scriptだけ。新H6専用module、全import closure、全研究の同等性を意味しない。
root CEMの有無はbranchで異なる。既存別版のEVIDENCE_MATRIX/summaryを参照し、新しい科学正本DBは作らない。

## P1から引き継ぐもの・引き継がないもの

採用するのはCLI→薄いcomposer→研究adapter→既存同checkout core、contextごとのSOURCE/input/state/cohort/law/compiler/metric、
STOP/欠損/UNQUALIFIED、科学modeの明示境界、mockと科学検証の分離という設計。
P1の`ScientificContext`はDF rank/partitionを必須にしているため、そのclassやSTAGESをPauliへ機械移植しない。
Pauli contextは既存snapshot/APIへ必要最小限に合わせる。P1のshared facade lazy化も継承しない。

H4 DF rank12/sector/state/geometry/生成順、DF PF/RTE law、DF signal/RPE会計、H4 compiler/cost scope、
P1未検証数値挙動をPauliの既定へしない。law名やq/r/Kが同じでも同一分布/seed/価格の証明にはならない。
Pauliではconstant/保存順/同一raw rayとexact norm、CDF/PRNG scope、探索/独立確認、rho/allowance、compiler/contextが研究固有。
CPU/RAM/workerの小さい既定capや承認証明/grantを新設せず、既存個別契約の制御はそのまま保持する。

## 次のP2実装で準備する限定技術test

既存 `tests/test_pauli_hybrid_rte_preparation.py` / `test_pauli_rte_event_extension.py` と新H6 adapterのtestをまずsource選定する。
readiness既存27件には人工回路build/compile/作用/oracleが含まれるため、27 PASSをそのまま今回許可のpure/mockへ転用しない。

1. fake frozen loaderでH2/H4/H6のsnapshot namespace・SOURCE/constant/label順/bit順/state/normの引数を保持（値を生成/評価しない）。
2. method/prefix、signed-event順序、q/r/K/law、compiler/metric/prep/controlのscope不一致を拒否。探索とconfirmationのseed/cohortを混ぜない。
3. mean/price/shot APIの呼出しはmock、欠損N/G/rho/guardと未資格を保持しzeroで補完しない。未観測rare eventを確率0へ変えない。
4. real modeはloader/trajectory/worker/出力前に拒否。script importがdispatch/既存runtime初期化を起こさないよう新入口だけを使う。
5. 固定旧runnerのAPI/namespaceは静的比較、旧SOURCE/result/STOP/output形式は不変。read-only saved-index案内と新実行pathを分離する。

実装前にユーザーがP2の研究単位/代表branchとscopeを指定し、最新正式handoffのSOURCE/STOP/APIを担当者の公開引継ぎから限定照合する。
並行研究の現行環境を調査して埋めない。不足情報は保留欄に残し、既存固定機能を同名新実装で置換しない。
元snapshot取得/restoreが必要なら科学実行とは別のbytes保全範囲を確認し、実path/非公開custodyを公開しない。

## 実装許可と科学検証許可

P2 composer/adapter/CLI/人工mock回帰の実装は次の明示的技術作業として許可を受ける。
新Hamiltonian/state/reference、mean/normalization/bias/shot/Gの数値再評価、sampling、実build/compile、探索/独立確認の追加、旧結果再ランキングは別の科学実行許可を要する。
旧SOURCE/STOP/one-shot/retryやB3保護は引き継ぎ資料の整備だけでは変更されない。
今回はP2 code/testの実装・起動0。mandatory STOP。
