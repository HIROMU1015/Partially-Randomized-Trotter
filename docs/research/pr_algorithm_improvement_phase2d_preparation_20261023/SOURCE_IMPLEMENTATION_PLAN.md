# 将来の最小prototypeと固定SOURCEへの接続案

**実装予定の設計のみ。以下の新ファイル・型・関数を今回は作成していない。** 最終SOURCE closureとtestsを固定する前はF1実行承認不可。

## 1. 三種類のSOURCE identity

| 役割 | 固定ref／状態 | 静的確認した内容 |
|---|---|---|
| 文書・実装起点 | ef2eb7d55ea37bf93e0b85773df20b63811d1f9b | 新namespace追加のbase、既存SOURCEを保全 |
| RTE/law参照 | 2978e2fea672b7a1ff20cac74269ec9a610159dc | rte.pyは起点とbytes同一、Component/TailLike/event/law |
| 回路・反復参照 | 同2978 | df_rte_qiskit.pyとdf_partial_s2_repeated.pyは起点とbytesが異なる |
| sequence support/full参照 | f29eafbf4412193e8e8660d9f051889c2cba684b | research_direction_sequence_policy.py、support_run_le_1 |
| arbitrary Pauli／toy参照 | ae7434452450e80ec007d65417335e18a063fa1e | construction_comparison.pyのPrimitive/apply_pauli |
| 将来の新prototype SOURCE | null | 未実装、commit／closure未確定 |

起点のdf_rte_qiskit.pyは16,363bytes／SHA256 `15395beab7f6e8d68589964516c03d1b513b87439490e1cb1d1487fbf4848aac`、参照2978は20,313bytes／SHA256 `c710de23b7bfef525199ab28b897c21b2a78723a01b29da6067ff64354dcd436`。起点には参照版のsequence builder／basis_plan接続をそのまま仮定できない。反復builderも28,698対30,013bytesで異なる。両者を混同せず、参照の共有・融合policyを新prototypeへ明示的に接続／portし、全4armへ同条件で与える。

## 2. 最小の追加予定

候補配置は `src/trottertracks/pr_dictionary_validation/fixture.py`, `prototype.py`, `reference.py`、F1専用 `scripts/run_pr_dictionary_f1.py`。production moduleを広く書き換えない。将来追加時にはPROJECT_MAP／scripts／src／docsの索引更新も別承認範囲で行う。今回それら既存文書は変更しない。

| 予定type／function | 接続・責務 |
|---|---|
| FrozenTaskIdentity / FixtureSpec | exact fraction、frame convention、state、scalar、全digestの検証 |
| MixedAtom / MixedTailView | RTEComponentとRTETailLikeへ接続。kind=FRAME_Z/PHYSICAL_PAULI/IDENTITYを別typed mapに保持 |
| make_fragment_alternatives / canonicalize_arm | exact JW代数、P fragment間collect、S origin保持、exact zeroのみ |
| build_mixed_event / build_component_probe | 既存event構造・sign accessor・application_sequenceを変更せずrouterへ渡す |
| MixedSequencePolicyBridge | same-basis run、support/full、I境界、phase集約、D半step融合を全armで共通化 |
| MixedEvolutionResult / build_mixed_hadamard_wrapper | strict DF resultへ偽装せず共通wrapper recipeへ橋渡し |
| independent_fixture_operator / reference_event / reference_target | candidate map/router/polyを共有しない小系CAR／Gamma(V)／独立高精度経路 |
| literal_recipe_price / expected_price_state_machine | fixed uncompiled RZ recipeの直接走査とDPを独立実装 |
| registered_case_plan / verify_case_identity | counts、case ID、SOURCE/input/IR hash、one-shot/dedupを固定 |

既存 `normalize_involutory_tail` やtoyのdense検証は小系referenceに限る。arbitrary PauliをDF metadataへ偽装しない。libraryのproduction APIが新routerを既に受け入れるとは主張しない。

## 3. Circuit recipe義務

frame Q=B Z_S B†は時間順 `B† → controlled central → B`。physical Pauliは既知局所Clifford＋parity compute→controlled central→actual uncompute。basisは非制御、中央と必要phaseだけ制御。supportが長いC=X0Y2Z3も扱う。negative product、I、paired -1、scalarはancilla phaseとして保持。

Gaussianは `U_to_qiskit_ops_jw_givens` のnative gate経路を参考にし、SOURCE内BogoliubovTransform(U.T)と本候補Bのconventionを検査する。`U_to_qiskit_ops_jw` のdense fallback、任意UnitaryGate oracleは許さない。vacuum global phaseを消してcontrol相対phaseを変えない。

WP06-bの `register_support_restricted_bases`／`make_run_threshold_basis_plan` は `research_direction_structure_pilot.support_restricted_unitary_completion` に依存する。helperを含むtransitive source closureが必要で、トップmoduleのhashだけでは不足。basis completionの非一意性は同じprovider／seed／precisionで固定する。未取得・未bindingのclosureはblockerとして残す。

既存Hadamard requestは `isinstance(DFPartialS2RepeatedCircuitResult)` を要求し、mixed resultをそのまま受け取れない。新adapterで同等のancilla-last、step metadata／fingerprint／時間、measurement-free evolution検査を保持するか、純粋なwrapper helperを新prototype内へportする。型偽装はしない。

wrapperの時間順は、system prep X0 X1、ancilla H、controlled evolution、YだけSdg、ancilla H、terminal measure。この規約でYは+Im。既存wrapperは `state_preparation_included=False` なので、full wrapper費用ではprepを明示的に加える。F2はfull wrapperを一度compileし、同じcompiled回路からterminal measurementだけ除いたoperatorで正しさを確認する。

## 4. 将来必要な検査と境界

meaningful testsはexact S/P algebraとcollect、sign/order、n0/n2、negative/I/scalar、control0/1、actual adjoint、full/sector、wrapperY符号、common sharing/fusion、priceDPvs独立列挙、hash/dedup/call拒否である。数値testの実行も科学処理に数える。実装だけの承認を得た場合にこれらを勝手に実行しない。

新SOURCEのcommit／全import closure／env lock／wheel hashes、input serialization、価格provider、interval provider、case countsをfreezeし、文書と同時にレビューしてからF1実行承認へ進む。今回SOURCEをimportせず、実装・test実行・生成・build・compileを行わない。
