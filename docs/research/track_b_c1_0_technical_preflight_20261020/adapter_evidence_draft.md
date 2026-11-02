# 最小adapter・Evidence仕様草案

**仕様のみ。IMPLEMENTATION_REQUIRED。API/class/schemaを実装・正式採択していない。** 2026-10-20 JST。
Gate結果・入力identityは[preflight](README.md)、保証と費用は[接続表](error_cost_connection.md)。新しいsamplerやcost oracleを定義しない。

## 1. 入力sidecarの必要情報

以下のfield名は説明用で、採用済みAPIではない。既存G7/F3のlawを保持し、DFとの意味論・identityだけを結ぶ。

| 区画 | 必要field・不変条件 | 取得状況 |
|---|---|---|
| version/source | schema/version、G6/F3/DF/measurement SOURCE commitとfile SHA、provider/runtime/precision policy、scope | 固定SOURCEは取得可能。新adapter SOURCEは存在しない |
| input | snapshot file SHA/bytes/blob、Hamiltonian/partition/preparation/tail hashes、model/state definition、state order、identity/threshold policy | snapshotと既存hash recordは取得済み。新exact model certificateは未取得 |
| labels | ordered `label_id ↔ component_id ↔ original_fragment_index/df_fragment_id`、retained/removed理由、support、sign、basis ID/hash | 保存V2 component列にcomponent/fragment/support/signがある。basis対応実体を含む完成sidecarはない |
| exact scalar | 採用した有理係数、正の絶対係数、λ、p、時間/σ、除外zero、sum(p)=1の証拠、元入力との差certificate | 変換・演算を今回はしない。REANALYSIS_REQUIRED |
| operator | system qubit順、P_S=ZまたはZZ、basis操作Bと実際のadjoint、scalar sign、identityの別扱い | SOURCEの規約を特定。保存basis gate列・native証拠は未取得 |
| schedule | T、outer PFの全ordered block、各tail occurrenceの時間、r_RTE、m_Taylor、前後phase、量子前全path選別の可否 | 草案値だけで実occurrence scheduleを取得・生成していない |
| precision | root/probability bit数、η/ρ、F3 angle区間policy、uniform provider/native error、state/readout仮定、familywise policy ID | 新C1用は未採択。旧値・confidenceをコピーしない |

### Exact入力の方針候補と境界

保存binary64係数をexact dyadic rationalとして採用し、λを**その有理絶対係数の和**、pをそのλで規格化する方針は検討できる。この定義なら、個々のfloat確率をFraction化してsum=1と仮定する処理を避けられる。ただし今回は変換しない。この方針は[ReturnKernel][KERNEL]の入力条件を満たすための既存仕様への対応候補で、新しいsampling法ではない。

元PR-2の`fsum`によるfloat λ/probabilityと同じ値・同じlawになるとは認定しない。また、保存DF factorからのfloat対角化・basis合成で得たcomponentを採用した有理Hamiltonianが、保存H_DFとexactに同じとも認定しない。threshold、係数変換、basis/model差、constant/one-bodyの扱いを別certificateで結ぶ必要がある。[DFTAIL][PRLOAD]

G6 labelの順序は保存component rowと明示mapで固定する。fragment順・係数順・basis group順へ暗黙に変えない。合法childは、非空parentなら**先頭の実label**を除き、空parentなら全label。係数型やbasis型が同じだけで別labelを排除・統合しない。zero係数を除く場合もID mapと原modelとの差を残す。

## 2. Eventと結果の必要schema

| record | 必須情報・意味 |
|---|---|
| Trial | trial ID、axis、schedule occurrence/path ID、input/source/policy/preflight IDs、σ、counter/domain、全N内の位置 |
| G6 event | arm、元group/Taylor degree、raw word、reduced ordered word、child実label、実proposal π、実target係数c_tilde、exact W=c_tilde/π、phase_i_power、ratioとangle enclosure、word時間順 |
| zero | raw nonreducedかcoin rejectか、そのbit/判定証拠。量子・合成・測定の未実施を記録。terminal failureとは区別 |
| path | 全occurrenceの順、各実π/W、path weightとzero判定、deterministic blocks/phaseの対応、量子初期化前に判定完了したか |
| native request | label/basis/control role、forward/actual-adjoint、ratio/angle区間・代表値とgate角規約、strict error目標、backend/tool identity、cache key |
| native receipt | 実gate列/IRのhash、actual adjointとphaseの証拠、strict joint error certificate、resource vector、取得/lookup/cache費用、coverage scope |
| measurement | X/Y、bit b→(-1)^b、signed weighted outcome、backend/readout/state evidence、失敗と実支払資源 |
| terminal | 完了全trial数、zero/accepted/測定完了/失敗を区別、全N完了、confidence対象、prefix/原本保存のidentity |

Fullの[既存F3][F3SRC]はinput/source/preflight/law identity、bit trace、event・parent・angleとterminal failureの証拠を持つ。これを再利用し、DF/quantum receiptをsidecarで結ぶ。元F3のcanonical非identity未対応は、P5等の法則が数学的に無効という意味ではない。各canonical経路にも正当なprecision/実law bindingが必要で、Fullの下界を流用しない。

**RTEEventへ偽装しない。** [RTE]はproduct数=taylor_order、phase=(-1)^(taylor_order/2)、float probability=coefficient/normalizationを要求する。G7 `_event`はreduced wordを持ちながら元degreeのphaseを保持する場合がある。これらの意味を、word長からTaylor次数を作り直して合わせない。独立sidecarから必要なprimitive/provider仕様を結び、old scientific APIは変更しない。

## 3. Operator・位相・時間順

DF builderでruntime操作列のunitaryをB_iと呼ぶ。unsigned componentは **B_i P_S B_i†**、signed involutionはs_iを掛けたもの。作用時間順は **B_i†→中央P_S→B_i**。[DFTAIL] `component_dense_operator` (907)、[DFBUILD] `_build` / `_append_basis`を根拠にする。G9のV記号からBの向きを推測しない。

G7のoperator記述は `i^phase exp(-i σ atan(ratio) Q_child) Q(word)`。回路上のword適用順は `reversed(word)`で、child回転はその後。`phase_i_power`は元degree/groupから保持する。wordをscalar係数計算で並べ替えられても、operator積は並べ替えない。[G7] `_event` (79)、[CQ] `controlled_event_ir`。

- signed productをunsigned P_Sで実装する場合のs_iはscalar位相としてancilla相対位相へ戻す。
- signed child回転をunsigned P_Sで実装するなら、signは角へ一度だけ反映する。既存RTE `unsigned_rotation_angle`が既にsignを掛ける規約を二重適用しない。
- G6のeven-parent phase、DF product sign、constant、抽出tail identity phaseを別fieldで追跡し、最終相対位相へ結ぶ。uncontrolledのglobal phaseとcontrolledのrelative phaseを混同しない。
- identity成分をtail外へ出す場合は元の時間係数に対応したscalar exponentialを保持する。G6の空parent/identity packetとDFの抽出identityを同じものとして合算・二重計上しない。

## 4. Controlとactual adjoint

既存の二つの経路は区別して仕様に固定する。今回どちらも実装しない。

| 経路 | 再利用可能な既存仕様 | C1で必要な追加証拠 |
|---|---|---|
| generic CQ/helper | [CQ]のouter-controlled word、helper-controlled child compute/actual-adjoint、2 RZ/2 CX/4 H等のliteral IR | labelごとのstrict CQ unitary、helper leakage込みjoint error、actual adjoint、追加workspace、実価格。query費用を0にしない |
| direct DF中央作用 | [DFBUILD]はbasisをcontrolせず中央Z/ZZと回転をcontrolし、basis reuse・相対phaseを保持 | G6 event用の独立binding、exact signed ordered targetとの対応、全basis/native誤差・event角certificate。既存RTEEventを生成したふりで接続しない |

片側だけgeneric helper、片側だけ最適化directを与える価格比較は、method差とprovider差を分けなければ公平でない。全方式へ適用可能なdirect lowering・basis reuse・cancel/cache・精度選択を対称に認める。

actual adjointは「同じ理想角の負角を別に合成したもの」と同じとは限らない。実際の取得列を逆順・各gateのadjointへ対応させ、scalar/global phase tokenも保持する。近似basisのinverseがどの実列のadjointかをnative receiptで結ぶ。既存builderの`gate.inverse()`という記述だけで、将来のClifford+T取得列のstrict一致まで認証しない。

## 5. 再利用と未実装の境界

| 再利用する既存内容 | 新規実装・取得を要する内容 |
|---|---|
| G6/Full/P5/ordinaryの既存generatorとlaw | G6 label→DF operator/provider sidecar、正確な入力binding |
| F1/F3のFull precision・実proposal/angle・terminal証拠 | canonical側の必要preflight、DF/native・測定・全path evidence結合 |
| DF registry/basis照合規約とprimitive契約 | 保存registryの再取得、実providerのstrict certificate |
| X/Y wrapperのS†→H、bit map | 既存DFPartialS2型に依存しないG6 wrapper接続、state/readout/全trial weighted accumulator |
| G8 on-demand/実際のadjoint記録 | C1のtool/precision/cache identity、全support/全N completionと費用契約 |

Evidence保存で全child lawを保持・再queryする現F3の費用を削除しない。案A/Bを採用せず、非列挙性は「全parent/event表が不要」という範囲に留める。full/compact Evidenceの再構成範囲、保持されない原文、再監査費用を明記する。

schema/hash/link確認は管理作業。sidecar生成、rational変換、basis再取得、人工科学testも今回実施していない。**新実装・新保証・再解析は別承認後。公開確認後STOP。**

[KERNEL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trottertracks/algorithm_codesign/return_aggregation.py
[DFTAIL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trotterlib/df_rte_tail.py
[PRLOAD]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trotterlib/pr2_new_series_validation.py
[F3SRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trottertracks/algorithm_codesign/g6_f3_generator.py
[RTE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trotterlib/rte.py
[DFBUILD]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trotterlib/df_rte_qiskit.py
[G7]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trottertracks/algorithm_codesign/g7_generator.py
[CQ]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trottertracks/algorithm_codesign/g7_provider.py
