# C1-0：G6–DF応用接続の実装前preflight

2026-10-20 JST。基点 `291e21218c1d1fde062c001a1586ba7c1bb9c988`。
**C1_0_STATIC_PREFLIGHT_COMPLETE_WITH_OPEN_GATES。C1の実行準備は未完了。科学的GO・新規性・投稿GOを判定しない。**

指定H4 snapshotは固定Git/GitHub/既知ローカルcopyで一致した。state・DF factorの既存digestと、生成順prefix3のcomponent記録も取得できる。一方、実行可能basis registryの独立した保存物、exact G6入力とのmodel差certificate、H4 strict provider、非列挙の認証native期待費用は未取得である。G8の既存finite-provider誤差枠組みやP5の群構造を再利用できる余地はあり、すべてを新理論の不足とはしない。

成果物：[adapter・Evidence仕様草案](adapter_evidence_draft.md)、[誤差・費用接続表](error_cost_connection.md)、[handoff](handoff.md)、[固定証拠・identity](evidence_inventory.json)、[管理照合](verification.json)。仕様草案は実装・計算契約の採択ではない。

## 1. 作業範囲とGate判定

既存文書・SOURCE・JSONのmetadata/schemaを読解し、opaque NPZのbytes/SHA256だけを照合した。NPZ/HDF5の開封・member読出し・数値配列解釈、係数変換、Hamiltonian/basis再構成、数値再解析、科学SOURCEのimport/実行/変更、科学test、新証明、adapter/回路/合成/benchmarkは0件。主agentだけで作業した。管理用コードは標準libraryによるGit・text/JSON・HTTP・hash・リンク照合に限定する。

依頼文が伝える「GPT科学レビューによるC1限定継続」を前提とする。任意添付の `Track_B_C1_応用接続_研究価値レビュー_2026-10-20.md` は既知のworktrees直下にはなく、原本未取得である。未読の原本による技術保証は認定しない。再レビューの開始や方向性の変更は今回行わない。

| Gate | 総合状態 | VERIFIEDとなる範囲 | C1適用の不足 |
|---|---|---|---|
| 1 保存入力・同一性 | **CONDITIONAL** | 指定snapshotの再取得・opaque bytes/hash、固定state/DF/component identity記録、生成SOURCE対応 | basis registry実体は検査したPR-2 packageでMISSING_SOURCE。array/member内部の独立再照合と新exact入力のbindingはREANALYSIS_REQUIRED |
| 2 誤差・confidence | **CONDITIONAL** | 既存G6/F1、G8 finite-provider、D1/D3積誤差・moment/rangeの適用条件を特定 | H4のPF/model/provider/state/readout certificate、精度・failure配分・全path bindingと数値評価が未完。十分N未算出 |
| 3 adapter・費用経路 | **IMPLEMENTATION_REQUIRED** | 型/語順/位相/basisの仕様差、全event列挙箇所、既存非列挙部品を静的特定 | adapter/weighted accumulator未実装。全supportのstrict/native費用coverageと認証期待費用はUNRESOLVED |

`VERIFIED`は表の限定した事実を指す。CONDITIONAL・MISSING_SOURCE・REANALYSIS_REQUIRED・PROOF_REQUIRED・IMPLEMENTATION_REQUIRED・UNRESOLVEDを、科学的な非改善と混同しない。新しい非自明な保証が必要と確定しない限り、データ/実装不足だけをPROOF_REQUIREDと呼ばない。

## 2. Gate 1：入力所在と再取得

### 指定snapshot

| 項目 | 確認結果 |
|---|---|
| 固定path | [artifacts/pr2_s0_s1_validation/2026-09-28/h4_1p00_rank12_development_v1.npz][SNAP] |
| 固定commit | `243eea8e840f3c472753f7935e510391626fd325` |
| Git blob | `39770ad86433a029d2f5157a90384f5ee37b12ce` |
| bytes / SHA256 | **8,652** / `3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a` |
| 取得照合 | Git blob・固定raw GitHub取得・方法ノートworktree内の同相対pathがbyte一致。数値開封なし |
| 過去の実行cwd | 結果は旧 `/home/abe/Project/Partially Randomized Trotter` を記録する。現在の実在を仮定せず、この旧pathを探索しない |
| モデルの意味 | linear H4 1.00 Å / STO-3G / DF rank12 / 8 system qubits / 生成順prefix3という**既存報告**のidentity。新しい物理・数理認証ではない |

前回調査の「本体未取得」は、このC1-0で許可されたopaque byte照合によって更新した。既存調査資料や過去の科学結果を変更していない。

### State・DF factor・Hamiltonian hashの根拠

既存結果[RESULT] `/stages/V1/development/digests`は、次のdigestをsnapshot raw hashと同じdevelopment recordへ結んでいる。

| identity | 保存済み値（今回再計算なし） |
|---|---|
| Hamiltonian | `de7a549238e3a21f15a84018bef28440c345b31030282c01cf874f3d1d212424` |
| full state array | `c9aca811b5c023772d148d0331c82958bac6824b593a367cb65a5f89937f4f63` |
| sector state array | `8a4b845362a63bd7d0352d7a02fbfb844e13d10ba6d5e6e451e626fa8804395d` |
| combined state | `31e63b0104126c85136ee173f1dce7642aee2d272924e70e8b120ac340ab45bd` |

Stateは同NPZの `state_vector` / `sector_state_vector`、DFは `constant`, `one_body`, `lambdas`, `g_matrices`、sectorは `sector_basis_indices`に保存する設計である。[PRLOAD] EXPECTED_LAYOUTと[S0] `write_snapshot`を静的確認した。これはmemberを今回開封して照合したという意味ではない。one-body/lambda/12個のg-matrix digestも既存結果にある。

[S0] `_array_hash`はdtype・shape・C-order bytesをhashし、`_state_hash`はfull/sector hashesと`largest_sector_amplitude_real_positive_v1`を結ぶ。一般ファイルSHAとは異なる。[DFHASH]のHamiltonian hashも配列hash、hex値、metadata・weight rule等を含むprovenanceであり、NPZ全体SHAと同じ値ではない。新たにこれらを再算出する処理はREANALYSIS_REQUIRED。

元V1–V3 SOURCEは `ef86868a95e9b77eb5cb521ec156f66c7d6c25f6`。resultに記録された主要 `df_partial_randomized_pf.py`, `df_partial_s2.py`, `df_rte_tail.py`, `pr2_new_series_validation.py`, `pr2_s0_s1_validation.py`のfile SHAは、元commitと243eeaの両Git blobで一致した（metadataの照合）。これらの実行や旧testsの再実施はしていない。旧V1の`two_load_digest_match`は過去の保存証拠であり、今回の独立load検証ではない。

### Componentとbasis

生成順は[S0] `generation_ranked_fragments` / generation partitionがoriginal indexを保ち、[PRLOAD] `_method_record`がB2-Gへ`explicit_ordered_partition`を渡す。既存結果のrank3 B2-G pointerは `/stages/V2/ranks/0/methods/B2-G`。

- ordered deterministic indices `[0,1,2]`、random indices `[3,4,5,6,7,8,9,10,11]`を記録する。
- partition hash `722ec0077b4505d963c03bccdd6311f9dd8b3d3a5ed1b9117f7a6f8fa85a7450`、preparation hash `749e99a787dfc8172c9c9748af05acc9a00d15d5099575d0eb1e303409dad557`、tail hash `ab5c095694f41496f9edd00342f3654425f190390aa8f02def76ffe1209b98cb`。
- `sampling_components`にcomponent ID、fragment ID、support、is_identity、signと既存float coefficient/probabilityがある。324 components、λ_R=0.5832704246339983は既存値の引用のみで、今回は合算・正規化・係数解析をしない。
- B2-Wは同じdeterministic prefixでもrandomized index順・partition/preparation hashが異なる。G/Wの同一Hamiltonian報告から、すべての来歴IDまで同一とみなさない。

[DFTAIL] `pre_threshold.sort`の順は**fragment ID文字列、support長、support tuple**であり、生成順indexや単純なcomponent ID sortそのものではない。保存component列を順序の根拠とし、G6の整数labelへ自動で再sortしない。同じ係数・basis型でも異なるcomponent/operatorを同一視しない。

identity policyはPR-2 SOURCEで`extract_identity_phase`、thresholdは`coefficient_atol=0.0`、basis構築は`fermionic_gaussian_jw`、diagonal sortは`descending_abs`。これらは[S2]と[PRLOAD]に固定される。tail scalarは`exp(-i*t*coefficient)`として保持し、constant/one-bodyの位相と別管理する。

**不足：** snapshot保存schemaは対角化後η・runtime basis gate列を保存しない。上記V2 component rowにもbasis ID/hash/操作列はない。tail hashは[DFTAIL]のpayloadにbasis metadataを含むが、hash値から操作列を復元できない。検査したsnapshot/PR-2 V1–V3 packageに独立したserialized registryは特定できなかった。他branch・非公開cache全体に存在しないという断定ではない。

再取得経路は[S2] `prepare_df_partial_s2`→DF block構築→[DIAG]の`np.linalg.eigh`等→Gaussian gate変換→[DFTAIL] `registry.register`。**これは数値再構成であり今回実行禁止。** 既存serialized registryの提供・所在確認ができれば先にそのbytes/hashを用いる。なければ別承認の限定再解析とruntime固定が必要となる。縮退空間・固有vector位相・sort/tie・library版によるbasis hash変化を、同一snapshotだけで排除しない。

## 3. Gate 2：適用可能性

対象は `z_H(T)=<ψ|exp(-iTH_DF)|ψ>`の単一時刻Re/Im。T=.8、q_outer=1、r_RTE=4、m=5、ε_total=.05、α_total=.05は**未採択草案**として参照するだけで、x、誤差、N、費用を新たに代入・評価していない。

適用表は[誤差・費用接続表](error_cost_connection.md)。重要なのは次の境界である。

1. 正有理pと0≤x≤1ならG6/F1の条件が使えるが、保存float coefficientからexact p/λを定義する処理と元Hとの差はまだ未確認。F1/F2自体を未完成理論へ戻さない。
2. [G8PROOF][G8BOUNDS]は全labelのunitary strict controlled-Q近似、actual adjoint、相対位相を仮定した既存誤差式を持つ。これは新証明を要する空白ではない。しかしH4のδ達成・全basis/rotation certificateは未取得。旧仮想δを達成値にしない。
3. [DACC]の非unitary finite meanの積、[D3]のTaylor保証は既存部品。D1実装はtoy input/steps/degree/stateを固定して拒否する。C1 outer PF・deterministic native・入力近似まで結合した完成certificateではない。
4. 旧PR-2のnumerical signal bias/shot設計、[RPE]のunit-radius/effective eigenstate/attenuationモデルは、G6のevent依存weight・zero-fillの認証Nへ転用できない。familywise配分と全N完了のbindingは未採択・未実装。

既存保証の仮定を満たす入力・provider・誤差上界が取得できれば、通常の適用・会計として閉じる部分がある。一般stateに対する新しいphase変換、未被覆angleへのuniform synthesis/費用保証、履歴依存lawなど、既存範囲外の非自明な命題を必要とする場合だけPROOF_REQUIREDとして切り出す。今回は証明しない。

## 4. Gate 3：adapterと非列挙費用

[G7] event dictはexact proposal/weightと還元word、元degree由来phase、`circuit_time_word`を持つ。[RTE] `RTEEvent`はproduct数=Taylor order、phase=(-1)^(order/2)、float coefficient/probability/normalization関係を要求する。任意G6 packetをその型へ偽装すると元degreeと還元word、実π/Wの意味を失うため、独立sidecarが必要である。

DFのruntime basis unitaryをBと呼ぶと、[DFTAIL] `component_dense_operator`は**B P_S B†**、[DFBUILD]の作用時間順はB†→中央P_S→Bである。このBをG9の記号Vと無条件に同じ向きで使わない。実装済みreferenceを今回呼んだわけではない。

全event列挙箇所と利用可能な部品は[費用表](error_cost_connection.md#2-全event列挙と既存の回避部品)に整理した。`collect_pending`の`list(reference_events(...))`だけでなく、`freeze_inventory`の全key集合と`evaluate`の全binding/平均費用・matrix検査まで依存する。[DPIPE]を参照し、列挙を単に別工程へ移して非列挙と呼ばない。

既存の候補部品は、G6局所生成/U、G8 on-demand cacheとfinite-provider上界、DF構造的instruction-size上界、P5少数群normalizer/angle式である。ただし**raw instruction上界≠strict native T上界、観測key cache≠全support coverage、群angle価格≠全wordの期待価格**。既存部品だけでH4の認証期待native費用まで閉じたとは確認できない。

## 5. 保全と次の承認

保全は本branchの基点に対する新規directory追加だけで確認し、既存tracked SOURCE/raw/grant/claim/ledger/marker/STOP、前回公開文書を変更しない。指定snapshotと既知ローカル参照のbytes/hash、依頼原本を限定照合する。他worktreeを凍結せず、共有設定・他processを操作しない。私的レビュー原本の転載なし。

次の具体的な技術作業は、serialized basis/preparation artifactの提供または限定再解析の許可範囲を固定すること、その後exact入力/model差とprovider/全path誤差の仕様を確定することである。詳細は[handoff](handoff.md)。新証明・再解析・adapter実装・native取得・科学計算は各々別承認を要する。**公開・保全確認後mandatory STOP。**

## 固定参照

リンクは固定commit。参照ファイルのbytes/SHA256と読解範囲は[evidence inventory](evidence_inventory.json)。過去SOURCE/結果のversionと現在の監査versionを混同しない。全体仕様は[方法ノートv1.1][NOTE]、C1の未採択条件は[前回契約草案][C1DRAFT]を参照する。

[SNAP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/artifacts/pr2_s0_s1_validation/2026-09-28/h4_1p00_rank12_development_v1.npz
[RESULT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/artifacts/pr2_new_series_validation/2026-09-28/pr2_v1_v3_result_v1.json
[PRLOAD]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trotterlib/pr2_new_series_validation.py
[S0]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trotterlib/pr2_s0_s1_validation.py
[DFHASH]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trotterlib/df_partial_randomized_pf.py
[DFTAIL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trotterlib/df_rte_tail.py
[S2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trotterlib/df_partial_s2.py
[DIAG]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trotterlib/df_trotter/decompose.py
[G8PROOF]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/docs/tracks/algorithm_codesign/g8_proof_contract_and_on_demand_scope_20261010.md
[G8BOUNDS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trottertracks/algorithm_codesign/g8_bounds.py
[DACC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trottertracks/algorithm_codesign/rte_development/accounting.py
[D3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/docs/tracks/algorithm_codesign/d3_guarantee_price_preregistration_v1_20261013.md
[DPIPE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trottertracks/algorithm_codesign/rte_development/pipeline.py
[G7]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trottertracks/algorithm_codesign/g7_generator.py
[RTE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trotterlib/rte.py
[DFBUILD]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trotterlib/df_rte_qiskit.py
[RPE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/src/trotterlib/rpe_resource_accounting.py
[NOTE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/docs/manuscripts/track_b_return_aggregation_native_resource_note_v1_1.md
[C1DRAFT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/291e21218c1d1fde062c001a1586ba7c1bb9c988/docs/research/track_b_ra_application_feasibility_20261020/minimal_validation_contract_draft.md
