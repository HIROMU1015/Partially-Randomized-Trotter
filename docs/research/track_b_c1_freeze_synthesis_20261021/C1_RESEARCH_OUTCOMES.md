# C1成果総括：確立範囲と未確立事項

今回の作業は既存記録の整理であり、新しい科学レビュー・検算・再解析ではない。採択済み方針は **C1追加研究を一時凍結し、G6の数学的・方法論的研究を維持する**。C1の恒久終了、G6の数学的否定、投稿GOを意味しない。[方針記録](research_state.json)。

## 入力・再現性

対象はLinear H4 1.00 Å、STO-3G、DF rank12、generation-prefix L_D=3、8 system qubits。C1-0で固定snapshotのopaque bytesを確認したがbasis/provider等は不足していた。C1-1はpartition hash不一致で停止し、basis/exact modelの取得は未到達だった。D0はfragment 6・7のweight bit差を特定したものの、元の完全payload/環境がないため唯一の原因とは断定していない。R1は新しいpartition来歴として採用され、旧失敗を変更せず先へ進んだ。[C1-0](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a48b3c8244fb5d932e0d6bf4a5de6bb573ce35ed/docs/research/track_b_c1_0_technical_preflight_20261020/handoff.md)、[C1-1停止](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4cfe1c257709f0ac09351eadce3952b81403cf93/docs/research/track_b_c1_1_input_basis_exact_model_20261020/handoff.md)、[D0診断](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8188c8a50f79ea9d17fddb987c651650b58aa22c/docs/research/track_b_c1_1_d0_partition_hash_20261021/handoff.md)、[R1結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2937e6b655ba3cd77f982d6901d766355a8b92c1/docs/research/track_b_c1_1_r1_new_partition_exact_model_20261021/handoff.md)。

| identity | 保存値・意味 | 固定根拠 |
|---|---|---|
| Snapshot SHA256 | `3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a` | [exact core](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2937e6b655ba3cd77f982d6901d766355a8b92c1/docs/research/track_b_c1_1_r1_new_partition_exact_model_20261021/result/exact_model_core.json) |
| Hamiltonian hash | `de7a549238e3a21f15a84018bef28440c345b31030282c01cf874f3d1d212424` | [input identity](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2937e6b655ba3cd77f982d6901d766355a8b92c1/docs/research/track_b_c1_1_r1_new_partition_exact_model_20261021/result/input_identity.json) |
| State vector hash | `c9aca811b5c023772d148d0331c82958bac6824b593a367cb65a5f89937f4f63`（dtype/shape/C-order規約） | [state identity](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/068f0ae8771fccf28c529bda12a84abe8b7e7648/docs/research/track_b_c1_3_r2_reference_recovery_c1_4_20261021/result/state_identity.json) |
| State metadata hash | `31e63b0104126c85136ee173f1dce7642aee2d272924e70e8b120ac340ab45bd`。上行やNPZ hashと別定義 | [input identity](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2937e6b655ba3cd77f982d6901d766355a8b92c1/docs/research/track_b_c1_1_r1_new_partition_exact_model_20261021/result/input_identity.json) |
| 新R1 partition | `16b35c1dde1a6807d39c733defd87e315e4ac7317a831b240d0f7214e744a834` | [partition](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2937e6b655ba3cd77f982d6901d766355a8b92c1/docs/research/track_b_c1_1_r1_new_partition_exact_model_20261021/result/partition_identity.json) |
| 旧expected partition | `722ec0077b4505d963c03bccdd6311f9dd8b3d3a5ed1b9117f7a6f8fa85a7450`。新hashへ上書きしていない | [D0 identity](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8188c8a50f79ea9d17fddb987c651650b58aa22c/docs/research/track_b_c1_1_d0_partition_hash_20261021/result/hash_identity.json) |
| R1 model SHA256 | `e96ea0faea31bc6a784dfb8dc6f7cf4856789b0256e4a1c53b63ce087c86dc49`（canonical core＋ordered chunk hashの定義。core file SHAとは別） | [model identity](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2937e6b655ba3cd77f982d6901d766355a8b92c1/docs/research/track_b_c1_1_r1_new_partition_exact_model_20261021/result/model_identity.json) |

R1は13 basis・324 labelを保存した。新binary64係数をexact dyadicとして採用し、正のp、Σp=1、λp sign=cのexact等式を保存している。旧PR-2とは135係数hexが異なり、ID順とsignの一致だけで同一実行入力とはしない。exact λ・全pの正本はcore/label chunksへ辿る。[basis index](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2937e6b655ba3cd77f982d6901d766355a8b92c1/docs/research/track_b_c1_1_r1_new_partition_exact_model_20261021/result/basis_registry_index.json)、[component比較](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2937e6b655ba3cd77f982d6901d766355a8b92c1/docs/research/track_b_c1_1_r1_new_partition_exact_model_20261021/result/component_comparison.json)、[exact core](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2937e6b655ba3cd77f982d6901d766355a8b92c1/docs/research/track_b_c1_1_r1_new_partition_exact_model_20261021/result/exact_model_core.json)。

保存full DFへの相対spectral残差 `4.609337135740557e-13` は既存許容差 `1e-10` 内で、保存分類は `MODEL_BINDING_NUMERICALLY_SUPPORTED`。浮動小数点の整合性確認であり、rigorous operator norm certificateはなく `MODEL_BINDING_UNCLOSED` を保持する。stateの厳密ground性・物理準備も未認証。[model binding](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2937e6b655ba3cd77f982d6901d766355a8b92c1/docs/research/track_b_c1_1_r1_new_partition_exact_model_20261021/result/model_binding.json)、[state checks](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2937e6b655ba3cd77f982d6901d766355a8b92c1/docs/research/track_b_c1_1_r1_new_partition_exact_model_20261021/result/state_checks.json)。

## Adapterと限定correctness

| 段階 | 残す成果・失敗 | 確立しないこと |
|---|---|---|
| C1-2〜C1-3 | G6–DF adapterと人工operator/actual adjoint/phase/X/Y/全trial意味論を実装・確認。H4はstate hash定義の取り違えで停止 | H4 correctnessの成功に書き換えない。[旧停止](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e72867b87e12d93d4fc1361f45f105febc6a94c6/docs/research/track_b_c1_2_c1_4_integrated_adapter_validation_20261021/handoff.md) |
| C1-3-R1 | dtype/shape/C-orderのstate hash修正は成立。続いてreference local bytes不一致で停止 | 6 event比較は未到達。[state修正・停止](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/49dae5247ff0ad2c4efb9056e095b70913cc9ddb/docs/research/track_b_c1_3_r1_recovery_c1_4_20261021/handoff.md) |
| C1-3-R2 | `real+1j*imag`がIEEE754 zero符号bitを変えるreference復元不備を修正。13 basis・412局所行列の保存bytes/hashが一致 | bitwise局所復元をexact physical unitary/native誤差certificateにしない。[reference修正](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/068f0ae8771fccf28c529bda12a84abe8b7e7648/docs/research/track_b_c1_3_r2_reference_recovery_c1_4_20261021/REFERENCE_REPAIR.md) |
| 固定6 event | controlled inactive/active/full matrix、相対位相、実adjoint、X/Y、F3 P0〜P5・実proposal/weight/angle guardを通過 | 選択labelは0/2/9/36、動的reference basisは2件。全324 label・全event/path・native coverageへ一般化しない。[correctness](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/068f0ae8771fccf28c529bda12a84abe8b7e7648/docs/research/track_b_c1_3_r2_reference_recovery_c1_4_20261021/CORRECTNESS_REPORT.md) |

固定6 eventのfull controlled Frobenius残差の保存最大は `2.868147051928566e-14`、既存基準は `1e-10`。global phaseでalignしていない。これは選択eventの数値診断であり、rigorous strict native保証ではない。[correctness summary](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/068f0ae8771fccf28c529bda12a84abe8b7e7648/docs/research/track_b_c1_3_r2_reference_recovery_c1_4_20261021/result/summary.json)。C1-4は誤差/confidenceと強い対照の**契約草案**を整理したもので、候補ε/αや十分Nを正式採択した結果ではない。[C1-4草案](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/068f0ae8771fccf28c529bda12a84abe8b7e7648/docs/research/track_b_c1_3_r2_reference_recovery_c1_4_20261021/C1_4_ERROR_CONFIDENCE.md)。

## Whole-path・費用接続から実分子探索へ

C1-4.5ではordered event/weight積、zeroの量子前判定、全trial分母、未完走meanの拒否、first failure/途中prefixの意味論を人工確認した。Ordinary/Partial/P3/Full/P5の5方式は人工L3 supportで接続され、15 tests PASSが保存される。これはH4全方式・測定器・confidenceの証明ではない。[whole-path人工接続](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/af75af03c0b334c29b5a4311bd2b4578a999460c/docs/research/track_b_c1_4_5_c1_5_technical_preflight_20261021/WHOLE_PATH_INTERFACE_AND_TESTS.md)、[5方式のSOURCE対応](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/af75af03c0b334c29b5a4311bd2b4578a999460c/docs/research/track_b_c1_4_5_c1_5_technical_preflight_20261021/SOURCE_CONNECTION_AUDIT.md)。

その後のC1-5AでOrdinary/Fullは各1024 path完走したが、旧P5はsetup後のworker突然終了でSTOPした。処理済trial数・直接のexit/signalはUNKNOWN。P5Rは別SOURCE/authorization/markerによるP5-onlyの1回・retry0で1024 pathを完走した。旧失敗をresume/成功へ変更したものではない。3方式の数値は[費用整理](C1_RESOURCE_EVIDENCE_AND_LIMITATIONS.md)に集約する。[旧失敗](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b022f2ea421649015a4c21ea3c904fe619d0c198/docs/research/track_b_c1_5a_h4_limited_resource_pilot_20261021/FAILURE_REPORT.md)、[P5R terminal](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bb9059bacb697e1105e63c1b3826b0c34886ad15/docs/research/track_b_c1_5a_p5_worker_recovery_20261021/result/terminal.json)。

D1はsampler自体へ参照表を渡さないが、価格取得側で全event/ratio inventory/評価を列挙する。G8 on-demandは観測keyのstrict RZ取得で、全support termination・DF provider費用を保証しない。Closed P5の群normalizer/angle共有からword別の認証native期待費用は得られていない。[非列挙費用の静的監査](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/af75af03c0b334c29b5a4311bd2b4578a999460c/docs/research/track_b_c1_4_5_c1_5_technical_preflight_20261021/NON_ENUMERATIVE_NATIVE_COST_AUDIT.md)。

## G6本体へ保持する既存成果

以下は既存方法ノートの要約で、新しい保証の採択・証明ではない。G6は正有理p、Hermitian involution、odd m、short step等の適用条件付きで維持する。

| 既存成果 | 正式な所在 | C1へ移す際の境界 |
|---|---|---|
| 形式returnの集約、正値parent/child、ordered operator meanの構成 | [v1.1 §§1–3](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/docs/manuscripts/track_b_return_aggregation_native_resource_note_v1_1.md) | channel一致や実分子総費用の改善ではない |
| 局所Green query・ordinary envelope、全global normalizerなしのFull生成 | [v1.1 §3](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/docs/manuscripts/track_b_return_aggregation_native_resource_note_v1_1.md#L130) | 古典query/selector/監査費用は残る |
| 実dyadic πと補正weightのexact関係、係数/角度bias・range・moment | [v1.1 §4/F1–F3](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/docs/manuscripts/track_b_return_aggregation_native_resource_note_v1_1.md#L212) | ideal meanへの無条件exact一致ではない。canonicalの一様認証はFull preflightと別 |
| 正当なzero/rejectionを含む全trial estimator | [v1.1 §§3–4](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/docs/manuscripts/track_b_return_aggregation_native_resource_note_v1_1.md) | technical failure・未完走prefixをzeroにしない。C1ではshotsなし |
| m=5のFull理想受理後ensembleとClosed P5群特殊化 | [v1.1 §5/§9](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/docs/manuscripts/track_b_return_aggregation_native_resource_note_v1_1.md#L336) | Fullはzeroを含む全trial lawが異なり、実digital law/係数近似/費用/Nも一致しない |
| G8の条件付きprovider/native誤差会計、D1–D3/W2b/W2cの接続条件 | [v1.1 §§9–10](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/243eea8e840f3c472753f7935e510391626fd325/docs/manuscripts/track_b_return_aggregation_native_resource_note_v1_1.md#L650) | 小synthetic provider・固定policyの成果をH4 uniform strict nativeへ移さない |

C1固有の肯定的成果は、R1 exact input packet、controlled adapterの限定correctness、whole-path技術接続、3方式の探索的観測である。現在の条件でlogical差が小さくzero削減を観測しなかったことは、その範囲の限定結果として保持する。保存DF指数taskへのrigorous binding、認証expected native cost、十分N、state/readout、全方式公平最適化後の総費用、他分子/時間/次数への移転は `NOT_ESTABLISHED`。G6の数学的構成の無効性・独立新規性の決着とはしない。
