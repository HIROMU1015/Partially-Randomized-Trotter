# 全18round：固定入力と再利用可用性

2026-11-04 JST。静的な保存literal・schema・byte/hash照合のみ。新H4包含・保存値の数値再解析は未実行。

**Partial全18roundの入力は揃い、m0〜m16の包含runnerを準備した。全18roundの局所条件が成立するかは未評価。m17は既存の両軸成立結果をそのまま再利用する。**

## 固定対象とSOURCE

Linear H4／隣接1.00 Å／STO-3G／保存legacy DF12、8 qubit・4電子sector70、Partial `L_D=3`、`δ=0.02`。保存physical stateのnormalized ray、`extract_identity_phase`、`coefficient_atol=1e-12`を維持する。Full L_D12は背景証拠のみ。

- [m17実行報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/34840ce9bf81d985b32764d9bc9d77645ed52e59/docs/research/rpe_m17_rigorous_mean_enclosure_execution_20261103/REPORT.md)：結果 `34840ce9bf81d985b32764d9bc9d77645ed52e59`、科学SOURCE `18fffce48e33136b91eb6ac5a6f9a5f77a08d36f`。
- [V2報告・CSV](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78c49d0b5f1d71e63b5959e78813237923bbc648/docs/research/pr_rpe_v2_branch_success_20261103/REPORT.md)：`78c49d0b5f1d71e63b5959e78813237923bbc648`。μ0はphysicalの`attenuated_event_mean_signal`。
- 元signal/model carrier `4591b8d75a2a6a55c74a51225f0c61648023c8ee`は保存物の固定保持先。元dirty科学実行のclean SOURCEと読み替えない。GitHub上の固定報告・CSV・SOURCEとローカルblobを照合した。

## 各roundの保存条件

以下はV2 CSVのPartial18行の転記。全行K=2であることを確認した。N/α/εは両軸に同じ値が保存されているため表ではまとめた。adapterは軸別に照合し、等しさを一般仮定にしない。β_PF=`0.014050012886294481`、β_RTE=`9.899584668737594e-08`、β_stat=`0.38594988811785885`も全行の保存literalである。

| m | q | r | K | μ0 Re | μ0 Im | N/axis | α/axis | ε/axis |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 1 | 2 | 0.9989255707627144 | 0.04330819375415813 | 510 | 2.8897922699779505e-08 | 0.266146579872968 |
| 1 | 2 | 1 | 2 | 0.9959766962772754 | 0.0865233243291444 | 491 | 5.6410662921482597e-08 | 0.26611036738029376 |
| 2 | 4 | 1 | 2 | 0.9844832938740641 | 0.17235042943249795 | 472 | 1.114810735575654e-07 | 0.26603795717573053 |
| 3 | 8 | 1 | 2 | 0.939502685390124 | 0.3393522369363202 | 454 | 2.2180178685403194e-07 | 0.26589319587097715 |
| 4 | 16 | 1 | 2 | 0.7675053551362614 | 0.6376446757872318 | 435 | 4.431641534385353e-07 | 0.26560390952889584 |
| 5 | 32 | 1 | 2 | 0.18247373758818888 | 0.9787914066646564 | 417 | 8.887836516415881e-07 | 0.2650262807151208 |
| 6 | 64 | 1 | 2 | -0.9247359528392534 | 0.35720745250785724 | 400 | 1.7916902293555355e-06 | 0.26387478899416617 |
| 7 | 128 | 1 | 2 | 0.7275394185256273 | -0.6606451478296738 | 387 | 3.644889634681192e-06 | 0.2615867928908603 |
| 8 | 256 | 2 | 2 | 0.09449303668679321 | -0.9781792465261092 | 366 | 7.450769839139177e-06 | 0.2615865346292552 |
| 9 | 512 | 2 | 2 | -0.9479057044908024 | -0.184862254812331 | 357 | 1.542821014260641e-05 | 0.2570696370881607 |
| 10 | 1024 | 4 | 2 | 0.8949920441076994 | 0.3628877670481872 | 335 | 3.2120735872292006e-05 | 0.2570695101600464 |
| 11 | 2048 | 8 | 2 | 0.6930510865999936 | 0.6725906890249251 | 312 | 6.785171715450883e-05 | 0.25706947842484645 |
| 12 | 4096 | 8 | 2 | 0.027941573675696562 | 0.9322794159963287 | 310 | 0.00014549464823564195 | 0.24826817288558517 |
| 13 | 8192 | 16 | 2 | -0.9310238465245358 | 0.055858061304341 | 284 | 0.00032566760951292903 | 0.2482681575604219 |
| 14 | 16384 | 16 | 2 | 0.8636852798032276 | -0.10401037428276866 | 295 | 0.0007487240052720257 | 0.23155920925457862 |
| 15 | 32768 | 32 | 2 | 0.8450540401972855 | -0.20652854424181677 | 261 | 0.0018373133741434265 | 0.23155920207811798 |
| 16 | 65536 | 32 | 2 | 0.6714622914006043 | -0.34905556143974903 | 297 | 0.004855672662800116 | 0.20143925382202432 |
| 17 | 131072 | 32 | 2 | 0.32902182375994465 | -0.46875529425442597 | 411 | 0.016957089147912167 | 0.15244325851620388 |

## τ・law・CDFの可用性

r/Kとqを照合してsignal gridのpointを選び、physical state labelでstateを選ぶ。V2のsignal selectorとopt2のmodel selectorは[fixed_inputs.json](fixed_inputs.json)に全18行分保存した。**signal側τとmodel側τは保存literalが異なる。平均lawにはm17と整合するsignal側を使い、model側τやnormalizationへ置換しない。** 差の新規数値分析・再最適化は行っていない。

| r | round | signal τ64（保存） | model τ64（保存） | order law/CDF/θ |
|---:|---|---|---|---|
| 1 | m0,m1,m2,m3,m4,m5,m6,m7 | 0.011665408492679006 | 0.011665408492679972 | 未保存。承認後に既存SOURCEで1 law replay |
| 2 | m8,m9 | 0.005832704246339503 | 0.005832704246339986 | 未保存。承認後に既存SOURCEで1 law replay |
| 4 | m10 | 0.0029163521231697514 | 0.002916352123169993 | 未保存。承認後に既存SOURCEで1 law replay |
| 8 | m11,m12 | 0.0014581760615848757 | 0.0014581760615849965 | 未保存。承認後に既存SOURCEで1 law replay |
| 16 | m13,m14 | 0.0007290880307924379 | 0.0007290880307924982 | 未保存。承認後に既存SOURCEで1 law replay |
| 32 | m15,m16,m17 | 0.0003645440153962189 | 0.0003645440153962491 | m17保存law/stepを再利用 |

保存gridにあるのはτ・normalization・signal等のscalarであり、他5群のfull order probability/CDF/θ payloadはこの入力集合にはない。将来の5回のreplayで得るbinary64 probability、NumPy CDF、θ hexを出力に固定する。これは新SOURCE replayとして定義するlawであり、過去のsampler payloadを復元できたという主張ではない。component law/CDFは全群で保存m17のものをそのまま再利用する。

## 共有入力と再利用境界

| 入力 | 利用方法・identity |
|---|---|
| H snapshot | H hash `56e4df83655aa2f2f8132126f2635996516dc2cfb1bf4cb8d95490184ad631e5`。SHA256と固定commitを照合、再load/再構成なし |
| B physical state | commit `e22d7718c47358ff650dbc04e8e72f6ab1c8bf41`、NPZ SHA256 `793300c86417f52569289ce86980c5f9a87292e130199355a0f4f7056225c9c4`。physical fingerprint `a07d6c7fdbbb23a035f9fc54b9956ee79e557a2185e29b1870748b771958be90`はgridと対応 |
| component/basis | m17 `source_replay.json`：218 component・11 basis・4 deterministic block。準備hash `89145d8324bf139eaec67265f3134a5c1a3cebb7ed97792d54ed3221d31f8919`、tail hash `65b31bcf432b703c523753ee4607236a2e9530d93401dcec9255e686cc3f04af`。q/r/Kは準備・basis定義を変えない |
| A、F/R | m17 ZIPの96.jsonに保存されたMatrixBallを半径ごと再利用。F/Rはδ/2の固定halfでrに依存しない。δのconstant/identity位相も既に含む |
| r32 step | 同ZIPのstepをm15/m16に再利用。m17 powered/meanは再計算しない |
| state norm² | m17 RESULTの96-bit区間を共通の同じexport stateに適用し、ψ†ψを再計算しない。既存の[byte対応調査](../rpe_m17_guard_proof_refinement_preflight_20261104/TECHNICAL_FINDINGS.md)を再利用 |
| export_adapter | SOURCE replayの定義とinput対応の根拠として再利用。新runnerではH/DF/stateやbasisを作り直す`replay()`を呼ばない |
| mean_adapter / ball_adapter | event_average・product/power・builder_phase_bound・strict classifyを変更せず再利用 |

exact ideal gates、独立uniform53によるcategorical choice、保存state rayの保証範囲を維持する。過去のcomponent payload同一性、固定PCG64の数学的IID、float64仮想測定、compiled回路、noise、実state preparationは未保証。

## 固定ファイル一覧

全entryのcommit/path/blob/size/SHA256は[fixed_inputs.json](fixed_inputs.json)に固定。以下のSHA256はbyte identity用であり、許可証明の仕組みではない。

| key | path | SHA256 |
|---|---|---|
| V2ROWS | `docs/research/pr_rpe_v2_branch_success_20261103/V2_ROUND_RESULTS.csv` | `e830057532c0d7b5700b4d723ddfb77e6b0734d714177d18706331996d1b42b0` |
| GRID | `artifacts/research_direction_ablation/2026-09-21/wp04_ld3_full_schedule_signal_grid_v1.json` | `56bca166b3fcbf6d5b235b77b8819b6bb0f51a99245fe87357479d3d55fa38d3` |
| OPT2 | `artifacts/research_direction_full_opt2/2026-09-25/wp11_all_r_opt2_coherent_analysis_20260925_065827.json` | `3a86b356a0e602506577141c1c1d7ced7acb06f629818ca5a6bc59fb33633a88` |
| STATE | `artifacts/rpe_fixed_request_measurement_b/2026-11-03/run01/physical_state.npz` | `793300c86417f52569289ce86980c5f9a87292e130199355a0f4f7056225c9c4` |
| H | `artifacts/rte_connected_cluster_cost_validation/h4_sto3g_d100_rank12_ld3_dt0p1_ref4_k2_connected_pilot30_max1500_hold1500_rare375_v1.hamiltonian.npz` | `13e4b10d2347ed900fe4aa4b2238128ebe735a48988e321b9e4fabac4d092a3b` |
| M17RESULT | `artifacts/rpe_m17_rigorous_mean_enclosure/2026-11-03/run01/RESULT.json` | `b58caa9dd8a0267eb20c049636e6d8c0b56d67125912e6b5a6a872334a1a2938` |
| EXPORT | `artifacts/rpe_m17_rigorous_mean_enclosure/2026-11-03/run01/source_replay.json` | `ad7fd3617a524207fbc86a77de471ed6222b010046822a11bd9675edcace1f08` |
| LAWS | `artifacts/rpe_m17_rigorous_mean_enclosure/2026-11-03/run01/categorical_laws.json` | `a75324a1ca014bedf56fbd4e8eacbe6c9829f4bfbc3f8ccf248b5db6cc897e91` |
| OPERATORS | `artifacts/rpe_m17_rigorous_mean_enclosure/2026-11-03/run01/enclosure_operators.zip` | `708095a92f7d3080f595388b80be6b6b9a26b4e4536f416edd8d7434731a5d56` |
| BALL_SOURCE | `docs/research/rpe_m17_rigorous_mean_enclosure_preparation_20261103/ball_adapter.py` | `c222319a65104dbd890f85f62b8ddd25af885ae1556609ffab2df065d0920588` |
| MEAN_SOURCE | `docs/research/rpe_m17_rigorous_mean_enclosure_preparation_20261103/mean_adapter.py` | `4df337055c1d22ee3778d0458230a76ea9f1f170ec59d860082a98a45b728183` |
| DISTRIBUTION_SOURCE | `src/trotterlib/rte.py` | `620a70077eb691281c01706ed726a91e625a98261c6e581c2d96502bc541e618` |

入力不足、固定hash/schema/tuple/physical identity不一致、未対応条件はUNKNOWN/STOP。新しい入力への自動置換・再生成・m17再実行を行わない。
