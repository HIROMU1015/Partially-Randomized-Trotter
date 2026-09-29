# PR-2 V4/S2 development resource comparison

## 結論

結果前に固定したamendment v5どおり、別系列`pr2-rebaseline-de7a5492-v1`のV4 correctnessを通過した後、
H4 linear 1.00 Å、STO-3G、DF rank 12のdevelopment snapshotだけでS2を実行した。正式statusは
`S2_TRANSFER_CANDIDATE_AWAITING_REVIEW`である。rank 6のB2はB0/B1を10%以上の基準で上回ったが、
B3に対する10%以上のmaterial advantageはなく、primary frontierはB2とB3の2候補となった。

この結果で`mandatory_stop_reached=true`、`S3_authorized=false`、`automatic_next_stage=null`である。
held-out H4 1.30 ÅのNPZはloadせず、signal、cost、rankingも評価していない。従ってこれはtransferの確認、
一般的なPR-2優位性、または最終RPE総costの結果ではない。

## 固定条件と証拠identity

- 系：H4 linear 1.00 Å、STO-3G、8 qubit、4 electron、DF rank 12。
- 物理時間：`T=0.8`、`delta=0.1`、`q=8`。
- signal精度：complex 0.05、axis 0.05/sqrt(2)、axis failure 0.025。
- primary cost：状態準備を除くfull measured Hadamard wrapperのcompiled RZ ×解析的shot数。
- compile：Qiskit 1.3.0、optimization level 1、seed 17、coupling mapなし。
- authorization commit：`33a1c0d0a7f6880972f4e0ffdb5b6e7ffe0358e0`。
- serial scientific source commit：`e098c54c78f589055082f9cfc2b13de50c90ca94`。
- parallel execution commit：`16331cc30299a42cc3ddbdd341827fc118c59b0c`。
- specification SHA-256：`a55d7878ab2c201b65cfaaf687806118c56df3a4135026d59811da8931056bcf`。
- machine-readable authorization SHA-256：`ff1283be5c3315777dfd878ec305d76e6013077b3e0c7e04c6b25242092c22a9`。
- V4 result fingerprint：`af79c26262232d7ba7f94b7d261fe730ebaa0b2699ac0455943d621b3ffa3d13`。
- V4 artifact SHA-256：`0396ac6729aa27e212be78cc02ba93ac814b939ea4c646b824da878f30eb6075`。
- S2 result fingerprint：`51fb92fdbcedb67299964eddd25e81c1faeaa55d7f7966765600a05e71d41a49`。
- S2 artifact SHA-256：`bbe665724af438d242569b020ab6148dacce84716a75681949bea22c632dd27c`。

S2は6 worker、`spawn`、各BLAS thread 1で実行した。並列層はcell内容、seed、32/96 trajectory拡張、
段階barrier、canonical結果順を変更しない。実行時sourceはcleanで、artifactが記録するcommitと一致した。

## Primary comparison

| 候補 | corrected bias | attenuation | shots | mean RZ/shot | no-prep RZ work |
|---|---:|---:|---:|---:|---:|
| B0 rank 6 discard | 0.0191207 | 1.000000 | 40,641 | 82,230.0 | 3,341,909,430 |
| B1 rank 12 deterministic | 0.000107029 | 1.000000 | 14,073 | 159,046.0 | 2,238,254,358 |
| B2 rank 6, r=1, K=2 | 0.000107029 | 0.999961 | 14,073 | 87,896.95 | 1,236,973,821 |
| B3 rank 0, r=32, K=4 | 0.000091914 | 0.624771 | 36,032 | 32,722.42 | 1,179,054,305 |

B2/B0のengineering ratio intervalは`[0.370062, 0.370218]`、B2/B1は
`[0.552534, 0.552768]`で、どちらも10%以上の優位基準を満たす。一方、B2/B3は
`[1.035433, 1.063175]`であり、B3のpoint workはB2より約4.7%小さいが、10% materiality基準には
達しない。このため`material_frontier`は`B2-rank6-r1-k2`と`B3-rank0-r32-k4`で、
materially dominating endpointはない。

状態準備を1 shot当たり共通に加える感度では、B2/B3のpoint break-evenは
`P*=2637.620854` RZ-equivalent/shotである。これはsecondary sensitivityで、primary順位を変更しない。

## rank controlと解釈上の注意

rank 6で選ばれた`r=1,K=2`を固定移送したcontrolでは、rank 3 B2のno-prep workは
`688,286,635`、rank 9 B2は`1,791,567,663`だった。rank 3はrank 6 B2より約44.4%、B3より約41.6%
小さい。ただしamendment v5はrank 3/9をcontrolと明記し、primary winner判定への混入を禁止している。
従ってrank 3を事後的なwinnerとは呼ばない。

このcontrolは、rank 6 anchorがglobal split optimumとは限らないことを示す診断である。またV1–V3で
B2-GとB2-Wはrank 3/6/9のordered prefixまで一致した。従ってPR-2を新しいprefix生成法としてではなく、
DF-prefix部分ランダム化がどのsplit・finite-RTE・状態準備cost条件で資源frontierに残るかを調べる
適用条件・resource-map研究として再定義する根拠が強くなった。

## 実行範囲と限界

- development NPZ load 1、held-out NPZ load 0。
- signal evaluation 28、full wrapper compile 3,204、random trajectory compile 1,600。
- molecular calculation 0、量子backend shot 0。
- dedicated testsは15 passed、fail/error/skip 0。JUnit SHA-256は
  `4cc2ff235e4d3f2da78806546e3fc466ad4f40b045c00a25f843d43671a58e29`。
- engineering intervalはmean ± 2 SEであり、formal confidence intervalではない。
- backend noise、coupling map、状態準備の実回路、held-out transfer、別分子・basis・PF、H12、長RPE、
  最終総costは未評価である。
- 本artifactはlocal validationであり、immutable CIまたは独立外部再現ではない。

## 次の判断

amendment v5が固定した停止点へ到達したため、次は追加計算ではなく研究方針の再検討である。
held-outを開く前に、少なくとも次の二案を明示的に選ぶ必要がある。

1. 現行rank 6 anchorを凍結したtransfer試験としてS3を別authorizationで行う。
2. rank 3 controlを受けて主RQをsplit/resource mapへ変更し、baseline、rank policy、state-preparation
   sensitivity、held-outの役割を結果前に再固定する。

現結果は第2案を支持するが、どちらを採る場合も本S2結果を見た後のprotocol変更として明示し、
同じheld-outを新しい設計の選択と最終評価の両方に使わない。方針を固定するまでS3を実行しない。
