# H2/H4/H6共通決定論PF探索：588候補の固定PR/PF費用比較完了

2026-10-22 JST。**H6の不足63 metadata取得と固定276候補の取得・適格性/shot・必要なcompile・会計・終了監査PASS。H2/H4を再計算せず参照結合し、共通588候補すべての評価が揃った。local commit後mandatory STOP、push0。**

以下は保存PR探索標本と有限共通格子PFの点費用比較であり、母平均順位の保証ではない。H6各精度の方向・exact費用は表とmachine-readable比較行を正とする。

## 固定条件・SOURCE

linear H2/H4/H6、隣接1 Å、STO-3G、tol-only DF threshold1e-8、保存rank3/10/19、T=.8、P=0、q=1/2/4/8、δ=.8/.4/.2/.1。S2/Yoshida4/m5_bestの元binary64係数・signed scheduleを維持。full cosine/sine Hadamard wrapper、symmetric-directional control、元scalar一回、Qiskit1.3.0・rz/sx/x/cx・opt1・seed17。backend/coupling/layout未指定。H2既存scalar guardは旧系列のまま、H6に新たに移していない。

| 系 | B0/B0I L_D | Full | 登録数 | 結果SOURCE |
|---|---|---:|---:|---|
| H2 rank3 | 0..2 | 3 | 84 | 1b35f9ac2efab4c15bc00ee2d3c81643aca422f9 |
| H4 rank10 | 1..9 | 10 | 228 | e2179f68f6979f3ad26392bff877fb422ddfb744 |
| H6 rank19 | 5..15 | 19 | 276 | bdf6772aade7063c65d69f24dfdc5e303a2819a2 |

H4 DF12、q16、未登録prefix、追加公式は含めない。最良は今回の固定共通格子内の最良。H2 prefix0/Full aliasを推測統合しない。保存PRは各系outer32/K4/prefix1/3/5のEXPLORATION 3公式×4qだけで、pilot・旧cohort・118確認標本を混合しない。新しいPR sampling・統計・単価・shot変更は0。

H6 metadata科学SOURCEは `cba2a82021b17cba0eec31b03fb19d80f35225f6`、共通科学SOURCEは `bdf6772aade7063c65d69f24dfdc5e303a2819a2`。起点0b1886f53、専用branch `pf-common-h6-metadata-execution-20261022`。旧H2/H4のSOURCE/raw/grant/ledger/marker/STOPはbyte同一のまま参照結合。旧H6 INPUT_STOPと失敗stageも正式履歴として保全した。

H6 SIGNAL/QUALIFYと中断前187価格のSOURCEは `b7662487185086aeaa1b298f82089e194b7270db`。回復22価格とACCOUNTは上記の新SOURCE。異なる取得SOURCEとgrantを混同せず、原科学入力と数値workerは同一性を確認した。

H6原snapshot SHA `99440a59d903dfd6330e786d84a956f1dd8a687a592295a1cd771c839769b005` がreceipt/bindingと一致。Hamiltonian hash `223628b06c794f4a2a7a84db532e19ed60915f7e86898de66347129a5aeaffa9`、state hash `664c63350c2510518b0ec561f804641447d39f75e68b960192c31d37540265f3` を原配列から検査し、保存reference・full preparation・runtime/compiler/provider sealと接続した。H6は元protocolで採用した保存weighted-Hermitian completionのDF Hamiltonianをそのまま使用し、化学入力を再生成した系列ではない。今回SCF/DF/state/reference新生成0。H2/H4は以前に科学配列hash一致を確認した再生成容器であり、原snapshot復元とは区別する。

## 不足63 metadataの正当な取得

旧B0I serializerはscalar信号のみ回転し、途中norm/closureを保存しなかった。保存B0 scalarから中間binary64 vectorを推測せず、旧STOPの63 IDだけを別契約で再観測。元merged S2/Yoshida4 schedule・compact_signal・独立recurrence_pathsを使い、native corrected/rawを別実行、各primitiveのsector leakage<1e-12を検査した。scalar/作用順序/referenceは元のまま。最大normは途中の原観測点を保持し最終B0I位相作用後も観測。closureは保存vectorからnorm(raw−corrected)を計算した実測値で、0補完ではない。

52候補は旧corrected/reference/signed biasをそのまま保存し、新観測だけ結合。原rawが未回収だった11候補は新signalとして明記し、原本復元とは呼ばない。保存u/marginを下回らない。独立監査では全63 archiveの原SHA復元、保存vector closure、旧52値のliteral不変を確認した。

- 旧corrected最大差：5.9337240e-14。
- reference最大差：2.7103316e-14。
- signed bias最大差：6.2973921e-14。
- 独立forward vector最大差：2.3596033e-12。
- 最大途中norm：1.0000000000001967。
- raw/corrected vector closure最大：0.0（別実行vectorの実測）。

metadata実使用はnative corrected63/raw63、独立forward63、native primitive build/作用各18736、forward作用9368、native outer scalar476/forward238、最終vector位相作用189、closure63、旧値比較52、独立sector matrix/eigh各16、tail scalar11、snapshot/reference/oracle/full preparation各1。63 worker全exit0・回収、wall約78.99秒。これらは従来102 signal予算とは完全別枠。

## H6取得・科学call・再利用

| 項目 | H6実使用 | 旧固定上限 |
|---|---:|---:|
| 旧観測の再利用 | 111候補 | 保存値のみ |
| 追加metadata | 63候補（旧52+新11） | 別承認63 |
| 本来の新signal | 102候補 | 102 |
| 基礎signal / B0I overlay | 51 / 51 | 51 / 51 |
| 独立forward | 51 | 51 |
| 適格性/shot | 1,380 endpoint | 1,380 |
| 固有の新価格候補/軸 | 209 / 418 | 固有pool 224 / 448 |
| compile予約の実呼出累積 | 438軸 | 448軸 |
| 同native保存price使用 | 30候補 | identity照合済みのみ |
| 新PR sampling | 0 | 0 |

H6全276候補・5精度の適格性が確定し、少なくとも1精度で適格な239候補の必要費用が揃った。全精度不適格37候補のcompileは不要。全588候補の評価完了は、不適格候補までcompileした意味ではない。共通段階の新signal全系計186、固有の新価格全系計378候補/756軸、compile予約の実呼出全系計776軸、適格性2940 endpoint。別metadata予算の新11signalと52再観測は別記録。

H6 SIGNALはsigned S2 piece1419、spectral/forward block作用各30588、prefix preparation9、expm構築448。同一行列/timeのexpm再利用で重複計算だけ削減、作用数・signed順序不変。COSTは旧中断分と新grant分を非返却で合算し、signed piece4854（旧3625から追加1229を承認）、native evolution/low/high/独立occupation各231（旧224から各追加7）、wrapper build452（旧448から追加4）、uncompiled作用441・compile438・compiled作用434（各448以内）。context各2、prefix preparation計17。固有の新価格209候補418軸は旧完成187＋新取得22で、compile実呼出438軸と区別する。正常stageのledger=reserved counter=receiptでgrant上限内。中断stageのledger/counterとSTOPは別保存し、返却0。

旧legacy217候補の価格は今回も同native identityを証明できず再利用0。H6同native保存36 pairのうち30適格pairを使い、残る6は今回不適格。H2/H4保存22を含む使用再利用は全系52 pair。旧profile/model推定を新しい直接実測へ代用しない。

## 共通政策と費用の定義

通常signed bias、ε=.05/.01/.005/.001/.0005、allowance倍率1、α_axis=.025。u=max(floor1e-9,10closure+64eps(1+norm)(1+stages),saved u)。log_B marginはfloor1e-12とroundoff式・saved margin以上、u/headroom≤.01。凍結binary64 shotと基本演算を段階ごとbinary64へ丸めるDecimal経路で照合。libm値は共有し、独立超越関数実装の照合とはしない。経験的allowanceはreference条件付きで厳密certificateではない。

G=N_cos×RZ_cos+N_sin×RZ_sin、状態準備費用P=0。PFは整数の決定論的会計、PRは保存exact有理数mean、不偏分散、paired cosine/sine共分散込みSEとnをそのまま参照。表の±は1 SEで母平均CIではない。量子測定shotを実行した結果ではなく、有限T複素signalのanalytic shot込みcompiled RZ予測。全energy/QPE/RPE round最適化の最終費用ではない。

## 比較B：3公式全体のPF最良対保存PR最良

PR削減率=100×(1−PR平均/PF費用)。負値はPF低費用。

| 系 | ε | PF最良 | PF RZ | 保存PR最良 | PR平均RZ ± SE | PR削減率 |
|---|---:|---|---:|---|---:|---:|
| H2 | 0.05 | S2 B0I p2/q1 | 6.3866960e+06 | S2 q1 | 1.5459953e+07 ± 1.756e+05 | -142.0650% |
| H2 | 0.01 | S2 FULL p3/q1 | 2.2650324e+08 | S2 q1 | 5.2550629e+08 ± 5.969e+06 | -132.0083% |
| H2 | 0.005 | S2 FULL p3/q2 | 1.4143274e+09 | Yoshida4 q1 | 2.5820199e+09 ± 2.136e+07 | -82.5617% |
| H2 | 0.001 | S2 FULL p3/q4 | 7.2245915e+10 | Yoshida4 q2 | 1.1106603e+11 ± 9.908e+08 | -53.7333% |
| H2 | 0.0005 | S2 FULL p3/q4 | 3.7010164e+11 | Yoshida4 q2 | 4.6314865e+11 ± 4.131e+09 | -25.1409% |
| H4 | 0.05 | S2 B0I p5/q1 | 7.5021842e+07 | S2 q1 | 1.2578331e+08 ± 1.237e+06 | -67.6623% |
| H4 | 0.01 | S2 B0 p7/q2 | 5.5687529e+09 | S2 q2 | 6.5413330e+09 ± 6.396e+07 | -17.4650% |
| H4 | 0.005 | S2 B0 p7/q4 | 3.6121389e+10 | Yoshida4 q2 | 4.1925721e+10 ± 2.830e+08 | -16.0690% |
| H4 | 0.001 | Yoshida4 B0I p7/q2 | 1.5485842e+12 | m5_best q1 | 1.3161914e+12 ± 3.561e+09 | +15.0068% |
| H4 | 0.0005 | S2 B0 p7/q8 | 8.4146011e+12 | m5_best q1 | 5.2697582e+12 ± 1.426e+10 | +37.3736% |
| H6 | 0.05 | S2 B0I p7/q1 | 3.7496872e+08 | S2 q1 | 5.7705136e+08 ± 4.808e+06 | -53.8932% |
| H6 | 0.01 | S2 B0 p11/q2 | 2.9717950e+10 | S2 q2 | 3.1110569e+10 ± 2.447e+08 | -4.6861% |
| H6 | 0.005 | S2 B0 p11/q4 | 1.6444846e+11 | S2 q4 | 1.7177421e+11 ± 1.787e+09 | -4.4547% |
| H6 | 0.001 | Yoshida4 B0I p11/q2 | 7.8694952e+12 | m5_best q1 | 5.4200842e+12 ± 1.286e+10 | +31.1254% |
| H6 | 0.0005 | m5_best B0I p11/q1 | 3.6785630e+13 | m5_best q1 | 2.1718526e+13 ± 5.152e+10 | +40.9592% |

## 比較A：同じ公式内

| 系 | ε | 同公式 | PF最良 | PF RZ | 保存PR q / 平均RZ ± SE | PR削減率 | n |
|---|---:|---|---|---:|---|---:|---:|
| H2 | 0.05 | S2 | S2 B0I p2/q1 | 6.3866960e+06 | q1 / 1.5459953e+07 ± 1.756e+05 | -142.0650% | 96 |
| H2 | 0.05 | Yoshida4 | Yoshida4 FULL p3/q1 | 1.7762949e+07 | q1 / 2.3003550e+07 ± 1.903e+05 | -29.5030% | 60 |
| H2 | 0.05 | m5_best | m5_best FULL p3/q1 | 6.2869592e+07 | q1 / 4.8382205e+07 ± 1.651e+05 | +23.0436% | 33 |
| H2 | 0.01 | S2 | S2 FULL p3/q1 | 2.2650324e+08 | q1 / 5.2550629e+08 ± 5.969e+06 | -132.0083% | 96 |
| H2 | 0.01 | Yoshida4 | Yoshida4 FULL p3/q1 | 4.6704208e+08 | q1 / 6.0486398e+08 ± 5.005e+06 | -29.5095% | 60 |
| H2 | 0.01 | m5_best | m5_best FULL p3/q1 | 1.5716008e+09 | q1 / 1.2094629e+09 ± 4.128e+06 | +23.0426% | 33 |
| H2 | 0.005 | S2 | S2 FULL p3/q2 | 1.4143274e+09 | q2 / 3.3125560e+09 ± 2.629e+07 | -134.2142% | 71 |
| H2 | 0.005 | Yoshida4 | Yoshida4 FULL p3/q1 | 1.9937162e+09 | q1 / 2.5820199e+09 ± 2.136e+07 | -29.5079% | 60 |
| H2 | 0.005 | m5_best | m5_best FULL p3/q1 | 6.2864840e+09 | q1 / 4.8379129e+09 ± 1.651e+07 | +23.0426% | 33 |
| H2 | 0.001 | S2 | S2 FULL p3/q4 | 7.2245915e+10 | q4 / 1.7101562e+11 ± 1.433e+09 | -136.7132% | 33 |
| H2 | 0.001 | Yoshida4 | Yoshida4 FULL p3/q2 | 8.9935928e+10 | q2 / 1.1106603e+11 ± 9.908e+08 | -23.4946% | 34 |
| H2 | 0.001 | m5_best | m5_best FULL p3/q1 | 1.5718181e+11 | q1 / 1.2096294e+11 ± 4.128e+08 | +23.0427% | 33 |
| H2 | 0.0005 | S2 | S2 FULL p3/q4 | 3.7010164e+11 | q4 / 8.7589411e+11 ± 7.338e+09 | -136.6631% | 33 |
| H2 | 0.0005 | Yoshida4 | Yoshida4 FULL p3/q2 | 3.7503876e+11 | q2 / 4.6314865e+11 ± 4.131e+09 | -23.4935% | 34 |
| H2 | 0.0005 | m5_best | m5_best FULL p3/q1 | 6.2882646e+11 | q1 / 4.8392812e+11 ± 1.652e+09 | +23.0427% | 33 |
| H4 | 0.05 | S2 | S2 B0I p5/q1 | 7.5021842e+07 | q1 / 1.2578331e+08 ± 1.237e+06 | -67.6623% | 255 |
| H4 | 0.05 | Yoshida4 | Yoshida4 B0I p7/q1 | 2.5509898e+08 | q1 / 2.4552941e+08 ± 2.714e+06 | +3.7513% | 54 |
| H4 | 0.05 | m5_best | m5_best B0I p7/q1 | 8.5488470e+08 | q1 / 5.2601430e+08 ± 1.423e+06 | +38.4696% | 33 |
| H4 | 0.01 | S2 | S2 B0 p7/q2 | 5.5687529e+09 | q2 / 6.5413330e+09 ± 6.396e+07 | -17.4650% | 143 |
| H4 | 0.01 | Yoshida4 | Yoshida4 B0I p7/q1 | 9.5594480e+09 | q1 / 9.1916591e+09 ± 1.016e+08 | +3.8474% | 54 |
| H4 | 0.01 | m5_best | m5_best B0I p7/q1 | 2.1387964e+10 | q1 / 1.3150757e+10 ± 3.558e+07 | +38.5133% | 33 |
| H4 | 0.005 | S2 | S2 B0 p7/q4 | 3.6121389e+10 | q4 / 4.2505737e+10 ± 4.839e+08 | -17.6747% | 62 |
| H4 | 0.005 | Yoshida4 | Yoshida4 B0I p7/q2 | 4.9157543e+10 | q2 / 4.1925721e+10 ± 2.830e+08 | +14.7115% | 33 |
| H4 | 0.005 | m5_best | m5_best B0I p7/q1 | 8.5636518e+10 | q1 / 5.2607815e+10 ± 1.423e+08 | +38.5685% | 33 |
| H4 | 0.001 | S2 | S2 B0 p7/q8 | 1.7852634e+12 | q8 / 2.1881646e+12 ± 1.563e+10 | -22.5682% | 48 |
| H4 | 0.001 | Yoshida4 | Yoshida4 B0I p7/q2 | 1.5485842e+12 | q2 / 1.3364622e+12 ± 9.021e+09 | +13.6978% | 33 |
| H4 | 0.001 | m5_best | m5_best B0I p7/q1 | 2.1580438e+12 | q1 / 1.3161914e+12 ± 3.561e+09 | +39.0100% | 33 |
| H4 | 0.0005 | S2 | S2 B0 p7/q8 | 8.4146011e+12 | q8 / 1.1285212e+13 ± 8.061e+10 | -34.1146% | 48 |
| H4 | 0.0005 | Yoshida4 | Yoshida4 B0I p7/q2 | 9.3426147e+12 | q4 / 7.6639679e+12 ± 5.896e+10 | +17.9676% | 33 |
| H4 | 0.0005 | m5_best | m5_best B0I p7/q1 | 8.7196107e+12 | q1 / 5.2697582e+12 ± 1.426e+10 | +39.5643% | 33 |
| H6 | 0.05 | S2 | S2 B0I p7/q1 | 3.7496872e+08 | q1 / 5.7705136e+08 ± 4.808e+06 | -53.8932% | 167 |
| H6 | 0.05 | Yoshida4 | Yoshida4 B0I p11/q1 | 1.1030436e+09 | q1 / 1.4840495e+09 ± 1.231e+07 | -34.5413% | 33 |
| H6 | 0.05 | m5_best | m5_best B0I p11/q1 | 3.4904357e+09 | q1 / 2.1644112e+09 ± 5.135e+06 | +37.9902% | 33 |
| H6 | 0.01 | S2 | S2 B0 p11/q2 | 2.9717950e+10 | q2 / 3.1110569e+10 ± 2.447e+08 | -4.6861% | 110 |
| H6 | 0.01 | Yoshida4 | Yoshida4 B0I p11/q2 | 4.9844355e+10 | q2 / 4.7936571e+10 ± 2.755e+08 | +3.8275% | 33 |
| H6 | 0.01 | m5_best | m5_best B0I p11/q1 | 8.7436771e+10 | q1 / 5.4115394e+10 ± 1.284e+08 | +38.1091% | 33 |
| H6 | 0.005 | S2 | S2 B0 p11/q4 | 1.6444846e+11 | q4 / 1.7177421e+11 ± 1.787e+09 | -4.4547% | 33 |
| H6 | 0.005 | Yoshida4 | Yoshida4 B0I p11/q2 | 2.0855166e+11 | q2 / 2.0114557e+11 ± 1.156e+09 | +3.5512% | 33 |
| H6 | 0.005 | m5_best | m5_best B0I p11/q1 | 3.5066354e+11 | q1 / 2.1649880e+11 ± 5.136e+08 | +38.2602% | 33 |
| H6 | 0.001 | S2 | S2 B0 p11/q8 | 7.8764969e+12 | q8 / 9.2899328e+12 ± 7.621e+10 | -17.9450% | 33 |
| H6 | 0.001 | Yoshida4 | Yoshida4 B0I p11/q2 | 7.8694952e+12 | q2 / 7.8258578e+12 ± 4.497e+10 | +0.5545% | 33 |
| H6 | 0.001 | m5_best | m5_best B0I p11/q1 | 8.9535948e+12 | q1 / 5.4200842e+12 ± 1.286e+10 | +39.4647% | 33 |
| H6 | 0.0005 | S2 | S2 B0 p11/q8 | 4.0345884e+13 | q8 / 5.8894876e+13 ± 4.832e+11 | -45.9749% | 33 |
| H6 | 0.0005 | Yoshida4 | Yoshida4 B0I p11/q4 | 3.8164238e+13 | q4 / 3.2773436e+13 ± 1.714e+11 | +14.1253% | 33 |
| H6 | 0.0005 | m5_best | m5_best B0I p11/q1 | 3.6785630e+13 | q1 / 2.1718526e+13 ± 5.152e+10 | +40.9592% | 33 |

各精度の6者（各公式のPF最良とPR点最良）の全順位は[15精度順位](../../artifacts/resource_applicability/track_a_h246_pf_common_final/2026-10-22/precision_rankings_15.json)、exact費用・保存PRの分散/SE/nは[60比較CSV](../../artifacts/resource_applicability/track_a_h246_pf_common_final/2026-10-22/comparisons_60.csv)と[完全JSON](../../artifacts/resource_applicability/track_a_h246_pf_common_final/2026-10-22/comparisons_60.json)。欠測候補を除いたsubset最良への置換なし。

## 旧比較との順位変化・内訳

旧保存PR同士の選択・費用は不変。以前のH2約47%、H4約53%、H6約63%削減は高次PR対S2 PRであり、最良決定論PFに対する主張ではない。旧m5疎格子PFだけを分母にした比較とも区別する。

H6旧m5部分集合に対する比較と、今回の全公式共通PF分母を比較すると次の通り。

| ε | 旧m5疎格子PF最良 | 旧PR削減率 | 新全公式PF最良 | 新PR削減率 |
|---|---|---:|---|---:|
| 0.05 | B0I p11/q1 | +83.4676% | H6_B0I_7_1_2nd | -53.8932% |
| 0.01 | B0I p11/q1 | +64.4194% | H6_B0_11_2_2nd | -4.6861% |
| 0.005 | B0I p11/q1 | +51.0145% | H6_B0_11_4_2nd | -4.4547% |
| 0.001 | B0I p11/q1 | +39.4647% | H6_B0I_11_2_4th | +31.1254% |
| 0.0005 | B0I p11/q1 | +40.9592% | H6_B0I_11_1_4th(m5_best) | +40.9592% |

分母変化を、旧m5疎格子を共通政策で評価した費用/旧保存費用、m5 prefix拡張後/旧m5疎格子、全公式最良/m5格子最良の3比へexact有理数で分解した。これらの積が新PF/旧PFに完全一致する。旧m5には同一native価格を使い、旧profile価格を移送しない。H4分も機械表へ保存。有限保存値の入れ子集合比較であり、探索範囲だけの単独因果効果や母平均保証は主張しない。

| ε | 政策・保存算術比 | m5 prefix拡張比 | S2/Yoshida4を含む公式拡張比 |
|---|---:|---:|---:|
| 0.05 | 1.00000000 | 1.00000000 | 0.10742748 |
| 0.01 | 1.00000000 | 1.00000000 | 0.33987932 |
| 0.005 | 1.00000000 | 1.00000000 | 0.46896368 |
| 0.001 | 1.00000000 | 1.00000000 | 0.87892018 |
| 0.0005 | 1.00000000 | 1.00000000 | 1.00000000 |

[旧比較15行](../../artifacts/resource_applicability/track_a_h246_pf_common_final/2026-10-22/old_vs_common_15.json)、[分母分解10行](../../artifacts/resource_applicability/track_a_h246_pf_common_final/2026-10-22/nested_PF_cost_change_decomposition_10.json)。

## 技術停止・資源・監査

前回のmetadata stageは0次元scalar shape検査の不具合でsnapshot_restore1後にSTOP。その消費・grant/marker/ledger/STOPを保全し返却0。今回の明示継続指示を新SOURCE/pack/grantへ結合した。修正後の63取得は新grantの一回実行。後続pack準備では汎用係数helperにm5がなく停止したが、科学grant前のため科学call0。元m5 signed scheduleを参照して新packへ封印した。消費済みgrantのretry/resumeなし。


その後のCOSTは完成187pairを残してprocessが消失し、元の停止時刻・exit/reap・原因は不明。旧STOPと394 compile予約を保持した。利用者の「計算を再開して」を直前に提示したfresh22追加内部予算案へ結合し、新SOURCE・新pack・新COST grantで22候補44軸だけを再取得した。SIGNAL/QUALIFYは旧成功receiptを参照し再計算0。ACCOUNTは新COST成功後に独立grantで実施。旧中断grantは再使用せず成功receiptを補作しない。自分専用の一時user serviceで起動元と実行を分離し、自動再起動・同grant retry/resumeなし。旧187pairは保存byteとvector/gateの受入監査に基づいて参照し、旧parentの正常exit/reapを証明したとは表現しない。

metadata監査readerの来歴修正を分離し、実行SOURCEはGitからarchive化した。作者による別数値経路チェックであり、独立GPT/Claude科学レビューや外部再現ではない。旧STOPを成功へ書き換えていない。

実affinity128、空きRAM約1TB・disk約124GBのfresh観測からmetadata63、signal51、旧COST127、新COST22 workerを選択。新COSTは残る22候補を全て並列化。内側BLAS/compiler1、共有immutable context/preparationを再利用。固定の低いworker/CPU/RAM上限や共有設定変更・他ユーザー操作は0。

H6 wall秒：SIGNAL 381.122、QUALIFY 1.613、新COST 6393.861、ACCOUNT 2.401。新COSTの22 workerについてexit0・同親join/reap・残存0を確認。旧187pairを取得した中断parentのexit/reapは不明のまま保全。受理raw archive 1254件の原byte SHA復元、全native gate/phase alignmentなし、保存shot integer-ceiling、別Fraction整数費用、80桁Decimal比、完全格子最小値、ledger上限を監査PASS。旧COSTのwall完了値は不明で補完しない。

H6 native誤差最大値は[終了監査](../../artifacts/resource_applicability/track_a_pf_common_H6_results/2026-10-22/end_audit.json)に各check別で保存。compiled vectorは元CT01<1e-7、他native gate<1e-9を保持し、allowanceへ流用しない。

## 統計的限界と終了

有限標本の探索選択効果は補正していない。保存SEが小さくても母平均順位を保証しない。全PR群でS4未観測、rare event費用/分散/tail上界はnullで0ではない。PFにはMonte Carlo SEがないが、reference/bias/normalizationの診断は経験的。今回の新PF分母との独立確認samplingは0、前回118確認はPR対PRであり新PF比較の確認ではない。

[metadata取得・旧52照合](../../artifacts/resource_applicability/track_a_h6_pf_metadata_completion/2026-10-22/ending_audit.json)、[H6全候補](../../artifacts/resource_applicability/track_a_pf_common_H6_results/2026-10-22/candidate_status.json)、[588登録](../../artifacts/resource_applicability/track_a_h246_pf_common_final/2026-10-22/candidate_status_588.csv)、[全契約・SOURCE・入力SHA・grant/ledger/marker](../../artifacts/resource_applicability/track_a_pf_common_H6_results/2026-10-22/execution_contract_projection.json)、[統合監査](../../artifacts/resource_applicability/track_a_h246_pf_common_final/2026-10-22/independent_saved_result_checks.json)を保存した。原private raw/vector/QPY/NPZ archiveと旧grant・STOPを保持。H2/H4はbyte参照結合のみ。

**588候補の固定比較・監査・保存を完了してローカルcommit後mandatory STOP。GitHub push、q16、候補追加、新規PR sampling、詳細科学レビュー、production、論文本文作成は0。**
