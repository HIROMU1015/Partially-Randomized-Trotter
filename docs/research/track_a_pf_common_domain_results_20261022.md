# H2/H4/H6共通決定論PF探索：追加取得・固定PR/PF費用比較

2026-10-22 JST。**H2 84候補・H4 228候補は全4 stageと終了監査PASS。H6 276候補は原metadata不足で科学実行前STOP。3系588候補の登録状態を保存し、ローカルcommit後mandatory STOP。**

共通域の3公式を含む最良PFと保存PR探索点最良を比較すると、H2は5精度すべてPFが低費用。H4はε=.05/.01/.005でPFが低費用、.001/.0005ではPRが15.01%/37.37%低費用だった。H6全域順位はUNRESOLVEDであり、取得subsetを全域最良へ置換しない。同じm5内のPR優位を、全公式PFに対する優位と混同しない。

## 固定条件と登録域

linear H2/H4/H6、隣接1 Å、STO-3G、tol-only DF threshold1e-8、保存rank3/10/19、生成順・signed weights・後段cutoff0、T=.8、P=0。δ=T/q=.8/.4/.2/.1、元S2/Yoshida4/m5_best binary64係数とsigned scheduleを保持した。Full cosine/sine Hadamard wrapper、symmetric-directional control、元constantとB0I tail scalarを一回。Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、backend/coupling/layout未指定。compiler options hash `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。H2は保存PRと同じ既存symbolic-zero compiler scalar guardを使い、gate stream不変をaxisごとに検証した。

| 系・DF rank | B0/B0I prefix L_D | Full | q | 1公式 / 3公式 |
|---|---|---:|---|---:|
| H2 / 3 | 0,1,2 | 3 | 1,2,4,8 | 28 / 84 |
| H4 / 10 | 1..9 | 10 | 1,2,4,8 | 76 / 228 |
| H6 / 19 | 5..15 | 19 | 1,2,4,8 | 92 / 276 |

H4 DF12・q16・未登録prefix・追加公式を混合せず、旧結果を保全。H2 prefix0/Fullのalias統合なし。「最良」はこの共通q≤8格子だけの最良で、歴史的な全域最良ではない。PRは保存outer32/K4、prefix1/3/5、3公式×4qのEXPLORATION cohortのみ。旧piece-r2/pilot/独立確認をpoolせず、候補・標本数・seed・SOURCE・単価・shot・統計は変更していない。

## SOURCE・承認・入力

基準commit `3728a618d7f09e67a070632ded1f27dab54b21a4`。H4/H6 branch `pf-common-domain-execution-20261022`、科学SOURCE `e2179f68f6979f3ad26392bff877fb422ddfb744`。H2は別worktree/branch `pf-common-h2-execution-20261022`、SOURCE `1b35f9ac2efab4c15bc00ee2d3c81643aca422f9`、結果commit `f36402f90bcfcbcc4d556c7a952837dc5d7dd291`。監査readerは科学コードと別commitで追加した。計算開始後に科学SOURCEを変更していない。

利用者の今回の包括承認を、新しい系別契約・承認receipt・段階別grantへ結合した。旧DRAFT・旧STOP・旧grantは不変。正常終了stageの成功receiptを条件に、SIGNAL→QUALIFY→COST→ACCOUNTへ各別grantで進行した。8 grantは全て異なるID・exclusive one-shot marker・非返却ledger・上限counter・終了receiptを持ち、同grant retry/resumeなし。H6はmetadata gateでSTOPし科学stage grantは0。

前回移行の再生成H2/H4 snapshotを再利用し、今回SCF/DF/state/reference新生成0。旧容器とSHAは異なるが、保存された科学配列、Hamiltonian/stateのhashは旧PR入力と完全一致。constant/one_body/lambdas/g_matrices/sector indices/full・sector stateの旧hash比較、前回forward/reference成功receipt、今回のnative保存signal照合を接続した。同一原本復元と表現しない。H6は原snapshot SHA一致だが、必須観測metadataの欠測は解消しない。

| 系 | 旧snapshot SHA | 現容器SHA | 同等性の位置づけ |
|---|---|---|---|
| H2 | 62c6481bd1f8b05ec3b399b3a5293873796737866409049d808e8fe432dcacaa | 8bb1aa35cf80266dd4aa1552cc65097770dfe67c690c2e292860923a5b4ae2f9 | 旧保存科学配列hash一致の再生成 |
| H4 | 12219b7d14dc2a275853314269e5e8628005ed72b42d16bc4d3831c8ea515c96 | ed9b5a136e54a47047b6458ebb7ccb6c9b9ef699cdd534ae1f9f18da5310a520 | 同上 |

元providerの固定SOURCE、主要runtime binary、Qiskit/numpy/Python、既存科学68-file系のsealを再照合した。H2 D70 blob `2b534c7425af6d67bc92c7423a422ac368d9ad07` のsnapshot/Hamiltonian/state/compilerはouter32 PRと一致し、48 signalを再利用。旧Yoshida4 prefix0の8値は単一block signed scheduleの相違を確認し、新予算内で再取得して原値を保全した。[H2詳細](track_a_pf_common_H2_results_20261022.md)に新旧差を記録。旧D70価格・shotは無条件転記していない。

## H6原metadata回収とSTOP

当初未回収28候補のうち17（B0 11・Full 6）を、固定Git blob `9bc570495d256d55868ddd658fff57e8a7c55669` から回収。原summary SHA256 `b9f6cb3dc1a168ddc409833c78a617a7ddd05a3a41e0665e9dec47346d516dfe` を確認した。残る11 B0Iだけでなく、既存52 B0Iの原serializerも`maximum_intermediate_norm`/`raw_corrected_vector_closure`を出力していないことが判明した。

計63候補について、共通allowanceに必要な原観測がない。旧scalar signalや保存shotから推測・0補完せず、今回150枠外の旧signal再生成を追加しない。利用可能な既存root・handoff/archive・固定Git資料を限定確認したが回収不能。H6独立科学stageをSTOP。原観測111、metadata欠落63、新signal未実行102の計276候補を区別した。[回収記録](../../artifacts/resource_applicability/track_a_pf_common_results/2026-10-22/H6/metadata_recovery.json)、[不足IDとSTOP](../../artifacts/resource_applicability/track_a_pf_common_results/2026-10-22/H6/H6_INPUT_STOP.json)。全H6 formula/全公式の5精度最良はUNRESOLVED。H2/H4実行を妨げていない。

## 費用identityと追加取得

旧217候補（H4 97/H6 120）のSOURCE・入力・preparation・scalar/control・provider/runtime/compiler証跡を読み取り照合した。同じcompiler options hashだけでは歴史的native stream/準備lowering/runtimeを現在系列へ接続できず、今回の再利用は0。これは旧測定値が誤りという意味ではなく、費用系列を同一視しない判断である。全217件の原価格証跡と未解決identity項目を[照合表](../../artifacts/resource_applicability/track_a_pf_common_results/2026-10-22/legacy_217_cost_identity_audit.json)へ保存した。

一方、もともと同じm5 native系列として保存されたH4 28 pairのうち、今回の適格候補22 pairを再利用。残り6は今回全精度不適格で使用しない。H6保存36同系列pairはH6 STOP後の新比較には使用していない。旧profile/model推定を新しい直接測定費用へ代用しない。

固定373 poolの旧reasonラベルは219 bridge/154 missingだが、H6の2件はmodel estimateだけで直接pairを持たない。直接pairベースでは指定どおり217/156である。[ラベル照合](../../artifacts/resource_applicability/track_a_pf_common_results/2026-10-22/historical_pool_label_audit.json)を保存し、旧契約・pool・科学予算は変更しなかった。

| 系 | 登録 | 再利用signal | 新signal（基礎/overlay） | 独立forward | 評価endpoint | 新compile候補/軸 | 使用した保存price pair |
|---|---:|---:|---:|---:|---:|---:|---:|
| H2 | 84 | 48 | 36（20/16） | 20 | 420 | 36 / 72 | 0 |
| H4 | 228 | 180 | 48（24/24） | 24 | 1,140 | 133 / 266 | 22 |
| H6 | 276 | 原観測111・欠測63 | 0（予定102未実行） | 0 | 0 | 0 / 0 | 0 |
| 計 | 588 | 完了系で228 | 84（44/40） | 44 | 1,560 | 169 / 338 | 22 |

H4新signal48≤承認150（H4/H6枠）、基礎24≤75、overlay24≤75、forward24≤75、endpoint1140≤2520、compile133≤373/軸266≤746。H2は別枠で新signal36≤84、基礎20≤48、overlay16≤36、forward20≤48、endpoint420≤420、compile36≤84/軸72≤168。H4系別149 compile poolにも133で収まる。予算間振替なし。

詳細内部callは封印契約とstage receiptに全記録。H4 SIGNALはsigned S2 piece990・spectral/forward block作用各10230・prefix preparation6、H2は705/各3390/3。COST signed pieceはH4 2089/H2 675、native evolution・独立occupation・low/high各133/36、wrapper/build/uncompiled作用/compile/compiled作用各266/72。contextは各系各stageでsnapshot/reference/oracle/full preparation各1、COST prefix preparation6/1、full-H reference新計算0。各実使用counter=ledger≤新grant上限。

H2は36候補、H4は155候補が少なくとも1精度で適格。全精度不適格の48/73候補には新compileを行わない。endpoint適格82/429、不適格338/711、未解決0。したがって**312候補で必要な評価が完了、H6 276候補が未解決**。全588候補をcompile済みとは表現しない。[全候補登録・状態](../../artifacts/resource_applicability/track_a_pf_common_results/2026-10-22/candidate_status_588.csv)。

## 共通評価政策と比較の定義

採択された今回の政策は通常signed bias、ε=.05/.01/.005/.001/.0005、allowance倍率1、α_axis=.025。uはfloor1e-9・安全倍率10・roundoff倍率64の既存m5経験式と保存uの最大、log_B marginはfloor1e-12・roundoff式・保存marginの最大、u/headroom≤.01。旧契約やuを遡及変更していない。凍結binary64 shot算術とDecimalで基本演算を段階ごとbinary64へ丸める別経路を照合した。sqrt/exp/logのlibm値は共有し、それらの独立実装検証とは主張しない。

費用G=N_cos×compiled RZ_cos+N_sin×compiled RZ_sin、P=0。PFは直接整数単価による決定論的会計。PRは固定保存費用CSVのexact有理数mean、不偏分散、paired cosine/sine共分散込みSE、nを再計算せず参照する。表の±は保存平均の1 SEで、母平均CIではない。共通政策に適合した旧PR cohortを参照し、新しいPR sampling・値の再価格取得・確認標本混合は0。

これは有限T複素signalのanalytic shot込みcompiled RZ予測であり、測定shotを実行した結果でも、全energy/QPE/RPE roundを最適化した最終総費用でもない。経験的allowanceはreference条件付き診断で、厳密certificate/全状態誤差上界ではない。

### 比較B：3公式全体の最良PF対保存PR最良

PR削減率=100×(1−保存PR平均/PF費用)。負値はPFの方が低費用。

| 系 | ε | 共通域PF最良 | PF RZ | 保存PR最良 | PR平均RZ ± SE | PR削減率 |
|---|---|---|---:|---|---:|---:|
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

H6の5行は全てUNRESOLVED。保存PR自体は旧値として存在するが、未完のPF分母との改善率を算定しない。

### 比較A：同じ公式内

| 系 | ε | 公式 | 共通域の同公式PF最良 | PF RZ | 同公式PR q / 平均RZ ± SE | PR削減率 | PR n |
|---|---|---|---|---:|---|---:|---:|
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

H2では同じS2/Yoshida4内でもPFが5精度とも低費用、同じm5内はPRが約23.04%低費用。H4では同じS2内はPF、同じYoshida4内はPRが3.75〜17.97%、同じm5内はPRが38.47〜39.56%低費用。ただしH4の緩い3精度では、別公式S2のPFがPR全体より低費用である。

精度ごとの6者（各公式のPF最良と保存PR最良）の点費用順位は[precision_cost_rankings.json](../../artifacts/resource_applicability/track_a_pf_common_results/2026-10-22/precision_cost_rankings.json)、全60比較は[comparisons_60.json](../../artifacts/resource_applicability/track_a_pf_common_results/2026-10-22/comparisons_60.json)。H2/H4全1560 endpointのexact整数費用・shot・bias allowance・価格は系別bundleのCSV/gzipを参照する。保存PRの分散・SE・nも各比較行に保持した。

## 旧比較との順位変化

保存PR同士のformula/q選択と数値は旧2026-10-21結果から不変。H2の従来約47.12%はYoshida4 PR対S2 PRで、対PF保証ではない。今回全公式PFに接続すると5精度ともPFが低費用となった。同m5内のPR低費用とは比較分母が異なる。

H4の旧分母は疎なm5決定論部分集合（B0I p8/q1）。共通格子ではm5最良がp7/q1へ移り、さらにS2/Yoshida4も含めると次の通り変わった。

| ε | 旧m5部分集合最良 | 旧PF RZ | 旧PR対m5部分集合削減率 | 新全公式PF最良 | 新PR削減率 |
|---|---|---:|---:|---|---:|
| 0.05 | B0I p8/q1 | 9.7330695e+08 | +87.0767% | S2 B0I p5/q1 | -67.6623% |
| 0.01 | B0I p8/q1 | 2.4339405e+10 | +73.1245% | S2 B0 p7/q2 | -17.4650% |
| 0.005 | B0I p8/q1 | 9.7401481e+10 | +56.9558% | S2 B0 p7/q4 | -16.0690% |
| 0.001 | B0I p8/q1 | 2.4439006e+12 | +46.1438% | Yoshida4 B0I p7/q2 | +15.0068% |
| 0.0005 | B0I p8/q1 | 9.8203405e+12 | +46.3383% | S2 B0 p7/q8 | +37.3736% |

拡張したprefix・q機会、S2/Yoshida4を含めた分母、今回採択した共通allowance/shot、新canonical native費用を区別する。検索範囲だけの単独因果効果として全差を説明しない。旧m5 p8/q1と新p7/q1は同じnative系列の直接値で比較できるが、旧profile価格を現在系列へ移す比較は採用していない。全旧行と新PF/PR対応は[old_vs_common_comparison.json](../../artifacts/resource_applicability/track_a_pf_common_results/2026-10-22/old_vs_common_comparison.json)。H6の旧m5部分集合に対する改善値は履歴として保持し、新全域結論へ延長しない。

## 統計的不確かさ・未観測event

PR費用の不偏分散/SEは保存された観測trajectory群を条件としたもの。同じcohortを5精度で使用し、q/formula選択効果を補正していない。全PR群でS4未観測、希少eventの費用・分散・tail上界はnullで、0とはしない。低い観測SEから母平均順位・希少event込み真の分散を保証しない。PF費用にMonte Carlo標準誤差はないが、bias/reference/normalizationは経験的政策に依存する。

母平均の数学的勝者保証、新PF分母に対する独立確認、全状態誤差certificateは主張しない。前回118 trajectory確認はPR対PRの固定比較であり、この新しいPR/PF順位を確認したものではない。今回は追加確認samplingを行っていない。

## 資源・終了監査

実CPU affinity128、各launch時idle CPU約127、空きRAM約1 TB・disk約127 GBを観測し、pending/idle CPU/RAM/disk proxyからworkerを選択。固定の低いCPU/RAM/worker上限を置かずH4 127・H2 36 workerで並行した。compiler processと内側BLASは各1、context/preparationを親で共有して重複生成を抑えた。共有設定、他ユーザーprocess、cgroup、venvを変更していない。観測値・実worker・壁時間は系別resource receiptsへ保存した。

| 系 | SIGNAL秒 | QUALIFY秒 | COST秒 | ACCOUNT秒 | COST workers |
|---|---:|---:|---:|---:|---:|
| H2 | 3.934 | 0.854 | 84.635 | 0.764 | 36 |
| H4 | 15.687 | 1.410 | 933.002 | 1.133 | 127 |

人工10 test method（全公式/q/prefix0・Fullの独立forward、overlay、integer、grant境界、toy native/cos/sin・2 fork worker）PASS後に科学実行。両系4 stage成功、計163 compiler worker全exit0・親でjoin/reap・残存0。新archive計1014（H2 216/H4 798）の解凍後原byte SHAを全照合。H4最大compiled vector誤差8.327e-11、他native最大1.089e-12。H2は2.060e-11/8.180e-14。CT01<1e-7・他native<1e-9、phase alignmentなしを全件確認。source/input/provider/runtime/compiler/preparation/scalar/control照合PASS。

保存shotの別整数ceiling不等式、全eligible endpointの別Fraction整数費用、80桁Decimal費用比、全登録格子最小、ledger/marker/grant上限、原archive SHAを独立経路で検証した。作者自身による別数値経路の終了監査であり、独立GPT/Claude科学レビュー・外部CI/外部再現ではない。H6は失敗stageをretryせず原metadata gateで科学call0 STOP。

## 保存と終了

- [H2 bundle](../../artifacts/resource_applicability/track_a_pf_common_H2_results/2026-10-22/README.md)：独立SOURCEと別branch結果をbyte同一で参照結合。
- [H4 bundle](../../artifacts/resource_applicability/track_a_pf_common_H4_results/2026-10-22/README.md)：契約・承認receipt・4 grant/ledger/marker・signal/native原値投影・終了監査・入力SHA。
- [統合bundle](../../artifacts/resource_applicability/track_a_pf_common_results/2026-10-22/README.md)：588候補、60比較、217費用identity照合、H6原回収とSTOP、旧比較との差。

原private raw・QPY/NPZ可逆archive・消費済みgrant・ledger・marker・STOPは各専用rootで保全。科学SOURCEの変更なし。旧研究の科学契約/u/費用/PR結果は遡及変更していない。追加call上限の増額、未登録候補、q16、新PR sampling、production、詳細科学レビュー、論文本文作成、GitHub pushは0。

[最終構造照合](../../artifacts/resource_applicability/track_a_pf_common_results/2026-10-22/structure_checks.json)は5,991項目PASS。新規科学callなしで588登録・60比較・8別grant・全入力seal・旧科学SOURCE/raw不変・H2別commitとのbyte同一を確認した。[統合STOP receipt](../../artifacts/resource_applicability/track_a_pf_common_results/2026-10-22/MANDATORY_STOP.json)を保存。

**ローカルcommit後mandatory STOP。この承認範囲の科学stageを再開・再実行しない。H6の継続には原63 metadataの回収または別枠で明示承認された対応が必要であり、今回のgrantを流用しない。**
