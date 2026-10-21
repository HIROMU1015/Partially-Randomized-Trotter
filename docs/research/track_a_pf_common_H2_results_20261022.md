# H2共通決定論PF探索：取得・固定PR/PF費用比較の結果

2026-10-22 JST。**84候補の共通域評価と4 stage・終了監査PASS。mandatory STOP。**

3公式を含む共通域最良PFは、5精度すべてで保存PR探索集合最良より低費用。同じm5_best内ではPRが約23.04%低費用だが、S2/Yoshida4も含めた最良PFに対する優位とは異なる。

## 条件と入力の接続

linear H2、1 Å、STO-3G、tol-only DF threshold1e-8・保存rank3、T=.8、P=0。B0/B0I L_D=0/1/2、Full=3、q=1/2/4/8、δ=.8/.4/.2/.1、S2/Yoshida4/m5_best各28候補。元binary64係数・signed schedule、symmetric-directional control、元constantとtail identityのscalar一回、cosine/sine full Hadamard wrapper。Qiskit1.3.0、rz/sx/x/cx、opt1、seed17。H2既存symbolic-zero scalar guardを保存PRと同様に使い、gate順序・angleの不変を各axisで検証した。

基準commit `3728a618d7f09e67a070632ded1f27dab54b21a4`。独立worktree/branch `pf-common-h2-execution-20261022`、科学SOURCE `1b35f9ac2efab4c15bc00ee2d3c81643aca422f9`。原D70 metadata blob `2b534c7425af6d67bc92c7423a422ac368d9ad07` と、新outer32 PR（固定結果a05f4ed19e87e1eb65f7fc346385b5b4eae69845、保存費用8613b3ecda2030a677b1690fe7aa3da23dec4873）のsnapshot/Hamiltonian/state/compiler bindingが一致した。

snapshotは前回承認済み移行再構築で得た容器を再利用し、今回のSCF/DF/state/reference新生成は0。旧snapshot SHA `62c6481bd1f8b05ec3b399b3a5293873796737866409049d808e8fe432dcacaa`、現容器SHA `8bb1aa35cf80266dd4aa1552cc65097770dfe67c690c2e292860923a5b4ae2f9`。同一原本の復元とは主張しない。constant/one_body/lambdas/g_matrices/sector indices/full・sector stateの保存配列hash、Hamiltonian/state hashが旧値と完全一致し、前回forward/reference検証receiptも接続した。今回native検証も保存signalとの照合に合格。

D70の48 signalだけを再利用し、旧Yoshida4 prefix0の8 signalは再利用しなかった。旧値は単一blockのsigned composition経路で今回の独立forward/expmと異なるため、許可されたH2予算内で明示signed scheduleから再取得した。[新旧差とbinding](../../artifacts/resource_applicability/track_a_pf_common_H2_results/2026-10-22/H2_D70_reuse_audit.json)を保存し、旧rawを修正していない。旧costはnative/preparation/runtimeの接続を証明できず再利用0。保存shotも無条件転記せず全420 endpointを共通政策で算定した。prefix0とFullは統合していない。q16は旧結果として保全し今回域へ含めない。

## 実取得と共通政策

| 処理 | 実使用 | 承認上限 |
|---|---:|---:|
| 基礎signal | 20 | 48 |
| B0I overlay | 16 | 36 |
| 新signal合計 | 36 | 84 |
| 独立forward | 20 | 48 |
| paired適格性/shot endpoint | 420 | 420 |
| 新compile候補/軸 | 36 / 72 | 84 / 168 |
| 新PR sampling | 0 | 0 |

内部callはSIGNALのspectral signed piece705、spectral/forward block action各3390、context各1・prefix preparation3。COSTはsigned piece675、独立occupation/low/high/native evolution各36、wrapper/build/uncompiled作用/compile/compiled作用各72、context各1・prefix preparation1。詳細上限を事前sealし、全stageでledger実使用=予約counter≤grant上限。新候補の人工10件PASS後に一回実行し、使用済みgrant retry/resumeなし。

通常signed bias、ε=.05/.01/.005/.001/.0005、allowance倍率1、α_axis=.025、u=floor1e-9・安全倍率10・roundoff倍率64・保存u以上、log_B margin floor1e-12・保存margin以上、u/headroom≤.01。凍結binary64 shot式とDecimalの段階別binary64丸めを照合。経験的allowanceはreference条件付きで厳密certificateではない。保存PRは元の同政策の探索cohort平均・不偏分散・SE・n・shot・費用をそのまま使用し、再sampling/再価格取得/確認標本混合は0。

84候補中36候補が少なくとも1精度で適格、残り48は5精度とも不適格なので新compile不要。420 endpointは適格82・不適格338・未解決0。84候補の取得完了は、全候補の適格性と必要な費用が揃ったことを意味し、不適格候補までcompileした意味ではない。

## 精度・公式別の比較

改善率は100×(1−PR平均費用/PF費用)。負値はPFが低費用。PFは決定論的な整数会計、PRの±は保存標本平均の1 SE。有限T複素signalのshot込みcompiled RZであり、全energy/QPE/RPE round最適化の最終費用ではない。

| ε | 比較域 | 最良決定論PF | PF RZ | 保存PR最良 / 平均RZ ± SE | PR削減率 | PR n |
|---|---|---|---:|---|---:|---:|
| 0.05 | S2 | S2 B0I p2/q1 | 6.3866960e+06 | S2 q1 / 1.5459953e+07 ± 1.756e+05 | -142.0650% | 96 |
| 0.05 | Yoshida4 | Yoshida4 FULL p3/q1 | 1.7762949e+07 | Yoshida4 q1 / 2.3003550e+07 ± 1.903e+05 | -29.5030% | 60 |
| 0.05 | m5_best | m5_best FULL p3/q1 | 6.2869592e+07 | m5_best q1 / 4.8382205e+07 ± 1.651e+05 | +23.0436% | 33 |
| 0.05 | 全公式 | S2 B0I p2/q1 | 6.3866960e+06 | S2 q1 / 1.5459953e+07 ± 1.756e+05 | -142.0650% | 96 |
| 0.01 | S2 | S2 FULL p3/q1 | 2.2650324e+08 | S2 q1 / 5.2550629e+08 ± 5.969e+06 | -132.0083% | 96 |
| 0.01 | Yoshida4 | Yoshida4 FULL p3/q1 | 4.6704208e+08 | Yoshida4 q1 / 6.0486398e+08 ± 5.005e+06 | -29.5095% | 60 |
| 0.01 | m5_best | m5_best FULL p3/q1 | 1.5716008e+09 | m5_best q1 / 1.2094629e+09 ± 4.128e+06 | +23.0426% | 33 |
| 0.01 | 全公式 | S2 FULL p3/q1 | 2.2650324e+08 | S2 q1 / 5.2550629e+08 ± 5.969e+06 | -132.0083% | 96 |
| 0.005 | S2 | S2 FULL p3/q2 | 1.4143274e+09 | S2 q2 / 3.3125560e+09 ± 2.629e+07 | -134.2142% | 71 |
| 0.005 | Yoshida4 | Yoshida4 FULL p3/q1 | 1.9937162e+09 | Yoshida4 q1 / 2.5820199e+09 ± 2.136e+07 | -29.5079% | 60 |
| 0.005 | m5_best | m5_best FULL p3/q1 | 6.2864840e+09 | m5_best q1 / 4.8379129e+09 ± 1.651e+07 | +23.0426% | 33 |
| 0.005 | 全公式 | S2 FULL p3/q2 | 1.4143274e+09 | Yoshida4 q1 / 2.5820199e+09 ± 2.136e+07 | -82.5617% | 60 |
| 0.001 | S2 | S2 FULL p3/q4 | 7.2245915e+10 | S2 q4 / 1.7101562e+11 ± 1.433e+09 | -136.7132% | 33 |
| 0.001 | Yoshida4 | Yoshida4 FULL p3/q2 | 8.9935928e+10 | Yoshida4 q2 / 1.1106603e+11 ± 9.908e+08 | -23.4946% | 34 |
| 0.001 | m5_best | m5_best FULL p3/q1 | 1.5718181e+11 | m5_best q1 / 1.2096294e+11 ± 4.128e+08 | +23.0427% | 33 |
| 0.001 | 全公式 | S2 FULL p3/q4 | 7.2245915e+10 | Yoshida4 q2 / 1.1106603e+11 ± 9.908e+08 | -53.7333% | 34 |
| 0.0005 | S2 | S2 FULL p3/q4 | 3.7010164e+11 | S2 q4 / 8.7589411e+11 ± 7.338e+09 | -136.6631% | 33 |
| 0.0005 | Yoshida4 | Yoshida4 FULL p3/q2 | 3.7503876e+11 | Yoshida4 q2 / 4.6314865e+11 ± 4.131e+09 | -23.4935% | 34 |
| 0.0005 | m5_best | m5_best FULL p3/q1 | 6.2882646e+11 | m5_best q1 / 4.8392812e+11 ± 1.652e+09 | +23.0427% | 33 |
| 0.0005 | 全公式 | S2 FULL p3/q4 | 3.7010164e+11 | Yoshida4 q2 / 4.6314865e+11 ± 4.131e+09 | -25.1409% | 34 |

各formulaの「最良」は今回のq≤8共通域内だけの最良。[全420 endpoint](../../artifacts/resource_applicability/track_a_pf_common_H2_results/2026-10-22/PF_endpoint_costs.csv)、[20比較の完全値・保存分散](../../artifacts/resource_applicability/track_a_pf_common_H2_results/2026-10-22/comparisons.json)にexact整数/有理数・shot・headroom・u・単価・保存PR不偏分散とSEを対応させた。欠測subset最良の代用なし。

## 旧比較との関係・統計的限界

旧2026-10-21報告のH2約47.12%改善はYoshida4 PR対S2 PRであり、PF対PRの主張ではなかった。保存PR同士の順位は変更していない。今回、比較可能な決定論PF分母を接続すると全公式PFが低費用になった。同じm5内のPR優位と、全公式PFに対する劣位は両立する。旧D70/q16の歴史的全域最良に勝ったという主張もしない。

PR群の有限標本・探索選択効果は補正していない。保存SEは観測されたtrajectory費用だけのSEで、未観測S4/rare eventの費用・分散上界はない。PF値に古典Monte CarloのSEはない一方、bias/reference/roundoffは経験的診断に依存する。母平均順位の数学的保証、全状態誤差保証、新しいPF分母に対する独立確認はない。

## 終了監査・保全

科学SOURCEとprovider/runtime/compilerのseal不変、Hamiltonian/state/preparation/scalar/control照合PASS。36 worker全exit0・同親join/reap・残存0、216新規archiveの解凍後原byte SHA一致。最大compiled vector誤差2.060e-11、他native最大8.180e-14、phase alignmentなし。164個の保存shot integer-ceiling検査、費用の別整数/Fraction経路と80桁Decimal比照合PASS。作者自身の別数値経路監査であり外部再現・詳細科学レビューではない。

[終了監査](../../artifacts/resource_applicability/track_a_pf_common_H2_results/2026-10-22/end_audit.json)、[SOURCE/入力SHA](../../artifacts/resource_applicability/track_a_pf_common_H2_results/2026-10-22/source_input_identity.json)、[契約投影](../../artifacts/resource_applicability/track_a_pf_common_H2_results/2026-10-22/execution_contract_APPROVED_projection.json)、[stage receipts](../../artifacts/resource_applicability/track_a_pf_common_H2_results/2026-10-22/stage_receipts.json)、[grant/ledger/marker](../../artifacts/resource_applicability/track_a_pf_common_H2_results/2026-10-22/controls_and_ledgers.json.gz)を保存。原private契約・raw・grant・ledger・marker・STOPは専用rootに保持。科学grantの再実行用コマンドは提示しない。

H4/H6とは別SOURCE/出力/grantで実行した。H2科学終了後はmandatory STOP、ローカルcommitのみ。GitHub push、q16追加、新候補、新PR sampling、詳細科学レビュー、論文本文作成を実施していない。
