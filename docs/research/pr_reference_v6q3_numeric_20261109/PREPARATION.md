# Q3-N：固定監査SOURCEと実行範囲

固定計画は[ba616902 §1〜4](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ba6169028d6d71105aa1f474dfcd285461339070/docs/research/pr_reference_v6q3_static_20261109/NEXT_VALIDATION_PLAN.md)。ユーザーが許可した1 campaignだけを、この専用branchのSOURCEで実施する。元SOURCE・provider・保存入力と算術条件を変更しない。

- [driver](../../../scripts/run_pr_reference_v6q3_n.py)：旧common()を1回、diagonal_primitivesを固定5 contextで各1回、元の左結合式によるscalarを1回。中央2角は保存V2 witnessのhexを読み、atanを呼ばない。N1は全D／全registryのmetadataと型・finite性を確認し、basis指数は評価しない。
- [checker](../../../scripts/check_pr_reference_v6q3_n.py)：保存出力だけの整数／Fractionによる共通格子・half・符号・coverage・SOURCE保全のN3検算。cacheをunpickleせずprimitiveやproviderを再実行しない。
- [人工test](../../../tests/test_pr_reference_v6q3_n.py)：half欠落、ratio改変、重複、型・非有限・shape・classical条件・metadata不一致、subnormal generatorのINCONCLUSIVE判定。科学cache・回路は使わない。

元科学SOURCEはf2638a7845685a119619382b850441f5d8d33713、providerはeaf70d886d42364a01b7e949ebc21f3b026bbe72。common.pkl.xz、A1、V2 witnessは元科学refから読む。fresh outputは`artifacts/pr_reference_v6q3/2026-11-09/N_run01`、旧cache等は変更しない。outputのcacheコピーはGit対象外とし、元archiveとhashで同定する。

対象はLinear H4／1.00 Å／STO-3G／DF12、元H厳密ground供給・状態準備費除外、Partial S2／LD3／h=.02／finite RTE K2、ordinary diag(I,V)、canonical independent finite-bit law、m17/q131072/r32/bits88、Qiskit1.3.0／compiler A。B′／Reference構成、energy精度1.5936e-4 Ha、failure .05、phase cap .4、αaxis1/720、旧Nを保持する。

共通格子の閉包は、固定SOURCEのbasis対・event文法と、旧correctly-rounded binary64 add/sub/remainderの算術モデルを条件とする。全supportの実Sを計算するのではなく、有限generatorの半角込みℓ≤1022という十分条件を監査する。ℓ>1022は実subnormal反例ではなくINCONCLUSIVE。古いM／B_normを再評価しない。

未知型・入力不一致・不足・非有限・文法不一致があれば部分結果とSTOPを保存し、自動再実行・scope拡張しない。全trajectory、m17 caller接続、回路build/compile/simulation、新phase評価、sampling、RPE／shot／総RZ／T、guard／合成は0件。N1〜N3と公開を完了したらmandatory STOP。

2026-11-09 SERVER 2 GPUA100-gleap02。既存project venvのQiskit1.3.0／NumPy1.26.4／SciPy1.14.1を再利用。事前importではcommon loaderの呼出し0を確認した。小さい有限集合に対する一度の監査であり、科学workerの重複起動・独自CPU/RAM恒常上限・共有設定変更・AI subagentは用いない。

実行前の人工8テストは全PASS。実科学入力を読まず、既存Q1/Q2科学runner・旧テストを再実行していない。このSOURCEをcommitして固定した後に、指定した有限campaignを1回開始する。
