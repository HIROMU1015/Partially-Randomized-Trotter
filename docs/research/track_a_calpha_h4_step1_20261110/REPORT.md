# C-α／α0-P H4 Step 1：sector適用確認での技術的STOP

2026-11-10 JST。担当：Codex。

**結果：`TECHNICAL_STOP_PRIMITIVE_SPIN_SECTOR_NOT_PRESERVED`。**

承認されたStep 1を開始し、既存入力の適用確認で停止した。固定された一般DF固有基底を使う非zeroの単一Z atomについて、46件でα/β混合の診断を得た。今回の指示・GPT理論レビューが要求する「各sampled primitiveのNα/Nβ保存」を確認できないため、**Γのmoment、m0/m17の信号、位相・半径上界へ進まなかった**。

これは指定された適用条件に対する技術的STOPであり、C-α一般の不可能性、固定r4の位相性能の負結果、総費用の劣位を示す結果ではない。演算子和のsector保存と、個別primitiveのsector保存は異なる。平均kernelの保存性だけで十分かという条件変更の理論判断は今回代行していない。

## 1. 成果物と科学条件

- [適用確認の全結果JSON](../../../artifacts/calpha_h4_step1/2026-11-10/run01/applicability.json)
- [実行log](../../../artifacts/calpha_h4_step1/2026-11-10/execution.log)
- [実行SOURCE](../../../scripts/run_calpha_h4_step1_applicability.py)

Linear H4、1.00 Å、STO-3G、cutoff前の元H_phys、保存DF12、generation-prefix LD3、Nα=Nβ=2／sector36、元Hの保存基底stateを厳密状態とする研究promise、状態準備費用除外。予定lawはpaired K0／α0-P、正時間partial-S2、δ=.02、r4、対象m0/m17、新scopeのsystematic上限.14 rad（stat .24、reserve .02、全体 .4）。入力適用確認に限定して実行され、二点の信号値は取得していない。

元H/state/DFのfile identityとarray identity、保存spin/JW順序、generation order、元HおよびDF再構成のhashは既存P1 loaderで一致した。stateの再正規化・phase変更やDF stateへの代用はない。新SCF・積分生成・DF fit・ground solveは行っていない。

## 2. SOURCEと再利用

作業branchは `codex/track-a-calpha-h4-step1-20261110`、worktreeは `.worktrees/track-a-calpha-h4-step1-20261110`。基点は `0d5ca357ccfd8fcc850bef800c5bcadfde2a153e`。既存の未commit編集があるroot checkoutと仮説Iのworktreeを使用しなかった。

実行SOURCEは新runnerとJSONの`SOURCE_SHA256`に記録した固定core。再利用したcoreは実行前に基点commitのbytesと一致し、実行後もrunnerを含む記録対象SOURCEのhashは不変だった。新runner・結果・本報告は未commitのlocal成果であり、immutable CIや外部再現済みの証拠ではない。commit/pushは実施していない。

主な再利用：

- `b2_p1_signal_validation.load_private_inputs`：既存private入力、保存積分、CAR構築、元H/state/DF/sector identity照合。`run()`は呼んでいない。
- `df_trotter.decompose.diag_hermitian`：既存generic DF diagonalizerの同じsort・固有基底。
- `df_rte_tail.exact_df_diagonal_coefficients`：同じI/Z/ZZ係数・identity抽出・係数ゼロ判定。係数cutoffは追加しない。
- 既存のCAR参照とS0 array hash。新しい回路builder・sampler・state solverは作成していない。

K2比較の固定入口は `ea2c0915ba2f7e11a6dac58dc01cf1af3c0d0ff5`、元信号SOURCEは `5ef55f923e73e61bada383d0cfe98283e7984e9d`。今回は新信号がないため数値差比較を行わず、K2再simulation・再会計もしない。

## 3. sector確認の方法と結果

原spin順は2p=α、2p+1=β。DF固有基底の列をu_jとすると、Gaussian変換後の単一Z atomは、対応する軌道のnumber operatorにより `Z'_j=I−2 a†(u_j)a(u_j)` と表される。そのone-particle projector `u_j u_j†` にα/β cross blockがある場合、単一Z atomはNα/Nβと可換でない。

各tail fragmentの非zero単一Z係数について、`u_alpha,j u_beta,j†`のmax-entry、α/βそれぞれのcolumn weight、元の係数を保存した。これは回路生成や状態trajectoryのsamplingを必要としない、既存辞書の必要条件の検査である。固有ベクトルのglobal phaseや単なる列のラベル変更ではこのcross blockは消えない。

既存P1の独立照合閾値1e-9を実行前に固定した診断基準として用いた。これを厳密interval証明とは呼ばない。46件がこの基準を超え、最初に記録された例は次のとおり。

| 記録された例 | 値 |
|---|---:|
| tail fragment（0始まり） | 3 |
| diagonal mode（0始まり） | 1 |
| 単一Zの係数 | −1.050571278693526e−17 |
| columnのα weight | 0.00014452733343994496 |
| columnのβ weight | 0.9998554726665599 |
| one-particle projectorのspin cross-block max-entry | 0.004590945778550787 |

**係数が小さい例を含む。46件という件数から、信号汚染の大きさ・頻度・総費用への影響を推論しない。** cross-block値も、係数を掛けたHamiltonian誤差や実回転のsector漏れ確率ではない。今回の停止理由は、非zero係数を保持する固定辞書について個別primitiveの保存条件を満たせなかったことである。微小係数のdrop、spin-aware基底への変更、平均保存性だけへの条件変更を黙って行わない。

同じ実行で、raw H_phys、明示したH_solve、DF再構成、one-body、および各DF fragmentについて、保存sectorから外へのmatrix blockのmax-entry診断はすべて0だった。**和のHamiltonian／fragmentがsectorを保つことは、各sampling atomが保つことを意味しない。** そのため、この停止を既存K2 P1の保存平均信号結果の取消しとは扱わない。

tailのλ_Rは0.5832704246339983、抽出identityは0.09261516589219072で、既存P1参照と一致した。別のtailや別の分割へ置換した検査ではない。

## 4. Hermitian参照と資格

入力の再構築ではraw H_physを保存し、`H_solve=(raw H_phys+raw H_phys†)/2`を明示した参照としてのみ区別した。両者のspectral norm差の今回の数値診断は5.418931789556987e−16 Ha。targetの変更や黙ったHermitian化はしていない。この診断を厳密なmodel上界へ読み替えない。

保存sector gapは数値診断として既存結果へ参照を残し、新しくsolveしていない。認証されたgap下界はnullのままである。sector gateで停止したため、gap依存側・非依存側の十分条件の数値評価はいずれも未実施。

今回得たのは既存inputとgeneric辞書に対する数値的な適用診断。ΓのPSD・moment・有限時間上界・有限bit誤差を資格化した結果ではない。

## 5. m0/m17：取得状態

| 項目 | m0 | m17 |
|---|---|---|
| raw複素信号 | UNKNOWN／未計算 | UNKNOWN／未計算 |
| 観測位相差・raw半径 | UNKNOWN／未計算 | UNKNOWN／未計算 |
| 減衰参照信号 | UNKNOWN／未計算 | UNKNOWN／未計算 |
| Γの期待値・分散 | UNKNOWN／未計算 | UNKNOWN／未計算 |
| ε_dyn・d(T) | UNKNOWN／未計算 | UNKNOWN／未計算 |
| 半径下界・位相上界 | UNKNOWN／未計算 | UNKNOWN／未計算 |
| .14 rad枠への適格性 | 未判定 | 未判定 |
| 保証資格 | 適用条件で技術的STOP | 適用条件で技術的STOP |

数値未取得へ0を代入していない。二点で良好／不良だったという性能結論も出さない。

## 6. 実行・検証・保全

- 適用確認の科学実行は1回。runner内runtimeは約0.288秒（import等を除く）。記録された入力・SOURCE検査を通過後、sector gateで終了した。
- 新しいcoherent信号call、ground solve、回路build、compile、sampling、shot会計はすべて0。Γ moment、全18round、r/LD変更、他変種、native価格・総費用、旧runのretryも実施していない。
- Python3.11.0rc1、NumPy1.26.4、SciPy1.14.1。CPU affinity32、GPU不使用。小さい8/36/256次元処理にはjob内のsingle-thread BLASを選択し、process poolやq回の展開を行わなかった。速度倍率は測定・主張しない。共有環境や他ユーザーjobは不変更。
- runnerの構文検査と、実行前後のSOURCE hash不変・出力JSONの構文・status・科学call記録・privateファイルのGit除外を確認した。新しい人工行列simulationや既存の大規模test suiteは実行していない。
- private入力は既存ファイルへのsymlinkを専用Git除外directoryから参照。公開対象の結果JSONにstate配列・積分・HDF5本体は含まない。
- commandのexit code 0は、runnerが予定されたgate STOPを記録して終了した意味であり、科学的PASSではない。

## 7. 次の担当とmandatory STOP

**現行の承認範囲ではここでmandatory STOP。** 個別primitiveのsector条件が不成立のまま信号計算へ進まず、sector70やspin-aware辞書へ変更しなかった。

次の担当は**ユーザー／GPT**。既存generic frameを保持し、平均kernel／参照減衰generatorの不変性で今回の信号定理を適用できるか、それとも個別primitive保存を満たす別辞書が必要かを判断する。前者は理論レビューの適用条件の扱い、後者は具体的law／frame・科学条件の変更になる。いずれも今回自動的に採用・実行しない。詳細理論レビューを新しく開始する場合も、科学実行とは別の指示が必要である。

**Codex**は、次の明示された固定範囲が決まるまで追加科学計算を行わない。入力・固定SOURCE・旧結果・STOPを保全する。
