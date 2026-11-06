# 研究C：H4・1.00 Åの頑健性・公平retuning・同J/U強D比較

2026-11-06 JST。ユーザーの指示によりH4・1.15 Åの追加検証・修復・再実行は行わず、完了したH4・1.00 Å側をまとめる。保存済みJSONの転記・算術照合のみで、新しいfit、分子評価、回路build/compile、NPZ/QPY decode、標本追加、重要科学レビューは行っていない。

**C0・64更新の主比較では、PR-aware DFの条件付き総RZ点推定がENTRYより、共通r32で51.08%、公平retuning後で45.47%低かった。公平retuningの全6比較でもPRが低い点推定となった。** これは固定モデル下の有限予算・有限標本の結果であり、実装全law、実際の化学精度達成、厳密な資源優位、他geometryへの移送性は認定しない。

## 条件と完了範囲

直鎖H4・隣接1.00 Å・STO-3G、number CDF rank8、Nα=Nβ=2/sector36、元順L_D=2、元g補正κ・nuclear scalar一回。partial ordered S2、s1、δ=h=.1、K0、m=0..12。候補固有own-DF ground oracleを用い、状態準備は費用から除外した。CA=.00159 Ha、DF予算CA/8=.00019875 Ha、βtotal=π/6、βstat=π/12、failure=.01、axis=.01/26。P0はQiskit1.3.0/opt1/seed17、rz/sx/x/cx、位相保持、fenced serial会計である。

C0は共通neutral start、P1/P2は事前固定の摂動start。3 start×3 strength(.03,.1,.3)×2 arm(PR/ENTRY)の18 trajectoryについて16/64更新の36 endpointを揃えた。「18」は今回の比較gridの積であり、それ自体に特別な意味や資源上限はない。17 trajectoryは既存の完了結果を再利用し、C0/ENTRY/.3の1本だけを新SOURCEで計算した。

- 全18 trajectoryが64受理更新まで到達し、36 checkpointすべてで物理DF予算PASS。各checkpointのJ一次基準はMET、U一次基準はNOT_MET。有限更新手順の比較であり、同時一次収束や大域最適性の比較ではない。
- 36 candidate×3r=108 policyを評価し、90 policyがnative価格評価へ適格、18 policyは不適格として保持した。
- 選択用nativeは各適格policy32件、計2,880件。選ばれた22個の(candidate,r)ペアへ独立final64件ずつ、計1,408件を実行した。selectionとfinal、過去の32/256/640件はpoolしていない。
- native合計4,288件と、採用候補に対応する同J/U強決定論D19件を保存した。選択・最終・Dの保存call summaryにcorrectness failureはない。
- 12個のPR−ENTRY比較のうち10比較が成立、P1の共通r32の2比較は未成立である。不適格を価格ゼロや手法の敗北として扱わない。

## 主比較：C0・64更新

候補はselection32の総RZで選び、独立final64で報告する。共通r32でもstrengthは両arm同じ3候補から選ぶ。公平retuningでは双方に同じ3strength×r={16,32,64}の機会を与えた。以下の総RZの単位は10^9。

| 比較 | 選択strength PR/ENTRY | r PR/ENTRY | PR総RZ | ENTRY総RZ | PR/ENTRY点比 | PR削減率 |
|---|---|---|---:|---:|---:|---:|
| 共通r32 | 0.1/0.1 | 32/32 | 93.727 | 191.600 | 0.489180 | 51.08% |
| 公平retuning | 0.1/0.03 | 16/64 | 93.007 | 170.560 | 0.545306 | 45.47% |

登録した主familyはもともと2 geometry×2 regimeの4差であり、1.15 Åを終了した後も**元の98.75%近似Welch差区間を変更していない**。PR−ENTRY差の区間は、共通r32が[-103.632, -92.114]×10^9 RZ、公平retuningが[-82.797, -72.308]×10^9 RZ。両方とも登録精度目標を満たし、保存判定は`NOMINAL_MODEL_PR_LOWER`である。ランダム分母からの比区間は作っていない。

公平retuningではENTRY側もr64を選び、総費用が下がったが、PRとの差は残った。最終結果を見てstrength/rを選び直していない。両regime間ではENTRYのstrengthも変わるため、その差をrだけの因果効果とみなさない。

## 費用内訳と同J/U強D

主指標は `C_total = W × mean_kernel_cost + shell_cost`。shots、反復重みW、kernel単価、shell費用を分離した。rやnative標本数をWへ重複乗算していない。主比較の保存値は次のとおり。

| regime | DF arm | r | shots | W | kernel平均RZ | shell RZ | 総RZ |
|---|---|---:|---:|---:|---:|---:|---:|
| common_r32 | PR | 32 | 14,732 | 12,522,400 | 7,484.734375 | 51,562 | 93,726,889,299.50 |
| common_r32 | ENTRY | 32 | 19,746 | 29,031,600 | 6,599.703125 | 69,111 | 191,600,010,354.75 |
| fair_retuned | PR | 16 | 16,142 | 16,932,304 | 5,492.890625 | 56,497 | 93,007,350,398.25 |
| fair_retuned | ENTRY | 64 | 16,082 | 16,741,456 | 10,187.875000 | 56,287 | 170,559,917,333.00 |

共通r32ではPRのkernel単価はENTRYより高いが、shots/Wが低く、総RZが小さい。したがってkernel単価だけの比較では今回の順位を説明できない。これは会計項の観察であり、λ・J・Uの個別の因果効果を切り分けた実験ではない。

各候補固有のJ/Uを使う全frame ordered S2/s1強Dを一度評価し、同じ候補のfixed/tunedで共有した。C0・64更新のPRでは、共通r32のPR/D点比は0.897911（約10.21%減）、retuned r16は0.891018（約10.90%減）。後者の個別nominal95%比区間は[0.874508, 0.907528]。同じPR候補のDはshots13,688、W9,467,006、kernel RZ11,026、shell RZ47,908、総RZ104,383,256,064である。

公平retuning・64更新でのPR対各自の同J/U Dは以下のとおり。D分母は候補ごとであり、start間・arm間に一つのDを流用していない。

| start | PR/D点比 | PR削減率 | 個別nominal95%比区間 |
|---|---:|---:|---|
| C0 | 0.891018 | 10.90% | [0.874508, 0.907528] |
| P1 | 0.735253 | 26.47% | [0.724766, 0.745741] |
| P2 | 0.778293 | 22.17% | [0.766086, 0.790501] |

採用したPR側の12 policyはすべて各自のDより低く、ENTRY側の10 policyはすべて各自のDより高い点推定だった。これは個別副比較の記述であり、全policy同時優位の主張ではない。共有統計APIの`RZ_PR_D_nominal`というキー名や`preselected r32 only`という内部説明は歴史的名称を含むため、実際のarm/r/cohortは外側のtask・r・final64 bindingで特定した。ENTRYについてこのキー名を「PR-awareの結果」と誤読しない。

## 3初期値・2更新予算での比較

下表は独立final64から保存された点比。初期値ごとにselection32で候補を選んだ結果であり、すべてのstrengthについて独立final検証した意味ではない。

| start | 更新予算 | 共通r32 PR/ENTRY点比 | 公平retuning PR/ENTRY点比 | 公平retuning PR削減率 |
|---|---:|---:|---:|---:|
| C0 | 16 | 0.492916 | 0.562856 | 43.71% |
| C0 | 64 | 0.489180 | 0.545306 | 45.47% |
| P1 | 16 | 未成立 | 0.404264 | 59.57% |
| P1 | 64 | 未成立 | 0.390448 | 60.96% |
| P2 | 16 | 0.401664 | 0.462920 | 53.71% |
| P2 | 64 | 0.403922 | 0.448389 | 55.16% |

公平retuningは全6セルでPRがr16、ENTRYがr64を選び、PRの削減率は43.71〜60.96%。C0・64更新以外の成立セルは個別nominal95%Welch差区間で、保存判定はすべて`NOMINAL_MODEL_PR_LOWER`、登録精度目標PASSだった。全セル同時の被覆・優位は認定していない。

P1/ENTRYのr32は、両更新予算・全3strengthでm12の`INFEASIBLE_PHASE`となり、ENTRY候補を選べなかった。したがってP1の共通r32は両予算とも`UNRESOLVED_MISSING_FINAL_PAIR`。r64では適格となり、公平retuning比較は成立した。計算障害や都合のよい標本除外ではなく、元の適格条件による未成立である。

主C0・64更新・strength=.1のexact tail λはPR .498794143081 Ha、ENTRY .945068647756 Ha。両DF予算PASSとJ MET/U NOT_METを保持する。λ低下だけを総費用改善の保証・因果説明にしない。

## 高速化と実行時間

全512入力/出力の行列を維持し、Aer CPU unitary(double)へ評価経路を変更した。raw・compiledと**実際に構築した両逆回路**をそれぞれ評価し、構造を仮定して逆回路検査を省略していない。candidate内の同一event行列を再利用し、Dの重複排除と独立候補の並列実行を行った。component/adjointの1e-10、complex signal2e-10、rtol0、位相規約を維持し、部分空間や少数stateへの置換はしていない。

人工CDF8回路における4行列評価の実測は、r16で48.84→7.82秒、r32で67.56→11.47秒、r64で102.11→18.30秒（約5.6〜6.2倍）。旧評価器との差は最大約3.42e-13で元の許容差内だった。これは人工回路の技術ベンチマークであり、科学実験全体の同倍率speedupを測ったものではない。

64 CPUのaffinityを独立候補36 processへ配分し、BLAS/OpenMPとAerは各worker内で協調した。専用fit/eval環境を分け、共有package/OS設定を変更していない。GPUは使用していない。

fresh実行は01:32:41 JSTに開始し、**開始から84.58分で1.00 Å側の全下流評価が完了**。これは17本の最適化結果を再利用した今回の実測であり、全18本の最適化を含む所要時間ではない。1.15 Å初期化中の不一致により02:57:57 JST、約85.26分で統合driverが停止した。

## 未資格事項と1.15 Åの終了

費用はexact native synthesis・ideal finite-RTE IID law・finite-bit標本価格をideal-law価格のproxyとする仮定に条件付けたモデル値である。実装lawの全範囲信号誤差・全law synthesis上界は未資格で、evidence laneのshots/total、actual_CA_achieved、保証付き資源優位はnullのまま。近似IID区間の被覆率、希少tail、費用support、一般分子・別geometry・実機・状態準備込みの優位は示していない。実際の量子shot抽選は0である。

開発selectionだけから移送strength PR=.03/ENTRY=.3を凍結した後に1.15 Å入力を生成した。P1のtensor-only初期復元で`INDEPENDENT_TERMINAL_VALUE_GRADIENT_MISMATCH`を検出し、元のmandatory STOPとなった。移送先の最適化・下流評価は完了しておらず、**移送性の結果はない**。今回ユーザーが追加実行不要と指定したため、この原因修復・再実行は行わない。過去のSTOPを成功へ書き換えず、凍結recipeも移送結果から変更しない。

重要科学レビュー、H6等への拡張、新surrogate、論文化GOは実施していない。今回依頼範囲の次の作業なし。

## 保存場所と来歴

- [1.00 Å geometry完了結果](../../artifacts/research_c_integrated/2026-11-06-picklefix/development/run01/result.json)
- [比較12セルTSV](../../artifacts/research_c_integrated/2026-11-06-picklefix/summary_h4_100/comparisons.tsv)
- [final64費用内訳・全6指標TSV](../../artifacts/research_c_integrated/2026-11-06-picklefix/summary_h4_100/final_costs.tsv)
- [同J/U D19候補の内訳TSV](../../artifacts/research_c_integrated/2026-11-06-picklefix/summary_h4_100/deterministic_costs.tsv)
- [全36 checkpointのDF/一次資格TSV](../../artifacts/research_c_integrated/2026-11-06-picklefix/summary_h4_100/optimization_checkpoints.tsv)
- [selection全108 policy TSV](../../artifacts/research_c_integrated/2026-11-06-picklefix/summary_h4_100/selection_rows.tsv)
- [機械可読要約](../../artifacts/research_c_integrated/2026-11-06-picklefix/summary_h4_100/summary.json)／[保存JSON転記スクリプト](../../artifacts/research_c_integrated/2026-11-06-picklefix/summary_h4_100/summarize_saved.py)
- [固定SOURCE](../../artifacts/research_c_integrated/2026-11-06-picklefix/technical/SOURCE.json)／[固定契約](../../artifacts/research_c_integrated/2026-11-06-picklefix/technical/contract.json)／[原STOP](../../artifacts/research_c_integrated/2026-11-06-picklefix/INTEGRATED_STOP.json)

SOURCE SHA256 `8113f23e66c5eca3c09a8ed1c91fc58c01eeaa91d0c07850b685a20c816539eb`、契約SHA256 `fd111609d72d2a2d4ded7415b3e622c0b3a8951f55591809794ef3e30c867f11`。今回の転記時にSOURCE closure386件のhash一致を確認した。コード・入力・raw・契約・STOPは変更していない。ローカル実行の結果を収録したもので、clean checkoutからの科学再実行や外部再現済みの証拠ではない。

## Gitへの収録範囲

ユーザーのコミット・プッシュ指示に従い、今回の実装・人工テスト、固定SOURCE/契約と必要入力、比較/最終評価/校正のJSON、TSV要約、技術ベンチマーク、過去および今回のSTOPを収録する。約14 GiBの生成済み全raw（QPY、dense配列、標本ごとの詳細診断、長いtrajectoryログ）は元のサーバーworktreeに保全し、Gitへの収録対象から外す。Gitの結果集は全rawの配布物ではなく、rawを参照する保存path/hashも元のままである。

収録直前の人工回帰確認は評価環境23件、fit環境8件がPASS。新しい分子計算、1.15 Åの修復・再実行、重要科学レビューは実行していない。
