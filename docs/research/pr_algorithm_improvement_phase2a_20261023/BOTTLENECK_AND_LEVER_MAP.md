# PR費用と改善自由度の対応

この表は保存された異なるtaskの費用構造を並置した資料で、新しい費用計算・支配率推定ではない。**PR全般について支配的と確定した単一項はない。** circuit単価、統計的反復回数、前処理を別々に扱う。異なる系・表現・target・compilerの値を足し合わせない。

区分Aは明記した保存条件で実際に費用差／支配要因を確認、Bは理論/SOURCE上必要だが寄与未測定、Cは局所削減は可能だが全資源寄与不明、Dは強いbaselineで対処済み。区分は項目・条件ごとで、研究の順位ではない。Aの局所証拠を全PRの支配性へ拡張しない。

|費用層・項目|証拠区分と判明範囲|改善自由度／未解決の問い|強い対照で除くもの|根拠|
|---|---|---|---|---|
|Hamiltonian：rank・項数・norm|A：REP-09固定toyでnorm節約よりbasis/central費増が大きい。B：分子一般の費用比は不明|RQ-01 表現・軌道と生成費の両方を変えられる構造|既知Pauli collection、DF/RC-DF・BLISS・GRADEの再命名|EV-0197/0200、EV-0179|
|Hamiltonian：表現誤差・sector|B：model差、shiftのtarget sector、scalar phaseを保持する必要。旧UWC値は未照合|RQ-01 同じ物理taskでの表現変更|異なるH/sectorの結果、stale Cgs、未照合UWCの採用|EV-0007/0148、P2-GAP2S|
|split/group：backbone対tail|A：PR-2固定集合で1-shot低下とshot増の交換。B：det/basis/random別compiled寄与は欠測|RQ-02 非prefix/groupの増分機構|prefixと異なるだけの新規性、同情報対照への探索禁止|P2-ATTR、EV-0066、EV-0073|
|commutator・order・rate|D：BMの同じ三次係数・cacheはcompact対照でも可能。B：別構造の有限step利益|RQ-02 既知評価を超える構造は何か|同式を整理し直しただけの方法差|EV-0084、P2-PD|
|PF/近似誤差・時間分割|A：PR-3はbiasを減らしてもvariance/backbone費で負けた。P-D/B-Fは強対照で選択一致|RQ-02/06 誤差を減らす価値がどの費用で相殺されるか|leading改善だけの優位、四次PF逆転原因の新規調査|EV-0068/0070、P2-PD、EV-0081|
|randomization：normalization・moment/range|A：SP/S4/G9の固定モデルで単価との交換。B：新lawの有限confidence・実測shotsは未確立|RQ-04/05 同mean生成と統計負担|normalization、必要shots、shot実測、compile標本数の混同|EV-0086/0091、EV-0131/0133、EV-0115|
|randomization：parent縮約|A：C1の固定H4では非空parentが少なくlogical差が小さい。各pathにchild回転が残る|RQ-04/05 変える項に十分な費用寄与があるか|degree-zeroを無料/no-opとする前提|EV-0142/0147、EV-0144|
|circuit：basis共有・support|C：WP06b短列で削減可能。P-A interval分割固有差は支持なし|RQ-03 共通共有後に残る異basis移動|全event support限定、一区間対照を除く弱い比較|EV-0027/0030、EV-0032/0037|
|circuit：control・step境界|D：actual M2 SOURCEではbasisは無制御、diagonalのみ制御、同blockの境界半sweep融合済み|RQ-03 追加で省ける操作の特定|全basis制御／未融合という架空baseline|P2-BUILD:214–352、P2-REPEAT:591–623|
|circuit：native T/CX/depth|A：G9固定syntheticではT-primaryとCXの利害が異なる。C：SPは加法費用、全wrapper未測定|RQ-05 辞書・allocation・合成位置の残る機構|TをRZ/CXへ同一換算、加法価格をactual全compile扱い|EV-0115/0119、EV-0086、EV-0129|
|実装：前処理・古典sampling・取得|A：C1にsetup/online CPU観測あり、方式間の実装・計測境界が異なる。B：公平な全N総費用は未知|RQ-04 安い実行可能な生成アクセス|CPU観測を最適算法順位とすること、未測定を0とすること|EV-0142、EV-0139、EV-0132|
|実装：prep/readoutと初期情報|B：多くの分子wrapperでprep除外。P感度は仮想単価。共通per-shot費でもNが違えば相殺しない|RQ-01/04/05/06の成立条件|共通準備を常に無視すること、oracleを無料にすること|EV-0061、EV-0072、EV-0134 §§7|
|複数round：時刻・精度・構成|A：WP固定局所設計でround配分に費用差。後続all-r opt2は完了し区間はなお重なる。D：r/K・beta/alpha配分は既存。B：表現/split変更の共通energy bridge|RQ-06 同じE0を読む条件と増分節約|単一Tやepsilon sweepを全RPE成功とすること、完了したall-r再最適化を再要求すること|EV-0038、P2-OPT2、EV-0072、EV-0134|
|接続性：routing・layout|B：主要資料はtopology-free。支配性も改善も未測定|RQ-07 既知mapping後にもPR固有問題があるか|記録不足を世界的新規性とすること|EV-0061、EV-0179、EV-0142|

## 保存条件付きの機構witness

|証拠|条件を保持した引用|そこから決められないこと|
|---|---|---|
|WP06b|H4 linear 1Å/STO-3G/DF12、L_D3、二次partial S2、δ=.02/K2、短列holdout長3/6、Qiskit1.3 opt1/seed17/topologyなし。run policyのpooled RZは保存報告で10.67%減|全wrapperの同率削減、全instanceへの移送|
|P-A後続|同H4/1Å/STO-3G/DF12/L_D3/δ=.02のforced-support Taylor order2、training15/blind15。全30 taskでcompile前planとcompiled metricsが一区間対照と同一|全ての区間最適化が不可能という定理|
|PR-2 M2|H4 linear1.30Å/STO-3G/DF12、T=.8。B2 L_D3 q1 δ=.8 r4/8 K2、B0 L_D6 q1、B1 L_D12 q1、B3 L_D0 q8 δ=.1 r32 K4。εcomplex=.05、32 paired cost trajectories、full measured wrapper・prep除外、Qiskit1.3 opt1/seed17/topologyなし。保存primary比は0.586090、engineering upper2SE .593584|held-out再最適化、rank/q最適性、formal CI、最終RPE費用|
|C1|H4 linear1Å/STO-3G/DF12/R1 generation-prefix L_D3、13 bases/324 labels/8 system q、T=4/5/q1/r4/m5、microstep1/5、各1024 path。TAIL_ONLY_PLUS_SCALAR。各pathのchild_rotations=4、valid zero path=0、logical差は小さい|native/N/全wrapper優位・劣位、CPU最速順位、一般RAの否定|
|PR-3|固定2-qubit HD=.9Z0+.7X0X1、HR=.31X0−.27Z0Z1+.19Y1、化学basis/DF/L_D非該当。T=.8、qDRIFT tail level4/8/16、固定state、complex RMSE .05。保存報告は外挿bias減少にもかかわらずtotal primitive workが通常最良の7.557倍|外挿一般の無効、化学PRの勝敗、exact-Hとの差の全帰属|

出典：[EV-0027](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/docs/research_direction_sequence_policy.md)、[EV-0032](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/docs/research_direction_joint_synthesis_mechanism_validation.md)、[EV-0061](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e3531956af6a6d359e2f1fb320e9a5179f41eb98/docs/pr2_matched_accuracy_m2_transfer_result_validation.md)、[EV-0066](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e3531956af6a6d359e2f1fb320e9a5179f41eb98/artifacts/pr2_matched_accuracy_m2_transfer_execution/2026-10-04/pr2_matched_accuracy_m2_transfer_result_v2.json)、[EV-0142](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/185ec4e9d6591f4ee8d7b047c2697bed7be7bc2e/docs/research/track_b_c1_freeze_synthesis_20261021/C1_RESOURCE_EVIDENCE_AND_LIMITATIONS.md)、[EV-0147](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bb9059bacb697e1105e63c1b3826b0c34886ad15/docs/research/track_b_c1_5a_p5_worker_recovery_20261021/result/summary.json)、[EV-0068](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/docs/pr2_pr3_minimal_pilot_validation.md)、[EV-0070](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/artifacts/pr2_pr3_minimal_pilot/2026-09-27/pr2_pr3_minimal_pilot_v1.json)。すべて保存値の転記で、比・分散・順位を新たに計算していない。

## 課題定義へ使う際の限界

PR-2のone-shot総RZからbasis、deterministic、random primitive別の費用比を作らない。C1のparent degree頻度から全量子利得率や母momentを推定しない。H8希少tailは既存の層別有限標本・条件付き感度の範囲で終了済みで、追加samplingをこの資料の必要条件にしない。C_useは実施δ窓の経験的envelopeであり厳密上界ではない。H12係数・最終総費用は未決定／未実施。
