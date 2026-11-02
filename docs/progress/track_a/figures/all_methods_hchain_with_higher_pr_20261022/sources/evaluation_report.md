# H2/H4/H6高次PFを用いたPR：保存値統計・測定込み総費用の結果

2026-10-21。状態：**SAVED_VALUE_EVALUATION_PASS_MANDATORY_STOP**。

通常signed・保存経験的allowance倍率1による有限探索比較では、緩い精度でS2が低費用、厳しい精度で高次PFが有利になった。ε=.0005の最良PRはH2がYoshida4/q2、H4/H6がm5_best/q1であり、各系のS2最良に対する削減率は約47.12%/53.30%/63.12%。これは保存探索標本の点費用結果であり、独立確認済み優位や母平均保証ではない。

## 1. 条件・契約・対象

線形H2/H4/H6、隣接1 Å、STO-3G、tol-only1e-8、保存DF rank3/10/19・prefix1/3/5、T=.8、状態準備費用P=0、K4、各outerの全piece合計32 short steps、q=1/2/4/8。S2、3段Yoshida4、固定m5_best binary64版のsigned scheduleを維持した。full Hadamard wrapper、cosine/sine paired、symmetric-directional、scalar once、保存Qiskit1.3.0・rz/sx/x/cx・opt1・seed17。H2だけに既存のcompiler位相帳簿guardがあり、その取得時監査も維持した。同一compiler SOURCEだったと読み替えない。

| 系 | 固定結果commit | 取得科学SOURCE | 使用した新探索trajectory / 既取得compile軸 |
| --- | --- | --- | ---: |
| H2 | a05f4ed19 | 4ec26921 | 525 / 1050 |
| H4 | 605ca07a6 | 5a448d94 | 793 / 1586 |
| H6 | 605ca07a6 | 5a448d94 | 607 / 1214 |

pilot・旧n8・確認標本は非混合。3公式×4q×3系のPR36群に加え、保存H4/H6 m5決定論64候補を同じ保存signal・適格性へ結合した。決定論単価は48候補96軸（新92/保存再利用4）が直接実測区分である。残る16候補の欠測単価や不適格状態を0補完しない。

実行契約は[こちら](track_a_h246_saved_cost_execution_20261021.md)。利用者の当初の保存値解析承認と、前報§3の採択案に対する「続きを行って」を、本解析だけの条件付き政策による再開指示へ結合した。旧未採択記録・実行前STOPを変更せず、共通v2全体やAD03の採択を変更していない。

## 2. shot・統計・費用の定義

5精度ε=.05/.01/.005/.001/.0005、軸精度ε/√2、α_axis=.025、保存経験的u倍率1、保存log_B+log_B_margin、headroom正かつu/headroom≤.01。凍結D0のbinary64演算順で整数ceilを計算し、LOG_DOMAIN_ONLYは整数shot・費用をnullにする。通常signedの保存適格性は全行で再現した。reference精度条件付きの経験的会計であり、厳密bias・normalization上界・全状態保証ではない。compiled vectorの1e-7受入閾値をuへ流用していない。

`G_i=N_cos C_cos,i + N_sin C_sin,i` のtrajectory平均・不偏分散・SEを保存した。cosine/sineはpairedで、分散には共分散項を含めた。単価・Gのmeanは整数十分統計から有理数で保持する。表の±は標本平均の1 SEで、母平均の厳密区間ではない。analytic shotを使う資源予測であり、今回は量子測定shotを実行していない。

## 3. 通常signed：公式ごとの4q内探索点最良

費用単位はcompiled RZ。各セルはq / mean G ± SE。改善率は `(G_S2−G_high)/G_S2×100` で、負値は高次PFの費用増加を表す。全軸bias・u・normalization・shot・単価・費用は[1000行表](../../artifacts/resource_applicability/track_a_h246_saved_cost_results/2026-10-21/candidate_precision_costs_1000.csv)、選択結果は[30条件比較表](../../artifacts/resource_applicability/track_a_h246_saved_cost_results/2026-10-21/formula_comparison_30.csv)を参照する。

| 系 | ε | S2 | Yoshida4 | Y4改善率 | m5_best | m5改善率 | 3公式内の点最良 |
| --- | ---: | --- | --- | ---: | --- | ---: | --- |
| H2 | 0.05 | q1 / 1.545995e+07 ± 1.756e+05 | q1 / 2.300355e+07 ± 1.903e+05 | -48.794% | q1 / 4.838221e+07 ± 1.651e+05 | -212.952% | S2/q1 |
| H2 | 0.01 | q1 / 5.255063e+08 ± 5.969e+06 | q1 / 6.048640e+08 ± 5.005e+06 | -15.101% | q1 / 1.209463e+09 ± 4.128e+06 | -130.152% | S2/q1 |
| H2 | 0.005 | q2 / 3.312556e+09 ± 2.629e+07 | q1 / 2.582020e+09 ± 2.136e+07 | +22.054% | q1 / 4.837913e+09 ± 1.651e+07 | -46.048% | Yoshida4/q1 |
| H2 | 0.001 | q4 / 1.710156e+11 ± 1.433e+09 | q2 / 1.110660e+11 ± 9.908e+08 | +35.055% | q1 / 1.209629e+11 ± 4.128e+08 | +29.268% | Yoshida4/q2 |
| H2 | 0.0005 | q4 / 8.758941e+11 ± 7.338e+09 | q2 / 4.631486e+11 ± 4.131e+09 | +47.123% | q1 / 4.839281e+11 ± 1.652e+09 | +44.750% | Yoshida4/q2 |
| H4 | 0.05 | q1 / 1.257833e+08 ± 1.237e+06 | q1 / 2.455294e+08 ± 2.714e+06 | -95.200% | q1 / 5.260143e+08 ± 1.423e+06 | -318.191% | S2/q1 |
| H4 | 0.01 | q2 / 6.541333e+09 ± 6.396e+07 | q1 / 9.191659e+09 ± 1.016e+08 | -40.517% | q1 / 1.315076e+10 ± 3.558e+07 | -101.041% | S2/q2 |
| H4 | 0.005 | q4 / 4.250574e+10 ± 4.839e+08 | q2 / 4.192572e+10 ± 2.830e+08 | +1.365% | q1 / 5.260782e+10 ± 1.423e+08 | -23.766% | Yoshida4/q2 |
| H4 | 0.001 | q8 / 2.188165e+12 ± 1.563e+10 | q2 / 1.336462e+12 ± 9.021e+09 | +38.923% | q1 / 1.316191e+12 ± 3.561e+09 | +39.850% | m5_best/q1 |
| H4 | 0.0005 | q8 / 1.128521e+13 ± 8.061e+10 | q4 / 7.663968e+12 ± 5.896e+10 | +32.088% | q1 / 5.269758e+12 ± 1.426e+10 | +53.304% | m5_best/q1 |
| H6 | 0.05 | q1 / 5.770514e+08 ± 4.808e+06 | q1 / 1.484050e+09 ± 1.231e+07 | -157.178% | q1 / 2.164411e+09 ± 5.135e+06 | -275.081% | S2/q1 |
| H6 | 0.01 | q2 / 3.111057e+10 ± 2.447e+08 | q2 / 4.793657e+10 ± 2.755e+08 | -54.085% | q1 / 5.411539e+10 ± 1.284e+08 | -73.945% | S2/q2 |
| H6 | 0.005 | q4 / 1.717742e+11 ± 1.787e+09 | q2 / 2.011456e+11 ± 1.156e+09 | -17.099% | q1 / 2.164988e+11 ± 5.136e+08 | -26.037% | S2/q4 |
| H6 | 0.001 | q8 / 9.289933e+12 ± 7.621e+10 | q2 / 7.825858e+12 ± 4.497e+10 | +15.760% | q1 / 5.420084e+12 ± 1.286e+10 | +41.656% | m5_best/q1 |
| H6 | 0.0005 | q8 / 5.889488e+13 ± 4.832e+11 | q4 / 3.277344e+13 ± 1.714e+11 | +44.353% | q1 / 2.171853e+13 ± 5.152e+10 | +63.123% | m5_best/q1 |

H2はε=.05/.01でS2、.005/.001/.0005でYoshida4が最小。m5も.001/.0005ではS2より低費用だが、今回の5精度の3公式比較では最小にならなかった。H4は.05/.01でS2、.005でYoshida4、.001/.0005でm5。H6は.05/.01/.005でS2、.001/.0005でm5。したがって、高次PFの導入による改善は精度と系に依存し、m5を常に最良とする結果ではない。

候補ごとのnは33〜255で異なる。[単価統計36群](../../artifacts/resource_applicability/track_a_h246_saved_cost_results/2026-10-21/unit_statistics_36.csv)に軸別mean/variance/SE/covarianceとnを対応させた。最良qは同じ探索標本で選択したもので、選択効果の補正・全格子母平均最適性の保証はない。

## 4. bias・shot・有効単価による費用分解

総費用比は、同じcohortの両軸を使う `shot総数比 × shot重み付き有効単価比` へ有理数で正確に分解した。有効単価は `G_mean/(N_cos+N_sin)` であり、cos/sin単価の単純平均で代用しない。[全60比較の分解表](../../artifacts/resource_applicability/track_a_h246_saved_cost_results/2026-10-21/cost_decomposition_60.csv)と全候補表にbias/headroom/normalizationを結合した。

| 系・ε | 比較（高次/S2） | shot総数比 | 有効単価比 | 総費用比 |
| --- | --- | ---: | ---: | ---: |
| H2 / 0.001 | Yoshida4/q2 ÷ S2/q4 | 0.867368 | 0.748759 | 0.649450 |
| H2 / 0.0005 | Yoshida4/q2 ÷ S2/q4 | 0.706198 | 0.748759 | 0.528772 |
| H4 / 0.001 | m5_best/q1 ÷ S2/q8 | 0.849790 | 0.707827 | 0.601505 |
| H4 / 0.0005 | m5_best/q1 ÷ S2/q8 | 0.659711 | 0.707827 | 0.466961 |
| H6 / 0.001 | m5_best/q1 ÷ S2/q8 | 0.756886 | 0.770838 | 0.583436 |
| H6 / 0.0005 | m5_best/q1 ÷ S2/q8 | 0.478399 | 0.770838 | 0.368768 |

これは公式自体の回路が常に短いという説明ではない。高次PFでは同じqの合成段数が増える一方、許容biasを満たす小さいqを使える場合がある。例えばH6 ε=.0005ではS2/q8に対するm5/q1のshot比0.4784と有効単価比0.7708が合わさり、総費用比0.3688となった。一方、同精度のYoshida4/q4は有効単価比1.0141とわずかに増えても、shot比0.5487によりS2より低費用だった。通常bias、normalization、headroomをまとめた会計上の説明であり、単一因子の因果効果を実験的に分離した主張ではない。

## 5. 非相殺診断

PRは各軸 `|outer_PF|+|finite_RTE|`、full PFは`|PF|`を使用し、通常signedとは別の行へ保存した。PRの保存成分closureを照合したうえで、同じ宣言済みshot政策を適用した。この診断でも3公式内の最良formula/qの選択は全15条件で通常signedと同じだった。一部の整数shot・費用はroundoff級の成分差によって変わるため、値を同一として省略していない。

B0/B0Iの保存signalはPFとdiscardの合計成分しかなく、それらを非相殺の別成分へ分ける証拠がない。56候補×5精度の280行は`MISSING_COMPONENTS`・費用null。m5決定論集合の非相殺最良は集合全体が未解決となるためnullであり、full PFだけへ候補集合を狭めた最良を代用しない。非相殺PRの安定性は、この保存成分と経験的allowanceの条件下の診断であり、厳密な非相殺bias上界ではない。

## 6. 同じnative系列に結合したm5決定論部分集合

H4/H6では固定conditionsのinput binding、compiler hash、候補method/prefix/q、preparation hashを信号・PR・実測単価へ照合した。新規実測と保存再利用を来歴で保持する。以下は**m5決定論部分集合だけ**の通常signed比較であり、旧強化S2/S4を含む全PF最良との比較ではない。

| 系 | ε | m5決定論部分集合の最良 | G_det | 最良PRのG | PRの対部分集合改善率 |
| --- | ---: | --- | ---: | ---: | ---: |
| H4 | 0.05 | B0I p8/q1 | 9.733070e+08 | 1.257833e+08 | +87.077% |
| H4 | 0.01 | B0I p8/q1 | 2.433941e+10 | 6.541333e+09 | +73.125% |
| H4 | 0.005 | B0I p8/q1 | 9.740148e+10 | 4.192572e+10 | +56.956% |
| H4 | 0.001 | B0I p8/q1 | 2.443901e+12 | 1.316191e+12 | +46.144% |
| H4 | 0.0005 | B0I p8/q1 | 9.820341e+12 | 5.269758e+12 | +46.338% |
| H6 | 0.05 | B0I p11/q1 | 3.490436e+09 | 5.770514e+08 | +83.468% |
| H6 | 0.01 | B0I p11/q1 | 8.743677e+10 | 3.111057e+10 | +64.419% |
| H6 | 0.005 | B0I p11/q1 | 3.506635e+11 | 1.717742e+11 | +51.015% |
| H6 | 0.001 | B0I p11/q1 | 8.953595e+12 | 5.420084e+12 | +39.465% |
| H6 | 0.0005 | B0I p11/q1 | 3.678563e+13 | 2.171853e+13 | +40.959% |

H2の新outer32系列へ同じidentityで結合した決定論signal集合は今回なく、対照比較はnull。旧H2成果を消去した意味ではない。旧H4 legacy DF12、別PF788研究、旧S2/S4のmodel推定を新たな直接実測として混合していない。全強化PF集合へ対するPR優位性の確定は、この部分集合比較からは主張しない。

## 7. 統計的不確かさと独立確認候補

全36群でS4は未観測。S2_multiはH6 Yoshida4/q1で1観測、他群では未観測だった。[144層記録](../../artifacts/resource_applicability/track_a_h246_saved_cost_results/2026-10-21/observed_strata_144.csv)に0観測層の条件付き費用・分散はnull、1観測層の分散はnullとして保存した。未観測費用を0へ置かず、観測SEが小さいことから母平均費用・tail・真の分散を保証しない。

標本のcos/sinはpaired、同じ候補の5精度では同じtrajectoryを共用する。異なる候補の費用差SEでは独立cohortを仮定し、候補間paired構造を仮定しない。測定shotのα_axis=.025は、古典的費用標本の多重比較・順位選択に対する誤り保証ではない。本報告に選択後補正、母平均CI、勝者保証は付けていない。

| 優先する研究判断 | 独立確認の候補・比較 | 判断材料 |
| --- | --- | --- |
| 厳しい精度での改善再現 | ε=.0005：H2 S2/q4 vs Yoshida4/q2、H4/H6 S2/q8 vs m5/q1 | 47.12%/53.30%/63.12%の探索点削減率を新cohortで確認する価値がある |
| 僅差で結論が変わり得る比較 | H4 ε=.005：S2/q4 vs Yoshida4/q2 | 探索点差は約1.36%。SE・選択効果の影響を優先して確認 |
| 最良4次公式の選択 | H4 ε=.001：m5/q1 vs Yoshida4/q2 | G約1.316e12 vs1.336e12の近接。両候補を事前固定して比較する候補 |
| 改善しない側の再現 | ε=.05：S2/q1 vs各4次/q1 | PRの高次化が常に有利ではないことも有効な確認結果になる |

これは候補・目的の提案で、n_Cを設計・採択していない。目的はまず独立標本の有限費用順位の再現であり、希少eventを含む母平均保証を目的にする場合は別のtail設計・根拠が必要。n48などを一律に要求せず、比較差・分散・希少層を踏まえて別設計とする。新sampling/compileは未実施・未承認。

## 8. 数値経路、SOURCE、終了監査

- 解析SOURCE：`9a77d2b0a0b87e246e85f085b37f5923c18e27bb`。plan SHA：`c2a9cce59b75abbc1669d60600e7f9fa5e8e016f2efa9a722e64e7f340057a71`。
- private科学結果SHA：`b6d1309fc615c6df2c3a65c81f0a62dfff327435e98fde3bc2d74f3b80504f5e`。終了監査原本SHA：`db71559e7761d5775b36ecaec68feacf45e15efcb1329800a229c75af9c829bc`。
- exact integer/rational十分統計とstdlib/中心化floatのmean・不偏分散・共分散照合PASS。Gの共分散恒等式とshot×有効単価の比の恒等式は有理数完全一致。
- 凍結binary64 shotと、Decimal基本演算を段階ごとbinary64へ丸める経路で整数完全一致PASS。sqrt/exp/logのlibm値は共有するため、それらの独立library実装の照合ではない。作者自身が行った二つの数値経路の検証で、独立AIレビューではない。
- 人工23件PASS。実行は候補100/精度bias行1000/まとめ30、64 thread、native exit0/reap済み。科学処理wall約2.23秒、外部wall約3.20秒、child maxRSS53664 KiB。新sampling/signal/simulation/回路生成/compile/fitは0。
- 最初の封印準備だけが組込みmathのfile属性で停止した。grant・科学root発行前で科学call0。旧準備STOPを保持し、封印方法を修正した別packで科学実行を一回行った。科学retry/resumeは0。

[独立数値照合](../../artifacts/resource_applicability/track_a_h246_saved_cost_results/2026-10-21/independent_numerical_checks.json)、[認可とcall範囲](../../artifacts/resource_applicability/track_a_h246_saved_cost_results/2026-10-21/grant_and_scope.json)、[入力来歴](../../artifacts/resource_applicability/track_a_h246_saved_cost_results/2026-10-21/input_provenance.json)、[終了監査](../../artifacts/resource_applicability/track_a_h246_saved_cost_results/2026-10-21/end_audit.json)、[データ索引](../../artifacts/resource_applicability/track_a_h246_saved_cost_results/2026-10-21/README.md)を保存した。seed一覧・private原本・QPY/NPZ・内部絶対pathは公開投影に含めない。旧SOURCE/raw/grant/ledger/marker/STOPは保全し、既存契約の遡及書換えはない。

[最終構造検証](../../artifacts/resource_applicability/track_a_h246_saved_cost_results/2026-10-21/structure_checks.json)では、全CSVと保存JSONの対応、null・exact有理数mean、SOURCE、旧worktree、リンク、公開投影を検査してPASSした。資料構造検査中に一度processがBus errorで終了し、その後のtool審査も一時的な利用上限で実行できなかった。科学計算・終了監査はそれ以前に正常終了しており、結果SHAは不変。再開後は資料の照合だけを実行し、科学解析・消費済みgrantの再実行は行っていない。

## 9. 終了と次の担当

**mandatory STOP。local commit後、公開pushは行わない。** AD03保留、詳細科学レビュー・最終freeze・論文本文作成は開始していない。

**ユーザー**：報告を確認し、独立確認で優先する系・精度・候補・目的と、公開対象範囲を判断する。**Codex**：別途指示があれば、その比較に必要な新cohort・n_C・固定shot/費用条件を設計する。今回のgrantで追加sampling/compileや新しい解析へ進まない。
