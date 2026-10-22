# PRアルゴリズム改善研究の横断地図

2026-10-23 JST。資料・SOURCE・保存結果の**回顧的な整理**である。新しい科学計算、保存rawの再解析/再ランキング、非自明な新証明、回路生成/compile、重要科学レビュー、研究の正式採択は行っていない。既存SOURCE・status・STOPは変更しない。

調査対象の研究を「方法のfamily」としてまとめ、retry・修正・独立確認を研究件数に加算しない。全体の関係を本書、各研究の詳細を[70研究カード](STUDY_CARDS.md)、機械可読情報を[カタログ](STUDY_CATALOG.json)で読む。資料不足は[EVIDENCE_GAPS](EVIDENCE_GAPS.md)、候補は[OPPORTUNITIES](PR_IMPROVEMENT_OPPORTUNITIES.md)。

## 評価層と研究間の関係

| 評価層 | 主な系列 | 接続するときの条件 |
|---|---|---|
| 元Hamiltonian・sector・表現 | REP-01〜09、QC-03/07/09〜14 | 同H、scalar、basis、rank/order、近似biasを保持 |
| PFとsplit | PF-01〜07、OPT-02〜05、QC-01/02/05/06 | 係数版・負stage・内部HD・同情報strong baseline |
| finite random mean・law | PR-01〜03、RB-01〜18 | ordered first moment、sign、全trial denominator、finite bits |
| controlled wrapper・native生成 | CIR-01/02、RB-03/07〜14/17 | ordinary control、actual adjoint、relative phase、同compiler/価格 |
| signal・shots・総資源 | OPT-01、ERR-01、EVAL-01〜03、RB-15 | bias/radius/range、共通confidence、状態準備の支払い単位 |

この表は依存関係であり、異なるtaskの倍率を掛け合わせて総利益を作るための表ではない。QPE/RPE統計誤差をPF係数Cへ混ぜず、C_useは実行delta窓の経験的envelopeとして扱う。H12係数や最終全round総費用は推定しない。

## 主要な肯定的成果と適用域

### run-level回路共有

H4の固定DF event列でfull/supportの使い分けが有効。P-A後続で消えたのはinterval subdivision固有claimであり、既存run-level改善は保持する。 [CIR-01 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/docs/research_direction_sequence_policy.md)

### 圧縮＋残差補完とmatched task

H4/STO-3G/DF12のdevelopmentと固定H4 1.30 Å transferで限定的なresource競争力を支持。新算法・全rank最適・全energy/RPE優位を示す結果ではない。 [PR-01 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e3531956af6a6d359e2f1fb320e9a5179f41eb98/docs/pr2_matched_accuracy_m2_transfer_result_validation.md)

### return構成と低次数特殊化

formal return/局所生成/finite bitsの構成成果を保持する。RA、General Full、Closed P5の役割を区別し、H4 logical pilotをnative/N保証へ格上げしない。 [RB-06 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f750a23f023d75c8c4fbcabc6937d6953ce61f2/docs/research/track_b_g6_v1_2_draft_review_integration_20261022/handoff.md)

### nativeでの条件付きpositive

固定3qubit/P5 provider・同finite target・登録policyでT-primaryの有効点があり、matched CTSにはCXの利点がある。全資源の優越ではない。 [RB-09 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/g9_v2_results_and_gpt_handoff_20261010.md)

### 同meanの保証・価格・境界の会計

D1–D3はfinite P5の同target比較、native相対位相、literal境界相殺、保証×価格の交換を保存。十分Nのforecastと実shot数を区別する。 [RB-13 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/d1_d3_prior_art_claim_evidence_and_gpt_handoff_20261014.md)

### 共有labelの限定理論

M1-SのC-sec理論とH4局所信号witnessは保持。phase biasとvisibilityを同時に見て、Task Aの単一時刻をTask Bの全RPEへ外挿しない。 [RB-15 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f079dbe67bb35eff35613857b169fdf812af4b8/docs/research/m1s_integration_20261017/integrated_research_summary.md)

### 表現比較基盤

H2のPauli/DF各PF/PRの固定比較が存在する。H4は技術準備までで、未観測Taylor次数の母平均費用は未知。 [REP-08 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c8990de7721545faff88b636108034dedb80a463/docs/non_df_h2_process_recovery_validation.md)

## 負の結果・停止理由が研究地図に与える制約

| 系列 | 保持する結果 | 次案へ持ち込む制約 |
|---|---|---|
| [CIR-02](STUDY_CARDS.md#cir-02) | 強い一区間baselineとの非退化比較で独立差なし。 | 既存共有policyの効果をDP固有の効果としない。 [CIR-02 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/docs/research/pa_joint_synthesis_v1_formalization.md) |
| [OPT-03](STUDY_CARDS.md#opt-03) | 局所geometry予測の後、stretch移送とtrackingが不支持。 | interpolation成功をPES全体・新geometryへ拡張しない。 [OPT-03 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/docs/research_direction_geometry_energy_difference_pilot.md) |
| [OPT-04](STUDY_CARDS.md#opt-04) | pilotでの選択差は、強い公平baselineで同じ選択へ。 | weak outer-stage対照だけの利益を新協調法としない。 [OPT-04 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/docs/research_direction_energy_tail_pareto.md) |
| [PR-02](STUDY_CARDS.md#pr-02) | tail bias低減がshot/backbone費で不利。 | biasやnormalizationだけで研究成功を決めない。 [PR-02 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/docs/pr2_pr3_minimal_pilot_validation.md) |
| [REP-09](STUDY_CARDS.md#rep-09) | 小さいlambda/shot利益をbasis/central費用が上回る。 | 表現変換のnative費用を全taskへ戻す。 [REP-09 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ae7434452450e80ec007d65417335e18a063fa1e/docs/research/representation_attribution_pair_results_20261010.md) |
| [RB-05](STUDY_CARDS.md#rb-05) | 指定辞書classではsaved CTS対照が優位。 | 任意sampler一般のno-goへ広げず同辞書の救済探索もしない。 [RB-05 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/g5_results_and_gpt_handoff_20261010.md) |
| [RB-11](STUDY_CARDS.md#rb-11) | 固定generic whole計画の限定費用分離。未確定条件も残る。 | 局所支持・追加weight1を総shot負担1としない。 [RB-11 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4301c0f87a16be45b620304c3fa935d5aa2d721a/docs/tracks/algorithm_codesign/df_correction_et_series_closure_20261012.md) |
| [REP-04](STUDY_CARDS.md#rep-04) | bias込みselectorはexact fallbackを選択。 | 表現近似の小回路をsame-accuracy利益としない。 [REP-04 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4c719a72eb8caae2f77fc746bd4672d610d97610/docs/research/n1_decode_recovery_v1_results_20261011.md) |
| [QC-05](STUDY_CARDS.md#qc-05) | 同情報既知法への還元と登録finite-step基準未達。 | 既知baselineへ同じ情報とcacheを与える。 [QC-05 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7af83266f96b714bb3fa2f1ef5e4bc89faf37db3/docs/research/qc_onebody_repartition_h_boundary_reaudit_v2_results_20261015.md) |
| [QC-08](STUDY_CARDS.md#qc-08) | 理論/checker成立後、強いcentered対照の十分stepが大きい。 | 新保証成立と資源改善を区別する。 [QC-08 固定資料](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f7e103bf3a0ffcee6162bd4b8d244f9a2f4502c5/docs/research/qc_a2_current_route_handoff_20261018/achievements_and_limits.md) |
| [PF-04](STUDY_CARDS.md#pf-04) | 固定S4はno-benefitで完了。 | negative completionを未完了や追加救済義務としない。 [PF-04 固定資料](https://github.com/HIROMU1015/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/blob/6a1e54d5e830a20b8817791f9f1f65729815fc12/PF_first_study_final_synthesis_20260925.md) |

## 技術停止・資料不足・未実行は別分類

旧PR-2 S0、G9/G10の旧実行、DF E/Tの保存停止、H2のthread競合、C1旧P5 worker停止はそれぞれの原statusを保持する。別SOURCE/認可の後続完了やsaved-only recoveryがあっても旧実行を成功へ書き換えない。主要な版の対応は[各カードの履歴](STUDY_CARDS.md)へ記録した。

UWCとstale DFは重要資料不足で成否未判定。M1-S旧行列、旧H4来歴、historical H01等の不足は、現在の限定成果を一括して無効化するものではない。未実施のstrict native/十分N/全RPEを「回収すれば必ず見つかるファイル」と扱わない。

C2/C3/C4、H1–H4、exact encoding、Gaussian積、J1/Q1/集団遷移等では、既知生成への還元、具体規則不足、固定ansatzの障害、未実装を分ける。研究全体の不可能性や世界的新規性の決着は認定しない。

## 分野別カタログ

次表の分類は調査専用。A/B/Cは資料十分性で、科学的成績の評価ではない。数値・比較の完全な条件は各カードと固定資料に置く。表の順番は優先順位ではない。

### REP

| 研究 | 実際の結果・停止理由 | 区分 | 資料 |
|---|---|---|---|
| [REP-01](STUDY_CARDS.md#rep-01) Pauli groupingとDF回路基盤 | 実装基盤を確認。最新PRや全energy taskの優位性とは分離する | 実装・評価基盤 | B |
| [REP-02](STUDY_CARDS.md#rep-02) 旧DF rank screening | 旧screening値は現行科学結果として利用不可。履歴として保存 | 重要資料不足・判断不能 | C |
| [REP-03](STUDY_CARDS.md#rep-03) BLISS型UWC＋grouped PF | 本文上の小差・悪化訂正は実行メモ。数値的な改善成否を確定しない | 重要資料不足・判断不能 | C |
| [REP-04](STUDY_CARDS.md#rep-04) N1 representation selector | decode修正後も固定全contextでexactが選ばれ、certified目的改善なし | 改善なし／現固定構成不支持 | A |
| [REP-05](STUDY_CARDS.md#rep-05) N2 signed charge・L1 fit | weighted entry-L1訂正は成立。旧charge回路の費用増とfit改善を混同しない | 肯定と否定が混在 | A |
| [REP-06](STUDY_CARDS.md#rep-06) N3 occupation／active-energy certificate | 限定理論・保存幅改善は保持。元精度で追加縮約なし、強い既存族との差不足 | 肯定と否定が混在 | B |
| [REP-07](STUDY_CARDS.md#rep-07) THRIFT／mixed relative evolution | ideal step削減はあるが追加算術を回収する登録予算不足 | 肯定と否定が混在 | A |
| [REP-08](STUDY_CARDS.md#rep-08) 非DF Pauli対DF fixed-signal比較 | H2固定4方式の条件付き比較完了。母平均rare-order費用は未同定、H4は人工準備のみ | 技術成果あり・科学的利益未確定 | B |
| [REP-09](STUDY_CARDS.md#rep-09) core吸収・physical pair表現 | 強いwhole-Pauliでcore中心の利得が消え、pairのshot節約はbasis/central費用に負けた | 改善なし／現固定構成不支持 | A |

### PF

| 研究 | 実際の結果・停止理由 | 区分 | 資料 |
|---|---|---|---|
| [PF-01](STUDY_CARDS.md#pf-01) Yoshida・Morales・新4次係数の比較 | PF family内の限定比較がある。高次数が常に有利とはしない | 限定条件で改善 | A |
| [PF-02](STUDY_CARDS.md#pf-02) Processed PF／YP8m8 | kernelとprocessed回路の検証・費用区分を保存。PR全費用への利益は未確定 | 技術成果あり・科学的利益未確定 | A |
| [PF-03](STUDY_CARDS.md#pf-03) PF固有位相・D6係数とsurvival代理量 | survival phaseとQPE RMSEは同じ量でない。C_useは実行窓の経験的envelope | 評価基盤・限定検証 | A |
| [PF-04](STUDY_CARDS.md#pf-04) 有限時間校正・第一研究S0–S4 | 機構分離は成立、固定S4診断はcomplete_no_benefit。研究未完了と解釈しない | 改善なし／現固定構成不支持 | B |
| [PF-05](STUDY_CARDS.md#pf-05) 有限時間校正・第二研究Direction C | H-chain限定支持と一般分子への不十分な移送、情報追加後のbudget利得なしを保持 | 肯定と否定が混在 | B |
| [PF-06](STUDY_CARDS.md#pf-06) FS-C旧入力回収とFS-R1校正follow-up | 旧回収失敗を保存。new-source Phase A freezeは成立、利益採点は未実行 | 技術成果あり・科学的利益未確定 | B |
| [PF-07](STUDY_CARDS.md#pf-07) BCH高次項・response・direct-scaling | 限定identityや予測成功はあるが、モデル増量の単調改善・サイズ普遍則・net資源利益は未確立 | 肯定と否定が混在 | B |

### CIR

| 研究 | 実際の結果・停止理由 | 区分 | 資料 |
|---|---|---|---|
| [CIR-01](STUDY_CARDS.md#cir-01) WP06-a/b full/support-union基底計画 | 局所列で削減を観測。一律support限定は混在列で悪化し得る | 限定条件で改善 | A |
| [CIR-02](STUDY_CARDS.md#cir-02) P-A interval-union DP | 初期blind削減は保持するが、後続非退化比較で一区間baselineとの差0。interval固有claimを停止 | 改善なし／現固定構成不支持 | A |

### OPT

| 研究 | 実際の結果・停止理由 | 区分 | 資料 |
|---|---|---|---|
| [OPT-01](STUDY_CARDS.md#opt-01) round別r/K・誤差/失敗確率配分 | 共通配分が大きく効き、強い対照とcompiler感度で頑健な区間優位は未確立 | 肯定と否定が混在 | A |
| [OPT-02](STUDY_CARDS.md#opt-02) P-B signal-aware selection | 現gridに実用的なselection差がない。一般signal問題の否定ではない | 改善なし／現固定構成不支持 | A |
| [OPT-03](STUDY_CARDS.md#opt-03) P-C geometry tracking／energy差 | 局所pilot支持。後続stretch予測破綻、tracked prefixは独立と同じ。主題候補を停止 | 肯定と否定が混在 | A |
| [OPT-04](STUDY_CARDS.md#opt-04) P-D energy/tail-aware高次PF選択 | pilot選択差は成立、強いB1b/B2/B4は同じ候補を選びfinite補正固有利得なし | 肯定と否定が混在 | A |
| [OPT-05](STUDY_CARDS.md#opt-05) R3 split／formulaのmulti-fidelity選択 | quantum-specific method deltaを固定できず数値pilotなし | 計画・理論／静的監査まで | A |

### ERR

| 研究 | 実際の結果・停止理由 | 区分 | 資料 |
|---|---|---|---|
| [ERR-01](STUDY_CARDS.md#err-01) FR 位相・半径分離 | 限定strict gainはあるが固定予算の実用認証差なし。直接帰結・反例・未解決を分離 | 技術成果あり・科学的利益未確定 | A |

### PR

| 研究 | 実際の結果・停止理由 | 区分 | 資料 |
|---|---|---|---|
| [PR-01](STUDY_CARDS.md#pr-01) PR-2 圧縮＋残差random補完 | development点最小と固定held-out構成のtransferを支持。一般rank最適・新算法・全RPE優位ではない | 限定条件で改善 | B |
| [PR-02](STUDY_CARDS.md#pr-02) PR-3 tail限定Richardson | biasは下がったがshotとbackboneを戻すと不利。一般外挿の否定ではない | 改善なし／現固定構成不支持 | A |
| [PR-03](STUDY_CARDS.md#pr-03) PR-1/4/5/6初期候補群 | 計画を実施済み成果へ数えない。後続に別名の対応候補があるか照合する | 計画・理論／静的監査まで | B |

### EVAL

| 研究 | 実際の結果・停止理由 | 区分 | 資料 |
|---|---|---|---|
| [EVAL-01](STUDY_CARDS.md#eval-01) Track A PM-1/PM-2 precision資源地図 | 近接discard補完後も限定B2点利得が残る。小差の統計winnerは未確定 | 評価基盤・限定検証 | A |
| [EVAL-02](STUDY_CARDS.md#eval-02) Track A H4–H8強対照と希少tail | 層別有限標本と残余費用の条件付き感度で採択。母平均優位保証・未観測費用0補完なし | 限定条件で改善 | B |
| [EVAL-03](STUDY_CARDS.md#eval-03) PR高次PF探索と次数逆転の既存証拠 | 既存逆転候補と政策/来歴差を記録。今回原因調査を開始しない | 技術成果あり・科学的利益未確定 | B |

### RB

| 研究 | 実際の結果・停止理由 | 区分 | 資料 |
|---|---|---|---|
| [RB-01](STUDY_CARDS.md#rb-01) B-F finite-RTE係数設計 | 保存cellの公平cross-scoreでF固有のdecision valueなし。未完了bridgeは未補完 | 改善なし／現固定構成不支持 | B |
| [RB-02](STUDY_CARDS.md#rb-02) B-M DF multirate／compact BCH | 固定adapterの三次係数は既知compact再帰と同じ。BM-1新手法検証なし | 計画・理論／静的監査まで | A |
| [RB-03](STUDY_CARDS.md#rb-03) BS block合成・SP synthesis placement | primitiveではstrict trade-off。wrapperでshot増加・不適格もあり一律有利でない | 肯定と否定が混在 | A |
| [RB-04](STUDY_CARDS.md#rb-04) R0/R1 finite-mean再配分 | 限定native Pareto上の有効点とtrade-offを保存。一般sampler最適・全PR利益は未確立 | 限定条件で改善 | A |
| [RB-05](STUDY_CARDS.md#rb-05) RA-D0 LP・G1–G5固定辞書 | 指定ideal/digital classで同辞書優位主線を閉じる。全乱択法のno-goではない | 改善なし／現固定構成不支持 | A |
| [RB-06](STUDY_CARDS.md#rb-06) G6 Return Aggregation／General Full | 方法論と限定理論を保持。RAは広い構成、General Fullは生成方式、Closed P5は内部特殊化 | 技術成果あり・科学的利益未確定 | A |
| [RB-07](STUDY_CARDS.md#rb-07) G7 conditional native economics | m5登録3対照への条件付きT利得、m3 closed対照には不利。provider/prep/古典費用が残る | 限定条件で改善 | A |
| [RB-08](STUDY_CARDS.md#rb-08) G8 on-demand synthesis／provider budget | on-demand接続は成立。一般termination・大L scaling・物理provider性能は未保証 | 技術成果あり・科学的利益未確定 | A |
| [RB-09](STUDY_CARDS.md#rb-09) G9 Closed P5 matched-native | 登録direct比較でT-primaryの有効点、CTSにはCX利点。全座標優越・任意IS最適ではない | 限定条件で改善 | A |
| [RB-10](STUDY_CARDS.md#rb-10) G10 degree-matched native比較 | within-degreeの条件付きtrade-offを取得。次degree/分子/全PRへ一般化しない | 限定条件で改善 | A |
| [RB-11](STUDY_CARDS.md#rb-11) DF希薄三次補正・generic whole | 局所支持・平均構成は保持。固定加法planの3条件でwhole下界が対照上界より高く、1条件未確定 | 肯定と否定が混在 | A |
| [RB-12](STUDY_CARDS.md#rb-12) Gaussian積生成 | 候補固有の追加手順が未定義で閉鎖。性能で敗れた実験結果ではない | 計画・理論／静的監査まで | A |
| [RB-13](STUDY_CARDS.md#rb-13) D1–D3 finite-target・境界相殺・保証価格 | 同targetの資源交換とboundary効果を保存。全方式に共通最適化、十分forecastであり新生成利益の証明でない | 肯定と否定が混在 | A |
| [RB-14](STUDY_CARDS.md#rb-14) Parent allocation／S4 KKT | 限定配分成果・訂正KKTを保持。commonはL1最小でありchild配分はL1改善自由度でない | 肯定と否定が混在 | A |
| [RB-15](STUDY_CARDS.md#rb-15) M1-S shared ± labels | 適用域付き理論と局所signal/整数witnessを保持。strong新primitive・whole-cost・多roundRPE未確立 | 技術成果あり・科学的利益未確定 | B |
| [RB-16](STUDY_CARDS.md#rb-16) G6 finite-bit／B3 live API・古典費用 | live接続と限定workload観測。全N・Closed P5/CTS同条件古典費用・新量子優位は未確立 | 技術成果あり・科学的利益未確定 | B |
| [RB-17](STUDY_CARDS.md#rb-17) C1 G6–DF移植・H4 logical pilot | 3方式探索を完走した記録、logical差は小さくzero削減未観測。strict native/N/全wrapperは未取得 | 技術成果あり・科学的利益未確定 | B |
| [RB-18](STUDY_CARDS.md#rb-18) interstep correlated randomization | marginal保存はcoherent mean保存でない。単純相関やISは既知対応、独立candidate未成立 | 計画・理論／静的監査まで | A |

### QC

| 研究 | 実際の結果・停止理由 | 区分 | 資料 |
|---|---|---|---|
| [QC-01](STUDY_CARDS.md#qc-01) C1 macrocycle／符号相殺 | 通常contextで事後oracleを含め登録優位なし。bad contextは適格不足で未判定 | 改善なし／現固定構成不支持 | A |
| [QC-02](STUDY_CARDS.md#qc-02) C2 branch付きPF外挿 | 式は条件付き成立、独立した安いbranch/remainder取得規則未特定 | 計画・理論／静的監査まで | A |
| [QC-03](STUDY_CARDS.md#qc-03) C3 self-energy／active space | 条件付きroot構造は整理、安い具体access未特定 | 計画・理論／静的監査まで | A |
| [QC-04](STUDY_CARDS.md#qc-04) C4 cost-aware proposal／whole path | q∝\|a\|/sqrt(g)等の既知法対応。moment×平均価格だけで全費用優位を示さない | 計画・理論／静的監査まで | A |
| [QC-05](STUDY_CARDS.md#qc-05) 一体再配分・exact-ground PF応答 | 生成法はREDUCES_TO_KNOWN、別scopeのfinite-step基準も4条件未達 | 改善なし／現固定構成不支持 | B |
| [QC-06](STUDY_CARDS.md#qc-06) H-E1・M-a/M-b PF小改変 | 既知処理への還元や生成規則不足。未実装を未検討扱いしない | 計画・理論／静的監査まで | B |
| [QC-07](STUDY_CARDS.md#qc-07) SOS-PF生成・K0/K0.5 | 生成探索の終了と保証理論の非採択を分離する | 計画・理論／静的監査まで | B |
| [QC-08](STUDY_CARDS.md#qc-08) A2 RF0/RF1/EW2有限step相対保証 | 理論・checkerは保持。固定DF pilotでは強いcentered対照の十分stepが大きい | 改善なし／現固定構成不支持 | A |
| [QC-09](STUDY_CARDS.md#qc-09) DF covariance縮約 | 同情報の既知対照が同じ演算列を選べ、独立した情報削減・計算量差は確認されない。 | 計画・理論／静的監査まで | B |
| [QC-10](STUDY_CARDS.md#qc-10) H1 先導固有値補正 | 既存式との重複とfinite-stepへの未接続。高body表示から一般RDM下界を主張しない | 計画・理論／静的監査まで | B |
| [QC-11](STUDY_CARDS.md#qc-11) H2 mixed correlation／filter | ratio恒等式だけでは独立情報取得法にならず、現時点で有利な生成は未確定 | 計画・理論／静的監査まで | B |
| [QC-12](STUDY_CARDS.md#qc-12) H3 frame／interaction picture | sampleされた和の指数・controlを無料にせず、独立した生成差を特定できていない | 計画・理論／静的監査まで | B |
| [QC-13](STUDY_CARDS.md#qc-13) H4 copies／phase reference | uncontrolled evolutionの定数位相不変性を確認。全測定方式の不能定理ではない | 計画・理論／静的監査まで | B |
| [QC-14](STUDY_CARDS.md#qc-14) exact fermion encoding／controlled構成 | 既知手順を超える具体的block生成則は未提示。adapter不足と算法不足を分ける | 計画・理論／静的監査まで | B |

### EXT

| 研究 | 実際の結果・停止理由 | 区分 | 資料 |
|---|---|---|---|
| [EXT-01](STUDY_CARDS.md#ext-01) P1* bus/CAR固定word | 固定wordは低次で不成立、後続紙上障害が固定solver結果を包含。一般bus不可能性ではない | 改善なし／現固定構成不支持 | A |
| [EXT-02](STUDY_CARDS.md#ext-02) J1 joint-loop・echo構成 | 成立したものは既知生成、独自候補は三次で不成立。追加探索保留 | 計画・理論／静的監査まで | A |
| [EXT-03](STUDY_CARDS.md#ext-03) Q1 integrable level insertion | 現候補の非一様性がchannel閉包を壊す。一般可解法の否定ではない | 計画・理論／静的監査まで | A |
| [EXT-04](STUDY_CARDS.md#ext-04) 集団遷移・pair collision・R1-P | 既知生成への還元や具体規則不足。量子優位の数値比較はない | 計画・理論／静的監査まで | B |
| [EXT-05](STUDY_CARDS.md#ext-05) G coherent adaptive subspace | 強い既知wrapperが候補評価・情報law・更新を再現でき、独立差未確認 | 計画・理論／静的監査まで | A |
| [EXT-06](STUDY_CARDS.md#ext-06) 固定条件付き測定 | 弱い分割対照との差はあるが既知公表生成へ具体還元。全測定下界ではない | 計画・理論／静的監査まで | A |
| [EXT-07](STUDY_CARDS.md#ext-07) B-01 unknown-gap／energy情報 | 目標依存と取得費を同時に閉じる独立規則未構成 | 計画・理論／静的監査まで | A |
| [EXT-08](STUDY_CARDS.md#ext-08) T1 logical accreditation・独立原理P1–P8探索 | 候補名を新規性と数えず、既知還元・構成欠如・不成立を分類 | 計画・理論／静的監査まで | B |

## 再利用と次判断

まず再利用できるのは固定入力・ordered law・phase-preserving control・same-target会計・強対照・compiler/provenanceの基盤である。既存数値の倍率を一般PRへ移すことは再利用に含めない。

[改善候補](PR_IMPROVEMENT_OPPORTUNITIES.md)は既存成果と重複を付記した概念提案。新規性・性能・正式優先順位は未確定である。資料回収と未実施検証を区別してユーザー/GPTへ判断材料を渡し、公開照合後mandatory STOPとする。
