# 既存証拠と重複判定：M1からC-Sへの限定準備

2026-11-08 JST／SERVER1。既存preflightを再利用し、指定資料と後続の固定結果・SOURCEだけを静的確認した。新科学計算・保存値の数値再解析・旧run再開は0。既存PASSの全面再試験をしていない。

## 最初に確認した資料

- 正式共通指示2026-10-25 §8.1/8.2/8.4/10/10.1/10.3、元workspaceと独立worktreeのAGENTS、PROJECT_MAP、研究概要全文、VALIDATION_STATUS/manifestの関連記録。
- [M1 preflight aeb9d184][preflight]のREPORT、O1_EXISTING_METHODS、O1_IMPROVEMENT_MECHANISMS、O1_MINIMUM_VALIDATION_PLAN。新たな文献調査や同じpreflightの作り直しは行わない。
- [3環境統合35714770][integration]のIMPROVEMENT_VERIFICATION_LEDGER、IMPROVEMENT_GAP_MATRIX、DUPLICATION_AND_REUSE_MAP、REMAINING_VALIDATION_MAP。関連行だけを読解し、全研究の再監査はしない。
- F2 SOURCE9468c708のprototype/algebra/circuits/gates/f2_resources/f2_statistics/pricingと関連test、[F2結果dc7378ab][f2]。F1修正版・PS48/Parent/研究Cの限定負結果と停止は統合台帳の固定chainから保持した。
- [SCV統合62c450a6][scv]、[O3計画66553d94][cs]。ユーザーの今回指示に従いFragment群・Signed Supportの追加shotは行わず、Product一致も保留する。

## 重複判定表

「ALREADY_VERIFIED」は**記載された元条件・保証範囲の中だけ**で使う。未資格部分までPASSへ広げない。IDは今回の作業単位で、実験数・test件数ではない。

| ID | 問い・機能 | 区分 | 既存証拠・今回の対応 |
|---|---|---|---|
| D01 | S/P代数・Hamiltonian再構成 | ALREADY_VERIFIED | F1登録correctness。make_fragment_alternatives/canonicalize_arm/make_tail、exact JWを再実装・再実行しない |
| D02 | controlled phase・actual adjoint・X/Y wrapper | ALREADY_VERIFIED | 修正F1/F2の元scope。位相修正や基本readoutテストをやり直さない |
| D03 | 4arm full-wrapper単回価格 | ALREADY_VERIFIED | F2の固定合成fixtureでは7指標でPP<SP<PS<SS。N=null、weighted_costs=nullを維持 |
| D04 | global Pauli collect・既存selector/DP | ALREADY_VERIFIED | canonicalize_armのP同word集約、select_dictionary/exhaustive_four、literal期待価格は実装済み。native総費用最適と読み替えない |
| D05 | 同物理H/T_eff/V_squareのmatched接続 | ALREADY_VERIFIED | 後続P3 Stage1の全Fock256・34検査PASS。旧静的資料の「adapterなし」だけを現在の状態としない |
| D06 | matched辞書別nominal mean/B | ALREADY_VERIFIED | Stage2の57整合項目・限定信号。finite bias成分UNRESOLVEDを保持 |
| D07 | matched finite-bit CDF/角度差 | ALREADY_VERIFIED | 後続有限bit診断はengineering範囲。native/compiled一様保証は未取得 |
| D08 | matched保存価格＋条件付きN/G | ALREADY_VERIFIED | Stage3 run04とStage4は完了。全5精度で保存点会計のPauli RZが小さい、ただし仮定付き。再価格/再会計しない |
| D09 | P3/F2とA1 N1のH4条件の互換性 | NOT_COMPARABLE | P3 tol-only rank10・T_effのみ保持、F2人工4mode、A1 legacyDF12/LD3は別条件。同じH4名から移植しない |
| D10 | native bridge・認証N・母平均価格tailの未資格 | REUSABLE_WITH_DELTA | 既存評価APIはあるが、新しい数値保証が必要。新算法の操作差ではないため今回は再計算・資格付与しない |
| D11 | singleton/PS48、混在辞書の正式停止family | FROZEN_OR_STOPPED | MIX-CLOSEの正式一時終了を保持。同poolの混在探索・singleton差替えを別名再開しない |
| D12 | SCV/研究C native fit/C1 | FROZEN_OR_STOPPED | 既存結果とNO-GO/凍結、今回のSCV一区切り・Product保留を保持 |
| D13 | canonical paired operator core・Parent位相/weight | ALREADY_VERIFIED | 既存CanonicalGenerator、native_plan/operator_target、event_statisticsを部品として再利用。Parent配分探索を再開しない |
| D14 | SP channel PAI＝coherent residualか | NOT_COMPARABLE | channel一致はHadamard coherent mean一致ではない。旧PAI/価格をC-Sの正解にしない |
| D15 | coarse＋finite residualを元の位置へ接続 | REUSABLE_WITH_DELTA | 新しいliteral splice adapterだけ追加。既存exact Angle/位相付きevent参照と人工toyで確認 |
| D16 | C-Sの実H4 strict合成後、N×full価格の追加利益 | UNTESTED_MECHANISM | 位置・normalization/rangeとnative価格の交換が未検証。今回科学実行なし |
| D17 | C-N/C-P/Multilevel | UNTESTED_MECHANISM | 次順位のまま。C-Sの技術差分が成立したため同時実装しない |

**D01–D08とD13の9作業単位を、原結果/既存実装の再利用で重複実施しなかった。** これは「9実験を省略」「過去test何件を新たにPASS」とする件数ではない。新規実行はC-Sの追加接口に対する人工test9件だけ。

## M1で見落としやすい後続証拠

既存preflightが参照するH4静的readinessは、その時点では新adapterなしと記録する。一方、今回の基点62c450a6に収録された`matched_partition_stage1/2/3/4.py`、`matched_partition_finite_bit.py`とその結果が既に存在する。旧資料を理由に同じadapterを新設しない。

| 固定結果 | 元科学SOURCE | 元条件と限界 |
|---|---|---|
| [Stage1 5708e370][p31] | da13170e593e6d845b014e3f110691be10e29d43 | linear H4/1Å/STO-3G/tol-only1e-8 rank10/full256、案A T_eff/V_square。時間発展なし、34項目数値整合 |
| [Stage2 8bcf5851][p32] | 4bfbbaad1f8178077489bf06b085377256d6fb80 | 同保存入力、T=.8/S2/q8/r4/K4、M_B/M_C。nominal信号/normalization、bias未解像 |
| [Stage3 e44e87a3][p33] | 1cfce5e961e3d063741a1e0656052b10c7b27392 | 同条件、独立32 trajectory/方式、full128価格。order4未観測、母平均tail未資格 |
| [Stage4保存REPORT][p34] | 2f0d31f75ab1ecacc73d9ee5c34ddfc997396102 | 同結果の5精度条件付きN/G。finite-bit gap=0/compiled rho=0等の仮定。新しい優越保証ではない |
| [finite-bit保存REPORT][p3bit] | d22ca967955a664260a3a9857df243b3f151610f | 同encoded modelの6信号診断。CDF/角度差候補、native bridge/compiled保証は未取得 |

F2は4mode/2粒子/occupied(0,1)/LD1/残差2density-square、T=1/5/q1/r2/K0。Qiskit1.3.0/opt1/seed17、prep込みfull-wrapperの単回価格であり、A1 no-prep価格との直接比較もしない。既報PPの低単価と未資格Nは両立する。

F2の7moduleを基点とblob照合し、6件は同一、circuits.pyだけ後続`reference_support`hook差があることを確認した。このhookはsingleton系列の来歴で、F2実行SOURCEと混ぜず、今回使用/変更/試験しない。元9468c708の証拠を当該版に結び続ける。

## 技術的な分岐

今回確認したM1のcollect・混在・selector・basis共有・matched接続は既存手法に還元され、追加の操作的工夫は特定できなかった。**M1全般の無効性や科学的NO-GOではなく、今回の実装課題として新コードを足さないという判定**。旧資格・価格tailを埋める科学計算も本依頼では行わない。

ユーザー指定の分岐に従いC-Sの未接続部分へ移る。新sampler/optimizer、Parent/S4の再実験、channel PAIの改名ではなく、既存coherent finite eventを元のslotへ挿入する接口を追加した。科学的追加利益・独立新規性はUNKNOWN。詳しくは[差分](ALGORITHM_IMPROVEMENT_DELTA.md)と[人工結果](IMPLEMENTATION_AND_ARTIFICIAL_VALIDATION.md)。

[preflight]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/aeb9d1847f9d500f2389db12b384bac6087117a9/docs/research/o1_hamiltonian_partition_preflight_20261106
[integration]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/3571477093441ac4852166fe829f8ccc5db5ec4f/docs/research/pr_algorithm_improvement_three_environment_integration_20261105
[f2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dc7378ab34015287215bcf4ce52147e46d94ecd9/docs/research/pr_algorithm_improvement_phase2d_f2_resources_rerun_20261024/EXECUTION_REPORT.md
[scv]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/62c450a6c22fc6cf228bd2fbaabda3c36a7cf169/docs/research/o2_scv_two_pilot_synthesis_20261107/REPORT.md
[cs]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/66553d948aa9df684987972789c6f3ee8a741312/docs/research/track_c_o3_pilot_plans_20261106/C_S_PILOT_PLAN.md
[p31]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5708e37080738c4ebf74590d799a3741246483b2/artifacts/df_pauli_matched_partition_validation/2026-11-02/stage1/REPORT.md
[p32]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8bcf5851f99f980dc08c5885ace54874b554c815/artifacts/df_pauli_matched_partition_validation/2026-11-02/stage2/run01/REPORT.md
[p33]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e44e87a306d05c8549f1a044b7fd056c2e1055a3/artifacts/df_pauli_matched_partition_validation/2026-11-02/stage3/run04/REPORT.md
[p34]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/62c450a6c22fc6cf228bd2fbaabda3c36a7cf169/artifacts/df_pauli_matched_partition_validation/2026-11-02/stage4/run01/REPORT.md
[p3bit]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/62c450a6c22fc6cf228bd2fbaabda3c36a7cf169/artifacts/df_pauli_matched_partition_validation/2026-11-02/finite_bit/run01/REPORT.md
