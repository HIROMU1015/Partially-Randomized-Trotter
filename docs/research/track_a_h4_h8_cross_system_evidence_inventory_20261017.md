# Track A H4〜H8：横断的証拠整理・比較条件監査

対象結果の固定日：2026-10-17。監査資料作成：2026-10-18 JST。監査コードSOURCE：`ad9d6a91dbf82f204f1f37298a1785a91847c5eb`（新しい科学SOURCEではない）。

2026-10-17。19の研究・評価系列を区別し、固定公開資料の値・状態222件を転記した。これは19個の独立実験を意味しない。親系列・標本・実測の再利用は明記し、重複加算しない。新しい科学計算、保存値の費用再会計、順位付け、統計推定は行っていない。

**現状の主要な結論は対照集合と会計条件に依存する。** H6のepsilon=.05ではp7単価の実測を用いた限定比較でもPRが高費用。H8の主要6軸反映後も、通常biasではepsilon=.05/.0005のPR削減率が負、中間3精度は正。全候補実測・PR母平均winner・厳密bias保証には拡張しない。

## 対象と証拠の読み方

対象は保存されたDF Hamiltonianとstateに対する `z(T)=<psi|exp(-i H_DF T)|psi>`。費用は原則P=0、解析的な各軸shot予算を掛けた測定込みfull Hadamard wrapperのRZ。実機でshotを実行した費用、energy精度Ha、QPE/RPE/FTのend-to-end費用とは区別する。

H4 geometryはlinear H4/STO-3G/legacy DF12/6距離/T=.8/S2/ordinary/structured Gaussian。H4固定anchorは1Å/native symmetric-directional/full S2/S4/固定16。H5/H7は1Å/STO-3G/ROHF共通real MO/DF-sector doublet/DF14・23/固定q1、H6/H8は1Å/STO-3G/tol-only1e-8/cutoff0/signed generation order/weighted Hermitian projection/DF19・28の別モデルである。H4/H6/H8の原軌道政策とcharge原欄は直接join未確認のためNOT_VERIFIEDを残した。

- [全条件CSV](../../artifacts/resource_applicability/track_a_h4_h8_cross_system_evidence/2026-10-17/comparison_conditions_v1.csv) / [系列inventory JSON](../../artifacts/resource_applicability/track_a_h4_h8_cross_system_evidence/2026-10-17/study_inventory_v1.json)：必須40項目とbias・normalization・軸配分・shot規則・稀少次数・SE・保証を別欄に収録。
- [evidence ledger](../../artifacts/resource_applicability/track_a_h4_h8_cross_system_evidence/2026-10-17/evidence_ledger_v1.csv)：各値を元fileの行またはCSV欄へ結合。分母・集合・SOURCE・result・SHA256・表示精度を保存。
- [source catalog](../../artifacts/resource_applicability/track_a_h4_h8_cross_system_evidence/2026-10-17/source_catalog_v1.json)：32ファイルのexact commit/path/Git blob/bytes/SHA256/GitHub固定URL。
- [比較可能性監査](../../artifacts/resource_applicability/track_a_h4_h8_cross_system_evidence/2026-10-17/comparability_audit_v1.md) / [CSV](../../artifacts/resource_applicability/track_a_h4_h8_cross_system_evidence/2026-10-17/comparability_audit_v1.csv)：必須10比較と補足2比較。
- [未解決13項目・既存レビュー履歴](../../artifacts/resource_applicability/track_a_h4_h8_cross_system_evidence/2026-10-17/open_evidence_items_v1.md) / [JSON](../../artifacts/resource_applicability/track_a_h4_h8_cross_system_evidence/2026-10-17/open_evidence_items_v1.json)。

## 完成済み系列の一覧

|study_id|完成した対象と保存結果|費用・選択の限定|根拠|
|---|---|---|---|
|H4_GEOMETRY|6距離x218候補、74784 logical wrappers、1812保存点でB2点最小|旧domain、S2のみ、同じ32標本で選抜/評価、明示u減算なし|E01;E03|
|H4_GEOMETRY_B0I|120構成240軸（234新+6再利用）、1812点の点最小B0I397/B21415|追加前と分母が変化。397/1415は勝率・独立実験数でない。driver native exit原欄null|E02;E03|
|H4_NATIVE_ANCHOR|固定16、532 compile、PR8x32。4精度PR/fullPF比 .506631/.471627/.453431/.510507|PRp3とfull12 S2/S4の固定panel。B0なし。H4geometryと別実装|E04;E05;E24;E03|
|H5_NEUTRAL_PILOT|DF14/sector100/p4/q1、20group/40compile/16PR trajectory完了|epsilon.05: PRK4=215276664、fullS2=388818084 RZ。8標本・固定6構成|E06;E27;E32|
|H6_CONFIRMATION|探索92/592group後、union8x独立32/512wrapper完了|fullPF比54.00/60.04/62.79/61.30%減はfull-rank対照のみ。B0/B0I追加後と区別|E07|
|H6_NATIVE_B0I|B0I12構成の適格費用充足。18取得済み軸、6不適格軸未測定|old32対照のPR削減29.35/49.69/53.27/56.14/50.37%。matched_B0_deltaはnull|E08;E07|
|H6_STRENGTHENED136|通常会計32→96→136、S4登録4軸PASS|5精度の136対照PR削減 −13.04/30.89/34.97/31.17/10.03%。選択対照は全て当初model推定|E09;E11|
|H6_P7_MEASURED_PAIR|p7q1S2各14492RZ・保存shot10216/14356、det356097424対PR403423200|epsilon.05限定pairのPR削減−13.29012029%。全136再順位や他4精度更新ではない|E10;E09|
|H6_FIXED_DELTA_T|固定δ=.2、T=.4/.8/1.6・q2/4/8、固定5構成完了|epsilon.001: 比.452914/1.756787/不適格、.0005:6.869224/不適格/不適格。強化136最適化と混ぜない|E12|
|H6_RARE_SAVED_AUDIT|保存832 PR trajectoryの次数/費用、選択320subset監査|新標本なし。未観測層平均null、全support compiled上界未証明|E13|
|H6_CONDITIONAL_S2|p5q8r2K2の条件付きs2 8本/16軸取得、局所層別費用+0.00734086%|元n32とpoolしない。K4・H8へ移送不可。全8本は実際にはorder2一個|E14|
|H7_NEUTRAL_PILOT|DF23/sector1225/p6/q1、20group/40compile/16PR trajectory完了|epsilon.05のPRK2/K4はfullS2対31.58/29.39%減。各8本、固定6。epsilon.005以下全不適格|E15|
|H8_ORIGINAL_MAP|探索92/592group・union6x独立32完了、DF28/sector4900|元4精度fullPF比54.29/64.37/64.38/65.76%減。B0比・B0I追加後と別分母|E16|
|H8_D0|保存値5精度/6指標/P感度へ拡張、epsilon.0005確認PR/旧det=.761667659|B0/B1のみの旧20対照、nativeB0Iは当時未追加。n8/n32は別|E17;E16|
|H8_PHASE_C_SIGNED|通常bias旧32/強化184、1900行/20比較完了|主要6軸置換前。184winner全model。現行正本はH8_CURRENT_SIGNED|E18;E25;E26|
|H8_NONCANCELLATION_COST|経験的非相殺別1900行/20比較完了|非相殺は成分絶対値和、厳密保証でない。現行正本はH8_CURRENT_NONCANCELLATION|E19;E25;E26|
|H8_KEY_SIX_AXES|B0Ip15q4S4=668636、B0Ip11q1S2=42268、B0p15q4S2=223940 RZ/軸、両軸PASS|登録6軸、誤差.216-.827%<2%。旧1+新5の別SOURCE、全184保証でない|E20|
|H8_CURRENT_SIGNED|6軸実測反映後。下表の5精度値が現行正本|通常winner単価は4精度実測/.001model。集合全体はmodel残存|E21;E22;E23;E20|
|H8_CURRENT_NONCANCELLATION|6軸反映後の非相殺。通常と合わせ3800行/40比較、winner変更0|非相殺winner単価は.001のみ主要実測、残4精度model。PR費用/SE/保存shot/旧32不変|E21;E22;E23;E20|

## 現行H8の5精度（保存された表示値）

Linear H8/1Å/STO-3G/DF28/T=.8/q1,2,4,8/184対照/PR union6各n32/P0/alpha_axis=.025。allowance倍率1。削減率は保存された `100*(1-G_PR/G_det)` で、正はPR低費用。以下はE21の丸め値の転記で、E22のexact有理数文字列を台帳に保存した。倍率10は同じ資料に別条件で保存しており、一部exact値は異なる。

|epsilon_sig|通常signed PR削減率|通常winner単価|経験的非相殺PR削減率|非相殺winner単価|
|---|---:|---|---:|---|
|.05|−8.0230%|B0I p11q1 S2：実測|16.6243%|B0I p15q1 S2：model|
|.01|33.2373%|B0 p15q4 S2：実測|34.6849%|B0I p15q4 S2：model|
|.005|31.4549%|B0 p15q4 S2：実測|35.1253%|B0I p15q4 S2：model|
|.001|22.9613%|B0 p15q8 S2：model|39.4150%|B0I p15q4 S4：実測|
|.0005|−50.8125%|B0I p15q4 S4：実測|−29.0910%|B0I p16q4 S4：model|

実測winnerの行でも、最小値を選んだ184集合にはmodel単価が残る。したがって「184候補をすべて実測して最良を確認」とは書かない。PRの単価はn32有限標本平均で、paired SEは記述統計である。非相殺と通常biasの違いは新しい近似誤差certificateではない。

B1のB0は3点fitと独立p21/q8 holdoutの両軸誤差0%であり、B0+Iの0%は8点training anchorへの残差である。後者を独立holdoutと呼ばない（E26）。B2は登録S4の4軸だけの局所検証で、旧B0取得run02と途中B2試行のSTOPは元のまま保持し、正常終了軸を別の受入契約で結合している（E25）。

## 証拠区分と主張の境界

|区分|支持する内容|支持しない内容|
|---|---|---|
|DIRECT_NATIVE_COMPILED|登録した実wrapper/軸の保存整数RZ、個別SOURCEでの取得|全候補の誤差保証、実機shots、population平均|
|SAVED_COMPILED_MEASUREMENT|以前の実測単価と保存shotの既存結果、原来歴の保持|今回再compileしたとの表示、欠測profileの推測補完|
|MODEL_ESTIMATED|固定式の保存推定値。外挿・検証点の範囲を明示|実測値、全grid2%保証、H6係数のH8移送|
|FINITE_SAMPLE_MEAN|指定候補・seed/層・本数内の保存平均、記述的paired SE|tailを含む母平均上界、選択補正CI、正式winner|
|MIXED_MEASURED_AND_MODEL|実測優先とmodel補完を用いた宣言domain内の保存点比較|全候補実測最小、一般PF方式に対する優位|
|NOT_AVAILABLE|未観測・不適格・未保証をnull/unknownとして保持|費用0、存在しないことの証明|
|人工検査/native probe PASS|対象の実装条件・指定probeでの数値的一致|厳密数学certificate、独立レビュー/immutable CI|

費用の種類と科学的保証は別軸である。H4 B0Iのepsilon=.05表は「B0IのB2に対する削減率」で、PRの決定論集合に対する削減率とは分母・向きが違う。元の符号を反転・再計算せず台帳に別metricとして保持した。

## 比較できる範囲と残る証拠

同じ保存候補・shot・PR標本を固定したH8の6単価置換、H6のp7限定pair置換は、意図した単価変更に限定して比較条件が揃う。H4 geometry内、H5/H7の固定pilot、H6/H8の強化対照は一致点と相違点を列挙した記述比較に留める。H4 geometryと固定anchor、通常と非相殺、固定T最適化と固定δT-panelは異なる契約である。

不足は全格子のmodel誤差、PR有限標本tail・選択効果、経験的u、旧profile/bit同値、H6旧B0I分解、H4イベント記録/親native exit、一部物理入力原欄、実state preparation/end-to-end費用。元資料が無いと断定せず、未join・未検証と区別した。新しい実験の採否は今回決めていない。

2026-10-13 H6 pilot後、10-14 H4 B0I、10-15 H8/D0の既存科学レビューはE30/E31/E29として固定し、当時の採否・SOURCEを引継ぎ表へ記録した。レビューの算術や科学判断をやり直していない。

## 技術検証・来歴・非干渉

資料作成の起点は `96e69a01d1fae7a402aac2c4efa570a59f8cc7a0`。旧v0.4比較表E03を再利用し、後続H8の完成状態を新しい別表へ追記した。元原稿・レビュー原本・SOURCE・結果・grant・ledger・marker・STOPは変更していない。

固定結果参照19 commitに加えてH5準備参照をGitHub originから取得し、ローカルGit objectと照合した。新branchの公開成否/remote SHAはpublication receiptと最終チャットに別記録する。固定commitを取得できることと新成果をpush済みであることは区別する。

[metadata専用検査結果](../../artifacts/resource_applicability/track_a_h4_h8_cross_system_evidence/2026-10-17/technical_validation_v1.json)では、SOURCE commit存在、固定blob/bytes/SHA、必須schema、系列/親関係、重複キー、222転記値の元行・欄一致、比較/未解決の参照、相対リンクを確認する。既存科学runner・accounting checker・付録算術・Qiskit・NumPyは実行しない。主担当による技術検査であり独立レビューではない。

作業は共有worktreeと異なる独立clone/branch/indexで実施した。共有Gitはimmutable object参照のみ。他チャットのH8希少次数pilotは未完了の別作業として扱い、専用SOURCE/input/output/grant/ledger/carry/marker/lock/STOP/CPU配置へアクセス・変更していない。広域再帰検索・他processへのsignal・共有環境変更・無関係worktreeの凍結は行っていない。固定公開資料と軽量metadataはキャッシュし、重複取得とI/Oを減らした。

新signal/SCF/DF/state/reference/native build/compile/sampling/fit/shot再計算/費用再会計/ランキング/平均SE再計算/統計解析は全0。今回の19系列・222行はmetadata/転記件数であり、新科学call数ではない。

## 次の担当とmandatory STOP

- ユーザー：本表の条件差と不足を踏まえ、後続の科学判断・追加検証の必要性を選ぶ。新しい主要claim/新規性は本監査では採択していない。
- 別チャットのCodex：その既存契約の範囲でH8条件付きpilotを担当。本監査は進行や完了を判定しない。
- Codex（本作業）：資料・技術検証をcommitし、安全に可能ならpush/remote SHA・公開blobを確認してmandatory STOP。新科学stageへの自動移行なし。

## 固定証拠索引

|ID|固定資料|result/reference commit|file SHA256|
|---|---|---|---|
|E01|[docs/research/track_a_h4_geometry_results_20261013.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7c4483378047b57f514c47dad98905e3f2a9a756/docs/research/track_a_h4_geometry_results_20261013.md)|`7c4483378047b57f514c47dad98905e3f2a9a756`|`a6034da2108ee6be4730989c1d77b02619c4c603f3f5813d8664ae993527186c`|
|E02|[docs/research/track_a_h4_b0i_map_results_20261014.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/98cdb35cd08469714f20d6cc3b982d9c80908f46/docs/research/track_a_h4_b0i_map_results_20261014.md)|`98cdb35cd08469714f20d6cc3b982d9c80908f46`|`f1c4d6bbbb4acb3cad5caeafa3bcf4b4c21f670204e9121425f39c0abb9616cf`|
|E03|[docs/research/track_a_resource_comparison_conditions_20261014_v0_4.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/47dc17a83e2fc9216e22b59317969952ef79028a/docs/research/track_a_resource_comparison_conditions_20261014_v0_4.md)|`47dc17a83e2fc9216e22b59317969952ef79028a`|`10c54b70a66e8ebb0fd07b1a96255d22caec86d1b41824e092100d4db8f424c4`|
|E04|[artifacts/resource_applicability/track_a_h4_common_anchor_result_v1/2026-10-13/README.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0f96a2e45ba58bd916508e1b1f32e6ff1ff65f41/artifacts/resource_applicability/track_a_h4_common_anchor_result_v1/2026-10-13/README.md)|`0f96a2e45ba58bd916508e1b1f32e6ff1ff65f41`|`db2b84d505362e8ec10a4ccb3f66151cb865c7d5d4795487d6aa1f20943e9416`|
|E05|[artifacts/resource_applicability/track_a_h4_common_anchor_result_v1/2026-10-13/result_summary.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0f96a2e45ba58bd916508e1b1f32e6ff1ff65f41/artifacts/resource_applicability/track_a_h4_common_anchor_result_v1/2026-10-13/result_summary.json)|`0f96a2e45ba58bd916508e1b1f32e6ff1ff65f41`|`d9d086cd7c5d0ef548d7a61dc82a80b0c989bfa79514881a18fb8bfe16fb10ca`|
|E06|[docs/research/track_a_h5_neutral_open_shell_pilot_result_20261016.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bde58ba23f1fecdf77ee950795cb80c199c0f86b/docs/research/track_a_h5_neutral_open_shell_pilot_result_20261016.md)|`bde58ba23f1fecdf77ee950795cb80c199c0f86b`|`413669a605545b9246a6bbcb23744eff5bfe51fa1564527384f7a80e52674f08`|
|E07|[docs/research/track_a_h6_independent_confirmation_result_v1.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/321cd8c3066194fe98741cbebbcdba1515d1deaa/docs/research/track_a_h6_independent_confirmation_result_v1.md)|`321cd8c3066194fe98741cbebbcdba1515d1deaa`|`6f66db8e5c3fcf14b50e2480d2d1404c58834215a3bf631362de0dabc6b400a7`|
|E08|[docs/research/track_a_h6_native_b0i_full_family_results_20261015.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/34efc914b2f485ec694c7d6016025d5c76829be6/docs/research/track_a_h6_native_b0i_full_family_results_20261015.md)|`34efc914b2f485ec694c7d6016025d5c76829be6`|`8cf60b6db0b9c11fab92865be08c5c5de35347fcb6e6ebb48aed030dd829384e`|
|E09|[docs/research/track_a_v1_reaccount_results_20261016.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f89c40af73289af9f4c2b70ed0b3966d95d47b06/docs/research/track_a_v1_reaccount_results_20261016.md)|`f89c40af73289af9f4c2b70ed0b3966d95d47b06`|`901cd26e196c028e44a9b721f4d81d908a30a936658cf872505eabfdce62198b`|
|E10|[docs/research/track_a_v1_p7_measured_reaccount_results_20261016.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f89c40af73289af9f4c2b70ed0b3966d95d47b06/docs/research/track_a_v1_p7_measured_reaccount_results_20261016.md)|`f89c40af73289af9f4c2b70ed0b3966d95d47b06`|`a8ad7b336ea9ab2c26e15350f31b3afc83e2de279593d2748922098b74e8ca62`|
|E11|[docs/research/track_a_v1_phase_b_run01_results_20261016.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f89c40af73289af9f4c2b70ed0b3966d95d47b06/docs/research/track_a_v1_phase_b_run01_results_20261016.md)|`f89c40af73289af9f4c2b70ed0b3966d95d47b06`|`3f539d2bd44bd8a801d218a644762917073b64bfca22cf39666e3b16a001e0f3`|
|E12|[docs/research/track_a_h6_fixed_delta_t_dependence_results_20261015.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/043f58ac9336d662a3181caff508526568f1e57a/docs/research/track_a_h6_fixed_delta_t_dependence_results_20261015.md)|`043f58ac9336d662a3181caff508526568f1e57a`|`5f2fee20797f85e521a3246bc9081f3c0626f91c37afa6992c39286e25b6cb25`|
|E13|[docs/research/track_a_h6_rare_order_cost_reliability_v1.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/eee1343f6e99aa7661892ddfbe1067e974e2b74a/docs/research/track_a_h6_rare_order_cost_reliability_v1.md)|`eee1343f6e99aa7661892ddfbe1067e974e2b74a`|`7905c1c98e5e1f9da54197e3b363a40950f0a484602742ff75c256c2070e09ff`|
|E14|[docs/research/track_a_h6_conditional_s2_pilot_result_v1.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0f96a2e45ba58bd916508e1b1f32e6ff1ff65f41/docs/research/track_a_h6_conditional_s2_pilot_result_v1.md)|`0f96a2e45ba58bd916508e1b1f32e6ff1ff65f41`|`9dd3c52d6c2b7d95b3cd93e0a3bbe99988512824c887af63dad918dade2a6b8b`|
|E15|[docs/research/track_a_h7_neutral_open_shell_pilot_result_20261016.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/162c256ea39b4ab3dadde65a94b3eec73cea4720/docs/research/track_a_h7_neutral_open_shell_pilot_result_20261016.md)|`162c256ea39b4ab3dadde65a94b3eec73cea4720`|`b374980d5e511484bd4c68e78f0bdad5e0ab2704b871bda509cd4d1d18229575`|
|E16|[docs/research/track_a_h8_resource_map_result_v1.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/681ec23c93fdd1090d325908f4b155a79115afdf/docs/research/track_a_h8_resource_map_result_v1.md)|`681ec23c93fdd1090d325908f4b155a79115afdf`|`ed44b5a3751181db050b5b6a3439590c2046f05602663d8c6aa3a247b6d0a706`|
|E17|[docs/research/track_a_h8_d0_saved_value_result_20261015.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3d910ee1de4597b8895e8506500553dbad5a5085/docs/research/track_a_h8_d0_saved_value_result_20261015.md)|`3d910ee1de4597b8895e8506500553dbad5a5085`|`b9b5501c35e83199fb9fbb6b62841c2ab88c04ff7d74e0ef7ee29707a2c169eb`|
|E18|[docs/research/track_a_h8_phase_c_run01_results_20261017.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6dc5bf0de01a0aa23cafe0ff022c0cd21fbf1086/docs/research/track_a_h8_phase_c_run01_results_20261017.md)|`6dc5bf0de01a0aa23cafe0ff022c0cd21fbf1086`|`a32a183f52b19bdc2ca588201b529756ac34685102c88bf80871057c5789fe9e`|
|E19|[docs/research/track_a_h8_noncancellation_cost_run01_results_20261017.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/23d46e489af054735d9d328aad0342ecdbddf305/docs/research/track_a_h8_noncancellation_cost_run01_results_20261017.md)|`23d46e489af054735d9d328aad0342ecdbddf305`|`48c68721006a5c81ee36d1230eec63219f3ea1a9869ea6a13cb94eb573a5a2d5`|
|E20|[docs/research/track_a_h8_key_compile_completed_results_20261017.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ce6cd8ead5c0311ec72165c04869e7df4050b75b/docs/research/track_a_h8_key_compile_completed_results_20261017.md)|`ce6cd8ead5c0311ec72165c04869e7df4050b75b`|`0fb7a36fd72cf809883e57042a650c6e8e746a7a08622ce5b110835c5e8d5732`|
|E21|[docs/research/track_a_h8_six_axis_reaccount_run01_results_20261017.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/96e69a01d1fae7a402aac2c4efa570a59f8cc7a0/docs/research/track_a_h8_six_axis_reaccount_run01_results_20261017.md)|`96e69a01d1fae7a402aac2c4efa570a59f8cc7a0`|`e7b8ecb25f48b72077d08dad6c02970b6aead5898ca3fdced7c6e9e13115b6ad`|
|E22|[artifacts/resource_applicability/track_a_h8_six_axis_reaccount_run01/2026-10-17/comparison_summary_exact.csv](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/96e69a01d1fae7a402aac2c4efa570a59f8cc7a0/artifacts/resource_applicability/track_a_h8_six_axis_reaccount_run01/2026-10-17/comparison_summary_exact.csv)|`96e69a01d1fae7a402aac2c4efa570a59f8cc7a0`|`ad117ee5f3afdb1c0b06882fdad5c37c7bd486c72683912f1eec230721c10a19`|
|E23|[artifacts/resource_applicability/track_a_h8_six_axis_reaccount_run01/2026-10-17/final_gate.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/96e69a01d1fae7a402aac2c4efa570a59f8cc7a0/artifacts/resource_applicability/track_a_h8_six_axis_reaccount_run01/2026-10-17/final_gate.json)|`96e69a01d1fae7a402aac2c4efa570a59f8cc7a0`|`d0359d0ddac21149e8eabdbf9f6f72a4f519da2e7354a70076af000d02992351`|
|E24|[artifacts/resource_applicability/track_a_h4_common_anchor_result_v1/2026-10-13/signal_metadata.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0f96a2e45ba58bd916508e1b1f32e6ff1ff65f41/artifacts/resource_applicability/track_a_h4_common_anchor_result_v1/2026-10-13/signal_metadata.json)|`0f96a2e45ba58bd916508e1b1f32e6ff1ff65f41`|`81adabc515701b28d0d55ea5c31a6de82747ddc37789ccce1dbd359add017045`|
|E25|[docs/research/track_a_h8_b2_completed_results_20261017.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fef2db7a79886ab682b652da7266b463c98c7d3a/docs/research/track_a_h8_b2_completed_results_20261017.md)|`fef2db7a79886ab682b652da7266b463c98c7d3a`|`9b23467f5e62ff31549200ee471d179f96d446d6d1d94790a1fbf7cf418a04b9`|
|E26|[docs/research/track_a_h8_phase_b1_run01_results_20261017.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/83a3c4a6688e66576f85471bf41a45e29cbfb705/docs/research/track_a_h8_phase_b1_run01_results_20261017.md)|`83a3c4a6688e66576f85471bf41a45e29cbfb705`|`8d05d3fc529c982e1b0e86012f056dbc173beca09c0e46c31a417b6f5d4f9076`|
|E27|[artifacts/resource_applicability/track_a_h5_neutral_open_shell_pilot_v1/2026-10-16/input_state_signal_v1.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bde58ba23f1fecdf77ee950795cb80c199c0f86b/artifacts/resource_applicability/track_a_h5_neutral_open_shell_pilot_v1/2026-10-16/input_state_signal_v1.json)|`bde58ba23f1fecdf77ee950795cb80c199c0f86b`|`3f68e5576a955ea85e1af5af5451a0118e8a64908268147b0ebe191ad0fd17c2`|
|E28|[artifacts/resource_applicability/track_a_h4_geometry_results/2026-10-13/summary.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7c4483378047b57f514c47dad98905e3f2a9a756/artifacts/resource_applicability/track_a_h4_geometry_results/2026-10-13/summary.json)|`7c4483378047b57f514c47dad98905e3f2a9a756`|`72c8ac70741f5399b1577e79a3164f3a90222b793d68bd6e811d29dcbeb45510`|
|E29|[docs/research/track_a_h8_d0_integrated_scientific_review_20261015.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3d910ee1de4597b8895e8506500553dbad5a5085/docs/research/track_a_h8_d0_integrated_scientific_review_20261015.md)|`3d910ee1de4597b8895e8506500553dbad5a5085`|`add57fd02f1df00bca495b3bf1a4185b8ca10462a0b4b9ceec61d49d0ed563c4`|
|E30|[docs/research/track_a_h6_s2_pilot_followup_scientific_review_20261013.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0f96a2e45ba58bd916508e1b1f32e6ff1ff65f41/docs/research/track_a_h6_s2_pilot_followup_scientific_review_20261013.md)|`0f96a2e45ba58bd916508e1b1f32e6ff1ff65f41`|`84211430f16f046ac2e2cf9cc5437f20dbf95950615a87f612ebb14a27b02719`|
|E31|[docs/research/track_a_h4_b0i_v04_scientific_review_20261014.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/34efc914b2f485ec694c7d6016025d5c76829be6/docs/research/track_a_h4_b0i_v04_scientific_review_20261014.md)|`34efc914b2f485ec694c7d6016025d5c76829be6`|`1c4fb797dbaed0580ab4bf188f8ef4adb4fa042a5ec42731e8a56ff5bd129703`|
|E32|[docs/research/track_a_h5_neutral_open_shell_readiness_20261016.md](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/803a24ef7a01d8c7394dde88357c74168795bb34/docs/research/track_a_h5_neutral_open_shell_readiness_20261016.md)|`803a24ef7a01d8c7394dde88357c74168795bb34`|`c9043dc407c630910fb7c5edae15b07f8efb32fc4984f46175b18227951b4b1d`|
