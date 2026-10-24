# 既存データ・コード・研究履歴の再利用索引

参照commit/path/SHAと関数位置は [source_catalog.json](source_catalog.json)。今回の読み取りはGit固定資料、保存schema、コードAST/本文と既知の文献に限定した。旧runnerは起動していない。

## データと役割

|固定資料|利用できる情報|本研究での役割・不足|
|---|---|---|
|eebdbfcb… 主比較60 endpoint・30比較・15方向、同一q8比較|H2/H4/H6、Full/PR、S2/Y4、5精度、C/N/G・bias/u/headroom・cohort|仮説形成用。保存ratioは既存証拠であって新特徴量ではない。複数epsilonは同じsignal/価格の再会計|
|d2536c4f… C/N/G22分解|軸別費用・shot・交差項、normalizationとheadroom関係|事後会計の再利用。恒等式の成立を事前予測の成功にしない|
|364970df… 同一q入力・候補|H4/H6のFull/PR q4/q8の登録条件|比較構造とavailability。既存の異なるqのpairと分ける|
|f75a7c25… dataset.json.gz・candidate_index.csv|588候補・2940保存精度endpoint、FULL/B0/B0I、S2/標準Y4/m5_best、q1/2/4/8|PF構成選択の開発域。選択・探索済み。PRを含むdatasetではない。全候補compile済みでもない|
|同上 systemsのscientific_identity_saved|Hamiltonian/state/snapshot、rank-policy・rank、compiler/native、原signal locator|条件の結合鍵。ground-state vectorやreference値は事前特徴量に渡さない|
|09e71910… 正式採用、4d117041… 原独立確認|H4/H6 epsilon=.001、S2/q8と標準Y4/q4、別cohort費用標本|第1研究の採用結果をread-only引用。仮説形成後の本研究の未使用testにはしない。保存shot/reference政策は条件付き|
|80b7a35c… qモデル採用、85433c48… 選択比較|H6/H8旧scope、保存モデル/実測混在価格から直接価格への置換|選択敏感性の先行内部資産。新共通PF588へ無条件転用しない。純モデル誤差・未使用予測試験ではない|
|8a2fbe94… 文献差分・501b4232… 旧文献索引|先行高次PR、次数選択、cost-aware sampling|既存調査を再利用し、今回は選択時の情報と取得費用へ焦点を限定|

PF588のmetadata内訳はH2 FULL12/B0 36/B0I36、H4 FULL12/B0 108/B0I108、H6 FULL12/B0 132/B0I132。これは登録schemaの件数確認であり、性能の再集計ではない。H2 rank3、H4 rank10、H6 rank19、tol-only DF1e-8。m5_bestと標準Y4は別公式。旧H4 rank12・旧q16等を合流させない。

PRのK4、outer32は法とallocationの指定であり、費用標本数nではない。探索、旧S2独立確認、新標準Y4独立確認をpoolしない。実行識別子・候補ID・cohort・軸対・物理条件・compiler/cost scope・reference/allowance政策を結合鍵に含める。

## 関数・SOURCEと再利用方法

|実在する関数・所在|確認した入出力/処理|再利用範囲・制約|
|---|---|---|
|h246_order_cost_analysis_backend.decompose @8a2fbe94…|S2/Y4の保存N・paired Cからexact費用分解|将来の事後説明のみ。Permitと別実行承認が必要。予測モデルではない|
|同 exact_row / explanation|保存G整合、normalization/bias/u/headroomの会計整合|将来の評価用。説明用labelを特徴量へ渡さない|
|scripts/h2_common_production_backend.py Backend.account @8a2fbe94…|bias・allowance・log_BからN、paired軸費用からG、結果へ書込|epsilon=.0005固定、H2固有。C0で呼出不可。P1は保存endpointを読み、新会計を避ける|
|u_aware_shots @c31ef424…|軸別bias/u、epsilon、log_B_upper、alpha→三状態・integer/log shot|新しい再会計が必要な後続のみ。証明は生成しない。現P1予算は呼出0|
|ax2a_preparation.axis_headroom @同上|epsilon_axis-bias-u。ELIGIBLE/INELIGIBLE/UNDETERMINED|適格性の定義。欠測・境界を不適格へ勝手に変換しない|
|src/trotterlib/h46_m5_plan.py weights / schedule @f75a7c25…|公式係数、signed_time=T*c/q、piece順、RTE配分|構造metadata用。allocationはfrozen_allocation経由で既存coreの同一関数を抽出。import経路を封印時に確認|
|scripts/h46_new2_core.py allocation @同上|絶対係数に比例したquota、整数切上げ配分、各piece最低1、総32|同じlawの将来構造proxyへ流用。結果を見て配分最適化しない|
|同 matrix_only_tail_coefficients|DF diagonalization・exact_df_diagonal_coefficients→scalar、lambda_R|回路なしでも新しいprimitive数値処理。C0では非実行。lambda未保存ならTier1取得計算として別承認|
|同 compare_prefix|deterministic spectra/state、tail spectrum、lambda、公式/q→signal/log_B/error|状態依存の高価な処理。名前やmatrix-onlyだけでcheap認定しない。C0非実行|
|src/trotterlib/rte_compiled_cost.py transpile_and_measure_cost / estimate_*|回路transpile、event/occurrence費用推定|Tier2取得。今回非実行。取得回数・paired law・full wrapper費用を後続契約で計上|
|pf_selection_cost_comparison.weighted / choose @85433c48…|保存整数shotと軸価格→exact G、有限候補の最小値・ties|P1の評価側算術を流用可能。70scope専用authorizeは流用不可。旧SOURCE/許可は不使用|
|pf_q_model_diagnostics.axis_errors / diagnose @80b7a35c…|保存予測/直接価格の誤差と内外挿区分|proxy校正・破綻診断の参考。qモデルは既存anchor compileとfitを取得費用に含める。新条件の保証なし|
|旧文書checker git / digest / check @501b4232…|固定Git本文読取・hash・文書check収集|今回の技術検査へAST抽出で再利用。科学runnerをimportしない|

関連テストはtest_h246_order_cost_analysis.py、test_h46_m5_plan.py（f75a7c25…）、test_pf_selection_cost_comparison.py（85433c48…）、u-aware提供側のテストをSOURCE索引に列挙した。旧テストの合格を本研究の科学的妥当性へ読み替えない。C0は人工の情報境界・契約・schema検査だけを実行する。

## 重複・否定的履歴

1. [R3停止記録](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8a2fbe9461d9cc262326fa30f9778b295adeb3f8/docs/research/r3_prior_art_and_minimal_contract.md)：ordered DF prefixの安価な選択・棄却・regret/評価数検証まで既提案。quantum固有bound/構造差分がなく停止。新C0の名前で再開しない。
2. 歴史的PR-2 M1-AのSELECTION_LIMITED：proxy frontierが選抜capに収まらず、compile前で結論不能だった履歴。action proxyをcompiled winnerと呼べない。旧authorizationやheld-outへのアクセスを流用しない。
3. Track B C1-1 R1（2937e6b6…）はReturnKernel/aggregation系の入力・basis・exact model接続に関する系列であり、本研究の構成選択法ではない。共有できるのは明示的な関数/入力形式の知見で、結果・grant・claimは独立させる。

不足は、PRの可変partition域、独立したcheap bias proxy、proxy誤差保証、候補別の取得実時間の比較可能記録、真の未使用条件である。これらを埋める計算は本C0では開始しない。
