# H4/H6 長時間PF・PR：次数・刻み幅・分割点・r/K比較

2026-10-25開始、2026-10-26取得終了の専用研究。科学計算・会計・独立標本確認・図表を保存。終了時の技術確認・公開状態は成果物内のcompletion_provenance.jsonとGit履歴を参照。
**最良はcompiled費用を取得した適格候補内の点最良。全登録域の大域最適値・母平均優位・厳密誤差保証ではない。**

## 条件

Linear H4/H6、隣接1.0 Å、STO-3G、tol-only DF threshold1e-8、rank10/19。保存Hamiltonian/state/scalar/DF fragment順を再利用し、SCF・DF・状態の再生成0。
H6は保存済みweighted-Hermitian completionを含むHamiltonianをそのまま使用し、completion前の別Hamiltonianとの同一性は主張しない。入力bindingは[scientific_provenance.json](artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/scientific_provenance.json)。
T=.8/1.2/1.6/2.0。基本qはT=.8/1.6で1/2/4/8/16、T=1.2では6、T=2では10も追加。δ=T/q。
S2、標準Yoshida4、m5_bestの既存signed係数を使用。Full、B0、B0+I、PRを同じfull-H referenceに照合。
H4 prefix=1/2/3/5/7/9、Full10。H6 prefix=3/4/5/7/11/15、Full19。PR R_outerはS2=4/8/16/32/64、Y4=8/16/32/64、m5=16/32/64。K=2/4/6（paired-cutoff、分子多項式の最高次数K+1）。
R_outerは1 outerの総short-step数。既存ABS_TIME_PER_OUTER整数配分・各piece最低1 stepを維持。全trajectoryのevent数はq R_outer。
ε=.05/.01/.005/.001/.0005、α_axis=.025、P=0。共通PF政策の経験的u floor1e-9、倍率1、log_B margin、u/headroom≤.01を使用。旧保存allowance/marginを減らさない。
主指標G=N_cos C_cos+N_sin C_sin。nativeはsymmetric-directional、scalar一回、actual adjoint、測定付きfull Hadamard wrapper。Qiskit1.3.0、rz/sx/x/cx、opt1、seed17。
analytic shot数であり、量子測定を実施した回数ではない。単一の有限時間signal費用で、全energy/QPE/RPE費用ではない。

## 取得範囲・重複回避

基本11,220構成は全signal評価完了。旧signal294構成・shot1,440条件・compiled費用候補203件を再利用候補として索引化し、選択された費用65候補は再build/compileしなかった。新アダプタに対する必要な旧T=.8代表照合4件のみ別記録（最大差約2.7e-13）。
q16でS2が探索境界に達したため、T=1.6/2.0のH4/H6・S2に限りq32を追加。追加412構成の全signalを評価。基本共通格子とS2限定追加域を区別し、Y4/m5のq32は未評価。
合計11,632構成・58,160精度endpoint。基本の費用取得対象481候補のうち65候補は保存価格を再利用し、新規416候補を取得。基本対象外でも互換な旧価格を残し、旧価格203＋新規416＋q32追加22＋固定δ不足1＝642候補を会計した。適格性・shot数と明示した作業proxyで取得候補を絞り、Full PF、S2 r4/K2、同じq/prefixのR32/K4対照を残した。proxyを実測費用や下界に使わない。
受入済み新規単価試行は基本2,624＋q32追加94＋固定δ不足9＋独立確認1,295＝4,022件。科学試行STOP/失敗0。固定δ=.2の全40比較候補は既存値を優先し、未取得の適格PR1候補だけ9標本を追加。
新規PR探索は9 trajectory/候補（粗い分散計画の名目基準）。独立確認は探索と別seed/cohort、旧有効確認標本は再利用。新確認nは探索分散×4、名目相対MCSE2%、最低33で取得前に固定し、確認後top-up・探索とのpoolingなし。
K0は任意候補として未追加。主候補K2/4/6を完了対象とした。

## ε=.0005の費用取得範囲内の結果

|系|T|方式|最良公式|q / δ|prefix|R_outer / K|G（1 SE）|
|---|---:|---|---|---|---:|---|---:|
|H4|0.8|PF|2nd|8 / 0.1|7|—|8.4146e+12|
|H4|0.8|PR|2nd|8 / 0.1|1|4 / 2|5.1495e+12 ± 6.86e+10|
|H4|1.2|PF|4th(m5_best)|1 / 1.2|7|—|8.74132e+12|
|H4|1.2|PR|4th(m5_best)|1 / 1.2|3|32 / 6|5.54978e+12 ± 4.59e+10|
|H4|1.6|PF|4th(m5_best)|1 / 1.6|7|—|8.66047e+12|
|H4|1.6|PR|4th(m5_best)|1 / 1.6|3|32 / 6|6.1726e+12 ± 4.25e+10|
|H4|2.0|PF|4th(m5_best)|1 / 2|7|—|9.58334e+12|
|H4|2.0|PR|4th(m5_best)|1 / 2|3|32 / 6|7.46061e+12 ± 4.63e+10|
|H6|0.8|PF|4th(m5_best)|1 / 0.8|11|—|3.67856e+13|
|H6|0.8|PR|4th(m5_best)|1 / 0.8|5|32 / 2|2.16718e+13 ± 1.31e+11|
|H6|1.2|PF|4th(m5_best)|1 / 1.2|11|—|3.70685e+13|
|H6|1.2|PR|4th(m5_best)|1 / 1.2|5|32 / 2|2.44644e+13 ± 1.07e+11|
|H6|1.6|PF|4th(m5_best)|1 / 1.6|11|—|3.52718e+13|
|H6|1.6|PR|4th(m5_best)|1 / 1.6|7|16 / 2|2.84996e+13 ± 4.45e+10|
|H6|2.0|PF|4th(m5_best)|1 / 2|11|—|3.99801e+13|
|H6|2.0|PR|4th(m5_best)|1 / 2|7|16 / 6|3.46305e+13 ± 5.85e+10|

全40系・時間・精度セルのうち、取得範囲内でPRの点費用が最良PFより低いセルは33/40。全5精度・公式別のq/δ/prefix/r/Kと未計測数は[best_configurations.csv](artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/best_configurations.csv)。

## signal・刻み幅・費用の区別

T=2・ε=.0005ではH6のS2 PF/PR、H4のS2 PRはq32で適格候補を取得。一方m5は両系PF/PRでq1の適格候補を持つ。H4 S2 PFにはq16の適格候補がある。この最小qはsignal適格性の観測で、費用最適qとは異なる。
高次化による少ないqと、signed piece増加・normalization・必要shot・単価の影響を分離する。公式別C/N/G比は[C_N_G_decomposition.csv](artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/C_N_G_decomposition.csv)。
同一q/prefixの共通R32/K4対照は[same_q_prefix_R32_K4.csv](artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/same_q_prefix_R32_K4.csv)。欠けたcompiled値は空欄であり0ではない。r/K感度は全signal・shotと取得済みcompiled費用を区別して[r_K_sensitivity.csv](artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/r_K_sensitivity.csv)に保存。

## m5とS2の厳しい精度での比較

ε=.0005について、各公式が取得済み適格域でq・prefix・PRのr/Kを選んだ探索点費用を比較する。次の比は独立確認後の再選択値ではなく、探索で固定した点比較。

|系|T|方式|S2のq|m5のq|m5 / S2 総RZ費用|
|---|---:|---|---:|---:|---:|
|H4|0.8|PF|8|1|1.0362|
|H4|0.8|PR|8|1|1.0025|
|H4|1.2|PF|16|1|0.5833|
|H4|1.2|PR|16|1|0.5525|
|H4|1.6|PF|32|1|0.3424|
|H4|1.6|PR|32|1|0.3880|
|H4|2.0|PF|32|1|0.3439|
|H4|2.0|PR|32|1|0.3249|
|H6|0.8|PF|8|1|0.9118|
|H6|0.8|PR|16|1|0.6902|
|H6|1.2|PF|16|1|0.6015|
|H6|1.2|PR|16|1|0.4168|
|H6|1.6|PF|32|1|0.2945|
|H6|1.6|PR|32|1|0.3348|
|H6|2.0|PF|32|1|0.3868|
|H6|2.0|PR|32|1|0.2475|

大きいδを使えることだけで費用優位は決まらない。緩いε=.05では両系・全TでS2が各PF/PR取得域の点最良だった。signed piece数、normalizationによるshot増、回路単価とbiasの釣合いを区別する。

## 旧T=.8結果との接続

旧共通PFのq≤8と旧PRのR32/K4等の固定比較に対し、本研究はq16（S2一部q32）、R/K・prefixの選択域を拡張している。T=.8でも同じ旧比較条件だけの再掲ではない。旧対応候補のsignal・shot・単価は再利用し、互換な旧価格203候補を新規取得対象外も含め保持した。新しい点最良や統計標本の違いを旧結果の誤りとは解釈しない。

## 独立確認と希少event

固定した独立比較59件のうち、確認費用が揃ったものは59件。探索と同じ点方向の再現は58件。
同じcandidateの標本を複数εで会計している比較があり、比較件数は独立実験数ではない。PR内のcos/sinはpaired、異なる候補をpairingしない。95%正規近似区間とBonferroni感度を保存するが、母平均・希少event tailの保証ではない。
PF対PRの探索優位33比較は33件とも独立点方向が再現。Yoshida4対S2 PRは8/8、m5対S2 PRは17/18。例外はH4・T=1.6・ε=.005で、探索のm5優位が独立標本では逆転した（独立G: m5約5.8103e10、S2約5.7216e10、差のBonferroni区間は0を含む）。探索で勝たなかった比較の全てを独立確認したものではない。
新規群はPCG64/NumPy choiceの元sampling lawを使用し、53-bit CDFの丸め・未観測次数の確率を記録。次数2/4/6の未観測費用を0とせず、未取得の条件付き費用・tail上界はnull。ゼロ標本分散も母分散0とはしない。
新しいK6だけ、H4/H6・正負時間の強制eventを既存native builderと独立occupation演算で確認（最大vector差約2.4e-15）。これは条件付き正しさ確認で、random費用標本へ混ぜていない。K0/K2/K4の旧検証は反復していない。

## 実装・再現性・制約

既存h46_new2_coreのspectral/paired polynomial/occupation、h46_m5_planのsigned係数・整数allocation、保存input loader、h46_new2_nativeのGaussian preparation・occupation oracle、ax2b native builder、fingerprint reuse、compiler、凍結shotとpaired統計を再利用。旧科学SOURCEを編集せず、新しい時間可変adapterを追加した。
signalは保存spin sectorの不変性を既存ladder演算で検査し、H4 36/H6 400次元でbatch演算。新referenceは同じfull-Hとstate、旧T=.8（H6は1.6も）は保存値を再利用。新T2 referenceはexpm_multiplyと別経路照合。
基本signalとnativeは実空きCPU/RAMから63 workerで開始し、内部BLAS/compilerは1 thread/process。q32は最初1 worker、復旧後63 workerで残りだけ処理。独立確認は64 worker。原子的なtask directory取得で同時実行の重複を防ぐ。
10月25日22:31頃に元workerと実行セッションが消失（原因未特定）。基本2,323件＋追加15件の完成値を保全し、未完了だけ同じseed・cohortで再開。中断中だった64試行は別directoryへ保全し、PRイベントの再現一致を確認して採用した。中断時の計算作業の重複はあり得るが、完成試行の再取得や二重標本計上はない。03:55に全取得完了後、報告保存だけでROOT未定義エラーが発生し、科学出力を再計算せずstudy.ROOTに修正した。元ログとSTOP状態は保全。
新規科学条件はこの専用研究のH4/H6・T≤2、登録公式/prefix/r/K内。旧run/grant/ledger/marker/STOP、R2-CS、Track A主表・Figure2・他研究・共有環境・既存venvを変更していない。
残る制約：compiled費用の全登録域最適性、未観測eventの母平均費用上界、厳密reference/bias保証、有限PRNGの独立性証明は未確定。保存数値allowanceは経験的である。歴史的Python/runtime差をbit完全同一と主張しない。

## 成果物

- [cost_vs_time.png](artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/cost_vs_time.png) / SVG / PDF
- [selected_delta_vs_time.png](artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/selected_delta_vs_time.png) / SVG / PDF
- [selected_formula_grid.png](artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/selected_formula_grid.png) / SVG / PDF
- [fixed_delta_time_dependence.png](artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/fixed_delta_time_dependence.png) / SVG / PDF
- [candidate_precision_accounting.csv](artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/candidate_precision_accounting.csv)
- [independent_comparisons.csv](artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/independent_comparisons.csv)
- private raw event/seed・代表QPY・全trial結果：`/home/AbeHiromu/projects/h46-long-time-pf-pr/20261025/`（Git公開対象外）

次の担当：ユーザーが本結果を踏まえた科学的採択・追加検証の要否を判断する。今回のCodex実行は技術確認・commit/push後mandatory STOP。詳細科学レビュー・主表変更・別研究への拡張は未着手。
