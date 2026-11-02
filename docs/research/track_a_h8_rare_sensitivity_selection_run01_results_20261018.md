# H8：希少層の2変数逆転境界・保存確認集合の選択効果（run01）

2026-10-18 JST。**両Stageを別々のgrant・one-shot・outputで各一回実行し、正常終了後mandatory STOPした。** SOURCE `3c0c9546701fd953de3b299be561344375c1a78b`。実行前後のSOURCE9 file・入力44 file・Python/runtime/config/grant照合PASS、外側親processのnative waitも両方exit0。追加PR sampling・native build/compile・signal・shot/bias再計算・fit・旧費用再会計は0。科学実行の再試行0、次stage自動実行0。

[準備と数値契約](track_a_h8_rare_sensitivity_selection_execution_readiness_20261018.md)、[結果bundle](../../artifacts/resource_applicability/track_a_h8_saved_analysis_run01/2026-10-18/README.md)、[summary](../../artifacts/resource_applicability/track_a_h8_saved_analysis_run01/2026-10-18/summary_v1.json)。準備資料のUNAPPROVED/INVALID案は履歴として保持し、利用者の「実際に実行するところまで行ってよい」とその後のStage別具体条件を、新しい独立grantへ結合した。有効grant原本・消費済registryは専用private領域に保持し、公開物は消費済契約・digest receiptのみ。

## 条件・固定入力

Linear H8、隣接1.00 Å、STO-3G、保存tol-only DF rank28、16 spin modes・sector4900、T=.8、prefix7/S2/r4、P=0、state preparationを除くshot-weighted full Hadamard-wrapper RZ。RARE対象K4/q4（R16/δ=.2）とK4/q8（R32/δ=.1）。SELECTIONは元確認union6（q2/4/8×K2/4）、各n32、192 trajectory。新しい量子軌道は生成しない。

元map `681ec23c…`、6軸反映後の対照正本 `96e69a01…`、s2 pilot `b21e90b8…`、準備正本 `0b68a2a2…` を再利用した。[入力44 file](../../references/h8_saved_analysis_inputs_v1.json)はcommit/path/blob/bytes/SHA256で固定。追加の保存shotはD0 `3d910ee1…` と非相殺 `23d46e48…` を6軸再会計の入力manifestへ結合した。元SOURCE・compiler/profile・seed・event pairは既存監査を継承。compiler SHA `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`、profile `622ce4a8391002558b877240b48d831d9bb0a72aeabba2e4d73bfec64b9c9f12`。今回はcompilerを起動せず、保存実測receiptとして使った。private原軌道・NPZ/QPYは不要。元selection receiptの原bytesは未取得で、公開hash・SOURCE順序・coverageによる証拠であり、その原fileの独立照合済みとはしない。

## RARE_SENSITIVITY

両bias（USUAL_SIGNED_SAVED / EMPIRICAL_NONCANCELLATION）、5精度、倍率1/10、固定PR2候補の計40条件。**32条件で境界を評価し、8条件は保存上不適格として境界nullのまま保持。** 未観測multi/s4平均は全てnull。層費用の仮定値、厳密上界、母平均保証は導入していない。

`A μ2_multi + B μ4 = C`、`A=π2 w`、`B=π4`、`C=G_det−π0 μhat0−π2(1−w) μhat2_single`。

wは保存count vectorのN2≥2の正のcellをDecimal80で加算し、single係数は宣言式どおり1−w。保存πやlawを再fit/再正規化せず、保存丸め残差を結果に記録した。μhat0は確認n32全次数0の保存平均、μhat2_singleは元s2条件付きn8が結果的に全N2=1だった保存実測の平均である。設計時からsingle限定だった実験とはしない。両軸の費用は別々の保存shotで重み付けし、未知層の両軸同値は仮定しない。

強化184候補の保存最良費用・費用来歴を固定し、対照の再最適化は0。q4ではA≈0.000570523456/B≈0.0000129936306、q8ではA≈0.000149727188/B≈0.00000162948630。以下は倍率1の代表条件のC（RZ単位）。全40行・両倍率・各候補の保存winner一致flagはJSON/CSVへ保存した。

| bias | PR固定候補 | ε | C（RZ） |
|---|---|---|---:|
| USUAL_SIGNED_SAVED | q4 | 0.01 | 3.030482729E+10 |
| USUAL_SIGNED_SAVED | q4 | 0.005 | 1.362263947E+11 |
| USUAL_SIGNED_SAVED | q8 | 0.001 | 4.577679064E+12 |
| USUAL_SIGNED_SAVED | q8 | 0.0005 | -4.786518421E+13 |
| EMPIRICAL_NONCANCELLATION | q4 | 0.01 | 3.232293702E+10 |
| EMPIRICAL_NONCANCELLATION | q4 | 0.005 | 1.606940172E+11 |
| EMPIRICAL_NONCANCELLATION | q8 | 0.001 | 9.992466076E+12 |
| EMPIRICAL_NONCANCELLATION | q8 | 0.0005 | -3.201581925E+13 |

非負費用領域では、C>0なら`A μ2_multi+B μ4<C`の側でplug-in PRが低費用、等号で同費用、上側で逆転する。C<0なら保存標本平均を使ったbaselineが既に対照を超え、非負の未知費用を足して低費用にはならない。数値境界が得られた32条件のうち16はC>0、16はC<0。負の切片を物理的な負の回路費用と解釈しない。

出力の座標軸切片は2変数境界の幾何学的情報であり、「もう一方の未観測費用を0と推定した値」ではない。未知費用を指定していないため単一の条件付き閾値はnull/statusで保持した。s0/singleの点推定と一部model対照を条件にした感度であり、厳密費用上界・PR優位性の証明ではない。q4/q8は固定候補であり、ε=.05の保存PR winner q2/K2への結論を代用しない。

## SELECTION_SAVED

元4精度 ε=.05/.01/.005/.001、USUAL_SIGNED_SAVED、倍率1だけを実行。ε=.0005・非相殺・倍率10は未実行。候補別の32 trajectoryをpairedにshot-weightし、20適格candidate-endpointの平均を保存exact費用と完全一致させた。不適格maskは固定し、値に都合よく候補を除外していない。各candidateのSD・SEはtrajectory費用から算出した記述量であり母平均保証ではない。

bootstrap2000回。候補別empirical distributionの積を用いる**明示した探索的仮定**の下、各candidateでtrajectoryを再抽選、同じ両軸を一緒に扱い、4精度で同じindexを共有。候補間replica indexのpairingは仮定せず、共分散の実測0や数学的独立性を認定しない。候補/phase別stream設計とseed非衝突は別のmetadata証拠として扱った。

2.5–97.5%は経験分位範囲で、family_alphaはnull。厳密な95%信頼区間・同時coverage・selection-adjusted intervalは出していない。all-support上界と妥当なmarginal intervalがないため、Hoeffding/Bonferroniの厳密認定もしない。

点差は「事前固定候補の確認平均−確認後点最小」（RZ）。分位範囲は両candidateのIDを固定した差の再抽選診断。最小値を取り直す診断とは区別して保存した。

| ε | 探索時固定 | 確認後点最良 | 点差 RZ | ID固定差の経験分位範囲 | 再抽選での点最良件数 |
|---|---|---|---:|---|---|
| 0.05 | q2/K2 | q2/K2 | 0 | [0, 0] | q2/K2: 1054/2000, q2/K4: 946/2000 |
| 0.01 | q4/K2 | q4/K4 | 198922136 | [-461337984, 884671610] | q4/K2: 560/2000, q4/K4: 1440/2000 |
| 0.005 | q4/K2 | q4/K4 | 969520146 | [-2.25003028e+09, 4.31335888e+09] | q4/K2: 560/2000, q4/K4: 1440/2000 |
| 0.001 | q8/K2 | q8/K4 | 9.34204901e+10 | [-2.89724111e+10, 2.16572432e+11] | q8/K2: 130/2000, q8/K4: 1870/2000 |

ε=.01/.005/.001では確認後K4が点最良だったが、ID固定のK2−K4差の経験分位範囲はいずれも0をまたぐ。これは母平均優劣の検定ではない。ε=.05も近いq2/K2・K4の間で再抽選の選択が変わった。winner件数はempirical resampling内の頻度であり、母平均最良の確率ではない。ε=.01と.005の件数が同じでも、4精度は独立実験ではなく共有trajectoryの結果として読む。

確認後の点最小、選択候補の母平均、union6の母平均最小、選択手続の母集団性能は別estimand。後3者の認定値はnullのまま。empirical law内のselection optimism診断も別labelで出し、真の母集団optimismとはしない。探索n8や条件付きs2 n8をn32へ混合せず、未観測s4/multi費用も補完しなかった。bootstrapは欠測tailを回収しない。

## 実行管理・検証・保全

135人工テストPASS（継承73＋追加62）。数式・負の閾値・zero係数・不適格・欠測、paired重み・endpoint共有、事前/確認後選択、retroactive ε拒否、grant/SOURCE/input/mask drift、one-shot/namespace衝突、first STOP、両stage人工e2eを検査した。内蔵_randomの封印をPython実行file hashへ結合する技術修正は科学実行前に完了。これは著者自身の検査で独立レビューではない。

実行本体wallはRARE 0.9034秒、SELECTION 1.7499秒（外側native waitはそれぞれ約1.03/1.87秒）。候補別resamplingの軸合計を一回だけ作り全endpointで再利用し、重複算術を削減した。この規模ではprocess起動・結合の方が大きくなるためbootstrapのworker追加は行わず、CPU/RAMの任意上限も設けていない。

既存SOURCE・結果・grant・ledger・marker・STOP、使用済みcallは不変、返却/reset0。独立clone/branchだけを書き込み、他worktreeを凍結しなかった。上位PSI・一時競合によるSTOP設計なし。既存準備cloneはHEAD `0b68a2a2…`・cleanのまま。新規試験用venvだけへpytestを導入し、共有package/cache/OS/cgroup/他processには変更0。各結果にfresh/final gate・terminal・native wait・消費済契約digestを保存。

branch `track-a-h8-saved-analysis-20261018`。公開時の結果commit/remote SHAはpublication receiptと最終チャットで照合する。認証不能時は認証設定を変えず正確な手動pushコマンドを返す。未公開をremote確認済みとはしない。

**次の担当：ユーザー・GPT**が今回の条件付き境界と探索的選択診断から、主張を保存有限標本benchmarkに限定するか、未知層測定・新独立holdout・解析拡張が必要かを判断する。**Codex**は本成果の保存・公開確認後mandatory STOP。新sampling/compile/fit・解析拡張・追加レビューへ自動移行しない。
