# PR α追加検証：D01/A05訂正とB-1/B-2の未承認計画

2026-10-18 JST。D01/A05の説明修正、B-1/B-2の実装・人工検証・計画案を保存した。**実H6/H8の二項確率計算、shot・費用・順位の新規再解析はすべて0。B-1/B-2は未承認で、実入力のprojectionと来歴が未sealのため実行不可。科学的な検証完了とはしない。**

基点はα契約監査 `e62ddb59173d100c113be7a33581515d78f152ee`。既存の11件のSOURCE照合と16件の人工検査は再利用し、同じ監査を再実行しなかった。[監査報告](pr_alpha_contract_audit_20261018.md)、[今回のmetadataと保全記録](../../artifacts/resource_applicability/pr_alpha_followup_preparation/2026-10-18/)を参照。

## 固定範囲と状態

|単位|今回の成果|実データ実行|
|---|---|---|
|D01/A05|説明metadata訂正と親会計参照をcommit `e08775aa` に保存|数値変更0|
|B-1|二項tail runner・人工検査・14条件の案|未承認・入力未seal|
|B-2|凍結shot関数による感度runner・人工検査・α grid案|未承認・入力未seal|

新規runnerのSOURCEは `16b737236facee33b8cdb0d9b47bf1806ced11f0`、branchは `track-a-alpha-followup-preparation-20261018`。planの`code_sealed=true`はこの7ファイルのclosureだけを意味する。`sealed=false`、`inputs_bound=false`、無効grantを保存している。後続の資料commitでHEADが変わっても、使うSOURCE/input/runtimeだけを照合する。他worktreeの通常更新を停止条件にしない。

対象は旧H4/RPEではなく、Track A native有限時間complex signalである。保存モデルはH6/H8 linear隣接1 Å・STO-3G、tol-only 1e-8、DF rank19/28、T=.8、保存state・L_D・p/q/r/K・symmetric directional順序をそのまま使用する。εはcomplex signal精度でありenergyのHa精度ではない。費用は保存された状態準備P=0のfull Hadamard wrapper RZ。同じ候補のα以外の科学条件を変えない。

## D01/A05：完了した訂正と残る不足

[説明追補](pr_alpha_projection_explanation_20261018.md)に元SOURCEと全親会計を固定commitで接続した。[変更receipt](../../artifacts/resource_applicability/pr_alpha_followup_preparation/2026-10-18/stage_d_changes.json)と[diff](../../artifacts/resource_applicability/pr_alpha_followup_preparation/2026-10-18/stage_d_diff.patch)が修正前後を記録する。

D01はH4_GEOMETRY/H4_GEOMETRY_B0IのJSON/CSVの`normalization_policy`各1欄のみを訂正した。保存normalizationを直接用い、h=ε/√2−bias、α_axis=.025、N=ceil(2 B² log80/h²)という旧契約を説明する。nativeのsaved log_B_marginや明示u差引きを持ち込まない。元の科学SOURCE・shot・RZ・順位は変更しない。A04のevent/component履歴と既知172候補のnormalization来歴不足は未解決。

A05は通常biasの簡略projectionからPhase C、D0/B0I/Phase Aの元accounting、6軸join、入力manifestへ参照を補った。通常signed biasと経験的非相殺を分離し、EMPIRICAL/CERTIFIED、実測/model RZ、n32のSEと測定αを区別する。親会計を参照できるようにした説明上の修正であり、欠けていた証明書を作ったわけではない。A03の数値bias・allowance・normalizationの厳密保証不足も保持する。

## B-1：測定モデルと整数境界

[計算関数](../../src/trotterlib/pr_alpha_followup.py)、[runner](../../scripts/run_pr_alpha_binomial.py)、[未承認plan](../../artifacts/resource_applicability/pr_alpha_followup_preparation/2026-10-18/B1_plan_UNAPPROVED.json)。native wrapperのH/U/(sineのみSdg)/H/measureとbit0→+1の根拠は[監査の固定SOURCE接続](pr_alpha_contract_audit_20261018.md)を再利用する。入力の符号規約が一致しなければ拒否する。

各量子shotでwhole trajectoryをfresh IID抽選する理想モデルについて、X=B_corr Y、Y∈{−1,+1}、E[X]=z_approx、|X|≤B_budgetを使用する。B_corrは実際の補正倍率、B_budgetはshot式の上界である。raw×B_corrとcorrectedの不一致が宣言allowanceを超える場合、B_budget<B_corrの場合、pが範囲外の場合は補正・clipせずSTOPする。保存headroom・allowance・B_budgetのlog結合も検査する。保存Nを使用し、B-1でshotを再計算しない。

保存binary64値を`Fraction.from_float`で正確な有理数に変換し、p=(1+z_approx/B_corr)/2とする。K~Binomial(N,p)、z_hat=B_corr(2K/N−1)について、

```
a = N/2 * (1 + (z_approx-h)/B_corr)
d = N/2 * (1 + (z_approx+h)/B_corr)
|z_hat-z_approx| >= h : K <= floor(a)    または K >= ceil(d)
|z_hat-z_approx| >  h : K <= ceil(a)-1   または K >= floor(d)+1
```

既存Hoeffding失敗事象に合わせ、実行案は保守的な`>=`を固定する。関数は双方を人工検査し、境界付近を丸めて整数へ吸着させない。Nは整数かつ0<N<2^52が必要。範囲を超える保存Nはbackendへ渡さない。

二項分布のCDF/SFを数値評価する。右tailは1−CDFによる桁落ちを避け、SF(K_min−1)を使う。[SciPy公式仕様](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html)に基づき、専用環境のSciPy1.17.0/NumPy2.5.4を固定した。小Nでは有理数の全和と照合、大Nでは人工N=10^12で列挙なしの経路を検査した。**「厳密二項」は分布モデルを指し、浮動小数点のCDF値は外向き丸め付き区間証明ではない。** 空でないtailのunderflow、非finite、α近傍の数値判定は未確定とし、0やPASSへ置換しない。正確なpとbackendへ渡すbinary64 pの差も数値評価上の限界である。

軸別二項確率、別計算のHoeffding上界とlog上界、宣言α_axis=.025への数値適合を出力する。2軸は周辺確率の和によるunion boundのみ。独立積による厳密joint probabilityはnull。bias・allowanceに条件付く値であり、厳密なsignal精度保証、候補選択後保証、複数signal/RPE全round保証へ昇格しない。

## B-1：代表条件と実行上の障害

代表IDは固定結果表の既存winner欄から機械的にコピーした。新しい費用比較や候補順位の計算はしていない。両familyを低精度・中間・高精度で含め、H8の既存境界診断ε=.000625も含める。これは全候補への一般化を目的とする無作為標本ではない。

|系|ε|決定論的PF|PR|
|---|---|---|---|
|H6|.05|B0I_p7_q1_2nd|B2_p5_q1_2nd_r4_K4|
|H6|.001|B0I_p11_q2_4th|B2_p5_q8_2nd_r2_K2|
|H6|.0005|B0I_p11_q4_4th|B2_p5_q8_2nd_r2_K2|
|H8|.05|B0I_p11_q1_2nd|B2_p7_q2_2nd_r4_K2|
|H8|.001|B0_p15_q8_2nd|B2_p7_q8_2nd_r4_K4|
|H8|.000625|B0I_p15_q4_4th|B2_p7_q8_2nd_r4_K4|
|H8|.0005|B0I_p15_q4_4th|B2_p7_q8_2nd_r4_K4|

案は通常bias・allowance倍率1・α_total=.05。14条件×2軸=最大28二項軸評価、CDF/SF最大56 callとlogCDF/logSF最大56 call。shot再計算・費用再会計・仮想測定は0。欠測行は評価せず`INCONCLUSIVE_SOURCE_MISSING`として残す。矛盾やα超過は当該B-1をSTOPし、部分結果を保存する。

必要な実入力はraw/corrected、B_corr/B_budget、reference、bias成分、u/log_B_margin、保存h/N、証拠分類と元フィールド位置。現時点ではcanonical scalar projectionと各原本へのフィールド対応を固定していない。既存inventoryが指すprivate H6 signal原本SHA256 `b9f6cb3dc1a168ddc409833c78a617a7ddd05a3a41e0665e9dec47346d516dfe` と旧H8 PR signal原本 `dac86904b9a9d5d72146956ce2ed7aa3f9da52545169f5bbecb2af45c8ae6706` の現在の入力pathも未結合。原本が存在しないと断定せず、今回は広域検索やprivate原本の公開をしなかった。

したがってIDとcall上限の案は完成したが、数値headroom/Nの入力hashまで固定済みとはしない。実行承認の前に保存値をコピー・joinするmetadata作業で入力をsealする必要がある。原本不足なら欠測を明記し、signal再生成や都合のよい代入へ進まない。

## B-2：総α感度の提案

[runner](../../scripts/run_pr_alpha_sensitivity.py)、[未承認plan](../../artifacts/resource_applicability/pr_alpha_followup_preparation/2026-10-18/B2_plan_UNAPPROVED.json)。提案gridは**α_total=0.01、0.05、0.10（未承認）**、各軸へα_total/2を均等配分する。非一様α最適化は含まない。

|条件|固定する案|
|---|---|
|H6候補|決定論136＋確認union8/n32のPR8＝144 ID|
|H8候補|決定論184＋確認union6/n32のPR6＝190 ID|
|H6 ε|.05/.01/.005/.001/.0005|
|H8 ε|.05/.01/.005/.001/.000625/.0005|
|bias|USUAL_SIGNED_SAVED、EMPIRICAL_NONCANCELLATIONを分離|
|allowance倍率|1のみ。倍率10は今回の主比較案に含めない|
|行数|入力3,720条件、各3αで最大11,160会計行（基準α行3,720を含む）|

候補集合は保存結果の全IDを使い、PRに有利な候補だけを残さない。H6の一部旧B0I非相殺値などが欠ける場合もIDを保持し、全候補の最小値はUNRESOLVEDとする。H8は保存された低精度・高精度・境界の条件を含む。ε=.000625のH6新規評価は提案しない。

SHA256 `ed3f57937a231719583c2893677b7e82f934dd98bda35cd37fc0f17f83d4279f` の既存`d0_frozen_shot_rule_v1.py`を変更せず再利用する。h=ε/√2−bias−f u、B_budget=exp(log_B+f margin)の既存算術順・ceil・最小1・log-domainを保持する。各入力で最初にα_total=.05を評価し、保存α・N_total・軸別全状態・証拠分類との完全一致を要求する。不一致なら新αへ進まない。

PRは保存確認n32の同じreplica/seed/event/compilerのreal/imagを対にし、G_i=N_real C_real,i+N_imag C_imag,iを集計する。平均は有理数、SEは32個のG_iの標本SEとし、64独立標本へ水増ししない。探索n8を混ぜない。deterministicの実測/model/軸別mixedとMISSINGを区別し、欠測を0にしない。

ELIGIBLE/INELIGIBLE/UNDETERMINED/LOG_DOMAIN_ONLY/MISSINGを保持する。共通の系・ε・bias・allowance倍率・αでのみ比較し、費用は有理数で最小化、同額はID辞書順とする。PR/PF比=G_PR/G_PF、PR削減率=100(1−G_PR/G_PF)。比較は有限標本の点費用で、母平均winner・モデル不確かさ込みの科学的優位性を認定しない。

出力は各軸shot、paired平均/SE、実測/model区分、費用比・削減率、α=.05からのwinner/rank切替、ceil前の連続値と丸め差の診断を持つ。二項確率計算、sampling、compileはB-2に含まれない。入力を一度parseし、grid全体で保存signalとpaired費用を再利用する。

## 人工検証・schema・保全

新規unittest41件がPASS（native exit0）。B_corr/B_budget、符号・raw対応、整数境界、二項/Hoeffding分離、equal α、基準shot再現、min/ceil/log-domain、証拠分類、paired共分散、費用区分、旧RPE拒否、scope/budget、grant/STOPを検査した。小SEを測定αの証拠へ流用する入力を拒否した。両runnerの人工parent/child実行と再実行拒否、および未承認plan表示・無効grant拒否のCLI8検査もPASS。[人工検査記録](../../artifacts/resource_applicability/pr_alpha_followup_preparation/2026-10-18/synthetic_test_result.json)、[CLI記録](../../artifacts/resource_applicability/pr_alpha_followup_preparation/2026-10-18/cli_synthetic_checks.json)。著者である主agent自身の検査であり、独立reviewとは呼ばない。

承認後の[result schemaと判定仕様](../../artifacts/resource_applicability/pr_alpha_followup_preparation/2026-10-18/result_schema_v1.json)は、B-1の条件付き確率とB-2の点費用比較を区別する。schemaは結果の構造であり、科学入力の正しさを証明しない。

[保全監査](../../artifacts/resource_applicability/pr_alpha_followup_preparation/2026-10-18/preservation_audit.json)で既存44固定objectと今回の変更範囲を記録する。旧audit/科学SOURCE/数値結果/grant/ledger/marker/STOPは保持した。変更した旧artifactはStage Dの説明欄とREADMEのみ。過去receiptはそのcommitの履歴として残す。今回の依存追加は専用`.alpha-preparation-venv`内のみで、共有packageや認証設定は変更しない。

## 次に必要な作業と承認

1. **Codex：**既存private入力の場所と固定hash、保存scalarのフィールド対応、baseline親会計、paired費用を結合し、B-1/B-2ごとのcanonical入力をsealする。数値再解析やsignal再生成は含めない。不足は欠測として提示する。
2. **ユーザー：**入力seal後のB-1の14条件・保存h/N・最大28軸call・SOURCE/runtimeを確認し、B-1専用の一回実行を別途明示承認する。
3. **ユーザー：**B-2のα grid案、144/190候補、ε・bias・倍率1、最大11,160会計行・SOURCE/runtimeを確認し、B-2専用の一回実行を別途明示承認する。B-1承認をB-2へ流用しない。
4. **Codex：**承認されたstageだけに独立grant/output/markerを発行し、使用依存をfresh照合して一回実行し、native終了・結果を記録してmandatory STOPする。

各stageは独立して停止・保存し、自動retry・scope拡大・次stage起動を行わない。共通の使用input/SOURCE/runtimeが破損した場合は影響するstageを停止する。他stageの固有障害や無関係なworktree更新だけで正常な作業を停止しない。今回の準備作業はここでmandatory STOPとし、有効grantを発行しない。
