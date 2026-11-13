# RPE decoderの有限shot統計・全RMSE検証基盤

2026-11-13 JST。**技術実装・人工検証・保存metadata受入まで完了**。B3216d27a69ac8532d2a22e78f5f94491d202d238から、研究範囲を区別するbranch `codex/rpe-decoder-statistics-20261113`を作成した。従来のB統合branchは基点のまま保全。A/C/Dのworktree/process/cacheには介入していない。

最も重要な確認点は、**既存decoderはpriorをbranch選択に使わず、zero/tieの未解決trialにはenergyもlossも定義していない**こと。従って成功trialだけのRMSEを全RPE RMSEとは扱えない。今回decoderの規則は変更せず、count入力とloss/統計評価を薄く接続した。

## 1. 実際に確認・再利用したSOURCE

以下はB基点3216d27a6の実装を読んで確認した。原fileはすべてbyte無変更である。

| SOURCE | 実装規則・今回の扱い |
|---|---|
| `pr_rpe_rmse.py:decode_supplied_coordinates` | raw測定平均X/Yを各々[−1,1]で検査、(0,0)ならUNRESOLVED_ZERO_COORDINATES、math.atan2から既存scalar decoderへ。最良/次点branch距離がabs_tol=1e−12・rel_tol=0で近いtrialはUNRESOLVED_BRANCH_TIE。energy=None |
| `rpe_four_round_phase_validation.py:reconstruct_rpe_phase` | q=1,2,4,…の連続倍増。初回を[−π,π)へwrap、以後全q枝のrootをwrapし、前推定との`abs(angle(exp(iΔ)))`が最小のbranchをnp.argminで選ぶ。scalar本体はtieでも先頭を選ぶがA wrapperが全体を未解決へ戻す。energy=−final_phase/h |
| 同moduleの`_marginal_branch_monte_carlo/_vectorized_reconstruction` | 旧4round検証では独立binomial・count-grid平均・vector化を使用し、zero/tieをfailureへ加える。phase成功確率を測る実装で、現在の元H energy RMSE/loss contractを既に解決しているものではない。今回は再実行・複製しない |
| `rpe_hadamard_failure_validation.py` | 既存binomial/有限shot検証とClopper–Pearson helperを確認。新集計の未解決率CIに`_clopper_pearson_interval`を再利用する |
| `pr_rpe_rmse.py:evaluate/nominal_axis_shots` | paper型の経験的nominal model。finite_B_diagnostic・予算/horizon・資格を分け、十分Nを返さない。Bでshot式やnormalizationを再実装しない |
| `pr_rpe_rmse_controller/integration/evidence` | V2の全schedule/identity/law/SOURCE・資格・null会計をそのまま再利用。cost/selectやprice providerへ統計診断を十分Nとして渡さない |

関連履歴も確認した。`origin/codex/rpe-float64-decoder-preparation-20261104`の4318f0bae追補は旧H4 DF12/L_D3/δ.02/18roundの部分準備で、count-grid/integer-sumの違いとFloat64全域保証の未解決を記録する。今回のDF10/13roundへ移植・再実行せず、runtime/丸めの経験的テストを数学的保証にしないという限界を引き継ぐ。

## 2. 新実装の最小範囲

新moduleは`src/trottertracks/rpe_decoder_statistics.py`。Aのdecoder wrapperを**直接呼ぶ**。独自branch selector、RMSE物理model、汎用controllerは作っていない。原A wrapperの通常importに伴う既存trotterlib依存もそのままで、circuit build/compile関数は呼ばない。

- count/outcomes：各軸の±1又はplus countと明示整数Nを入力。任意精度整数の(2k−N)/Nを保持してbinary64に一回変換する。極端に不均衡なX/Y・巨大Nをfloatへ丸めて保存しない。NumPy binomialのint64 domain外はsimulationを拒否し、shotsを縮小しない。
- count callerの区別：新しい明示conversion ID=`EXACT_SIGNED_INTEGER_SUM_THEN_BINARY64_RATIO_V1`。旧marginal callerの`2.0*k/N-1.0`と数値演算順が異なる。N=3,k=1等でbit差を検査した。新callerの技術対応は科学的採択ではなく、次の実験でどのcallerを評価するか固定が必要。
- mean/normalization：Aのraw meanを±1確率p=(1+μ)/2へ使う。corrected mean/logB・bias・precision・state/target/law/uncertaintyはSOURCE付きで保持するが、normalizationを確率・N・価格へ二重乗算しない。mean_from_aは元physical requestとcanonical identityを検査してcopyするだけ。
- readiness：全M+1roundとX/Yを保持。欠測はNOT_EVALUATED、物理未資格はdefaultで実行しない。明示した条件付き人工/診断実行でも、unknown systematicを0へ変えず十分N/global boundを返さない。
- 統計：明示IID policyによるbinomial countのvector生成と元scalar decoderのtrial別呼出し。count列挙は小人工分布だけをcaller指定の有限scopeで実行。PR event/trajectoryの実samplingやphysical mean計算ではない。
- loss：energyと元targetの実直線上の差の二乗。branch/aliasの大誤差を円周phase差に置き換えない。未解決trialはenergy=Noneのまま全分母に残し、未解決率、解決条件付きRMSE、全試行への解決分MSE寄与を別欄に出す。
- 未解決loss：既存仕様ではUNDEFINED。未解決が観測された場合、全trial RMSEはnull。SOURCE付きBOUNDED_INTERVALを入力した場合だけlossの区間を計算し、energyを代入しない。0件観測でもpopulation failure確率を0と認定しない。
- finite-MC不確かさ：未解決率は既存Clopper–Pearson。全riskのCIは、独立反復・confidence・一様squared-loss support・未解決loss区間を明示したときだけ、Hoeffdingの両端片側boundで計算。UNDEFINEDではpopulation RMSE CIを出さない。binary64実装の条件付き統計CIで、全物理誤差や量子十分Nの証明ではない。

詳細I/Oは[HANDOFF](HANDOFF.md)。priorは入力metadataと所属診断だけに保持し、branch変更・clipping・energy shiftは加えていない。通常principal energy範囲は(−π/h,π/h]。priorが狭くても全出力supportの証明には使わない。branch errorそのもののoracle定義/資格は別問題としてnullを保持する。

## 3. 人工・解析可能な検証

**新16＋元A decoder1＋既存V2回帰106＝123件PASS**。failure/error=0。[TEST_DECODER](TEST_DECODER.json)／[log](TEST_DECODER.txt)、[TEST_REGRESSION](TEST_REGRESSION.json)／[log](TEST_REGRESSION.txt)。作者自身の技術確認であり、実H4の科学的性能や独立reviewではない。

ideal phase・有限X/Y count/outcomes・zero・branch tie・alias境界・prior端点・不均衡N・巨大整数・missing・物理未資格・identity/law/source不一致・normalization/価格の非乗算・CI前提を検査。原decoderへの同じ座標入力との出力一致も確認した。

解析例は**分子と無関係の人工M=0,h=1,target=0,rawX=rawY=0**。N_X=N_Y=2なら全count格子9点を列挙し、未解決質量1/4、解決分MSE寄与17π²/64、解決条件付きMSE17π²/48と一致。未解決lossが未定義なら全RMSE=null。loss区間[0,π²]を人工入力すると全MSEは[17π²/64,33π²/64]となる。N_X=N_Y=1の解析MSE5π²/16ともbinomial Monte Carloが合致し、loss/精度から算出した有限反復数でCIを検査した。これらの列挙はSciPy binary64 PMFで、厳密enclosureではない。

既存dependencyのinvalid escape sequence DeprecationWarningは検証中に出たが、無関係な旧SOURCEは修正していない。8次PF検証method・旧S0 producer suiteは実行対象から除外した。実科学値をdecoder/statisticsへ渡さないことをmetadata CLIのtestで確認した。

## 4. 実H4はmetadata準備だけ

[metadata01/PREPARATION.json](metadata01/PREPARATION.json)：正本configuration `587d20f7ed906d25841615dc9746674af872675b895bd438c06d3e74abf0e911`、元H/state/snapshot/prior/decoder/M12/h.1/共通exactκとpolicyをcopy。A3e035c0dのnominal m0=64・m1=60・m6=38だけを配分metadataとして転記し、他10roundはnull。

**実meanを統計入力へ輸入していない**。準備表の全13mean欄はNOT_SUPPLIED_TO_STATISTICSであり、Aに保存meanが全く存在しないという意味ではない。targetの数値も新評価せずnull。十分N/global RMSE bound/empirical H4 RMSE/Gはnull。実H4 decoder Monte Carlo、保存signal/N再解析、新作用/reference/normalizer、十分N認定、実global bound、compile/fit/Gは0。

## 5. D最新coverage

既存`rpe_d_evidence_intake`へ小さな`accept_receipt_coverage`を追加し、D公開e7dca5f7301c8633b17a59b6a7161331303aef2dのCOVERAGE_V2/RECEIPTSを読取受入れした。receiptの再実行やRZ再計数はしない。

[D_RECEIPTS_ACCEPTANCE](metadata01/D_RECEIPTS_ACCEPTANCE.json)：136 Full PF候補、X/Y各価格EXACT73、CONDITIONAL0、欠測63。旧92ff17deのEXACT56/CONDITIONAL17/欠測63と各値・physical/source/cohort、非価格signal/reference/旧資格は保全し、17件のaxis receiptによる更新だけを照合した。standard4の元physical labelとB canonical IDも別欄で確認。旧56/17/63は上書きしていない。

EXACTは保存compiled価格の物理SOURCE分類で、現在のRMSE・menu採択ではない。136候補を自動登録せず、DIRECT/MODEL/Sの区分を維持し、実Gを計算しない。新しいschema管理frameworkはない。

## 6. 次の担当とSTOP

[SCIENTIFIC_PLAN](SCIENTIFIC_PLAN.md)に1固定configuration×13round×1固定shot配分の有限実験案を記載した。未解決lossとloss上界・精度/confidenceからsimulation反復数を設計し、根拠のないtrial数・成功率を採択しない。H4の実nやRMSEを今回は数値評価していない。

**A**は全13round physical mean/normalization・局所/数値/state/target/law証拠。**B**は全shot統計・decoder・alias/未解決loss・energy RMSEの統計検証。**C/D**は平均compiled価格/残差。**ユーザー**はshot配分、count caller、joint law、未解決loss/support、精度/confidence/有限scopeを判断し、新科学実行を別途許可する。MinkowskiのL2契約・十分N・4方式比較条件と実G資格は未解決のまま。

commit/push、remote SHA/tree/主要artifact照合後**mandatory STOP**。新しい科学simulationへ自動進行しない。
