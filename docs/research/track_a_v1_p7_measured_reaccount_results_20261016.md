# V1：p7/q1 S2の実測費用によるε=.05限定比較

**保存shot数と両軸の実測RZを接続した決定論費用は356,097,424 RZ。保存PR確認点403,423,200 RZに対するPR費用削減率は−13.29012029%。この限定比較では、実測費用でもPR優位の逆転が維持された。**
通常signed signal bias・経験的allowanceの条件付き比較であり、PR母平均の優劣、厳密bias保証、全候補の実測最良、energy/QPE/RPE/FT費用を認定する結果ではない。

利用者の「次の作業に進んで大丈夫」を、[既存比較計画](../../artifacts/resource_applicability/track_a_v1_p7_readiness/2026-10-16/future_accounting_plan.json)の別途再会計への承認として新しいgrantに記録した。[実compile結果](track_a_v1_p7_compile_results_20261016.md)は公開commit `a73ce48410264305f257ebe5d7c1c2c2c527fe47`とremote SHAの一致を確認した。実行コードはcommit `e078a7e6a8b53226f16994f405732785917a0779`の[stdlib限定runner](../../scripts/track_a_v1_p7_measured_reaccount.py)へ固定した。

[機械可読結果・入力hash・承認receipt・終了監査](../../artifacts/resource_applicability/track_a_v1_p7_measured_reaccount_run01/2026-10-16/README.md)。元の計画・未承認draft・旧compile grant・ledger・lock・rawを変更せず、新grantと新one-shot lockで一度だけ実行した。

## 固定した対象と入力

linear H6、間隔1.00Å、STO-3G、保存tol-only DF rank19、12 modes、sector400、T=.8、δ=.8、B0+I、prefix7、q1、S2、P=0/RZ。比較対象は`B0I_p7_q1_2nd`と保存PR確認候補`B2_p5_q1_2nd_r4_K4`の1組だけ。allowance倍率1と10の保存条件を各1行報告する。

- Phase A保存shot：cosine10,216、sine14,356、合計24,572。倍率1・10とも同じ。shotの再導出・丸め・最適化は行っていない。
- 実compile：cosine/sine各14,492 RZ。各軸の固定model予測14,524.2 RZからの誤差は161/72460＝0.222191554%で、各軸独立に2%未満PASS。
- PR費用：保存確認union8/n32の403,423,200 RZ。探索72/n8、母平均、理論期待費用へ読み替えない。
- 既存symmetric-directional control、単一scalar phase、保存full-H reference、経験的allowance、compiler profileと保存適格性を維持した。

## 実測を用いた算術

`G_det = 10216 × 14492 + 14356 × 14492 = 356097424 RZ`

|allowance倍率|決定論費用RZ|保存PR費用RZ|PR/決定論|PR削減率|
|---|---:|---:|---:|---:|
|1|356,097,424|403,423,200|1.132901203|−13.29012029%|
|10|356,097,424|403,423,200|1.132901203|−13.29012029%|

比は厳密に`25213950/22256089`。削減率`100*(1-G_PR/G_det)`は`-295786100/22256089`%。負値は、この決定論候補より保存PR点費用が高いことを意味する。PRに対する決定論側の削減率へ同じ百分率を流用しない。

旧固定modelによる候補費用は356,888,642.4 RZ、PR削減率−13.03895727%だった。今回、この候補の2軸費用だけを実測値へ置き換えると、逆転の符号は変わらない。旧全136候補の再ランキング、他4精度の更新、別bias ruleの最良選択は実行していない。

## 実行・検証・保全

新grantをscope `p7_measured_cost_single_comparison`、新規会計1比較・2倍率行に固定。8個の会計入力をbytes/SHAで照合し、公開入力7個をa73ce484…のGit blobへ、保存p7 signalを既存receiptへ結合した。登録2軸の全PASS・正常終了・全回収と保存shot/PR/candidate identityを確認した後に会計を予約した。

1process/thread1、idle CPU5、AS512MiB/RSS256MiB、CPU30秒/wall60秒の上限で実行。会計runner wall0.010498秒、current-exec peak RSS18MiB、exit0、mandatory STOP。CPU記録2.482606秒にはexec前のmetadata gateも含まれる。再実行・自動retryなし。

実行後にDecimalによる軸別重み付き和、総shot×同一軸費用、比と削減率の整数cross-product、旧model値と旧削減率の再現を照合した。これはprimary agent自身によるローカル算術・保全検証であり、新しいGPT/Claudeレビュー・独立再現・CI全体の成功ではない。

SOURCE22（旧21＋新runner）、provider230、runtime34,273、旧科学入力15receipt、新会計入力8を終了後にも照合した。旧p7 compile raw receipt 22個と、H5/H7を含む旧grant/ledger/carry/marker等663ファイルは不変。既存結果・V2・H5/H7を保全した。

新compile・signal/reference/scalar/primitive/SCF/DF/state生成・PR sampling・shot再計算・fit・候補再選択・V2は0。経験的allowance、旧B0I12候補の非相殺bias分解欠測、厳密bias certificate未取得、全grid費用誤差未保証、旧binary bit同値未確認、PR母平均費用未保証は未解決のまま。

**実測に基づくε=.05の限定比較完了・mandatory STOP。次の科学計算や追加探索へ自動移行しない。成果物はlocal commitに保存し、pushは利用者が行う。**
