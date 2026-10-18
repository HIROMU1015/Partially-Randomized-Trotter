# PR α B-2：総失敗確率の感度、一回実行結果

2026-10-18 JST。**固定H6/H8候補に対し総α=.01/.05/.10を各軸へ均等配分して一回再会計した。完全3,660条件で基準α=.05の保存shot会計が完全一致した。比較可能な17条件では、基準から他2αへの34比較すべてでPR/PF順位が変わらず、最良候補の変更も0だった。**

PR削減率の基準αからの最大変化は**0.00471243844648パーセントポイント**。H6非相殺の60欠測は除外せず、5精度×3αのPF最小値とPR/PF順位をUNRESOLVEDに保った。全66比較が確定したという意味ではない。経験的bias・allowance・normalizationと保存費用に条件付く有限標本の点比較であり、厳密な成功保証・母平均の優位性の認定ではない。

[結果bundle](../../artifacts/resource_applicability/pr_alpha_sensitivity_run01/2026-10-18/)、[66比較CSV](../../artifacts/resource_applicability/pr_alpha_sensitivity_run01/2026-10-18/comparisons.csv)、[全11,160行を含む保存結果gzip](../../artifacts/resource_applicability/pr_alpha_sensitivity_run01/2026-10-18/result.json.gz)、[summary](../../artifacts/resource_applicability/pr_alpha_sensitivity_run01/2026-10-18/summary.json)。gzipはprivate出力の完全な無損失コピーで、raw signal・reference・private grant/plan原本は含まない。

## 実行範囲と来歴

- 科学SOURCE：`93145a4b24aa2e77f1eefdb39c2d70c56f9bddcb`。B-1結果commit `446ace970b2caa60858e3115d5f316b0b6870eb1` のGitHub公開を実行前に確認した。実際に使う5 file、Python executable・使用stdlibと固定inputのhashを実行前後に照合した。B-2はstdlibのみを使う。
- 入力：run03のB-2 pack/plan。[入力seal](pr_alpha_input_binding_20261018.md)、[公開binding](../../artifacts/resource_applicability/pr_alpha_input_binding/2026-10-18/run03/B2_input_binding_UNAPPROVED.json)。元のUNAPPROVED planを変更せず、直前に提示したB-2一回実行への利用者の「次の作業に進んで大丈夫」を、新規B-2専用grantへ結合した。B-1のgrantを流用していない。
- 対象モデル：H6/H8 linear隣接1 Å、STO-3G、tol-only1e-8、DF rank19/28、T=.8、保存Hamiltonian/state。split L_Dは各IDのp、PF formula/q・PR r/Kを固定。H6は決定論136（p5〜15とfull rank19）＋確認PR8（p5）、H8は決定論184（p7〜21とfull rank28）＋確認PR6（p7）。PF次数2/4、q=1/2/4/8、outer step T/q=.1〜.8。PR short step T/(qr)はH6=.05〜.2（r2/4）、H8=.025〜.1（r4）、K2/4。新しいδ・geometry・basis・DF・stateの走査はない。
- ε：H6=.05/.01/.005/.001/.0005、H8はさらに.000625を含む。有限時間complex signalの精度であり、energyのHa精度ではない。通常signed biasと経験的非相殺biasを分離、allowance倍率1、候補・入力値・単価を全αで固定した。

実行は20:34:41.947〜20:34:43.899 JSTのdispatch1回。driver wall1.440285秒、launcher wall1.952393秒、両native exit0・wait完了。`COMPLETE_WITH_INCONCLUSIVE_MANDATORY_STOP`は既知欠測を保持して完了した状態である。科学的な不一致によるSTOP・自動retryはなかった。[binding/hash receipt](../../artifacts/resource_applicability/pr_alpha_sensitivity_run01/2026-10-18/execution_binding.json)、[terminal](../../artifacts/resource_applicability/pr_alpha_sensitivity_run01/2026-10-18/terminal.json)、[native wait](../../artifacts/resource_applicability/pr_alpha_sensitivity_run01/2026-10-18/native_wait.json)。

## α・shot・費用の会計

総αは**単一complex signalの2軸に対するunion boundの配分**で、α_real=α_imag=α_total/2。理想fresh whole-trajectory IIDモデルの1 shot推定量X=B_corr Y、Y∈{−1,+1}に対する範囲上界とB_budgetを前提とする。[契約監査](pr_alpha_contract_audit_20261018.md)を再利用した。今回、二項確率・仮想測定・物理shotは評価していない。

軸bに対してh_b=ε/√2−bias_b−u_b、B_budget=exp(log_B+saved margin)を使い、凍結関数の演算順で

\[
N_b=\max\!\left(1,\left\lceil\frac{2B_{\rm budget}^{2}}{h_b^2}\log\frac{2}{\alpha_b}\right\rceil\right)
\]

を評価した。各入力で**最初にα=.05を照合してから**他αを計算する。基準照合はα_axis・N_total・軸別全状態・evidence_kind・eligibilityの等値で、3,660/3,660件が一致した。headroom≤0・UNKNOWN・log-domain等の状態を有限shotや0費用に変換しない。今回の実入力にはUNDETERMINED/LOG_DOMAIN_ONLYはなかったが、既存人工検査の対象である。

PRは保存確認n32の同じreplica/seed/event/compilerの2軸費用を対にし、G_i=N_real C_real,i+N_imag C_imag,iの有理数平均と標本SEを出力した。PFは保存単価を使う。eligible 5,619行の費用分類はPR paired実測300、決定論実測840、決定論model4,479。軸別mixed分類も実装で保持するが、今回のeligible行にはない。model推定の新fit・compileは0。

α=.05から.01への連続近似の共通倍率はlog(400)/log(80)≈1.36728、.10へはlog(40)/log(80)≈.84182。これはshot・総費用の増減である。同じαをPR/PFへ適用するため、連続近似ではこの因子が費用比から消える。保存結果にはceil前の値と整数丸め差も含まれ、今回の小さな削減率変化はこの会計と整合する。αによる絶対費用の変化と、PR/PF順位の変化を区別する。

## 結果

|項目|結果|
|---|---:|
|入力条件|3,720（完全3,660、既知欠測60）|
|基準会計完全一致|3,660|
|出力行|11,160|
|凍結shot会計を評価した行|10,980（基準3,660＋他α7,320）|
|ELIGIBLE / INELIGIBLE / MISSING|5,619 / 5,361 / 180|
|比較|66（PR点費用小36、PF点費用小15、未確定15）|
|他α対基準の確定順位比較|34、変更0|
|他α対基準の未確定順位比較|10、未確定を保持|
|他α対基準の最良候補比較|PR44・PF34、変更0|

以下はPR削減率100(1−G_PR/G_PF)の百分率表示。負値はPRの点費用が大きい。全行でα_total=.01/.05/.10の順。最良候補ID・有理数費用・paired SE・実測/model区分はCSV/JSONに保存した。

|系|bias|ε|α=.01|α=.05|α=.10|
|---|---|---:|---:|---:|---:|
|H6|通常|.05|−13.038095|−13.038957|−13.038607|
|H6|通常|.01|30.890684|30.890515|30.890538|
|H6|通常|.005|34.965248|34.965256|34.965232|
|H6|通常|.001|31.167695|31.167695|31.167695|
|H6|通常|.0005|10.031850|10.031850|10.031850|
|H6|非相殺|全5精度|未確定|未確定|未確定|
|H8|通常|.05|−8.023396|−8.022970|−8.027682|
|H8|通常|.01|33.237398|33.237274|33.237429|
|H8|通常|.005|31.454886|31.454900|31.454866|
|H8|通常|.001|22.961345|22.961345|22.961341|
|H8|通常|.000625|−0.497175|−0.497174|−0.497174|
|H8|通常|.0005|−50.812476|−50.812475|−50.812474|
|H8|非相殺|.05|16.623696|16.624251|16.621383|
|H8|非相殺|.01|34.684855|34.684864|34.684773|
|H8|非相殺|.005|35.125301|35.125323|35.125288|
|H8|非相殺|.001|39.415041|39.415041|39.415040|
|H8|非相殺|.000625|11.283196|11.283196|11.283196|
|H8|非相殺|.0005|−29.091008|−29.091008|−29.091008|

最大変化はH8通常ε=.05で、基準−8.02296973078%からα=.10の−8.02768216922%への変化。境界H8通常ε=.000625でも、3αすべてでPFの点費用が約0.497%小さいという関係を保った。ただし、この差の母平均での有意性を示す結果ではない。

H6非相殺は、旧B0Iの12候補×5精度に分解/baselineがなく60入力欠測、3αで180出力欠測。確認PR集合の点最小値は比較できるが、PF full-domain最小値が未確定のため削減率を出していない。

## 検査・保全と結論の限界

[保存結果checker](../../scripts/collect_pr_alpha_sensitivity_run01.py)は科学runnerを呼ばず、固定scope/grid、基準会計、欠測、正費用、shot/費用の単調性、保存値からの点最小値・有理数比・切替、native wait/markerを照合した。無損失gzipの展開bytes/SHAも一致。[検査記録](../../artifacts/resource_applicability/pr_alpha_sensitivity_run01/2026-10-18/verification.json)。SOURCEが同じため人工62件とCLI8件は既存記録を再利用し、再実行していない。主agent自身の検査であり、独立reviewではない。

[保全監査](../../artifacts/resource_applicability/pr_alpha_sensitivity_run01/2026-10-18/preservation_audit.json)は、旧準備・入力seal・B-1公開artifact、private sealed pack/plan、B-1 grant/marker/結果の計70 fileの実行前後hash一致を記録する。使用SOURCE5 fileも不変。旧STOP・grant・ledgerを上書きせず、他worktreeを変更・凍結していない。新規signal/sampling/compile/fit・二項確率評価・RPE解析は0、自動再実行0。

全10,980会計行の証拠分類はEMPIRICAL、`certificate_verified_by_this_function=False`、`familywise_winner_certified=False`を保持した。n32費用SEは費用平均の有限標本の不確かさであり、Hadamard測定に配分したαの証拠ではない。αを変更してもbias/u/normalizationの証明書不足やmodel費用の不確かさは解消しない。

今回確認したのは固定条件での総α感度である。複数signalの同時保証、候補選択後の保証、RPE全round、最終energy成功確率・総資源へα=.05を拡張しない。D01/A05説明訂正・B-1代表28軸診断・B-2固定gridは各指定範囲で完了したが、H4 law/provenance不足・H6欠測・厳密bias等の未解決点は保持する。

## 次の担当

- **ユーザー：**今回の報告とcommitを確認し、必要な場合はこのbranchを手元のGit認証でpushする。追加の科学検証を行うかは、残る欠測・保証範囲を踏まえて判断する。
- **Codex：**今回承認されたB-2は回収・記録で完了しmandatory STOP。自動で実行する次の科学作業なし。
