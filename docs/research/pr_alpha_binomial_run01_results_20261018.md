# PR α B-1：保存signalの二項失敗率、一回実行結果

2026-10-18 JST。**固定したH6/H8代表14条件・28軸を一回評価し、28軸すべてがα_axis=.025以内、各complex signalの2軸union boundも.05以内となった。結果は理想fresh-IID Hadamardモデルと保存bias/allowance/normalizationに条件付く数値診断である。**

最大の軸別失敗確率は0.0021448205475243535、最大2軸union boundは0.00214496460418531。いずれもH8 `B2_p7_q2_2nd_r4_K2`、ε=.05で、軸最大はimag。underflow・数値未確定・α超過は0軸。

[結果bundle](../../artifacts/resource_applicability/pr_alpha_binomial_run01/2026-10-18/)、[軸別CSV](../../artifacts/resource_applicability/pr_alpha_binomial_run01/2026-10-18/axis_failures.csv)、[保存値をそのまま投影したJSON](../../artifacts/resource_applicability/pr_alpha_binomial_run01/2026-10-18/binomial_results_public_v1.json)を参照。private estimator p・整数cutoffとraw scalar pack・grant原本は公開しない。完全なprivate結果のhashを公開receiptへ結合した。

## 実行範囲と来歴

- 科学SOURCE：`93145a4b24aa2e77f1eefdb39c2d70c56f9bddcb`。入力seal/報告commit：`2c0e60803374a0a355a9513cf2a3315d678f138f`。実行前に両commitの来歴と報告のGitHub取得を確認した。
- 入力：run03の固定B-1 pack・private plan。[入力seal報告](pr_alpha_input_binding_20261018.md)、[固定14条件・予算](../../artifacts/resource_applicability/pr_alpha_input_binding/2026-10-18/run03/B1_input_binding_UNAPPROVED.json)、[保存h/N](../../artifacts/resource_applicability/pr_alpha_input_binding/2026-10-18/run03/B1_saved_shot_scope.csv)。候補やεを再選択していない。
- 承認：直前に提示した「B-1の14条件・最大28軸の一回実行」に対する利用者の「次の作業に進んで大丈夫」を、B-1専用grantへ結合した。元の未承認plan/packを変更せず、別の承認記録と一回grantをprivateに保存した。B-2への承認拡張はない。
- 対象モデル：H6/H8 linear隣接1 Å、STO-3G、tol-only1e-8、DF rank19/28、T=.8、保存Hamiltonian・state。split L_Dは各IDのp、PF formulaとq、PRのr/Kを固定。PF outer stepはT/q（保存q=1,2,4,8）、PR short stepはT/(qr)（保存r=2,4）。新しいδ走査はない。εは有限時間complex signalの精度で、energyのHa精度ではない。
- 統計条件：通常bias、allowance倍率1、各軸α=.025、軸許容幅ε/√2。X=B_corr Y、Y∈{−1,+1}、fresh whole-trajectory IIDモデル。B_corrとB_budgetを分離し、保存h/Nを使用した。

20:18:51.785〜20:18:53.320 JSTの一回dispatchで完了。driver wall1.024689秒、launcher wall1.535935秒。driver/launcherともnative exit0、parentのwait完了を保存し、mandatory STOPした。[実行binding](../../artifacts/resource_applicability/pr_alpha_binomial_run01/2026-10-18/execution_binding.json)、[driver terminal](../../artifacts/resource_applicability/pr_alpha_binomial_run01/2026-10-18/terminal.json)、[native wait](../../artifacts/resource_applicability/pr_alpha_binomial_run01/2026-10-18/native_wait.json)。

## 保存shotに対する結果

下表のreal/imagは保存N/hでの事象 `|z_hat−z_approx| >= h` の二項分布数値評価である。union boundは周辺2確率の和であり、軸独立性を仮定した厳密joint確率ではない。表は表示丸め、JSON/CSVは保存binary64値を保持する。

|系|ε|候補ID|real失敗確率|imag失敗確率|2軸union bound|
|---|---|---|---|---|---|
|H6|0.05|B0I_p7_q1_2nd|1.2652899e-08|0.00055019852|0.00055021117|
|H6|0.05|B2_p5_q1_2nd_r4_K4|1.1184024e-05|0.00079337605|0.00080456007|
|H6|0.001|B0I_p11_q2_4th|1.6961715e-08|0.00050514619|0.00050516315|
|H6|0.001|B2_p5_q8_2nd_r2_K2|1.8978127e-07|0.00058979388|0.00058998367|
|H6|0.0005|B0I_p11_q4_4th|1.7125083e-08|0.00050449789|0.00050451501|
|H6|0.0005|B2_p5_q8_2nd_r2_K2|1.8969638e-07|0.00058954057|0.00058973027|
|H8|0.05|B0I_p11_q1_2nd|2.3360184e-18|0.001836044|0.001836044|
|H8|0.05|B2_p7_q2_2nd_r4_K2|1.4405666e-07|0.0021448205|0.0021449646|
|H8|0.001|B0_p15_q8_2nd|5.2375849e-23|0.0019150597|0.0019150597|
|H8|0.001|B2_p7_q8_2nd_r4_K4|3.0805096e-14|0.0019849407|0.0019849407|
|H8|0.0005|B0I_p15_q4_4th|5.3609283e-23|0.0019145912|0.0019145912|
|H8|0.0005|B2_p7_q8_2nd_r4_K4|3.0681798e-14|0.0019845402|0.0019845402|
|H8|0.000625|B0I_p15_q4_4th|5.3678715e-23|0.0019154047|0.0019154047|
|H8|0.000625|B2_p7_q8_2nd_r4_K4|3.0688627e-14|0.0019848375|0.0019848375|

全28軸で二項失敗率は保存したHoeffding上界以下、各Hoeffding上界も.025以下だった。二項確率とHoeffding上界は別欄に保存している。28軸のCDF/SF呼出しは最大枠56内、logCDF/logSFも56内。実際のcutoff・空tail処理からfrontend call数を56/56と復元したが、backend内部の呼出数を実測したcounterとは呼ばない。

## 確認できた範囲と限界

今回確認できたのは、事前固定した14条件の保存shot契約について、宣言した理想測定モデルの軸別二項失敗確率が予算以内という点である。`>=`の整数境界は保存binary64値の有理数表現で決定し、SciPy1.17.0/NumPy2.5.4の固定backendでCDF/SFを数値評価した。Python executable、使用stdlib、packageのRECORDと記載ファイルhashを実行前後に照合した。

「厳密二項」は分布モデルを指す。浮動小数点CDFを外向き丸め付き区間証明とせず、保存N自体も既存`BINARY64_DIAGNOSTIC`のまま。bias・u・log_B_marginはEMPIRICALで、厳密なsignal誤差証明へ昇格しない。A03の証明書不足は未解決である。

物理Hadamard shotやRTE trajectoryを新たにsamplingした検証ではない。評価モデルはfresh-IIDで、固定n32回路集合を反復測定した場合への保証移送をしていない。n32費用SEは測定αの証拠に使っていない。

α_total=.05は各complex signalについての条件である。14 signal同時成功、候補選択後、RPE全round/energy成功確率、PR母平均費用やPR/PFの科学的優位性を認定しない。主資源表のshot・RZ費用・順位は変更しない。

## 検査・保全・STOP

[検査記録](../../artifacts/resource_applicability/pr_alpha_binomial_run01/2026-10-18/verification.json)で14条件/28軸、整数shotとevent、各α、no-underflow、input/SOURCE hash、markerとplan、native終了を確認した。結果の回収・表作成では二項確率やshot・費用を再計算していない。SOURCEを変更していないため、既存の人工62件・CLI8件PASSを再利用し、無意味な再実行はしなかった。主agent自身の技術確認であり、独立reviewではない。

新しい二項軸評価28、scientific dispatch1。shot再計算・費用再会計・signal生成・sampling・compile・自動retry・B-2 dispatchはすべて0。使うSOURCE/input/runtimeだけをgateにし、無関係なworktreeのHEAD/statusや一時的な共有CPU競合を停止理由にしなかった。共有設定・他ユーザーprocess・他worktreeを変更していない。

[保全監査](../../artifacts/resource_applicability/pr_alpha_binomial_run01/2026-10-18/preservation_audit.json)に旧input/結果・B-2 plan/input・SOURCEの不変を記録した。消費済みgrant・marker・全private結果は保持し、再使用しない。B-1の数値評価単位を完了しmandatory STOP。

## 次の担当

**ユーザー：**総α=.01/.05/.10・均等2軸配分でのB-2を実施するか判断し、必要ならB-2専用の一回実行を明示承認する。対象はH6 144/H8 190候補、固定ε・通常/非相殺・倍率1、最大11,160出力行（完全3,660条件×3αの最大10,980会計call、欠測180行を含む）。H6非相殺60条件の不足は保持し、full-domain winnerを補完しない。

**Codex：**B-2の承認がある場合だけ、独立grantと未使用output/markerを結合して実行し、結果を保存してSTOPする。B-1の完了を理由にB-2を自動開始しない。
