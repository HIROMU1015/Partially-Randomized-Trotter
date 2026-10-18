# H4/H6 new_2 native・同精度代表比較

前回数値診断7198bbde…を保全した独立追加検証。利用者の追加検証指示を根拠に、人工14件PASS後、H4 prefix3/H6 prefix5でS2 q4・Yoshida q2・new_2 q1、K4、T=.8、ε=.001を固定。各8軌道・paired2軸、合計96 compile。保存状態とgeneration順は維持。既存venv/install・他job・GPU操作なし。

[plan](plan.json)、[入力](input_binding.json)、[source closure](source_closure.json)、[環境/compiler](environment_compiler_profile.json)、[人工検査](artificial_tests.json)、[係数次数](coefficient_order_audit.json)。新Gaussian構造化/ordinary controlの費用系列であり、旧dense費用へ混合しない。原SOURCE c31ef42…のsourceを変更せず、専用process内のadapterを使用。

保存8桁new_2係数は立方和7.346136322761508e-9が残り、厳密4次ではない。70桁toy検査で実用4次から漸近2次への遷移を確認。係数は補正しない。

独立occupation oracle・native branch・Hadamard・compiled vectorを代表軌道で照合。order0/2/4の正負time forced fixtureは費用標本へ入れない。8軌道の費用平均/SEは小標本の点診断であり、希少層・母平均・厳密bias・全RPE energy費用/全候補最良の証明ではない。P=0・empirical allowance1e-9・logB margin1e-12を事前固定。実行後、結果を別commitへ固定してSTOPする。

初回SOURCE8b0f6cf…はgeneric4x4 unitaryのcompileでstatevector差1.32e-6を検出してSTOP。全6childを回収し、10compile entry＋切分け1を保持。固定1e-9は緩和せず、explicit XXPlusYY/number Phaseへ交換し14人工検査PASS。旧costを新系列に混ぜず、新SOURCE/run02で96compileを行う。[失敗・切分け](run01_stop_and_diagnosis.json)。

## 最終結果（取得完了）

[報告](../../../../docs/research/track_a_h46_new2_native_validation_20261018.md)、[比較JSON](representative_cost_summary.json)、[CSV](representative_cost_summary.csv)、[終了監査](final_completion_audit.json)、[引継ぎ](parallel_handover_receipt.json)。H4 SOURCE687679a4…とH6 executor SOURCEf0dd2a9a…を分離し、seed namespaceと科学helper8filesを同じに保持。H4完了48軸を再実行せず、H6は24軌道並列・48軸を完了。途中のH6単価は採用平均へ混ぜず、重複13軸の完全一致だけを照合した。

採用96軸・数値222件PASS/最大1.57e-10・人工18件PASS。実compile entry121を返却せず記録。最終の採用processはexit0/wait回収、旧handover中断は中断のまま保持。[checker](verify_saved_result.py)は保存hash・SOURCE・coverage・paired費用会計だけを再検査し、科学計算を再実行しない。

[係数図](coefficient_order.png)、[費用図](representative_cost.png)、[plot helper](plot_saved_results.py)。8桁係数の厳密4次条件不成立、n8/代表q/empirical/P=0の制限を保持。旧global campaign/対決定論の結論へ外挿しない。完了後STOP、pushは行っていない。
