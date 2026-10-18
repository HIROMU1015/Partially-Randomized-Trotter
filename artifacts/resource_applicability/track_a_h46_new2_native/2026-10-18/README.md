# H4/H6 new_2 native・同精度代表比較

前回数値診断7198bbde…を保全した独立追加検証。利用者の追加検証指示を根拠に、人工14件PASS後、H4 prefix3/H6 prefix5でS2 q4・Yoshida q2・new_2 q1、K4、T=.8、ε=.001を固定。各8軌道・paired2軸、合計96 compile。保存状態とgeneration順は維持。既存venv/install・他job・GPU操作なし。

[plan](plan.json)、[入力](input_binding.json)、[source closure](source_closure.json)、[環境/compiler](environment_compiler_profile.json)、[人工検査](artificial_tests.json)、[係数次数](coefficient_order_audit.json)。新Gaussian構造化/ordinary controlの費用系列であり、旧dense費用へ混合しない。原SOURCE c31ef42…のsourceを変更せず、専用process内のadapterを使用。

保存8桁new_2係数は立方和7.346136322761508e-9が残り、厳密4次ではない。70桁toy検査で実用4次から漸近2次への遷移を確認。係数は補正しない。

独立occupation oracle・native branch・Hadamard・compiled vectorを代表軌道で照合。order0/2/4の正負time forced fixtureは費用標本へ入れない。8軌道の費用平均/SEは小標本の点診断であり、希少層・母平均・厳密bias・全RPE energy費用/全候補最良の証明ではない。P=0・empirical allowance1e-9・logB margin1e-12を事前固定。実行後、結果を別commitへ固定してSTOPする。
