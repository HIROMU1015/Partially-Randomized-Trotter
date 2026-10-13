# 図の再利用用caption

4図ともlinear H4/STO-3G/legacy固定DF12/8 modes/sector36/T=.8/delta=.8/q/S2、
元ordinary-controlled structured JW Gaussian cost series、所定Qiskit1.3.0 opt1/rz,sx,x,cx/seed17、P=0。
302点は保存32 paired costの精度再投影であり独立実験ではない。

1. **Registered method costs.** 6geometryについて登録B0（PM-1補完を含む）/B1/B2/B3の適格点最小費用。空欄は不適格。全1812点で元登録集合内の点最小はB2。
2. **Precision/configuration map.** 登録deterministic最小に対する削減率とB2点最小q/r/K（prefix3）。geometryはカテゴリ行であり物理補間ではない。全点に±2SE区間の重なる近接候補がある。
3. **Cost factors.** epsilon=.05。左は距離別再選択の登録deterministic比をN比×C_effective比へ分解。右はp3/q1/r4/K4を固定し.70 Å基準で比較。effective-RZはaxis shot比で加重される。
4. **Unmeasured B0+I sensitivity.** 左は追加compiled RZ差Δ=0の補助対照参考比、右は全候補・axis・geometry共通非負Δという仮定下の点数。補助再選択とzero-Δ構成固定を分ける。実compiled費用は未測定、正式baselineではない。

PNG表示とSVG/PDF exportを併記。図4の感度を図1〜3の元比較値へ混入させない。
