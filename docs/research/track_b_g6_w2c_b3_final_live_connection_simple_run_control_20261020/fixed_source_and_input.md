# 固定SOURCE・入力・runtime

直接基点 `841ac8920e41e93e2e79574f6ecdc6213a570f88`、旧測定 `2405f5e2e5fb0dc5add39b102e3ccae3d56df0a4`、新測定 `a1c4f41b6c29baf691006a8ef39e87b8ab9aa8d8`。
科学参照 `76ca3c4089b201d55da0fede08e1f77733ac38a5`。Core6はGit bytes/SHA256/ASTだけで照合し、import/loadしなかった。
旧114 SOURCE pinsと10 input pinsは旧測定・基点公開HEAD・新worktreeで等しい。
詳細は[source inventory](source_inventory.json)と[静的監査](acceptance/static_preparation_audit.json)。

P5_general_order：p=(1/5,3/10,1/2)、x=5/7、m=5、sigma=+1。
root256/H160、eta=rho=1e-12、epsilon_axis=1/200、primitive=1e-6、alpha_axis=1/320、familywise=1/20。
A6 tau=1e-6/K_phi=32/J=32、math-only。Nativeへ拡張しない。
Python executable、実際にloadedしたstdlib/shared library、worker runtime/schema/import pathsをSHA256で固定する。
正式共通指示7eff8a33の§8.4を実際に読み適用。空き資源観測を静的監査JSONへ保存。
保護4 worktreeは既知pathのHEAD/status観測だけ。無関係な変更を実行拒否条件にしない。


**実科学未実行、実行未承認。mandatory STOP。次の担当：ユーザー・GPT。**
