# V2 S1：固定保存データの補正fit・既知データ評価

2026-11-10 JST。[事前設計](../h4_df10_price_v2_design_20261110/REPORT.md) §4–6（`df1dcd7bf0469a724493e0e4193b107ad3a157f0`）に対する利用者の実行許可の範囲を実施する。新たな承認証明・grantは作らない。

- 入力：carrier `eec5bdf9cc391fe49b55b5673f3e385deeb20d42`のV1 DEVELOPMENT/HOLDOUT JSONをGit objectから読み取る。原trajectory、QPY、NPZ、provider、SCF/DF/ground solver、元runnerを呼ばない。
- 条件：Linear H4/1Å/STO-3G・保存tol-only DF10/sector(2,2)・保存DF-H ground target/数値state条件付き/P=0。S2、δ=.1、PF B0/p7、PR p1/R4/K2、元symmetric_directional・測定付きfull Hadamard wrapper、Qiskit1.3.0/rz,sx,x,cx/opt1/seed17の**保存価格**。化学精度.00159360010199040、αtotal=.05/方式、βtotal=.4、h=.1、13roundは不変。
- 校正：PF q8の1 pathでMq、PR探索q8の9 pathでMq/MqBの計3 fit。Mqはq列だけ、MqBはqと保存tail_basis_groupsの2列。応答はcompiled−V1 structural。追加intercept/特徴探索/regularization/希少order係数なし。
- 既存`track_d_c7_4_models.fit`を同じcarrierの関数本文のまま再利用する。ASTでfitだけを読み、無関係なD moduleのimportを避ける。equal-q最小二乗、rank検査、X/Y二応答は不変。PF/PRを別fit、X/Yを独立nにしない。
- 評価：凍結M0/Mq（PF）、M0/Mq/MqB（PR）を、開発と既知V1使用済みPF q16/PR確認33へ適用する。確認データを未使用独立holdoutへ再認定しない。実装と本仕様を科学実行前にcommitし、3 fitと評価を一括実施する。
- 出力：RESULT.jsonにSOURCE・入力digest・係数/rank・cohort/method/axis別のbias/MAE/RMSE/残差範囲/相対誤差/PR残差SD、保存basis/order2層別を記録。PATH_RESIDUALS.csvに丸めない全path/axis/model予測と残差を保存する。PR平均誤差SEはIIDを仮定しfitを条件付けた記述診断で、係数不確かさや系統差を含むCIではない。trainingには同SEを付けない。
- rank不足・非有限値は評価不能、負価格は診断として残しclampしない。追加候補を自動探索せず、他の有効候補を固定仕様で評価する。既存出力を上書きしない。
- 人工テストのみでrank、二応答、校正/既知価格分離、axis/feature binding、zero/負値/非有限値を確認。科学評価はこの承認された保存範囲だけ。
- 新構造予測、再synthesis、bootstrap、law期待値、sampling、実回路build/compile、長q価格、signal/N/N×C/G、候補最適化、詳細科学レビューは0。V1モデル・入力・結果・不一致判定、旧SOURCE/raw/grant/ledger/marker/KEEP/STOP、他worktree/processを保全する。
- 結果と限界を公開、remote SHA・主要公開成果物を照合してmandatory STOP。長q/独立汎化/全RPE providerへ自動採用しない。
