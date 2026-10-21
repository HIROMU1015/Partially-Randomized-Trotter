# H8希少RTE：保存値統合・残余感度の固定契約

2026-10-21。現ユーザー指示が、固定H8の2条件に限る保存値科学解析1回を明示承認している。追加承認待ちは不要。実行SOURCEを先にcommitし、別の契約・承認receipt・新grant・非返還ledger・one-shot markerへ結合する。旧C-1等のgrantは使わない。成果物は[固定定義](../../artifacts/resource_applicability/h8_rare_rte_saved_integration/2026-10-21/analysis_definition.json)と同directoryの機械可読記録。

段階B `3d5ab5bb1cf0d78a3157ad7f742cfa5c7b58efbb`、C-1結果 `f377f8869c697c4846e3614b0ae7bc9d3ad5cb3c`、C-1 SOURCE `383dd9068504b8f578944984fc45a60dd4ed18b2`を固定する。線形H8/1Å/STO-3G/tol-only DF28/T=.8/P=0/通常signed bias/allowance1。

|条件|固定PR|既存S0|新S2_SINGLE|固定PF / G|
|---|---|---:|---:|---|
|ε=.05|B2_p8_q2_2nd_r4_K4|48|43|B0I_p11_q1_2nd / 1267321444|
|ε=.0005|B2_p7_q16_2nd_r4_K4|48|43|B0I_p15_q4_4th / 94193016652860|

元input/preparation/analytic shots・full Hadamard wrapper/compiled RZ会計・Qiskit1.3.0/rz,sx,x,cx/opt1/seed17の**保存結果**を使う。今回Qiskitや分子runtimeをimportしない。cohort行のpool、候補再選択、law再形成・再正規化を行わない。段階Bのcanonical層確率100桁文字列を有理数入力として使う。その小さな丸め誤差を正規化によって隠さない。

`Khat=p0*m0hat+p1*m1hat`、`E[G]=Khat+pM*muM+p4*mu4+delta_K`。`delta_K=p0*(mu0_canonical-m0hat)+p1*(mu1_canonical-m1hat)`であり、標本誤差とsampling lawのtarget差を含み得る。未知層平均・分散はnullのまま保持する。残余の全support費用上界、全母平均、母平均の順位保証を新設・採択しない。

層内iid/有限分散、旧Pの観測S0本数48に条件付けた抽出と新pilotの独立性を仮定した推定分散は`p0²*s0²/48+p1²*s1²/43`。paired G内の両軸を独立とは仮定しない。seed非衝突は実装証拠で、独立性の数学的証明ではない。独立性を置けなければ共分散項が未知となる。±1/2/3 SEは境界の感度座標で、信頼区間・正規分布適合・母平均保証ではない。全層の分散・SEも未評価である。

逆転境界は`pM*muM+p4*mu4=D=G_PF-Khat`。非負費用領域で線の下/上/線上を条件付きPR低費用/PF低費用/同費用と読む。各層の軸切片と、残余budgetの配分t=0,1/4,1/2,3/4,1の5座標、k=0/1/2/3 SEでの境界移動を保存する。軸切片の他方0は仮想座標であり未知費用の0補完ではない。残余混合平均は確率重み付きの定義で、2層同費用の仮定ではない。段階B比で新たに測定側へ移る確率質量と境界を比較するが、従来未知だった分散の減少量を有限数値で主張しない。

C-1は保存binary64を厳密整数比にする条件付き抽出を使用し、canonical次数確率との最大差1e-12 engineering gateを通過済み。この既存receiptを転記する。次数差・component正規化・PRNGを含む費用law全体がcanonicalと厳密同一という証明、期待費用biasの数値上界はない。今回の統合はその整合証拠に条件付け、unknown costによるlaw差の費用保証を付与しない。

科学call予算はcampaign1、2条件のprimary各1、別算術各1、決定論的表出力1。各呼出前に非返還予約する。既知層4行・境界10行・SE感度8行・条件概要2行を出力し、主要値を別経路で照合する。主経路は元C05のcentered Fraction統計を再利用し、別経路は別CSVから整数総和/二乗和とDecimal140、isqrtで検査する。閾値は相対/絶対1e-85。担当Codex自身の別実装であり、別reviewerや数学的certificateではない。

実行前に固定Git blob/SHA・保存SOURCE・使用input・現在の標準ライブラリ/実行ファイルSHA・保全目録を照合し、人工検証・構造preflightを通過する。数値演算はtiny2条件・各91行のため1数値processでまとめ、共通入力を1回readして重複処理を減らす。CPU/RAM/workerの恣意的capは設けず、実空き資源を記録する。共有設定・他process・他worktreeは変更しない。

科学load前に新markerをO_EXCL消費し、失敗/中断時は予約・部分結果・native exitを保持してSTOP、同grant再試行/resumeを禁止する。数値不一致や科学定義の変更が必要ならSTOP。成功後もローカルcommitと終了監査後mandatory STOP。新sampling/simulation/signal/回路/compile、production、詳細review、pushは行わない。
