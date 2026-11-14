# PR-DF LD1：保存24軸pairのexact残差較正とm4予測固定

2026-11-15 JST。今回の利用者指示により、指定保存値の集計・2軸fit・m4予測2件の実行を許可済み。追加承認証明・grantは作らない。

- 入力：c99d511e2のC受入済みP2 fresh m0/m6各4path×X/Yと、D2 2592d442aのm3 fresh4path×X/Y、計24軸pair。m3実行SOURCE2f423befb・原結果4031dbab。
- 受入：INPUTS.jsonの24行は元S/C・receipt・cohort/index/seed/event digest・physical IDを個別照合。m3の元scalar/new ZZ/compiler/profile/測定・preparation・gate・full-wrapper statevector=falseを保持。P1/replayは参照のまま。
- 科学処理：各pairの整数C−S、各cell/axisの4path算術平均6件。特徴量[1,q,qR]の3×3 exact有理数系を既存price_model_validation._solveでX/Y各1回解く。非零pivot3個の成立でrank3を確認する。較正一致は算術確認で独立検証ではない。
- 予測：保存E_I[S]のm4 X/Yの各binary64値をexact有理数として保持し、各残差モデルのq16/R32予測を足す。表示floatで再構成しない。元の構造区間・law/SOURCEも保持する。2予測のみ。
- m4：physical 3df0cfd3356c358aeda8、T1.6/q16/R32、元fresh4pathのcohort/seedをD2取得前INPUTSから参照。価格は読まず、将来のOは各軸4path平均。P−O、絶対差、符号付き/絶対相対差（分母O）、O=0の相対null、欠測時n4保持。許容差・PASS/FAILを新設しない。
- 実装再利用：CのSOURCE-bound reader・前段path照合、既存exact Gauss-Jordan solver/predict。固定24組/3×3のためprocess並列化の対象となる重い数値処理はなく、保存文書のreader cacheを再利用する。科学call中の段階開始/終了/所要時間を保存する。
- 実行前に公開ref/関連path/job名・既知D2 namespace名を確認。確認範囲の限界をPREOPEN_CHECKへ保持し、同等fit/予測やm4取得の疑いがあれば解決前に実行しない。
- 不一致・欠測・rank不成立は部分結果を保持してSTOP。閾値緩和・補完・自動refit/retryはしない。実行はfresh run01一回、元入力/旧結果は上書きしない。
- IとQの差、有限cohort・rare・共有誤差は未知、CI/適格E[C]/G/winnerはnull。

**SOURCE固定→指定集計/fit/予測→係数/予測固定・公開・D2引渡しでmandatory STOP。m4取得/開封/判定、他round予測、sampling/compile、全law期待価格、CI、N/Gへ進まない。**
