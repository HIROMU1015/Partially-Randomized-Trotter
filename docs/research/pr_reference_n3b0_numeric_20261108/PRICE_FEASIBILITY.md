# N3-B0：価格評価へ進むための条件

2026-11-08 JST。[REPORT](REPORT.md)。固定preflight`385c24a00fb9b47d6639fea1e7aecb0c37d55b23`の範囲内で、価格条件を整理した。**本段階で新しいT価格の和／最大値／期待値は計算していない。**

## 固定された辞書仕様

H4・1 Å・STO-3G・DF12、Partial S2/LD3/h=.02/K2、m7/q128/r1/73 bits、元H ground供給・状態準備費除外、compiler A／Qiskit1.3.0、最終native epsilon=10^-11を維持する。選択結果はb=36、j=-36…6、43 key、各基本wordのFrobenius割当10^-13。既存3 archive内の一致recordはなく、全43件が取得必要である。

この結果は**有限取得対象の仕様**であり、その取得が成功する保証ではない。pygridsynth2.0.0の全入力停止未資格を維持する。今回の内部precision・数値選択は、未知のT価格やcacheの成功しやすさを使っていない。

## 残る価格条件

各基本wordを実際に取得・資格し、その採用T+T†数をt_jとする場合、単純連結の一様上界候補は

\[
B_{dict}=\sum_{j=-36}^{6}t_j.
\]

43というkey数からt_jやこの和を推定しない。20,000文字拒否、経験的T則、存在する別word、最小T下界を採用価格へ代入しない。全43件の価格は未取得として扱う。

CT-4B既存C0を優先してN1の固定価格を保持するpolicyには

\[
B_P=\max(B_{dict},B_{cache})
\]

を覆う証拠が要る。B_cacheは、実際に優先採用する既存wordのTを安全に覆う値である。今回は既存recordのT metadata整合だけを検査し、この最大値も新集計していない。既存wordを辞書で勝手に置換して旧N1閾値を使わない。

保存[N1 RESULTS](../../../artifacts/pr_reference_n1_numeric/2026-11-08/RESULTS.json)のpointwise十分条件は、同じcompiler条件C／共通policy／全word資格の下で**B_P < 173790**。等号では厳密優越を保証しない。今回の判定はPRICE_BOUND_CONDITIONALであり、実T上界が閾値以下になったという結果ではない。

期待閾値`1602930468830332421754729/9223372036854775808`も保存値の引用だけ。一様capの移送又はcanonical joint lawの期待価格証拠が必要であり、真空位相分布・bit確率・期待Tは計算していない。保存2 trajectoryの平均や最大値を全support capへ使わない。

## 科学的保証との関係

実基本wordが各割当を満たすだけでなく、実emitterに対する[ERROR_BUDGET](../../../artifacts/pr_reference_n3b0_numeric/2026-11-08/ERROR_BUDGET.json)の`e_emit < R_emit_lower`を資格する必要がある。その要求値は約5.55121e-13であり、実誤差の上界は未取得。未証明の0を代入してT優越や全support CT RPE成功を認定しない。

後続で粗い価格capが閾値を超えても、Referenceの一般的劣位を意味しない。改善を得るためにb／precision／辞書／fusionを自動拡張しない。新規取得・guard・emitter検証・価格会計・N3-B1は別途のユーザー判断・実行scopeを必要とする。

今回の数値scopeは完了し**mandatory STOP**。次担当はGPT／ユーザーであり、Codexによる追加価格取得・全18round総T計算は開始しない。
