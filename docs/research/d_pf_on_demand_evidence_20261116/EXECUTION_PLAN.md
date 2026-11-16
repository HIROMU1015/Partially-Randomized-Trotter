# 選択された決定論PF入力の補完

対象は保存H4直線1Å/STO-3G/tol-only DF10、h_RPE=.1、M12、P0除外。最新依頼の許可内で、旧17不足構成からFull PF/S2/要求幅2h (9e96ed75f7f10b5f993ba9e2841684e52ad97b7d4e9df07b7b0070f02c852981) を通常入口への受入確認に使う。探索器の最適候補と認定した選択ではない。17構成・全pの一括科学計算はしない。

保存信号m0/1/2/3/12を再利用し、保存step・保存referenceからm4..11の8 complex meanのみ取得する。信号のscalarは総Tに一回。新reference、sampling/event、Q3は0。元数値gate・B/C bindingを維持する。

価格は既存C受入direct m0..3 X/Yの8軸を再利用する。実幅.2のm2/q2、m3/q4の保存4軸を軸別alpha+beta*q較正へ用い、m5/q16を見る前に係数と予測をcommitする。新規はm5 X/Y最大2 measured full controlled wrapper compileだけ。Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、保存native/lowering/control/scalar・作用gateを維持する。

価格モデルはこの固定幅・prefix・公式専用の比較用reference。holdoutはfitへ戻さず、符号付き差と相対差を保存する。新しい2%合否規則は採択しない。長q誤差保証・CI・正式G資格はUNKNOWN。直接価格・条件付きモデルを各軸で区別する。異なる幅/pへ係数を移さない。

§10.5では元438d6c9a4の17不足表、既存幅coverage、156価格表、旧trial/catalog、該当wrapperのprivate locator、現在の関連job、Bのp7 jobを照合する。既存同等値があれば再利用、未解決重複は該当分を保留する。Bのp7 pendingはDから再投入しない。SOURCE/入力/namespaceをcommit後に開始し、旧run/STOPを保全する。

p0..9は保存fragment順0..p-1を保持したB0I、p10はFull PFの同一configuration alias。T1全pのshift/scalarとT2 p7/8/9の42cellをpriority metadataに使い、Q3判定に代用しない。新prefixの実preparation成功を人工testだけで主張しない。任意正有理数幅は候補化できるが、同実幅で較正2点＋later holdoutが取れない幅は別の取得方策が必要であり、他幅のmodelを借用しない。

通常optimizerのseed_rules・shotsearch・decoder/Q3 workerは再実装しない。completion registryを通常入口へ渡し、自動生成された提案を最大新Q3=0で確認する。既存のFullPF探索保存結果をresumeし、Q3/shot最適化はB2に戻す。Cには標準C_RESULT_JOIN locatorと最小consumer差分を渡す。

入力/gate/SOURCE/保存失敗時は部分結果を保存し自動retryなし。今回の有限確認・公開後は追加候補やQ3を自動開始しない。
