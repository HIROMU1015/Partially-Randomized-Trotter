# 未解決事項と主張への影響

|ID|事項|制約の種類|固定研究への影響|より強い主張に必要な証拠案|
|---|---|---|---|---|
|U01|H4 legacy DF12 / H6 tol-only DF19|FIXED_CROSS_SYSTEM_INTERPRETATION|各系内の固定比較は保持。単純な系サイズscalingと呼ばない。|同じDF生成契約での比較が必要なら別研究設計。|
|U02|H4 PR n32 / H6 PR n8 の有限標本|FIXED_INFERENCE_LIMIT|保存標本に条件付ける。点順位を母平均順位としない。|独立標本や統計設計が必要。今回は実施しない。|
|U03|H6別研究の確認n32を混合していない|DATASET_SEPARATION|現在のpanelの標本は探索n8のまま。欠測補完に別研究を使わない。|統合する場合は入力・比較条件・標本phaseを再定義する別承認。|
|U04|PR母平均保証・未観測希少RTE費用上界なし|FIXED_INFERENCE_LIMIT|期待理論値と有限標本compiled平均を区別。現panelの条件付き観測は記述可能。|全supportの費用上界又は事前登録された統計的証拠。追加samplingだけで厳密上界としない。|
|U05|経験的bias/allowance・保存shot|FIXED_COMPARISON_CONDITION|保存shot条件付き費用。厳密なsignal精度保証・shot最適性を主張しない。|厳密bias/allowance又は独立数値検証の別設計。|
|U06|固定7候補外の最適性未検証|STRONGER_GENERAL_CLAIM_ONLY|指定pairの比較は記述可能。全候補最良の宣言は不可。|候補集合の事前固定・公平な探索と確認が必要。|
|U07|理論countとcompiled countの残差、lowering/opt1未分離|FIXED_ATTRIBUTION_LIMIT|残差は観測事実。opt1単独の因果効果へ帰属しない。|同一保存回路・seedでopt0/opt1とpass記録を比較する限定compile案。件数・grantは未採択。|
|U08|旧compiler出力のbinary完全再現未証明|STRONGER_REPRODUCTION_CLAIM_ONLY|保存compiler hash・profileへ結合した値。新hostでのbit完全再現や外部再現と呼ばない。|旧環境のbinary closureと固定wrapper出力を比較する別検証。|
|U09|全wrapper wordのbit完全同一性は未認定|STRONGER_EQUIVALENCE_CLAIM_ONLY|8/28 metricのcount一致をword全体の一致へ格上げしない。|固定wordのcanonical byte/angle/control/phase比較を別承認で設計。|
|U10|状態準備費用P=0|FIXED_OBJECTIVE_LIMIT|no-preparation full-Hadamard RZ/CX費用。状態準備込みやend-to-end優位性は未評価。|状態準備契約と費用入力が必要。今回P感度も再会計しない。|
|U11|backend/noise/routing/耐故障論理資源は未評価|STRONGER_PLATFORM_CLAIM_ONLY|rz/sx/x/cx、topologyなし。logical qubits/T-count/T-depth/surface codeを測定したとしない。|hardware又はfault-tolerant mappingの別契約。|
|U12|理論lower/upperはcompiled母平均の信頼区間や上界ではない|FIXED_MODEL_INTERPRETATION|元の区間を保持し、中点化・CI化しない。|保証を主張するならモデル誤差・sampling・compilerの別証明が必要。|
|U13|最新表に継承されたUNATTRIBUTED列|HISTORICAL_LABEL_INTERPRETATION|expectation表のattribution等をそのまま保持。S4 count差の解消は新acceptance/primitive証拠で別記する。|元ラベルを一括置換せず、count原因と残るcompiled残差を分けて記述。|

追加計算案は目的と必要証拠の候補だけ。正式計画・grant・実行は未承認。S4の8行count不一致自体は解消済みで、U07/U08/U09を未解決のまま維持する。
