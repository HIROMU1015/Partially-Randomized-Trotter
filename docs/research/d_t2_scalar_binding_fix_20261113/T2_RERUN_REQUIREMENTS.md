# 技術修正後のT2条件（科学実行未承認）

今回許可されたscalar request接続の修正と人工・metadata回帰は完了した。正しいexpected totalが保存input sourceから生成され、既存診断とhex一致した。これはT2の実preparation gate通過・巨大回路実行可能性を新たに確認した結果ではない。

後続で必要なもの：

1. **ユーザーの別の科学実行承認**。対象candidate/round/axes/path数・cohort/index/seed規則・最大compile数・SOURCE・出力先を有限指定する。旧T2の未使用compile枠を現在の技術許可へ移さない。
2. **新しいSOURCE/request/出力先**。修正済み`request_for`で正しいtotal/sourceを束縛し、旧run01/PLAN/STOPを上書きしない。旧one-shot callerをそのまま再実行せず、必要な小差分だけを新scopeに固定する。
3. **元の科学条件**。同H/state/snapshot/DF10/prefix2、S2/ΔPF=.1/K2/h=.1/M12/元exactκ、q4096/R2559/T409.6、元fragment順・scalar一回、P0、新ZZ/元compiler/full measured X/Yを保持する。値やSOURCEが合わなければSTOP。
4. **runtimeの実照合**。復元H/preparationのλ・scalar・hash・fragment順を元のbinary64 gateで確認。失敗時は同一呼出しの値を記録し、guard緩和・再seed・別path/roundへの変更を行わない。

未取得の証拠は、修正SOURCEでの実context成功、m12 full native生成・hash・X/Y compileの実行可能性、wall/RSS、同path C/S/Dとfingerprint/6指標である。前回はsampling前に停止しており、巨大回路の性能実績はない。科学承認が来た場合だけ、必要な一path等の有限T2を実行する。

T2が成功しても一pathの結果は適格E[C]/CIではない。ideal/sampler law差、rare寄与、有限標本・model/systematic、X/Y共分散と同時区間、全roundのN/G会計は別の課題として残る。今回これらの値・精度・n・confidenceを採択せず、fit/外挿や他roundへ自動進行しない。

**ユーザー**が次の有限scopeを判断し、**Codex D**は明示承認後のみ科学取得を担当する。今回の修正PASSを実行許可にしない。**mandatory STOP。**
