# Decision Handoff：Q1–Q7と次の承認

状態：TECHNICAL_ADOPTION_PREPARATION_COMPLETE_PENDING_USER_DECISIONS。
実science／authorization／claim発行・消費0。既存driver・SOURCE・契約を変更せずmandatory STOP。

**Q1：具体Angle候補はどこまで出せたか。**
A6=(τ10^-6 rad,Kφ32,J32)、A9=(10^-9,48,64)、A12=(10^-12,64,96)。
既存F3のroot/半角/alternating remainderを保守的に代入し、全r∈[0,2]に対する十分なradius上界を示した。
実event探索・精度最適化・F3/G7数値実行はしていない。R1はA6を提案し、まだ採択していない。

**Q2：旧error/confidenceのまま数学的angleを閉じられるか。**
既存SOURCEのdomain/postconditionの下で閉じられる。代表値は物理RZへ渡さず、新error allocationは不要。
旧H/K_root/η/ρ/ε_axis/primitive error/α/Nは不変。実G7 law/context/preflight/trace成立は未確認で、
別途承認された初回correctnessで確認する。runtime failureはSTOPでありprecision retryしない。

**Q3：技術的確定と採択待ちは何か。**
旧seed文字列規則、同一prefix再生、X/Y識別、24 constructor/3 prepare/9 canonical preflight、
X/Y保持state共有、memoなし、元guard/requery、二pass、compact保持と64GiB/256MiB上限の実装を確認した。
それらを新しい実費workloadとして正式に使う採択は残る。R1に一式をまとめ、代替cacheや保存方式を増やしていない。

**Q4：Admission/STOPの人工検証と未実装部分は何か。**
新30検査でbinding不一致、未採択／未承認、旧claim、同scopeの32 thread／32 owned-process raceで各1winner、partial reserve/trial、
保存失敗、output衝突、STOP後retry/resume拒否を確認。既存84人工検査も別集計する。
本番external port、実authorization照合、live scope/claim store、本番process/reboot耐久性は未実装・未検証。
mockは本番driverのadmit_onceを持たず、本番で使えない。人工費用を実CPU値として報告しない。

**Q5：次の承認文はどう書くか。**
次の2段階を混ぜない。

1. ユーザーの採択・本番admission技術準備承認案：
   「公開された本候補のpackage R1（math-only、τ=10^-6 rad、Kφ=32、J=32、既存seed/setup/cache/保持/二pass）を採択する。
   科学実行をせず、文書のbinding/atomic/STOP仕様に従う本番admissionを実装・人工検証し、SOURCEと正式契約を公開固定することを承認する。
   この段階では実authorization/claim発行・消費と科学実行は承認しない。」
2. 上記実装freeze後、ユーザーの最後の科学実行承認案：
   「完全SOURCE commit `<PRODUCTION_RELEASE_SHA>`、正式契約digest `<FORMAL_CONTRACT_SHA256>`、
   実在承認参照 `<AUTHORIZATION_REFERENCE>` と新one-shot scopeを結び、必要なfresh authorization/claimの発行・一回消費と、
   correctness256＋全N baseline15,855,296＋instrumented15,855,296、総31,710,848 sampling calls
   （correctnessを含む）、accepted毎のB angle/requery、constructor24/preflight/保存を一束one-shotで承認する。
   initial correctness不合格・技術失敗はprefix保全しSTOP、retry0/resumeなし、完了後もmandatory STOP。」

後者のplaceholderは現時点では存在しない値であり、有効な実行承認文として使わない。
実装・公開・技術照合が済んだらCodexが実値を埋めて提示し、それを最後にユーザーが承認する。

**Q6：科学計算なしで解決できる技術課題は残るか。**
今回許可された候補具体化・静的照合・人工検証は完了。本番port実装/接続と採用storageでの人工crash/process検査、
正式契約freezeは科学計算なしで可能だが、今回は明示的に未承認のため実施していない。
次の担当はCodexで、上の技術準備承認後にのみ行う。実入力correctness/実費はその後の科学実行承認が必要。

**Q7：ユーザーが選ぶ最小件数は何か。**
実行条件の**内容の採択は1件**（R1一括。別profileを選びたい場合だけ指定）。
これと別に、SOURCE/正式契約freeze後の**科学実行承認が1件**必要。
本番admissionの準備承認は採択時に併記でき、未実装SOURCEを見ずに科学実行を先行承認する形にはしない。

次の担当：ユーザーがpackage採択と必要な本番技術準備を承認するか判断。
承認後の担当：Codexが本番portを準備・freezeし、最後の実行承認用資料を提示。
現在は追加science・科学レビュー・正式採択・実authority発行へ進まずmandatory STOP。
