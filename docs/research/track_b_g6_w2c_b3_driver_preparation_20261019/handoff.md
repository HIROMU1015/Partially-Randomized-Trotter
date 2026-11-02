# Decision Handoff：Q1–Q5

技術状態：**TECHNICAL_DRIVER_PREPARED_ADOPTION_PENDING**。実科学実行0、mandatory STOP。

**Q1：driverは完成・人工tests合格か。**
実行engine、lazy G7/F3接続、source/input/N/axis guard、共通監査、角度再query、
新しい所有childへの分離、排他的CPU/wall、stream保存・hash・prefix/STOPを実装した。
新36＋既存人工48件がPASS。mock全schedule800 trialの接続検査を含む。
これは人工endpointだけであり、実G7成立性／correctnessはNOT_ESTABLISHED。
外部の正式承認portはinterfaceのみで、実authority発行・実claim消費は行っていない。

**Q2：固定N・4方式・A/B・X/Yの公正条件は閉じたか。**
保存N・16 cell・呼出数・X/Y state共有・cache・全trial分母・保存課金の具体仕様は候補として閉じた。
独立childでarm間HWM混入を避け、最小observer込みbaselineと同一traceのinstrumentedを分ける。
一方、τ/Kφ/J・seed・保持/2pass scheduleの正式採択と実入力検査が残るので、
**採択済み・実入力検証済みの比較として完全には閉じていない**。未取得canonical overheadを0にはしない。

**Q3：何がユーザー採択待ちか。**
数学的angle enclosureだけを測るscopeと具体τ/Kφ/J、新namespaceのSHA-counter固定bitstream、
cache無効＋X/Y保持state共有、correctness16＋全N二pass、compact hash保持と完全prefix保存、
output/flush条件、公開payload完全SHA、新一回admissionのbindingが残る。
代替の物理RZ代表値利用は新error整合条件を要し、現driverは拒否する。C1 provider選択は求めない。

**Q4：何を明示承認すれば実行できるか。**
ユーザーが上記未決値を採択し、公開payload／正式契約digest／新claimを固定した上で、
「初回correctness256 sampleと、4方式A/B・X/Yの保存全N baseline＋instrumented、
合計31,710,848 sample、acceptedごとのB angle/requery、constructor24、retry0/resumeなし／mandatory STOP」
の一束one-shotを別途明示承認する。Codexはその承認後に正式契約と外部admissionを結び、
初回correctnessから実施する。新precision条件やphysical保証を無断で追加しない。

**Q5：実行後に何が分かり、何が残るか。**
当該固定workload・環境・scopeのprocess CPU、wall、setup/online/save、R/C/A経路、
実call数、RSS/HWM/行・output bytes、observer付き／なしの差を記述できる。
W2c-Aの式へ観測全trial平均とsetup/output差を入れ、当該workloadの費用関係を述べられる。
母期待μの認証、一般入力計算量、正確bit演算数、一般速度倍率、物理CQ/native全費用、
state/readout/backend込みEnd-to-End、研究GO/NO-GOは判定しない。

次の担当は**ユーザー**：候補とτ/Kφ/J等を採択し、実行する場合だけ別途one-shot承認。
その後の担当は**Codex**：正式条件固定・外部admission準備・承認bundle実行・保存公開・STOP。
現時点では次scienceを開始せずmandatory STOP。GPT重要科学レビューも開始していない。
