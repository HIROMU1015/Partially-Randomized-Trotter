# Track D O5 C1：保存実H4・q4限定移送PLAN

2026-11-08 JST。利用者が送付した実行指示の範囲を一回だけ実施する。基点は人工結果 `14276fe0e2dadef54ef105debe6d6d99b43b6ecb`、C6a実行SOURCEは `6b6667026ec8b670f610e41141d195883a9eb513`。以前の原本回収・再利用記録 `2f316fb229149117177367720a6eaa5aa8a94b76` とそのSTOPを保全する。承認証明・grantを新設しない。

## 固定科学条件と境界

Linear H4／隣接1.00 Å／STO-3G／Nα=Nβ=2／元H／Number-CDF8／元frame順L_D2／ordered partial S2/s1/r32/K0／h=δ=.1。保存C6a `q4_sample00` 一件、4 complete steps各32 events、complete step2後cut。元H厳密groundを利用できると仮定し、準備費を除外する。ground solve・DF fit・新sampling・別path・別q・cut探索・モデルfit・N算定・全13-round再会計は0。

全input identity・compiler・数値条件・科学call上限は [PLAN.json](PLAN.json) に固定する。SOURCEをcommitし、実行file hashとともにevent decode前に記録する。出力は `artifacts/track_d_head_tail_h4/2026-11-08/run01/`、既存directoryの上書き・retry・resumeを認めない。

変更はhead出口G_H→tail最初のfull-frame入口G_T†のみ。正しいnative span、actual rotation／gauge、8 system wire順0..7、ordinaryではancilla8・physicalでは8 wireだけを確認し、既存StrongBuilder.transitionのG_T†G_Hへ置換する。除去するbasisの宣言phaseを差し引いて橋のphaseを一回composeする。Basis ID・近似rotationによる相殺はしない。tail出口→head入口、内部異frame run、step/global cut、CSWAP越し、control中心には新規則を加えない。

新module `track_d_head_tail_h4.py` は2-mode専用guardを置き換える明示opt-in入口。旧人工adapter・C6a/StrongBuilder/provider/gadget/compiler/checkerのdefaultは無変更。Strongの3 step境界融合・head relative/tail same-basis共有・scalar集約、Referenceの内部2境界融合／global cut損失1／actual A†／χ／二層CSWAPを維持する。

## 検証とcall上限

1. 既存archiveと全20 member、events、H packet、CDF endpoint、sector・signed atoms・lawのhashを照合する。原本はread-only、private copyだけを作る。S0/R0の旧raw全256入力PASS・4 P0/postcheck・価格・wire/axis/SOURCE/profileを資格付ける。不適格なら追加baseline compileをせずSTOP。
2. 既存C3.prepare_stepsとsaved-event readerでOFF core/halfのcanonical IRを再構成し、宣言phase込みで保存QPYと完全照合する。OFF wrapper callbackは照合済み保存wrapperを返す。再構成用のidentity target placeholderは計算・比較に一切使わず、保存CARのVとordinary_rawをbindingして返す。新しいOFF wrapper生成・compileは0。
3. 各stepのhead→tailをS1/R1双方へ同じ規則で適用する。新ON coreは各1、新wrapperは合計4。Reference A†は融合後A.inverse()。局所8-mode橋の全Fock位相付きunitaryと独立exterior liftを検査する。拒否・不一致は保存して依存処理STOP、別pathへ移らない。
4. 既存C3/C6a checkerを使い全256 system入力／全reference出力を検査する。V、S1=S0、両Strong branch、W0†W1=V、AB=V、actual inverse、half vacuum、χ、scalar/identity・signed events、full coherence・X/Y observable・leakage・bit規約を確認する。component abs1e−10、observable Frobenius/signal2e−10、rtol0、phase fitなし。
5. raw全PASS後だけS1/R1×X/Yの最大4新P0。Qiskit1.3.0／opt1／seed17／rz,sx,x,cx／approx1／initially_zero=False／topology-free、既存MPFR160 phase保持。OFFと同一IRの場合は旧compiled再利用を優先する。新postcheck最大4。新構成に旧postcheckを流用しない。
6. 4構成×X/Yの全7資源をfull-wrapperで比較する。ΔS、ΔR、前後R−S、差の差を保存し、Strong自身／Reference自身／Reference相対差の効果を分ける。局所削減の足し算をfull-wrapper資源に代用しない。

数値処理には既存C7-3のofficial Aer small-unitary/exact local cache・BatchedStateActionsと既存nofusion adapterを実行前から選択する。これはsimulationだけの選択でP0を変更しない。全入力・全出力を保持し、17-qubit Operatorを構築しない。事後FAILを救う設定切替や閾値緩和はしない。

空きCPU/RAMを既存観測関数で確認し、4独立compile/postをprocess並列化する。BLAS重複を抑え、native state-actionへ観測budgetを配分する。固定の低いCPU/RAM上限を追加せず、共有scheduler/venv/cache・他job/process/affinity/fileは変更しない。q64の進捗・未完了出力を照会しない。

## 公開・mandatory STOP

原本events/QPY/NPZ/private入力・運用情報は非公開。公開は科学結果、全7資源、correctness要約、SOURCE、hash索引、報告書とSTOPだけ。改善・0・悪化・不成立のいずれも保存し、専用branch `codex/track-d-o5-c1-h4-q4-20261108` へcommit/push、remote SHA／tree／公開bytesを照合する。その後mandatory STOP。C1追加探索、O1/O2/O3実装、C2、全RPEや詳細科学レビューは開始しない。次の判断はユーザー／GPTへ戻す。
