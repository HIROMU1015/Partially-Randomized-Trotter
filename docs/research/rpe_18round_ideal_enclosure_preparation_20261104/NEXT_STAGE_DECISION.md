# 次段判断：Partial18roundのideal categorical包含へ

2026-11-04 JST。
**固定入力・最小adapter・限定runner・人工検証は揃った。公開SOURCEをユーザーが確認し、別途明示承認すればm0..m16の包含評価を実行できる。全18roundの合格は未評価である。**

対象はlinear H4/1 Å/STO-3G/保存legacy DF12、保存physical stateのnormalized ray、4電子sector70、Partial L_D3/δ=.02。
全保存K2、rは1/2/4/8/16/32の6群。r/K/N/α/β/εを変更せず、m17の既存包含・両軸十分条件を再利用する。

- [入力可用性](ROUND_INPUT_AVAILABILITY.md)：18行のμ0・N/α/ε、signal/model τの区別、全固定commit/path/hash。
- [科学SOURCE・実行範囲](ENCLOSURE_EXTENSION_PLAN.md)：96 bit・最大17 complexセル、5新order law、6群のstep共有、STOP・出力。
- [全体成功までの証拠と不足](RPE_GLOBAL_EVIDENCE_GAPS.md)：局所ideal mean、条件付きbranch/energy、実測定全体を区別。
- [人工テスト](test_round_extension.py)：新18件＋既存25件、計43件PASS。実H4はbyte/schema照合のみ、新数値計算0。

入力不足はないが、他5群のorder law/CDF/θは未保存であり、承認後に新SOURCE replayとして生成する。
過去payload同一性は保証しない。model側とsignal側の保存τの違いを解消したことにもせず、包含lawにはsignal側を固定する。

最終energy保証には、正の半径と共通E0 phaseへの保証付き接続、物理state/PF固有stateの区別、strict branch条件とfloat64 decoder丸め、外部alias windowと境界margin、保存αのexact和、真のenergy referenceとの誤差が残る。
V2の保存中心点のfloat64二項確率を新ν_catの保証へ移送しない。
元float64仮想測定のguard系列は保留し、compile/noise/state preparation/実PRNGの保証も未完了のまま。

次の担当と範囲:

1. **ユーザー**：このbranchの公開SOURCE・[固定限定計画](ENCLOSURE_EXTENSION_PLAN.md)を確認し、m0..m16の96-bit包含（m17再利用、5law replay、sampling/compileなし）を別途明示承認するか判断する。
2. **Codex（承認後のみ）**：同じSOURCE/入力を照合し、実行hostの資源を確認して1 passを実行。各セル未判定・中断・不一致は部分結果を保存してSTOP。N/条件/精度を自動変更しない。
3. **Codex（別の後続技術指示後）**：局所結果からbranch/energyへの最小幾何・alias/reference・α和証明のSOURCEを準備する。その新科学再解析はこの包含実行許可に含めない。

この準備で`RPE_tolerance_certified=false`、`scientific_status=UNDETERMINED`、`U_virtual_IID_upper=null`、`U_exec_upper=null`を保持した。
旧CONDITIONAL_SUCCESS、SOURCE/raw/grant/ledger/marker/STOPは無変更。他branch・SERVER2 H8・共有環境には干渉していない。

**専用branchへの公開・remote SHA／公開blob照合後、mandatory STOP。今は科学実行しない。**
