# Scope・seed・比較条件の静的監査

16 cell=4 arm×2 layer×2 axis。3phaseはcorrectness/baseline、profile/baseline、profile/instrumented。各条件でcorrectness16×16=256、baseline16n、instrumented16n、計256+32n。S/P1二条件では512+64n。

|Profile n|一条件|二条件|技術上の位置付け|
|---|---:|---:|---|
|64|2,304|4,608|短い入口案。setup/分解能/coverage不足の可能性|
|256|8,448|16,896|固定prefixの複数blockを記述する候補|
|1,024|33,024|66,048|長いprefix案。科学的十分性は保証しない|

独立global planでoffset/arm/layer/axis/phaseの算術を照合。二条件のglobal identityにはconditionとbundle digestの境界が必要であり、既存bundle-local trial IDだけを二条件で再利用して一意としない。現在SOURCEは一条件のfake projectionだけをadmitする。16,896の二条件実科学one-shot bridgeは未実装。公開正常caseはprofile16、768 fake callsであり、16,896を実行した証拠ではない。

Proposalは同じ入力・precision・fused codec・CachedObserver・F01/trace/保存・passをS/P1へ使う意図。schedulerのSはworkers1、P1は8 pair並列。proposalのworkers8だけからS実配置を確定しない。新しい逐次S projection768 fake callsも独立再読で通ったが、二条件を一つのglobal予算/claimで結ぶ実装の証明ではない。

SHA256 counterはnamespace/arm/axisをdomainとし、A/B、correctness/profile、両pass、S/P1は同じprefixを比較する案。X/Y・armは別domain。実行identityはlayer/phase/pass/conditionを区別するが、その全項目をseedへ入れる案ではない。IID_claim=false。proposal新namespaceは未採択の宣言である。人工fixtureのARTIFICIAL_R1_PARALLEL seedを、将来の科学namespace実行済み証拠としない。

correctness16/cellは保持案。accepted不足やraw-zero/coin-reject/accepted等の未出現pathを未検証として残す。coverageのためlaw/seed/bitsを変更したり無承認でcallsを追加したりしない。

n256はCPU母平均や量子confidenceの十分Nではない。同一prefixのblock分割は独立反復ではない。clock分解能・計測負担、setup支配、整数bit長/GC/cache/IOの変化、S→P1順序交絡を記述する必要がある。全N外挿は別の仮定であり、旧31,710,848契約のN・grantは変更していない。
