# C7-3：高速化SOURCEと承認済み残taskの自動引継ぎ

2026-11-07 JST。利用者の「高速化を行って、切り替えまで行ってよい」という追加指示により、C7-3の技術SOURCE切替えと残task継続を実施する。新しい科学条件や標本数を追加する許可ではない。承認証明書・新grantは作らない。

## 保全と切替境界

旧SOURCEは `35bfd0423d08114b0a24107501ce545cdfb28f32`、旧PLANは[PLAN.md](PLAN.md)。q32/cut16/fresh8 path/256 occurrence/8192events、同H4/CDF8/LD2/S2/s1/r32/K0/h=.1、元H厳密ground仮定・状態準備除外、同law/seed/モデル/価格幅/familyを維持する。

旧controllerだけへSIGINTを送り新規投入を終了する。投入済みworkerにはsignalを送らず、実行中rawを完了までdrainする。旧実行SOURCE・HEAD・raw/結果・STOPを変更しない。旧controllerはKeyboardInterruptによるSTOP_FAILUREと、途中までの集約値を記録する場合がある。その記録を修正せず、controller終了時に保存された各pathの実raw PASSを別に確認する。科学的不一致を原因とする停止は引継ぎ対象外。

調整停止後、全8 raw PASSと旧compile0を実ファイルで確認した。保存済みevent、全raw QPY、conditions/fusion、参照NPZ、raw検査・構造特徴、旧SOURCE/seed/model/event hash索引・旧STOP等を[固定inherited_evidence][INPUT]へ封印する。payloadは非公開、公開はrelative path/bytes/hashのみ。

高速化SOURCE・人工結果・引継ぎhash索引を専用branchへcommit/pushし、tree/bytes照合後に新controllerを起動する。全8 raw PASS、科学failureなし、old SOURCE/input/位相条件、event/回路bytes、既存compile/postの完了状態を再確認してから未着手処理を開始する。新側の科学実行前後にこの同じbindingを照合する。

旧集約のraw件数やsource_input_preserved=falseを後から書き換えない。新側の実ファイル照合・再利用件数・新側前後保全結果と併記する。旧8 rawは再構築・再検証しない。既に始めたcompileがあれば、そのpostまで完了済みのものだけを再利用し、不完全な既着手処理へ無断retry/resumeしない。

## 科学実行の上限

|処理|旧新合算/今回の扱い|
|---|---|
|canonical sampling|既存256 occurrence/8192eventsをそのまま使用、追加0|
|raw correctness|既存8 pathの完了を待って再利用、新raw実行0|
|P0 full-wrapper compile|旧新合算最大32、未着手wrapperだけを各一回|
|postcheck|同じ32 wrapper、未着手検証だけ|
|単体A/B/shell compile・別q/別cut・ground solve/DF fit|0|
|モデル再fit・追加calibration・N/全RPE評価・詳細科学レビュー|0|

全8が揃わない場合に有利なcomplete-case集計を作らない。CX候補違反/価格幅外は有効結果として、correctness等が成立する限り固定の診断を完了する。

## 高速化の内容と再利用

1. **局所行列のexact cache**。既存simulation_circuitの「3qubit以下は公式Qiskit Operator、より大きいcompositeは公式definitionを辿り宣言位相を保持する」変換を再利用する。標準gateの実class/control state/角度のbinary64 bytesをkeyにし、同じ局所行列と検証済みUnitaryGateをsimulation用に共有する。零/微小角・符号を丸めず、arbitrary gateはcacheせず従来どおり変換する。cost circuitを変更しない。
2. **小さい全演算子のnative評価**。従来のQiskit circuit Operator評価を、9qubit以下に限り公式AerSimulatorのdouble/unitary methodへ切り替える。全行列要素を保持し、宣言global phaseを明示的に除去/復元する。位相fitやprobeへのcoverage削減ではない。17qubit全Operatorは禁止のままで、Referenceは既存double/statevectorの全256 input column検証を使う。
3. **既存BatchedStateActionsの再利用**。その入力batch/native executor/宣言位相規約は維持し、simulation変換だけをexact cache版に明示選択する。新numerical simulatorを実装せず、公式Aerを使う。新worker内だけで選択し、旧process/packageファイル/共有環境は変更しない。
4. **compile/postの別dispatch**。既存compile_oneとverify_oneをそのまま別taskとして呼ぶ。compileは元と同じ1 threadのP0、postは両方式ともnative thread budgetを使える。既存allocatorのmemory/CPU判断を再利用し、postの大きなpeak見積りをcompile段階へ一律に課さない。独立最大32 compileを並列投入し、完了したものからpostを重ねる。

compileの初期memory estimateは完了q16 direct工程peakをq長比2で拡大した値、Reference postは既存q16 Reference peakの同じ拡大値を使う。いずれも経験的admission見積りで厳密上界ではなく、実MemAvailable/cgroup/CPUを確認する。歴史的max-worker数を恒常上限にせず、未着手task数と利用可能資源で決める。CPU使用を妨げる共有設定を変更しない。

P0はQiskit1.3.0/opt1/seed17/rz,sx,x,cx/approximation1/initially_zero=false/topology-free、既存MPFR160宣言位相保持の同じ関数。raw/compiled cost QPYやgate列は高速化のために書き換えない。価格は元のP0出力から取得する。小演算子検証のbackend method選択は上記のとおり変わるが、binary64・全行列coverage・component1e−10/observable2e−10/rtol0・phase fitなしは維持する。

## 人工検証とSTOP

人工3/5/9qubitの標準/符号付き/零/微小角/controlled/composite位相・actual inverseで、既存Operatorと新full matrixを全要素比較する。control state/cache keyの衝突拒否、arbitrary同名gate非共有、input不変、既存Reference state-actionとの一致、17qubit全Operator拒否を確認する。9qubit人工例で高速化を測定したが、その倍率を実H4全体やP0 compileのspeedupとはしない。

固定196608命令の9qubit人工identityでも、元のcomponent1e−10/Frobenius2e−10を検査した。新規人工11件と旧回帰50件、合計61件がPASS。

停止済みtaskの再利用、既着手不完全taskの拒否、旧STOP保全、compile一回、raw PASS依存、32compile/32postのmock dispatchも検査する。人工PASSを実H4性能や全sampling lawの保証にはしない。

新コード・人工結果・固定input bindingを科学実行前にcommitで固定する。旧SOURCE/結果を改変せず、new source hashとold/new生成来歴を保存する。数値検証失敗時に従来法へ自動fallbackして同じ検証をやり直すこともせず、partialを残してSTOPする。threshold/seed/cut/モデル/標本数は変更しない。

価格recordは旧event source/旧raw request fingerprintと、新compile/post sourceを分けて記録する。execution_sourceを新compile側へ束縛する場合も、raw request自体とそのhashを併記し、元event recordを変更しない。科学的なpath・law・compiler条件の相違をSOURCEの違いで隠さない。

最終はraw8再利用、旧新合算compile/post32、paired8、SOURCE/input前後照合を確認して集計する。raw event/QPY/NPZ/参照配列・native metadata/CPU/RAM/絶対pathはlocal保全。公開は科学scalar/価格/共分散/CX残差/構造feature/hash索引/旧STOPと新STOP/報告書。成果公開・remote SHA/tree/bytes照合後mandatory STOP。q≥64/新fit/確認的保証/全RPE/actual N/詳細科学レビューには進まず、判断をユーザー/GPTへ戻す。

[INPUT]: ../../../artifacts/track_d_c7_3_q32/2026-11-07/accelerated_technical/inherited_evidence.json
