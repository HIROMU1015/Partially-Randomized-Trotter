# N4-B：共通policyとwhole consumerの接続

2026-11-08 JST。[REPORT](REPORT.md)。実装は[pr_reference_common_word_consumer.py](../../../src/trottertracks/pr_reference_common_word_consumer.py)、固定入力は[N4-A targets](../../../artifacts/pr_reference_n4a_coverage/2026-11-08/INTEGRATION_TARGETS.json)。

## 1. 選択・emissionの層

1. 実際のcompiler後RZパラメータをbinary64のまま読む。
2. 既存DictionaryProvider.provide(theta)を使う。abs(theta).hex()の固定C0が優先、missは固定43-word辞書、域外は拒否。
3. Complete positive_sequenceをsequence-only carrierへ置く。このcarrierはbackend record／新guardの偽装ではない。
4. 旧lower_native(segment, allocation, carriers, common_native)を呼ぶ。common_nativeは旧インターフェースの固定labelで、word選択へのrole条件ではない。
5. 負RZは旧lower_nativeのlocal.inverse()がcomplete wordを1回だけ反転する。signed_sequenceを先に渡して二重反転しない。

Emitter、provider、C0／基本recordを変更せず、arm引数を新設していない。Wは価格0でも捨てず、gate列とglobal scalarを共に保持する。

SOURCE inverseとword inverseは別の層である。保存basis_inverseはactual SOURCE inverseの実native loweringを使用し、forward loweringの単純inverseへ置き換えていない。

## 2. Shellの不変条件

Outerは元のqubit/clbit object列から新規作成し、同じregisterを登録する。元numeric global_phaseを1回設定する。Native命令を順に読み、rz/sx/x/cxをphase0のsegmentへappendする。Reset／measure／barrierに達したらsegmentを旧lower_nativeでemitし、outerへinplace composeしてから元の非unitary命令を同じwireへappendする。

この帰納的処理により、各fence以前のunitary命令がすべて先に出力され、以後の命令はその後に出力される。Gate展開で命令indexは変わるが、reset/readoutをunitaryの前後へ移動しない。Clbit対応とbarrierのwire順も保持する。Unitary命令の期待展開全文と実出力の一致を局所全件で検査した。

OutputはH/S/Sdg/T/Tdg/X/CXと元fenceからなる。Native SXは旧lower_nativeのHSH。新control／compiler pass／Target／routing／phase fitを加えない。条件付き命令、free parameter、native外gateを拒否する。旧SOURCEへこの出力を戻してcompileする経路はない。

## 3. 検証対象

| 固定入力 | 実施内容 |
|---|---|
| N4-A保存85 key | positive／negative certificate、同一word、hash、W、strict上界、role対応 |
| 指定frame／b2、±0 | 実emitter、actual adjointのexact algebra、Operator診断、命令列／位相参照 |
| 両armの保存basis／basis_inverse | 全local命令をconsumerで実emit。元literalと順序を保持。8-qubit gate productを既存compressed kernelで診断 |
| 両armのp_symbolic | 保存RZ(symbolic_correction) templateを指定2 key±と±0で局所bind。新PF build/compileではない |
| CT-4B保存8 native QPY | hash、実phase、全RZ literal多重度、alphabetを旧metadataと照合。先頭4＋末尾4＋全非unitaryの固定局所投影だけをconsumerへ通す |
| 技術shell fixture | 途中reset、barrier、非連続clbit測定先、符号、tau境界、複数unitary segment |

保存whole本体をCTへ全展開していない。Localで確認したconsumerのループ不変条件とword証明をSOURCE閉包へ条件付きで移送するのであって、8件の有限PASSを全support実行証拠と呼ばない。

## 4. Phase欠損とQPY

Local catalogueにSOURCE/global phase metadataがないことはそのまま保存した。Local検査対象は書かれたnative gate productで、同じ任意の未変更prefactorを両側に置く。Diagnostic用のphase0 conventionを欠損SOURCE値の推定に使用しない。

Wholeの元phaseは保存CT-4B metadataと入力QPYから取得した。Float payloadのhashと読込後hexが一致し、全8 phaseは0<=phi<tauである。Qiskitのconstructor/setterのrem_euclidはこの範囲で恒等なので、この固定入力ではphase bitが変化しない。QPY出力／再往復／一般のtau境界入力の資格は付加しない。

Consumerの内部ではQPY、to_instruction、default copying compose、whole inverse、新しいcompilerを使わない。負RZだけに旧lower_native内のactual inverseを用いる。[位相操作と上界](PHASE_ACCOUNTING.md)を別会計とした。

## 5. 保証する範囲

全85のrole wordと共通policy、固定runtime下の局所consumer／shell、保存contextのliteral一致は資格済み。Compiler条件CとC_phiを維持するSOURCE閉包に限り、全必要roleがC0または固定辞書域に入るという既存静的議論と接続可能となった。Vacuum reachable keyを新列挙したわけではない。

全位置のjoint synthesis disk、既存compiler/vacuum diskとの合算、199-shot成功条件の新しい数値検算は未実行。**FULL_SUPPORT_CT_RPE_UNQUALIFIED**。[次scope](NEXT_EXECUTION_SCOPE.md)は提案のみで、今回の終了後はmandatory STOP。
