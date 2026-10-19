# H6 full m5_best：compiler位相伝播の追加調査と修正候補

2026-10-19。**固定runtimeのBasisTranslatorで、parameter付き等価則の定数global phaseが落ちる最小人工例を再現した。問題の経路を避ける明示native→basis変換は、小さい人工回路のraw演算比較にPASSした。実H6の不一致が解消したという結果ではない。** 実分子QPY/NPZ復元・新compile・新再生・新grantは0。旧STOPと58受理軸を保持し、local commit後mandatory STOP。pushは行わない。

人工prototype/evidence SOURCE：`9fe4981ac21da17888c7f5dcadc094257c282e4e`。prototypeの最初の固定commit：`662b8464`。起点結果：`3f0ce05668b1f85249b45269ba087b6e2edf7b55`。branch：`track-a-h6-m5-phase-repair-design-20261019`。

[保存位相台帳の結果](track_a_h6_m5_phase_ledger_run01_results_20261019.md)、[前段の公式source調査](track_a_h6_m5_phase_ledger_static_20261019.md)、[今回のbundle](../../artifacts/resource_applicability/track_a_h6_m5_phase_repair_design/2026-10-19/summary.json)。現在の承認は追加調査・修正計画と人工検証の範囲であり、新しい実H6計算への承認ではない。

## 1. 確定したことと残る因果関係

対象とする旧不一致はH6直鎖・隣接1 Å/STO-3G/tol-only1e-8 DF19、T=.8/P=0、full m5_best/p19/q2、symmetric-directional/full Hadamard cosine、保存low初期状態1つ。m5のIDは`PF_M5_BEST_20D64C2_BINARY64_V1`。旧raw差は約1.504e-9、保存角度のdeterminant台帳差は約1.486e-9 radだった。これらは既存結果の転記で、今回の再解析値ではない。

自前nativeは各signed pieceのscalarを0にし、外側scalarを一度だけ追加する。保存構造にもancilla Pは一度だけで、scalar二重挿入を示す証拠はない。固定Qiskitの位相処理を人工回路で分離したところ、binary64加算誤差だけとは別の、定数phase伝播の欠落を確認した。

ただし、**今回の人工例のπ/2差と、旧H6の約1.5e-9差は同一の現象だと認定していない。** 旧H6にはpassごとのsnapshotや実際に選ばれた等価経路がない。保存XXPlusYY数116,160は4の倍数だが、それだけで実runの位相の相殺や微小残差の発生原因を確定できない。旧58軸の影響を新たに判定・再会計することもしていない。

## 2. 最小人工再現とsourceの該当分岐

既存の固定Python3.12.3/Qiskit1.3.0/NumPy1.26.4を使った。共有SessionEquivalenceLibraryは変更せず、process内に独立したEquivalenceLibraryを作成した。人工parameter付き1qubit gateの等価則を`exp(iγ) RZ(θ)`とし、θ=.2でBasisTranslatorを直接呼んだ。

|等価則のγ|期待global phase|出力global phase|3qubit全演算のraw差|
|---|---:|---:|---:|
|定数の数値0.123|0.123|0|0.34767727253431857|
|parameter式θ/2|0.1|0.1|0|
|0|0|0|0|

この固定runtimeでは定数の数値phaseだけが伝播していない。parameter式の例とゼロの例は対照として一致した。[人工証拠JSON](../../artifacts/resource_applicability/track_a_h6_m5_phase_repair_design/2026-10-19/artificial_evidence.json)。

既取得の公式1.3.0 release commit `896d1c8b72888a52a46081a6d1f4de8c56db8aea`でも、この挙動に対応する分岐を確認した。

- [replace_nodeの609行](https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/crates/accelerate/src/basis/basis_translator/mod.rs#L609)：parameterなしの分岐は665行でtargetのglobal phaseを加算する。
- [770行](https://github.com/Qiskit/qiskit/blob/896d1c8b72888a52a46081a6d1f4de8c56db8aea/crates/accelerate/src/basis/basis_translator/mod.rs#L770)：parameterありの分岐は`ParameterExpression`の場合にだけ801行の加算へ進む。数値の`Param::Float`を加える分岐がない。

公式sourceの該当SHAは以前の取得manifestと一致した。installed binaryをこのsourceから再buildしたattestationはないため、sourceの欠落分岐とinstalled runtimeの人工再現を別の証拠として記録する。共有のQiskit・Rust binary・Python moduleは変更していない。以前の一般的な「basis translatorが位相を伝播する」という静的説明は、この型と分岐の制約を付けて読む必要がある。

## 3. 修正候補と人工比較

[prototype](../../scripts/m5_phase_repair_prototype.py)は既存native/compile経路に未接続で、既定の呼出しは`ARTIFICIAL_ONLY`で拒否する。このflagは誤使用を避けるためのもので、科学実行の認可機構の代わりにはならない。

### Pだけの置換案は単独修正として不採用

`P(θ)=exp(iθ/2)RZ(θ)`により、各PをRZに換え、scalarは元binary64値を正確な有理数としてまとめて一度だけ与える。異なるqubit上のPは削除せず、gateの順序・位置・角度を保つ。この変換自体は人工回路で演算・制御付き演算が一致したが、残るXXPlusYYの等価変換経路でraw差が出た。そこで、この失敗を検出する回帰例を残し、P-onlyをH6の修正として採用しない。

### 明示native→basis変換の候補

XXPlusYYの既存定義を展開し、RYは`S H RZ(θ) H Sdg`（行列の積順）として表す。H/S/Sdg/SXdg/RZZも固定の式でRZ/SX/X/CXへ展開する。PとClifford由来のscalarは正確な有理数で蓄積し、最後に一度だけbinary64へ丸める。高精度moduloには数学的πを使うが、展開に使う固定Clifford角度はbinary64の`math.pi`であり、厳密なClifford誤差certificateではない。

これにより、parameter付きR/RY等価則の定数phaseが落ちる経路を回避する。opt1内の1qubit再合成やphase加算は残るため、実H6で1e-9未満になる保証はない。gate角度のfit、保存差に合わせたbranch選択、compile後の位相補正は行わない。

人工回路は3qubit、H/P/RZZ/XXPlusYY/CX/SXを含む固定例で、実分子データから作っていない。

|人工経路|元回路に対するcompile後のraw全演算差|人工1e-12基準|
|---|---:|---|
|元の等価変換経路|4.0|FAIL|
|P-only|4.000000000000002|FAIL|
|明示native→basis候補|2.5768813629570786e-15|PASS|

明示候補のcompile前の差は2.162286713063666e-15、制御付き全演算の差は2.399172112373377e-15だった。小さい人工回路では位相を含めて一致する。今回の人工例のcompiled gate countsは3経路で同じだったが、実H6の費用や回路サイズが同じになるとは推論しない。人工1e-12はtest専用であり、実H6の原raw基準1e-9未満を置換しない。

## 4. pass観測のprototypeと限界

callbackはDAGを変更せず、pass名・順番・時間・global phase・P角度の正確な総和・gate counts・topological metadata SHAをコピーして残す。小さい人工opt1では24 callbackと入力/出力の計26記録を得た。callbackあり/なしで出力の順序・parameter・global phaseが一致した。DAGやoperationへの参照を保持しない。

未知gate・classical conditionは完全なdeterminant証拠とせず、scalar欄を`null`にする。terminal measurementの判定は当該wireの後続を使い、他wireの独立gateとtopological順が入れ替わることを許容する。DAGのtopological SHAをQPYの保存命令順序hashと混同しない。観測器自体はgate matrixやdeterminant moduloを評価しない。

現prototypeは各callbackで走査するため、大回路での費用は未測定。実adapterでは、analysis passの再走査を避けるか、登録したtransformation境界だけを精密に観測する方針を固定してからsealする。未観測区間があれば、最初の原因passを確定したと表示しない。

## 5. 次の最小実験案：固定cosine一軸

[DRAFT計画](../../artifacts/resource_applicability/track_a_h6_m5_phase_repair_design/2026-10-19/pilot_plan_DRAFT.json)と[無効grant](../../artifacts/resource_applicability/track_a_h6_m5_phase_repair_design/2026-10-19/grant_INVALID.json)を保存した。**この段階では実分子用adapter/launcher・fresh資源見積り・実行SOURCE closureは未完成で、起動できる準備完了状態ではない。** 有効grantは発行していない。

同じ旧wrapper QPY・同じ保存初期low状態・同じ元wrapperの保存vectorを再利用し、H6 full m5/p19/q2 cosineだけを対象にする。新候補探索・signal・SCF/DF/state/reference生成・sampling・shot会計は不要。明示native→basis変換、新compile一回、変換後/compiledの作用各一回、計2再生を最大範囲とする。sineは含めない。

各errorはfiniteかつ**独立に1e-9未満**を要求する。

1. 変換後の作用対、元wrapperの保存vector。
2. compiledの作用対、変換後の作用。
3. compiledの作用対、元wrapperの保存vector。中間2検査だけで原end-to-end基準を置換しない。

QPY/結果は判定前に新しい専用領域へ保存し、FAILでも旧証拠を上書きしない。PASSしても一状態・一軸の技術pilotで、全状態同値や両軸費用の受理ではない。結果費用は診断用で会計へ接続しない。失敗なら即STOPし、auto retry・追加compile・phase alignment受入・閾値変更・再fitはしない。

Qiskit1.3.0・rz/sx/x/cx・opt1・seed17という選択肢は維持するが、**compilerへ渡す入力の生成政策を変える**。新native-policy IDとSOURCEを含む別のpipeline identityが必要であり、旧compiler option hashだけで旧費用へ互換だと表示しない。共通プロトコル、AD03採用保留、既存58軸・旧grant・旧STOPを変更しない。

資源は現在の利用可能量と展開後の命令上界、元runの実測peakから見積もり、RAM/disk/fresh input/SOURCE/runtimeを確認する。過去のworker数・GiBを新たな恒常上限へ流用しない。一軸compileにはproductive worker一つ、観測processを別に用意する。候補内の依存処理を無理に並列化せず、再利用・block cache等の軽量な高速化を検討する。host/上位cgroupの一時的なPSI・CPU競合や無関係なworktree更新ではkillせず、実OOM・own異常・使用入力/結果の破損に対する保護は維持する。

## 6. 検証・保全と次の担当

人工18 test methodがPASS（0.137秒）。正負の小さい回転、制御付き演算、巨大な人工scalarの正確な総和、symbolic/nonfinite/未知gate拒否、偽P名の拒否、測定順、callback不変性、定数phase欠落の再現と回避を含む。失敗経路を検出するtestのPASSを、そのcompiler経路の受入PASSとは数えない。初回は人工例のAPIを修正し、続く2件のraw FAILをきっかけに上の最小再現と明示候補を作成した。過去のtest logはprivateに保持した。[検証記録](../../artifacts/resource_applicability/track_a_h6_m5_phase_repair_design/2026-10-19/artificial_validation.json)。作者自身の確認で、独立レビューではない。AI subagent0。

前段の2324保護fileに、前回の新結果・新grant/ledger/marker・資料を加えた**2369fileを保全**する。固定runtime/旧SOURCEのbyte照合、今回のsource・文書・DRAFT/無効grant・索引・inventoryの構造検証を行い、既存の履歴は変更せず追記する。実H6科学処理・有効grant・共有環境変更・他worktree書込は0。

**Codex（次の準備作業）**：固定一軸用adapter/launcher、保存vectorのlineage結合、監視・排他的な保存・one-shot拒否、fresh資源見積り、完全なSOURCE closureを仕上げる。

**ユーザー（その準備報告後）**：新しい入力生成政策を伴う一軸pilotを実施するか、明示的に承認する。現在の追加調査承認や人工PASSを新compile承認へ流用しない。

今回の追加調査・prototype・修正計画はlocal commit後mandatory STOP。公開push、実H6への自動適用、後続Core・再会計は行わない。
