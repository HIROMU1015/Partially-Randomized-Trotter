# H2 m5：停止保存・位相保持による技術修正

利用者の段階1〜6実行承認の範囲で、初期化STOPとcompile STOPを保存した別SOURCE・別one-shotを用いる。旧grant/ledger/marker/STOPは変更しない。H4/H6 pilotの再実行承認へ流用しない。

## 保存された停止

- SOURCE `cbdf57ba0509542193abca5fb7199d53e06b09b5`：scalar配列のmetadata形状規約が元H2と異なり初期化STOP。signal/sampling/compile 0。元のcontiguous規約へ修正した。
- SOURCE `ffb62a445c22e01e1ccd832463407e2dec1c2970`：signal101完了、費用取得の先行cosineでraw compiled差が約2となりSTOP。accepted pair0、統合未実施。61 workerを回収した。先行保存例はafter ≈ −before、負符号残差約1.23e-12だが、位相合わせして受理しない。

保存QPYの再生診断では、Qiskit Statevectorのglobal_phase処理が共有入力配列を変更し得るため毎回コピーして照合した。コピー後は保存before/afterをそれぞれ再現し、compiled側のraw差約2は残った。この診断中の入力共有問題と、科学runのcompile不一致を混同しない。

## 修正方法と互換性の範囲

固定Qiskit1.3.0の[replace_node実装](https://github.com/Qiskit/qiskit/blob/1.3.0/crates/accelerate/src/basis/basis_translator/mod.rs)は、parameter付き置換のglobal_phaseをParameterExpressionの場合だけ加算する。定数となった位相が落ちる人工最小例を再現した。

専用process内のBasisTranslator呼出しだけに `H2_QISKIT130_SYMBOLIC_ZERO_SCALAR_GUARD_V1` を適用する。各parameter付き入力gateを、同じgateと値0の補助scalar parameterを持つ私有等価則で包む。合成templateの位相にlive parameterを残し、既存translatorの位相加算経路を通す。出力状態・overlap・測定値から位相を推定したり、compile後に合わせたりしない。

各翻訳passで元translatorも走らせ、全gate名・数値角度・qubit/clbit順序のdigestが修正経路と一致することを必須gateにする。異なればSTOPする。H/SX/X/CX/RZ等のgate列を変更して低費用化する修正ではない。元のQiskit version、basis、opt1、seed17、compiler設定hashを保持し、追加policyを別欄へ記録する。SessionEquivalenceLibrary・共有runtime・元providerは不変。後続最適化も既存opt1のまま。

人工3qubitの最小例でraw演算子差約4→約7.46e-15、最終gate列・角度・個数の同一性を確認した。12符号/角度条件、定数位相の独立等価則、例外時復元、既存native/会計等を含む24 testがPASSした。これは作者自身の技術検証であり独立レビューではない。

raw compiled状態差1e-7未満、native/oracle/Hadamard等1e-9未満を維持する。過去のH2 S2/S4費用の受入検証を後から全状態保証へ昇格しない。新m5のpolicy区分と、各passで確認したgate列互換性を統合表と共に掲載する。人工結果だけから実H2の完了を認定しない。

## 新one-shot

保存signal101はSOURCE・入力SHA・生成関数ASTの同一性を照合してbyte再利用し、再計算しない。元H2探索・確認・H4/H6・前2契約のseedを除外した新探索cohortを使用する。候補集合・n8・係数・r政策・経験的shot式は元H2追加契約を維持する。

適格選択はm5決定論15pair＋PR528 trajectory、最大543pair/1086軸。これは事前のD35/PR66を同精度両軸のsignal適格性で選択した結果であり、費用による事後除外ではない。並列数はfresh CPU/RAMから決める。新SOURCE/input/compiler・byte復元・native・台帳・exit/reapが全てPASSした場合だけ、既存D/A会計のexact再現と統合比較へ進む。各段階のmarkerと停止記録を保存し、段階7へ自動進行しない。

次の担当：Codexが承認済み統合比較まで実行・終了監査しmandatory STOP。ユーザーが独立確認・AD03の採否を別途判断する。旧H4/H6 pilotのSTOPは保持する。
