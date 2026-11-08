# 次の最小科学scope案：固定pathでの費用資格

状態：DRAFT_NOT_ADOPTED / NOT_EXECUTABLE。人工で有利・不利の両条件が得られたため、実PRに同じ利益が残るかを少数の固定pathで確認する価値はある。独立新機構や科学的GOを採択する文書ではない。

## 次の判断と入力

判断するのは、selected fullのfusionを保った既存controlled PRに比べ、追加8-qubit真空referenceとCSWAP/routingを含む同信号の回路単価・depthが下がるかだけ。実H4の全supportやQ期待費用の保証をこの段階で狙わない。

候補入力は保存Linear H4／1.00 Å／STO-3G／legacy DF12、8 system qubits、L_D3、m0/q1/delta=.02/r1/K2、coefficient_atol=0、N1 actual finite-grid Qとselected full policy。N1結果744e1a09661ebba6841fe5faf456f88c41c47780、N1科学SOURCE b6dbec9f80da35a88fba28795ad02ca74c9adfa0を固定候補とする。今回そのpayloadを読み込んでいないため、対象path identity・入力blobは未選定。旧measurement bit/state/cache/streamを流用しない。

対象pathは事前固定したn=0の1件と、既存support上の正次数1件までを上限候補とする。metadata-onlyの選定方法・path IDを実行前に固定し、費用を見て選び直さない。別order/roundへの探索はしない。各pathのX/Yを比較する。既存block列の中央に最も近いapplication境界でR/Lへ1回だけ分ける案で、split探索は行わない。

## 必要な最小技術作業

- 1–3 qubit限定prototypeをそのまま科学runnerにしない。既存serializerでL/Rと全scalar/event phaseを保持する小さな実入力adapterを別SOURCEへ固定する。
- 実gate列にRZ/RZZ/XXPlusYY以外の未対応演算があれば先にSTOP。gatewise vacuum証明とexact binary64角度ledger、inverseとphase補正の一様誤差を適格化する。
- correctnessは全入力operatorを対象とする正当な証明/誤差境界で行い、数stateの一致や代表pathを全support保証へ拡張しない。N1 preparation、state solve、旧engine認証の再計算は不要。新state/Bornの実行は初期案に含めない。
- 対照は同じselected/fused controlled、同じpath splitの既知SE、候補。fanoutは同じように認めるが、最小案では非fanoutを固定。17 qubits対9 qubits、明示的8-reference resetとsignal readoutを計上。fanoutを採るなら24 qubitsと追加reset/CXを別契約にする。

## 有限callと費用

提案上限は2固定path×X/Y×3方式×2 topology（all-to-allと固定line）=24結果行。同一SE/candidate IRは価格cacheで再利用し、cold compileは同一IR当たり一度だけ。N1 lawの再計算、RNG、shot、state/Born、train/calibration/productionは0。

compile profileは科学実行前に採択する。既存A1価格との比較を行うならQiskit1.3.0、rz/sx/x/cx、optimization level2、seed17を全方式に同一適用し、新しいphase-preservation監査を通す案。今回の人工結果は既存P0 level1なので、価格を混ぜない。routing、initial/final mapping、reference reset、相対P、full interrogation readoutを含める。

RZ/CX count、各depth、qubits/reset/measure、cold/warm compile・path/phase生成時間・memoryを記録。TはFT synthesisと誤差配分を定義しない限りUNKNOWN。状態準備費は共通項として除外し、target state消費は一shot一個で保持する。物理reset/readout時間や実行errorはhardware契約がなければUNKNOWN。

## STOP・追加承認

入力/SOURCE/演算子/phase/policy不一致、未対応gate、非有限値、必要な誤差資格が得られない、compileのphaseやroutingを追跡できない、出力衝突、共有環境非干渉の不成立でSTOP。retry0、対象・split・profileの事後選び直しなし。

最初に必要なのは、path選定規則・固定identity・profile・新adapter誤差方式・最大callを確定する技術準備。その後、上記有限scopeへの明示的科学実行許可が必要。全RPE改善、hardware noise/postselection、期待費用やshot数の比較、重要な科学的新規性の判定はさらに別課題で、今回自動開始しない。
