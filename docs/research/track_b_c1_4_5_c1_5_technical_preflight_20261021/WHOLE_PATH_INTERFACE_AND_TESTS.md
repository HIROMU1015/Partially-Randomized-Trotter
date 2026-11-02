# Whole-path interfaceと人工検証

実装は[family.py](family.py)、[whole_path.py](whole_path.py)、[native_receipt.py](native_receipt.py)。`SYNTHETIC_ONLY`・`synthetic:` model・最大3 labelを入口で要求する。実H4入力のloader/runnerを追加していない。旧adapterのlogical `append_q`とHadamard wrapperを再利用し、RTEEventや既存DFPartialS2RepeatedCircuitResultへ型偽装しない。

## schemaと状態遷移

- Microstep receipt：method、exact p/x/mを含むtarget hash、model/label identity、SOURCE SHA、precision、certificate種別、provider/測定規約、trial ID、実event/proposal/weight、bit trace、formal queryの区別。FullはF3原Evidenceも保持する。
- WholePath：ordered step context、schedule ID、独立性のcaller仮定、step IDとEvidence digest。step IDは`trial:index`。全eventを古典的に先決定し、ordered actual weightとproposalを積にする。raw/coin zeroが一つでもあればpath weight=0。zeroの総proposalをaccepted eventの積で偽装しない（`proposal=None`）。
- AllTrials：事前に一意な全trial IDとaxisを固定。`prepare`が既存sampler→whole-pathへ接続し、失敗時は最初のfailureを保持してterminal。各accepted pathにはbit0→+1/bit1→-1の観測整数を一つ受け取り、weightを掛ける。zeroには観測を受け取らない。
- 完了：予定IDすべてが一度ずつ処理されfailureがない場合だけ`full_trial_mean=weighted_sum/planned`。未完了・重複・technical failureでは正式meanを出さない。confidenceは常にfalse。外部provider/測定/保存の失敗もcallerが`fail`へ渡す必要があり、productionの永続transactionは未実装。

accepted-only分母、formal queryのsample扱い、事後conditioning、未知zero reason、改変context/digest、実πW不整合を拒否する。FullはF3の記録済みbit列を再生し、canonicalも固定generatorのtraceを再生して実eventを照合する。再生では外部乱数を追加取得しないが、係数queryやEvidence検査は再度必要であり、古典費用に含める。Evidence hashは内容同一性の検査で、独立性や外部証明の真実性を認証する署名ではない。

## operator・phase・measurement

回路時間順はstep0→step1…、行列積はV_last…V_0。各event内はreversed reduced parent word→child回転。canonical iid語はreductionで長さが減っても**元のraw Taylor次数の位相を保持**する。例：ordinary degree2 raw(0,0)は空reduced wordだが位相は-1。Full用の「child!=first」をordinary/Partial/P3のiid tailへ強制しない。

人工whole-pathでは別のcontrolled deterministic before/afterとscalar相対位相も結合し、独立にraw wordから構成したsigned-Q行列積とfull controlled matrixを比較した。XはH、YはSdg→H、bit0は+1で、weighted X/Yを全trial平均へ接続する。float angle代表値による小さい残差はstrict native certificateではない。

全pathを量子実行前に決めた場合だけzeroで回路構築・quantum callを回避する。途中実行architectureでは`spent_prefix`に非負の既支払費用を要求し、後続zeroでも消さない。そのarchitectureの実executorは未実装で、`circuit`は明示的に拒否する。zeroのclassical処理費は0としない。

## 比較family契約

全familyはmethod/exact p・label/m、normalizer取得、ideal/digital law、actual proposal/weight、zero/合法性、誤差certificate、DF provider、measurement、setup/online、SOURCE/Evidence/terminalを保持する。normalizer・setup/selector等の方式別費用は[SOURCE表](SOURCE_CONNECTION_AUDIT.md)と[native監査](NON_ENUMERATIVE_NATIVE_COST_AUDIT.md)に記録し、UNKNOWN価格を数値0にしない。

Full以外のreceiptに`CONDITIONAL_EXISTING_NOT_F3_CERTIFIED`を付ける。synthetic trace guardはcanonicalの一般入力uniform finite-bit certificateの代用にならない。Full/P5の同理想ensembleと、digital実law/normalizer近似/weightの相違を両方保持する。

## 人工テスト結果

最終[15件raw log](evidence/test_run_03.stderr.log)・[SOURCE/runtime記録](evidence/test_run_03.json)は全PASS。先行14件の2回のPASSも無加工で保存した。追加再実行は実law trace再生、意味的event価格key、sampler失敗のterminal伝播を確認するためで、科学run/retryではない。

| 必須事項 | test名/検査 |
|---|---|
| 全trialとvalid zero | 01：accepted 1・raw zero 1・coin zero 1、分母3 |
| failureとzero | 02/14/15：失敗でzero数を増やさずterminal、再prepare拒否 |
| weight積 | 03：actual weight/proposal積をexact Fractionで照合 |
| phase/order | 04/13：両sigma、非可換parent・step、deterministic halves/scalar、raw reduction後の-1 |
| X/Y | 04/05：人工stateのwrapper expectationと独立行列、±1 weighted accumulator |
| identity | 06/10：SOURCE/model/precision/digest/actual coefficientの改変拒否 |
| 完走/未完走 | 01/07/15：完全時だけmean、途中failure/重複はterminal |
| conditioning | 08：formal query/accepted-only改変拒否 |
| 共通target/異law | 09/12：同finite target、Full/P5 digital lawを同一視しない |
| 同event価格の公平性 | 10：Full/P5の対応root eventが同意味key、methodを価格keyに含めない |
| zero quantum費用 | 11：事前zeroはcircuit None、既実行CX prefix=7は保持 |

[family_checks.json](evidence/family_checks.json)は人工L3有限supportだけを列挙し、Ordinary273、Partial264、P3+tail258、Full63、P5 63 eventの実πW=αとfinite polynomial mean・二step積を診断した。独立referenceはraw wordと人工Qから構成し、adapter出力をexpectedに使っていない。試験の微小float残差・同じ結果は一般定理や完全なdigital law同値の証明ではない。

未実施：実H4 whole-path、物理state準備、実測定/shot、IID実証、統計Monte Carlo、十分N/CI、native価格・全support認証、全round RPE。実入力適用にはtarget/全schedule・runtime・保存終端・各family certificateを別途固定し、新しい承認を取得する。
