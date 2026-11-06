# Track D C7-1以降：最小追加検証の実行条件案

2026-11-07 JST。**未採択・未実行。C7-0の許可は本案の作成までであり、本案に科学実行の許可はない。** [構造解析報告](REPORT.md)を踏まえ、ユーザー／GPTが目的と範囲を選び、Codexへ別途明示的に実行を指示するための最小条件を示す。旧C6a/C6bの再開契約ではない。

## 1. 推奨する最小scopeと未決事項

最初の科学domain候補は**q=16のみ**。最初の未検証qで、q8からの長さ移送・global cut・数値精度・compiler残差を限定して検査できる。cut候補は結果非依存のcomplete step8後、whole Vに一つのReference gadget。これは採択候補であり、今回はrequest/seed/event/回路を生成していない。

|ユーザー／GPTが実行前に採択すること|候補・判断基準|
|---|---|
|目的/estimand|同qの直接paired期待価格、固定予測値の誤差、population校正線、方式別絶対期待価格を分ける。用途が同Nでの方式差ならpairedを優先する案|
|探索的/確認的|まず有限観測と名目的診断までの探索的検証にするか、有効な母平均/モデル精度保証を要求するか。後者はlaw/独立性/tail等の根拠が別途必要|
|資源・family|RZ/CX count・各depth・total depth・sizeを個別記録、qubitsは幅。主資源を無断に固定しない。確認的familyと多重性は用途に合わせて事前選択|
|価格幅・信頼度|許容する価格予測誤差と、資源差の符号/実用的削減の判定を分ける。小さいpaired差を分母とした相対幅を避ける。新しい幅/alphaは未決|
|nと停止規則|whole-path標本数n16は未決。C6bの32を保証数として引き継がない。必要なら標本設計の限定数値作業を許可scopeに含める|
|モデル|CXの固定/bulk/boundary候補＋P0残差、RZ/sizeの固定＋bulk＋path残差、depthはaffine対最小lane候補。全path定理は未認定|
|校正データの扱い|C6a/C6bを設計材料にするか、既存凍結モデルを変更せず診断するか。新fitは別途許可し、旧係数/幅を上書きしない|

本案ではn・価格幅・信頼度の新しい数値を決めていない。探索的nは知りたい残差/変動と実行負荷から採択し、確認的nは有効な推論法と精度用途の根拠から設計する。q8のSDやmin/maxだけを長qの母分散/support上界にせず、任意停止・都合のよい追加samplingで精度PASSを作らない。

## 2. 固定する科学条件・SOURCE・law・seed

共通条件はlinear H4/1.00 Å/STO-3G/Nα=Nβ=2、number-CDF8/元frame順LD2、ordered partial S2/s1/r32/K0、h=δ=.1、元H厳密ground仮定・準備費除外。packet/endpoint/lawの三つのSHA256は[REPORT §1](REPORT.md#1-固定条件確認資料保全)に一致させる。新ground solve・DF fitは不要で、own-DF signal/Nを元Hへ転用しない。

採択後に新しい専用branch/SOURCE/出力namespaceへ、次を固定する。

- 採択qとcut、n、sample/occurrence IDと順序、モデル仕様・係数の来歴、estimand/family/幅/信頼度・判定規則。
- C6a/C6bの既使用domain/seedとは別の価格検証domain。C6bの名前に残る`unused`を未使用と誤認しない。完全seed keyとPRNG版、canonical draw順序を固定する。
- 一度生成した同じwhole-pathを両方式・X/Yで共有する。価格による標本/seed/cutの変更・除外なし。全予定event列を科学価格評価前に保存/hash bindingする。
- 元C science SOURCE `ba0e2bf2d1d6f0dc78616d3392f3ff5b6ba7d5b7` とC6b最終公開 `fed5f458be2f625de4e1b8c98b8cc5da6e551b14` の関係、新しい実行SOURCE、再利用元ファイル版、固定入力/lawを記録する。
- P0 Qiskit1.3.0/opt1/seed17/rz,sx,x,cx/approximation_degree1/initial-zero false/topology-free/位相保持、既存component abs1e−10・observable Frobenius/signal2e−10・rtol0・phase fitなし。

SOURCE/input/hashは科学的来歴のために固定する。ユーザー承認の証明書、承認原文保存/hash、grant発行・認証サーバー等を新たな要件にしない。既存の旧grant/ledger/marker/STOPには触れない。

## 3. 必要最小限の実装・許可対象

|処理|既存再利用先|許可が採択された場合だけ行う最小差分|
|---|---|---|
|新q request/seed/event ingestion|C6a/C6b bindingとcanonical `configure_rte/sample_occurrence/iter_sample_rte_events`|旧q2/4/n16、q8/n32のguardを保持し、新q/n/domain用の薄い接続。sampler再実装なし|
|whole-path/CAR|C3 `prepare_steps`、C6a CAR context、既存B2 head/atom cache、StrongBuilder/Reference adapter|同じ組立に新しいbound stepsを渡す。強いbaseline、位相、actual inverse、normalization会計を保持|
|full-wrapper価格/P0|C5 `wrapper_for/_checked_price`、C2 compile/postcheck、C3 pairedχ|旧資格を緩めず新scopeの価格recordを追加。raw PASS pathだけ最大4 wrapper。compiler/precision変更なし|
|構造特徴|既存conditions/fusionの科学scalar、builder内の境界数/cut/phase|必要な場合だけrun/support/lane/interface等を保存する最小拡張。追加の局所回路生成/compileは明示された許可範囲に限る|
|高速化|C6b `BatchedStateActions`、同Aer変換/native/double、cacheとpipeline|新workerで選択し、版適合を必要な人工回帰で確認。全256入力・全出力振幅・位相復元を維持|
|統計/モデル|C6a/C6b paired統計、共有共分散、generic fitterとfrozen診断の設計|採択estimand/n/familyへ最小接続。新fitが許可されない場合は固定モデルだけを診断し、旧結果を変更しない|

科学call予算は採択nに対し、**canonical occurrence samplingは16n、whole-path raw検証n、P0 full-wrapper compile最大4n、対応postcheck最大4n**というscope案。途中raw不成立pathはcompileしない。X/Yを二倍の独立標本へ数えず、既存有限同価格だけを理由に一軸を省略しない。既存q1/2/4/8のcompileを再実行しない。

追加calibration、局所dictionary/境界compile、長qモデルの新fit/数値予測を必要とするなら、その対象とcall範囲を別途明記して許可を得る。q16検証の許可に含まれたとは推定しない。既存q8は新モデル設計へ使った時点でその新モデルの独立holdoutにはならない。

## 4. 検証順序・GO/STOP

1. **目的固定・SOURCE適合確認。** 対象q/cut、モデル/estimand/n/判定、SOURCE/input/law/seed/非重複、P0・価格境界、strong公平性を確認。旧STOPと既使用domainを保全する。
2. **採択された実装・人工回帰。** q/seed binding、AB・actual adjoint・χ、旧guard、paired単位、frozen/校正不確かさの区別、必要な高速化互換を確認。人工PASSを分子性能証拠にしない。
3. **新scopeのsamplingを一回だけ実施。** 全予定eventを保存してidentity固定。同じpathを両方式/axisへ渡す。旧cohort巡回・prefix再利用・短path反復でfresh標本を補わない。
4. **pathごとのraw correctness。** 独立CAR、AB=V、actual A†、vacuum ray/χ、scalar/identity/sign、strong branch/coherence、X/Y誘導observableを全256 system入力で既存精度照合。
5. **PASSしたpathだけcompile/postcheck。** 最大4n wrapperでCAR/raw、paired X/Y、χ、reference leakage、declared phaseを照合。追加register/CSWAP/cut損失/shellを含む価格だけ比較。
6. **完全な予定標本の整理。** 個別価格/差/SD/方式間・資源間・axis間covariance、採択したモデル診断、SOURCE/input前後照合とSTOPを保存。欠測complete-caseで有利なモデル判定をしない。
7. **公開可能な結果・SOURCE・hash索引を公開照合してmandatory STOP。** raw payloadの別公開許可がない限りローカル保全のみ。追加q/calibration/fit/N/総費用へ進まない。

技術的GOは依存処理へ進める条件で、全path保証や研究上の採択を意味しない。SOURCE/input/seed/law/cut/phase/compiler/precision/公平性の不整合、correctness不一致、実際の継続不能な資源不足、破損・証拠欠落があれば該当処理をSTOPしpartialを保全する。無断retry/resume、閾値緩和、弱い対照への置換はしない。

モデル診断が幅外・精度不足なら、そのモデルの当該domainへの移送をSTOPし、結果を理由に同holdoutへ再fit・幅拡張してPASSにしない。探索的結果を確認的保証へ格上げしない。構造モデルのP0 bridgeが不足した場合も全path公式のclaimをSTOPするが、保存された有限観測はその資格のまま保全する。

## 5. 計算負担と直接compile削減

q16では長いnative IR/逆回路とP0 DAG、CAR matrix積、Reference全入力state-action、配列保持・QPY I/Oが主な候補ボトルネックである。q8からwall/RAMを比例外挿した数値見積りは本案で作っていない。qubit幅一定でもgate列と変換copyの負担が一定とは限らない。

実空きCPU/RAM/GPU・適用quotaを実行前と進行中に確認し、独立path/compileの並列化、raw PASS後のpipeline、CAR/head/atom/局所変換cache、既存batch/copy削減を優先する。CPU/RAM/worker数に恣意的な低い上限を設定せず、観測されたmemoryと非干渉に応じて配分する。旧q8 process peakは見積り資料で、長q上界ではない。共有OS/venv/scheduler/他ユーザー・他jobの変更は禁止。

直接compileを減らす案は、SOURCE上の局所正規形/境界/CSWAP/lane依存と、P0 full-wrapperとの同値または有効な残差包絡が得られた部分だけに適用する。P0後DAGを取得してから価格を読む方法は価格の説明には使えてもcompile省略にはならない。native DAGを代理にする場合はP0 bridgeを別に検証する。fenced kernel＋shellや単体CSWAP和へ黙って価格境界を変えない。

## 6. q16の次と全13-roundへの残課題

q16でPASSしてもq32〜4096は未資格のまま。次のqは、構造モデルの有効域、critical pathの切替や長runの未確認域、必要な価格精度、計算実現性を基準に**別途採択**する。全てのqを直接compileすることも、少数qから全域を外挿することも自動前提にしない。さらに離れたqの独立検証を選ぶ案と、全域へ適用する解析的/P0残差包絡を得る案を比較する。未観測域の根拠がない場合、総費用は未評価として返す。

追加q2/q4 calibrationは、採択されたpopulation-line/絶対価格の係数不確かさが目的精度を妨げる場合の別案。q8の別診断4量幅外だけで必要数を確定しない。C6bを見た新fit・新モデルは探索版として保存し、独立検証用domainを新たに分ける。

全RPEへは、各m/axisの同law期待価格、共有係数とcross-round共分散、model bias/domain、ideal/finite-bit law bridgeが必要。B5の全path/全round実装誤差、joint-law、actual N共通化も別に未解決であり、B4-2の保存十分shot数14,738を掛けてactual総費用を作らない。qubitsはpeak幅として扱う。

**次担当：ユーザー／GPTが§1の目的・scope・n決定法と必要な新科学実行を採択/許可する。Codexはその許可後だけ採択範囲を実装・検証する。本案作成後はC7-0のmandatory STOPを維持する。**
