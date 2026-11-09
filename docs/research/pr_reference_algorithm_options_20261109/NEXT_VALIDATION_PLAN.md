# 次の最小検証案：実行待ち

2026-11-09。**本書は次の実行範囲の提案で、実行承認ではない。** 今回は静的準備まででmandatory STOP。[調査結果](REPORT.md)、[式と障害](ALGORITHM_IMPROVEMENT_OPTIONS.md)。既存runner／固定SOURCE／STOPは一切再開しない。

## 1. 判断する順序と変更しない条件

最初にQの「未知半径を含めても単一方向の測定から必要なenergy幅を識別できるか」を診断する。回路を作らずに反証できるためである。Iの最新PR×Referenceは独立案として残す。両案を同時に変更し、改善の原因を混ぜない。

今回の固定成果は1cd1fd29949c2079cd74a6d25855f9f1e54ce1d2、V6-P結果は5f672ce6884bd49231366875a9e24a539fcaec7a、元入力／D0はf2638a7845685a119619382b850441f5d8d33713。Linear H4 1Å/STO-3G/DF12、元H exact ground供給、状態準備費除外、Partial S2 LD3 h=.02 K2、18round、元のq/r/bits／canonical law、energy1.5936e-4 Ha／failure .05／phase .4を引き継ぐ。これらを変更する提案ではない。未知E0/gap/state vectorをpolicyへ入力しない。

**推奨する次の依頼は§2だけ。** 保存入力の数値再解析・有限区間診断の明示承認が必要。Qの新しい成功保証の正式採用を判断する重要レビューは別であり、この文書で開始しない。Qが技術診断を通っても、§3の回路／RPE campaignへ自動移行しない。

## 2. Q0：m17の有限識別・shot境界診断（回路なし）

### 2.1 入力とSOURCE

使用する保存入力を次に限定する。現行branch名ではなく固定commitから読む。欠損field、意味の異なるsignal scale、元Hへのphase接続の不一致ならUNQUALIFIEDで停止し、0や推測値で補わない。

| 固定入力 | 用途 |
|---|---|
| 5f672ce6：`artifacts/pr_reference_v6p/2026-11-09/P_run01/RESULTS.json` | m17のB′およびReference-B1について、保存ρ下界・β下界・raw disk・Nの採用値。旧Reference630も履歴として保持するが主対照にしない |
| 同`SOURCE.json`／同`CHECKS.json` | 原入力のpath／blobとfieldの来歴。V6-P再実行には使わない |
| f2638a78：`artifacts/pr_reference_bprime/2026-11-07/ERROR_RPE.json` | m0〜16の元保証・m16までのprefix条件。真のsignalを設計に使わない |
| f2638a78：`artifacts/track_c_guarantee_cn_price/2026-11-06/A_guarantee/verified/GROUND_ALIAS.json` | 元energy branchの既存条件。ground solveを実行しない |

既存`pr_reference_bprime_accounting.error_and_shots`、`rpe_resource_accounting._hoeffding_shots`、`rpe_four_round_phase_validation.reconstruct_rpe_phase`、`rpe_hadamard_failure_validation`の二項／座標・位相tail処理を**関数の意味と統計単位の再利用先**とする。H4をload/buildする既存entry pointを呼んで数値を再取得しない。必要差分は保存field読取とannulus／区間反転の小さいadapterで、generic optimizerや承認基盤ではない。

### 2.2 科学callの有限範囲

対象はB′／Reference-B1の2条件、**m17のみ**。nは旧m17両軸合計`2×626`以下の正整数だけ。これは許可範囲の有限候補集合でありCPU/RAM/worker制限ではない。条件が不利でもn範囲やρ情報を自動追加しない。

1. **誤差と半径を同じ物理meanへ戻す。** ρLは保存raw radius lower、ρUは無重みphysical coherenceの1という外側制約。`B=.4−βstat_lower`を安全側に囲み、元のPF＋disk由来のphase chargeと同じ意味を持つことを確認する。diskを再加算しない。実半径や真の位相を計算しない。保存βに単純な位相モデルへ戻せない条件があれば、その入力ではQを資格化しない。
2. **prefixからの区間。** 既存m0〜16の成功event上で、m16の出力に対するenergy半幅`w=.4/t16`を使用する。点推定の正しさを仮定せず、残差`u=−t17(E−Ec)`の全区間`[−t17 w,t17 w]`を扱う。既存alias範囲も保持。Ecは将来の実outcomeから得る量で、保存E0ではない。
3. **まず理想feedbackによる必要条件を調べる。** `δφ=0`とする診断は明示した楽観的screenだけであり、未評価のfeedback誤差を0認定する意味ではない。この条件でもsin枝が閉じない／半径識別幅が目標を超えるなら、このQ案をSTOPする。合格しても実装成功・RPE保証・総費用改善とは判定しない。
4. **正のfeedback余裕を境界として返す。** 新providerを仮定する代わりに、目標幅とshot節約を許す`δφ`の上限を外向きの区間で報告する。探索域は元の位相cap内`[0,.4]`、単調な包含判定を使用。192-bit区間を既存算術として使い、上下界が判定を分ける場合はINDETERMINATEで停止する。新しい合成precisionや誤差予算を採択しない。実feedbackの誤差がこの上限以下である証拠は後続の別資格である。
5. **固定回数のconfidenceと集合反転。** one-directionの各nでk=0..nの全countを対象に、±1 Hoeffdingの安全側区間を使う。全可到達energy／ρ／biasと、その区間に整合するcountに対し、反転集合hullの径が`2·.4/t17`以下であることを囲む。数点のθ／ρ gridや点値で一様保証としない。空集合／複数枝／幅超過のcountを捨てない。全域の囲みが閉じなければ未資格で終了する。
6. **同じ総失敗予算。** prefix34軸の各1/720を保持し、最終round合計2/720をQと対照の両方へ与える。Qは固定nの一方向、対照は同じ合計予算で整数X/Y配分を許す。prefix失敗を条件付けたまま消さず、元の総.05へ戻す。データを見ながらnを変える方式や固定n式によるpeekingは含めない。

二項tailの既存コードは、固定nの正確な周辺二項評価とconfidence反転の意味が一致する部分だけ再利用する。真の保存pにおけるtailが小さいことを、一様confidenceへ読み替えない。未知pの区間上での保証が閉じない場合、Hoeffdingによる対照と候補の比較に限定し、二項改善を0補完しない。

### 2.3 強い対照・公平性

最低限、次を同じ2 armで比較する。前段区間・未知ρ集合・全phase域を候補と対照で共用する。

- 旧X/Yの626/axisは保存値として引用。
- X/Y固定方向でm17総失敗予算内の整数配分を許す対照。既存の同じconfidence方式を使い、αの整数境界も外向きに扱う。
- 前段区間を使いX/Yから有利な方向を選ぶ、**Clifford方向のみ**の強対照。候補だけが前段情報を使う比較にしない。Ecにより向きが変わるため、未知の中心phase全域も区間で確認する。Pauli方向だけで同等の結果なら任意feedbackの追加利益は未成立。
- Qの任意feedback方向。二項confidenceが正当に利用可能なら他対照にも同じ機会を与える。不等式変更だけの差とreadout変更による差を分ける。

Q0では期待RZ／CX／depth／Tを再集計せず、n境界と許される追加readout価格の**記号的break-even式**を示す。少ないnを得ても、追加RZ／Tがその節約内に収まるかは未評価。B′にQを与えずReferenceだけと比べる採用は禁止する。

### 2.4 必須検算・終了条件

- 人工ρL=ρU、bias=0でsin反転と標準Hadamard式が一致。φ=0/π/2でX/Y符号を確認するが、量子回路はbuildしない。
- 小n全countの手計算可能な人工例、ρ区間・bias・feedback域を広げると可行集合が縮まらないこと、alias境界・空集合・枝分裂を検査する。
- 独立した単純区間実装または解析端点と照合し、有限gridの未検査内部をPASSにしない。著者の技術検算であり独立科学レビューとは呼ばない。
- prefix＋m17失敗予算、同じdiskの二重加算防止、両armのinput binding、境界を含むceil／整数比較を確認する。
- SOURCE不足、量子化／feedbackの必要条件不足、単調枝不成立、理想screenで強対照以上のshot、数値囲み未決、有限scope完了のいずれかでmandatory STOP。別precision／radius校正／分子／PFへ拡張しない。

Q0で許可を求める科学callは、**この保存2 armのm17有限区間／統計境界再解析のみ**。新しい回路・native価格・strict guard・合成・実sampling・18round総費用は含めない。実行時はfreshな専用出力へSOURCE・入力identity・結果・検算・STOPを保全する。今回それらの実行成果物は作っていない。

## 3. Q0が通った場合にのみ検討する後続（今回もQ0承認にも含めない）

実feedback角の生成・誤差、履歴条件付きの一様成功保証、実mean許容集合とenergy区間の包含は未資格。QをRPEの新しい成功保証として採用する重要判断をGPT／ユーザーへ戻す。既知適応PEを使うこと自体と、PRのnuisanceを含む保証が成立することは別である。

採択する場合の最小実装は、旧coreにP(−φ)を付ける薄いreadout adapterとconfidence集合decoder。既存caller／producerは変更せず、別の明示φ interfaceとfresh trialを使う。人工2-modeの複素coherenceについてφ={0,π/2,−π/2,π/4}、B′型controlled／Referenceを確認し、common global phaseとbranch phaseを分ける。実装する場合も新しいgadgetやsamplerは作らない。

それから対象を事前固定する少数の実入力fixtureとcompiler contextを選び、SOURCEのphase、lower後のoperator／Xφ、reset、追加RZ/CX/depth/size/qubitsを検査する。**この実入力／角度集合はまだ採択しておらず、今の計画をbuild/compile許可として使えない。** m17全回路やH4新samplingは最初のcallに含めない。

全18roundへ進める場合は、m0〜16の費用不変を確認し、Qの両armと、同じ機会を与えた強Full／Truncatedの元H／schedule／compilerを固定する必要がある。資源jごとに`Σ N E[C_j]`、追加校正／feedforward費、資源の資格（exact／bound／MC）を分離する。新Clifford+T評価はさらに別scopeで、feedback word／joint disk／T価格を含める。native勝利から総T勝利を認定しない。

## 4. I0/I1：最新PR×Referenceの接続案（代替の選択肢）

基準はc5a3c102の`extended/POLICY.json`、同`ROUNDS.csv`、`SOURCE.json`、`RAW_INDEX.json`と公開raw archive、canonical／Closedの同logical-word意味論、625 joint catalogue、diagonal network。新Aの旧h=.02 consumerを緩めず、別adapterの対象を一組に限定する。Iは既存の組合せ提案の具体化で、他担当の研究を新Aへ移管しない。

**I0の静的bindingで未確定なのは実fixtureのidentity。** 公開archiveにはcorrectness／path資産があるが、今回word列を展開・再分析して「最新policyと同時に適合する一つの短trajectory」を新たに認定していない。将来は既存の公開correctness記録にある最短のcomplete occurrenceを、価格・信号を見る前に`(t,n,record ID,raw member,hash,law,weight,context)`まで固定する。scalar・非可換product・actual inverse・joint mappingを含まなければ補完生成せずUNQUALIFIEDで止める。このbindingは既存担当の後続結果有無の確認を含む。

その後のI1案は同じlogical対象でcanonical／Closed×strong controlled／Reference×X/Yの最大8 wrapper。既存IRからuncontrolled diagonalを出す差分だけを実装し、`reference_core`、partial Gaussian operations、compiler A、旧CAR／sector／observable検査を再利用する。旧C/D/newAのbuilderを改変しない。量子回路build/compile/simulation／有限価格取得には、この実fixtureを固定した別の明示承認が必要。

correctnessでは、whole ordered operator、signed phase、全system入力でのX/Y誘導observable、vacuum rayとχ、actual inverse、proposal×weight、controlled branchを保持する。価格は同じfull-wrapper scopeのRZ/CX/各depth/size/peak qubitsを別々に取得。局所emitterの差やfinite fixtureの価格差を全law期待値とみなさない。

一致不足、既存強controlと同等／不利、cut損失が支配、入力欠損ならSTOP。成功してもjoint disk／weighted range／必要N／全round価格への移送は別scope。独立の次段階では強Full／Truncatedにも同じ合法control／fusion機会を与える。長q価格、最適cut、law／h/rate再探索、総Tへ自動進行しない。

## 5. 承認・担当・今回の停止

| 次作業 | 担当と必要な判断 |
|---|---|
| Q0を選ぶか | **ユーザー／GPT**：既知適応PEのPR向け応用診断を優先するか判断。選ぶ場合、§2の保存数値再解析・区間／統計境界評価を明示許可 |
| Q0実装・診断 | **SERVER 2 Codex**：選択後の固定scopeだけ実施。通常の補助実装・人工テスト・debugは同じ範囲でまとめて行い、結果公開後STOP |
| Qの保証採用・後続回路scope | **GPT／ユーザー**：Q0結果を見て判断。詳細レビューを開始するならその明示指示が必要。Codexが正式採択しない |
| Iを先に選ぶ場合 | **ユーザー**：新Aと既存C/D担当の作業範囲を指定。CodexはI0のbindingを完了してから、有限I1 callを具体化。I0段階の不足を推測で埋めない |
| 他の新規機構を待つ／保留 | **ユーザー／GPT**：Qが既知法応用に留まること、Iが既提案であることを踏まえ、独立新規機構を優先して保留する選択も可能 |

Q0・I1・重要レビューは今回実行していない。V6-P/Q保証厳密化、実H4の4 kernel fixture、V6-P再計算、m17 strict guard、D3 inverse defect、composite合成、43-key全面資格、全18round再会計は引き続き保留。別候補名で旧STOPを解除しない。

**今回の3文書をcommit/push・remote内容照合後mandatory STOP。次担当はGPT／ユーザー。**
