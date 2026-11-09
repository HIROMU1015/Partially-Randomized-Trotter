# 新A V6-Q：保存m17 callerと既存価格保証の接続

2026-11-09、SERVER 2。**保存18roundへ結ぶ薄い生成adapterを実装し、人工13テストPASS。新しい科学計算は0件。** 固定m/q/r/bitsを検証して旧F/G builderへ渡し、明示opt-inのm17だけ既存B1位相をancilla relative Pへ渡す。Q3-N・Compiler Aの構造的保証は、以下の固定SOURCE／外側consumer条件を保つ範囲で接続できる。本番m17 wholeを実行した結果ではない。

## 1. SOURCE・schedule・接続点

| 関数・SOURCE | 接続内容 |
|---|---|
| [新module](../../../src/trottertracks/pr_reference_v6q_schedule.py)のsaved_schedule | 元科学ref `f2638a7845685a119619382b850441f5d8d33713`のSTAGE4_RPE.jsonをGitから読む。m/q/r/bitsだけを投影し、shots・誤差・費用を再評価しない |
| bind_round | 保存rowとcallerのm/q/r/bitsを照合。欠損・重複・未知round・不正mode／型・不一致を拒否。別scheduleの読込口や補完tableは追加しない |
| assemble_reference_for_saved_round | 検証済みq/r、同一cfg/dist/eventsを旧transfer.assembleへそのまま渡す。得た同一F/Gを旧reference_coreへ渡す |
| phase_for_round（既存Q1） | 明示`mode='exact_binary_m17'`かつm17だけ選択し、1回呼ぶ。exact和／RN／subnormal reject等は不変更 |
| reference_axes_for_saved_round | 上記coreを1回生成し、同じcoreを旧wrapperのX/Yへ渡す。compile・reset・samplingは行わない |

schedule正本は[STAGE4_RPE.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2638a7845685a119619382b850441f5d8d33713/artifacts/pr_reference_transfer_qualification/2026-11-06/STAGE4_RPE.json)、Git blob `0003b6ca217eea3df208045ebcec499c5f0c8519`。18rowの対応をmetadataとして確認し、m17=(17,131072,32,88)と一致した。他roundのr/bitsはこの保存rowを使用し、手入力の独立scheduleを作っていない。

全roundでdefaultはlegacy。opt-in policyを選んでもm0〜m16はlegacyである。legacy時はassembleが既に計算した旧phaseを再利用し、providerを再評価しない。m17のopt-in時だけ新phaseで置換する。旧assemble内部のlegacy phase計算自体は保全されるため残るが、新callerがそれを再計算することはなく、B1呼出しも1回である。この残るclassical費用は今回測定していない。

既存SRC／scriptsを検索した範囲では、全18roundの既存経路は会計用であり、`sample()`は有限価格cohort用だった。保存scheduleを消費してReferenceを生成する本番RPE runnerは見当たらなかった。したがって旧assemble/sample／Q2 runnerを書き換えず、**旧assemble→B1選択→旧core→旧X/Y wrapper**を接続するadapterを追加した。sampler・backend実行・shot loop全体は新設していない。

## 2. 固定条件と人工検証

固定条件はLinear H4／1.00 Å／STO-3G／DF12、元H厳密ground供給・状態準備費除外、Partial S2／LD3／h=.02／finite RTE K2、ordinary diag(I,V)、canonical independent finite-bit law、Qiskit1.3.0／compiler A／rz,sx,x,cx。B′4-CX controlled-RZZ、ReferenceのF/G・actual SOURCE inverse・vacuum P・2 register・二つのCSWAP cascade、D融合・basis共有・scalar／signed event・X/Yを維持する。energy精度1.5936e-4 Ha、failure .05、phase cap .4、αaxis1/720、旧Nも変更していない。

[人工test](../../../tests/test_pr_reference_v6q_schedule.py)を`python3 -B tests/test_pr_reference_v6q_schedule.py -v`で実行し、**13テスト全PASS**。既存Q1のduck-lane／instruction／独立整数和helperを再利用し、旧20テストsuiteは再実行していない。

- 保存18rowのmetadata、m17限定opt-in、全roundのlegacy default／他17roundのlegacy保持、未知mode／round／型・m/q/r/bits不一致・欠損／重複scheduleの拒否。
- cfgのr／h／K、distのorder集合とstep parameter整合、event長q*rの確認。不一致はbuilder前で拒否する。
- 同一cfg/dist/eventsとF/Gオブジェクトの受渡し、入力値・gate順・operation identity不変、B1のexact位相と1回呼出し、legacy phaseの再評価なし。
- 人工gadgetでPの位置・wireを保持し、変更が**P角だけ**に限られること、X/Yが同じcore／選択位相を共有すること。
- 旧transfer／pilot／provider／compiler／finite-bit helper／D/RTE builderのbyte不変更。新module・testはQiskit／NumPy／trotterlibをimportせず、科学cacheを読まない。

人工m17 testではq*rの長さだけを持つmockを使い、eventの生成・列挙を禁止した。mockのPASSは実回路correctnessの代用ではない。実回路の有限根拠には[Q2の保存4構成](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9560e310e9f848fe59a1dafce4b506d5a08aae76/docs/research/pr_reference_v6q_q2_20261109/REPORT.md)をそのまま参照する。新しいsimulationやQ2/Q3-N再実行はしていない。

## 3. Q3-NとM／B_normへの接続

[Q3-N 543b711a](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/543b711a4da6cb291af1bb5665db602d609ad272/docs/research/pr_reference_v6q3_numeric_20261109/REPORT.md)の13 basis／135 generator、ℓ=65、`PASS_IN_FIXED_ARITHMETIC_MODEL`は既存結果として継承する。新callerは同じ旧assembleとprep／basis registryの利用経路を保持し、eventを引き直す・並べ替える・F/Gを再生成する・scalar/global_phaseを変更する処理を追加していない。取得したF/Gはそのまま既存phase_for_roundへ渡り、Q1算術を不変更で使う。

従って、**同じ固定cacheとcanonical law由来の適格cfg/dist/events**が渡される限り、SOURCE文法・basis対の相殺・旧correctly-rounded binary64 add/sub/remainderモデルを保ち、Q3-Nのnormal域と保存M／B_normを同じ位置で継承できる。旧MやB_normを再計算・削除せず、`δ_B1≤B_norm+2uM`の適用条件を広げていない。relative Pをcommon joint scalarとして捨てない。

**残る入力契約**：このadapterはevent samplerではない。bitsを保存scheduleと照合しても、与えられた1本のevent列だけからfresh independent bitsによる生成法は証明できない。固定cache／tail・λ・registry・canonical lawの来歴は外側consumerの条件である。旧NumPy条件付き価格cohort samplerを量子shot用canonical samplerへ昇格せず、events_for等を新たに呼ぶ経路も追加していない。任意の外部cfg／event列にℓ=65や平均信号保証を無条件適用するものではない。

## 4. Native RZ単価の構造的接続

旧[reference_core／wrapper](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2638a7845685a119619382b850441f5d8d33713/src/trottertracks/pr_reference_interference_pilot.py#L120)、[Compiler A](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2638a7845685a119619382b850441f5d8d33713/src/trottertracks/track_c_compiler_a.py)、[context／価格資格](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2638a7845685a119619382b850441f5d8d33713/docs/research/pr_reference_bprime_20261107/STRUCTURAL_APPENDIX.md)を再利用する。

本adapterでは、同じF/G・SOURCE inverse・register・CSWAP・Pの位置とwire・X/Y wrapperを既存関数が生成する。追加gateはない。Pは角0でも省略せず、形状は同じ1 parameterである。新旧の違いはm17のPへ渡すfloat角だけになる。

固定compilerは、初段のinverse cancellationでP/RZを対象とせず、definition loweringは同じsymbolic P(φ)→RZ(φ)を使い、最終cancelはCX/Xだけである。角度cutoff、数値commutation、1qubit数値合成や反復最適化を追加していない。従って**同じ既存compiler contextのadapter出力について、whole native RZ数は不変**という既存補題へ構造的に接続できる。新しいcontextは導入しておらず、compile／新価格取得は不要だった。T価格やdepthの不変性を意味しない。

外側consumerでは、旧sampleのcompile前wrapper→固定compiler→compile後のreference register resetという規約を保持する必要がある。本adapterはcompile前のbody／X/Y回路までを返し、reset・backend・追加controlを実装しない。別compiler、追加control、別register、gate再構成、角度依存passを外側で使えばこの移送資格の対象外となる。実m17 wholeの新価格を測ったという主張でもない。

## 5. 判断・保全・mandatory STOP

| 項目 | 今回の到達点 |
|---|---|
| Caller技術実装 | **完了**：保存schedule→旧assembly→m17 B1→旧core／X/Yの薄いadapter。全RPE runner／samplerは未接続・未実行 |
| 全support保証への接続 | **固定入力・旧算術モデル下で接続**。任意event law／cache／wheel形式証明へ拡張しない |
| Native RZ価格保証 | **既存contextで構造的に接続**。外側consumerの固定compiler／reset等を保つ条件が残る。新compileなし |
| V6-P研究結果の採用 | 正式採用はGPT／ユーザー判断。今回、既存の条件付き数値を再計算・実験結果へ格上げしていない |

[V6-Pの保存値](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f672ce6884bd49231366875a9e24a539fcaec7a/docs/research/pr_reference_v6p_results_20261109/REPORT.md)は位相上界5.172494657527835e-08 rad、m17 Reference 630→626 shots/axis、全18round総RZはB′比少なくとも10,360,815,280.326削減という**固定モデル・価格不変条件付き結果**である。今回その整数N・費用を再評価していない。

既存のSOURCE・Q1/Q2/Q3-S/Q3-N結果・raw・grant・ledger・marker・STOPを不変更で保全した。変更は新module・人工test・本REPORTの3ファイルだけ。研究階層ではアルゴリズム改善／新A reference干渉／V6-Qの継続作業で、リソース評価への接続を既存保証で確認した。Git履歴上の親は543b711aであり、同じ作業線の既存branch `codex/pr-reference-v6q3-numeric-20261109`をfast-forwardで継続する。branch名を研究上の親子関係と同一視せず、rename／merge／削除／force pushは行わない。

最新版の正式共通指示`/home/AbeHiromu/projects/research_project_instructions_unified_2026-11-09.md`（特に§10.4）を読んで適用した。共有環境・他利用者・他Trackは変更せず、AI subagentは使用していない。新科学位相評価・m17 F/G／core build・compile・simulation・sampling・RPE／総RZ・Clifford+T評価はすべて0件。

**公開・remote SHA／3ファイル内容一致の照合後mandatory STOP。** 次担当はGPT／ユーザー：条件を明示した既存V6-P結果を研究成果として採用するか、外側consumerへさらに接続するか判断する。今回の薄い接続と固定context資格を確認するための追加科学計算・compileを必須とはしない。別contextが必要となった場合だけ、その局所P／X/Y等の最小scopeと別途承認を固定する。Codexは新科学計算や総T評価へ自動進行しない。
