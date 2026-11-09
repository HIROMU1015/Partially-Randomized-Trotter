# Q1後の最小検証案：Q2局所readout correctness・native追加価格

2026-11-10 JST。**未実行・別途の明示的実行承認が必要。** Q1の[REPORT](REPORT.md)／[接続条件](FEEDBACK_READOUT_CONTRACT.md)を前提にする。以下は有限科学fixtureによる接口検証案であり、実m17 feedback／全support／全18round費用の正式検証scopeを採択したものではない。

## 1. 問い・SOURCE・保全

問いは「保存B′／Reference coreへQ1 wrapperを付けたとき、有限角度でobservableの位相・符号を保持し、追加native価格を取得できるか」。Q0のshot最適化や既存CT campaignを再実行する問いではない。

基準は `34327b904540556bce7509f2a3168cbff36daab7` と、それを親に持つ本Q1成果commit（実行承認時には完全SHAで指定）。Q1 SOURCEは[新module](../../../src/trottertracks/pr_reference_feedback_q1.py)／[test](../../../tests/test_pr_reference_feedback_q1.py)、旧SOURCE hashは[新SOURCE.json](../../../artifacts/pr_reference_feedback_q1/2026-11-10/SOURCE.json)を再利用する。

| 再利用SOURCE／asset（旧34327b90内） | 用途・制限 |
|---|---|
| `pr_reference_ct1.py` の `sample_rows` と保存 `PREDECLARED.json` | 保存q1/r1/j0、rare0/1のfixture同定。CT stage entry pointを起動しない |
| `pr_reference_bprime.build_saved`／`reconstructed`、transfer `assemble`、pilot `reference_core`／`wrapper` | 同一保存eventから既存raw coreを再構成。PF・law・scalar・F/G・actual inverseを変更しない |
| pilot `raw_prices.tar.xz`／`native_circuits.tar.xz`、CT1 `native_qpy.tar.xz`／`NATIVE_INPUT_CHECKS.json` | 保存X/Y回路・比較metadata・価格を再利用。旧価格が一致contextでなければSTOP |
| `pr_reference_interference_pilot.ground` と既存state-action処理 | 保存groundのendian写像を再利用。新ground solveをしない |
| `track_c_compiler_a.py`、既存Reference reset-prefix処理 | 同一Compiler A／Qiskit1.3、whole jointの既存規約。lower後に追加controlを付けない |

CT1の `stage_c`／whole合成runner、strict guard・word取得・T集計経路は呼ばない。必要なnative-only関数を選んで再利用する。旧SOURCE・raw・STOP・cacheを読取専用とし、別出力namespaceへ保存する。

## 2. 固定fixture候補

既存CT1で固定済みの**q1/r1、j0、rare0/rare1**を使う。新しいtrajectoryをsamplingしない。下表は保存metadataの転記で、今回eventを生成・評価した値ではない。

| fixture | 保存seed | ordered event SHA256 | event |
|---|---:|---|---|
| rare0（nonrare） | 4216572573356262833 | `bc3dfe95d2e57df0d6dcdc323d5876e9053b9122725f918da46b8d8a20f3758b` | order0、rotation `df-fragment-5:Z3Z7`、productsなし |
| rare1 | 4841711708811669773 | `54873b01217f8347d33f424e6aee2e65c77a432d8e499fa5c95cefc1b25b97f5` | order2、rotation `df-fragment-3:Z0Z4`、products順 `df-fragment-3:Z2Z3` → `df-fragment-3:Z1Z4` |

旧 `artifacts/pr_reference_ct1/2026-11-07/PREDECLARED.json` のvacuum phase hexはrare0=`0x1.8fd112eb44b80p+2`、rare1=`0x1.8b84c9c9f67a8p+1`。保存cacheのcompressed SHA256は `85d7e21a41e770596b92327d909f38f0ae091c875e2dad3651abeb660f26ae16`、pickle SHA256は `68269aa0415f048a0220beb4054daa1bba715c6eb29fbbe9d5ba779baf0346f2`。実行時は既存SOURCEとのbindingを確認し、不一致を補修・推定で埋めない。

各fixtureについてB′とReferenceの2 arm。ReferenceはB1 scheduleを保持し、**m0の旧phase経路**を使う（B1の変更対象はm17）。このq1試験をm17 vacuum domainの検証と呼ばない。

有限readout角はQ1人工testにも含まれる次の4個を提案する。既に表現されたbinary64値をparameterとし、実H4の未知m16履歴から生成された角度とはみなさない。

| 名称 | parameter literal | exact binary64 hex |
|---|---|---|
| X照合 | `0.0` | `0x0.0p+0` |
| Y照合 | `1.5707963267948966` | `0x1.921fb54442d18p+0` |
| 一般正角 | `0.213` | `0x1.b4395810624ddp-3` |
| 一般負角 | `-0.719` | `-0x1.7020c49ba5e35p-1` |

**2 paths×2 arms×4 angles＝16新wrapper context**。π/2という数学定数とそのbinary値との差を混同しない。有限角での受渡し・lowering確認は、全φまたは実adaptive feedback角の一様保証ではない。missing assetや不一致があれば他fixtureに取り替えず停止する。

## 3. 必要な限定callと判定

承認する場合の最小call案は次のとおり。科学実行は新namespaceで1 campaign。既存の完了campaignを再開しない。

1. 保存cache／stateを1回読込。保存2event列と4 raw core（2arm×2path）を既存関数で再構成し、上表・旧X/Y metadataとの同一性を確認。新Hamiltonian・PF・trajectory・reference stateを作らない。
2. Q1 wrapperで固定16回路をbuildし、測定前のancilla P/Hの作用と符号を比較。expected coherenceは同じ保存入力・state・phaseにbindingした旧raw expectedを再利用する。bindingできなければSTOP。新たなphase fit・normalization補修をしない。
3. Compiler Aでこの16 contextだけnative compile。Referenceのresetは従来どおりcompile後にprefixし、物理実行順はresetが先。準備費は除外。旧X/Yは保存native QPYを使い、**baseline再compileは0**。保存名は `rare{0,1}_{Bprime,reference}_{X,Y}` を照合する。
4. 同じ保存ground入力で16個のnative期待値を既存state-action処理により評価し、`Re(exp(−iφbin) z_saved)` と直接比較。raw／native間の位相・符号、φ0と旧X、φbin≈π/2と旧Yを照合。symbolic P/H作用は小さい局所operatorで確認し、実17-qubit全operator構築を要求しない。
5. 各16回路のRZ／CX／size／native DAG depth／qubitsを取得し、保存X/Yとの差を別々に記録。raw PがどのRZ・global phaseへlowerされたか、cancel／fusionの有無とcontextを記録する。新価格の有限per-shot表だけを保存し、平均全law価格や総費用へ外挿しない。

旧native validationの `1e−10` 浮動小数点照合規約を有限信号・局所operator検査に再利用する。実行前にそのSOURCE上の規約との一致を確認し、別条件なら変更せず停止する。この数値PASSを全supportのoperator norm誤差保証、Q0上限に対する厳密なδφ証明とみなさない。保存state以外の任意入力へは拡張しない。旧CT1のfull joint probeを、Reference resetを含む物理RPE入力へ読み替えない。

φbin=π/2のケースでは旧Sdgとbinary Pの差がある。数学的なY対応、binary渡し、数値比較の許容を別々に報告する。全jointのcommon global phaseは信号不変だが、controlled branchのrelative phaseは保持する。数値phase fitによる合わせ込みを禁止する。

## 4. 誤差・費用へ進む条件

このQ2で得られるのは**有限core・有限角でのreadout接続correctnessとnative価格**。実m17角生成・prefix受理確率・全support readout誤差・価格移送は残る。実m16測定履歴はまだなく、E0や保存真signalを使ってそれらしい履歴を作らない。

実装移送には `δarith+δparameter/lowering/provider ≤ .000748774257 rad` を満たす角度誤差保証が必要。common phase以外の一般joint／raw diskが生じた場合、radへ直接足さず、保存bias／radiusとの正しい単位での再接続が別途必要。旧diskと新追加分を区別し、2度chargeしない。Q2のfloat差だけでこの保証を採用しない。

費用は資源jごとの `340 μQ,j+Aj < 626 μX,j+626 μY,j` が最終判断条件。Q2の2path価格はE_law[価格]ではなく、m17や全18roundに掛けない。Ajには必要な校正・feedback費を残し、0とみなさない。depth・peak qubits・classical latency・Tは互いに換算しない。B′にも同じ16context内の機会を与え、Referenceだけを改善した対照にしない。

## 5. 継続判断・STOP・担当

承認が必要なのは§2〜3の保存科学入力読込、新core／wrapper build、16回のCompiler A、16固定科学期待値の評価、有限native価格取得である。**今回Q1ではすべて未実行**。

次のいずれかでmandatory STOPする。

- 固定SOURCE／cache／event／core／保存expected／baseline contextが適合しない、または必要な値が欠ける。
- 指定correctnessの不一致、relative phase不一致、未知のlowering／reset問題がある。
- 新word／precision／sampling／別compiler／別物理条件等への変更が必要となる。
- 有限16contextと検算が完了し、結果・SOURCE・STOPの公開・remote照合が終わる。

失敗から角度・trajectory・toleranceを追加探索しない。Q0再実行、実m17 whole回路、全support列挙、追加sampling、shot再最適化、全18round会計、Clifford+T／strict guard／総Tへ進まない。

**次担当：GPT／ユーザー**がQ2案の採否と、prefix受理保証・実feedback誤差の資格を次のどのscopeに置くか判断する。採用する場合はQ1完全commitと上記有限callを指定して実行承認する。**Codex**はその承認を得るまでQ1のmandatory STOPを維持する。Q2を完了しても後続scopeへ自動進行しない。
