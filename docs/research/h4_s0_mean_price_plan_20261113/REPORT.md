# H4 S0固定候補：平均compiled価格の検証計画（2026-11-13）

**m0/qPF1/R2とm6/qPF64/R128に一致するcompiled価格は、調査した公開済みSOURCE・価格索引では確認できない。次は同じH4 preparationに対する閉状態診断（新compile 0）を優先する。閉包が未確認のまま直接価格が必要なら、2候補×normal 2path×X/Y＝最大8 full-wrapper requestの有限pilotを別案とする。** いずれも未実行で、全law平均価格の取得完了とは扱わない。

今回行ったのはSOURCE/identity/domainの静的照合、B V2 metadataの再構成、技術テストと計画作成だけである。新しい実S/平均S、保存数値再解析、sampling、build/compile/transpile、価格fit/予測、C−S評価、N/G、κ/M最適化は0件。S0aは再実行していない。実価格・物理閉包・モデル資格は従来どおり未取得である。

## 固定条件とBへの対応

A正本は[`7da8a35f…:CONFIGURATION.json`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7da8a35f68e9fdc2b46c361dd2c68e2a799b1e75/docs/research/pr_rpe_rmse_s0_results_20261113/run01/CONFIGURATION.json)。Linear H4/1 Å/STO-3G/保存tol-only DF10、S2、L_D=1、ΔPF=0.1、K=2、h=0.1、M=12（m0〜12）、`finite_B_diagnostic`、P=0を維持する。保存数値stateの条件付きground仮定であり、新たな厳密ground証明や十分N・全体RMSEの資格はない。

κの正本は次の文字列をそのまま使用した。表示近似24.30527458を入力へ戻していない。

```text
13112614441153360648422410673503886836581176429267589536059228160/539496659426909122096437389293855970805348475396548904610992201
```

hの元binary64 exact値は`3602879701896397/36028797018963968`。[PRICE_REQUESTS.json](PRICE_REQUESTS.json)に元H/state/snapshot、分割、λの保存出典、phase/scalar契約、compiler、全B identity/lawと13roundのmetadataを保存した。H/state/snapshotのhashと元SOURCEは[SOURCE_MAP](SOURCE_MAP.md)に記載する。binding内の`T=0.8`は継承した元D1参照metadataであり、今回の回路Tは各physical candidate/identityのTである。`profile_is_reference_only`も保持し、参照時刻を回路時刻として使用しない。

| round | physical candidate | T | qPF | R_outer | signed allocation | event位置数 | 保存DIRECT/MODEL | 現mean C |
|---|---|---:|---:|---:|---|---:|---|---|
| m0 | `53e00d8b07d1547d19a7` | 0.1 | 1 | 2 | S2: [2] | 2 | 該当なし / 該当なし | null |
| m6 | `2a8c6718220eec59d419` | 6.4 | 64 | 128 | S2: [128] | 8192 | 該当なし / 該当なし | null |

元A S0 packetは旧`PR_RPE_RMSE_V1`、ID `6cc89360a60b6f5b6554d044193a28f4c92cafa0208e77efb4d1f2ce9f80181c`である。これを新V2へrelabelしていない。公開B `6a9d7fd…`の`generate/a_spec`を読み取り、固定A inputsから**metadataだけ**を新規構成した。指定された最新B `af284d8…`のcontroller/integrationはその固定版とbyte一致した。

再構成したV2 configuration IDは`587d20f7ed906d25841615dc9746674af872675b895bd438c06d3e74abf0e911`。m0のB candidate IDは`RMSE:e9ac70f2efddf2b25d621da1a2ee0f3d43c1248eb15ef8ddabeaae3c840954c8`、m6は`RMSE:a4ed2fac90ddbd6bf8e7d5e0077157a66eade4a657d8db01ebf4ad973781def0`。

これは**Cによる固定Bコードでの暫定再構成**であり、Bが今後公開する正式な固定S0 requestの代わりではない。元physical IDと新B IDを併記し、後からconfiguration全体・identity・lawを完全照合する入口を用意した。Bの未公開作業・checkpointは参照していない。

## 保存価格のcoverageと再利用判断

今回の調査範囲は固定Git objectと、直前Cの[118候補/236軸の公開metadata索引](../bv2_mean_price_validation_design_20261113/PRICE_COVERAGE.json)、その出典であるB coverage、C1/H4 Mq、元有限時間価格SOURCE・Full-H/Track D等である。既存の価格平均・残差・covarianceは再計算していない。private cache/rawの広域探索や回路fingerprintの新規生成もしていない。したがって「調査した公開索引に一致価格がない」という結論であり、未公開cacheを含む全保存領域の不存在証明ではない。

| 既存資産 | 使えるもの | 今回対象への制限 |
|---|---|---|
| C1 H4 q8/16×R4/8 | 元preparation、Accounting、full-wrapper compiler、cohort/role、保存paired C/S・残差のSOURCE | m0/R2、m6/R128に一致しない。係数・保存予測を移植しない。`qualified_for_candidate=false`を保持 |
| H4 Mq/独立q検証 | 同DF10/S2/δ.1/p1/K2、固定R4の方法・保存記録 | PRの直接検証q16まで。PF B0/p7のq32は別回路。q64/R128は未検証 |
| 有限時間H4 direct/探索・確認 | 正確なcandidate/compiler/cohortに結び付いた保存価格 | 旧T/分割/R/lawが異なる。時間又はqR一致だけでは使わない |
| C2準備 | 長q/Rの旧実行案・wrapper/cache等の再利用設計 | 未実行計画であり、m6/R128を含む案の存在は直接価格の存在ではない。40/16案は非実行 |
| S0a run02 | 保存4path・XY8件のStreaming実装parityとinput/law検査 | 元C1の別q/R。単一pathのSを平均S/Cや独立holdoutへ昇格しない。run01失敗/STOPも保全 |
| 旧connected-cluster | 局所1〜3event form、校正→凍結→holdoutの方法 | 主にH4 DF12/H5 DF9・uncontrolled RTE内部。現DF10/full wrapperへ旧係数・閾値を移植しない |
| Track D / Full-H RTE等 | 方法、元compiler・SOURCEへの索引 | DF/回路/対象量が異なる。Strong/Referenceは制御構造比較。Full-H RTEはone-bodyもrandomizeし、今回PR tail-onlyとは別 |

[PRICE_REQUESTS](PRICE_REQUESTS.json)の13行は保存AのPR scheduleとBの対応である。全roundのpriceはnull、mean compiled資格false。m0〜6は旧索引に同じphysical recipeがあるが価格cohortは欠測、m7〜12は新しいκ由来Rで旧索引のrecipe自体が一致しない。後半Rは255/510/1020/2040/4080/8160で、R=2qの単純延長にしない。これらは静的coverageであり、全Gを算出したものではない。m0/m6だけ取得しても残り11roundのCと、全体に適格なN/誤差契約が必要である。

## 平均構造価格の方式比較

目標は**同一law・全回路条件での**E[C]=E[S]+E[C−S]。いずれの項も今回計算していない。[前段METHODS](../bv2_mean_price_validation_design_20261113/METHODS.md)の実装・論点をそのまま再利用する。

| 方式 | 固定H4で必要な証拠 | 計算量と現在の状態 |
|---|---|---|
| 閉状態 `MeanKernel/full_wrapper_mean` | 合法support上でpending basis/Z、tail active/phase、piece/outer継続、scalar/terminalの価格同値類が閉じること。R2とR128のlawを別定義 | dが一定ならO(d³[log R+log q])、kernel memory O(d²)。kernel構成費は別。実d・閉包・物理kernelは未確認 |
| 局所期待値・connected-cluster | 同じlawの局所S、又はpaired D=C−Sと全境界。打切り・非局所compiler残差・移送誤差の独立検証 | 固定窓ならqR全展開を避けられる候補。現H4の係数/境界/誤差は未取得で、未採用 |
| Streaming参照器 | 元event順・binary64・head/tail・scalar・一回finishを維持 | n本でO(n[qR+q×head量])。consumerのevent全保持は不要だが全eventを処理する。S平均とC平均は別 |
| full-wrapper直接C | exact candidate、同trajectory X/Y、元compiler。保存一致があれば再compileせずcopy | 実回路/DAG全体の保持が必要。有限cohort価格を取得できても全law期待値の保証は別 |

### 閉包の可能性と未解決事項

SOURCEでは`extract_identity_phase`によりrandomized supportからidentityを除く。この条件なら非identity K2 eventのapplicationはorder0で1個、order2で3個であり、tail相対phaseはTaylorのπと負productのπの逐次和になる。以前の人工identity-angle反例を実H4の到達例と扱うことはできない。この制約はSOURCE上の確認で、実supportを新計数した結果ではない。

一方、`Accounting`はbasisの完全operations/inverseの一致とwireごとの逐次binary64 Z和を保持し、`angle != 0.0`でflushする。tail内phaseの値・加算順、tail終了のp gate、固定scalar −cTとX/Y wrapperの境界も残る。`global_phase`はSの分岐に戻らない箇所があるが、report/controlled native phaseの意味まで省略してよいとはしない。価格カウンタを加算rewardへ分離できることと、残りの状態が小さく閉じることは別である。

必要なのは、全合法continuationに対して同じ遷移・S増分・terminal rewardを与える同値性、又はq/Rに依存しない一定長flushの証拠である。固定9wire/有限bitのfloatだから小状態になる、phaseをmod 2πにすればよい、S0aの8軸件が一致したから閉じる、とは結論できない。Rを変えるとτ=λΔPF/Rとorder確率/角度が変わるため、R4/8のkernelをそのまま使えない。m0/m6は同じS2の1pieceだが、tail長とtotal T/scalar境界の双方を照合する。

現時点の判定は**物理小状態閉包UNKNOWN、全event生成なしの実E[S]は未実装・未資格**。不成立を証明したわけでもない。近似が必要なら別仮説として校正・検証する。

### 残差と二重計上

C1 Sは既にRTE内部・PF–RTE・outer境界のbasis相殺とliteral Z fold、relative phase、元controlled構成のhead/tail、scalar一回、Hadamard/測定/terminalを含む。元wordはwrapperのHの後、各outerでforward headの`block(False)`→tail→reverse headの`block(True)`を実行し、最後にscalar p(−cT)、Yのsdg、H/measure、finishを各一回適用する。tail/piece/outerごとにAccountingを再初期化しない。旧raw κ(C)をSへ足さない。S基準では同じtrajectory/window/境界のD=C−Sを使用し、**凍結C1 affineか新local Dのどちらか一つ**を選ぶ。local RTE内部だけで全wrapper価格としない。

未知のorder2寄与・sampler対ideal law距離・長q/R systematic・係数/共有covariance・total CIはnull。X/Yは同pathのpaired値で、nを2倍にしない。C1の保存paired統計は元domainのまま再利用候補とする。今回の2点はqとRが同時に変わり、3係数θ0,θq,θLを識別できない。X/Y応答もdesign rowを増やさない。2点を新fitの校正＋holdoutと呼ばない。

## m6の直接価格経路の負荷

元`Backend.price_requests`はrequestsとevent dict列を保持し、元native compose→full Hadamard wrapper→fingerprint/cache→Qiskit1.3.0 compileで価格を取得する。m6は8192event位置、K2の各位置は1又は3applicationで、Gaussian basis等のmetadataも伴う。回路命令列/fingerprint/DAGは展開されたnative長に依存し、qubit数が9でもこの保持量は消えない。Streaming Sだけを導入してもcompile経路全体はstream化されない。

将来はpath単位の独立並列とcacheを再利用し、同pathのX/YとSを共有し、原本を上書きしない。既存Backendは価格のみの経路で、`full_wrapper_statevector_validation=false`を保存している。新しいfull-state検証まで完了しているとは記載しない。SOURCE/input/phase/回路照合の既存条件を維持し、必要な追加検証は別に定義する。RSS/秒/速度向上率は未測定で、m6実行可能性を実証したとは扱わない。資源の任意上限や共有環境の変更は加えない。

## 次の限定科学検証

詳細な有限scopeとSTOPは[SCIENCE_PLAN.json](SCIENCE_PLAN.json)。いずれも未承認・未実行で、科学実行leaf SOURCE/runnerはまだ固定していない。

1. **推奨A：閉状態診断。** 同H4 preparation descriptor 1個、R2/R128のlaw descriptor 2個、S2の1pieceとq1/64の境界を対象とする。sampling/新full-path S/compile/fit/平均価格/N/Gは0。保存preparation/sourceを優先し、必要な合法support/価格状態の物理解析だけを別許可で行う。全continuation同値性を示せなければUNKNOWNでSTOPし、無制限のstate探索や近似採用へ進まない。
2. **代替D：固定二時点の直接価格pilot。** 閉包未確認でも価格参照が必要な場合だけ、normal n=2/候補、同path XY、計4path/最大8 wrapper request。新paired Sも最大8件、C−Sも同じ8件、cohort meanは最大4軸値。既存exact cacheはbinding確認後reuseする。n=2は標本変動を記述できる最小pilotであり、価格精度・全law平均・rare寄与の保証を意味しない。fit/モデル予測/Gは0。sampling seed/cohort、科学leaf SOURCE、check/STOPを実行前に固定し、承認範囲に含める必要がある。
3. **B：平均S近似のdevelopment。** 物理kernel又は局所近似が定義された後に選ぶ。元C1 R4/8の保存S/cohortは方法開発の別domainとして利用できるが、target R2/R128の参照値にはならない。既存summaryを優先し、再集計も別許可。Dを実施した場合はその同path Sを参照に再利用し、新S/samplingを重複しない。有限cohort一致をE[S]の厳密性としない。
4. **C：残差の校正/独立検証。** Dのpaired値はまずdevelopment資料とする。新local Dをfitするなら、PF/RTE・tail/outer・scalar/wrapperを含む窓、識別可能なdesign、別のfresh full-wrapper holdoutを先に固定する。必要ならq-only/R-only controlを別案とするが、今回の二時点や旧C2から自動追加しない。凍結C1仮説を試す場合も指標/予測手順を価格取得前に固定し、MODEL falseを維持する。

Aのnominal Nや旧高確率Nから価格許容誤差を勝手に決めない。全体十分N/採用policy・必要な価格精度が未確定なので、現案Dは取得可能性/有限cohort参照のpilotに限定する。精度保証が必要なら、分散上界/law/信頼policyと必要標本数を別に定義して承認範囲を決める。旧40/16 wrapper、S0a再試行、H6、他path/round、全Gは自動開始しない。

## 実装・テスト・引渡し

実装SOURCEは`bdd2b93565c0994ab5682e131dbccc74912197d3`。[h4_s0_price_plan.py](../../../src/trottertracks/h4_s0_price_plan.py)は固定A reader、B metadata再構成、既存coverage照合、将来の正式B publication照合だけを行う。既存`V2MeanPriceLookup`とB controller/integration・test loaderを再利用し、新しい価格provider/controller/N側を実装していない。

[7技術テストPASS](TESTS.txt)。固定A metadata・人工変異でexactκ/旧新ID/physical recipe、law/制御profile/全scheduleの不一致拒否、26軸価格null、S0a quantity拒否、未資格/共分散null、最新B coreのbyte一致を確認した。N/S/mean/G/selection入口を例外化した準備テストもPASS。sampling/compile/fitの入口は実装・呼出しがないことをSOURCEで確認した。既存12testの結果は前段記録を参照し、同じ回帰を重複実行していない。これは作者自身の技術確認で、独立科学レビューではない。

[HANDOFF](HANDOFF.md)にB正式configurationとのjoinと将来recordの条件を記載した。今回全ての価格値を欠測に保ち、C1係数/資格、C1/C2/connected-cluster、固定SOURCE/input/raw/cohort/grant/ledger/marker/KEEP/STOP、S0a run01失敗/run02成果を変更していない。A/Bの実行中領域/process/cacheへ介入していない。

成果をcommit/pushしremote SHA/tree/主要成果物を照合した後mandatory STOP。次は**ユーザー**がA又はD等の有限科学範囲を選ぶ。**Codex C**は選ばれた範囲の実行leaf SOURCE/手順を具体化し、明示された科学実行許可の範囲だけを処理する。**Codex B**は正式な固定H4 V2 configurationを公開し、Cの暫定metadataと照合する。新科学計算には自動進行しない。
