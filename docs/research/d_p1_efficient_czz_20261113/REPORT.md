# 効率的controlled ZZへの移行とP1 paired pilot（Codex D、2026-11-13）

**状態：D-1（新回路SOURCE）とD-2（技術検証）は完了。P1は、run01のrunner STOPを経て、ユーザーの明示指示によるrun02で完了した（`P1_COMPLETE`、full-wrapper compile 8/8件、4 path）。**

- run01：compileの前に、runnerの技術不具合でSTOPした（compile 0件、retryなし）。記録は保全してある。
- run02：
  - 修正済みSOURCE `20fe1194`で、同じPLAN・seed・lowering・compiler・予算のまま実行した。
  - 実行前に、run01の`events_PRIVATE.json`と、再生成したeventのSHA-256が4 pathすべてで一致することを確認した。
  - phase Aのparityは全件PASSした。
- 結果：
  - 全4 pathで、残差D＝C_optimized−Sは**正**だった。X＝Y。
  - m6（q64/R128）では、Dは約6.5万〜7.9万RZで、Cの3.6〜4.3%にあたる。
- 対象外：P2/P3、fit、全lawのE[C]、N、RMSE、Gには進んでいない。**mandatory STOP。**

## 1. D-1：効率的controlled ZZのSOURCE

[efficient_controlled_zz.py](../../../src/trottertracks/efficient_controlled_zz.py)（lowering ID `CONTROLLED_ZZ_CX_CRZ_CX_V1`）。

- **実装**：provider `c31ef424`の`QiskitDFRTEEventCircuitBuilder`の**subclass**として作った。
  - 上書きするのは、**controlledかつ2-wire支持のrotationだけ**。生成するのは`cx(s0,s1)`、`CRZ(2u)(anc,s1)`、`cx(s0,s1)`。
  - それ以外（1-wire rotation、product/CZ、uncontrolled、検証・例外）は、元のstatic methodへそのまま委譲する。
  - 戻り値（相対phase 0）も元のrotation分岐と同じ。
- **切り替え**：`efficient_controlled_zz(module)`でprovider moduleの属性を明示的に差し替える。例外時も含め、必ず元に戻す。二重の有効化は拒否する。
  - PRの`h46_long_time_native.compose`とFull-H RTEの`h46_full_h_pure_rte`は、どちらも呼出し時にbuilderをimportするので、両方へ届く（ASTで確認）。
- **変更しないもの**：provider・carrierのfile、決定論PF block、K/R/q/law/scalar/wrapper/compiler。
- **新旧の区別**：
  - 新：`CONTROLLED_ZZ_CX_CRZ_CX_V1`
  - 旧：`CONTROLLED_ZZ_GENERIC_RZZ_CONTROL1_PROVIDER_C31EF424`
  - 新回路はwrapper fingerprintが異なるため、旧compile cacheと取り違えることはない（run02のcache hitは0件）。

## 2. D-2：技術検証（人工・小規模）

[TESTS](TESTS.txt)：**44 PASS**（新10件、P0回帰6件、残差adapter 13件、C mean price 15件）。

| 確認項目 | 内容 |
|---|---|
| unitary | 角度±0.142・±0.00224・−1.3、support (0,1)/(2,0)で、旧`RZZ.control(1)`と新形のoperatorの差 < 1e−12（global phaseを含む） |
| 制御分岐 | ancilla=0のblockは単位行列、非対角blockは0。新形のgate数は`cx 2, crz 1` |
| 委譲 | 1-wire rotation、product（±）、uncontrolledの命令列と戻り値は、旧実装と完全一致。不正なsupport・roleでは同じ例外 |
| 有効化 | 差し替え・復元・例外時の復元・入れ子の拒否。旧classは無変更 |
| tail parity関数 | 計画どおりの置換だけを受け入れる（角度符号・他命令の違いを検出） |
| 接続 | PR compose（`155529e2`）とFull-H（`9b83b96b`）で、builderのimportが関数内（呼出し時）にある |
| 旧SOURCE | provider fileのbytesが不変 |
| P1 plan | 決定性、m0/m6・各2 path、seed＝carrier `trial_seed`の再計算、予算8 |
| 技術修正 | 固定Backendが読むplan keyに`source`が含まれることと、run01 planに欠けていたことの再現。空の要約の扱い |

主要guard（角度符号、制御方向、委譲範囲）を無効化すると、testが失敗することを確認した（mutation 3件）。

## 3. P1の固定計画（[PLAN.json](PLAN.json)、科学実行前に`bb475af8`で公開）

| 項目 | 内容 |
|---|---|
| 対象 | 固定H4 DF10、S2、L_D=1、K2、ΔPF=h=0.1、canonical `587d20f7…`。m0：physical `53e00d8b…`（T0.1/q1/R2）。m6：`2a8c6718…`（T6.4/q64/R128） |
| cohort・seed | `D_P1_EFFICIENT_CZZ_20261113`、path 0/1。carrierの`trial_seed`でseedを固定（m0は各1、m6は各64） |
| 取得経路 | 固定C1 `Backend`（`1a87e533`、pinned blobから読込み）の`context`/`price_requests`。保存入力・preparation・providerの照合、carrierの`sample_requests`（PCG64/CDF law）、Accounting S、compose→full wrapper→compile |
| compiler | Qiskit 1.3.0、rz/sx/x/cx、opt1、seed17（settings hash `1239304c…`） |
| 予算 | full-wrapper compileは最大8、trajectory 4。旧回路のfull-wrapper compileは0 |
| gate | phase A（全path）：seed照合、eventの集計、旧・新tailの命令置換parity（全request）、m0はtail全体のoperator一致。phase Aが全件PASSの場合に限り、phase B（compile）へ進む |

## 4. run01：compile前のSTOP（[run01/STOP.json](run01/STOP.json)、[実行log](EXECUTION_RUN01.txt)）

- SOURCE `bb475af8`で実行した。phase Aは全4 pathで通過した（根拠は制御フロー。詳細件数は保存されていない）。
- phase Bでは全4 pathが失敗した。
  - 原因：固定C1 `Backend.price_requests`は`STRUCTURAL.json`に`self.plan['source']`を書くが、P1 planにこのkeyがなく、KeyErrorになった。
  - compileは0件。
- runnerの要約関数も、空集合で0除算を起こした。
- event・seed・lawはprivate `run01/`に保全し、SHA-256を記録した。retryはしていない。
- 技術修正（`20fe1194`、科学条件は不変）：
  - `execution_plan`で、実行SOURCEを`source`としてBackendへ渡す。
  - 固定Backendが読むplan keyを、ASTで事前に検査する。
  - `PHASE_A.json`をcompileの前に保存し、失敗時は要約より先に`STOP.json`を書く。
  - 要約関数は空集合でも落ちないようにした。
- run02の提案：[RUN02_PLAN.json](RUN02_PLAN.json)。

## 5. run02：実行と照合（ユーザーの明示指示による）

**実行条件**

| 項目 | 内容 |
|---|---|
| SOURCE | `20fe11942369253361068dd2b00f98b26350ba8d`（HEAD一致・clean、preflight PASS） |
| PLAN | [PLAN.json](PLAN.json)（SHA-256 `7580ed3b…`、run01と同一） |
| 実行 | phase A・Bとも、pathごとにfork processで並列実行。開始12:47:01、終了13:41:25（wall 54分24秒） |
| 資源 | runner全体のmax RSSは2.93 GB。m6 workerのpeakは2.86 GB / 2.81 GB。共有環境の設定は変更していない |
| 出力 | public [run02/](run02/)（[PHASE_A.json](run02/PHASE_A.json)、[RESULT.json](run02/RESULT.json)）、[実行log](EXECUTION_RUN02.txt)。private `/home/AbeHiromu/projects/d-p1-efficient-czz-20261113-private/run02/` |

**run01との連続性**（[run02_precheck/](run02_precheck/)）

- 実行前（[continuity_precheck.py](run02_precheck/continuity_precheck.py)→[CONTINUITY_PRECHECK.json](run02_precheck/CONTINUITY_PRECHECK.json)、compileなし）：
  - 4 pathのtrajectoryを、runnerと同じ経路で再生成した。
  - Backendが`events_PRIVATE.json`として書く際のbytesのSHA-256が、run01の記録とrun01の保存fileの両方に一致した。seedも一致し、PASS。
- 実行後（[POST_RUN_CHECK.json](run02_precheck/POST_RUN_CHECK.json)）：
  - run02の`events_PRIVATE.json`のSHA-256は、run01と一致した。
  - event digestは、precheck＝PHASE_A＝RESULTで一致した。
  - run01のprivate fileは不変。
  - compileは8件、cache hitは0件、旧回路のcompileは0件。

**phase A（全件PASS）**

| path | request | 置換したcontrolled ZZ | 命令数（旧→新） | parity失敗 | m0 operator差 |
|---|---:|---:|---|---:|---:|
| m0/0 | 1 | 2 | 342→346 | 0 | 0.0 |
| m0/1 | 1 | 1 | 172→174 | 0 | 0.0 |
| m6/0 | 64 | 7,547 | 904,338→919,432 | 0 | — |
| m6/1 | 64 | 7,458 | 906,092→921,008 | 0 | — |

## 6. P1の結果（finite development cohortのみ、[RESULT.json](run02/RESULT.json)）

全pathでX軸とY軸の値は同一（C・S・CX/depthとも一致）。D＝C_optimized−S。

| path | N_ZZ | N_Z | event数 | C_optimized（RZ） | S | **D** | D/C | CX | CX depth | RZ depth | total depth | size |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| m0/0 | 2 | 0 | 2 | 1,906 | 1,845 | **+61** | 3.2% | 710 | 365 | 500 | 1,127 | 3,717 |
| m0/1 | 1 | 1 | 2 | 1,674 | 1,529 | **+145** | 8.7% | 642 | 331 | 444 | 994 | 3,235 |
| m6/0 | 7,547 | 645 | 8,192 | 1,826,433 | 1,747,585 | **+78,848** | 4.3% | 557,546 | 271,556 | 482,432 | 1,081,197 | 3,653,039 |
| m6/1 | 7,458 | 734 | 8,192 | 1,815,119 | 1,750,316 | **+64,803** | 3.6% | 554,536 | 270,985 | 481,312 | 1,078,443 | 3,629,793 |

- order-2 eventは全pathで0。
- 記述統計（4 pathのみ、推定量ではない）：
  - m0：Dの平均103、標本分散3528。
  - m6：Dの平均71,825.5、標本分散98,631,012.5。
- trajectoryの識別情報：
  - event digest：m0/0 `9349ff16…`、m0/1 `1aaca481…`、m6/0 `0c6458db…`、m6/1 `296162cf…`。
  - seedはPLANどおり。X/Yのwrapper fingerprintは[RESULT.json](run02/RESULT.json)に記録した。
- full-wrapperのstatevector検証は、計画にないため実施していない（`full_wrapper_statevector_validation=false`）。

**compile負荷**

| path | X compile | Y compile | price全体 | worker peak RSS |
|---|---:|---:|---:|---:|
| m0/0・m0/1 | 0.1 s | 0.1 s | 3〜4 s | 0.27 GB |
| m6/0 | 241.3 s | 241.8 s | 3,139 s | 2.86 GB |
| m6/1 | 255.9 s | 258.9 s | 3,187 s | 2.81 GB |

m6の1 pathにかかる時間（file時刻からの概算）：

| 段階 | 時間 |
|---|---|
| 構造S | 約6分 |
| compose＋X wrapper（fingerprint計算を含む） | 約22分 |
| X compile | 約4分 |
| 照合とY wrapperの構築 | 約12分 |
| Y compile | 約4分 |
| 照合 | 約3分 |

**律速はQiskitのcompileではなく、固定C1 Backendの中で行うcompose・wrapper構築・fingerprint計算（pure Python）である。** 今回はSOURCEが固定されているため、この部分は変更していない。

## 7. 観察（数値の観測。理論保証やmodelではない）

- **新loweringでも、Dは正のまま残った。**
  - m6のDは、eventあたり9.6／7.9、N_ZZあたり10.4／8.7。
  - 前回のC1残差（旧lowering、q8/16・R4/8の領域）はθL≈18/eventだった。今回はこれより小さいが、領域とloweringが異なるため、同じ尺度での比較とはいえない。
- **旧loweringとの参照比較（旧回路はcompileしていない）。**
  - P0の局所差（controlled ZZあたり+17〜19 RZ）×N_ZZを、旧loweringでのCの参照値とした。
  - 新loweringによる削減は、m6で約12.7万〜14.3万RZ（旧参照値の6.5〜7.3%）に相当する。
  - この参照値に基づくと、旧loweringでのDはeventあたり約23〜27になる。
- **DはN_ZZだけでは説明できない。**
  - m6の2 pathを比べると、N_ZZの差は89、N_Zの差は−89だが、Dの差は約1.4万あった。
  - m0では、N_ZZ=1・N_Z=1のpath（D=145）のほうが、N_ZZ=2のpath（D=61）よりDが大きい。
  - eventの種類と数だけで決まらない成分（basis対の相殺、head/outer境界、wrapper、fragmentとsupportの組合せなど）が残っていることを示唆する。ただし、どの成分が効いているかは、今回の4 pathでは識別できない。
- **Sは安定していた。**
  - m6の2 pathでSの差は0.16%。Cの差は0.62%。
  - path間のばらつきは、主にDから来ている。

## 8. E[S]との接続と資格（変更なし）

- \(\mathbb E[C_{\rm optimized}]=\mathbb E[S]+\mathbb E[C_{\rm optimized}-S]\) を評価するための技術経路は揃っている。今回、そのpaired Dを4 path分取得した。
- CのE[S]はideal finite-K lawに対する期待値である。P1のDは、実際のPCG64/CDF samplerから取ったpath標本である。両者のlawの距離は未資格。
- 全lawのE[C]、十分N、全RMSE bound、実Gは、すべて未資格・未評価のまま。fitも新たな平均価格も作っていない。

## 9. P2へ進むために足りない証拠

- **Dの成分の識別**：basis対の相殺、head/outer境界、wrapper、1-wire Z rotationの寄与を分離できる設計（前回SCIENCE_PLANのP2 ladderなど）が必要。4 pathでは識別できない。
- **N_ZZとDの関係**：m6の2 pathの比較から、Dは単純にN_ZZに比例しない。path数を増やすか、条件を制御した設計が必要。
- **sampler lawとideal lawの距離**：PCG64/CDF samplerとideal finite-K lawの関係。
- **大規模pathの計算負荷**：m6の1 pathに約53分かかり、その大半はcompose・fingerprint（pure Python）だった。P2で大規模pathを増やすなら、X/Y並列化やfingerprint計算の高速化を、別のSOURCEとして検討する価値がある（今回は未実施）。
