# 効率的controlled ZZへの移行とP1 paired pilot（Codex D、2026-11-13）

**状態：D-1（新回路SOURCE）とD-2（技術検証）は完了。P1の科学実行run01は、compileに入る前にrunnerの技術不具合でSTOPした。**

- run01の範囲：4 trajectoryのsamplingと、全4 pathのtail parity（phase A）までは通過した。
- STOPの原因：その後、固定C1 `Backend.price_requests`がplanの`source`欄を要求するのに、P1 planにその欄がなかった。全4 pathが構造Sの記録直後に失敗し、**compileは0件**だった。
- 対応：
  - 部分結果（4 pathのevent・seed・law）はprivateに保全した（[run01/STOP.json](run01/STOP.json)）。
  - 指示§7に従い、**retryはしていない**。
  - 技術修正とrun02案（未実行）を公開した。
- 再実行にはユーザーの指示が必要。P2/P3・fitには進まない。

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
  - 新回路はwrapper fingerprintが異なるため、旧compile cacheと取り違えることはない。

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
| 予算 | full-wrapper compileは最大8、trajectory 4。旧回路のfull-wrapper compileは0 |
| gate | phase A（全path）：seed照合、eventの集計、旧・新tailの命令置換parity（全request）、m0はtail全体のoperator一致。phase Aが全件PASSの場合に限り、phase B（compile）へ進む |

## 4. run01の結果とSTOP（[run01/STOP.json](run01/STOP.json)、[実行log](EXECUTION_RUN01.txt)）

- **phase A：全4 pathでPASS**。
  - 根拠は制御フロー：phase Bは、全jobがseed・parity・operator照合を通った場合にしか開始されない。
  - ただし、run01のrunnerは記録を最後にまとめて書く構造だったため、**parityの詳細な件数は保存されていない**。
- **phase B：全4 pathでFAILED**。
  - `Backend.price_requests`は構造Sを数えた直後に、`STRUCTURAL.json`へ`self.plan['source']`を書く。P1 planにこのkeyがなく、KeyErrorになった。
  - axisの`*.started.json`は0件で、**compileは0件**。Sの値も保存されていない。
- **runnerの第二の不具合**：要約関数が空集合で0除算になり、RESULT/STOPを書けずに終了した。
- **保全**：private `run01/`に、4 pathの`events_PRIVATE.json`（sampling event・seed・law）を保全し、SHA-256を記録した。m6は各約246 MB。これらはPLANのseedから決定論的に再生成される。

## 5. 技術修正（科学条件は不変）

- `execution_plan`：固定PLANに、実行SOURCEを`source`として加えたものをBackendへ渡す（C1と同じ意味）。PLAN file自体は変えない。
- 固定Backendが使うmethod（`__init__`/`context`/`price_requests`）の中で読む`self.plan[...]`のkeyを、pinned SOURCEのASTから求める。不足があれば、実行前にエラーにする（現在のkeyは`menus`/`runtime`/`source`）。
- `PHASE_A.json`をcompileの前に保存する。phase Bに失敗があれば、要約より先に`STOP.json`（traceback付き）を書く。要約関数は、PASSがないroundを`NO_PASSED_PATHS`として扱う。
- [RUN02_PLAN.json](RUN02_PLAN.json)（未実行）：PLAN・seed・lowering・compiler・予算はそのまま。出力先だけ新しくする（run02）。再生成したeventがrun01の`events_PRIVATE.json`と一致することを確認する。

## 6. E[S]との接続と資格（変更なし）

- \(\mathbb E[C_{\rm optimized}]=\mathbb E[S]+\mathbb E[C_{\rm optimized}-S]\) を評価するための技術経路は揃っている。
- CのE[S]はideal finite-K lawに対する期待値である。P1のpaired Dは、実際のPCG64/CDF samplerから取るpath標本である。両者のlawの距離は未資格。
- 全lawのE[C]、十分N、全RMSE bound、実Gは、すべて未資格・未評価のまま。

## 7. P2へ進むための不足証拠

- P1そのもの（run02の実行）。m0/m6のC_optimized、S、Dと、m6のcompile負荷が未取得。
- Dの符号・大きさが、旧版でのC1残差（θL≈18/event）からどれだけ縮むか。P0の値（17〜19 RZ×N_ZZ）を参照値として、P1で比較する。
- 残る両符号の項（basis対の相殺、head/outer境界）について、識別可能な設計（前回SCIENCE_PLANのP2 ladder）と、sampler lawとideal lawの関係。
