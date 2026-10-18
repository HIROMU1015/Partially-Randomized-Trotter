# D01〜D10：最終静的照合結果

[入口](README.md)。基準は[P1 checklist][CHECK]、旧`c8cbeb18...`から新`e4af6c06...`への公開差分のみ。[machine判定](delta_inventory.json)の分類は技術監査上の状態で、数値correctness・実行許可・科学的GOではない。

| ID | 分類 | 新公開版で確認した内容 | 保留・差分／次の扱い |
|---|---|---|---|
| D01 | `ADDITIONAL_BINDING_REQUIRED` | 新driver SHA、旧6 frozen pins、補助SOURCE・initの同一性。CLIはpreflightのみ、`execute→Boundary.enter→load_modules`の順。候補未採択・angle未定・外部portなしでscience import前に拒否 | 外部`admit_once`のatomic永続claim・正式契約/source-release bindingはinterfaceのみ。mock portはauthorityではない。F01修正も別途必要 |
| D02 | `MATCHED_WITH_SCOPE_LIMIT` | P5 input・σ=+1、4方式、16 cells、保存N、H160/K256・η/ρ・旧confidence/bias/lawを継承。`load_saved`でraw SHA/input/N/αを確認しbudget再評価なし | G9のClosed P5/CTS・8ρ/22 axes/resource配分は別。保存Nをworkload長に使うことと、この固定SHA traceのIID/信号confidenceを実証することは別 |
| D03 | `ADOPTION_PENDING` | Layer A=frozen generator、B=Full旧F3＋canonical audit。Meter粗stageとFunctionObserver細profileを別々にouter totalへ照合。baseline/instrumented二pass。CPU/wall/memory/output座標を分離 | 候補は固定workloadの記述的費用。母期待μのcertificate・重み付き総費用は未採択。観測差を純observer overheadとして差し引かない |
| D04 | `ADOPTION_PENDING` | constructor24、Full prepare3（内包constructor）、canonical preflight9という公開候補。X/Y state共有、各pair4 identity checks、arm/layer/phase-pass独立child。新law/packet/angle memo無効、全trial hash/first16保持、stream/hash/flush | 新seed、二pass、共有/保持/保存scheduleは正式採択前。OS cacheはuncontrolled。failureのinflight記録にF01。G9参照CPUは移植不可 |
| D05 | `ADOPTION_PENDING` | Aはratioのみ、BはacceptedごとのF3 enclosure。Fullの旧event→packet再query、canonicalの原event再queryを維持。`validate_angle`はmath-only modeを要求 | τ/Kφ/Jはnull・defaultなし。物理RZ代表値適用は拒否。固定H/Kや保存RZ εから3値を自動決定しない。実radius成立性は未検証 |
| D06 | `MATCHED_WITH_SCOPE_LIMIT` | 公開sample予算：correctness256、baseline15,855,296、instrumented15,855,296、total31,710,848。保存N load、固定loops、accepted依存angle/requery、retry0/resumeなし、first correctness込み一束 | call数は候補の引用で、新N計算ではない。fresh claim外部実装・承認未発行。failure記録F01を解消する必要。未出現pathを追加samplingで補わない |
| D07 | `NOT_APPLICABLE_TO_B3_SCOPE` | B3は古典費用。物理Q/providerを選ばず、G7抽象CQ仮定を保存。新driverにnative実行経路なし | 新C1へG9明示3-qubit/I1/providerをbindingする選択は別。B3実行のためだけにprovider選定・native compileを要求しない |
| D08 | `MATCHED_WITH_SCOPE_LIMIT` | 補助SOURCE/旧provider価格schemaは不変。新driverはRZ cache/compileを呼ばず、19 key・18 G7来歴の前回再利用分類は変わらない | G9 event価格→新primitive schema、phase/actual adjoint/tool/provider等のbindingは依然未実施。metadata一致は物理再認証ではない |
| D09 | `CONTRACT_DIFFERENCE` | B3は4方式の固定workload CPU/wall/setup/online/save/R-C-A/observer。G9は固定provider/native T/CX/1Q/workspaceの別契約・期待会計 | 異なるN/confidence・価格粒度のまま総量を統合不可。Closed P5/CTSの監査込み全trial古典費用、depth、C2等は未取得 |
| D10 | `ADOPTION_PENDING` | 新engine・人工tests・公開記録と、候補未採択/未認可/実入力未実行を分離。旧claim不使用。`Boundary`は成功/失敗ともSTOP | F01修正→新SOURCE固定→具体条件採択・正式契約→外部admission固定→別途科学承認が必要。今回これらを代行しない |

根拠：D01/D02は[DSRC][DSRC] `candidate_errors/load_saved/Boundary/execute/main`、[DC][DC] `/source_pins,/input,/policy,/cells`。D03/D04は`Meter/FunctionObserver/LegacyEndpoint/run_pair/isolated_pair/Stream`と[測定仕様][DMEASURE]・[angle/cache仕様][DANGLE]。D05は`validate_angle/LegacyEndpoint._angle_for`、D06は[schedule][DSCHED]・DC `/call_budget,/one_shot`、D07〜D09は[前回crosswalk][PCROSS]・[再利用表][PREUSE]と新source差分。D10は[preflight保存記録][DPRE]・[handoff][DHAND]。人工testsは[原本][DTEST]を静的参照しただけで再実行していない。

## F01：失敗したtrialのevent文脈が前trialを指し得る

[DSRC 516行][DSRCFAILSTART]で`result = endpoint.sample(...)`を代入し、[572〜573行の分岐][DSRCFAIL]は`'result' in locals()`なら`result.event`を`inflight_event`として保存する。trial開始時に`result`を消去していない。

従って、先行trialが成功した後、次trialの`endpoint.sample`が**return前に例外**を出すと、`result`は先行trialのまま残る。失敗した現trialのeventが得られていないのに、そのeventとして誤表示し得る。XからYへの移行時にも変数寿命は続く。これは静的なPythonデータフローの指摘で、実科学失敗を観測した主張ではない。

**必要な最小修正案**：各attempt開始時にcurrent resultを未取得へ初期化し、axis/trial_id・取得済みか・最後に保存完了した位置を区別してfailureへ記録する。成功後の次sample例外・axis移行直後例外・angle/save例外で前eventが現trialへ混入しない人工回帰を追加する。今回SOURCE変更・そのテスト実行は行わない。別途依頼されたCodexが新versionで修正・固定すべき技術課題である。

加えて、`digest.update`は536行、`stream.emit`は547行、完了counter加算は548行である。保存例外時の`trace_sha256`は生成済みinflight eventまで含み得る一方、counterは保存完了prefixを指す。これを一つの「保存完了prefix hash」と解釈せず、trace・committed JSONL chain/file SHA・inflight文脈のdomainを明示する必要がある。streamのcommitted hashは別に保持されており、失敗を正常完了へ格上げしない。

## 外部admissionの残る責務

`Boundary.enter`はportからのreceipt fieldを検査するが、atomic排他・永続消費・ユーザー承認の根拠・source release commitの照合実体はport/launcher側である。`source_release`等が非nullというだけでそれらを実装済みとはしない。

また`execute`はadmission後にSOURCE/raw/moduleを検査し、その後に出力領域をexclusive作成する。`target`未作成時の例外ではdriver自身のfailure fileは書かれない。正式launcher/admissionは、**消費済みclaimと、出力作成前の失敗情報**も保全する必要がある。これは既存設計の外部binding要件であり、今回新しいgrant/claimを作って補わない。

静的照合を完了しても、F01・外部binding・採択待ちが残るため「即実行可能」とはしない。**mandatory STOP。**

<!-- FIXED_REFERENCES -->

[CHECK]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/568bf486805dfbd2fd4a13ef5a25771351981ce2/docs/research/track_b_g7_g9_p1_public_crosswalk_20261019/p1_final_delta_checklist.md
[DANGLE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e4af6c064db805cbec611bcc8f0ff75a4754d38b/docs/research/track_b_g6_w2c_b3_driver_preparation_20261019/angle_cache_output_contract.md
[DC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e4af6c064db805cbec611bcc8f0ff75a4754d38b/docs/research/track_b_g6_w2c_b3_driver_preparation_20261019/execution_contract_candidate.json
[DHAND]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e4af6c064db805cbec611bcc8f0ff75a4754d38b/docs/research/track_b_g6_w2c_b3_driver_preparation_20261019/handoff.md
[DMEASURE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e4af6c064db805cbec611bcc8f0ff75a4754d38b/docs/research/track_b_g6_w2c_b3_driver_preparation_20261019/driver_and_measurement.md
[DPRE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e4af6c064db805cbec611bcc8f0ff75a4754d38b/docs/research/track_b_g6_w2c_b3_driver_preparation_20261019/preflight.json
[DSCHED]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e4af6c064db805cbec611bcc8f0ff75a4754d38b/docs/research/track_b_g6_w2c_b3_driver_preparation_20261019/comparison_schedule.json
[DSRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e4af6c064db805cbec611bcc8f0ff75a4754d38b/src/trottertracks/algorithm_codesign/g6_w2c_b3_driver.py
[DSRCFAIL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e4af6c064db805cbec611bcc8f0ff75a4754d38b/src/trottertracks/algorithm_codesign/g6_w2c_b3_driver.py#L572
[DSRCFAILSTART]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e4af6c064db805cbec611bcc8f0ff75a4754d38b/src/trottertracks/algorithm_codesign/g6_w2c_b3_driver.py#L516
[DTEST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e4af6c064db805cbec611bcc8f0ff75a4754d38b/docs/research/track_b_g6_w2c_b3_driver_preparation_20261019/artificial_test_results.json
[PCROSS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/568bf486805dfbd2fd4a13ef5a25771351981ce2/docs/research/track_b_g7_g9_p1_public_crosswalk_20261019/b3_c1_g9_identity_crosswalk.md
[PREUSE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/568bf486805dfbd2fd4a13ef5a25771351981ce2/docs/research/track_b_g7_g9_p1_public_crosswalk_20261019/native_evidence_reuse_matrix.md
