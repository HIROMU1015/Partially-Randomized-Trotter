# 研究C：PR-aware DF C2-Bridge・保存CAR referenceとの限定科学実行

2026-10-27 JST。Repository：`HIROMU1015/Partially-Randomized-Trotter`。

**固定SOURCEを変更せず、fresh Bridge-run01を一度実行した。47/47 logical units COMPLETE、correctness FAIL・retry・resumeは0。** 両armの全216 atom/K0 event、保存corrected-RTE reference、raw kernel、全26roundのnormalization/signal replayを確認した。Atom平均と保存raw CAR referenceの差をsector36で数値観測したが、厳密Bridge上界、whole-law P0 synthesis、実PCG64のIID性、CA・保証shots/totalは未資格のままである。

## 1. 固定SOURCE・契約・科学条件

| 項目 | 固定値 |
|---|---|
| 科学実行SOURCE | `3f64842d69d54a70a5b736486b5faefa398903b8` |
| Branch | `codex/research-c-pr-aware-df-c2-bridge-execution-20261027` |
| Scope | `C_PR_AWARE_DF_C2_REFERENCE_BRIDGE_V1` |
| 契約 | [固定47-call契約](research_c_pr_aware_df_stage_c2_bridge_contract_draft_20261027.json) |
| 契約SHA256 | `e4f0326c1769f27b615da558af8f48370c6b02fb3192a410d502e8c3ce41ed8f` |
| SOURCE manifest SHA256 | `9dbfe8a3e7886a73fcac79238e4f432f308ae4c7456080e2ad5734fa93c97408` |
| 保存input | 契約の20件。J/U・B1 generators/corrected/ground、C2-B結果・SOURCE/STOP等 |
| 原本 | [Bridge-run01：44ファイル](../../artifacts/research_c_pr_aware_df_c2_bridge/2026-10-27/Bridge-run01/) |
| 技術準備 | [再利用・数学的対応・人工/回帰85 PASS](research_c_pr_aware_df_stage_c2_bridge_synthesis_readiness_20261027.md) |

保存直鎖H4・隣接1.00 Å・STO-3G、通常number CDF rank8のBaseline/Proposed、Nα=Nβ=2/sector36。元frame順LD2、元g補正κ・核scalar一度、partial symmetric S2、δ=.1/s4/h1/40/r16/K0、m0..12、own-DF ground oracle/状態準備除外を維持。CA=.00159 Ha/DF reserveCA/8と共通Gaussian/P0政策は継承metadataで、今回CAや回路費用を再判定していない。

ユーザーのチャット上の明示許可後に実行した。歴史的draft名・UNAPPROVED metadata、旧技術STOPは変更せず、承認記録・承認原文保存・新grantは作成していない。専用worktreeとtask-local cacheを用い、旧A/B1/B2/C1/C2-Bに書き込んでいない。

## 2. 全gateと科学call

| 登録分類 | COMPLETE |
|---|---:|
| Saved JSON/provenance/input/SectorCAR context | 各1、計4 |
| Arm arrays/model-table replay/atom context/atom audit/K0 mean/head reference/kernel bridge | 各2、計14 |
| Round bridge：2arm×13 | 26 |
| Arm summary | 2 |
| Overall summary | 1 |
| **合計** | **47** |

[Journal](../../artifacts/research_c_pr_aware_df_c2_bridge/2026-10-27/Bridge-run01/science_calls.jsonl)の47 attemptと47 COMPLETEはkind/labelごとに一対一。FAIL/inflight0、retry/resume0、[STOP](../../artifacts/research_c_pr_aware_df_c2_bridge/2026-10-27/Bridge-run01/STOP.json)はfailed=false/mandatory_stop=true。Overall statusは`C2_REFERENCE_BRIDGE_NUMERICAL_PILOT_COMPLETE`。

SOURCE82/input20/依存版/保存来歴、保存ground正規化とsector bit mapping、元frame/符号/basis/support/係数・確率表、K0/r16、λ/τ/normalization/identity、216 atomのHermiticity/involutionとtail再構成、全event unitarity/mean恒等式、保存corrected/kernel replayが両arm PASS。**両armのkernel gate完了後だけ**26roundへ進んだ。保存ground/aliasのPASSを再利用し、新しいeigsolveはない。

固定allowanceはcomponent max-entry 1e-10、complex-signal replay 2e-10、normalization 1e-12、mass rel1e-12/abs1e-14、rtol0/global-phase-fitなし。RoundのBridge operator2/Frobenius/signal距離は診断値であり、このcomponent gateとは別欄である。観測値へ事後的なPASS閾値や厳密上界を設定していない。

## 3. Atom・event・kernelの数値対応

下表は各arm内の最大component残差。全桁と個別216件は原本JSONに保存した。丸めを含む数値観測であり、数学的enclosureではない。

| 対応 | Baseline | Proposed |
|---|---:|---:|
| Atom Hermiticity | 1.6132928327e-15 | 1.8377226824e-15 |
| Atom involution | 4.4408920985e-15 | 5.4400928207e-15 |
| 係数和と保存raw CAR tail | 4.1633363423e-16 | 3.8857805862e-16 |
| K0 event unitarity | 1.1102230246e-16 | 1.1102230246e-16 |
| 全atom weighted meanとcos/sin恒等式 | 1.1102230577e-15 | 5.5511162415e-16 |
| Corrected-RTE再生と保存reference | 6.6613381478e-16 | 1.2213241756e-15 |
| Atom平均raw kernelと保存raw CAR kernel | 1.6990318144e-14 | 1.1436831776e-14 |
| 全round normalization power/exp-log経路差 | 4.8616806405e-13 | 8.1408529101e-13 |
| 保存raw signal再生差 | 0（binary64出力の再生値） | 0（同左） |

既存のsigned CAR atomsと`event_unitary`、保存C2-Bのideal atom確率をbinary64へcastした有限平均を使用した。Samplingも16-event全word列挙も行わず、全216 eventを各arm一度ずつ平均した。二つのh/2 headは保存G0/F0/F1を既存ordered exponentialで構成した。核scalarはG0内で一度、tail identityだけを外部phaseへ一度含め、追加の核phaseは掛けていない。

同じfinite-Taylor処理によるcorrected再生と保存Cの照合、`K_atom`対`A_1*C`、`(A_1*C)^n`対`A_m*C^n`、保存own-groundのraw signal再生を区別した。正規化・符号・位相の対応に、登録allowanceを超える不整合は観測しなかった。Signal replayの0は参照計算の誤差が0という意味ではない。

## 4. 全26roundの観測距離

`n=4*2^m`。Operator列はsector36の`||K_atom^n−A_m*C^n||_2`、signal列は同じ差のown-ground complex matrix elementの絶対値。いずれもdimensionless raw単位で、Haではない。Frobenius、normalization経路、保存PF/RTEと条件付きlawの添付値を含む全桁は[26round CSV](../../artifacts/research_c_pr_aware_df_c2_bridge/2026-10-27/Bridge_rounds26.csv)を参照。

| m | Baseline operator2 | Baseline signal | Proposed operator2 | Proposed signal |
|---:|---:|---:|---:|---:|
| 0 | 6.795397e-14 | 6.314985e-14 | 4.599516e-14 | 3.775547e-14 |
| 1 | 1.358249e-13 | 1.261370e-13 | 9.209247e-14 | 7.562150e-14 |
| 2 | 2.715435e-13 | 2.521308e-13 | 1.836851e-13 | 1.512037e-13 |
| 3 | 5.416996e-13 | 5.033920e-13 | 3.663332e-13 | 3.021410e-13 |
| 4 | 1.079248e-12 | 1.003749e-12 | 7.290638e-13 | 6.035656e-13 |
| 5 | 2.140507e-12 | 1.995594e-12 | 1.443868e-12 | 1.204273e-12 |
| 6 | 4.216243e-12 | 3.943490e-12 | 2.843259e-12 | 2.396384e-12 |
| 7 | 8.214310e-12 | 7.699836e-12 | 5.576644e-12 | 4.744542e-12 |
| 8 | 1.560621e-11 | 1.467675e-11 | 1.092110e-11 | 9.298978e-12 |
| 9 | 2.828385e-11 | 2.666257e-11 | 2.095061e-11 | 1.786040e-11 |
| 10 | 4.676274e-11 | 4.399634e-11 | 3.859945e-11 | 3.294378e-11 |
| 11 | 6.531476e-11 | 5.989835e-11 | 6.558466e-11 | 5.604143e-11 |
| 12 | 6.452790e-11 | 5.551124e-11 | 9.468157e-11 | 8.108597e-11 |

最大operator2はBaseline m11の6.531475943733856e-11、Proposed m12の9.468157197482842e-11。最大signalはBaseline m11の5.989835105946907e-11、Proposed m12の8.108596756914505e-11。[2arm集計CSV](../../artifacts/research_c_pr_aware_df_c2_bridge/2026-10-27/Bridge_arm_summary2.csv)。これらの距離の差を、物理精度・総資源の順位や機構の因果効果へ転用しない。

## 5. 得られた資格と残る不足

| 項目 | 状態 |
|---|---|
| 保存J/U→general-K→signed atom→全K0平均とB1 raw reference | 固定sector36/binary64 numerical correspondenceを確認 |
| 符号・frame/spin/bit順・identity/scalar・normalization | 登録gate PASS |
| 全26roundのoperator/signal距離 | 観測値を取得。誤差原因ごとの寄与分解や丸めenclosureは未取得 |
| C2-Bのcategorical law | 保存IID53bit条件付き上界を添付。再計算・資格変更なし |
| 厳密Bridge/reference uncertainty | null。観測距離のままではexact対象間の上界にならない |
| Whole-kernel P0の全law synthesis/shell/actual inverseの一様誤差 | 未資格。今回native回路をbuild/compileしていない |
| 実PCG64 joint streamのIID性 | 未証明。宣言IIDモデルと区別 |
| 費用support・母平均保証・C1 actual CA/保証shots/total | 未資格/null、旧C1 evidence不変 |

C2-Bの約1e-9値は同一exact unitary mapを二つのlawで使う条件付きcategorical項である。今回の観測Bridge距離をそのまま厳密項として足したり、未知synthesis/referenceを0で埋めたりしていない。B1保存PF/RTEは引用だけで、physical Hamiltonian・DF誤差・PF誤差を再評価していない。C1の約83%削減は元の仮定付き点推定のまま保持した。

## 6. 資源・再利用・事後照合

[Preflight](../../artifacts/research_c_pr_aware_df_c2_bridge/2026-10-27/execution_preflight.json)：SOURCE82/input20・契約・保存来歴・fresh不存在をPASS。既存read-only venvのPython3.11.1、NumPy1.26.4/SciPy1.14.1/Qiskit1.3.0等の9依存を照合した。準備時85人工/回帰PASSのSOURCEを変更していないため、同じテストを重複実行していない。

起動時affinity32、MemAvailable 56,315,408,384 bytes（約52.45 GiB）、load average 1.13/1.32/1.50、CPU/memory/I/O pressure avg10/60=0。局所gateは依存順を維持し、両arm gate完了後の2本のround chainを既存resource_planで並列実行した。Round内BLASはjob-localで1threadへ協調し、他段階のlibrary threadsは制限していない。小sector行列でGPU不要、CPU/RAM/worker/wallの人工上限なし。共有venv/cache/OS/他jobの変更なし。

[Resources](../../artifacts/research_c_pr_aware_df_c2_bridge/2026-10-27/Bridge-run01/resources.json)：runner wall 2.5941043104976416秒、process CPU差分40.527394821秒、peak RSS364620 KiB。CPUはnative library threadsを含む累積で、wallやspeedupではない。準備・出版・remote照合時間は含まない。

[事後audit](../../artifacts/research_c_pr_aware_df_c2_bridge/2026-10-27/execution_audit.json)：SOURCE82/input20、旧src/scripts/PR-aware証拠1799件がbyte不変。原本44件のhash、journalの47件対応、2arm/26roundとnull資格を確認した。CSVは生成済みJSONのserializeだけで、科学関数を再実行していない。新fit/optimizer/rank探索/sampling/native build-compile/DF-PF誤差再計算/shots/総費用/C1再解析は0。旧契約、raw、grant/ledger/marker/STOPは保全した。

## 7. 次の担当とmandatory STOP

保存atom/event平均とB1 raw referenceの対応を、元のsector36モデルの数値pilotとして縮小検証できた。厳密reference/Bridge上界や全law P0上界の取得には至っていない。

次は**研究C／ユーザー**が、whole-P0の境界横断1q再合成・Gaussian/phase/referenceの丸めを覆う最小技術作業を依頼するか判断する。**Codex**は別の具体的指示後に対象を絞り、新しい実分子評価は新SOURCE・範囲を固定して明示許可後だけ行う。C2-A、追加sampling、C1 shots/総資源再解析、別分子、重要な詳細科学レビューは今回の実行に含めない。

SOURCE・raw・call/STOP・報告をcommit/pushし、remote SHA/成果物照合後は**mandatory STOP**。既存Bridge-run01への再実行や残枠再利用は行わない。
