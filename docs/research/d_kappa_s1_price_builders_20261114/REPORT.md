# D：Full-H κ2のS-1/P-1実行準備とTruncated PF・Pauli tailの価格接続

2026-11-14 JST、Codex D。**caller・接続・SOURCE照合・人工loader・metadataまでを行った。** 実H4の新mean、sampling、native build、compile、価格取得、N/G、κ・K・decoderの採択は行っていない。

固定SOURCE（caller・接続・test）：**`04bead115233641233ae1fc409b2c4d89ebc2483`**（branch `codex/d-kappa-s1-price-builders-20261114`）。

[次の科学実行・compileの対象とSTOP](NEXT_EXECUTION.md)／[HANDOFF](HANDOFF.md)／[人工test結果](TESTS.txt)

## 作業場所

新branch `codex/d-kappa-s1-price-builders-20261114` を、C `5fb2c32e`（Truncated T2公開head）から作成した。理由は次のとおり。
- S-1は、R448と同じC数値SOURCE（`5cc7c07e` とbyte一致）とC `execute` を使う。`execute` は数値module等のbyteを実行treeで照合する。
- D既存branch（`codex/d-compiled-residual-c-minus-s-20261113`）はB基点で、C pipeline moduleを持たない。共通moduleも別版である。
- C branchへはcommitしていない。D既存branchの成果（κ helper、新ZZ、S1/V1、DF-head Pauli調査）は固定commitから参照・byte copyした。κ helperは `26d4f23e` とbyte一致。

## 作業A：S-1 caller（κ2の13 mean）

[`scripts/run_full_h_kappa2_signals.py`](../../../scripts/run_full_h_kappa2_signals.py)。

- **計画**：固定候補 `26d4f23e:KAPPA_CANDIDATES.json` のκ2を、published B/D/C route（`full_h_rte_spec(kappa)` → `layouts` → `configure` → `I.plan`）で再生成する。configuration（`54ac3eb3…`）全体、13 physical ID、q/R/s、reference descriptorが一致しなければ停止する。R448の13 physical IDとの重複も拒否する。
- **数値**：C数値2 moduleが `5cc7c07e` blobとbyte一致し、R448 callerが `5a172629` とbyte一致することを確認する。そのうえでR448 callerの保存reference readerをそのまま使い、`I.produce_signals`（C `execute`/`evaluate`）を `saved=()` で呼ぶ。新reference・scalar抽出は0。
- **追加gate**：全roundで次を満たさなければ `STOP_POST_GATE_UNRESOLVED` とし、B packetを書かない。
  - 0<logB≤nextafter((λT)²/s)＋`normalizer_one_ulp_log_sensitivity`（既存の1 ULP感度をallowanceとして保持）。
  - ID、q、R、sの一致。
  - B・attenuationとlogBの整合。
- **mode**：既定は**metadata preflight**（入力fileを開かず、数値は0）。`--execute-approved-s1` だけが科学実行で、一回のbatch、最初の失敗でSTOP、retryなし。出力先は `docs/research/d_full_h_kappa2_s1_results_20261114/run01/`。

## 作業B：P-1（κ2 m3、新controlled ZZ）

[`scripts/run_full_h_kappa2_p1.py`](../../../scripts/run_full_h_kappa2_p1.py)。

- **cell**：κ2 m3、T=0.8、q8/R71/K2、`FULL_H_4de6130f0d3a66e5d695`、cohort `D_FULL_H_KAPPA2_NEW_ZZ_M3_WIRING_V1`。8 outerのseedは元Full-H `trial_seed` で再計算し、固定値と一致することを確認する。
- **再利用**：元Full-H script（9b83）をGit blobのまま実行する（`setup/context/prepare_full_h/full_oracle`）。long-time nativeの `sample_requests/compose/cost_trial/compile_cached` とwrapperも使い、D新ZZ（01e068af）のswapは `cost_trial` の範囲だけで有効にし、必ず元に戻す。
- **2 process構成**：人工loaderで、元 `setup()` が要求する `h6_hash_reuse_v1` がこのtreeに無いことが判明した。この依存はprovider root（c31ef424）にしか無い。元の研究と同じく、**子processで `install()` を最初に呼び**、provider rootのmoduleをc31ef424 blobとbyte照合してから実行する。親processはmetadataと価格sidecarを担当する。
- **実行時の検査**：
  - index0の既存gate（low control、leakage、occupation oracle、X/Y Hadamard、compiled vector）。
  - 新ZZ builderが有効であること。旧 `crzz` が0個で、`instruction_parity` がPASSすること。
  - one-body成分が実際にsampleされたこと。
  - scalarがc＋Full-H抽出identityで、ancilla phaseとして一回だけ適用されたこと。
- **sidecar**：price recordとC guard用sidecar/proofを作る（1 pathなので `NOT_A_LAW_MEAN`）。
- **mode**：既定はpreflight。実行は `--execute-approved-p1`（別承認、compile最大2、legacy replayのcompose 1回はcompileしない）。

## 作業C：Truncated PF p7の価格builder

[`scripts/run_truncated_p7_prices.py`](../../../scripts/run_truncated_p7_prices.py)。

**新しいdeterministic builderは作っていない。** Full PFと同じlong-time native path（`native_context/compose/cost_trial` → `build_deterministic_native` → measured X/Y wrapper → 固定compiler）を、B0I・p7のphysical rowで呼ぶ。

- **保持block**：`prepare(ham,7)` はone-bodyとfragment0..6を元順で保持する（native_contextのorder check）。builderも保持block数＝p＋1と順序を検査する。
- **scalar**：c＋`prep.extracted_identity_coefficient`（捨てたtailのidentity）を総Tで一回適用する。実行時には、B0I signalで使った保存prefix7 scalarとの一致も必須とする。
- **signed piece・実幅**：C `configure/prepare` で3構成（`a6e55777…`、`defc5e88…`、`424eecbf…`）を再生成し、T2 PLAN_METADATAのIDと一致させる。T3 cellのm4/m5/m6はpiece数がS2 7/14/28、Y4 9/18/36、M5 11/22/44。early roundは別の実幅を保持する。
- **sidecar**：TRUNCATED_PF、`NOT_APPLICABLE_NO_RTE`、native `H46_LONG_TIME_NATIVE_COMPOSE`、`HAMILTONIAN_CONSTANT_PLUS_B0I_TAIL_IDENTITY`。quantityはFull PFと同じFULL_WRAPPER_MEAN_COMPILED_PRICEである。
- **T3の最大18 wrapper**（3公式×m4/m5/m6×X/Y）は次回の実行候補として固定しただけで、実compileは未承認・未実行。

## 作業D：Pauli tailの価格接続

[`src/trottertracks/pauli_tail_price_connection.py`](../../../src/trottertracks/pauli_tail_price_connection.py)。

- **辞書の受取**：A P0の辞書（4220f3006）を、全round packet（714247619）の `dictionary_identity`（canonical JSONのSHA-256）で照合して読む。184項を再生成・再cutしない。
  - λ_P＝Σ|c|のbinary64 fsum、hex係数、sign、確率、identity除外を検査する。
- **event生成**：component/labelは既存 `PauliSignalAdapter` の規約（辞書順）に従う。eventは既存 `sample_rte_events` で作り、`append_pauli_event` でfull 2^n上に置く。
- **制御とhead**：D2 `central_control_body` で中央回転だけを制御し、native DF head（`build_partial_native(..., controlled_tails)`）へ渡す。各stepはscalar=0とし、c＋s_Rを総Tで一回適用する。
- **区別**：DF tail用factoryや旧DF IDへ偽装しない。DF tail descriptorは拒否する。
- **LD1/m0 pilot metadata**：q1/R3/K2、8 qubit、184項、seed 1個、scalar c＋s_R＝2.7108187593826942。canonical `d905d219…`。
- **未対応**：C guardはこのPauli routeをまだ知らないので拒否する（人工testで確認）。guardの拡張はC担当である。

## 人工・metadata検証

[TESTS.txt](TESTS.txt)：**30 test PASS**。内訳は、S-1 7、P-1 parent 3＋circuit 4、Truncated parent 3＋circuit 5、Pauli 8。

- **S-1**
  - toy 4次元Context（実binding/constant）、toy reference、人工normalizerで13 roundを流し、B packetの生成とenvelope gateを確認した。
  - 1 ULP allowanceが必要な場合とgate違反を確認した。
  - 第一失敗でのSTOPとretryなし、context失敗、候補改ざん、R448 ID混入を確認した。
  - **実κ2のnormalizer・meanは計算していない。**
- **P-1**：toy 4-qubit DF Hamiltonianを元prepare_full_hへ通し、cost_trialを（compileを置換して）end-to-endで実行した。
  - 新旧の全制御行列が一致し（global/control phaseを含む）、low-controlは恒等であり、high branchが独立oracleと一致した。
  - scalarの一回会計、X/Y符号、旧 `crzz` が0個であること、one-bodyの扱いを確認した。
  - C guardへ渡すsidecar形式を確認した。旧loweringとscalar不一致は拒否された。
- **Truncated**：toy DF（3 fragment）でB0I p2の2nd/4th/m5を確認した。
  - 保持block数と順序、scalar（tail identityあり）、signed piece（負時間あり）、index0 gateを確認した。
  - Full PFでは同じpathでscalar＝cだけになることを確認した。
  - 保存scalarの不一致、保持Hの取り違えでSTOPすることを確認した。
  - sidecarは変更なしのC guardを通り、RTE lowering IDを付けると拒否される。
- **Pauli**：toyの全制御行列（ordinary/symmetric、K0/2、非可換productのorder phase）、scalarの一回会計、X/Y符号、seedの再現性、辞書不正の拒否を確認した。
  - 実LD1辞書・packetは読取照合だけ行い、H4回路は作っていない。

作者自身の技術確認であり、独立レビューではない。既存module・C/B/Aのbranchとworktree、実行中のprocessは変更していない。

## 残るUNKNOWN

- S-1：実H4でのκ2のfinite logB、raw/corrected、finite-K bias（実行後に判明する）。
- P-1：実H4のcontext・入力の読込、新ZZ回路のcompile時間とRSS、one-bodyのsample数。子processのprovider root経路はtoyで確認済みだが、実入力では未通過。
- Truncated：実prefix7のtail identityと保存scalarのbit一致（実行時gate）、m6 holdoutの規模、C価格modelの受け方。
- Pauli：実H4でのfull-256 index0の独立oracle（A P0のfull operatorとbit順変換の扱い）、実行caller、C guardのPauli route拡張、価格policyの採択。
- 共通：長時間roundの価格model、law/rare、N/G。

commit/push・remote照合後 **mandatory STOP**。
