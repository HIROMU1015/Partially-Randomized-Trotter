# 個別所見（2026-11-14 総合監査）

[REPORT](REPORT.md)の各所見の詳細である。区分は、S＝科学的正しさ、E＝計算効率、C＝実装構造、I＝統合上の問題、V＝検証不足。
根拠の種類は【実測】【静的】【既存記録】【推定】で示す。実測の原本は[evidence/](evidence/)にある。

branchの略称：A＝`codex/pr-r-dependent-n-model-20261111`、B＝`codex/rpe-decoder-statistics-20261113`（B本線`codex/r-variable-rpe-cost-pipeline-20261111`の後継）、C＝`codex/r-variable-compiled-price-design-20261111`、D＝`codex/d-compiled-residual-c-minus-s-20261113`。

## 0. end-to-end経路（G＝Σ N·E[C] までの実装）

| 段階 | 現行の実装（branch） | 依存する固定SOURCE | 同じ計算の別実装 |
|---|---|---|---|
| 物理入力（H、DF10、state、sector） | `h4_pr_physical_inputs`（A）、`h46_m5_tol_saved_input.load` | carrier `155529e2`、私的snapshot（SHA照合） | — |
| tailのscalar・λ_R | `h46_new2_core.matrix_only_tail_coefficients` | carrier、`df_rte_tail` | provider `extract_df_diagonal_tail`（Dの診断で値がbinary64一致） |
| PF/PRのsignal | `pr_rpe_signal_provider`（A、direct expm＋行列累乗） | `rte.finite_taylor_operator`、`finite_rte_distribution`（`c31ef424`） | `h46_long_time_signal.evolve_batch`、`h46_new2_core.compare_prefix`、`h6_df19_rpe_d2.evolve`、`h4_df10_rpe_batch_provider.evolve_h4`、C `baseline_signal_numerics`、A `pf_effective_hamiltonian` |
| event数・R | `pr_rpe_rmse.paper_event_mapper`（R＝max(J, ⌈⌈κλ²T²⌉/q⌉)） | `h46_m5_plan.allocation` | — |
| shot数N | nominal：`pr_rpe_rmse.nominal_axis_shots`。risk：`rpe_deterministic_risk`（B）、MC：`rpe_decoder_statistics`（B） | decoder `rpe_four_round_phase_validation`、`rpe_decoder_fast` | 旧`u_aware_shots`（高確率型、別方針） |
| 構造価格E[S] | `h4_structural_kernel`（C、区間付きMarkov kernel） | `h4_df10_price_v1.Accounting`（`8457c443`） | path単位S：C1 `structural`、D `compiled_price_fast.structural_pair` |
| compiled価格C | D `candidate_price_adapter`＋C1 `Backend.price_requests`（`1a87e533`） | `df_rte_qiskit`、`ax2b_native_df_v5`、`rte_compiled_cost`（`c31ef424`）、D `efficient_controlled_zz` | `pf_common_run_native.cost_one`、`h46_long_time_native.cost_trial`、`h46_full_h_pure_rte` |
| 会計・選択 | B `pr_rpe_rmse_controller`、`rpe_shot_allocation`、C `rpe_mean_price_model`、`rpe_optimization_pipeline` | `pf_pr_rpe_pipeline`（3版） | — |

正式なGは、どのbranchでもnullのままである（N・価格とも未資格）。この扱いは一貫していた。

## S. 科学的正しさ

### S0. 中核計算の照合結果（不具合なし）

| 照合対象 | 定義 | 実装 | 結果 |
|---|---|---|---|
| S2 1 step | 先頭blockから順に半step、tail全step、逆順に半step | `evolve_batch`、`d2.evolve`、A `_macros`、C `pf_step` | 4実装とも同じ順序【静的】。人工行列で総当たりと一致【実測】 |
| 符号付きpiece（4次） | t＝T·w/q、wは負もあり得る | normalizerは\|τ\|、回転角はτの符号を保持 | 負のτでも event lawの恒等式が成立【実測】 |
| 有限K RTE | Σ_{k偶数≤K} (τ^k/k!)·√(1+(τ/(k+1))²) をbとし、E[U]＝poly_{K+1}/b | `_paired_order_weight`、`_make_event`、`polynomial` | 全event列挙と一致、差 ≤ 2.9×10⁻¹⁴【実測】 |
| raw／corrected | corrected＝raw·B、log B＝Σ q·count·log b | 全実装で独立に伝播し、closureを検査 | 総当たりと一致【実測】 |
| scalar | exp(−iT(c_H＋s_R)) を全時間の後に一回 | 数値側は全実装で一回。回路側はancillaの`p(−scalar·T)` | 数値側は一致【実測】。回路側はI4参照 |
| event数 | r*＝κλ_R²T² | `paper_event_mapper` | 式どおり【静的】 |
| nominal N | ⌈e^{2 log B}·(11＋4(M−m))⌉ | Decimalの二精度でceiling | 式どおり【静的】 |
| 測定確率 | p＝(1＋raw)/2 | `EXACT_AFFINE`は有理数で厳密 | 式どおり【静的】 |
| PF係数 | Σw＝1、Σw³＝0 | Yoshida4、M5 best | 成立（Σw³は10⁻¹⁵以下）【実測】 |

- 検証script：[RTE恒等式](checks/check_rte_mean_identity_artificial.py)、[PR平均kernel](checks/check_pr_mean_kernels_artificial.py)。
- 限界：回路（native emitter、wrapper）と数値発展の作用素一致は、今回は再検証していない（V4）。

### S1. Full-H pure RTEの価格と新loweringの不整合 — 重要度：高

- **対象**：D `full_baseline_v2_requests.py`（Full PF／Full-H RTEの価格request）、保存Full-H価格（`9b83b96b`、`7a8bc7f1`）、D `candidate_price_adapter.py:86`、B `rpe_shot_allocation._price_metadata`、C `rpe_mean_price_model.context`。
- **根拠**【静的】【既存記録】：
  - PRの新しい価格取得（D）は`CONTROLLED_ZZ_CX_CRZ_CX_V1`を強制する。
  - Full-H RTEのrequestにはcontrolled ZZのlowering欄がない。保存Full-H価格は旧lowering（汎用control、CCX経由）で取得されている。
  - compiler設定のhash（`1239304c…`）は両loweringで同じなので、hashでは区別できない。
  - Bの`_price_metadata`は、configurationに`lowering_id`がない場合（現行のcanonical configurationはこれに当たる）、価格側のextensionがどのloweringを名乗っていても受け入れる。extension自体がない場合は`MISSING_PRICE_PROFILE_EXTENSION`を記録する。Cは明示のlowering profileを必須にしている。
- **研究への影響**：Full-H RTEはtailがすべてcontrolled回転なので、旧loweringの影響を最も強く受ける。P0の実測では、controlled ZZ 1個あたり+17〜19 RZである。旧価格のFull-H RTEを新価格のPRと比べると、baselineを不当に高く見積もる。決定論PFには影響しない。
- **改善案**：
  1. RTE tailを含む価格record（PR、Full-H RTE）すべてに`lowering_id`を必須にする。旧recordには`CONTROLLED_ZZ_GENERIC_RZZ_CONTROL1_PROVIDER_C31EF424`を明示的に付ける。
  2. 4方式比較に入れるFull-H RTEの価格は、新loweringで取り直す（要承認の科学計算）。取り直すまでは比較表で旧lowering値と明記する。
  3. loweringをcanonical configurationの識別に含めるかどうかを、B/Cで統一する。

### S2. 決定論的RPE risk上界が実用的な情報を持たない — 重要度：中

- **対象**：B `rpe_deterministic_risk._bound`、結果`pr_ld1_ld2_deterministic_risk_results_20261113`。
- **根拠**【既存記録】：上界は「all-good終端寄与＋P(any unsafe)×L_bad」で、L_bad≈1127.77 Ha²（主値範囲いっぱいのloss）。公開結果はLD1で約0.2399 Ha²（平方根0.49 Ha）。all-good寄与は約4.7×10⁻⁷ Ha²なので、上界のほぼ全部がunsafe項である。計算自体は式どおりで誤りはない。
- **原因**【静的】：
  1. unsafe（安全集合の外）を、すべて最大lossで評価している。
  2. zero座標・tieの履歴は推定値を持たず（energy＝None）、同じ最大lossを当てている。
- **研究への影響**：この上界では「十分N」を資格化できない。Bは11/14に状態集約DPで改善を始めている（`rpe_mse_state_refinement_20261114`、実H4へは未適用）。
- **改善案（解析的な補助上界、未検証の提案）**：
  - 元decoderは、round mで前の推定値に最も近い候補を選ぶ。候補は2π/2^m間隔なので、測定値が何であっても \|θ̂_m−θ̂_{m−1}\| ≤ π/2^m が成り立つ（円周上の距離）。
  - round m−1までがgood（\|θ̂_{m−1}−θ\| ≤ ρ_{m−1}/2^{m−1}）で、round mで初めてunsafeになった場合、最終推定値は \|θ̂_M−θ\| < (2ρ_{m−1}＋2π)/2^m を満たす。
  - したがって、その履歴のenergy lossは ((2ρ_{m−1}＋2π)/(2^m h))² 以下になる。lossはroundが進むごとに1/4になり、unsafe確率の大きい後半roundほど小さい。
  - 必要な条件：(a) zero・tieでも推定値を返す推定量の定義（例：前roundの推定値を引き継ぐ）、(b) binary64の丸めに対する余裕、(c) 主値の折返しが起きない範囲の確認（前半roundでは従来の最大lossを使う）。
  - 計算量はO(M)で、DPの状態数に依存しない。
- **注意**：保存結果への適用は数値再解析にあたるため、今回は式の提示にとどめた。数値はBが承認のうえで評価する。

### S3. 長時間のspectral経路でunitarity誤差が蓄積する — 重要度：低

- **対象**：`h46_new2_core.spectrum`／`spectral_apply`（carrier）、A `pf_power_stability`。
- **根拠**【既存記録】：T＝409.6、q＝1024の1セルで、spectral経路のoperator unitarity誤差は1.0×10⁻⁹（固定gate 10⁻⁹に不合格）、direct expmは5.9×10⁻¹²。complex meanの経路差は4.3×10⁻¹¹。
- **研究への影響**：signalの差は10⁻¹¹で、科学的な結論には影響しない。ただし、このgate不合格でbatchが止まり、診断と再承認に1往復を要した。
- **改善案**：新規の数値gateは、科学量（signalの誤差）の許容値から決める。作用素normのgateはq（step数）とともに増えるので、長時間roundではqに応じた閾値にする。新規作業ではdirect expm＋行列累乗（A `pr_rpe_signal_provider`）を使う。

### S4. energy targetの範囲（明示事項）

- 比較の対象は「保存DF10 Hamiltonianの同sector最低固有値」（保存solverのproxy）である。分子Hamiltonianに対するDF打切り誤差は、どの方式の誤差予算にも入っていない。全方式で同じHを使うので比較の公平性は保たれるが、論文での主張は「DF10 Hamiltonianのenergy推定」に限られる。
- AのH_eff診断（Full PF S2、Δ＝0.1）では、行列の最低固有値とproxyの差は8.9×10⁻¹⁶ Haだった【既存記録】。

## E. 計算効率

### E1. `candidate_id`が呼出しごとにSOURCEをparseする — 重要度：高（効果大・変更極小）

- **対象**：`src/trottertracks/pf_pr_rpe_pipeline.py:candidate_id`（A/B/C/Dの3版すべて同じ実装）。
- **根拠**【実測】：
  - 1回7.1 ms。parse結果を保持すれば0.0047 ms（約1500倍、出力は同一）。
  - 遅い回帰テスト11本をprofileした結果、合計808秒のうち623秒（77%）がこの関数だった。

    | テスト（branch） | 全体 | `candidate_id` | 呼出し回数 |
    |---|---:|---:|---:|
    | `check_rpe_r_price_intake`（B） | 139.4 s | 126.1 s（90%） | 16,676 |
    | `check_h4_s0_price_plan`（C） | 88.7 s | 79.1 s（89%） | 10,380 |
    | `check_rpe_s3_intake`（B） | 84.6 s | 72.0 s（85%） | 9,152 |
    | `check_pr_rpe_rmse_pilot`（A） | 49.7 s | 42.0 s（84%） | 5,920 |
    | `check_pr_rpe_rmse_s0_v2`（A） | 60.0 s | 49.7 s（83%） | 6,608 |
    | `check_rpe_new_evidence_intake`（B） | 79.6 s | 62.5 s（79%） | 7,747 |
    | `check_rpe_provider_integration`（B） | 46.8 s | 37.2 s（79%） | 3,260 |
    | `check_rpe_signal_statistics_caller`（B） | 47.2 s | 37.3 s（79%） | 3,172 |
    | `check_rpe_d_price_connection`（C） | 57.2 s | 42.1 s（73%） | 5,363 |
    | `check_h4_pr_rmse_completion`（A） | 82.3 s | 58.4 s（71%） | 4,570 |
    | `check_pr_r_detailed_n_p1`（A） | 72.8 s | 16.6 s（23%） | 2,199 |

  - testだけでなく、productionの`_validate`（B controller、C pipeline等）が同じ経路を通る。
- **現在の計算量**：呼出し1回ごとに、約23 KBのfileの読込みとAST parse。
- **改善案**：module内でparse済みの`cid`を1回だけ保持する（file内容が変わらない前提は現在と同じ）。数行の変更。
- **科学的再検証**：不要（同じ関数を同じ入力で呼ぶ）。IDの一致は既存testで確認できる。
- **研究上の利益**：各担当が作業のたびに回す回帰（1 branchあたり直列で約5〜10分）が大きく短くなる。前回監査のR2と同じ内容で、未対応のままである。

### E2. X軸とY軸のfull-wrapperを別々にcompileしている — 重要度：高

- **対象**：C1 `Backend.price_requests`（`1a87e533`）、D `candidate_price_adapter`、`ax2b_native_df_v5.build_native_hadamard_wrapper`。
- **根拠**：
  - 【既存記録】D P2のm6 replay（1 path、857秒）で、Y側のwrapper構築・fingerprint・mutation検査・compileは計318秒（37%）。
  - 【静的】X wrapperとY wrapperの違いは、回路の最後でancillaに`sdg`が1個入るかどうかだけである。opt1のpassは1q runの再合成と隣接CXの相殺で、最後のancilla runの後には測定しかない。
  - 【実測】人工回路2000個（研究と同じcompiler設定）で、compile後のXとYは、ancillaの最後の1q runを除いて全wireで同一だった。RZ数の差は−1/0/+1のみで、CX数は全件一致した（[script](checks/check_xy_wrapper_locality_artificial.py)）。
  - 【既存記録】P1（4 path）とP2（fresh 8 path）の保存結果では、全pathでC_X＝C_Yだった。
- **改善後の見込み**：片軸だけcompileし、他軸は「同値±3 RZ」（ZSXの1 runに含まれるRZの上限）として扱う。m6でCは約1.8×10⁶なので、相対誤差は2×10⁻⁶以下になる。1 pathの時間は約4割減り、compile件数の予算は半分で済む。
- **変更の大きさ**：中。価格取得の新SOURCEと、「X/Yを個別に取得する」という現行契約の変更（ユーザー判断）が必要。
- **科学的再検証**：小。保存済みのX/Y対（すべて同値）との整合と、小roundでの両軸compileによる確認で足りる。
- **成立条件**：compiler profile（opt1、layout/routingなし）とwrapperの形が変わらないこと。変える場合は人工検証をやり直す。

### E3. 同じ巨大回路のfingerprintを繰り返し計算している — 重要度：中〜高

- **対象**：`canonical_qiskit_circuit_fingerprint`（provider）、`build_native_hadamard_wrapper`の事前照合、C1 `price_requests`のmutation検査。
- **根拠**【既存記録】【静的】：m6の1 pathで、wrapper構築時のevolution再照合（X/Y各1回）、wrapperのfingerprint（各1回、約24秒）、compile後のmutation検査（各1回、約24秒）がある。明示的なものだけで6回・約144秒（857秒の17%）。compose内の部分・全体fingerprint（107秒の内数）を含めるとさらに多い。
- **無駄と判断する理由**：
  - `qiskit.transpile`は入力回路を書き換えない。compile後の再hashは、起こらない変化の検査である。
  - wrapperのfingerprintはcompile cacheのkeyだが、新lowering・新cohortではcache hitは0件だった（P1、P2）。
- **改善案**：wrapperごとにfingerprintは1回にする。mutation検査は命令数など軽い不変量に置き換える。C1 Backendは固定SOURCEなので、Dの新しい取得経路に入れる。
- **見込み**【推定】：m6で約100秒/path（約12%）の削減。価格は変わらない。

### E4. 直接compileで測れるroundには上限がある — 重要度：中（計画への示唆）

- **根拠**：
  - 【既存記録】m6（8,192 event）：365万命令、peak RSS 2.8 GB、compile 242秒/軸。
  - 【既存記録】event数はκλ_R²T²でroundごとに4倍。m7は32,640、m12は33,423,360。
  - 【推定】比例計算では、m8で約5,800万命令・約45 GB・約65分/軸、m9で約2.3億命令・約180 GB・約4.3時間/軸、m12で約1.5×10¹⁰命令となる。m10以上は保持もcompileもできない（hostのRAMは503 GB）。
  - 【既存記録】nominal×E[S]の重みは、m10〜12で96.9%、m12だけで66.2%（BのP0〜P4）。
- **研究への影響**：Gを決めるroundのcompiled価格は直接測れない。直接測定は「価格model（E[S]＋残差）の検証点」としてだけ意味を持つ。
- **示唆**：m0・m6で観測された残差はCの3〜9%だった。方式間のGの差がこれより十分大きければ、中間roundのcompileを増やしても結論は変わらない。大規模なcompile計画（例：P2の296 compile案）は、必要な価格精度を先に決めてから規模を決めるのがよい。

### E5. exact有理数の全count列挙はNに対して急に重くなる — 重要度：中

- **対象**：B `rpe_deterministic_risk.round_distribution`、`verify_good_transitions`。
- **根拠**【実測】（人工のraw値、1 round、X/Y同じN）：

  | N | cell数 | 時間 | peak memory |
  |---:|---:|---:|---:|
  | 12 | 169 | 0.26 s | 9 MB |
  | 64 | 4,225 | 0.43 s | 7 MB |
  | 256 | 66,049 | 35.7 s | 280 MB |

  二項係数の分母はN=1024で52,225 bit、N=4096で208,897 bitになる。
- **現在の計算量**【静的】：cell数は(N+1)²、各cellで巨大整数の積を作って辞書に保持する。隣接roundの閉包検査は、good集合の全対（最悪でO(N⁴)）を調べる。
- **研究への影響**：現行の配分（N≤64）では数秒で問題ない。riskを下げるために後半roundのNを数百以上に増やす探索では、この実装が先に限界になる。
- **改善案**：
  1. good確率は、X軸の各countについてY側の累積和を使えば、積の表を持たずに求まる（memory O(N)）。
  2. 閉包検査は、実数での十分条件（2ρ_{m−1}＋ρ_m＜π）に丸めの余裕を加えた判定で置き換えられる（S2の幾何的議論と同じ）。全対の列挙は小Nでの照合用に残す。

### E6. event列のJSON保存が大きい — 重要度：中

- **根拠**【既存記録】：m6の1 pathの`events_PRIVATE.json`は246 MB（8,192 event、1 eventあたり約30 KB）。各eventがfragmentのbasis変換列を重複して持ち、indent付きで書かれる。
- **無駄と判断する理由**：event列はseedから決定論的に再生成できる。P1 run02で、再生成したbytesのSHA-256がrun01の保存fileと4 pathすべてで一致した。
- **改善案**：大きなroundではseed・event digest・event数だけを保存する。保存が必要な場合は（次数、成分番号）の整数列にする。m8以上を直接扱うなら必須になる。

### E7. `trotterlib`のimportが重い — 重要度：低

- **根拠**【実測】：`import trotterlib.rte`だけでも約2.6〜3.3秒・345 MB。package初期化がpyscf、openfermion、cirq、matplotlib、pandas、numbaを含む3,677 moduleを読む。numpy＋scipy＋qiskitだけなら0.57秒・105 MB。
- **影響**：test fileごと・起動ごとに約3秒。fork workerの基礎memoryにもなる。
- **対処**：providerは固定SOURCEなので変更しない。新規moduleでは、純関数1個のために`trotterlib`を経由しない設計にする。

### E8. 効率が良好と確認できた部分

| 計算 | 根拠 | 評価 |
|---|---|---|
| PR/PF signal（A provider） | 1 outer stepの行列を作り、行列累乗（O(d³ log q)）。H4はd＝36 | 良好。`d2.evolve`はevent数に比例するloop（O(q·R·d²)、毎回normを検査）なので、新規作業では使わない |
| 構造平均E[S]（C） | 局所alphabetのMarkov kernelを二乗法で合成。全11roundで4分38秒 | 良好。区間演算の厳密さは価格modelの不確かさ（数%）に比べて過剰だが、費用は小さい |
| 決定論的risk（B、N≤64） | 2構成で約7秒 | 良好 |
| 並列化 | path単位のprocess並列、compiler内は1 process | 妥当 |

## C. 実装構造

### C1. 規模と到達範囲

【実測】D headでの集計（型stubを除く）：

| 区分 | file数 | 行数 |
|---|---:|---:|
| `scripts/` | 1,298 | 177,744 |
| `src/trotterlib/` | 248 | 118,025 |
| `src/trottertracks/` | 746 | 127,994 |
| `src/trottercert/` | 30 | 3,641 |
| `tests/` | 742 | 135,172 |
| 合計 | 3,064 | 562,576 |

- 現行系列（11/10以降にA/B/C/Dが追加・変更したfileのうちtrack_d等を除く）は133 file・約2.2万行。
- そこからimportと固定path文字列をたどって到達するのは385 file・約11.5万行で、非test code（2,388 file・43.8万行）の26%にあたる。この値は`trotterlib/__init__`の一括importを含むので、実際に計算で使う範囲はさらに小さい。
- 残り約74%は過去研究の保全分である。固定科学成果の一部なので、削除候補とはしない。

### C2. 重複

【実測】（[集計](evidence/CODE_STRUCTURE.txt)）：

- 本体が完全に同じ関数が1,304組あり、余分な複製は2,107個・約2万行。多いのは`digest`（40）、`_git`（40）、`_canonical_json`（25）、`_fingerprint`（21）、`_compiler`（16）。
- 同名helperを定義するfile数：`sha` 193、`read` 170、`write` 170、`require` 145、`digest` 127、`save` 120、`git` 106。
- git objectを直接読むfileは327、`sys.path`を操作するfileは854、`main`を持つrunnerは836。
- 同じ科学計算の別実装：S2＋tailの有限平均が7実装（§0の表）、Hadamard wrapperのbuilderが8、decoder系の実装が14 file、compile呼出しが78 file。

**評価**：過去分は触る利益がない。現行系列で意味があるのは、(a) 新規作業で使う実装を1つに決めること（signalはA provider、価格取得はDのadapter）、(b) 固定blobを読むloaderを共通化すること（I1の原因の一部）である。

### C3. 識別・照合処理の比重

- 例：`pr_rpe_rmse.py`（25 KB）のうち、数式にあたるのはRの写像とNの式の十数行で、残りはidentity・SOURCE・configurationの照合である。
- 【実測】遅いtestでは`deepcopy`が1 fileあたり140万〜480万回呼ばれ、時間の5〜17%を占める。E1を直した後は、これが次の律速になる。
- **評価**：照合は取り違え（I4のscalar等）を実際に止めており、不要ではない。ただし、同じconfigurationを段ごとに全体copy・全体比較する設計は、候補数が増えると二次的に重くなる。IDの比較で足りる箇所は多い。

### C4. 資料の量

【実測】11/10〜11/14に追加された日付付き研究資料directoryは129（11/13だけで74）、1,284 file・80.7 MB。REPORT 135、HANDOFF 99、TESTS 72、STOP 14。
1つの有限作業に、準備・計画・結果・引渡しの各directoryが作られている。研究上必要なのは、結果の原本、条件、来歴である。同じ内容の再掲（索引4〜7 fileの同時更新を含む）は、`AGENTS.md`の「重要度に応じた資料更新」の方針に照らして減らせる。

## I. 統合上の問題

### I1. 初回の実実行が配線の誤りで止まる — 重要度：高

【既存記録】11月13日の1日に、次の7件があった。

| 時刻 | 系列 | commit | 止まった原因 | 分類 |
|---|---|---|---|---|
| 00:45 | C S0a | `3f2a696e` | 固定blobをexecするscopeに`ast`がなく`NameError` | exec scope |
| 02:44 | C | `eeccdf18` | descriptor APIの不一致 | interface |
| 10:10 | C K0 | `19b95d8b` | B configurationの参照先の誤り | 参照先 |
| 12:38 | D P1 | `20fe1194` | 固定Backendが要求する`plan['source']`の欠落 | plan key |
| 15:50 | A 20セルscreen | `37e30194` | `scripts/`がimport経路になく`ModuleNotFoundError` | import経路 |
| 16:02 | A PF screen | `69917830` | spectral経路のunitarity gate不合格（S3） | 数値gate |
| 20:53 | D T2 | `752b9747` | tail-onlyのscalarをtotal scalarの欄に渡した（I4） | 量の取り違え |

- **共通点**：7件中5件は、数値計算に入る前の配線の誤りである。事前の人工testは実loaderをmockしていたため、検出できなかった（Aの報告に明記されている）。
- **費用**：「無断retry禁止」の契約のもとで、各STOPは保全・報告・再承認の往復を1回必要とした。計算時間ではなく、この往復が研究の進行を律速している。
- **改善案**：実行scriptに「配線preflight」を標準で持たせる。実際のloader・保存入力・固定blobを使い、context構築と引数の束縛まで進めて、数値kernelの直前で止める。新しい科学値は生成しないので、技術検証の範囲で実行できる。科学実行の承認は、preflightが通った後に求める。
- **補足**：前回監査のR4（execで読む固定SOURCEの自由変数検査）は、DのP1 runnerに部分的に入っただけである。

### I2. 変更され続ける共有moduleを、作業tree上のbytesで固定している — 重要度：高

- **対象**：B `src/trottertracks/h4_s0_v2_intake.py:176-184`、各branchのtest（`check_pr_r_dependent_n`、`check_h4_df10_rpe_batch`、`check_h6_price_pipeline_coverage`、`check_r_variable_compiled_price`、`check_h4_s0_v2_intake`）。
- **根拠**【実測】：
  - `h4_s0_v2_intake`は、作業treeの`pf_pr_rpe_pipeline.py`が、S0時点の版と（H6 factoryを除いて）ASTで一致することを要求する。
  - このmoduleは`030f3f54`（11/13 02:32）で追加された。35分後の`f0c4bb55`（03:07）で、B自身が`pf_pr_rpe_pipeline.py`に標準Yoshida4を追加した。
  - `check_h4_s0_v2_intake`は`030f3f54`では10件PASSし、`f0c4bb55`以降は`S0_H4_PIPELINE_SOURCE_CHANGED`で失敗する。productionの関数が同じ例外を出すので、S0 V2 intakeは現在のheadでは再実行できない。
  - BとDのheadでは、上の5 test fileが失敗する（[一覧](evidence/TEST_MATRIX.txt)）。AとCのheadでは、現行系列のtestはすべて通る。
- **研究への影響**：各報告の「新12＋回帰45 PASS」は、選んだtestの結果であり、branch全体の健全性を示さない。失敗が既知か新規かを、毎回人が判別する必要がある。
- **改善案**：
  1. 過去時点のSOURCEとの同一性は、作業treeではなく、固定commitのblob（`git show`）に対して検査する。
  2. 共有moduleを拡張するときは、追加だけにとどめ、旧版に対する互換testを同じcommitで更新する。
  3. branchごとに「現行系列の全test」を1 commandで回せるようにする（I6）。E1を直せば数分で終わる。

### I3. loweringの識別がconfigurationの外にある — 重要度：中〜高

S1の統合面である。canonical configuration（例：`587d20f7…`）はloweringを含まず、新旧どちらの価格も同じconfiguration IDに結び付く。区別は価格側の付帯情報（profile extension）だけに依存し、旧recordはそれを持たない。対処はS1と同じ。

### I4. scalarという名前の欄に、意味の違う量が入る — 重要度：中

- **根拠**【既存記録】：Aのdescriptorの`scalar`はtailから抽出したidentity係数s_R（0.2224）、Dの実行時contextの期待値はHamiltonian定数を含むtotal（c_H＋s_R＝2.5155）である。Dはs_Rをtotalの欄に渡し、binary64の完全一致guardで停止した（`752b9747`）。Dは量の種類を明示する修正を入れた（`e8273676`）。
- **評価**：guardが働き、誤った位相の回路は作られていない。数値側のsignal実装は、すべて同じ定義（c_H＋s_Rを一回）を使っている（S0）。
- **改善案**：Dの修正と同じく、A・Cのdescriptorでも`TAIL_EXTRACTED_IDENTITY_COEFFICIENT`／`HAMILTONIAN_CONSTANT`／`EXPLICIT_TOTAL_SCALAR`のように量の種類を欄名で区別する。

### I5. 実行されるprogramが、多数の固定commitにまたがる — 重要度：中

- **根拠**【実測】：
  - 現行133 fileのうち77 fileが40桁のcommit IDを持ち、参照先は107 commit。mainの祖先は0、A/B/C/Dいずれのheadの祖先でもないものが17ある。
  - 26 fileが固定blobを`exec`で読み込む。
  - C1 Backendは実行時に`sys.path`の先頭へ私的provider rootを入れ、`trottertracks.__path__`にも追加する。同名moduleがどちらから読まれるかは、importの順序で決まる。
- **現状の評価**：providerと現行treeに共通する215 moduleのうち、内容が違うのは2件（旧`pf_delta_validation.py`と`trottertracks/__init__.py`）だけなので、実害は確認されていない。物理kernel（`rte`、`df_*`、`rte_compiled_cost`）は10/14から1版である。
- **危険**：今後、共通moduleを現行branchで変更すると、実行時にはproviderの旧版が黙って使われる可能性がある。
- **改善案**：providerに存在するmoduleは現行branchで変更しない、という規則を明文化する。または、Backendのcontext構築時に「読み込まれた全moduleの出所」を記録する。

### I6. testの起動条件がbranchごとに違う — 重要度：中

【実測】Cの新しい6 test fileは`PYTHONPATH=src`がないと`ModuleNotFoundError`になる（付ければ全件PASS）。他のtestは自分で`sys.path`を設定する。3 fileは`docs/research/`がcheckoutされていないと失敗する（sparse worktreeで起きる）。
1 commandでbranchを検証する方法がなく、どのtestを回したかは担当の報告に依存する。

### I7. branch・worktree・objectの量 — 重要度：低〜中

【実測】remote branch 250、local branch 664、登録worktree 217、tag 103、共有`.git` 4.0 GB、loose object 7,581（自動gcの既定閾値6,700を超えている）、reflogからしか到達できないcommit 120。tree内の`artifacts/`は1.79 GB・48,899 fileで、full checkoutは1つ約2.3 GBになる。

- mainは11/04で止まっており、11/05以降の研究はすべて未mergeのbranchにある。
- 自動gcが走ると、reflogの期限（既定30〜90日）を過ぎた到達不能commitは消える。保全したいcommitがreflogだけに残っていないかは、各担当で確認が必要である（今回の120件の内容は判定していない）。

### I8. 監査中の副作用

`git fetch origin --prune`により、remoteで削除済みのbranchに対応するremote-tracking refが消えた。詳細と確認結果は[REPORT §5](REPORT.md)に記載した。

## V. 検証不足

| ID | 未検証の内容 | 重要度 | 必要な最小の検証 |
|---|---|---|---|
| V1 | m7以上のcompiled価格。Gの約97%を占めるm10〜12は直接測定できない | 中〜高 | 価格modelの外挿誤差の上界か、方式間の差に対する感度の評価 |
| V2 | Full-H RTE・Truncated PFの誤差予算（bias＋統計）を、PRと同じ枠組みで評価した結果 | 中〜高 | 各方式のbias（H_effの固有値shift、打切り）を同じtargetに対して評価する。CがB0I・Full-Hの26 round signalを公開済み（`367dc98d`、内容は未確認） |
| V3 | 実sampler（PCG64/CDF）の法則とideal有限K法則の距離 | 中 | Cの整合資料（`finite_k_sampler_law_alignment_20261113`）の条件を、価格の期待値差に結び付ける |
| V4 | 回路と数値発展の作用素一致。確認済みは小規模（m0のtail、人工event）だけで、full wrapperのstatevector検証は全価格実行でfalse | 中 | 小さいq/Rでfull wrapperのstatevectorとsignalを照合する（要承認） |
| V5 | 旧kernel（spectral、event loop）で計算した保存済みの長時間mean。direct expmとの照合は1セルだけ | 低 | 代表2〜3点の再照合（要承認）。差は10⁻¹¹級の見込み |
| V6 | 実loaderを通る経路のtest | 高（I1） | 配線preflight |
| V7 | X/Yの価格の同値性は観測に基づく | 低（E2） | 人工検証は今回実施。採用時に保存X/Y対と照合 |

## 前回監査（2026-11-13）の提案の状況

| 提案 | 内容 | 状況 |
|---|---|---|
| R1 | controlled ZZの効率的lowering | 実装済み（D `bb475af8`）。P1/P2で価格を取得 |
| R2 | `candidate_id`のmemo化 | **未対応**（E1。影響を再計測した） |
| R3 | 閉形式のdecoder | 実装済み（`rpe_decoder_fast`、旧実装とのtie規則も保持） |
| R4 | exec loaderの自由変数検査 | 部分的（DのP1 runnerのみ）。同種のSTOPが続いている（I1） |
| R5 | 共有moduleの版管理 | **未対応**（I2。失敗testは1件から5 fileに増えた） |
| R6 | D inventoryのcache | 未確認 |
