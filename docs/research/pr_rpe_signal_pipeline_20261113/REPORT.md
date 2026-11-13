# PR全RPEラウンドのphysical signal provider

2026-11-13 JST。設計SOURCE：`821c5c1ffd7c95e1690d45d2b6e8ff4ccb15bd1c`。

**直接expmの決定論headと既存finite-K RTEを接続し、B V2が生成した一つのPR configurationの全M+1roundを評価する数値入口を実装した。新10＋既存7の人工・metadata回帰test PASS。実H4の新signal・保存数値再解析・新reference・N/C/G・候補探索は0。** 人工PASSは科学的性能や全RMSE保証を示さない。公開照合後mandatory STOP。

実装：[pr_rpe_signal_provider.py](../../../src/trottertracks/pr_rpe_signal_provider.py)／[人工test](../../../tests/check_pr_rpe_signal_provider.py)。[HANDOFF](HANDOFF.md)／[未実行の有限科学案](SCIENCE_PLAN.md)。

## 1. 再利用と最小差分

| SOURCE・既存処理 | 今回の使い方 |
|---|---|
| Track A `155529e20a8fac39e00786c40b57830b71203c4f`、`h46_long_time_signal.cid/evolve_batch`、`h46_m5_plan` | 元physical ID・fragment適用順・signed schedule・ABS_TIME allocationを保持。有限Tのwinner/Paretoや価格を採択しない |
| B `3216d27a69ac8532d2a22e78f5f94491d202d238`、`pr_rpe_rmse_controller` | generate/validator・全round identity/lawはBの正本。AでID・κ mapper・候補管理を作り直さない。公開branchのSHA一致も確認 |
| `pr_rpe_signal_plan.plan`（821c5c1） | constructorで全M+1round・SOURCE・physical recipeをmetadata照合する。数値leafは呼ばない |
| `h4_pr_rmse_completion.load_context`、保存入力INDEX `7a8bc7f132e166390c943e6b32c92cc6abe1baf4` | 明示factoryで既存snapshot/hash/layout検査とsector行列復元を再利用。SCF/DF fit/state生成なし。旧launcherやprocess-wide BLAS方針を起動しない |
| `trotterlib.rte.finite_taylor_operator/finite_rte_distribution` | paired Taylor degree K+1と有限normalizerをそのまま利用。Taylor再実装・sampler生成なし |
| `pr_rpe_power_mean`／S1＋S2 `169d3316dd008eb98dd9be56fe4ecefc5c1a7d60` | matrix-power方式とraw/corrected独立伝播を引き継ぐ。旧spectral kernelとの比較は人工行列だけで実施 |
| `pf_power_stability.power_diagnostics`、実H4直接expm確認 `b72ea6b30e48e6a2f5d805903b7bd750b965cf16` | headはscipy expm、exact-tail macro/powerのnorm・unitarity検査を再利用。実H4の二経路検証を繰り返さない |
| `h4_four_method_screen_mean.allowance_formula`、元数式SOURCE `7c3e1e5db0c55dd8249e9030a4c7e9954496fa7e` | R-aware allowanceの算術部分だけを再利用。旧Hoeffding solver、RMSE nominal N、decoderは呼ばない |
| 物理入力契約 `bba150834ca30bdb186fbfd58c55a8d51177b855` | 保存H/state bindingと全13roundの独立reference pairをlookup。今回確認したのはschema・所在であり、新数値診断ではない |

追加は数値provider一つと人工test一つ。新CLIやcontrollerは作らず、Bの既存codeから明示的に呼べる`SignalBatch`を入口にした。旧科学SOURCE・結果・norm欠測・STOPを変更していない。

## 2. 対応範囲

| 項目 | 対応・制限 |
|---|---|
| PR構成 | uniformなL_D/formula/ΔPF/KとBが生成した全round qPF/R。S2、4次M5 best、K2/4/6。共通κの意味・整数化はBから受け取る |
| prefix | generic contextは非空head・非空tailの1≤L_D<rank。保存H4 factoryはdescriptorがある1/2/3/5/7/9のみ。4/6/8等のλを新計算しない |
| 時間分割 | B V2のactual δ=T/qPFを使用。初期T<指定幅でもrowの実幅を保持。q_RPE=2^mとqPFを区別 |
| M | 全0…Mを要求。人工M12の13roundをS2/M5各一構成で検証。今回Mやκの実menuは未採択 |
| 未対応 | PR標準Yoshida4（B正本が未対応）、H6初期prefix5/R32例外、round別に任意のL/formula/Kを混ぜる構成、Full PF/Full-H pure RTE endpoints |
| K横断 | K2/4/6の個別signal評価は可能。K別configuration間のN/G比較・同資格判定はBの仕様であり、Aは横断selectorを追加しない |

generic人工contextの対応と保存H6への接続完了は別である。H6実入力factoryは今回追加していない。

## 3. 数値経路と保存する値

各signed pieceで、元順のhead半ステップを直接expmし、tailを挟み、逆順headを適用する。tailはA_p=Σtail H_l−s_p I。τ=λ_R t_piece/n_pieceの有限normalizer bを既存式から取得し、既存paired Taylor Pを使う。

- exact-tail：`expm(-i t_piece A_p)`。
- corrected：`matrix_power(P, n_piece)`。
- raw：**独立に**`matrix_power(P/b, n_piece)`。最後にcorrected/Bを代入しない。

piece積の三つのmacroを独立にqPF乗し、最後に共通scalar `exp[-i(c+s_p)T]`を一回だけ適用する。負時間の符号をP・expmに残し、allocationは元11piece全体のRを分配する。Taylor primitiveは積dt·Aだけに依存するためphysical A・dtで呼び、normalizerのλを省略したことにはならない。

whole logBは`qPF Σ_piece n_piece log(b_piece)`、attenuationはexp(−logB)。各head expmとexact-tail expm、同macroはcontext内でcacheする。計算量はeventの逐次列挙でなくmatrix powerのlogarithmic multiplicationで増える。量子回路価格が無料になるという意味ではない。BLAS/worker/CPU/RAMの独自上限を追加していない。

出力はraw/corrected/exact-tail/reference、三つのsignal-space bias、B/logB/attenuation、piece time/event数、closure、norm/unitarity、empirical allowance、reference/state証拠、B/physical identityとSOURCE。新しいphase/radius/N評価はproviderの責務に含めず、B/A既存診断への入力を返す。

## 4. 数値gate・失敗保存

独立reference pair差≤1e−10、入力state norm差≤1e−12、head・exact-tail powerのnorm/unitarityと独立closureは既存1e−9基準。corrected/rawのfinite-K operatorは一般にunitaryではないため、そのunitarityをexact-tailの条件で検査しない。raw座標の範囲はempirical allowanceと共に確認し、clip/再正規化しない。

normalizerのb≤1による丸め未解決、非有限、B overflow、attenuation underflowも停止理由として保存する。logBのone-ULP感度を従来marginとmaxで結ぶ。最大normはmacro/power出力境界の観測であり、全event中間normの厳密上界ではない。従来のlogical event数を使うR-aware allowanceは保守的な経験式で、実matrix乗算数の誤差証明ではない。

数値gate失敗でも取得済みsignal・残差・piece診断を消さず`NUMERICAL_FAILURE`として保存。例外時は取得済み診断・例外型/理由を保存し、signalを無理に埋めない。batchは最初の失敗で部分結果と未開始roundを保存する。fresh output必須で自動retry/fallbackなし。SOURCE/入力不一致は数値性能FAILと区別する。

保存inventoryの一致は「受入完了」を意味しない。`SignalBatch.run`は新規mean要求のみを扱い、保存mean/保存FAILを含む場合は既存acceptanceへ返してSTOPする。旧spectral meanや旧N packetを新direct-sourceの結果へ再ラベルしない。

## 5. 技術検証と資格

[TESTS_PROVIDER.txt](TESTS_PROVIDER.txt)：人工10件PASS。[TESTS_PLAN.txt](TESTS_PLAN.txt)：既存metadata7件PASS。人工2×2非可換11block、prefix1/2/3、S2/M5、K2/4/6、負時間、逐次productとの一致、旧finite kernelとの一致、scalar一回、raw破損による独立closure検出、数値gate失敗の残差保全、13round境界、SOURCE/candidate誤用、架空ファイルだけのH4 loader/reference橋渡し、passive importを検証した。テスト用λ・κ・H/stateは架空で、実探索menuではない。

十分N・global RMSE・actual sampler law・ground同定は未資格/null。新しい実H4 mean/reference/normalizer、保存科学数値の再解析、N/decoder/C/G、候補探索・最適化は行っていない。

## 6. 次に必要な作業

**ユーザー**が有限menu・exactκ・外側条件・新数値SOURCEとsignal-only scopeを指定/承認する。**Codex A**がその範囲の全round signalを評価し、**B**がN/decoder/同RMSE資格、**価格担当**が同candidate/scheduleのC coverageを扱う。最小の単一configuration13round案と二構成比較への拡張を[SCIENCE_PLAN](SCIENCE_PLAN.md)に分離した。今回この選択や科学実行は行わず、公開・remote照合後mandatory STOP。
