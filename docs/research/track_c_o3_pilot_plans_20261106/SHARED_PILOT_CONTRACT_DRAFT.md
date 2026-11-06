# Track C：共通pilot科学契約案

2026-11-06 JST。**DRAFT / 計画作成のみ / 科学実行・実装は未承認。**
この文書は固定実行契約として発効していない。数値を「既存条件からの採用案」として示す箇所も、今回の正式採択・計算開始許可を意味しない。

## 1. 目的・固定資料・実行境界

元の物理Hamiltonian `H_phys` の厳密基底状態を入力する。`H_phys |E₀⟩ = E₀ |E₀⟩`。状態準備のみを除き、各候補とbaselineに同じ入力・情報アクセス・energy精度・成功条件を与える。`H_D`、近似DF、PF有効Hamiltonianの基底状態を代用しない。

正本は[Track Cプレフライト][preflight]、固定commit **`4b625e5bef06ac51f2b05cafe66f709df2f8bddb`** の3文書。補助は3環境統合`3571477093441ac4852166fe829f8ccc5db5ec4f`、先行調査`e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8`。正式共通指示2026-10-25の§8.1/8.2/8.4/10/10.3と適用AGENTS/PROJECT_MAPを確認した。

今回許可されたのは静的読解・計画・文書公開だけ。科学module/研究testの起動、NPZ数値配列のdecode、保存値の再解析、simulation/sampling、回路build/compile、合成、algorithm実装は行っていない。旧研究のSOURCE/input/raw/result/契約/grant/ledger/marker/STOPを変更しない。

## 2. 共通入力：H4を優先する案と、その限界

### 2.1 既存資産から確認できたこと

| 項目 | 既存の記録・再利用候補 | 今回の位置付け |
|---|---|---|
| 分子・空間 | linear H4、隣接1.00 Å、STO-3G、8 system qubit、4電子sector dimension70 | 新分子・geometryを追加しない共通anchor候補。spin-sectorの追加投影を勝手に行わない |
| 保存Hamiltonian | legacy DF rank12、生成順prefix。Hamiltonian hash `56e4df83655aa2f2f8132126f2635996516dc2cfb1bf4cb8d95490184ad631e5` | 以下では`H_DF,legacy`と呼ぶ。元の未近似電子Hamiltonianとの同一性をこのhashだけで認定しない |
| snapshot | [carrier 4591b8d…のNPZ][snapshot]、ファイルSHA-256 `13e4b10d2347ed900fe4aa4b2238128ebe735a48988e321b9e4fabac4d092a3b` | loaderの保存項目はconstant/one_body/lambdas/g_matrices/metadata。元ERIと物理groundを保証する証明書ではない |
| 保存state | [e22d7718…のphysical_state.npz][state]、ファイルSHA-256 `793300c86417f52569289ce86980c5f9a87292e130199355a0f4f7056225c9c4` | 保存DFのphysical ground-state rayに由来。配列を今回開いておらず、元H_phys groundへの距離は未認定 |
| 既存state取得法 | [finite_rte_signal_validation.py][finite]は保存DFをsector行列化し`numpy.linalg.eigh`で最低固有stateを得る。後続m17は保存rayを用いる | `H_phys`の正当なgroundと同定できる場合だけ共通入力にできる。normalized rayは数学的厳密groundと同義ではない |
| 分割 | Partial LD3：one-body/correctionとDF prefix[0,1,2]をD、残り[3..11]をR。Full LD12はtail-free対照 | `LD=3`と`12`は共通採用案。別rank/非prefix/新表現の探索は本pilotに入れない |
| finite-RTE | signed paired Taylor、偶数K、original tailのrはoccurrenceごとのshort-step数 | K2、既存round別rは候補初期値。高次PFの負時間・occurrence数に同じrを無条件流用しない |

snapshotは`4591b8d75a2a6a55c74a51225f0c61648023c8ee`、stateは`e22d7718c47358ff650dbc04e8e72f6ab1c8bf41`の固定objectとして所在を確認する。これはSOURCE/入力の参照で、科学callの許可ではない。

### 2.2 元Hと厳密stateの判断を先に残す

**推奨する計画上の扱い**は、H4の既存geometry/basis/sectorとDF/native資産を優先しつつ、`H_phys → H_DF,legacy → 保存state`の対応が未確定なら主要科学評価を開始しないこと。

1. 元の一体/二体積分・nuclear/scalar・orbital ordering・fermion mapping・sectorの固定入力と生成来歴を指定する。今回確認したDF snapshot/stateだけではその対応を閉じていない。
2. 再利用に必要な同一性が静的provenanceで確定できれば、その範囲を採用する。行列の再構築・差のnorm評価・固有状態残差/距離の新評価が必要なら、それ自体を将来の科学callに含める。
3. 近似DFなら`H_phys-H_DF`の作用と入力stateを分離して扱う。DF groundからenergy差を戻すだけでは、厳密`H_phys` groundを入力する条件を満たさない。主要比較は同じ`|E₀(H_phys)⟩`へ各近似operatorを作用させる。
4. 既存stateを使えなければ、指定された既存H4元入力からsectorのground-state参照を一度取得する案を別途選ぶ。SCF/新DF fitや新分子生成を自動追加しない。有限精度のstate近似誤差は保証/経験的診断を区別して保存する。
5. 来歴を閉じないまま`H_DF,legacy`を「元H」と改称しない。DF model上だけの機構診断を希望する場合は、ユーザーが別の限定目的として明示選択し、本来の元H energy task達成と区別する。

これは新しい重要科学レビューの開催やDF結果の否定ではなく、今回の研究前提へ移すための未決条件の指定である。

## 3. 既存条件からの共通採用案

| 条件 | 採用候補・出典 | 変更が必要になる理由／比較可能性 |
|---|---|---|
| energy目標 | 旧WP01/V2の暫定 `εE=.00015936001019904 Ha`（CA/10） | 新目標として確定しない。元H/DF/合成誤差の予算を加える場合、従来のresolution余裕だけでは不足し得る |
| 成功条件 | 旧`α_total=.05`、`β_RPE=.4 rad` | 条件付きIIDモデルから現実の全実装保証へ昇格しない。価格推定のconfidenceとenergy失敗率を別欄にする |
| RPE時刻 | 共通base `δ0=.02`、`m=0..17`、`q=2^m`、`t_m=q δ0`、終端q131072 | [V1/V2][v1]の既存候補。PF内部step hとinterrogationのbase δ0を別記する |
| PF microstep | C-Nでは旧検証にあるh=.02/.01の2点を候補にする。いずれも同じt_mを整数stepで満たす | RPE時刻・aliasを固定してPFだけを比較できる。旧P-Dのδ=.4 score探索を再実行しない |
| original tail | 旧opt2のK2、r列 `[1×8,2×2,4×1,8×2,16×2,32×3]` | C-S/C-Pの固定入力案。C-Nの多occurrenceはsigned時間に対応する整数配分を別に固定する |
| β/α配分 | 元opt2の**Partial/Full別**配分を初期対照にし、[元JSON][opt2-json]の`/coherent_reoptimization/best_by_ld/{3,12}/best`を参照。βは同object、α/Nは`rounds/*/axes` | 二候補に同じ配分表があると誤記しない。新armへのコピーは暫定診断。公平比較では全armに同じ配分自由度を与える |
| native profile Q | Qiskit1.3.0、basis `rz,sx,x,cx`、opt2、seed17、backend/coupling/layout/routing指定なし、ordinary control | 既存full-RPEに最も近い。RZ/CX/depthを別々に報告。T合成済み価格ではない |
| native profile T | D1/D3のstrict RZ合成・actual adjoint・Clifford+T/CXの既存条件を候補とする | C-SのT trade-offにはこちらが必要。profile Qの旧価格を混ぜない。同じprofile Tを全C-S armに与える |
| synthesis precision | 旧SP/D1の1e-6、D3取得1e-7等は既存候補の例 | **長roundの全誤差予算への適合は未確認**。同じ値を自動採用しない。選ぶのは全interrogationの合成誤差予算と適合する一つのprecision policy |
| threshold | 既存m17例はcoefficient_atol=1e-12。保存gridには非ゼロthreshold errorあり | 無視して0補完しない。元値を保存してbudgetに含めるか、変更を選ぶ場合は科学条件差として記録 |
| 主資源 | Qでは旧compiled RZ、Tでは既存Tに加えCX/depthの個別比較が候補 | 新しい主指標をこちらで採択しない。ユーザーが主張したい指標を選ぶ。異profile間の勝者を作らない |

上表は大きな直積gridではない。採用後は元H/target/profileを一つずつ選び、各pilotの有限arm表だけを評価する。backend/noise/routingの新研究は含めない。

## 4. 比較するfull wrapperと情報境界

費用対象は、元H groundの供給後の全操作：ancilla準備、物理processor、controlled evolution、basis/Givens往復、scalar/identity相対phase、合成、X/Y readout、測定。追加register/補助操作を方式が要求すれば含める。状態準備除外をprocessor無料へ拡張しない。

`exp(-itH)`、cos=Re、sine=Sdg→Hによる+Imを既存と揃える。constant/extracted identityは既に含むものを二重加算しない。system-global phaseでも制御後は相対phaseになり得る。全space controlledのcontrol0=I/control1=対象operator、または明示された測定等価性を別々に検査する。

E₀、gap、exact bias、exact priceは**診断側**へ隔離する。選択側は指定入力/既存の利用可能bound・公開係数・実取得価格だけを使用する。同じfinite比較集合の全候補・baselineへ同じ情報/探索機会を与える。exact参照で事後に良いarmを拾った場合はoracle diagnostic frontierとして表示し、実行可能selectorの利益にしない。

## 5. 共通signal・shot・branch会計の計画

計画する共通データは、各arm/roundの元H reference、ideal PF、finite mean、native実装meanを分けた複素信号、実半径または根拠を持つ下界、各誤差源、normalizer、実proposal、weight、phase/branch、価格と不確かさである。データ形の拡張だけを計画し、今回コードを実装していない。

\[
G^{(a)}=\sum_{m,b}N_{m,b}\,\mathbb E[C^{\mathrm{full,no\text{-}prep},(a)}_{m,b}].
\]

1. **誤差源**：元H/DF近似、PF、finite-RTE、coefficient/synthesis、数値評価、統計を別々に残す。Cδ²を一般高次/processing/合成へ代入しない。誤差の単純なnorm差引きで因果寄与率を作らない。
2. **元H stateと半径**：ideal PFの信号半径を1と仮定しない。正半径下界と系統位相差を正当化できないarmは、経験的signal診断までに限定する。`C_use`や保存最小値はrigorous boundではない。
3. **normalizationの規約**：raw `Zraw=Zcorr/B` の±1測定と、corrected `Y=W X` の測定会計を別ルートとする。rawなら観測半径へBを入れ、correctedならrange/momentへ入れる。両方に同じ負担を掛けない。
4. **十分shot**：軸内独立で`Y∈[-L,L]`なら既知Hoeffding式 `N≥ceil(2 L² εcoord^(-2) log(2/α_axis))`を使用できる。raw±1はL=1。weighted lawでL不明なら単位range式を使わず未資格。momentだけでfinite-confidence最適とはしない。
5. **位相・energy**：実signalと系統誤差から座標余裕を決め、X/Y別αと全round unionを確認。既存nearest-branch decoderはE₀を受け取らない。外部alias windowとstrict branch条件、元Hへのenergy変換を別に満たす必要がある。
6. **Nの扱い**：保存Nのままのablationは局所機構確認。全Gの主比較は同じtarget/成功条件で各armの必要Nを評価する。PF/合成/半径変更後の旧Nを正当化なしに移送しない。

[既存accounting][account]はS2/基準半径1に限定される。再利用は軸・round・provider・集計の境界であり、一般mean/誤差model/weighted十分Nの受渡しを最小差分として計画する。[V1][v1]・[V2][v2]の条件付き成功、[m17包含][enclosure]の指定ideal categorical結果を全実装成功へ拡張しない。

sector行列を使うのは、対象block/meanがそのsectorを保つと正当化できる場合だけ。native loweringの中間gateや個別Pauli pathを毎回sectorへ射影してleakageを消さない。必要なcorrectnessはfull system/joint spaceで行い、計算短縮と科学的target変更を区別する。

## 6. 最小段階・科学callの範囲案

将来の一つの明示許可範囲内では、下表を通常の技術作業として続けられる設計とし、各段階に追加の承認証明・grant発行・独立レビューを挿入しない。未決の科学条件が変わる場合だけユーザーへ戻す。今回はどのcallも実行しない。

| 段階 | 将来必要なcall・範囲 | 再利用・省略できるもの |
|---|---|---|
| 共通入力 | 指定H/stateのload・sector/basis対応、必要な一度のH/state参照評価、DF/basis/primitive preparation | 同じ入力版なら3pilot間で共有。元Hを新規生成するcallは現案に自動追加しない |
| mean/phase | 各pilot有限arm集合のone-step operator/finite mean、同じstateへの全18round signal、半径/誤差/branch診断 | sector行列とbinary powerをcache。同じmeanからX/Yを取る。全trajectory列挙・仮想量子shot samplingを初期必須にしない |
| native価格 | q1/2の較正と未使用q8の代表full-wrapper holdoutを候補、価格lawのrare strataも評価 | 同一実現のfull/分割価格をpairedに取得。標本数・精度は観測差を判別する設計値を実行前に選ぶ |
| 長round接続 | 全roundの価格モデルとscope/不確かさ、N、総G。必要な場合だけ終端近傍の有限な追加価格情報 | q>直接域を自動PASSしない。長q価格が閉じなければ局所結果＋条件付きGで止める。全長回路の大量compileを必須にしない |
| 精度・統計 | 合成/finite-bit/mean誤差の適合、軸別十分N・union/branch確認 | 固定m17用包含adapterは参考部品。新armの包含、保存再会計、samplingは別の新科学callとして数える |

各pilotの価格標本nと量子shot Nは別。rare stratum未観測を費用0にしない。finite lawからの条件付き層別価格とprobability重み、残るtailの上界/未知幅を分離する。期待価格の誤差が選択差を覆う場合はUNDETERMINEDとする。

## 7. 共通再利用と最小差分

| 資産・固定SOURCE | 再利用する接口 | 計画する差分／実行しない入口 |
|---|---|---|
| `f29eafbf4412193e8e8660d9f051889c2cba684b`：PF-ORDER/RTE-BASE/DF-CONTROL/RPE-COST | `iter_pf_steps`、`finite_rte_operator_moments/compose_finite_rte_occurrences`、DF builder、`RPERoundCostProvider` | 旧runner全体はS2・元SOURCE条件付き。高次macro/一般meanと実半径/weighted会計の小さい接続差分 |
| `4ccc1832f63cd323b23fc05623f7e0b58cf34f37`：Parent訂正版 | `event_statistics/error_accounting/native_plan/adjoint_plan` | providerと与えられたfinite表の部分を使う。新sampling law/Parent旧toy再開はしない |
| `b190fe28b98d2f94e1509e19cd3d36a63b2bb99e`：b2d_adapter | `published_math_sequences/actual_adjoint`、DContext、DF/native-v5のphase保持方法 | toy evaluator/launcherをH4 runnerとして使わない。H4 block/context接続のみ必要 |
| `4591b8d…`：保存opt2、BのRPE後続 | 18round時刻、保存input/条件・decoder/価格依存scope | 元execution SOURCEとcarrierを分離。保存再解析runnerは今回起動しない |

旧SOURCEは読むだけで、将来採用された変更は新しいTrack Cの科学SOURCEへ固定する。管理目的の汎用frameworkや承認認証機構は作らない。

## 8. ユーザーが決める項目と記録方針

| 未決条件 | 実行前に必要な判断 |
|---|---|
| 元H/state | 元H4積分とDF/stateの対応を閉じる入力を指定。追加の元H ground取得を含めるか。限定DF診断だけなら目的の縮小を明示 |
| target/成功 | 旧CA/10・α=.05・β=.4・18round案を採るか。元H/合成誤差まで含めたbudgetと外部alias条件 |
| profile/資源 | QのRZ/CX/depthか、TのClifford+T比較か。主指標と他指標の役割。C-SでTを選ぶ場合の全arm同profile |
| precision/比較自由度 | 全interrogation合成予算、数値・state誤差の扱い、β/α再最適化範囲、native price推定精度とdecision materiality |
| pilot固有 | C-Nの3formula/2h案、C-Sの同位置residual scope/精度policy、C-PのD-context kernelとPの向き・端点実装案の採否 |

将来の科学結果は、SOURCE/input/state/sector/係数/実現列・compiler/tool・law/seed・複素signal/誤差・weight/N・軸別価格/paired不確かさ・G・判定を必要十分に保存する。旧値、欠測、null、未資格、技術失敗、負結果を混同しない。承認原文や証明書の保存は求めない。

CPU/RAM/worker数を恣意的に制限しない。将来の独立arm・価格層を利用可能資源に応じて並列化し、共有のH/state/basis cacheを重複生成しない。他ユーザーや共有設定を変更せず、正常な他jobを停止しない。AI sub-agentは使用しない。

**今回のmandatory STOP：5計画文書の公開・remote照合で終了。次の科学計算・回路コンパイル・実装は、ユーザーの別途明示許可後だけ。**

[preflight]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/4b625e5bef06ac51f2b05cafe66f709df2f8bddb/docs/research/track_c_o3_technical_preflight_20261106
[snapshot]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/rte_connected_cluster_cost_validation/h4_sto3g_d100_rank12_ld3_dt0p1_ref4_k2_connected_pilot30_max1500_hold1500_rare375_v1.hamiltonian.npz
[state]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e22d7718c47358ff650dbc04e8e72f6ab1c8bf41/artifacts/rpe_fixed_request_measurement_b/2026-11-03/run01/physical_state.npz
[finite]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/src/trotterlib/finite_rte_signal_validation.py
[v1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4b625e5bef06ac51f2b05cafe66f709df2f8bddb/docs/research/pr_rpe_v1_e0_state_radius_20261103/REPORT.md
[v2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4b625e5bef06ac51f2b05cafe66f709df2f8bddb/docs/research/pr_rpe_v2_branch_success_20261103/REPORT.md
[opt2-json]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/research_direction_full_opt2/2026-09-25/wp11_all_r_opt2_coherent_analysis_20260925_065827.json
[account]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f29eafbf4412193e8e8660d9f051889c2cba684b/src/trotterlib/rpe_resource_accounting.py
[enclosure]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4b625e5bef06ac51f2b05cafe66f709df2f8bddb/docs/research/rpe_m17_rigorous_mean_enclosure_execution_20261103/REPORT.md
