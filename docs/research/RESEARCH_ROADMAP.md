# 統合研究ロードマップ

Repository：`HIROMU1015/Partially-Randomized-Trotter`。進捗統合担当：C（利用者指示、2026-11-16）。科学的採択の原本と公開済み進捗の同期を分離する。

この文書は、研究の目的・現在地・不足している証拠・次の進行順序を一か所で把握するための入口である。科学的な結果・SOURCE・検証内容の正本は、リンク先の個別REPORTである。本書は新しい科学計算・数値再解析・採択を含まない。

## 0. 現在地サマリー

<!-- C_AUTO_PROGRESS_START -->
現在の研究段階は、保存raw・Q3判定と価格規則を束縛した、保存候補集合内の条件付き比較である。
方式別の最新費用・配分・SOURCEは[C比較表](h4_saved_rpe_cost_catalog_20261116/run01/REPORT.md)と
[機械可読catalog](h4_saved_rpe_cost_catalog_20261116/run01/CATALOG.json)を正本とし、本書に数値を重複記載しない。

|方式|価格付き適合点|Q3待ち|価格待ち|公開探索の状態|
|---|---|---|---|---|
|FULL_PF|あり|なし|なし|COMPLETED_METHOD_FINITE_SEARCH|
|TRUNCATED_PF|あり|なし|なし|保存評価のみ|
|PR_PAULI|あり|なし|なし|INTERRUPTED_SEARCH_POLICY_CHANGE|
|PR_DF|あり|なし|あり|保存評価のみ|
|FULL_H|あり|なし|なし|保存評価のみ|

公開結果の結合に未解決項目：あり。
理由と固定SOURCEは[同期状況](h4_saved_rpe_cost_catalog_20261116/SYNC_STATUS.md)を参照。
Full PFの完了とPR-Pauliの方針変更による途中終了は区別する。進行中jobのprivate rawを同期処理から読まない。
正式E[C]、十分N、物理的global RMSE、G、winnerの資格はこの集計で変更しない。
<!-- C_AUTO_PROGRESS_END -->

## 1. 文書の役割

| 文書 | 役割 |
|---|---|
| 本書（RESEARCH_ROADMAP.md） | 研究目的・進捗・不足・依存関係・次の手順 |
| [PROJECT_MAP.md][MAP] | 実装・研究資料・成果物への入口 |
| [RESEARCH_RESULTS_INDEX.md][INDEX] | 過去の研究系列と原結果への索引 |
| [VALIDATION_STATUS.md][VS] | 検証結果の状態と利用制約（このcheckoutに登録された範囲） |
| 個別REPORT | 科学的結果・SOURCE・検証内容の正本 |

**読むときの注意**
- `main`（`546bca7ed`、2026-11-04）は11月5日以降の成果を含まない。それ以降の成果は各研究branchにあり、本書は固定commitのURLで参照する。
- リンク先の「現在」「次段階」は、その報告の作成時点の記述である。最新の状態は本書の§5・§10と、そこから辿る最新報告で確認する。
- 本書の数値は元報告からの転記であり、条件は必ずリンク先で確認する。

## 2. 研究全体の目的と構造

**全体の目的**：Hamiltonianの一部を決定論的な積公式（PF）で、残りを有限のランダム化Taylor展開（RTE）で実装する部分ランダム化（PR）について、同じ精度・成功条件の位相／エネルギー推定を、どの条件でどれだけ安く実行できるかを明らかにする。単回の回路の短さではなく、PFのbias、finite-RTEのbias、normalizationによる減衰、必要shot数、全roundの回路価格を接続して評価する（[研究目的][GOAL]、[方法][METHOD]）。

### 2.1 研究の階層

```text
研究全体（repository）
├─ リソース評価
│   ├─ 有限時間の資源地図：H4/H6/H8 DF、Pauli-JW、長時間B2/B3（〜11/02）
│   ├─ RPE全roundの総資源：H4 DF10の5方式比較（11/10〜、現在の中心）
│   └─ 一般性：H6/H8への条件移送（保存成果あり、現比較とは別契約）
└─ アルゴリズム改善
    ├─ return集約・sampling・推定器（Track B／O2）
    ├─ 表現・分割（元Track A／A-O1、研究C）
    ├─ Closed P3・共同選択（Track C／C-O3）
    ├─ Reference／control（Track D／D8）
    └─ 適応readout（新A）
```

- 11月1日時点の全体像は[科学的到達点と証拠の地図][LAND]と[主張・証拠・未解決課題の対応表][GAPS]にある。
- 11月10日時点の系列間の接続は[PR＋RPE共同最適化への統合調査][INTEG]と[対応表][INTEGM]にある。
- 本書はこれらを再利用し、11月10日以降の進展を加える。

### 2.2 同じ名前で別のものを指す語

| 名前 | 指すもの |
|---|---|
| 作業担当 A／B／C／D（11/13〜） | H4総資源比較の役割分担。A：物理信号・Full-H・Pauli設計。B：RPE推定器・MSE・必要N。C：価格・モデル・証拠受入。価格担当はD：決定論PF、D1：Full-H、D2：PR-Pauliと後続PR-DF LD1の取得。各branch・SOURCE・科学契約は区別する |
| 旧Track A（〜10/26） | 既知PRの資源評価（リソース評価系列）。PR-2 M1/M2、H4/H6/H8、長時間B2/B3 |
| 元Track A／A-O1（11/06〜08） | アルゴリズム改善。辞書変更（O1-M1）、core＋residual（O1-M2）、B2-P1 |
| 新A | PR Reference／適応readout（V6、Q3、Q4） |
| Track B（B/O2） | return集約・局所生成・有限bit・sampling |
| 研究B／B2-D | D内部PF・outer分割。Track Bとは別 |
| Track C（C-O3） | Closed P3、joint basis、h／resolution／αの共同選択 |
| 研究C（RC） | Hamiltonian表現・PR-aware DF（CDF、ENTRY）。Track Cとは別 |
| Track D（D8） | Reference／control。固定CDF8条件の13round条件付き会計 |
| 方式名 B0／B0I／B2／B3 | DFの方式ラベル（下表）。作業担当Bとは無関係 |

| 方式ラベル | 内容 |
|---|---|
| Full PF | 一体項と全DF fragmentを決定論PFで保持 |
| B0／B0I（Truncated PF） | prefixを保持し、tailを捨てる。B0Iは捨てたtailのidentity成分をscalarへ一回戻す |
| B2（PR-DF） | 一体項＋非空prefixを保持し、tailをfinite RTEで補う |
| B3 | 二体prefix 0。一体項は保持する。全Hの純RTEではない |
| PR-Pauli（DF head＋Pauli tail） | DF prefixを保持し、tailをPauli辞書のRTEで補う |
| Full-H pure RTE | 一体項を含む全Hをランダム化する。B3とは別方式 |

## 3. H4リソース評価の最終目的

### 3.1 評価対象

状態準備を除いた、RPE全roundの総計算資源を評価する。

$$
G=\sum_{m=0}^{M}\sum_{a\in\{X,Y\}} N_{m,a}\,\mathbb E[C_{m,a}]
$$

- $N_{m,a}$：対象精度・推定器・誤差契約の下で使うshot数。
- $\mathbb E[C_{m,a}]$：対象lawに対する、1 shot当たりの測定付きfull Hadamard wrapperの期待compiled RZ数。制御付き時間発展、scalar（総時間に一回）、X/Yの基底変換、測定を含む。
- 古典の価格標本数 n と normalization B は、N や C に重ねて掛けない。

**現行の検証で共通に使っている条件**（物理条件の出所は[4方式menu][MENU4]・[ベースライン設計][BASEOPT]。主精度評価の現在の採択状態は、ユーザー確認と[Bの固定REPORT][B_PF_FIXED]・[Pauli REPORT][B_PAULI_3Q]による。全候補集合・最終比較policyの採択とは区別する）

| 項目 | 内容 |
|---|---|
| 系 | Linear H4、1.00 Å、STO-3G、保存tol-only 1e−8 DF10（weighted-Hermitian completion）、sector (2,2)・36次元 |
| target | 保存DF10 full DF-Hのsolver energy proxy。厳密ground energyの包含・state誤差は未認証。DF前のHamiltonianへは変更しない |
| RPE | h=0.1、M=12（13 round、$T_m=h\,2^m$）、状態準備費 P=0 |
| 採択済み主精度評価 | Z1＋T1 total decoderのQ3直接モデルMSE。保存ideal finite-K raw・IID二項モデル、閾値 $\tau=(0.00159360010199040\,\mathrm{Ha})^2$。物理的なglobal RMSE保証とは別 |
| compiler | Qiskit 1.3.0、basis rz/sx/x/cx、opt1、seed17。lowering IDも価格の条件として一致させる |

### 3.2 比較方式：正式採択と検証用shortlistの区別

| 区分 | 内容 |
|---|---|
| 方式の集合 | PR-DF、PR-Pauli、Full PF、Truncated PF、Full-H pure RTE。作業はこの5方式で進んでいる |
| 候補（schedule）の集合 | **正式採択された候補集合はない。** 11/13の11 schedule案は提案であり、最適候補群でもない（[4方式menu][MENU4]） |
| 現在の検証用候補 | Cの5方式・14 schedule・364軸の[固定対応表][C_ALIGNMENT]と§5。対応表完成は、最終候補選定や適格N・E[C]の完成を意味しない |
| 価格のshortlist（決定論PF） | S2・幅16h/7、Y4・幅16h/3、M5・幅16h（[D 保存4件binding][D_SAVED4]）。Full PFの局所予算内候補（[A追加5点][A_FPF_FU]）に基づく検証用の集合 |

### 3.3 二つの比較水準

| 水準 | N の意味 | 現状 |
|---|---|---|
| 第1水準 $G_{\mathrm{paper}}$ | 原論文型のnominal shot数 $\lceil B_m^2[11+4(M-m)]\rceil$。経験的な名目値で、精度の保証ではない | 会計の実装は人工値でPASS（[C 価格経路][C_ROUTES]）。実データでの計算は0件 |
| 第2水準 $G_{\mathrm{qualified}}$ | 採択した精度条件の下で根拠のあるshot数 | 主精度のQ3直接モデルMSEは採択済み。5方式の特定候補に固定配分での条件付き適合はあるが、物理的に十分なN・global RMSE・正式Gは未確定 |

**nominal N を十分 N として扱わない。** 両水準の出力は別fieldで管理されている。

### 3.4 直近の到達点：評価済み固定配分での全13round比較

各方式から少なくとも1つのQ3適合済み物理候補を使い、その候補で**実際に評価した** $N_X,N_Y$ とm0〜m12・X/Y計26軸の測定付きfull controlled Hadamard wrapperのcompiled RZ価格を結合する。これは第1水準のnominal Nでも、十分Nを認定する第2水準でもない。既存の二水準の定義は変更しない。

決定論PFでは直接価格と採択済み条件付き閉包価格を区別して総額が公開済み（[B2][B2_PF_TOTAL]）。ランダム化方式では対象lawの期待価格、その評価誤差・適用条件が不足する。有限path平均・凍結予測・期待残差差の上界はそのまま別種の証拠として示し、互いを置換して同じ期待総額や順位を確定しない。別の条件付きモデルシナリオや上界による比較を採用するなら、law・26軸の根拠・不確かさと主張範囲を先に明示し、別判断を受ける。

状態準備P=0、候補/round/axis/law/compiler/lowering/scalarの束縛を維持する。十分Nの認定や物理的global RMSE保証を、この最初の比較の前提として追加しない。一方、固定配分でのモデル適合を物理保証・正式G・5方式のwinnerへ昇格しない。

## 4. 研究課題と実装・検証の対応（旧棚卸しの証拠map）

この節の到達点はad6ab45bb時点の保存記録であり、現在地は§0・§5を参照。価格・配分の数値更新はC比較表のみで管理する。

実装pathは、併記した固定commitに存在することを確認した。`main` には無いものが多い。

### 4.1 量子回路実装（実回路・loweringのcorrectness）

| 項目 | 内容 |
|---|---|
| 研究上の問い | 価格を測る回路が、意図した制御付き時間発展・scalar一回・X/Y測定を正しく実装しているか |
| 実装 | D `acea1118a`：`src/trottertracks/pauli_tail_price_connection.py`、`scripts/run_full_pf_shortlist_prices.py`、`scripts/run_truncated_p7_prices.py`、`scripts/run_pauli_ld1_m0_pilot.py`。A `7a0c9468b`：`scripts/run_full_h_kappa2_p1.py`。新ZZ lowering：`26d4f23e7`の`src/trottertracks/efficient_controlled_zz.py` |
| 検証 | [Full-H P-1再検証][A_P1]（ancilla命令列の完全一致など23条件）、[Pauli B0 run02][A_B0]（固定の実H4数値gate）、[p7保存回路のX/Y識別][C_ADM]（保存QPYのfingerprint）、[PR-DF P2 direct pilot][D_P2] |
| 到達点 | PR-DF：LD1 m0/m6に続きm3・m4のreceipt接続を完了、m4のC受入8/8・比較2/2（[C][C_PRDF_M4_ADMISSION]）。新しい数値作用gateは元P2 fresh同様未実施。PR-Pauli：LD1 m0、LD2 B1-Wに加えCAL18 compile・独立m5/m6 holdout6 compileの対象pathで実装gateを確認（[D2 CAL][D2_CAL]、[holdout][D2_HOLDOUT]）。決定論PF：m9・m0・m1/m2も元作用gate・C implementation guardを通過（[D m9][D_PF_M9_RESULT]、[m0][D_PF_M0]、[m1/m2][D_PF_M12]）。Full-H：D7全18軸のD1最終検査PASS、Cも10較正record・元18軸の保存証拠を受入済み。index0の7数値gateをindex1へ拡張していない（[D1][D1_FINAL]、[C][C_FULLH_FREEZE]） |
| 科学的資格 | 検査したpath・cellに限った配線の証拠。価格の期待値や他roundの正しさは示さない |
| 不足 | PR-Pauli LD2のorder2 product・非自明event位相は未観測UNKNOWN。長いroundの直接検査・モデル外挿資格は未確定 |
| 次の作業 | 完了した配線・取得の証拠を再利用する。Pauliの記述的holdout完了を未観測eventへ一般化しない。各取得系列の後続範囲は§8・§10による |
| 担当・依存 | D（決定論PF）、D1（Full-H）、D2（Pauli・PR-DF LD1の別取得系列）、A（設計・検証）、C（guard受入）を区別する |
| 完了条件 | 価格を使う全ての回路族について、少なくとも代表cellの配線証拠がある |

### 4.2 物理信号

| 項目 | 内容 |
|---|---|
| 研究上の問い | 各方式・各roundの平均信号（raw、補正後、normalization）は何か |
| 実装 | A `714247619`：`src/trottertracks/pr_rpe_signal_provider.py`、`df_head_pauli_tail_p0.py`／`_p1.py`／`_completion.py`、`pf_effective_hamiltonian.py`、`scripts/run_h4_full_pf_heff_scan.py`。A `7a0c9468b`：`baseline_signal_pipeline.py`、`baseline_signal_numerics.py`、`scripts/run_truncated_fullh_signals.py`、`scripts/run_full_h_kappa2_signals.py` |
| 検証 | [PR-DF 26round][A_PRDF]、[PR-Pauli 26round][A_PAULI]、[B0I p7・Full-H R448 各13round][A_TRSIG]、[Full-H κ2 13round][A_S1]、[Full PF H_eff格子][A_FPF_SCAN]・[追加5点][A_FPF_FU]、[Truncated T1][T1]・[T2][T2]、[決定論PF shortlist6候補完成][A_PF_COMPLETE] |
| 到達点 | 上記の既存packetを保全。加えてFull PF・B0I p7の各3候補が各13round完成（保存30再利用、新規48 PASS）、B intake6/6 PASS。[科学SOURCE][A_PF_SOURCE] `fb9a1878b`、[raw carrier][A_PF_RAW] `06a15ee45`、[packet索引][A_PF_PACKETS]と[HANDOFF][A_PF_HANDOFF]を参照 |
| 科学的資格 | 保存した数値stateとideal finite-K lawに条件付いた解析的平均。基底状態の認証と、実samplerのlawへの移送は未取得 |
| 不足 | 今回のshortlist6候補について信号不足・B受入待ちは解消。補助候補Full PF S2・幅0.1の未完備状態は別候補として維持する |
| 次の作業 | 完成packetを再利用する。Aの取得作業はmandatory STOP。追加meanの取得をこの文書から指示しない |
| 担当・依存 | A（信号）、C（baseline metadata）、B（受入） |
| 完了条件 | 比較に使う各物理候補について、全13roundの信号packetがSOURCE付きで受入済みである |

### 4.3 誤差評価（推定器・MSE）

| 項目 | 内容 |
|---|---|
| 研究上の問い | 与えたshot配分の下で、エネルギー推定の誤差（MSE）はどこまで抑えられるか |
| 実装 | B `999304918`：`src/trottertracks/rpe_total_decoder.py`、`rpe_total_risk.py`、`rpe_deterministic_risk.py`、`rpe_total_binding.py`、`rpe_c_interfaces.py`、`pr_rpe_rmse_controller.py`。test：`tests/check_rpe_total_decoder_risk.py` |
| 検証 | [精度契約W0][B_W0]、[Z1＋T1/Q3][B_Z1]、[PR-DF旧部分decoder DP][B_DP]、[Pauli元control][B_Q3]・[3/4配分][B_PAULI_3Q]、[PF固定6評価][B_PF_FIXED]、[PR-DF/Full-H元control][B_PRDF_FULLH]、[Full-H κ2・2倍][B_FULLH_2X]・[1.5倍][B_FULLH_3O2] |
| 到達点 | PF共通固定shot6/6、Pauli3/4配分2件、PR-DF固定control2件が適合。Full-H κ2は元control不適合・固定1.5倍適合・保存2倍適合の3成果がある（[B][B_FULLH_3O2]）。1.5倍は1 call完了・mandatory STOP、原API/資源記録保存済み。PF6件は保存componentから判定復元、実行記録部分保存・最終API return/CPU/peak RAM欠測のまま |
| 科学的資格 | 採択済みQ3直接モデルMSEでの条件付き適合。保存solver proxy・ideal raw・IID・未認証allowanceの範囲に限る。物理的な精度保証・十分Nではない。PFの科学worker完了とdriver記録UNKNOWNを区別する |
| 不足 | 物理的な資格・必要N最適化は未完了。Full-H R448・補助候補・未評価vectorへ結果を移さない。PR-DF/Full-H新評価のAPI・資源記録は保存済みだが、PF6件の欠測を補うものではない |
| 次の作業 | 各固定REPORT/HANDOFFを再利用。Full-H初回比較は1.5倍配分を使い、追加shot探索・Q3再評価を必要作業に加えない。PFの記録欠測を埋める再実行も要求しない。未評価vectorや物理的資格は別判断 |
| 担当・依存 | B。信号（§4.2）に依存 |
| 完了条件 | 採択した契約の下で、比較する全候補のMSE評価が同じclaim水準で揃う |

Full-H κ2の3配分の保存結果（[B REPORT][B_FULLH_3O2]からの表示値の転記、Ha²）。採否は元REPORTのexact端点で判定済みであり、本書で再計算していない。

| 配分 | 保存MSE区間（概数） | 採択済みQ3閾値への適合 |
|---|---|---|
| 元control | [1.6244310×10⁻⁴, 1.6336187×10⁻⁴] | 不適合 |
| 固定1.5倍 | [2.0984341×10⁻⁶, 2.1058026×10⁻⁶] | 適合・初回比較の第一候補 |
| 保存2倍 | [7.6023913×10⁻⁷, 7.6038351×10⁻⁷] | 適合・保存参照 |

### 4.4 必要shot数 N

| 項目 | 内容 |
|---|---|
| 研究上の問い | 目標精度を満たすために、各round・各軸で何shot必要か |
| 実装 | B `999304918`：`src/trottertracks/rpe_shot_allocation.py`、`pr_rpe_rmse.py`（nominal）、`rpe_total_binding.py`の`evaluate`入口。C `855573064`：`paper_nominal_cost.py`の`nominal_packet` |
| 検証 | 旧[有限計画][B_NPLAN]・[準備報告][B_N]を履歴として保持。完了成果は[Pauli3/4配分][B_PAULI_3Q]、[PF6候補][B_PF_FIXED]、[PR-DF固定control][B_PRDF_FULLH]、[Full-H κ2・2倍配分][B_FULLH_2X]・[1.5倍配分][B_FULLH_3O2] |
| 到達点 | X/Y各軸共通、m0〜m12順。PF6候補＝`[59,55,51,47,43,39,35,31,27,23,19,15,11]`（[B][B_PF_FIXED]）。Pauli LD1/LD2＝`[48,45,42,39,36,33,29,26,23,19,16,13,9]`（[B][B_PAULI_3Q]）。PR-DF LD1/LD2＝`[64,60,56,52,47,43,38,34,30,25,21,17,12]`（[B][B_PRDF_FULLH]）。**Full-H κ2の初回比較＝1.5倍 `[96,90,84,78,71,65,57,51,45,38,32,26,18]`、allocation `FULL_H_KAPPA2_TOTAL_Q3_USER_SCALE_3O2_01`**（[B HANDOFF][B_FULLH_3O2_HANDOFF]）。保存2倍 `[128,120,112,104,94,86,76,68,60,50,42,34,24]` も適合。元control（PR-DFと同じvector）は不適合。十分N・最適Nではなく`sufficient_N=null` |
| 科学的資格 | 評価済み配分の条件付きモデル閾値適合のみ。物理的に十分なN・global RMSE保証は未資格 |
| 不足 | 未評価vectorへの補間、最小N、cost-aware配分、物理的誤差・law移送の資格。旧有限計画全体が実行済みになったわけではない |
| 次の作業 | Bは各固定評価の完了後mandatory STOP。Full-H初回比較に追加倍率・round別探索は不要。1.5倍は全round/両軸で2倍よりshot数が少なく両者適合だが、未評価vectorのMSE単調性・最小Nは推論しない。後続は同じκ2の価格26軸との結合 |
| 担当・依存 | B。§4.3に依存。価格とは独立に進められる |
| 完了条件 | 各候補について、採択した契約で閾値を満たすwhole-vectorが、根拠とともに保存される |

### 4.5 compiled価格（直接価格・有限cohort）

| 項目 | 内容 |
|---|---|
| 実装・受入 | Dの既存caller、C `price_implementation_guard` / `rpe_d_price_connection` / `price_context`を再利用。固定compiler/loweringとfull measured wrapper/P=0の束縛を保つ |
| 決定論PF | [m3受入][C_PF_M3]、[較正][D_STAGE1]、[m6][D_PF_M6]、[m9][D_PF_M9_RESULT]、[m0][D_PF_M0]、[m1/m2][D_PF_M12]で**直接96/156軸**。残60軸は新compileしておらず、[Cの条件付き導出表][C_PF60_TABLE]で埋めた。m0/m1/m2はモデル判定なし、m6/m9の各12/12 PASSは指定点だけ |
| PR-DF | LD1 m0/m6 P1各2・P2 fresh各4 pathと、m3較正/独立m4各fresh4 pathの有限価格。m4はC receipt8/8・比較2/2受入、保留0（[C][C_PRDF_M4_ADMISSION]）。LD2の同条件compiled価格は確認範囲で未同定 |
| PR-Pauli | LD1 B0のm0 X/Y各1 pathはwiring専用。LD2 W m0 X/Y各1 path、CAL5非configuration cell・18 compile、独立m5/m6の3 fresh path・6 compileは公開済み（[W][D2_W]、[CAL][D2_CAL]、[holdout][D2_HOLDOUT]）。m9は予測固定済み、取得開始/結果公開未確認（[準備][PAULI_M9_PREP]） |
| Full-H κ2 | D7較正5 cell・9 path・18 compile、D1検査PASS18・C受入10 record完了（[D1][D1_FINAL]、[C][C_FULLH_HANDOFF]）。新holdoutはm5のfresh2 path・4 compile・取得/必要gate PASSを公開済み（[D1][D1_M5_STATUS]）。m6 fresh1 path・2 compileも取得/必要gate PASSを公開済み（[D1][D1_M6_STATUS]）。最終4比較の完了は未確認。旧P-1 m3はwiring専用 |
| 科学的資格 | RTEの各有限cohortは観測価格。X/Yは同pathの2応答で標本数を二倍にしない。Pauli/Full-H較正の非configuration cellを通常13roundの軸価格へ付け替えない。Full-H G36223の保存4 compile再利用は18の内数・追加0、paired P2R566は親G566と非独立 |
| 次・完了条件 | 既取得の再compileは不要。取得中の限定holdoutの固定成果だけを後続受入へ渡す。全26軸期待価格の資格・計画は§4.6/§8で別管理する |

### 4.6 期待価格モデル（E[C]）

| 項目 | 到達点・不足・次の依存 |
|---|---|
| 決定論PF | [C推奨][C_PF60_REC]の閉包条件はユーザー採択済み、[60価格公開][C_PF60]、[B2全13round集計][B2_PF_TOTAL]まで完了。60軸のclassificationはMODEL_PREDICTION、evidence_kindはFIXED_COMPILER_CLOSURE_DERIVATION_CONDITIONAL。障壁/反復/終端構造の持続、installed Rustと参照sourceの対応、未調査規模分岐の前提に限る。無条件QUALIFIED_MODEL・直接観測にはしない。2%保証・誤差上界・CIはnull、元モデルqualified_for_candidate=falseを保全 |
| PR-DF LD1 | E_I[S]26/26、m0/m3/m6による軸別残差2モデルとm4予測2件の凍結・記述的比較/C受入は完了。全13roundの[計画][D2_PRDF_EC_PLAN]・[F0受入][C_PRDF_F0]・[26軸条件付き残差差上界][D2_PRDF_RANGE]を追加。上界は保存binary64 ideal weightのlaw Iと**独立一様53-bit格子law**の間のTV×SOURCE envelopeによる差の絶対上界であり、符号付き補正・有限標本誤差・モデル誤差ではない。実PCG64のseed ensemble/joint streamへの接続は未確定、actual PCG64 price shiftはnull。新26上界のC受入は未確認 |
| PR-DF LD2 | 同条件の構造期待値・compiled価格・凍結モデルは確認範囲で未同定。LD1の数値を移送せず、LD2の保存候補/辞書/入口に合うkernel・価格設計の技術照合はLD1/PFを待たずに可能 |
| Pauli LD2 | 2モデル凍結後のm5/m6は4/4有効・DESCRIPTIVE_COMPLETE。m9案B採択・同係数のX/Y2予測凍結公開済み、取得開始/結果未確認。長qの有限cohort比較を増やしても、母law平均・有限標本/共有係数誤差・未観測round移送は別に残る。Pauli LD1へ係数/辞書を移植しない |
| Full-H κ2 | D7の軸別3係数2モデル・m5/m6の4予測は凍結済み、CALIBRATED_UNVALIDATED・qualified_for_candidate=falseを維持（[C][C_FULLH_HANDOFF]）。m5・m6は取得公開、最終比較は未確認（[m6][D1_M6_STATUS]）。対象law・全26軸の期待価格根拠、有限cohort/paired依存・共有不確かさ・長round移送は未資格。1.5倍shot適合は価格資格を付与しない |
| 共通の境界 | PR-DF m4・Pauli m5/m6・Full-H採択規則は数値許容差null・モデルPASS/FAILなし。決定論PFの2%を移さない。希少イベント単独の追加調査は今回の優先作業から外し、後続の期待価格設計では全supportを含める必要を保持する |
| 担当・完了条件 | C/C2（モデル・受入）、D/D1/D2（別契約の取得）、A等の理論/設計担当。独立比較完了と全round/lawの価格資格は別判断。今できる静的準備と新しい科学実行の境界は§8.2 |

### 4.7 総資源会計（G）

| 項目 | 内容 |
|---|---|
| 既存実装 | C `paper_nominal_cost` / `rpe_optimization_pipeline`、Bのbinding/会計に加え、[B2固定配分集計][B2_PF_TOTAL]の`allround.py`・入力`43f87397a`。新しい管理システムは不要 |
| 到達点 | PF6候補は同一固定Q3配分で直接96＋条件付き60軸を結合し、全13round総compiled RZ数・条件付き順位を公開済み。既知8round小計[540b54ab2][B_PF_PARTIAL]を入力として再利用。C価格公開待ち・B2集計待ちは解消、各担当STOP |
| 不足 | 他3方式は同一物理候補・対象lawの26軸期待価格と不確かさ/適用範囲。初回比較の評価済み配分は既にある。十分N・物理的global RMSE・正式G/5方式winnerは別の未解決事項 |
| 次・依存 | ランダム化各方式の期待価格計画・候補/軸/配分metadata照合は先行できる。各方式の価格表公開を待つのは数値の最終結合であり、全準備を一括待機させない。PR-DFとPFの差額比較には双方の同じclaim水準の総額が必要だが、PF側総額は今回受領済み |
| 資格 | PFの条件付き固定配分総額を第2水準Gへ昇格しない。他方式の有限path平均・点予測・条件付き上界を同じE[C]として混在させず、欠測0補完・新許容差の事後採択をしない |

## 5. 物理候補ごとの進捗

現在地は§0の依存状態、候補別・配分別の正本は[C比較表](h4_saved_rpe_cost_catalog_20261116/run01/REPORT.md)／[CATALOG](h4_saved_rpe_cost_catalog_20261116/run01/CATALOG.json)を参照する。公式・要求幅・実幅・L_D・κ・配分を保存し、方式名だけで統合しない。旧固定配分も履歴として保持する。

Full PFの有限探索完了、PR-Pauliの途中終了、p7の価格待ち解消は別の進捗として扱う。直接receipt、条件付き閉包、有限cohort、モデル外挿、保存総額のみの証拠を区別する。未取得単価や全law E[C]を補完しない。

## 6. 証拠の状態の区分

| 区分 | 意味 | 例 |
|---|---|---|
| 実装済み | コードがある | 第1水準の会計 |
| 人工・回帰テストPASS | 架空の入力で動作を確認した | guard、bridge、価格モデルの検査 |
| 実物理条件での検証PASS | 実H4の回路・数値でgateを通過した | P-1再検証、Pauli B0 |
| 科学的数値結果の保存あり | 承認された実行の結果が固定SOURCE付きで保存されている | 各方式の13round信号、Pauli Q3の2評価 |
| 特定条件下で利用可能 | 明示された条件の範囲で使える | 決定論PFのm0〜m6・m9直接価格96軸、評価済みvectorでの条件付きモデル閾値適合 |
| 最終比較へは未資格 | 値はあるが、最終比較に必要な資格が無い | 単一path価格、E[S]、control vectorでのMSE |
| 調査済み範囲では欠測 | 確認した範囲に記録が無い | PR-DF LD2の価格 |
| 原本・同等性が未確認 | あるかどうか、同じ条件かどうかを判断できていない | 補助候補の旧価格locatorと現行wrapper receiptの対応 |
| 科学結果保存・実行記録部分保存 | 保存componentで科学的判定を確認できるが、完全な実行記録はない | PF固定shot6件。最終API return・CPU・peak RAM欠測を併記し、再実行しない |
| 指示作成／送付／実行／成果受領 | 各段階を確認できた記録に従い区別する | PF60価格・B2集計は公開完了。Full-H m5/m6取得公開と最終比較待ちを区別。Pauli m9は案採択・予測固定公開、取得開始/結果未確認 |
| モデル凍結／独立比較完了 | 観測前の係数・予測固定と後続の独立比較を分ける | PF m6/m9・PR-DF m4・Pauli m5/m6は比較完了。Full-Hは4予測固定・m5/m6取得公開、最終比較未確認。Pauli m9は予測固定のみ。元凍結JSONは遡及変更しない |
| 記述的比較完了・モデル判定なし | 有効な独立観測と予測差を報告し、数値許容差によるモデル判定は行わない | PR-DF LD1 m4のC比較2/2受入とPauli LD2の4/4有効比較はいずれもDESCRIPTIVE_COMPLETE。モデルPASSではない |
| STOP記録の保全 | 個別契約の完了後STOPと後続作業を区別する | A/B/Cの完了成果、D m9・m0・m1/m2、D1 D7較正、D2各取得の旧STOPを保全。後続の限定holdout・計画/F0/上界公開はそれぞれ別scopeとして記録する |
| 次段階の別判断・実行指示待ち | 現在の指示の範囲に含まれない | RTEの全26軸期待価格評価・追加取得/fit、正式G。Full-H holdout・Pauli m9の限定scopeは既に別指示あり。PF60価格公開/集計待ちは解消 |

**区別を保つ点**
- 人工テストのPASSは、科学的な検証の完了ではない。
- **有限cohortの平均、構造期待値 E[S]、compiled期待価格 E[C] は別の量である。**
- 単一pathの観測を、lawの期待値として扱わない。
- 旧SOURCEの科学的資格を、新SOURCEへ自動で移さない。
- 「確認した範囲で未同定」は「未実施」ではない。
- D2の過去の[SCOPE_DEVIATION.md][D2_SCOPE]に記録された範囲外数値検査は保全し、W・CAL・今回のholdoutの科学的証拠・資格には採用しない。各観測pathのgate PASSと、その過去記録を混同しない。

## 7. 未解決課題と、解くと何が判断できるか

- 比較可能なのは指定した保存候補集合・採用価格規則・評価済み配分の条件付き費用であり、全探索での最適解や物理的winnerではない。
- 新しいQ3または価格を待つ候補、schema未対応・physical/law/contract不一致は[同期状況](h4_saved_rpe_cost_catalog_20261116/SYNC_STATUS.md)に固定SOURCE付きで残す。他の成果は独立に統合する。
- 実PCG64とideal lawの接続、有限cohort誤差、長q/Rモデル移送、共有誤差・CI・正式E[C]・十分N・global RMSEの未資格は保持する。
- 厳密ΔPF=1/5のPR-Pauliはm0/m1の同recipeとm2以降のprovider未対応を区別する。[SOURCE診断](pauli_exact_step_price_binding_20261116/REPORT.md)を参照し、一律丸めで価格を転記しない。

## 8. 依存関係と進行順序

A/B/B2/D/D2の公開固定成果 → Cのmetadata/契約照合 → Q3と価格の同構成・同配分join → C比較表。片側だけなら待ち状態を保持し、後着成果から更新する。進行中roundの完了を統合の前提にしない。

次工程は、§0と[同期状況](h4_saved_rpe_cost_catalog_20261116/SYNC_STATUS.md)の不足証拠を担当の固定REPORTで確認すること。保存受入・adapter差分で解消するものと、新しい科学実行・方針判断を要するものを分ける。同期処理は新しい科学call・retry・モデル採択を実行しない。

## 9. アルゴリズム改善系列とその他の資源評価

### 9.1 アルゴリズム改善系列

詳細は[11/01の全体地図][LAND]、[対応表][GAPS]、[11/10の統合調査][INTEG]にある。ここでは目的・到達点・残課題と、リソース評価への接続だけを示す。

| 系列 | 目的 | 主な到達点 | 残課題 | H4総資源比較への接続 |
|---|---|---|---|---|
| Track B／O2（return集約・sampling） | 同じ平均を安い回路・少ないshotで実現する | G6〜G10の構成、Closed P3/P5、固定syntheticでの資源交換、H4 C1の3方式1024path（[全体地図][LAND]§4）。11/08までのO2の成果は[O2統合][O2] | 独立した新規性と、認証N・native総資源での優位は未確立 | Closed P3はTrack Cへ接続済み。現在の5方式比較には入っていない |
| 元Track A／A-O1（表現・分割） | 辞書やcore＋residualで単価やλを下げる | O1-M1は単価低下を長roundの減衰が相殺する負結果。B2-P1は18roundの条件付きshot会計（[HP報告][A_HP]、[後続調査][ASURVEY]） | O1-M2のmodel適格性。H4のLD4〜7等は未実行 | PR-Pauliの[準備][A_PAULI_PREP]で、O1-M1とP3のSOURCEを照合している。旧結果の数値は現比較へ移していない |
| Track C／C-O3（Closed P3・共同選択） | h・resolution・α・整数shotを全roundで共同選択する | DF12・18roundで強Full／Truncatedと比較。最適化したClosed PRは強Fullより高価という結果（[共同選択][C_O3]、[統合報告][C_O3INT]） | 別条件（DF10、h=0.1、13round）への移送は未資格 | 強い決定論対照を落とさないという比較方針の根拠。数値順位は現比較へ混ぜない |
| Track D／D8（Reference／control） | whole-pathのReference構成でCX・depthを下げる | 固定CDF8条件で13roundの条件付き会計まで接続（[総計][D8_TOTAL]）。r16の有限価格校正（[報告][D8_R16]） | 絶対費用の大半が長qのproxy。全体CIとactual成功は未資格 | 測定同値性・十分shot・価格の三接続の型を再利用できる。A/F/S価格モデルは別条件へ移さない |
| 新A（適応readout） | 適応的な読み出しでshotを減らす | canonical条件で固定モデル内の保証と18roundのnative費用（[Q3][NEWA_Q3]）。Q4-1は実装と人工検証まで（[準備][NEWA_Q4]） | 強Full／Truncated／Closedへの公平な移送 | shot数の削減を直接扱う。現5方式のNには未接続 |
| 研究C（表現・PR-aware DF） | Hamiltonian表現を変えてλや誤差を下げる | CDF／ENTRYの3方式比較（[報告][RC]） | own-DFの結果を元Hのgroundへ移す根拠 | 未接続。target・stateの資格が先に必要 |
| C-α、閾値反射（元Track Aの後続） | gate lawの資格化、中心化による改善 | 固定gate lawで18roundの条件付きshot会計（[C-α][CALPHA]）。中心化は数学的な差を確認、総費用は未評価（[報告][THRESH]） | 全roundのcompiled期待価格 | 未接続 |

**接続できる可能性があることと、科学的資格を得て統合済みであることは別である。** 上表の「接続」は前者を述べている。異なる科学契約・STOPを統合していない。

### 9.2 H4以外・過去のリソース評価

| 対象 | 保存成果 | 現比較との関係 |
|---|---|---|
| 有限時間の資源地図（旧Track A、Pauli-JW、長時間B2/B3） | [全体地図][LAND]§2・§5 | 単一時刻の N×C。全RPEの比較ではない。決定論PFの保存4価格（`155529e20`）の出所でもある |
| H4 DF10の旧13round N、H4/H6のN/C coverage | [N batch][H4_NBATCH]、[coverage][H46_COV] | 旧契約の条件付きN。現在の比較のNへは流用しない（[4方式menu][MENU4]の方針） |
| H6 DF19のRPE m7信号と条件付きshot | [報告][H6_M7] | 一般性検証の入力候補。今回のH4比較には不要 |
| Full-H pure RTEのH4/H6/H8有限時間 | [T2最適化][FH_T2]、[H8][FH_H8] | 現κ2 scheduleの設計の出発点。旧loweringの価格は使わない |
| 実装監査・技術修正 | [repo全体監査][AUDIT] | 中核の物理・統計計算に、科学的結論を変える不具合は報告されていない。正式なG比較の前の注意点として、Full-Hの旧lowering価格とPRの新lowering価格を直接比べないこと、Gの大半を占めるm10〜m12の価格は直接測れない規模であることを挙げている |

## 10. 実験・研究計画の管理欄

現在の進行状態と次工程は§0・§8、公開探索の完了/途中終了はC比較表の各レコードに記録する。費用改善だけでは本書を変更しない。モデル資格・依存段階・停止理由のクラスが変わる場合だけ現在地を更新する。

研究方針・精度規約の採択は§2・§3・§6および各固定科学契約が正本であり、同期処理が新たな採択を行わない。旧指示・計画・SOURCE・raw・凍結・STOPは個別資料に保全する。

## 11. 出典不足・確認できなかった事項

本書の旧棚卸し正本は[ad6ab45bb](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ad6ab45bbcc7139bf499760d28efad794e4eae9b/docs/research/RESEARCH_ROADMAP.md)。当時の状態を現在の進捗として再掲しない。今回の公開branch範囲・固定commit・未取込み理由は[PUBLICATIONS](h4_saved_rpe_cost_catalog_20261116/PUBLICATIONS.json)と[同期状況](h4_saved_rpe_cost_catalog_20261116/SYNC_STATUS.md)を参照。

## 12. 更新規則

**更新責任と引渡し**

- A/B/C/D等の個別研究担当は、自分の固定REPORT・SOURCEを公開し、原結果を保全する。作業完了時に、研究目的への達成状況、実装・検証の進捗、科学的資格・適用範囲、未解決課題、次に必要な作業、依存関係・担当、正式採択・実行承認・STOPへの重要な影響を判断する。
- 重要な影響がある場合、最終報告または既存REPORTに、変更すべき研究課題、完了した実装・検証、新しい証拠と固定SOURCE・REPORT、解消した課題、残る未解決事項、次の作業・依存関係・承認要否を簡潔に示す。これは統合担当への引渡しであり、管理専用の新しいファイル・台帳は不要である。
- 本書は、統合担当として明示的に割り当てられたCodexが各固定REPORT・SOURCEを照合し、重要な変更だけを更新する。個別担当は、別途明示指示がない限り本書を直接編集しない。
- 更新は重要な成果が揃った時点でまとめて行ってよい。個別担当の作業完了条件に、本書の即時編集を追加しない。同一文書への競合編集を避ける。

**更新する契機**

- 新しい固定科学結果の公開。
- 他担当による成果の正式な受入。
- 重要な検証のPASS／FAIL／UNKNOWN。
- 科学的資格・適用範囲の変更。
- ユーザーによる研究方針・比較policyの採択。
- STOP条件や依存関係の変更。
- 軽微なバグ修正、人工・回帰テストの追加、同じ結論の再確認、研究上の進捗に影響しない文書修正、実行条件・科学的資格を変えない内部整理では、原則として本書の更新は不要である。必要なら最終報告で「ロードマップ更新不要」と示す。

**更新の仕方**

- 原則として、影響する行（§0：現在地、§5：進捗、§7：未解決課題、§10：計画管理）と依存関係（§8）だけを直す。元REPORTと固定SOURCE・commitへのリンクを加える。
- 必要な場合は冒頭の最新確認日時を、実際に確認した範囲に合わせて更新する。運用規則の整備だけを、研究状況を再確認したこととして扱わない。
- 日々の小さなテストやcommitのたびに全体を書き換えない。
- 過去の状態は個別REPORTに残っているので、本書に履歴を積み上げない。
- 2026-11-16の利用者指示により、Cが公開成果の検出・比較表と現在地の更新を担当する。専用C checkoutと利用者自身のcron内で定期同期を行う。研究方針・科学契約・承認・STOPは変更しない。承認の証明基盤や科学実行の自動起動は作らない。

**科学的な資格と実行状態**

- §6の区分を維持し、実装済み／科学的検証済み、人工テストPASS／実物理条件でのPASS、保存結果あり／最終比較に使用可能、有限標本の観測／期待値・保証を区別する。
- 提案／正式採択、実行準備完了／実行承認／実行完了、未取得／所在未確認／科学的同等性未確認、mandatory STOP／再開条件成立も区別する。本書の記載変更だけで、科学的資格・実行許可・STOPが自動的に変わることはない。

**本書と個別証拠の関係**

- 本書は研究全体の管理・判断の入口であり、科学的結果の正本である各REPORT・raw・固定SOURCEを要約で置き換えない。異なる研究系列、物理候補、科学契約、SOURCEを無条件に統合しない。
- 個別REPORT・検証文書・研究ノートは各担当が維持する。研究全体の概要・索引は必要な場合に限り、manifestは科学的な証拠の構成・状態が変化した場合に更新し、既存契約で必須の文書更新を保つ。各作業のたびに全体を全面更新しない。
- 新しい科学計算・compile・実験の前には、共通運用指示 `research_project_instructions_unified_2026-11-14.md` §10.5の既存成果との重複照合を適用する。ロードマップの更新は、その実行承認や科学的な方針・契約の変更を代行しない。

---

作成：2026-11-14 JST、Claude Code（主agentのみ。独立レビューではない）。branch `codex/research-roadmap-integration-20261114`（基点 `main` `546bca7ed`）。実施したのは、公開済み資料・SOURCEの読取、Markdownの作成、リンクの確認だけである。新しい科学計算、保存値の再解析、sampling、回路build/compile、MSE・N・E[C]・Gの計算、採択は行っていない。**mandatory STOP**：本書を根拠に、科学計算・compile・価格取得・方針変更・`main`への統合へは自動で進まない。

<!-- links -->
<!-- 固定commitの参照。科学実行のSOURCEは各REPORTの記載を使う。 -->
[GAPS]: pr_repository_claim_evidence_gap_matrix_20261101.md
[GOAL]: 研究目的・研究課題.md
[INDEX]: RESEARCH_RESULTS_INDEX.md
[LAND]: pr_repository_scientific_landscape_20261101.md
[MAP]: ../../PROJECT_MAP.md
[METHOD]: 研究方法・解析手順.md
[VS]: ../../VALIDATION_STATUS.md
[A_PF_COMPLETE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1adb7e65a9e7c3adc32491d853f31d124393ead7/docs/research/pf_shortlist_mean_completion_20261115/REPORT.md
[A_PF_SOURCE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/fb9a1878b677b7ee116b52e804e5a01cdaa96004/scripts/run_pf_shortlist_mean_completion.py
[A_PF_RAW]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/06a15ee45beb177c018f6260ea4a5ac8cf66cfe6/docs/research/pf_shortlist_mean_completion_20261115/run01/RESULT.json
[A_PF_PACKETS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1adb7e65a9e7c3adc32491d853f31d124393ead7/docs/research/pf_shortlist_mean_completion_20261115/PACKET_INDEX.json
[A_PF_HANDOFF]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1adb7e65a9e7c3adc32491d853f31d124393ead7/docs/research/pf_shortlist_mean_completion_20261115/HANDOFF_C.md
[B_PF_FIXED]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8dba096dc39788b8948af9eafe394f03d386c520/docs/research/pf_shortlist_q3_fixed_results_20261115/REPORT.md
[B_PF_HANDOFF]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8dba096dc39788b8948af9eafe394f03d386c520/docs/research/pf_shortlist_q3_fixed_results_20261115/HANDOFF.md
[B_PAULI_3Q]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/50be68b803a90fd893a1499ebce74e033225144d/docs/research/pauli_q3_three_quarter_20261115/REPORT.md
[B_PRDF_FULLH]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a82b0f0ec7f0f11d9981ddb2343fcd5378a87503/docs/research/pr_df_fullh_q3_fixed_results_20261115/REPORT.md
[B_PRDF_FULLH_HANDOFF]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a82b0f0ec7f0f11d9981ddb2343fcd5378a87503/docs/research/pr_df_fullh_q3_fixed_results_20261115/HANDOFF.md
[B_FULLH_2X]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/422db93fc7bb599a7ea46c5b58e5dec0fda9ef08/docs/research/full_h_kappa2_scale2_q3_results_20261115/REPORT.md
[B_FULLH_2X_HANDOFF]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/422db93fc7bb599a7ea46c5b58e5dec0fda9ef08/docs/research/full_h_kappa2_scale2_q3_results_20261115/HANDOFF.md
[C_PRDF_ADMISSION]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8eef0ad7a9b04e04d766f5e05b3b32eee50a794e/docs/research/pr_df_ld1_q3_price_admission_20261115/REPORT.md
[C_PRDF_HANDOFF]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8eef0ad7a9b04e04d766f5e05b3b32eee50a794e/docs/research/pr_df_ld1_q3_price_admission_20261115/HANDOFF.md
[C_PRDF_NEXT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8eef0ad7a9b04e04d766f5e05b3b32eee50a794e/docs/research/pr_df_ld1_q3_price_admission_20261115/NEXT_VALIDATION_PLAN.md
[D2_PRDF_M3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2592d442ae8f36f71e42820a26e847b1fa38b59d/docs/research/d_pr_df_ld1_m3_results_20261115/HANDOFF_C.md
[C_PRDF_FREEZE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3cac53da1a4a8b2228245593c74c1ff9ec1f5212/docs/research/pr_df_ld1_residual_freeze_20261115/REPORT.md
[D2_PRDF_M4]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/063f2b3b6ea271a179a469be91252fa4e5a7d096/docs/research/d_pr_df_ld1_m4_holdout_results_20261115/HANDOFF_C.md
[C_PRDF_M4_ADMISSION]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9edfa5d3201f22c23dd77569d102801be80dacf3/docs/research/pr_df_ld1_m4_holdout_admission_20261115/HANDOFF.md
[C_PF_M3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3b1301738507d612694bf865d904b9376e038f69/docs/research/pf_m3_saved_price_admission_20261115/REPORT.md
[C_PF_M9]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/04a68f5932f53377595b2d69bbf8f46e02e82a01/docs/research/pf_m9_prediction_freeze_20261115/REPORT.md
[C_PF_M9_HANDOFF]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/04a68f5932f53377595b2d69bbf8f46e02e82a01/docs/research/pf_m9_prediction_freeze_20261115/HANDOFF_D.md
[D_PF_M6]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a55813373b84b9128cbd10c6812c0d26143a4280/docs/research/d_pf_stage2_holdout_20261115/REPORT.md
[D_PF_M6_HANDOFF]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a55813373b84b9128cbd10c6812c0d26143a4280/docs/research/d_pf_stage2_holdout_20261115/HANDOFF.md
[D_PF_M9_EXEC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c02e2416ad8ffd63b800a4d1039e8ec1e6cc75aa/docs/research/d_pf_m9_holdout_20261115/EXECUTION.md
[D_PF_M9_PROGRESS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7ed93d24dbd2c9d49d8951549c976c9eb463d4e5/docs/research/d_pf_m9_holdout_20261115/ACCELERATION.md
[D_PF_M9_RESULT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0a35101914c6833244b6e0bad7bc2767a49b5a47/docs/research/d_pf_m9_holdout_20261115/HANDOFF.md
[D_PF_M0]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c95353a44b4ce99ed86e949e408d522d6ad7e7fe/docs/research/d_pf_m0_direct_prices_20261115/HANDOFF.md
[D_PF_M12]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6e110c083ec2db2530cdda3f7b5c84d2a7d3f31c/docs/research/d_pf_m1_m2_direct_prices_20261115/HANDOFF.md
[D1_PARTIAL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4768ca3cd8a89684650d0af90900457d54ed2a77/docs/research/d_full_h_p2_calibration_resume_20261115/REPORT.md
[D1_PARTIAL_HANDOFF]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4768ca3cd8a89684650d0af90900457d54ed2a77/docs/research/d_full_h_p2_calibration_resume_20261115/HANDOFF_C.md
[D1_CONTINUATION]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6376e8fddc0828e274989a66f707638e9b8b0e3d/docs/research/d_full_h_p2_calibration_resume_20261115/CONTINUE.md
[D1_FINAL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e628c0d69c7da559369d99f323e3656a96c0728a/docs/research/d_full_h_p2_calibration_resume_20261115/HANDOFF_C.md
[C_FULLH_FREEZE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/81596ce7c561d7b75890cfe2758a0d4a12fb6b47/docs/research/full_h_d7_model_freeze_20261115/REPORT.md
[C_FULLH_HANDOFF]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/81596ce7c561d7b75890cfe2758a0d4a12fb6b47/docs/research/full_h_d7_model_freeze_20261115/HANDOFF_D1.md
[C2_FULLH_RULE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f0765067a5a8b39888e36f3a8b80f1bdbda69223/docs/research/full_h_d7_holdout_rule_proposal_20261115/PROPOSAL.md
[D2_CAL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5dac74959ed9ea49465d393fc792ef5b4727edba/docs/research/d_pauli_ld2_b1_cal_results_20261115/REPORT.md
[D2_FROZEN_MODELS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b74e4e536392c2bbe94cd371436615fc56c0564f/docs/research/d_pauli_ld2_b1_cal_results_20261115/c2_freeze/run01/FROZEN_MODELS.json
[D2_HOLDOUT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/705f59ea863e1b9fe90e023f1cb1057d00366b16/docs/research/d_pauli_ld2_b1_holdout_results_20261115/REPORT.md
[D2_HOLDOUT_HANDOFF]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/705f59ea863e1b9fe90e023f1cb1057d00366b16/docs/research/d_pauli_ld2_b1_holdout_results_20261115/HANDOFF_C2.md
[C_ALIGNMENT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0b4b2d941cd669aaa45231e1236bc2d3158a9d12/docs/research/h4_five_method_candidate_alignment_20261115/REPORT.md
[C_FREEZE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f0fc056464b63551d8f7795766bc0983d5863f95/docs/research/pf_stage1_model_freeze_20261115/REPORT.md
[C_FREEZE_HANDOFF]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f0fc056464b63551d8f7795766bc0983d5863f95/docs/research/pf_stage1_model_freeze_20261115/HANDOFF_D.md
[C_FROZEN_MODELS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/12372fa6b53c878dc5dd124e036572ba0cd82061/docs/research/pf_stage1_model_freeze_20261115/run01/FROZEN_MODELS.json
[D_STAGE1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b190bd815c886446291fac8b281c3c7df7343372/docs/research/d_pf_stage1_calibration_20261115/HANDOFF_C.md
[D_STAGE1_REPORT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b190bd815c886446291fac8b281c3c7df7343372/docs/research/d_pf_stage1_calibration_20261115/REPORT.md
[D1_STOP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/df0e11fd1e8d0f09166db71646d3df101ab13233/docs/research/d_full_h_p2_calibration_20261115/REPORT.md
[D1_STOP_HANDOFF]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/df0e11fd1e8d0f09166db71646d3df101ab13233/docs/research/d_full_h_p2_calibration_20261115/HANDOFF_C.md
[D1_RESUME]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d323fee1ca8cd6893c4b705f990cdc931bdb5c87/docs/research/d_full_h_p2_calibration_resume_20261115/SOURCE_PREPARATION.md
[D2_W]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1e2639a1be46ebea618b15591fb1e886bc499d33/docs/research/d_pauli_ld2_b1_w_results_20261115/REPORT.md
[D2_HANDOFF]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1e2639a1be46ebea618b15591fb1e886bc499d33/docs/research/d_pauli_ld2_b1_w_results_20261115/HANDOFF_C.md
[D2_SCOPE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1a7eed91dd96992cc6abcbfa1cd473a2044db16c/docs/research/d_pauli_b1_caller_preparation_20261114/SCOPE_DEVIATION.md
[ASURVEY]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/162f3d3f0f42973048fd14a7a6624e147a1e961a/docs/research/track_a_algo_improvement_survey_20261109/README.md
[AUDIT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/51d324ea9c83bc12f1e866da0fa409cef926f74a/docs/research/pr_resource_repo_wide_audit_20261114/REPORT.md
[A_B0]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/00f74f86d16cc6878e1d4c5977eb9abd9ce9c076/docs/research/d_pauli_ld1_m0_pilot_results_20261114/RECOVERY_REPORT.md
[A_B1PLAN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7a0c9468b2499fac9fa1cd279ae2980c9dab048e/docs/research/a_full_h_p2_pauli_b1_price_design_20261114/A2_PAULI_B1_PLAN.md
[A_DESIGN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7a0c9468b2499fac9fa1cd279ae2980c9dab048e/docs/research/a_full_h_p2_pauli_b1_price_design_20261114/REPORT.md
[A_FPF_FU]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/714247619afb4714fd663eb15c0470ef32bdc7af/docs/research/h4_full_pf_followup_results_20261114/REPORT.md
[A_FPF_SCAN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/714247619afb4714fd663eb15c0470ef32bdc7af/docs/research/h4_full_pf_heff_scan_results_20261114/REPORT.md
[A_HP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e162ef708d7f721cd9c95ae07a833f7301d2b449/docs/research/track_a_b2_p1_hp_branch_shots_20261107/REPORT.md
[A_P1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/a04470c0ad489eea1798856e648f863f9092cfb1/docs/research/d_full_h_kappa2_p1_results_20261114/run02_revalidation01/REPORT.md
[A_P2PLAN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7a0c9468b2499fac9fa1cd279ae2980c9dab048e/docs/research/a_full_h_p2_pauli_b1_price_design_20261114/A1_FULL_H_P2_PLAN.md
[A_PAULI]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/714247619afb4714fd663eb15c0470ef32bdc7af/docs/research/df_head_pauli_tail_all_round_results_20261114/REPORT.md
[A_PAULI_PREP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/714247619afb4714fd663eb15c0470ef32bdc7af/docs/research/df_head_pauli_tail_preparation_20261114/REPORT.md
[A_PRDF]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/714247619afb4714fd663eb15c0470ef32bdc7af/docs/research/pr_rpe_first_signal_results_20261113/REPORT.md
[A_S1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ce6aacfaca8249c510baa2d958849cffc7649cdc/docs/research/d_full_h_kappa2_s1_results_20261114/REPORT.md
[A_TRSIG]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7a0c9468b2499fac9fa1cd279ae2980c9dab048e/docs/research/truncated_fullh_signal_results_20261114/REPORT.md
[BASEOPT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/331a6ffebe551577eb9574aca354e76175697743/docs/research/rpe_baseline_optimization_design_20261113/REPORT.md
[B_DP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/99930491819cabd94609c42005a8637ac06a787d/docs/research/h4_dp_risk_fixed_results_20261114/REPORT.md
[B_N]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/99930491819cabd94609c42005a8637ac06a787d/docs/research/rpe_required_n_cross_method_preparation_20261114/REPORT.md
[B_NPLAN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/99930491819cabd94609c42005a8637ac06a787d/docs/research/rpe_required_n_cross_method_preparation_20261114/FINITE_N_PLAN.md
[B_Q3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/caf1417ef20d0a5ab2030a2d957ee4f09443be43/docs/research/pauli_total_q3_accelerated_results_20261114/REPORT.md
[B_READY]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/99930491819cabd94609c42005a8637ac06a787d/docs/research/rpe_required_n_cross_method_preparation_20261114/READINESS.json
[B_SIGAPP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/99930491819cabd94609c42005a8637ac06a787d/docs/research/rpe_required_n_cross_method_preparation_20261114/SIGNAL_APPLICABILITY.md
[B_TOTAL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/99930491819cabd94609c42005a8637ac06a787d/docs/research/b_total_q3_canonical_integration_20261114/REPORT.md
[B_W0]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/104dacdc64eb01c7f1b5207a32c4a4d1c6a83a37/docs/research/rpe_accuracy_contract_w0_20261114/REPORT.md
[B_Z1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f3c84aa854583e78a88b9cbf260cce709325d342/docs/research/rpe_total_decoder_q3_risk_20261114/REPORT.md
[CALPHA]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e03824a2a76f0067844043d25a23ec8dd79cb91e/docs/research/track_a_calpha_h4_step1_20261110/SAMPLING_LAW_RPE_RESULT.md
[C_ADM]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/855573064b2c17cfe20eb94a01ece1a554141b0b/docs/research/truncated_p7_admission_pr_df_evidence_20261114/REPORT.md
[C_BRIDGE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c628e9e8447c11c984064b922cbce6c52dc72d48/docs/research/paper_g_price_routes_20261114/PAULI_BRIDGE.md
[C_CELLPLAN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/855573064b2c17cfe20eb94a01ece1a554141b0b/docs/research/truncated_p7_admission_pr_df_evidence_20261114/CELL_PLAN_V2.json
[C_ES]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/855573064b2c17cfe20eb94a01ece1a554141b0b/docs/research/h4_structural_schedule_results_20261113/REPORT.md
[C_FULLADM]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/855573064b2c17cfe20eb94a01ece1a554141b0b/docs/research/m5_m4_receipt_admission_20261114/REPORT.md
[C_O3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c5a3c102199fe97f4d5c21837c059ae2684f5ef2/docs/research/track_c_closed_p3_joint_opt_20261108/REPORT.md
[C_O3INT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b3c81c2677988c7b9491eeee31fbcc0f9b29b078/docs/research/track_c_o3_integrated_20261108/TRACK_C_O3_INTEGRATED_REPORT.md
[C_PRDFCOV]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/855573064b2c17cfe20eb94a01ece1a554141b0b/docs/research/truncated_p7_admission_pr_df_evidence_20261114/REPORT.md
[C_PRDFCSV]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/855573064b2c17cfe20eb94a01ece1a554141b0b/docs/research/truncated_p7_admission_pr_df_evidence_20261114/PR_DF_COVERAGE.csv
[C_PRDFPLAN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/855573064b2c17cfe20eb94a01ece1a554141b0b/docs/research/truncated_p7_admission_pr_df_evidence_20261114/PR_DF_VALIDATION_PLAN.md
[C_PVPLAN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c628e9e8447c11c984064b922cbce6c52dc72d48/docs/research/paper_g_price_routes_20261114/PRICE_VALIDATION_PLAN.md
[C_ROUTES]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c628e9e8447c11c984064b922cbce6c52dc72d48/docs/research/paper_g_price_routes_20261114/REPORT.md
[D8_R16]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d68cf3c19685da586ceadbbac8193f55b9ba0fa0/docs/research/track_d_r16_price_calibration_20261110/REPORT.md
[D8_TOTAL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8828b298975c88a2c0d1fd57ff6ed905233840f5/docs/research/track_d_bc_total_20261108/REPORT.md
[D_NEXT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/acea1118a4df4e727e13398ce0f8ae11511c7396/docs/research/d_pf_price_saved4_revision_20261114/NEXT_EXECUTION.md
[D_P2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/26d4f23e79e91385087669d4ee091632efa0c87c/docs/research/d_p2_direct_pilot_20261113/REPORT.md
[D_SAVED4]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/acea1118a4df4e727e13398ce0f8ae11511c7396/docs/research/d_pf_price_saved4_revision_20261114/ADMISSION_UPDATE.md
[FH_H8]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/605d78e8501411592b678f53e2a12824fa40d2db/artifacts/resource_applicability/df_h8_full_h_pure_rte_light/2026-11-10/parallel_v1/REPORT.md
[FH_T2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7a8bc7f132e166390c943e6b32c92cc6abe1baf4/artifacts/resource_applicability/df_full_h_pure_rte_t20_opt/2026-11-09/REPORT.md
[H46_COV]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4cb61ef4633467446122954ed6cc43462e0306a9/docs/research/h6_provider_integration_nc_coverage_20261111/REPORT.md
[H4_NBATCH]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/89c91c692e1d35de94d2e7afbe890aabe6d80b7e/docs/research/h4_df10_rpe_batch_results_20261110/REPORT.md
[H6_M7]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/14c4f871fb9e8d4a23a6cc3244f5435ebc65a7ce/docs/research/h6_df19_rpe_m7_results_20261110/REPORT.md
[INTEG]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/20b3c6b10f7195ea847afae31d2989076a2aaffc/docs/research/pr_rpe_joint_integration_20261110/REPORT.md
[INTEGM]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/20b3c6b10f7195ea847afae31d2989076a2aaffc/docs/research/pr_rpe_joint_integration_20261110/INTEGRATION_MATRIX.md
[MENU4]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/51500b724a75f9535a2c831d50bcfe09eabb9326/docs/research/h4_four_method_minimal_menu_20261113/REPORT.md
[NEWA_Q3]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/08f9cbe77456ec29d50a992d7165c4d726df7bf7/docs/research/pr_reference_feedback_q3cd_20261110/REPORT.md
[NEWA_Q4]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d5bc47f762cc2dcbf0839fc0c8c6445356b43293/docs/research/pr_reference_feedback_q4_1_20261110/PREPARATION_REPORT.md
[O2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d28164b5d51e0ca5a0ade74ded8ff176b443dab8/docs/research/o2_research_results_integration_20261108/REPORT.md
[RC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/776e0a16ac98762631fe9307672725fdfee3b73d/docs/research/research_c_three_method_z_relocated_comparison_20261108/REPORT.md
[T1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7a0c9468b2499fac9fa1cd279ae2980c9dab048e/docs/research/truncated_pf_t1_results_20261114/REPORT.md
[T2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7a0c9468b2499fac9fa1cd279ae2980c9dab048e/docs/research/truncated_pf_t2_results_20261114/REPORT.md
[THRESH]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ecafc5deaba250ad170a4324cf1be48b14e56a9d/docs/research/track_a_threshold_centering_square_avoidance_20261110/REPORT.md
[B_FULLH_3O2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6c5569c3479013b080b2ba53842585f1d670d3c9/docs/research/full_h_kappa2_scale3o2_q3_results_20261115/REPORT.md
[B_FULLH_3O2_HANDOFF]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6c5569c3479013b080b2ba53842585f1d670d3c9/docs/research/full_h_kappa2_scale3o2_q3_results_20261115/HANDOFF.md
[B_FULLH_3O2_JOIN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6c5569c3479013b080b2ba53842585f1d670d3c9/docs/research/full_h_kappa2_scale3o2_q3_results_20261115/C_RESULT_JOIN.json
[C_PF60_REC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c3f761f0af5b21207e64051c9c619858064450fe/docs/research/pf_closure_price_recommendation_20261115/HANDOFF.md
[C_PF60]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c383105b16db7004db39b1ae270b220627d0f3b4/docs/research/pf_closure_prices_20261115/HANDOFF_B.md
[C_PF60_TABLE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e4458cd7548977e6e11a5a7b0c8dca0d049535d4/docs/research/pf_closure_prices_20261115/run01/PRICES.json
[B2_PF_TOTAL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1002f1dfea34de439ca1b704a0192bca473c5387/docs/research/pf6_allround_cost_20261115/HANDOFF.md
[B_PF_PARTIAL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/540b54ab2aa01a164f913d5b18497e711ae4a29c/docs/research/pf6_eight_round_cost_20261115/REPORT.md
[D2_PRDF_EC_PLAN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/950577f416c17a20d309a8053a4bb016027b724d/docs/research/d_pr_df_ld1_ec_plan_20261115/REPORT.md
[C_PRDF_F0]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/259a0b2d7357d740502d6837db2cf3c2a8b75c5e/docs/research/pr_df_ld1_f0_admission_20261115/HANDOFF.md
[D2_PRDF_RANGE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dcf905fc5c0abb05966019bcae8b0206af9aa952/docs/research/d_pr_df_ld1_f0_rz_range_20261115/HANDOFF_C.md
[PAULI_M9_PLAN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2d513c2755cfd88cfcf30cba4352a6075d9a9c4b/docs/research/pauli_ld2_next_price_evidence_20261115/HANDOFF.md
[PAULI_M9_PREP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3f6e6604a4877dc882fd1ee7538a854464160a5b/docs/research/d_pauli_ld2_m9_longpoint_results_20261115/SOURCE_PREPARATION.md
[D1_M5_STATUS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c511f6b4f521f0e1dd7853bea02e0b0058272640/docs/research/d_full_h_d7_holdout_20261115/run01/M5_ROUND/AXIS_STATUS.json
[D1_M5_INPUTS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/c511f6b4f521f0e1dd7853bea02e0b0058272640/docs/research/d_full_h_d7_holdout_20261115/run01/M5_ROUND/C_INPUTS.json
[D1_M6_STATUS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b5fac0ee4eac83f027f883e5ea5630623a98bf29/docs/research/d_full_h_d7_holdout_20261115/run01/M6_ROUND/AXIS_STATUS.json
