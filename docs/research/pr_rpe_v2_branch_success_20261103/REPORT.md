# RPE V2：最終18roundの分枝復元・成功率・energy精度

2026-11-03 JST / LOCAL / 主エージェントのみ。preflight後の明示実行許可に基づき、保存値だけの固定V2解析を完了した。

**Partial／Fullとも、保存平均の18round nearest-branch復元は一意で、別oracle診断と全roundで一致した。平均信号復元のenergy誤差は約5.34e-6 Haで保存暫定目標内。保存平均を真の測定平均とするIIDモデルでは、共通E₀への全round位相条件の失敗率union boundはPartial約2.82e-6、Full約1.68e-7となり、同じ保存成功率条件を支持する。総合判定は両候補とも`CONDITIONAL_SUCCESS`。**

V1の`CONDITIONAL`、Full半径の7点の不一致、state identity・単位半径下界の未保証は維持する。現実の量子実験・状態準備・backend/noiseまでの厳密保証、全18round decoderの厳密joint失敗確率、最終費用優位は認定していない。sampling、RNG replay、物理信号生成、compile、費用再会計は0件。

## 成果物・固定条件

- [V2_ROUND_RESULTS.csv](V2_ROUND_RESULTS.csv)：Partial／Full各18行。元H/state参照、q/r/K、保存物理信号・N/α/β/epsilon、確率、半径の区別、平均信号branch、V2判定。
- [BRANCH_RESULTS.json](BRANCH_RESULTS.json)：実際の保存平均復元と別oracle診断、初期window、各round選択・曖昧性、最終位相・energy。
- [SUCCESS_PROBABILITY_RESULTS.json](SUCCESS_PROBABILITY_RESULTS.json)：軸別・round別の二項確率、Hoeffding、union bound、条件・数値精度・未保証部分。
- [VALIDATION.json](VALIDATION.json)：固定入力/SOURCE、限定再利用した関数、人工テスト、実行範囲、資源・保全情報。
- [run_saved_v2.py](run_saved_v2.py)、[人工テスト](test_saved_v2.py)：元関数を変更せず限定再利用するadapter。

対象はlinear H4、隣接1.0 Angstrom、STO-3G、legacy DF rank12、8qubit、保存4電子sector、δ=.02、m0..17、q1..131072。Partial L_D3は最終coherent opt2の選択済みr/K、Full L_D12はtail-free r0/K0。N・α・β・coordinate epsilonは元opt2 JSONの保存値を固定した。新schedule・候補・配分は作っていない。

V1は固定`e520e6e396f7e0def9e522ea1cff58eb10ddfa8f`、元signal・opt2・SOURCE carrierは固定`4591b8d75a2a6a55c74a51225f0c61648023c8ee`。保存暫定energy目標はWP01の`.00015936001019904` Ha、beta_RPE=.4、alpha_total=.05を引き継ぐ。CA/10を新規に正式採択していない。[V1][V1REPORT]、[opt2][OPT2JSON]、[暫定目標][WP01]

## 1. 入力と元モデルからの変更範囲

共通E₀は元DF Hamiltonianの4電子sector基底エネルギー、保存値`-2.1663874486347616` Ha。H hashは`56e4df83655aa2f2f8132126f2635996516dc2cfb1bf4cb8d95490184ad631e5`、snapshot参照SHA-256は`13e4b10d2347ed900fe4aa4b2238128ebe735a48988e321b9e4fabac4d092a3b`。V1入力hash・36tuple・元signalとの一致を確認した。H/state/referenceを再生成していない。

測定確率に使ったのは、Partialの`physical_df_ground_state/attenuated_event_mean_signal`、Fullの`signal_diagnostics/12/points/*/signal`である。Partialのnormalization補正済み信号やPF有効固有状態controlを主測定平均へ混ぜていない。[Partial][ABLGRID]、[Full][ABLJSON]

元資源モデルはPF有効固有状態・単位基準半径を仮定する。今回の評価は、その保存N/α/β/epsilonを固定し、**元H物理基底状態に対する保存測定平均**で確率を評価した条件付きモデルである。観測半径に合わせてNやepsilonを作り直していない。

Full診断自身のstate fingerprintとδ=.02のPF有効固有状態controlは未記録／未発見のまま。元SOURCE/provenanceによる対応を使い、別資料のfingerprintをFull診断自身の保存欄として補わない。元入力の条件付き性はV1から引き継ぐ。

## 2. 保存平均による18round分枝復元

RPE4の`reconstruct_rpe_phase`を変更せず限定再利用した。各roundで`arg(z_m)`からq個の候補を作り、前roundの推定位相に円周上で最も近い候補を選ぶ。**この関数にはE₀を渡していない。** E₀は復元後の評価と、独立した`_oracle_branch`診断だけに用いた。[元SOURCE][RPE4SRC]

両候補の保存平均では全18roundの選択が一意、曖昧性なし、別oracleと全round一致。最終q131072の選択branchはともに904。全候補列挙はこの少数の平均信号復元にだけ使い、試行数×qの巨大arrayや新しい高速分枝アルゴリズムは導入していない。

| 保存平均復元の結果 | Partial L_D3 | Full L_D12 |
|---|---:|---:|
| 最終base phase | .04332764224934449 | .043327642039047376 |
| 共通E₀ referenceとの円周位相誤差 | 1.0672335060135651e-7 | 1.0693364771441338e-7 |
| 最終位相許容幅 beta_RPE/q_max | 3.0517578125e-6 | 3.0517578125e-6 |
| 平均信号からのE_hat（Ha） | -2.1663821124672245 | -2.166382101952369 |
| 保存E₀との絶対誤差（Ha） | 5.336167537173253e-6 | 5.3466823928260965e-6 |
| 平均信号復元の判定 | NUMERICALLY_VALIDATED | NUMERICALLY_VALIDATED |

これは保存された平均値の決定論的復元であり、新しい量子測定結果ではない。N有限時の評価は後述の解析確率と分けた。

### 符号・初期branch・alias条件

元SOURCEは`U(t)=exp(-itH)`、`z=<psi|U|psi>`であり、復元base phaseは`phi=-E0*delta`に対応する。energy変換は`E_hat=-phi_hat/delta`。scalar/identity位相は元signalに含まれており、二重補正していない。

初期m0は元SOURCEどおりbranch0、phase windowは`[-pi,pi)`。対応するenergy windowは`(-pi/delta,pi/delta]`、今回約`(-157.07963267948966,157.07963267948966]` Haである。保存referenceはこのwindowの内部にあるが、**対象energyがこのalias windowに入ることは外部条件**として明記する。別aliasを物理的に除外した新しい証拠ではない。E₀を使ってdecoderのbranch offsetを補正していない。

保存目標へのnominal resolutionは`beta_RPE/(q_max*delta)=.000152587890625` Haで、保存暫定目標`.00015936001019904` Ha内。正しいalias・共通phase・branch条件なしに、この式だけを最終energy保証とはしない。[方法][METHOD]、[ACCOUNT][ACCOUNTSRC]

## 3. 固定N・α・βによる確率評価

各軸は元SOURCEと同じ±1測定モデル、`p_plus=(1+mu)/2`。保存NはPartialの各軸261〜510、Full144〜476をそのまま使用した。各α・coordinate epsilon・統計phase予算も固定。全72軸で確率clippingは不要だった。

RPE4で使われた以下の関数を変更せず再利用した。

- `_axis_distribution`：保存muとNから二項PMF全点を評価。座標失敗事象は`abs(sample_mean-mu)>=epsilon`。
- `_round_exact_phase_probability`：cos/sinの二項product格子から位相失敗確率を評価。zero推定信号または`phase_error>beta`を失敗とする。

[元failure SOURCE][FAILURESRC]

二種類のround評価を分けた。

1. **統計位相誤差**：物理保存平均のphaseから、保存beta_statを超える確率。
2. **共通E₀への全位相誤差**：同じ確率格子を用い、phase referenceを共通E₀のround位相、許容幅をbeta_RPE=.4とした確率。PF/RTE系統差を含む評価である。

後者でE₀を使うのは確率事象の評価であり、decoderのbranch選択には渡していない。

### 軸・round・全roundの結果

以下は固定Bernoulliモデルの解析確率をfloat64で評価した値。union boundは全roundの失敗率上界として使うが、数値丸めまで外向きに保証する区間計算ではない。

| 評価 | Partial | Full |
|---|---:|---:|
| 各候補36軸の座標失敗率union bound | .001877290236513795 | .0007411817333623419 |
| 固定epsilon/NによるHoeffding座標union bound | .0496919287376424 | .049110879645390515 |
| 各roundの統計phase失敗率union bound | 3.742642147368975e-6 | 3.632233215714606e-7 |
| 共通E₀への各round全phase失敗率union bound | 2.8153520984407623e-6 | 1.679843848066986e-7 |
| 保存αの合計（数値評価） | .05 | .05 |

全軸で二項座標失敗率はその軸の保存α内、全roundで統計phase・全phase失敗率は保存round α_c+α_s内だった。単独のαを実際のround失敗率として扱っていない。

固定Nの離散格子では、全36roundで「両座標successなら保存beta_stat以内」の含意違反は0件。共通E₀のbeta_RPE以内という含意も違反0件だった。これは指定N・mu・epsilonの格子上の確認であり、一般の単位半径下界の証明ではない。

### 全energy成功率とbranch失敗率の位置付け

既存のstrict branch条件`beta_RPE<pi/3`と、全roundで共通E₀へのphase条件を満たす事象を用いた。union bound自体にはround間独立を追加要求しない。初期alias windowと局所energy変換の条件の下で、当該十分事象から評価するenergy成功率下界の数値値は、Partial`.9999971846479015`、Full`.9999998320156152`。ともに保存α_total=.05に対応する成功条件を支持する。

同じ十分事象のunion boundから得るbranch失敗率上界の数値値も、それぞれ約2.82e-6、1.68e-7である。**全18roundのdecoderの厳密joint branch失敗確率を求めた値ではない。** 平均信号のoracle一致、round位相失敗率、branch十分事象による上界、energy成功の条件付き評価を別欄に保持する。[既存strict条件][ACCOUNTSRC]

元ACCOUNTの`conditional_energy_error_bound`などは`certified`時だけ値を返す。その制御は変更せず、今回の値はV2専用の条件付き数値モデル欄へ記録した。元scientific statusをcertifiedへ変更していない。

## 4. 成功率に必要な条件とV1の半径問題

今回の数値確率には次の条件がある。

- 保存complex値が、記録された物理入力に対する正しい真のmarginal測定平均を表すこと。
- 各軸内の独立有界shot、およびcosine/sine間の独立した測定実験という二項productモデル。
- Partialで仮想的な各量子shotごとにfresh-IID RTE trajectoryを用いること。今回この実装監査やtrajectory生成はしていない。
- 外部alias-free energy window、同じ共通E₀と符号、既存nearest-branch十分条件が適用されること。
- 元H物理基底状態を利用できるという入力条件。Full診断のstate fingerprint未記録等のprovenance上の制約も残る。

V1のFull半径差7点（m6,7,8,9,10,11,14）はそのまま残した。V2では実際の保存muから固定Nの分布を評価したため、これを理由にNやepsilonを変更していない。固定離散格子の含意は成立したが、near-one半径を厳密な下界1と認定する理由にはならない。

PF有効固有状態controlと元H物理基底状態も同一と認定しない。Fullの同δ controlを新たに生成せず、元のPF係数の経験的性質、状態準備の仮定、厳密単位半径未保証を保持する。従ってV1は`CONDITIONAL`のまま、V2はこの測定モデルに対する`CONDITIONAL_SUCCESS`である。

## 5. 数値精度・実装再利用・人工検証

NumPy1.26.4、SciPy1.14.1、Python3.11.0rc1の既存環境をread-onlyで使用した。元SOURCEのFunctionDefを限定抽出し、そのASTを変更せず実行した。元module全体のimport・runner呼出しはしていない。選択関数名・元line・text hashをVALIDATIONに記録する。

人工テスト10件PASS。元scalarとvectorizedの小q一致、wrap両側、tie/ambiguous、oracle分離、既知小NのPMF、独立した小格子和、zero信号、元のmean範囲・丸め、tuple/stateの取り違えを確認した。少数の保存点と2round prefixのsanity後、全36roundへ適用した。

元axis処理のPMF直接和を主評価とし、同じ離散失敗maskをCDF/SFでも照合した。最大差はPartial`2.8189256484623115e-18`、Full`1.4094628242311558e-18`。PMF総和の1からの最大差は、それぞれ`7.882583474838611e-15`、`1.9984014443252818e-15`だった。

これらは浮動小数点の整合性診断で、厳密な誤差区間ではない。underflowやPMFの0を数学的な確率0の証明として扱わない。保存mean自体の数値誤差を区間として伝播した評価でもない。成功率の小数全桁を物理的な確実性として解釈しない。

元のsignal tolerance1e-10、branch ambiguity1e-12、mean範囲1e-12、budget比較参照1e-15は用途を分けて保全した。閾値を変えてPASSにする処理はない。Clopper–Pearson区間、Monte Carlo推定、RNG seed／反復数は新設していない。

確率格子は(Nc+1)×(Ns+1)でvector化し、独立roundを32 workerで評価。branchのq候補列挙は平均信号の少数復元だけ。観測wall timeは約2.65秒、peak RSSは244972 KiB。人工CPU/RAM/worker上限、共有設定変更、他ユーザーjob操作は行っていない。

## 6. 旧検証との違い・V3の意味

旧RPE4はδ=.1/r4/K2/q1,2,4,8で、物理signal、厳密二項、fresh-IID1572 trajectory、10万回の周辺分布branch検証を行った。今回その新samplingは再実行せず、最終δ=.02・18roundの保存平均とN/α/βへ解析的処理だけを適用した。

旧WP04には長round信号と旧配分の統計診断があった。今回の評価は最終coherent opt2の別N/α/βを固定している。WP04全runnerにはshot探索・費用counterfactualが含まれるため実行せず、V2に必要な信号対応・確率定義だけを再利用した。元結果・元判定は変更していない。

同じ暫定energy目標・成功率条件を満たすという**条件付き固定モデル比較**には科学的な意味があるため、ユーザーがこの条件付きscopeを選ぶ場合にV3の価格移送・不確かさ検討を提案できる。ただし、今回の成功でq>32価格移送・状態準備・compiler/backend・総費用優位が確定するものではない。既存約4.858%の費用点差と区間重複を維持し、新しいN×C会計や費用順位付けをしていない。

現時点で追加samplingはこのV2の条件付き判断に必須ではない。全decoderのjoint失敗率そのもの、実fresh-IID実装、物理mean/stateの独立確認を別の主張として要求する場合に、何を判別するかを限定して提案する段階である。V2結果からその実行を自動開始しない。

次担当はユーザー。入力state／半径・alias・IIDという主張条件と、V3へ進む必要性を判断する。追加科学計算・sampling・compile・詳細科学レビュー・中心claim採択は別承認の対象で、今回実行していない。

## 原本保全・公開・Mandatory STOP

AGENTS.md、PROJECT_MAP.md、正式共通運用指示2026-10-25の§8.1・§8.4・§10をpreflightで実読した。旧V1と他研究のgrant/ledger/marker、one-shot/retry、mandatory STOPを保全した。承認証明書や管理基盤は新設していない。

固定27 objectのcommit/path/blobと取得byteを記録し、V1の元input hash・物理tuple・signalを照合した。公開carrierと元dirty execution SOURCE/provenanceを分け、旧SOURCE/rawからのclean再生成や完全な推移的依存監査とはしていない。元コード・資料・raw/QPYを変更していない。

出力は独立Git領域の新branch `codex/pr-rpe-v2-branch-success-20261103` に置き、親はV1固定commit e520e6e396f7e0def9e522ea1cff58eb10ddfa8f。原repoでcheckout/reset/clean/gc/prune/fetch・削除・cleanupは行っていない。内部host/IP・絶対パス・秘密情報・private rawを公開しない。

再現は固定objectと必要な実行許可のある環境で、既存NumPy/SciPyを利用して行う。自動fetchや環境installはしない。

```bash
<research-python> docs/research/pr_rpe_v2_branch_success_20261103/run_saved_v2.py --evidence-repo '<repository-containing-fixed-objects>'
```

原本状態の観測では、HEADと開始時から存在するstatus項目は維持された一方、本作業外の整理・研究準備と整合する新規未追跡ファイル2件と研究refs更新を検出した。V2が使用する27固定objectは引き続きbyte一致。全worktree・全refs不変とは報告せず、観測した旧／新OID等は私有作業記録に保持し、他作業の停止・巻き戻しは行っていない。

公開前の機密情報・成果物整合性・原本状態確認後にcommit/pushし、remote SHAと公開各file本文を照合する。最終SHAと全文一致の結果は最終チャットで報告する。

**Mandatory STOP：公開・照合後、V2を終了する。** V3、shot/α/β/schedule最適化、新H/state/operator/物理signal、sampling/RNG replay、compile、N×C再会計、詳細科学レビュー、最終資源優位の採択へ進まない。

[V1REPORT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e520e6e396f7e0def9e522ea1cff58eb10ddfa8f/docs/research/pr_rpe_v1_e0_state_radius_20261103/REPORT.md
[V1ROWS]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e520e6e396f7e0def9e522ea1cff58eb10ddfa8f/docs/research/pr_rpe_v1_e0_state_radius_20261103/V1_ROUND_RESULTS.csv
[V1VALID]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e520e6e396f7e0def9e522ea1cff58eb10ddfa8f/docs/research/pr_rpe_v1_e0_state_radius_20261103/VALIDATION.json
[V1HELPER]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e520e6e396f7e0def9e522ea1cff58eb10ddfa8f/docs/research/pr_rpe_v1_e0_state_radius_20261103/reanalyze_saved_signals.py
[V1TEST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e520e6e396f7e0def9e522ea1cff58eb10ddfa8f/docs/research/pr_rpe_v1_e0_state_radius_20261103/test_saved_signal_reanalysis.py
[ABLGRID]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/research_direction_ablation/2026-09-21/wp04_ld3_full_schedule_signal_grid_v1.json
[ABLJSON]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/research_direction_ablation/2026-09-21/wp04_finite_rte_statistical_ablation_v1.json
[OPT2JSON]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/research_direction_full_opt2/2026-09-25/wp11_all_r_opt2_coherent_analysis_20260925_065827.json
[WP01]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/research_direction_prevalidation/2026-09-21/wp01s_model_conditional_screening_v1.json
[RPE4JSON]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/rpe_four_round_phase_validation/2026-09-20/h4_sto3g_d100_rank12_ld3_dt0p1_r4_k2_q1_q2_q4_q8_physical_branch_v1.json
[HORIZONJSON]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/rpe_target_round_horizon_validation/2026-09-20/h4_sto3g_d100_rank12_ld3_dt0p1_r4_k2_ca_over_10_horizon_v1.json
[METHOD]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/docs/research/研究方法・解析手順.md
[PLAN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/9a0bc7f17f47823e46c3945734127e9f0d09154c/docs/research/数値実験・評価計画.md
[RPE4SRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/rpe_four_round_phase_validation.py
[RPE4TEST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/tests/test_rpe_four_round_phase_validation.py
[FAILURESRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/rpe_hadamard_failure_validation.py
[FAILURETEST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/tests/test_rpe_hadamard_failure_validation.py
[ACCOUNTSRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/rpe_resource_accounting.py
[ACCOUNTTEST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/tests/test_rpe_resource_accounting.py
[ABLATIONSRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/research_direction_ablation.py
[ABLATIONTEST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/tests/test_research_direction_ablation.py
[FINITESRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/finite_rte_signal_validation.py
[FINITETEST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/tests/test_finite_rte_signal_validation.py
[OPT2EXTSRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/research_direction_full_opt2_extension_analysis.py
[OPT2EXTTEST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/tests/test_research_direction_full_opt2_extension_analysis.py
[HORIZONSRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/rpe_target_round_horizon_validation.py
[HORIZONTEST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/tests/test_rpe_target_round_horizon_validation.py
