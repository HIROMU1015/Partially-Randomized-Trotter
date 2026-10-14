# H4固定一体項再配分PF：許容h境界・強対照比較の一回限定検証準備（2026-10-15）

完了済みGPT科学レビューの「限定継続、許容h境界へ一度まとめて進む」判断に従い、新scope/source/有限探索/独立checkerを準備した。実入力の新しいeig、RDM、expectation、PF評価、旧数値の新規集計、compileは実施していない。35人工テストPASSとidentity-only preflightを、分子性能結果と区別する。

## 正本・不変の対象

受領レビューは[原文](../reviews/quantum_chemistry_pf/qc_onebody_repartition_finite_step_pilot_v1_scientific_review_2026-10-15.md)、43818bytes、SHA256 `aade00b1cfbdb2845a4414daeb8632576ef6178a931432276ce50eb7fd4b2486`。別のGPT科学レビューを開始していない。v1結果正本 `31ea58449eb8e535b8f5d5f62956fb410d35f0ae`、旧source19d5bd74と36セル・NO_UNIFORM_REGISTERED_COMPONENT_GAIN・消費済み認可・STOPを保持した。

本source commit `5e3341c12bd84f30f19072cefa590f4499c36c38`、[freeze42files](../../artifacts/qc_onebody_repartition_h_boundary_preparation_v1/2026-10-15/source_manifest.json)、manifest SHA256 `35cb38263997eb26f636959799691660c64c2512f903673521f0e6b614d10b7b`。[契約](../../artifacts/qc_onebody_repartition_h_boundary_preparation_v1/2026-10-15/contract.json) SHA256 `f1417c0eac36ed2dae935fc3b5aff74f3bfa3adc77b10e74006161250de2fe05`。[入力identity](../../artifacts/qc_onebody_repartition_h_boundary_preparation_v1/2026-10-15/input_identity.json)と旧9重要filesのbyte/SHAを固定している。

linear H4 / 1.00Å / STO-3G / 保存DF rank12・generation順0..11、8spin / 4spatial、Nα=Nβ=2、sector36dim。全12densityをdeterministic PFに用い、PR L_D/delta探索を変更しない。H = constant I + dΓ(one_body) + Σλ_l dΓ(g_l)²、追加1/2やnormal-order重複補正なし。入力pack SHA256 `3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a`、Hamiltonian hash `de7a549238e3a21f15a84018bef28440c345b31030282c01cf874f3d1d212424`。

v1 [selector seal](../../artifacts/qc_onebody_repartition_finite_step_pilot_v1/2026-10-15/attempt_1/selector_seal.json) SHA256 `98f6ff69923ceca02448f750cf387bfe443a8795cf66b45034a481e32e0bc008`、raw NPZ `815e31a4c305a7bf13b380c53f1e99808920aaaf72d03836ee5af76dd058c81c`。response/known shift、leading、one-body/density frame、cached exact stateとtruthを継承する。新selector、方向、更新fragment、係数、分子、rankは追加しない。厳密E0をcandidate生成へ与えず、現在は生成自体0。旧DF由来の追加物理認証も行わない。

## 全手法に共通のh探索

共通**比較h**はh_anchor=`0.002745119306432635`から**1 Ha⁻¹**まで。4%増のcoarse点と、安全な位相追跡のためのguard bridge点を含める。全12method（3config×S2/S4/CS2/EXT2）に同じ比較hを与える。

Richardsonには各比較hのh/2が必要で、これとv1 h1 anchorを**補助h**として追加する。補助点は全6 primitive streamの追跡にも評価するが、PF単独の追加選択機会にしない。比較点と補助点を区別することで、外挿だけh/2を欠いたり、原S2だけ追加hを使ったりする不公平を避ける。Richardsonは同一構成の正確な(h,h/2)、weights(−1/3,4/3)のみで、新node/外挿次数を増やさない。

一次scheduleは**比較368点、補助込み736点**。係数boundによる**S4のsigned-stage絶対時間和 β=4p+|1−4p|、p=1/(4−4^(1/3))、K=βΛ_common**を全streamへ共通に用いる。隣接h間隔はπ/(8K)以下。Λ_commonは旧保存値45.53536150763562で新数値fitなし。[全静的schedule](../../artifacts/qc_onebody_repartition_h_boundary_preparation_v1/2026-10-15/static_schedule.json)。補助込みh範囲は旧h1=`0.0013725596532163176`から1。域の上端でのwinnerをglobal最適と呼ばない。

各指数のunitary微分norm triangleから||U'(h)||≤K。孤立branchなら位相変化のboundにも使える。ただし有限個のprojector照合は区間内部での連続branch存在の厳密証明ではない。その限界を保ち、枝の曖昧性や数値上限を未判定として保存する。

## 結果前に固定した細分化

全method・両targetのeligible/ineligible変化またはbranch-valid変化を収集する。相対幅1%を超える区間を、相対幅の大きい順、左hの小さい順で一意に並べ、各round最大32区間の幾何中点と必要なh/2を追加する。**2roundのみ**。全methodのunionを同じ比較点へ反映し、特定methodのoracleだけを優遇しない。

最大追加比較64点・補助込み128点、**比較432点 / 全864点**。最初の不適格点でより大きいhを切り捨てず、coarse domain全体を評価する。単調性は仮定しない。狭い未観測islandまで除外したcontinuum最適性は主張しない。細分化枠不足・残るtransition・数値/branch・上端は理由付き未判定、域延長・precision rescue・再実行なし。

## phase wrap・ground branch・数値正確性

旧h1におけるhash-bound eigenpairをanchorとし、各config/orderを別streamとしてhを追跡する。phaseは円上でcluster化するため±πを跨ぐclusterを分断しない。exact入力stateの最大overlap clusterを使い、同点差<=1e-6、rank変化、スペクトルdefect>1e-9を曖昧とする。

integer aliasは**前のunwrapped phaseに最も近い値**で決める。E0に最も近いenergyや最小biasからaliasを選ばない。位相増分KΔh+1e-8、projector transition>=.95、競合projectorとの差>1e-6を確認する。新細分化点は既評価の直近左と、有効な直近右のalias/projectorにも整合を求める。左branchが不明なstreamを大hで自動再開しない。その他のstreamは共通domainを維持する。

E0は旧truthのhash-bound再利用に限り、bias/eligibleの**性能oracle**にのみ使用する。運用可能なtruth-free h selectorとは呼ばない。候補Xは元のsealのまま。exact-state overlap .99以上の適格性・error budget・branch repeat方針はv1と同じ。

double precision safety allowanceはfloor1e-10 Haとspectral/eigenvector/unitarity residual、cluster幅から構成する。独立checkerはJW/Padé再構成と別の円上cluster・alias・費用計算で照合する。外向きintervalの厳密保証ではない。小hの数値床からc4をfitしない。CS2残差は直接finite-h energy−h²d、EXT2も直接二点から評価し、数値幅を伝播させる。宣言precisionで曖昧ならそのまま保存し、未登録高精度再計算を追加しない。

## 強い対照・10%判定・情報費用

primary候補はv1同様responseのS2/CS2。controlはその他全eligible：original raw/CS2、known raw/CS2、全3構成S4、全3構成EXT2。response S4/EXT2もcontrolに含まれる定義を維持し、secondary同family比較（16 target/price/family条件）を別記する。secondaryをprimary成功へ置換しない。

target10^-4 / 10^-5Ha、budgetdet/stat/synth/info=(.5,.25,.125,.125)、P1/P2=(4,1,2)/(8,1,4)、uniform4context・10%gate ratio<.9−1e-10を維持する。価格はcontrolled Givens/linear/pair primitiveの仮説で、compile/T/hardware測定ではない。全手法へfixed-N scalar、basis sharing、同fragment fusionを許す。

coherent queries=ceil(π/(h εstat))、repeat、Richardsonの二experiment/L1 precision費用をv1と同じ配分で算入する。別の最適precision配分は今回primaryへ追加しない。cached affine repeated-query grammarを全h/target/priceに適用し、人工fixtureでv1の直接QR/fusion費用と一致を照合した。

全4条件のdiscrete-grid判定を保持し、さらに両winning methodの最大eligibleと次の明瞭なbias-ineligible点の相対幅<=1%を要求した境界診断を別記する。上端・branch・overlap・数値曖昧・幅不足は `UNDETERMINED_BOUNDARY_COVERAGE`。10%thresholdを変更せず、観測点でのgainと境界の情報不足を混同しない。有限gridの局所bracketであり、global continuum費用の上下界ではない。

原rawは不要なγ/d0取得を強制しない。CS2のd精度要求η_d<=ε_info/h²を記録し、responseの追加情報に使えるcomponent差を別記する。γ/d取得の量子費用はUNKNOWN、無償とはしない。exact-state情報を理想的に継承する機構scopeを保持し、量子測定backend・全QPE/RPE confidence・total費用・state preparation・実T/compileは未評価。tiny synthesis allowanceは予算表示のみ。

## 固定実装・call・資源上限

[runner](../../scripts/run_qc_onebody_repartition_h_boundary_v1.py)、[protocol](../../src/trottertracks/ground_pf_repartition/qc_onebody_repartition_h_boundary_protocol_v1.py)、[pilot](../../src/trottertracks/ground_pf_repartition/qc_onebody_repartition_h_boundary_pilot_v1.py)、[独立checker](../../scripts/verify_qc_onebody_repartition_h_boundary_v1.py)、[人工テスト](../../tests/test_qc_onebody_repartition_h_boundary_v1.py)。旧sourceを改変しない。

| 項目 | 一回実行の上限 |
| --- | ---: |
| 新規PF seed / Schur | 各5,172 |
| cached anchor cells | 12（v1 h0/h1×3config×2order） |
| 独立PF再構成 | 5,184（cached anchor含む新domain全cell） |
| 独立Padé expm | 648,000（worst-word countからの保守的上限、実cache回数を記録） |
| method records / cost rows | 5,184 / 20,736 |
| saved raw科学array load | producer1、checker1 |
| 新H/spatial eig、RDM、leading、selector、optimizer、分子生成、ground準備、compile/T | 0 |
| worker / thread | 1 / 1 |
| peak RSS / output / log | 512MiB / 512MiB / 2MiB |
| wall/CPU cap | なし |
| producer/audit attempts、retry | 各1、0 |

36dim cached sectorとsealed actor framesを利用し、新しいdiagonalization・expectationは不要。spectral/Padé LRUは512entriesでbounded、eigenvectorsはcellごとにstream保存する。v1 timingは選択根拠でありspeedup保証ではない。[性能plan](../../artifacts/qc_onebody_repartition_h_boundary_preparation_v1/2026-10-15/performance_plan.json)、[残るI/O費用](../../artifacts/qc_onebody_repartition_h_boundary_preparation_v1/2026-10-15/performance_limitations.json)。CPU32、RAM/GPU状態は[環境](../../artifacts/qc_onebody_repartition_h_boundary_preparation_v1/2026-10-15/hardware.json)の確認範囲。GPU利用や多worker speedupを仮定しない。caller logは少量のpass/stop JSONのみで、future launchの実測log/RSS/outputを保存する。

## 準備監査・承認境界

35登録外fixtureテストPASS、source/header/SHA preflight PASS。phase wrap/tie/rank/left-right alias、非単調islandの共通refinement、未判定、引継ぎanchor再利用、independent malformed price/alias/numerical証拠拒否を検査した。旧27/37test suiteは再実行しない。新registered arrays deserialize/eig/RDM/expectation/PF/compile0。current CI/外部科学再現ではない。

[preflight](../../artifacts/qc_onebody_repartition_h_boundary_preparation_v1/2026-10-15/preflight.json)、[人工テスト記録](../../artifacts/qc_onebody_repartition_h_boundary_preparation_v1/2026-10-15/synthetic_tests.json)、[承認拒否テスト](../../artifacts/qc_onebody_repartition_h_boundary_preparation_v1/2026-10-15/execution_gate_tests.json)、[480files保護監査](../../artifacts/qc_onebody_repartition_h_boundary_preparation_v1/2026-10-15/preparation_audit.json)、[未発行approval target](../../artifacts/qc_onebody_repartition_h_boundary_preparation_v1/2026-10-15/authorization_target.json)。

新scope`QC_ONEBODY_REPARTITION_H_BOUNDARY_V1_20261015`の別明示one-shot認可だけを受け付け、clean isolated worktree・source/input/environment/capsを検証してからattempt directoryを先に消費する。古い36セル認可は拒否、未発行targetはfalse、今回は認可発行0・attempt消費0。数値checkerも別の一回audit markerを要求する。

480旧files/154旧manifest records/non-result fields/global description FAIL/root138pathsを保全する。N1STOP/N2HOLD/N3CLOSED/THRIFT現template CLOSED、C1–C4/Track A/B/第二候補を再開しない。原文と完了GPT判断を記録するが、独立新規性・正式採択・終了や新GPTレビューを代行しない。

GitHub公開・再取得一致後Mandatory STOP。次は[固定契約への別実行承認](qc_onebody_repartition_h_boundary_execution_handoff_v1_20261015.md)。通常source準備ができたという理由で同じ詳細GPTレビューを再要求しない。承認後の一回結果・独立監査・公開で再びSTOPし、重要な新科学的判断は別GPT開始承認へ戻す。
