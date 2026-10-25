# 研究C・C1：保存H6・修正codec round-trip限定実行

2026-10-25 JST。固定公開SOURCE `43941ed8fbb0f12f0b028b2c379b872f5ba50f75`。
Branch `codex/research-c-c1-h6-codec-execution-20261025`。

**20/20 logical unitsで正常完了。rank14／18ともJ・各frameのdensity projector・tensor・sector operatorの同値性と、DF予算CA/8の保持がPASSした。det=-1の列符号と、前段で不一致だったdet=+1のframe9を含む保存H6の初期値を許容差内で往復できた。新fit／optimizer／retry／resume0。修正後fitの成功や総資源優位は未評価。公開照合後mandatory STOP。**

[実行要約](../../artifacts/research_c_c1_h6_codec_roundtrip/2026-10-25/execution_summary.json)／[4状態比較](../../artifacts/research_c_c1_h6_codec_roundtrip/2026-10-25/state_comparison.csv)／[全gate](../../artifacts/research_c_c1_h6_codec_roundtrip/2026-10-25/gate_summary.csv)／[rank別frame32記録](../../artifacts/research_c_c1_h6_codec_roundtrip/2026-10-25/frame_diagnostics.csv)／[科学call](../../artifacts/research_c_c1_h6_codec_roundtrip/2026-10-25/calls_summary.csv)／[原本・J/U・生成子・parameters・STOP](../../artifacts/research_c_c1_h6_codec_roundtrip/2026-10-25/run01/)／[固定契約](research_c_c1_h6_codec_roundtrip_contract_unapproved_20261025.json)／[技術準備](research_c_c1_h6_codec_roundtrip_preparation_20261025.md)／[旧初期変換の診断](research_c_c1_h6_initialization_execution_20261025.md)。

## 許可範囲と来歴

ユーザーが直前の固定SOURCE・20-call限定評価をチャットで許可したため、独立worktree・fresh run01で1回実行した。契約のunapprovedという歴史的ファイル名・statusを改変せず、承認原文保存・承認証明書・新grantは追加していない。

直鎖H6・1.00 Å・STO-3G、中性singlet、Nα=Nβ=3、sector400。通常number表現、元g補正κ・scalar・元frame順、actual rank14／18を保持した。CA=0.00159 Ha、DF予算CA/8=0.00019875 Ha、既存`sector_operation_scale_guard8_proposal_v1`は変更していない。PFのL_D split・時間窓は今回非適用で、PF/RTE・回路・shots・費用は評価していない。

元H6入力／XDFのSOURCEは33c380d8…、保存初期値をXDF prefixと照合した前段SOURCEは14cf3558…、前段結果commitは53f45cacf…である。今回は保存済み`R14_before.npz`／`R18_before.npz`を読み、元`XDF_full.npz`の対応prefixとJ/U成分差0を再確認した。DFの再分解・新しい初期値生成はしていない。今回before snapshotのbytesも保存初期値と一致した。

## 物理sector誤差とDF予算

| rank | 変換前sector norm Ha | 変換後sector norm Ha | guard Ha | 変換後norm+guard Ha | CA/8 |
|---|---:|---:|---:|---:|---|
| 14 | 1.0247385485275444e-4 | 1.0247385484858992e-4 | 1.1384527781887395e-6 | 1.0361230762677866e-4 | PASS |
| 18 | 1.731868467770461e-9 | 1.7318670383582787e-9 | 1.366146019993636e-6 | 1.3678778870319943e-6 | PASS |

変換前も両rankがguard込みで予算内であり、変換後も保持された。beforeのsector normは前段の保存scalarと固定1e-10 Ha以内で再現した。rank18ではguardがraw normより大きいが、guardを除いた値へ判定を差し替えていない。これはDF予算の数値判定で、最終energyのCA達成や厳密rounding enclosureではない。

| rank | sector operatorの変換前後差 norm Ha | 同max-entry | 同Frobenius |
|---|---:|---:|---:|
| 14 | 1.241107775535217e-14 | 7.549516567451064e-15 | 9.177608023842878e-14 |
| 18 | 1.2411041218591138e-14 | 7.549516567451064e-15 | 9.177595208577705e-14 |

同値性gateには既存`component_check`のmax-entry1e-10／rtol0を使い、spectral norm差も別に報告した。max-entry toleranceをそのままoperator norm上界へ読み替えていない。κ・scalarを含む同じ元物理Hamiltonianに対し、既存`SectorCAR`／`rank_assessment`で各状態を独立に評価した。

## J・回転・projector・tensor

Jのround-trip成分差は両rankで0。float64 dtype、rank、frame順、ffsim parameter長（rank14は504、rank18は648）も保持された。負detの7／9 frameは最後の列の符号を反転したrepresentativeへ移るため、元Uとの直接差は合法的に残る。元Uの各column density projectorと、復元representativeの誤差は分けて確認した。

| 指標 | rank14 | rank18 |
|---|---:|---:|
| 最大representative U復元max-entry | 2.8865798640e-15 | 2.8865798640e-15 |
| 最大density projector max-entry | 3.3306690739e-15 | 3.3306690739e-15 |
| tensor round-trip max-entry | 9.4368957093e-16 | 9.4368957093e-16 |
| tensor round-trip Frobenius | 5.6291061954e-15 | 5.6291036645e-15 |
| tensor F許容差 | 2.7729823971e-10 | 2.7729899399e-10 |

元gに対するtensor残差Fはrank14でbefore7.729927307028549e-5→after7.729927307042256e-5、rank18で1.4210209588800006e-9→1.4210213706448963e-9だった。元gへの近似誤差と、変換前後のtensor同値性残差を混同しない。全量とmax-entryは4状態CSV／raw JSONへ保存した。

det=+1のframe9（0-based）は両rankに共通で、復元Uのmax-entry差2.7755575616e-15、projector差2.0816681712e-15、Schur block-exponential残差1.1102230246e-15だった。保存した回転平面の1つは角度3.1415926535897736で、π近傍を含む。前段で観測したこのframeの不一致は、今回の固定数値allowance内で解消した。

32は14+18のrank別frame記録数であり、prefixが重なる32個の別frameではない。今回の保存初期値での成功を、すべての直交入力や将来のfit経路への保証に一般化しない。

## 全gateと未評価事項

| gate | rank14 | rank18 |
|---|---|---|
| 保存初期J/UとXDF prefixの同一性 | PASS | PASS |
| J・rank・frame順・dtype／ffsim形式 | PASS | PASS |
| real Schur／反対称生成子／expm残差 | PASS | PASS |
| 各density projector・tensor同値性 | PASS | PASS |
| 保存before sector normの再現 | PASS | PASS |
| sector operator同値性 | PASS | PASS |
| before／afterのDF予算 | PASS／PASS | PASS／PASS |

修正codecは保存H6の指定初期値を保持できた。旧変換の不忠実性への対処として、この範囲の数値的証拠が得られた。一方、旧CDF optimizerの結果は書き換えておらず、修正codecを実際のffsim fitter初期化へ接続した新fitはしていない。`fit_success`、`CA_achieved`、`total_resource_superiority`はnullである。生成子の非一意性・branchを跨ぐ写像の連続性・厳密enclosureの資格も今回の結果だけで確立しない。

## call・資源・保全

| logical unit | 実績／最大 |
|---|---:|
| saved_codec_load | 1/1 |
| saved_initial_identity | 2/2 |
| corrected_codec | 2/2 |
| tensor_assessment | 4/4 |
| snapshots_load | 1/1 |
| sector_context | 1/1 |
| physical_sector | 1/1 |
| sector_assessment | 4/4 |
| saved_sector_replay | 2/2 |
| fidelity_and_budget_gate | 2/2 |
| 合計 | **20/20** |

codec側9＋sector側11、20 attempt／20 COMPLETE／unexpected FAIL0。新fit・optimizer iteration・初期DF分解・rank探索・gradient・PF・native build/compile・shots・総費用・retry/resumeはすべて0。旧科学実行の残call・grant・STOPを再利用していない。

実行前はSOURCE42、保存入力14、ffsim util hash、両専用venv版のpreflightと人工54 testsがPASSした。固定SOURCEは無変更で、実行後もSOURCE／入力、旧初期化raw43・旧H6 raw108・H2 raw40、今回raw32のbyte hashとcall整合がPASS。報告は保存JSONの整形のみで、追加のNPZ decode・norm／tensor再計算はしていない。原本・shared venv・ffsim/SciPy本体・他ユーザーjobに変更なし。

起動時affinity32 CPUのidleは約0.78%、空きRAM約23GiBだったため、このjobだけnice10にして既存jobへCPU優先度を譲った。人工的なCPU/RAM quotaは追加せず、固定の4-state resource-aware並列／process-local BLAS1を使った。wall11.596177秒、親process lifetime CPU11.102450秒、親peakRSS547,320 KiB／child peakRSS425,424 KiB。CPU値は親子合計ではなく、peakも同時使用量の和ではない。メモリpressure/OOMの異常は観測せず、共有設定を変更していない。speedupは主張しない。

契約SHA256 `5e4ba1812e6da0419b43941e239cc06dfeb141507de82749c77cfc33b3fbdfbe`、SOURCE manifest SHA256 `2ee53559c0164077bd8910f9a075c8f750bf82acab00671e365b89ff850cffa7`。元H6 input SHA256 `b82ee9864d9c76071944834ae32d3d6eea7e95959708c2e19bca011936bce447`。過去のSOURCE・結果・grant・ledger・marker・STOPは保全し、local実行の公開証拠を外部独立再現やCIの科学再現とは扱わない。

## 次の担当とmandatory STOP

ユーザー／研究Cが、修正codecをffsimの初期化へ接続する技術準備と、最小1fit条件の検討を次に依頼するか判断する。Codexは具体的依頼後にoptimizer遮断の人工接続試験と新SOURCE／条件を準備する。新fitの実行、PF・回路・shots・総費用評価や重要な科学レビューは、それぞれ該当する別の明示許可後だけ行う。

結果・証拠をcommit・pushし、remote SHA・SOURCE・成果物を照合後、**mandatory STOP**。今回の許可で次のfitへ自動進行しない。
