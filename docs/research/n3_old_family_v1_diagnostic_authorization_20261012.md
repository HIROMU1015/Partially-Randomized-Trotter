# N3旧条件族診断 v1：固定S1二例・一回の実行認可

2026-10-12 JST。利用者の「次の作業に進んで大丈夫」を、直前に提示した固定planに沿うS1二例の一回診断、独立保存監査、HIROMU1015への公開・remote再取得、終了後Mandatory STOPへの明示認可として記録する。
[新authorization](n3_old_family_v1_diagnostic_authorization_20261012.json)、[固定plan](n3_old_family_v1_execution_plan_20261012.json)。

frozen source13 `1aa12ab0b777b28f1428d9bb8417af93c4376aae`、source fingerprint `46baa537269db923dd8b8c95e4788555471c60bad1572d3a01ec1e70245c6488`。
plan SHA `ef3c66d2e01e47027d9d382afc324aab9b7263aab3d550c3576c31e0f5bd6ff5`、input fingerprint `81e90cb760d6e6e9774aad808ed40953bae0b461c18532bff78d7ead6e4e584d`。
準備publication `eddb5c003197b07d72f1c294273bc4b62d8ce0fd`から専用execution branchを作成する。
準備時plan/templateのfalseと旧STOPは当時の履歴として保存し、新認可を独立に与える。

対象はS1_edge_pair/S1_distributed_threeの旧族のみ。star、実対称one-body＋非負unordered density、一外部mode8、full10→active9、ν4、元δgoal=.005、scale1.25。
化学geometry/basis/DF rank/policy/L_D/PF窓なし。同じ保存U/c/vと保存OCCの正の有理上端δを維持する。
旧族の全κ∈[0,c−U+δ]・許容μを対象とする数学契約の下で、primal回収／strict dual分離／未判定を区別する。
旧族を数値的に探索できなくても不可行性としない。NORM不成立とFIVE包含を別に保つ。

2 spawn workers、numeric threads各1、analytic優先、endpoint/abscissa cache。23 archived synthetic PASSとsource identityを再利用する。
eigh64/primal96/ray80/混合1024 per case、有理4096bits、explicit演算2,000,000、AS2GiB/worker、sampled aggregate4GiB。
output/log制限・attempt1/retry0を保持し、wall/CPU time capなし。失敗・部分証拠も保存し、sourceや予算を変えて再試行しない。

新input、U/c再生成、OCC探索、真ground/gap/E_forced、quantum compile/shots、旧batch再実行、N1/N2再開は0。
元NO_ADDITIONAL_ACCEPTANCE_OBSERVED/2-base未達/旧結果/STOPとroot dirty・BentoSlide・別Trackを保全する。
科学的結果の意味・中心テーマ・新規性・最適旧幅との差率は採択しない。全結果後Mandatory STOP、次段false、中心テーマnull。

[実行前source/input identity](../../artifacts/n3_old_family_v1/2026-10-12/execution_preparation/source_bound_preflight.json)、
[凍結resource確認](../../artifacts/n3_old_family_v1/2026-10-12/execution_preparation/resource_configuration.json)。
