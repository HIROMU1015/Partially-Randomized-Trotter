# B/C統合：全13-round条件付き総資源会計の固定PLAN

2026-11-08 JST。ユーザーの今回の明示指示に基づく**一campaign**。主評価A・感度F/Sを実行前に[SCENARIOS.json](SCENARIOS.json)で固定し、結果で変更しない。新規H4 sampling/build/compile、価格fit/bootstrap、N再算定、actual保証、詳細科学レビューは行わない。

## 科学条件と価格採択

Linear H4/隣接1.00 Å/STO-3G/Nα=Nβ=2、元H、number-CDF8/元frame順LD2、ordered partial S2/s1/r32/K0、h=δ=.1。元H厳密groundを仮定、準備費除外。packet1d6087f6・CDFa8813aca・lawc347d7f2の完全hashはB6b adapter bindingを継承する。

B4-2固定07d18bfeの26 axis行・13round q1..4096をそのまま使用。energy .00159 Ha/failure .01、βtotal π/6/βstat π/12/αaxis .01/26/prior[-10,10]/decoder/normalization/attenuationは変更しない。14,738は条件付きの保存**十分shot数**であり、必要最小値またはactual保証ではない。B6bの全適格path ideal blueprint、正しいχ/actual adjoint/V-ray/測定、同じideal product lawと各軸内fresh independent trialsを明示仮定にする。

q1はStage A 3dc9a61の保存32平均・event16 cutを独立scopeで採用。q2/4/8/16/32はC6a/C6b/C7-1/C7-3固定版の各16/16/32/16/8平均。既存paired_statisticsを読み、新しいpath価格統計の再推定をしない。q≥64はC7-4公開64072ba8・SOURCEdd31e1e2の保存モデル：A=affine、F=元C6a凍結affine、S=1/q/sqrt(q) design。basis/lane-runは使用しない。モデル単価は整数へ丸めず、全6資源の同じscenarioを使う。

全価格はsingle_hadamard_interrogation_without_state_preparation、同P0（Qiskit1.3/opt1/seed17/rz,sx,x,cx/approx1/initially_zero=false/topology-free/位相保持）。既存strong最適化とwhole-path Referenceの保存価格を保持。切替えたSOURCEのsampling/compile来歴は元記録のまま参照する。

## 再利用と実行

既存B6b `read_b4`/`price_catalog`/`common_N`/`aggregate_registered`を再利用し、元`aggregate_totals`へ委譲する。旧SOURCEは変更しない。新薄いscalar adapterは保存平均/model係数をscenarioごとに52 recordへ展開し、finite/非負/identity/P0/SOURCE/cut/単位/全6資源を検査する。人工テスト後、SOURCE・本PLAN・scenario・出力schemaをGit commitで固定し、明示runner `--execute`を一回だけ起動する。`--preflight`は固定metadata/schema/平均対応の照合だけで、モデル予測や総和をしない。

出力は`artifacts/track_d_bc_total/2026-11-08/run01`。SOURCE/input_bindings/preflight、A/F/S各prices52.json・contributions.csv（312行）・paired_rounds.csv（156行）・result.json、campaign result/STOPを保存。各scenario完了時にpartialを逐次保存する。各資源の方式別G・ΔG・削減率、13-round/axis寄与、q1/q2..32/q64..4096寄与を記録。peak logical qubits9/17は非加算。attenuation追加乗算、RZ→T、累積depth→実機時間はしない。

## 不確かさの固定規則

保存mean-price共分散だけを線形伝播し、新bootstrap/fitやpath再解析はしない。q2..32についてC7-4 integrated_statisticsの保存36量joint covarianceを使い、元paired_statisticsの採択平均と全量一致を照合する。S_q/nの名目的mean-price変動は保存pseudo-IID解釈に条件付きで、IID証明ではない。

A/Sは保存係数共分散を一つの共有blockとして全七つの長qへ伝播し、method/axis/資源/round依存を保つ。Fは元凍結係数と保存48係数共分散を使用。Fのβは同じq2/q4平均の可逆線形変換なので、同じβからq2/q4直接寄与と長qを結んだjoint componentも保存する。このcomponentを元q2/q4/長q componentへ重ねて加算しない。Fの保存cross-q=0は既存の明示したpseudo-stream仮定としてのみ継承する。

q1 full joint covariance、A/Sの直接平均とfit係数間、他direct-q間、scenario間のjoint covarianceは不足。**total covariance/SD/CIはnull**とし、既知blockの条件付き変動を全体の保証や下界と呼ばない。model bias/law/長q外挿誤差もnull。モデル間差は感度であり信頼区間ではない。

会計はbinary64、恒等式照合はrelative2e−12/absolute2e−7。共分散の減算で負の丸め残差が出た場合は64ε|L||Σ||Lᵀ|（diagonal、最小1e−14）以内だけSD表示を0へ丸め、元の共分散・負残差・allowanceを保存する。unknownを0にする規則ではない。超過はSTOPし、実行後に閾値を変更しない。

## 保全・STOP

価格/coefficients/input/SOURCEの不一致、欠損、非finite/負の絶対価格、会計恒等式不成立、結果を変える修正・新仮定が必要な場合はpartialと原因を保全しSTOP。科学実行後の無断retry/resumeはしない。unknown共分散やbiasはnull表示できる限り点推定の停止理由にしない。

保存scalarと小行列の会計であり、大規模回路計算はない。重複読取りをcacheし、小行列はNumPyで伝播する。プロセス分割の費用が支配するため単一campaign processで十分で、CPU/RAMの人工上限は設けない。必要な空き資源の観測はローカル非公開。共有OS/venv/scheduler、他job/worktree/cacheに変更しない。

旧B4/B6b/C6a〜C7-4のSOURCE・raw・結果・STOPは読取専用。公開は新scalar会計・SOURCE/hash・報告書とSTOPだけで、内部絶対path/資源観測やevent/QPY/NPZを追加公開しない。専用branchへcommit/push、remote SHA/tree/public bytes照合後**mandatory STOP**。次の担当はユーザー/GPT、後続計算・詳細レビューを自動開始しない。
