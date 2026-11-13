# S3最終科学実行計画案：固定H4 PRの全13ラウンド

2026-11-13 JST。**提案・未採択・未実行。** 今回の許可は計画の作成・公開までであり、以下の科学条件の採択やS3実行許可を意味しない。新しい承認証明書・grant発行は不要。ユーザーが採択した条件と有限範囲への明示許可を受けた後だけ実行する。

基礎は[既存SCIENTIFIC_PLAN](../rpe_decoder_statistics_20261113/SCIENTIFIC_PLAN.md)と[S3_DECISIONS](../rpe_s3_readiness_20261113/S3_DECISIONS.md)。旧計画の「nominalは3roundのみ」という入力状況は、B bf515で全13round受入済みに更新された。過去の計画・SOURCE・結果・STOPは書き換えない。

## 1. 固定SOURCE・入力・対象

| 役割 | 固定commit・読取対象 |
|---|---|
| B実行コード | `ab12b69120323379b0aae572e7f62692a5819125`：`rpe_decoder_statistics.py`、`rpe_decoder_fast.py`、元scalar decoderと依存コード |
| B受入済み入力 | `bf515a976f1987ad6a19c61d14689b27b0894542`：`docs/research/rpe_s3_readiness_20261113/metadata01/STATISTICS_INPUTS.json`、`SOURCE_FILES.json`、`S3_READINESS.json` |
| canonical V2 | 同Bに保存された`docs/research/h4_s0_v2_integration_20261113/run01/CONFIGURATION_V2.json`。元公開SOURCE `030f3f548a35d98a88e62e92a35f132e0243a293`も保持 |
| A物理誤差契約 | `bba150834ca30bdb186fbfd58c55a8d51177b855`：`docs/research/h4_pr_physical_error_contract_20261113/PHYSICAL_INPUTS.json`、HANDOFF |
| A全13round証拠 | `169d3316dd008eb98dd9be56fe4ecefc5c1a7d60`：`docs/research/h4_pr_s1_s2_rmse_results_20261113/run01/N_HANDOFF.json`、`result.json`。新10roundと旧S0/m1の混合来歴をそのまま保持 |
| targetの元証拠 | `89c91c692e1d35de94d2e7afbe890aabe6d80b7e`：`docs/research/h4_df10_rpe_batch_results_20261110/run01/result.json`、selector `records[5].diagnostics`。A物理packetとB入力の参照先 |

canonical IDは`587d20f7ed906d25841615dc9746674af872675b895bd438c06d3e74abf0e911`。Linear H4/1.00 Å/STO-3G、保存tol-only 1e−8 DF10/weighted-Hermitian completion、sector(2,2)、元H/state/snapshot/fragment順、S2/L_D=1/ΔPF=.1/K2/h=.1/M12/元exactκ、prior/decoder/law/compiler/profile/full-wrapper/P=0を固定する。κの表示近似からscheduleを再生成しない。

εE=.00159360010199040 Ha、εqpe=εtrot=εE/2、minkowski/finite_B_diagnosticは変更しない。これらは目標budgetであり、達成済み上界ではない。今回の候補はPR1個だけで、4方式比較の完了ではない。

各m=0..12のX/Y両軸の経験的nominal Nは

`64,60,56,52,47,43,38,34,30,25,21,17,12`。

この配分だけを使い、十分Nへの昇格、旧HP型Nの採用、shot/M/κ/分割の探索はしない。

## 2. 採択を求める推奨セット

| 条件 | 初回S3の推奨案 | 理由・ユーザー判断 |
|---|---|---|
| 対象 | 固定1 configuration・全13round/26軸・上記1 nominal配分 | 全scheduleのbranch挙動を見る。N最適化を混ぜない |
| target | 保存solver energyを条件付きproxyとして採用 | 新ground計算なしで診断可能。厳密groundのRMSEとは呼ばない |
| shot law | 保存raw meanを中心とするideal finite-K/fresh IIDのbinomial plug-in model | 各shot・X/Y・round・反復の独立性を仮定。actual sampler等へ外挿しない |
| count caller | `EXACT_SIGNED_INTEGER_SUM_THEN_BINARY64_RATIO_V1` | B既存exact整数和→binary64をそのまま使用 |
| decoder | `batch_decoder=rpe_decoder_fast.decode_sampled_counts` | B ab12の人工scalar互換性を再利用。旧vectorizedのtie規則は選ばない |
| 主評価の未解決loss | `UNDEFINED`。条件付きRMSE・未解決率・解決分MSE寄与を報告 | 元decoderを変えず、未解決energyを補完しない |
| 副評価（感度） | 別ラベルで未解決squared loss∈[0,U]という仮定のrisk envelope | 全trialに定義された特定推定器のRMSEではない。**この追加仮定も採択が必要** |
| 数値不確かさ | 供給binary64 mean/target固定のplug-in診断。Aのallowance/bias/unknownを付記 | 誤差0の認定やallowanceの新伝播をしない |
| support候補 | U=4096 Ha²、energyの保守域[−32,32] Ha（下記条件付き） | prior clippingなしでalias大lossを残す。未解決loss域は感度仮定 |
| 精度・confidence候補 | 副評価のMSE端点へのMC加算幅η=0.01U=40.96 Ha²、各CI 97.5%、joint≥95% | **粗い初回診断**用。化学精度のMSE検定ではない |
| 未解決率の解像度候補 | 零観測時、97.5%二側CP upper≤10^−4を設計目安 | 成功率目標やloss寄与保証には採択しない |
| 反復数・RNG候補 | n=43,819固定、PCG64 seed=20261113 | 第6節の設計式で決める。CIや結果を見て追加しない |

上記は一括して採択できる具体案であり、自動的に正式条件にはならない。副評価を採らない選択も可能。その場合は主評価と未解決率CIだけを報告し、MSE精度保証を付けない。主評価の条件付きRMSE点推定に対するCIは現APIにないため、勝手に付け加えない。

## 3. Target・shot law・count caller

proxy値はB入力の`target.saved_truth_proxy.reference_solver_energy_ha`から無変更で読む。保存値は−2.1663874486347634 Ha（転記）。`ground_state_certified=false`、`state_residual_ha`は保存solver診断であり、基底状態同定やtarget energyの厳密区間ではない。評価対象は供給state/meanとこのproxyを条件としたenergy誤差。元DF-Hの厳密ground、真の分子Hamiltonian、実backend性能の保証とは区別する。energy shift=0、proxyをpriorやbranch選択へ使用しない。

採択後の新しいrun記録にproxy採用を記す。過去の`energy_ha=null`、`truth_proxy_adopted=false`やAの元packetは上書きしない。proxy採用は物理meanの`QUALIFIED`化ではない。

各round/axisのplus countを

\[
k_{m,X}\sim\mathrm{Binomial}(N_{m,X},(1+\Re\mu_m^{raw})/2),\qquad
k_{m,Y}\sim\mathrm{Binomial}(N_{m,Y},(1+\Im\mu_m^{raw})/2)
\]

とする。これはfresh ideal-law trajectoryとBorn outcomeを毎shot独立に使う測定の**周辺分布model**。RTE trajectory自体をsamplingする実験ではない。有限cohort再利用、跨shot/軸/round相関、actual samplerとideal lawの距離、compiled noiseは含まない。全meanを共有する数値系統誤差が独立になるわけではない。

実装は`NUMPY_PCG64_BINOMIAL_BINARY64_P`。保存rawを(1+μ)/2へ変換するのは次の科学実行時だけ。binary64確率・PRNG/binomial実装を利用し、厳密実数lawの数値認証とはしない。confidenceもこのIID統計modelを条件とする。

rawは既にattenuated。corrected=B×rawをBernoulli平均に使わず、Bを再乗除しない。nominal Nに含まれるnormalization負担を二重計上しない。countはPython整数の`2*k−N`とFraction比を作ってからbinary64へ一回変換し、旧`2.0*k/N−1.0`へ戻さない。PCG64は元のround→X→Y順で全count arrayを生成し、その後高速hookへ渡す。

## 4. 未解決trial・loss・出力の意味

DECODEDのtrial iについて実直線上のsquared lossを`ℓ_i=(Ehat_i−Eproxy)^2`とする。円周phase差でalias誤差を消さない。Rは解決trial数、F=n−Rとする。

- 主評価：`sqrt(sum_resolved ℓ_i/R)`（R=0ならnull）、未解決率F/nとzero/tie別件数、解決分全trial MSE寄与`sum_resolved ℓ_i/n`、prior外件数。解決分寄与を報告することは、未解決lossを0と認めることではない。
- 元decoderは1roundでもX=Y=0ならzero未解決を優先。branch近接判定は元のabs_tol=1e−12、完全tie選択はscalar規則。未解決trialを除去して成功率を計算し直さず、energy=Noneを保持する。
- F=0でもpopulationの未解決確率を0としない。元APIの`all_observed_trials_RMSE`や、零観測時の`all_trial_MSE_interval`は**観測標本の診断**。主評価のpopulation RMSE/CIは未定義・null。
- 副評価：ユーザーが未解決loss区間[0,U]を採用した場合のみ、経験的MSE端点は`[sum_resolved ℓ_i/n, sum_resolved ℓ_i/n+(F/n)U]`。energy fallbackを生成せず、許容するloss補完familyの感度区間として表示する。平方根を取っても元decoderの単一の全試行RMSEにはならない。

**特定の全試行RMSEを定義する追加契約**は、未解決事象でのenergy出力又は固定loss、全事象でのtarget/loss、prior/alias規則とそのSOURCEである。decoder変更や別推定器の採択なら別研究条件として許可が必要。今回の推奨案はそれを行わない。副評価のrisk envelopeも、同一targetの有効L2誤差成分が揃ったMinkowski/global boundではない。

## 5. 物理bias・数値不確かさ・support

Aの`outer_PF_or_discard_bias`、`finite_RTE_bias`、`total_signed_bias`はsignal空間の保存分解。plug-in rawによる評価には、そのrawに表れた近似の影響が入るが、各成分をenergy L2へ移す保証はない。biasをphase/Tへ換算したり、decoder RMSEへ足し引きしてεtrot達成と判定しない。保存state/full-H referenceは、厳密groundの単一位相と同一ではない。

Aの経験的allowance、reference pair、normalization/logB、m3/m4の観測norm欠測・SOURCE_GATE_LIMIT_NOT_OBSERVED、ground同定誤差・actual law差・実装誤差のunknown/nullをそのまま参照する。独立Gaussian誤差とみなしてsamplingせず、confidenceへ吸収せず、未知量を0にしない。共通有効Hamiltonianやbranch不連続点を跨ぐL2/couplingの証拠も未取得。

**supportの推奨候補**：固定h=.1のwrap後principal phaseからの出力域を外側に広げ、解決energyについて[−32,32] Haを用いる。proxyも同域内という契約なら`|Ehat−Eproxy|≤64`からU=4096 Ha²。これは狭いpriorをdecoder出力域と誤認する方法ではない。元scalar/fastのwrap・有限binary64演算に余裕を持たせた運用上の保守域候補で、interval arithmeticによる認証結果を新たに主張しない。

全到達出力がこの域に入るというsupport前提の採択が必要であり、観測値が域内だっただけでpopulation supportを証明しない。実行時もfinite/raw/energy/supportを検査し、域外をclip・破棄せずSTOPする。supportの不確実性を許容できない場合、[0,U]感度評価と有界loss CIは採らず、UNDEFINED主評価だけを行う。Uを小さくするためのpriorへのclippingや保存meanの再解析はしない。

## 6. 有限反復数・精度・同時confidence

以下の整数は**実行計画の汎用算術**であり、H4 mean/確率/energy lossの評価ではない。量子shot NとMC反復数nを区別する。

副評価の端点変数L_−/L_+が0≤L_−≤L_+≤U、反復IIDという採択済み条件の下で、既存`trial_budget`/`summarize`のHoeffdingを利用する。端点の上下片側を合わせた失敗確率α_MSE、各端点へのMC加算幅ηについて

\[
n_{MSE}=\left\lceil{U^2\over2\eta^2}\log{2\over\alpha_{MSE}}\right\rceil.
\]

ηはMSEの単位Ha²。risk envelope全幅や条件付きRMSEの半幅ではなく、端点へのMCの余裕幅である。平方根変換後の幅は別表示し、未解決率由来の識別不能幅も残す。

推奨pilotはα_MSE=.025、η/U=.01なので`n_MSE=21,911`。U=4096ならη=40.96 Ha²と非常に保守的・粗い。**εE²やεqpe²付近を識別できる精度ではない**。主目的は条件付きRMSEの観測値・未解決/alias挙動を得ることで、目標精度達成判定は行わない。

未解決率CIには既存二側Clopper–Pearsonをconfidence=.975（α_fail=.025）で使う。零観測でのupperをp_*=10^−4以下にする設計条件は

\[
n_{zero}=\left\lceil{\log(\alpha_{fail}/2)\over\log(1-p_*)}\right\rceil=43,819.
\]

この解像度を推奨する理由は、仮にCIがp_fail≤p_*を支持した場合、仮定上の未解決lossの最大寄与U p_*がηの1%となるため。解像度をη/Uの100分の1とする倍率はpilot設計の選択肢であり、理論から決まる定数ではなく、ユーザーの採択対象である。

したがって推奨は`n=max(n_MSE,n_zero)=43,819`の**1回の固定実験**。零観測を前提に実行を打ち切る設計ではなく、F>0なら同じnまで実行して実際のCP区間を報告する。p_*は設計解像度であり、未解決率やenergy誤差を小さいと事前認定する閾値ではない。

両summaryのconfidenceを.975にし、同じ未解決率CIを重複した独立証拠として数えない。CP familyとMSE端点familyに各.025を配分し、union boundでjoint≥.95とする。これは供給mean/target・IID・support・副評価loss契約を条件とする名目confidenceで、FP認証/物理系統誤差を含まない。主評価の条件付きRMSE点推定に95%CIが付くという意味でもない。

副評価を不採択にする場合でも、同じn=43,819とCP97.5%だけを採択できる。その場合ηやMSEのconfidenceは主張しない。nを変更する場合は上記のp_*・α・ηの変更として事前に決め、結果を見て増減しない。

**化学精度のMSE確認を求める場合の別案**：旧計画の候補η_confirm=.1εE²と適格Uを同じ式へ入れ、得られるnをそのまま有限scopeとして検討する。今回そのH4確認用nは評価していない。worst-case Uでは極めて大きくなり得るため、43,819へ恣意的にcapして同じ精度を名乗らない。pilotからの分散/rare-event解析、適応法や別CIの開発・実行は後続の別判断であり、自動追加しない。

## 7. 実行APIと出力（採択後のみ）

既存`simulate_prepared`→`simulate`→高速hook→`summarize`をそのまま使用する。R3人工互換性60件と性能は[統合REPORT](../rpe_decoder_r3_integration_20261113/REPORT.md)に固定済み。科学実行時に新scalar比較simulationを自動追加しない。

`allow_conditional_unqualified=True`は、供給経験的meanを固定する今回の条件付きscopeをユーザーが許可した場合にだけ指定する。元preparedの資格を変更してgateを通さない。proxyの採択・confidence/lossの採択は新run metadataへ記録し、旧packetと未資格/nullを保全する。

新規run directory候補は`docs/research/rpe_s3_final_plan_20261113/run01/`（今回未作成）。既存pathを上書きしない。最低限、実行SOURCE/input参照・採択条件、全trial/status/branch/energy、主summary、副summaryを採った場合はその仮定、N/global bound/Gがnullである資格、完了又は部分STOPと未完了件数を残す。全trialを保持し、巨大な既存API出力を理由にstatusを省略しない。保存形式・圧縮は科学内容を変えない実装判断としてよい。

seed=20261113/PCG64・round→X/Y順・runtime/NumPy版・binary64 count変換と確率モデル・`decoder_source`/`decoder_acceleration_source`・実行commitを記録する。今回の計画公開commitとコード固定ab12を区別する。CPU/RAM/worker capを追加せず、既存NumPy batch処理を使う。共有設定・他ユーザー/担当processは変更しない。

## 8. 完了・STOP・科学資格

許可後の有限実行の技術的完了は、入力identity/SOURCE/全13round/26軸Nの一致、固定nの全trial会計、loss/CIの前提とラベルの保持である。RMSEが小さいことや未解決0件を完了条件にしない。

欠測/混在SOURCE・law・canonical不一致、raw非有限/範囲外、非整数N、prior/decoder/callerの変更、support超過、未知status、loss overflow、結果書込失敗・入力破損など継続不能なら部分結果を保全してSTOPする。decoderのzero/tie自体は通常の観測として数え、捨てずに続ける。単なる共有CPU競合を理由に科学workerを停止・再起動しない。

固定n完了後は結果が不都合でも自動再seed・追加反復・shot変更・別lossへの差替え・mean再評価をしない。失敗時の再試行/resumeは停止原因と許可済みscopeを確認してから判断し、無断で初めから再実行しない。既存の一回性・retry/STOP条件も保全する。

全13round十分N、元Hへのmathematical global RMSE bound、actual-sampler資格、compiled C、G、最適候補はnull/未資格を維持する。主評価・副評価とも、PR原論文のfinite-K保証を新たに正式採択した証拠ではない。

**次の担当**：ユーザー＝第2節の推奨セット又は変更点を採択し、固定SOURCE・有限1実験を明示許可。Codex B＝許可された範囲だけS3を実行・報告してSTOP。A＝必要なら後続の物理L2/target/law証拠、C/D＝価格/残差の各既存scope。本計画からA/C/Dの追加計算を開始させない。
