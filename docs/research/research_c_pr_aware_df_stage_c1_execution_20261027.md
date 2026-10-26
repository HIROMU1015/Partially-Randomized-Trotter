# 研究C：PR-aware DF Stage C1・保存値の条件付きshots／serial総資源評価

2026-10-27 JST。Repository：`HIROMU1015/Partially-Randomized-Trotter`。

**固定SOURCEを変更せずfresh C1-run01を一度実行し、164/164 logical unitsがCOMPLETE、correctness FAIL・retry・resumeは0。** 証拠経路は全26roundで`UNRESOLVED_SIGNAL_ERROR`となり、shots／総費用をnullのまま保持した。登録済み2シナリオでは両armの全13roundが条件付き十分条件を満たし、Proposedの総RZ点推定はBaselineより約83.1%小さかった。これは仮定付きの数値pilotで、実装のCA達成・保証付きshots・母平均資源優位の認定ではない。

## 1. 実行SOURCE・入力・固定条件

| 項目 | 固定値 |
|---|---|
| 実行commit | `b1c735da394d4b0a444cbc618ead9781ef4d9134` |
| 実装commit | `e9f704e330e8d281f3f136524997f8e156efe8ff` |
| Branch | `codex/research-c-pr-aware-df-stage-c1-execution-20261027` |
| 契約 | [C1固定契約](research_c_pr_aware_df_stage_c1_contract_draft_20261027.json)、scope `C_PR_AWARE_DF_STAGE_C1_CONDITIONAL_RESOURCES_V1` |
| 契約SHA256 | `d2256190c1b0e5c5daf4a51e90e13d727e5836f7a0218321356f2eab85fd08ee` |
| SOURCE manifest SHA256 | `bc2620634a6560e0492337547db0574451dd6576749bcf4e502e4d1ab99088f0` |
| 元H4入力SHA256 | `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50`。今回はdecodeせず、保存来歴だけを再利用 |
| Baseline final J/U SHA256 | `f3a3141059e29f79f39c3f6d3721562abfa7221ad97ce06a118331f6583e1a02` |
| Proposed final J/U SHA256 | `f47a9fbf86918458ea0db6549468f2041b3b87bba6649ff70234467a442cd227` |
| 出力 | [C1-run01原本96ファイル](../../artifacts/research_c_pr_aware_df_stage_c/2026-10-27/C1-run01/) |

チャットの明示的な実行許可に基づいた。歴史的な契約名・`UNAPPROVED_PROPOSAL` metadataと技術準備STOPは変更せず、承認証明書・承認原文保存・新grantを追加していない。

対象は保存直鎖H4・隣接1.00 Å・STO-3G、Nα=Nβ=2／sector36、Stage A最終の通常number CDF rank8のBaseline／Proposed。元frame順prefix LD2、元g補正κ、核scalar一回を維持した。Partial symmetric S2、δ=0.1、s=4、h=1/40、r=16、K=0、m=0..12。CA=0.00159 Ha、DF reserve CA/8、βtotal=π/6、βstat=π/12、各候補failure .01／X/Y各軸 .01/26、own-DF ground-state oracle、状態準備費用除外。両armの物理DF誤差は異なる保存値のまま扱う。

B1科学SOURCE `7da0fdb3ff7c2e5e22c7d8658f6980077993c5d9`のPF／補正RTE／attenuation／ground-alias、B2科学SOURCE `a1cf0046967c206f414dbad568a75884ee076ebf`のrun02だけの32標本/armとshellを再利用した。B2 sampling digestは`cdf79ed569127f5cb6c7ee017b76bce6422166a656bb3ae866de6e86917c45bf`。旧run01の標本をpoolしない。共通Gaussian・相対融合・P0 Qiskit1.3.0／opt1／seed17／rz,sx,x,cxの保存価格であり、追加最適化は導入していない。

## 2. 全gate・科学call実績

| 処理 | COMPLETE |
|---|---:|
| saved bundle／identity／target-horizon | 各1、計3 |
| arm attachment／cost calibration | 各2、計4 |
| serial cost：2arm×13round×X/Y | 52 |
| signal qualification：2arm×3lane | 6 |
| round accuracy：2arm×3lane×13round、shots反転を内包 | 78 |
| energy／total aggregation／shared-calibration uncertainty | 各6、計18 |
| lane comparison | 3 |
| **合計** | **164** |

[Call journal](../../artifacts/research_c_pr_aware_df_stage_c/2026-10-27/C1-run01/science_calls.jsonl)のattempt164件とCOMPLETE164件はkind／labelごとに一対一、FAIL・inflight0。[STOP](../../artifacts/research_c_pr_aware_df_stage_c/2026-10-27/C1-run01/STOP.json)はfailed=false／retry0／resume0／mandatory_stop=true。[事後照合](../../artifacts/research_c_pr_aware_df_stage_c/2026-10-27/C1_execution_audit.json)でSOURCE60、保存入力408、既存SOURCE・script・PR-aware証拠1644ファイルのbyte保全を確認した。

Identity、arm/export/seed、既存raw/compiled/reference/adjoint/X-Yの保存判定、共通policy、固定13round horizonはPASS。これらは保存gateの再確認であり、新たなoperator／回路検証ではない。Numerical range異常・十分条件不適格は今回なし。未資格の証拠経路はcorrectness FAILとして扱わない。

## 3. 証拠経路と明示仮定シナリオ

| 経路 | 仮定／資格 | Round／energy／資源 |
|---|---|---|
| `evidence` | 全law synthesis／implemented-law raw-signal bound不明 | 両arm各13/13 UNRESOLVED、energy UNQUALIFIED_ROUNDS、shots／total null |
| `ideal_exact_native` | counterfactual exact synthesis、ideal finite-RTE IID law、B2 finite-bit標本平均をideal費用proxyとする仮定 | 両arm各13/13 CONDITIONAL_ROUND_PASS、CONDITIONAL_CA_PASS、条件付き点推定のみ |
| `assumed_raw_signal_budget` | 毎roundでsynthesis≤.001 A、law差≤.001 Aという未検証の一様raw-signal仮定、費用proxyの定常性 | 両arm各13/13 CONDITIONAL_ROUND_PASS、CONDITIONAL_CA_PASS、条件付き点推定のみ |

未知誤差を実際に0と認定していない。第1シナリオの0はcounterfactualの定義、第2の.001 Aは実測boundでもgate toleranceでもない。標本最大残差を全law上界へ昇格させず、`CA_achieved`、`shots_guaranteed`、`total_resources_guaranteed`、`cost_support`等はトップレベルでnullを維持した。

既存評価は同一raw complex-signal normで `d=A(εPF+εRTE)+εsynth+εlaw` を作り、`ρ=A−d`と登録phase roomからHoeffding十分shotsを算定する。情報理論的最小shotsではない。DFのHa誤差はsignalへ混ぜず、phase→energy後に一度だけ加えた。両シナリオのenergy十分評価値はBaseline `0.0012783276208742716 Ha`／Proposed `0.0012783296188530234 Ha`、candidate failure upperは各.01。実測energy誤差・厳密enclosure・両候補同時failure .01ではない。

## 4. 条件付きshotsと総RZ点推定

以下はX/Y・全13roundの合計。Shotsは算定数であり、量子測定やtrajectoryを新たに抽選していない。

| 仮定シナリオ | arm | 合計shots／executions | 条件付き総RZ点推定 | 記述的校正SE（RZ） |
|---|---|---:|---:|---:|
| ideal_exact_native | Baseline | 53,136 | 3,194,333,274,946 | 40,995,736,145.65 |
| ideal_exact_native | Proposed | 19,574 | 540,290,160,109 | 9,112,902,093.39 |
| assumed_raw_signal_budget | Baseline | 53,380 | 3,209,642,597,218 | 41,192,214,372.54 |
| assumed_raw_signal_budget | Proposed | 19,658 | 542,769,406,705 | 9,154,718,756.64 |

登録したProposed/Baseline総RZ点推定比はidealで`0.16914019721944434`（83.08598%低い）、感度シナリオで`0.16910587090769935`（83.08941%低い）。B2のkernel単価差約8.32%だけでなく、保存attenuation／補正RTE誤差から生じるshotsと長roundの反復重みの違いを含む結果である。個別機構の因果寄与を切り分けた追加実験ではない。

### RoundごとのX/Y各軸shots

| m | ideal Baseline | ideal Proposed | .001 A各項 Baseline | .001 A各項 Proposed |
|---:|---:|---:|---:|---:|
| 0 | 512 | 512 | 514 | 514 |
| 1 | 512 | 512 | 515 | 514 |
| 2 | 513 | 512 | 516 | 515 |
| 3 | 516 | 514 | 518 | 516 |
| 4 | 520 | 516 | 522 | 518 |
| 5 | 527 | 519 | 529 | 521 |
| 6 | 542 | 526 | 544 | 528 |
| 7 | 570 | 538 | 572 | 540 |
| 8 | 635 | 565 | 637 | 567 |
| 9 | 787 | 623 | 790 | 626 |
| 10 | 1,212 | 760 | 1,217 | 763 |
| 11 | 2,889 | 1,133 | 2,901 | 1,138 |
| 12 | 16,833 | 2,557 | 16,915 | 2,569 |

証拠経路の同じ26roundはすべてnull。[全78round](../../artifacts/research_c_pr_aware_df_stage_c/2026-10-27/C1_round_accuracy78.csv)／[156 lane-arm-round-axis行](../../artifacts/research_c_pr_aware_df_stage_c/2026-10-27/C1_interrogation_axes156.csv)／[6 arm-lane資源行](../../artifacts/research_c_pr_aware_df_stage_c/2026-10-27/C1_conditional_resources6.csv)。CSVは保存済み出力のserializeで、評価関数の再実行ではない。

### 副指標：ideal_exact_nativeの条件付き総費用

| 指標 | Baseline | Proposed | Proposed/Baseline |
|---|---:|---:|---:|
| RZ count | 3,194,333,274,946 | 540,290,160,109 | .1691402 |
| RZ depth | 1,268,642,747,401 | 219,452,620,009 | .1729822 |
| CX count | 1,601,103,148,210 | 280,340,936,400 | .1750924 |
| CX depth | 1,240,683,073,505 | 221,280,584,700 | .1783538 |
| total depth | 2,768,469,723,089 | 478,863,343,531 | .1729704 |
| circuit size | 6,209,963,917,154 | 1,045,946,875,231 | .1684304 |

感度シナリオを含む全6指標・差・比は[result.json](../../artifacts/research_c_pr_aware_df_stage_c/2026-10-27/C1-run01/result.json)に保存した。Depthは登録serial fence付き段のshots加重和であり、実長回路全体を再compileしたdepthや最小depthではない。

## 5. 費用モデルと不確かさ

各interrogationは`n_m=4*2^m`回のcomplete native kernelと、ancilla Hadamard head＋X/Y shell一組。保存B2の共通P0 kernel価格を線形に合成し、shellは一度だけ加算した。各実shotの各coherent occurrenceで独立のfresh trajectoryを使うモデルであり、同じ標本回路の反復で費用を過小評価する政策ではない。状態準備、long-roundのglobal compiler最適化、実deviceのnoise等は含めない。

各armの同じ32校正標本を全13roundへ再利用するため、総費用の記述的SEは単一平均のSEに`W=Σ_m,axis N_ma n_m`を掛ける。保存Wはideal Baseline614,494,240／Proposed113,366,400、感度シナリオ617,439,296／113,886,608。Round間でsqrt(13)縮小せず、32／64をquantum shots倍率にしていない。

SEは保証CI・母平均の囲いではない。全law費用support／ideal-law平均への転送は未資格で、すべての費用intervalとguaranteed advantageはnull。今回の大きな点推定差を、それだけで保証付き母平均差、実装同CA優位、他分子への優位、新規性へ昇格させない。

## 6. 実行環境・資源・保全

[Preflight](../../artifacts/research_c_pr_aware_df_stage_c/2026-10-27/C1_execution_preflight.json)は固定commit・契約hash・SOURCE60・保存入力408・fresh output不存在・依存版を照合した。Python3.11.1／NumPy1.26.4／SciPy1.14.1／threadpoolctl3.6.0の既存evaluation venvをread-onlyで使用。CPU affinity32、MemAvailable約52.7 GiB、CPU／memory pressure avg=0、実行は6独立scalar task並列。CPU/RAM/worker/wallの人工上限は追加していない。

最初のpassive資源記録だけ、存在しないcgroup v2 cpu.maxの無条件readで中断した。Optional情報をnullとして記録し直し、科学SOURCEは無変更、科学output／callはまだ0だった。これは科学retry／resumeではない。本科学runnerは一度で正常終了した。旧packageのeager importによるMatplotlib cache warningは保存logに残り、書込不能な共有設定ではなく/tmpの自動一時cacheが使用された。共有venv/cache・OS設定・他jobは変更していない。

Runner wall `1.4919445412233472 s`、保存CPU_seconds `9.866716421 s`、peak RSS `349656 KiB`。CPU_secondsは既存runnerのprocess_time終値で、start差ではない。Preflightから公開までの全作業wallではなく、speedup比較でもない。

元の人工23＋回帰50＝73 PASSは[技術準備](research_c_pr_aware_df_stage_c_readiness_20261027.md)の固定SOURCEに対する保存結果。本実行では科学SOURCE変更がなく、同じtestを重複実行していない。代わりに今回のcall・全出力・null／qualification・保全・hashを事後照合した。旧SOURCE/raw/grant/ledger/marker/STOP、H2/H4/H6決定論比較、B2 run01の技術STOPは保持した。

再現用の一回の実行command（消費済みrun01へ再実行しない）：

```bash
env PYTHONDONTWRITEBYTECODE=1 \
 XDG_CACHE_HOME=/tmp/prt-c-pr-aware-df-stage-c1-execution-20261027/.cache/c1 \
 /tmp/prt-c1-stage2-preparation-20261025/.venv/bin/python -B \
 scripts/run_research_c_pr_aware_df_stage_c.py \
 --contract docs/research/research_c_pr_aware_df_stage_c1_contract_draft_20261027.json \
 --output artifacts/research_c_pr_aware_df_stage_c/2026-10-27/C1-run01 --execute
```

## 7. 取得できた結論と次の担当

固定H4・rank8・LD2・δ.1/s4/r16/K0・own-DF oracle／状態準備除外・共通P0・serial費用モデルにおいて、2つの登録仮定下ではshots加重後にもProposedの点推定が小さいことを確認した。Actual全law synthesis／implemented-law誤差／費用supportは未取得で、証拠経路は未解決のままである。

次は**ユーザー／研究C**が、[C2非実行草案](research_c_pr_aware_df_stage_c2_design_draft_20261027.json)のうち全law synthesisとimplemented-law差の資格付けに向けた最小の技術調査を依頼するか判断する。**Codex**は具体的な次指示後に既存emitter／P0／finite probability tableの再利用を優先してscopeを具体化し、新しい実分子評価はその固定範囲への別明示許可後だけ行う。追加samplingだけで資格不足が解消するとは扱わない。

本報告は固定契約の実行・証拠整理であり、重要な詳細科学レビューやC2の採択・実行を含まない。結果・SOURCE・保存入力のcommit/push・remote照合後に**mandatory STOP**。追加fit、sampling、build／compile、上界評価、別条件、重要科学レビューへ自動進行しない。
