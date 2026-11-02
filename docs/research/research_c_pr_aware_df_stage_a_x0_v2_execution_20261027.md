# 研究C：PR-aware DF Stage A・qualified x0 V2限定科学実行

2026-10-27 JST。Repository：HIROMU1015/Partially-Randomized-Trotter。

**固定V2契約を無変更で一度実行し、30/30 logical unitsがCOMPLETE、correctness FAIL0、各arm 1 fit・計2 fit、retry/resume0。Stage Aの登録判定は`STAGE_A_PROXY_AND_DF_PASS`。両終了値が物理DF予算内で、Proposedのexact tail massはBaseline比34.7246449%小さかった。** これは固定DF／代理目標の数値pilotであり、PR総資源削減・最終CA・科学的新規性の判定ではない。

## 固定SOURCE・実行範囲

| 項目 | 固定／実行条件 |
|---|---|
| 公開科学SOURCE | `e1085c95f5c01b68566e17d543ace5832cb94061` |
| 数値実装SOURCE | `5a185964090f2cb8d19d445d41cf520c4d8ed2c7` |
| Scope | `C_PR_AWARE_DF_PAIRED_STAGE_A_X0_V2` |
| [契約](research_c_pr_aware_df_pilot_contract_x0_v2_unapproved_20261027.json) SHA256 | `610be0acc2357f3e9fb0e618d67825ea8d7039292a0fe34fcd81b021e9a9fc5d` |
| [SOURCE manifest](../../artifacts/research_c_pr_aware_df/2026-10-27/technical/SOURCE.json) SHA256 | `5deabdfab70029ba6ef784e326908fb7a8d8ffdef6b90e0142887c0e59e9a600` |
| 系 | 保存直鎖H4・隣接1Å・STO-3G、Nα=Nβ2、sector36 |
| 元input SHA256 | `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50` |
| 保存初期CDF8 export SHA256 | `c33f782d3224aa80968e1e6ea70f5ae3f764b872b5bce545c38875c83f3c4e08` |
| DF／変数 | 通常number CDF、rank8、一般実対称J／実直交U、元frame順prefix LD2 |
| 目的 | `Fγ=f/f*+γ(Mε/M*)²`、Baselineγ0／Proposedγ.1、ε1e−8 Ha、loss floor1e−14 Ha² |
| 初期化 | 共通保存J/U→修正real Schur codec一回→qualified x0をそのまま既存ffsimへ渡す |
| optimizer | ffsim0.0.83、既存L-BFGS-B、options=None。`tol=1e−8`はffsim初期化側の設定でSciPy tol指定ではない |
| 物理予算 | sector norm＋既存engineering guard≤CA/8=.00019875 Ha |
| 元規約 | 元g補正κ固定、核scalar一度、spin／frame順維持 |
| fresh output | `artifacts/research_c_pr_aware_df/2026-10-27/run01` |

今回のユーザーの明示許可に従って実行した。`UNAPPROVED_PROPOSAL`という歴史的契約metadataは書き換えていない。承認証明・承認原文保存・新grantは作成していない。旧18-call STOPをresumeしたのではなく、別途許可された新V2範囲を実行した。旧出力・旧契約・旧SOURCE・STOPは保持する。

既存runner／ffsimのloss・gradient／修正codec／general-K／SectorCAR／rank assessmentを再利用し、SOURCE・scientific条件は実行中に変更しなかった。Baselineも共通初期lossで正規化した新fitであり、旧H4 CDF fitの再現実験ではない。

## Preflight・実行順序・資源

[preflight](../../artifacts/research_c_pr_aware_df/2026-10-27/execution_preflight.json)ではSOURCE29件、保存初期export／metadata／SOURCE3件、元input・契約・manifest、fresh output、旧STOPを確認した。fit／evaluation双方の契約package版と固定ffsim2sourceもPASS：[fit環境](../../artifacts/research_c_pr_aware_df/2026-10-27/fit_preflight.json)。旧準備の人工61 tests＋29 subtestsを再実行せず、固定済み成果を利用した。

affinity32 CPU、空きRAM約52.74 GiB、load平均約1.02、CPU／memory pressure avg10/60/300は0。既存resource planで2個の独立fit processを並列起動した。小行列BLASは既存planどおり自job内1 thread、JAXのCPU並列・CPU/RAM/worker/wallに新しい人工上限は設けていない。cache/tempは独立worktree内、共有venv・他job・共有設定は変更していない。

byte preflight→共通初期化／tensor／scale→初期sector／表現／予算gate→並列2 fit→両終了tensor／sector／表現→比較→mandatory STOPの順に実施した。固定runnerは一回だけ起動し、実行時SOURCEを`run01/SOURCE.json`へ保存した。

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /tmp/prt-c1-stage2-preparation-20261025/.venv/bin/python -B \
  scripts/run_research_c_pr_aware_df.py \
  --contract docs/research/research_c_pr_aware_df_pilot_contract_x0_v2_unapproved_20261027.json \
  --input /tmp/prt-c1-h4-b3-execution-20261026/.external/P-H4.npz \
  --output artifacts/research_c_pr_aware_df/2026-10-27/run01 --execute \
  --fit-python /tmp/prt-c1-h6-corrected-fit-preparation-20261025/.venv-fit/bin/python
```

上記は履歴であり再実行の許可ではない。計算後にSOURCEを変更して再実行していない。

## 初期値と終了値

元初期値とcodec後のtensor Frobeniusは約6.609029e−5。共通initialのsector normは7.666027761495033e−5 Ha、guard3.278760734406151e−9 Ha、和7.666355637568474e−5 Haで初期予算PASS。codec忠実性とsector36 general-K／raw-CARの独立表現gateもPASSした。

初期exact massは両armとも2.2093719370656357 Ha、smooth massは2.2093719370722074 Ha。共通`f*=2.1839632053132894e−9 Ha²`、`M*=2.2093719370722074 Ha`を変更せず使用した。

| 終了指標 | Baseline γ=0 | Proposed γ=.1 |
|---|---:|---:|
| 実rank | 8 | 8 |
| exact tail mass [Ha] | 2.2097618367788057 | 1.4424298855756243 |
| smooth tail mass [Ha] | 2.2097618367853338 | 1.4424298974848968 |
| tensor Frobenius | 5.607958698344648e−9 | 6.285301032493658e−9 |
| tensor max-entry | 9.689323562562511e−10 | 1.350095923946526e−9 |
| ffsim raw tensor objective | 1.572459914037971e−17 | 1.9752503728467674e−17 |
| combined objective Fγ | 7.200029332968556e−9 | .04262376967322518 |
| 物理sector norm [Ha] | 7.018855038843621e−9 | 9.039161828383144e−9 |
| numerical guard [Ha] | 3.278781198621627e−9 | 3.2564531608037766e−9 |
| norm＋guard [Ha] | **1.0297636237465247e−8** | **1.229561498918692e−8** |
| DF予算 .00019875 Ha | **PASS** | **PASS** |
| general-K／raw-CAR max-entry | 3.552713678800501e−15 | 2.7755575615628914e−15 |

exact massの登録削減率は**.34724644911133484＝34.72464491113348%**で1%基準を通過した。smooth量を判定へ代用していない。両終了tensor残差・物理誤差は初期値より小さいが、Proposedの終了tensor／sector誤差はBaselineよりわずかに大きい。係数massと物理誤差を同じ目的量として扱わない。

表のraw tensor objectiveはffsim/JAX経路の診断値。独立NumPy再構成に基づく`.5*Frobenius²`も各`tensor.json`に保存しており、tensor／sector／guard／massを混同しない。厳密丸めenclosureは未資格、DF予算PASSは登録engineering数値条件のPASSである。

## Optimizerの停止状態と残る限界

| 指標 | Baseline | Proposed |
|---|---:|---:|
| success／status | true／0 | true／0 |
| nit | 97 | 1129 |
| nfev／njev | 107／107 | 1228／1228 |
| 初期combined objective | 1.0 | 1.1 |
| 初期gradient L2 | 32834.019268206124 | 32834.23292490374 |
| 終了gradient L2 | 3.104515097647238 | 4.673093479443538 |
| 終了gradient infinity | .7423484410886623 | 1.4825504139166679 |

両armの生messageは`CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH`。**successは目的値の相対減少停止によるもので、gradientの小ささや停留点・大域最適性を証明しない。** 非ゼロの終了gradientを隠さず保存した。今回の固定契約はoptimizer successとDF予算・1% exact mass基準を独立判定するため、判定を事後変更せずPASSを報告する。収束条件変更・精度変更・追加fitは行わない。

両armの初期・終了parameters／gradient、生optimizer x/jac、最終J/U、success/status/message/fun/nit/nfev/njevをNPZ／JSONへ保存した。x0修正は実ffsim経路で通過し、以前の`QUALIFIED_X0_CHANGED`は今回発生しなかった。

## 科学call・成果物・保全

prepare6＋Baseline5＋Proposed5＋初期／終了sector・表現／比較14＝**30/30 COMPLETE**。correctness FAIL・未完了call0、実optimizer呼出し2・完了fit2、retry/resume0。内部iteration／nfevをlogical unitに換算したり、追加fitへ流用したりしない。旧18-callを含む本系列の累積は48 unitsで、旧2 FAILも取消していない。

- [全raw・SOURCE・各processのjournal／STOP](../../artifacts/research_c_pr_aware_df/2026-10-27/run01/)
- [両armの比較CSV](../../artifacts/research_c_pr_aware_df/2026-10-27/paired_fit_metrics.csv)
- [実行summary](../../artifacts/research_c_pr_aware_df/2026-10-27/execution_summary.json)
- [SOURCE／input／旧証拠postflight](../../artifacts/research_c_pr_aware_df/2026-10-27/execution_postflight.json)

runner wall4.15409秒／親CPU1.75613秒／親ru_maxrss276648 KiB。prepare wall.72846秒／peak368912 KiB、Baseline wall.81065秒／CPU1.59879秒／peak367884 KiB、Proposed wall1.35905秒／CPU2.56508秒／peak377932 KiB。各processの原値で、job全体の合算peakや速度向上率ではない。

実行後もSOURCE29件・保存input3件・元input・契約／manifestは同一。旧run01とtechnical証拠・旧契約・初期export等53ファイルの不変性も確認した。旧H2/H4/H6・研究A/B・C4・Gaussian singletonの結果／SOURCE／grant／ledger／marker／STOPは変更していない。共有環境・他ユーザーへ影響する操作はしていない。

## Q1〜Q4と次の判断

1. **Exact tail massは減ったか：はい。** 固定prefix LD2・rank8・同一初期値でProposedはBaseline比34.7246449%小さく、登録1%目標を通過した。
2. **物理予算は保持したか：両arm PASS。** norm＋guardは約1.03e−8／1.23e−8 Haで予算.00019875 Ha内。厳密CA保証ではない。
3. **停止規則と未確立事項：相対目的減少停止。** gradient収束・global optimum・別初期値やγへの頑健性は未検証。今回の単一paired結果から一般化しない。
4. **Stage Bの技術的情報価値：次の機構検証を設計する材料は得られた。** 物理予算を守りtail massを減らす例が得られたため、新θでlawを再生成し、固定政策の減衰・PF／補正RTE・native価格を両arm公平に調べる検証には具体的対象がある。ただしtailから決定論prefixへ負担が移った可能性やUの変化によるPF誤差／Gaussian価格の変化は今回測定していない。mass削減をshots・総費用改善へ読み替えない。

Stage Bでは新J/Uに対応するsampling law・identity phase・finite-bit適合を確認し、同じPR政策・Gaussian精度／合法fusion／compilerで比較する必要がある。既存の強baseline結果や旧PR価格を新分解へ転用しない。全law合成誤差・law差・費用support、最終CA、shots、期待総RZ/CX/depth、独立新規性は引き続き未資格で、結果schemaではnullを維持した。

研究C／ユーザーがStage Bの採否・科学範囲を判断し、Codexは具体的指示後に必要な最小技術準備を行う。今回の許可でStage B/C・新fit・目的再設計・追加候補・重要な科学レビューは開始しない。成果公開・remote照合後、**mandatory STOP**する。
