# 研究C・update064/r32：独立native価格校正と母平均差の統計検証の技術準備

2026-11-03 JST。**TECHNICAL_GO_FOR_SEPARATE_SCIENTIFIC_PERMISSION**。既存Aの統計・native pipelineを再利用し、update064専用の独立固定256件、95%近似t区間、半幅/D≤1%を提案する。最大**2,374 logical calls**。人工・mock・回帰**82 tests PASS**、passive preflight PASS。**実H4の新sampling/build/compile、保存raw再解析、追加母平均推定は0、科学run01未作成**。統計条件は採択・科学実行許可待ちの未承認提案で、公開照合後mandatory STOP。

[未承認科学契約](research_c_update064_r32_price_statistics_contract_unapproved_20261103.json) ／ [新adapter](../../src/trottertracks/c_update064_r32_price_statistics.py) ／ [再利用したA統計engine](../../src/trottertracks/c_r32_price_statistics.py) ／ [entry](../../scripts/run_research_c_update064_r32_price_statistics.py) ／ [追加test](../../tests/management/test_research_c_update064_r32_price_statistics.py) ／ [scalar計画](../../artifacts/research_c_update064_r32_price_statistics/2026-11-03/technical/scalar_planning.json) ／ [技術検証](../../artifacts/research_c_update064_r32_price_statistics/2026-11-03/technical/verification.json) ／ [SOURCE](../../artifacts/research_c_update064_r32_price_statistics/2026-11-03/technical/SOURCE.json)。

## 1. 問い・固定資料・対象条件

問いは、**固定update064/r32の実装sampling lawにおけるcomplete controlled kernelのRZ母平均を新holdoutで推定し、同じupdate064 J/Uの強Dとの条件付き費用差の符号と不確かさを評価できるか**である。現状の約3.73%の点差を母平均優位と認定する作業ではない。同等性・非劣性marginは未採択のためnull。

| 固定対象 | commit/SOURCE・identity |
|---|---|
| B起点・最新下流結果 | `cca6b487c8e1ed3362f8825e95c937ebbcc87755`、科学SOURCE `928fda0a6df260b66eba1bbea63b51f2734e7b54`、[固定B報告][B] |
| A技術コード | `62fa4c0144b9e36f9d96b7dff67628051946632d`、[固定A報告][A] |
| A元code Git blob | `64ee0b23f550fe0a4fb22263424f135e3164c058`（公開本文と照合） |
| A test Git blob | `087eff0c0bc220e03f71bbe449fe4dd1399ae00f`、Bへの取込み後も無変更 |
| update064 candidate SHA256 | `53d8f5027ce9e61bc72f6943c0268666bfcf87515154850a9a99326687b26ed3` |
| 同candidateのU更新結果 | `ea54b9b8ba42ff993d803ec6f5c57e4e622fdee6`、科学SOURCE `025d495166311a58ec9c24c43f07ed6e40f65f42` |
| 共通h/g/constant snapshot SHA256 | `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db` |
| 旧update005資料 | r再調整 `f200e45afecc0a80b9f8a895b68c75ba4dda5dd5`、強D `38b8d0df39b8943178ecef611194819cff1c2169`。来歴/旧seedの参照だけでBの比較分母にはしない |

直鎖H4・隣接1.00 Å・STO-3G、通常number CDF8、一般signed対称J・実直交U、Nα=Nβ=2/sector36、system8/full256/controlled512。元g補正κ・核scalar一回・元spin/frame順、LD2・partial ordered symmetric S2、s1/δ=h=.1/r32/K0/m0..12。CA=.00159 Ha、DF reserve=CA/8=.00019875 Ha、既存RPE βtotal=π/6・βstat=π/12・failure=.01・各round軸=.01/26。Own-DF ground-state oracle、状態準備費用除外。

P0はQiskit1.3.0/opt1/seed17、rz/sx/x/cx、approximation degree1、qubits_initially_zero=false、MPFR160 declared phase・phase fitなし。既存Gaussian/head相対融合・tail exact basis_key共有、fenced serial＋shell会計を保持する。Dは同一update064 J/Uの全9因子ordered symmetric S2/s1。DF fit/U更新/r探索/PF次数/compiler/RPE/sampling law変更0。

[最新Evidence Map](research_c_evidence_map_updated_20261102.md)、[科学的貢献候補](research_c_contribution_positioning_20261102.md)、[Claim統合](research_c_claim_contribution_alignment_20261102.md)の条件・資格を継承する。これらの過去の「新selected」はupdate005を指す箇所があり、今回のupdate064とは区別する。詳細科学レビュー・新規性/論文化の正式判断は行っていない。

## 2. Estimandと保存scalarの扱い

一つの独立校正trajectoryのcomplete kernel RZをXとし、`μ_impl=E_impl[X]`を対象とする。実装lawは、固定64-bit seedからPCG64を初期化し既存r32/K0 samplerと固定P0で価格Xを得るlaw。事前固定のhash由来seed群を、このlawに対する独立擬似乱数replicationというモデルで解析する。固定PRNG実行の数学的独立性・理想lawとの距離を証明するものではない。

`C_PR = 13381920 × μ_impl + 52528`

`C_D = 105038112319`、`Δ=C_PR−C_D`、`R=C_PR/C_D`。

W/shell/D/shotsは保存済みideal RPE会計の固定条件である。Dは同じupdate064の固定kernel/P0価格なので、今回の古典的trajectory校正に由来する第二の標本分散は導入しない。Welchの二標本式やランダムな分母の比推定は不要で、同じ一標本区間をaffine変換する。Dの合成・PF・丸め・actual CAの不確かさを0と認定した意味ではない。

保存B報告の引用は、PR総RZ101118440083、shots15008、W13381920、kernel mean7556.34375・記述SE95.1649629863、総SE1273489921.49、shell RZ52528、D総RZ105038112319/shot13706/kernel11026、差−3919672236/比.962683333226である。これらは**探索・設計用の引用scalar**で、旧32件を再集計して新CI/p値/母平均推定を追加していない。

価格区間が得られても、ideal RPEの重みにμ_implを入れた**条件付き資源モデル**に限定する。Ideal sampling lawの期待価格、全law synthesis/実装law信号誤差、actual CA、厳密全law費用・一般的PR優位は引き続き未資格。

## 3. 再利用方針と最小変更

Aの固定Git objectから`c_r32_price_statistics.py`と元testだけを新B branchへ取り込んだ。Aの実行中worktree・branch・契約・outputへはアクセス/書込みしていない。統計・native処理を別名で再実装していない。

| 部分 | 再利用・今回の差分 |
|---|---|
| mean/variance/記述SE・t区間・affine差/比 | A `price_statistics`/`secondary_summary`を再利用。`statistics`と固定`expected_n`を引数で受ける最小拡張。A既定N640/精度.005とそのtestは保持 |
| native256の実行 | A `execute`に固定studyのmetadata/seed/accounting hooksとNを渡す。Aのnative32×batch loop、cache、QPY/Calls、全件完了後の統計処理を共有 |
| 科学identityと会計 | B adapterがcandidate53d8・同064D105038112319・W13381920・S52528を固定。005 identity/旧D105532782860は混在としてSTOP |
| saved Hamiltonian/ground | B617の保存J/U/G/H・ground/aliasを再利用。元A `ret.prepare`は005の資料に固定され、旧B `downstream.prepare`は新ground solveを行うため直接呼べない。新adapterはload/一致/General-K/CAR/ground residualだけを接続し、fit/DF norm/PF/RTE/ground eigensolve/RPE shots再計算0 |
| sampler/compiler/full512 | `c_pr_integrated_performance.native`、Stage B/B2、StrongBuilder、P0、actual adjoint/X-Yをそのまま呼ぶ。新PF/RTE/native compilerなし |
| SOURCE・STOP | 既存Calls/journalを再利用。新SOURCE/input manifest/fresh出力を別namespaceへ固定。承認証明・新grantなし |

A共有moduleの変更関数はAST照合で`price_statistics`と`execute`の2個だけ。差分は29行追加/16行削除。A元testはbytes不変で22件PASS。新B sourceはAの実行中sourceと同一物とは扱わず、公開commit/manifestで別に識別する。

## 4. 推奨N・精度・計算と保存費用

既報の総SE1273489921.49とn0=32だけから、計画式`H(n)/D=t_(.975,n−1)×SE_old×sqrt(32/n)/D`を用いた。[NISTの平均区間式][NIST]と既存[SciPy1.14.1 t分位点][SCIPY]を利用する。これは許可されたscalarベースの技術計画であり、新母平均推定・正式power解析ではない。

| prospective N | 予測95%半幅/D | 予測半幅/既報点gap | 新compile数 | 新QPY数 | logical calls |
|---:|---:|---:|---:|---:|---:|
| 64 | 1.7132% | 45.91% | 64 | 256 | 640 |
| 96 | 1.3896% | 37.24% | 96 | 384 | 929 |
| 128 | 1.1996% | 32.15% | 128 | 512 | 1218 |
| 160 | 1.0709% | 28.70% | 160 | 640 | 1507 |
| 192 | 0.9763% | 26.16% | 192 | 768 | 1796 |
| 256 | 0.8441% | 22.62% | 256 | 1024 | 2374 |
| 320 | 0.7543% | 20.21% | 320 | 1280 | 2952 |
| 640 | 0.5324% | 14.27% | 640 | 2560 | 5842 |

**推奨N256＝32×8組**。半幅/D≤1%の最小計画Nは184だが、32単位のN192は予測半幅.976296%で、同じ精度に対する将来SD増加の余裕は約2.43%しかない。保存32件の分散見積りの不確かさも考慮し、予測半幅**.844146%**・既報gapの約22.62%、将来SD約18.46%増までの計画上の余裕がある256を選ぶ。N192から64 compile/578 calls増の折衷であり、最も大きいN640を選ぶ理由はない。これはSDや精度の保証ではない。Aの約1.16%点差に対する640/.5%目標をそのまま移植していない。

1%は推定精度の**技術提案**であり、同等性・非劣性・科学的に重要な差のmarginではない。Confidence95%、N256、精度と診断閾値はユーザー／研究Cによる採択・別実行許可まで未承認。分散の変化、希少tail、平均のD側への移動によって精度未達・符号未確定となり得る。予測半幅を実験後の半幅保証、既報gapを真の効果、N256を成功確率保証とは扱わない。

256 compile、1024 QPY、256 full256 reference NPZを保存し、32件ごとに同一armのCAR/Gaussian/shell cacheを共有する。保存B617実行（32PR＋Dほか）はwall969.720秒・raw120677678 bytesだったが、今回とは作業/共有cache/同時負荷が異なるため、所要時間・保存量・speedupを単純比例で断定しない。相応の数十分〜数時間、数百MB以上の負担を見込み、実行前に空きdisk/CPU/RAMを再確認する。時間・RAM等の人工上限は追加していない。

## 5. 近似区間・診断・判断規則

主解析は固定256件の`xbar ± t_(.975,255) s/sqrt(256)`。W>0/D>0なので端点をW×端点+S、−D、÷Dへ変換する。同じkernel平均を13roundに共有するため`SE_total=W×SE_kernel`であり、さらに√13で縮めない。RPE failure=.01と価格alpha=.05は別の条件。

正規IIDならt式に対応するが、本価格分布が正規であるとは採択していない。擬似乱数独立replicationモデルと十分な近似性を前提にする**nominal95%近似区間**で、validated coverage guaranteeではない。有限Nや分布screenだけで未観測tailや近似の妥当性を証明しない。

| 返却状態 | 事前に固定する規則 |
|---|---|
| NOMINAL_MODEL_PR_LOWER | 精度達成・screen clear、Δ区間上端<0 |
| NOMINAL_MODEL_PR_HIGHER | 精度達成・screen clear、Δ区間下端>0 |
| UNRESOLVED_SIGN_NOT_EQUIVALENCE | 区間が0を含む。同等性の証明ではない |
| UNRESOLVED_PRECISION_TARGET | 近似半幅/D>.01。Nを増やさずSTOP |
| UNRESOLVED_DISTRIBUTION_DIAGNOSTIC | abs moment skew>2、または最大delete-one平均変化×Wが区間半幅の25%超 |
| UNRESOLVED_ZERO_OBSERVED_VARIANCE | 観測分散0。ゼロ幅保証を出さない |
| correctness/data FAIL | wrong candidate/D、欠測/非整数/負値/非有限値、seed重複、full512等のFAIL。依存call停止/partial保全 |

Delete-one変化量は既存Aの`max|xi−xbar|/(N−1)`を使用する。Bでは半幅の25%超を工程診断として提案し、skewが相殺する対称の高influence人工例でも保留する。元Aの既定閾値はそのまま保ち、Bの固定policy引数で指定する。この25%やskew2は被覆率/CLTを認証する定理ではない。正当な高価格観測を削除・winsorize・再標本化しない。

主推論はideal-weight RZ一件だけ。他5指標とassumed laneは平均・SE・affine totalsの記述にとどめ、同時区間・多重優位判定は追加しない。Observed min/maxはsupportではなく、true support/Hoeffding型価格保証はnull。区間が0を含むことを同等性・非劣性の認定へ変えない。

## 6. 選択・cohort・seed

既存update064 native32は探索/計画用で、新holdoutへpoolしない。Aのupdate005 prospective640件ともpoolしない。r32や有限update064を選んだ過去の価格・最適化情報から、固定した対象だけを新しい独立holdoutで検証する。対象選定後の条件付き一比較であり、全r/全DF探索の母平均最小性やfamilywise保証ではない。Aが完了してもその結果を自動的に新B契約へ取り込まない。

新master **2026110303**、cohort `update064_r32_prospective_v1_batch00..07`、axis `UPDATE064_R32_PRICE_HOLDOUT_ONLY`。組内sample0..31、global ID0..255。既存SHA256→64bit seed関数とPCG64は変更しない。Master/cohort/新SOURCE digest・arm/candidate・round=-1/axis/sample/occurrenceへseedを束縛する。[NumPyの並列乱数説明][NUMPY]の擬似乱数streamに関する前提を踏まえ、非重複を数学的独立性の証明とは呼ばない。

Passively照合した旧keyは、005探索128、064探索32、A固定契約予定640の計**800**。新256とのhash seed衝突・cohort ID重複0、PRNG draw0。A640は固定公開code/契約の予定namespaceから導出したもので、実行中の進捗・seed rawを読んだ意味ではない。異なるseedで同じevent列になる反復はlawの正常な観測として保持し、棄却しない。

## 7. Stage・科学logical calls・停止

| 部分 | 最大calls | 処理と進行条件 |
|---|---:|---|
| A metadata/seed/会計＋native準備 | 23 | saved gate1、seed gate1、accounting1、input/JU/reference/binding/sector/model6、General-K9、保存ground2、law3。全資格PASSのみnativeへ |
| B 共通native＋8組 | 2347 | 初回共通35、各組289＝9種類×32＋校正summary1。全512 raw/compiled/actual-adjoint/XY。最初のFAILで停止・開始済みtask回収 |
| C 全件validation/主統計/副記述/summary | 4 | 256件がすべて正当なデータとしてそろってから一度だけ実行 |
| 合計 | **2374** | **20+3+35+8×(9×32+1)+4** |

Nativeの9種類はsample/build/reference/raw gate/compile/compiled gate/actual-adjoint pair gate/XY/metrics。共通35はCAR1/raw9/full-sector1/reference context1/law1/Gaussian9/relative4/shell load6/shell assessment3。過去617/5845の未使用枠は流用しない。内部primitive/iteration/量子shotとlogical unitsを混同しない。

新ground eigensolve/DF norm/PF-RTE/RPE shots計算は0。許可後に保存candidate/ground/Hの残差・normalization/alias、General-K・lawを確認し、価格だけを新たに校正する。成分allowance1e-10、signal2e-10、rtol0、phase fitなしを維持する。

組別進捗は完了件数と既存の記述的calibrationのみ。途中CI・rank・符号/精度に基づく打切りをしない。256件未完了なら主推定なし。精度未達/符号未確定/screen保留は正しさFAILと区別するが、いずれも追加標本を自動投入しない。Fresh exclusive run01、retry/resume0、失敗標本の置換0、partial JSON/QPY/Calls/SOURCE/STOPを保全し、終端でmandatory STOP。

## 8. 人工・mock・回帰検証

**82 unique tests PASS**。追加B40、A元test22、既存retuning/native20。Aの実行中環境・SOURCEを変更せず、B自身の小さい人工/mock suiteを逐次に実行した。実H4の科学的性能・母平均・実際の被覆率の証拠ではなく、ローカル技術検証である。

| 検証群 | 結果・主要範囲 |
|---|---|
| [B＋A統計/adapter](../../artifacts/research_c_update064_r32_price_statistics/2026-11-03/technical/adapter-statistics-tests.log) | 62 PASS/7.67秒。正負差/0含み/精度未達/ゼロ分散/skew/対称高influence、t式/affine/共有SE、candidate005混在/旧D分母/ground/W/STOP/law、cohort/seed衝突、source/input drift、partial/missing/FAIL/fresh/no-opt-in |
| [既存native/retuning](../../artifacts/research_c_update064_r32_price_statistics/2026-11-03/technical/reused-native-tests.log) | 20 PASS/12.22秒。既存native32 dispatch、同arm cache、QPY保存、r32/K0、raw/compiled/adjoint/X-Y callbackを人工/mocksで確認 |

Mock256 end-to-endは、8組完了前の統計処理なし、poolなし、wrong/missing sampleやnative correctness FAILでpartial STOP、再開禁止を確認した。追加の人工J/U/G/Hでは新064タグ/既存signed law/r32/保存groundへの接続を検査し、ground eigensolve呼出を禁止してPASS。量子状態は人工入力で、保存実H4 NPZ/QPYは開いていない。

CLIはdefault passive。`run`も明示opt-inなしでは作業前に拒否する。`--execute`/opt-inは起動意思の安全な入口であり、チャット許可をプログラムで認証する仕組みではない。実H4科学実行には別のユーザー明示許可が必要であり、承認証明・保存原文・ID/署名/新grantを要求しない。今回、科学CLIの`--execute`を起動していない。

## 9. SOURCE・passive preflight・Aへの非干渉

SOURCE96件、local input/固定結果/ground/shell等16件、ea54 peer3件。SOURCE SHA256 `8f1864e900f5c1eae5e9453bc8fe4c3c4fdce02852c6f60278be213a31248c5d`、契約SHA256 `af507d6961440ece2d11a4fdfb8474f33d3c3bc39b41435bae72e0d608c77883`。`saved_inputs`/`peer_inputs`がinput manifestである。Source内の固定N/統計/候補/D/W/S/cohort/compiler/seed政策は実行後に変更しない。

数値ライブラリimportを禁止したpassive確認でもPASSした。新run01は未作成で、seed計算はhashだけ、PRNG draw0。次のcommandは今回実行したpassive入口である。

```bash
/tmp/prt-c1-stage2-preparation-20261025/.venv/bin/python -B \
 scripts/run_research_c_update064_r32_price_statistics.py \
 --contract docs/research/research_c_update064_r32_price_statistics_contract_unapproved_20261103.json \
 --candidate-root /tmp/prt-c-u-step-control-execution-20261103
```

専用branch `codex/research-c-update064-r32-price-statistics-preparation-20261103`、worktree `/tmp/prt-c-update064-r32-price-statistics-preparation-20261103`。候補peerは固定ea54をread-onlyで参照し、既存SOURCE/raw/契約/grant/ledger/marker/STOPを変更しない。

Aの可視job（PID1501576、Python、当時elapsed2h04m55s、CPU1605%、RSS2486352KiB、95threads）は**プロセス情報だけ**を読み取り確認した。A liveworktree/output/cache/lock/markerのアクセス・変更、job停止/凍結/再実行/監視変更は0。固定Git object62faと公開codeを参照しただけで、Aの現在の統計結果は利用していない。元A source/契約を064用に置換せず、A branchへpushしない。

B検証時affinity32 CPU、MemAvailable約50GiB、memory pressure0、disk空き約217GB。Bの小規模人工/mockを逐次実行し、process-local BLAS1/task-local MPL cache/tempを使った。大きな人工native32校正も今回追加していない。未来の科学実行では実際の空き資源/A jobを再確認し、B自身の投入時刻・資源を調整する。他jobやOS/共有venv/cacheを変更せず、CPU/RAM/worker/wallに恣意的上限は置かない。

正式共通運用指示2026-10-25版§8.1/8.2/8.4/10.1/10.3はroot側現物で確認した。AGENTS/PROJECT_MAPの意味付けに従い、IDEのbranch-cleanup選択を今回の作業指示として使わない。共有索引・Evidence Map・validation manifestは依頼どおり統合担当へ引き渡し、今回変更しない。詳細科学レビュー・新規性・研究方向の正式判断へ拡張していない。

## 10. 技術判断・未承認事項・次の担当

**技術的GO**：update064/r32の同J/U強D分母、独立N256/95%近似t/1%精度提案、native/full512、2,374-call範囲、source/input、fresh/retry/STOPを固定でき、人工検証・passiveがPASSした。未承認の科学契約は実行許可があれば同SOURCEで一度実行できる状態である。

未資格は、実際の価格母平均の符号、今回Nでの精度達成、近似被覆率、真のsupport/希少tail、理想lawへの移送、全law合成/実装law信号誤差、actual CA、一般的PR優位、最適r、目的関数だけの因果効果。これらを技術PASSから推定しない。

**ユーザー／研究C**は、Aの結果とBの提案N256・confidence95%・1%精度・skew/influence政策・2,374-call契約を確認し、採択と科学実行可否を判断する。別設計を望む場合は実行前にSOURCE/契約を固定し直す。**Codex-B**は明示実行許可がある場合だけfreshに一度実行し、近似統計の結論/未資格/partialを保全・公開してSTOPする。A完了を自動的なB実行許可にしない。**今回は公開照合後mandatory STOP。**

[A]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/62fa4c0144b9e36f9d96b7dff67628051946632d/docs/research/research_c_r32_price_statistics_preparation_20261103.md
[B]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/cca6b487c8e1ed3362f8825e95c937ebbcc87755/docs/research/research_c_update064_r32_downstream_execution_20261103.md
[NIST]: https://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm
[SCIPY]: https://docs.scipy.org/doc/scipy-1.14.1/reference/generated/scipy.stats.t.html
[NUMPY]: https://numpy.org/doc/1.26/reference/random/parallel.html
