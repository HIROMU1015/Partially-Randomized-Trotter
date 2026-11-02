# Track A AX-1b 保存値解析・結果レビュー（2026-10-09）

本実行は `AX1B_COMPLETE_WITH_DECLARED_NA`、読み取り専用検証は `READ_ONLY_VALIDATION_PASS` で完了した。launchは1回、retry/resumeは0回。以下は登録された出力の転記・表示であり、追加fit、別条件の性能再評価、研究契約の変更を行っていない。`mandatory_stop=true`、`next_stage_authorized=false` を維持する。

## A. Launchと検証

- 実行時HEAD / authorization source: `fc297cd9ab840018c4f35b2764d0b8be07c57285`。
- 実装・170 synthetic testsのprovenance: `fe312600191cbed33fcf9626d2cde8dbb2270820`。両commit間の対象source/test/bundle bytesは同一である。環境の完全一致を確認して既存170 passed / 0 failed / 0 skipped auditを再利用した。今回の独立CI証拠ではない。
- branch: `pr2-v4-s2-parallelization-20260928`、origin: `git@github.com:HIROMU1015/Partially-Randomized-Trotter.git`。
- 最新identity compatibility manifestのcanonical SHA-256: `cdc33b9da8c9e714274fd5956c0d23dbc258fe05147784e75641e55a9f08896c`（raw: `ca9f6db4e901de4a919f5dbc7bbbb9d05b01c8d89e4f5368dc269710788e07e5`）。古いSTOP/draftを保存した。
- 条件付きレビュー判定 `APPROVE_AX1B_EXECUTION_CONDITIONAL_ON_FINAL_BINDING` と明示的ユーザーlaunch指示を保持し、15項目すべての最終gateが成立した後に限り実行用 `APPROVE_AX1B_EXECUTION` を記録した。AX-1b解析のみtrue、新しい科学生成はfalse。
- [authorization](execution_authorization_v1.json) canonical SHA-256: `ca2075814c016757608d280c10fb78f980ccd8ccf2826d600f4d7638ff00fabf`。起動前は `/tmp` に置いてHEADを保持し、ここには実行後に同一bytesをコピーした。内部の絶対パスは当時の記録であり、書き換えていない。対応表は [execution_evidence_manifest_v1.json](execution_evidence_manifest_v1.json) にある。
- environment fingerprint: `314b059957876f54e3e82d97d0368ccd3ef9157aec335bb04d057123c2fc10f5`。Python 3.11.1 final、NumPy 1.26.4、SciPy 1.14.1、pytest 9.0.3。executable/package origin/metadata/NNLS hashを照合した。
- 上限: CPU 1 core、affinity `[0]`、1 process、BLAS 1 thread、address space 8 GiB、wall 300 s、output 512 MiB、GPU禁止。共有サーバーの専有予約ではなくプロセス上限である。直前のCPU/メモリ/disk/scheduler必要性を確認し、固定runnerの制限を適用した。
- 古典計算実測: runner wall 8.737193 s、authorization込み 8.866924 s、max RSS 310702080 bytes（約296.31 MiB）。登録12出力合計 10325366 bytes。量子回路資源とは別の計測である。
- unchanged runnerを観察用wrapper経由で1回実行した。関数やgateは置換せず、実行段階・関数entry・成功入力read・RSSのみを記録した。execute/analyze各1、45入力read、64 axis/model fitすべてFIT_OK、新しい科学生成用importの試行0。wrapperのhashはlaunch receiptに固定されている。
- 45入力hash/schema、M1 210 / PM-1 8 / M2 5 membership、canonical compiler identity、各PM-1候補の3件B0 anchor、train-only scalingと全fold training membership、固定KKT基準、12出力hash/schema、authorization/source/terminal bindingを照合した。
- 197件の有限分布のorders/probabilities/weightsを保存値と固定toleranceで照合した。解析上界のslackは固定tolerance内で成立し、PM-2のshot/eligibility/RZ-work 67,346行を再現した。出力後の検証ではnormalizationやfitを再計算していない。

## B. モデル性能

評価範囲は線形H4、STO-3G、DF rank12、二次PF、canonical finite-RTE、T=0.8。M1は1.00 Å、prefix `L_D=0,3,6,9,12`、q=1/2/4/8、delta=0.8/0.4/0.2/0.1、random K=2/4、保存された(q,r,K)・R=qr候補のみ。PM-1は同geometryのB0 prefix4/5追加8候補。M2は1.30 Åの既観測固定5構成であり、新しいheld-out geometry探索ではない。

目的変数は実コンパイルfull Hadamard wrapperのone-shot RZ（cosine/sine、ancilla/control/phase/measurement込み、state preparation除外）。保存compilerはQiskit1.3.0、basis rz/sx/x/cx、optimization1、seed17、couplingなし。これは特定compilerでのnative資源評価である。学習にはM1 210候補だけを使い、PM-1/M2はfitに入れていない。

主表はcosine/sineをまとめた保存summaryの値。MAREは平均絶対相対誤差、過小率は予測費用がreference費用より小さい割合、過小>10%率はその過小幅が10%を超える割合。signed relative errorとlog errorの各候補値、method/prefix/q/R/K別の保存診断は [cost_metrics.csv](../run_v1/cost_metrics.csv) にある。

| 診断 | モデル | MARE | 中央絶対相対誤差 | 最大絶対相対誤差 | 過小率 | 過小>10%率 | Coverage |
|---|---|---:|---:|---:|---:|---:|---:|
| IN_SAMPLE_DEVELOPMENT | SINGLE | 72.1751% | 41.7081% | 247.0857% | 60.4762% | 53.3333% | 100.0000% |
| OBSERVED_GEOMETRY_DIAGNOSTIC | SINGLE | 58.6569% | 44.1957% | 183.7326% | 60.0000% | 60.0000% | 100.0000% |
| OBSERVED_PREFIX_DIAGNOSTIC | SINGLE | 43.1147% | 43.2888% | 47.0105% | 100.0000% | 100.0000% | 100.0000% |
| IN_SAMPLE_DEVELOPMENT | FEW | 8.6875% | 2.9091% | 80.2503% | 51.9048% | 4.2857% | 100.0000% |
| OBSERVED_GEOMETRY_DIAGNOSTIC | FEW | 5.7600% | 2.2401% | 19.8906% | 20.0000% | 20.0000% | 100.0000% |
| OBSERVED_PREFIX_DIAGNOSTIC | FEW | 3.4205% | 3.4999% | 5.4111% | 0.0000% | 0.0000% | 100.0000% |
| POOLED_OUT_OF_FOLD_COST | SINGLE | 72.7665% | 41.2006% | 247.3816% | 58.5714% | 53.3333% | 100.0000% |
| POOLED_OUT_OF_FOLD_COST | FEW | 8.9096% | 2.7954% | 86.1507% | 52.3810% | 4.7619% | 100.0000% |

固定complexity gateは **PASS**、採用は `PRED_BASE_FEW_PARAM`。gateに用いた4 q-fold均等平均はSINGLE **72.7447%**、FEW **8.9043%**。全foldで改善しcoverageは一致した。主表のpooled OOF MARE（72.7665% / 8.9096%）は候補数で重みが変わるためgate平均と異なる。pooled OOFは異なるfold学習に基づくcross-fitted内部診断であり、単一凍結モデルの独立予測性能ではない。

各foldの保存summary（cosine/sine同値なので片軸表示）：

| Fold | モデル | MARE | 最大絶対相対誤差 | 過小>10%率 | Coverage |
|---|---|---:|---:|---:|---:|
| leave_one_method_out:B0 | SINGLE | 45.6973% | 53.3111% | 100.0000% | 100.0000% |
| leave_one_method_out:B1 | SINGLE | 52.9825% | 53.8798% | 100.0000% | 100.0000% |
| leave_one_method_out:B2 | SINGLE | 48.0191% | 69.2645% | 93.7931% | 100.0000% |
| leave_one_method_out:B3 | SINGLE | 249.5685% | 302.4167% | 0.0000% | 100.0000% |
| leave_one_prefix_out:0 | SINGLE | 249.5685% | 302.4167% | 0.0000% | 100.0000% |
| leave_one_prefix_out:12 | SINGLE | 52.9825% | 53.8798% | 100.0000% | 100.0000% |
| leave_one_prefix_out:3 | SINGLE | 29.2389% | 118.3630% | 52.8302% | 100.0000% |
| leave_one_prefix_out:6 | SINGLE | 33.6974% | 50.4441% | 69.2308% | 100.0000% |
| leave_one_prefix_out:9 | SINGLE | 43.8837% | 61.1895% | 84.6154% | 100.0000% |
| leave_one_q_out:1 | SINGLE | 70.2077% | 247.3816% | 52.8302% | 100.0000% |
| leave_one_q_out:2 | SINGLE | 69.7273% | 220.8590% | 53.8462% | 100.0000% |
| leave_one_q_out:4 | SINGLE | 71.1748% | 211.7723% | 53.8462% | 100.0000% |
| leave_one_q_out:8 | SINGLE | 79.8689% | 241.5651% | 52.8302% | 100.0000% |
| leave_one_random_K_out:2 | SINGLE | 82.0820% | 253.9416% | 44.8980% | 100.0000% |
| leave_one_random_K_out:4 | SINGLE | 73.1243% | 241.1695% | 50.0000% | 100.0000% |
| leave_one_method_out:B0 | FEW | 3.7917% | 9.1364% | 0.0000% | 100.0000% |
| leave_one_method_out:B1 | FEW | 0.1545% | 0.2799% | 0.0000% | 100.0000% |
| leave_one_method_out:B2 | FEW | 4.6778% | 18.0216% | 0.0000% | 100.0000% |
| leave_one_method_out:B3 | FEW | 48.3696% | 135.9014% | 18.3673% | 100.0000% |
| leave_one_prefix_out:0 | FEW | 48.3696% | 135.9014% | 18.3673% | 100.0000% |
| leave_one_prefix_out:12 | FEW | 0.1545% | 0.2799% | 0.0000% | 100.0000% |
| leave_one_prefix_out:3 | FEW | 4.1242% | 13.9131% | 7.5472% | 100.0000% |
| leave_one_prefix_out:6 | FEW | 4.0368% | 12.9847% | 0.0000% | 100.0000% |
| leave_one_prefix_out:9 | FEW | 3.6601% | 9.8646% | 0.0000% | 100.0000% |
| leave_one_q_out:1 | FEW | 7.0580% | 67.6934% | 1.8868% | 100.0000% |
| leave_one_q_out:2 | FEW | 7.7759% | 55.9222% | 3.8462% | 100.0000% |
| leave_one_q_out:4 | FEW | 8.9187% | 69.3822% | 3.8462% | 100.0000% |
| leave_one_q_out:8 | FEW | 11.8646% | 86.1507% | 9.4340% | 100.0000% |
| leave_one_random_K_out:2 | FEW | 10.2406% | 82.4021% | 9.1837% | 100.0000% |
| leave_one_random_K_out:4 | FEW | 8.3991% | 74.7252% | 2.0833% | 100.0000% |

B3を除外した学習ではFEWでもMARE48.3696%、最大135.9014%が残る。prefix0とB3、prefix12とB1のholdout集合は重複しており、独立した追加証拠として数えない。全体誤差の改善からmethod外挿や全候補の誤差保証を導けない。R/q/K別の周辺診断も候補構成と混在するので、因果効果として解釈しない。

Full210 fitのSINGLE係数は343.72166538309943。FEWの係数はintercept=0、E_rand=77.86907618903217、n_det=793.2211607640618、n_fixed=213.1315997001303、q=121.25057418130754。固定列・NNLS・train-only scalingを維持し、結果を見たfeature追加やmethod別救済をしていない。これらを物理的な因果係数として解釈しない。

`PRED_BASE_ACTION_INDEX` は順位診断のみ。保存SpearmanはM1 0.8399048582154811、PM-1 1.0、M2 0.8207826816681233（各軸同値）。action indexをRZと同じ単位として扱わず、未校正indexでRZ誤差やselection regretを計算していない。`AXM1_FINITE_NORMALIZATION` は有限分布と保存normalizationの照合・解析上界slackの診断であり、one-shot RZ fitとは別の対象である。

入力は保存action/q/有限分布等のI1情報、M1費用学習のI2情報、資源選択のreference shot/eligibilityを含むI4 conditional-oracle情報。保存値の再利用は可能だが、新しい系のDF・lambda_R・basis/event情報の取得費用はUNKNOWNであり、今回の実行時間から推定していない。詳細basis/eventを要する `PRED_BASE_STRUCT_ACCOUNT` はN/A。

## C. 条件付き資源・selection

評価対象は `G_conditional = sum_a N_reference[a] C_predicted[a]`。regretはこの予測で選ばれた候補のreference RZ-workと、同じ候補集合・精度におけるreference最小値との相対差である。reference shots/eligibilityを使う **CONDITIONAL_ORACLE** である。各行は [conditional_oracle_selection.csv](../run_v1/conditional_oracle_selection.csv) の既存selection_diagnosticから転記した。

| 診断 | epsilon | モデル | 登録候補 | Eligible候補 | Conditional regret | 最小G_ref |
|---|---:|---|---:|---:|---:|---:|
| IN_SAMPLE_PLUS_OBSERVED_PREFIX | 0.05 | SINGLE | 218 | 214 | 7.4952% | 130774896.65625 |
| OBSERVED_GEOMETRY | 0.05 | SINGLE | 5 | 5 | 0.0000% | 111753794.4375 |
| CROSS_FITTED_INTERNAL_GROUP | 0.05 | SINGLE | 210 | 206 | 7.4952% | 130774896.65625 |
| IN_SAMPLE_PLUS_OBSERVED_PREFIX | 0.05 | FEW | 218 | 214 | 1.4753% | 130774896.65625 |
| OBSERVED_GEOMETRY | 0.05 | FEW | 5 | 5 | 0.3577% | 111753794.4375 |
| CROSS_FITTED_INTERNAL_GROUP | 0.05 | FEW | 210 | 206 | 1.4753% | 130774896.65625 |
| IN_SAMPLE_PLUS_OBSERVED_PREFIX | 0.01 | SINGLE | 218 | 163 | 2.9633% | 6462349562.531248 |
| OBSERVED_GEOMETRY | 0.01 | SINGLE | 5 | 5 | 0.0000% | 6814896455.25 |
| CROSS_FITTED_INTERNAL_GROUP | 0.01 | SINGLE | 210 | 163 | 2.9633% | 6462349562.531248 |
| IN_SAMPLE_PLUS_OBSERVED_PREFIX | 0.01 | FEW | 218 | 163 | 0.5631% | 6462349562.531248 |
| OBSERVED_GEOMETRY | 0.01 | FEW | 5 | 5 | 0.3510% | 6814896455.25 |
| CROSS_FITTED_INTERNAL_GROUP | 0.01 | FEW | 210 | 163 | 0.5631% | 6462349562.531248 |
| IN_SAMPLE_PLUS_OBSERVED_PREFIX | 0.005 | SINGLE | 218 | 151 | 3.5642% | 40969763267.62501 |
| OBSERVED_GEOMETRY | 0.005 | SINGLE | 5 | 1 | 0.0000% | 108267309639.5625 |
| CROSS_FITTED_INTERNAL_GROUP | 0.005 | SINGLE | 210 | 151 | 3.5642% | 40969763267.62501 |
| IN_SAMPLE_PLUS_OBSERVED_PREFIX | 0.005 | FEW | 218 | 151 | 0.8122% | 40969763267.62501 |
| OBSERVED_GEOMETRY | 0.005 | FEW | 5 | 1 | 0.0000% | 108267309639.5625 |
| CROSS_FITTED_INTERNAL_GROUP | 0.005 | FEW | 210 | 151 | 0.8122% | 40969763267.62501 |
| IN_SAMPLE_PLUS_OBSERVED_PREFIX | 0.001 | SINGLE | 218 | 101 | 0.6208% | 2071908920387.6562 |
| OBSERVED_GEOMETRY | 0.001 | SINGLE | 5 | 1 | 0.0000% | 2867495920366.5 |
| CROSS_FITTED_INTERNAL_GROUP | 0.001 | SINGLE | 210 | 101 | 0.6208% | 2071908920387.6562 |
| IN_SAMPLE_PLUS_OBSERVED_PREFIX | 0.001 | FEW | 218 | 101 | 0.0000% | 2071908920387.6562 |
| OBSERVED_GEOMETRY | 0.001 | FEW | 5 | 1 | 0.0000% | 2867495920366.5 |
| CROSS_FITTED_INTERNAL_GROUP | 0.001 | FEW | 210 | 101 | 0.0000% | 2071908920387.6562 |

218候補のIN_SAMPLE_PLUS_OBSERVED_PREFIXはfull210 fitを使う開発診断であり、pooled210 cross-fittedとは学習と支持集合が異なる。この4 anchorではregretが同値でも同じ検証ではない。q-fold別selectionは各holdout集合内だけの診断であり、pooled全体の選択と区別する。epsilon=.005のq=1、epsilon=.001のq=1/2のfoldはeligible集合が空で、両モデルとも `REF_ELIGIBLE_EMPTY`（計6診断）、regretはN/Aである。

PM-1追加8候補はepsilon=.05で8件eligible、それより厳しい3 anchorでは0件eligible。追加によってreference最小G_refは低下せず、4 anchorで選ばれた候補もPM-1ではない。これは固定anchor・固定候補集合内の結果であり、discardの一般的な価値を否定する結果ではない。

M2固定5構成ではSINGLE regretは4 anchorすべて0、FEWはepsilon=.05/.01で約0.3577%/0.3510%、.005/.001では0。後者2 anchorはeligible候補が1件だけなので、regret0を予測力の証拠としない。M2費用MAREはFEWで改善していても、selectionはSINGLEを一律には上回らない。

Reference eligibilityの三状態を維持した。今回は未確定eligible 0、予測欠測0、common-support除外候補0。各domainのfull-set、common-support、known-eligible subsetを独立の記録として保持しており、今回の同一値は欠測がないことによる。144 standalone selection診断のうち138はVALID_CONDITIONAL_ORACLE、6はREF_ELIGIBLE_EMPTYである。

Paired costはcosine/sineのcovarianceを保持した。197 random候補は各32 trajectories、26 deterministic候補は各1標本、6指標で計1,338行の保存統計。点±2SEはengineering intervalのみ。formal simultaneous CIではなく、rare-event tailとcross-candidate covarianceは未解決である。

## D. 研究上の制約と終了状態

- Operational bias predictorがないため、predicted shots、operational eligibility、operational total work、operational selection regretはN/A。詳細basis/eventがないSTRUCT_ACCOUNTもN/A。
- H4内のfit/group/pooled診断は独立移送の証拠ではない。PM-1/M2は既観測診断であり、M2は固定5構成内の結論に限定する。
- epsilon=.001は保存bias/normalizationからの条件付き会計であり、新しい高精度signal実験ではない。finite-time signal accuracyを化学的energy estimation精度へ同一視しない。
- 32 trajectoriesのrare-event寄与、cross-candidate covariance、正式な同時信頼区間は未解決。
- 特定compiler/native full wrapper費用とreference shotによる解析であり、原論文のQPE総資源と直接比較していない。PRの一般的優位、最終総資源評価、未検証系での予測保証を主張しない。
- AX-2、H6/H8、新Hamiltonian/state/signal/trajectory/compileを実施していない。モデル/feature/solver/tolerance/seed/候補の変更もしていない。次段階のGOを出していない。

## E. 成果物とGit保存

登録 [run_v1](../run_v1/) の12出力はすべて存在し、改変していない。runnerが生成した短いreport.mdもそのまま保存し、本詳細レビューを別の実行監査directoryに置いた。

Output manifest raw SHA-256: `823d07c35fd4a03ddf09ed9dbf595c208edf3bdfa433d906c6c76929f668a8e9`。11出力のhashをmanifestが保持し、manifest自身を含む12hashは [output_validation_audit.json](output_validation_audit.json) と実行証跡manifestに保持した。入力45件のhashは [input_identity_audit.json](../run_v1/input_identity_audit.json)、終端は [terminal_status.json](../run_v1/terminal_status.json) にある。

結果commitにはこの12出力と必要な実行監査だけを明示stageする。実行時source commitと後続の結果commitは別であり、結果commit SHA/push照合は最終報告で示す（commit自身のSHAをcommit内に書き込む循環は作らない）。既存21 tracked dirty、35 untracked、87保護pathは実行前記録と照合して保護し、既存dirtyをstageしない。AX-0/AX-1a契約、M1〜PM-2結果、旧STOP、原稿v0.1、Track B、未コミットの文書整理を変更しない。

**終了状態: AX1B_COMPLETE_WITH_DECLARED_NA / mandatory_stop=true / next_stage_authorized=false。結果レビュー待ちで停止する。**
