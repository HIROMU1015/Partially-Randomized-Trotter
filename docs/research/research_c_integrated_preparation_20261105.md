# 研究C：頑健性・公平retuning・条件外移送の統合検証準備

2026-11-05 JST。**技術準備のみ。科学実行0、重要GPTレビュー0。契約案は未承認。公開照合後MANDATORY STOP。**

中心の問いは「PR-aware DFの条件付き総資源の追加利益が、初期値・正則化・有限最適化予算・公平な実行政策選択・未使用の物理条件で維持されるか」。新しい最小λ、新surrogate、全law保証の完成、一般的優位・新規性の確定を今回の到達条件にしない。

正本は[未承認の統合科学契約](research_c_integrated_contract_unapproved_20261105.json)。実装・技術検証は[頑健性](research_c_integrated_robustness_preparation_20261105.md)、[retuning/下流](research_c_integrated_retuning_preparation_20261105.md)、[1.15 Å入力/移送](research_c_integrated_transfer_preparation_20261105.md)を参照。固定コード版・今回の変更ファイル・契約/保存inputのbytesは[新SOURCE](../../artifacts/research_c_integrated/2026-11-05/technical/SOURCE.json)で接続する。

## 出発点と資料の確認

`PROJECT_MAP.md`、研究概要・現状、初期PR-aware設計、10月31日Evidence Synthesis、11月3日Evidence Map、結果統一索引を読んだ。共通運用指示は作業rootにある `research_project_instructions_unified_2026-10-25.md` の§8.1/8.4/10/10.1/10.3を適用した。公開取得可能性を仮定せず、実際にアクセスできたlocal原本を使用した。

収集版main `546bca7ed2cece59b714b986a264d7b0b0871a01` は、本文やSOURCE退避版を集めた閲覧入口であり、今回必要な一部moduleはroot実行pathにない。原作業領域には多数の既存dirty/untrackedがあるため触らず、独立branch/worktreeで準備した。A/B/Cも独立worktreeで調査・人工検証し、統合側で必要な差分だけ取り込んだ。

実装基点にはユーザー指定の結果収録版 `beb8de5cd8b71f80bb69556f92299435ff431c3e` を採用した。これは過去の科学SOURCEの改称ではない。最適化SOURCE `70a98adfb701a209db2c65ea60fb789ff62b1fe7`、下流SOURCE `4df8eca29c59ffd55e2dd898f05da16bb246ddbe`、旧移送adapter `f2a9baea08eeeeb2af7b4a87445acdc61932c911` を区別する。旧移送moduleを同原本から取り込み、task-local引数の転送だけ拡張した。GitHub connectorで固定optimizer原本、移送原本、共通ideal baseline報告の公開本文とlocal Git blobの一致を確認した。

[既存PR/ENTRY報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/beb8de5cd8b71f80bb69556f92299435ff431c3e/docs/research/research_c_near_practical_r32_downstream_execution_20261103.md)の約50.6%差は、直鎖H4・1.00 Å・STO-3G、number-CDF8/元順LD2、Nα=Nβ=2/sector36、partial S2/s1/r32/K0、δ=h=.1、m0..12、own-DF ground oracle・状態準備除外、P0/serial条件における16更新の有限endpoint比較として引用する。新規に再算定していない。両armのU一次未達、有限native価格、全law・actual CA未資格という制限を保持する。

別系列のupdate064/r32の約4.48%対同J/U強D差は、別の候補・ground・shots・分母・独立256校正である。前者の約50.6%差と一つの候補の結果にしない。

## 一体的な実験設計

| 要因 | 固定した候補 | 判断できること・限界 |
|---|---|---|
| 初期値 | C0、P1、P2の3点。P1/P2は固定PCG64の2方向、全frame合計skew Frobenius .05 rad、U右乗算後に中立tensor-only restore各1回 | 単一起点の依存性。初期値分布に関する統計的一般性はない |
| 正則化 | 初期penalty寄与 .03/.1/.3。各arm・paired startの初期massで正規化 | 同じ数値の係数を無条件に公平としない。初期寄与を合わせても曲率・gradientは同じでない |
| 有限最適化 | 同一64更新軌道の16prefixと終端。共通512trial/513 U点 | 4倍の更新機会と実計算負担の依存性。旧16/128実験の再現ではない |
| 実行政策 | 両armともr16/32/64、s1/K0/h=.1/LD2/既存law/PF/compiler固定 | 共通r32と公平retuningを別estimandとして比較。集合外最適性はない |
| 移送 | H4・1.15 Å。method別strengthを開発結果の登録集約で凍結 | recipe移送の一条件。全研究でblindな距離、一般性の証明ではない |

開発は18trajectoryから最大36endpoint、移送は凍結したmethod別strengthによる6trajectoryから最大12endpoint。tensor/sector誤差・guard、exact λ、目的のtensor/penalty成分、J/U一次残差、受理/棄却/停止、nfev/njev/nhev・VG/HVP・inner solve・CPU/wallを保存する。自己目的値をarm間で直接順位付けしない。同更新数を同CPU時間・同最適化精度と称さず、共通の実work指標も併記する。高終端が不適格でも既に適格な16prefixを保存する。

物理DF予算はsector operator norm＋既存engineering guard≤.00019875 Ha（CA=.00159 Ha）を維持する。元gによるκ・核scalar一回・signed J・identity位相・元frame順を保持する。未収束であることと、独立数値correctness/物理DF適格性を分ける。

## 選択と最終評価

全36開発endpointに同じ3rの探索機会を与え、各candidate/rでselection native32件を取得する。各start/budget/armの共通r32は3strengthから、retunedは3strength×3rから、適格なselection総RZ最小を選ぶ。この二つでDF候補自体が異なり得るため、最終評価は最大24(candidate,r)ペアとなる。tieはstrength昇順、次にr32/r16/r64。

移送strengthは、開発update64における各startの最小r費用を3startでmedian集約し、methodごとに最小のstrengthを選ぶ。両arm・全3startに適格rがあるstrengthだけを集約対象にする。結果が欠けたstartを都合よく除いて成功しやすいrecipeにしない。共通の完全集合がなければ移送recipe未確定とし、targetを開かない。

選択を保存してから独立final64件（既存native32を2batch）を取得する。selection/旧cohortを混ぜず、結果後のN追加・候補追加・外れ値除外をしない。固定r32とretunedのinput/candidate/rが同一のときだけfinalを共有し、独立反復とは数えない。各候補の全13roundを使うW、shots、kernel平均/SE、shellを分離し、`Ctotal=W mean_kernel+Sshell`で全6指標を保存する。rや校正Nを重複乗算しない。

主推論はC0・update64の共通r32/retuned×2geometry、計4比較。family α=.05、各98.75%の近似Welch差区間を用い、既存5%半幅/ENTRY・skew/delete-one診断を維持する。RPE failure=.01とは別の統計政策である。残りの初期値/予算と全strength gridは記述的頑健性として保持し、全cell同時優位としない。未選択strengthの価格はselection32の記述的評価であり、独立final64を取得済みとは表現しない。

有限bit実装lawの価格proxy・近似IIDモデル内の判定である。価格CIはvalidated coverage/全law保証でない。Evidence laneの未知synthesis/law誤差はnullのまま、実CA達成・一般的優位・全資源支配へ格上げしない。

## 1.15 Åへ移すもの

旧移送準備のSystem/objective_specs/recipe_arm/PhysicalContext/handoffを再利用した。元h/g/scalar生成は既存PySCF RHF/ao2mo本体へ新metadataだけ渡す。STO-3G、canonical RHF、spin0、Nα=Nβ=2、rank8/LD2/sector36、元g補正κを固定する。

研究C用の1.15 Å packetと中立CDF8は未生成である。新SCF/積分、explicit Cholesky XDF8からのneutral fit、二摂動restoreを将来の科学scopeに明示した。1.00 Åの数値J/U・f*・λ・ground・価格を流用しない。C0自身のgamma0 restoreにはraw XDF8の正のtensor objectiveを初期scaleとして用いる。C0完成後、外側PR/ENTRYとP1/P2 restore用f*はC0の正のtensor lossから一度だけ、各目的のmassは各paired startから一度だけ固定する。両scaleを区別して保存し、未完成C0の値を初期fitに要求しない。target結果を見たstrength/seed/solver/retuning規則変更は認めない。

1.15 Åには別PF研究のrank12/LD3/sector70記録がある。今回のPR/ENTRY recipeには未使用だが、全研究から秘匿された条件とは扱わない。旧PF結果を研究Cのsector36物理入力に代用しない。rank8でDF予算を満たせるかは未検証で、失敗時にrank ladderへ自動拡張しない。

## baselineと解釈

最終選択候補ごとに同J/U全frame強決定論S2/s1を評価する（最大36候補）。Dにも候補自身のPF・ground・native・shotsを用い、PRのr不適格でもDの資格を独立に記録する。これは同J/Uの強い対照であり、全決定論政策の最適解ではない。

保存CDF8/XDF8は[共通ideal結果39ee9705](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/39ee9705b7a459c40c205c542d660b5d2ab90d33/docs/research/research_c_xdf_cdf_common_ideal_lane_execution_20261103.md)を、元入力来歴・精度・compiler・会計の対応を確認して1.00 Åの固定task基準に再利用できる。別DF/別最適化予算であり、同J/U対照や1.15 Åの費用ではない。旧数値を今回再集計・再ランキング・再統計評価していない。

| 取得結果 | 後の研究判断へ渡す解釈 |
|---|---|
| 初期値/強度/予算と公平retuning後にも利益が残る | 登録有限recipeの頑健な条件付き利益を支持。一般的優位・新規性の採択は別 |
| λ改善のみ、総RZ利益は消失 | PF/RTE・shots/W・kernel単価のどこで失ったかを同じcandidate identityで切り分ける |
| retuning後にENTRYとの差が縮む | 固定rの利益と目的関数の追加利益を分離する |
| 16/64で順位が反転 | 古典最適化予算への依存性として報告。都合のよいbudgetだけを選ばない |
| 1.15 Åで利益が再現しない | recipeの一条件での移送不支持。一般的NO-GOにはしない |
| DF/round/統計精度等が未資格 | 未解決として保存し、欠測や未知誤差を0にしない |

これらは事前の解釈区分であり、今回の技術準備で研究方針・論文化GOを採択していない。

## 依存、call数、資源

| 段階 | 最大logical units |
|---|---:|
| 開発入力/初期化/共通setup | 27 |
| 開発18trajectory | 57,834 |
| 開発selection/final/同J/U D/統計 | 63,896 |
| 移送recipe凍結 | 1 |
| target新入力/初期化/共通setup | 26 |
| target6trajectory | 19,278 |
| target下流 | 31,772 |
| 統合集計 | 1 |
| **合計** | **172,835** |

1trajectoryは3inner＋2precision＋1admissionを513点、独立terminal/physical各65、driver1、二checkpoint診断/保存各2、計3213。下流はcandidate48×1268＋final pair48×612＋D36×141＋final統計48×6＋両geometry共通等64＝95,668。最大native標本はselection4,608＋final3,072＝7,680、強D36kernel。各logical unit内部の最適化反復・VG/HVPは別実測で、量子shotsの実績とは異なる。未使用枠を別条件やretryへ流用しない。

同一候補のr32でPF/exact-Hを作り、r16/64へ共有する依存は必要である。選択前にfinalを開かず、development recipe凍結前にtargetを生成しない。これらの依存に中間の再承認は不要。correctness破損は影響する依存を止め、物理/round未資格のセルはnullとして残し、独立した登録セルは継続できる。

既存のpaired16更新はwall約93秒、native64＋D2の下流は約930秒（過去の測定値の引用）。新gridはnativeだけでも数十時間級、最適化とneutral fitは追加であり、時間・speedupを保証しない。CPU/RAM/worker/wallの人工capを置かない。独立候補のprocess並列、各候補内round/native32並列、既存cache/ベクトル化を使い、逐次line search/refinementとQiskit一時global-state mutexを保つ。

Passive確認時はCPU affinity32、空きRAM約50 GiB、空きdisk約21 GiB。GPUはnvidia-smiがdriverへ接続できず使用可能とは扱わない。fit/eval既存venvの計17package pinsはmetadataで一致した。新moduleの実分子性能は未測定。既存raw保存規模から新gridは数十GiBになり得るため、現ディスクの容量で完走できるとは認定しない。実行直前に実quota/空きCPU/RAM/GPU/disk/pressureを再確認し、必要なら対象jobの保存先を十分な容量のuser領域へ決める。旧証拠削除・他job停止・共有設定変更で容量を確保しない。

## 最小差分、検証、保全

新しいのは、paired start/チェックポイント、候補bindingとr dispatch、既存会計/統計の固定N拡張、1.15 Å input dispatch、selection/受渡し、passive preflightである。optimizer、Hamiltonian再構成、native compiler、shots式、資源会計は再実装していない。旧APIのdefault16/128、N32/95%は保持した。

科学実行の新CLIは追加せず、候補ごとのcallable adapterを用意した。`scripts/prepare_research_c_integrated.py` はbytes/JSON/保存先のpassive検査のみで、`--execute`を持たない。将来の許可範囲は同じ契約に従ってこれらのAPIを組み合わせるもので、旧hardcoded run/consumed grant/STOP実験を再開するものではない。実H4でのend-to-end適格性は未実行であり、人工PASSだけで成立とは扱わない。

実施したテスト一覧・件数は[技術検証記録](../../artifacts/research_c_integrated/2026-11-05/technical/verification.json)と同directoryのlogsに固定する。人工配列、mock optimizer/PySCF/native、既存数値APIに渡す人工64行、独立processのpassive import、実ffsim接続をoptimizer反復0で試験した。保存実H4のnpz decode・新科学fit・PF/shots/native・実H4統計は0。ローカルテストでありCI・外部科学再現・科学GOではない。

旧SOURCE/input/raw/契約/grant/ledger/marker/STOPは元のまま。今回の科学output二箇所は未作成、retry/resume0。変更は独立新branchの新SOURCEと必要な最小API拡張・文書・索引・技術記録だけで、元dirty worktreeは保存した。研究結果・正式研究判断・evidence statusを変えていないため、科学manifestや研究概要・過去研究ノートの一律更新は行わない。

## 次にユーザーが判断すること

ユーザーはこの候補集合・比較/移送政策・最大172,835 logical units・fresh output・失敗/未資格/STOPを含む**新しい統合科学scopeを実行するか**を明示的に判断する。承認証明書、原文保存、新grantの発行は不要。科学実行とは別に重要GPTレビューを開始する許可も今回与えられていない。

次担当のCodexは、その別途明示許可があった場合だけ、SOURCE/環境/実空き資源と出力容量をpassive照合し、固定依存順で実験を実行・保存・報告する。今回の公開確認が完了した時点ではMANDATORY STOP。追加科学計算、重要レビュー、論文化GOへ自動進行しない。
