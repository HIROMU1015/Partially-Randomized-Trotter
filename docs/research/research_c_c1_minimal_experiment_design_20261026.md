# C1：次期最小実験A／Bの独立した技術設計（未採択・未実行）

2026-10-26 JST。設計起点：`5385c6ef01879ee8a70734bfc0d5ae8999dd6186`。
[比較・公平性監査](research_c_c1_h2_h4_h6_comparison_audit_20261026.md)の補助資料。

**本書は科学実行契約案の技術設計であり、実行可能なrunner／検証済み新科学SOURCEを固定した完成契約ではない。** 今回は科学call0。A／Bは別実験・別fresh output・別実行許可であり、片方の採択から他方へ進まない。SOURCEとcall案は、選択された案の薄いentryを実装・人工検証した後、実行前に固定する。新しい管理基盤や承認証明書・記録・新grantは不要。

## 案A：H4保存XDF prefixのDF予算適格性

### 問いと入力

保存P-H4／既存OpenFermion生成順のprefix1..9のうち、CA/8を満たすものはあるか。既に評価済みのrank10を再計算しない。新CDF fit・新XDF分解・別順序への再rankingは0。

| 保存資料 | 固定identity／用途 |
|---|---|
| 元P-H4・λ/G/κ/c | commit `86b8b002704aced8ca239414080f53793de81658`、`artifacts/non_df_h4_pilot/2026-10-23/run01/input_run/input_generation/data/snapshot.npz`、SHA256 `8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50` |
| 入力生成SOURCE | `0c329b30a81accb35c212be8defa66fee0dd0f12`のh4_molecular_inputs.py。rank10・eigh重み順・hermitizationの来歴を維持 |
| 元physical sector行列 | 起点commitの`artifacts/research_c_c1_stage1_eval_only_science/2026-10-25/run01/operators/physical_sector.npz` |
| 既存rank10／CDF8のscalar値 | 同runの`result.json` recordsのsector_DF_XDF／sector_DF_CDF。rank10のnorm・allowanceとStage3の予算判定を引用 |
| 再利用評価SOURCE | 起点commitの`c1_stage1.saved_xdf_sector`、`legacy_df_module`、`operation_lengths`、`allowance`、既存PhysicalSector／df_linear_operator |

P-H4は公開Git objectからbytesだけを取得可能なことを今回確認した。数値読込は将来の許可後のみ。Stage1保存行列のbit／sector順序と同じ`bit_reverse_indices`／sector indicesを引き継ぐ。元inputと保存physical行列のSOURCE／hash／順序を固定し、無条件に別H4行列と混ぜない。

### 評価順序と判定

1. 新entry／SOURCE／入力hash／環境をpassive preflightし、fresh `artifacts/research_c_c1_h4_xdf_prefix_budget/<execution-date>/run01`を確保する。旧runは起動しない。
2. 保存inputとphysical sectorを読み込み、schema、finite、Nα=Nβ2／sector36、κ・scalar・元順序を確認する。κの元g補正は保持し、prefixごとに再補正しない。
3. rank10の保存norm／allowanceをanchorとして引用する。新しいrank10 normやCDF normは求めない。
4. r=1..9の全prefixを登録順に評価する。`saved_xdf_sector(one,λ[:r],G[:r],c,...)`と既存sector matvecを再利用し、保存physical行列との差のnorm2と固定allowanceを取得する。最初の予算PASSで残りを省略せず、9行を揃える。rank単調性を仮定しない。
5. `norm_r+allowance_r ≤ 0.00019875 Ha`を数値pilotの適格性とする。全9結果と保存rank10に対し、保存順でのfirst-passingを報告する。費用最適rankとは呼ばない。
6. 結果・partial evidence・call・STOPを保存、公開照合してmandatory STOP。

allowanceはH4の`binary64_operation_scale_guard8_v1`を再利用する。`operation_lengths(n=4,d=36,rank=r)`のphysical／xdf長と、既存式のphysical scale `|c|+Σ|spin_h|+Σ|spin_g/2|`、XDF scale `|c|+Σ|one|+Σ|λ|*(Σ|G|)^2`を使い、`allowance(bp,bx,mphysical,mxdf,36,Hphysical−Hxdf)`で算定する。これらのscale算定は将来の科学call内の処理であり、今回未実行。既存rank10 allowanceを低rankへそのまま転用しない。κ照合・Hermiticity等は既存1e-10／rtol0のcorrectness条件を事前固定し、必要な人工正常／順序不一致・非有限テストを実装時に行う。工程guardは厳密丸めenclosureではない。

### 科学call案：最大356 logical units

| call | 最大 | 定義 |
|---|---:|---|
| saved_bundle_load | 1 | 固定P-H4、physical sector、保存scalar／来歴の読込 |
| sector_context | 1 | 既存PhysicalSector・同じbit／basis順序 |
| input_kappa_scalar_identity | 1 | 元g由来κ・核scalar・schema／finite確認 |
| physical_reference_attachment | 1 | 保存36行列を再利用し、元inputのformation scale・来歴を接続。新operator生成なし |
| xdf_prefix_model | 9 | 元順prefixのDFHamiltonian／linear operator作成 |
| xdf_matvec | 324 | 各prefix36基底action。既存helperの計上を保持 |
| sector_error_norm | 9 | 各差行列のnorm2。内部SVDを追加探索としない |
| numerical_allowance_budget | 9 | scale・guard・Hermiticity／finite・CA/8判定を含む1prefixの複合評価 |
| prefix_selection_report | 1 | 9行と保存rank10からfirst-passing報告。追加normなし |
| **合計** | **356** | 新norm9、basis action324、新fit／分解／回路／PF／shots／費用0 |

SOURCE同一性・input／κ／sector順序・referenceのcorrectness不一致では依存処理をSTOP。あるprefixのDF予算NOT_METはcorrectness FAILではなく、残りの登録prefixを続ける。全1..9 NOT_METも有用な正常結果である。retry／resume0、rank0／11以上・再ordering・新分解・別分子へ拡張しない。

計算規模は36×36行列と324 matvec。既存Python sector backend・immutable inputを共有し、利用可能資源に応じてprefixを並列化する。新しいfull256／controlled512回路や巨大行列は不要。実装前の固定秒数予測やspeedup主張はしない。CPU／RAM／worker／wallの人工上限は設けず、実際の空き資源と非干渉で構成する。

### 結果後の判断

低prefix適格があれば、ユーザー／研究CがH4のより強いbaselineとして採用し、PF・ground／alias・native・合成・固定shots margin／費用を再評価する別実験を設計するか判断する。旧XDF10の52round／shots／QPYを低ranktargetへ流用しない。全prefix不適格なら保存順rank10の位置付けが明確になる。別XDF方式との優劣や、旧27%の一般的保証は得られない。

## 案B：H6 rank13・修正codecによる最小1fit

### 保存rank13の状態

[旧保存CDF13](../../artifacts/research_c_c1_h2_h6_science/2026-10-25/H6-run01/CDF_rank_13.json)はsector norm `0.00081132667707527 Ha`、guard `1.131698121763783e-6 Ha`。[旧XDF prefix13](../../artifacts/research_c_c1_h2_h6_science/2026-10-25/H6-run01/XDF_rank_13.json)はnorm `0.0004632769171816072 Ha`、guard `1.081524262161987e-6 Ha`で、既存選定では両方NOT_MET。

[旧optimizer](../../artifacts/research_c_c1_h2_h6_science/2026-10-25/H6-run01/CDF_13.optimizer.json)はsuccess=true、status0、fun `2.1066494419957358e-7`、nit349／nfev373、relative reduction停止。生の最終x/jac・njevはその旧workerで保存していない。既存の終了J/Uとscalarを保全する。

H6の旧codec非保持が実測されたのはrank14／18である。rank13は同じ旧経路だが、その初期変換でどれだけ悪化したかは未測定。旧CDF13を「正しい初期化から最適化しても予算に届かない証拠」や「修正すれば成功する証拠」と扱わない。

### 使用する入力・SOURCE

| 資料 | 固定identity／用途 |
|---|---|
| 元H6 | `artifacts/research_c_c1_h2_h6_science/2026-10-25/H6-run01/input.npz`、SHA256 `b82ee9864d9c76071944834ae32d3d6eea7e95959708c2e19bca011936bce447` |
| 保存XDF_full | 同runの`XDF_full.npz`、SHA256 `50623b80baa69a8fece6d0923ad4128634bfb12ba46ff1269d86bf5aa1316abd`、requested21／actual18。その元順prefix13を初期候補とする |
| 旧rank13比較 | CDF_13.npz／CDF_13.json／CDF_13.optimizer.json、CDF_rank_13.json／XDF_rank_13.json。新しい旧fit再解析は含めず保存scalarを引用 |
| 再利用SOURCE | `c1_real_orthogonal_codec.py`、`c1_ffsim_corrected_codec.py`のrank-generic fit、`c1_molecule_extension.SectorCAR/rank_assessment`、`c1_h6_corrected_fit.py`のprepare／diagnostic／decision |
| 既存成功SOURCE | rank14修正fit `33432e78659b8b3e654e39f1392328cb0ba5c487`、全資料snapshotは今回起点commit |

rank13 initializerと保存full prefix13の成分同一性は、rank14／18の過去のidentityテストでは直接確認していない。将来の限定実行では**同じ固定ffsim `_double_factorized_explicit_cholesky(g,tol=1e-8,max_vecs=13)`を非最適化で1回だけ再現**して照合し、一致後に保存prefixをcached initializerとして返す。新たな条件探索ではない。違えばSTOPし、別の初期値へ交換しない。今回はその呼出しをしない。

直鎖H6・1.00Å・STO-3G、Nα=Nβ3／sector400、number表現、元g補正κ／scalar／frame順、actual rank13固定。codec=`real_schur_column_gauge_v1`、ffsim0.0.83、L-BFGS-B、tol1e-8、options=None／callback=None／jac=True、既存JAX x64目的関数・解析gradientを維持する。追加rank、rank14／18fit、best-of、再最適化は計画しない。

### rank14契約との必須差分

旧`c1_h6_corrected_fit.preflight/run`はrank14を固定し、initial_budget_failでfit前STOPする。**そのrunner／契約を変更せず再起動してはいけない。** 保存XDF13が予算外なので、そのままではrank13の圧縮可能性を検証できない。

新rank13案は、codecのJ／projector／tensor／sector保持と有限性を必須gateにし、開始DF誤差は予算外も含めて診断記録する。**開始予算PASSをfit前提としないことは新しい科学条件であり、ユーザーの採択が必要**。終了時は同じCA/8と`sector_operation_scale_guard8_proposal_v1`で判断する。これはrank14契約のSTOP解除ではない。

既存codec component max1e-10、tensor F≤1e-10×max(1,||g_before||F)、objective照合1e-12＋1e-8×|expected|、qualified x0のexact一致を維持する。optimizer success、tensor残差の増減、sector誤差の増減、最終DF予算PASSを別々に報告する。開始点が悪くても、結果後に目的関数・options・DF予算を変更しない。

### 科学call案：最大21 logical units

| 群・call | 最大 | 範囲 |
|---|---:|---|
| prepare：保存load／保存初期schema・順序identity | 2 | 同じg、XDF_full prefix13、旧scalarの固定来歴 |
| prepare：rank13 initializer再現・同一性 | 1 | 固定非最適化Cholesky1回。保存prefixと成分・符号・rankを照合 |
| prepare：修正codec／tensor assessment2 | 3 | pack/unpack、J／projector／tensor fidelity、変換前・後tensor残差 |
| fit：load／実fitter／initial value-gradient／final tensor／final value-gradient／objective照合 | 6 | 最大1fit。内部iterationsはその1unit、前後diagnosticは別計上。x/jac/status/message/nit/nfev/njev保存 |
| sector：snapshots load2／context／physical | 4 | SectorCAR400とphysicalを一度だけ作り共有 |
| sector：before／restored／final assessment | 3 | 同一物理sector、元κ・scalar・guard |
| sector：開始codec同値gate／終了decision | 2 | 開始予算は診断、終了予算とoptimizer成功は独立判断 |
| **合計** | **21** | 新最適化fit最大1、非最適化initializer1、sector assessment3 |

fresh output案：`artifacts/research_c_c1_h6_rank13_corrected_fit/<execution-date>/run01`。旧fitと旧grant・call残枠を流用しない。新しい薄いrank13 entry・人工mockは将来の採択後に準備し、実際の科学SOURCE／input hash／21-call定義を固定してから別途明示許可を受ける。今回の資料に実行可能性や承認済みを示すmetadataは付けない。

入力／SOURCE不一致、codec非保持、非有限値、objective不一致はfit前または直ちにSTOPし、旧codecへfallbackしない。optimizer例外はpartial evidenceを保全してSTOP。有限のunsuccessful returnには登録された終了tensor／sector診断だけを実施する。終了DF予算NOT_METはその単一条件の未達としてSTOPし、初期値採用・rank変更・retry／resume・追加fitは0。optimizer success=trueでも予算外なら適格としない。予算PASSでもPF・native／CA／shots／総費用へ進まない。

計算規模は6軌道・13 frame・sector400、準備2状態は独立に並列化可能、物理行列／sparse CARを共有する。4096²／8192²の密unitaryを作らない。fitは既存JAX x64のまま、実資源に合わせたlocal BLAS/thread設定を使う。旧rank13の349iterationと修正rank14の1iterationのどちらからも、修正rank13の時間・収束を予測しない。

### 結果後の判断

最終DF予算PASSなら、ユーザー／研究Cが返却CDF13を新しい候補として採用するか判断する。その後もCDF13自身の一般K／ground／alias、13round PF、precision native／P0、合成誤差、shots／serial費用の独立資格が必要。CDF14の保存52セル・QPY・shots・費用は新targetの証拠を代替しない。XDF14の変更しない保存結果は、入力・target・数値条件を照合した上で対照として再利用可能。

NOT_METなら今回の1fitの情報を保存してSTOPする。全rank13、他初期値、別optimizerの不可能性を主張しない。どちらの結果でも自動探索・自動次段階は行わない。

## 推奨順序と担当

最初は**案Aのみ**を推奨する。ユーザー／研究Cがこの技術提案の採否と科学実行範囲を判断する。Codexは選ばれた案について最小entry・人工検証・未承認契約の固定までを実施し、固定後の明示科学実行許可に従う。重要な科学的解釈の採択や詳細レビューは別の依頼・許可として扱う。

今回の公開照合後は**mandatory STOP**。本書のcall案やfresh pathを使用して未許可の計算を起動しない。
