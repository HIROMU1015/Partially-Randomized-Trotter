# 研究Cと既存DF・Hamiltonian表現改善研究の重複整理

2026-11-07 JST。Codex主エージェントによる静的調査。独立新規性、論文化GO/NO-GO、研究の開始・終了の正式判断ではない。

**現在確認した範囲では、既存O1・既存DF改善系列から別立てにすべき研究Cの独立課題を特定できなかった。研究管理上は案B「現成果で一区切り」を基本とし、必要な部品だけ案C「既存O1等へ技術的接続」として使うことを推奨する。** 未解決事項がない、研究Cの成果が無効、今後の改善が不可能、という判断ではない。研究Cを「共同最適化」と改名するだけでは、既存の問いと方法の重複を解消できない。

**重要な時点差：指定されたstructural準備版では実H4 fitは未実行だが、後続原本には同じ科学SOURCEでの1 fit/21-call完了と、tail-P4のPF/RPE/native32・449-call完了がある。後者の条件付き総RZは旧PR-aware対照より約17.64%増だった。** この既存負結果を無視して、hard supportを未実行の改善案として提案し直さない。

## §1 目的・取得した固定資料

### 1.1 今回の範囲と運用

研究C、O1-M1/M2、混在辞書F1/F2、DF structural、3環境統合の固定資料・SOURCEテキストを比較し、何が同じ処理で何が異なるtaskかを整理した。新科学計算、保存数値の再解析・再集計・再ランキング、DF fit、回路計算、研究アルゴリズムの実装、重要科学レビューは行っていない。数値は既存報告の引用である。

正式原本 `/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/research_project_instructions_unified_2026-10-25.md`、`/home/AbeHiromu/projects/AGENTS.md`、worktreeの`AGENTS.md`を確認した。同じ固定mainの`PROJECT_MAP.md`・既読の研究概要全文・`VALIDATION_STATUS.md`・中央manifestと、最新Cの個別状態記録を併読した。旧概要・中央台帳の時点を最新の全研究状態としない。AI-agent delegationは0。

専用branchは `codex/research-c-existing-df-improvement-overlap-20261107`、独立worktreeは `/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/research-c-existing-df-improvement-overlap-20261107`。基点はGitHub mainと一致した `546bca7ed2cece59b714b986a264d7b0b0871a01`。新規成果物は本Markdown一件のみである。

### 1.2 固定資料・取得範囲

開始時にO1公開ref `refs/heads/codex/o1-hamiltonian-partition-preflight-20261106` をGitHub接続で実取得し、HEAD **`aeb9d1847f9d500f2389db12b384bac6087117a9`** を確認した。以下では可変branchリンクを固定commitへ置き換えて引用する。O1に後続実行がないという全historyの不存在証明はしていない。

| ID | 固定commit・資料path | 取得・再利用した範囲 |
|---|---|---|
| C-OLD | `417fa459b823509be5c2010c5836e58955088282`: `docs/research/research_c_integrated_h4_100_results_20261106.md` | 同一会話で取得・読解済みの固定原本を再利用。旧辞書の目的arm・有限recipe・公平retuning・同J/U D・資格 |
| C-Z | `3015d4fb342d4d98b58e41d57553e75539a07655`: `docs/research/research_c_h4_relocated_downstream_results_20261106.md` | 同じ固定原本を再利用。再配置後の主比較・統計診断・引継ぎ来歴 |
| C-D | `91417fb3ef75e7eec0849ad39f0649d170754664`: `docs/research/research_c_h4_strong_d_relocation_results_20261107.md` | 同じ固定原本を再利用。CDF8/XDF8強D、ideal/数値lane、最新status/manifestも関連箇所確認 |
| C-BASE | `77ded5c12db21572933653cb26758436da3d9ecd`: `docs/research/research_c_baseline_comparability_20261106.md` | 既読の固定原本§2–9を再利用。RC-DF/SCDF/UWCの外部調査・実装探索を再実施しない |
| O-R/O-E/O-M/O-P | `aeb9d1847f9d500f2389db12b384bac6087117a9`: `docs/research/o1_hamiltonian_partition_preflight_20261106/` の `REPORT.md`、`O1_EXISTING_METHODS.md`、`O1_IMPROVEMENT_MECHANISMS.md`、`O1_MINIMUM_VALIDATION_PLAN.md` | 4本文をGitHub接続で取得。M1/M2の対象task・既知法・再利用・NO-GO境界 |
| F2 | `dc7378ab34015287215bcf4ce52147e46d94ecd9`: `docs/research/pr_algorithm_improvement_phase2d_f2_resources_rerun_20261024/EXECUTION_REPORT.md` | GitHub本文取得。修正SOURCE・4arm correctness・単回価格・N未資格 |
| DS-PREP | `59e186ad5132ffde66684a781a47a2bd11a77ed4`: `docs/research/research_c_df_structural_design_20261028.md` | GitHub本文取得。準備時点のsupport仕様・人工検証・未承認契約。**実行後の状態とは区別** |
| I-R/I-G/I-U/I-V | `3571477093441ac4852166fe829f8ccc5db5ec4f`: `docs/research/pr_algorithm_improvement_three_environment_integration_20261105/` の `REPORT.md`、`IMPROVEMENT_GAP_MATRIX.md`、`DUPLICATION_AND_REUSE_MAP.md`、`REMAINING_VALIDATION_MAP.md` | GitHub本文取得。RQ-01/02/03、MD/DF/RTE資産、修正鎖・保留を重点読解。全資産143群を新しく再監査したのではない |
| DS-FIT | `e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8`: `docs/research/research_c_df_structural_support_execution_20261028.md` | 必要な後続証拠として追加取得。1 fit/21-callとDF適格性 |
| DS-PERF | 同`e8b19ff6…`: `docs/research/research_c_df_structural_pr_performance_20261028.md` | 追加取得。tail-P4の449-call・13round・native32・条件付き負結果 |
| C-NO | 同`e8b19ff6…`: `docs/research/research_c_pr_native_cost_improvement_feasibility_20261102.md` | 追加取得。過去support/rank/surrogate/共有/round等の結果と限定技術NO-GO |
| SHARED | 同`e8b19ff6…`: `docs/research/research_c_shared_basis_df_design_20261029.md` | 関連する既存の代数的同値・政策区分を参照。新しい証明・PoCを実行しない |

`e8b19ff6…`や3環境統合commitは閲覧用carrierであり、各科学SOURCEへ付け替えない。追加したDS-FIT/PERFは、指定準備版の「未実行」を現在の状態として扱わないために必要な限定追跡である。

静的SOURCEは主に、DS-PREPの `c_df_structural_support.py` / `c_df_structural_execution.py`、F2実行SOURCEの `pr_dictionary_validation/prototype.py` / `f2_resources.py` / `f2_statistics.py`、C-Z/C-Dの `c_h4_relocated_downstream.py` / `c1_stage3_connection.py` / `c_pr_aware_df.py` / `c_near_objective.py` を固定Git objectから読んだ。科学moduleをimport・実行していない。

### 1.3 状態分類の意味

`ALREADY_VERIFIED`は元の固定条件での検証記録、`KNOWN_METHOD_APPLICATION`は既知手法の適用・組合せ、`DISTINCT_MECHANISM_POSSIBLE`は比較した操作・情報・費用経路に具体差があること、`INSUFFICIENT_EVIDENCE`は未解決、`PREVIOUSLY_STOPPED`は既存の停止・保留を表す。これらは新規性や実行GOを発行するラベルではない。同じ論点でも、操作は既知、局所結果は検証済み、一般化は証拠不足という複数状態があり得る。

## §2 各研究の中心課題

| 系列 | 固定するもの | 変更するもの・問うこと |
|---|---|---|
| 研究C・旧目的比較 | 元積分、物理DF予算、CDF8/LD2、初期値群・有限更新recipe・同等tuning機会 | 一般対称signed Jとframe別直交U、PR-aware/ENTRY目的・strength。候補ごとに近似H_DFが変わり得る。各自のDF groundからの条件付き下流費用 |
| 研究C・Z再配置 | 各保存J/UとそのH_DF、元frame順LD2 | tailの線形Zをone-bodyへ移す。全Hは保持し、H_D/H_RとPF split・law・費用を変える。再fitなし |
| O1-M1 | 同じ元H、同じH_D/H_R、元Hの厳密groundアクセス、共通energy task | 同じ物理tailをS/collected P等で表す。理想補正meanを保つ範囲でλ/B/raw半径/単価を交換 |
| O1-M2 | 元H・その厳密groundアクセス・元H energy task | 実装しやすいcoreを選び、`H−H_core`を完全残差として保持。core軽量化に伴う残差費用を支払う |
| F1/F2 | 合成H、occupied state、固定D/Rと入力、finite target | 2 tail fragmentのS/P表現4arm。F1は登録correctness、F2は修正版full-wrapper correctnessと確率加重単回資源 |
| DF structural | 元g・κ/scalar、rank8/LD2、全U可変、既存目的/scale | tail JだけP4 supportへ制約。soft penalty変更ではなく実行可能集合の制限。後続でfitと資源評価まで実行済み |

O1は既知応用候補のpreflightで、分子・rank・成功条件を新しく固定した実行契約ではない。Cのown-DF oracleを元Hの厳密groundへ置き換えることも、逆の置換も、同じ無料アクセスとはみなさない。[O-E §1/5][OE]、[O-M §2–3][OM]

## §3 科学条件の対応表

研究C欄は最新H4結果、structural欄は準備と確認できた後続結果を併記する。O1に未固定の数値条件を研究Cから輸入しない。

| 比較項目 | 研究C | O1-M1 | O1-M2 | F2 | DF structural |
|---|---|---|---|---|---|
| 元Hamiltonian | H4直鎖1Å/STO-3G、元h/g/cとκの保存来歴 | 選択対象の元物理H。個別新対象は未採択 | 同左 | 4mode/2粒子の合成H。分子geometry/AO basis非該当 | 同じH4保存元g/κ/c |
| 対象state | 各候補のown-DF sector ground。prep費用除外 | 元Hの厳密groundを受け取る想定。古典vector/E₀を無料選択oracleにしない | 元Hの厳密ground。core groundではない | occupied(0,1)固定。物理H厳密groundとする資格ではない | 各DFのown-ground。後続P4で新groundを評価 |
| 近似Hの扱い | 候補H_DFを物理sector norm＋guard予算でadmit。fit残差をPR tailに戻す構成ではない | H_D/H_Rそのものを保持する辞書同値 | `H_core+(H−H_core)=H`。fit残差を捨てない | 同じH/D/R、共通finite targetの数値照合 | hard maskと初期projection/fitで近似Hが変わる。誤差は別評価 |
| DF rank・表現 | number-CDF8一般signed J/U、元順LD2。C-Dは保存CDF8/XDF8強Dも対照 | S/P/mixedの同値辞書候補。新分子rank未固定 | 既存DF/XDF/RC-DF等のcore候補、適合未確定 | 残差density-square2fragment、SS/SP/PS/PP。分子DF rank非該当 | CDF8、head full J、tail対角＋path edge、outer/inner rankを区別 |
| J/U最適化 | 旧目的比較では可変。Z下流・強D比較では保存値固定 | 物理split固定の辞書比較自体には不要 | 保存coreの再利用を先行。新fit採択なし | なし | 1 fit実施済み。既存loss/AD/codec/SciPyを使用 |
| Hamiltonian split | 元順LD2。Z再配置でhead/tail演算子も変わる | H_D/H_R固定 | core/residual選択により変更 | 固定LD1・2残差fragment。辞書変更だけ | 元順LD2固定でも新J/Uにより演算子が変わる |
| 表現辞書 | 元Z+ZZ、再配置後tail I+ZZのうち非identityをRTEへ | 同じH_RにS/P/collect等 | coreと完全残差の実装可能辞書を選ぶ必要 | frame Sとphysical P、P側global collect | 元の回転Z/ZZ、tail係数位置216→144。再配置後版は未取得 |
| One-body・scalar | 元g補正κ、核scalar一回、tail I抽出。再配置はΔκと対応scalarを追加 | 同じD/R/scalarを保持することが比較条件 | core由来補正と元Hとの差分を整合する必要 | faithful identity tail・元scalar・D/Rを固定。CのI抽出と同じ政策と仮定しない | 元κ/c固定、新JからI再抽出。禁止Jをone-bodyへ移す操作ではない |
| PF/RTE | ordered partial S2/s1/h=.1/K0、m0..12。r32共通またはr grid retuning | 同じsplit/paired構成を用いる範囲で補正mean同値を確認する計画 | 新split自身のPF/有限RTE評価が必要 | T=1/5,q1,δ1/5,r2,h1/10,K0 | 後続P4はS2/s1/h=.1/r128/K0、13round |
| Normalization | law固有B/Aを評価。有限bit価格proxyとideal lawを区別 | λ変更→B/A/raw半径・Nの交換が対象 | residual λが増えるとcore軽量化を相殺 | arm別λ/Btotを取得。SSのλ/Bは小さくPPの単価は低い | P4でλ増・attenuation悪化を実測。少ないatom位置を少eventとしない |
| Shot政策 | CA=.00159Ha、DF CA/8、β=π/6/stat π/12、failure .01/axis .01/26。条件付き13round会計 | 共通元H energy・成功task。数値配分未採択 | 同左。元H stateでの半径・branchが必要 | 認証N/γ/統計資格未成立、N=null | 後続P4は同CA系の条件付きshots。evidence lane未資格 |
| 評価資源 | W×mean complete kernel＋shell、RZ主・6指標、P0/fenced serial | Σround/axis N×full no-prep価格、指標別 | 同左、fit/collect/較正費も別記録 | 両軸別full wrapperの1-shot期待7指標。X0X1準備・readout/measure込み | P4 complete kernel native32、shell/serial総費用。旧対照とは非paired |
| 実装・検証済み範囲 | 旧/再配置後のDF目的arm比較、同J/U D、保存強D2候補、有限native correctness | preflight・既存機能/計画。元H分子全RPE性能は未実証 | preflight。完全残差＋元H stateの新C-core実証は未取得 | SOURCE9468…、全登録operator/signal/4arm meanと単回価格PASS | SOURCE59e…で1 fit/21、SOURCE4ac0…で449-call・13round/native32完了 |
| 未資格・STOP | PR全law/implemented-law/価格移送/actual CA/FT/母平均。P/r16診断UNRESOLVED。1.15Å STOP | 新実行GOなし。F2やCの別資格を転用しない | 新fit/残差計算GOなし。元H task未接続 | 資格STOP・旧compiled失敗は保持。修正版成功でもN未解決 | 固定P4負結果・STOP、後続C-NOの限定NO-GOを保持。別mask一般の不可能性ではない |

C側compilerはQiskit1.3.0/opt1/seed17/rz,sx,x,cx、位相保持・relative Gaussian/P0。F2も同名設定だが、basis provider、identity、state準備、full-wrapper構成、lawと会計単位が異なる。同じRZ単位で研究間の新価格順位は作らない。

## §4 既存の改善機構と実証済み範囲

### 4.1 研究Cの最新結果を必要最小限で再利用

[C-OLD][COLD]、[C-Z][CZ]、[C-D][CD]の既存表を引用する。旧辞書→再配置後の差の因果分解や、研究間の新比率を算出していない。

| 固定結果 | 保存された観測 | 保持する限定 |
|---|---|---|
| C-OLD C0/64更新 | PR-aware目的由来DFはENTRY由来より、共通r32で51.08%、公平retuning後45.47%低い条件付き総RZ | 36 checkpointはDF予算PASS、J一次MET/U一次NOT_MET。有限recipe比較、大域最適/同時収束でない |
| C-Z保存3候補 | 公平retuningではP/r16=87,514,417,699、ENTRY E2/r16=75,621,395,471 RZ。主比較共通r32もENTRY低値 | 保存J/U再配置であり再配置後目的での再fitではない。主2差は名目モデル支持、P/r16個別PR/D等の影響度診断はUNRESOLVED |
| C-Z同J/U D | E2＋PR/r16は同J/U Dの105,217,244,042 RZより低い点値 | DF目的の優位とPR利用の利益は別。個別名目モデル、同時/母平均保証なし |
| C-D共通LD2再配置強D | E2＋PR/r16=75,621,395,471、XDF8＋強D=101,780,388,260、CDF8＋強D=108,561,283,488 RZ。`E2_POINT_BENEFIT_REMAINS` | 保存2候補、Dはordered S2/s1/h=.1、13round。ideal lane点比較のみ。C1/C2の区間を新対照へ付け替えない |

C-DのDは数値合成laneも通り、idealと整数費用が一致したが合成誤差は非零。PR側の全law/implemented-law/ideal-law価格移送・actual CAは未資格のままである。Dの数値pilotをPRへ移さない。C-OLDの1.15Å停止は移送成功ではなく、今回修復・再開していない。

科学SOURCEは、C-OLDが`artifacts/research_c_integrated/2026-11-06-picklefix/technical/SOURCE.json`（SHA256 `8113f23e66c5eca3c09a8ed1c91fc58c01eeaa91d0c07850b685a20c816539eb`）、C-Zが`cfc64b70861a6482dddd042da8a110abebd6ad7a`、C-Dが`e6652ab841aae5ae700286bf1ab249ad4fb85461`。固定報告のcommitと区別する。

### 4.2 O1とF2

O1-M1/M2は`KNOWN_METHOD_APPLICATION`として整理されたpreflightで、強UWC等を超える全費用利益は未実証である。これは実装不在ではなく、既存部品の適合と共通taskでの利益が未接続という状態である。[O-R][OR]、[O-P][OP]

F2はSOURCE `9468c7089727f5648d9f59cddbaebe0a42f5f0a0`、結果`dc7378ab…`。同じ合成fixtureで13,676 raw/compiled operator・signal検査、4arm corrected meanが全件PASS。単回7指標はPP < SP < PS < SS。期待RZはSS70.825335、SP45.161578、PS53.950984、PP35.445140という**既存報告の丸め表示**であり、Nを掛けた価格ではない。[F2原報告][F2]

SSはλ/Btotが最小でも単回価格は最大だった。これにより、normの大小と実回路単価が別であることは固定条件で観測済みである。一方、γ・認証N・physical同精度総費用は未取得。`f2_resources.execute_resources`は`N=None, weighted_costs=None`を明示し、`f2_statistics.qualified_shots`も資格未解決を拒否する。関数の存在を資格取得へ読み替えない。

F1 metadata失敗→修正版correctness、F2資格STOP／旧compiled3 FAIL→別SOURCEで修正版単回資源PASS、PS48局所費用→正式一時終了という鎖を保持する。修正版成功で旧STOPやN未資格が消えたとはしない。[I-G][IG]、[I-U §2–3][IU]

### 4.3 DF structuralの時点差：準備だけではなかった

| 段階 | 固定文書・科学SOURCE | 保存された到達点 |
|---|---|---|
| 準備版 | [DS-PREP][DSP]、commit`59e186…` | head full/tail P4、128→110の固定注入、旧目的/scaleを保持。人工60＋29 subtests。**このsnapshotでは実H4未実行** |
| DF fit | [DS-FIT][DSF]、閲覧版`e8b19ff6…`、科学SOURCE **`59e186ad5132ffde66684a781a47a2bd11a77ed4`** | 新fit1、21/21 COMPLETE、終了DF予算PASS、`DF_ELIGIBLE_NOT_PERFORMANCE_GO`。同じSOURCEを使った後続実行が存在 |
| 下流性能 | [DS-PERF][DST]、閲覧版`e8b19ff6…`、科学SOURCE **`4ac0e5f96a5ad5009f6273d0fc6ff5657ff80aff`** | 449/449 COMPLETE、13round、native32、raw/compiled full512/actual inverse/X-Y PASS。主lane総RZ275,422,425,481.5対旧234,117,788,135、約17.64%増 |
| 既存統合判断 | [C-NO §4.2/6–7][CNO]、閲覧版`e8b19ff6…` | P4負結果を含め、新cost-aware fit準備候補を選ばない`TECHNICAL_NO_GO_CURRENT_EVIDENCE`。独立新規性や研究C全体終了の判断ではない |

DS-FITはhead `[10,10]`、tail各7上三角、tail係数144を実現したが、λは旧1.4424298856から新1.5161659324 Haへ増えた。Optimizer successは相対目的減少による終了で、停留点・最適性の証明ではない。DS-PERFはs1/r128/K0・own-ground・当時の元辞書での結果であり、11/6のZ再配置済みP/E候補やr16比較ではない。

**216→144は係数位置数であり、r128のevent数・Gaussian種類数を同じ割合で減らしていない。** 後続でPF誤差・shots/W・kernel平均も悪化したという固定観測がある。開始点projection・J/U・lawも変わるため、疎化だけの純粋な因果効果ではない。全supportの不可能性、再配置後にも必ず不利、という結論へ広げない。

以上から、依頼文3.4の「実H4新fit未実行」は指定準備版を説明する場合には正しいが、今回確認できた後続系列全体の現在地には適用できない。古い準備契約の`unapproved`名や原本は変更せず、消費済みの21-call/449-callを再起動しない。

### 4.4 過去の検討済み案・STOP理由

個別値・出典は[C-NO §4–7][CNO]にまとまっているため、その調査を再実施せず以下へ対応付ける。

| 案・系列 | 既存の結果・不足 | 今回の扱い |
|---|---|---|
| basis switch/overlap/composite目的 | cohort内相関はあるが、登録surrogate基準・第三DF transferが不成立。相関はJ/U介入効果ではない | `PREVIOUSLY_STOPPED`：旧特徴の改名・η追加を新候補にしない |
| 凍結event価格weighted mass | 既提案。whole-P0の加算性と使える固定価格が未資格 | `KNOWN_METHOD_APPLICATION`＋`INSUFFICIENT_EVIDENCE`。未実行を負結果としないが、新loss作成の根拠にも使わない |
| shared-basis DF | 一般対称JではG共有groupの表現集合は通常CDF≤G。専用fitter・重複fitは不要 | 既存[SHARED][SH]を引用。同Hのatom化/PF政策の差は残るが、O1-M1/M2/RQ-02/03と重なる |
| rank削減/CDF7 | 既存固定候補でnative/総費用改善を得ていない | rank sweepを別名の第一候補に戻さない。全rank一般のNO-GOではない |
| 固定U tail λ LP | 既存の小さいλ改善はあるがnative/総費用未測定 | λ最小化を新しい資源改善機構としない |
| relative Gaussian/中央結合 | 既存complete人工でRZ増または不変、外側head-tail境界は未測定 | 局所gate削減をcomplete価格へ移さない。未測定境界を必ず有望としない |
| singleton/PS48 | 局所correctness/価格結果はあるが一律削減・未使用selector未成立、正式一時終了 | O1やCへ改名して再開しない |
| core/pair・N1 | 強whole-Pauliでcore利得消失、pairはbasis/中央費に負けた。N1はfixed contextでexact fallback | 元H補償を考える際の反例。新C候補の不可能性にはしない [O-E §3][OE] |
| 非prefix/P-C/P-D/R3 | 局所支持と広域破綻、強対照と同候補、具体的方法差不足等 | 既存RQ-02の未解決を研究C独自の方法と呼ばない [I-G][IG] |
| round/sampling | 旧C条件のround割当はD全選択、sampling surveyは限定`NO_DISTINCT_FREEDOM` | 旧条件を最新r16へ一般化せず、同じ停止familyも再開しない |

11/7のC-Dでは別候補・再配置・r16でPRの点利益がある。これは旧r128のnative-fit NO-GOを一般化して「PRは常に不利」とする根拠をなくすが、停止済みsurrogate/solver案を自動再開する許可や、新価格機構を得た証拠ではない。

## §5 研究Cとの具体的重複

### 5.1 操作を同じ「辞書変更」にまとめない

| 操作 | 保存するもの／変わるもの | 研究C・既存研究との対応 |
|---|---|---|
| global physical Pauli collect | 同じ物理operatorの同一word係数を加算。異frameの同じZ文字列を同一operatorとはしない | F2の`canonicalize_arm`。既存O1-M1経路 |
| DF frameのPauli化 | 回転Z/ZZをphysical Pauliへ展開。同じH_Rを別辞書で表す | F2 S/P、O1-M1。J/U fitではない |
| 線形Z→one-body | 候補H_DF全体は同じ、H_D/H_Rは変わる。κとscalar・head固有basisを変える | C-Zの`prepare`。O1-M1のsplit固定操作とは異なり、O1-M2のsplit変更という分類に近い。ただし元Hとのfit残差補償ではない |
| symmetry shift | target sectorで補償付き同値、off-sector Hは変わり得る | SCDF/BLISS。Cの全Fock同値なZ再配置と異なる [C-BASE §4–5][CB] |
| hard support fit | Jの禁止位置を厳密0にし、近似tensor/Hを変え得る | DSの固定注入。単なる辞書同値やZ移送ではない |

C-Zでは`rows_ti=Σ_j J_tij`、`Δκ=Σ_tail U_t diag(rows_t) U_tᵀ`、`scalar_shift=−Σ_tail,i rows_ti`を既存headへ加え、tailの線形RZだけ除く。frame I/ZZは保持し、同じ再配置modelからDも作る。`c_h4_relocated_downstream.strong_d`はD化のためLDをfullへ変更しても、LD8で再配置をやり直していない。[固定SOURCE][ZSRC]

この具体操作はF2の4armには含まれていない。したがって順位反転はF2結果の重複再現ではなく、異なる操作・物理条件での独立した既存観測である。ただし「norm/辞書/splitが価格とshotへ影響する」という大きな問いは既存RQ-01/02に属する。操作が異なることだけで独立論文の新規性を認定しない。

### 5.2 目的関数と既知法

[C-BASE §2–5/8–9][CB]の対応をそのまま使う。新しい文献探索や同値性証明は行っていない。

- 旧PR-awareはtensor fitと、線形写像後の平滑Z/ZZ mass二乗を使う。ENTRYはordered diag5/4・off3/2のentry mass二乗で、零点floorを合わせても平滑化は全点で同じではない。finite recipe/start/scaleも性能解釈に必要。
- Z再配置後の非平滑 `M_ZZ=Σ_tail(Σ_i<j |J_ij|+¼Σ_i|J_ii|)` は、ordered diag1/4・off1/2のweighted entry L1。`M_ZZ²`はRC-DFのweighted L1二乗族に含まれる。head mask、smoothing、scale、solverまで原法と一致するとはしない。
- 保存ENTRYの重みはw_ZZではない。重み比がdiag/offで違うため、strengthだけ替えて同じ目的へ戻せない。逆にmask・重み・smoothing・scale・solverまで同じなら、別名の二手法を重複fitしても方法差を検証できない。
- UWCはglobal OO/BLISS、可変Pauli head/tail、head費用＋tail²、RPE資源の設計を既に扱う。固定CDF frame maskで独立J/UをfitするCとは同じ最適化問題ではないが、「表現とPR費用を共同最適化する」という説明自体は重複する。
- SCDFのrank-one/shiftと一般signed Jのtail supportは別。制約が違うことを、新アルゴリズムとしての追加価値の証明にはしない。

### 5.3 B1–B6の比較分類

| 問い | 分類 | 根拠・判断できる範囲 |
|---|---|---|
| B1 Z再配置反転とO1/F2 | `ALREADY_VERIFIED`（Cの固定観測）、`DISTINCT_MECHANISM_POSSIBLE`（操作差） | 全H保存でもsplit変更あり。F2の同H_D/H_R辞書変更と別。一般的な表現依存性という問題は共通 |
| B2 目的の一致 | `KNOWN_METHOD_APPLICATION`、一部`INSUFFICIENT_EVIDENCE` | w_ZZ L1²族対応は既存に確定。旧ENTRY/原法solver/UWC全体と同一ではなく、実性能の公平比較は未完 |
| B3 M2への保存C fit再利用 | `KNOWN_METHOD_APPLICATION`＋`INSUFFICIENT_EVIDENCE` | J/U・元g補正・CARは再利用候補。完全残差と元H ground上のphase/半径/全費用はC結果にない |
| B4 hard supportの追加価値 | `ALREADY_VERIFIED`（旧固定P4の実行・負結果）、`PREVIOUSLY_STOPPED`（関連fit選別） | softとhardは異なる許容集合。旧P4で適格fit/費用悪化まで実証。別mask・再配置後版の利益は未知で、一般否定でも新候補GOでもない |
| B5 F2単価とC総RZ | `ALREADY_VERIFIED`（各元条件）、`INSUFFICIENT_EVIDENCE`（共通因果） | 両者ともnormだけで資源を決められない。F2はNなし、CはWμ＋shell。F2の価格因子をCへ移植して説明しない |
| B6 強Dに対するC利益 | `ALREADY_VERIFIED`（固定C-D ideal点比較） | 実H4・再配置・ENTRY由来CDF＋PR対保存強Dという具体問を閉じた。O1の元H厳密state/full task、F2認証N、UWCへの優越までは閉じない |

## §6 研究Cで新たに判明したこと

ここでの「新たに」は、指定された他系列の固定証拠だけでは直接得られない研究Cの観測、という意味である。科学的新規性・文献初出の判定ではない。

1. **目的armの順位は保存DFの実装表現とsplitに依存した。** 同じ保存J/UへZ再配置を適用すると、旧PR-aware低費用からENTRY由来低費用へ変わった。F2はJ/U目的比較もone-body移送もしておらず、この具体結果を持っていない。
2. **PR利用利益とPR-aware目的利益を分離できた。** ENTRY由来DF＋PRが自分のDや保存CDF8/XDF8＋共通LD2強Dより低いideal点費用を示した。これをPR-aware目的関数の成功としては扱えない。
3. **Hamiltonian同値はPF/RPE資源同値ではなかった。** C-Dでは再配置によってDのkernel RZが減っても総RZは旧政策より増えたという既存報告がある。同じHだけでPF split/shot/費用の再利用を正当化できない。
4. **実H4 fit・物理DF予算・13round・native価格を接続した限定証拠がある。** F2の単回synthetic価格やO1 preflightから直接出せる結果ではない。ただしown-ground・ideal law・有限標本という射程を超えない。
5. **古い「疎化未実行」を更新する証拠も既にある。** P4の適格fitと負の資源結果は、supportを研究Cの新しい第一案に戻す前に読む必要がある。これは11/7の再配置結果でP4を再計算したことではない。

これらは成果として保全できる。新たな独立方法が残らなかったとしても、過去の有効な限定実証・負結果が失効するわけではない。

## §7 独立課題候補

**今回追加する独立課題候補は0件。現在確認した範囲では独立課題を特定できなかった。** 未解決の問いは残るが、具体化できたものは既存O1-M1/M2、既存C-BASEの比較課題、既存structural/native設計の後続に対応する。既存法の応用自体に研究価値がないという判断ではない。

| 候補に見える案 | 具体的な既存対応・残る差 | 今回独立候補へ残さない理由 |
|---|---|---|
| 「再配置後massでJ/Uを再最適化」 | w_ZZのRC-DF型L1²と物理admission、C-BASE §3/8/10に既存。保存ENTRYとの重み差はある | 未実行の新比較であって新しい方法差を示す証拠ではない。同一目的になる場合は重複fitも不要。元目的の再命名だけでは不足 |
| 「固定DFでS/P辞書を選び総RPEを下げる」 | O1-M1/F1/F2、same-H_R/mean・λ/B/価格の交換 | 問い・操作・必要証拠が既存O1-M1と同じ。CのH4値でF2のN不足は閉じない |
| 「圧縮DF＋元H残差をPRで補償」 | O1-M2そのもの。保存C候補はcore素材にできる可能性 | 新しいfitではなく完全残差/元H state接続が未取得という既存課題。別研究C名で二重実装しない |
| 「tail supportを減らす」 | P4の実fit・全下流負結果、既知support制約CDF | 準備版を未実行と読んで同条件を繰り返す価値はない。別maskだけの探索は具体的な価格機構不足 |
| 「shared/near basis、switch、weighted event価格」 | 通常低rankとの同値、既存surrogate転送不支持・complete価格反例・加算性未資格 | 新自由度の名称だけでは不足。既存停止やNO-GOを解除する新証拠を今回は取得していない |
| 「PR全law/actual CAを認証する」 | Cの証拠資格の不足を閉じる課題、O1/既存RPEの未解決にも重なる | 重要な検証課題にはなり得るが、新表現改善機構と独立研究としての主題を今回特定したことにはならない |

別の元H state、誤差資格、表現クラス、制約、native変換を選べば検証結果が変わる可能性はある。しかし変更を列挙するだけでは、既存強baselineで再現されない具体差や反証可能な追加機構を得たことにはならない。

## §8 最小検証・SOURCE再利用可能性

独立候補が0件なので、新しい実行計画や契約は作らない。以下は**ユーザーが既存課題を選ぶ場合にだけ研究判断を変え得る最小の確認**であり、全実施必須のbacklogではない。今回実装・科学実行していない。

### 8.1 既存課題へ戻す場合の情報価値・反証・負担

| 既存の問い | 最初に必要な確認・何を固定するか | 強対照と反証／STOP | 再利用と新処理の境界 |
|---|---|---|---|
| O1-M1：同じH_Rで辞書のN×価格交換が有用か | 元H/state・H_D/H_R/scalar・finite target・r/K/成功条件を共通化。まず既存S/P両端点の完全operator/meanと正しい半径/shot政策を接続 | collected-P/rotated-Sに同じphase/control/fusionと情報を与える。全taskでは適用可能UWC/強D。Nまたは価格の増加で利益消失、識別不能、target変更なら当該claimをSTOP | F2 typed辞書/mean/compileを利用。CのZ移送を「split固定」の操作には使わない。分子adapter/新law/半径・N/価格は新科学処理。旧F2資格STOPを改変しない |
| O1-M2：保存coreの軽量化が完全残差を払っても残るか | 保存C/XDF/CDFの元g/κ/c/sectorを静的に対応付ける。新fitを先行せず、適合core一つと元Hとの差を実装できるか確認。その後元H stateでmean/phase/半径を確認 | 同じcore/residualを使う既知PR、元Hに整合した強D・collected-P/適用可能UWC。残差λ・PF・basis・shotsが利得を相殺、元H task不適合/無料oracle追加ならSTOP | CのJ/U/general-K/CARと既存RTE/nativeを利用。任意fitted core＋完全残差・元H stateのprovider境界は未接続。新残差生成/信号/compileは別科学許可 |
| C-BASE：既知正則化族を超える方法差があるか | mask/ordered weights/power/smoothing/scale/solver/初期/有限予算を静的に揃え、同一最適化問題かを先に判断 | 同一定義なら重複fit不要。差を調べる場合は既存weighted-L1²/full-L2/原法solver等のどの差を問うか一つ選ぶ。同条件既知法で再現されれば独自方法claimを止める | tensor/AD/codec/stable経路・既存RC2と確認済み外部実装が出発点。新fitは別許可、既存ENTRY重みを黙ってw_ZZに置換しない |

これらは肯定なら既知応用の成立条件、否定なら同条件の不利・同値・適合不足が分かる。どちらの結果も有用になり得るが、研究Cという独立枠を維持する理由とは別に判断する。structural P4をこの表の第四の新検証へ戻さない。再配置後のsupport等を希望する場合も、旧負結果と違う具体的な問いをユーザーが選んでから扱う。

### 8.2 部品・runner・testの再利用表

| 機能 | 固定SOURCE・既存関数／入口 | 再利用できること・制限 |
|---|---|---|
| J/U objective/solver | C-D版`c_pr_aware_df.make_value_and_grad/fit`、`c_near_objective.entry_weights/mass`、ffsim codec・stable J/U資産。C-BASE §8 | 新optimizerの一からの実装は不要。新penaltyはADだけでなく既存解析gradient/Hessian/admissionとの整合が必要 |
| hard support | DS-PREP=`59e186…`の`c_df_structural_support.Support.project/reduce/expand/assess`、`pullback`、`compact_optimizer`、`fit`、`c_df_structural_execution`、専用runner/tests | 固定線形注入`x=Ey`、gradient `Eᵀ∇F`は実装済み。準備コードの存在と実fit/P4負結果を両方保持。消費済みrunnerを再起動しない |
| Z再配置 | C-Z/C-D版`c_h4_relocated_downstream.prepare/raw_generators/tail_gate/strong_d` | 既存κから移送、独立CARと全H同値を確認する部品。`prepare_candidate`はH4/CDF8/保存input/LD2/own-groundへ固定され、汎用元H importerではない |
| F2辞書 | 科学SOURCE=`9468c708…`の`prototype.make_fragment_alternatives/canonicalize_arm/make_tail/configure`、`algebra.py`/`fixture.py`、F1 tests | exact Fraction・physical-word collect。`canonicalize_arm`は2fragment/F0-F1とSS/SP/PS/PP固定。任意CDF8へ無変更で適合とはしない |
| F2 full wrapper/価格 | 同SOURCEの`circuits.build_event_sequence`、`gates.compiled_wrapper/component_derived_mean`、`f2_resources.execute_resources`、F2回帰tests | 修正済み位相保持、全operator/referenceと実単回価格を再利用。旧位相修正を再実装しない。literal `pricing.py`はnative oracleでない |
| Shot/RPE | 同F2 `f2_statistics.require_adopted_statistics/qualified_shots`、C `c1_stage3_accuracy`、既存radius/shot/decoder、O1 O-P §5 | F2資格をCの条件付きNで置換しない。古いDFPartialS2限定・基準半径1のAPIを元H新splitへ無条件適用しない。対象半径/誤差providerだけを必要時に接続 |
| General-J/physical reference | C `c1_general_k_adapter.from_ffsim_number_arrays`、`c1_stage1.corrected_kappa`、`SectorCAR`/`rank_assessment`、`c1_stage3_reference` | 元g補正・signed J・spin/JW/sector・scalarを保つ。core fit誤差と完全補償を混同しない。dense oracleを大系の無料アクセスにしない |
| Native complete kernel | C `StrongBuilder`/`c1_stage3_native_bridge`、F2 `BasisProvider`、既存phase/actual inverse/X-Y tests、I-UのDF-CONTROL/MD-CONTROL/RTE-BASE | 既存relative融合・exact-key run共有・compiler簡約を双方へ与える。singleton、near-U共有、cross-head/tail、旧価格を新policyへ無断輸入しない |

新しいscience pipeline全体、DF solver、CAR実装、Gaussian合成器を作り直す必要はない。必要になり得るのは、**保存coreと完全残差・元H state、一般入力とtyped辞書、対象信号の半径と既存会計**の限定adapterである。適合しない固定runnerのgateを外して汎用化することは再利用ではない。

### 8.3 資源見積りの限界

既存DS-FITは1 fit/21-call、whole runnerのwall3.79秒、DS-PERFは449-call/native32でwall17分18.59秒と報告している。これらは当時の特定入力・solver・環境・政策の実績であり、再配置後の新fitやO1分子実行の所要時間予測へ転用しない。新残差辞書・全law/全roundの実行費は未測定である。CPU/RAM/workerの人工的な低い上限、新しい科学call契約は作っていない。

## §9 研究Cの今後の3案

| 判断軸 | 案A：独立再構成 | 案B：現成果で一区切り | 案C：既存O1等へ技術接続 |
|---|---|---|---|
| 独立した科学的課題 | 現時点で具体的な別立て課題を未特定。新しい具体差と反証可能性が条件 | 新しい独立課題を作る必要なし | O1-M1/M2等の既存課題として扱う。C独立性を必要としない |
| 既存手法との重複 | 再命名だけでは残る。w_ZZ L1²、support/共有等を再点検する必要 | 重複実験を増やさない | 既存の担当課題・同じ部品へ接続し、二重実装を避ける |
| 現在の数値証拠 | H4条件付き結果はあるが、独自方法の追加性を証明していない | 旧辞書・反転・強D点比較・P4負結果を一組の成果として保持 | 入力/候補素材と機構の参考。元H stateや新辞書の性能には移さない |
| 追加実装 | 現段階では目的と差が未固定で着手根拠なし | 不要 | 課題を選んだ後だけ限定adapterの必要性を判断 |
| 追加科学計算 | 採択前には不要。差を検証するなら別の明示許可 | 不要 | 残差/元H state/新law/価格を評価する場合だけ別許可 |
| 強baseline比較 | 新たな課題固有の公平条件が必要。既存2強Dだけでは全法優越でない | 既存比較の限定をそのまま保存 | 同core/同辞書pool/同情報/元H taskをbaselineにも開放 |
| 既存成果の活用 | 新課題を決める材料 | 保存結果・負結果・限界として活用 | J/U、再配置、CAR、RPE/native、SOURCEを再利用 |
| 継続理由 | 操作・費用機構・新証拠が具体化した場合のみ | 新しい独立課題未特定なら合理的な区切り | 既存O1の問い自体に価値を認めたときだけ。Cの延命を目的にしない |

案Bと案Cは両立する。研究Cの現在の成果を区切り、既存O1が必要とする部品を将来使うことでよい。Track AのPR資源評価論文への全面統合とは別案である。

## §10 推奨と根拠

**案Bを基本、案Cを必要時に限定する。案Aは、既存O1/既知正則化/構造制約/既存NO-GOと異なる具体的な問いが提示されるまで選ばない、という研究管理上の提案である。** 正式終了・研究開始・論文化判断はユーザー／GPTへ戻す。

根拠は、(1)旧PR-awareの低費用がZ再配置後へ移らない、(2)ENTRY＋PRの点利益は残っても目的関数固有利益ではない、(3)w_ZZ目的が既知L1²族に含まれる、(4)辞書と残差補償はO1の既存2課題、(5)structuralは後続実行・負結果まであり、未実行案として戻せない、(6)新しいcomplete価格機構へ直結する証拠が不足している、という固定資料の関係である。

継続する場合に必要なのは、さらにloss名や自由度を増やすことではなく、どの既存taskで、何を固定・変更し、同情報の強対照が再現できない差または有益な成立条件を調べるのかという選択である。O1の未解決はそのままO1で扱える。新実装・科学計算・重要レビューを、今回の報告を完成させる必須工程にはしない。

一区切りにする場合も、研究CのH4条件付き結果、Z再配置の全H同値、目的arm反転、固定強D比較、hard support負結果、資格・STOPを保全する。結果の無効化や過去証拠の書換えは必要ない。

## §11 残る判断事項

### 11.1 Q1–Q12への回答

| 質問 | 回答 |
|---|---|
| Q1 CとO1-M1/M2の共通・差 | 共通は表現/split→λ/B/shot/native資源。M1は同H_D/H_R辞書、M2は元H完全残差、Cは候補H_DF/own-groundとZ split変更（§2–3） |
| Q2 Z再配置は同じ操作か | 同じではない。F2の辞書代替/collectと異なりD/R演算子を変え、全Hは保つ。SCDF sector shiftとも別（§5.1） |
| Q3 F2でどこまで扱えるか | typed S/P代数・full-wrapper correctness/単価は部品として再利用可。J/U fit、one-body移送、実H4・13round・認証N/総費用は既存F2の結果外（§4.2/8.2） |
| Q4 hardとsmoothの追加価値 | 数学的には許容集合とpenaltyの差。ただしP4は適格fitと総RZ悪化まで実行済み。別条件の追加価値は未確定（§4.3） |
| Q5 RC-DF/SCDF/UWC重複 | 数学的対応・変数/task/API差はC-BASEで整理済み。公平な同条件性能比較や独立新規性は未解決。同じ目的を別名で二重fitしない（§5.2） |
| Q6 C固有の既存H4結果 | 目的armの旧/再配置後順位、ENTRY＋PRの同J/U/保存強Dに対する条件付き点利益、実分子の13round/native接続。F2/O1 preflightにこの直接結果はない（§6） |
| Q7 残る独立課題 | 今回特定0件。新しい機構の不存在証明ではない。具体化できた未解決は既存O1/C-BASE等へ対応（§7） |
| Q8 既存法で再現され得るか | あり得る。w_ZZ L1²、通常低rank、同pool辞書選択、同core residual法を強対照へ与える。同一定義なら重複実行不要（§5/8） |
| Q9 判断を変える最小検証 | 新独立課題の選択はまず静的重複判断。O1を選ぶなら、同H_R辞書の共通成功条件＋N×full価格、または保存core＋完全残差の元H state適合が分岐点。今回は実行不要（§8.1） |
| Q10 再実装必要か | 新DF optimizer/science pipeline全体は不要。選択したtaskで既存入力/provider境界が不足する場合だけ限定adapter（§8.2） |
| Q11 適切な位置付け | 管理上はBを基本、Cを必要時に限定。Aは別立て具体課題が提示されるまで未選択。正式な終了/GOは判断しない（§9–10） |
| Q12 次の担当 | ユーザー／GPTがBかO1への技術接続を選ぶ。Codexは別依頼後だけ必要な整理/adapterを担当。追加科学実行・重要レビューはそれぞれ必要と選んだ場合だけ明示許可（下表） |

### 11.2 次の担当と許可の境界（未実行）

| 担当 | 次の選択・作業 | 現在の状態 |
|---|---|---|
| **ユーザー／GPT** | 研究Cを現成果で区切るか、既存O1-M1/M2のどちらを検討するかを選ぶ。独立再構成なら既存と異なる具体的な問いを指定 | 今回の資料から判断する次段階。正式採択をCodexが代行していない |
| **Codex** | 選ばれた既存課題について保存core/辞書/元H/state/SOURCEの適合を整理し、必要な場合だけadapterを準備 | 別の具体的依頼後。今回新実装0。全基盤の作り直しは不要 |
| **ユーザー** | 新残差生成・保存値再解析・fit・PF/RPE/native等を望む場合、対象SOURCE/input/条件/scopeを明示して科学実行を許可 | 今回不要・未実行。消費済み枠や停止familyの自動再開は禁止 |
| **ユーザー／GPT** | 正式な独立新規性・論文化・研究終了を詳しく審査したい場合、その重要レビューの対象と開始を別に許可 | 本報告の完成条件ではない。今回は開始していない |

Codexの自動的な次作業なし。新しい課題を作るためだけの追加計算・レビューを提案しない。

## §12 保全・MANDATORY STOP

本報告の検査は、固定文書の取得・Git blob/SOURCE定義・出典・時点の静的照合、Markdown参照と差分確認、commit・公開・remote SHA/tree/本文一致の確認である。別担当による独立科学レビューや、科学結果の再検証とは呼ばない。

科学数値の再計算・再集計、J/U/Hamiltonian/stateのdecode、DF fit、PF/RTE/RPE、sampling、回路build/compile、科学runner/test実行、新アルゴリズム/目的/solver、科学契約/grant、科学的な新保証・新規性/GO/終了判断は全て0。原稿・既存図表を変更していない。

既存C全SOURCE/契約/raw/結果/STOP、O1の4固定文書/STOP、F1/F2の旧失敗・修正結果・資格STOP、structural準備・後続結果と元契約、3環境原台帳、Track A S1方針・原稿・図、H2/H6/1.15Å停止、grant/ledger/marker、他worktree/process/cache・共有環境は不変更。今回のtree差分は本Markdown一件に限定する。他チャットの通常更新を凍結せず、引用は固定commitによる。

主要資料の固定GitHub本文とローカルblob、引用SOURCEの所在、12節・Q1–Q12・文書参照を確認して公開する。Git認証設定は共有変更せず、前回の通常push認証失敗を踏まえ、既存の承認済みGitHub接続を使う。公開commit SHA/tree/本文一致と一Markdownのみの差分の実績は最終チャットに記す。自己commit SHAを本文へ埋めるための追加commitは不要。

**公開照合後MANDATORY STOP。** 独立再構成、新科学実行、停止family再開、Track A全面統合、重要科学レビュー、正式終了・論文化判断へ自動進行しない。

[COLD]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/417fa459b823509be5c2010c5836e58955088282/docs/research/research_c_integrated_h4_100_results_20261106.md
[CZ]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3015d4fb342d4d98b58e41d57553e75539a07655/docs/research/research_c_h4_relocated_downstream_results_20261106.md
[CD]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/91417fb3ef75e7eec0849ad39f0649d170754664/docs/research/research_c_h4_strong_d_relocation_results_20261107.md
[CB]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/77ded5c12db21572933653cb26758436da3d9ecd/docs/research/research_c_baseline_comparability_20261106.md
[OR]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aeb9d1847f9d500f2389db12b384bac6087117a9/docs/research/o1_hamiltonian_partition_preflight_20261106/REPORT.md
[OE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aeb9d1847f9d500f2389db12b384bac6087117a9/docs/research/o1_hamiltonian_partition_preflight_20261106/O1_EXISTING_METHODS.md
[OM]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aeb9d1847f9d500f2389db12b384bac6087117a9/docs/research/o1_hamiltonian_partition_preflight_20261106/O1_IMPROVEMENT_MECHANISMS.md
[OP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/aeb9d1847f9d500f2389db12b384bac6087117a9/docs/research/o1_hamiltonian_partition_preflight_20261106/O1_MINIMUM_VALIDATION_PLAN.md
[F2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dc7378ab34015287215bcf4ce52147e46d94ecd9/docs/research/pr_algorithm_improvement_phase2d_f2_resources_rerun_20261024/EXECUTION_REPORT.md
[DSP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/59e186ad5132ffde66684a781a47a2bd11a77ed4/docs/research/research_c_df_structural_design_20261028.md
[DSF]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/research/research_c_df_structural_support_execution_20261028.md
[DST]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/research/research_c_df_structural_pr_performance_20261028.md
[CNO]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/research/research_c_pr_native_cost_improvement_feasibility_20261102.md
[SH]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e8b19ff6baf79e21b9b658c0f38dcf66b7e180d8/docs/research/research_c_shared_basis_df_design_20261029.md
[IG]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3571477093441ac4852166fe829f8ccc5db5ec4f/docs/research/pr_algorithm_improvement_three_environment_integration_20261105/IMPROVEMENT_GAP_MATRIX.md
[IU]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3571477093441ac4852166fe829f8ccc5db5ec4f/docs/research/pr_algorithm_improvement_three_environment_integration_20261105/DUPLICATION_AND_REUSE_MAP.md
[IV]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3571477093441ac4852166fe829f8ccc5db5ec4f/docs/research/pr_algorithm_improvement_three_environment_integration_20261105/REMAINING_VALIDATION_MAP.md
[ZSRC]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3015d4fb342d4d98b58e41d57553e75539a07655/src/trottertracks/c_h4_relocated_downstream.py
