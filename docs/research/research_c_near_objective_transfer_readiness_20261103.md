# 研究C：PR-aware対weighted L1・未使用条件へのrecipe移送準備

2026-11-03 JST。技術準備・実験設計。**実分子の新科学計算・保存raw数値再解析・重要科学レビューは0件。**

**Adapter/比較IIの設計は技術的GO、分子科学runはNOT_READY。** 人工・回帰 **72 unique PASS（新31＋既存41）**、数値moduleのimportを禁止したpassive preflightはSOURCE167 files・metadata5件でPASS。第一推奨は **H4・1.15 Å/STO-3Gのrecipe移送**。代替は反対側のH4・0.85 Åと、既存の元入力・中立DFがあるH6・1.00 Å。いずれも科学条件の正式採択ではない。

[未承認設計JSON](research_c_near_objective_transfer_contract_unapproved_20261103.json)は **`UNAPPROVED_DESIGN_NOT_EXECUTABLE`**、`selected_system/science_entry/maximum_scientific_calls=null`。H4別距離の元h/g/scalar packetと中立初期値、rank/LD・scale/強度・有限予算・校正政策が未固定なので、数値を恣意的に埋めた実行可能契約にはしていない。

[入力候補matrix](../../artifacts/research_c_near_transfer/2026-11-03/readiness_matrix.json) ／ [adapter](../../src/trottertracks/c_near_transfer.py) ／ [passive CLI](../../scripts/prepare_research_c_near_transfer.py) ／ [人工test](../../tests/management/test_research_c_near_transfer.py) ／ [技術SOURCE](../../artifacts/research_c_near_transfer/2026-11-03/technical/SOURCE.json) ／ [preflight](../../artifacts/research_c_near_transfer/2026-11-03/technical/passive_preflight.json) ／ [静的照合](../../artifacts/research_c_near_transfer/2026-11-03/technical/static_verification.json)。

## 1. 固定版と保全する科学証拠

| 用途 | 結果/資料commit | 科学SOURCE・資格 |
|---|---|---|
| 最新Evidence Map・旧42 Claim | [169e327ae…][MAP] | C01〜C29/N01〜N10/G01/Q01〜Q02を固定参照。今回新しい移送成功Claimは追加しない |
| PR/ENTRY有限手順 | [2ceb0a882…][OPT] | `70a98adfb701a209db2c65ea60fb789ff62b1fe7`。共通tensor-only開始・各16受理更新・両DF予算PASS、J MET/U NOT_MET |
| 二endpoint共通r32下流 | [beb8de5cd…][DOWN] | `4df8eca29c59ffd55e2dd898f05da16bb246ddbe`。条件付き費用・nominal Welch判断、全law/actual CA等未資格 |
| 近接目的とUWC/RC-DF/SCDF | [既存比較設計][NEAR] | 方法差と概念的重複を保持。ENTRYは原法全体の再現ではない |
| H6元input | [既存input identity][H6INPUT] | 生成SOURCE `33c380d8aa792e5119bb3adec5dd07d1e1377214`、SHA256 `b82ee9864d9c76071944834ae32d3d6eea7e95959708c2e19bca011936bce447` |
| H6修正codec・tensor-only fit | [保存報告][H6FIT] | SOURCE `33432e78659b8b3e654e39f1392328cb0ba5c487`。直鎖H6・1Å/STO-3G、中性singlet・Nα=Nβ3/sector400、CDF14、元g補正κ/scalar一回、PF split/時間非適用。保存DF予算PASSは引用のみ |
| H6 native・CA会計資産 | [Native精度修正][H6NATIVE]・[CA/serial][H6CA] | Native SOURCE `3d7b56b44832a2a90a8d3e500bf55e856f36da76`。有限probeと数値/仮定laneの資格を、PR全lawへ移さない |

起点はF16とR32の実装がそろう `beb8de5cd8b71f80bb69556f92299435ff431c3e`。最新Map169e327aeは別branchの固定URLで参照する。独立branch `codex/research-c-near-objective-transfer-readiness-20261103`、worktree `/tmp/prt-c-near-objective-transfer-readiness-20261103`。旧branch/worktreeを変更せず、既存raw/NPZ/QPYを複製・decodeしない。

正式運用指示 `research_project_instructions_unified_2026-10-25.md` は依頼元workspaceの原本、特に§8/8.4/10を適用した。技術準備の許可を、新科学実行・重要レビュー・科学条件採択へ拡張していない。承認証明・原文保存・新grantは作成しない。

## 2. 調査した未使用条件と入力の実体

「未使用」は**このPR/ENTRY有限recipe比較に未使用**という範囲である。別研究で既に計算したgeometryを、全研究でblindな条件とは呼ばない。他研究の停止を解除しない。

### 距離移送：H4

固定tree `e098c54c78f589055082f9cfc2b13de50c90ca94` に、旧P-Cの0.80/0.85/1.15/1.20 Åの[PF記録][H4META]がある。`hamiltonian`と`provenance`からmetadataだけを確認した。記録はrank12/LD3・総電子数sector70であり、研究CのNα=Nβ2/sector36とは異なる。元h/g/scalarを含むpacketや、今回用の中立CDF8 J/U/x0ではない。

記録された生成commitは `898da8487aca30a176ce5931d34a923c28c34ac9`、証拠statusはlocal dirty-worktree validation。Hamiltonian hashやPF結果があることから、元tensor・全生成SOURCE・修正codec適格初期値の存在を推定しない。調査範囲は指定固定tree・既存資料/既知pathであり、全cache/全ディスク/全履歴にpacketが存在しないという証明ではない。

1.15 Åは保存された異なるgeometryのうち近い伸長側で、サイズ・orbital数・sectorを維持できるため第一提案とした。同じ距離差の圧縮側0.85 Åを代替に残す。旧PF係数やPR/ENTRYの勝敗予測を選定基準に使わず、四条件同時の探索は提案しない。

元packetを来歴付きで回収できなければ、将来の許可済み段階で新SCF/積分生成が必要。旧`chemistry_hamiltonian.call_geometry/jw_hamiltonian_maker`は距離指定を持つがOpenFermionの係数順序と保存形式を使う。一方、研究Cは`c1_molecule_extension.integrals`のPySCF RHF/ao2mo・chemist `(pq|rs)`/h/g/constant経路を使う。後者のwrapperはH2/H6・1Å固定なので、H4/別距離の生成dispatchはまだ接続していない。別辞書のPF係数を代用品として持ち込まない。

### サイズ移送：H6

元h/g/constant packet・input identity、保存XDF14、修正real-Schur codec、tensor-only CDF14、sector400 CAR/一般K、ground/PF、native state-action/精度修正・serial会計がそろう。保存中立候補はXDF14（SHA256 `559dc43bf923a421ccc62d05c4cefa0bdf39a48ba68824abf3c8f770c04642b1`）と修正tensor-only CDF14（`cd4de687f99d32bd6f644ac33100fb52d4f7543e710e6c273723a0908b87dc45`）。両armへ同じ候補を使う必要があり、正式な初期化選定は未了。

H6には強い入力資産がある一方、rank14、LD、scale、sector400、12 system modes/controlled8192、native検査方式が変わる。H4距離移送と同じ一般化実験にはしない。既存H6のfinite-probe correctnessは全operator/full512検査と同じ資格ではなく、新PR law/trajectoryへの接続も未実施である。

| 必要な接続 | H4別距離 | H6・1Å |
|---|---|---|
| 元入力/生成来歴 | PF metadataのみ。元h/g packetの回収または新生成・固定が必要 | 元packet/生成SOURCE/hashあり |
| 初期化・修正codec | 新条件の中立CDF初期化と物理予算は未評価 | 修正codec・tensor-only CDF14/XDF14保存候補あり。旧結果を再計算しない |
| Rank/LD | rank8/LD2は変更を抑える提案。DF予算成立は未知 | rank14は既存資産由来の候補。LDの科学選択は未了 |
| PR/ENTRY/optimizer | 既存coreをそのまま利用可能。H4固定IO/scaleを分離 | 新adapterで6軌道shape・scale・既存arm呼出しを人工確認。閾値の意味・最終policyは未固定 |
| General-K/CAR・DF norm | sector36、元g補正κ、scalar一回の既存engine | 同じ既存sector engineの400を利用可能 |
| Ground/alias | 新候補固有に再評価・window固定が必要 | 新PR/ENTRY固有に必要。旧CDF/XDF groundを流用しない |
| PF/RTE・RPE | 既存engine、候補固有13roundを新規に接続する設計 | sector経路を再利用し、候補/LD/rの新接続が必要 |
| Native/serial | 既存P0・full512・shell/serial。ただし新J/Uの価格が必要 | 既存8192 state-action/precision資産を使う設計。H4 full512 wrapperは転用不能 |
| Runtime/資源 | 既存fit/eval環境、入力生成/初期化は新段階 | 既存2環境＋Aer/state-action。大きなdense全operatorを標準にしない |
| 比較可能性 | geometry一要素の移送を優先できる | size/rank/LD/精度・会計backendの影響が重なる |

## 3. 第一実験案：recipe移送I・有限手順比較II

移送Iは新条件で共通中立初期値から両目的を解くこと。最適化済みH4 endpointを直接移す「パラメータ移送II」とは別であり、今回は後者を主対象にしない。「比較II」は目的の一般的な最適能力を問う比較Iとも別の分類である。

| 項目 | 第一提案と採択前の条件 |
|---|---|
| 移送先 | Linear H4、隣接1.15 Å、STO-3G、中性singlet・Nα=Nβ2/sector36。正式採択前の提案 |
| 既存入力 | PF metadataへの固定参照のみ。元h/g/scalar packetと生成SOURCEを回収/新規固定する |
| 表現 | Number CDF、一般signed対称J・実直交U。rank8/元順LD2を最小変更候補にするが実行条件はnull |
| 初期化 | 新条件の非正則化tensor-only CDFを両arm共通にする。修正codecでJ・density projector・tensor/x0同値性を確認。旧PR-aware endpointからのwarm startなし |
| Scale | 新しい共通中立点のf0、PR Mε0、ENTRY Bη0を一度だけ固定。旧H4 f*/M*/B*の数値を使わない。f0=0はUNRESOLVED、恣意的floorを入れない |
| 強度・smoothing | 初期正規化penalty寄与.1/arm・ε=1e-8 Haは元recipeを移す提案。η=ε(2n+1)/(1.5n−1/4)。初期寄与/零floor一致は勾配・曲率・最適強度の公平性ではない |
| 有限予算 | 元wrapperの各16受理U更新・128 total trial・方向ごと32 halving、cold J/2 refinement・同じadmission/独立終端を提案。正式budgetはnull。Armごとの有利な追加更新なし |
| 物理判定 | 元g補正κ・nuclear scalar一回、元frame/spin/bit順、既存sector norm＋engineering guard、共通CA/8=.00019875Haを提案。初期/各受理/独立終端を候補固有に評価 |
| 下流 | 双方同じ固定r32/s1/K0/h=.1/m0..12はrecipe移送の提案。最適rと呼ばない。公平な有限retuningは別の科学設計 |
| 指標 | Exact λ・B0/cancellation、DF/J/U到達度、候補固有PF/RTE・A・shots/W、native単価、shell込み条件付き総RZ・事前固定した価格不確かさ |
| 再利用 | `c_near_objective/c_near_stable/c_near_practical`、修正codec、`SectorCAR/rank_assessment/representation_gate`、既存accuracy/RPE/native/serial |
| 不足実装 | H4別距離のinput生成/保存dispatch、中立初期化の固定・IO、候補固有のdownstream dispatcher、N/精度とcall/SOURCE/runtimeを閉じる最終runner |
| 計算負担 | 新SCFが必要なら一共通packet、新中立fit、二armのJ solve/refinement、候補固有13round、採択後の価格校正。今回の所要時間を科学runへ外挿しない |
| 情報価値 | 肯定なら別geometryに同recipeの限定支持、否定/物理不適格なら固定政策の移送限界を得る。いずれも条件を変えて勝者を作り直さない |
| 制約 | 一geometryの有限比較で、大域最適・独立新規性・全分子優位・actual CAを認定しない。H6サイズ移送の代用にもならない |

両armがU一次未達でも比較II自体を無効にしない。双方の到達度・raw optimizer status・棄却/未資格trialを報告し、比較Iの目的能力へ格上げしない。初期物理予算未達や片arm不適格なら登録状態として保存し、rank/LD/r/精度を片側だけ結果後に変更しない。

## 4. 実装の再利用範囲と残る接続

新しい数値optimizer・PF engine・shots solver・native builderは作っていない。実装SOURCEは **`bd38fbbfa2c5fa79d4fb416aa4ecc7769ee8e5ee`**。既存科学コードの編集0件で、追加はadapter・passive CLI・人工testの3ファイル。

| 新adapterの責務 | 既存処理・適用範囲 |
|---|---|
| `System/validate_arrays` | Geometry/rank/LD/input identity、閉殻spin-sector、signed J・U直交性・binary64 shapeを分離。原PF記録のsector70を36として読まない |
| `objective_specs` | 既存tensor再構成・PR mass/Objective・ENTRY initial_specを使用。新条件の共通scaleだけを組み立てる。人工入力で確認 |
| `recipe_arm` | 既存`c_near_practical.run_arm`全体へ委譲し、既存stable reduced/inner/refinement/Armijo・独立終端・固定finite policyを使用。旧global定数のmonkeypatchなし |
| `PhysicalContext` | H4固定のwrapperをdimension指定へ置換。`SectorCAR/physical/rank_assessment/representation_gate`本体はそのまま。元g・scalar一回、chemist ERI対称性、入力hashを確認 |
| `downstream_handoff` | input/candidate/rank/DF予算を結び、新しいown-ground/aliasへ渡すschema。旧ground/PF/RTE/shots/W/native値は埋めない |
| `optimization_call_model` | 既存call instrumentationから任意の提案A/Tの上限を算定。科学runを実行しない |
| `preflight/CLI` | Source/metadata bytesと非実行設計・fresh/retry-resumeを確認。`--execute`はdesign非実行として拒否 |

今回の技術SOURCE167件はF16の108件とR32の139件の和集合＋新3件を現在起点で固定したもの。F16 manifestからは`c_pr_aware_df_m4.py`、`c_pr_integrated_performance.py`、`c_rc2_pr_stage_b.py`の3件だけがR32の既存版へ進んでいる。差分はrのopt-in指定、追加raw signal、sampling/prepared-context等の下流拡張で、今回のarm optimizerはこれらの下流関数を呼ばない。旧SOURCE manifestを変更・再発行せず、技術manifestに両hashと版差を明記した。

167件は**技術adapter/testの依存閉包**であり、未採択の新SCF/初期fit/全下流の実行SOURCEではない。最終科学runnerの入出力・stage orchestration・physical worker dispatch・SOURCE/input closureは対象採択後に閉じる必要がある。

## 5. 下流設計・精度資格・資源

`candidate identity → physical DF → own tail law/λ → own ground/alias → PF/RTE → A/phase → RPE shots/W → native → shell込み総RZ` の順を保持する。DF誤差はenergyへ一度、PF/RTEはdimensionless signalへ、同じkernel価格のSEはW倍とする。rや校正標本数をWへ重複乗算せず、13roundを独立価格標本と数えない。

旧H4の設定はpartial ordered S2・s1/r32/K0・δ=h=.1・m0..12、CA=.00159Ha・DF reserve1/8、βtotalπ/6・βstatπ/12・各候補failure.01/軸.01/26、own-DF ground oracle・状態準備除外。P0はQiskit1.3.0/opt1/seed17/rz,sx,x,cx、approximation1、initial-zero false、MPFR160 declared phase・位相fitなし、head/X/Y shell込みfenced serial。新条件でも意味を確認して採択する提案で、RPE/alias適格性を仮定でPASSにしない。

固定rを双方へ移す検証と、双方に同じ有限r選定機会を与える検証を分ける。今回の第一案は前者。H4でのr32が他条件で最適とは仮定しない。新candidateには新価格が必要であり、旧kernel平均/shots/Wは再利用しない。

価格N・precision・固定統計政策はnull。候補と実行政策を凍結してから独立prospective cohortを採る設計とし、32/256/640を自動選択しない。母平均差・比のどちらを問うか、精度未達時のUNRESOLVED、同時比較の範囲を契約前に選ぶ。必要なら価格pilotも別の許可済み科学段階にする。今回、新しいNやCI/検定を保存rawから計算していない。

Evidence laneは未知の全law synthesis/implemented-law signalをnullとし、未資格shots/費用を確定しない。Idealは反実exact-native/ideal finite-RTE IID、価格はfinite-bit proxyという資格を保持。Actual CA、厳密enclosure、ideal-law期待価格移送、一般的優位は未認定。

H6はsector400の疎CAR・因子後のsector制限・既存PFの反復二乗/指数cacheを優先する。単一complex128行列はsector400で2,560,000 bytes、controlled8192で1,073,741,824 bytesになるため、全operatorの多数同時dense化をH4から移さない。既存`c1_h6_state_action`、Gaussian精度/declared-phase・native資産を再利用する設計だが、有限probe coverageと全operator coverageを混同しない。Native32を機械的に複製するH6 runnerは準備していない。

Fitは既存ffsim0.0.83/NumPy1.26.4/SciPy1.14.1/JAX0.4.38/Qiskit2.4.2/PySCF2.14.0、evalはQiskit1.3.0/OF1.6.1/PySCF2.7.0等の既存環境をread-onlyで利用。Artifactに環境metadataを記録する。技術検証時はaffinity32 CPU・空きRAM約54GB。独立fit/eval試験を並行し、小行列BLASは各processで1 threadとしてJAX等との過剰並列を避けた。科学用のCPU/RAM/worker/wall上限は設けず、将来の重い実行前に実資源・quota・圧力と支配負荷を再確認する。共有venv/cache/OS/他jobには変更しない。

## 6. Logical callsの導出と未確定部分

Prepared neutral inputを使う二arm optimizerについて、T=各armのtotal trial cap、P=1+T、A=受理更新capとする。

| 群 | 既存instrumentationからの上限 |
|---|---:|
| 共通load/identity/context/common物理/setup/比較/export | 7 |
| 一armのJ solve・refinement/precision/admission | 3P＋2P＋P＝6P |
| 一armの受理時＋最終独立assessmentと物理判定 | 2(A＋1) |
| 一armのdriver/endpoint診断/export | 3 |
| **二arm合計** | **7＋2[6(1＋T)＋2(A＋1)＋3]** |

A16/T128なら1629という旧上限を再導出できるが、これは提案recipeの整合確認であって、旧645の実績・未使用callを転用するものではない。新SCF/積分・中立fit・codec/identityの追加callはこの式に含まれない。初期値の経路が未採択なので合計を固定しない。

下流H4の**旧full-operator構造を保てる場合だけのworksheet**は、rank R・LD・round数M・固定校正N/armとして、共通9＋二arm×`A+B+C+D+E`。

| Stage | 旧実処理からの条件付きcall式 |
|---|---|
| A：candidate/因子/ground/law | R＋9 |
| B：13round/reference・3lane accuracy/energy等 | 6M＋12 |
| C：native trajectory・shell・serial・aggregation | 2R＋2LD＋9N＋2M＋23 |
| D：同J/U強D・両lane | 4R＋5M＋44 |
| E：binding/価格統計/summary | 6 |

旧R8/LD2/M13/N32では1231に一致する。新条件のNの根拠ではなく、実処理内訳の照合である。現下流wrapperにはrank8/LD2/full512/samples32等の固定点があり、任意R/LD/Nでこの式どおり動く実行実装は未完成。H6のstate-action/検査範囲変更には別instrumentationが必要。最終契約では必要なstageの実callから再算定し、未知の合計を埋めない。

## 7. 人工検証・passive preflight・残る採択

**72 unique PASS：新31＋既存41。** [Fit/旧回帰63 PASS](../../artifacts/research_c_near_transfer/2026-11-03/technical/fit_tests.txt)、[評価環境3 PASS](../../artifacts/research_c_near_transfer/2026-11-03/technical/evaluation_tests.txt)、[追加rank14 codec1 PASS](../../artifacts/research_c_near_transfer/2026-11-03/technical/codec_shape_test.txt)、[handoff等8 PASS](../../artifacts/research_c_near_transfer/2026-11-03/technical/handoff_tests.txt)。最後の8にはgeometry metadata3件の再確認が含まれるため、延べ75をunique件数へ加算していない。試験失敗0。

新試験はgeometry/sector/rank/LD、signed J・直交性、新初期scale/η、6軌道PR/ENTRYのJ gradient/Hessian-HVP・U tangent、rank14の504 parameter修正codec、既存arm委譲、人工sector400のCAR/一般K/元κ/scalar/物理予算、wrong input/candidate/rankの下流拒否を確認した。既存試験からinner solve/refinement・Armijo・独立終端・不適格/partial STOP・source/input drift・fresh/retry-resumeを再利用した。

6軌道微分試験は人工rank3、codecは人工rank14、sector400物理試験は人工rank1。実H6rank14のPR/ENTRY最適化や下流価格を検証したとは扱わない。保存分子arrayの`numpy.load`を禁止した試験と、完全な人工packetだけを読む既存mock worker試験を区別する。新しいnative build/compileや実分子のRPE/shots計算は0。

```bash
/usr/bin/python3 -B scripts/prepare_research_c_near_transfer.py \
  --contract docs/research/research_c_near_objective_transfer_contract_unapproved_20261103.json
```

Preflightはbytes/JSONだけでPASSし、NumPy/SciPy/JAX/ffsim/Qiskit/PySCF/OFのimportを禁止したprocessでも通った。これは未使用分子の数値資格ではなく `numeric_input_qualification=NOT_RUN`。CLIの`--execute`は非実行設計として拒否される。この設計JSONをそのまま科学runへ使うコマンドはない。

| 次の担当 | 残る具体的な作業・境界 |
|---|---|
| ユーザー／研究C | H4距離移送を第一にするか、H6サイズ移送にするかと、初期化/rank/LD/scale・強度/finite政策の科学条件を選ぶ。今回の提案を正式採択とは扱わない |
| 次のCodex | 採択条件の元packet回収/生成仕様、中立初期化、target IO/worker/downstream・校正N/精度、最終SOURCE/input/call/STOP/runtimeを閉じて実行可能な未承認契約へ進める |
| ユーザー／研究C | 完成した固定範囲の新科学実行を別途明示許可する。必要な重要科学レビューは科学実行と独立した許可で行う |

今回の技術準備と提案は完了。旧42 Claim・各科学SOURCE/raw/契約/grant/ledger/marker/STOP・元dirty checkout・他jobは変更しない。共有Evidence Map/研究概要/validation manifestは編集せず、technical pathを見つけるPROJECT_MAP・docs/scripts/library索引だけに最小追記する。Commit/push・remote SHA/tree/公開本文/主要リンク照合後、**mandatory STOP**し、新科学計算や正式レビューへ自動進行しない。

## 固定参照

[MAP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/169e327ae50960173bb2690ab98ea92c85e7a5ed/docs/research/research_c_near_objective_xdf_evidence_integration_20261103.md
[OPT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2ceb0a8825785ebd4f21241d8ef0f20cb321bf69/docs/research/research_c_near_objective_practical_comparison_execution_20261103.md
[DOWN]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/beb8de5cd8b71f80bb69556f92299435ff431c3e/docs/research/research_c_near_practical_r32_downstream_execution_20261103.md
[NEAR]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/2ceb0a8825785ebd4f21241d8ef0f20cb321bf69/docs/research/research_c_uwc_rcdf_near_objective_comparison_design_20261103.md
[H4META]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e098c54c78f589055082f9cfc2b13de50c90ca94/artifacts/research_direction_geometry_energy_difference_pilot/2026-09-25/inputs/h4_sto3g_d115_rank12_ld3_v5.json
[H6INPUT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/beb8de5cd8b71f80bb69556f92299435ff431c3e/artifacts/research_c_c1_h2_h6_science/2026-10-25/H6-run01/input_identity.json
[H6FIT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/beb8de5cd8b71f80bb69556f92299435ff431c3e/docs/research/research_c_c1_h6_corrected_fit_execution_20261025.md
[H6NATIVE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/beb8de5cd8b71f80bb69556f92299435ff431c3e/docs/research/research_c_c1_h6_native_stable_phase_execution_20261025.md
[H6CA]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/beb8de5cd8b71f80bb69556f92299435ff431c3e/docs/research/research_c_c1_h6_ca_resources_execution_20261026.md
