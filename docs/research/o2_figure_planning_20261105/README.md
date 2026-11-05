# O2：図表候補と保存データの対応計画

2026-11-05着手・2026-11-06完了 JST / LOCAL。図表作成・追加分析の対象をユーザーが選ぶための技術準備。
基準台帳commit `2779710a86d380bb484accc51d3dccba183370b8`、研究統合commit `3380f6de0d86c0ac1db9aeb6d97ab5d328dd43bd`。
正式共通指示 `research_project_instructions_unified_2026-10-25.md` の§8.1/8.4/10を確認した。
既存23scope・24候補表参照・28literal・24比較・8matching・84固定参照をread-onlyで再利用する。
`CURRENT_CLAIM_SCOPE_EVIDENCE_SUFFICIENT` は引き継ぐ記述的判定で、今回再審査していない。

## 対応表の使い方

| 成果物 | 用途 |
|---|---|
| [figure_candidates.md](figure_candidates.md) | 各候補の問い・役割・優先理由・重複・限界。caption/Results本文ではない |
| [figure_data_plan.csv](figure_data_plan.csv) | study/comparison/candidate alias・固定selector・元列・軸/系列・条件/cohort・単位を辿る |
| [figure_readiness.csv](figure_readiness.csv) | 保存値の利用可能性・不足・追加分析の要否と限定範囲 |
| [existing_figure_assets.csv](existing_figure_assets.csv) | 実在を確認した旧図/元表/静的コードと再利用条件。空asset_refsは図の存在を主張しない |
| [VALIDATION.json](VALIDATION.json) | 基準と追加固定commit/path/blob/bytes、技術検査。科学PASSではない |
| [validate_figure_plan.py](validate_figure_plan.py) | 既存checkerと最小拡張によるmetadata/literal検査のみ |

figure_candidate_idは計画の仮IDで、元研究・実験・図枚数の新認定ではない。
`candidate_selectors` の各bindingは `ref_id → selector → columns/pointers` の順に読む。
ref_idは原台帳のSOURCE_INDEX、追加refだけVALIDATIONのsupplemental_fixed_refsで解決する。
CSV where/where_inは文字列完全一致・条件の積、JSON pointerは絶対RFC6901。
text selectorは原reportの節/表のlocatorであり、機械的に抽出済みの数値表ではない。
`literal_preview` は対応確認用の一行/field引用。全候補表や全観測の複製ではない。
`catalog_candidate_refs` はTABLE_REFERENCE/LITERAL_EXTRACTへのaliasで、図に使う独立標本でも新しい候補でもない。
実際の表示範囲はcandidate_selectorsと原比較policyで決め、全結合して新winnerを選ばない。

## 利用可能性と優先順位

図表候補は33件。これは候補の列挙であり、作成枚数は確定していない。

| 分類 | 候補数 | 意味 |
|---|---:|---|
| `READY_WITH_CONDITIONS` | 26 | 保存値に原条件・仮定・cohort・表示限定を伴う |
| `READY_FROM_SAVED_DATA` | 2 | 保存単回価格・既存診断を直接表示できる |
| `UNCONFIRMED` | 1 | 共通bindingの対応が未確認 |
| `INSUFFICIENT_COMPARABILITY` | 2 | 意図する同条件比較として不適切 |
| `NEEDS_NEW_ANALYSIS` | 2 | 意図する量が未保存で数値再解析/識別検討が必要 |

優先候補はDF03/04の強PF・時間/精度依存、DF07のB3負結果、DF09のH8固定確認、
PA01–03の最新Pauli分子別比較、XP02のP3条件付き横断、XP04の定義差、ME01/03–06の限定機構例。
順位は可用性と研究上の役割に関する定性的な編集準備であり、科学候補の新ランキングではない。
情報が重なるものはoverlap_withで示す。主図/補足/形式/枚数/配置は未確定。

## 比較と表示で保つ条件

Nは解析上の十分量子shot数、nは古典価格trajectory標本数。
Cは測定付きfull Hadamard wrapperのcompiled RZ/shot、G=N_cos C_cos+N_sin C_sinは元保存RZ×shot。
G_SEもRZ×shot。P=0、C_eff・PF誤差係数C・native T/CX/runtimeを同一価格としない。
単位が異なるN/C/G/B/bias/headroomは別軸/表列の候補とする。
時間Tの単位は元SOURCEの規約を優先し、別研究の無次元時間を原子単位へ自動換算しない。
複数精度の価格共有は同じ標本の再引用。元shared_price_groups、cohort、結果版を保持する。
探索/固定確認/確認後min、DIRECT/MODEL/EXTRAPOLATED、旧33/40と拡張33/40と境界R、旧H8aliasを混合しない。
空欄/null/NOT_ACQUIRED/UNRESOLVEDはゼロでない。分数・decimal表示を勝手に丸めず、未保存B/C/比を逆算しない。

横断候補は原df_pauli_matchingのラベルと全matching_recordを引き継ぐ。
XP03のMATCHEDは同一Pauli programのcontrol価格だけ。表現単独の因果効果でない。
XP01/P3はCONDITIONAL_MATCHED、P3のfinite-bit gap=0/rho=0/B候補coverage/normalization仮定を保つ。
DF B3 one-body保持とPauli純RTE全非identityランダム化はCONTEXTUAL。
DF12/DF10、H8T1.6/別分子PauliT.8の同条件結合は不適切。最新best同士のbindingはUNCONFIRMED。

## 既存図の再利用と不足

旧H4の精度/transfer図、H4/H6長時間図、Pauli event図は固定blobの存在を確認した。
元条件の図としての再利用候補であり、今回はレンダリングや座標の科学的再監査を行っていない。
H6 event図の旧Hybrid水平線を最新r32へ再ラベルしない。B3の既存q/delta図は費用比較図とは別。
P1/P2/P3/H8C/境界Rは保存表・JSON/reportを優先し、未確認plotを存在する図として登録しない。
生成器にはG/SE/比/選択の再計算を含むものがあるため、静的確認だけで起動していない。

ME10のPR平均Cはaggregate未保存（個別価格はある）。完全な三量図を選ぶ場合だけ別途限定再解析が必要だが、
DF09のN/G図や個別価格の表示は保存値で準備できる。ME09の因果寄与率は未同定で、必要対照のscope自体が未決定。
XP05は共通H/state/reference・候補集合・cohort対応未確認。新科学計算を自動の次工程にしない。
境界R完全SOURCEのprivate保全等、原台帳のアクセス制約を引き継ぐ。

## 検査・保全・引継ぎ

`python3 docs/research/o2_figure_planning_20261105/validate_figure_plan.py --repo .`

既存validate_catalog/validate_synthesisを変更せず再利用し、ID・参照・固定blob・列/selector・literal・matching・原status保全だけを検査する。
新科学計算、数値再解析、N/C/G/SE/比/平均/順位、sampling、build/compile、詳細科学レビュー、図生成、論文執筆は実施しない。
原台帳・35行map・SOURCE/raw/QPY/contract/grant/ledger/marker/STOPは無変更。
新directoryと入口2箇所だけを公開し、remote SHAと公開全変更fileのblob/本文を照合後mandatory STOP。

次の担当はユーザー：作成したい候補、図表で答える問い、固定条件/cohort、原仮定の表示、既存図の流用か保存値からの描き直しかを選ぶ。
追加分析を望む場合はその具体的範囲を別途判断する。Codexはこの指示から図生成・分析・原稿・レビューへ自動移行しない。
