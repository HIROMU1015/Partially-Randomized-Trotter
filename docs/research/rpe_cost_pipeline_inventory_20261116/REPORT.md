# RPE方式別コスト比較：自動化済みの工程と人手が残る工程の棚卸し

2026-11-16 JST、B2（Claude Code、主agentのみ。作成者自身の確認で、独立レビューではない）。保存資料とcodeの読み取りだけで作成した。新規の科学計算・compile・Q3・最適化器の実装は0件。

目標は次のとおり。ハミルトニアンHと要求精度を入力すると、各方式（決定論PF・PR-Pauli・PR-DF）が構成とshot配分を探索する。そして、共通の精度条件を満たす中で、方式ごとの最小達成費用と採用構成を返す。この目標に対して、H4で実際に使われた経路を固定commit/pathでたどった。工程ごとの機械可読表は [PIPELINE_INVENTORY.csv](PIPELINE_INVENTORY.csv) にある。

## 0. 結論

- **自動化済みなのは「固定した1候補の中」の計算である。** 具体的には次の3つ。
  - 配分生成（`generate`）
  - Q3評価（閉包・total DP・保存・検査まで）
  - 固定した有限menu内の逐次探索と停止（`search`）
- **候補の「外側」はすべて人が決め、起動し、受け渡している。** 外側とは、公式・幅・L_Dの選択、信号・価格の取得、配分族・予算・上限の設定、menu外の次候補、集計への登録である。
- 各段階の実行は「固定callerとINPUTSを書く→commit・push→1回実行→REPORT・HANDOFF→mandatory STOP」の形で進む。次の段階は、ユーザーの新しい指示で始まる。
- 方式別最安の表は、Cの集計器が出す（`saved_rpe_cost_catalog`）。ただし入力（固定commit/path）は結果ごとに人が追加している。新しいschemaが来るたびに、読み取りcodeも追加している。
- **別のHに替えると、最初にH・状態の読込で止まる。** 読込関数が `n_qubits=8`・電子数(2,2)を固定し、続く検査が `rank==10`・状態36次元を要求するためである。その先でも次の点で止まる。
  - 入力fileのprivateな絶対pathとhashを手で固定している。
  - PFの信号は保存PF stepに依存している。
  - PRの分割descriptorはH4のcontext fileに保存されたものを使う。
  - 精度契約・targetの定数がH4固有である。
- 改修は、手作業の受け渡しを減らすもの（共通の結果schemaと集計の自動取込）から始めることを勧める。次に、方式に依らない配分・Q3探索driver、候補空間の宣言と分割の導出、共通の価格規則、の順で進める。別Hへの一般化は、H4で外側のloopが回ることを確認してから行うのが既存実装を最も活かせる（§5）。

分類の記号：
- **自動**：入力が決まればcodeが判断・実行する。
- **人手**：実装はあるが、人が入力・起動・受け渡しをする。
- **未実装**：codeが無い。
- **決定**：研究者が決める事項。
- **未確認**：今回たどれなかった。

## 1. Hの入力から最終比較表までの実際の経路

H4の現行比較で使われた順にたどった。行頭は共通の工程、方式で異なる箇所は「PF／PR-Pauli／PR-DF」に分けた。PFは決定論PFの2系列（Full PF、Truncated PF B0I p7）を指す。Full-H pure RTEは今回の範囲外とした。

### 1.1 Hと対象状態の準備（分類：人手。別Hでの生成は未確認）

- **共通**
  - 保存H4 tol-only 1e−8 DF10のsnapshotとreceiptを使う。置き場所はprivate directoryで、`/home/AbeHiromu/projects/h4-followup-validation/20261018/private/inputs/tol_only_snapshot.npz`、sha256 `12219b7d…`。
  - 読込関数は `155529e20:scripts/h46_m5_tol_saved_input.py` の `load`（16〜31行）。receiptとhashでH・stateを照合し、`PhysicalSector.spin_sector(n_qubits=8, nelec_alpha=2, nelec_beta=2)` を固定している。
  - 入力fileの絶対path・sha256・byte数は、各runのINPUTSに手で固定している。例：`e0b3cad2a:docs/research/d_pr_df_ld_split_results_20261115/ACQUISITION_INPUTS.json` の `saved_input_files`、private provider root、承認済みprivate config。
  - snapshot自体を分子から作る工程は、今回の経路には無い（未確認）。
- **PF**：信号moduleは同じ読込関数を使う（`1adb7e65a:src/trottertracks/pf_shortlist_mean_completion.py` の `load_inputs`）。
- **PR-Pauli**：DF tailからP0 Pauli辞書を作る。LD1/LD2は `fd111f989` の `df_head_pauli_tail_p0.py`。LD3/LD5は同じ手順を分割ごとに凍結したrunnerで実行した（LD5は `d6385593e`→辞書 `6255fbc42`）。A自身の計画（`a50dce8f1`）に「P0と信号のcodeはprefixを(1,2)に固定しているが、一般化の差分は小さい」とある。
- **PR-DF**：各分割のdescriptor（λ_R、s_R）は、保存された `H4_context.json` の `prefixes.<L_D>` を使う（`f89fdc9d4:docs/research/d_pr_df_next_split_choice_20261116/REPORT.md`、context SHA-256 `8396605f…`）。runtimeでは `run_pr_df_ld_split.py` の `restore` が `rank==10`・`len(state)==36` を要求する（D2 head `460d77d8f`、87〜89行）。

### 1.2 精度条件・target（分類：決定。以後の各INPUTSへの転記は自動）

- 次の項目をユーザーが採択した。各runのINPUTS（例：`b4296b110:docs/research/pf6_cost_budget_q3_results_20261115/INPUTS.json` の `adopted_evaluation`）と、Cの比較条件（`4afda9e48:docs/research/h4_saved_rpe_cost_catalog_20261116/INPUTS.json` の `comparison`）に同じ値が入っている。
  - 十進 εE＝0.00159360010199040 Haと、そのexactな二乗τ
  - Z1＋T1 total decoder
  - 全事象平方loss
  - 保存ideal raw・IID二項law
- targetは保存DF10 full DF-Hのsolver energy proxy −2.1663874486347634 Ha（`bba150834` の `PHYSICAL_INPUTS.json`）。これはH4で計算済みの値で、別Hで計算し直す工程は今回の経路に無い。

### 1.3 方式別の候補空間・候補の選択（分類：決定。codeによる選択は未実装）

| 方式 | 現在の候補 | 誰がどう決めたか（根拠） |
|---|---|---|
| PF | Full PF・B0I p7 × S2（16h/7）・Yoshida4（16h/3）・M5（16h）の6候補。粗幅p7 Yoshida4（32h/5）を追加 | 6候補はDの保存4件binding（`acea1118a`）とAのfollow-upに基づく検証用shortlist。Bの幅計画（`7f27e18f0`）は29の保存configurationを `WIDTH_SIGNAL_COVERAGE.json` に索引したが、別幅の信号・価格が足りないとした。粗幅Y4はDの計画（`d01c780d0`）。幅・公式を列挙して選ぶcodeは無い |
| PR-Pauli | LD1、LD2、LD3、LD5 | L_Dは「headに置く二体DF fragmentの本数」と定義された（`a50dce8f1`）。LD3はAの計画が隣接分割として提案した（同）。LD5は段階1（`bf0a930fd`）で構成されたが、選んだ理由は固定資料に記載が無い（未確認、ユーザー指示と推定） |
| PR-DF | LD1、LD2、LD7、LD9 | L_D 0〜10の棚卸しはD2の計画（`f42f6e9b9`）。LD7はD2の次分割計画（`f89fdc9d4`）が「LD3対LD7」に絞って選んだ。LD9は、LD7の結果に対して計画の事前分岐「LD7がLD2より安ければLD9」を適用した（`5c5d87ab3` REPORT） |

### 1.4 configurationとschedule（分類：生成器は自動、起動と受け渡しは人手）

- **共通**：common κのmapper `paper_event_mapper`（`510fbacd0`。Bの `pr_rpe_rmse_controller.generate`、`b9b9ffa62`）が、`target=κλ_R²T²`、`R=max(J, ceil(r/q))` で各roundのRを決める。RPEは h=0.1、M=12（13 round）に固定。
- **PF**：A・Bの記録にある保存configurationとPLANを使う（例：`06a15ee45:docs/research/pf_shortlist_mean_completion_20261115/run01/<候補>/PLAN.json`）。表示幅からqを作り直さない（`7f27e18f0` §1）。新しい幅のconfigurationは、別途の生成と信号が必要。
- **PR-Pauli**：Aの段階1 runnerが辞書・λ_P・configuration・13 roundのscheduleを作る（LD5：`39a1fc7df`、configuration `4d6e2c9c…`）。
- **PR-DF**：D2がdescriptorとmapper入力を固定し（`04090592a`）、元のmapperでconfigurationを1件作る（`33a308bd7`、LD9）。

### 1.5 信号（13 round平均）（分類：人手）

- **PF**：Aの `scripts/run_pf_shortlist_mean_completion.py`／`src/trottertracks/pf_shortlist_mean_completion.py`（`049f6e1f8`→保存 `06a15ee45`）。保存PF step（`OPERATORS.npz`、`714247619`・`5fb2c32e3` の保存cell）から平均を作る。module自身の説明が「No new PF step」で、新しい幅・新しいHではstepの生成から必要になる。
- **PR-Pauli**：Aの段階2で13 roundの平均を作る（LD3：`8c8ce542e`→packet `cdbce383f`。LD5：`c67469ad4`→packet `ac46d100e`）。
  - Bが取り込むには、分割ごとにintakeの拡張が必要だった。LD3では `rpe_pauli_signal_intake.accept_saved` を固定SOURCEに拡張した（`2de33a4f4:docs/research/pauli_ld3_vp_q3_results_20261116/REPORT.md` 31〜33行）。LD5でもassemblyと辞書のdispatchを最小限追加した（`b6d4dea49` SCOPE_AND_PREFLIGHT）。
- **PR-DF**：D2が元のnormalizerのgate付きで自身の信号を作る（LD9：`d1d2b847f`→`2f717a807`）。

### 1.6 価格規則の選択（分類：決定。方式間で規則が異なる）

| 方式 | 規則 | 根拠 |
|---|---|---|
| PF | 実際に観測したaxisは直接compileの価格。未観測のm7/m8/m10〜m12は、ユーザー採択の条件付き閉包価格 | D `acea1118a`、C `e4458cd75`（60軸）、B2 `1002f1dfe`（156軸を結合） |
| PR-Pauli | 較正cellでの分解 `C=A+H·q+E·q·R`。S0-r1の監査を通す | LD3：cell Q8R660（`f25a05f8a`）。LD5：cell Q8R4（`b8bcdf97b`） |
| PR-DF | full wrapperの2点モデル `C=α+β·q`（m0・m6で較正、m3は記述的holdout） | `4326422a4`（LD7） |

- どの方式もE[C]・CIの資格は無い。ランダム化方式はモデル参照価格である。PFは直接価格と閉包価格の混合。同じ証拠水準で方式を比べる規則は、まだ決まっていない。

### 1.7 価格の較正・取得と26軸への適用（分類：人手）

- **PF**：Dのcompile script（`acea1118a` の `scripts/run_full_pf_shortlist_prices.py`、`scripts/run_truncated_p7_prices.py`）でreceiptを作る。Cが閉包を計算し、B2が `AXIS_TERMS.json`（`1002f1dfe`）に結合する。
- **PR-Pauli**：Aの較正cell driver（LD3：`158bbcc7b`→binding `3074433d1`。LD5：`1097c37f9`→`aedf55d9a`）で取得し、`PRICE_26AXIS` を作る。
- **PR-DF**：D2の `run_pr_df_ld7_prices.py`／`freeze_pr_df_ld7_prices.py`（`8d4503e00`）で較正、凍結、holdoutの順に進め、`AXIS_PRICES.json` を作る。

### 1.8 配分生成（分類：生成は自動。入力は決定）

- `generate`（Bの整数価格版 `06f4210e1`、正の有理数版 `e094b2d60`）。各軸の上限・価格・予算・規則（A_SHAPE＝形を保つ、B_PRICE＝単価で重み付け）を与えると、breakpoint列から予算内で最大の点を返す。D2（`freeze_pr_df_ld_split.py` の `GENERATOR`）とB（`a844ac25f`）も同じ関数を使っている。
- 上限（N0・VP・V0）、予算（V0費用の3/4、PFの達成費用）、配分族（A/B、局所の±shot）は研究者が決めた。各callerに定数として書かれている（例：`b4296b110` の `scripts/run_pf6_cost_budget_q3_fixed.py` の `N0`）。

### 1.9 Q3（分類：固定caseの中は自動、run単位の準備は人手）

- **共通の評価器**：kernel `710aa69b6`（Z1＋T1、Q3）と高速total DP（`9e2947782`、8 worker）。worker自身が原API・partials・資源を保存する。保存結果の検査は `validate_result`（`e7d24c219` の `run_p7_s2_local_shot_search.py`）。
- **方式別の経路**
  - PF：`b4296b110` の `scripts/run_pf6_cost_budget_q3_fixed.py`。
  - PR-Pauli：`7733089b8` の `scripts/run_pauli_ld3_vp_q3_fixed.py`、`a844ac25f`、`b6d4dea49`。
  - PR-DF：D2の `scripts/run_pr_df_ld_split_q3_resume.py`（`005dc8c01`）。固定SOURCE view `27e70efe5` のworktreeとPython 3.11.0rc1が必要。
- runごとに、固定callerとINPUTSをcommit・pushしてから1回だけ実行する運用である。callerは対象の列・費用・ラベルをassertする。

### 1.10 次候補の選択（分類：menuの中は自動。menuと構成は決定）

- **配分**：Bの `search`（`e7d24c219` の `run_p7_s2_local_shot_search.py`、`a844ac25f` の `run_pauli_ld3_pf_budget_q3_fixed.py`、B2 `cf19fc36f`）が、事前に固定した有限menuを費用順または固定順に評価する。最初の適合点・call上限・UNKNOWNで止まる。menuの外へ進む規則は無い。
- **構成（幅・L_D）**：計画文書の規則（`a50dce8f1` の隣接分割、`bf0a930fd` の損益分岐の目安、`f89fdc9d4` の事前分岐）を人が書き、ユーザーが採択する。codeは無い（未実装）。

### 1.11 停止（分類：run内は自動、段階の切れ目は運用）

- 停止はrun内のcallerの規則による（`FIRST_FEASIBLE_IN_COST_ORDER`、`SCIENTIFIC_CALL_LIMIT`、`MAJOR_GATE_OR_PERSISTENCE_FAILURE`）。段階ごとのmandatory STOPは運用の規則で、次の段階はユーザーの指示で始まる。

### 1.12 集計（方式別最安）（分類：比較は自動、入力の登録は人手）

- Cの `scripts/collect_saved_rpe_costs.py`＋`src/trottertracks/saved_rpe_cost_catalog.py`（実装 `a171839a1`、現在のhead `4afda9e48`）。固定した入力集合の中で、適合・モデル不適合・価格待ち・Q3待ち・不一致を区別し、exactな最安点を返す。
- 入力は `INPUTS.json.sources` に結果ごとに手で追加する（HANDOFFに「開始後の新成果を自動的に追跡せず」とある）。未対応のschemaでは止まる（`saved_rpe_cost_catalog.py` 186行）。新しいproducerのschemaには最小限の読み取りcodeを足してきた（LD7・LD3 PF予算・LD9の各追記）。

### 1.13 受け渡し（分類：人手）

- 各producerがREPORT・HANDOFF・`C_RESULT_JOIN`／`C_HANDOFF`／`C_COMPARISON_JOIN` を、それぞれ別のschemaで出す。次の担当の入力は、ユーザーの指示文書に書かれた固定commit/pathで指定される。

## 2. 分類のまとめ

| 分類 | 工程 |
|---|---|
| 既に自動 | 固定case内のQ3（閉包・DP・保存・検査）、`generate`、固定menu内の `search` と停止、mapperによるscheduleの計算、Cの集計内の比較と最安点の選択、各INPUTSへの精度契約の転記 |
| 実装はあるが人が入力・起動・受け渡し | H4入力の読込と入力pathの固定、辞書・descriptor・configurationの生成runner、信号の生成（3方式別）、価格の取得・凍結・26軸適用（3方式別）、Q3 callerの作成とcommit、Cへの入力追加とreaderの拡張、分割ごとのintakeの拡張 |
| 未実装 | 候補空間の列挙と次構成の選択（幅・公式・L_D）、menu外へ進む配分探索規則（例：適合している間は下げ続ける）、未評価候補に必要な信号・価格・Q3の自動列挙、別Hでのsnapshot・descriptor・PF step・targetの作成の接続、方式共通の価格規則 |
| 研究者が決める事項 | H・状態・DF方針、target、εE・decoder・loss・law、方式の定義（L_Dの意味など）、候補空間の範囲、価格規則とその証拠水準、配分族・上限・予算、科学計算の予算、停止と報告の範囲 |
| 未確認 | H4 snapshotの元の生成手順、target proxyの計算手順、LD5を選んだ理由 |

## 3. 具体例

### 3.1 p7 S2・16h/7：m12各4→3→2→1

| 段階 | 結果 | 誰が何を決めたか | 既存codeだけで同じ判断ができるか |
|---|---|---|---|
| m12=4（B_PRICE、435,481,090 RZ） | 適合（`b4296b110`） | Bが計画（`7f27e18f0` §3：β=3/4×2規則）を提案し、ユーザーが採択した。列は `generate` が決めた | 予算・規則が与えられれば同じ列を作れる。予算の割合と規則の採択は人 |
| m12=3（415,721,884） | 適合（`0b33b4afd`） | Bが近傍探索を計画し（`9992c594d`：S−e12、S−2e12、S−e11、S−2e11、振替、最大8件）、ユーザーが採択した（`e7d24c219`）。`search` が固定順の1件目を評価し、「初めて安い適合点」で停止した | 計画したmenuの中ではできる（停止を含む） |
| m12=2（395,962,678） | 適合（`93ab9dbbe`） | 計画menuの2件目（S−2e12）だったが、探索はm12=3で止まっていた。**ユーザーが別途この1件を指示**した（`f60fafeca` SCOPE） | できない。「適合したらさらに下げる」規則が無い。callerは列と費用を定数で固定している |
| m12=1（376,203,472） | 不適合（`cbc471c91`） | **ユーザーの指示**（`cdf3b1f91`）。計画menuの外（変更量は1〜2が上限） | できない。menuの外で、列・費用（`376203472`）をcallerにassertしている |
| （同じ手順をFull PFへ） | m12=2適合、m12=1不適合（B2 `a006b8e15`・`7349f6f9b`） | ユーザーがp7の結果を見てFull PFにも指示した | できない。候補をまたぐ転用の規則が無い |

この連鎖で、codeが決めたのは「menu内の順序と停止」だけである。menuを越える一歩はすべてユーザーの指示だった。

### 3.2 PR-Pauli：LD3とLD5

| 段階 | 結果 | 誰が何を決めたか | 既存codeだけでできるか |
|---|---|---|---|
| LD3を候補にする | `a50dce8f1` | Aが保存descriptorとLD1/LD2の関係から隣接分割を提案し、ユーザーが採択した | できない（選択の規則は文書にあり、codeに無い） |
| LD3 段階1（辞書・λ_P・configuration・schedule） | `e77c37989`（configuration `dd5b881d`） | Aがrunnerを凍結し実行した | runnerはあるが、prefixの一般化と起動は人 |
| LD3 段階2（13 round信号） | `029866330`（packet `cdbce383f`） | ユーザーが承認し、Aが実行した | 人手 |
| LD3 VP Q3 | 適合（`2de33a4f4`） | ユーザーの指示でBが実行。intakeを分割用に拡張した | 拡張codeが必要だった |
| LD3 価格（Q8R660） | `f25a05f8a` | Aが較正cellと規則を凍結し、ユーザーが承認した | 人手 |
| LD3 PF予算4列 | FULL_PF__B_PRICE適合（`fb4d78781`） | ユーザーが2つの予算を指定し、Bが逐次探索した（`a844ac25f`） | 予算が与えられればmenu内の探索は自動 |
| LD5を候補にする | `bf0a930fd` | 固定資料に選択理由の記載は無い（未確認）。段階1でAが「信号より先にheadの価格を測る」順序を損益分岐の目安で勧めた | できない |
| LD5 段階1〜3 | 信号 `68fa77b1b`、価格 `b8bcdf97b` | Aが実行（runnerはLD3のものを再利用） | 人手 |
| LD5 Q3（LD3と共通の2列） | 2列とも適合（`1c4a83d1b`） | **2列をユーザーが指定**した（`b6d4dea49` SCOPE）。intakeのdispatchを最小限追加した | できない。列の選択とintakeの拡張が人 |

LD3とLD5の比較（FULL_PF__B_PRICE：541,936,296対541,743,906 RZ、VP：1,051,862,455対872,241,518 RZ）は、既存codeでは「同じ列を両方の価格で数える」までしか自動化されていない。分割の選択・配分の選択・intakeの接続は人が行った。

## 4. H4固有の依存と、別Hで最初に止まる箇所

| 依存 | 場所（固定commit:path） | 別Hでの扱い |
|---|---|---|
| H・状態の読込が8 qubit・(2,2)を固定 | `155529e20:scripts/h46_m5_tol_saved_input.py`（`load`、26行） | **最初に止まる**。snapshotとreceiptのschemaが同じでも、sector定義が合わない |
| rank 10・36次元の検査 | D2 `460d77d8f:scripts/run_pr_df_ld_split.py`（89行） | 直後に止まる |
| 入力fileのprivateな絶対path・hash | `e0b3cad2a:.../ACQUISITION_INPUTS.json`、各runのINPUTS | 手で差し替えが必要 |
| PFの保存PF step | `1adb7e65a:src/trottertracks/pf_shortlist_mean_completion.py`（`OPERATORS.npz` 依存、「No new PF step」） | step生成が必要（今回の経路に無い） |
| PRの分割descriptor | `H4_context.json`（`prefixes.<L_D>`、`f89fdc9d4` が参照） | 新しいHのdescriptor生成が必要。Pauliでは保存の無いLD4がそもそも作れなかった（`a50dce8f1`） |
| target・事前分布・τ | `bba150834` の `PHYSICAL_INPUTS.json`、Cの `comparison`（system=H4、snapshot/H/state hash、energy prior、target） | 再計算と再採択が必要 |
| 固定候補・ラベル・配分ベクトル | Bの `P.LABELS`・`N0`（`b4296b110`）、p7の局所探索の `LABEL`・費用（`e7d24c219`）、D2の `V0`（`freeze_pr_df_ld_split.py`）、Pauliの `VP` | 定数の書き換えが必要 |
| PR-DFのQ3経路 | D2の固定view worktree（`/…/d2-pr-df-ld-split-q3-source-20261115`、`27e70efe5`）とvenvのpath | 環境依存 |
| 価格の証拠 | PFは候補ごとのcompile receiptとCの閉包、PRは較正cellごとの取得 | すべて取り直し |
| Bの接続module | `h4_df10_rpe_connection.inspect_saved_point(expected_system='H4', expected_rank=10)`（`b9b9ffa62`） | 既定値がH4 |

## 5. 開始時の決定事項、プログラムが決める事項、改修の順序

### 5.1 利用者が開始時に決める事項

- H（分子・幾何・basis・DF方針・sector）と、対象状態およびtargetの定義。
- 精度条件：εE、decoder、loss、law（ideal IIDなど）。
- 比較する方式と、その定義（L_Dの数え方、headの並べ方など）。
- 探索範囲：PFの公式と幅の格子、PRのL_Dの範囲（「導出できる全分割」でもよい）、配分族、各軸の上限、予算の決め方。
- 価格規則とその証拠水準：較正cellの数、holdoutの有無、外挿をどこまで許すか。
- 科学計算の予算：信号・compile・Q3の最大件数。
- 停止と報告の範囲：有限集合内の最安なのか、などの表現。

これらを一度に承認すれば、その範囲内では段階ごとの承認を求めずに進める。範囲を広げる場合だけ、利用者に戻る。

### 5.2 承認された範囲内でプログラムが決める事項

- Hのfragment分解と方式の定義から、選べる分割と候補を列挙する。
- 各候補のconfiguration・scheduleを生成する（既存のmapper）。
- 次にどの候補・配分を評価するかを規則で選ぶ。下限による枝刈りや、損益分岐の目安の自動計算を含む。
- 配分の生成（`generate`）、Q3の実行、検査、停止。
- 保存結果の再利用（同一の候補・列・契約）。
- 未評価候補に足りないもの（信号・価格・Q3）の一覧を出す。モデル不適合とは区別する。
- 方式別の最安表を作り、探索範囲を明記する。

### 5.3 価格の扱い（研究目的に合わせた提案）

方式間を同じ物差しで比べるため、価格は方式ごとに別の規則を採るのではなく、共通の規則で扱うことを勧める。規則の中身は次のとおり。

- 1 shotの価格を `C(m,a)=A+H·q_m+E·q_m·R_m` の形で表す。PFは `E=0`、PR-DFの現行2点モデルはR=1での特殊例とみなせる。
- 各候補につき少数の較正cellを直接compileし、1 cellの独立holdoutを記述する。この手順を全方式・全候補に同じ形で適用する。
- 探索中の順位付けはこのモデル価格で行う。そのうえで、外挿が占める割合（例：m7〜m12の割合）とholdoutの相対誤差を必ず併記する。
- 方式ごとに選ばれた最安候補だけ、長いroundの直接compileで確認する。PFの現行の直接価格と閉包価格は、この確認の資料として再利用できる。

これでE[C]の資格は与えないが、証拠水準の揃った方式間比較になる。PFだけを直接価格、PRをモデル価格で比べる現状の非対称は、なくすか、少なくとも明示する。

### 5.4 改修の順序（既存実装を最も活かす順）

1. **結果schemaの共通化と、集計の自動取込**
   - 対象：Cの `saved_rpe_cost_catalog`、各producerのjoinファイル。
   - 内容：producerが共通の `C_RESULT_JOIN` v1（候補・配分・26軸価格・Q3の正本locator・状態）を出す。集計器は指定したdirectoryやbranchから取り込み、「信号待ち／価格待ち／Q3待ち／モデル不適合／判定保留」を出力する。
   - 効果：科学計算は不要。手で入力を追加する作業と、schemaごとのreader追加がなくなる。
2. **方式に依らない配分・Q3探索driver**
   - 対象：Bの `search`・`generate`・`cached_record`・`validate_result`、各方式のQ3経路。
   - 内容：Q3経路を方式別に登録する（PF：`run_pf6_cost_budget_q3_fixed`、Pauli：`run_pauli_ld3_*`、PR-DF：`run_pr_df_ld_split_q3_resume`）。そのうえで、配分族×予算、局所の降下規則（「適合している間は下げる」を含む）、call上限、停止をdataで指定できるようにする。
   - 効果：run単位のcallerを手で書く工程がなくなる。3.1の連鎖は、承認された範囲内で自動で進む。
3. **候補空間の宣言と分割の導出**
   - 対象：A・D2の辞書／descriptor・configuration runner、`paper_event_mapper`、Bの幅索引（`7f27e18f0` の `WIDTH_SIGNAL_COVERAGE.json`）。
   - 内容：方式ごとの候補空間（PF：公式×幅の格子、PR：fragment数から導いたL_Dの範囲）を宣言する。各候補の工程状態（configuration・schedule・信号・価格・Q3）を管理する。分割ごとのintake拡張は、辞書identityを引数にとる形へ一般化する。
4. **共通の価格規則**
   - 対象：AのS0-r1分解監査、D2の2点モデルと凍結・holdout、Dのcompile、Cの閉包。
   - 内容：5.3の規則で、較正計画→取得→fitting→holdout→26軸適用を1つの入口にする。方式ごとの最終確認compileも同じ入口から出す。
5. **外側の候補選択の規則化**
   - 対象：Aの損益分岐の目安（`bf0a930fd`）、D2の事前分岐（`f89fdc9d4`）。
   - 内容：次に評価する構成を、隣接分割、下限による枝刈り、未評価の最も安い下限などの規則で提案し、承認済みの科学予算の範囲で実行する。
6. **別Hへの一般化**
   - 対象：`h46_m5_tol_saved_input.load`、trotterlibのDF・Hamiltonian生成、PF step・descriptor・targetの生成。
   - 内容：sectorとrankをreceiptから読む形にする。H・snapshot・descriptor・PF step・targetを入口から作る。H4で1〜5が回ることを確認してから行う。保存成果のあるH6で1件通すのが次の確認になる。

1と2が、現在の作業でいちばん手間がかかっている受け渡しを直接減らす。3〜5で「何を次に評価するか」をcodeに移す。6はそれらがH4で回ってから進めるのが、作り直しが最も少ない。

## 6. 範囲・資格

- 今回は保存資料とcodeを読んだだけである。新規の科学計算・compile・Q3・最適化器の実装は0件。各結果の科学的な資格（モデル判定、E[C]・CI無し、十分N・G未認定）は元のREPORTのとおりで、変更していない。
- 固定commitは2026-11-16 03:25 JST頃のremote（Bのhead `1c4a83d1b`、Cのhead `4afda9e48`、D2のhead `23160cc87`、Aのhead `68fa77b1b`・`b8bcdf97b`）で確認した。以後の更新は反映していない。
- Full-H pure RTEは今回の範囲外とした。
