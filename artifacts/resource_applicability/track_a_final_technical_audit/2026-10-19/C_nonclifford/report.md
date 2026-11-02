> 公開用projection。private内部パス、原入力・回路・運用記録、ローカル補助scriptは含まない。本文に記す照合件数は原調査の報告値であり、未公開原資料を再公開した意味ではない。

# C：非Clifford RZ数の再集計可否（静的調査）

2026-10-19 JST。状態：**調査完了。既存のcompiled RZ費用を非Clifford RZ費用へ読み替える証拠は未確認。** 新しい回路生成・QPY読込・transpile・角度分類・数値再集計は実施していない。

調査結果は、公開artifact、SOURCEの保存動作、資料が明示するprivate領域でのfilename／schema／hash確認に限定する。未知の保管先を含む「どこにも最終回路が存在しない」という断定はしない。

## 1. 判定

| 確認事項 | 判定 | 得られた証拠・限界 |
|---|---|---|
| 主なA_COST H4/H6/H8にfinal RZ角度別receiptがあるか | 確認したschemaにはない | `measured_{cosine,sine}.json`はRZ/CX/depth/size、seed/event digest、compiler hash、fingerprint、wallのみ。公開CSVも角度列を持たない。SOURCEも同じprojectionを保存する |
| 指定したprivate A_COST出力にQPY等があるか | filename検査範囲では未発見 | H4/H6原run、H8原run＋継続runを限定走査。QPY/QASMや専用角度列のfilenameは見つからない。任意JSON内部の全dataや未知のbackupは網羅していない |
| H8主要PFの保存QPYがあるか | 確認済み | 3構成×2軸の6本が存在し、各descriptorのSHA256・bytesと一致。ただし全て**compile前のvalidated wrapper**である |
| H8のQPYが完成済みcompile結果そのものか | 該当しないことをSOURCEで確認 | validated wrapperをdumpし、そのQPYをloadした後にcompileする。compile後のcircuitは戻り値中には一時存在するが、保存receiptは集約metricのみ |
| native lowering／保存preparationに角度classがあるか | 調査範囲では未確認 | H6 archiveにnumeric operationを含むcodec、H8 exportに`name/parameters/qubits/matrix_sha256`がある。Clifford／非Clifford分類やfinal optimizer後のRZ angle列はない |
| 保存済みmetricだけで非Clifford数を復元できるか | 不成立 | RZ総数・gate種別count・hash・depthから個々の角度を復元できない。hashは照合情報であり角度列ではない |
| 新たなcompileなしで取得できる可能性 | 条件付き | 正しく結合された**最終compile後**回路／全RZ角度列が別途取得できればparser-only再集計を検討できる。今回、そのcoverageは確立していない |
| 非Clifford指標でもPRの結論が同じか | 未確認 | 新規集計禁止に従い評価していない。既存RZ点比較から推定・認定しない |

## 2. SOURCEと保存動作

### 2.1 新A_COST（主Track A）

固定入口は`64c85f7a833b5950ef2ccae623708b73081edcd6`の[完了報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/64c85f7a833b5950ef2ccae623708b73081edcd6/docs/research/track_a_pr_cost_batch_completed_results_20261018.md)と[新A統合報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/64c85f7a833b5950ef2ccae623708b73081edcd6/docs/research/track_a_pr_new_a_total_cost_results_20261018.md)。前者は公開対象をscalar測定表等に限定し、NPZ/QPY・raw event列等を非公開と記す。この一般的な非公開方針を、QPYが実際に生成・保管された証拠にはしなかった。

実行SOURCE `31ef62a12407010e9c1a07f9b37facedb8dff5fe`の[pr_cost_batch_science.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/31ef62a12407010e9c1a07f9b37facedb8dff5fe/scripts/pr_cost_batch_science.py)は`native.compile_task`を呼ぶ。同SOURCEの[pr_cost_native.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/31ef62a12407010e9c1a07f9b37facedb8dff5fe/scripts/pr_cost_native.py)は次を保存する。

- `trajectory.json`：event replay、seed、distribution、探索cohort。
- `measured_cosine.json`／`measured_sine.json`：整数metric、compiler hash、入力wrapper fingerprint等。
- `task_result.json`：paired axisとbinding。final circuitや角度列のserializerはこの経路にない。

H8の追加112軌道はSOURCE `87f1f0d528478ad212ba98159db06eec3c78433e`。[継続adapter](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/87f1f0d528478ad212ba98159db06eec3c78433e/scripts/pr_cost_candidate_parallel.py)は元native kernelのSHAを確認して同じ`compile_task`を利用する。元16軌道と追加112軌道、旧確認n32は区別される。

原本のA_COST technical seal → payload → native config → provider closure →代表metricというmetadata chainを追跡した。H4/H6のaccepted result、H8継続accepted resultの所在と既登録SHAも記録した。`metadata_observations.json`（原調査の非公開照合記録）に、SOURCE・input/hamiltonian/state/compiler/profileの識別子と参照pathを保存した。全raw値の再検証・科学的独立性認定ではない。

条件はlinear H4/H6/H8、隣接1Å、STO-3G、T=.8、H4 legacy rank12、H6/H8 tol-only rank19/28、symmetric-directional control、探索n8。compilerはQiskit1.3.0、`rz/sx/x/cx`、optimization1、seed17、backend/couplingなしの保存profileに結合される。compiler hashは`1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。H8 profile `622ce4…`は継承された識別子として扱い、元profile fileの現存bytesを改めて取得したとの主張はしない。

### 2.2 H8主要PFの6本のQPY

固定公開receipt [six_axis_results.json](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/64c85f7a833b5950ef2ccae623708b73081edcd6/artifacts/resource_applicability/track_a_h8_key_compile_completed_six/2026-10-17/six_axis_results.json)と、そこへ対応するprivateのdescriptorを照合した。対象は以下の両軸。

- B0+I p15/q4 S4：元SOURCE `352581103b7dbd4fa58e28af7a6ca65570d36e8e`で準備、継続runへQPY2本をcopy。
- B0+I p11/q1 S2：SOURCE `41e466e13f32e9224bb37790cc16f0ff6621c399`で準備。
- B0 p15/q4 S2：同SOURCEで準備。

private入口は`[非公開保管先]`。全6本の実在・非symlink・size・SHA一致をbyte-onlyで確認した。QPYをdeserializeしていない。公開版では個別のprivate path・descriptor等を除外した所在要約を[artifact_inventory.csv](artifact_inventory.csv)に示す。詳細な原本照合は非公開記録に保持する。

SOURCE `41e466…`の保存段階は明確である。

1. [h8_key_compile_controls.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/41e466e13f32e9224bb37790cc16f0ff6621c399/scripts/h8_key_compile_controls.py)の`build_case`がnative wrapperを検証し、`dump_wrapper`でQPY化する。
2. [h8_key_compile_handover.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/41e466e13f32e9224bb37790cc16f0ff6621c399/scripts/h8_key_compile_handover.py)の`import_prepared`が旧QPYをSHA一致でimportする。
3. [h8_key_compile_adapter.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/41e466e13f32e9224bb37790cc16f0ff6621c399/scripts/h8_key_compile_adapter.py)の`compile_axis`が`load_wrapper`後にcompileを呼ぶ。
4. [h8_b0_anchor_adapter.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/41e466e13f32e9224bb37790cc16f0ff6621c399/scripts/h8_b0_anchor_adapter.py)の`compile_wrapper`はfinal circuitから得たmetricのみをdictとして返す。

したがって「QPYがあるから追加compileなしで**保存compiled費用と同じ段階の**非Clifford RZ数を集計できる」とは言えない。compile前QPY内のnative RZを数える別指標も今回未実施であり、optimizer後のcountと同一視しない。

### 2.3 compiler本体・cache

A_COST封印provider closure中の`src/trotterlib/rte_compiled_cost.py`を実ファイルSHAと照合した。`transpile_and_measure_cost`（843行以降）は`count_ops()`でpost-transpile gate名ごとの数を取得し、`rz_count=post_counts.get("rz",0)`を返す。角度を判別する分岐はない。戻り値の`transpiled_circuit`はメモリ中には存在するが、上記callerの保存projectionへは渡されない。

同fileのpersistent payload（524行以降）はpre/post gate counts、depth/size/global_phase、compiler/hash等に限定される。persistent cacheを読む際は`transpiled_circuit=None`になる（607行）。cacheがあるという理由だけでは角度列を復元できない。

fingerprintは浮動小数点parameterを`float_hex`等でcanonicalizeしてhashする（217行以降）。fingerprintが一致すれば入力identityの照合に役立つが、保存hashのみを逆変換して角度列を取得する経路はない。

### 2.4 高次outer-PF PR系列

`1fc58aa3f5568887e0678a9be939551b0ed6523d`の[統合報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1fc58aa3f5568887e0678a9be939551b0ed6523d/docs/research/track_a_h46_higher_pf_integrated_run01_results_20261018.md)はraw NPZ/QPYの非公開を一般的に記すが、[取得報告](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1fc58aa3f5568887e0678a9be939551b0ed6523d/docs/research/track_a_h46_higher_pf_acquire_run01_results_20261018.md)は**「QPY・新NPZ出力はない」**と明記する。SOURCE `d77e439a…`の[h46_acquire_native.py](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/d77e439a16d6953cb17e6d1ef91b0d9430144848/scripts/h46_acquire_native.py)でも、`events_PRIVATE.json`とscalar metricを保存後、`del cost,wrapper`を実行する。

指定private `runtime/run01/`のfilename確認と代表2条件のschema照合もこの保存動作に整合する。この系列はordinary controlの高次**PR**探索であり、主Track Aの強化決定論PF格子の実測coverageを増やす証拠にはしない。

## 3. π/2の整数倍の分類に必要な条件

最終回路のsingle-qubit RZについて、global phaseを除くClifford角はπ/2の整数倍である、という分類を採用する場合でも、入力と判定policyが必要になる。

- symbolicなπ倍や、生成由来のexact class証拠が残れば、丸めによらない分類の候補となる。しかし今回のnative経路はnumeric parameterを要求し、H6 codecやH8 exportにはfloat hex／parameter値が保存されている。**Clifford classの証明書はない。**
- binary64数値がπ/2に「近い」場合、丸め・周期・absolute/relative tolerance・threshold近傍を未確定とする規則を事前固定する必要がある。新たなtoleranceを科学結果を見ながら選んではならない。
- upstreamの`rz`／`xx_plus_yy`、controlled diagonalやscalar phaseのparameterと、basis translation／optimization後のRZ列は別段階である。native loweringの構造だけから最終countを確定しない。
- sx/x/cxと個々のRZ角度の分類を定義しても、それはそのcompiler出力に対するgate-level指標である。最適なClifford+T synthesis費用やT数への変換は今回の証拠に含めない。

実角度の分類、閾値試験、Clifford／非Cliffordの個数比較は実施していない。人工テストもこの調査には不要と判断し、新たに実施していない。

## 4. 何を追加承認すればよいか（未実行）

**経路C1：final artifactが取得できる場合のparser-only作業。** 対象candidate/axis/cohortを固定し、complete-final-circuitまたはfinal RZ角度列のpath・SHA・compile後段階を確認する。SOURCE/input/compiler receiptとの対応、角度class規則、未確定角の扱い、取得した回路だけのcoverageを事前登録した上で、保存回路の新規非Clifford科学集計を別承認する。新compileを承認に含める必要はない。現時点ではこの前提を満たすfinal artifactは未取得である。

**経路C2：現在確認できた保存物しかない場合。** 最小対象は論文で用いる保存winnerとPR確認／探索を分けた既存trajectoryであり、全grid再探索へ広げない。保存event replay・preparation/input hash・wrapper fingerprintと同一回路を復元し、同じQiskit/runtime/compiler profileで一回compileし、final circuit／final角度列と分類receiptを新領域へ保存する必要がある。sampling・signal・SCF/DF/state生成は不要とする契約を作り、回路復元／生成、transpile、非Clifford集計、照合用の元RZ再現検査を明示承認の対象とする。**今回実行していない。**

C2でも欠測PF候補の新規全格子compile・新ランキング・費用最適化・shot再計算は含めない。モデル単価だけのPFについて、実測を新たに追加しない限りcompiled非Clifford数を直接実測済みとは扱えない。どの候補を実測するかの判断は別に必要である。

既存のcompiled RZ指標と、その適用範囲・非Clifford感度未確認を明記して**原稿準備へ進める技術的余地はある**。非Clifford指標での勝敗維持を論文で主張するには追加証拠が要る。これは論文の科学的採否を認定する結論ではない。

## 5. coverage・保全・成果物

旧確認n32全回路、全強化決定論PF格子、全過去run／backupのfinal artifactについて、今回の限定調査で完全coverageを確立していない。固定6commitのtree listingにはQPY/QASM拡張子pathはなかったが、これはprivate全体の不存在証明ではない。実測・model区分は維持し、未取得を読了済み／PASSへ変換していない。

- [artifact_inventory.csv](artifact_inventory.csv)：20行の所在・保存段階・可否一覧。
- `metadata_observations.json`（原調査の非公開照合記録）：A_COST SOURCE/input/compiler chainとH8の6QPY byte binding。
- `private_filename_observations.json`（原調査の非公開照合記録）：対象を限定したfilename/stat観測。
- [source_manifest.json](source_manifest.json)：固定commit:path、使用した実local fileのSHA256。
- `private_evidence_stability.json`（原調査の非公開照合記録）：実際にbyte確認した59fileのbefore/after SHA一致。調査対象外のfileや他jobの全体凍結を意味しない。
- `metadata_audit.py`（原調査の非公開照合記録）：標準libraryだけのmetadata照合script。Qiskit import／QPY load／角度集計なし。

書込先はC専用新領域のみ。既存SOURCE／raw／grant／ledger／marker／STOPへの書込、他process操作、環境変更、公開pushは0。private内部pathやmetadataを含むため、この調査folderをそのまま公開する承認は得ていない。研究の数値再計算、新compile、レビュー、独立確認へ進まず、このsubtaskは停止する。

必須資料の読取は共通AGENTS、固定commit AGENTSとPROJECT_MAPによる入口確認を実施した。大きな現状 synthesis は主担当の分担指示に従い1551–2059行を担当し、読了内容を主担当へ共有した。個人による全文読了とは記録しない。歴史的CPU／memory制約は現在の利用者policyへ移送していない。

次の作業（今回未実行）：**主担当Codex**がA/B/D/Eへ本結果を統合してmandatory STOPする。**ユーザー**は非Cliffordの新集計を論文化条件とする場合のみ、C1またはC2の具体的対象と分類policyを別に承認する。
