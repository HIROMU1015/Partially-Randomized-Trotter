# PR価格規則の適用判定（D2、2026-11-16）

新しいPR-Pauli／PR-DF configurationに対し、同じ物理条件の直接価格、既存モデル価格、または具体的な不足を返す受動的な入口を実装した。保存モデル6構成・156軸と保存総額12件をexactに再現し、保存資料・人工構成を使う17検査を通過した。新規path・回路build・compile・fit・Q3は各0件。期待価格やCIの資格は追加していない。

実装SOURCEは `c1f95e0806f5478a1ee5ce9ecb148ce9d10461aa`。コードは `src/trottertracks/pr_price_rules.py`、固定登録表は `configs/pr_price_rules/h4_saved_rules.json`、JSON入口は `scripts/quote_pr_price_rules.py`。本成果の公開commitは、このSOURCEとは別の最終公開commitを使う。

## 判定と再利用

| 入力の変更 | 処理 |
|---|---|
| shot配分だけが変わる | 単価をそのまま再利用。供給されたshotでexact加重和を返す。配分の生成・精度判定は行わない |
| κや設計Δの宣言・canonical IDが変わるが、生成済み物理条件が同じ | 実際のT/q/R/step、event recipe、piece配分、head/tail、辞書、scalar規則、実装・compilerが一致すれば再利用。κからscheduleを推測しない |
| q/Rが変わり、同じ実step・同じpiece構造の既存規則で扱える | 候補固有モデルを適用し、較正q/R範囲と外挿する変数を返す。eventのstep条件が較正Rと異なることも表示 |
| 実PF step・角度条件が変わる | 当該stepでの係数根拠がないため価格不足。同じ式を無条件には延長しない |
| formula、signed piece、相対event配分が変わる | head/piece境界・event係数の根拠が不足。新公式に対応する較正／監査が必要 |
| split・辞書・scalar・SOURCE・lowering・wrapper・compiler条件が変わる | 不一致箇所を返す。同名LD2でもPauliとDFを混用しない |

照合対象は保存H4 linear 1.00 Å／STO-3G／tol-only DF10、P0の測定付きfull controlled Hadamard wrapper。Qiskit 1.3.0、rz/sx/x/cx、opt1、seed17、symmetric-directional control、総Tでのscalar一回適用を保持する。PR-DFは新ZZ lowering、PauliはD2 central-rotation loweringを別契約として要求する。H/state/snapshotと元fragment順は固定binding・SOURCEで照合する。SOURCEの別名や変更を自動的に同等と認定しない。

## 登録した方式別根拠

| 系列 | モデル価格／不足 | 保存直接価格の入口 |
|---|---|---|
| DF LD1 | 保存E_I[S]＋凍結残差からなる26軸表をそのまま再利用。新しい物理点ではE_I[S]が不足するため、構造量を捏造して係数へ加算しない | m0/m3/m6のP2 fresh較正、独立m4。P1・replayをpoolしない |
| DF LD2 | 自身のfull-wrapper `θ0+θq q+θE qR`。較正m0/m3/m6、q=1..64、R=1..40 | 凍結モデルの保存4 path平均・元path receipt |
| DF LD7 | 自身のfull-wrapper `α+βq`、R=1。較正q=1/64。Rが変わるとβのhead／tail分離が不足 | m0/m6較正各4 path、独立m3各軸2 path |
| DF LD9 | 26軸モデルなし。LD7係数を転用しない | m6各軸4 path平均のみ。AUTOでは2軸が価格付き、残り24軸と総額はnull |
| Pauli LD1 | 26軸モデルなし。A/H/Eまたは候補固有のfull-wrapper係数が不足 | B0 m0の1 path。配線証拠の役割を保持 |
| Pauli LD2 | C2凍結 `β0+β1q+β2qR`。較正q=1..8、R=66..4205。長いqへの適用はモデル外挿 | B1-W m0、5較正cell、独立m5/m6/m9。paired q2の相関を保持 |
| Pauli LD3 | AのS0-r1監査による `A3+H3q+E3qR`。較正cell q8/R660 | Aの2 path×X/Y。cellはm=Noneのまま |
| Pauli LD5 | AのS0-r1監査による `A5+H5q+E5qR`。較正cell q8/R4 | Aの2 path×X/Y。m3やm12のreceiptへ付け替えない |

全登録の固定commit・path・selectorは登録表と [C_HANDOFF.json](C_HANDOFF.json) にある。Pauli LD3/LD5ではAのwrapper終端角・非零marginの確認範囲も保持し、新しいTにはその監査不足を返す。較正範囲の内側であっても検証済みとは扱わない。DF LD1の構造＋残差と、他モデルのfull Cを重ねて加算しない。X/Yは同じpathの別軸であり、独立標本数に加算しない。

## 不足の返し方

各不足行にはm/axis/q/Rと、`needs.reasons`、`missing_axes`、`calibration.coefficients`、埋めるための作業を返す。

- DF LD1の新物理点：不足量はその軸の保存定義によるE_I[S]。既存残差の適用可能性とは分ける。
- DF LD7のR変更：不足は別々のhead q傾き・tail qR傾き。R=1のβはその和のみ。同じ候補・stepで独立なq/qR特徴行を持つ較正が必要。
- DF LD9：m6の4 path平均を再利用し、全R=1を維持するなら別qの候補固有較正で切片／傾きを識別する案。Rも変える場合はfull-rankの(1,q,qR)が必要。
- 新しいPR-DF分割／step／公式：当該候補のfull-wrapper平均で係数を識別する較正と、独立した記述的確認。単一q/Rから3係数を作らない。
- 新しいPauli分割／step／公式：A/H/Eのwrapper・head境界・normal tail本体を分解する候補固有監査、または対応するfull-wrapper較正。LD3/LD5の数値係数は転用しない。

これらは不足内容の提示であり、追加取得を投入する指示ではない。件数・seed・cohortの実行範囲は後続指示で固定する。

## B2向け仕様と検査

`PriceRules.quote(request)` はconfiguration本体、明示的な実装契約、任意のshot列を受け取る。CLIではconfigurationを `{commit,path,selector}` で渡せる。返却は各軸の `DIRECT_FINITE_COHORT`／`MODEL_REFERENCE`／`NEEDS_PRICE_EVIDENCE` とexact単価、両方の代替証拠、外挿、不足、元locator。全軸が価格付きの場合だけ加重和を返す。

`policy=MODEL` は凍結モデル価格を保持する。保存総額の再現とB2のモデル比較にはこれを使う。`AUTO`（省略時）は同一物理条件の直接平均が一意なら優先する。`DIRECT` は直接値だけを返す。直接cohortが複数ならpoolせず、`direct_cohort`を指定する。m4/m3等のholdoutをモデルへ再投入しない。直接値に有限cohort以上の資格を与えない。

[SAVED_REPLAY.json](SAVED_REPLAY.json) は156軸とDFの9配分＋Pauliの3保存配分のexact一致を記録する。[TEST_RESULTS.txt](TEST_RESULTS.txt) は17検査の結果。shot・κ宣言のみの変更、q/R適用と外挿、実step変更、辞書・law・scalar・formula・piece・lowering・compiler変更、LD7の識別不足、LD9の部分価格、Aのcellとroundの区別、paired相関、不正なidentity／shot、非正モデル価格を確認した。CLIの本番読取・出力保存経路も通した。保存26軸の新しいfitや価格規則の変更はない。

## §10.5・ロードマップ・終了

起点 `23160cc876807c0300be3de811f767b5810640a7` のcleanな既存D2 branchで作業した。Git全branchの関連履歴、B2の `rpe_method_optimizer.py`／`saved_rpe_cost_catalog.py`、Cの `price_model_validation.py`／`rpe_mean_price_model.py`、AのLD3/LD5監査・価格表、D2の既存モデル／直接receiptを照合した。既存B2の保存単価読取と同じexact単位・加重順を使い、探索器・Q3実行器は増設しなかった。Git blobの読取cacheを再利用する実装方針とし、producerをimport・起動しない。

差分観測ではB2の候補生成／optimizer作業が進行中だった。これは本入口の外部利用側であり、当該worktree・job・出力を変更していない。同等のPR価格適用判定の公開履歴は確認範囲では未同定。Aの最新LD5成果は新規較正せず固定参照で再利用した。

統合への差分は、B2が候補を価格付き／価格不足に区分し、取得依頼の内容まで機械的に出せること。signal・Q3の再利用可否や候補の生成はB2側のまま。正式E[C]、CI、PCG64 joint-lawへの移送、順位・winnerの認定は行わない。ロードマップ本体、旧raw・SOURCE・モデル・独立holdout・旧STOPを変更せず、公開確認後に停止する。
