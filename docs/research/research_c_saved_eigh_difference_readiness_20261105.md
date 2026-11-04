# 研究C：保存HとH_eighの独立差分診断・科学契約準備（2026-11-05）

**独立した4-call診断のSOURCE・未承認実行契約を固定し、passive preflightを通過した。実H4の配列decode・差分評価は0件。公開照合後MANDATORY STOP。**

Branchは `codex/research-c-saved-eigh-difference-readiness-20261105`。起点は指定された `56d2f85e471d2ab3927c9c9568f31034df143b26`、tree `eb8f142a354c5ce08413e6e95d5ad870ab1b0a32`。既存Hermitian bridge、入力、producer記録が揃っており、このcommitを採用した。独立したsparse worktree `/tmp/prt-c-saved-eigh-difference-readiness-20261105` で作業し、元checkout・並行Compiled→raw作業を変更していない。

## 固定する対象・保証

対象は当該PR endpoint（SHA256 `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`）の保存sector Hだけ。

|役割|固定path・array|SHA256／schema|
|---|---|---|
|H_saved|`artifacts/research_c_near_practical/2026-11-03/run01/physical/PR_update016.npz`、`H`|`ebdebe8f2c3f156c8202834181e143298d667490b3f33fc611dc0fab97b976cd`、`<c16`／36×36|
|sector順序|`artifacts/research_c_near_practical/2026-11-03/run01/physical_reference.npz`、`sector_basis_indices`|`92a8d27bdebb53937f6361c106a13ef90b9110930d4bf4ddb577796cfb208739`、`<i8`／36|

Linear H4・1.00 Å・STO-3G、Nalpha=Nbeta=2、number CDF8／元順LD2の当該保存入力を使う。既存partial S2／s1/r32/K0／h=.1／m0／own-DF oracle等の政策は変更しないが、今回は回路・oracle・RPE評価を実行しない。

保存complex128のreal/imag各binary64成分をexact rationalとした行列をAとする。固定NumPy1.26.4のproducerは `np.linalg.eigh(matrix)`、UPLO未指定＝Lである。このsemanticsに対応する行列Bを

\[
B_{ij}=(H_{\mathrm{eigh}})_{ij}=\begin{cases}
\Re A_{ii}&i=j,\\A_{ij}&i>j,\\\overline{A_{ji}}&i<j
\end{cases}
\]

と定義する。研究C全体の `H_target` は未採択のまま。今回の対象は **D=A−B**、

\[
S=\sum_{ij}\big((\Re D_{ij})^2+(\Im D_{ij})^2\big)\quad\text{（exact rational）},\qquad
\|D\|_F=\sqrt S\in[L,U],\quad \boxed{\|D\|_2\le\|D\|_F\le U}.
\]

Arb192の外向きsqrtを使い、`0<=L<=U`、`L²<=S<=U²` もexact arithmeticで検査する。Uはoperator normのrigorous上界であって厳密値ではない。大きな差でも数値資格が成立すればそのまま記録する。小ささの閾値は設けず、非zero成分を0へ丸めず、平均化・元ファイル上書き・近似同一視を行わない。Aを非Hermitianなまま扱えるノルム診断であり、Aのgroundやgapは定義・評価しない。

## Producer確認・再利用

保存Hは `c_near_practical` のphysical callback → evaluation worker → `c_pr_integrated_evaluation.Evaluation.assess` が、元gからの補正κ、核反発、8個のCAR frameを足した `sum(G)` を保存したもの。数値ground側は `prepare_arm → ground(sector H) → loader.ground → MolecularBackend._ground → np.linalg.eigh(matrix)`。詳細とNumPy仕様は [先行技術報告](research_c_hermitian_oracle_bridge_readiness_20261105.md) およびそのproducer_provenanceを参照する。

今回も保存producer SOURCEの9ファイル、H／基底／endpointとSOURCE・契約・既存provenance・旧STOPの9入力／metadataをbytesで照合した。現コードが保存SOURCEと同じ生成経路を示すこと、ground契約のPR referenceが固定Hであること、candidate・basis・own-DF policyを確認する。原因や差の数値的大小について新しい判断は行わない。

|再利用|今回の用途|
|---|---|
|`c_hermitian_oracle_bridge.matrix`／`matrix_id`|exact component解釈と数学的matrix identity|
|`eigh_problem(..., uplo='L', dtype='<c16')`／`hermitian`|既存L規則による構成とexact Hermitian検査|
|`difference_bound`／既存Arb sqrt|exact Sとoutward U。ノルム計算を再実装しない|
|`c_direct_m0_oracle.exact_real`|bridge内部のbinary64→Fraction。ground資格付け関数は呼ばない|
|`c1_stage1_execution.sha/read/save/array_record`|bytes binding・exclusive証拠保存・配列identity|
|`c1_stage3_integrated.Calls`|既存attempt／COMPLETE／FAIL journal・first failure STOP|
|既存source manifest／Git固定commit照合方式|新しい独立SOURCEを固定し、実行前にコード・入力を照合|

旧Direct m0のrunner・固定SOURCE・契約は変更していない。既存の汎用source freeze関数は見つからず、既存manifest形式・hash・Git照合を用いて今回の小さな実行経路を固定した。承認証明・grant発行等は追加しない。

新規 [c_saved_eigh_difference.py](../../src/trottertracks/c_saved_eigh_difference.py) はNPZ・基底schema、exact変換、既存bridgeへの接続、Frobenius区間下端、証拠出力を追加する。読込時はhashを確認した同じbytesをBytesIOからdecodeし、Hとoccupation labelsだけを取得する。physical_referenceのH、endpointのJ/U、保存ground vectorはdecodeしない。保存基底と固定36 occupation labelsの順序まで一致を要する。

## 固定科学SOURCE・未承認契約・calls

- 科学SOURCE commit：**`4a3228881173710f9c6b4d0d13085a77fbc59668`**（tree `a7105e2d57eef9744090e6d195bc432430e4d048`）。
- [SOURCE.json](../../artifacts/research_c_saved_eigh_difference/2026-11-05/technical/SOURCE.json)：SHA256 `0653edb67d03da3e1ea49abfb6780d9d36bbbbea996cc15e0d4305a7e0ec86ca`。
- [新規契約](research_c_saved_eigh_difference_contract_unapproved_20261105.json)：SHA256 `9b5c16785bebc3c047ef3f5feadee6fbacdbf8ebcdb475d1bddf5305703eca6b`。
- 契約状態：**`UNAPPROVED_EXECUTABLE_AFTER_SEPARATE_PERMISSION`**、executable=true、実行経路ready。これは科学実行許可ではない。

SOURCEにはentryを含む実行経路14ファイルとproducer対応9ファイルを固定した。人工実行で観測したlocal runtime module13件が含まれることも確認した。未使用のoracle・回路・CA関数は、同じmoduleからimportされても今回の科学scopeに含めない。Python3.11.1、NumPy1.26.4、python-flint0.8.0／FLINT3.3.1、Arb192を固定する。SOURCE commitに準備時のNOT_READY契約が残るが、実行する契約は上記の最終契約であり、同commitの不変コードとSOURCEへbindする。

**最大4 logical calls、各1回、次の順で固定する。**

|順序・kind|処理内容|主な証拠|
|---|---|---|
|1 `metadata_load`|開始済みjournal内で保存producer／NumPy L規則／input／旧STOPのmetadataを読込み・bind|`producer_binding.json`|
|2 `arrays_load_exact`|固定bytesからHと保存基底を読込む。shape／dtype／有限性／基底順序を検査しexact変換|`arrays.json`、`H_saved_exact.json`|
|3 `eigh_semantics`|固定L規則でBを構成し、保存Aとのidentityとexact Hermitian性を検査|`H_eigh_exact.json`|
|4 `difference_norm`|既存bridgeでexact SとUを取得。Lと包囲を検査し、exact Dと結果を保存|`difference_exact.json`、`result.json`|

各stageは前段の結果に依存する。exact変換をarray loadへ、差分とsqrtを単一norm評価へまとめ、同じ仕事を別callに計上しない。SOURCE・input bytesのpassive preflight、contract/sourceのsnapshot、journal／STOP／resource出力は管理処理として別callを消費しない。開始した診断のmetadata読込みは、既存Callsと同じ粒度で1 callとする。完全成功時は4 attempts／4 COMPLETE。大きな差も正当な診断結果であり、失敗扱いにしない。

Fresh出力は **`artifacts/research_c_saved_eigh_difference/2026-11-05/science-run01`**。今は存在しない。mkdirはexclusive、既存directoryがあれば拒否し、retry/resume=0。旧Direct m0の残12 callsを一切転用しない。

実行前のpassive入口（今次実行済み）は次のとおり。

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 /tmp/prt-c-pr-aware-df-zsxx-enclosure-preparation-20261027/.venv/bin/python -B scripts/run_research_c_saved_eigh_difference.py
```

**別途ユーザーがこの固定4-call scopeの科学実行を明示許可した後だけ**、同じcommandへ `--execute-science` を付けて一度実行する。追加設計やtarget採択は不要。今回の依頼は準備許可であり、この実行commandを起動していない。

## Failure・資源・数値資格

SOURCE/input hash、固定scope、NumPy／Python package metadata、SOURCE commit、fresh出力をpreflightで検査する。未来の実行では固定array schema、有限成分、保存基底順序、exact変換、L規則、Hermitian性、差分identityと包囲を検査する。最初の例外・不整合・包囲失敗でdependent callsを止め、partial_failure・journal・既存の完成証拠・STOPを保存する。KeyboardInterruptも対象。SIGKILL等のprocess消失ではjournalを保全してINCOMPLETEとし、STOPが残らなくてもresumeしない。自動alternate candidate／shift／再実行はない。

支配処理は36×36、1296個のcomplex成分のexact差分・二乗和でO(n²)、sqrtは1種類。各段階に逐次依存があり、この単一小行列へprocess poolやGPU転送を加える利点は見込まない。既存変換・差分関数を一度ずつ使い、重複再計算を避ける。人工Hamiltonianによる経路検証を先に済ませた。実H4の性能測定やspeedupは主張しない。

検査時CPU affinity32、RAM約62GiB／available約52GiB、可視 `/dev/nvidia*` deviceなし。新worktreeはdisk残量に合わせsparse展開した。CPU／RAM／worker／wall-timeの人工上限は設定しない。実quota・空き資源・非干渉を実行前に確認し、他job・共有venv・global設定を変更しない。

## 人工検証・passive結果・保全

**新規人工32 PASS。既存関連127 PASS／継承4 FAIL。合計159 PASS／4 FAIL、新規failure 0。** 起点commitの既存worktreeでも4 FAIL／127 PASSを再現し、失敗node IDが同じと確認した。旧pre-executionテストが消費済み `science-run01` をfreshと扱えず `FRESH_NO_RETRY_RESUME` で止まる4件であり、旧raw・SOURCE・テストを変更してPASS化しない。

新規テストはexact Hermitian差0、上三角変更、下三角変更、対角虚部、複素共役、0.1と最小subnormalのexact binary64変換、微小差2^-100・大きな差2^40、既知のexact S、operator norm包含、非平方数sqrtの外向き幅、wrong hash/schema/basis、包囲失敗時のpartial STOP、permission前拒否、fresh拒否、retry/resume拒否を検査した。eigensolver／oracle関数を禁止した人工経路も完走した。人工NPZだけを数値decodeし、実H4データはhash・JSON・NPY headerのmetadata確認だけである。

[validation.json](../../artifacts/research_c_saved_eigh_difference/2026-11-05/technical/validation.json) とbaseline/current pytest原本参照。[preflight.json](../../artifacts/research_c_saved_eigh_difference/2026-11-05/technical/preflight.json) は `execution_path_ready=true`、`scientific_execution_permitted=false`、`scientific_calls=0`、`arrays_decoded=0`、missing=[]。数値library importを禁止した子processでもpassive preflightを検証した。future science output・journalは作成していない。

共通運用指示の原本 `research_project_instructions_unified_2026-10-25.md` の§8.1・§8.4・§10を実際に確認した。原本は元checkoutの未収録ファイルであり、固定公開版と偽らない。原本hash、旧SOURCE/input保全、変更pathの検査は [preservation.json](../../artifacts/research_c_saved_eigh_difference/2026-11-05/technical/preservation.json) に残す。旧NPZ・SOURCE・raw・契約・grant/ledger/marker/STOPと並行作業を保全し、旧15-callの3 attempts／2 COMPLETE／1 FAILを維持する。

技術 [STOP.json](../../artifacts/research_c_saved_eigh_difference/2026-11-05/technical/STOP.json) は実H4科学0・未承認・mandatory STOPを記録する。公開後にremote SHA/tree、報告全文、SOURCE／契約／主要blobを照合し、[public_verification.json](../../artifacts/research_c_saved_eigh_difference/2026-11-05/technical/public_verification.json) に記録する。これらは科学的provenanceと成果物照合であり、許可を証明する仕組みではない。

**次の責任：ユーザーがこの独立4-call差分診断の実行を明示許可するか判断し、許可後にCodexが固定契約を一度実行してSTOPする。今次はMANDATORY STOP。** Ground保証、target採択、Direct m0、X/Y、期待bias校正、Stage C、actual CA、重要科学レビューへ自動進行しない。
