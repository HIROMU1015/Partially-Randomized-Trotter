# 受入済みDIAGNOSTICS専用COMPARE：技術仕様

この分岐は `COMPARE_FROM_ACCEPTED_DIAGNOSTICS`。旧 `COMPARE_REUSE_PR`、実行済みDIAGNOSTICS、科学SOURCE35444654…を変更しない。新runner SOURCEは `46a3b33e3717c4fadf11846e922caf17d10156d9`。比較の科学実行は未承認・未実行、PR witness不足により契約はBLOCKED。

## 固定入力とscope

元契約digest `f71e52ce6f42cc17d1a97f0cbdfac311da57c80b49ee8ea69ebe34fb4d45bfd9`、元completion SHA `cc7ebfee979b2e34eb4327aa8a8c6e20ea53bc46871f97c83d0391db5e7ec90f`、元SOURCE35444654…を固定する。linear H2/1 Å/STO-3G/tol-only DF1e-8/rank3、T=.8、ε_sig=.0005、P=0、α_axis=.025、system4/ancilla4、Full prefix3/PR prefix1/K4/outer32、指定4候補だけ。共通QR native・経験的adaptive allowance・旧FP reference条件付き利用の政策IDを継承し、別精度・別q・別系に拡張しない。

元DIAGNOSTICSの実行承認は履歴証拠として読む。新実行の承認には使わない。現在の実パス・SOURCE・入力・runtimeを持つ[COMPARE_contract_BLOCKED.json](COMPARE_contract_BLOCKED.json)は新しい契約digestを持ち、`reuse_witnesses=[]`のため実行不可。

## 現在許可するmetadata検査と将来の科学load

`h2_accepted_diagnostics.inspect_origin` はstdlibだけで、以下を確認する。

1. 元契約digest、SOURCE、4候補/物理/compiler/bindingと元completion全byte SHA。
2. 元approval・grantファイルとCONSUMED marker内grant、approval digest、政策採択、元ledgerのkind/連番/非返金/amount/予算。
3. 元SOURCE268 fileのGit blobとSHA、元SOURCE/input/runtimeの封印byte。
4. 2 prep receiptと全4候補のJSON/NPZ、各result SHA、candidate/prefix/prep/operations hashとoperation streamのmetadata対応。
5. corrected/raw列、reference ID/保存bias schema、log_B、adaptive inputの単位/coverage/欠測/元input digest。uやN/Gは評価しない。
6. ZIP/NPY headerのmember名・shape・dtype・payload長、occupation basisとwire mappingの整数metadata byte。

NPZのprimary/independent複素vector本体は今回decodeしていない。NPY headerは6次元occupation表現とFull1列/PR2列を要求する。physical spin sectorの4次元とoccupationの6次元を混同しない。basisは[3,5,6,9,10,12]、bit-reversed qindicesは[12,10,6,9,5,3]。

将来の新grantの`bundle_load=1`だけが `load_for_backend` へ入れる。ここで初めてallow_pickle=falseのNumPy decode、fresh setupのbasis/qindicesとの同一性、finite/schemaを検査する。保存independent vectorをFull native correctnessへ、保存AdaptiveInputs/bias/log_Bを会計へ渡す。pathの再propagation、exp(log_B)での再正規化、reference再計算/再結合、norm/workの再測定はしない。元resultは変更せず、新runへコピーする場合も全byte同一とし、`accepted_diagnostics_reuse.json`へ元SOURCE/digestと継承fileを明示する。

新setup1/prepare2は、原preparationの完全live objectが保存されていないため、後続native buildに必要なobjectを構成し、既存prep/operations hashと照合するために必要。元診断8 pathを計算し直すためではない。

## 新分岐のexact call予算

| kind | call | 単位・内包関係 |
|---|---:|---|
| bootstrap | 1 | 外部承認・新grant消費後の科学backend import |
| setup | 1 | 保存snapshot restore/H/state/occupationの構成とidentity検査 |
| prepare | 2 | prefix3/1のfresh object作成と元prep receipt照合 |
| bundle_load | 1 | 4候補を含む既受入bundleのdecode/schema/provenance検査・adapter復元 |
| diagnose / path_signal / reference_rebind | 各0 | 元4/8/4を再実行しない |
| native_validation / compose | 各2 | Full S2/Y4のみ |
| lowering | 10 | S2 q4の4 piece＋Y4 q2の6 piece |
| wrapper / compile / transpile | 各4 | Full2候補×cos/sin。transpileはcompile内部 |
| simulate | 12 | candidate当たりnative low/high2＋各軸before/after4 |
| allowance / accounting | 各4 | 固定4候補の新共通会計。今回は実行0 |
| ranking | 1 | 全4結果がeligibleで完全な場合だけ |
| sampling / request_factory / rte_event / occupation_trajectory | 各0 | 保存PR単価を完全identity成立時だけ再利用 |
| 新full-H reference / quantum shots / q探索 | 各0 | 実装対象外 |

上位と内部のcallを併記しており、表の和を独立実験数とはしない。[機械可読予算](call_budget.json)が正。追加の曖昧なreserveはない。bundleのmetadata/hash事前検査は科学callではなく、科学vector decodeはload1へ明確に分離した。元grant・元診断の枠は新予算へ移さない。

## 科学backendと停止

新backendは旧backendを別fileへ固定コピーし、新controlへ接続した。setup/prepare/steps/allowance/account/rankのAST一致を人工テストで確認した。診断/referenceメソッドを除去し、native検証はFullだけを許可する。Fullの独立expected vectorは保存archiveから読む。sampling側に入る候補はこの分岐では拒否する。

PRはprepare/load後にもfresh prep/operation/scalar/schedule/runtime identityを照合する。PR witnessが欠測・不一致なら、科学import前のpreflightで開始を拒否する。会計の欠測・不適格が一つでも出ればrankingへ入らずpartial STOP。途中失敗時の枠は返金せず、retry/resumeしない。

新承認はUSER_EXPLICIT、**新契約digest**、固定政策、exact budgetに対応する新unused grant・新run_rootを必要とする。元grant IDの再使用、使用済みroot、予算変更、module/native混入、結果改変を拒否する。現在はgrant/approvalを発行しておらず、scientific execution authorizationはfalse。旧mockのexecute拒否も維持した。

## PR witness形式

`H2_PR_COMPILED_IDENTITY_BYTE_JOIN_V1` は旧16 fieldの一致条件を保持するlegacy projectionと、16 field全てのhistorical-to-common証拠、元accepted resultの実byte hash、完全event payloadを要求する。保存単価は固定Git cc7404ed…のoriginal projectionに結合し、現在のworktree上の可変metadataを正本にしない。

eventはSOURCEの生成規則通り、piece順の`[RTEEvent.to_dict()]` nested listのcanonical JSON digestを各taskの保存event_digestへ結合する。order/component/phase、application index/role/順序、pieceごとのevent数、cohort/paired cost/correctness scopeを検査する。N2/N4/stratum要約をfull bodyとして受け入れない。

runtime証拠には版表示だけでなくworker observed load manifestを要求する。runtime identityはinterpreter/package inventoryとsys_pathを対象にし、run固有の未使用pycache prefixを除く。現在の保存Python3.12と採択Python3.11には宣言byte差があり、この差をpolicy名変更で隠さない。別runtimeで比較すること一般を禁止する結論ではなく、今回の厳格な保存単価identity条件の未解消事項である。より狭いcompiler関連closureの同等性を採用するなら、証拠と別の明示判断・SOURCE/契約が必要。

人工witnessの完全例と各拒否条件は新testで検証したが、実PR witnessを作成/採択したわけではない。証拠fileの意味・時点・歴史的anchorを手作業の技術監査でも確認する必要があり、形式的JSONの追加だけで科学identityが成立するとはしない。
