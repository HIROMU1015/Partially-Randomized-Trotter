# H4 D2：Pauli control方式比較の準備（科学実行なし）

2026-10-24 JST。D1と同じ保存Pauli-PF singleton／Pauli-PR trajectory_index0について、generic whole-body controlをcentral-rotation controlへ置き換える候補を実装した。**人工12件PASS、metadata検査8件PASS。実H4 QPY読込・build・compile・科学grant・N/G再会計はすべて0。**

[無効契約](../artifacts/non_df_h4_d2_preparation/2026-10-24/contract_DRAFT.json)は2対象・将来の新compile最大2回を提案する。実行権限はなく、production dispatcherも未実装。今回の「次の作業に進んで大丈夫」は、直前に提示したD2準備への承認として扱った。科学実行へ自動移行せず、mandatory STOP。

## 基点・固定条件・既存証拠

基点 `4e172d69271e8956e095b8e284766de0a9f7e563` とそのD1 result_summaryをGitHub connectorで実取得した。remote/local Git blob `e96d9e06a9f7ac9849278d9f131c4f3aac7b4c90` が一致する。前回のpush失敗記録は当時の履歴として変更しない。

元H4科学SOURCE `0c329b30a81accb35c212be8defa66fee0dd0f12`、保存bodyを構築したD1 SOURCE `48d4d27a05a529e05b11599f5f3c671302826617`、完了D1 SOURCE `9843b803878db6c3aee7ab5e7c6bb4818572c944`。今回のbranchは `codex/non-df-h4-d2-preparation-20261024`。新しい**科学SOURCE/grant/runtime sealは未発行**であり、今回のcommitは準備コードと資料の来歴である。

Linear H4、隣接1.00 Å、STO-3G、system8 qubit＋ancilla8、S2、T=.8、q8、r4、K4、cosine。Pauli184非identity項・PR prefix27を維持する。比較背景のDFはtol-only1e-8/rank10・PR prefix1＋one-bodyで、D2ではDFを再構築しない。保存した分割・law・event列・角度・wire・wrapperを変更しない。Qiskit1.3.0、basis rz/sx/x/cx、opt1、seed17、backend/coupling/layout/routing指定なし。状態準備費用を除くfull Hadamard wrapperである。

| 対象 | 保存trajectory ID | D1 generic RZ（引用） | D2 central RZ |
|---|---|---:|---|
| Pauli-PF singleton | `d761627d56a4454e644358fef8907b8a7db28c4feeb44130224c20e4e50aae4d` | 255500 | 未測定 |
| Pauli-PR index0 | `93ef5e4e28b499472c8ba9329f6cb139f63c5de071279783b3cca3319793fc97` | 9652 | 未測定 |

価格は[D1 metrics](../artifacts/non_df_h4_d1_source_execution/2026-10-24/metrics_table.csv)の引用で、PR256本平均ではない。D2価格・差・比は計算していない。D1で旧full価格4件が再現済みなので、generic対照の再compileを提案範囲に含めない。

## 実装候補と位相の対応

[candidate module](../src/trotterlib/h4_d2_pauli_control.py)は保存bodyの明示的なoperandだけを順に読む。real H4には今回適用していない。許可する命令は、既定LieTrotter設定の非identity・単一項PauliEvolutionとliteral X/Y/Zだけである。未知の命令、複数項、非既定synthesis、symbolic/複素/非finite値、classical操作は拒否する。名前だけがxである任意GateをXと認定しない。lazy `.definition` を要求しない。

SOURCEの各回転 `exp(-i c t P)` は、Pauli基底変換Bとparity compute Vに対して

`(I⊗B†V†) controlled-RZ(2ct) (I⊗VB)`

へ変換する。右端が先に作用する。XはH、YはSdgの後HでZ基底へ移し、active wireの最後をparity targetにする。control=0ではcompute/uncomputeが相殺しI、control=1では元の回転になる。Pauli labelは右端がsource qubit0で、各gateの実際のqargsへ写す。複数の非可換命令を交換・整列しない。

productのX/Y/Zは同じ位置と順序でCX/CY/CZへ写す。bodyの集約済みglobal phase φは**P(φ)をancillaへ一度だけ**置く。定数・negative-product・次数由来phaseをeventから再加算しない。`U=e^(iφ) U_m…U_1` に対し、control liftは積を保存するため、最終bodyは理想演算として `diag(I,U)` となる。元のcosine wrapperを `already_controlled=True` で再利用する。

source instructionごとの出力範囲・label/係数/時刻/wireと集約phaseをsidecarに残す。保存SOURCEのfloat operandを使うが、CRZ角度生成やcompileの丸め誤差certificateを新設したものではない。この局所恒等式・人工数値比較を、全H4 compiled programの厳密保証に格上げしない。

既存のPauli emitterは非制御body、DF diagonal emitterはRZ/RZZ中心で、任意Pauli wordの制御転送をそのまま提供しないため、局所liftだけを新設した。wrapper/compiler、D1人工fixture・RTE保存event変換、SHA/provenance、既存native witness/予約機構の設計は流用する。旧sourceは書き換えていない。

## 人工検査と適用範囲の制限

[tests](../tests/test_h4_d2_pauli_control.py)、[最終12件](../artifacts/non_df_h4_d2_preparation/2026-10-24/artificial_tests.json)、[log](../artifacts/non_df_h4_d2_preparation/2026-10-24/artificial_tests.log)。最大3 system qubitの人工回路で、全2qubit非identity Pauli label×符号付き係数/時刻、XYZとwire置換、非可換順序、負productと集約phase、D1人工PF/PR・RTE次数0/2/4、scalar phase、両control branch、QPY往復、元wrapper/profile、source非変更を確認した。phase削除・誤control・角度符号・Y基底の誤りを検出するnegative witnessも含む。

sampler・event/DF preparation生成・compile・generic whole-body control・PauliEvolution lazy定義を禁止するmock下でも変換を確認した。人工Operator比較には浮動小数点許容値を使っており、certificateではない。これらは主agent自身の検査で、別reviewerによる独立レビューではない。

初回は2 FAIL/1 ERRORで、[初回log](../artifacts/non_df_h4_d2_preparation/2026-10-24/artificial_attempt01.log)も保存した。全identity `PauliEvolution('II', c=-.7, t=-.21)` のこのruntimeの局所定義phaseは `-.294` で、理想式 `-ct` と異なり、候補との不一致が見つかった。固定Pauli SOURCEはidentityをconstantへ分離するため、この経路を**対象外として明示拒否**した。元runtime・保存body・科学条件を修正していない。複素係数のnegative fixtureも、Qiskit constructorが先に拒否していたため、既存gateのoperandを人工的に破損させ、adapter自身の拒否を検査する形に直した。修正後12/12 PASS。

[runtime observation](../artifacts/non_df_h4_d2_preparation/2026-10-24/runtime_observation.json)は人工検査に使ったQiskit1.3.0と局所sourceのSHAであり、将来の科学runtime sealではない。

## 将来の最小実行案（未承認）

| call | 提案上限 | 今回消費 |
|---|---:|---:|
| 保存Pauli body validation | 2 | 0 |
| central controlled full body build | 2 | 0 |
| 費用top-level compile | 2 | 0 |
| generic対照のbuild/compile | 0 | 0 |
| sampling・state/signal・N/G | 0 | 0 |

1. CodexがD1のauthority／call予約／native witnessを新D2 namespaceへ接続し、人工検査する。新production科学SOURCE、使用input/body/joinsとD1対照、compiler/runtimeをsealする。既存grantは消費済みのまま。
2. この2対象・各1回のbuild/compile・保存対照との差の限定解析について、ユーザーの明示実行承認を結合した新grantを一度だけ発行する。draftのflag変更だけでは起動できない。
3. 両bodyのSHA、D1で採用済みのSOURCE基準Bとtoken/phase対応を確認してから変換する。未知gate・phase/wire/operand不一致でSTOP。新実装のinput hashをgeneric input hashに一致させるgateは置かず、新QPY自己整合性とsemantic対応を別に保存する。Pauli-PRのhistorical hash FAILと原因UNKNOWNは維持する。
4. 共通semantic gate後、2つの独立taskを利用可能資源に応じて並列化し、予約・出力・receiptを分離する。各targetは一度だけcompileし、input/output QPY・hash・histogram・RZ/CX/depthを保存する。旧wrapperにcallbackがないためpass traceは未取得を明記し、後付けcallbackでprofileを変更しない。
5. 両方の結果と自分のnative exit/reapを確認後だけ、central−genericの符号付き差と明記した分母の比を算出する。失敗でretry/resumeしない。保存・旧証拠保全を確認しmandatory STOP、D3へ進まない。

この差が測れた場合も、**固定した同一bodyでのcontrol実装＋compiler応答**の差である。Pauli/DFの表現だけの効果、一意の因果寄与率、母平均・N/G・精度保証を意味しない。

## 保全・公開・引渡し

[preservation](../artifacts/non_df_h4_d2_preparation/2026-10-24/preservation_before.json)の旧937file、input86、既存code42＋新3のSHAを[stdlib検証](../scripts/verify_h4_d2_preparation.py)で確認した。Qiskitをimportせず、無効flag・2対象・対照価格/trajectory・契約SHAを検査する。[author check](../artifacts/non_df_h4_d2_preparation/2026-10-24/author_check.json)はflag改変5種の拒否、`--launch`拒否(exit2)、syntaxとquantum importなしの計8件PASS。

元H4 native UNKNOWN、旧Pauli-PR hash FAIL/原因UNKNOWN、旧20行N/G null、C/L COMBINATION_BLOCKED、旧compiler MISSING_EVIDENCEは変更していない。新STOP・receiptで旧監視証拠を代替していない。別系列H2/Yoshida4、高次PF解析、他worktree・他ユーザー・共有環境への変更は0。

[公開状況](../artifacts/non_df_h4_d2_preparation/2026-10-24/publication_status.json)に専用commit/push結果を記録する。既存値・旧原本を置換せず、新規準備資料と索引更新だけをcommitする。

**mandatory STOP。次はユーザーが2対象・最大2compileのD2実行を判断する。Codexは承認された場合に限り、production接続・SOURCE/seal・新one-shot実行・終了監査へ進む。**
