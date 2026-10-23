# H4 D1限定診断：Stage Aでfingerprint不一致・mandatory STOP

2026-10-24 JST。**4つのfull入力を再構成した結果、Pauli-PF・DF-PF・DF-PRの旧recursive input fingerprintは完全一致し、Pauli-PR index0だけが不一致だった。D1はStage Aで停止した。費用計測用top-level compileは0/9回、Stage B/Cと部分回路診断は未実行。**

取得済み証拠の保存・終了監査は完了したが、保持部・event部の定量的費用診断には到達していない。
不一致をPauli-PRの高費用・不適格・意味的誤りと読み替えない。
Rename・hash条件緩和・科学retry・D2/D3は実行していない。

[結果summary](../artifacts/non_df_h4_d1_execution/2026-10-24/result_summary.json)、
[Stage A原本](../artifacts/non_df_h4_d1_execution/2026-10-24/run01/stage_A.json)、
[STOPとnative回収](../artifacts/non_df_h4_d1_execution/2026-10-24/run01/STOP.json)。

## 固定条件・承認・SOURCE

Linear H4、隣接1.00 Å、STO-3G、system8 qubit＋ancilla8、共通保存stateの来歴。
Pauli184非identity項・PR prefix27、DF tol-only1e-8／rank10・PR prefix1＋one-body。
S2／T=.8／q8／r4／K4、cosineのみ。Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、
backend／coupling／layout／routing指定なし。Full Hadamard wrapper、状態準備費用除外、
DF raw_concatenationとoccurrence内の隣接equal-basis reuseを維持する。

今回の[承認原文](../artifacts/non_df_h4_d1_execution/2026-10-24/approval.txt)はadapterから最大9費用compileまでを一括承認する。
全4full hash一致後だけcompileし、全4旧RZ一致後だけablationへ進む条件は変更していない。

| 役割 | commit |
|---|---|
| 基点 | `3282f74cd4d219141e4358c215a225dc2e3ec9c3` |
| 新D1科学SOURCE | **`48d4d27a05a529e05b11599f5f3c671302826617`** |
| 元H4 | `0c329b30a81accb35c212be8defa66fee0dd0f12` |
| C | `e80d7510880f8db993b777c627bb8e95b05a070d` |
| L | `eb195bf2dedd0aed9d80ad7f84948b59ea53d854` |
| C/L | `0880f4af98b125d62599e970db89fc33634d1b61` |
| 条件付きadapter／結果 | `b1984dcde06a874d0fd6e9fe6395b7f3fb7504d8` ／ `5600f0e4b28c140a6a50b8aba47425860972a932` |

基点commitと計画・無効契約をGitHubから取得しGit blobを照合した。
[取得記録](../artifacts/non_df_h4_d1_execution/2026-10-24/github_verification.json)。
旧descriptorの来歴は前段階の取得と42-input SHAを継承し、新sealへ結合した。
新SOURCEは実H4生成前にcommitし、結果を見てadapterを変更していない。

## 再利用・人工検証・新grant

[adapter](../src/trotterlib/h4_d1_adapter.py)はJSONから既存gate・event型へ復元する薄い接続。
Pauli append、DFBasisRegistry・sequence builder・basis逆順、diagonal append、
Hadamard wrapper、recursive hash、元compiler関数を再利用した。
保存matrixをUnitaryGateに置き換えず、元RZ／XXPlusYY型とparameterを復元する。

Pauliのgeneric whole-body control、DFのcentral-only controlと元phase順序を維持。
保存eventのorder・angle・sign・phaseを直接使用し、sampling／_make_event／atan再生成を呼ばない。
C保存のfull matrix／stateを計算へ持ち込まず、回路に必要なfieldだけを接続した。

[controller](../src/trotterlib/h4_d1_control.py)と[runner](../scripts/run_h4_d1.py)は、
既存CallBudget・durable save・native subreaper witnessを再利用する。
Witnessは新receiptのscope名だけを接続時に変更し、原helper SOURCEは改変していない。

[adapter人工8件](../tests/test_h4_d1_adapter.py)＋[authority人工5件](../tests/test_h4_d1_authority.py)がPASS。
人工2qubitの両control branch・global／relative phase、basis inverse／reuse、
event order0/2/4・順序、QPY roundtrip、sampling／再準備／費用compile拒否、
scope拡張・grant欠測／改変拒否、one-shotと非返還予算を検査した。
初回の人工roundtripテストではtupleとJSON listの比較方法を修正し、最終8件がPASSした。科学runのretryではない。
主agent自身の確認であり、独立レビューや実H4意味的同等性のcertificateではない。

[新seal](../artifacts/non_df_h4_d1_execution/2026-10-24/seal.json)にSOURCEのcommitted code、
入力SHA、参照runtime manifest、承認SHAを固定した。
[runtime6,999file](../artifacts/non_df_h4_d1_execution/2026-10-24/runtime.json)は既存Cのlegacy closureを再利用。
原科学依存28source moduleは0c329b30のbyteと一致、authority helperは基点版を固定した。
旧611-file保全mapも再利用し、今回複製していない。

[active contract](../artifacts/non_df_h4_d1_execution/2026-10-24/active_contract.json)、
[新grant](../artifacts/non_df_h4_d1_execution/2026-10-24/run01/grant.json)、
[CONSUMED](../artifacts/non_df_h4_d1_execution/2026-10-24/run01/CONSUMED.json)、
[ledger](../artifacts/non_df_h4_d1_execution/2026-10-24/run01/call_ledger.json)を保存して実行した。
Grant ID：`b4a61a9064c687d00df10f49bbc24e90d62f47074dbc655727a6ab33e4e17ba8`。

Active contractに継承されたDRAFTの説明文は実測statusではない。実行後の状態と消費数はrun01のledger／STOPを正本とし、新grant本体は別fileに保存している。

| 項目 | 上限 | 消費 |
|---|---:|---:|
| full input build | 4 | 4 |
| ablation／empty build | 5 | 0 |
| 費用top-level transpile | 9 | 0 |

Stage Aの4独立processを並列実行した。Affinity64 CPUと十分な空きmemoryを観測し、
CPU/RAMの人工上限を追加せず、共有設定・他ユーザー・別研究のprocessを変更しない。
資源競合による中断はない。
元generic control内部のunroll／BasisTranslator等は回路構築として実行された。
**費用compile0はtop-level価格測定が0という意味**で、実H4回路構築も0という意味ではない。

## Stage Aの結果と保存証拠

以下の旧RZはsingleton／index0の参照値で、新しい測定値ではない。
PR256本平均を再現性判定へ流用していない。

| full入力 | 旧input hash | 旧cosine RZ参照 | 新RZ／CX／depth | RZ一致 |
|---|---|---:|---|---|
| Pauli-PF singleton | PASS・完全一致 | 255500 | 未測定 | 未確認 |
| Pauli-PR index0 | **FAIL** | 9652 | 未測定 | 未確認 |
| DF-PF singleton | PASS・完全一致 | 135355 | 未測定 | 未確認 |
| DF-PR index0 | PASS・完全一致 | 23021 | 未測定 | 未確認 |

Pauli-PR旧hash：`8a8adeded83edfddead525494906ccbd2a3796b11ed038f1d53f113b61468986`

今回のhash：`b677002ac37cd37a3f8a0ec84320963f7e44d3dcd29c271c369b1e6ab49fcd21`

4つのinput QPY、Pauli2つのcontrol前body QPY、pretranspile histogram、
sidecar tag、DF basis switch／reuse counterを[variants](../artifacts/non_df_h4_d1_execution/2026-10-24/run01/variants)へ保存した。
Pauli tagはbody内、DF tagはstep内のforward／event／reverse piece内の座標。
Nested basis／diagonal／controlを独立の加算費用として二重計上しない。

全4hash一致gateが成立せず、Stage B/Cは未実行。
費用開始marker・metrics・postcompile QPYは0件。
Retained-only／event-only／emptyもbuildしていない。
新価格・非加法性を0で埋めず、null／未測定とする。未実行は不適格の判定ではない。

## 不一致の確認事実と原因候補

**事実**：保存event／trajectory hashとdescriptorの来歴は維持され、
固定runtimeで元append／control／hashを用いたが、Pauli-PR fullのrecursive hashが異なった。
他3fullは完全一致した。
旧input QPY／hash preimageの保存は確認できず、新QPYを旧原本の代替にはしない。

今回のPauli-PR body／base_gate名は `circuit-162`、
controlled gate名は `ccircuit-162`、body global phaseの保存表示は0.2651822507334482。
これは今回の値で、旧名との相違を直接確認したという意味ではない。

**有力な候補**：recursive hashはoperation.nameを含み、
元 `h4_pilot_science.event_gates` はPR trajectory前にforced-order検証用QuantumCircuitを生成する。
今回のreplayはその科学検証を再実行せず直接bodyを作るため、自動名counterの履歴が異なり得る。
元to_gate()はbody名をgeneric gateへ継承する。
準備段階の人工検査でも、名前だけの差がこのhashを変えることは確認した。

ただし**旧名／旧recursive payloadがないため、不一致が名前だけに限るとは証明していない**。
phase・definition等の他の差を完全に排除した比較も未成立。
意味的に同じという判断でPASSにしたり、RZだけへ判定を緩めたりしていない。
Rename、counter操作、名前総当たり、hash field無視は0。
科学的意味の同等性は未検証のまま扱う。

不足が判明したのは「保存eventを新科学計算で補う必要」ではなく、
**現在の旧hash厳密一致gateが閉じないこと**である。
別の再現性基準を使う場合は、別契約と利用者の明示承認が必要。

## Q1〜Q7と次の研究判断

| 問い | 今回の到達点 |
|---|---|
| Q1 Pauli保持部はどれだけ安いか | Stage C未実行で定量未回答。保持27がZ/ZZのみという既存構造証拠は維持 |
| Q2 event列の費用差 | 未測定。保存eventは追加抽選なしで復元したが価格比較は行っていない |
| Q3 DFのbasis関連構造 | fullのsidecar／counterを保存。保持／eventにbasisがある構造と、basis単独のcompiled価格を区別 |
| Q4 fullと隔離回路の非加法性 | CF−CR−CE+C0の入力がそろわず未測定。推定・0埋めなし |
| Q5 逆転の主要候補 | 保持内容＋event粒度＋controlの連動という前回仮説を維持。新しい定量的支持は得ていない |
| Q6 残る交絡 | Pauli-PR旧payloadとの差、保持／event／basis／control／compiler寄与率、256本平均への外挿 |
| Q7 D2／D3の価値 | baseline対応が先。D2/D3を今回のSTOPを迂回する手段にしない |

次の第一候補はPauli-PRの厳密hash gateの扱いに関する限定判断。
旧名・program証拠の回収か、SOURCE／descriptor／phase／control対応による
意味的再現性の別基準を認めるかを利用者へ返す。
本報告は基準変更・追加科学実行の承認ではない。
価格baselineが未確認なので、D2/D3の効果や優先順位を新しい実測として断定しない。

## Native終了・保全・閉鎖・公開

新engineはnative exit0、4taskもnative exit0でdirect parentが回収し、
外側subreaper witnessも終了を永続化した。
[engine receipt](../artifacts/non_df_h4_d1_execution/2026-10-24/run01/native/worker_native_exit.json)、
[task receipts](../artifacts/non_df_h4_d1_execution/2026-10-24/run01/native_tasks)、
[最終STOP](../artifacts/non_df_h4_d1_execution/2026-10-24/run01/STOP.json)。
これはprocess／保存の正常終了で、Stage Aの科学gateのPASSではない。

Source・入力・runtime・旧611fileの終了時保全PASS。
[run出力50fileのSHA inventory](../artifacts/non_df_h4_d1_execution/2026-10-24/run01/output_inventory.json)も照合した。
旧N/G null、C/L COMBINATION_BLOCKED、旧compiler MISSING_EVIDENCE、元native UNKNOWNは維持。
新native証拠で旧監視終了を補完しない。

Active contract／grantを後書きせず、CONSUMED＋STOPで再起動を拒否する。
未使用の費用callを別runやD2/D3へ転用しない。
科学retry、sampling／RNG replay、SCF／FCI／DF、state・signal、N/G、certificate受理は0。
seal後のcode変更も0。
資源・保全は最新版方針§8.4、再利用は§10に従い、所在確認済み版にない§10.3については添付が明記する必要十分な技術管理を適用した。

専用branch：`codex/non-df-h4-d1-execution-20261024`。
SOURCEは48d4d27a0。結果commit／push／remote SHAは[公開記録](../artifacts/non_df_h4_d1_execution/2026-10-24/publication_status.json)。
今回の実行・取得済み証拠整理・終了監査・commit/pushまでで **mandatory STOP**。

次の担当は**ユーザー**：旧Pauli-PR payloadの回収、または意味的再現性を用いる別契約を認めるか判断する。
**Codex**：その明示承認後だけ新契約・必要な証拠・新SOURCE/grantを準備する。
今回は新基準実装、科学再実行、追加compile、D2/D3へ進まない。
