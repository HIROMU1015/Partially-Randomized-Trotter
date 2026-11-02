# Track A Codex 2：高次PF B_ACQUIRE_MISSING run01

2026-10-18。**全12群の取得と統合比較準備を完了した。新しい費用ランキングは未実行。mandatory STOP。**

## 1. 実行範囲と結果

利用者添付指示459d1234…を、固定12条件・各n_E8・96 trajectory・192 cost-wrapper軸の一回承認として受領した。SOURCE/input/seed/control/環境/資源をsealした後に取得した。

| 条件 | 取得trajectory | 検証済みcost軸 | 状態 |
|---|---|---|---|
| H4_S2_q8 | 8 | 16 | ACQUIRED |
| H4_Y4_q4 | 8 | 16 | ACQUIRED |
| H4_Y4_q8 | 8 | 16 | ACQUIRED |
| H4_N2_q2 | 8 | 16 | ACQUIRED |
| H4_N2_q4 | 8 | 16 | ACQUIRED |
| H4_N2_q8 | 8 | 16 | ACQUIRED |
| H6_S2_q8 | 8 | 16 | ACQUIRED |
| H6_Y4_q4 | 8 | 16 | ACQUIRED |
| H6_Y4_q8 | 8 | 16 | ACQUIRED |
| H6_N2_q2 | 8 | 16 | ACQUIRED |
| H6_N2_q4 | 8 | 16 | ACQUIRED |
| H6_N2_q8 | 8 | 16 | ACQUIRED |

実compile entryは192、検証済み費用軸は192、通常trajectory完了は96。別の科学compileは0。call ledgerは候補/replica/axisごとにexclusive createし、失敗entryを返却しない。

## 2. 科学条件と既存6群

linear H4/H6、隣接1Å、STO-3G、DF rank12/19、PR prefix3/5、保存Hamiltonian/state/fragment ordering、T=.8、K4。外側1step当たりRTE短step32、全体32q、signed絶対値largest-remainder配分・generation orderを維持。元8桁new_2・S2・Yoshidaだけを使用し、projected/厳密係数版へ置き換えていない。

ε=.001、各軸ε/√2、α_axis=.025、経験的allowance1e-9、logB margin1e-12、P0。shotはB_SAVED_ONLY 17570742…の保存値へ結合した。今回shotを再計算せず、B_SAVED_ONLYも再実行していない。

既存ordinary control＋explicit XXPlusYY/number Phase系列、Qiskit1.3.0/rz,sx,x,cx/optimization1/seed17/backend・couplingなし。compiler hash1239304c…。旧dense・symmetric-directional単価やH6→H4係数移送を混入しない。

旧6群はH4/H6それぞれS2/q4・Yoshida/q2・new_2/q1、各n8。原本のbytes/SHAとSOURCEを保持し、新規n8と同一候補の確認標本として混ぜない。今回の12群は旧6群に存在しない適格qだけである。

## 3. Control・seed・回収

各群sample0についてnative低/高ancilla、全particle-number occupation oracle、scalar相対位相、cosine/sine Hadamard、同じ取得軸のtranspile後full-vectorを1e-9で検査し、PASSした群だけreplica1..7を解放した。new_2のforced order0/2/4・正負時間はstatevector controlのみで、通常sampleにも追加compileにも入れていない。

合計数値検査636件、最大差9.952552433743547e-10。全完了時の636件は、各trajectoryのscalar/lambda等5×96、sample0固有7×12、forced両control12×6から成る。forced eventは全完了時36 instance・72検査で、compile0。保存18人工/222実数値のnative検証を再利用し、新旧42人工/mockも起動前PASSした。作者自身の検査であり、独立reviewではない。

新探索namespaceはSOURCE/候補/replica/outer/piece/入力hashを含める。新1,824 seedは内部重複0、既知旧1,505 seedとの衝突0。失敗した元native namespaceも除外集合へ入れた。追加の旧seed/provenance証拠は固定reference契約が参照するものだけに限定し、未知の無関係jobの全seedを把握したとは主張しない。新データは全てEXPLORATION、C確認sampleではない。cosine/sineは同じevent列を共有する。

Native launcher exitは0、wait回収済み。Pool worker回収はTrue。wallは7674.813秒。実際のpool worker数31、fresh CPU idle約60.84core相当から最大60を設定した。process内でpreparation/DF primitive/oracleをcacheし、trajectoryごとに再利用した。compiler内部は保存profileどおり1thread/1process。今回のworker数を新規作業の恒常上限にしない。

旧RSSをqで拡大したtask別予測で並列admissionを行い、private replay/軽量費用の保管に約1.29GiBの事前見積りを用意した。QPY・新NPZ出力はない。host PSIは観測のみ、他job・OS・共有venvは変更しない。

## 4. 統合比較は準備だけ

`h46_acquire_integration.py` と `run_h46_acquire_comparison.py` を用意した。全取得が成立すれば、metadataで18群の同一input/preparation/compiler・paired event・SOURCE/seedの結合を確認できる。`assemble_rows` の実値評価や既存比較kernelの実18条件へのdispatchは今回実行していない。

将来は凍結shotを用い、`G_i=N_real*C_cos,i+N_imag*C_sin,i`、paired標本平均/SE、CX補助、正確な平均→小さいq→IDのtie-breakを適用する。欠測を0補完せず、不一致でfail closed。独立確認標本を混入せず、n8のSEを母平均の厳密保証にしない。

統合入力状態は`METADATA_BOUND_NO_ACCOUNTING`。publicにはraw event・private科学入力・QPY/NPZ・credential・有効grantを含めない。新96trajectoryの軽量projectionとfingerprint・compiler/event digest・RZ/CX、SHA・消費grant receiptだけを公開する。active grantと原本はprivateに保持。

将来のB_INTEGRATED_COMPAREは未承認plan/無効grantを作成し、実CLIが入力読込前にexit2で拒否することを確認した。新ランキング、最良qの数値選択、公式間の優劣判定は0。

## 5. 来歴・保全・次の作業

SOURCE：`d77e439a16d6953cb17e6d1ef91b0d9430144848`。Branch：`track-a-codex2-h46-higher-pf-b-acquire-20261018`。Provider：`c31ef424175c02379eafebe8fc6c7548c1d9aabe`。元kernelは687679a4…/f0dd2a9a…、保存値比較17570742…を再利用した。封印306fileを起動前後照合し、既存SOURCE/input/結果/grant/ledger/marker/STOP・他worktreeに書き込んでいない。旧STOPをPASSへ書き換えていない。

[公開bundle](../../artifacts/resource_applicability/track_a_h46_higher_pf_acquire_run01/2026-10-18/README.md)。Private領域は`/home/AbeHiromu/projects/h46-higher-pf-acquire/20261018/private/`および`runtime/run01/`。

**ユーザー**：取得結果が全PASSなら、別にB_INTEGRATED_COMPAREの保存18適格条件の費用比較を一回承認するか判断する。失敗なら、partial evidenceとSTOPを確認して対処範囲を判断する。

**Codex**：別承認後に限り新しい統合比較one-shot grantとSOURCE/input sealを固定し、ランキング・q選択を実行してSTOPする。C独立sampling、係数変更、q>8、α感度、B0/B0+I・RPE全round評価は別範囲である。
