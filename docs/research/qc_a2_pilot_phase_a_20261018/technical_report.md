# A2実入力pilot Phase A：接続SOURCE・未採択契約の準備

2026-10-18 JST。**`PHASE_A_IMPLEMENTED_ARTIFICIAL_PASS_AWAITING_CONTRACT_ADOPTION`。** 実入力adapter、別途承認後だけ使えるone-shot runner、保存機構、未採択の科学契約草案を作成した。新36＋既存98、計134 test methods・6,777 subtests PASS。**実入力の係数decode・certificate生成・step比較は0件。Phase Bは未承認であり、ここでmandatory STOPする。**

[科学契約草案](execution_contract_draft.md)／[machine草案](execution_contract_draft.json)／[入力候補と不足情報](input_candidate_and_provenance.md)／[SOURCE・guard・費用](source_guard_and_resource_design.md)／[次の承認・引継ぎ](decision_handoff.md)／[テストJSON](synthetic_test_results.json)／[全log](synthetic_test_log.txt)／[静的検査](static_inspection.json)／[SOURCE再利用監査](source_reuse_audit.json)／[固定SOURCE](source_freeze.json)／[証拠索引](evidence_index.json)／[公開receipt](publication_receipt.json)。

## 1. 承認範囲と固定資料

[ユーザー指示原本](inputs/user_instruction_original.txt)をbyte同一で保存した。Phase Aは接続実装・人工テスト・科学契約草案・公開まで承認、Phase Bはユーザーの後日の契約採択と別途実行承認がある場合だけ実施する、という二段階を適用する。表題の「限定科学実行まで」を今回の計算承認とは読まない。

基点は[34d9a2f1a25bd35c03566ce0635e8b66665ad996](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/34d9a2f1a25bd35c03566ce0635e8b66665ad996)。旧SPEC-01〜07のSOURCE・98件の人工検査記録・報告・API、[設計7af83266…](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/7af83266f96b714bb3fa2f1ef5e4bc89faf37db3/docs/research/qc_a2_certificate_connection_design_20261018)、[EW2監査db181958…](https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/db1819587df36ee8833b37227ca0083faa0408ec/docs/research/qc_relative_form_ew2_step_audit_20261017)、既存[preflight](../qc_a2_practical_preflight_20261017/practical_preflight_report.md)と入力writerを実際に確認した。既存保証の再証明、新しい科学レビューは行っていない。

SOURCE固定commit Sは **`8d483f9bfa5983bb52fef55887e2f9f73a790511`**。新namespace [src/trottercert/a2_pilot/](../../../src/trottercert/a2_pilot/)、[launcher](../../../scripts/run_qc_a2_fixed_df_pilot.py)、[人工test](../../../tests/test_a2_pilot_phase_a_synthetic.py)。31個の新旧module・package初期化・launcherのbytes/SHA256を固定し、Git commit Sのblobと照合した。契約・公開資料はSの後の別commitで固定する。SOURCEを後から書き換えて契約へ合わせる方式ではない。

## 2. 入力候補

既存preflightで追跡されていたH4 snapshotを第一候補とする。3589 bytes、SHA256 `13e4b10d2347ed900fe4aa4b2238128ebe735a48988e321b9e4fabac4d092a3b` が旧記録と一致した。ヘッダー上は一体8×8、λ12個、g12×8×8。constant、一体、全平方が同じcontainerにある。ファイル名・既存metadataに基づく名目条件はH4 linear／1.00 Å／STO-3G／保存DF rank12で、正式採択はしていない。

今回確認したのはcontainer・NPY member・array bytesのhash、ヘッダー、既存metadata、writer/provenanceだけ。**浮動小数点の係数値はdecodeしていない。** λの符号・exact Hermiticity・sector実適合・PSD値・U/L/stepは未評価。旧metadataのenergyやtruncationをU・ground truth・scalar correctionへ用いていない。

調査した保存パスでは、より小さいH2のfull DF snapshotは特定できなかった。H2の集計結果JSONから入力を逆生成しない。PR-2 development/held-outの別snapshotを新たに開いたり流用したりもしない。H4は最小と確認できた候補であり、repository全体に他の小入力が存在しないという主張ではない。

詳細な来歴・不足情報・proposalは[入力資料](input_candidate_and_provenance.md)。SOURCEにあるfull DF保存規約を使い、固定DFモデル自体をtarget候補とする。元電子Hamiltonianへの厳密対応や過去のHermitizationのモデル誤差が認証されたとはしない。

## 3. 接続実装

旧認証SOURCEのraw guardは `SYNTHETIC_TECHNICAL` 専用なので、そのまま実入力へ適用できない。旧guardは変更せず、新版で正直に `FIXED_DF_PILOT` とする。実入力へ人工tagを付け替えて旧facadeを通す方式は使わない。

SPEC-01〜03のdecode/独立再decode・cap/PSD/scalar/PF検査、SPEC-04〜07のcertificate構築・独立検査を、scope・version・provenanceを変えた薄い版として保存した。算術bodyは、明示した二つの最適化を除き、docstring・scope literal・timing observerを除外したAST比較で既存と一致する。[再利用監査](source_reuse_audit.json)は変更箇所と元SOURCE hashを列挙する。全面的に新しい認証算法を作り直したものではない。

adapterはstdlibだけでNPZ/NPYを読み、object/pickle・未対応dtype・Fortran順序・不足/重複block・規約不一致を拒否する。元のbyte列をC-order・宣言mode順・保存block順のdescriptorへ移すだけで、cast・Hermitization・再DF・係数修正は行わない。N保存・exact Hermiticity・signed scalar恒等式はその後のgenerator/checkerで検査する。通らなければ旧rawを保全してSTOPし、別モデルを黙って作らない。

raw/container/array manifest、mapping、SOURCE、model/sector/partition/PF、certificateのIDを結ぶ。論理model_idはexact decodeなしでは計算できないため、草案は `DERIVE_EXACT_FROM_PINNED_RAW_AND_MAPPING` を明示する。Phase Bで固定raw＋mappingからモデルを生成し、独立SPEC-01〜03 checkerが確認したmodel_idを後段へ渡す。Phase A中にmodel_idを得るためだけの隠れた係数評価は不要である。

## 4. Guard・保存・計測

runnerは採択済み契約、未解決`UNSET`なし、31 SOURCE hashとcommit S、raw manifest、外部から独立に指定されたcontract/auth hash、別authorizationのscope・文言hash・SOURCE/input親を照合する。contract自体の `contract_grants_execution` はfalseのまま。明示execute flagだけで許可されたとは扱わない。人工test用authorization schemaはproduction入口が拒否する。

同じ契約についてcanonical実行repositoryとledgerを固定し、input IO前にexclusiveな永続claimを作る。run_id変更・既存/中断claim・再実行は拒否し、古い記録を上書き・削除しない。実行repositoryはこの専用checkoutの絶対pathを草案へ固定し、契約全体の採択待ちとした。Phase Aでは実科学inputをrunnerへ渡しておらず、物理的な科学markerも0件。

generator提案→独立checker→次段、という順序を維持し、途中raw、元snapshot、提案、拒否理由、検証済みrecord、元contract/auth bytes、stdout/stderr、SOURCEと入出力hash、呼出数、各段CPU/wall、process累積peak RSS、certificate Ratのbit長と保存量を記録する。K_SOSとCbar2の生成・検査時間は別々に記録する。final log/result書込みを計測範囲から除くことを結果へ明示し、取得費を0扱いしない。

byte上限は**各record／各log**として後日採択する。集合全体の保存量は別に測る。失敗・中断・上限不適合でもclaimを解放せず、partial recordと理由を保全しSTOP。SIGTERMをowned launcher内で例外に変え、KeyboardInterruptも保存する。disk故障・SIGKILL・電源喪失では全record保存まで保証できず、消費済みclaimと既に保存済みの部分を保持するbest-effortである。自動retryはない。

## 5. 人工検査と性能scope

**新36 methods／4,148 subtests、既存98 methods／2,629 subtests、合計134 methods／6,777 subtests PASS。failure/error/skipped 0。** subtestはunit-test内のパラメータ例で、独立科学実験数ではない。最終suiteはwall約4.37秒、CPU/RSS・SOURCE/test hashは[JSON](synthetic_test_results.json)に記録した。最終SOURCEとテストのbyte同一性も確認する。

正常/複素NPZのexact byte mapping、ヘッダー/endian、旧人工tagとの分離、再hashした入力改変、欠落/追加/重複・dtype/order・scalar役割、exact Hermiticity拒否、非有限値、元モデルID一致、同じgrouped構造、guard全段、UNSET、外部pins改変、SOURCE不一致、claim衝突、異なるrun_idによる再実行、checker拒否、KeyboardInterrupt、部分保存、保存失敗、出力/log上限を検査した。authorization vectorとclaim storeは人工・in-memoryであり、本番文書や実行済みmarkerを発行していない。filesystemのexclusive/open/fsyncはmockで検査した。

高価な箇所には、generator側の離散Pauli maskによる交換するpairの除去と、checker側のoccupation行列からの**exact trace一括計算**を用意した。generatorのJW経路とcheckerのoccupation経路は共有しない。旧の直接Pauli交換子と3mode全4,096組、旧literal traceと0〜4mode、8modeの人工Pauli作用へ照合した。新しい数学保証ではなく、同じexact積・`2^-n Tr(PX)`を別順序で評価する実装改善である。

CPU32/affinity32、利用可能RAM等をread-only確認した。小さい人工suiteはpatch isolationのため単一runner。科学SOURCEはCPU/RAM/worker/wallの人工capを設定せず、BLAS/GPUや共有設定も変更しない。pilot pipelineは前段の独立受理を次段の条件とする。高価なtraceにはbatch処理を適用したが、実H4でのruntime・memory成功・speedupを確認したとはしない。詳細な計算量と限界は[resource資料](source_guard_and_resource_design.md)。

## 6. 未決条件・保全・停止

科学target、mode/N/mapping、全fragmentとPF順序、trial、d/α、sqrt精度、root幅/反復予算、record/log byte上限、canonical実行場所と正式契約の採択が残る。実入力がexact Hermiticityを満たすかも未検証であり、初回技術gateで停止し得る。state preparation/native費・best-known envelope全体・元分子保証・総資源改善は未評価。人工testの設定をscienceの既定値に移していない。

旧SOURCE、raw、科学契約、authorization/grant/ledger/marker、親worktree、他研究を保全。索引・概要・validation inventoryだけにPhase A追加を記録し、既存科学結果・既知FAILを変更しない。[保全監査](preservation_audit.json)と[公開照合](publication_receipt.json)を参照。

**実科学係数decode 0、実入力certificate 0、科学step比較0、科学authorization/grant/marker発行0、新数学保証・新科学レビュー・SPEC-08・QPE/native/state実装0。** A2 `NOT_ADOPTED`、A1 `CLOSED`、K1-F未承認・未実行。公開後mandatory STOP。次はユーザーによる科学条件・正式契約の採択判断であり、Phase Bにはさらに別途の明示実行承認が必要である。
