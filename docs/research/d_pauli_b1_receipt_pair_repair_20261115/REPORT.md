# D2：非configuration価格の証拠接続と保存親eventの直接連結

2026-11-15 JST。**限定実装2点を完了し、人工・toyの26 test methodがPASS。実LD2の較正18 compileは引き続き保留、技術成果の公開後mandatory STOP。**

実装・test SOURCE：`13ab1acfaebe5f33958e1db67862f5c45725ab91`。branch `codex/d-pauli-b1-caller-prep-20261114`、計画基点`043c7567451e8f502a1cf0f425d32efe0a50f5dc`、旧caller SOURCE`bd1b1b2862da6d174fa521e7612051810a636e7f`。旧W成果`1e2639a1b`・raw `6c4709f85`、旧B0、SCOPE_DEVIATION・ログは無変更で保全した。

## 変更と再利用

| 対象 | 結果 |
|---|---|
| [B1 caller](../../../scripts/run_pauli_b1_prices.py) | 非configuration cellに公開compile receipt・軸別実装証拠を出し、公開後bindingとCのfit前読取APIを追加。pairedの親は公開locatorから読む。 |
| [receipt reader](../../../src/trottertracks/pauli_b1_receipts.py) | cell/axis/path/request・辞書・cohort/seed・metrics/fingerprint・公開SOURCEをJSON/Gitから照合する小さな専用module。fitや回路処理はしない。 |
| [connection](../../../src/trottertracks/pauli_tail_price_connection.py) | 保存親のeventを既存decoderで復元し、label/係数/order/product/normalization・D2 token/phaseをbuild前に検査。samplerは呼ばない。 |
| [B0 core](../../../scripts/run_pauli_ld1_m0_pilot.py) | optional `saved_step_data`引数だけを追加し、既存head/tail/oracle/scalar/X/Y処理を再利用。通常時は従来のsamplerへ進む。 |

D1 `d323fee1c`の親receipt・直接event復元の経路も参照した。DF用request構成をPauliへ移植せず、既存`h46_same_event.decode_event`をそのまま再利用し、Pauli専用のlabel・D2 token/phase検査を追加した。native lowerer、D2 emitter、compiler、wrapper、RTE library、decoder自体は変更していない。新しい回路生成器、fit、承認機構は作っていない。

実行code SHAと結果公開commitを別々に保持する。raw RESULT/C_INPUTS/receipt/evidenceを固定した後、公開用record/sidecarから元結果を`{commit,path,selector}`で参照する。m=Noneと元`PAULI_PRICE_CELL:`は保持し、canonical roundへ偽装しない。Cの別worktreeは変更していない。

paired P2R66はS66 path0→path1の保存eventを直接使う。親SOURCE・request・辞書・cohort/index/seedの不一致は子process開始前に拒否し、event/order/product/D2 phaseの不一致はbuild前に拒否する。各stepのscalarは0、子total Tのscalarを最後に一回だけ適用する既存gateを保つ。通常normal samplerの経路とB0の既定結果はtoy回帰で確認した。

## 限定検証

| 実行したsuite | 方法 | 最終結果 |
|---|---|---:|
| [check_pauli_b1_receipts.py](../../../tests/check_pauli_b1_receipts.py) | 一時Git repoの保存・再読。LD2計画5 cell×X/Yの架空record、fake compilerのreceipt・明示的な人工artifact bytes。全10 recordから正しいpath/axis/receiptへ到達し、軸・欠落・fingerprint・辞書/seed/metrics・公開SOURCEの取違えを拒否。 | [7 PASS、2.701s](metadata-receipts.txt) |
| [check_pauli_b1_saved_parents.py](../../../tests/check_pauli_b1_saved_parents.py) | 4-mode toyを保存JSON境界で再読。order2と非ゼロD2 step phaseを明示的なtoy eventで検査。paired sampler2入口とも0回、childのcontext復元前phase拒否、親順序・辞書・event phase/D2 phase・scalar不一致でbuild/compile0。通常sampler経路と、fake compilerの保存QPY cache hitも検査。 | [5 PASS、8.679s](toy-direct-parents.txt) |
| [check_pauli_b1_prep.py](../../../tests/check_pauli_b1_prep.py) | 既存metadata/架空価格の限定回帰。非configuration公開receipt全体の検査は上記の新suiteで行う。 | [11 PASS、3.458s](regression-b1-metadata.txt) |
| [check_pauli_b1_circuits.py](../../../tests/check_pauli_b1_circuits.py) | 既存4-mode toy・fake compiler。旧B0 default結果同値などを確認。 | [3 PASS、3.122s](regression-b1-toy.txt) |

Qiskit1.3.0を持つ既存project Pythonで上記4入口だけを`-B -W ignore`付きで実行した。新toyの初回は保存JSONのtuple/list表現差に対するassertが失敗し、比較を既存のJSON表現へ正規化して修正した。科学値・precision・gate条件は変更していない。上表は固定SOURCEの最終logである。作者自身のlocal技術検証で、独立agentレビューやCIではない。

今回のB1-V/CALの新規科学call、H4 preparation/sampling/build/compile/statevector、保存H4 P0演算子の数値再検査、実価格再解析・C受入、fit、holdout、E[C]/N/Gはすべて0。toyのstatevectorとfake compilerを実H4や実transpileの証拠として数えない。`check_pauli_ld1_m0_pilot_prep.py`の親suite、unfiltered discovery、`--include-saved-numerical`は起動していない。

## 残る確認とロードマップ影響

技術的な2不足は人工データで解消した。実取得後に初めて確認できるのは、実CALのparent/phase/normalizer gate、全5 cellのX/Y compile/cache receipt、実データの公開SOURCEを用いたC読取、較正後のmodel・独立holdout適合等である。**実LD2のorder2 product・非自明位相はUNKNOWNのまま**。Wはorder0 component34/40の1 pathであり、今回のtoyから実観測へ昇格しない。

統合ロードマップでは「D2の証拠接続・paired直接連結の技術修正待ち」を「限定SOURCE完成・26人工/toy PASS、実LD2較正18は未採択/未実行のまま」へ更新できる。順序は較正取得→C fit・凍結→別指示で独立holdout。B0/Wの価格をCALへ混入せず、Pauliの許容差・全law E[C]/CI・正式G資格は未確定。ロードマップ本体は編集していない。

APIと入力例は[HANDOFF](HANDOFF.md)。**実装・検証のcommit/push・remote照合後mandatory STOP。** この実装完成はCAL/Vの実行許可ではない。
