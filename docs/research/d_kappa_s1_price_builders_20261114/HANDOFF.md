# D → C/B/A：S-1/P-1 caller、Truncated・Pauliの価格接続

固定SOURCE `04bead115233641233ae1fc409b2c4d89ebc2483`（branch `codex/d-kappa-s1-price-builders-20261114`、C `5fb2c32e` 基点）。[REPORT](REPORT.md)／[次の実行](NEXT_EXECUTION.md)／[TESTS](TESTS.txt)。他担当のbranch・worktree・moduleは変更していない。

## C担当へ

- S-1はCの数値SOURCE（`5cc7c07e` とbyte一致）と、R448 callerのreference readerをそのまま使う。追加はscopeの差替えとpost-gate（logB包絡＋1 ULP、IDの一致）だけである。S-1結果のB packetは既存の `I.signal_input` 形式で出力する。
- `price_implementation_guard` には変更が不要だった。
  - Full-H κ2（新ZZ）とTruncated p7（`NOT_APPLICABLE_NO_RTE`）のsidecarは、既存の契約どおりに作る。各callerの `price_bindings` が `IMPLEMENTATION_EVIDENCE.json`（selector付きproof）を生成する。
  - 公開後のcommitをproofのlocatorへ入れるのは、実行後の公開手順で行う。
- **Pauli D2 route**（`PAULI_D2_CENTRAL_ROTATION_CONTROL_WITH_NATIVE_DF_HEAD`）は、現行guardの `_rte_source` に無いので拒否される（人工testで確認）。DF新ZZのIDを付けて通すことはしていない。representation/routeごとのpolicy拡張はC側の作業である。
- T3（最大18 wrapper）のcallerは決定論の直接価格とsidecarまでを担当する。m4/m5での係数固定、m6 holdout判定、B0/B0I同値、fitはCのscopeで、本callerは実行しない。
- Full PFとTruncated PFは同じlong-time nativeを使い、quantityも同じである。Full PFの既存receiptがpf_common系を含む場合は、Cの既存同一性判断に従う（Dで再監査していない）。

## B担当へ

- S-1が承認・実行されると、κ2の13 roundのB packetが `docs/research/d_full_h_kappa2_s1_results_20261114/run01/FULL_H_K2_QDYADIC_KAPPA2/B_SIGNAL_INPUT.json` に出る。各roundの `law_binding.kappa` はκ2を持つ。RMSE policyはR448・PRと同じA S0 policy。
- P-1とT3のprice recordは単一pathの配線証拠、又は決定論の直接価格である。law平均・全roundのE[C]・Gとしては使わない（`expectation` fieldで区別した）。

## A担当へ

- Pauli接続はA P0辞書（4220f3006）と全round packet（714247619）をcanonical SHA-256で照合して読み、184項・順序・係数・s_Rを変えない。eventはfull 2^nで扱い、sector射影したPauliは使わない。
- LD1/m0 pilotの実行callerに必要なのは、full-256 index0 oracleとして使うP0 `head.npy` の基底順（保存mode-0-MSB）の扱いの確認である。Aの記録した `mode_mapping` に従ってbit反転で変換し、`Pauli_nonidentity.npy` との一致を実行前gateにする案である。

## ユーザーへ

承認が必要な事項：(1) S-1の実行、(2) P-1の実行（compile 2）、(3) T3の実行（compile最大18、Cのfit scopeを含めるかどうか）、(4) Pauli pilot callerの実装と実行。現在はすべて未実行で、mandatory STOP。
