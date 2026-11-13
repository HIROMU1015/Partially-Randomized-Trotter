# D：Pauli LD1/m0 pilot callerとFull PF同条件価格callerの完成

2026-11-14 JST、Codex D。**caller・接続・人工検証・SOURCE照合まで行った。** 実H4のtrajectory sampling、回路build、compile、価格取得、model fit、N/Gは行っていない。

- 固定SOURCE（caller・接続・test）：**`8b28dce02cc5b5da1a8f4f1718f0b9f855759978`**
- branch：`codex/d-kappa-s1-price-builders-20261114`（前回 `04bead11`／`5ccc1dfc` の続き）

[HANDOFF（C・A）](HANDOFF.md)／[次の有限compile](NEXT_EXECUTION.md)／[test結果](TESTS.txt)／[保存P0のbit順照合](P0_BIT_ORDER_CHECK.json)（[script](CHECK_P0_BIT_ORDER.py)）

完成済みの `run_full_h_kappa2_signals.py`、`run_full_h_kappa2_p1.py`、`run_truncated_p7_prices.py` は**byteを変更していない**。preflightとtestも再実行してPASSした。

## 作業A：Pauli LD1/m0 pilot caller

[`scripts/run_pauli_ld1_m0_pilot.py`](../../../scripts/run_pauli_ld1_m0_pilot.py)、接続 [`pauli_tail_price_connection.py`](../../../src/trottertracks/pauli_tail_price_connection.py)。

| 確認事項 | 実装 |
|---|---|
| 1 辞書 | A P0 `LD1_DICTIONARY.json`（4220f300）を、packet（714247619）のcanonical SHA-256 `dbdde5a1…` で照合して読む。184項の順序・hex係数・sign・確率・λ_P（fsum）を検査し、再生成しない |
| 2 bit順 | 保存P0はmode-0-MSB、回路とlabelはQiskit little-endian（mode q→qubit q）。変換 `saved_to_qiskit` はAの `reverse_bits` 置換（自己逆）。実行前gate `bit_order_checks` で、ラベル反転、Σc·P(saved)＝保存Pauli、変換後＝Σc·P(qiskit)、head/tail＝block和、tail＝Pauli＋s_Rを確認する |
| 3 full-2^n oracle | 保存headブロック（one-body、fragment0）を変換したうえで別々に厳密指数化し、dense Pauli event（積順・符号・order phase）を適用する独立oracle。sector射影はしない。index0で回路の高制御ベクトルと比較する |
| 4 D2 | `append_pauli_event`→`central_control_body`（中央回転のみ制御）。product/order phaseを保持 |
| 5 DF head | `native_context('H4',(1,))` の `prepare(ham,1)`。one-bodyとfragment0、保存順を検査する |
| 6 scalar | c＋s_Rを最後のancilla phaseとして一回だけ適用。s_RはnativeのDF tail identityとbit一致を必須とし、二重適用は拒否する |
| 7 seed/law | 既存long-time `trial_seed`（family `H46_LONG_TIME_20261025`、af284d85からAST読込）を、canonical candidate IDで使う。eventは既存 `sample_rte_events`（K2のnormal law）で生成する。seedは `14788159701244282917` |
| 8 wrapper/compiler | native `build_native_hadamard_wrapper`（測定付きX/Y）、`compile_cached`（Qiskit1.3.0/rz,sx,x,cx/opt1/seed17、hash照合）、compiled vectorの1e−7照合 |
| 9 C guard | sidecarは `method=PR`、`route=PR_PAULI_TAIL`、`PAULI_D2_CENTRAL_ROTATION_CONTROL_9C4006BF`（source 9c4006bf `h4_d2_pauli_control.py`）。scalar SOURCEはA辞書の `saved_DF_scalar`。C `cfc0fae6` のguard（blobから読込、mergeなし）で受理を確認した |

- **2 process構成**：子processで元 `install()` を最初に呼び、provider root（c31ef424とbyte一致）の上で動かす。provider rootに無い `pauli_signal_adapter`/`h4_d2_pauli_control` は、9c4006bf blobを `trotterlib` 配下へ登録する（現treeとbyte一致を確認）。
- **mode**：既定はmetadata preflight。`--execute-approved-pauli-b0`（別承認）だけがsampling・build・compile（最大2）を行う。
- **legacy physical candidate**：`261e184236a8b95246f7` を作るが、辞書を含まないため、照合の正本はcanonical identity `RMSE:62a9a6b6…`（`d905d219…`）とする。CのphysicalID bridgeとの接続条件はHANDOFFに記載した。

## Aへの照合結果：保存P0のbit順

[P0_BIT_ORDER_CHECK.json](P0_BIT_ORDER_CHECK.json)。時間発展・signalは計算していない。保存演算子の表現の照合だけである。

| 項目 | LD1 | LD2 |
|---|---:|---:|
| `label_in_saved_tensor_order`＝`label_qiskit` の反転（184項） | 全一致 | 全一致 |
| ‖保存Pauli − Σc·P(saved label)‖ | 6.2e−16 | 5.8e−16 |
| ‖reverse_bits(保存Pauli) − Σc·P(qiskit label)‖ | 6.2e−16 | 5.8e−16 |
| 変換しない場合の差（参考） | 4.75 | 3.62 |
| ‖head − Σblocks[0..p]‖、‖tail − Σblocks[p+1..]‖ | 0、0 | 0、0 |
| ‖tail − Pauli − s_P I‖／‖tail − Pauli − s_R I‖ | 7.1e−15／1.8e−15 | 2.2e−15／1.5e−15 |

保存arrayはmode-0-MSB順で、Aの `reverse_bits` で正しくQiskit順へ移る。変換を省くと大きく食い違う。

## 作業B：Full PF同条件価格caller

[`scripts/run_full_pf_shortlist_prices.py`](../../../scripts/run_full_pf_shortlist_prices.py)。

- **再利用**：Truncated T3 callerの検査関数をimportして再利用する（T3ファイルは不変）。同じlong-time native決定論path・wrapper・compiler・価格quantityを使う。差分はrowの `method=FULL`、`prefix=10` だけ。
- **Full PF固有の検査**：
  - 全11 block（one-body＋fragment0..9）を保持すること。
  - `prep.extracted_identity_coefficient==0`。tail identity補正を加えない。
  - scalarはHamiltonian定数cだけであること。
- **構成**：C COVERAGE（3a8b37f1）のconstructorで再生成する。S2 16h/7 `11cf8ca6…`、Y4 16h/3 `a799697f…`、M5 16h `70f7b317…`。T3_CANDIDATESのfull_pf_shortlistとも一致する。
- **cellの役割**：m4/m5を較正、m6をholdoutとし、Cの計画（cfc0fae6 PRICE_VALIDATION_PLAN）と一致させた。

  | 公式 | m4 | m5 | m6 |
  |---|---|---|---|
  | S2 | q7 | q14 | q28 |
  | Y4 | q3 | q6 | q12 |
  | M5 | q1 | q2 | q4 |

  physical IDはCのCOVERAGEおよびNEXT_PRICE_VALIDATIONと一致した。Truncated p7とはq・T・実幅・piece数が同じ対になる。
- **保存receipt**：M5 m4（`0cd1527c…`）はX/YともEXACTの保存direct（155529e20 exploration）があるため、**再compileから除外**し、Cの受入用locatorとして渡す。新規は**8 cell／16 wrapper**。
- **出力**：FULL_PFのprice record・sidecar（`NOT_APPLICABLE_NO_RTE`、`HAMILTONIAN_CONSTANT`）、proof、Cのfit入力（`price_model_validation.record`、features one・q、CALIBRATION/HOLDOUT）。Truncated p7のPRICES.jsonも、同じ関数でfit入力にできる。

## 人工・技術検証

[TESTS.txt](TESTS.txt)：**新規16件・回帰30件、すべてPASS**。

- **Pauli（toy 4-qubit DF＋toy辞書）**：toy "P0" はnativeと独立なJordan–Wigner（保存mode-0-MSB順）で構成した。元 `prepare(ham,1)` のnative headとPauli D2 tailの回路が独立oracleと一致した。
  - low-controlの恒等性、leakage、X/Y符号、scalarの一回会計、compile経路（compileは計数のみで置換）を確認した。
  - 拒否・検出の確認：bit変換を省いた場合、二重変換したoracle、headの取り違え、辞書（saved label・保存Pauli）の不一致、s_Rの不一致、scalarの二重適用。
  - 実P0の照合（上表）、凍結した計画とseed、C guard（受理と、route欠落・ZZ lowering・scalar不一致・DF用PR policyの拒否）。
- **Full PF**：toyで全block（S2/Y4/M5の負時間pieceを含む）、scalar＝c、tail identity=0を確認した。保持blockの不足・B0I row・tail identityのあるprepは拒否された。
  - 実構成のcell、ID、役割、保存receiptの除外、Truncated p7との対を確認した。
  - Full＋Truncatedを同じpolicyでC guardに通し、scalar quantityの取り違えが拒否されることを確認した。
  - C fit入力の形式（架空の値で `fit` のinterfaceだけ確認）。
- **回帰**：Pauli接続8（seed規則とC route IDの更新後）、S-1 7、P-1 3＋4、T3 3＋5。

作者自身の技術確認であり、独立レビューではない。

## 残るUNKNOWN

- **Pauli**：実H4のcontext読込、P0 block[1]がfragment0であること（head和とは一致。個々の順序は実行時のoracle一致で確認する）、実回路の規模とcompile時間、C physical-ID bridgeの最終形、平均単価のlaw期待値（B1以降）。
- **Full PF**：保存M5 m4の数値受入（C）、新規8 cellの実compile結果、q-affineの外挿資格、holdout許容差（ユーザー）。
- **共通**：長時間round、law/rare、N/G。

commit/push・remote照合後 **mandatory STOP**。
