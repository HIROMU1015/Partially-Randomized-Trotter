# C-S V0：保存入力との技術binding

2026-11-08 JST／SERVER1 GPUA100-gleap-01。**静的schema・bytes確認と人工検証だけ。実H4 slot選択・角度導出は未実行。**

## 固定対象と正本

保存linear H4／1.00 Å／STO-3G／legacy DF12／8 system qubits／全4電子sector参照／L_D3／m0,q1／δ=.02,r1,K2／coefficient_atol=0。N1 actual finite-grid Qとselected full policyを変更しない。N2 stateは参照identityだけで、vectorを読まず、使用しない。

- N1 `744e1a09661ebba6841fe5faf456f88c41c47780`：PREPARATION/LAW/POLICY/CONFIG/RESULT/SOURCE/STOP。
- N2 `5e9bd08068738cdf1070bdc43bee8194b113ae33`：SOURCE/STATE_RESULT/STOP。
- 旧engine `fdf00cee3c87d804845497afc89ac6a71f3b3f59`：保存UNIFORM_ERRORのD half-step記述、SOURCE/STOP。**旧engine上界をC-Sへ転用しない。**

13入力のpath/bytes/SHA256/Git blobは[SOURCE_AND_CHECKS.json](SOURCE_AND_CHECKS.json)。全ファイルが上記固定commitとbyte一致。静的preflightは全入力bytes・20 SOURCEファイル・runtime・依存package lock・pygridsynth/mpmath sourceを照合してPASS。入力合計bytesは保全用で、科学的な再集計ではない。

## 保存schemaと元SOURCE

| 内容 | 保存path／元実装 | V0での扱い |
|---|---|---|
| Dの順序 | PREPARATION.deterministic_blocks配列/order_index/block_id | forward sweepの保存順。reverse sweepを選び直さない |
| Dのdiagonal gate | UNIFORM_ERROR.inventory.deterministic_half_step_records[].half_step_gates | 元df_partial_s2._append_blockとdf_trotter.ops.append_diagonal_primitivesが保存したcrzz→crz→p。数値primitiveを再生成しない |
| basis境界 | PREPARATION.basis_operations、block.basis_change_operations、保存gate前後 | 先頭inverseと末尾forwardをschemaで対応。RZ/XXPlusYYのbasis gateもFIXEDで、名前だけでdiagonalに分類しない |
| Rの順序/role | LAW.label_masses、PREPARATION.components | 最初のpositive actual-grid label、order0 eventのrotation role。productはeligibleでない |
| Rのselected basis | POLICY.definitionsのsource_basis_id/hash/support→definition | 一意対応必須。原full basisへ戻さず、保存selected definitionをFIXED境界として返す |
| Rの実角度 | rte._make_event、df_rte_qiskit._append_central_application | 将来resolverで元のbinary64 atan→sign→2倍の順序を再利用。新しい回路は作らない |
| scalar/identity/event phase | D central p、保存aggregated scalar phase、R event phase/符号 | 全てFIXED。削除・periodicization・projective一致への変更なし |

Dのcentral gateはbasis配列の長さを使って前後を分離してから識別する。原SOURCEの順序はrzz一覧→rz一覧→scalar pで、保存D gate配列からそのまま読む。R0を選ぶのは元Q全体をorder0へ変更する操作ではなく、局所angle classの代表specを事前指定するだけである。V0では元pathをsampleせず、Q/CDFも作り直さない。

## 一意な選択規則と角度

`cs_v0.bind_slots(prep,law,uniform,r_angle,policy=...)` は**将来science call**。今回、実入力では呼んでいない。

1. D：canonical forward block順、各blockの保存central gate順で最初のcrz/crzz。
2. R：保存LAW順で最初のpositive label、order0の最後のrotation application。最初のlabelが非適格ならSTOPし、後続labelへ差し替えない。
3. 各logical controlled rotationを、joint Pauli上の `R_(Ianc P)(θ/2) R_(Zanc P)(−θ/2)` と記述する。**V0は各々factor0だけ、最大D1/R1＝2 joint局所spec**を資格付ける。factor1も返却してFIXED partnerとして保存する。両factorを全部合成する4spec scopeへ拡大しない。
4. Stable ID：`D/forward/<block_id>/central<index>/factor0`、`R/order0/rotation/<component_id>/factor0`。元bytesとpolicy identityへ結ぶ。word順はancilla=q[n]を左端とし、続いてq[n−1]…q[0]。

角度は保存されたcanonical float.hexをFraction.from_float相当で**厳密なbinary64有理数**に変換し、既存`wrapper_sequence.Angle(pi,rad)`へ入れる。数学的atan値を新たな高精度値で置換しない。Rだけは元event-angle処理によるbinary64パラメータ取得が必要なので、V0へ残した。これは旧state/engine認証の再計算ではない。

このbindingはfull circuit IRを再構築しない。非回転のbasis/Givens、wrapper、scalar phase、他slotの費用/誤差は今回の局所価格から除外して**未取得**と表示する。partnerを含む全列の測定等価性・full cost・N/Gを保証しない。

## 再利用と重複排除

基点c077de7のsame_position_residual.pyと9test、Angle、CanonicalGenerator、event_statistics、operator_target、error_accounting、actual adjoint/native_accountingを無変更で再利用する。Parentは`4ccc1832...`、nativeは`546bca7e...`のblobと一致。3環境統合`35714770...`の再利用地図と旧C-S計画`66553d94...`を確認した。

新規はslot/schema binder・元R角resolver・単一coarse規則・finite tableの外部誤差接続、strict角型/provider、static preflightだけ。F1/F2・P3・Parent/S4・SCV・C1/PS48の再実験、旧9testの再実行はない。正式共通指示2026-10-25 §8.1/8.2/8.4/10/10.1/10.3、AGENTS、PROJECT_MAP、概要全文・目的・方法・関連STOPを確認した。
