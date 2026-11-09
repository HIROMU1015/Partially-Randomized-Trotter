# Phase精度確認と保存q8へのNative資源評価移送

2026-11-10 JST。**保存q8_sample00のStrong／Reference X/Y、計4 wrapperで、rawから先に固定した全7資源の予測28値が保存P0と一致した。** 価格補正・最適化rule追加は0。q4の約4.5e−9 radの宣言global phase差は逐次binary64集約が主因で、同じphase項のMPFR160集約により約4.7〜9.4e−12 radへ縮小した。q8の修正後差は約1.3〜1.8e−11 rad。全工程の160-bit精度や新しいH4作用保証は認定しない。

[Phase結果](PHASE_PRECISION.md)／[全28資源](Q8_RESOURCE_COMPARISON.md)／[性能とq4比較](PERFORMANCE.md)／[次段階案](NEXT_EXECUTION_PROPOSAL.md)／[PLAN](PLAN.md)／[STOP](STOP.md)。

## 固定条件・選定・来歴

Linear H4／隣接1.00 Å／STO-3G／Nα=Nβ=2、元H／number-CDF8／元frame順LD2／ordered partial S2/s1/r32/K0／h=δ=.1。元H厳密基底状態仮定・準備費除外。q8のcomplete中央cut4、C1 head-tail追加融合OFF、既存強いsymmetric-directional対照とwhole-path一gadget Referenceを維持した。初期state-vectorを使わず、保存raw bytesを入力とした。

P0はQiskit1.3.0／opt1／seed17／rz,sx,x,cx／approx1.0／qubits initially zero=False／topology-free／既存宣言global phase規約。新P0は0。

対象はC6b event_bindingsのsample ID昇順先頭 **q8_sample00**。候補価格を読まず、raw4／compiled4／eventの9 memberのbytes／hash、SOURCE／law／packet／compiler、保存conditionsを確認し、[SELECTION.json](../../../artifacts/track_d_phase_q8_transfer/2026-11-10/run01/SELECTION.json)をcommit固定した。元のq4は別C6a calibration pathであり、同じstepを倍にした構成ではない。q8の1 pathを32本全体の標本統計へ外挿しない。

q8 event／raw生成SOURCEは23725fc6f6d6d0c2bec93e31e32074711804e250。保存完了SOURCEは44a58b9fba5d3aba7a31d0a6b12e3aa25be9def4。compiled4には両SOURCEの生成物が含まれ、各memberのgenerated_by_sourceを索引通り保持した。入力packetは1d6087f6…、lawはc347d7f2…、CDF endpointはa8813aca…で旧条件と同一。完全hashはSELECTION／元索引に保存する。

q8のevent SHA256は `9cfedf5c557af6609ad75c6218314be636076fe6ae23aabb453c9e01d124094c`、8 step×32 event。保存元の原本をread-onlyで使い、SOURCE／input／raw／compiled／旧結果／grant／ledger／marker／STOPは変更していない。原本のコピーやseedからの再生成も行わなかった。

## 実行の固定と技術修正

既存リソース評価branchを継続し、基点は1d03ba9cbd04ff68ded88ec33aac3ae7553d4c46。最初のselection preflightは、保存hの文字列 `1/10` と数値.1を直接比較した入口のschema不備で停止した。科学条件の差ではない。停止記録と修正前SOURCEを **491109160** に保全し、有理数として厳密に1/10と比較する修正を行った。数値実行・予測はその時点では未開始だった。[FAILED_select](../../../artifacts/track_d_phase_q8_transfer/2026-11-10/run01/FAILED_select.json)を削除していない。

修正済み実行SOURCEは **ac5f7327bb2812b664b9205de4ef8db0323db9e6**。旧lowerer／T0／1Q／CX／q4観測器／既存MPFR compilerはbyte不変。新しいphase ledgerとcallerだけを追加した。局所合成角・命令順・最適化rule・閾値は変更していない。

q4 phase4とq8予測4を独立8 taskで並列処理し、**0cd6de18fea06c858592f60d3b310e67eb759ce8** でq8予測とSOURCEを固定した。その後、別compare処理でq8保存prices／compiled4を開いた。予測関数へ価格やcompiled列を渡さず、価格補正定数も使わなかった。初回のq8数値予測で28値が一致し、数値retryは0。

## 結果と保証の範囲

| 方式 | 軸 | 資源一致 | Native命令 | Bridge反復 | Worker時間 | Peak RSS |
|---|---|---|---:|---:|---:|---:|
| Strong | X | 7/7 | 429400 | 20 | 55.207 s | 491.238 MiB |
| Strong | Y | 7/7 | 429405 | 20 | 55.431 s | 491.461 MiB |
| Reference | X | 7/7 | 430996 | 20 | 56.426 s | 491.316 MiB |
| Reference | Y | 7/7 | 431001 | 20 | 56.634 s | 494.023 MiB |

q4位相調査とq8を含む全8 taskのwall timeは59.219秒。これはq8単独4並列の計測ではない。q4旧実行の約25秒と直接speedup比を作らず、各worker／stageと実行条件を区別して比較した。

q4では旧binary64 phaseを再現し、新旧native／bridge wordは4 wrapperとも完全に同一、全7価格も不変だった。修正は完成wrapperのglobal phase集約だけであり、P(χ)等の相対位相gateを変更しない。完成wrapper全体のglobal phaseはX/Y確率から消える。一方、相対χの変更は測定へ影響することを小人工テストで区別した。新しいH4全入力simulation／observable／reference leakage検査は0で、保存correctness PASSを新PASSに数えない。

人工・技術回帰は新4＋旧49＝重複なし53件PASS。旧54人工caseの固定PASS結果と旧SOURCEも不変維持した。新P0 transpileは人工分も0。未知のphase残差や一般P0 domainを価格一致で適格化しない。

## 研究上の判断材料

q8でも大規模compiler再実装なしで28値が一致したため、限定的な構造評価の継続には価値がある。反面、q4の10反復からq8の20反復へ増え、bridge時間はXで約3.4倍、記憶は約1.75〜1.76倍となった。異なるpath／cutと数値会計・並列条件もあるため、qだけの因果効果や全q則にはしない。

次の焦点は、既存ruleを変えず反復・履歴・再処理を減らせるかであり、全P0の複製ではない。q1024〜4096の実用速度／RAM／P0対応・全law期待価格は未確認。A/F/Sは変更せず条件付きモデルとして残し、限定改良の費用対効果が低ければPR＋RPE共同最適化へ重点を移す代替案を提示する。Codexによる正式な科学的GO／研究終了の採択は行っていない。

新sampling／whole-path生成／P0／ground solve／DF refit／全入力simulation／q16以上／fit／N／全13-round更新／詳細科学レビューは0。公開は科学scalar・hash・SOURCE・技術コードだけで、raw/QPY/NPZ／native角列／phase項／内部保存先は公開しない。公開ref／tree／bytes照合後mandatory STOP。次の担当はユーザー／GPT。
