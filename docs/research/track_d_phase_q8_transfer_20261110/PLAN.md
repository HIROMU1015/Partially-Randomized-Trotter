# Phase精度確認＋保存q8一path移送：実行前PLAN

2026-11-10 JST。基点1d03ba9cbd04ff68ded88ec33aac3ae7553d4c46。既存Track Dリソース評価branchを継続し、新namespaceだけで実施する。旧SOURCE／結果／STOPは変更しない。

q8対象はC6b保存event_bindingsのsample ID昇順先頭、q8_sample00。価格やcompile資源を選定に使用せず、4 raw／4 compiled／eventのhash、科学SOURCE／law／packet、P0 profileを確認してSELECTION.jsonをcommit固定する。欠落や不整合時は別pathへ黙って変更しない。

科学条件はLinear H4／1 Å／STO-3G／Nα=Nβ=2、元H／number-CDF8／元順LD2／ordered partial S2/s1/r32/K0／h=.1、q8中央cut4、C1 head-tail追加融合OFF、元H厳密ground仮定・準備費除外。Strong symmetric-directional／whole-path一gadget ReferenceのX/Y。P0はQiskit1.3／opt1／seed17／rz,sx,x,cx／approx1／initially_zero=False／topology-free／宣言phase保持。

位相精度の最小修正は、既存gateのphase出力をexact binary64比からMPFR160へ変換して集約し、最後に160-bitの2πで正規化する独立ledgerだけ。局所gate／1Q合成の精度・角・順序・ruleは変更しない。旧binary64逐次値も保存する。q4保存raw4を再利用して旧位相差を再現し、native／bridge wordが旧private保存列と同一かを検査する。新full-wrapper／state simulationは作らない。

既存lowerer／T0／1Q／CX／q4観測器を無変更で再利用する。新callerはphase項のside recording、保存q8 I/O、並列task orchestrationだけ。人工位相テストと旧回帰を確認後、q4 phase4＋q8予測4の独立8 taskを実空き資源に応じて並列処理し、時間／peak RSS／native長／反復／frontierを記録する。q4再処理は位相原因調査のためで、過去の価格・性能結果を上書きしない。

q8予測workerには価格・compiled QPYを渡さない。予測SOURCEとPREDICTIONS.jsonをcommit固定してから別compare処理で保存P0を照合する。初回予測や不一致を保持し、価格合わせのrule追加／補正定数／閾値緩和はしない。完成wrapperのglobal phaseとreadoutに効く相対χを区別し、全入力correctnessや一般P0／全q保証を新認定しない。

新sampling／whole-path生成／P0／ground solve／DF refit／全入力simulation／q16以上／fit／N／全13-round更新／詳細科学レビューは0。旧grant／ledger／marker／STOP、他worktree／process／共有設定を変更しない。角列・phase項・raw/QPY/NPZはprivate localに保全し、公開はscalar／hash／SOURCEと技術コードだけ。公開ref/tree/bytes照合後mandatory STOP。次の採択はユーザー／GPTへ戻す。
