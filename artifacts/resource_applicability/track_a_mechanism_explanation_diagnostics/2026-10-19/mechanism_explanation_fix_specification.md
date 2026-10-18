# 技術修正仕様

新SOURCE：`b6716396acc5760fde7cf7cc6f2481b183a08f10`。元SOURCE `5c256067db29a326bcaa0a0e244b38f6b87bd138`、core module、旧runner／preparer／人工64件、結果2bf5d832…は不変。

追加したfileはmechanism_explanation.py、run_mechanism_explanation.py、prepare_mechanism_explanation.py、test_mechanism_explanation.py。旧core算術を編集せず、説明専用の別入口を設けた。

|項目|仕様|
|---|---|
|step／累積|step_b、step_count、log_B_nominalを別名で保持。有限RTEはR*log(b)との対応guard、PFは保存nominal log_B=0と決定論SOURCE。PFのRTE専用fieldを補完しない|
|shot上界入力|元と同じbinary64順序でlog_B_nominal + 1.0*log_B_margin。EXP_OF_LOG_UPPER経路を事前固定し、結果に応じて経路を選ばない|
|saved upper|明示保存upperを使う別schemaはmath.log(saved_upper)==log_B_upperを厳密維持。今回のbindingはEXP経路|
|headroom|epsilon/sqrt(2)-abs(saved signed bias)-saved allowance。元保存値とのabs差1e-12を維持|
|bound|元u_aware_shotsと同じlog式、2*B_upper**2/h**2*log(2/alpha)の順序とfallback。saved log boundとのabs差1e-12|
|整数|log_Nstar<log(2**52)の保存BINARY64_DIAGNOSTICだけceil整合性を検査。保存Nを置換・最適化しない。大指数はLOG_DOMAIN_ONLY、整数未保存を区別|
|因子|nominal normalization²、empirical margin²、effective upper²、headroom²、alpha-log、連続shot、丸めを区別。表現不能なexp比はnull＋log値を保持|
|cohort／axis|methodごとのidentityにsystem/DF等binding、candidate、ε/factor/bias、PR cohort、H4 stage、両軸を結合。pooling・候補再選択を拒否|
|core継承|固定SHAの旧results.jsonを読み取り、32 IDsと既存受入flagsを確認して値をコピー。G／単価平均／比率を再計算しない|
|欠測と失敗|A82_OLD_PR.PRだけ既知欠測を許容。その他の追加欠測・数値UNRESOLVEDはfail-closed STOP。欠測sideが他sideの数値失敗を隠さない|
|認可|別scope TRACK_A_MECHANISM_EXPLANATION_PRIMARY32。旧消費grantはscopeで拒否、新しい明示承認receiptとone-shot grantが必要|
|I/O／native|新規専用run root、exclusive marker/ledger/worker claim、native exit0/wait/reap、結果hash・core転記一致・side statusを親も検査。失敗時は記録保持、retry／返却なし|

科学的な新仮定や新しい保証は採択していない。normalization_upperやallowanceは元のEMPIRICAL conditionalという意味のまま。1e-12を緩和せず、不明な不一致を通過させない。修正の実値への適用、残るceil境界・headroom・log guardの一致、最終的な説明成立は別承認後に判定する。
