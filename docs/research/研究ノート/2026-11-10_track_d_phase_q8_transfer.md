# Track D Phase精度とq8への限定移送

2026-11-10 JST。[結果・条件](../track_d_phase_q8_transfer_20261110/REPORT.md)。既存リソース評価branchを継続。入口のh表記1/10に対するschema不備を491109160に保全し、有理数比較へ修正。数値実行SOURCEac5f7327、予測固定0cd6de18でraw4を先に固定し、後から保存P0と比較した。

q4の約4.5e−9 rad差は逐次binary64集約が主因で、局所出力を変えないMPFR160 ledgerにより約4.7〜9.4e−12 radへ縮小。新旧native／bridge列・価格は同じ。q8_sample00／cut4／C1 OFFでも全28資源が一致、phase残差は約1.3〜1.8e−11 rad。全工程160-bit化や新H4作用PASSとはしない。53人工・技術回帰PASS、旧54人工結果は不変維持。

q8は約43万native命令・20反復、各worker55〜57秒・491〜494 MiB。次は全P0再実装ではなく、同じruleで再走査／履歴を減らす技術案を判断する。A/F/Sを保ち、費用対効果が低ければPR＋RPE共同最適化へ重点を移す代替案を示す。科学的GO／研究終了は採択しない。

新sampling／whole-path生成／P0／ground solve／全入力simulation／q16以上／fit／N／13round更新0。旧SOURCE／結果／STOP・非公開原本を保全し、公開ref／tree／bytes照合後mandatory STOP。次の担当はユーザー／GPT。
