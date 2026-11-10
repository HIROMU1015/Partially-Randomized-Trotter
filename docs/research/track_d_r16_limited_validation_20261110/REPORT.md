# Track D × PR＋RPE：r32→r16 限定科学検証

**S1全13roundと、S2の固定q4・1 path検証が条件付きPASS。** このr16 pathではReferenceがStrongよりCXとdepthを削減した。一方、r16の長roundでは減衰と十分Nが増え、PR＋RPE全体の総費用優位は今回判定していない。

## 固定条件と実施範囲

Linear H4／隣接1.00Å／STO-3G／Nα=Nβ2、元H exact ground仮定・状態準備費除外、number-CDF8／元frame順LD2／ordered partial S2/s1/r16/K0／h=δ=.1。13round m0..12/q1..4096、εE=.00159Ha、αtotal=.01、各X/Y軸α=.01/26、βtotal=π/6、βstat=π/12、prior[-10,10]と旧decoder/φ=−hEを維持。

元H packet `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db`、CDF endpoint `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`。新A Q、Closed P3、α最適化を組み込まない。実行前条件は [PLAN](PLAN.md)。

## S1：元H誤差と十分shot数

全13roundで条件付き十分条件が成立。r非依存の保存PF誤差・DF norm＋guardを再利用し、r16 RTE law・kernel・誤差・attenuation・Nを新評価した。own-DF groundの信号/Nは転用せず、ground solve、DF再fit、state-vector評価は0。

q4096のA=.726109048236092、元H corrected operator誤差=.15553812061881345、半径下限=.6131714115091349、deterministic phase上限=.15617218029996552rad、phase slack=.10562720749918392rad。Nは各軸1359（保存r32は910）。条件付きenergy誤差上限=.0012783173232380342Ha、統計失敗上限=.01。詳細と13round表は [R16_RPE_ACCURACY](R16_RPE_ACCURACY.md)。

理想finite-RTE IID／exact-native／数値sector norm前提への条件付きで、rigorous enclosureやactual implemented RPE成功保証ではない。

## S2：固定r16 q4・1 path

S1 PASS後だけsamplingを開始。`r16_q4_sample00`、事前固定4 seed・64 event、whole-path一gadget、complete step2 cut、C1追加融合OFF。raw全22チェック＋paired X/YがPASS。StrongのGaussian共有・directional制御・head relative・tail basis-run共有・step境界融合・scalar集約を維持した。

P0 compile4回、postcheck4件、最終paired X/Y・χ照合が全PASS。原precision、MPFR160宣言phase保持、component abs1e−10／observable Frobenius2e−10を維持。X/Y共通の価格：

|資源|Strong|Reference|Reference−Strong|
|---|---:|---:|---:|
|RZ count|13,042|13,333|+291|
|CX count|5,434|4,874|−560|
|RZ depth|3,114|1,594|−1,520|
|CX depth|2,752|1,072|−1,680|
|Total depth|7,495|3,568|−3,927|
|Circuit size|26,861|26,784|−77|
|Logical qubits|9|17|+8|

Referenceの追加register、16 CSWAP、χ補正、cut融合損失、readoutを含むfull-wrapper価格。CX約10.3%減・depth約52.4%減とRZ増・幅増が同時に観測された。詳細は [R16_CIRCUIT_RESOURCES](R16_CIRCUIT_RESOURCES.md)。1 pathの技術・価格証拠であり、law母平均やr変更だけの因果効果とはしない。

## SOURCE・再利用・失敗保全

- S1 SOURCE：`9ec3da171ca7065217cd59887b2666d433479974`、S2 SOURCE：`08f15308ef16429da7e375b1f82e7799510d2578`。
- 既存canonical PR/RTE、B1 operator transport、B4 shot API、C3 whole-path、C6a/B2 CAR reference、位相保持compiler、既存batched checkerを再利用。旧r32読取専用resource interfaceはr16へラベル替えしていない。
- 人工回帰はS1/B1/B4の58件、S2/C3/state-actionの11件がPASS（共通3件を含む）。初回test起動時のPYTHONPATH不足と既存fixture環境変数不足は記録し、テスト本体・閾値を変更していない。
- S2最初の技術preflightは不要なgmpy2 metadata要求で停止。sampling／回路／compileは0。初回SOURCE・traceback・空directoryを保持し、実際のMPFR処理が使うSymEngineへmetadataだけ修正して別private subdirectoryで初回科学実行を行った。科学retry／resampling／閾値緩和は0。
- SOURCEとinputの前後hash、保存J/U・sector・PF kernelの対応を確認。既存SOURCE/raw/result/grant/ledger/marker/KEEP/STOP、元main・他worktreeは無変更。

S1約2.86秒、S2約272.09秒。raw入力全256、Reference出力131072 amplitudes/columnを既存Aer checkerで検査した。H4の大きなfull17-qubit Operatorは構築していない。

## 保全・公開・限界

新namespace `artifacts/track_d_r16_limited_validation/2026-11-10/run01/`。公開はSOURCE、input/hash索引、law、精度・N、correctness、資源比較、STOP。ordered event、QPY、NPZ、内部環境情報・詳細tracebackはlocalのみ。公開raw payloadの追加はない。

今回の上限はS1＋1path／4compileで終了。別qの回路、別条件探索、r16期待価格model、A/F/S refit、全13round総価格、actual成功確率、ground solve、DF fit、詳細科学レビューは行っていない。

推奨は、r16を候補として残す技術証拠を受け取り、N増加を埋める期待価格改善を追加校正する価値をユーザー／GPTが判断すること。r32既存A/F/Sはその条件内の参照軸に保つ。正式GOや研究方針をCodexが採択しない。[次段階案](NEXT_EXECUTION_PROPOSAL.md)。

専用branch `codex/track-d-r16-limited-validation-20261110` へcommit/pushし、remote SHA・tree・今回公開filesのbytesを照合する。最終SHAと照合結果はチャットとlocal publication記録へ残す。科学処理の[STOP](../../../artifacts/track_d_r16_limited_validation/2026-11-10/run01/STOP.json)を保全し、公開照合後もmandatory STOP。次の担当はユーザー／GPT。
