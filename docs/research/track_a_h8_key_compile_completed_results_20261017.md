# H8主要3構成・6軸：直接compileの完了結果（2026-10-17）

固定6軸はすべてPASS。2026-10-17 **22:15:38 JST**に正常終了し、最終gateもPASS。
今回の主要3構成では、固定RZ費用モデルとの差は0.216〜0.827%で、全軸が2%未満の契約条件を満たした。

## 条件と実測

Linear H8、隣接間隔1.0 Å、STO-3G、保存DF rank28、16 spin modes、sector dimension4900、T=0.8。
prefixは11または15、qは1または4（基本刻みT/qは0.8または0.2）。
保存signed generation ordering、unrotated full-H reference、B0+Iのown-prefix full-Fock scalarを維持。
S4は負の中間stepを含む非結合3段S2 Yoshida。compiler/profile/optionsは固定のまま。

各構成でcosine/sineの両軸を測定した。以下のRZ・CX・depthは**1軸あたり**で、各構成の両軸で一致した。

| 候補 | 実測RZ/軸 | 実測CX/軸 | depth/軸 | 固定モデル相対誤差 | 判定 |
|---|---:|---:|---:|---:|---|
| B0+I p15/q4 S4 | 668,636 | 297,968 | 221,226 | 0.216902% | cosine/sineともPASS |
| B0+I p11/q1 S2 | 42,268 | 18,576 | 13,786 | 0.827036% | cosine/sineともPASS |
| B0 p15/q4 S2 | 223,940 | 99,600 | 73,930 | 0.215874% | cosine/sineともPASS |

誤差定義は `abs(prediction-measured)/measured`。予測は順に4670300/7、298323/7、1564196/7 RZ。
事前固定の係数を変更せず、追加fit・許容差緩和・追加候補compileは行っていない。
実測値はp15/q4の2構成でモデルより僅かに大きく、p11/q1 B0+Iでは僅かに小さい。

## 検証と解釈の範囲

native方向制御と保存corrected/independent-forwardとの誤差最大は順に
1.95685e-13、1.37080e-14、5.05408e-14で、1e-8未満。
各軸のcompile前後のwrapper fingerprintが一致し、native worker exit0・reapedを確認した。
SOURCE closure26件・入力11件のbyte照合、軸raw結果とnative receiptのSHA、過去61証拠の不変、
logical/new-actual/reuseの会計を保存値から限定監査しPASS。
終了時runtime全体の照合は既存のfinal gateが実施済みであり、今回の要約作業では再実行していない。

主要候補のRZ単価について、大きなモデル誤差はこの6軸では認められなかった。
ただし、この結果は184候補すべての直接実測や、全候補実測最良の証明ではない。
**PR削減率の更新は未実施**。通常signed bias・経験的非相殺のshot重み付き費用再会計には別途実行承認を要する。
厳密bias保証・PR母平均の優位は主張しない。

## 実行と来歴

- 最終実行SOURCE：`41e466e13f32e9224bb37790cc16f0ff6621c399`。
- 旧SOURCE：`352581103b7dbd4fa58e28af7a6ca65570d36e8e`。
- 旧runで完了したB0+I p15/q4 S4 cosineの1軸と、同構成の検証済みQPY2本を独立受入。
  cosineのSOURCEは元のまま保持し、残り5軸を新SOURCEで取得した。
- 新run：`attempt02-parallel-handover`。新launcher wall **24分23.886秒**（引継ぎ待機を含む）。
  初回run開始から最終終了までは **約1時間8秒**。並列化だけの速度比は測っていない。
- 新runの最大単一process RSSは **9.643 GiB**。これは同時稼働processのRSS合計ではない。
- 新runの数値準備は残る2構成を並行実行し、各構成の2軸workerも並列化。
  compileは固定優先順で1軸ずつ、各軸のモデル判定後に次軸へ進んだ。
- native driver/launcherともexit0、全worker回収、cleanup error0、監視292 sample。
- 新run actual compile5＋再利用1＝登録6軸。旧runのcompile予約2のうち中断1も元記録に保全し、返却・リセットは行わない。
- 旧STOP・grant・ledger・markerと他jobは変更していない。既存venv変更・GPU query/use・新SCF/DF/state・
  新signal/reference・PR sampling・fit・費用再会計は0。

[全6軸の軽量結果](../../artifacts/resource_applicability/track_a_h8_key_compile_completed_six/2026-10-17/six_axis_results.json)、
[保存値監査manifest](../../artifacts/resource_applicability/track_a_h8_key_compile_completed_six/2026-10-17/result_audit_manifest.json)、
[引継ぎ記録](track_a_h8_key_compile_parallel_handover_20261017.md)。

raw private入口：`/home/AbeHiromu/projects/h8-key-candidate-compile-execution/20261017/attempt02-parallel-handover/`。
新しい科学計算・再会計は開始せずmandatory STOP。

次の段階：ユーザーが実測6軸を用いた通常・非相殺費用の限定再会計を承認するか判断し、承認後にCodexが実施する。
