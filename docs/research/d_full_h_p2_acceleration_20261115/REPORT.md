# Full-H P-2のprocess並列化・高速化

**現在の最終状態：** 旧SOURCE887は保存結果を保全して計画的に終了し、SOURCE0dcの残検査へ移管したが、同workerが予期せず終了して [mandatory STOP](../d_full_h_p2_completion_20261115/REPORT.md)。18 compile保存済み、最終path検査16軸PASS/2軸UNKNOWN。以下の実行中という記載は過去の観測である。

**再発防止の追補：** [RELIABILITY_FIX.md](RELIABILITY_FIX.md)でcheckpoint wrapper内部の遅いfingerprintへのfallbackを人工testで再現し、適用範囲の最小修正後に9 test PASSを確認した。実行中SOURCE887には未適用で、workerは継続中。新H4試運転・追加compile0。

**現在の状態：許可済みcheckpoint引継ぎを実施し、新SOURCE887で実行中。** [TRANSFER.json](run01/TRANSFER.json)に旧workerの終了・保存checkpoint・新namespaceへの引継ぎを記録した。path0は旧sampling/compileを再利用し残り検査を実行、path1は初回samplingとX/Yの並列compileを行う。旧parentのSTOP_CELL/exit2は計画的な引継ぎであり、科学gate失敗ではない。以下の未適用・拒否・旧worker継続の記載は、その時点の履歴として保持する。取得済み4 cell・14 compile PASS、D7全体の引渡しは未完了。

**追補：ユーザーがcheckpointからの引継ぎを許可した。** 元の拒否記録は以下に保全する。新controller `scripts/continue_full_h_p2_checkpoint.py` と [FROZEN_CONTINUATION.json](FROZEN_CONTINUATION.json) を固定し、path0 native trialの両compile・7数値gate PASSを引き継ぐ。path1は未着手、新compile上限2、D7累計上限18。[TRANSFER_TESTS.txt](TRANSFER_TESTS.txt) は9 PASS（H4追加試運転なし）。実際の切替結果はrun01のTRANSFER.jsonと継続結果に別途記録する。

**実装・synthetic検証済み。実行中G36223への切替は未実施。**

ユーザーの「並列化と高速化を行って」に対応した。稼働中SOURCE `d323fee1ca8cd6893c4b705f990cdc931bdb5c87` を保全するため、同SOURCEから専用worktree/branch `codex/d-full-h-p2-acceleration-20261115` を作成した。既存D1 branch・旧SOURCE・旧raw・旧STOPは変更していない。

## 1. 実装

| 対象 | 変更 |
|---|---|
| path | 各pathを独立processへ分離。providerの可逆swapやseed状態をprocess間で共有しない |
| X/Y | 既存nativeのcompile callbackからX/Yを別processで一回ずつcompile。先行生成したYのfingerprintと、後で既存nativeが生成する実際のYを完全一致で照合する |
| compiled-vector gate | representative compiled circuitを既存native QPYで親へ渡し、既存のcompiled-vector検査を維持する |
| controlled rotation | 同じsupport数・binary64角度のcontrolled gateの構築をmemoize。prototypeをcopyして使い、命令順・qubit・parameter・global/control phaseを維持する |
| legacy fingerprint | 既存の同一canonical bytesを出すfast writerを、trial外のlegacy再構築にも適用する。既存contextが有効なら重ねて切り替えない |
| 計時 | buildごとの時間と、X/Y compiler processの時間を残す |
| checkpoint再利用 | 保存済みnormal eventからrequestを直接復元。元trialの数値gateを引き継ぎ、元compile recordとのwrapper fingerprint一致を確認して、parity・scalar/event phase・X/Y検査を完了する。sampling/compileは再実行しない |

新entry：[run_full_h_kappa2_p2_accelerated.py](../../../scripts/run_full_h_kappa2_p2_accelerated.py)。既存P2 cell executorとrecord producerを再利用する。helper：[full_h_p2_acceleration.py](../../../scripts/full_h_p2_acceleration.py)。元P-1/P-2、native、provider、wrapper、lowering実装、compiler設定のファイルは変更していない。

worker数は独立path数、各pathのcompiler worker数はX/Yの2つで決まる。CPU/RAMへ新しい人工上限を設定していない。Qiskit 1.3.0、rz/sx/x/cx、opt1、seed17、P0、測定付きfull wrapper、new ZZ、scalar、固定D7 physical/cohort/seedは維持する。単一回路のtranspiler設定は既存と同じである。

## 2. 検証

[TESTS.txt](TESTS.txt)：**8 PASS**。主agent自身によるlocal検証であり、独立レビュー・CIではない。

- X/Yを別PIDで動かし、barrierにより実際の同時実行を確認した。
- legacy controlled gate再利用のcanonical fingerprint完全一致と、toyの全unitary一致（global/control phaseを含む）を確認した。
- checkpoint復元中はsampler/compileの両入口を拒否するmockを置き、どちらも0回で元events/fingerprintとphase検査が一致した。
- pathも別PIDで同時実行し、path index順でreceiptを収集した。
- 旧workerが生きている場合、元pathが既に着手されている場合、元native STOP／不完全数値gateの場合の継続を拒否する。
- 人工compile失敗で再試行せず、既に開始したworkerを回収してbuilderを復元した。途中で見つかったclosed pipe再読取の不具合は修正済み。
- **小さい4-qubit toy＋ancilla**では実Qiskit transpileもX/Y並列で2件行い、QPY受渡しとcompiled-vectorを含む7 gateをPASSした。これは実H4の試運転ではなく、H4 compile件数は追加していない。

100回同一controlled-ZZをappendするtoy benchmarkでは、gate構築100回を1回＋copy再利用99回へ削減した。時間の実測はTESTS.txtにある。**この比率を実H4全体の速度改善率と扱わない。** 新SOURCEでの実H4 wall/RSSはまだ未測定。

## 3. 実行中jobと適用境界

G566/P2R566/Q8R566/G4528は旧callerで完了し、計14実compileがPASSした。G36223は旧worker PID 3155960で継続中。現在のjobに新しいcodeを差し込んだり、科学SOURCEを書き換えたりしていない。完了2 cellのreceipt gzip化は並列に行い、元bytes一致を確認した。

新entryは、正常な旧workerがまだ動いている間の同じpathの起動を拒否する。したがって、旧workerを走らせたまま同じpathを別processへ再投入する方法は使っていない。

### 提案するcheckpoint引継ぎ

1. G36223のpath0 native trialがPASSし、X/Y両compileと7数値gateが保存されたことを確認する。途中のcompileを切らない。
2. まだpath1が始まっていない境界で、元job・SOURCE・events・compile record・QPYと必要なhashを保全する。path1が始まっていれば、不完全な計算を捨てず、次の保存済み境界を待つ。
3. **ユーザーが旧workerの終了を明示的に許可した場合に限り**、本人の該当processを終了して新SOURCEの継続へ引き渡す。旧terminal receipt・旧STOPを保持し、科学的gate失敗と計画的切替を区別する。
4. 新namespaceで、path0の保存sampling/compile/数値gateを再利用して残り検査を進める。同時に未着手path1を固定seedで一回取得し、X/Yを並列compileする。path0で2 compile取得済みなら新compileは最大2、全D7累計最大18のまま。
5. 実際のgate失敗・証拠不一致ならmandatory STOP。retry・別seed・条件緩和はしない。

これは新candidate・追加path・HOLDOUTを増やす提案ではない。G36223が自然完了した場合はその保存結果を使い、取得を繰り返さない。

## 4. 自動承認レビューによる制限

上記の旧workerをSIGSTOP/SIGTERMして継続へ切り替えるscript案は、**実行前の自動承認レビューで拒否された**。理由は、進行中の科学計算を損失・中断させるリスクがあり、「高速化」の許可だけではworker終了の許可が明確ではないため。拒否されたscriptは作成・実行していない。迂回もしていない。

旧workerは現在も継続している。**コードの実装・検証・公開まで完了させたうえで、適用に必要な旧worker終了の許可だけを確認する。** 承認証明書・grant・承認用台帳等は作成していない。

## 5. 科学的範囲・ロードマップ

固定D7 5 cell・最大18 H4実compile、有限cohortのみを維持する。H4の追加試運転、fit/係数凍結、m5/m6 HOLDOUT、長時間外挿、law E[C]、N/Gは0。CI/confidence、sampler/ideal-law距離、rare寄与、m7–m12資格は未解決のまま。

ロードマップへの影響は、取得経路のprocess並列化・legacy構築高速化の技術実装が整ったこと、実H4での適用と性能確認は未完了であること。ロードマップ本体は編集していない。旧計算が完了、または許可された引継ぎ後に、元のC向け価格・実装証拠の公開作業を完了する。
