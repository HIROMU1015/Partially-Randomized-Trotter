# SOURCE接続・guard・保存・resourceの技術設計

[固定SOURCE S](source_freeze.json)、[再利用監査](source_reuse_audit.json)、[静的検査](static_inspection.json)、[テスト](synthetic_test_results.json)。新SOURCEは [a2_pilot](../../../src/trottercert/a2_pilot/) に隔離し、旧SPEC-01〜07のSOURCE/結果は変更していない。

## 接続と独立性

`adapter.inspect_snapshot` はfile/member/array bytesとNPY header・既存Unicode metadataの検査だけ。`adapt_snapshot` は後日の認可済みinput変換callであり、Phase Aでは人工NPZにだけ適用した。数値library/np.load/pickle・暗黙cast・Hermitizationを使わない。NPY v1/v2、C order、明示endianのf8/c16、full保存値だけを受ける。

scope01/raw/core01は既存SOURCEのscopeとprovenanceを明示的に版上げしたcopy。scope04/core04も既存facadeを再利用し、必要なscope/version/独立prerequisiteとcost observerだけを接続した。旧人工tagを実入力へ偽装しない。全元SOURCE hashとnormalized AST比較は再利用監査に記録する。

| 層 | generator | checker | 共有・限界 |
|---|---|---|---|
| bit/PSD | 旧整数bit decode・entry cap・signed shift | 旧別bit decode・Hermitian pair cap・scalar/PF再構成 | 新adapter/NPY parser・schema/hash・Python/Fractionは共有 |
| U/W | bit prefix CAR・action norm² | occupied list parity・A²行列対角 | 同じ固定係数・determinant・既知恒等式 |
| K_SOS | 旧raw辞書/streaming/sqrt proposal | 旧別列挙/explicit B/sqrt不等式 | fixed K0・有理表現は共有 |
| Cbar2 | JW Pauli構築、離散mask交換子 | occupation行列・exact traceのbatch | norm式/contextは共有。JW積をcheckerへ流用しない |
| step | 旧η/Bの有理式 | 旧の正分母を払った不等式 | 同じ保証・d/α・domain。科学GOなし |

原NPZ→raw mappingは両経路の共有trusted前処理であり、独立に書いた二つのNPZ parserではない。file/member/array pinと人工bit比較で検査したが、parser/mapperの形式証明を得たとはしない。独立checkerも完全独立開発やplatform認定ではない。

## exact処理の高速化と計算量

旧generatorの`AB−BA`を各Pauli word pairで直接の離散交換子へ変え、可換pairをexact zeroとして先に落とす。mode masksは係数/分子に依存しない数学的cacheで、物理結果やground情報のcacheではない。相殺はgroup全体で保持し、norm前の集約を変えない。

旧checkerのtraceは各Pauli×全matrix非零entryを走査する。新版はoccupation行列の `(row XOR column)` ごとにまとめ、`(-1)^(row dot sign_mask)` の和をexact Walsh transformで一括評価する。同じ `2^-n Tr(PX)` で、float FFT/BLASを使わない。非零数z、異なるXOR数Fとすると、その部分は旧O(4^n z)からO(z+F n 2^n)の算術構造になる。出力そのものやbit演算の費用は別で、実H4 speedupを測定したという意味ではない。

H4 proposalの8 modesならfull Fock dimension 256、Pauli basis 65,536という構造であり、N=4 sectorだけの行列へ黙って縮めない。先にPauli/JW展開・grouped行列積が高価になる可能性もある。sparse fill-in、Pauli語数、有理数bit長とmemoryは実係数に依存し、今回未測定。SOURCEの規約上実行できる小規模候補だが、実行成功・安価な証拠取得の保証ではない。

32 CPU/affinity・利用可能RAMをread-only確認し、人工suiteは約4.37秒なのでpatch isolationを保つ1 runnerとした。CPU/RAM/worker/wall制限や共有設定変更はしていない。科学pipelineは証拠受理後に次段へ進むfail-closed逐次依存を保ち、重い部分に上記batch/bit高速化を適用する。process並列group backend、GPU、float置換、sector投影fallbackはこの版にはない。未知のruntimeを根拠なく短く見積もらない。

## 事前gateと一回実行の保存

`run_qc_a2_fixed_df_pilot.py` は明示execute flagがなければ文書・入力を読まない。flagがあっても契約採択/UNSET/別authorization/外部pins/SOURCE Git blobチェックが通らなければinput IO・claimを行わない。authorizationはこのSOURCEでは発行せず、後日の明示承認を受けた別文書を必要とする。schemaや文言hashはユーザー意図の暗号学的証明ではなく、独立に固定したpinと会話上の承認来歴の確認が別途必要。

契約はcanonical repository絶対pathとledger位置を固定する。同一契約hash directoryのexclusive mkdirが消費済みmarkerであり、run_idを変えても再利用できない。既存directoryや書込み途中のclaimを削除しない。recordはO_EXCL・0600、claim directoryは0700で、append-only保存とfsyncを使う。途中書込み・OS障害まで完全に回復するtransaction engineではなく、残ったclaim/partialを保全してSTOPする。

rawfile/input commitとworking bytes、全array manifest、mappingを照合してからcertificateへ接続する。各generator/独立checkerの出力を順に保存し、拒否された場合に後段を実行しない。SIGTERMはowned launcher内の例外、SIGINT/KeyboardInterruptも記録対象。SIGKILL・電源喪失・disk故障では保存保証の限界を明示し、未保存をPASSにしない。

stdout/stderrは採択したbyte上限で保存し、切り落としたbytesを記録する。certificateとlog各fileの上限超過は拒否/STOP、科学閾値を変更しない。集合bytes・各record SHA/最大Rat bit長、SOURCE/親ID、CPU/wall/累積RSSを保存する。timing observerはK_SOS/Cbar2の生成・検査を分ける。whole-runのfinalization前時間も保存し、最終log/result IOが未測定であることを明記する。古典費用以外の未知量は未評価のままである。

## Phase Aの検査境界

新36 testsはliteral/seeded人工NPZとin-memory store/人工auth vectorのみ。実snapshotへのrunner起動、物理的な実行済みmarker、科学authorization/grantは0件。productionは人工auth schemaを拒否し、filesystem操作はmockでexclusive/権限/fsync/上限を検査した。旧98 testsは旧SOURCEでそのまま実行。ローカルの人工検査であり、immutable CIや実入力独立再現ではない。

**実入力の初回correctness/Hermiticity/resource gateはPhase Bの一回分に含まれる。Phase A終了後mandatory STOP。**
