# 同一固定点を保つstream bridgeの実装

基点817a1cd54da11ac20dc5a84477bf0f742a567214。[PLAN](PLAN.md)。[新module](../../../src/trottertracks/track_d_stream_bridge_fast.py)は、旧lowerer／1Q／CX／T0／MPFR160を変更せず再利用する。全P0 compilerや新しい量子回路規則は作らない。

## 分析と選択

旧処理は、初期native列をlistへ保持し、各roundで全列を1Q処理してfolded listを作り、CX処理のnode list／wire stackを作ってから次のword listへ移す。CXの各feedは中間価格をNativeCounterで再計数するが、その値をrewriteの判断にも最終価格にも使っていない。大半が不変の1Q runも毎roundで行列積と公式合成を再実行する。

今回の実装は、処理順と浮動小数点演算順を保ちやすい三点を優先した。

1. 1Q flushの出力をgeneratorで同じCX feedへ直結し、folded全listを作らない。初回入力もstreamのまま一度validation／T0計数し、元native全wordのlist化を避ける。
2. 1Q runの結果を、wire／opcode／parameterのfloat64全bitによる完全一致で再利用する。cache missは旧oneq_runsをそのまま一つのrunに適用する。独自Euler合成・近似key・角の閾値変更はない。
3. 旧CX feedの相殺判断をそのまま呼び、既に検査済みのnative列に対する中間価格の再計数だけを省く。最後にT0で全7資源を計数する。消費済みの内部list参照を解放し、callerの入力listや原本を変更しない。

worklistを先に採用しなかったのは、処理順が変わると1Q runの分割／結合とphase項の加算順も変わり得るためである。今回、全word再走査が不要になったとは主張しない。全roundを同じ順序で走査しながら、不変runの再合成と不要な中間処理を減らした。

## 同一性を維持する条件

1Qは旧実装と同じpending wireを持ち、非1Q命令の対象wireをtuple順でflushし、最後にwire昇順でflushする。旧folded列が完成してからCXへ渡す場合と、同じ列を順次渡す場合で、CX feedが見るprefixは同じである。

cache missは旧oneq_runsそのものを呼び、同じrun行列積順・Qiskit1.3 target-free ZSXX・strict-shorter判定を使う。cache hitは同一入力／wire／versionの不変結果だけを再生する。signed zeroも区別する。logicalな置換回数とphase項はhitでも同じ順に記録する。

cacheは直前passと現passのものだけを保持する。変更したrunを古い位置／object IDで同じと見なさず、bit keyで再照合する。不変runは入力objectをcacheに抱え続けず、現在のrunをそのまま返す。新しく合成したNativeInstructionはimmutableなので、同じ値の複数出現でobjectを共有してもgateの個数や順序は変わらない。

CXは旧両wire top／同方向判定と相殺を使い、RZの可換移動や新しいparity ruleを追加しない。初回入力のT0 validationと、旧1Q関数が返す検査済みNativeInstructionによって、中間Counterの省略を限定する。未検査入力を汎用CX APIへ無条件に受け入れる変更ではない。

各roundのsize／置換数／CX相殺数、最終native word hash、旧binary64 phase、MPFR160 phase、全7資源を旧保存結果と比較する。位相差をfitせず、価格が一致するだけの別回路へ置換しない。相対χ／signed coefficient／actual adjointは固定native列に既に含まれ、その命令を変更しない。

## 記憶と残る走査

frontierの固定上限は設けず、tombstoneも同じpass内では保持する。合法な将来相殺の候補を切り捨てない。各pass完了後にCX containerを解放する。cacheの保持期間はsemanticな二passであり、恣意的なRAM／entry数の小上限ではない。

O(L)のword／node／cache保持と、最悪O(RL)=O(L²)の反復走査は残る。cacheの異なるkeyが多い場合は追加記憶を使い、常に一定量のfrontierで済む保証はない。scope内の実測効果は[PERFORMANCE](PERFORMANCE.md)に分ける。

## Lowering接続

iter_raw_nativeは既存NativeLowererを1 gateずつ呼び、有限gate chunkをyieldする。全native列を一度listにする入口を省ける。固定q4では旧保存native列と完全hash一致、q8では固定raw hash・旧phaseに一致することを確認する。

最大chunkは今回のStrongで44命令、Referenceで53命令。ただし保存raw QuantumCircuitと、そのgate.definition等のcacheは残る。chunkが小さいことを全pipelineのO(1)記憶やQC不要と混同しない。未展開descriptorからfull-wrapper QCなしに全PRをemitする一般driverは今回追加していない。

## 計測adapterの修正

最初のbenchmarkは、旧baseline関数の引数phaseを共通呼出し側がglobal_phaseというkeywordで渡したため、旧側optimizerに入る前に停止した。改善側8件はPASSし保全した。旧runnerと失敗を変更せず、[互換adapter](../../../scripts/complete_track_d_stream_bridge_benchmark.py)が同じ値を位置引数で渡すだけの修正を行い、未実行の旧8件だけを完了した。

このため旧新の測定waveは分かれる。source・入力・観測器は同じだが、厳密な同時paired実験とは称さない。修正後のkeyword回帰1件も含め、人工・技術回帰は61件PASS。旧価格・正しい結果を合わせるrule変更や閾値緩和は0。

この条件差を残さないため、同じ8 wrapper／同じ旧新関数を16 fresh processの共通poolで一度だけ再測定した。主性能比較はrun02_pairedを使い、初回の別wave結果／失敗も保持する。新fixture・新規P0・科学条件変更は0。

## Phase-producing streamへの接続

[stream resource adapter](../../../src/trottertracks/track_d_stream_resources.py)は、producerの終了後にnative phaseが確定する場合にも対応する。rewriteは全体global phaseを判断に使わないため、入力終了後に旧binary64加算順を再構成し、全logical roundのphase_additionsからMPFR160 ledgerを再構成する。cache missの合成callだけをphase項として数えてはならない。

この接続は小人工2件で旧関数と比較した。初回のテスト期待値が「入力phase項のみ」と「補正項も含む最終ledger」のterms数を混同して失敗したため、旧optimize_with_ledgerの結果をoracleとする期待値へ修正した。adapter本体や最適化規則の修正は不要だった。初回logを保全し、最終の人工・技術回帰は合計63件PASS。新adapter単体の実H4 end-to-end再計測は行わず、主benchmarkのSOURCE／結果を変更していない。
