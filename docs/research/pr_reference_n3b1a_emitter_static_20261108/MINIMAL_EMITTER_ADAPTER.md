# N3-B1a：既存emitter再利用と必要最小限の差分

2026-11-08 JST。[REPORT](REPORT.md)、[記号的誤差条件](PHASE_ERROR_BOUND.md)。**設計のみ。新adapter／emitter／policyを実装していない。**

## 1. 最初の選択肢は既存lower_nativeの維持

C_phiと有限phase経路の証明を採用できるなら、Wを1文字ずつ加える現emitterにもSOURCE由来の上界を与えられる。W数が未取得という理由だけで新emitterが必須とは結論しない。

完成した正角のcomposite written wordを既存record interfaceへ渡せば、lower_nativeは従来どおり全文字列を逆順にemitし、負角にはそのlocal circuit全体のinverseを使う。43 chunkをemitするたびに別の負角合成を呼ぶ必要はない。

将来必要になるのは、基本wordを固定順へ連結し、N3-B0の量子化・基本guard・emitter証明を結び付ける**composite record／providerの限定実装**である。これは既存emitterのgate意味論を書き直すこととは分ける。現validate_recordのpygridsynth spec／guard_methodを未評価のcompositeへ偽装せず、証明経路を識別する最小限のprovenanceを持たせる。旧recordは変更しない。

## 2. Integer W ledgerを使う場合の最小案

SOURCE条件の確認を容易にするため明示的ledgerを選ぶなら、既存処理のW部分だけを次へ置き換える設計を検討できる。

| 処理 | 最小差分 |
|---|---|
| 入力word | 保存sequenceをそのまま読む。旧word／hashを上書きしない |
| 非W文字 | 既存のwritten product逆順とnamed H/S/T/Tdag/X emissionを維持 |
| W | 整数Kを数え、整数でk=K mod8へする。float(K)へしない |
| 具象化 | local phaseをkPへ1回設定するか、既存のP加算を高々7回使う。kPの表現可能性を証明する |
| 負角 | positive連結word全体を完成してからactual inverse。chunkの順序も反転 |
| jointへの追加 | 現out.compose経路を維持し、そのscalar増分を誤差bookkeepingへ入れる |

Wはscalarなので整数指数の集約はexact operatorを保つ。C_phiの8状態証明の下では、このper-word ledgerは現emitterと同じlocal binary phaseになる。単にWを削除する方法ではない。内部counter化は旧C0のword選択／T数を変更せず、input sequence／hashも保持できる。ただし将来の新SOURCEの適合確認を省略する根拠にはしない。

全jointのWをまとめてから位相を最後に設定する広い書換えは、今回の最小案ではない。compose順序やserializationを変える場合は、別にSOURCE境界と誤差を資格する必要がある。

## 3. Exact carrierとbare Qiskit circuitの区別

integer Wとnamed gate列を組にして`W^k * circuit`と解釈すれば、数学的なexact representationを作れる。しかし、Wを落としたbare QuantumCircuitだけが同じstrict operatorになるわけではない。sidecarを使うならconsumer、compose、inverse、保存、読込がすべてその整数位相を保持する必要がある。

この組表現での誤差0を、Qiskit float global_phaseへ変換した後のe_emit=0と混同しない。symbolic ParameterExpressionへ置くだけでもQPYを含む全経路は自動資格されない。[SOURCE監査](EMITTER_SOURCE_AUDIT.md)のとおり、最初の既定手段として新symbolic engine／serialization frameworkを追加する必要性は認めていない。

## 4. 取得後の最小correctness検証

実装を別途許可された場合は、既存exact_word_matrix、strict_guard、CT-1のactual adjoint／emitter照合、CT-4Bの局所correctness資産を再利用する。

- 実43 recordのsequence／W／guard／hashとfixed keyを対応させる。内側要求は10^-13のまま。
- 正負のcomplete word、W residue／wrap、zero、選択bit順、全体adjointの対応を確認する。
- 新providerを使ってもC0の採用sequence／価格を置換しない。
- Qiskitへの具象化経路、compose回数、inverse回数、必要QPY往復を固定し、C_phiのoperation bookkeepingへ対応させる。
- exact wordの論理行列・scalar phaseの保証と、浮動Operatorの診断を分ける。数値toleranceだけを厳密残余のcertificateとしない。

実basic word取得前に、既存wordを使ったSOURCE適合・phase定数・保存形式の有限検査を先に行う余地がある。これも今回実施していない。新しいgate gadget、PF、F/G、sampling、native compileを追加する理由にはしない。

## 5. 変更の採否

既存emitterのC_phiと固定経路を満たせる場合は、emitter自体の維持を優先する。C_phiが成立しない／consumerがexact phaseを失う／必要経路が証明範囲外なら、MINIMAL_ADAPTER_REQUIRED又はUNQUALIFIEDとして戻す。integer ledgerの実装を自動開始しない。

この設計は旧SOURCE／回路／価格を変更した結果ではない。旧結果・STOPを保全し、後続の具体的SOURCEとcall範囲への承認まで停止する。
