# 既存emitterからnative streamへの接続

[結果と条件](REPORT.md)。実装：[track_d_native_emitter.py](../../../src/trottertracks/track_d_native_emitter.py)。旧SOURCEの回路構築規則を変更せず、新しいsinkとbounded chunk interfaceを追加した。

## 接続経路

| 既存資産 | 再利用方法／確認範囲 |
|---|---|
| C3 `mixed_head`／head_recipe | 元functionのcode・defaults・closureをそのまま使い、コピーしたglobal名前空間内のQuantumCircuit constructorだけをNativeSinkへbind。first／boundary／lastを2-mode人工contextで確認 |
| StrongBuilder relative transition／directional中心 | 既存mixed_headから元builder／Gaussian gateを呼ぶ。同basis共有・half-sweep境界融合・scalarの構築順を変更しない。C1 head-tail追加融合は導入しない |
| `build_event_sequence` tail | ローカルimport型constructorのため小さな既存QCを `circuit_chunk` で逐次lowering。元event／sign／basis-run共有を保持。新sampler・tail builderなし |
| physical A/B・actual inverse | 人工検証のbounded segmentをStreamBlockへ。逆は既存命令列を逆順に辿り、各gateのactual inverseと負のdeclared phaseを使う。max-plus matrixを逆行列にしない |
| `reference_gadget` | 既存の二cascade／A†／B／χ補正をconstructor bindして実行。共通ancillaを保持、二registerのwire mappingはinjectiveに検査 |
| `hadamard_wrapper` | H・X/YのSdg・measureを既存順でemit。古典bitもT0へ渡す |

`legacy_to_stream`はmoduleのglobalやファイルを変更しない。Python functionがglobal QuantumCircuitを参照し、整数registerでsinkを一つ生成して返す範囲に限定する。ローカルimportや複数中間QCの一般builderには適用せず、明示的に拒否する。この小さい入口を設けた理由は、既存SOURCEにsink protocolがなく、全builderの複製／書換えを避けるためである。

`NativeSink`自体に `.data` はなく、命令ごとにconsumerを呼ぶ。T0なら `consumer=counter.feed`、既存CX bridgeなら `consumer=bridge.feed` に接続できる。StreamBlockはforwardと明示actual-adjoint callbackを持ち、replay可能な記述だけを保持する。固定request／eventを逆順に読む全H4 driverは今回実装・実行していない。

## Native loweringと位相

標準rz/sx/x/cx/measureだけをそのままT0へ渡す。H／P／Sdg／RZZ／CRZ／controlled-RZZ／XXPlusYY／CSWAP等はQiskit1.3の標準gate definitionを再帰展開し、definitionのglobal phaseも加える。fundamental Uの定義がない場合だけ、公式 `OneQubitEulerDecomposer('ZSXX')`を2×2入力・simplify=Falseで使う。多qubit dense合成や新しいGaussian合成器はない。未知custom gate／symbolic／条件付き／directiveは拒否する。

価格だけでなく命令順・signed parameter・control/target・wire・phaseを保持する。global phaseは浮動小数点の宣言値で、modulo丸めやphase fitを行わない。Referenceテストのχは元人工モデルのscalarとevent角から解析的に定め、ABのvacuum rayを1e−10で確認してからX/Y wrapperの位相付き作用を比較した。

## QCとdense targetを回避した範囲

Strongのhead→tail→boundary→tail→lastをcounterへ送る経路は、全wrapper QC／C3 `prepare_steps`／全CAR targetを生成しない。Referenceもshellとreadoutの全wrapper QCを省ける。ただし既存tail／Gaussian／physical segmentの小さいQCは残る。今回の人工fixture準備は既存toy.contextの小dense oracleを使い、correctness比較も最大5qubitのQC／Operatorを作った。これらを本番stream経路の記憶量へ混ぜたり、「全処理でQC／dense行列ゼロ」と表現したりしない。

保存54case試験は原本の小raw QCを読み、loweringとbridgeを検証する別経路である。新H4 emitterを動かした証拠ではない。原本は公開済みhashで固定し、欠落を再生成で補っていない。

## 実装境界

Qiskit1.3.0を検査し、別版のprivate APIを黙って使用しない。`.control()`やgate.definition生成が行う局所定義展開は再利用するが、full P0とは区別する。人工suiteと保存54runnerは `qiskit.transpile`／`qiskit.compiler.transpile` を例外に置き換えて完走した。1Q試験では最適化passの `.run` も禁止し、2×2合成だけを呼んだ。

既存frame／gate definition cacheは元providerのものを使う。native word全体を新cacheへ保存する機構は追加しない。replay時の位相・wire検査を省く新しい高速経路や、旧科学SOURCEへの差込みは行っていない。
