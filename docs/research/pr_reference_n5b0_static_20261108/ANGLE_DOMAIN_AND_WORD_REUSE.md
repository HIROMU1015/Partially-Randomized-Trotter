# N5-B0：m17角度域と保存word再利用

2026-11-08 JST。[REPORT](REPORT.md)。Actual signed literalと保存条件は[M17_INPUTS](../../../artifacts/pr_reference_n5b0_static/2026-11-08/M17_INPUTS.json)へ引用した。新しい角度表、到達集合、wordを生成していない。

## 1. 域の状態

| 対象 | 保存値／literal | 現policyとの関係 |
|---|---|---|
| 固定辞書 | L=82.49247654978845、b36、j=-36…6、43 basic、final10^-11 | N3-B0がm7の保存包絡を元に固定した仕様 |
| Reference m17 vacuum | 保存包絡32751.887444214277 rad、symbolic:vacuum_phase[m=17] | L内を証明できていない。包絡超過は実到達域外の証明ではない |
| B′ m17 scalar | **-0x1.86e032ee41d40p+12** | 保存post-compiler literalは域外。q依存の固定scalarで、random到達性で除外する項ではない |
| B′ event/r32 | 3rまでの逐次Pi0加算から得た保存96-key安全superset | うち70 keyが固定域外というN5-A metadata分類。各candidateのcanonical到達性は別問題 |
| B′ central/r32 | ±0x1.7e4088e024c5fp-12、±0x1.fdab627b8d37ap-14 | 域内。R1のhalf-angle keyとは異なる |
| Reference central/r32 | ±0x1.7e4088e024c5fp-11、±0x1.fdab627b8d37ap-13 | 域内。対応するfull-angle keyで、r1の価格を流用しない |
| 固定basis frame／D等 | 既存の有限SOURCE／inverse template、C0とN4-B frame certificate | 域内・literal一致でも、m17の十分capへのguard＋emitter適合は別問題 |

Vacuum包絡は、固定SOURCEのF/G phase逐次減算、finite event／basis/run alphabet、IEEE arithmeticを旧CT-0が囲った結果。新しい生成方式や平均角に置換していない。

現providerはdomain判定をC0 lookupより先に行う。従って必要literalが域外なら、別の旧poolにwordが存在する可能性だけでは現policyが実行可能にならない。SOURCEのangleをmod縮約したり、理想πへsnapして拒否を回避することもしない。

## 2. 再利用と不足の分類

| 分類 | 既存資産 | 次に必要な証拠 |
|---|---|---|
| 保存word再利用候補 | Literalが一致するC0299・basic43・role85 | 同じword意味、actual adjoint、目標精度、emitter、C0優先の適合 |
| 同じwordのguard資格不足 | 保存logical guard上端が十分capを超える、又はemitter込みで入らない | 上端超過だけでは実誤差の下限にならない。必要なら同じwordの保証補強を別途承認 |
| Current domain外 | B′ scalar、event supersetの一部、vacuumの未被覆包絡 | 域／policyの保証拡張又は必要target域のより強い証拠。まだ方式未選択 |
| 新しいbasic取得の可能性 | 旧43 wordでは保証できないprecision／指数範囲を選ぶ場合 | 新しい有限key集合・precision・guard／停止条件を先に固定する。今回取得なし |
| SOURCE／compiler context不足 | 保存templateと異なる経路を将来使う場合 | 条件C／新経路との対応。今回は新compileをしていない |

固定43 wordの取得とonline有限停止の証明は再利用できるが、現提供域は変えない。域を広げる方法として同じwordを繰り返す／新指数を加える／別表現を使う等は、いずれも現policyの資格外。どれを選ぶかも、基本wordが追加で必要かも今回確定しない。

より厳しいprecisionを将来採択する場合、保存basicの実guardが一部適合しても、C0優先、量子化、全chunk和、emitter、価格anchorとの整合が必要。Guardを再評価すれば必ず改善する、あるいは再合成すれば必ず十分になるとは扱わない。

## 3. Literal閉包とphase保証

条件Cは固定Qiskit1.3.0／同値則／post-unroll template、actual SOURCE inverse、RZに作用しない最終CX/X取消し等を含む。同じthetaに両armで同じpositive word、負側はそのcomplete actual adjointを使う。SOURCE逆方向をforward native列の単純inverseへ置換しない。

N5-B0のphase因子分離候補は**compiler後のjoint word具象化**についてで、SOURCE scalar／event／vacuumのliteralを変える許可ではない。Pure W具象化がcommon scalarであることと、RZ(theta)のthetaを変えてよいことは全く別である。

統括兼Dの13-round adapter／別native経路に今回のCを仮定しない。必要になった場合に公開SOURCEを参照して対応を確認し、今回重複実装・検証を行わない。

## 4. 新規合成に関する結論

現証拠でm17を実行可能と認定できず、追加の保証又はpolicy対応が必要。ただし**新規Clifford+T合成が必須という結論は出ていない**。同じwordのSOURCE-based評価改善、既存record再利用、域の保証の補強が先に検討対象となる。

新規合成が必要と判断するには、選択した目標保証／域／policyの下で、保存wordが不足する具体的な有限key・precisionを特定する必要がある。現在はその新仕様を採択していない。M17の新RPE成功判定・T価格比較は未実行、mandatory STOP。
