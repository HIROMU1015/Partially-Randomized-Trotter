# PO・F1 gateへのSOURCE対応

全行の科学statusは **OPEN_NOT_EXECUTED**。SOURCE／testの存在はgate PASSの証拠ではない。

| 義務／gate | SOURCE | 定義済みtestの対象 |
|---|---|---|
| PO-01 / H-ID | algebraのexact S/P、prototype collect、reference original_operators、execution H_ID | exact JW、四armの元H再構成、微小非零保持 |
| PO-02 / typed metadata | MixedAtom/MixedTailView、RTEComponent bridge、LAW metadata検査 | arbitrary PauliをDF supportへ偽装しない、serializer、zero認証 |
| PO-03 / LAW・FINITE-MEAN | 既存law/config/event、gates component_derived_mean、独立P3 | actual control1 blockからのmean、90/128参照一致 |
| PO-04 / CONTROLLED・SIGNAL | circuits central/router、actual inverse、Hadamard adapter、gates | 8motif×4arm、n0/n2、負係数、I/c、order、adjoint、X/Y |
| PO-05 / PRICE | literal_rz_count、direct scan、EMPTY/SINGLE/MULTI DP、selector | 小catalog直接列挙vsDP、unknown price拒否、tie/fallback |
| PO-06 / finite bit | exact fraction、make_tail変換loss拒否、LAW residual、interval provider、ceil | exact zero、ceil境界、二精度reference、固定compiled operator gate |
| PO-07 / statistical policy | intervals score_enclosureの共通N式、qualification record要求 | interval/ceil拒否、未資格fallback。IID quantum shotの実行なし |
| PO-08 / fairness | four-arm canonicalizer、共通BasisProvider／sequence policy／compiler、二端点selectorと既知全列挙 | 4arm共通検査、同情報cache、conservative tie |
| PO-09 / full register・sector | CAR/full16、controlled32、H-sector診断、中間projectionなし | 元Hのfull比較、Pauli support、actual controlled block |

P0のH、finite mean、controlled operator、Hadamard signal、literal priceは別handlerとreportを持つ。compiled gateは5e-10、未compile controlは1e-10、mean/signalは2e-10、Hは1e-11をSOURCEに保持し、rtol0／arbitrary phase fittingなし。90/128digitsの一致を厳密certificateと呼ばない。

F1候補はT=.2/q2/delta.1/r2/K2、F2候補は同fixtureのT=.2/q1/delta.2/r2/K0で変更していない。F2実行runner、trajectory費用評価、shot抽選は今回の実装対象にしない。SOURCEにF2 contextを表現できることは実行権限ではない。

科学callの最終case IDs、metadataの全coverage、literal provider、env/precision資格はB後にbindingする。test定義をF1登録callへ重ねて余分に実行せず、最終契約で対応と回数を確定する。一般証明、独立新規性、performance GOは全て今回の到達点に含めない。
