# M案K0：GPT科学判断用引継ぎ

2026-10-15 JST。ユーザー指定のA→Bを完了した。Aは[短い既存成果引継ぎ](qc_onebody_he1_archival_handoff_20261015.md)、Bは[限定技術監査](qc_m_fixed_circuit_k0_technical_audit_20261015.md)。[完了GPTレビュー](../reviews/quantum_chemistry_pf/qc_post_he1_research_direction_scientific_review_20261015.md)を42,646 bytes／SHA256 `326622956c9eca9d28fc8cc9d500ad3cf1d45e7b2809ab06eea23aac1c750fca` の原文で保存した。新しいGPT科学レビューは開始していない。

**M-aのcounterterm／射影生成coreは `REDUCES_TO_KNOWN`。M案固有の生成上の優位・CS2後の費用利益は未特定で、全体は `UNDETERMINED_NOT_READY`、K0 `gate_pass=false`。** K1、pilot準備・数値検証・M-b探索に進まず停止した。

| 科学判断へ渡す問い | 技術監査の回答 |
| --- | --- |
| h²係数修正自体は新しいか | modified potential／modified operator splittingに具体的対応がある。固定slot角度変更とBCH/HS係数fitも既知。Carenaのspatial lattice改善とは誤差源が異なり完全同一としない |
| 固定回路で何を実装できるか | G1は固定frameのdensity pair-phase、G2はrank-one平方の接方向。sourceはゼロ角を省略するため、任意Wにgate数不変を与えられない。lower-level RZ/RZZ表現は存在するがM adapter・productionは未実装 |
| 数学は成立するか | `V=D+ΣW_l`、`T=E+D'[W]+ΣR_l` の形式展開は整合。etaをh²で変更して平方構造を保つ場合のh⁴Rを落とせない。V=0ならoperator4次、ground期待値の相殺だけでは同じ保証にならない |
| 指数sectorの射影を計算せずに定義できるか | 固定body-rankではCARと一様sector traceの組合せ式でGram・右辺を生成し、標準線形問題へ落とせる。実コピー取得不要のtraceだが、ground RDMではない。粗い多項式上界は実装速度・bit保証ではない |
| CS2より良くなる機構はあるか | 非scalar Wはoff-diagonal couplingとh⁴Tを変え得る。しかしHS誤差減少から補正後残差・overlapの改善は従わない。Wがscalarだけなら元CS2と同じ。現M-aに改善保証はない |
| 強い対照に対する独立差は残ったか | 同じ合法slot・BCH次数・sector trace・情報・価格を許す既知counterterm/係数fitで同じ射影coreを再現できる。DF固有の計算量・生成規則・保証改善は特定できなかった。Mansurogluの公開degree2目的関数とS2三次目的を完全同一とはしていない |
| 数値pilotを始める技術根拠はあるか | 現案はK0進行条件を満たさない。未計算であることや特定手順の未掲載を継続根拠にしない。数値救済、別候補、自動追加文献探索は起動しない |

次にGPTが判断するのは、既知原理の特殊化と残る未確定点を踏まえて、M案への独立研究投資を見送るか、別に与えられる具体的な差があれば再検討するかである。**Codexは正式採択・終了や別研究への移行を決定していない。** 重要な科学レビューには別の開始承認、新しい科学計算には別の固定scope・実行承認が必要。

第一候補は終了推奨を保存するが、正式状態は `AWAITING_USER_ADOPTION_DECISION`。H-E1は現定義を独立算法研究として追究しないという受領レビュー推奨を記録し、未実施K1〜K3や第1/第2研究への移管を起動しない。一般的不能性は主張しない。

[固定資料identity](../../artifacts/qc_m_fixed_circuit_k0/2026-10-15/reference_identity.json)、[一次文献6本文・確認式](../../artifacts/qc_m_fixed_circuit_k0/2026-10-15/literature_registry.json)、[scope](../../artifacts/qc_m_fixed_circuit_k0/2026-10-15/scope.json)、[技術status](../../artifacts/qc_m_fixed_circuit_k0/2026-10-15/method_status.json)。新科学計算・数値射影・保存数値再集計・compile・pilot・production・GPTレビューは0。旧source/result/認可/STOP・manifest FAIL、N1 STOP/N2 HOLD/N3 CLOSED/THRIFT現template CLOSED、Track A/B/C1〜C4を保全する。

独立branch `qc-m-fixed-circuit-k0-audit-20261015` でcommit・push・GitHub再取得照合後 **Mandatory STOP**。同様の候補探索を繰り返さず、科学判断の承認境界へ戻す。
