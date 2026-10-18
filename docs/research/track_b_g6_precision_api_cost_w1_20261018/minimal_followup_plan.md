# W1後の最小技術作業案

**提案であり開始承認ではない。** W1の静的差分仕様は本資料で完了する。新理論研究の開始は推奨しない。selector実装を新しい研究成果と扱うことも提案しない。以下は不足の解消へ進む場合の最小単位であり、全部を必須の準備段階にするものではない。

| 順序/案 | 作業・固定入力 | 担当／技術的完了条件 | 必要な承認／STOP |
|---|---|---|---|
| F1 入口と事前精度の通常仕様 | D01〜D04/P01〜P03/L01〜L03/E03。既存[MATH]/[BIT]、K45–110/138–236、G103–143/216–251を使い、一般有理domain、arm差、identity責任、η/ρ/τ、固定K/H・全到達parent条件、3 failure区分を一つの呼出契約に結ぶ | **Codex**。既存lemmaとguardに全条件の根拠があり、固定defaultsを一様成功としない。必要な科学的選択（精度目的/confidenceを変えるか）は勝手に追加しない | **ユーザーが具体的な次の技術作業を明示指示**。新しいH/K具体式・selector実装へ進む場合は範囲を別途許可。既存lemmaだけで正当化できない条件が現れたら命題を限定しSTOP、新証明は始めない |
| F2 全bit費用の追加会計 | C01〜C03/K01〜K13。入力/係数/丸め後有理数・root整数/一時配列のbit費用を、採用する標準算術modelと既存countへ対応付ける。setup/最大trial/期待trial/全Nを分ける | **Codex**。既存の引用評価と追加解析を分離し、isqrt・sort・law再検査・Fraction/weight・angle・出力の費用が未記載のまま総費用と呼ばれない。CPU/bitとT/CXを別座標にする | **ユーザーによる追加の複雑性解析の明示指示**。新しい非自明定理・強い比較の証明が必要なら対象と不足だけを返してSTOP。実測・benchmark・samplingは許可に含めない |
| F3 必要な通常実装の接続 | F1で固定した範囲だけ。identity処理、全体入力validator、事前precision出力/根拠、既存angle enclosure手順との接続。既存K/G/CPを保全し新versionが必要 | **Codex**。数学存在と実装の対応が再現可能、失敗をzero化しない。どのSOURCEを変えるか、許可する人工testを先に限定する。ここでは実装しない | **ユーザーのSOURCE実装・人工testに関する別途明示承認**。sampling、実入力、回路、合成はさらにそれぞれ範囲を固定して承認。failure後のretryを自動採用しない |
| F4 物理provider（今は保留） | A02/G07。exact controlled-Qの条件を特定native providerで実証する作業は[B-HANDOFF]の別scope | **ユーザー**が必要性/対象を判断し、許可後のCodexが証拠取得。strict位相/actual adjoint/単価/誤差を取得 | 今回は`OUT_OF_SCOPE`。Bで未採択の物理target/route等をW1から決めない。未取得値を0へ置換しない |

F1/F2は既存保証の使用可能範囲と費用の説明を具体化する通常の技術作業であり、重要科学レビューを形式的な前提として自動開催しない。新たな科学判断が実際に必要になった場合だけ、**ユーザー**へ限定課題を返す。GPTの重要レビューは別の開始承認後とする。

## W2/C2へ渡す条件（未開始）

本W1の出力は、一般次数のFull/Partialに関する新比較命題の採択でも作成でもない。後続作業を別途指示する場合、少なくとも次を入力として渡す。

- 同じfinite M、p/x/m、σ、exact CQ、fresh-bit条件。armごとの実装domainとidentityの別経路。
- η/ρ/τ、root/probability精度の全law適合。未完成なら条件付きのまま比較し、片側だけ無視しない。
- ideal normalizer、U enclosure、digital moment/range、共通factor/bias・正の残余幅、既存confidence・ceil Nを分離。未知B_RA/Zを無料oracleにしない。
- setupと全trial古典費用、prequantum zero、accepted eventのword/angle/provider価格、期待値とhard cap。具体bit上界が未取得の成分は未取得と表示。
- 既知対照にも同じ情報・合法な最適化/streaming/cache・precisionを認める。canonicalのm≥3制約を考慮する。W1から低次数対照や一般比較の結論を作らない。

**次の担当はユーザー**：F1/F2の追加技術仕様・会計へ進めるか、現時点の仕様で保持するかを判断する。W2/C2、新証明、SOURCE変更/科学test、科学計算、レビュー、論文執筆にはそれぞれ別の明示指示/承認が必要。新実行契約・grant・marker・claimは作らない。

共通運用§8.4を維持し、必要な管理処理を効率化しても、共有設定・他者job・既存固定契約を変更しない。**mandatory STOP**。

<!-- FIXED_REFERENCES -->

[B-HANDOFF]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4cf2a948ed348f842f3c338f6b46091124742430/docs/research/track_b_g6_e2e_interface_contract_audit_20261018/handoff.md
[BIT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g6_finite_bit_and_access_audit_20261010.md
[MATH]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g6_independent_mathematical_audit_20261010.md
