# 完成範囲と残る判断

**F1/F2は、既存lemmaと通常の代数/標準算術の合成で、条件付き技術仕様として閉じた。新理論研究としての追加投資は現段階では不要。** これはSOURCE実装・任意入力での実行成功・実CPU保証・物理native認証の完成を意味しない。必要十分精度やtight complexity、新規性を認定した結果でもない。

| 区分 | 完成したもの／残るもの | 次に進む場合の扱い |
|---|---|---|
| 技術仕様として完成 | Input/arm/helper domain、identity責任、Δ/hに対するP0〜P5/R、全valid parentの列挙不要な精度十分条件、canonical有限law条件、3 failure区分、mean/confidence前提 | [F1](preflight_precision_contract.md)を固定仕様として使える。具体的H/Kを選んだことや実入力で確認したことにはしない |
| 標準モデルの費用として完成 | Q/R/D/S、係数/normalized interval/π/Wのbit長、Full setup、local L²再sum、raw/coin/accepted、最悪trial、angle既存手順のJ/K_φ費用、generator memory、全Nの会計、canonicalの同条件 | [F2](full_bit_complexity_accounting.md)の計算モデル・入力変数・未実装層・外部費用を明示して引用する。粗い上界を実費/優位にしない |
| 通常実装が必要 | 入口validator、identity dispatch、与えたK/Hの事前述語/evidence、原因別failure保存、angle enclosure接続 | 数学的条件は仕様化したがcodeは未変更。実装/人工testの別許可後にCodexが担当。新selectorを作る許可と混同しない |
| 標準的追加解析が有用（必須の新理論でない） | 実装backendに即したQ/Rの鋭い評価、shared denominatorを活かしたtightening、出力/十進文字列化/保存policyのbit費用、具体的allocator/serialization制約 | 出力形式・backend・必要な精度を固定してから限定解析。実測は別許可。現条件付き上界を理由なく無効扱いしない |
| 非自明な新証明 | **必須と確認した命題は0件**。P条件は既存下界/丸めguardの通常整理、costは既存count＋標準算術 | 不明/未実装というだけで新命題を作らない。将来、仮定を越える量化/停止規則/精度domainを求める場合に限り命題を限定して別承認へ返す |
| 物理providerが必要 | actual CQ/native strict error、event依存T/CX/depth、angle合成、state preparation/測定実費 | `PHYSICAL_PROVIDER_UNCERTIFIED`。[B-HANDOFF]へ参照。exact query仮定を物理誤差0の測定結果としない |
| 別研究scope | Codex AのP_U、一般normalizer/Full-Partial比較、W2b/W2c、DF/PR/RTE binding/backend/accumulator、論文化判断 | `OUT_OF_SCOPE`。今回それらを再監査/証明/実装/採択しない |

## 未取得・未検証を区別する

必須W1資料、固定G6/G7 SOURCE/数学監査/G7契約/U、B入口・関連固定報告、共通運用指示を取得確認した。標準算術の一次資料・Python公式仕様は限定取得し、version/実読範囲/bytes/SHAをinventoryに記録した。必須資料の未取得による穴はない。

未検証なのは、今回作っていないpreflight/angle実装のcorrectness、実入力でP条件が成立するか、CPython実秒・実peak bytes、実出力/物理provider費用である。保存値を評価していないので、何bitなら実入力で十分か、何shots/何秒/何Tかという値はない。Codex Aの未公開結果も取得していない。

## 過大主張を避ける条件

「F1完成」は、**条件を満たす提供H/Kなら**全到達parentで既存precision guardを満たす仕様を指す。H/Kを返す自動手順やfixed defaultsの一様成功は主張しない。

「全bit費用を具体化」は、明示された標準モデルのgeneratorと条件付き追加層の評価を指す。物理oracle・出力/保存・全Nを無料にせず、未実装angleは別項である。1trialの多項式性を、log mのみ・精度記述長のみ・全N/実nativeを含む多項式時間へ拡張しない。

現行の物理target・confidence・精度値・P_U投資・研究採択・論文化について新しい判断はしていない。旧D＋E、Parent、G6〜G10/DF/Gaussian等のSTOP/CLOSED/HOLDを変更しない。**mandatory STOP**。

<!-- FIXED_REFERENCES -->

[B-HANDOFF]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4cf2a948ed348f842f3c338f6b46091124742430/docs/research/track_b_g6_e2e_interface_contract_audit_20261018/handoff.md
