# 費用座標・固定証拠・取得不足の対応

[入口](README.md) / [契約候補](execution_contract_candidate.md)。既存値の参照/引用と新計算を区別する。新しい順位・scalar weightingは作成していない。

| 変数・座標 | 現在の分類 | 固定証拠/不足 | 将来どの判断を変えるか |
|---|---|---|---|
| N_a、元moment/range/ceil | **既存証拠で判定可能** | G7 P5の保存budgetを引用、W2b条件を継承。再計算0 | CL/Qの全trial倍率。これだけで費用順位は決まらない |
| z_a、既存label CQ係数、RZ切片 | **既存証拠で判定可能**（旧粒度） | 保存small-support reference/affine total。再列挙不要 | 旧literal期待Tの限定結論をそのまま使える |
| accepted論理実行、total CQ/helper | **既存証拠で判定可能**（conditional IR） | W2c-A Ordinaryとの構造非劣、P5保存結論 | native費用へ無条件変換しない |
| S_a：setup/preflight/guard/load | **新規測定によって判定可能**＋実driver未接続 | F2上界/補助adapterあり。実G7では未実行 | CLのΔ_aのsetup差。共有回数も同時固定 |
| μ_a：all-trial CPU | **科学条件が不足**（estimand/statistics） | 固定全N観測なら記述的、母期待値なら新confidence設計が必要 | CL1/CL2の実測又は期待費用を区別して判定 |
| raw/coin/accepted prefix | **新規測定によって判定可能** | 旧経路/sourceと新StageLedger。coinはpacket/law/sort込み | Full追加処理がtrial削減を超えるかの原因内訳 |
| F3相当canonical監査 | **補助実装済み・実入力接続未検証** | finite root/law witness＋共通overlay。group意味は固定SOURCE依存 | Fullだけ監査/比較側0という不公平を避ける。追加費用は両側測る |
| angle τ/K_phi/Jと取得schedule | **科学条件が不足** | 既存F3 APIはあるがこの比較の具体allocation未採択 | μ_aのangle/requery差、同一G7物理error互換性 |
| Ω_a：output/cache/flush | **新規測定によって判定可能**＋schedule未採択 | 共通retention/cache政策を選ぶ必要 | CLの保存差。streamingを無料扱いしない |
| 正確bit operation | **既存上界のみ・実数未取得** | F2 Q/R/D/Sのsymbolic上界。CPU instrumentationはbit演算数ではない | 上界比較と実費比較を別に保つ。新tight保証を作らない |
| peak memory | **新規測定によって判定可能**（観測値） | per-process RSS/allocator/buffer/cacheの定義が必要 | N倍領域と誤認せずPareto座標で比較 |
| word/child/control/direction価格 | **物理入力・実装不足** | Q/encoding/physical provider未固定。source interface/price schemaのみ準備 | Q1/AFF。役割別価格なら既存係数粒度不足も確認 |
| RZ synthesis | **既存証拠を再利用可能**（有限keyのみ） | G7 cache/G8履歴。新合成なし。metadata guardは独立誤差再証明ではない | 同一key/ε/compilerなら既存価格を再利用 |
| T/CX/1Q総native | **物理入力・科学条件不足** | CQ/native誤差0の仮定と実providerの互換性、価格poolが必要 | Q条件に接続。RZだけの優位を総nativeへ格上げしない |
| serial depth/whole-circuit depth | **比較不能・未取得** | primitive加算とwhole-circuit scheduleは別。word/angle相関の粒度が必要 | 個別C1/B2追加承認後のみ判断可能 |
| ancilla/workspace | **一部既存・一部未取得** | outer/helper2は既知。provider/state追加領域は未知 | peak領域をN z倍しない |
| prep/readout/backend/failure | **C2実装・費用不足** | 未知はnull。旧実measurement0、全N accumulatorなし | 現B3/C1でEnd-to-End総優位を判定しない |

CLの接続は、同じ層/単位で

\[
(N_a-N_F)\mu_a\ge N_F(\mu_F-\mu_a)+\Delta_a,
\quad\Delta_a=S_F-S_a+\Omega_F-\Omega_a.
\]

観測全N workloadを選ぶ場合は、実際に足した各trial費用を使う。**固定bitstream観測平均を母期待値μのcertificateと呼ばない**。実測CPUは環境依存であり、比較対象の認証下界を持たないF2 upper-vs-upperで置き換えない。

量子接続は、全trial digital πを分母とし

\[
N_F\sum_e\pi_{F,e}c_{F,d}(e)\le
N_a\sum_e\pi_{a,e}c_{a,d}(e).
\]

現在保存された係数で表せる共通literal価格なら既存B1/B2を再利用する。別のcontrol/direction価格、depth、whole-event最適化で情報粒度が足りない場合、**追加B2縮約**を独立承認に含める必要がある。新cache価格を取得しただけでこの和の全情報が自動的に揃うわけではない。

strict/nonworse/tie/未解像はW2c-A分類を継承する。未知価格/区間重複をtie/0にしない。各座標を別vectorで扱い、価格換算・scientific GO・最適性を自動判定しない。**mandatory STOP**。
