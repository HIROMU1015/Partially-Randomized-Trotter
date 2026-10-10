# G10：入力・ensemble・費用・保証に基づくclaim監査

採用GPT review §13-A。研究採否・独立新規性・投稿十分性はGPT判断。
旧[G6 claim監査](g6_prior_art_and_method_delta_20261010.md)と数学監査の範囲を保持し、
既知component、G9の実装証拠、G10で残る比較を具体化する。

## 一次本文の今回の確認

- [Aomoto–Kato一次PDF](https://www.numdam.org/item/10.5802/aif.1123.pdf)：pp.62–65、§1 Eqs.(1.6)–(1.14)、Lemma 1.1の解析text。
  別旧URLの取得は失敗したがこのURLで取得した。全スペクトル定理の監査ではない。
- [Zhao–Yuan arXiv:2103.07988](https://arxiv.org/pdf/2103.07988)：§4.2 Eqs.(24)–(29)の関連全文を今回確認。
- [Wan–Berta–Campbell arXiv:2110.12071](https://arxiv.org/pdf/2110.12071)：Appendix C Algorithm 2、Appendix E.2のfinite truncation記載を確認。
- [CTS出版本文](https://www.nature.com/articles/s41534-025-01168-w)：Theorem 1 Eqs.(5)–(7)、finite truncation/normalizationとnative比較の範囲。
  層sampling/補足Note3/5の詳細は既存G6監査を参照。今回は補足全頁を再精読していない。
- [Resource-optimal IS v1](https://arxiv.org/html/2603.13495v1)：Theorem1 Eqs.(9)–(13)、§IV.1/2のZeroFill/Discardを確認。
- [元PR v2](https://arxiv.org/pdf/2503.05647v2)：Appendix A.2 Eqs.(A18)–(A26)のRTE/normalizationを確認。

版・該当節の主張に限定し、引用ネットワーク全体の網羅的priority監査とはしない。

## Claim単位の比較

| Claim | 既知の入力/出力/費用・保証 | Track B側の限定差候補と証拠 | 非claim・未解決条件 |
|---|---|---|---|
| Green係数query | 自由積のGreen倍率、returnによる他因子のspectral shift | Z2 involution確率への特殊化。G6に代数的対応あり | 母関数・再帰を新しいrandom-walk定理としない。finite samplerとの接続の非自明性は未確定 |
| ordinary finite RTE | order+IID labelの非列挙生成、隣接even/oddの回転pair、phase、finite truncation | 同じ有限P_mをtargetにfull形式returnを先にcollectし、parent依存角度を使う | order/IIDやEuler identity自体は既知。ordinaryへ全word tableを要求して古典勝利を作らない |
| return/identity吸収 | modified Taylorはanti-commutation、低次単位ary/identityへの高次寄与吸収とLCU誤差を扱う | 現familyはQ_i²=Iだけで、順序付きfree-wordの全returnをcollect。source proofが同じ有限P_mを保つ | 吸収原理/P3/P5計算だけのpriorityは未確定。高次tail近似やLCUと対象・access・誤差を揃える必要 |
| literal CTS | Pauli情報、real correction + common-angle odd rotations、finite CTS norm/step会計 | G9は同じfull P5に特殊化しcheap I1も公開、direct native/相対位相/十分shotsを共通化 | CTS全般への勝利なし。identity吸収/別group/層CTS/精度最適化は本固定辞書の外 |
| cost-aware IS | fixed protocolでcost×second moment下界と最適proposalが既知 | G10-Aは保存digital辞書/同Bernstein規則で任意proposal下界を独立確認。partial/P3にも同じ原理 | 下界は実行可能lawや達成optimalityでない。固定precision/辞書外へ移さない |
| rejection/zero-fill | success/weight補正とdiscard normalizationは既知 | full localはordinary envelope/Uでbudgetし、unknown global B_newを要求しない | zero-fill自体の新規性なし。accepted-call期待値とtail/hard attempts、古典coin費用を区別 |
| P5特殊化 | 有限次数の記号代数・DP・group selectorは既知計算手段 | 同ideal familyの低次数fast path、group/root O(L²)算術。G9のnative証拠あり | sorting/bit/root/selector費用を含む全初期化/一試行O(L)とは言わない。full localとの差はgenerator/budget役割分担 |
| general degreeの追加価値 | 低次吸収＋tailは強い対照 | G10はm7のclosed P5+6/7 ordinary pairとfull returnを同provider/targetで比較する | 準備時点で未取得。m7での勝利から全次数/DF/PR-QPE総costを主張しない |
| 最終PR task | 元PRはdeterministic/random split、normalization、QPE信号/総費用を扱う | Track Bはfinite single-block operator ensemble・native資源のmethod候補 | deterministic側、Taylor remainder、複数step、state preparationの全会計は未接続 |

この表の「差候補」は一次文献とローカルsourceを照合した推論であり、独立新規性の確定ではない。

## Claim/evidenceの現在値

| 項目 | 証拠状態 |
|---|---|
| 有限奇数次数の集約恒等式・非負性・局所query | G6限定数学証拠。既知Green式との対応を保持 |
| P5/direct native利益 | G9登録source-bound local evidence。cheap I1 synthetic context、fixed policy/compiler |
| 固定literal CTSおよびordinary/partial/P3の任意proposal分離 | G10-Aの保存有理値post-hoc独立下界。固定P5価格/policyに限定 |
| closed P5+tail / general P7 / matched P7 CTS資源 | 実装・off-domain semantic testsまで。登録比較は未実行 |
| production非列挙性 | sourceと参照依存遮断tests。将来64固定interface試行/armはdiagnosticのみ |
| degree/structure transfer、scaling/DF/全PR cost、priority、投稿十分性 | 未確定。GPTがG10一束後に採否とscopeを判断 |

B-F/BM/SP/旧G5主線のSTOPは保持する。新しい恒等式や別対象を理由に過去STOPを解除しない。
