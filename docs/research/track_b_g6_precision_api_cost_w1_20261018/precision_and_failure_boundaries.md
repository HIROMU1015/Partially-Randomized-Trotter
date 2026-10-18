# 精度・support・failureの境界

本書は[BIT]§§3–6、[K] / [G]のguard、[G7C]の既存規約を対応付ける。H/Kの新しい選択式・十分定数・adaptive refinement・retryは作らない。義務IDは[主表](precision_api_obligation_map.md)を参照する。

## 1. 存在保証と実行条件

正の明示有理p、正確なΣp=1、0<x≤1、有限奇数m、valid parentという既存域で、[BIT]§5は次を既に与える。

| 理想量 | 既存一様下界 | API上の使用箇所 |
|---|---|---|
| order q_l=b_l/B_o | t_(m−1)/3 | K208–212のnormalized root intervals→K55–82 |
| label | p_min>0 | G `_law`75–76/222 |
| valid-parent条件付き受理A_u | 1/(2√2)>1/4 | K228–229→K85–94 |
| 有るchild q(i∣u) | p_i/2≥p_min/2 | K230–231 |
| parent d_u | t_(m−1)p_min^(m−1)/2 | K224–227のrelative midpoint検査 |

これらのlog inverseの既存評価と、所要精度が入力bit長・**次数m**・log(1/η)・log(1/ρ)に多項式で取れるという論証を再利用する。これは固定defaultsが全入力で成功する保証ではない。一般入力から具体的K/Hを出すSOURCEはなく、今回もその式・アルゴリズムを補っていない。

下界は「どの全parentにも正しい意味の十分精度を選べるか」という数学存在を閉じる。現SOURCEと結ぶには、選んだK/Hが以下の**すべての実guard**を満たすことを、sampling開始前に同じ入力/誤差条件について保証する必要がある。これは既存一様lemmaの適用義務であり、別の一様存在定理が欠けているという指摘ではない。全語列挙をpreflightに要求する仕様でもない。

## 2. 実guardが検証すること

`root_interval` [K]45–52は非負の正確有理radicandと正整数Kを受け、整数division/isqrtで分母2^Kの外向きroot区間を返す。rootが正確に表現可能なら一点区間。区間幅が小さいだけでは、非常に小さい正数の相対精度やlo>0を保証しない。

`order_intervals`208–212はb_l rootの端点とその総和からnormalized区間を作る。`dyadic_distribution`55–82の呼出し元は、その区間が**同一の真のnormalized law**を包む根拠を持つ必要がある。Σlo≤1≤ΣhiはSOURCEの必要検査であり、任意の外来区間データの数学的正しさを認証するものではない。Gのlabel/childは正確有理lawの一点区間を使うため、root幅の要因はない。

dyadic処理は、正の全lo、0<η<1、正しいH、normalized条件の後、midpointを再正規化してfloor、余り順sort、counts調整を行う。その後の

`q̃_i >= (1−η) hi_i` かつ `q̃_i <= (1+η) lo_i`

を**全要素**で検査する。これが通れば既存のrelative error条件とfull supportを実lawへ接続できる。粗いHだけでなく広いinput intervalも失敗原因になる。単にq_i>0、sum(q̃)=1だけを満たしても十分ではない。

`digital_parent`224–227ではd区間lo>0かつ `max(mid−lo,hi−mid) <= ρ lo` を検査する。これはabs(d̃−d)≤ρdに対する既存の保守的受入条件である。A²のroot区間はhiを1へclipした後、`dyadic_bernoulli`でlo>0、hi≤1と下向きthresholdの相対下界を検査する。thresholdはlo以下なので上向きの過剰受理を行わない。区間不足を受理確率0へ変更する処理はない。

H/Kの正整数性、0<η<1、ρ>0、合法word等は全体契約の前提とする。低水準関数の全てが同一の型/範囲validatorを備えるわけではない。例えばHはshift操作にも使われ、ρはweight rootの比較に使われるが、fullのconstructorには包括的なρ validatorがない。入口仕様の不足を、新しい数理domainの採択で埋めない。

直接の`event`/`rational_event_weight`呼出しも、登録group index、合法word/label、child indexとpacket/lawの一致を呼出側が満たす必要がある。Kの`_word`はlabel範囲を検査するが、Gのcanonical `event`162–191は同じ包括検査を持たず、group/label/childのPython indexingに依存する箇所がある。負index等を含む任意の不正呼出しが必ず意味的に拒否されるAPIとは認証しない。`sample`が自身のlawから生成する経路と、外部から自由に呼ぶhelperの契約を分離する。これも入口仕様/validationの不足であり、新しいdomainや定理を必要とする根拠ではない。

## 3. 三つの状態を混ぜない

| 状態 | 固定SOURCEの経路 | 統計・保全上の扱い |
|---|---|---|
| 正当なzero trial | G240–241：raw語が既にreducedでない。G243–244：dyadic coinで不受理 | X=0として**全trial N**に含める。reduce先へremapしない。量子前棄却でも古典draw/query/coin費用は残る |
| precision technical failure | K65/79–81、92–93、226–229等のpositive endpoint/相対誤差不足 | 正当なlawのzeroではない。部分結果・失敗を保持しSTOP。固定N完了としない。今回retry/自動bit増加を許可しない |
| 不正入力・契約違反 | p/和/m/x、word parity/reduced/label、bit integer範囲、非dyadic law、残余幅等の不正 | fail closed。precision不足と区別する。多くはValueErrorなので例外classだけでは原因分類できない。例外をcatch-allでzeroへ変換しない |

`root_interval`の入力不正、positive-support不足、coefficient precision不足は原因に応じて区別すべきだが、今回SOURCE例外型・保存schemaを変更していない。分類は将来の呼出契約の要件である。

## 4. sampling前に必要な一様条件

G fullはconstructorでorder/labelを検査し、local d/accept/childは`sample`中に検査する。したがってconstructor成功だけで、未訪問parentの検査成功は認証されない。既存の全trial mean/range/Bernstein保証を「この固定APIが完走する」という保証まで接続する場合、**同じ事前固定K/Hが、そのlawで到達可能なすべてのvalid parentのroot/accept/child条件を満たす**ことが必要である。

既存の一様下界とbounded-construction論証は、この条件を満たす精度が存在する根拠になる。SOURCEへの具体的設定・証拠の接続は未実装。実際に訪問したparentでguardが通ったことだけから、全lawに対する事前保証を作らない。途中例外があり得るまま、完了したrunだけを選んで無条件1−α保証と呼ぶことにも、既存契約は根拠を与えない。失敗runの記録は有効だが完全なconfidence結果ではない。

この不足に対し、新しいfailure probability配分、conditional successの定理、resampling、適応的精度救済を今回考案していない。まず既存保証の事前適用を接続するのが最小である。

Canonicalのconstructorは既存group root精度を確認するが、`conditional`/`first_distribution`/root-childのdyadic lawは試行時にも作る。[G]153–212。Full用parent下界をそのままcanonicalの全分布の下界と呼ばない。canonicalでは登録group・正確な条件付きlawの式と到達domainを別途直接照合する仕様が必要であり、新しいFull/Partial比較命題の問題ではない。

## 5. identity・境界の責任

x=0のM=Iは[MATH]§1で解決済み。一方K constructorの0≤xと、G7全生成APIがidentityを返すことは同じでない。K221–222は明示拒否、canonicalは初期化拒否、fullにはidentity dispatcherなし。今回API域を広げず、identityは別の呼出経路が必要と記録する。

L=1/m=1/x=1を主定理の未解決境界にしない。Fullのm=1とcanonicalのm≥3を区別する。s=0 pure-parentのK規約があることから、fullのchild前提経路を任意のpure-parent packetへ拡張してよいとは推論しない。

## 6. mean・残余幅・confidence

実πと正確有理W̃からπW̃=c̃が成立しても、d midpointの係数biasは消えず、ideal angleの下で対象はM̃。angle/native誤差は別。range・momentは[U]§7.4、[G]266–284の既存式を使い、B_o/Uのhiを使う。U.loをB_RA下界にしない。

G7 `budget`の共通biasは `2*[3ρ+3(1+ρ)*2 primitive_error]`、`remaining=epsilon_axis−common_bias>0`。exact CQと2 Rzの既存誤差規約が条件である。負のprimitive_errorやη/ρ domain不正を許す一般budget仕様ではない。SOURCEのremaining検査だけでは全前提を認証しない。

固定規約はα_axis=1/320、16 axes family≤1/20、ln640の上向き有理評価とceil N。一般αの数学式と、任意αを受け取るAPIは異なる。新しいconfidence、十分N、最適予算は作っていない。fresh IID bits/測定モデル、全trialの正しいlaw、事前precision条件、十分N完了が必要で、SHA-counter traceや例外を除いたprefixは代用にならない。

本記録は新しいscience contractではない。元Hamiltonian、Taylor remainder、native provider、PF/RPEへの拡張はBの固定監査と別途判断へ戻す。**mandatory STOP**。

<!-- FIXED_REFERENCES -->

[BIT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g6_finite_bit_and_access_audit_20261010.md
[G]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ab2549f41b3546fb3940342a2162dd9ee93699c4/src/trottertracks/algorithm_codesign/g7_generator.py
[G7C]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ab2549f41b3546fb3940342a2162dd9ee93699c4/docs/tracks/algorithm_codesign/g7_mathematical_and_execution_contract_20261010.md
[K]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/28cfabb1d47e0e1824bce1e154a9c289738fa2b9/src/trottertracks/algorithm_codesign/return_aggregation.py
[MATH]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bc649c592c8c2b625b49094b85c27e380faa3cf4/docs/tracks/algorithm_codesign/g6_independent_mathematical_audit_20261010.md
[U]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/101bf3426fcd04cc0c01c0d202d0916268765d97/docs/research/track_b_G6_scientific_review_20261010.md
