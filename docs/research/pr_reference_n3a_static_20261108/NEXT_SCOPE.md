# N3-A：次に判断する最小scope

2026-11-08 JST。[REPORT](REPORT.md)。**提案であり、正式採択・実行承認・開始ではない。** 静的範囲は完了してmandatory STOP。新契約／grant／承認証明基盤は作らない。

## 1. GPT／ユーザーが先に決めること

RZ加法性を使う有限辞書は条件付きで構成可能であり、全SU(2) netやSK再帰を先に実装する必然性はない。一方、辞書の実価格がN1の要求値に合うかは未評価。これは研究継続GOや新規性判定ではない。

次へ進む場合は、まず**GPT／ユーザー**が以下を具体化する。

- 最初に求める保証をpointwise一様capとするか、期待capとするか。期待capには別のcanonical質量評価が必要で、最初の辞書取得に自動的に含めない。
- b、固定丸めtie規則、J_b／M_b、基本角度keyと内部精度の有限集合。全体epsilon_G=10^-11を変更せず、量子化・基本word和・emitter余裕が同じ局所予算に入ること。
- CT-4Bの全採用wordを優先する共通policyとfreezeした表。辞書経路以外も含めたT capの証拠をどう取得するか。

今回はbや辞書サイズを数値評価していないため、**即実行できる固定keyリストはまだない**。承認後に都合よくb／precisionを拡張する計画にはしない。上の具体化が数値探索を要する場合も、別途その有限範囲を承認する必要がある。

## 2. 基本keyと再利用候補

本候補の基本角度は`2^j, -b <= j <= J_b`。literal binary64として非ゼロ・有限・正確に表現できる範囲に限定する。実key一覧は、このパラメータを固定した後でだけ生成・保存する。

[CT-4B採用inventory](../../../artifacts/pr_reference_ct4/2026-11-07/KEY_REUSE_AND_SYNTHESIS.json)の既存key読取では、直接のpower-of-two候補として

```text
angle:    0x1.0000000000000p-61
exponent: -11
key:      b199cd712e6abbf24cbd9ddd09cb3b1ef0492ae538bbced87a82051db26ef743
```

がある。保存routeはstrict_clifford_fast_path、guard方式とsequence SHAも保存されている。基本指数域に入るか、保存Frobenius上界が将来割り当てるg_j予算に入るかは未判定。e=-11のラベルだけで基本word積の全体precisionを資格しない。旧C2-B全cacheのpower-of-two集合を新しく走査・集計した結果でもない。

他の基本角に再利用wordがあるか、新規wordが何件必要かは未確定。実辞書／新keyを生成して件数を埋めていない。再利用時もtarget literal、word/hash、W、actual adjoint、guard方式と実上界、採用精度を照合し、legacy欠損を補作しない。

## 3. 最小の追加実装と検証候補

**Codex**へ別途固定scopeを明示承認する場合の候補は、次の関連作業をまとめた有限pilotである。

1. exact binary64→有理数→整数最近接grid→固定bit順の分解と、正側連結／負側全体adjointを行う小さいadapter。既存emitterとrecord検査を再利用し、新しいcompilerや合成法を発明しない。
2. 事前固定した基本key／内部precisionの保存word照合と、missingだけのstrict合成。全dict完成／有限成功／不成立を区別する。旧cacheを更新せずfresh領域へ保存する。
3. 各基本wordのexact行列・strict guard、W付きactual adjoint、有限連結wordのemitterとの一致を必要最小限に確認する。zero、符号反転、丸めtie、grid境界／端点、cache hit/missを含む有限検査を事前固定する。合成後のwordへ新controlを付けない。
4. 全域の被覆・Frobenius合算・online停止は記号証明と完成辞書の全recordへ対応させる。局所の数値成功だけを全support証明の代用にしない。
5. 実採用基本Tとcache優先capを有限に会計し、保存N1閾値と厳密比較する。新しい期待分布評価、全support trajectory列挙、199-shot積／18round総Tは含めない。

内部指数precisionが既存acquire形式へ収まらない場合は、古いrecordのspecを偽装せず、新adapter側の入力／certificate形式を限定して定義する。連結certificateは既存単語の実guardを再利用し、量子化／基本誤差の証明経路を識別する。未評価の連結wordに既存guard_methodをコピーしてPASSとしない。

これらの実装・合成・guard・回路emission検証・価格会計は**今回実行していない**。新しいnative compileは、post-compilerのRZ置換で足りる限り既定作業にしない。必要になった場合は対象と理由を固定scopeへ含める。word合成controllerの無期限待ちを、単なるタイムアウト表示で解決済みとしない。backendの全入力停止は依然未資格である。

辞書の一部だけを取得するpilotを選ぶ場合、その結論はPARTIAL_COVERAGE／実装検査に留める。全域policyを資格するには、選んだ指数域の全基本recordと余裕付き誤差証明の完成が必要。

## 4. N2とRPEは別問題

[CT-5B](../pr_reference_ct5b_20261107/FINITE_KEY_AND_UNKNOWN_PRICE.md)のcommon frame未取得key例`0x1.08ba32d420f2cp+2`とB′ b2 event key`0x1.921fb54442d18p+2`の欠損は維持する。b2のcanonical到達性も未資格。N1は新規合成を行っていないため、そこで欠損が解消したことにはならない。

vacuum辞書の角域を他roleに自動移送しない。仮に同じ表が他targetを覆う可能性があっても、対象域・policy・strict誤差を資格し、N2のscope判断を経る必要がある。今回N2を再開しない。全support CT RPE成功や18round総Tも別途の証拠が必要。

## 5. 反証・STOP条件

後続候補で、対象のliteral表現が崩れる、Frobenius予算／phase／actual adjointが満たせない、必要keyが取得不能、emitter対応が未資格、旧採用word置換が必要、又は固定条件変更・対象外の新callが必要なら、その結果を保存してSTOP。指定有限scopeを完了してもSTOPする。

粗い価格和がN1閾値を超えた場合は「そのcapでは十分条件を資格できない」と報告する。Reference劣位や一般的no-goとは判定しない。改善を得るためのb／precision／新辞書／word policy／fusion探索へ自動拡張しない。

次担当は**GPT／ユーザー**。別途承認まではCodexによる実装・合成・guard・価格評価・科学レビューを開始しない。旧STOP、共有SERVER 2、他ユーザー／別Trackを保全する。
