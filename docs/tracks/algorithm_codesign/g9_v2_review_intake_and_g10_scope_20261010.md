# G9 v2 GPT review受領とG10 bundleの作業範囲

利用者が2026-10-10に[GPT科学review](../../research/track_b_G9_v2_scientific_review_20261010.md)を
「こんな感じで進める」と共有し、今後のTrack B方針として採用した。
本記録はその受領・仕様整理であり、G10の結果前契約・source freeze・実行の完了報告ではない。

## 採用方針

return集約familyを優先候補として限定継続する。closed P5は低次数の標準実装と位置付け、
一般局所生成器は低次数特殊化で代替できない範囲の価値を調べる。
独立新規性、一般優位、主algorithm採択、投稿十分性は未確定。
研究方針と科学的解釈はGPT側、以下の固定scope内の技術作業はCodex側が担当する。

G10で問うのは、同じp・x・providerについて、低次数return吸収にordinary tailを加えた対照後にも、
一般次数のfull returnに資源上の追加価値または有効条件が残るか、である。
各次数内部の同一有限Taylor first operator momentを比較する。
次数の違う費用を、同一accuracyのexact exponential性能として順位付けしない。

## G9とレビューの証拠境界

- G9 v2基点：`c95736fd2990f5ef6dd1cb5866421fd78bb28687`。
  [原handoff](g9_v2_results_and_gpt_handoff_20261010.md)と原result/authorization/marker/STOPは保持する。
- `G9_MATCHED_NATIVE_RESOURCE_MAP_COMPLETE`は登録known/development synthetic providerの
  source-bound local資源予測。実量子測定、独立再現、immutable CI、分子/DF、PR/QPE最終総costではない。
- review §7の任意sampling CTS固定policy下界はGPT側の追加導出。
  固定literal CTS辞書・取得列・共通bias・同じBernstein policy・full-support proposal・共通非負prep Tに限定される。
  G9の事前登録certificateや原classificationへ組み込まない。
- 下界はG10-Aで原resultのrational係数/native費用から独立確認する。
  T=0 real eventsを残し、理想Pauli係数・rational angle・digital係数を分ける。
  本受領ではreview末尾の再現添付を追加取り込みせず、その再実行・独立検証も行っていない。
- canonical CTSの表示削減率を任意proposalに対する保証削減率としない。
  identity吸収、別辞書、精度再配分、stratification、別confidence方式は当該下界の外。
- 同familyのclosed P5/general local full差はnormalizer・zero-fill・十分予算の差として扱う。
  selectorの線形走査を含む現在の実装全体をO(L)と呼ばない。
- native Tをprimaryとし、CX/1Q/workspace、期待量/accepted tail/hard attempts、古典取得費用を別々に残す。
  partial/P3のnative cost-aware最適proposal分離は未完了である。

## G10を一束で準備・実施する範囲（review §13）

| 区分 | Codexの作業 | 結果前の固定・報告事項 |
|---|---|---|
| A：保存証拠とclaim | CTS固定policy下界の独立再導出、partial/P3 native cost-aware対照の不足、prior-art差分の具体化 | 登録値/下界/候補値を区別。未分離なら未分離と記録。既存lawの再探索や新合成はこの部分に不要 |
| B：次数比較 | p=(1/5,3/10,1/2), x=5/7、同じG9 3-qubit provider/direct lowering、m=3/5/7 | m=5は既存anchorを再利用。各m内部で同一P_m、同じ情報accessと予算規則 |
| C：非列挙性 | production local queryと参照列挙を分離、低次数group構築/selectorの会計 | angle/cache/finite-bit/termination/abortと古典取得費用を記録。参照表をproductionへ注入しない |
| D：まとめて固定 | 同target/provider/主要比較を維持した型修正・区間・tests/source整理 | 各testごとに研究承認を分割せず処理。bundle完了/失敗後にmandatory STOPしGPTへ返す |

Bの対照はstreaming ordinary、partial P3+ordinary tail、closed P3+ordinary tail、
closed P5+ordinary tail（m>=5）、general full、同じP_mのliteral matched CTS。
P3/P5には閉形式を使い、一般器を強制しない。
m=7では特にclosed P5+次数6/7 ordinary pairとfull returnを比較する。

cost-aware有限proposalを加える場合は全方式へ同じ情報access/有限構成規則を認める。
T=0を正しく扱い、leading ISのinfimumを実行可能予算と呼ばない。
共通prep Tを後から選んでwinnerを作らず、T切片/Kまたは成立領域を保存する。

## G10実施前に残る固定事項

G10のmachine-readable contract、source identity、row/axis総数、failure配分、key inventory、
native angle/precision/backend/compiler、bias、CPU/RSS/key/output上限はまだ固定していない。
key上限は静的group/angle見積りからCodexが決めるというreviewの委任に従う。
G9 authorization/消費済markerをG10へ転用しない。実施は新しい固定記録へ結び付ける。

今回の受領で変更したのはreview写し、受領記録、日付note、manifest、文書索引の追記だけ。
新しいscientific run、合成、sampler、LP、matrix、circuit、DF/分子/NPZ/GPUは実行していない。
G10 runner/実装/contractがあること、launch可能であること、下界検証が済んだことを主張しない。

## GO/縮小/STOPを判断する段階

- 低次数対照・sampling自由度後にも一般次数の利益/有効条件が残る：GPTが方法claimと次の構造対比を判断。
- closed P5+tailで説明できる：一般器を性能上の主役とはせず、低次数実装と一般構成の限界を整理。
- 既知再構成に吸収される：GPTがmethod deltaを縮小。名称変更で同じ探索を続けない。
- 未分離：認証不足と真の非改善を分ける。certificate精密化を無期限の目的にしない。

結果後に5%/10%閾値を逆算しない。G10終了/失敗で必ずSTOP。
m=9、新p/x/provider/seed、精度探索、旧G5主線、全面v4、DF/分子/QPEへ自動拡張しない。
その時点でGPTが原稿claimの集約、構成noteへの縮小、必要な一般化一つの採否を判断する。
