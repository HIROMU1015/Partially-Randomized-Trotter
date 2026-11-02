# 限定構成検討の引継ぎ：新算法候補なし・既知DF shiftへ還元

2026-10-17 JST。[主報告](limited_construction_report.md)、[固定証拠・検査台帳](evidence_index.json)、[公開receipt](publication_receipt.json)。研究方針の正式採択や実行契約ではない。

## 現在地

今回の分類は **`REDUCES_TO_KNOWN_METHOD`**。一つのspin-free・number-preserving DF平方blockに対し、fixed-N0の恒等式で基準空間軌道の係数を消す書換えを検討した。controlled位相と実行順序まで紙上で追跡したが、既知shift＋通常のdiagonal compilerにない差は残らなかった。

結論を変える直接の証拠は [Güntherほか、arXiv:2503.05647v2 Appendix C.3、式(C14)](https://arxiv.org/pdf/2503.05647v2)。本案の `D_s=D−s(N̂−N0)` は同式の `f=−s` である。基準係数が0になるparameter選択と通常のzero回転除去を、独立した新生成規則へ格上げしない。

また、`B−B_s=2κsL(N̂−N0)−κs²(N̂−N0)²` は、[Loaiza–Izmaylov、arXiv:2304.13772v2 §II、式(6)–(9)](https://arxiv.org/pdf/2304.13772v2) のnumber差とnumber-preserving一体operatorによるshiftに対応する。これらの既知恒等式を参照するために、LCU/block-encoding/qubitizationを新研究へ採択したわけではない。

## 残すべき区別

| 記述 | 証拠区分・限界 |
| --- | --- |
| fixed-N0上での式(3)–(4)の一致 | 主報告の局所的な紙上式変形。独立した新定理の科学的受理ではない |
| 基準α/β modeが対角phaseの支持から外れる | 記号的な係数支持の比較。Gaussianからそのmodeを除去・qubit削減したのではない |
| 相対位相を保つRZ/RZZ/Phaseへの展開 | 既知回路生成規則の非実行記述。実装済み・数値PASSではない |
| 素朴なunshifted DFより回転候補が少なくなり得る | native利益・depth・T数・総費用は未測定。強い既知shift baselineは同じ処理を行える |
| 新規則が得られなかった | このblock・この案の限定結果。encoding研究全般の不可能性ではない |
| 手順と式がある | 独立した新候補はないため、別の「新規則仕様書」は作成しない |

既存preflightで確認したtapering/GSE/Paldusの違い、mode順、global phase条件はそのまま適用する。固定N、Nα/Nβ、Sz、S²、Z2 parity、GSE codeは同一ではない。本案のspin-free条件のもとで保存できる量と、一般のN保存Gaussianで保存できる量を混同しない。

## 未取得・未検証

- 実DF入力、state sector品質、丸め・合成誤差のcoverage、数値correctness、runtime、native資源、shots、RPE全roundの利益は未検証。
- PR論文はversionを固定したAppendix Cの必要箇所、BLISSは§IIの必要箇所、SCDFは§§2.3–2.4を読解した。全論文の証明・出版版差分・全関連文献を独立認証したわけではない。
- 前回preflightで取得済みのPaldus本文範囲は維持。未取得の過去Claude原本等を今回取得したことにはしない。
- これらの未取得情報を埋めても、式(C14)への直接対応が消えるという根拠はない。native費用が欠けていることを理由に、新規性判定を単に保留へ戻さない。一方、実務的な応用価値まで否定しない。

## 状態と承認境界

| 対象 | 今回後の状態 |
| --- | --- |
| 新しい独立算法・テーマ | 未採択、独立した候補なし |
| 今回の紙上設計作業 | 承認範囲で完了 |
| 新SOURCE・実験契約・実行 | 未準備・未承認・未実施 |
| 新authorization/grant/marker | 0件 |
| ④停止系列、②Gaussian積等のclosure | 変更なし。名前を変えて再開しない |
| ③M1-S、①資源評価 | 今回の対象外。再解析・進捗判断・方針変更なし |
| mandatory STOP | 維持 |

## 再開時に最初に読むものと次の担当

次のCodexは、まず[主報告](limited_construction_report.md)の§§4・7を読む。今回の案を新規則として再提案せず、既知のnumber shiftとcontrolled scalar処理を再度「新しい接続」として実装しない。SOURCE・raw・科学契約・authorization・grant・ledger・markerの保全を確認する。

**次の担当はユーザー。** この限定結果を確認し、必要なら別の具体的な構成課題を新たに指示する。現時点で自動的に始める作業はない。今回の結論に対する重要科学レビュー、別候補設計、正式採択、SOURCE実装、数値/compile実行はそれぞれ今回の公開完了からは承認されない。

候補classを広げたり、既知法の応用評価へ目的変更したりする判断もCodexが代行しない。**公開・保全確認後、mandatory STOP。**
