# C1未達原因・異符号機構：GPT科学判断への引継ぎ v1

2026-10-13 JST。基準commit `3dd94637e8da77ce6e1f724f12f6a67dd4ded7ed`。
[静的技術監査](qc_c1_sign_mechanism_technical_audit_v1_20261013.md)、[保存literal索引](../../artifacts/qc_c1_sign_mechanism_audit_v1/2026-10-13/saved_literal_index.json)、
[形式恒等式検査](../../artifacts/qc_c1_sign_mechanism_audit_v1/2026-10-13/symbolic_identity_checks.json)を主資料とする。
保存pilotは非分子2qubit Pauli4H/39方法/6h、h=1/(8Λ2^s),s0…5、Λ=fragment係数L1、target10^-5/10^-6 model units、controlled価格3/6。
geometry/basis/DF rank-policy/L_D/固定粒子sectorは適用なし。
元940calls・独立監査・16通常比較改善0は既存結果で、今回再生成していない。

## 今回解消した点と残した点

| 項目 | 判定・科学判断の材料 |
|---|---|
| 主要な未達原因 | **SUPPORTED**：真3kernel dは全4点で正なので正時間leading相殺なし。S1/S4で近似dだけが負、S4では近似macro d=0でも保存真dθは正。符号誤認と追加controlled費用が具体的な問題。 |
| 数式で説明できる同符号領域 | **CONDITIONAL**：同Pauli族の十分弱い結合で共通の正のζ²L項と明示的剰余boundを導出。4点の全てが十分条件に入るかは未代入・未判定。 |
| 現族の構造制約 | **SUPPORTED**：groundで⟨B⟩≤0、α,y≥0から2dABC+dACB≥0。二つが同時に負になることはないが、各々が正になる証明ではない。 |
| 現辞書の異符号相殺 | **CONDITIONAL**：nonzero d_j,d_kは異符号が必要十分。rootは解析的に一意。固定4aは四つの特定ratioだけでexact cancellation可能。 |
| 現pilot族内の異符号可能性 | **UNDETERMINED**：correlation/ground quadratic formを係数で閉じる条件が未成立。正/負inputの探索は行っていない。 |
| 一般の三fragment辞書 | **SUPPORTED（数学的可能性のみ）**：τ未代入のPauli形式familyでopposite signs。pilot族/native template外、H自体が単一Pauliなので資源優位例ではない。 |
| selectorとmacro全体の区別 | **SUPPORTED/UNDETERMINED**：近似符号誤認は確認。未選択24macroの最良有限h frontierやselector regretは、新posthoc集計をしないため未判定。 |
| 情報・precision保証 | **SUPPORTED**：6/32validはPF精度保証の件数で、16通常費用比較の改善0と異なる。gap=nullのS2/S4と、gap有のS3 strict-target nullを区別。bound内のdominant項は未分解。 |
| 既知法との差 | **NOT_SUPPORTED_IN_FIXED_SCOPE**：同catalog/情報/score/fusionを与えたknown-compositionとgeneratorが同じ。processing/conjugationやreverse-isospectral対は真eigenphaseの独立改善を生まない。 |
| 有限hの悪化原因 | **SUPPORTED（保存二例に限定）**：近似相殺を選んだ二例では保存高次remainderが主要biasより小さく、真主要項が残る説明を支持。全候補/grid外への一般化なし。 |
| 次の境界 | **GPT科学判断の開始承認待ち・Mandatory STOP**。新science/posthoc/compile/C4/正式採択を自動起動しない。 |

## 作業Eの七つの問いへの回答

1. **説明できる主要因**：真の同符号、近似stateでのみ発生した負符号/zero、正時間・同じfusionの費用制約、強いsingle/subdivision/S4/C2対照。
   保存通常比較の未達をbranch失敗に帰す根拠はなく、bad-state overlap条件と区別する。
2. **現辞書・現族で異符号は成立し得るか**：一般辞書の一律排除はできない。現Pauli族ではpartial必要十分条件とweak非相殺領域までは得たが、係数だけの族内存在/不存在は未決着。
3. **真値なし利用の道筋**：full Dの取得可能な期待値・不確かさでzero crossingを排除し、固定aに対するjoint幅とfinite-hを認証する方法は条件付きで書ける。
   安い符号certificate、測定/state費用、強対照との総費用回収を具体化できたとはしていない。
4. **同符号・加法費用制約を回避する別機構**：現sharingのみでは回避しない。有限h/角度依存合成/追加fusion等は一般的な例外になり得るが、今回は新しい具体的有利機構を特定していない。
5. **判断を変え得る情報**：族内異符号を係数で確定する構造、または安い運用sign certificateが第一。
   別認可の保存catalog oracle/regret診断は「selector不足か、有限h潜在利益なしか」を分け得るが、情報価値をGPTが先に判断する。
6. **判断が変わりにくい追加作業**：現all-positive4点の正時間係数を増やすだけ、reverse-isospectral対の再導入、同一known selectorの別名化、singleに合法fusionを与えない比較。
   情報取得/剰余/controlled費用をzeroにする仮定も研究根拠を強くしない。
7. **確定/未確定**：保存scopeの改善0、近似符号誤認、partial sign関係、weak-coupling十分条件は記録できる。
   全族同符号定理、族内異符号例、未選択catalogの不能性、分子/総QPE費用優位、新規性、継続/終了/C4は確定していない。

## 新しい数値実行を必要とする場合

今回は禁止なので、監査本文の最小案を提案に留める。
保存4H/24macro/6hのoracle frontier/regretだけを別namespaceで取得する案は、fresh eig/新H/compileを使わず既存rowsと価格のみを扱う。
それでも新しい数値的事後集計であり、別の明示実行認可が必要。
保存vectorからG_AやPauli correlationを取得する案も新しい物理量計算になるため、今回実施していない。
この監査を追加実行の自動承認と解釈しない。

## 運用上の終了点

Codex技術監査は二つのMarkdown、形式恒等式、literal/provenance/未決着表の公開・remote照合までで完了する。
新しいGPT科学レビューは未開始。C1の正式継続・再設計・終了、C4への移行、中心テーマはGPT判断に残す。
N1 STOP/N2 HOLD/N3 CLOSED/THRIFT現template終了、旧source/results/auth/STOP、既存global manifest FAILを保持する。

## 公開・remote確認（2026-10-14 JST）

監査本文のpayload `fb448c7d968edc7d90b8eac8b42e6f433b73c94b` をGitHubから独立bare cacheへ取得し、23変更blobs、既存280files、固定source/input20files、一次結果9files、追加38relative linksをbyte/SHAで照合した。[取得記録](../../artifacts/qc_c1_sign_mechanism_audit_v1/2026-10-13/publication/receipt.json)。旧134manifest records/non-result fields/global description FAIL、原指示CRLF/SHA、root側既存131filesの編集を保持した。
これはGitHubからの取得一致で、外部科学再現・新しい数値評価・GPT科学レビューではない。新科学/posthoc/費用再計算/compile/retry/C4はzero、Mandatory STOP。
