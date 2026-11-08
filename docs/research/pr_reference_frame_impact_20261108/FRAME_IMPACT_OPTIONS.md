# 43 frame literal：変更できる保証と保存wordの比較案

基準340670387dbe58bdda5606fdf222408774be1932。[REPORT](REPORT.md)の静的照合と既存SOURCEを前提とする。以下の数値比較・選択・runtime拡張は未実行である。

## 1. Template／context／多重度

完全な43-key表は[STATIC_CROSSWALK](../../../artifacts/pr_reference_frame_impact/2026-11-08/STATIC_CROSSWALK.json.xz)の`candidates_43`にある。各keyに現word・保存非劣ID・元record metadataと、arm／forward template／context／position／wire／positive literal／出現数を接続した。`S1_template_max_quoted`は旧S1のscalar引用であり、新weighted寄与ではない。

| 範囲 | 43-key再選択による変更可能性 | S1との関係 |
|---|---|---|
| SOURCE `8b103…` forward | 6 literal、両arm各4 context | template和とそのcontext-maxが変わる可能性 |
| SOURCE `cb0c4…` forward | 26 literal、同上 | 同上。B1の1 recordはここに属する |
| SOURCE `e6adc…` forward | 11 literal、同上 | 同上 |
| SOURCE全7 basisのinverse | 保存hitなし | 現word／上界を変更しない |
| 他4 SOURCE basisのforward | 保存hitなし | 現word／上界を変更しない |
| D／actual SOURCE inverse D | 保存hitなし | Dの不足は解消しない |
| products／CSWAP | 保存role metadataもframesのみ | 既存式・word・上界を保持 |

上の省略IDはREPORTの完全IDへ対応する。CurrentとcandidateでSOURCE RZの位置・wire・符号・多重度を変えず、positive keyに対応する保存guardだけを変更する。新角度、modulo同一視、basis共有やD融合の追加削減を導入しない。

S1はforward／inverseを各々context-maxにしてからpairを足す。そのため、forwardの一部keyを改善しても、inverse寄与、未変更key、最大contextの切替、他basisの最大値が残る。最大basis/contextの識別は保存証拠にないため、新しいargmax診断を承認後に行う。今の段階で寄与順位・削減率・最良basisを決めない。

## 2. 保存frontierの有限比較

43 literalそれぞれについて保存S2の`nondominated_ids`をそのまま使う。全375 literalのfrontierを再計算せず、43以外のwordは固定する。3規則は全て同じpoolから選び、tieは明示的に固定する。

| 規則 | 選択順序 | 見るべきtradeoff |
|---|---|---|
| G | guard昇順、T昇順、record ID辞書順 | 最も小さい保存guardが局所Tを増やす場合も残す |
| T | T昇順、guard昇順、record ID辞書順 | 最安局所Tが保証を緩める場合も残す |
| NT | 現T以下の保存非劣候補＋現wordから、guard、T、ID順 | Tを増やさずどこまで保証を改善できるか。適切な候補がなければ現wordを明示的fallback |

実際の選択は今回0件である。Gを科学的勝者として採用せず、3規則の実行可能性・frame上界・非中央残余・local T変化を並べる。同じ選択mapならaliasを記録し重複評価を省く。これらは有限のper-key比較であり、全体のT最適性を主張しない。

単位は区別する。KeyのTは保存wordの離散T+T†字段、local template Tはそのnative出現数で重み付けした和である。Guardは保存strict word上界で、実誤差ではない。Context別のguard-maxとT-maxを同じpath／contextだと仮定しない。Local Tを3qrやshotsで掛けて全RPE価格とする処理は含めない。

B′／Referenceは同じpool、選択規則、record mapを使う。N_B626／N_R630、既存alphaは保持し、共通frameコストを総費用から相殺しない。真E0／gap／ground古典記述などの無料情報を片armへ与えない。

## 3. B1 overlayを使う際の最小差分

再利用するSOURCEは[SingleSavedRecordOverlay／SavedRecordCertificate](../../../src/trottertracks/pr_reference_s2_record_overlay.py)、[lower_native](../../../src/trottertracks/track_c2_b_verify.py)、[consumer](../../../src/trottertracks/pr_reference_common_word_consumer.py)、B1人工test／固定4 fixtureの証拠である。旧SOURCEは変更しない。

B1 classのconstructorは単一literal、precision=-13、strict seed17 specを固定する。保存候補のlabelは-13,-12,-11,-10,-9,-8,-7,-6であるため、そのまま43候補全てを受け入れるとはいえない。将来runtime化するなら次の最小差分を新SOURCEとして資格する。

1. 元literal・元exponent・backend version・strict spec・record/sequence hash・guard methodの検査を保持し、許すexponentを上記保存集合へ限定する。
2. 同じcertificate表現を使うimmutable literal→record tableにする。`provide(theta)`にarm/role依存の選択を加えない。
3. 43以外は現providerの条件とwordへ委譲し、C0優先・domain制限を勝手に変えない。
4. Emit時はpositive sequenceだけをsequence-only carrierへ渡す。Transport -11はbackend specを上書きしない。Negativeはunchanged lower_nativeでcomplete actual adjointを一度だけ実施する。
5. W、SOURCE relative phase、wire、prefix reset／terminal、inplace joinを保持し、選んだ新経路だけを有限correctness対象にする。

今回はこのruntime拡張を実装・buildしていない。準備したnumeric SOURCEはmetadata guard/Tの置換だけで、providerや回路を生成しない。保存strict record再利用で合成を省ける可能性はあるが、全recordのemitter／shell資格まで得たことにはならない。

## 4. D・S3・全体費用への残項

Dは保存164 literalのfrontierで現wordが非劣という結果を維持する。精度とTのtradeoffは既存保存点を参照し、今回再選択・ランキングをしない。D templateのkey別／context別寄与、forwardとactual SOURCE inverseの相違は、43 frame診断の結果を見て次のscopeを定める。

既存[S3条件付き共役式](../pr_reference_s3_s2_reuse_20261108/IMPROVEMENT_OPTIONS.md#1-共役式と資格の段階)は

`||A D_tilde C - B D B†|| <= eps_D + min(eps_f+eps_i, min(2,|t|W)eps_f + eta)`、`eta=||C-A†||`

である。SOURCE inverse、個別negative RZのactual adjoint、basis全体のactual adjointは別の資格である。全pairのeta=0やcenterのnear-identityを仮定しない。A1の2 pair defect評価と、basis全体をactual adjoint化する別構成は今回は実行しない。

現加法証明ではDだけでもjoint予算を超えるため、frame改善単独をRPE成功へ結び付けない。中央・domain・emitter・shell・terminal、旧diskとsynthesis diskの単位、位相／半径／整数shots、全18round価格は後続課題である。局所保証が改善しても、全体Tの悪化やstrong B′との公平性を確認する前に正式policyを採択しない。
