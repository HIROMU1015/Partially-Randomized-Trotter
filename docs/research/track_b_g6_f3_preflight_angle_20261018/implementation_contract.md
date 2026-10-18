# F3実装契約：一般有理入力とterminal facade

本資料は新しい科学実行契約ではない。基点[F1](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dd03eb08b6d768f5578d743ab5ad390bd9c93b07/docs/research/track_b_g6_f1_f2_precision_bit_cost_20261018/preflight_precision_contract.md)と[A01〜A10](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dd03eb08b6d768f5578d743ab5ad390bd9c93b07/docs/research/track_b_g6_f1_f2_precision_bit_cost_20261018/implementation_acceptance_requirements.md)の通常実装である。旧SOURCEはbyte単位で保全する。

## 1. APIとexact入口

`g6_f3_generator.prepare(p,x,m,*,arm='full_return',sigma=1,eta,rho,root_bits,probability_bits,angle_tau,angle_root_bits,angle_terms)`が`GeneratorSession`を返す。精度パラメータにdefaultを設けない。H/Kを出力するselectorは存在しない。

pは非空の明示tuple/listのみ。各値とx/η/ρ/τは、**厳密な型のint、Fraction、またはintのnum/den対**を受け付ける。den>0。float、文字列、bool、Decimal、暗黙のiteratorを拒否する。既に作られたFractionがどの物理入力から来たかは推定できず、`Fraction(float)`の来歴や元物理係数の誤差を認証しない。

p_i>0、exact sum=1、0≤x≤1、odd int m≥1、σはint ±1、0<η<1、ρ>0、τ>0、H/K_root/K_phi/Jはpositive int。零labelは呼出し前に利用者が除去し、元label対応を別途保持する。F3内部では明示pの順番をlabel0…L−1として維持し、並べ替え/再正規化しない。入力identityと精度policy identityを分離する。dimensionsにはp/x/精度の整数対bit長、mの数値とbinary長、H/K/Jの数値を記録する。

認識armは`full_return`、`ordinary`、`partial_return_tail`、`closed_P3_tail`。後3つの既存domain m≥3を保持する。**非zero canonical preflightは今回未対応**で、`CANONICAL_PREFLIGHT_NOT_IMPLEMENTED`を返す。Fullの下界をcanonicalへ転用しない。

## 2. Identityとpreflight

合法な入力でx=0なら`IDENTITY` sessionを返す。正のpreflightは`NOT_APPLICABLE`、`sample(None,trial_id=...)`はdeterministic I。generator constructor、ReturnKernel、parent、乱数、rejection、weight、angle、controlled providerを呼ばない。identityでも入力/policy/SOURCE/σとnonempty文字列trial_idを保持する。Full m=1が可であっても、canonical m=1のdomainを広げない。

x>0ではP0〜P4をexactに評価し、不成立ならconstructorやbit取得前にSTOPする。成立後に固定FullReturnGeneratorを同じp/x/m/H/K/η/ρで構築し、P5を結ぶ。arithmetic assessment単体では`ready=false`、P5は`NOT_BOUND`。実law binding後だけ`ready=true`となる。[証拠仕様](preflight_certificate.md)を参照。

## 3. Session出力とhelper

| API | 出力/制約 |
|---|---|
| `sample(bits,trial_id=...)` | 外部`bits.bits(H)`からstrict int 0≤v<2^Hを受け、旧sampleを1回呼ぶ。`EVENT` / `ZERO` / identityをimmutable Evidenceとして返す |
| `event(word,child,trial_id=...)` | 合法な選択parent/childの**形式的deterministic query**。sampleされたtrialとは表記しない。旧eventを1回照合対象にする |
| `parent_packet(word)` | 局所parent、a/s/masses、d区間、midpoint、acceptance、child law、tangent、phase/time-orderをexact evidenceで返す |
| `angle_for_event(record)` | 同じsession provenanceのEVENTを確認し、旧eventを局所再計算して内容一致後にangle enclosureを返す。新たな乱数なし |
| `preflight` / `last_failure` / `status()` | immutable証拠、最初のterminal失敗、READY/IDENTITY/STOPPED。`run_complete=false`、confidence未認定を常に保持 |

wordは明示tuple/list、合法int label、even長≤m−1、隣接同labelなし。非空wordのchildはfirst(word)以外、空wordは全label。negative/bool indexや外来packetを旧helperへ渡さない。

EVENTには旧辞書を値そのままで保持する。word/raw_word/child、r=s/a、parent phase、逆時間word、実π、c_tilde、W_tildeと`π W_tilde = c_tilde`をexactに検査する。σは外側に保持する。meanは一般に近似M_tildeで、irrational係数の丸めやangleの誤差を消さない。返却dictを利用者が変更しても元Evidenceは変化しない。

## 4. Zeroと失敗

Aは`ZERO/RAW_NONREDUCED`と`ZERO/COIN_REJECT`のみ。raw wordをreduce先へ再配分しない。bit traceはdraw段階/整数/幅、raw word、degree、draw数/bit数を保持し、追加乱数を取得しない。

Bは`TECHNICAL_PRECISION_FAILURE`（preflight不足、angleτ不足等）。Cは`INVALID_INPUT`（型/domain/index/bit値やinput/source/policy/証拠内容不一致）。予期しない環境/実行障害は`UNEXPECTED_TECHNICAL_FAILURE`。P条件を通過した固定処理のValueError guardは`CONTRACT_INCONSISTENCY`としてstageと元exception・SOURCE位置を保存する。例外型だけでA/B/Cを分類しない。

全失敗は最初のevidenceを保持してsessionをterminal STOPにし、その後のcallはbitを取得せず拒否する。trial/parent/child/trace（存在する範囲）、source/input/policy/preflight ID、source location、未完了prefixを保持する。不正なtrial_idは型を記録し、非exactなobjectをevidenceへ直入れしない。B/Cをzeroにしない。結果を救済する増精度・retryはない。prefixは全Nの完了confidenceではない。

prepare前のinvalidにはまだ合法input IDがない場合がある。evidenceは呼出し側へ返し、全runファイル保存・checkpoint・再開機構を勝手に実装しない。OS killなどPython例外として捕捉できない障害の永続保存を保証しない。

## 5. 証拠・SOURCE・cost境界

Evidenceはcanonical JSON文字列とSHA256を凍結して持つ。intを`$int`のhex、Fractionを`$rat`のhex整数対、tupleを`$tuple`、非文字列key辞書を`$map`として明示する。CPython十進桁数制限を全体設定変更で解除せず、20,000-bitの人工整数もexact roundtripする。`read()`は新しい値tree、`to_dict()`はJSON保存用表現を返す。証拠のSHAは改変検出・provenanceであり、秘密鍵署名や悪意あるprocessへのattestationではない。予約されたtagは内部encoding用で、任意外部JSONを一般purpose schemaとして採用しない。

各入口で固定旧3SOURCEと新3SOURCEの実bytes hash、input/policy、certificate、kernel/law stateを照合する。old providerはhashだけ読み、import/呼出ししない。実fileとimport済みcodeの敵対的差替えやmonkeypatch全般へのsecurity sandboxとはしない。

[F2](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dd03eb08b6d768f5578d743ab5ad390bd9c93b07/docs/research/track_b_g6_f1_f2_precision_bit_cost_20261018/full_bit_complexity_accounting.md)の費用は旧generator本体のまま。raw zeroはraw draw/predicateまで、coin zeroは局所packet・2 roots・**coin前child law/sortとL²再sum**まで、acceptedはさらにchild draw/weightまでを支払う。traceは実際に通ったprefixを区別するが、root/GCD全operationの実測profilerではない。

F3追加費用を無料扱いしない。input/P検査とevidenceのexact serialization/hash、各callのSOURCE file読込とstate再hash、P5用のorder/label law再計算、監査trace用のorder/label bit-map再照合、各event invariant、angle時の旧event/packetの1回再計算、half-angle/series、出力保存が別途加わる。監査用bit-mapは乱数を再取得しない。F2式をF3の実CPU/RAM/全N費用実証へ読み替えない。全parent cacheや全N保存を実装しない。

## 6. A01〜A10受入対応

| ID | 今回の実装/検証 | 保持する境界 |
|---|---|---|
| A01 | strict exact validation、domain、σ、dimensions、word/child、missing/unknown精度argument拒否、T1/T3 | 元物理入力の誤差やlabel除去対応表はcaller責任 |
| A02 | dispatcher/identity、constructor/bit/angle禁止spy、T2 | positive sampler成功と区別 |
| A03 | exact P0〜P5/R evidence、全parent列挙なし、T3 | H/K selectorを作らない |
| A04 | frozen law再照合・state seal・実bit-map、T3/T4/T6 | canonical非対応。fresh IIDは外部前提 |
| A05 | local packet/旧event、πW=c_tilde・phase/σ/time-order、T6 | meanはM_tilde。外来pure-parent packet不可 |
| A06 | terminal failure/trace/context、T4 | 無限run・OS kill保存を保証しない |
| A07 | exact enclosure/τ/±σ/provenance、T5/T6 | native/physical certificateなし |
| A08 | 全recordが未完了/confidence未認定、旧budgetは呼ばず保持 | fixed αの旧budgetを変更せず、任意α/estimatorを提供しない |
| A09 | dimensionsとraw/coin/event prefix、bit-draw内訳、旧F2＋上記F3追加費用 | 新benchmark、最適化、全N資源比較なし |
| A10 | exact hex JSON evidence、出力費用/保存を別項、型/serialization failureを明示 | arbitrary-sizeメモリ/出力成功やplatform性能の保証なし |

**新しい数学保証は追加しない。F3技術PASS・科学的GO・実行承認を区別し、mandatory STOP。**
