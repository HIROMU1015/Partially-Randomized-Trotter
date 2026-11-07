# 実H4 Signed Support metadata-only：限定科学実行契約草案

2026-11-07 JST / SERVER1。**DRAFT・未採択・未実行。本書の公開は科学実行許可ではない。**
入力・hash・SOURCEの具体値は[固定binding](SIGNED_SUPPORT_H4_METADATA_BINDING.md)を本契約の一部とする。今回は3文書の作成だけで、新runnerは実装しない。

## 1. 問いと結果の意味

保存actual finite-grid Qにおいて、u=s_rotation|S_rotation|/2がB1-OLのh1で非定数かを、exactなΣ1>0で判断する。これにより確認するのは**metadataに基づく構造的適格性だけ**である。

Σ1>0はpopulationのfeature variationを意味するが、finite trainでのrank2、X/Y測定bitとの相関、残差分散低下、学習費回収、compiled RZ改善を意味しない。それらの統計的有効性は本実行後もUNKNOWNとする。構造的適格でもfresh train/paired評価へ自動進行しない。

## 2. 固定科学条件と入力

| 項目 | 固定する条件 |
|---|---|
| モデル/geometry/basis | 保存linear H4 / 1.00 Å / STO-3G |
| Hamiltonian/system | legacy DF12 / 8 system qubits / 全4電子sectorを後段state参照として維持 |
| split/PR | L_D=3、m0、q1、δ=.02、r1、K2、coefficient_atol=0 |
| law | N1 `744e1a09661ebba6841fe5faf456f88c41c47780` のLAW.json、2^53 grid、orders=(0,2)、products→rotation |
| catalog | 同N1のPREPARATION.json全bytes。component_id/fragment/sign/supportを同一行から取得 |
| 層/label | 全5正層、a=`df-fragment-3:Z0Z1`、b=`df-fragment-6:Z0`、定義・順序・質量の変更なし |
| basis policy | N1 selected full policy fingerprintを固定参照。再構築・compileなし |
| feature | b23b3c0の固定version、u=s|S|/2∈[−1,1]。絶対値・角度・state・bitを追加しない |
| state/engine | N2 `5e9bd08068738cdf1070bdc43bee8194b113ae33`、認証 `fdf00cee3c87d804845497afc89ac6a71f3b3f59` を参照のみ |
| 軸 | metadata-only評価は軸非依存で1組。将来のX/Yは保持するが、本評価を軸ごとに二重実行しない |

元NPZ、N2 state、gate列、engineは数値ロードしない。Hamiltonian/partition/preparation/CDF/policyを再計算しない。旧224/752 bit・係数・乱数・cacheは入力に含めない。

## 3. SOURCE・環境と実行前条件

1. b23b3c0の既存SOURCEを変更せず、[binding §4–6](SIGNED_SUPPORT_H4_METADATA_BINDING.md)のI/O・記録・STOPだけの薄い入口を次回実装する。新しいsampler/科学kernel/managerは不要。
2. 保存された106人工PASSを参照し、追加入口に対しては人工bytesでcall一回・hash/schema不一致・constant h1・output衝突・禁止関数非起動・STOP保存を確認する。実H4をsmoke testに使わない。
3. 新入口・人工テスト・実行設定を専用branchでcommitし、remote SHA・本文・blob/treeを照合してから実入力を投入する。実行SOURCE SHAはこの時点で固定する。現在のb23b3c0はkernel基準であり、新入口のSHAではない。
4. Python候補3.12.3、mpmath1.3.0と実体、SOURCE hash、環境変数・platform・UID、実際の空き資源を記録する。NumPy/SciPy/Qiskitやengineの数値kernelを呼ぶ必要はない。共有venv/package/設定を変更しない。
5. ユーザーが**metadata-only契約の採択・薄い入口準備と一回の科学実行**を明示許可する。今回の文書作業の許可を代用しない。承認証明書・grant発行・管理者権限は要求しない。

出力先の候補は `artifacts/scv_signed_support_h4_metadata/2026-11-07/run01/`。次回の実行設定に最終pathを固定し、初回にfresh directoryを排他的に確保する。既存directory・STOP・同名結果があれば何も上書きせずSTOPする。旧attemptを再開しない。

## 4. 有限call範囲：正常一回の場合

| 操作 | 上限/予定回数 | 内訳と注意 |
|---|---:|---|
| metadata-only入口の科学attempt | 1 | retry/resumeなし。途中失敗も消費したattemptとして保全 |
| LAW/catalogのファイル読込 | 各1 | 同じbytesを照合とbinderへ渡す。元データを再取得してやり直さない |
| 直接入力のSHA256照合 | 固定2入力 | 外側とscalar binderの重複digest検査は可。再取得・再計算の科学retryではない |
| signed metadata_from_bytes | 1 | scalar binder1回を内部で呼ぶ |
| ConditionalSampler.from_record | 1 | CategoricalLaw2個、OccurrenceLaw1個、sampler1個、既存ControlBank3個 |
| scalar.Metadata / signed.Metadata構築 | 各1 | Componentは保存catalogの各行から一度構築。全path列挙なし |
| JSON解析 | LAW計2 / catalog計2 | binder内部はLAW1/catalog2。外側のsaved五層照合に元LAW bytesを追加1回解析する。ファイル読込の再実行ではない |
| cells('B1-OL') | 計2 | scalar.Metadata内部の全5正層検査1回＋外側で保存用/照合用に一括取得1回。結果を再利用 |
| pattern_mass | 計10 | 上の2回×5層。有限controlの既存inclusion-exclusionであり、新CDFやevent列挙ではない |
| SignedControl構築・signed moment算出 | **1** | k=1,2のlabel有限和を既存__post_init__で一度だけ計算 |
| 全5層ν/m₂/Σ・snapshot取得/保存 | 1組 | 計算済み属性を参照するだけで、別のmoment実装を呼ばない |
| h1非定数性判定 | 1 | exact Σ1の符号による一度の分類 |
| state solve / eigh / spectral認証 | 0 | N2再実行なし |
| engine初期化 / state作用 / Born | 0 | 参照identityのみ |
| sampling / measurement / 科学PRNG / seed | 0 | ConditionalSamplerはlaw表現・検査として構築するだけ。drawやStreamsを起動しない |
| gate/circuit build / compile / price | 0 | selected policyを再構成しない |
| OLS / train / paired / cal / 十分N / 新RZ比較 | 0 | 後続の別契約 |
| 科学retry | 0 | 修正や別seedで救済しない |

`pattern_mass`は固定4-control patternの有限和で、1回の5層検査は4+8+16+8+4の40個のjoint_mean呼出しを含む。正常経路2回では計80。これはSOURCEを静的に展開した内部呼出し数で、H4の新momentや物理的性質を評価した数値ではない。digest/fingerprintやFraction構築は内部で複数回起こるため、それらを「全内部演算一回」と記述しない。

外側整合検査・保存・hash読戻しは、計算済み結果の検査とI/Oである。SignedControlを再構築して検証することは禁止。失敗時のcompleted回数は上表より少なくなり得る。CALL_START/RETURN/EXCEPTIONを残し、未RETURNを正常完了へ換算しない。

## 5. 入力schemaと前方検査

固定入力は[bindingのcommit/path/byte hash](SIGNED_SUPPORT_H4_METADATA_BINDING.md)に完全一致すること。通常のJSONとして読むだけで、整形・subset化・新catalog生成はしない。

| 検査 | 実装箇所/要求 |
|---|---|
| SOURCE/環境/2入力bytes | 新入口の実入力前照合。異なるSHAへ自動追従しない |
| law schema | 既存from_record：M=2^53、products_then_rotation、order(0,2)、exact rational非負正規化質量、grid整数性 |
| labelとcatalog | 既存完全label集合検査に加え、外側でlawのlabel順とcatalog順を比較。未知/重複/欠落/順序不整合はSTOP |
| H/preparation | scalar binderの保存field比較。partition/tail/policyは固定N1 bytes/provenanceへ結ぶ |
| a,bと5層 | 既存positive support、a≠b、全5正層。元LAWのfive_patterns/five_strata[].exactと外側で照合、Σ_h p_h=1 |
| sign/support | 同一rowの整数±1・canonicalなI/Z/ZZ support・system domain。既存Component/Metadata検査と矛盾するschemaを受け入れない |
| feature/moments | [-1,1]、ν∈[-1,1]、m₂∈[0,1]、Σ=m₂−ν²≥0をexactに確認。h0のΣ=0、h2–4のν/m₂/Σ一致も確認 |

float_preview、係数比、CDF hexを使ってQを作り直さない。LAW内のexact文字列を正本にする。各labelからfeatureを構成する処理はSignedControl内部で一度のみ。全event/pathの列挙、正labelの除去や無断再正規化、新control選択は行わない。

## 6. 出力schema・保存順序

以下は将来の出力仕様であり、現在これらの実H4値を作成していない。exact値の正本はFractionから得るcanonical文字列（整数なら`0`等、非整数なら既約`numerator/denominator`、正分母）。小数表示は省略でき、保存してもpreviewとしてのみ扱う。

| ファイル | 必須内容 |
|---|---|
| SOURCE.json | 実行SOURCE commit、kernel commit b23b3c0、実行入口と関連SOURCE hash、Python/依存環境、固定科学条件、N1/N2/engine参照identity、feature version、軸非依存scope、attempt/output identity |
| INPUT_IDENTITY.json | N1 commit、LAW/catalogのrelative path・byte SHA256・Git blob、H/partition/preparation/tail/policy、a,b、qubits=8、5pattern。実際に参照したbytesのhash |
| METADATA_RESULT.json | signed metadata/control identity、snapshot、全5層のexact p_h/ν_h/m₂_h/Σ_h、検査結果、metadata_structural_eligibilityと理由、h1判定、statistical_effectiveness=UNKNOWN、science_ready=false |
| CALLS.jsonl / CALL_COUNTS.json | 入口・binder・SignedControl・判定等のSTART/RETURN/EXCEPTION、正常/未完了の区別、上記有限callの実数、禁止callは0 |
| REPORT.md / STOP.json | 成否・適格性・UNKNOWN・取得済み結果と未完了理由、retryなし、後続未開始、MANDATORY_STOP |

METADATA_RESULTの型は次のように固定する。これは値を埋めないschema説明であり、placeholderを0や実測値と解釈しない。

| field | 型/意味 |
|---|---|
| schema | 固定version文字列 `scv_signed_support_h4_metadata_result_v1` |
| metadata_identity / control_identity | 計算成功時のSHA256文字列。不成立時はnullと理由 |
| feature_version | 固定の `rotation_coefficient_sign_times_diagonal_support_size_over_2_v1` |
| feature_bounds_exact | `['-1','1']` |
| cells | id='0'…'4'の順、要素は{cell, pattern, mass_exact, mean_exact, second_moment_exact, variance_exact}。exact fieldは上記rational文字列 |
| h1_nonconstant | 真/偽。判定まで到達しなければnull |
| metadata_structural_eligibility | `ELIGIBLE` / `INELIGIBLE_CONSTANT_H1` / `UNKNOWN_ERROR` |
| reason / validation_checks | 人が読める理由と各検査のPASS/FAIL/UNKNOWN |
| statistical_effectiveness | 常に`UNKNOWN`。相関/分散削減/費用改善を未取得とする |
| science_ready / next_stage_started | false / false。metadata適格でも測定へ進まない |
| state_loaded / engine_loaded / old_measurement_data_used | すべてfalse |

保存順は、fresh出力確保→SOURCE/input identity→call開始記録→計算→exact結果→call完了/例外→REPORT/STOP。tmpからのatomic replaceとflush/fsyncを用い、計算済み結果を保全する。読み戻しで保存bytes/hashを検査してよいが、SignedControlの再実行はしない。旧campaignの保存機構から必要なI/O部分だけを再利用し、engineをimportするためにcampaign全体を起動しない。

失敗前に取得できたfieldだけを保存する。moment未取得時に空欄を0で補完しない。process消失等でcall末尾が不明ならUNKNOWNとし、同一attemptを再実行して欠測を埋めない。衝突先・旧STOP・入力ファイルには書き込まない。

## 7. 適格性とmandatory STOP

- **ELIGIBLE**：全identity/schema/law/moment検査がPASSし、exact Σ1>0。構造的適格性だけを記録してSTOP。
- **INELIGIBLE_CONSTANT_H1**：整合検査はPASS、exact Σ1=0。h1限定の次段階へ進まずSTOP。別feature/別層を試さない。
- **UNKNOWN_ERROR**：SOURCE/input/schema/component/正層/moment不整合、range外、負variance、call範囲超過、出力衝突、実際の共有環境非干渉不成立等。条件を緩めず取得済み証拠を保全してSTOP。

Σ1>0の判定へ任意のtoleranceや浮動小数のゼロ判定を導入しない。非定数性の存在判定と、有限trainで十分な多様性が得られるかは別である。今回は後者の数値評価、rank推定、shot見積りをしない。

科学attemptは成功・不適格・中断のいずれも一回で終了する。新しいmetadata数値評価、retry、control/層/targetの変更、train/paired/cal/費用評価へ自動進行しない。旧N1/N2/engine/224/752/中断結果・SOURCE・grant/ledger/marker/STOP、元main dirty/untracked、O1/O2高速化SOURCE・成果物、C1凍結を保全する。

CPU/RAM/workerへ恣意的な固定上限は設けず、実際の空き資源と非干渉を確認する。単一の短い逐次依存処理であり、科学worker・高負荷manager・GPU・全体設定変更は不要。共有job/processへ操作しない。

## 8. 採択・承認と後続の分離

次回ユーザーが採択・明示許可する必要があるのは、**固定2入力・既存signed kernel・この一回のmetadata-only scope、薄い入口の準備/人工確認/公開固定、出力とSTOP**である。実行SOURCE SHAと最終fresh outputは準備時に固定し、公開照合したものを使う。

metadata-onlyにはshot数・seed/stream・学習/評価配分・λ・数値range guard・価格条件は不要であり、今回それらを設定しない。構造的適格性の取得後、fresh train/paired評価を希望する場合だけ、それらと科学call上限・STOPを別契約へ具体化し、改めて明示許可を得る。旧224条件を継承しない。

正式共通指示§8.1のとおり、この承認は会話で対象と範囲が明示されれば成立する。証明書・grant・承認用ファイル・不要な管理システムを新設しない。今回は本契約草案と関連3文書の公開照合後にmandatory STOPし、次回のSOURCE実装も開始しない。
