# A0 normalization受入準備とC1接続（2026-11-11）

既存B branch上に、固定A0 24候補の**受動的な診断受入adapter**を追加した。既存`build_menu/request_context`・`SavedArtifacts`と公開済み候補整合表を再利用し、N/C provider・総費用会計器は変更していない。新8＋既存39＝**47技術test PASS**。実A0結果は消費しておらず、新normalization/signal/N/C/再解析/予測/fit/sampling/compile/実G/R最適化は0。

## 固定参照と実装

- A0準備：[`6a75ebf96` REPORT](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/6a75ebf967aaae2698ce6d295bf6fbec9d5e9929/docs/research/pr_r_normalization_phase0_preparation_20261111/REPORT.md)、同commitの`PLAN.json`/`HANDOFF.md`。producer実装SOURCEは`6ed8c6873311085bb3b39799b268105b0bc97211`を読み、出力schemaを確認した。
- B基準：[`328cb26fa`候補整合](../r_screening_price_alignment_20261111/REPORT.md)と同commitの`ALIGNMENT.json`/`INTEGRATED_PLAN.json`。過去の計画statusを今回のAの実行状況へ書き換えていない。
- C1：[`8457c4435`固定計画](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/8457c443554af5fadaa3ba36f97f6316ca3c5050/docs/research/r_variable_compiled_price_design_20261111/MINIMAL_SCIENTIFIC_PLAN.md)。8セル/最大80 wrapperのscopeを変更しない。
- 新API：[`rpe_normalization_intake.py`](../../../src/trottertracks/rpe_normalization_intake.py)。新CLI：[`prepare_rpe_normalization_intake.py`](../../../scripts/prepare_rpe_normalization_intake.py)。[受入仕様](HANDOFF.md)、[受入準備metadata](INTAKE.json)、[技術test記録](TESTS.txt)、[静的確認](CHECKS.txt)。

## 受入動作と資格

`A0Intake`は固定PLANから24件を復元し、既存Bのmetadata APIでcandidate ID、m/T/qRPE/qPF/δ/R、formula/prefix/K、signed weights・piece allocation・event/scalar規則、H/state/snapshot、configuration、compiler/control/wrapper/P0、lawを照合する。H4 δ=.1、H6 δ=1.6、元候補の範囲を保持する。共通identityは一致するが、Aの法則宣言とBのN provider用lawタグは異なるため、元requestとB側law_bindingを併記し、aliasやN recordへの変換はしない。

公開HANDOFFの`scientific_source`が指すGit SOURCE9件を固定PLANの科学SOURCE hashと照合する。normalization内のmethodと保存tail SOURCEも照合する。これは科学的来歴の確認であり、許可証明や実行中入力の再検査ではない。生の配列・tail値の再計算・Aのlive pathへのアクセスはない。

HANDOFFだけでも受け取れるが、診断共有を有効にするには、同じ公開commitの`result.json`を併せて読み、status/SOURCE/recordsの一致、固定source_files/input_hashes、`source_preserved=true`、`input_preserved=true`を確認する。さらにroot statusが`PHASE0_NORMALIZATION_RECORDED`、24件の数値記録が揃い、HANDOFFの`normalization_evidence_valid=true`であることが必要。AのHANDOFFには保全booleanがないため、companion不足は未資格のまま保持する。

出力は全24件を元順で保持し、未着は`record=null`/`NOT_RECEIVED`、明示未評価は`normalization_status=NOT_EVALUATED`とする。正常記録、部分完了、STOP、全件存在する数値未資格を区別し、root status/errors/保全状態を原文のまま残す。STOPや精度未資格を科学的FAILにしない。重複/未知ID、candidate・SOURCE・law不一致、N/C/G/PASS/推奨値の混入は拒否する。

`log_B`、2log B、attenuation、shot inflation、1 ULP感度、小τ診断は**再計算せずdeep copy**する。overflowのnull、underflowの0と専用status、数値未解決、欠測を保存する。表示用指数値がoverflowしても有限logと元statusがあれば記録を保持できる。欠測log/感度、未知表示statusなどは診断共有を資格しない。JSONにNaN/Infinityを新たに書かない。1 ULP感度を厳密誤差区間に昇格せず、nullを0へ補完しない。

`normalization_evidence_usable`/`diagnostic_share_eligible`は上記の**診断消費条件だけ**を表す。十分N、actual実装、価格、全law、全RPE成功、科学的PASS/FAIL、推奨Rの資格ではない。出力の`N_records=[]`/`C_records=[]`、N/C/Gと推奨値はnull、全RPEとR採択はfalse。`run_missing`や費用・R選択entryを呼ばない。巨大Nを読んで浮動小数点へ変換する経路も追加していない。

## C1との対応

既存Bの固定8セルをそのまま参照する。H4 qPF=8/16・R=4/8、H6 qPF=1/2・R=16/32。H4 R8/q8とR8/q16は`EXPLICIT_C_CALIBRATION_VALIDATION_ONLY`を保全し、A0へ追加しない。

| A0/C1重複候補 | 条件 | C1 role |
|---|---|---|
| `2d2af6ca830dff14a695` | H6 m4、T=1.6、qPF=1、δ=1.6、R16、prefix7/K2 | 保存校正 |
| `4269eafc8c0ef8922c23` | H6 m4、T=1.6、qPF=1、δ=1.6、R32、prefix7/K2 | 新校正 |

`for_c1`はこの2 IDだけを対象とし、同じnormalization SOURCEを明示し、full requestと元A law_bindingが一致する場合に診断を返す。SOURCE/law差を自動同一視しない。STOP等では診断recordを残して共有資格falseを返す。他6セルへ補完しない。A0 24候補とC1 8セル/最大80 wrapperは独立scopeであり、C1実行の前提として追加normalizationやNを要求しない。

C1小qからA0の長時間m12/qPF4096・256へ価格資格を移送しない。固定Rモデル、finite cohort、全law期待価格、希少event、長q外挿、cross-R共分散・共有校正誤差/CI、全RPE成功は既存の未資格を維持する。normalizationを価格へ再乗算せず、shot inflationを十分NやGへ変換しない。

## テストと残作業

新8件は、固定保存PLAN24の整合、人工24件の最終受入API、12件/0件の欠測・途中STOP、明示未評価、overflow/underflow/null/精度未資格、candidate/H/state/snapshot/法則/tail/SOURCE不一致、companion/保全不足、C1の厳密2点接続、N/C/G混入拒否・数値backend非起動を確認した。定数の人工診断を使い、分子のnormalization式は評価していない。producer自体もimportせず、standalone受入準備でnumpy/scipy/qiskitがimportされないことを確認した。

既存は候補整合7、R候補選択12、provider統合8、PF/PR pipeline12を再実行して全PASS。初回新testでは未存在SOURCE pathのGit例外型が想定と違ったため、明示的なSOURCE不一致ValueErrorへ統一して修正した。数値coreの変更や科学結果の再解析はない。これは実装者による技術確認であり、独立科学レビューではない。

次の最小作業は、**Codex AがA0 HANDOFF.json/result.jsonをcommitして公開し、Codex Bがその公開SHA/pathを指定して本adapterを実行すること**。実行中Aの完了を待つ処理や自動pollは設けていない。SOURCE/schemaが変わった場合は差分を技術確認し、無断で固定条件を緩めない。その後のsignal/phase/radius/N、R依存C、長q/全law/共分散検証、実N×C/R選択は別の明示科学実行範囲に属する。

A/Cのworktree/process/cache/raw/checkpointを読み書きせず、既存SOURCE/input/raw/結果/grant/ledger/marker/KEEP/STOPを保全した。成果をcommit/pushし、remote SHA・主要artifactを照合してmandatory STOPする。新しい科学実行へ自動進行しない。
