# N3-B1a：43 key取得前後に必要な最小後続scope

2026-11-08 JST。[REPORT](REPORT.md)。**提案であり未実行。N3-B1aはmandatory STOP。** 科学計算・実装の許可を新たに作ったものではない。

## 1. 取得前に解決できること

**GPT／ユーザー**が、[SOURCE条件C_phiと記号証明](PHASE_ERROR_BOUND.md)を固定経路に適用するか判断する。既存emitterのW loopを無条件に不適合とする必要はない。固定binary定数と8状態閉包により、W数を推測しない上界を構成できる。

採用するSOURCE・runtimeで、CPython pi／Rust PIの同一binary値、通常IEEE条件、numeric global_phase、named gate保持、compose／inverse／QPYの有限経路が成立することを対応させる必要がある。必要な最小適合検査を、実43 wordの取得前に既存recordや人工的なW residue例で実施する選択肢がある。

実行経路はまず**1つのcomplete wordを既存lower_nativeでemit**する形を優先し、不要なchunk circuitの再構成を増やさない。integer W ledgerは必要性がある場合の限定差分とし、旧SOURCEを直接改訂しない。

## 2. N3-B1bを指示する場合の有限入力

基準は本N3-B1aの固定成果と[N3-B0 FIXED_PARAMETERS](../../../artifacts/pr_reference_n3b0_numeric/2026-11-08/FIXED_PARAMETERS.json)。b=36、j=-36…6、43 literal positive key、基本Frobenius要求10^-13、既存のstrict backend／guard／actual adjoint規約を変えない。

保存coverageは0/43であり、基本keyは全件が新規取得対象となる。ただし取得開始前に、emitter経路と証明条件を固定する。別の成功しやすい角度、内部precision、bを選び直さない。C0はCT-4Bの採用wordを優先し、D／CSWAP／scalar／中央価格を保持する。

最低限必要な実装は、固定43基本recordを読む辞書、exact binary丸めと固定bit順のcomposite provider、complete-word actual adjoint、量子化／guard／emitter証明の対応である。word取得・この実装・数値correctness検証は現在いずれも未承認／未開始である。

## 3. 明示承認が必要なcall範囲

後続を実施するなら、次を具体的に固定して承認する。

1. 既存SOURCEと必要最小限の新provider／必要時のledger adapter。旧emitterを使うか、per-word W処理だけの新差分を使うか。
2. 固定43 keyのstrict合成／guard。既存wordを再利用する機会は同じliteral・実保存上界で判定し、missing対象やprecisionを後から拡張しない。
3. 実recordのexact matrix、sequence hash、global W、T/Tdag metadata、strict Frobenius上界、正負actual adjointの確認。
4. 既存word／取得wordを用いた必要最小限の局所emitter検証と、宣言した固定順のcomposite検証。zero、wrap、負角、grid端点／最大chunk数の人工検査を実supportの到達観測と混同しない。
5. 必要なQPY往復とjoint相対位相の確認。wordへ新controlを付けず、元のnative／jointへの埋込みを比較する。Qiskit Operatorは診断として使い、exact行列／scalar境界の保証を代用させない。

SOURCE定数の有限確認、strict guard、回路materialization、Operator／simulation等の数値検証は**今回行っていない**。native build/compileはSOURCE適合を保ったpost-compiler RZ置換で不要なら追加しない。必要なら対象を事前固定する。

T/Tdagは取得recordの実出力metadataに対応させる。新T和／cache cap／N1比較まで数値会計する場合は、その範囲も明示する。今回の静的調査から価格評価の許可を推定しない。全support期待T、199-shot積、18round総T、T-depthは既定scopeへ含めない。

## 4. 取得後でなければ確定しない証拠

全基本wordの実sequenceと終了、実guard／word hash、W指数、実T/Tdag、chunk依存閉包、composite certificateと実providerの一致が必要。SOURCEのphase閉包証明は、pygridsynthの全入力停止や43取得の成功を保証しない。未取得metadataへ推測値を入れない。

他のC0 wordも含むwhole接続では、保存guardとemitter増分を合わせた局所許容、初期global phase、reset／measurement、compiler／合成diskの同単位会計を確認する。N2のcommon frame／b2の欠損は別問題として保全し、今回の結果で解消したことにしない。

## 5. STOP条件

固定scope完了、word／phase／順序／actual adjointの本質的不一致、C_phi不成立、residual条件不成立、固定precision／key／compiler変更、追加の対象外callが必要なら、結果を保存してmandatory STOPする。

8状態条件やphase経路が使えない場合は、一般のW数／操作数に依存する境界又は最小差分へ判断を戻す。証拠がないのにe_emit=0へしたり、余裕を得るためにN3-B0を再選択したりしない。取得が完了しない場合もSK／N2／別backendへ自動移行しない。

次担当は**GPT／ユーザー**。既存emitter再利用の条件と残る適合確認を評価し、必要なN3-B1b scopeを判断する。**Codex**はその明示指示の範囲内だけで実装・取得・検証を行い、完了時STOPする。
