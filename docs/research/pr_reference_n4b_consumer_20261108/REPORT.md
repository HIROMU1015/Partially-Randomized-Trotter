# 新A N4-B：共通word policy・consumer局所correctness

2026-11-08 JST、SERVER 2。基準 **547532cbdc51de0721d333bcc8dfc9dbfab1f76c**、branch `codex/pr-reference-n4b-consumer-20261108`。

**固定85 positive literal keyすべてを既存43-word辞書で資格し、共通policyと薄いconsumerを実装した。指定範囲を完了しmandatory STOP。** 新規Clifford+T合成、strict guard、native PF build/compile、RPE再会計は行っていない。

判定は **ROLE_WORD_COVERAGE_QUALIFIED／COMMON_POLICY_IMPLEMENTED／LOCAL_CORRECTNESS_PASS／CONSUMER_PHASE_QUALIFIED**。最後の判定は、固定C_phi/runtime下の各RZ置換と別記したshell位相上界についての資格である。Compilerへの移送は **CONDITIONAL_ON_COMPILER_CONTEXT**。**FULL_SUPPORT_CT_RPE_UNQUALIFIED** を維持する。

## 固定条件

Linear H4・1.00 Å・STO-3G・DF12、元H厳密ground供給・状態準備費除外。Partial S2／LD3／h=.02／finite RTE K2、ordinary diag(I,V)、canonical independent finite-bit law、m7/q128/r1・保存73 bits。Qiskit1.3.0/compiler A、最終native RZ strict Frobenius10^-11。b36・j=-36…6・43基本word・基本10^-13、固定C0全299採用wordを維持した。

B′4-CX controlled-RZZ、Reference F/G・actual SOURCE inverse・vacuum P・2 register・8-CX CSWAP、D融合・basis共有・scalar/event・X/Yも不変更。保存199 shots/axis、alpha1/720、phase cap.4、failure.05、energy1.5936e-4 Haは引用であり、新たな成功判定ではない。Exact W、up_to_phase=False、全文字列のactual adjointを保持し、phase fit／π snap／無条件2π key同一視／後置controlを用いない。

## 結果と証拠

| 対象 | 結果 | 範囲・留保 |
|---|---|---|
| 共通RTE frame 84 key＋B′ b2 1 key | 85/85 strict certificate取得 | 保存numeric superset。canonical全trajectory列挙ではない |
| 固定C0優先 | 全299 sequenceを保持 | 旧guard・価格は不変更、再合成・再guardなし |
| 同じpositive literalのword | arm／role非依存、正負で同じpositive sequence | 負側のcomplete actual adjointを旧lower_nativeで1回だけemit |
| 局所検証 | 34ケースPASS | 指定2 key±、±0、両armの保存basis／basis_inverse／P template、phase／reset／clbit境界 |
| 保存whole入力 | 8件のQPY hash・phase・全literal多重度・native gate集合が旧metadataと一致 | 既存QPY読込のみ。巨大whole CT回路の展開・dense Operator・状態simulationなし |
| Whole consumer接続 | 上記8件の固定局所投影でPASS | 実prefix reset・実X/Y末尾・measure先・qubit/clbitとphaseを保持。full bodyの実行試験ではない |
| 全support CT RPE | 未資格 | 新たなjoint誤差／raw disk／199-shot数値再会計を実行していない |

[ROLE_WORD_CERTIFICATES](../../../artifacts/pr_reference_n4b_consumer/2026-11-08/ROLE_WORD_CERTIFICATES.json)は85個の実positive sequence、hash、両符号のcertificate、role、W、T/Tdaggerを保存する。量子化＋保存基本guard＋既存emitterの合計は全件10^-11未満。最大の保存上界は約5.6221e-12。これは局所certificateの値で、全回路・信号・RPE成功の値ではない。

特にframe `0x1.08ba32d420f2cp+2` とB′ b2 `0x1.921fb54442d18p+2` を解消した。b2を0／理想2πへ置換していない。b2のcanonical到達性は未確定のままだが、安全なsupersetへのword準備には支障がない。

[COMMON_POLICY_VERIFICATION](../../../artifacts/pr_reference_n4b_consumer/2026-11-08/COMMON_POLICY_VERIFICATION.json)にC0保持と85 keyの同一選択を保存した。既存[DictionaryProvider](../../../src/trottertracks/pr_reference_dyadic_dictionary.py)を変更せず使用し、word選択へarm／role分岐を追加していない。旧43基本word／C0全299のarchive・manifest・metadata検証を再利用し、guard計算は起動していない。

## Consumerと局所検証の意味

新しい[consumer](../../../src/trottertracks/pr_reference_common_word_consumer.py)は、native回路をunitary segmentとreset／measure／barrierへ分け、segmentだけを変更していないlower_nativeへ渡す。Qubit/clbitとregister対応、gate順、非unitary位置の意味、X/Y wrapperを維持し、全joinはinplace=True。unsupported gate／条件付き命令／free parameter／域外角は拒否する。SOURCEを再buildせず、compiler後のbinary64 literalを読む。

[LOCAL_CORRECTNESS](../../../artifacts/pr_reference_n4b_consumer/2026-11-08/LOCAL_CORRECTNESS.json)は実emissionの全文命令順とwire対応、整数W residueから独立に作ったbinary phase参照との一致を検査する。指定2 keyではexact cyclotomic actual-adjoint恒等式とQiskit Operator診断も実施した。Operatorの誤差toleranceはstrict certificateの代用にしていない。

保存basis／basis_inverseは両armとも指定archive memberの実loweringを使用した。8-qubit native gate productと数学的wordによるcompressed置換の局所Frobenius診断は最大約2.073e-10。joint Frobenius保証という意味ではなく、実jointの保証はspectral embedding／telescopingで扱う。保存local recordで欠けているSOURCE global phaseは0で補完せず、同じ未知prefactorを保ったgate product置換だけを検査した。

[SAVED_WHOLE_INTEGRATION](../../../artifacts/pr_reference_n4b_consumer/2026-11-08/SAVED_WHOLE_INTEGRATION.json)ではB′9 qubit／Reference17 qubit、各1 clbitを維持した。Referenceの実resetは先頭8命令でq8…15、測定はq16→c0。B′の測定はq8→c0。固定投影は「元nativeの先頭4＋末尾4命令＋全非unitary」で、全resetと実readout末尾を残す。別の2-qubit技術fixtureで途中reset、barrier、clbit入替、複数segmentも確認した。

保存QPYのglobal phaseは全8件で旧hexとbit一致し、[0,tau)内のため読込setterによる変化がないことを確認した。新しいQPY出力／往復や任意QPY経路の資格ではない。詳細は[CONSUMER_INTEGRATION](CONSUMER_INTEGRATION.md)と[PHASE_ACCOUNTING](PHASE_ACCOUNTING.md)。

## 位相・Compiler・価格の解釈

各物理RZ位置に既存certificateを1回適用する。Segment内へのword joinはそのcertificateのemitter項で覆い、consumerが別に行うouter初期phase設定1回と非empty segment→outer join S回には、既存C_phiの記号上界 `(16+64S)2^-53` を別項として与えた。これは局所1-qubit scalarのFrobenius上界をspectralへ移す規約で、joint Frobeniusへdimension係数なしに移すものではない。Whole全体に局所emitter上界を1回だけ適用したり、frame誤差を取消したりしない。

固定CT-5B条件C、同値則／compiler pass／post-unroll template、Dのevent非依存性、共通frame pair multiset、actual SOURCE inverseを前提にSOURCE閉包を移送する。選んだ保存contextと全8 native入力ではliteralの不整合は見つからなかった。これは任意contextの新compile証明ではなく、条件C付き資格である。

N1の **K=160222、d0=106、d2=114、B_P=6072<173790** は保存値の引用。新frameを旧C0価格や0として扱わず、条件C下の同一literal multiset＋共通policyなら任意の共通word価格の差が消えるという既存の記号論を維持する。**価格取消しは誤差取消しではない。** 新しい全support価格／総T／T-depth／18round改善率を計算していない。

## 再現・保全・終了

[検証SOURCE](../../../scripts/validate_pr_reference_n4b_consumer.py)、[SOURCE／入力](../../../artifacts/pr_reference_n4b_consumer/2026-11-08/SOURCE_INPUTS.json)、[環境](../../../artifacts/pr_reference_n4b_consumer/2026-11-08/ENVIRONMENT.json)、[CHECKS](../../../artifacts/pr_reference_n4b_consumer/2026-11-08/CHECKS.json)を保存した。Runtime確認とlower_native／exact word matrixはN3-B1b-0 helper、record validatorとcompressed kernelは旧SOURCEの関数ASTを再利用した。合成・guardや無関係な科学runnerをimport／実行する経路は追加していない。

固定runtimeはPython3.12.3／Qiskit1.3.0／pygridsynth2.0.0／numpy1.26.4／mpmath1.3.0。64 CPU affinityと空きRAMを確認し、重い保存8入力読込だけを8 workerで並列化した。局所行列は既存compressed kernelを使い、全wholeの巨大行列は不要だった。共有設定・他ユーザー・他Trackに変更なし。著者Codexによる技術検証であり、独立科学レビューではない。

再現入口は `PYTHONDONTWRITEBYTECODE=1 <既存固定venv>/bin/python scripts/validate_pr_reference_n4b_consumer.py`。実venv pathはENVIRONMENTに保存。過去SOURCE／raw／結果／grant／ledger／marker／STOPは不変更。正式共通運用指示2026-10-25、親／worktree AGENTSとPROJECT_MAPを適用した。

**[MANDATORY_STOP](../../../artifacts/pr_reference_n4b_consumer/2026-11-08/STOP.json)。** 次担当は **GPT／ユーザー**：[NEXT_EXECUTION_SCOPE](NEXT_EXECUTION_SCOPE.md)の固定m7/q128/r1の誤差再会計を判断・別途承認する。CodexはRPE再会計、新規合成、N2／SK、N3-B1、18round総T等へ自動進行しない。
