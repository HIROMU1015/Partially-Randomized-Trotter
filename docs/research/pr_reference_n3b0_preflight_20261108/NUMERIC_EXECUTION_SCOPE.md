# N3-B0：固定する限定数値実行scope

2026-11-08 JST。基準 **`79dfa01e6796d1ecab50e30694f5b3f73b95f4ab`**。[PREFLIGHT](PREFLIGHT.md)、[選択規則](PARAMETER_SELECTION_RULES.md)。本書は今回新規作成したpreflight成果物である。**preflight完了時のmandatory STOPを維持し、本turnでは以下を実行しない。**

## 1. 固定条件と入力

H4・1 Å・STO-3G・DF12、元H ground供給／準備費除外、Partial S2・LD3・h=.02・finite RTE K2、ordinary diag(I,V)、canonical independent finite-bit law、m7/q128/r1/73 bits、最終native RZ epsilon=10^-11を固定する。Qiskit1.3.0/compiler A/rz,sx,x,cx/全結合、B′4-CX controlled-RZZ、Reference F/G／actual inverse／vacuum P／2 register／8-CX CSWAP、D融合・basis共有・scalar/event・X/Yも不変更。

保存199 shot/axis、energy1.5936e-4 Ha、failure.05、phase.4、alpha1/720を維持する。W、相対位相、up_to_phase=False、負角の連結word全体adjointを必須とする。

科学入力は[INPUT_REFERENCES.json](../../../artifacts/pr_reference_n3b0_preflight/2026-11-08/INPUT_REFERENCES.json)の基準commit/path/hashに固定する。主な保存入力はCT-0 PHASE_SUPPORTのm7/L、N1 RESULTSの保存閾値、CT-4Bの採用inventoryとstrict archive、旧C2-B・CT-1の保存strict archiveおよび各BUNDLES。新しい科学条件・合成record・cacheは追加しない。

数値runtimeはCT-4B保存ENVIRONMENTと同じ既存project専用`/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/track-c-h4-parallel-pilots-20261106/.venv/bin/python`（Python3.12.3、mpmath1.3.0）を読取再利用する。分布metadataの確認だけをpreflightで行った。system pythonにはmpmathがないため、黙って置換・共有installしない。compiler／Qiskitは実行せず、旧runtime／SOURCE／結果を変更しない。

## 2. 実行する計算の閉じた範囲

1. 既存SOURCE／入力のhash・schema・m7条件と保存record provenanceを読取検査する。科学module全体や旧runnerを起動しない。
2. 192-bitのsqrt2閉区間を1回作り、端点をFractionへ直接移す。端点の二乗による包含を検算する。既存区間helperとstrict_guard／有限bit helperの端点変換規約を再利用し、strict_guard自体は呼ばない。
3. b=0…1074について、整数N_b／J_b／M_b、Q_bの有理区間、成立判定、最も粗い十進要求p_bを選択規則どおり評価する。tuple(p_b,M_b,b)で主仕様を一意選択する。これは固定された有限数学判定であり、T価格・backend成功に応じた探索を行わない。
4. 主仕様だけのpositive basic key集合を作る。`2^j`のbinary64表現を整数比と照合し、0／Infへの丸めを拒否する。負角は実adjoint使用を指定し、負角wordを生成しない。候補全体のkey catalogueは作らない。
5. 主仕様の一様基本要求g_j、量子化区間、emitterへ要求する残余区間を厳密Fractionで保存する。実e_emitは未測定／未資格とし、残余を満たす条件を明記する。
6. 固定した保存poolのみで基本keyのcoverageを照合する。保存word本体／hash／W／strict metadataと実上界を確認し、適合候補・予算不適合・method未資格・指数域外・missingを区別する。保存T／Tdaggerの文字数検査を行う場合もmetadata整合の確認に留め、T価格和／最大値を計算しない。
7. 規模、再利用候補、missing、予定内部precision、N3-B1で別途取得が必要な有限対象を報告する。価格未取得部分はnull／UNQUALIFIEDとし、0や経験則で埋めない。

数値化するのはパラメータ・誤差要求・keyと保存coverageまで。丸め／分解／連結のword policy adapter、dictionary generator、新しいemitterは実装しない。必要なN3-B0会計用の小さいSOURCEは、上記の保存値処理だけに限定する。

## 3. 必要な検算

- Lを保存binary64から正確な有理数へ移したこと、sqrt2端点の包含、Q_bと残余の符号を確認する。
- N_bのfloorを整数比較で、J_bをbit lengthで検算し、`M_b=J_b+b+1`、必要全指数、最大基本word使用数を対応させる。
- p_bの成立と、1段粗い要求の不成立を固定Q_b上端に対して検算する。主仕様が固定tuple順で最初の成立候補であることを確認する。
- 人工的な正負・zero・最近接tieと、保存Lを使う両端の整数丸め検査を行う場合、実trajectoryの到達観測やsimulationと扱わない。選んだ基本角はすべて厳密に2^jへ一致することを確認する。
- 主仕様の`Q_up + M*g_alloc + R_emit_lo = epsilon`を厳密検算し、**e_emit < R_emit_lo**という未資格の実装条件を残す。未知e_emitを0へ代入しない。
- 同じ保存recordが複数archiveにある場合は科学的内容を対応させる。非本質的なJSON整形差を不一致としないが、word／phase／specの食違いはPASSにしない。
- 小数表示は外向き区間の補助のみ。判定と保存正本は整数／Fractionとする。

## 4. 出力と資格

freshな独立領域に、REPORT、FIXED_PARAMETERS、全bの判定表、BASIC_KEY_COVERAGE、ERROR_BUDGET、PRICE_FEASIBILITY、SOURCE・INPUTS・ENVIRONMENT、検算とSTOPを保存する。元のpreflightと旧結果は変更しない。

主仕様を固定できればPARAMETER_SCOPE_FIXED、保存wordが部分適合ならEXISTING_WORD_COVERAGE_PARTIAL、missingがあればDICTIONARY_ACQUISITION_REQUIREDとする。再利用候補が揃っても、実policy／emitter／全support CT RPEの資格は別。emitter条件未解決はUNQUALIFIEDとして残す。

N1のpointwise閾値173790と期待閾値の厳密Fractionは**保存値の引用のみ**。`max(B_dict,B_cache)<173790`の条件を記述するが、このscopeでは実T和・cache最大値・期待vacuum価格を新しく数値評価しない。PRICE_BOUND_CONDITIONALを維持する。期待分布と価格には別scopeの証拠が必要。

## 5. 実行しないこと

新規Clifford+T合成／word取得／strict guard、連結word生成／回路emission検証、native build/compile、sampling/simulation、真空位相分布／期待T／価格極値、precision・shot・alpha・fusionの性能最適化、N2、CT-3B/SK、N3-B1実装、199-shot積／18round総T／T-depthへ進まない。実wordがない箇所へ新価格を推測しない。

## 6. STOPと既存承認の扱い

本turnはpreflightを公開した時点でmandatory STOP。数値実行の許可を受領済みでも、これをpreflight後の自動実行の根拠にはしない。承認証明書・grant・追加の承認認証は不要。

後続で本scopeに従う数値実行を再開する場合、許可の対象はb・内部誤差要求・有限key・保存coverageという範囲で整合している。**actual emitter資格、word policy実装、新規合成、価格取得はその許可へ含めない。** 新しい研究上の選択が必要なら、固定規則を無断改訂して実行しない。

数値実行は、指定範囲完了、成立候補なし、key表現不成立、SOURCE／record／phaseの本質的不一致、科学条件変更、対象外callが必要な時点でmandatory STOP。coverageや価格が不利でもb／precision／poolを事後拡張しない。未資格のemitterを補うN3-B1実装へ自動進行しない。
