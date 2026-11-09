# 新A V6-Q：B1 provider実装・科学検証準備

2026-11-09、SERVER 2。**Q1実装・人工20テストPASS、Q2の保存4構成を特定、Q3静的整理を完了。Q2／Q3科学実行は0件。** 添付指示§2・§8・§12に従い、今回の許可範囲を完了し科学実行前でSTOPする。

[実装と条件付き証明](B1_IMPLEMENTATION_AND_PROOF.md)／[固定4構成の次検証計画](Q2_VALIDATION_PLAN.md)。今回の技術検算は著者自身によるもので、独立科学レビューではない。

## 1. 固定資料・条件

- [V6-P結果 5f672ce6](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/5f672ce6884bd49231366875a9e24a539fcaec7a/docs/research/pr_reference_v6p_results_20261109/REPORT.md)。条件付き数値結果を引用するだけで再実行しない。
- [V6-Q計画 e5d6e388 §3](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e5d6e388ee2543b47ebcc1dfd9e19862d08ac327/docs/research/pr_reference_v6_static_20261108/NEXT_VALIDATION_PLAN.md)、V6-P実行SOURCE a2c54558e1b4cc052aeb835a9e6d3b2796767e98。
- 元科学SOURCE・保存入力は **f2638a7845685a119619382b850441f5d8d33713**。共通運用指示2026-10-25版、AGENTS.md、PROJECT_MAP.mdと既存契約を保持する。前段で読解した不変の研究総括・既存保証を再利用し、旧記録や他Trackを更新しない。

Linear H4／1.00 Å／STO-3G／DF12、元H厳密ground供給・状態準備費除外、Partial S2／LD3／h=.02／finite RTE K2、ordinary diag(I,V)、canonical independent finite-bit law、Qiskit1.3.0／compiler A／rz,sx,x,cx。B′4-CX controlled-RZZ、ReferenceのF/G・actual SOURCE inverse・vacuum P・二つのCSWAP cascade・2 register、D融合／basis共有／scalar／signed event／X/Yを不変更。energy精度1.5936e-4 Ha、failure .05、phase cap .4、αaxis 1/720、保存scheduleを保持する。

## 2. Q1の実装と旧pathとの差

[新module pr_reference_v6q.py](../../../src/trottertracks/pr_reference_v6q.py)だけを追加し、旧pilot／transfer／builder／compilerを変更していない。

| 新関数 | 役割 |
|---|---|
| `exact_binary_phase(f,g)` | 実際に渡された両laneのglobal_phaseとRZ/RZZ実literalをexact二進有理数へ移し、半角・F/G合計をexactに評価。最後に一度だけfloat変換する |
| `_verify_nearest_even` | 返却floatと隣接floatの中点を有理数で比較し、最近接・偶数丸めでなければ拒否。単にruntimeを信頼して小さい観測誤差を保証としない |
| `phase_for_round` | defaultは旧transfer同様の `vacuum_phase(F)+vacuum_phase(G)`。新modeは明示したm17だけ受け付け、他roundへの新mode指定は拒否 |
| `reference_for_round` | 旧`reference_core(F,G,phase)`へ渡す新opt-in callsite。F/G builder、inverse、CSWAP、wire、wrapperを再実装しない |

既存のFraction会計を調査し、標準`Fraction`と`as_integer_ratio`を再利用した。旧naive provider自体は要求する一回丸めを実現せず、V6-P adapterは実phase生成を含まないため、純粋算術の新moduleが必要だった。既存のrunner・RPE estimator・価格providerを複製していない。

float subclassは格納binary64をそのまま取得し、整数はbinary64へexactに移せる場合だけ受け付ける。decimal／symbolic／complex／非有限値、未対応gate・parameter形状、条件付きoperation、lane幅不一致、overflow、非零subnormal exact出力を拒否する。未知の入力を文字列化や近似変換で補完しない。後者のrejectを含め、**本番全supportでproviderが完了することは未資格**である。

## 3. 人工テスト・runtime確認

[test_pr_reference_v6q.py](../../../tests/test_pr_reference_v6q.py)の**20テストPASS**。Qiskit回路を作らず、人工duck-circuitとmock gadgetのみを使った。

大きな正負相殺、0和、正規化済みglobal phase、実binary半角、片側ずつの丸めが残差を失う例、合計一回丸め、wire／gate順序の不変、supported family形状、未対応／非有限値、整数変換の丸め拒否、overflow／subnormal拒否、正負のtie-to-even、誤った最終丸めの検出、全18roundのlegacy default、m17限定opt-in、旧gadgetへの委譲を確認した。

参照和はproviderのFraction walkを呼ばず、独立の整数common-grid方式で計算した。最終丸めは既知の隣接binary64と中点tieのfixtureでも確認。Python3.12.3／GCC13.3.0／x86_64／glibc2.39、radix2・mantissa53・double8byte・rounds1を観測し、標準`Fraction.__float__`が整数true divisionを使うSOURCEもread-only確認した。各返却値のexact中点検査により、仮に変換が要求RNでなければ近似値を返さず停止する。

このPASSは実H4のcorrectness／phase／全support成功を意味しない。旧H4科学値、ground、保存trajectoryをproviderへ渡していない。

## 4. Q2入力の所在

[Q2_INPUTS.json](../../../artifacts/pr_reference_v6q/2026-11-09/preparation/Q2_INPUTS.json)にarchive／memberのhash、seed、SOURCE blobと共通pickleのbyte来歴を固定した。4件とも実在し、UNKNOWNのsampleを作成・置換していない。

| q/r・stratum・j | 確認したarchive member |
|---|---|
| 1/1 nonrare j0 | pilot `prices/q1_rare0_00.json` |
| 1/1 rare j0 | pilot `prices/q1_rare1_00.json` |
| 4/1 nonrare j0 | transfer `raw/q4_r1_rare0_0.json` |
| 4/1 rare j0 | transfer `raw/q4_r1_rare1_0.json` |

確認はtar／JSON metadataのread-only照合であり、event再生成・phase評価・価格再解析ではない。共通pickleはbyte hashを既存INPUTS.jsonと照合しただけで、unpickleしていない。

## 5. Q3の成立条件と未解決事項

受け付けた入力に対してexact変換・半角・和と最終RNを実装し、返却値のRNをexactに検査する。|S_F|,|S_G|≤Mが適用され、Sが0または許容normal範囲なら、|RN(S_F+S_G)−(S_F+S_G)|≤2uM。旧normalizationを保持すればV6-Pの式へ条件付きで接続する。

残る重要項目は、実Qiskitのliteral型／gate形状／copy・compose・Pへの転送、全finite-bit supportでの受入れ・normal出力範囲、保存Mとnormalizationの同じSOURCEへの適用、呼出側が実m17 scheduleをopt-inへ正しく結ぶこと。有限4構成がPASSしても全support資格にはならない。新modeを全18roundへ適用する実装はしていない。

価格不変性は[既存STRUCTURAL_APPENDIX](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2638a7845685a119619382b850441f5d8d33713/docs/research/pr_reference_bprime_20261107/STRUCTURAL_APPENDIX.md)のsymbolic P→1 RZ、固定context、RZを取消しないpass構成を再利用できる。新moduleは同じ旧gadgetへfloat角だけを渡すため、**F/G・context・gate型等の前提を保つ場合**の構造的価格不変補題を整理した。局所gate数だけで未確認consumer全体を資格せず、新compileは実施していない。

V6-Pの630→626と総RZ差は引き続き条件付き引用であり、今回新たに再計算・昇格していない。classicalな長round走査／Fraction費用、m17 CT、domain／emitter／word保証、総Tの課題も未解決。

## 6. 保全とSTOP

[SOURCE](../../../artifacts/pr_reference_v6q/2026-11-09/preparation/SOURCE.json)／[技術検算](../../../artifacts/pr_reference_v6q/2026-11-09/preparation/TECHNICAL_CHECKS.json)／[準備STOP](../../../artifacts/pr_reference_v6q/2026-11-09/preparation/STOP.json)。変更は専用branchの新module・人工test・3文書・準備証跡4ファイルだけで、旧SOURCE／raw／結果／grant／ledger／marker／STOP・科学契約は不変更。

利用可能資源を観測し、affinity64 CPU、MemAvailable 515737768 kB、CPU／memory PSI avg10=0.00。小さい人工算術とmetadata確認を重複並列化せず、独自資源上限や共有設定／他jobへの変更を加えていない。AI subagentは使用していない。

**commit/push・remote SHA／内容照合後mandatory STOP。** 次担当はGPT／ユーザー：固定4構成に対するQ2の科学build・phase・state simulationを許可するか、全support受入れ証明の不足を補うQ3範囲をどう固定するか判断する。添付指示§2・§8が別途承認を求めており、今回それらを開始していない。
