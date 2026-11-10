# A/C成果の受入れ・次の実行scope

[REPORT](REPORT.md)／[ALIGNMENT](ALIGNMENT.json)／[実行依存関係](INTEGRATED_PLAN.json)。今回の成果は計画統合のみで、科学実行許可・R採択・価格model採用ではない。

## 再現と候補lane

```bash
/home/AbeHiromu/projects/h6-environments/20261011/venv/bin/python -B \
  scripts/plan_r_screening_price_alignment.py --output-dir <fresh-dir>
```

指定Git commitのJSON/文書と既存metadata関数だけを読む。normalization/signal/N/C/fit/sampling/compile/Gを呼ばず、値の再集計もしない。新frameworkやactor別候補台帳はない。

Aの元86 PR descriptorと全menuを比較対象として保持する。priority_menuは計算queueであり、未評価候補を削除しない。Cの8セルはCSVのID/roleで管理し、H4新R8/q8・q16の2点は`EXPLICIT_C_CALIBRATION_VALIDATION_ONLY`として独立にrequestする。CをA自動Rへ置換したり、AのR上限・採択候補と見なしたりしない。

Cの校正cohortはH4(q8,R4)探索9/(q16,R4)既存33、H6(q1,R16)探索9/(q2,R16)既存8を明示選択。保存確認cohortや旧modelは別の証拠として保持する。H4 q16の元独立q成果も今回の新modelで校正に使うなら既知点であり、新holdoutへ再ラベルしない。C−Sは同じpath/source/wrapper/axisで対応付け、旧Sを不要に再計数しない。不足したdescriptorは別途明示した科学scopeに含める。

## law・N/C recordの受渡し

候補IDだけでなくfull identity（H/state/snapshot、configuration、T/qPF/δ/prefix/R/K/formula、allocation/負時間/scalar、compiler/control/wrapper/P0）とaxisを照合する。H6 qPF1のC校正点はm4/T1.6であり、m0..3のqPF1・別δ/p5/R32へ結合しない。

A screeningとB統合のlaw宣言はraw equalityでは一致しない。ALIGNMENTは両方のlaw_bindingを保存し、A/Cのrte.py定義が同一blobであることだけを確認した。実samplerの浮動小数点CDFとideal lawの距離は未資格のまま。

受入れ時は、Bの既存`build_menu/request_context`で該当物理候補のrequestを作り、実際のSOURCE/input/law証拠が一致した後に、producerがそのrequestに対応するN/C recordを返す。元A宣言とSOURCEも参照として保持する。raw文字列を無断aliasしたrecordをBの`RMenuBatch`へそのまま渡さない。今回の照合だけでlaw資格や数値PASSを作らない。

NはA既存RBatchから受け、FAIL/未実行/actual未資格・phase/radius/allowance SOURCEを残す。巨大NはPython int/decimal stringのまま、H4旧値は補正済みN_HANDOFFを正本にする。新Rへ旧Nを移植せず、R比較baselineの同R-aware allowance scalar再解析は別途許可して別結果として保存する。

Cは既存common recordを返し、direct/model/null/外挿、finite cohort/path/model mean/law期待値を区別する。新modelのfull candidate domain、校正SOURCE/cohort、独立(q,R)集合、shared coefficient group、paired XY、未知R/q systematic・cross-R covariance・CIを明示。小qで値が計算可能という理由だけで長qをqualifiedにしない。未知値は0にしない。

実価格の再利用はcandidateに加えてkind/unit/cohort/path/event digest、wrapper fingerprint/compilerを照合する。Cのjoin_keyはidentity/axisのmetadataであり、trajectory cache hitの十分条件ではない。X/Yは同じwhole-pathの2応答で、独立標本数を倍にしない。forced rare対は別unitで、無条件cohortへpoolしない。

## 許可を判断するpackage

- **A0**：元24件のnormalization-only。保存tail SOURCE/binding/prefix、閾値なしの記録、数値未解決時の扱いを固定。N/C/G・候補採択なし。
- **C1**：元Phase 1の8セル・新32 whole-path/X/Y最大64＋rare対応最大16＝80 wrapper。sampling前のSOURCE/seed/cohort、rare対の正確な(q,R)/位置/legal word、path C−S対応と評価指標をCが固定。3セル校正→係数等のcommit固定→独立qH/R1検証は一つの許可scope内の工程。n8を全law精度保証にしない。
- **A1**：元m12 N例の新mean2＋新R/baseline scalar計4候補・8軸。保存referenceが資格しなければ別scopeを判断し、無断再計算しない。C1はこの長qの価格を与えない。
- **任意J0**：Cと同じ小q点でN×Cを確認するなら新R N4＋baseline同policy scalar4。C1の前提ではないので、自動でN計算を追加しない。実partial費用評価も別途明示する。
- **C2/G**：未使用R、長q、allocation段差、全law/rare、共有誤差、全RPE成功を含む不足だけを次の有限範囲にする。C1の承認をこれらの追加実行へ拡張しない。

A0/C1は依存しないため、SOURCEと各scopeが承認されれば並行実施できる。一つの明示許可で両packageをまとめて扱うこともできるが、24 scalarと80 wrapperは別unitで数える。A1はA0内の4点をSOURCE一致で再利用する。Cの候補をA0へ増やす場合は追加scopeを明記する。

次の担当：ユーザーがpackageと必要精度を判断し、A/Cが許可範囲の結果を供給、Bが既存coverage/exact会計へ受け渡す。Bの比較は元全menu/未探索rangeを保持し、NだけでRを選ばない。本担当は公開照合後mandatory STOP。
