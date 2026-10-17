# SPEC-04〜07 API・証拠形式・独立性

[技術報告](technical_report.md)／[引継ぎ](decision_handoff.md)。実装は[新namespace](../../../src/trottercert/a2_spec04_07/)。stdlibのみ。読み込む数学対象は旧SPEC-01〜03のexact raw envelopeと受理された証拠で、実入力loader・CLI・数値eigensolver・指数関数・科学runnerはない。

## 呼出し

```python
from trottercert.a2_spec04_07 import generate_spec04_07, check_spec04_07

proposal = generate_spec04_07(
    raw_synthetic, base_spec01_03_evidence,
    particle_number=declared_N, occupied_modes=declared_occupied_indices,
    sqrt_bits=explicit_dyadic_bits, comparison_policy=explicit_synthetic_policy,
    fragment_order=declared_order, expected_model_id=intended_model_id,
)
checked = check_spec04_07(
    raw_synthetic, base_spec01_03_evidence, proposal["evidence"],
    expected_particle_number=declared_N,
    expected_occupied_modes=declared_occupied_indices,
    expected_sqrt_bits=explicit_dyadic_bits,
    expected_comparison_policy=explicit_synthetic_policy,
    expected_fragment_order=declared_order, expected_model_id=intended_model_id,
)
```

この例の変数はcallerが独立に指定するもので、実科学データへ適用する許可ではない。既存APIと同じくraw/evidenceはdictまたはstrict JSON文字列を受ける。SOURCEの自動動作やimport時の計算は行わない。generatorの成功flagをcheckerへ渡しても証拠としては受理しない。

- N・occupied indices・p・max_bisectionsはcanonical decimal strings。occupiedは0-based昇順list、JW bitはmode0 LSB、Pauli文字列はmode0が左。
- `comparison_policy`はNone、またはexact keys `d, alpha_spec, root_width, max_bisections`。dはHa、root_widthはHa^-1、alphaは無次元。Ratは既存の `{num: canonical Int, den: positive canonical Nat}`、既約形。`d>0, 0<alpha<1, root_width>0, max_bisections>0`。callerが指定しない科学値を推測しない。
- `sqrt_bits>0`を明示する。これはsqrt witnessのdyadic gridを指定する技術パラメータで、binary64やplatform精度の認定ではない。
- 科学scopeのraw tagは旧SPEC-01〜03 guardで拒否する。ただし自己宣言tagはsecurity attestationではない。SOURCE・科学入力・科学契約のfreezeと別実行承認は将来別途必要。

## Recordとstage

generatorは `EVIDENCE_GENERATED_UNVERIFIED` を返す。evidenceはnamespace `spec04_07_evidence_v1` のcanonical hash node `{id,payload}`。payloadは次を持つ。

| field | 内容 |
|---|---|
| schema/scope/completion | A2_SPEC04_07_V1 / SPEC04_07_SYNTHETIC_TECHNICAL_ONLY / SPEC04_07_SYNTHETIC_ONLY |
| source | 新8 moduleのbytes SHA256。旧sourceはbase証拠checkerが別検査 |
| context | model/sector/partition/PF/base_evidence/raw_inputの全identity |
| sqrt_bits/comparison_policy | callerの期待値とexact一致を要求 |
| upper_certificate | kind/parents/trial/data。全列・signed内訳・L/U/W |
| ksos_certificate | kind/parents/data。全f、位置、raw辞書、unique sqrt witnesses、行和、上界 |
| commutator_certificate | kind/parents/data。group全体のPauli展開、centered/構造norm候補とmin、1/24・1/12和 |
| step_comparison | kind/parents/certificate_refs。ground条件・trial非ground、policy、全routeのprobe/bracket/判定 |
| cost_ledger | 両側共通情報pool、未測定generator/checker費はnull＋理由 |
| limitations | 原分子・native・state・platform・best-known envelopeの未確立/未評価、正式状態、authorization false |

個々のtyped nodeにもcontent hashを付け、step nodeは検査した3 certificateのIDへ結ぶ。単位・SOURCE・期待trial/policy・LIMITS・costも検査対象。改変後hashだけを付け直しても意味論の独立再検算で拒否する。

checker成功は `SPEC04_07_VERIFIED_EXACT_SYNTHETIC`。これは人工対象の特定証拠をSOURCEが受理したという限定verdictで、platform数値certificateや実DF分子の `VERIFIED_EXACT_FOR_FIXED_DF` を今回取得したという主張ではない。step policyなしでもSPEC-04〜06は検査し、stepは未評価とする。policyがあっても`UNRESOLVED_ARITHMETIC`を保存でき、bundleの正しい未解像recordとstep優位を混同しない。scientific GO/NO-GOは出力しない。

失敗はcode/path/stage、提出input/base/evidenceのdigest、可能なcontext/partial proposals、空verified_quantitiesとLIMITSを返す。SOURCE I/Oはtechnical error。OS killや全例外を回復するrunnerではなく、永続保存はcallerの責務。partialのgenerator proposalをchecker受理済みと扱わない。

## 実装経路の分離

| SPEC | generator stage | checker stage | 共有する前提 |
|---|---|---|---|
| 04 | producer_algebra.upper：bit-prefix符号、作用norm² | verifier_algebra.upper：ordered occupied list parity、occupation行列A²の対角 | CAR規約、旧checker受理model、Fraction/representation |
| 05 | producer_algebra.k_sos：raw寄与列挙、整数isqrt上側grid、streaming row | verifier_algebra.verify_k_sos：全tupleの別列挙、exact sqrt不等式、明示Bとrow | 固定K0定義、表示順/caps、Rat/hash |
| 06 | producer_algebra.commutators：JW ladderと局所Pauli積table | verifier_algebra.commutators：occupation行列の積・差、全Pauliのtrace | exact model、group式、標準norm不等式 |
| 07 | producer_steps.compare/predicate：有理ηとB式 | verifier_steps.compare/predicate：正の分母を払った別不等式 | 固定数学定義、domain、同じpolicy/context |

低水準stage関数はnamespace内部の実装で、科学的に自由使用可能なrunnerではない。facadeで必ず旧SPEC-01〜03を先にcheckし、全contextを共通にする。checker計算moduleはproducer/generatorをimport/callしない。package `__init__`は両public facadeをexportするため、importされるmoduleの完全分離を主張しない。

`common.py`はstrict encoding・ID、期待context、trial/policyのschema、source hash、入力展開だけを共有する。U、BCH、commutator、step算術は共有しない。ただし全体を別チーム・別runtimeで再実装した独立検証でも形式証明でもない。現人工testがcheckerの全bugを排除する保証はない。

## 複雑度と未取得費

generator Uはおよそ `(r+1)n²` の係数処理と疎遷移集約。K0 row生成の算術項数はO(M³)、row作業vectorはO(M)だが、今回保存するraw辞書はO(M³)、unique root cacheはO(M²)。checkerはK0でO(M²)のBも使う。巨大有理数のbit演算とJSON保存は別費用。

JW多項式の語数は積・group展開で増大し得る。occupation checkerは最大2^n次元、Pauli traceは4^n個を列挙するため指数的。sparse matrix積・非零数・整数bit長でも費用が変わる。今回は小さな人工referenceとして意図しており、実分子のscalableなcertificate checkerと称さない。任意nを安価に扱えること、memoryの事前成功保証はない。新しい人工的CPU/RAM上限を設定して解決したことにもしない。

stepは指定された有理数instruction budget内のbisectionで停止し、未達は未解像。これは科学runtime capではない。将来の算術幅/反復数は科学契約に明示採択が必要。test内の16-bit sqrt、128回等を科学の既定値へ流用しない。

実行時間・RSSはテストsuite全体の観測で、各certificate費、実分子での実用性、native費は未測定。相対側と対照側に全証拠を公開し、構造対照の取得費を除外して優位を作らない。費用の単一尺度への合算や償却回数は未採択。

## Guardと終了境界

人工scopeの検査だけを許し、ファイル型science adapter、実入力CLI、execution authorization/grant/markerの作成・消費、旧ledger更新、SPEC-08は追加していない。science contractは作成していない。SOURCE hash一致は実行許可を意味しない。

Taskの完了後はmandatory STOP。A2未採択、A1終了、K1-F未承認を保つ。新しい科学条件・実分子input・QPE/native・reviewは別途明示承認を要する。今回の人工検査合格だけでその承認条件を満たしたと扱わない。
