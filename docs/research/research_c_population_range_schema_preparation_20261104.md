# 研究C：Stage Rのschema修正・新規実行準備

2026-11-04 JST。**保存Stage AのeventタグとIIDモデル設定を分けて照合するopt-in入口を追加した。新schema人工48件、適合回帰込み313件PASS。新契約は未承認256 logical units、実H4科学callは0。** 旧失敗試行のretry/resumeや残246 callsの転用は行っていない。

## 修正と再利用

起点は固定失敗commit `e73f79de6e46d65e76ac7126f4eaf8a9b3568c0b`。独立branch `codex/research-c-stage-R-schema-repair-20261104`、独立worktreeで作業した。正式共通指示2026-10-25版の§8.4・§10を適用し、技術実装・人工検証・新契約準備だけを行った。承認証明・原文保存・新grantは作成しない。

旧科学SOURCE `6c1fbe5399b2418e680acaa4f6e8aaa5e27d6dbd` の[10/256-call部分失敗](research_c_population_range_stage_R_execution_20261104.md)は保全した。旧runner、人工fixture、SOURCE manifest、契約、raw、journal、STOPの原本を変更しない。

| 保存側のschema | 新しい照合先 |
|---|---|
| `ideal_event_mean.json.model`：文字列`ideal_saved_atom` | 理想eventタグの文字列定数 |
| `grid_event_mean.json.model`：IIDモデルIDの文字列 | `identity.model.id`と同じ固定IIDモデルID |
| Stage A `result.json.model`：全IID設定の辞書 | 契約`identity.model`の辞書および固定`MODEL` |

[c_per_word_range_schema.py](../../src/trottertracks/c_per_word_range_schema.py)は、JSONの型を含めて独立性・53 bits・mapping・inverse規約・PCG64未資格設定を照合する。`True`と`1`、`53`と`53.0`も同一視せず、文字列→辞書の変換やfallbackを行わない。保存ideal/gridのangleは既存binary64値を保持し、bit同一性を確認する。新しいangle算定・law再生成はない。

旧`execute_range`の既存call dispatcherを拡張点に使い、`native_model`の返値を受けて、`event_catalog`の処理だけを明示的に新schema照合へ置き換える。誤った旧metadata closureは呼ばない。全CAR/head producer、signed atom/frame/support/係数binding、grid support、Arb norm、13round、call journalはそのまま再利用する。Frozen SOURCEに変更を入れず、数値pipelineを複製しないための小さいadapterである。旧方式へ戻す例外処理や保存JSONの偽装はない。

新入口は [prepare_research_c_population_range_schema.py](../../scripts/prepare_research_c_population_range_schema.py)。既定動作はSOURCE/input hash・package・JSON schemaだけのpassive preflight。新しいNPZ/QPY decode、CAR生成、norm評価、科学runnerの起動を含めない。科学flagには別途明示許可が必要。

## 入力gateの確認・テスト

[Schema監査](../../artifacts/research_c_population_range_schema/2026-11-04/technical/schema_audit.json)は実保存JSONのkey/typeとproducer/consumerの静的確認に限定した。

- Atom一覧は`atoms`内のID/frame文字列・support整数list・係数字符列を照合し、重複IDも拒否する。既存のsigned係数一致検査は維持する。
- Tableは`grid`内のcount整数list・denominator整数・bits整数・mappingを確認する。Count和・正supportの科学評価は従来の`support_gate`に残す。
- Origin contract/result/SOURCE/STOPは既存`evidence`を再利用する。B1/B2側のmodel辞書と`random_model` ID文字列も、それぞれ対応するschemaと比較していた。
- 保存sector/basis NPZは既存loaderの宣言keyとproducerだけを確認し、新たな配列読込を行わない。Head bit一致は将来Stage Rの診断事項のまま。

この対象範囲では、ほかにeventタグと全モデル辞書を混同するgateは見つからなかった。全repoや数値correctness全般を保証する監査ではない。

[新テスト](../../tests/management/test_research_c_population_range_schema.py)の人工fixtureは実Stage A JSONのtop-level/report key・型に合わせ、値は人工入力とした。正しいタグ/モデルのPASS、誤タグ・誤IID ID/設定・辞書/文字列混同・bool/int/float混同・非有限角・angle不一致・count型・入力/SOURCE不一致を検証した。Dispatcherで旧closureを呼ばないこと、失敗時の後続停止・部分保存・mandatory STOP・fresh拒否も確認した。

人工の8-frame/4-orbital J/U、216 atomsで、既存CAR→新schema→norm→13roundのend-to-endが256/256 COMPLETEとなる。これは人工データの技術結果であり、実H4のpopulation a/Mや科学的PASSを意味しない。

- [新schemaテスト](../../artifacts/research_c_population_range_schema/2026-11-04/technical/schema_tests.txt)：**48 PASS**。
- [関連回帰込み](../../artifacts/research_c_population_range_schema/2026-11-04/technical/regression_tests.txt)：**313 PASS、6 deselected**。
- 除外2件は、旧消費済みStage R契約にfreshを要求するcold testと、既知の誤schemaを持つ旧end-to-end fixture。原本を改変せず、正しいschemaの新cold/end-to-end testで置き換えた。
- 残る4件は前段から除外されていた歴史的SOURCE drift依存preflight test（IID bridge 1、B1/B2 scope 1、P0 uniform 2）。[既存のparent確認記録](../../artifacts/research_c_iid_grid_expected_bias/2026-11-04/technical/historical_parent_checks.txt)を再利用した。全履歴契約の回帰PASSとは主張しない。

既存Arb環境Python3.11.1/NumPy1.26.4/SciPy1.14.1/Qiskit1.3.0/python-flint0.8.0を読み取り利用し、共有venvを変更していない。Passively確認したCPU affinityは32、MemAvailable約52 GiB、CPU/memory pressure avg10は0。既存の独立3 head resource planとworkerあたりBLAS1を維持し、少量scalar Arbはprecision lock付きの既存経路を使う。新しいCPU/RAM/worker/wall capや共有設定変更はない。性能speedupは測定・主張していない。

## 新SOURCE・未承認契約

[新契約](research_c_population_range_schema_contract_unapproved_20261104.json)、scope **`C_PR_R32_LITERAL_POPULATION_NORM_STAGE_R_SCHEMA_V2`**、最大**256 logical units**。旧256-call契約は消費済みSTOPのため再利用不可。今回は新しい独立した完全Stage Rの契約を準備しただけであり、実行許可はない。旧246残枠からの移管は0。

[新SOURCE](../../artifacts/research_c_population_range_schema/2026-11-04/technical/SOURCE.json)は旧154 fileを無変更で継承し、新module/runner/testを加えた157 file。保存入力は旧34件を維持し、schema照合用`grid_event_mean.json`を1件追加した35件。Stage A resultは元から固定入力に含まれている。契約に記録したSOURCE manifest SHA256と各file hashが科学実装を固定する。

対象・数値条件は元のまま：直鎖H4/1.00 Å/STO-3G、number CDF8、Nα=Nβ2・sector36/full256/controlled512、元順LD2・partial ordered S2/s1/r32/K0、h=δ=.1、m0..12、CA=.00159 Ha・reserve CA/8、own-DF ground oracle・状態準備費用除外。EndpointはSHA256 `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`。Literal U、元g補正κ/scalar/frame順、192-bit Arb、P0条件は変更しない。

| 新規契約内の区分 | logical units |
|---|---:|
| evidence/input/candidate読込、binding、modelデータ、CAR context | 6 |
| raw head producer | 3 |
| event catalog、grid table、保存head読込、head binding | 4 |
| exact support | 1 |
| head norm | 3 |
| frame Gram/norm | 8 |
| event norm | 216 |
| population組立 | 1 |
| 13round growth | 13 |
| summary | 1 |
| **合計** | **256** |

Schema読込・照合は既存event_catalog unit内とpassive gateへ置くため、科学的norm評価の追加は0、上限は同じ数となる。旧head配列は保存前に停止しており、部分計算をpoolして省略せず、新契約では全pipelineをfreshに評価する設計である。これを旧試行のresumeとして起動しない。

Fresh出力は **`artifacts/research_c_population_range_schema/2026-11-04/stageR-run01`**。現在存在しない。[Passive preflight](../../artifacts/research_c_population_range_schema/2026-11-04/technical/passive_preflight.json)はPASS、科学call/配列decode/実行開始は0。Cold testでも数値libraryのimportを禁止してPASSした。

実行許可後の入口は次の一つだけである（今回は未実行）。

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python scripts/prepare_research_c_population_range_schema.py --execute-stage-R
```

Runtimeは契約の固定packageに適合する既存環境を選ぶ。実行前にSOURCE/input/schema・実際の空き資源を確認し、上記順序で一度だけ評価する。Correctness/input/SOURCE/schema/runtime FAILでは依存処理を停止し、部分証拠を保存する。Retry/resume0、threshold/target/数値allowanceの事後変更・fallbackなし。正常終了でもmandatory STOPする。

## 残る未資格・停止点

実H4のa/M、216 event norm、13round growth、B2とのhead bit bindingは未取得のまま。新しいper-word誤差、confidence/期待bias、shots/energy/費用、actual CAを得ていない。Stage P/C/Aは非実行、全q compiler totality・useful synthesis bound・CA transportを今回の修正から認定しない。

[技術照合](../../artifacts/research_c_population_range_schema/2026-11-04/technical/verification.json)で旧SOURCE154 file・旧契約/manifest・失敗run6原本の計162 fileが起点と同一であることを確認した。[今回の技術STOP](../../artifacts/research_c_population_range_schema/2026-11-04/technical/STOP.json)は新科学0を記録する。旧契約・raw・grant・ledger・marker・STOPを変更しない。

公開時にbranch remote SHA/tree、主要SOURCE/契約/報告/証拠blobを照合する。**その後mandatory STOP。** 次の担当はユーザー／研究C：この新契約の科学実行可否を判断する。Codexは別途明示許可を受けた場合だけ固定V2 Stage Rを実行し、Stage Cや重要科学レビューへ自動進行しない。
