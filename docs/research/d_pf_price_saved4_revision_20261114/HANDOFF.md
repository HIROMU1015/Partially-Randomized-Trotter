# C/D接続：保存4件・新規20＋12件（p7受入待ち）

**現在の状態は[保存4件binding完了の追補](ADMISSION_UPDATE.md)を参照。** Cのp7受入・新版CELL_PLAN・共通指示の取得は解消し、SOURCE22ccf3972へ接続済み。policy/許容差は未採択で科学実行STOP。以下は初版0501096bd時点の記録を保全したもの。

改訂SOURCE：`2b9983590a642e67fda794521baac7e5eaf0496f`。
現時点では**技術準備のみ、Stage 1開始不可、mandatory STOP**。
[REPORT](REPORT.md)／[実行前条件](NEXT_EXECUTION.md)／[coverage提案](CELL_PLAN_PROPOSED.json)。

## Cから必要な固定成果

Full PFの受入bundle 3368805575／公開2fa88065は従来どおり再利用する。
追加p7成果の公開SOURCEは未確認。DはCのbranchを変更していない。
新版C CELL_PLANは、旧36行のうちp7 M5/m4 X/Yのactionだけが変わる表を想定する。
`plan_rows`は固定blobと全field比較し、configuration・role・physical ID等の変更は拒否する。

`scripts/pf_price_stages.py`の未設定箇所：

```python
P7_CELL_PLAN = None  # {'commit': <C新版表commit>, 'path': <CELL_PLAN path>}
P7_ADMISSION = None  # {'bundle': <入力/価格/proof commit>,
                    #  'admission': <bound sidecar/guard commit>, 'directory': <bundle directory>}
```

値を仮置きしない。公開後にDが正本を照合して接続SOURCEを固定する。
Cの出力formatが異なれば、実証拠を読んで必要な最小接続差分だけを作る。
人工fixtureのcommit/pathを実行SOURCEへ流用しない。

現在のFull受入と同じper-route interface：

| file | 必要な内容 |
|---|---|
| SAVED_CALIBRATION_INPUTS.json | 既存D fit_inputs/C recordによるX/Yの元入力、route・cell_id・SOURCE・profile・features |
| PRICES.json | 元trialからcopyした決定論direct record、元cohort・physical/configuration・axis保持 |
| SIDECARS.json | 固定IMPLEMENTATION_EVIDENCEに束縛済み、価格SOURCE/context/law/profile/scalar/axisの対応 |
| GUARD_CHECK.json | C628 guardのstatusと両軸checked（技術互換性、科学policy採択ではない） |
| IMPLEMENTATION_EVIDENCE.json | 元証拠との接続を示す固定assertion（sidecarがbundle commitを参照） |

元trial/receipt、hash、cohort、SOURCEはCのrecordに保全し、Dは新cohort・平均へ置換しない。
Full scalarはc、p7 B0I scalarはc＋discarded-tail identityという別quantityを維持する。
両routeは決定論PF、RTE loweringはNOT_APPLICABLE_NO_RTE、nativeはH46_LONG_TIME_NATIVE_COMPOSE。
PR用ZZ/Pauli loweringとの混在や、compiler hashだけによる同一認定を行わない。

## C_INPUTSと12モデル

science後の既存output `C_INPUTS.json`は、routeごとに6 groupを持つ。
keyは`route|formula|width_over_h|axis`、featuresは`one,q`である。

- S2・16/7・X/Y：新規m4＋新規m5。
- Y4・16/3・X/Y：新規m4＋新規m5。
- M5・16・X/Y：**元SOURCEの保存m4＋新規m5**。Full/p7両方でこの形となる。

保存recordは`saved_records[].record`、新規recordは`new_records[]`に分ける。
保存の`references`には入力/価格/bound sidecar/guardの固定locatorとaxisを保持する。
`check_group`でm4/m5以外・別route/formula/幅/axis・重複を拒否する。
Cの`_domain`単体にはformula/幅の独立keyがないため、このgroup分離を外してpoolしない。
実C_INPUTSは未作成で、正例は人工試験だけである。

HOLDOUTはm6のみ。同じnamespaceの再実行、CALIBRATIONの未公開/改変/未bind、
C凍結modelの不足/混合/m6混入/別SOURCEなら、childより前に拒否する。
前Stageの実価格は読取参照し、fitやholdoutの数値判定はDでは行わない。

## 重複検査と次の作業

`duplicate_preflight`は各new cellの両軸について、固定保存原表・catalog・元trial path・
既存stage結果・関連実行中jobを調べ、追加証拠/未解決重複を検出したらSTOPする。
旧price tableに存在しないことだけで、未知のprivate archiveや別SOURCEの同値性を認定しない。
現在のsnapshotは[DUPLICATE_PREFLIGHT](DUPLICATE_PREFLIGHT.json)。本番直前に再確認する。

**C**はp7受入と新版表を公開、**ユーザー**は比較policy・許容差と共通運用指示の所在を確定、
**D**は固定SOURCE結合と開始可否の再確認を担当する。
最大34件という旧承認・固定STOPは保持するが、今回の指示はscience開始を許可しない。
受入不成立時は理由を保存し、当該2件を再compileへ自動復帰させない。
