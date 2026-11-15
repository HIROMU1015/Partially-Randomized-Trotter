# PR-DF LD1/LD2 段階的有限比較：価格取得完了・新Q3前の技術STOP

保存H4 linear 1.00 Å／STO-3G／tol-only DF10、one-bodyを含めない元DF順prefix L_D=1/2、S2、Δ=0.1、K2、元exact κが対象。計画f42f6e9bの採択範囲を実施した。**LD2の12 normal path・24 compile、2軸fit・26参照価格、候補別4配分の固定まで完了。新Q3はSOURCE/AST gateで科学call前に停止し、4配分とも未評価。比較全体の完了ではない。**

| 分割 | Q3適合を確認できた配分のうち最小のモデル費用 RZ | 選択X配分 m0..12 | 選択Y配分 | 保存Q3 MSE区間 Ha² |
|---|---:|---|---|---|
| LD1 | 261,314,830,475.559718 | V0 `[64,60,56,52,47,43,38,34,30,25,21,17,12]` | 同じV0列（別軸） | [4.72462588934653801951643E-7, 4.72462588934653960729534E-7] |
| LD2 | 63,914,076,114.980849 | V0 `[64,60,56,52,47,43,38,34,30,25,21,17,12]` | 同じV0列（別軸） | [4.72409731948051054908399E-7, 4.72409731948051194818804E-7] |

上表は既存V0適合2件を再利用した範囲。A_SHAPE/B_PRICEを含む6件の全評価済み最小とは呼べない。MSE上限閾値は15136964828156312001/5960464477539062500000000 Ha²、元total decoder・target・全事象lossを維持。価格は期待値の認定ではなく探索用モデル参照である。LD2のV0モデル費用はLD1より小さいが、配分を選び直した後の比較は未完了。全L_D最適性・単調性・正式G/winnerは未認定。

## 実施・保留件数

| 項目 | 上限/予定 | 再利用 | 新規完了 | 保留・未実施 |
|---|---:|---:|---:|---:|
| LD2 normal path | 12 | 0 | 12 | 0 |
| X/Y full-wrapper compile | 24 | 0 | 24（receipt有効24） | 0 |
| LD2軸別モデル／価格 | 2／26 | 0 | 2／26 | 0 |
| LD1価格／fit | 26／0 | 26価格 | 0 | 0 |
| A/B整数配分 | 4 | 0 | 4 | Q3の評価は保留 |
| V0 Q3 | 2 | 2 | 0 | 0 |
| 新Q3 | 最大4 | 0 | **0科学call** | **4** |
| 新信号／LD3／独立価格holdout／科学retry | 0 | — | 0 | 対象外 |

count-onlyは4件とも通過。新列のclosure・MSE・DPは0件なので、旧V0のclosure PASSを新列へ引き継がない。4件にモデル不適合やINCONCLUSIVEの数値判定を付けず、`HELD_BEFORE_SCIENCE_SOURCE_GATE`、MSE欄nullとする。

## 取得条件と元receipt

[取得前INPUTS](ACQUISITION_INPUTS.json)・[差分照合](DELTA_CHECK.json)を固定し、計画後の同等LD2結果・fit・Q3・同cohort jobは未同定。旧prefix2 q4/R4価格、LD2 m12のsampling前STOP、PR-Pauli LD2は除外した。Full-H m0・p7 width-hの他担当作業は条件が異なる。新outputと空の専用cacheを使い、cache hit0、出所不明cacheからのfallback0。

取得はD e8273676の`ld12_price_requests.request_for('PR-LD2',m)`→`NativeRuntime.context`→同adapter `acquire`。matching fast SOURCEもe8273676から専用名で読み、旧C1のprefix1固定`Backend.context('H4')`は使わない。元P1/P2のAccounting・Gaussian・compose・efficient ZZ・元samplerを再利用した。NumPy1.26.4／Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、symmetric-directional control、P0、測定付き9-wire full controlled Hadamard wrapperを各軸で別compileした。

入力H/state/snapshot/rank、元configuration・physical候補、request、q/R/T、cohort/index/outer seed、実preparation、λ、head/tail、total scalar、同path event digest、各軸の6指標/fingerprint/compiler・mutationを確認。headはone-body＋DF fragment[0,1]、tail[2..9]。preparation hashは`fb3ee708f92ab1e02ffcd2918ba0535bd65ec6d650b1a0adb4fe76b51288810b`、total scalar=2.515477156730702 (`0x1.41fb27ccff28cp+1`) を総Tに一回適用した。新しい作用保証は付加しない。数値action gateとfull-wrapper statevectorは元P2 fresh同様未実施。

cohortは`D_PR_DF_LD2_SPLIT_CAL_20261115_V1`。各4 pathのindex0..3、元carrier trial_seedのouter seed列は取得前INPUTSから参照できる。X/Yは同一pathの別軸であり独立8 pathとは数えない。

| m / physical | q/R | path0 C(X,Y) | path1 C(X,Y) | path2 C(X,Y) | path3 C(X,Y) | 各軸exact平均 C |
|---|---|---|---|---|---|---|
| 0 / ce7ef02e7926c1a6c924 | 1/1 | (2479,2479) | (2525,2525) | (2557,2557) | (2585,2585) | 5073/2 |
| 3 / 334b1fd35cb2f1ad32b0 | 8/5 | (21427,21427) | (22835,22835) | (22685,22685) | (23101,23101) | 22512 |
| 6 / fe234d98b289ae91869c | 64/40 | (551648,551648) | (552354,552354) | (548893,548893) | (550379,550379) | 1101637/2 |

[COMPILE_ROWS.json](COMPILE_ROWS.json)は24行でS/C/D・元request・seed・event digest/観測内訳・軸fingerprintと固定receipt locatorを結ぶ。[raw RESULT](acquisition/RESULT.json)、各`m{m}/p{i}/PATH_RECEIPT.json`、`cosine.json`/`sine.json`が原本。実eventは全12 pathでorder0のみ、order2/productは未観測である。強制eventや追加seedは使わず、希少事象がないとは判断しない。元event JSON bytesは専用private directoryに保全し、同じbytesのgzipを公開。[EVENT_STORAGE](EVENT_STORAGE.json)に両方のhashとpathを残した。

## 価格の固定

LD2は軸別に`C_ref=theta0+thetaq*q+thetaE*qR`、モデルoffset=0。head込みfull-wrapper CをfitしたためSや残差を再加算しない。exact square solveは既存d83737b6の`_solve`を再利用。point-lawを同一と偽装せず、各cellのlawとreceiptを保持したprefix2 schedule-family modelである。

| 軸 | theta0 | thetaq | thetaE |
|---|---:|---:|---:|
| X | 430645/966 | 19822871/10304 | 737005/4416 |
| Y | 430645/966 | 19822871/10304 | 737005/4416 |

[凍結モデル](FROZEN_LD2_MODELS.json)と[52価格表](AXIS_PRICES.json)を公開済み。2 fit・26 LD2予測を一回実施し全て正・finite。3点3係数なので較正残差0は予測検証ではない。m1/2/4/5は未検証の補間、m7..12はq64/R40を越える外挿。LD1は85dab58dの構造期待値＋残差モデルによる26価格をそのまま再利用し、m4観測への差替え・再fitは0。

fit後のV0 JSON読取でPythonの4300桁制限に当たった。既に保存した2モデルと26予測を保全し、B2同様に桁数制限を解除して**配分生成だけ**継続した。fit/予測の繰返しは0。[読取修正記録](ALLOCATION_READ_CORRECTION.json)参照。

## 候補別配分と費用

B2 e094b2d6の正の有理数版`generate`をAST抽出して無変更再利用。各LD自身のV0総額の3/4を予算に、最大finite breakpointを選んだ。A重みV0、B重みV0/C_ref、1≤N≤V0、各軸価格は丸めずexact積和。PF金額を予算に使わない。X/Yが同じ数値の価格になったことも、LD間で同じ列になったことも結果であり、配分を共通に強制していない。

| LD | 配分 | 実際のモデル総額 RZ | Q3状態 |
| 1 | V0 | 261,314,830,475.559718 | 保存V0適合を再利用 |
| 1 | A_SHAPE | 193,529,481,006.191629 | 未評価：SOURCE gateで科学call前STOP |
| 1 | B_PRICE | 189,232,849,610.910739 | 未評価：SOURCE gateで科学call前STOP |
| 2 | V0 | 63,914,076,114.980849 | 保存V0適合を再利用 |
| 2 | A_SHAPE | 47,334,382,243.300078 | 未評価：SOURCE gateで科学call前STOP |
| 2 | B_PRICE | 46,341,978,336.212733 | 未評価：SOURCE gateで科学call前STOP |

V0 = `[64,60,56,52,47,43,38,34,30,25,21,17,12]`。A_SHAPE = `[48,45,42,39,35,32,29,25,22,19,16,12,9]`。B_PRICE = `[64,60,56,52,47,43,38,34,30,25,21,17,7]`。いずれもX/Y別配分で同じ整数列となった。exact lambda・3/4予算・残額・次breakpoint・26軸単価は[ALLOCATION_INPUTS](ALLOCATION_INPUTS.json)、[MENU.csv](MENU.csv)と[SUMMARY](SUMMARY.json)参照。表の費用は予算額ではなく実配分の費用。

## Q3技術STOPの原因と保存範囲

既存route27e70efe、kernel710aa69b、accelerator9e294778、C c628e9e8を専用detached worktreeから読んだ。metadata/canonical bindingと新列count-onlyは確認し、Aの遷移上界77,967,141、B137,062,954で固定141,559,036以内。Bのm0は元64/64 latticeと同じため既存complete first-round state countを再利用し、他roundは新列のcountを使った。

最初の投入は動的module未登録のPicklingErrorでworkerに入らず、科学call0。module登録を修正し、同supervisorでfake evaluatorを子processに渡す回帰を通した。旧[q3/COMPLETION](q3/COMPLETION.json)を保全し、科学評価の再試行は行っていない。

次のworkerは全4件とも**Accelerator constructorのSOURCE/AST照合で停止し、T.evaluateには入っていない**。科学call0は各`WORKER_PERSISTENCE_ERROR.json`で確認できる。[q3_calls/COMPLETION](q3_calls/COMPLETION.json)・START/資源記録を保全した。API_RETURN/closure/DP/MSEは存在しない。

[静的原因調査](Q3_SOURCE_STOP_ANALYSIS.json)：kernelファイルbytesは固定710aa69bと一致する。Python3.12がFunctionDefのASTに`type_params=[]`を追加するため、AST hashはeb3150a5…となり元の3d043e71…と一致しない。診断用のAST文字列から空の追加fieldだけを除いたhashは固定hashに一致した。**実gateにはこの除去を適用せず、hashの変更・guard緩和・別backendへのfallbackは0。以後の投入を停止した。** 本体SOURCEが変更されたと結論してはいない。

人工試験のworker本体mockではconstructorを通していなかったことが適用漏れだった。最小再発検査で実constructorの停止を再現し、callerのmetadata preflightへ同constructor検査を移した。次の技術対応は元AST契約と合うruntime、または科学bodyを変えないAST互換処理の明示SOURCE化・人工検証。現在のSTOPを越える科学callは行わない。

## 確認済みV0のround寄与と次の一案

| 分割 | m0..m6モデル費用 RZ | m7..m12モデル費用 RZ |
|---|---:|---:|
| LD1 V0 | 198,979,534.093441 | 261,115,850,941.466277 |
| LD2 V0 | 66,223,444.937371 | 63,847,852,670.043478 |

[SUMMARY.round_differences](SUMMARY.json)は13 round別のLD2−LD1を保存。m0/m1ではLD2が約77,482／71,280 RZ高く、m2から小さい。モデル上の差は長round側で増え、m12だけでLD2−LD1≈−130,823,719,406.114070 RZ。新配分の適合は未判定なので、これは保存V0同士のモデル寄与比較である。

**現在のV0とモデル寄与に基づく暫定推奨は、LD2の長round価格確認である。** LD3を増やす前に、ほぼ全費用と分割間の差を支えるLD2 m7..12外挿を確認する方が、現在の結論を直接検査する。最初はm9等の一つの長roundで、prefix2/head/tail/角度・同path X/Yを保つ独立normal pathの実取得案を別途固定し、記述的にモデルとの差をみる。m9の確認だけでm12や母期待価格を資格しない。今回この追加取得は未採択・未実行。

実務上は先にQ3 SOURCE/runtime接続を解決し、現在の四つの未開始callをどう再開するか別指示で確定する必要がある。新しい配分や追加signalを作らず、凍結価格・整数列・予算・元signal・target/loss/閾値をそのまま再利用できる。Q3再開前に関連branch/jobの差分と初期constructorを再確認する。

Full PF 615,304,502 RZとB0I p7 435,481,090 RZはb4296b11の方式間参照値。探索モデル費用との数値的大小はあるが、ここから正式E[C]・実PCG64 joint law・CI・G/winnerの認定には進まない。Q3は保存ideal finite-RTEのIID raw measurement条件、価格の較正は元normal samplerの指定有限cohortという根拠の違いも保持する。

## 検証・資源・保全

人工検証はprefix2/total scalar/X,Y receipt 4件、exact solve/有理数generator 4件、Q3 process persistence・実constructor gate再現4件の**distinct12件PASS**。各群は実際の専用SOURCE contextを分けたprocessで実施した。元H4 P0演算子再検査を含むsuiteは起動していない。初回loader import失敗と読取・dispatch失敗は記録し、人工PASSを実Q3成功と扱わない。

12独立pathを並列、X/Yはpath内順次、既存のRZ projection／共通prefix／fingerprint再利用を使用。全取得wall262.287853秒、path CPU合計1053.660353秒、単一worker最大RSS1,223,216 KiB。m0は2.72–2.94秒/path、m3は11.42–13.74秒、m6は239.29–259.31秒。これは単回実測で速度倍率試験ではない。RSSはtree同時peakではない。pathごとのOS PID/exit signalは今回runnerで別欄保存しておらず、成功Future/receiptを根拠に記録した。この観測限界もSUMMARY.resourcesに明記した。Q3科学CPUは0、supervisorの2回の技術投入は各0.017108／0.043053秒で停止。

LD1 m3/m4原本・m4独立holdout・凍結モデル・既存F0・旧SOURCE/raw/STOPを変更していない。他担当branch/worktree/jobも変更していない。SOURCE固定後の実行中code変更・無断sampling/compile retryは0。価格原本公開SHAと実行code SHAは下記のとおり分離した。

- 取得code: `ea9e66092465d39dfcd4aeb6fea8e9f09808cabb` / `scripts/run_pr_df_ld_split.py`
- 取得INPUTS: `1ef521ba55ab4b128d22eca4970c4eba773f2dc4` / `ACQUISITION_INPUTS.json`
- raw公開: `b030ea97a6062a35a73069988c2ed616b6869bc4` / `acquisition/RESULT.json`
- fit code: `8e298f484be31af867d61503d00d5bdbec179b6c` / `scripts/freeze_pr_df_ld_split.py`
- モデル公開: `3ac3e09b73602ed705a00fe5e94858ca1846f48c` / `FROZEN_LD2_MODELS.json`
- 配分公開: `4ef70274261cf18bd9356a15d780eb5b6b61c60c` / `ALLOCATION_INPUTS.json`
- Q3 loader code: `a6d732fe35ba819f44cdbafd522aa857dd3faa0f` / `scripts/run_pr_df_ld_split_q3.py`
- Q3固定入力: `4d3aa8a666e153188addfb3a4f2b8ef10733a3dd` / `Q3_CALL_INPUTS.json`

## ロードマップ差分・mandatory STOP

統合担当は「LD2価格なし／全段階未実行」から「3セル24 receipt有効、モデル・26価格・4配分固定、Q3は技術SOURCE gateで4件未評価」へ更新できる。六つの配分を全て評価したことにせず、モデル上でLD2が勝つという最終結論にも進めない。参照比較の次の研究優先はLD2長round、運用上の未完了依存はQ3 runtime/AST接続である。ロードマップ本体は変更しない。

[HANDOFF](HANDOFF.md)と機械可読表をcommit・pushし、remote SHA/tree/主要artifact bytesを照合後mandatory STOP。LD3・追加holdout・別配分・追加Q3・科学retryは実施しない。
