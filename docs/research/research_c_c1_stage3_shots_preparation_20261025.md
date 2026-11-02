# C1 Stage 3：shots下界スクリーニングの技術準備

2026-10-25 JST。起点 `2eef01a4256df4c146324ef571c0074cd0ea5b08`。
Branch `codex/research-c-c1-stage3-shots-preparation-20261025`。

**実装・人工21件＋既存回帰60件＝81 testsがPASS。保存260セルのshots再解析は0。未承認の次回再解析は最大282 logical units。公開照合後mandatory STOP。**

[未承認実行契約](research_c_c1_stage3_shots_contract_unapproved_20261025.json)／[SOURCE・テスト・byte preflight・STOP](../../artifacts/research_c_c1_stage3_shots_technical/2026-10-25/)。新規実装はscalar module・専用CLI・人工testの3ファイル。旧SOURCE・科学結果・grant・ledger・marker・STOPと元workspaceを変更していない。正式共通指示2026-10-25版は確認済みのSHA256 `e1e3e2625e2b728977402bded6583658e7fb762160f5e787fd45572af0095720`と同じ。

## 算定値の意味と再利用

対象は保存P-H4（直鎖1.00 Å・STO-3G、Nα=Nβ=2、sector36／full256）、CDF保存rank8 number・元g補正κ／XDF保存rank10。共通CA0.00159 Ha・DF reserve CA/8、δ0.1、13 round m=0..12、決定論ordered symmetric S2 s1/4・PR L_D2／s1/4／r4/16／K0/2の既存20候補を保持する。own-DF sector ground-state oracle、状態準備費用除外、ideal finite-RTE IID modelを維持する。

`src/trottertracks/c1_stage3_shots_screen.py`は、旧`c1_stage3_accuracy.assess_round`の誤差から信号・phase・coordinateへの変換と、`rpe_resource_accounting._hoeffding_shots`を再利用する。旧`c1_stage3_feasibility.classify_round/summarize`で保存分類の整合を確認し、`c1_stage3_integrated`の20候補順序・call journal・partial progress、`c1_stage1_execution`のbyte／JSON／fresh出力helperを使う。旧関数の変更、新しいRPE・fitter・Gaussian合成器・数値計算ライブラリの実装はない。

保存PF誤差をePF、補正RTE誤差をeRTE、attenuationをAとする。e=ePF+eRTEを保ち、**追加の合成誤差・law誤差が仮にない場合だけ**、既存式の

```
rho_opt = A * (1 - e)
beta_det = asin(e)
epsilon_axis = rho_opt * sin(beta_stat) / sqrt(2)
N_axis = ceil(2 * log(2 / alpha_axis) / epsilon_axis**2)
```

を計算する。A>0およびe≤sin(beta_total−beta_stat)という既存十分条件も保持する。βtotal=π/6、βstat=π/12、alpha_axis=0.01/26をX/Y各軸へ固定し、余ったphaseやfailure予算の再配分は行わない。failure配分は各候補内の26軸であり、20候補全体の同時保証を新しく主張しない。

この値は、**採用済みRPE/Hoeffding十分shots算定式における楽観的な必要数**である。同じ非負の追加誤差予算を加えれば式の所要数は減らない、という意味のスクリーニング用下界で、情報理論的・普遍的なshots下界ではない。実信号に対する最少shotsや成功確率の新しい保証でもない。保存operator誤差自体が数値pilotであり、丸めを含む値を厳密な区間証明へ昇格させない。

出力の`actual_synthesis_signal_error`／`actual_law_signal_error`／`guaranteed_actual_shots`／`CA_achieved`はnullのまま保持する。仮定を記した`optimistic_*`欄だけに算定値を入れ、実装誤差の資格付けやCA達成とは扱わない。

## 全20候補・減衰・巨大整数

主対象は前段階の潜在的適格14候補。残る6候補も削除せず、20候補×13roundを処理し、各roundにX/Yの共通値または不適格・未解決の理由を保存する。元PF/RTE/Aを各roundへコピーし、全13roundの値が得られる対象だけ26軸分の候補合計を表示する。前段階不適格／未解決候補の合計はnullとし、短roundだけの部分和を完全な合計として示さない。登録外候補は追加しない。

同じPF/RTE誤差・failure配分でA=1とした診断値と、その合計、元Aでのshotsとの差のlog10を保存する。これは減衰の影響を見る対照であり、新候補・誤差の再評価ではない。候補順序は固定のまま、回路費用や最終資源順位は算定しない。実行結果からの対象選択やshots閾値も今回採択していない。

極小Aで旧式のA×誤差やcoordinate²がunderflowしないよう、`assess_round`をA=1へ規格化して半径／phase幾何を再利用し、最後に保存Aを掛ける。通常範囲は既存Hoeffding helperを呼ぶ。算定値が2^53以上またはfloat範囲外なら、同じ式を`Fraction.from_float(unit_coordinate)`・保存A・binary64のlog定数から有理数積／二乗／ceilで計算する。通常と同じ式の整数表現域を拡張した最小処理であり、新しい推定理論ではない。

- shots・候補合計はPython任意精度整数で合計し、全て十進文字列でJSON保存。int64超過を別flagで示し、clipしない。
- coordinateのbinary64表示が0へunderflowしてもflagを記録し、countは非零有理数積で計算する。保存Aそのものが0なら旧binary64-zero不適格を保持し、失われたlog Aを復元しない。
- log10は表示・比較用診断値。整数へのceilは有理数処理で行うが、sin／log／保存binary64入力の厳密enclosureではない。規格化で直算と最下位bitが異なることはあり得る。
- NaN／Inf／負誤差／不正AはFAIL・STOP。欠測は未解決とし0で補わない。旧phase受理と保存の十分条件が境界丸めで食い違えばUNRESOLVEDとし、閾値を変更しない。

## 人工検証・preflight

| 検証 | 結果 |
|---|---|
| 新規人工scalar／end-to-end／mock | 21 PASS |
| 既存Stage3精度・feasibility・統合回帰 | 60 PASS |
| 通常範囲の旧式一致、PF/RTE保持、A=1診断と減衰効果 | PASS |
| 人工A=1e-20／1e-160／1e-300／binary64最小正subnormal | PASS。任意精度ceil、JSON roundtrip、int64超過・underflow表示を検証 |
| 人工20候補×13round、14対象／6除外、282 call・全候補保持 | PASS |
| 欠測候補保持、phase境界、科学的FAIL時partial progress、fresh／resume拒否 | PASS |
| 禁止経路のmock sentinel | PASS。operator準備・CA判定・総費用集計を未呼出し |
| 実保存データ23点のbyte hash・SOURCE11・依存5版 | PASS。数値JSON decodeをmockで禁止したpreflight |
| 既定CLI | byte preflightのみPASS、科学call0 |

実ffsim／H4 fit・NPZ decode・operator・PF/RTE再計算・実回路build／compile・shot sampling・総費用は全て0。保存260値に対する実shots算定・再解析も0。人工PASSから分子候補のshots値や資源優位は得ていない。

独立worktreeとtask内cacheだけを使用し、前段階の専用venvをread-onlyで再利用した。新しい依存は追加していない。科学条件を数値計算する部分は小さな260-row scalar loopなので、process起動／大きなimportの多重化を避ける逐次処理を選んだ。CPU／RAM／worker／wall-timeへ人工上限は設けない。後続に高負荷処理は含めず、共有環境・他ユーザーのprocess／設定を変更しない。

## 未承認再解析の実行条件

[新契約](research_c_c1_stage3_shots_contract_unapproved_20261025.json)は保存feasibility科学commit `2eef01a4256df4c146324ef571c0074cd0ea5b08`の23 scalar JSONを固定する。元のHamiltonian入力・fit・Stage1／2の科学結果を再decodeせず、旧SOURCE／結果／STOPも保全する。新SOURCE11ファイルと旧の依存lockを記録し、推移的SOURCEは起点commitで固定する。

| 科学call | 上限 |
|---|---:|
| 保存scalar bundle decode・schema／来歴確認 | 1 |
| 20候補×13roundの仮定付き評価（X/Y共通算定とA=1診断を内包） | 260 |
| 候補合計・前分類整合 | 20 |
| 登録順スクリーニング表 | 1 |
| 合計 | **282 logical units** |

各logical unitは内部の単純scalar計算・helper呼出しを含み、軸や乗算ごとの二重計上はしない。byte preflightは0。科学的FAIL・保存分類不整合で後続を即停止し、partial progress／failure／call／resources／STOPを保存する。不適格／未解決は理由を保持して他の登録候補だけ続ける。fresh output `artifacts/research_c_c1_stage3_shots_science/<date>/<fresh-run-id>/`、新fit／optimizer／retry／resumeは0、成功・FAILともmandatory STOP。

CLIは既定でbyte preflightだけを行う。

```bash
/tmp/prt-c1-stage2-preparation-20261025/.venv/bin/python \
  scripts/run_research_c_c1_stage3_shots_screen.py \
  --contract docs/research/research_c_c1_stage3_shots_contract_unapproved_20261025.json
```

将来、固定SOURCE・契約・保存input・fresh outputへの明示科学許可を受けた場合だけ、同じentryへ`--execute --output <fresh>`を渡す。今回はその実行をしていない。許可はチャットで足り、証明書・承認記録・新grantは不要。

次はユーザー／研究Cが準備SOURCE・算定値の意味・282-call契約を確認し、保存データの限定再解析を許可するか判断する。Codexは別明示許可後だけ20候補の表と対象候補のshots合計・減衰診断を取得する。回路費用評価へ自動進行しない。今回はcommit・push・remote照合後**mandatory STOP**。
