# Full-H κ2 F1：全額はnull、m6モデル部分額を保存

**F1実行完了。m0-X/Yの負の参照価格により、全13 roundモデル参照総額 `T_ref` と両PF基準に対する全額の順位反転境界 `D_PF` はnull。** 26軸のうち24軸はSOURCE・構成・非負性gateを通過し、2軸は `OUT_OF_DOMAIN_NEGATIVE_MODEL_REFERENCE`。価格を0へ丸めず、保存観測値や別軸で置換していない。

m6の有効な保存予測だけによるモデル部分額は **633,699,606.0515325… RZ**。B0I p7 S2基準を40,144,254.0515325… RZ上回り、Full PF S2基準を204,993,973.9484675… RZ下回る。これは `MODEL_SCENARIO_ONLY` の部分額であり、母期待値の下界・全round総額・正式な順位ではない。

## 実行SOURCE・固定入力

- 科学実行SOURCE：`417526460f22f9f49dfcf166442740c17a00105b`、`scripts/run_full_h_f1_diagnostic.py --execute`。
- F1計画・26軸正本：`61f249e6e18d5afab9ba8797bef5e8f9f14e29de:docs/research/full_h_d7_holdout_admission_plan_20261115/` の `NEXT_SCIENCE_PLAN.md` / `ADMISSION.json`。
- 元モデル：`b5165f11bb63f9eb2a6bfe385ba75d8003b7e724:docs/research/full_h_d7_model_freeze_20261115/run01/FROZEN_MODELS.json`。
- D1独立holdout：`06956c64380107392f1b24fe54b9273f0f5ae1c0`。保存4予測を利用し、観測平均や誤差の再集計はしない。
- Bの固定Q3・1.5倍配分：`6c5569c3479013b080b2ba53842585f1d670d3c9:docs/research/full_h_kappa2_scale3o2_q3_results_20261115/`。`N_X=N_Y=[96,90,84,78,71,65,57,51,45,38,32,26,18]`。
- PF2基準：`1002f1dfea34de439ca1b704a0192bca473c5387:docs/research/pf6_allround_cost_20261115/RESULT.json#rows[0],rows[2]`。593,555,352 RZ（B0I p7 S2）、838,693,580 RZ（Full PF S2）。元の固定配分・条件付きcompiler閉包の資格を保持する。

対象はLinear H4 1.00 Å、STO-3G、保存tol-only DF10・H/state/snapshot、Full-H pure RTE（one-bodyを含む全非scalar H、決定論headなし）、finite K2、κ2、q grouping=2^mの元schedule。scalar −0.33147781341681615をtotal Tで一回、new `CONTROLLED_ZZ_CX_CRZ_CX_V1`、symmetric-directional、測定付きfull controlled Hadamard wrapper、P0。Qiskit1.3.0、rz/sx/x/cx、opt1、seed17。Full-H qをPF回数とみなさず、PR-DF/prefix0やR448へ置換しない。

## 26軸の適用結果

入口は [run01/RESULT.json](run01/RESULT.json) の `rows[26]`、簡潔な表は [run01/AXES.csv](run01/AXES.csv)。X/Yを別行に保存し、physical ID、元/total configuration・candidate ID、axis、N、q/R、exact T、元係数・特徴量・B_join・ADMISSION selectorへ戻れる。

| 範囲 | 軸数 | 結果 |
|---|---:|---|
| 全26軸 | 26 | 固定SOURCE・physical/law/compiler・特徴量単位のmetadata照合PASS |
| m0 X/Y | 2 | 非負性gate不適合。signed値をそのまま保存 |
| m1〜m12 X/Y | 24 | 非負のモデル参照値。全law価格の資格を与えない |
| m5/m6 X/Y | 4 | b516のexact保存予測を転記。predict呼出し0 |
| その他11 round X/Y | 22 | 元特徴量・元係数からexact predictを各一回 |

不適合の両軸は physical ID `FULL_H_2a9a68a11d08af234a8c`、m0、T=0.1、q=1、R=9。

`C_ref(m0,X)=C_ref(m0,Y)=−3078724851127 / 2514388957 ≈ −1224.4425599133746 RZ`。

これは低Rでの固定加法モデルの不適合であり、物理回路が負の価格を持つことやFull-H方式の科学的FAILを意味しない。残り24軸の非負性も、較正domainの移送精度・期待価格の資格を証明しない。D7の5非configuration較正cell、paired P2R566依存、m5/m6独立holdoutという役割は不変。

## 全額gateとΔ

全26軸が揃わないため、全額の和は評価していない。

| PF基準 | 保存T_PF | 全13 round T_ref | D_PF=T_PF−T_ref | 全額順位反転境界 |
|---|---:|---|---|---|
| B0I p7 S2 | 593,555,352 | null | null | null |
| Full PF S2 | 838,693,580 | null | null | null |

したがって、理想finite-K2 lawの未知総額とT_refとの差Δに対する数値境界を提示しない。一般には有効なT_refがあればΔ<D_PFでκ2が低費用、等号で同額、Δ>D_PFで高費用だが、このrunにはその有効な全額がない。

## m6の独立した部分額診断

m6はphysical ID `FULL_H_4ec8d73571760a6830b8`、T=6.4、q=64、R=566、N_X=N_Y=57。

元保存値：`P_6X=P_6Y=111815248527103419 / 20115111656 RZ`。

`T_ref,m6=57(P_6X+P_6Y)=6373469166044894883 / 10057555828 RZ`。

| 基準 | exact `T_PF−T_ref,m6` | 表示RZ | モデル部分額と基準 |
|---|---|---:|---|
| B0I p7 S2 | −403753076296703427 / 10057555828 | −40,144,254.0515325… | m6部分額が上 |
| Full PF S2 | 2061738337390289357 / 10057555828 | ＋204,993,973.9484675… | m6部分額が下 |

この表の差は**部分額診断**であり、nullとした全額のD_PFやΔ境界の代わりではない。元1 pathの実測をモデルの代わりに使っていない。

## 次の片側順位判断に必要な証拠

全roundの実compiled価格は非負である。将来、同じideal law・回路・compiler条件のm6母期待価格に対する下界 `L_6X,L_6Y` が得られ、`57(L_6X+L_6Y)>T_PF` を証明できれば、他12 roundを取得せずにκ2の固定shot期待総額がそのPF基準を上回ると判断できる。

必要な下界の和の閾値は、p7 S2に対して `197851784/19 ≈ 10,413,251.7894737 RZ`、Full PF S2に対して `838693580/57 ≈ 14,713,922.4561404 RZ`。これは母下界に課す条件であって、現在その下界が得られているという意味ではない。

- p7 S2については、モデルm6部分額が基準を超えるため、**m6の母期待部分和に対する片側下界検証**を次の優先候補にできる。保存モデル値も有限1 path値もその下界ではなく、κ2の敗北は認定しない。
- Full PF S2については、現在のm6モデル部分額は基準未満である。m6下界が上記のより高い閾値を超える証拠、または別roundを含む母期待部分和の証拠が必要。現比較からκ2がFull PFより低費用とも結論できない。
- 下界は、全supportを対象にした正当な解析、または事前固定した標本law・独立性・停止/信頼規則を満たす統計的片側評価と、その適用条件が必要。n=1の保存観測や小さいholdout誤差から分散/CI/2%保証を作らない。
- Bと比較する期待値はideal finite-K2 law I。D7はPCG64/CDF有限cohort Qであり、I/Qの距離・価格全support変動幅・稀なorder2・共有係数/軸誤差が未資格。PR-DF F0の上界はFull-Hへ移送しない。
- 全26軸の数値的期待価格へ進む場合は、まずm0低Rの適切な価格証拠・モデル適用範囲が必要。今回の負値を事後clamp・refitして回避しない。長q/Rへの移送精度も別に必要。

追加科学実行・lower-bound policyの採択・path数・compile件数は今回確定/実行しない。上の問いに沿った限定案を後続指示で具体化する。D7元モデルは `CALIBRATED_UNVALIDATED/qualified=false`、holdoutは `DESCRIPTIVE_COMPLETE/NOT_ASSESSED_NO_ADOPTED_TOLERANCE/tolerance=null` のまま。

## 実行・技術確認・STOP

新6＋既存保存受入回帰6の人工テスト12件PASS（[TESTS.txt](TESTS.txt)）。保存結果の再読取で26軸・22新評価/4保存reuse・全額null・SOURCE不変を確認（[CHECKS.json](CHECKS.json)、科学再実行ではない）。実F1はfresh run01一回、終了code0、段階log [PROGRESS.jsonl](run01/PROGRESS.jsonl) の完了elapsed約0.168秒。これはこのscalar診断だけの実測であり、回路compileの高速化倍率ではない。Git JSON cache・元APIを一回読み、qRのevent列を生成しない。

新規path/build/compile/fit/Q3はすべて0。正式E[C]/CI/G/winnerはnull。SOURCE・元receipt・旧STOP・係数・モデル資格を保全した。§10.5の重複照合範囲と再利用は [SOURCE.md](SOURCE.md) に記録。作者自身の人工/技術確認であり独立レビューではない。

ロードマップ差分：Full-H F1完了、24非負参照・m0両軸不適合、全13 roundモデル額はnull。m6参照部分額の対p7/Full S2比較を取得したが母下界ではない。次の優先候補はp7基準に対するm6母期待部分和の片側証拠。ロードマップ本体は統合担当へ委ねる。公開・remote照合後mandatory STOP。
