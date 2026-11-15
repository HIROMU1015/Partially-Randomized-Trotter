# C→B/D1/統合担当：Full-H κ2 F1結果

**全13 roundのモデル参照総額・反転境界はnull。m0-X/Yの負値を補正せずSTOP。**

- branch：`codex/m5-m4-receipt-admission-20261114`。
- 実行SOURCE：`417526460f22f9f49dfcf166442740c17a00105b`。
- 機械可読正本：[run01/RESULT.json](run01/RESULT.json)。26軸：[run01/AXES.csv](run01/AXES.csv)。[REPORT](REPORT.md)。
- 元26軸・B_join：61f249e6eのADMISSION。モデルb5165f11b、D1独立holdout06956c643、B1.5倍Q3配分6c5569c34、PF基準1002f1dfeを保持。

## 結合キー・算出結果

`rows` の `(original_configuration_id, physical_id, m, axis)` から `admission_source` / `B_join` / `model_source` へ戻る。total decoder view・N・allocation ID・元law/profileも保持。X/Yは別行、Full-H grouping qとPF回数を混同しない。

- 元exact predictを22回、保存m5/m6の4予測を再利用。新fit0。
- SOURCE/構成/特徴量のmetadata gateは26軸通過。24軸は非負の `MODEL_SCENARIO_ONLY`、m0-X/Yは `OUT_OF_DOMAIN_NEGATIVE_MODEL_REFERENCE`。
- m0のphysical ID `FULL_H_2a9a68a11d08af234a8c`、q1/R9。両軸 `C_ref=−3078724851127/2514388957`。0丸め・別軸/実測置換なし。
- `T_ref=null`、両基準の全額 `D_PF=null`、Δによる全額の順位反転境界もnull。

## m6モデル部分額（母下界ではない）

`57(P_6X+P_6Y)=6373469166044894883/10057555828 ≈ 633,699,606.0515325 RZ`。

| 保存PF基準 | `T_PF−m6参照部分額`（表示RZ） |
|---|---:|
| B0I p7 S2：593,555,352 | −40,144,254.0515325… |
| Full PF S2：838,693,580 | ＋204,993,973.9484675… |

exact差・基準locator・m6保存予測locatorは `RESULT.json.m6`。有限1 pathの観測値は参照部分額へ代入していない。

次の有用な問いは、**m6のideal-law母期待部分和の下界だけでp7 S2基準を超えるか**。正当な `L_6X+L_6Y>197851784/19` が得られれば非負の他roundを使わずに片側比較可能。Full PF S2にはより高い `838693580/57` の閾値、または別roundの下界が必要。現モデル部分額・保存1 pathは下界ではなく、現在の順位は未判定。

母law Iとsampler Q、有限cohort・稀なevent・共有誤差を扱う証拠と事前固定した判断法が必要。追加標本数・compileやCI policyをここで採択せず、後続の有限案・明示承認へ戻す。D7の2モデル・4比較の旧status/qualification/toleranceは変更しない。

## ロードマップ差分とSTOP

「F1完了、26軸評価中m0両軸負値、全モデル額はnull。m6部分額をPF2基準へ比較したが母下界ではない。次はp7基準に対するm6母期待部分和の片側証拠が優先候補」。本体は統合担当が更新する。

人工/回帰12 PASS、実run01一回・exit0。path/build/compile/fit/Q3=0、正式E[C]/CI/G/winnerはnull。固定成果公開・remote照合後mandatory STOP。
