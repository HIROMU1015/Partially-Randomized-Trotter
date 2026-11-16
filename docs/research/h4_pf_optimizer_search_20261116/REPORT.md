# H4方式別PF探索：Full PF完了・B0I p7継続中

Full PFは**244新規Q3で終了**。確認済み最安は **480,258,064.00 RZ**、M5・要求幅16h（h=0.1、ΔPF=1.6 Ha⁻¹）。B0I p7は別jobで継続している。本REPORTはFull PFの先行公開であり、二方式の終了報告ではない。

| Full PF | 開始時 | 終了時 |
|---|---|---|
| 公式・要求幅 | S2・16h/7 | M5・16h |
| X/Y各軸 m0..m12 | [59, 55, 51, 47, 43, 39, 35, 31, 27, 23, 19, 13, 2] | [20, 25, 26, 24, 16, 20, 17, 15, 12, 11, 8, 7, 2] |
| 条件付きcompiled RZ | 559,457,386.00 | 480,258,064.00 |
| MSE下界 Ha² | 2.402471387794261e-06 | 2.5395541121403294e-06 |
| MSE上界 Ha² | 2.402471387794261e-06 | 2.5395541123957259e-06 |

終了点のX/Y列は同じ。開始時からの削減は 79,199,322.00 RZ（14.15645302%）。判定には十進εE=0.00159360010199040のexact二乗τを使用した。新規244件は適合23/不適合221、UNKNOWN/held 0。保存11件を開始時に再利用し、探索中の同一job再取得は0件。原API・partials・資源・identity・process exit・既存validatorの結果を保全し、244件の保存/gate/閾値整合を確認した。選択点のAPI returnとRESULT.recordは完全一致。

## 探索範囲と停止

固定最適化器 `53bb0a1d7b1da70644da1c22552224291205af78`。専用checkoutで実行し、B2の進行中checkoutは変更していない。方法間は並列、方法内は固定FINITE_LOCAL_DESCENT_V1の順序で結果取込→次候補を選んだ。UNLIMITEDの有限規則（既定100000 step）を採用し、実際は244 stepで停止した。科学call数で打ち切っていない。

停止理由は **NO_IMPROVING_PROPOSAL_INPUTS_MISSING**。保存signal/26価格/Q3経路が揃う3構成では固定規則内の改善案が尽きたが、範囲内17構成に不足がある。大域最適、全幅の最適、十分Nとは呼ばない。不足は[NEEDS](run01/FULL_PF/NEEDS.json)と[SUMMARY](SUMMARY.json)に保持。旧PF6の部分保存状態や旧STOPは変更せず、同一配分を再評価していない。

## 固定PR-Pauli点との比較

PR-Pauli LD5 FULL_PF__B_PRICEの保存点（Q3接続 `1c4a83d1b`、全26軸MODEL価格 `b8bcdf97b`、C catalog `417a73e73`）は 541,743,905.57 RZ。モデルは C_ref=A+Hq+EqR、qualified_for_candidate=falseのまま。新しいPRのsignal/Q3/価格取得・再fitは行っていない。

この固定点に対し、探索後Full PFの条件付き費用は 61,485,841.57 RZ、11.34961389%低い。固定された価格モデルと評価済み配分での算術比較であり、PRを同等に最適化した方式全体の優劣や真のE[C]の保証ではない。Full PFにも直接/条件付き閉包価格の資格・未取得CIを引き継ぐ。

H4/1.00 Å/STO-3G/DF10・sector(2,2)、h=0.1/M12、全13round、保存full DF-H solver proxy、Z1+T1 total decoder、全事象二乗loss/IID二項・P=0。科学的ground/実sampler移送/厳密数値certificateは未認証。AのδEを加算せず、十分N/global RMSE/G_qualified/winnerはnull。

## 保存と引渡し

原結果snapshot：`4b28d35da197c7a8dcde0a8d60a6a0f9ed917d75`、本directoryの`run01/FULL_PF/`。API_RETURN・RESULT・coarse/lemma/DP partials・source・resource・supervisionを全244件保存。[VERIFICATION_FULL_PF.json](VERIFICATION_FULL_PF.json)と[機械可読C索引](C_RESULT_JOIN.json)参照。Full PF計算wallは14817.9秒（約4.12時間）。CPU/RSSは同verificationに測定範囲付きで保存。

B0I p7の進捗・結果は別のrun01/TRUNCATED_PFに保存されており、終了後に追補する。ロードマップ本体は統合担当へ委ねる。
