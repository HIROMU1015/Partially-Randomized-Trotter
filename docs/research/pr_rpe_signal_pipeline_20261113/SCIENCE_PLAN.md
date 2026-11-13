# 未実行：直接expm PR providerの最初の有限科学実行案

2026-11-13 JST。今回は技術実装・人工testまで。以下の実H4処理は**別途ユーザーがmenu・SOURCE・範囲を承認した場合だけ**実施する。承認証明書・新grantは要求しない。旧grant・STOPを再利用/変更しない。

## 1. 固定する条件

対象は保存Linear H4 1.00 Å/STO-3G/tol-only DF10/sector(2,2)/dimension36、同H/state/snapshot・fragment順・scalar・P0。全保存DF-Hをtargetとし、元分子Hamiltonianの厳密ground精度や保存stateの厳密ground同定を保証しない。

最初の**単一configuration**について、ユーザーがL_D、S2/M5、ΔPF、K2/4/6、exactκ、h/M、元prior/decoder/RMSE予算を固定する。Bが既存APIで全scheduleを生成し、保存λ/scalar SOURCEを含むcanonical IDを確定する。今回L_Dやκの値、初回menuは選ばない。

元設計821c5c1の案を維持する場合は、baselineのprefix1/S2/Δ.1/K2、元exactκ/h.1/M12/誤差契約と、別prefix一つp∈{2,3,5,7,9}を同条件で比較する。ただし**これは二構成の拡張案**であり、単一configurationの承認に自動的に含まれない。Track Aのwinner・L7/8/9・m12だけの案を正式選択と扱わない。

## 2. 実行前の保存再利用

1. 今回公開する数値SOURCEを固定し、B validatorと`pr_rpe_signal_plan`で全M+1roundを照合する。candidateとnumerical recipeの対応を保存する。
2. 既存INDEXのprefix-specific λ/scalar・input hashとbba物理契約の保存reference pairを使う。新λ fit・reference action・H/state生成は0。SOURCE/field欠測でSTOPし、範囲外計算を追加しない。
3. 旧169dの全13meanは旧spectral numerical SOURCEの証拠として保全する。今回のdirect-head結果へ再ラベルしない。旧結果との新しい数値比較やallowance受入を行う場合は、その再解析も承認範囲へ明記する。
4. 保存meanの受入とfresh mean生成は別経路。今回のbatchは保存hitを自動再解析しないため、新規部分の実行/保存部分の受入を計画上で分ける。旧m3/m4の観測norm欠測を0で埋めない。

## 3. 最小の全round signal実行

M12をユーザーが維持した場合の上限。Mを変える場合は13をM+1に置き換える。

| 処理 | 単一configuration | 二構成比較を別途承認する場合 |
|---|---:|---:|
| round-candidate評価 | 13 | 26 |
| raw/corrected/exact-tailのcomplex出力 | 39（各候補3） | 78 |
| finite normalizerのpiece評価 | S2は最大13、M5は最大143 | S2二構成は最大26。一般形はΣ(M+1)×piece数 |
| 新reference action | 0 | 0 |
| 新SCF/DF/state/ground solve | 0 | 0 |
| 新nominal/十分N・decoder sampling・C・G | 0 | 0 |

同prefix/δ/timeのhead・exact-tail expmはcacheし、finite polynomial/event power・outer powerを使う。巨大event列・trajectoryは生成しない。全roundを一つの有限batchとして処理し、原則SOURCE/数値失敗時に残りをSTOPする。独立configurationの並列化が必要なら別context/cache/outputを用い、他担当のworktree/processや共有設定へ触れない。

取得値は三mean・reference、logB/B/attenuation、piece allocation、signal-space bias、gate/closure/empirical allowance、source/binding。local phase/radius・N政策を後で評価する場合も別の明示scopeとして扱い、signal-only実行から拡張しない。

## 4. 成功・STOP条件

全roundが`LOCAL_FINITE_MEAN_NOT_RMSE_QUALIFIED`で、B/physical identity・入力SOURCE・保存pair、元norm/unitarity/closure条件とnormalizer精度が成立したら、全round physical input取得完了とする。比較候補の優劣、十分N、global RMSE、最適L/κ、G winnerは認定しない。

SOURCE/λ/scalar/law/profile/候補の不一致、必須field欠測、pair gate失敗、normalizer丸め未解決/overflow/underflow、非有限、norm/unitarity/closure失敗では、取得済み値・例外/残差・未開始roundを保存してmandatory STOP。自動retry・閾値緩和・別R/別prefixへの差替え・新reference取得なし。

## 5. その後の分担

**B**がraw分布と同RMSE契約に基づく全roundのshot/decoder資格を検証し、**価格担当**が同physical scheduleのdirect/model C coverageを返す。AのsignalだけでN×Cを最適化しない。**ユーザー**が最初に承認する最小事項は、一つの固定configurationとその全round signal-only有限範囲。二構成・旧保存値比較・N/RMSE・価格・候補探索は必要になった段階で別途指定する。
