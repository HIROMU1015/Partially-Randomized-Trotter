# 次の有限科学実行案：固定H4の全RPE shot統計

2026-11-13。**未実行・科学条件未採択**。本計画は新simulationの実行許可ではない。Aの物理作用・mean評価をBで再実装しない。

## 1. 最初に必要な入力と判断

正本B030f3f548のconfiguration `587d20f7ed906d25841615dc9746674af872675b895bd438c06d3e74abf0e911`を維持する。Linear H4/1Å/STO-3G・保存tol-only DF10/元H/state/snapshot、S2/L_D1/ΔPF=.1/finite K2、h=.1/M12/元exactκ、無energy shift、元prior・decoder・P0、εE=.00159360010199040 Ha、εqpe=εtrot=εE/2、minkowski/finite_B_diagnostic。

**Aが供給するもの**：m0..12のSOURCE付きraw/corrected complex mean、有限normalization logBと数値精度、bias/誤差範囲、state/元H target energy・reference誤差、law/実samplerとの対応、各round identity/law/元physical request ID。normalizationはAの値を受け、Bは新計算しない。nominal Nは現在m0=64、m1=60、m6=38だけ。残10roundのNを内挿・経験則で自動生成しない。

**ユーザーが判断するもの**：

1. 全26軸の固定shot配分とその出典。nominalを試すのか、別の有限menuを試すのか。技術準備はN探索・十分N認定を含まない。
2. 測定平均の数値caller。今回の集計器はexact整数和(2k−N)/Nをbinary64へ一回変換する。旧marginal Monte Carloの`2.0*k/N-1.0`とはbit単位で同一でない。decoder本体は同じだが、どちらの測定callerを科学検証対象にするかは固定が必要。今回これを正式採択していない。
3. shot・X/Y・round・simulation反復の独立性と対象law。独立binomialは明示IID policyに限る。PRで毎shot独立のtrajectoryを使うmodelと、cohort/trajectoryを再利用するmodelを混同しない。law距離・実装誤差は未取得ならunknown。
4. 未解決試行のloss。既存decoderはzero/tieでenergyを出さない。したがって現仕様のままでは全population energy RMSEを単一値として定義できない。下記の選択肢を決める。
5. finite-simulation精度・信頼水準・loss supportと、数値演算の扱い。経験的CIを数学的/実装上の全RMSE保証へ昇格しない。

## 2. decoderと未解決loss

固定するdecoderはA `decode_supplied_coordinates` →既存scalar `reconstruct_rpe_phase`。mごとatan2、初回[−π,π) wrap、q=2^mの全branch、前推定との円周距離が最小のbranch、zero又は最良/次点距離の差がabs_tol=1e−12以内なら未解決。energy=−final_phase/h。priorは現在の実装のbranch選択には使われておらず、Bはclip/alias移動を加えない。

未解決の選択肢：

- **現decoderを維持してloss未定義**：未解決率と条件付きRMSE、解決分の全試行MSE寄与だけを報告。全RMSEはnull。未解決0件の観測でも真の未解決確率0とはしない。
- **明示loss区間で感度評価**：未解決時のsquared lossをSOURCE付き区間[u_min,u_max]とする。energyを代入せず、全MSE/RMSEの区間だけを評価する。これは特定のenergy estimatorのRMSEを定義したことにはならない。
- **別の完全な推定器を科学的に定義**：未解決時の出力・prior/alias対応を定義し直す研究判断。既存decoder変更になるため今回実装せず、別SOURCE・許可で比較する。

energyを返すtrialについても、DECODEDは正しいbranchを意味しない。lossは(estimated energy−original-H target)^2という**実直線上の差**で、円周phase差でalias誤差を消さない。通常のprincipal energy範囲は(−π/h,π/h]だが、これは対象binary64全域の厳密enclosure証明ではない。狭いpriorを出力supportに使うには、decoderが出力をそこへ制限する証拠が別途必要。

## 3. 有限実行の順序と規模

最初の科学scope案は**1個の固定PR configuration × 全13round/26軸 × 1個の固定shot配分**。M/κ/方式を最適化しない。実Gを集計しない。独立反復i=1..nごと全26軸のplus countを生成し、元decoderを一回実行する。各軸のplus countは、raw信号座標μに対しBinomial(N,(1+μ)/2)。corrected meanやBを確率へ代入しない。RTE trajectory自体のsamplingではなく、供給meanを条件としたmarginal Hadamard測定simulationである。

**nは固定定数を先に決めず、lossと必要精度から設計する。** squared loss Lが全populationで0≤L≤U（Uの単位Ha²）を満たすSOURCEがあり、MSEの絶対半幅η、片側2本を合わせた失敗確率αを指定するなら、Hoeffdingから

\[
n\ge\left\lceil\frac{U^2}{2\eta^2}\log\frac{2}{\alpha}\right\rceil.
\]

実装`trial_budget`はこの**simulation反復数**だけを提案する。量子shot数Nを決める式ではない。未解決loss区間の両端の平均に上下片側を一つずつ適用し、union boundで合計αとする。MSE区間を非負で切り、平方根を取る。入力Uが無根拠・無限ならこのnを出さない。

精度の候補としてη=0.1 εE²、confidence=0.95を**検討用**に挙げるが、未採択。判定境界付近ではRMSE半幅との非線形対応を平方根区間で確認する。Uはprincipal energy範囲と元target/reference範囲、未解決loss区間から全到達出力を被覆して決める。H4のU/nを今回は数値評価していない。狭いprior幅を勝手にUへ使うとrare aliasの大lossを落とす。

未解決確率は既存Clopper–Pearson二側区間を使用する。0件でもupper>0。未解決確率p_*未満を零観測で調べる別の有限設計には、n≥log(α)/log(1−p_*)という式が使える。p_*は「十分成功率」の採択ではなく、loss寄与U·p_*と所望MSE精度から設定すべき入力。全RMSE CIと未解決率CIは個別confidenceであり、同時confidenceを主張するならαの配分を先に固定する。

worst-case Uによるnが非実用的なら、有限pilotで分散/rare構造を調べるか、検証済み有限格子/集中不等式を別設計する。今回のn式を小さな恣意的capへ置換しない。pilot/本試験の二段階、seed/反復数/停止規則は別の有限実行scopeに固定し、CIが通るまで無制限に追加しない。複数shot配分を比較する場合は有限menuと同時誤差管理が別途必要。

## 4. 出力・成功条件・STOP

出力：全trial数、zero/tie/その他status数、未解決率とCI、prior外推定数、解決分MSE寄与、解決条件付きRMSE、全試行RMSEの定義状態・interval、target/source/law、seed/runtime/count変換、入力の系統誤差証拠。未解決trialを保存し、勝手に除外しない。

技術成功条件：全入力/identity/sourceと26軸shot配分が一致し、全反復が成功又は未解決として会計され、確率model/精度/CI前提が満たされること。RMSEの点推定が目標未満であることを十分Nの証明にしない。明示された統計モデル/未解決loss/physical meansを条件としてupper CIが目標を下回る場合も、その条件付き統計結果として報告する。

STOP：欠測/混在SOURCE・law、非有限又は[−1,1]外raw座標、Nの非整数/サポート外、prior/decoder/count caller不一致、loss support超過、数値未解決の無断補完、未定義lossを全RMSEと呼ぶ状況、固定scope外の反復追加。未解決decoderそのものは捨てる理由ではなく、そのstatus/loss policyを記録する。入力破損以外の単なるCPU競合で他担当の計算を停止しない。

## 5. 科学資格と実Gまでの残件

Aのfinite-K/PF mean bias・保存数値state/target・normalization誤差・law距離・実装誤差はunknownを0にしない。plug-in simulationのCIは、供給raw meanの真値やsampling実装の正しさを保証しない。Minkowski配分には、各成分が同一targetと同一joint lawの有効L2上界であること、全branch/alias/未解決事象を含むことが必要。局所biasの値や平均同士の差だけではdecoderの不連続なloss移送を証明しない。

役割：**A**は物理mean/誤差/referenceと不足作用、**B**は固定shot統計/decoder/全loss評価、**C/D**はcompiled価格と残差。全13roundの十分N/global RMSEは未資格。PRの適格平均C、C−S/law/rare/長q/共分散、他3方式の正式menuとN/Cも未取得。Dの保存Full PF価格73 EXACTはその物理SOURCE分類であり、比較menu/RMSE採択ではない。

**ユーザー**が入力条件・loss policy・精度/信頼水準・有限反復scopeを判断して別途科学実行を許可し、**Codex B**がその範囲だけ実行する。今回は人工検証・計画・metadata受入までで、公開照合後mandatory STOP。
