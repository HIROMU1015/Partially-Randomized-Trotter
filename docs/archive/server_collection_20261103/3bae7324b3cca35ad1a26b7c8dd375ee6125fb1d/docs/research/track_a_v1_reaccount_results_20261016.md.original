# V1：5精度の決定論費用再評価・完了STOP

**通常signal bias・宣言した固定費用モデルの条件下では、細格子を加えるとε=.05でPRの点費用優位が反転した。残る4精度ではPRが約10–35%低費用だった。**
新しい最良決定論候補はいずれもモデル推定であり、全prefixでの費用誤差保証・母集団最良・厳密bias保証は得ていない。保守的な解釈に必要な非相殺診断の全domain最良値は、旧B0+Iのsigned分解未保存により未解決のまま。

[完全な4320会計行・60比較・終了監査](../../artifacts/resource_applicability/track_a_v1_reaccount_result/2026-10-16/README.md)、[事前固定した実行契約](track_a_v1_reaccount_readiness_20261016.md)、[Phase B全4軸PASS](track_a_v1_phase_b_run01_results_20261016.md)。
利用者の「その方針で進めて」に基づく専用one-shotをSOURCE `2261a3ad1d6c23521474b31204572a6c3a51258b` で実行した。通常会計と保存値の2170件照合が全て一致。native exit0・全回収・`COMPLETE_MANDATORY_STOP`で終了した。新signal/primitive/native回路/compile/fit/sampling/V2実行は0。

## 条件と比較集合

linear H6、間隔1.00Å、STO-3G、tol-only DF rank19、12 modes、sector400、保存Hamiltonian/state、T=.8、q=1/2/4/8（δ=.8/.4/.2/.1）。ε=.05/.01/.005/.001/.0005、経験的allowance倍率1/10、α_axis=.025・既存shot算術順序・u/headroom≤.01を保持した。
主費用はP=0、shot重み付きcompiled measured full-wrapper RZの点費用。保存compiler条件はQiskit1.3.0・rz/sx/x/cx・opt1・seed17・topologyなし・symmetric_directional、hash `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。今回compilerを呼んでいない。

旧比較集合32候補、S2格子追加後96候補、さらに打切りS4追加後136候補を別々に選択した。全scopeにfull S2/S4を保持する。B0/B0+IのS2追加prefix6–14（旧10は再利用）、S4追加prefix11–15、各q1/2/4/8。厳密な有理数のweighted RZ、次にcell ID辞書順で選び、許容差tieは使わない。
PR主比較は既存確認union8・各32 paired標本。探索72候補・各8は別表へ保存し、確認標本と混ぜない。prefix3/4の外側S2診断は正式な費用候補に追加していない。

## 5精度の主結果

以下はallowance倍率1の通常bias会計。倍率10も最良候補と優劣の符号は同じで、完全な値は[CSV](../../artifacts/resource_applicability/track_a_v1_reaccount_result/2026-10-16/comparisons.csv)に保存した。
「PR削減率」は`100*(1-G_PR/G_det)`。負値はPRの方が高費用であることを表す。

|ε|旧比較集合のPR削減率|S2追加後|S4も追加後|
|---:|---:|---:|---:|
|.05|29.35%|−13.04%|−13.04%|
|.01|49.69%|30.89%|30.89%|
|.005|53.27%|34.97%|34.97%|
|.001|56.14%|31.39%|31.17%|
|.0005|50.37%|15.09%|10.03%|

|ε|S4追加後の最良決定論候補|点費用G（丸め表示）|費用の種類|
|---:|---|---:|---|
|.05|B0+I p7 q1 S2|3.56889e8|固定モデル推定|
|.01|B0 p11 q2 S2|2.82872e10|固定モデル推定|
|.005|B0 p11 q4 S2|1.56588e11|固定モデル推定|
|.001|B0+I p11 q2 S4|7.47720e12|固定モデル推定|
|.0005|B0+I p11 q4 S4|3.62659e13|固定モデル推定|

PRの最良は順にB2 p5 q1 r4 K4、p5 q2 r4 K2、p5 q4 r4 K2、p5 q8 r2 K2、p5 q8 r2 K2。旧比較と同じ保存確認候補から選ばれた。標本平均とpaired標本SEを保存したが、選択補正済み信頼区間や母平均の最良保証は作成していない。

## 格子とS4の寄与、報告trigger

[格子効果の全表](../../artifacts/resource_applicability/track_a_v1_reaccount_result/2026-10-16/grid_effects.csv)。倍率1の最良決定論費用の変化は次のとおり。

|ε|S2格子を細かくした費用削減|S4追加による追加の費用削減|
|---:|---:|---:|
|.05|37.50%|0%|
|.01|27.20%|0%|
|.005|28.15%|0%|
|.001|36.07%|0.322%|
|.0005|41.55%|5.617%|

ε=.001/.0005では最良がB0 p11 q8 S2からB0+I p11 q2/q4 S4へ変わった。ε=.001の差は小さく、モデル誤差に対する順位の頑健性を認定していない。

事前登録した報告triggerを満たした。緩い精度ε=.05でPR優位の符号が変わり、PR削減率は42.39 percentage points低下した。中間精度ε=.01/.005/.001の低下はそれぞれ18.80/18.31/24.97 percentage pointsで、10 points以上の条件に該当する。両allowance倍率で同じtriggerが成立した。
これは結果を明記する条件であり、追加compile・自動再fit・新たな科学比較条件の採択を許すものではない。

## 非相殺診断とモデルの限界

保存signed分解の各軸絶対値和を使う経験的な非相殺診断も計算した。旧B0+Iの12候補は分解が未保存のため、全scope・全精度の決定論最良とPR比は`UNRESOLVED_FULL_DOMAIN`/null。欠測候補を除いた最良を「保守的な最良」として報告していない。厳密なbias上界・これに基づく適格性もnullのまま。

例えば通常biasでε=.05の最良となったB0+I p7q1 S2は非相殺診断でも適格だが、点費用は通常の約3.57e8から診断の約4.88e9へ増える。この差はbias相殺に対する感度を示す。非相殺診断の全domain最良と比較していないため、この値からPRの保守的優劣を決めない。細格子での相殺を一般的な精度改善と解釈しない。

固定model SHA `406c6f36f93dac317f9709e68b042c4563edf5ec4b06219e1b60de660b695211` は変えていない。保存実測を優先し、Phase Bの4軸も実測として使った。他の追加候補は固定S2式または別S4合成仮説で推定した。生成したmodel軸200件のうち4件を実測で置換し、最終のmodel行は196件。実測とmodel推定を同じ来歴にしていない。

特にε=.05のp7は、fitに用いたp10/15のアンカー範囲外のprefix外挿である。登録S4の4軸PASSは、このS2 p7の費用や全候補の誤差2%未満を証明しない。旧B0 binary profile未在庫・旧compiler出力bit同値未確認も維持する。今回の通常biasでの優劣はこの宣言モデルに条件付きの点結果であり、一般的なPR優位/非優位の証明ではない。

## 終了・再現・保全

保存JSON/CSVを一度parseし、各精度/biasの会計を全scopeで共有した。accounting実評価4040回、保存4320行（探索PRと確認PRを分離）、60比較。旧104候補×10条件、Phase A追加104候補×10条件、確認PR8候補×10条件、旧最良比較10条件の計2170件が一致した。
既存V1関連71人工testsがpinned PythonでPASS。[保存結果の再照合script](../../artifacts/resource_applicability/track_a_v1_reaccount_result/2026-10-16/verify_public_results.py)は、gzipの完全復元、4320行のweighted費用、固定model式・実測優先、全60最良と比、CSV/JSON・終了状態の整合をstdlibのみで確認した。primary agentによるlocal照合であり、外部再現・CI・別担当の独立レビューではない。

監視・回収を含むwall6.024秒、会計と実行後byte確認3.264秒。native wait0、driver不在、all_reaped=true、first_stop=null、cleanup_errors=[]。peak総RSS約126MiB、監視最大間隔1.003396秒、stderr0 bytes。
終了後にSOURCE19ファイル、runtime34,273ファイル、入力26receipt、既存grant/ledger/carry等653ファイルをhash照合し、全て不変だった。H5・V2・Phase A/Bのrawとgrant/ledgerを保全し、新しい科学計算・compile・fit・samplingを起動していない。

結果・実装・監査をcommit・pushし、remote SHAを確認してmandatory STOPする。上記triggerを含めた科学的な採否や追加測定は、次の明示指示で判断する。V2は保留を維持する。
