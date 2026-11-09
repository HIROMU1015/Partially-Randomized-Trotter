# Q3-A後：保証を閉じるための最小検証案

2026-11-10 JST。**未実行。保存科学入力を使う数値診断は別途の明示承認が必要。** 基準は本Q3-A公開commit（承認時に完全SHA指定）とQ2 `b6b74a3f0a73bb5984d8ee978705eb0e498b86c2`。[保証接続](GUARANTEE_CONNECTION.md)の未資格を対象とし、価格改善を得るための探索は行わない。

## 1. 先に戻す技術・科学判断

全prefix受理には、旧binary rankingの誤差ηと区間端点slackνが必要。人工PASSの追加だけで代替しない。GPT／ユーザーが次のrouteを判断する。

- **旧経路を保つ：** `math.atan2`、NumPyのwrap／exp／angle、candidate生成について、固定runtimeの一様数値契約を一次SOURCE・検証済みboundから与える。有限gridで近いという結果を、連続引数全域の証明にはしない。
- **必要最小の数値実装を改める：** 既存nearest-branch規則をinterval主体で選択する薄いadapterを検討。`(qm θhatprev−pm)/(2π)`の最近整数cellを包含で特定する案なら、S上で`|qm θhatprev−pm−2πj|≤3c<π`という既存marginを使える。c・成功event・precision・lawを変えず、Q1原本を保全し、Q0 decoder／readoutを再利用する。今回は実装していない。

後者を採用する判断とproduction SOURCEの固定が先である。前者のまま次項のcount監査だけをPASSしても、旧`np.exp/angle` rankingの一様保証は自動的に閉じない。

## 2. N1案：保存scheduleの有限count-domain監査

### 固定入力・SOURCE

- f2638a7845685a119619382b850441f5d8d33713のB′ `artifacts/pr_reference_bprime/2026-11-07/ERROR_RPE.json`、Reference `artifacts/pr_reference_transfer_qualification/2026-11-06/STAGE4_RPE.json`。**m0〜16の保存Nmをそのまま**使う。
- Q1固定f3705321の`AxisCounts`／`RoundCounts`／`_count_phase`／`_feedback_phase`、Q0固定34327b90の同じ192-bit constructor。Q3-Aの数学的marginと人工testを再利用する。
- Runtimeは保存CT1のPython3.12、mpmath1.3.0／NumPy1.26.4。上記の固定関数と保存metadataを使用し、物理event producerは呼ばない。production精度は192-bitを維持する。

### 有限call範囲

各arm、m0〜16について、整数`0≤kX,kY≤Nm`の全組から非ゼロvectorを監査する。偶数Nmの `(Nm/2,Nm/2)` はゼロとして別記し、都合よく成功としない。domain数は

`Σarm,m [(Nm+1)²−1{Nm even}]`

で固定され、今回この個数や保存Nの新しい集計は実行していない。重複した分母・count pairの同じarg評価は再利用できるが、両armのcoverageを明記する。

必要な出力は、192-bit arg包含・幅、binary atan2との周期対応、必要なslack bound。m16に限り既存`_feedback_phase`の戻り値と算術包含が共通cap内か確認する。解析的な一様boundが先に閉じた部分を、理由なく全件再計算しない。点oracleだけを包含の厳密証明と呼ばない。全17履歴の直積、E0を使った成功historyの選別、実測history生成は行わない。

### 判定・STOP

- containment・幅・period・算術capに不足があれば、入力identityとwitnessを保全してSTOP。
- 全非ゼロdomainを覆えても、それだけで全履歴受理・native誤差・340 shotsを採用しない。§1のranking routeと整合する条件付き結果として報告する。
- 固定domain完了・検算・公開でmandatory STOP。n・precision・count範囲を追加探索しない。

これは**保存scheduleを使った新しい科学的数値診断**になるため、現在は未承認・未実行。H4 cache・state・回路・samplingは必要ない。

## 3. N2案：保存Q0余裕とenergy-hull丸めの最小接続

必要条件は、Q0の点中心をQ1の区間中心へ移す際の余裕である。入力は34327b90の完全RESULTS／CHECKSに保存されたn340の2arm下側PASS bracket δlo、Q1のδsafe、N1または静的論証で得た中心幅κ・hull数値slack boundのみ。

新しいcallは**2armの保存bracketについてのexact rational余裕計算と、κ・丸めslackとの比較**に限定できる。[保証接続§6](GUARANTEE_CONNECTION.md#6-confidence-hullと有限precisionの残余)のclipped-affine幅補題を使い、旧全count反転・全n scan・margin二分探索を繰り返さない。旧δloの下側PASSを、実行可能な新価格や実RPE結果とみなさない。

判定は`κ t17+ωE t17+ωQ0 ≤ δlo−δsafe`を安全側で証明できるか。κ等が一様に未資格なら、0を代入せず未資格でSTOP。既存δsafe・phase cap・failure条件を変更しない。全17roundのshotやRPEを再計算しない。

この保存値の新しい再解析も、対象SOURCE・入力・boundの版を固定した別途の実行承認が必要。N1が未完ならN2を有効な新保証として採用しない。

## 4. Native signalの保証：価格より先に必要な条件

既存Compiler A／Qiskit1.3を維持し、P→U1→RZの同一parameter経路と、whole jointのcore誤差chargeが全対象contextで保持されることを確認する。Q2の16回路を再実行せず、pass traceと固定SOURCEを先に再利用する。accelerated translatorの全context移送を、経路の存在だけで認定しない。

追加compileが必要と判断した場合は、固定local template／context／angle familyと科学call数を**実行前に**列挙する。現時点では全context certificateのrouteが未確定なので、この項を承認済み有限compile scopeとは呼ばない。m17 wholeやtrajectory samplingで問題を迂回しない。

historyごとに同じreadout parameterと純粋角度差を全trajectoryへ渡せることも必要。trajectory依存のcompile誤差を、単一eφのモデルへ無条件にまとめない。

純粋角度差ならQ0 capへ加算可能。一般operator／raw diskなら単位を揃えた誤差再接続が別途必要で、radへ直接代入しない。新precision・compiler・word・law・phase-provider変更が必要ならSTOPして判断を戻す。

## 5. 費用接続・終了

保証条件が閉じるまでm17価格取得を始めない。次に必要となる価格SOURCEは[GUARANTEE_CONNECTION §8](GUARANTEE_CONNECTION.md#8-費用接続の境界)に記した旧m17会計・V6のbindingから準備する。旧m17相殺・provider emission・全lawを資格化せずにQ2の`+1 RZ`を全supportへ移さない。

将来の判定は資源jごとに`340 μQ,j+Aj < 626 μX,j+626 μY,j`。この計画では新μ・Aj・全round総費用・総Tを求めない。両armへ同じ改善機会を与える。

**次担当はGPT／ユーザー：** §1のroute、重要な保証の採否、N1／N2の実行順と固定scopeを判断し、必要な科学数値診断だけを明示承認する。**Codex：** 現Q3-Aの公開・remote照合後mandatory STOP。N1／N2／新compile／正式科学レビューを自動開始しない。
