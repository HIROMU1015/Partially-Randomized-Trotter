# H6数値条件修正とfresh再実行（2026-11-06）

基点commit `30c6f47529d708a0dd1f67818fb455325b084fc8`。ユーザーの「修正を行って再実行してよい」に基づく数値処理修正。同じH6・1.00 Å/STO-3G、CDF14・元順LD4・保存C0、PR/ENTRY×strength .03/.1/.3、64受理更新/512trial、selection32/final128、r32/64/128、subset26/800入力/full8192出力、218350 unitsを維持する。旧runのretry/resume/STOPを再開しない。

[新固定契約](research_c_h6_conditionfix_contract_20261106.json)／[新SOURCE](../../artifacts/research_c_h6_conditionfix/2026-11-06/execution_preflight/SOURCE.json)／[人工検証・保存入力構造診断](../../artifacts/research_c_h6_conditionfix/2026-11-06/technical/)。

## 修正の理由と範囲

親runは全6本が初期点で不適格、受理更新0、checkpointなし。5本はJ基準未達、全6本でheadの構造資格とrefinement間U勾配差が不適格だった。18 inner solveの17回が1000反復上限に達した。保存結果・SOURCE・STOPは旧worktreeへ保全する。

- **head資格**：同じ物理係数写像をlongdoubleで組み、既存SVDの再構成・左右直交性の残差を独立に評価する。誤差guardはlongdoubleモデルのguardと観測残差の大きい方。従来の安全係数256/1024を維持する。保存C0の最小非gauge特異値約2.65e-9に対し、旧要求約1.51e-7、新しい残差確認後の要求約8.21e-11。実際に数値不確かさを検査した上で分離を判断し、単に旧閾値を下げない。真の追加nullは拒否する。厳密enclosure/certificateではない。
- **早期拒否**：構造が不適格ならJ solveより前に記録してUNQUALIFIEDを返す。失敗値・勾配をouterへ渡さない。setup/guard macroは1unitで記録し、optimizer未起動を3solves実施と数えない。
- **座標**：直交Uでの既知LD−1 gaugeについて開始代表を固定する。残る全nongauge方向を保持し、フルの可逆metricで前処理する。weak directionの切捨て・新penalty・rank/LD変更はない。gauge写像残差1e-10を検査し、実binary64 Jへの座標復元をlongdoubleで蓄積する。
- **inner solver**：既存の解析smooth-mass HessianとSciPy1.14.1 `trust-exact`をPR/ENTRY共通で使う。maxiter1000を保持し、内部gtol1e-13。固定した開始点からの目的差を安定に計算し、元のFも独立に保存・照合する。これは数値solverの変更であり、Newton-CGを実行したとは記録しない。
- **高速化**：探索方向用の同じ解析Hessian式をBLAS binary64で組む。人工問題で元のlongdouble式と照合する。最後の実z/Jに結び付いたprimalだけをcacheする。受理用primal/gradient、独立照合とstateは従来どおりlongdouble＋binary64で検査する。

**保持した受理条件**：元のJ閾値1e-5とengineering margin、refinement間U勾配差1e-6、値/勾配独立照合の許容差、DF予算.00019875 Ha、original-g κ/nuclear scalar、元outer Armijo・物理gate。raw solverのsuccess=falseを自動的なscientific FAIL/PASSへ置換しない。solver差分・raw centered fun/centering origin・元F・既知gauge数を保存する。

既知gaugeの固定は厳密直交の数学モデルで等価な代表の選択であり、追加の物理方向の削除ではない。有限精度では実配列の値/勾配・物理DF検査を維持する。新方式でも任意の悪条件点を必ず適格にできるとは主張しない。

## 再利用と接続

新 `c_h6_fit_conditioning.py`だけがhead/metric/innerの修正を提供し、既存の係数写像、smooth objectives、reconstruction/adjoint、precision state、guarded Oracle、Armijo controllerを利用する。`c_integrated_fit_worker.run_cell`は明示H6 backend指定がある場合だけ接続する。H4既定経路と旧solverは変更しない。汎用Oracleのearly precheckは任意backendのhookで、既定では従来どおり。

新契約・新SOURCE・新出力先を独立worktreeで定義し、旧契約ファイルと旧SOURCE manifestには上書きしない。実行入口は同じ `scripts/run_research_c_h6.py`で、新worktreeの既定契約を使う。両専用Pythonを明示する。

## 検証と限界

[verification.json](../../artifacts/research_c_h6_conditionfix/2026-11-06/technical/verification.json)に最終件数とSOURCEを記録する。人工の弱いhead方向/追加null、full-objective/gradient/Hessian照合、2軌道と6軌道rank14の実3solve/oracle/元J・U安定性gate、壊れたsolver勾配・SVDの拒否、既存H4/fit/RPC/native/state-action/統計の回帰を検査する。作者自身の人工・技術検証であり、重要科学レビューではない。

保存C0については構造の数値残差だけを新方式で診断し、両armで適格を確認した。科学最適化、PF/RTE、native価格の成功や優位をこの診断から推論しない。実分子のinner精度不足・refinement不安定・DF予算/RPE不適格は、そのままUNRESOLVEDとして残す。

## 実行と来歴

- SOURCE152ファイル、SHA256 `d177976d3a8d644f19bec393c2231eb08928fe4e27f5385716769cd80eecdf0c`。
- 新契約 `docs/research/research_c_h6_conditionfix_contract_20261106.json`。
- fresh output `artifacts/research_c_h6_independent_validation/2026-11-06-conditionfix/run01`。
- 旧input/C0 hash、fit/eval package pinsは親契約から不変。新SCF/積分/中立fitなし。
- 実行時にaffinity/quota/空きCPU/RAM/diskを再観測し、候補並列とBLAS/OpenMP/Aer threadsを協調させる。人工的な新しい資源上限は置かない。
- 旧STOPを保全し、新run内retry/resume=0。初回登録native batchの実測から時間/保存見積もりを更新する。価格未検査ラベル・条件付き統計資格は維持する。

科学実行は今回のユーザー指示で許可されたfresh scopeに限る。新SOURCEと契約を固定・公開照合してから起動し、正常終了/不適格/異常終了のいずれもpartial/STOPを保全する。重要レビューや別科学条件へ自動拡張しない。
