# H6固定U solverの限定診断と修正候補（2026-11-06）

対象はH6直鎖1.00 Å/STO-3G、保存CDF14 C0、元順LD4、PR/ENTRY、strength .1の固定Uだけ。旧stablemetric runのSOURCE・入力・raw・STOPとretry/resume=0を維持し、fresh診断を行った。outer U更新、DF norm、PF/RTE、回路/native価格、頑健性6本の再実行は行っていない。

## 結論

**修正候補はまだ本計算へ採用できない。** 凸形式＋Newton仕上げではPRの固定UのJ stationarityと最新refinement対のU安定性が元の閾値を満たした。ENTRYは不合格であり、3回目のrefinementはSciPyのArmijo探索内のゼロ除算で停止した。solverの`Solved`や座標勾配の小ささを科学的な受理へ置き換えない。

[診断要約と配列hash](../../artifacts/research_c_h6_trustmodel/2026-11-06/summary.json)に全結果を記録した。元の固定契約を変更せず、各限定診断のSOURCE・scope・STOPを別々に保存した。これらは科学性能・費用の評価ではない。

## trust-region停滞の再現

`diagnose_research_c_h6_trustmodel.py`はSciPy1.14.1の既存trust-region/subproblemを再利用し、radius、予測減少、実減少、ratioを観測した。両armについてC0で100反復、旧保存最終点で1反復を実行した。

PRの2回目の提案は予測減少.205に対し実減少−415.6、ENTRYは.217に対し−195.5。100反復付近ではradiusがPR4.77e-7、ENTRY2.38e-7へ縮小した。epsilon=1e-8の平滑化項の曲率は非常に局所的で、差分幅を1e-10まで下げるとHessian-vectorの相対差は概ね1e-6〜1e-5まで下がった。大きい差分幅での不一致を、そのまま解析Hessianの数式誤りとは判断しない。

## 同じ目的を保った探索候補

1. `c_h6_factored_newton.py`：解析Hessianを残差因子Dで表し、`D.T@D`の正規方程式を組まず、列スケーリング＋full QRでNewton方向を求める。勾配の正規方程式右辺も避け、元の残差ベクトルから直接解く。既存SciPy Armijoを使用。ridge、方向の切捨て、係数のhard boundは加えない。丸め後のJが動かない場合は探索失敗として返し、物理gateに委ねる。
2. `c_h6_conic_diagnostic.py`：`t_i >= hypot(c_i, epsilon_i)/M0`、`m=sum(t)`の二次錐制約により、同じ`loss/(2f0)+gamma*m²`を解く。gamma>0なので最適点でこのepigraphは元の目的に一致する。大域段階後、同じ元の目的のQR Newtonで仕上げる。元のepsilon・J/U gate・許容差を維持する。

Clarabel0.11.1は公式PyPI wheelのSHA256を照合し、worktree専用`.diagnostic-packages/`へ配置した。旧fit/eval環境は変更していない。[配置記録](../../artifacts/research_c_h6_trustmodel/2026-11-06/technical/conic_package.json)／[公式の二次錐接口](https://clarabel.org/stable/python/getting_started_py/)。既存numpy1.26.4/scipy1.14.1を利用する。小さな独立問題の診断をarm別2 process、各BLAS/OpenMP1 threadで実行し、共有環境の設定には触れていない。

## 実H6の限定結果

QR Newton単独の1,000反復はPRのJ勾配13.44、ENTRY615.18でともに不合格。ENTRYではJ normが約3.00e6へ増大し、binary64/extendedの値照合も不合格だった。この候補を採用しない。

凸形式＋Newton仕上げの結果は次のとおり。

| arm | 固定Uブロック | 元のJ gateで用いる最大勾配 | 元の閾値 | 最終2回のU勾配差 | 状態 |
|---|---:|---:|---:|---:|---|
| PR | 3 | 9.0833e-7（第3回） | 1e-5 | 1.7263e-7（閾値1e-6） | 当該2 gateを満たす |
| ENTRY | 2完了、第3回で例外 | 2.0473e-3（第2回） | 1e-5 | 第3回未完了 | 不合格・STOP |

PRのcone反復数は各24、Newton仕上げは10/11/8、3ブロック全体約35.45秒。J normは約35.31。値、返却jac/独立pullback、binary64/extended値、独立AD/forward-pullbackの既存検査はPASS。これは固定Uでの進展であり、outerの64更新、全strength、全下流条件の資格を意味しない。

ENTRYはcone19/20反復で`Solved`を返したが、J norm約7.77e4、extended J勾配約7e-4、binary64 J勾配約2e-3で元のgateを満たさなかった。例外は第3回Newton仕上げの`scalar_search_armijo`の補間計算で発生した。旧runは再開しない。後続の技術修正は別SOURCEで行う。

## 人工検証と残る限界

新QRテスト6件＋既存conditioning回帰15件＝21件PASS。二次錐の目的/制約恒等式と小次元問題は4件PASS。一方、n=6/rank14/LD4、近接head、非零tail、微小残差の人工例2件は元のJ閾値を満たせず、**2件FAILを保持**する。J normは約4.56e7/7.33e7となり、丸め後の勾配は約.073/.083だった。[当時の検証XML](../../artifacts/research_c_h6_trustmodel/2026-11-06/technical/conic_tests.xml)。許容差を緩めてPASSにしていない。

大きな相殺係数と丸めの影響は重要だが、binary64での適格点の存在が不可能だと証明したわけではない。高精度化だけで全問題が解決するとも判断しない。ENTRYの係数表現・残差/勾配の精度をさらに切り分ける必要がある。固定契約のJ dtype、受理gate、科学条件を変更する案は、具体的な差分と影響を提示し、明示許可なしで採用しない。

## 次の段階

Codex：Armijoの技術的例外を別SOURCEで失敗扱いへ修正し、人工回帰を行う。ENTRYの適格化は未完了として残す。

ユーザー：この証拠を踏まえ、ENTRYの高精度表現を含む追加の限定診断を行うか判断する。現状で頑健性6本を再投入しても、両armが適格になる根拠はない。重要科学レビューは実施していない。
