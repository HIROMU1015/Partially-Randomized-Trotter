# H2/H4/H6高次PF：保存結果評価の実行契約

利用者の「次の作業に進んで大丈夫、ここまでで集計したH2,4,6について評価して（すでにしてあるものは再度しなくてよい）」を、完了pilotの限定数値評価に対する承認として記録する。新しいsampling/compile/signal/SCF/DF/state生成は0。既存SOURCE・raw・grant・ledger・marker・STOPを変更しない。

H2はcommit `2d415da2b09a73fab5580eded80e822fc44a25e1` の5精度統合比較・n8平均/SE・固定集合内の点選択を転記するだけで、再計算・再ランキング・独立確認をしない。H2はtol-only rank3、元H2のr/K格子とm5のPER_PIECE法則を保持し、H4/H6 outer32/n33とpoolしない。H2に存在しないYoshida4 PRを補わない。

新評価はH4/H6各12群×33本＝各396 trajectoryのDESIGN_PILOTだけ。H4は元SOURCE de5485d4の正常終了、H6は同元SOURCEの保存74本を独立受入した証拠と修正SOURCE 0ff0f704の新322本・正常終了を結合する。元H6全runのSTOPはそのまま。群・sample ID・input/preparation/compiler・paired両軸・native8gate・結果SHA・lawを受け入れ、部分成功や別cohortを混ぜない。既存の終了byte監査をSHAで結合し、QPY/NPZの再シミュレーションをしない。

H4/H6直鎖1 Å/STO-3G/tol-only1e-8/rank10・19、prefix3・5、T=.8、P=0、S2/Yoshida4/m5_best、q1/2/4/8、K4、ABS_TIME_PER_OUTER=32。既存の係数・compiler・scalar/control・CT01等は維持し、保存bias/適格性を再評価しない。主比較へのm5採用AD03・正式shot政策は保留。

各群で軸平均、2×2標本共分散、軸別MCSEと相対MCSE、未加重paired単価の平均・分散・MCSE、観測min/max、S0/S2_single/S2_multi/S4の件数・条件付き平均/分散を評価する。33本未満・重複・非finite・pairing欠測・未解決SOURCEはSTOP。1観測の層内分散と未観測の条件付き費用はnull。ゼロ観測分散は母分散0とせず、n_Eをnullとする既存規則を維持する。

未採択の設計感度は既存案ρ=5%/2%/1%、分散倍率f=1/2/4。群別の `max_axis ceil(f*s²/(ρ*mean)²)` を名目n_E案として出し、4q共通案は各公式内の最大値とする。これは提案のみで正式採択せず、未知tail上界や母平均順位保証と呼ばない。未解決qを除外して共通nを成立させない。

保存lawの3確率（orders0/2/4）とpiece整数step数から、理想的独立stepモデルの排他的4層確率、33本の理想期待観測数と1回以上観測する確率を計算する。binary64丸めの範囲の確率和を80桁Decimalで正規化し、4状態recurrenceで相殺を避ける。実samplerは変更しない。これはfinite PRNGの厳密独立性の証明でも、未観測層の費用・分散・tail上界でもない。費用0を仮定せず、tail cost boundはnullのまま。

H4/H6は独立plan/grant/消費marker/output/STOPを持ち、別processで並列評価する。各processは最大12個の固定群を利用可能CPUに合わせたthreadで読み取る。共有環境・他jobの設定を変更しない。一方の数値FAILから他方の科学条件を変更・retryしない。共通SOURCE/入力不整合はfail closedで扱う。

正常/異常終了後はSHA・native exit/worker回収・元入力保全を記録してmandatory STOPする。新しい本探索・独立確認・自動top-up・shot計算・総費用会計・再ランキング・AD03採用・最終evidence freezeへ進まない。次はユーザーがMCSE目標、名目分散感度、未観測tailの扱いと本探索n_Eを判断し、実取得を別途承認する。
