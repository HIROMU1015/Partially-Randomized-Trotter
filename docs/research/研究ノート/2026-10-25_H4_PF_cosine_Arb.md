# 2026-10-25 H4 PF cosine Arb局所包含評価

利用者のPF cosine1件・64→96→128bit各1回/初回最大3状態作用の明示承認に従った。
基点5c69e1339955c77273b356df30e94ec3c57458b1、専用branch
`codex/non-df-h4-central-arb-pf-cosine-20261025`。

grant前SOURCE bbb29c868では旧Cと現system zlib SHA差でsetup停止、科学0/grant0/run01なし。
変更主体/時点はUNKNOWN。旧SOURCE/STOPを保存し共有環境は変更しなかった。
FLINT数値package103の旧hash一致を保持し、現在runtime197を新SOURCE92a5f104a2e9962353870e87f1d910d236adba41へ固定した。
最終構成は新production11＋再利用25＝36 PASS。修正前35件・追加回帰11件の記録を両方保存。

新one-shot grant発行/消費1。64bit・状態作用成功1/総試行1、局所差scalar1、約47.2747秒。
compiled signal区間幅54849/4611686018427387904（約1.18935e-14）。
rho上端180675711/18889465931478580854784（約9.565e-15）≤保存予算約2.649e-6。
raw ballの微小負下端を保持し、定義rho>=0との共通部分[0,u]を報告。rho=0の確定や旧nullの0置換ではない。
64bitで比較が決まったため96/128bitは実行せず。新C-target/PR/sine/DF/compile/sampling/N_G/R2/正式受理0。

worker652280は00:24:17 JST native exit0/reap、controller exit0。今回STOPは元H4 native UNKNOWNを置き換えない。
保存値・SHA監査65項目PASS、旧証拠412/入力9/runtime197/code19/旧manifest186行保持。
局所数値上界と正式受理を区別し、LOCAL_BUDGET_MET=false、旧N_G/C-L/PR全law/tail等は保持。

次：ユーザーがPR個別C-targetとcompiled包含の対象・軸・call範囲を別承認後、Codexが別SOURCE/未使用grantで限定実施。
現在PR拡張未実行。公開処理後mandatory STOP。

## 公開試行

結果commit b0a8bb28c8ba78c11b6be2bbc67b575fa92e9fd8。
origin push1回はHTTPS Username取得不能でexit128、remote専用branch検索0件、remote SHA=null。
公開記録を後続local metadata commitへ保存する。結果とSOURCE/grant/native/STOPは不変。
ユーザーが認証可能な環境で同branchをpushできる。mandatory STOP、科学計算への自動移行なし。
