# Track D B4-2：固定13-roundの元H向け条件付き統計評価

2026-11-06 JST。実装基点はB4-1公開 `31cd67c91abda3c9c5d4b643c434c040b8ceaeca`。ユーザーの今回の明示科学実行許可に従い、B1保存operator誤差を既存adapterへ接続する。過去のSTOP・SOURCE・結果は変更しない。

固定：linear H4／1.00 Å／STO-3G／Nα=Nβ=2、number-CDF8／元frame順LD2／ordered S2/s1/r32/K0、h=δ=.1。元H厳密groundを仮定し準備費除外。εE=.00159 Ha、αtotal=.01、βtotal=π/6、βstat=π/12、αaxis=.01/26、m0..12／q1..4096、prior[-10,10] Haを今回の条件として採用する。旧DF energy reserve・Weyl加算は再適用しない。

科学SOURCEは本PLAN・runner・人工テストをcommitした新revision。B1科学SOURCE `8c6eeb02e89f6cf14870789719493e75482f62f8`、保存結果 `d3913866e6ade4ab97ac02645fed91b2d5fcfa75`、CDF8科学SOURCE `4df8eca29c59ffd55e2dd898f05da16bb246ddbe`、保存carrier `3649b75af191916daf44abb91e14d34c43b83e8f` と役割を区別する。

元H packet SHA256 `1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db`、CDF endpoint `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`、physical H reference `92a8d27bdebb53937f6361c106a13ef90b9110930d4bf4ddb577796cfb208739`。B1 SOURCEのinput binding、保存law・normalization・同sectorと固定生成SOURCEを照合する。NPZはhash/header確認のみで配列数値解析をしない。physical ground JSONの共通state文字列をbinding根拠にしない。

alias priorの包含根拠は同じ元Hの保存binary64最低固有対・residual、分枝余裕は採択したprior／h／βから評価する。厳密包囲や新ground solveとはしない。ideal IID／exact nativeをモデル定義として追加raw誤差0とし、actual law／compiled／Reference実装誤差は未知nullのまま分離する。

既存 `OriginalHRoundEvidence.from_b1_row`、`assess_round`、`assess_energy`、Hoeffding／horizonを使用し、adapter・旧SOURCEを変更しない。保存geometryとの比較許容差2e−15はbinary64 scalar演算順序差の技術照合に限る。物理的phase予算やfailure和には許容超過を追加しない。decoder条件は整理するが測定データ復元・simulation・成功率評価は行わない。

明示runner `scripts/run_track_d_b4_2.py --execute` の一回だけで、出力は `artifacts/track_d_b4_2_rpe_shots/2026-11-06/run01`。既存出力があれば起動しない。入力・SOURCE不一致、alias不足、horizon不足、正半径／phase／failure不通過、数値範囲異常は証拠を保存してSTOPし、無断retry・事後目標変更をしない。

処理は13行の短いscalar評価であり、新しいmatrix計算・worker並列化は不要。CPU/RAM/workerへ人工capを設けず、実空き資源を観測し、既存project専用venvを読取専用で使う。共有設定・C4・他jobを操作しない。

許可scopeの結果／SOURCE／各軸N／energy・failure・alias判定／ideal-actual資格／実行・保全・STOPを保存し、専用branchへ公開、remote SHA・tree・bytesを照合する。原科学SOURCEのlocal commitとGitHub接続で生成する公開SOURCE commitのSHAが異なる場合は、同一tree・コードbytesの対応を明示する。新grant・承認証明は作らない。

**完了・不成立ともmandatory STOP。B2、sampling、shot simulation、ground solve、DF refit、回路build／compile、C4介入、actual保証、全RPE総費用、詳細科学レビューへ進まない。次担当はユーザー／GPT。**
