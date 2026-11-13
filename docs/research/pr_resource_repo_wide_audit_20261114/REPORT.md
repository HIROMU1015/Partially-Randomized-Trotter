# PRリソース評価研究：リポジトリ横断・実装総合監査（2026-11-14）

対象は、PRリソース評価の現行4系列（A/B/C/D）と、それが依存する過去の固定SOURCE・共通実装・旧pipelineである。
今回は調査と人工データでの技術検証だけを行った。production codeの修正、新しい科学計算、保存科学結果の数値再解析、実回路のbuild/compile/samplingは行っていない。

- 個別所見の詳細（対象file・commit・根拠・改善案）：[FINDINGS](FINDINGS.md)
- 計測値の原本：[人工検証と計測](evidence/ARTIFICIAL_CHECKS.txt)、[回帰テスト一覧](evidence/TEST_MATRIX.txt)、[profile](evidence/PROFILES.txt)、[コード構造の集計](evidence/CODE_STRUCTURE.txt)
- 検証script：[checks/](checks/)
- 前回監査（A/B/C/D限定、2026-11-13）：[REPORT](../pr_resource_implementation_audit_20261113/REPORT.md)

## 1. 結論

**中核の物理・統計計算に、科学的結論を変える不具合は見つからなかった。問題は「接続」と「無駄」に集中している。**

1. **計算の正しさ**：PF/PRの時間発展、有限K RTEの平均とnormalizer、PFの係数表、物理kernelの版管理を、定義式と直接照合した。うち2件は人工データで独立に再実装して突き合わせ、一致した（差 ≤ 3×10⁻¹⁴）。
2. **科学上の注意が必要な点は3つ**ある。いずれも計算の誤りではなく、比較条件と評価法の問題である。
   - Full-H pure RTEの保存価格は旧lowering、PRの新価格は新loweringで、そのまま比べるとbaselineが不利になる（S1）。
   - 決定論的RPE risk上界は、現行の配分では約0.49 Haで、目標（10⁻³ Ha級）に対して情報を持たない（S2）。
   - Gの約97%を占めるm10〜12のcompiled価格は、直接には測定できない規模である（V1、E4）。
3. **最大の非効率は、計算時間ではなく「初回実行での配線ミスによるSTOP」である。** 11月13日の1日で7件あり、それぞれが承認の往復を1回消費した（I1）。
4. **計算時間の無駄で効果が大きいのは3つ**で、いずれも実測に基づく。
   - `candidate_id`の毎回parse：遅いテスト11本の実行時間の77%（E1）。前回監査で指摘済みだが未対応。
   - X/Y両軸のcompile：m6の1 pathの37%。人工回路2000個で、両軸の差が最後の1q runに限られることを確認した（E2）。
   - 巨大回路のfingerprintの繰返し：1 pathで少なくとも6回、約17%（E3）。
5. **実装構造**：Pythonは3,064 file・56万行ある。現行系列から静的に到達するのは非test codeの26%で、残りは過去研究の保全分である。現行の133 fileは107個の固定commitを参照しており、mainには1つも含まれない（C1、I5）。

## 2. 優先順位

### 2.1 科学的に危険な問題（正式なG比較の前に対処）

| 順位 | 所見 | 重要度 | 何が起きるか | 対処 |
|---:|---|---|---|---|
| 1 | **S1** Full-H RTE価格とloweringの不整合 | 高 | 旧lowering価格のbaselineと新lowering価格のPRを比べると、baselineがcontrolled ZZ 1個あたり17〜19 RZ（P0実測）不利になる | RTE tailを持つ全価格に`lowering_id`を必須化する。Full-H RTEは新loweringで取り直すか、旧価格と明示して比較から外す |
| 2 | **I2** 生きた共有moduleのbyte/AST固定 | 高 | Bの`h4_s0_v2_intake`は、B自身の35分後の変更で動かなくなった。B・Dのheadで5 test fileが失敗している | 固定は作業tree上のfileではなくcommitのblobに対して行う。branchごとに現行系列の全testを通す |
| 3 | **S2** 決定論的risk上界が緩すぎる | 中 | 上界約0.24 Ha²は、unsafe確率2.1×10⁻⁴に一律の最大loss約1128 Ha²を掛けた値。Nの資格判定に使えない | 未解決（zero/tie）時の推定量を定義する。最初にunsafeになったroundに応じた幾何的なloss上界を使う（FINDINGS S2） |
| 4 | **V1** 長時間roundの価格domain | 中 | m10〜12は直接compileできず、価格modelが結論を決める | 価格modelの外挿誤差を、方式間の差の大きさと比べて必要精度を決める |

### 2.2 改善効果の大きい非効率

| 順位 | 所見 | 実測・根拠 | 改善後の見込み | 変更の大きさ | 科学的再検証 |
|---:|---|---|---|---|---|
| 1 | **I1** 初回実行の配線STOP | 11/13に7件。原因はimport経路、exec scope、plan key、参照先、scalarの取り違え等 | STOP 1件につき承認往復1回を削減 | 小：実loaderで数値計算の直前まで進む「配線preflight」を標準化 | 不要（数値計算に入らない） |
| 2 | **E1** `candidate_id`の毎回parse | 7.1 ms/回。遅いテスト11本で623/808秒（77%） | 0.005 ms/回（実測、出力同一）。該当テストは数分の一に | 極小：数行のcache | 不要（同一出力） |
| 3 | **E2** X/Y両軸compile | m6の1 pathで857秒中318秒（37%）がY側。人工2000回路で差は最後のancilla runだけ、\|C_Y−C_X\|≤1 | 1 pathの時間を約4割削減、compile予算を半減 | 中：新SOURCEと契約変更 | 小：保存済みX/Y対との照合 |
| 4 | **E3** fingerprintの繰返し | m6の1 pathで少なくとも6回×約24秒＝約144秒（17%） | 1〜2回に削減（推定で約100秒減） | 中：C1 Backendは固定なので新SOURCE | 不要（価格を変えない） |
| 5 | **E4** 直接compileの適用範囲 | m6で365万命令・2.8 GB・242秒/軸。event数はT²に比例 | m9以上の直接測定を計画から外し、作業を価格modelの検証に集中 | 計画の変更のみ | — |
| 6 | **E5** exact有理数の全count列挙 | N=64で0.43秒/round、N=256で35.7秒・280 MB | 周辺和の利用でmemory O(N)。閉包検査は幾何条件＋余裕で代替可能 | 中 | 小：既存の全列挙と小Nで一致確認 |

### 2.3 修正する価値が低いもの

- 過去研究のhelper重複（`sha`が193 file、`read`/`write`が各170 file等）：固定科学成果の一部で、触る利益がない。
- `trotterlib`の一括import（1 processあたり約3秒・345 MB）：providerは固定SOURCEなので変更しない。新規moduleで依存を増やさないだけでよい（E7）。
- H4/H6のAccounting複製、構造Sの純Python実装（m6で95秒/path）：現状の規模では律速でない。

## 3. 確認できたこと（変更不要）

| 項目 | 確認方法 | 結果 |
|---|---|---|
| 有限K RTEの平均とnormalizer | 人工のZ/ZZ成分で全eventを列挙し、`b·E[U]`とpaired Taylor多項式を比較。時間の符号は正負とも | 差 ≤ 2.9×10⁻¹⁴。bは閉形式と一致 |
| PRの有限平均kernel | carrierのspectral版、Aのdirect-expm版、独立の総当たり実装を、人工行列で比較 | 差 ≤ 7.6×10⁻¹⁵。S2の順序、負の係数、event配分、log B、scalar一回適用が一致 |
| PF係数表 | 対称合成の次数条件を計算 | 標準Yoshida4・M5 bestとも Σw=1、Σw³=0（4次） |
| 物理kernelの版 | 実行時に使う私的provider rootの252 fileをcommit `c31ef424`と比較 | 全file一致。現行treeと共通の215 moduleで内容が違うのは2件（旧検証moduleと`__init__`） |
| 再現性 | 現行codeが固定する107 commitがremoteにあるか | すべてremoteのbranch/tagから到達可能 |
| 数値経路の効率 | 既存の実行記録 | signalは行列累乗（O(d³ log q)）、全11roundの構造平均は4分38秒、決定論的riskは7秒。律速ではない |
| 前回監査R1・R3 | SOURCEの確認 | R1（効率的controlled ZZ）とR3（高速decoder）は実装済み。R2（`candidate_id`）とR5（共有moduleの版管理）は未対応 |

## 4. 調査範囲と限界

- **基準commit**（2026-11-14 01:04 JST時点）：A `878000d8`、B `06db9aef`、C `367dc98d`、D `01e068af`。テストとprofileは、その直前のhead（A `69ee0ff8`、B `6106e4d9`、C `6306fb3a`、D `01e068af`）で実行した。
- **固定SOURCE**：provider `c31ef424`、carrier `155529e2`、Full-H `9b83b96b`、C1 `1a87e533`、構造counter `8457c443`。
- **過去系列**：全file・全関数を集計・重複検出・到達解析の対象にした。精読したのは、現行経路から呼ばれる部分と、同じ計算の別実装である。O2・pr-reference・research-C等の独立系列の科学内容は監査していない。
- **未確認**：repository外の私的入力（保存snapshot、compile cache）の中身、他サーバーの成果物、GitHub上で既に削除されたbranchの内容。
- **実測の性質**：すべて人工データ、既存の実行記録、静的解析による。「推定」と書いた値は、実測値からの比例計算である。
- **自分の検査であること**：すべて同一agent（Claude Code）による確認であり、独立した査読者によるreviewではない。

## 5. 作業中に生じた副作用（報告）

監査の最初に`git fetch origin --prune`を実行し、共有`.git`から、remoteで既に削除されていたbranchのremote-tracking ref（`origin/track-b-*`等）を消してしまった。branchの削除・統合・履歴変更はしていないが、意図しない変更なので記録する。

- **影響の確認**：GitHubの活動記録から、過去90日に削除された1,128 branchの最終commitを取得し、すべてlocalに存在し、localのbranch・tag・その他のrefから到達できることを確認した。到達不能になったcommitは0件。`git fsck`でもdanglingなcommitは0件だった。
- **復元**：消えたのは名前だけで、commitは保全されているため、復元操作はしていない。必要なら、活動記録の名前とSHAから再作成できる。
- **再発防止**：今後は`--prune`を付けない。

## 6. 引渡し

- **ユーザー／GPT**：2.1と2.2の各項目の採否と担当を判断する。S1とE2は科学契約（価格の定義・X/Yの扱い）に関わる。
- 今回は修正も追加の科学計算も行っていない。公開後はmandatory STOP。
