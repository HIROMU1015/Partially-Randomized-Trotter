# Truncated PF：有限科学検証の段階計画（未承認・未実行）

2026-11-14 JST。[REPORT](REPORT.md)の§4〜§5を段階表にしたもの。**ここに書いた計算はすべて未承認で、実行していない。** 各段階の後はmandatory STOPとし、次の段階には自動で進まない。

## 共通の固定条件

- 対象：Linear H4/1.00 Å/STO-3G、保存tol-only 1e−8 DF10、weighted-Hermitian completion、sector(2,2)/36次元。
- 固定するもの：保存H/state/snapshot、元のfragment順、h（元binary64の0.1）、M12、P0。
- target：元のfull DF-Hの同sector基底エネルギー。保存solver proxyは `−2.1663874486347634 Ha`、Aの保存固有系の最小値は `−2.1663874486347643 Ha`。保持Hの基底エネルギーをtargetにしない。
- 方式：B0I（切り捨てたtailのPauli identity係数を、総Tで一回だけ位相に入れる）。B0はB0Iから定数`−c_tail`だけずれるので、解析的に導く（新しい計算はしない）。
- 公式：既存の3公式（S2、標準Yoshida4、M5 best）。
- 幅の梯子（hの倍数、exactな有理数）：S2 {1, 2, 16/7, 8/3}、Y4 {4, 16/3, 32/5}、M5 {8, 16}。Aが評価済みのFull PF格子と揃えてある。
- schedule：Bの既存規則`q_m=ceil(T_m/Δ)`、early roundの実幅を含む。全99 configurationと154 step cellを既存generatorで生成済み（[PLAN_METADATA.json](PLAN_METADATA.json)）。c546のB0I p7 S2/.1とAのFull PF 3候補のconfiguration IDは一致を確認した。

## 段階表

| 段階 | 担当 | 新しい科学処理（上限） | 再利用 | 出力 | STOP・UNKNOWN |
|---|---|---|---|---|---|
| T1：保持Hの切断シフト | C（またはA） | 下記の3処理 | C numericsのContext/loader（`86f37d25`＝`5cc7c07e`とbyte一致）。保存scalar p∈{1,2,3,5,7,9}。Aの元H固有系cache `69ee0ff8` | 各pの`δE_trunc`、元H基底群との重なり、保存stateの重み、λ_R、c_tail。B0の値は`−c_tail`で導出 | 入力hash・SOURCEの不一致は全体STOP。固有系gateの不成立や対応の曖昧さ（最大trace重なり≤0.5、または最小群と最大重なり群の不一致）はそのpだけUNKNOWN |
| T2：PF H_eff cell（主対象） | A（評価器の所有者）。callerはCも可 | T2で使えるpごとに14 cell、最大140 cell（p=0..9）。各cellでPF step 1、Schur/H_eff 1、二時点のdirectとspectralのmean各1組（初使用roundとm12） | Aの評価器 `878000d8d`、`expm_step` `dd24dd329`、Context.pf_step。p=10はAの既存Full PF cellを再利用（再計算0） | 各cellの`δE_total`（評価器）、`δE_PF|ret`（分解helper）、群の対応・重み・分枝診断・二時点照合 | 照合gate：p7/S2/hのcellから13roundをspectral再構成し、保存B0I p7 S2/.1（`367dc98d`）の13 directと元gate（1e−9）内で一致しなければ全体STOP（実装の不一致）。数値・分枝のgate不成立はcell単位でUNKNOWN。閾値の緩和・window変更・retryはしない |
| T2b：境界challenge | 同上 | 梯子の最も粗い幅が主screenを通ったとき、その(p,公式)ごとに隣接する粗い幅を1つだけ（S2 16h/5、Y4 8h、M5 32h）。最大3×|P*| | 同上 | 同上 | **別承認**。自動では実行しない |
| T3：価格 | D（取得）、C（接続・guard） | 選んだtemplate（p,公式）ごとに、ある幅でq-affineの2点を校正し、別の幅・大きいqで1点holdout。X/Y両軸で最大6 wrapper。上限6 templateで最大36 wrapper（保存分は差し引く）。B0とB0Iの同値検査1組 | D native・receipt、C `price_implementation_guard`（Truncated＝`NOT_APPLICABLE_NO_RTE`）。保存B0I p7 m3価格（`155529e2`）は、domainが一致すれば受け入れる | 各templateの価格modelと、holdoutの残差 | holdout誤差が事前に定めた比較差の1/3を超えたら、modelを採らずSTOP（refitしない）。Dのtruncated builder／receiptが未対応なら、取得前に技術拡張へ戻す |
| T4：全13round signal | C | finalistごとに13 mean、最大6 finalistで78（保存B0I p7 S2/.1は再利用） | 既存`C.execute`＋保存reference pair（`bba15083`） | Bが読むsignal packet | 既存gate、`367dc98d`と同じ規則 |
| T5：risk・N | B | finalistごとに、採択した契約（W0のD1〜D3）でのDP評価 | Bのdeterministic risk／DP | モデルMSEの上界・下界とN | W0の採否が前提。未採択なら実行しない |
| T6：G | B/C | finite menu上の集計 | 既存`W.compare`／`B.select`＋guard | 区間つきのG | 区間が重なれば「未識別」とする |

## T1の具体的な件数（上限）

- context load：1。
- c_tailとλ_Rの抽出：p∈{0,4,6,8}の4件を、既存の`matrix_only_tail_coefficients`で行う。保存のある6件は再計算して保存値と照合する（6件。不一致ならSTOP）。
- 保持Hの固有系：10件（36×36のeigh、p=0..9）。保持Hは`c_0 I＋one-body＋fragment 0..p−1＋c_tail I`。

## T2→T3のfinalist選択規則（事前固定の案）

1. 契約の判定を2通り記録する。
   - Q1（Minkowski）：`|δE_total|≤ε_trot`。
   - Q2（直接モデルMSE）：`δE_total²<ε_E²`。小さいほど有利で、N proxyは`1/(1−(δE_total/ε_E)²)`。
2. 相殺依存の判定：L3（`|δE_total|`）では通るがL2（`|δE_trunc|+|δE_PF|ret|`）では落ちる候補には`CANCELLATION_DEPENDENT`を付ける。主比較に入れるなら、隣接幅でもL3が成立することを条件にする。
3. 構造上の費用proxy：m12のsigned piece数×(p+1)。ブロック数に比例するという仮定で、T3で検証する。
4. 各公式からproxy上位2つ、合計最大6 templateをT3へ送る。p=10（Aの3候補）を同じ公式の対照として必ず残す。

proxyは計算順を決めるためだけに使う。候補を恒久的に除外する根拠にはしない。

## 実行しないもの

- 係数単位のdrop（Track C型）、round別のprefix、fragmentの並べ替え。新builderと新しい科学scopeが必要。
- B0S（sector内のtraceをscalarにする案）。新しいscalar規則になる。
- h/Mの探索。
- PRの新しいLD/κ探索（別担当）。
